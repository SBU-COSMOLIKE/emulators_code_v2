"""Diagonal scores and target-law transforms for CMB spectra.

The classes in this file provide the interface used by the shared training
loop: encode a raw target, decode a network output, compute one score per
cosmology, and reduce those scores to a training objective. The wrapped
``CmbDiagonalGeometry`` standardizes each multipole with the positive scale
stored in the covariance product. The plain score is the sum of squared
standardized residuals over the multipoles.

The ``none`` law applies no analytic amplitude transform. The
``as_exp2tau_ref`` law multiplies each raw spectrum by

``f = (A_s_ref / A_s) * exp(2 * (tau - tau_ref))``

before centering and standardization. This removes the dominant primary-CMB
amplitude trend but does not make the remaining target independent of all
amplitude information. ``A_s`` and ``tau`` are read from named parameter
columns. The reference values are persisted scientific facts, and ``f`` is
one at the reference cosmology.

Encoding multiplies by ``f``. Decoding therefore divides by ``f`` to return
physical ``C_l``. The standardized network residual also contains ``f``, so
the score divides the residual by ``f`` before squaring. At a fixed physical
error, the reported score is then independent of this chosen target
transformation. ``AMPLITUDE_LAWS`` maps the stored law name to its class, and
``make_cmb_chi2`` constructs the selected implementation.
"""

import math

import torch

from .core import CosmolikeChi2


def _require_finite_cmb_tensor(value, *, stage, shape):
  """Require one exact, finite floating tensor in the CMB amplitude law."""
  expected = tuple(int(x) for x in shape)
  if not torch.is_tensor(value) or not torch.is_floating_point(value):
    raise TypeError(
      "CMB amplitude-law stage " + repr(stage)
      + " must be a real floating-point Torch tensor, got "
      + type(value).__name__ + ".")
  if tuple(value.shape) != expected:
    raise ValueError(
      "CMB amplitude-law stage " + repr(stage) + " must have exact shape "
      + repr(expected) + ", got " + repr(tuple(value.shape)) + ".")
  if not bool(torch.isfinite(value).all()):
    raise ValueError(
      "CMB amplitude-law stage " + repr(stage)
      + " contains NaN or infinity.")
  return value


class ResidualRoughness:
  """
  Band-explicit high-pass penalty on the whitened residual.

  CMB spectra are smooth multipole by multipole, so an emulator residual
  that OSCILLATES on short periods (much shorter than the acoustic peak
  spacing, ell_A ~ 200-300) is a network artifact, never physics. This
  term measures exactly that content: it smooths the residual with a
  triangular kernel (a boxcar of width `period_cut` applied twice) and
  penalizes the sum of squares of what the smoothing removed:

      r  (B, n_ell)  whitened residual, r = pred - target
         │  smooth = boxcar(w) ∘ boxcar(w)   (reflect-padded, w odd)
         ▼
      rem = r - smooth(r)                     the short-period remainder
         │  square, sum over multipoles
         ▼
      c_rough  (B,)  per-sample roughness penalty

  (legend: B = batch rows; n_ell = multipole count; w = the kernel width
  in multipoles, period_cut rounded to the nearest odd integer; rem =
  the high-frequency remainder the term penalizes.)

  Band behavior (why this satisfies the term's two cautions): the double
  boxcar's smoothing response is ~ sinc^2(w / P) for an oscillation of
  period P, so the REMAINDER carries full weight at P << w and nearly
  none at P >= ~4 w. With the default period_cut 50 that separates the
  penalized band (P <~ 50) from the acoustic band (P ~ 200-300, where a
  shifted peak or a lensing-smoothing misfit lives) by the ruled factor
  >= 4 — those misfits belong to the plain chi2, and this term barely
  sees them. Acting on the RESIDUAL (never the prediction) makes lensing
  neutrality structural: the penalty is identically zero when the
  prediction equals the lensed truth, however smooth its peaks.

  Arguments (constructor):
    period_cut = the multipole period below which residual oscillations
                 are penalized with full weight (the YAML knob; rounded
                 to the nearest odd kernel width, >= 5).
    device     = device the kernel lives on (the geometry's).
  """

  def __init__(self, period_cut, device):
    w = int(round(float(period_cut)))
    if w < 5:
      raise ValueError(
        "loss.roughness.period_cut must be >= 5 multipoles (got "
        + repr(period_cut) + "); shorter periods cannot be separated "
        "from noise on an integer ell grid")
    if w % 2 == 0:
      w += 1   # odd width keeps the kernel centered (no phase shift)
    self.width  = w
    self.kernel = torch.ones(1, 1, w, device=device) / float(w)

  def per_sample(self, r):
    """Per-sample roughness c_rough from the whitened residual.

    Arguments:
      r = (B, n_ell) whitened residual (pred - target).

    Returns:
      (B,) sum of squares of the residual's short-period remainder.

    Raises:
      ValueError when the spectrum is too short for the kernel (the
      reflect padding needs n_ell > width // 2).
    """
    pad = self.width // 2
    if r.shape[1] <= pad:
      raise ValueError(
        "roughness penalty needs n_ell > period_cut/2 (n_ell = "
        + str(int(r.shape[1])) + ", kernel width " + str(self.width)
        + "); a spectrum this short has no band to separate")
    x = r.unsqueeze(1)                       # (B, 1, n_ell)
    # boxcar applied twice = a triangular smoothing kernel by
    # composition (C0-smooth response, no boxcar ringing), each pass
    # reflect-padded so the edges see no artificial jump.
    for _ in range(2):
      x = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
      x = torch.nn.functional.conv1d(x, self.kernel)
    rem = r - x.squeeze(1)
    return (rem * rem).sum(dim=1)


# The imposed-amplitude-law registry: law name -> the extra config keys
# it needs (persisted by name in the artifact, resolved values, never a
# code default). "none" learns the raw C_ell shape and needs nothing;
# "as_exp2tau_ref" imposes the DIMENSIONLESS order-one factor
# f = (A_s_ref / A_s) * exp(2 (tau - tau_ref)), f == 1 at the persisted
# fiducial, and needs the two named input columns it reads A_s /
# tau from PLUS the two reference values (as_ref, tau_ref).
AMPLITUDE_LAWS = {
  "none":           (),
  "as_exp2tau_ref": ("as_name", "tau_name", "as_ref", "tau_ref"),
}

# The retired raw-factor law: f = exp(2 tau) / A_s carried an
# arbitrary ~1e9-scale normalization (raw A_s ~ 2.1e-9), so its encoded
# target and float32 conditioning were unit-porting defects. A config or a
# persisted artifact naming it is refused with the retrain instruction --
# never silently reinterpreted under the new order-one convention.
_RETIRED_AMPLITUDE_LAWS = {
  "as_exp2tau": "the raw-factor law exp(2 tau)/A_s carried an arbitrary "
                "1e9-scale normalization; retrain under 'as_exp2tau_ref' "
                "(the dimensionless (A_s_ref/A_s) exp(2 (tau - tau_ref)), "
                "f == 1 at the fiducial), supplying data.cmb.as_ref and "
                "data.cmb.tau_ref.",
}


def reject_retired_amplitude_law(law):
  """Refuse a retired amplitude law with its retrain instruction.

  The shared adjudicator, called wherever a law name is resolved: the config
  build (make_cmb_chi2 below), the geometry h5 rebuild
  (geometries/cmb.py CmbDiagonalGeometry.from_state), and the staging /
  prediction dispatch. One message source so an old-convention name is a loud,
  identical error at every boundary, never a silent reinterpretation.

  Arguments:
    law = the amplitude-law name (from a config or a persisted artifact).

  Raises:
    ValueError naming the retired law and how to retrain; returns None for a
    live or unknown-here name (the caller's own registry check handles those).
  """
  if law in _RETIRED_AMPLITUDE_LAWS:
    raise ValueError(
      "amplitude law " + repr(law) + " is retired: "
      + _RETIRED_AMPLITUDE_LAWS[law])


class CmbDiagonalChi2(CosmolikeChi2):
  """
  Plain diagonal chi2 for a CMB spectrum (the "none" amplitude law).

  Holds a CmbDiagonalGeometry (self.geom) instead of a
  DataVectorGeometry and overrides only the metric: the per-sample chi2
  is the sum of squared whitened residuals, sum_l (pred_l - target_l)^2
  over the multipoles. Because the geometry whitens by the
  cosmic-variance scale sigma_l = 1/sqrt(cinv_l), that sum already is the
  cosmic-variance chi2 sum_l cinv_l (C_pred_l - C_target_l)^2 (the sigma
  cancels the cinv), so there is no covariance to contract and no mask to
  unsqueeze through, exactly as for ScalarChi2. Everything else is
  inherited from CosmolikeChi2:

    - encode / decode forward to self.geom (center-and-scale each way);
    - dest_idx / total_size forward to self.geom (arange(n_ell) / n_ell),
      so the loop sizes the model to n_ell with no CMB branch;
    - loss(...) and _reduce(...) are the shared trim / mode transform /
      focal reduction, so every knob applies to the CMB chi2 unchanged.
  """

  # the optional residual-roughness term: absent (None, the class
  # default) = the loss path is byte-identical to the plain chi2 path;
  # configure_roughness attaches it. lam is a fixed per-run multiplier (a
  # constant in the compiled graph, unlike the per-epoch trim/focus
  # tensors).
  _rough     = None
  _rough_lam = None

  def configure_roughness(self, lam, period_cut):
    """Attach the residual-roughness term (train_args.loss.roughness).

    Arguments:
      lam        = weight of the per-sample roughness penalty added to
                   the per-sample chi2 (c_total = c_chi2 + lam*c_rough);
                   > 0 (an absent block, not lam 0, states OFF).
      period_cut = the penalized-band edge in multipoles (see
                   ResidualRoughness).

    Returns:
      self (so the caller can configure in one expression).
    """
    self._rough_lam = float(lam)
    self._rough     = ResidualRoughness(period_cut=period_cut,
                                        device=self.geom.center.device)
    return self

  def chi2(self, pred, target, full=False):
    """Per-sample chi2 = sum of squared whitened residuals.

    The targets live in the whitened (unit-cosmic-variance, independent)
    space the geometry's encode produced, so the cosmic-variance
    Mahalanobis distance is the plain squared L2 residual summed over the
    multipoles. full is accepted so the signature matches
    CosmolikeChi2.chi2 but has no effect: a diagonal geometry has no
    masked entries to scatter through. The roughness term never enters
    here — chi2 is the evaluation metric; the penalty is a TRAINING
    objective addition, applied in loss only.

    Arguments:
      pred   = (B, n_ell) network outputs in the whitened space.
      target = (B, n_ell) whitened targets, same shape.
      full   = accepted for interface parity with CosmolikeChi2.chi2;
               ignored (no mask / covariance on the CMB path).

    Returns:
      (B,) per-sample chi2.
    """
    r = pred - target
    return (r * r).sum(dim=1)

  def _penalty_residual(self, pred, target):
    """The residual the roughness penalty is measured on.

    The plain diagonal residual pred - target is already physical, so this
    returns it unchanged. The imposed-amplitude subclass overrides it to
    divide the per-row factor out, keeping the penalty law-neutral.
    """
    return pred - target

  # Executed metric: a plain sum of n_ell squared whitened residuals. A
  # length-n_ell (= kept width) diagonal reduction, the SAME depth as the
  # base dense r^T Cinv r. So the chi2-domain band's per-row term count
  # (_chi2_n_terms = the kept width w) is inherited from CosmolikeChi2
  # unchanged; retired the redundant override that returned the same
  # width.

  def loss(self, pred, target, mode="sqrt", trim=0.05,
           focus=0.0, focus_scale=1.0, berhu_knot=None, berhu_cap=None,
           berhu_s=None):
    """Training loss; adds the roughness penalty per sample when present.

    With no roughness term this delegates to CosmolikeChi2.loss unchanged
    (byte-identical, the off-identity rule). With one, the composition
    is c_total = c_chi2 + lam * c_rough per SAMPLE, before
    the shared reduction — so trim / focus / berhu / the mode transform
    all act on one number per sample and compose with the penalty
    exactly as they compose with the plain chi2 (one reduction path, no
    second ladder). Both pred and target hold any imposed amplitude
    factor, so the residual (and hence the penalty) is law-neutral.

    Arguments: identical to CosmolikeChi2.loss (mode / trim / focus /
    focus_scale / berhu knots), forwarded to the shared _reduce.

    Returns:
      a scalar loss tensor.
    """
    if self._rough is None:
      return super().loss(pred, target, mode=mode, trim=trim,
                          focus=focus, focus_scale=focus_scale,
                          berhu_knot=berhu_knot, berhu_cap=berhu_cap,
                          berhu_s=berhu_s)
    c = self.chi2(pred=pred, target=target)
    c = c + self._rough_lam * self._rough.per_sample(
      self._penalty_residual(pred, target))
    return self._reduce(c=c, mode=mode, trim=trim, focus=focus,
                        focus_scale=focus_scale, berhu_knot=berhu_knot,
                        berhu_cap=berhu_cap, berhu_s=berhu_s)


class CmbFactoredChi2(CmbDiagonalChi2):
  """
  Diagonal chi2 with the imposed amplitude law (the "as_exp2tau_ref" law).

  The target presented to the network is the amplitude-rescaled spectrum

      target' = C_ell * (A_s_ref / A_s) * exp(2 (tau - tau_ref))

  a per-row scalar factor f = (A_s_ref / A_s) * exp(2 (tau - tau_ref))
  multiplied into every multipole (A_s and tau are the same for all l of
  one cosmology). Measuring against the persisted fiducial (A_s_ref,
  tau_ref) makes f a dimensionless order-one number, exactly 1 at the
  fiducial. encode bakes f into the target before the geometry centers it
  and divides by the stored per-multipole scale;
  decode divides f back out, so the emulator returns physical C_ell. The
  chi2 also DIVIDES the per-row factor back out of the whitened residual
  before summing, so the reported metric is the physical cosmic-variance
  chi2. Prediction and target for one row share the same f, so a plain sum
  would carry f^2 (not cancel) and bias delta-chi2, the threshold
  fractions, and selection by cosmology.

  This mirrors RescaledChi2 (losses/core.py) with a per-row scalar factor
  in place of the per-element analytic R, and, like RescaledChi2, divides
  that factor out of the metric. It is NOT neutral in the residual.
  needs_params = True, so the training loop hands encode / decode /
  chi2 / loss the whitened input params; _factor decodes them to physical
  through the param geometry and reads A_s / tau by column. Build by
  wrapping a geometry, CmbFactoredChi2(geom), then configure_law(...) to
  attach the param geometry, the A_s / tau column names, and the fiducial
  reference pair (as_ref, tau_ref).
  """

  # capability flag the loop reads (getattr default False elsewhere): this
  # loss's encode / decode / chi2 / loss take the whitened params, from
  # which the amplitude factor is built. Mirrors RescaledChi2.needs_params.
  needs_params = True
  # stash so the inherited loss reduction (which calls self.chi2 without
  # params) is reused unchanged; the plain chi2 ignores it, but the field
  # keeps the RescaledChi2 shape and documents the contract.
  _params = None

  def configure_law(self, param_geometry, as_name, tau_name, as_ref, tau_ref):
    """Attach the amplitude-law state (call once, after wrapping geom).

    Resolves the A_s and tau column positions in the raw parameter vector by
    name off the param geometry, so _factor can read them from the decoded
    physical params, and stores the fiducial reference pair (as_ref, tau_ref)
    that makes the factor dimensionless and order-one. A name the geometry
    does not carry is a loud error naming it and the available columns; a
    non-finite reference, or a non-positive as_ref, is refused.

    Arguments:
      param_geometry = ParamGeometry whose decode maps the whitened model
                       inputs back to physical params (what the law reads);
                       the same object passed to run_emulator.
      as_name        = the raw linear amplitude column name (e.g. "As"). A
                       run that samples logA materializes a raw A_s column
                       in the generator, so the law always reads a linear
                       amplitude (recorded in families-scalar-cmb.md).
      tau_name       = the optical-depth column name (e.g. "tau").
      as_ref         = the fiducial linear amplitude A_s_ref (> 0, finite):
                       the reference the factor is measured against, so
                       f == 1 at (A_s, tau) == (as_ref, tau_ref).
      tau_ref        = the fiducial optical depth tau_ref (finite).

    Returns:
      self (so make_cmb_chi2 can build and configure in one expression).
    """
    names = list(param_geometry.names)
    for nm, role in ((as_name, "amplitude (as_name)"),
                     (tau_name, "tau (tau_name)")):
      if nm not in names:
        raise ValueError(
          "CmbFactoredChi2.configure_law: the " + role + " column "
          + repr(nm) + " is not among the parameter columns "
          + repr(names) + "; the as_exp2tau_ref law reads A_s and tau from "
          "named input columns.")
    as_ref_f  = float(as_ref)
    tau_ref_f = float(tau_ref)
    if not (math.isfinite(as_ref_f) and math.isfinite(tau_ref_f)):
      raise ValueError(
        "CmbFactoredChi2.configure_law: the fiducial reference pair must be "
        "finite; got as_ref=" + repr(as_ref) + ", tau_ref=" + repr(tau_ref)
        + " (data.cmb.as_ref / data.cmb.tau_ref).")
    if as_ref_f <= 0.0:
      raise ValueError(
        "CmbFactoredChi2.configure_law: as_ref must be a positive linear "
        "amplitude; got " + repr(as_ref) + " (data.cmb.as_ref).")
    self.param_geometry = param_geometry
    self.as_name  = as_name
    self.tau_name = tau_name
    self.as_idx   = names.index(as_name)
    self.tau_idx  = names.index(tau_name)
    self.as_ref   = as_ref_f
    self.tau_ref  = tau_ref_f
    return self

  def _factor(self, params_whitened):
    """Per-row DIMENSIONLESS amplitude factor for whitened inputs.

    Decodes the whitened params to physical, reads A_s and tau by their
    resolved column positions, and forms the order-one factor

        f = (A_s_ref / A_s) * exp(2 (tau - tau_ref))

    which is exactly 1 at the persisted fiducial (as_ref, tau_ref). The
    factor is the same for every multipole of one cosmology and is returned
    as a column so Torch broadcasts it across the multipole axis.

    Arguments:
      params_whitened = (B, n_param) whitened model inputs (the tensor the
                        loop / predictor pass).

    Returns:
      f = (B, 1) amplitude factor on the params' device.
    """
    if not torch.is_tensor(params_whitened) or params_whitened.ndim != 2:
      raise ValueError(
        "CMB amplitude-law encoded parameters must be a rank-two Torch "
        "tensor with one row per cosmology; got "
        + repr(getattr(params_whitened, "shape", None)) + ".")
    batch = int(params_whitened.shape[0])
    width = len(self.param_geometry.names)
    params_whitened = _require_finite_cmb_tensor(
      params_whitened,
      stage="encoded parameters",
      shape=(batch, width))
    phys = _require_finite_cmb_tensor(
      self.param_geometry.decode(params_whitened),
      stage="decoded physical parameters",
      shape=(batch, width))
    a_s  = phys[:, self.as_idx]
    tau  = phys[:, self.tau_idx]
    if not bool((a_s > 0.0).all()):
      raise ValueError(
        "CMB amplitude-law parameter " + repr(self.as_name)
        + " must be strictly positive for every row; got "
        + repr(a_s.detach().cpu().tolist()) + ".")
    f    = (self.as_ref / a_s) * torch.exp(2.0 * (tau - self.tau_ref))
    f = _require_finite_cmb_tensor(
      f.reshape(-1, 1),
      stage="amplitude factor",
      shape=(batch, 1))
    if not bool((f > 0.0).all()):
      raise ValueError(
        "CMB amplitude factor must be strictly positive for every row; got "
        + repr(f.detach().cpu().reshape(-1).tolist()) + ".")
    return f

  def encode(self, dv, params_whitened):
    """Raw C_ell -> amplitude-rescaled, centered, whitened target.

    Squeeze, multiply by the per-row amplitude factor f (into target'
    space), then center and whiten.

    Arguments:
      dv              = (B, n_ell) raw physical C_ell rows.
      params_whitened = (B, n_param) whitened inputs (set f per row).

    Returns:
      (B, n_ell) encoded (rescaled, centered, whitened) target.
    """
    geo = self.geom
    f = self._factor(params_whitened)
    batch = int(f.shape[0])
    dv = _require_finite_cmb_tensor(
      dv,
      stage="physical spectrum before scaling",
      shape=(batch, int(geo.dest_idx.numel())))
    scaled = _require_finite_cmb_tensor(
      geo.squeeze(dv) * f,
      stage="physical spectrum after scaling",
      shape=(batch, int(geo.dest_idx.numel())))
    return _require_finite_cmb_tensor(
      geo.whiten(scaled - geo.center),
      stage="encoded spectrum",
      shape=(batch, int(geo.dest_idx.numel())))

  def decode(self, pred, params_whitened):
    """Network output -> physical C_ell (inverse of encode, divides f).

    Un-whiten and un-center to target' space, then divide the amplitude
    factor back out so the returned spectrum is physical C_ell.

    Arguments:
      pred            = (B, n_ell) whitened network output.
      params_whitened = (B, n_param) whitened inputs (set f per row).

    Returns:
      (B, n_ell) physical C_ell: (unwhiten(pred) + center) / f.
    """
    geo = self.geom
    f = self._factor(params_whitened)
    batch = int(f.shape[0])
    pred = _require_finite_cmb_tensor(
      pred,
      stage="model spectrum before inverse scaling",
      shape=(batch, int(geo.dest_idx.numel())))
    numerator = _require_finite_cmb_tensor(
      geo.unwhiten(pred) + geo.center,
      stage="physical spectrum before inverse scaling",
      shape=(batch, int(geo.dest_idx.numel())))
    return _require_finite_cmb_tensor(
      numerator / f,
      stage="physical spectrum after inverse scaling",
      shape=(batch, int(geo.dest_idx.numel())))

  def chi2(self, pred, target, params_whitened=None, full=False):
    """Per-sample chi2 in the PHYSICAL (factor-corrected) metric.

    encode multiplies the spectrum by the per-row amplitude factor f before
    whitening, so the whitened residual pred - target is f * (the physical
    whitened residual). A plain sum of its squares would therefore report
    f^2 * chi2_physical. Since f = f(A_s, tau) varies by cosmology,
    that biases delta-chi2, every threshold fraction, and best-epoch
    selection toward small-f cosmologies at a fixed physical error.
    Prediction and target for one row share the same f, so subtracting them
    leaves f in the residual and squaring produces f^2. This method
    DIVIDES the factor back out of the residual before summing, so the
    metric is the cosmic-variance chi2 of the physical spectra, independent
    of (A_s, tau) at a fixed physical error.

    Arguments:
      pred            = (B, n_ell) network output, whitened space.
      target          = (B, n_ell) whitened target (holds the factor).
      params_whitened = (B, n_param) whitened inputs; the amplitude factor
                        is read from them (REQUIRED because the metric cannot be
                        computed without it). The eval path passes it
                        explicitly; the loss reduction, which calls chi2
                        without params, reads the value loss stashed.
      full            = accepted for interface parity; ignored.

    Returns:
      (B,) per-sample physical chi2.
    """
    p = params_whitened if params_whitened is not None else self._params
    if p is None:
      raise ValueError(
        "CmbFactoredChi2.chi2 requires params_whitened: the amplitude "
        "factor f divides the whitened residual to recover the physical "
        "metric (a plain sum would report f^2 * chi2_physical), so the chi2 "
        "cannot be computed without the parameters. Pass params_whitened, "
        "or call through loss(), which stashes them.")
    r = (pred - target) / self._factor(p)
    return (r * r).sum(dim=1)

  def _penalty_residual(self, pred, target):
    """The factor-corrected whitened residual (pred - target) / f.

    The roughness penalty must see the PHYSICAL residual, not the
    f-scaled one. Without the division the penalty would carry f^2 like
    the uncorrected chi2 did, so it would depend on (A_s, tau) at a fixed
    physical roughness. Reads the parameters loss stashed (the
    roughness term runs only inside loss, after the stash).
    """
    return (pred - target) / self._factor(self._params)

  def loss(self, pred, target, params_whitened, *args, **kwargs):
    """Training loss: the inherited reduction, params stashed for the metric.

    The amplitude factor is NOT neutral in the metric: the chi2
    divides it back out of the residual, and the roughness penalty (when
    present) measures the same factor-corrected residual. loss stashes
    params_whitened so both consumers can read it: the chi2 that the reduction
    calls without params and _penalty_residual. It then forwards the reduction
    controls positionally.

    Arguments:
      pred            = (B, n_ell) network output.
      target          = (B, n_ell) whitened target.
      params_whitened = whitened inputs; stashed so the chi2 and the
                        roughness penalty read the amplitude factor. The
                        stash is PRIVATE to this reduction: it is cleared
                        the moment loss returns, so a public caller (a
                        diagnostic, a gate) that omits params gets the loud
                        refusal, never the last batch's stale factor (unit
                        70). Every public chi2 path passes its own params.
      *args, **kwargs = mode / trim / focus reduction controls, forwarded
                        to CmbDiagonalChi2.loss unchanged.

    Returns:
      a scalar loss tensor.
    """
    self._params = params_whitened
    try:
      return super().loss(pred, target, *args, **kwargs)
    finally:
      self._params = None


def make_cmb_chi2(geom, law, param_geometry=None, as_name=None,
                  tau_name=None, as_ref=None, tau_ref=None):
  """
  Build the CMB chi2fn (loss + geometry wrapper) for run_emulator.

  The CMB analogue of make_chi2 / make_scalar_chi2: it dispatches on the
  imposed amplitude law (persisted by name in the artifact). The "none" law
  needs only the geometry; the "as_exp2tau_ref" law also needs the param
  geometry, the two named input columns it reads A_s / tau from, and the
  fiducial reference pair (as_ref, tau_ref) that makes the factor order-one.
  The retired raw-factor law "as_exp2tau" is refused with the retrain
  instruction.

  Arguments:
    geom           = CmbDiagonalGeometry for the spectrum (its .state() is
                     what save_emulator persists; pass the geometry, not a
                     chi2fn).
    law            = the amplitude-law name, a key of AMPLITUDE_LAWS ("none"
                     or "as_exp2tau_ref").
    param_geometry = ParamGeometry whose decode maps whitened inputs to
                     physical params; required for "as_exp2tau_ref", unused
                     for "none".
    as_name        = the raw amplitude column name; required for
                     "as_exp2tau_ref".
    tau_name       = the tau column name; required for "as_exp2tau_ref".
    as_ref         = the fiducial linear amplitude A_s_ref (> 0); required
                     for "as_exp2tau_ref".
    tau_ref        = the fiducial optical depth tau_ref; required for
                     "as_exp2tau_ref".

  Returns:
    a CmbDiagonalChi2 ("none") or a configured CmbFactoredChi2
    ("as_exp2tau_ref") to pass to run_emulator as chi2fn.
  """
  reject_retired_amplitude_law(law)              # as_exp2tau -> retrain error
  if law not in AMPLITUDE_LAWS:
    raise ValueError(
      "unknown amplitude law " + repr(law) + "; the CMB law registry has "
      + repr(sorted(AMPLITUDE_LAWS)) + " (persisted by name in the "
      "artifact, never a default).")
  if law == "none":
    return CmbDiagonalChi2(geom=geom)
  # as_exp2tau_ref: the imposed order-one primary-amplitude scaling.
  missing = []
  for nm, val in (("param_geometry", param_geometry),
                  ("as_name", as_name),
                  ("tau_name", tau_name),
                  ("as_ref", as_ref),
                  ("tau_ref", tau_ref)):
    if val is None:
      missing.append(nm)
  if missing:
    raise ValueError(
      "amplitude law 'as_exp2tau_ref' needs " + repr(missing) + "; it reads "
      "A_s and tau from named input columns through the param geometry and "
      "measures them against the fiducial (as_ref, tau_ref) so f == 1 there.")
  chi2fn = CmbFactoredChi2(geom=geom)
  chi2fn.configure_law(param_geometry=param_geometry,
                       as_name=as_name,
                       tau_name=tau_name,
                       as_ref=as_ref,
                       tau_ref=tau_ref)
  return chi2fn
