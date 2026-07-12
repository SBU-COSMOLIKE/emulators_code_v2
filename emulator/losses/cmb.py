"""Chi2 losses and the imposed-amplitude-law registry for CMB spectra.

The CMB sibling of losses/core.py's CosmolikeChi2 and losses/scalar.py's
ScalarChi2: it wraps a CmbDiagonalGeometry and exposes the exact loop
interface (encode / decode / a per-sample chi2 / loss), so the shared
training machinery (trimming, the focal hardness weight, the sqrt / berhu
ladder, EMA, the L2-SP anchor) composes unchanged. Because the geometry
whitens each multipole by its cosmic-variance scale, the per-sample chi2
is the plain sum of squared whitened residuals, and that already is the
cosmic-variance chi2 (no covariance to contract, no mask to unsqueeze
through).

Two laws share this file. The "none" law (CmbDiagonalChi2) is the plain
diagonal chi2: the network learns the raw C_ell shape directly. The
"as_exp2tau" law (CmbFactoredChi2) imposes the primary amplitude scaling
rather than learning it: the target the network sees is the
amplitude-rescaled spectrum target' = C_ell * exp(2 tau) / A_s, and the
decode multiplies the law back so the emulator returns physical C_ell.
A_s and tau are read from named input columns (the factored-IA
philosophy: an exactly-known scaling is imposed in closed form, not
fit), so only the shape is learned and the amplitude generalizes for
free. AMPLITUDE_LAWS is the small registry (persisted by name in the
artifact, never a code default); make_cmb_chi2 builds the right loss.

PS: a whitened residual is (pred - target) after each multipole has been
put in units of its cosmic-variance error bar (see geometries.cmb's PS);
its square summed over the multipoles is the per-sample chi2 this loss
reduces. "The loop interface" is the small set of methods the training
loop calls on the loss object: encode (raw target -> network space),
decode (its inverse), chi2 (per-sample metric), and loss (the scalar
objective). needs_params (True only for as_exp2tau) is the flag the loop
reads to decide whether to hand the loss the whitened input params (the
law reads A_s / tau from them); it mirrors RescaledChi2's contract.
"""

import torch

from .core import CosmolikeChi2


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
# "as_exp2tau" imposes target' = C_ell * exp(2 tau) / A_s and needs the
# two named input columns it reads A_s and tau from.
AMPLITUDE_LAWS = {
  "none":       (),
  "as_exp2tau": ("as_name", "tau_name"),
}


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
    c = c + self._rough_lam * self._rough.per_sample(pred - target)
    return self._reduce(c=c, mode=mode, trim=trim, focus=focus,
                        focus_scale=focus_scale, berhu_knot=berhu_knot,
                        berhu_cap=berhu_cap, berhu_s=berhu_s)


class CmbFactoredChi2(CmbDiagonalChi2):
  """
  Diagonal chi2 with the imposed amplitude law (the "as_exp2tau" law).

  The network learns the SHAPE of the spectrum, not its primary
  amplitude: the target it sees is the amplitude-rescaled spectrum

      target' = C_ell * exp(2 tau) / A_s

  a per-row scalar factor f = exp(2 tau) / A_s multiplied into every
  multipole (A_s and tau are the same for all l of one cosmology). encode
  bakes f into the target before the geometry centers and whitens it;
  decode divides f back out, so the emulator returns physical C_ell. The
  chi2 stays the plain sum of squared whitened residuals (the factor is
  already in both pred and target, so it does not enter the metric),
  exactly as the legacy trainer computes the loss in the rescaled-target
  space with a fixed cosmic-variance weight.

  This mirrors RescaledChi2 (losses/core.py) with a per-row scalar factor
  in place of the per-element analytic R, and inheriting the plain chi2
  (like ResidualBaseChi2) rather than dividing the factor back out of the
  loss. needs_params = True, so the training loop hands encode / decode /
  chi2 / loss the whitened input params; _factor decodes them to physical
  through the param geometry and reads A_s / tau by column. Build by
  wrapping a geometry, CmbFactoredChi2(geom), then configure_law(...) to
  attach the param geometry and the A_s / tau column names.
  """

  # capability flag the loop reads (getattr default False elsewhere): this
  # loss's encode / decode / chi2 / loss take the whitened params, from
  # which the amplitude factor is built. Mirrors RescaledChi2.needs_params.
  needs_params = True
  # stash so the inherited loss reduction (which calls self.chi2 without
  # params) is reused unchanged; the plain chi2 ignores it, but the field
  # keeps the RescaledChi2 shape and documents the contract.
  _params = None

  def configure_law(self, param_geometry, as_name, tau_name):
    """Attach the amplitude-law state (call once, after wrapping geom).

    Resolves the A_s and tau column positions in the raw parameter
    vector by name off the param geometry, so _factor can read them from
    the decoded physical params. A name the geometry does not carry is a
    loud error naming it and the available columns.

    Arguments:
      param_geometry = ParamGeometry whose decode maps the whitened
                       model inputs back to physical params (what the
                       law reads); the same object passed to
                       run_emulator.
      as_name        = the raw linear amplitude column name (e.g.
                       "As"); the law divides the target by it. A run
                       that samples logA materializes a raw A_s column
                       in the generator, so the law always reads a
                       linear amplitude (the simpler ruling; see the
                       CME resume).
      tau_name       = the optical-depth column name (e.g. "tau"); the
                       law multiplies the target by exp(2 tau).

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
          + repr(names) + "; the as_exp2tau law reads A_s and tau from "
          "named input columns.")
    self.param_geometry = param_geometry
    self.as_name  = as_name
    self.tau_name = tau_name
    self.as_idx   = names.index(as_name)
    self.tau_idx  = names.index(tau_name)
    return self

  def _factor(self, params_whitened):
    """Per-row amplitude factor f = exp(2 tau) / A_s for whitened inputs.

    Decodes the whitened params to physical (the form the closed law
    reads), then reads A_s and tau by their resolved column positions.
    The factor is the same for every multipole of one cosmology, so it
    is returned as a column to broadcast across the multipole axis.

    Arguments:
      params_whitened = (B, n_param) whitened model inputs (the tensor
                        the model consumes; the loop / predictor pass it).

    Returns:
      f = (B, 1) amplitude factor on the params' device.
    """
    phys = self.param_geometry.decode(params_whitened)
    a_s  = phys[:, self.as_idx]
    tau  = phys[:, self.tau_idx]
    f    = torch.exp(2.0 * tau) / a_s
    return f.reshape(-1, 1)

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
    return geo.whiten(geo.squeeze(dv) * f - geo.center)

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
    return (geo.unwhiten(pred) + geo.center) / f

  def chi2(self, pred, target, params_whitened=None, full=False):
    """Plain sum-of-squares chi2: the factor is already in the target.

    Accepts params_whitened (the loop passes it for a needs_params loss)
    but ignores it: the amplitude factor is baked into both pred and
    target by encode, so it cancels in the residual and does not enter
    the metric. Delegates to CmbDiagonalChi2.chi2.

    Arguments:
      pred            = (B, n_ell) network output, whitened space.
      target          = (B, n_ell) whitened target (holds the factor).
      params_whitened = accepted for the needs_params call convention;
                        ignored here.
      full            = accepted for interface parity; ignored.

    Returns:
      (B,) per-sample chi2.
    """
    return CmbDiagonalChi2.chi2(self, pred=pred, target=target, full=full)

  def loss(self, pred, target, params_whitened, *args, **kwargs):
    """Training loss: the inherited reduction, params accepted and dropped.

    The amplitude factor is already baked into the encoded target, so the
    chi2 (and hence the reduction) needs no params; loss accepts
    params_whitened only to match the needs_params loop call, stashes it
    for interface parity with RescaledChi2, and forwards the reduction
    controls positionally.

    Arguments:
      pred            = (B, n_ell) network output.
      target          = (B, n_ell) whitened target.
      params_whitened = whitened inputs (stashed; not used by the plain
                        chi2).
      *args, **kwargs = mode / trim / focus reduction controls, forwarded
                        to CmbDiagonalChi2.loss unchanged.

    Returns:
      a scalar loss tensor.
    """
    self._params = params_whitened
    return super().loss(pred, target, *args, **kwargs)


def make_cmb_chi2(geom, law, param_geometry=None, as_name=None,
                  tau_name=None):
  """
  Build the CMB chi2fn (loss + geometry wrapper) for run_emulator.

  The CMB analogue of make_chi2 / make_scalar_chi2: it dispatches on the
  imposed amplitude law (persisted by name in the artifact). The "none"
  law needs only the geometry; the "as_exp2tau" law also needs the param
  geometry and the two named input columns it reads A_s and tau from.

  Arguments:
    geom           = CmbDiagonalGeometry for the spectrum (its .state()
                     is what save_emulator persists; pass the geometry,
                     not a chi2fn).
    law            = the amplitude-law name, a key of AMPLITUDE_LAWS
                     ("none" or "as_exp2tau").
    param_geometry = ParamGeometry whose decode maps whitened inputs to
                     physical params; required for "as_exp2tau" (the law
                     reads A_s / tau from it), unused for "none".
    as_name        = the raw amplitude column name; required for
                     "as_exp2tau".
    tau_name       = the tau column name; required for "as_exp2tau".

  Returns:
    a CmbDiagonalChi2 ("none") or a configured CmbFactoredChi2
    ("as_exp2tau") to pass to run_emulator as chi2fn.
  """
  if law not in AMPLITUDE_LAWS:
    raise ValueError(
      "unknown amplitude law " + repr(law) + "; the CME registry has "
      + repr(sorted(AMPLITUDE_LAWS)) + " (persisted by name in the "
      "artifact, never a default).")
  if law == "none":
    return CmbDiagonalChi2(geom=geom)
  # as_exp2tau: the imposed primary-amplitude scaling.
  missing = []
  for nm, val in (("param_geometry", param_geometry),
                  ("as_name", as_name),
                  ("tau_name", tau_name)):
    if val is None:
      missing.append(nm)
  if missing:
    raise ValueError(
      "amplitude law 'as_exp2tau' needs " + repr(missing) + "; it reads "
      "A_s and tau from named input columns through the param geometry.")
  chi2fn = CmbFactoredChi2(geom=geom)
  chi2fn.configure_law(param_geometry=param_geometry,
                       as_name=as_name,
                       tau_name=tau_name)
  return chi2fn
