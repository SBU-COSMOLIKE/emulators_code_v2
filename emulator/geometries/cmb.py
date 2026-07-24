"""Output geometry for CMB power-spectrum emulators (TT / TE / EE / pp).

This geometry maps one spectrum row on a stored multipole grid to the
standardized target predicted by the network. The covariance product supplies
one positive scale ``sigma_l`` for each multipole. The geometry subtracts the
training mean and divides by that scale. It performs no rotation and stores no
dense inverse covariance.

The meaning of ``sigma_l`` depends on the spectrum and on the covariance
producer. TT, TE, EE, and pp do not share one variance formula. The geometry
therefore treats the stored scale as an input fact rather than re-deriving it.
Dividing by ``sigma_l`` expresses a residual in units of the stored fiducial
error scale. It does not guarantee empirical unit variance, equal physical
importance, or uniform learning difficulty across multipoles.

The imposed amplitude law is owned by ``losses/cmb.py``. For a row-dependent
factor ``f``, the complete path is raw ``C_l`` -> ``f C_l`` -> subtract the
training mean -> divide by ``sigma_l`` -> network residual. The loss divides
the residual by ``f`` before reporting the physical diagonal score. Decoding
reverses the standardization and divides by ``f`` to return raw ``C_l``.

Every multipole is kept. ``dest_idx`` is ``arange(n_ell)`` and ``total_size``
is ``n_ell`` so the shared training loop can obtain the output width through
the same interface used by masked data-vector geometries.

    dv (B, n_ell)           one spectrum's raw C_ell, l = 2..ellmax
       │  squeeze            every l kept (dest_idx = arange, a copy)
       ▼
       │  - center           subtract the training-mean target
       ▼
       │  / sigma            divide by the per-l cosmic-variance scale
       ▼
    t  (B, n_ell)           whitened target the network predicts

(legend: B = batch rows; n_ell = number of multipoles l = 2..ellmax, the
width the network emits; center / sigma = per-multipole training mean and
cosmic-variance scale. decode runs the arrows bottom-up, exactly
inverting each step. dest_idx = arange(n_ell) and total_size = n_ell give
the loop the same output-width surface the data-vector geometries expose
through their mask, with no CMB-specific branch. When an amplitude law is
in force the dv fed here is the amplitude-rescaled target the chi2 wrapper
built; for the "none" law it is the raw C_ell.)
"""

import numpy as np
import torch


class CmbDiagonalGeometry:
  """
  Diagonal cosmic-variance geometry for one CMB spectrum.

  Whitening (see module PS) is applied per multipole, so the network
  sees unit-cosmic-variance targets instead of C_ell that span orders of
  magnitude across the acoustic peaks and the damping tail. One instance
  owns the encode/decode between a spectrum's C_ell vector and the
  whitened target. It holds:

    - spectrum: which spectrum this is ("tt" / "te" / "ee" / "pp").
    - ell: the multipole grid l = 2..ellmax (integer, ascending).
    - center: the per-multipole training mean (the target zero-point).
    - sigma: the positive per-multipole scale read from the covariance
      product. The precise variance expression belongs to that producer and
      differs among TT, TE, EE, and pp. The plain sum of squared standardized
      residuals is the diagonal score defined by these stored scales.
    - fiducial_cl: the stored fiducial C_ell the cosmic-variance
      diagonal was built from (persisted so the units convention and
      the diagonal are reproducible, never a default).
    - units: the C_ell units convention ("muK2" for T/E temperature and
      polarization, "dimensionless" for the lensing pp), read by the
      cobaya adapter when it assembles the C_ell dict.
    - law / as_name / tau_name: the imposed amplitude law by NAME plus
      the two parameter columns it reads (empty strings for "none") —
      persisted HERE because the artifact records its law; the
      chi2 wrapper in losses/cmb.py executes it.
    - as_ref / tau_ref: the fiducial (A_s_ref, tau_ref) the order-one
      "as_exp2tau_ref" law measures the sampled (A_s, tau) against, so
      the amplitude factor is 1 at the fiducial. Resolved floats
      persisted with the artifact (None for "none"); the geometry only
      carries them — the loss reads them through make_cmb_chi2.

  Build from an analytic fiducial C_ell + the training-mean target at
  training time (from_fiducial) or from saved tensors at inference time
  (from_state); the transform travels with the weights, exactly like the
  parameter and data-vector geometries. There is no mask / rotation /
  probe: dest_idx and total_size are the trivial identity (arange(n_ell),
  n_ell), present only so the training loop's output-width surface is
  unchanged.
  """

  def __init__(self,
               device,
               spectrum,
               ell,
               center,
               sigma,
               fiducial_cl,
               units,
               law,
               as_name,
               tau_name,
               as_ref=None,
               tau_ref=None):
    """Place the diagonal geometry tensors on the device.

    Plain constructor: stores fields only; from_fiducial builds them
    from a fiducial C_ell and from_state from a saved dict. as_tensor
    accepts numpy (training) or cpu tensors (a saved state), so both
    paths share this code. dest_idx and total_size are derived from the
    multipole count (an identity, not a stored knob), so from_state need
    not persist them.

    Arguments:
      device      = device the tensors live on.
      spectrum    = spectrum name ("tt" / "te" / "ee" / "pp"); the
                    cobaya adapter keys its C_ell dict by it.
      ell         = (n_ell,) integer multipole grid l = 2..ellmax,
                    ascending; stored long, kept for the adapter's
                    C_ell assembly (the l array cobaya expects).
      center      = (n_ell,) per-multipole training mean of the target
                    (the amplitude-rescaled C_ell when a law is in
                    force, the raw C_ell for the "none" law).
      sigma       = (n_ell,) per-multipole cosmic-variance scale, the
                    whitening unit (strictly positive).
      fiducial_cl = (n_ell,) fiducial C_ell the cosmic-variance
                    diagonal was built from (persisted resolved).
      units       = C_ell units string ("muK2" / "dimensionless").
      law         = the imposed amplitude-law name (a key of
                    losses/cmb.py's AMPLITUDE_LAWS registry:
                    "none" or "as_exp2tau_ref"), persisted here because
                    the LAW is an artifact fact; the chi2
                    wrapper (losses/cmb.py) is its executor.
      as_name     = the raw linear amplitude column name the law
                    reads ("" for the "none" law).
      tau_name    = the optical-depth column name ("" for "none").
      as_ref      = the fiducial linear amplitude A_s_ref the order-one
                    "as_exp2tau_ref" law measures A_s against (a resolved
                    float, persisted; None for "none", which has no
                    reference). The law's chi2 wrapper reads it through
                    make_cmb_chi2 at build / rebuild.
      tau_ref     = the fiducial optical depth tau_ref (a resolved float;
                    None for "none").
    """
    self.spectrum = str(spectrum)
    self.units    = str(units)
    self.law      = str(law)
    self.as_name  = str(as_name)
    self.tau_name = str(tau_name)
    # the order-one law's fiducial reference pair: a resolved float when
    # the law carries one, None for the "none" law (no reference). Stored
    # here only so it PERSISTS with the artifact (state / from_state); this
    # geometry never uses it — the loss's _factor reads it via make_cmb_chi2.
    self.as_ref  = None if as_ref  is None else float(as_ref)
    self.tau_ref = None if tau_ref is None else float(tau_ref)
    self.ell = torch.as_tensor(ell,
                               dtype=torch.long,
                               device=device)
    self.center = torch.as_tensor(center,
                                  dtype=torch.float32,
                                  device=device)
    self.sigma = torch.as_tensor(sigma,
                                 dtype=torch.float32,
                                 device=device)
    # sigma is the whitening unit: whiten divides by it, so a zero sigma
    # makes every whitened target infinite and a non-finite sigma makes
    # it not-a-number, silently poisoning the run until the finite
    # contract aborts with no multipole named. Both real build paths
    # (from_fiducial's fixture route and the production covariance .npz
    # handed straight to this constructor) meet here, so this is the one
    # place that catches a zero-variance or corrupt sigma at build time.
    if not bool(torch.isfinite(self.sigma).all()):
      raise ValueError(
        "CmbDiagonalGeometry: sigma (the per-multipole cosmic-variance "
        "whitening scale) contains non-finite values. This is a corrupt "
        "covariance; repair or regenerate it before building the geometry.")
    if not bool((self.sigma > 0.0).all()):
      raise ValueError(
        "CmbDiagonalGeometry: sigma (the per-multipole cosmic-variance "
        "whitening scale) must be strictly positive everywhere; got a "
        "zero or negative entry. A zero-variance multipole would make the "
        "whitening divide by zero. Check the covariance .npz.")
    self.fiducial_cl = torch.as_tensor(fiducial_cl,
                                       dtype=torch.float32,
                                       device=device)
    # output-width surface for the training loop: with no mask every
    # multipole is kept, so dest_idx is the identity arange and
    # total_size the multipole count. Derived from ell (not persisted);
    # the loop sizes the model by dest_idx.numel(), so this reuses the
    # data-vector sizing path with no CMB branch.
    n_ell = int(self.ell.numel())
    self.total_size = n_ell
    self.dest_idx   = torch.arange(n_ell, device=device)

  @classmethod
  def from_fiducial(cls,
                    device,
                    spectrum,
                    ell,
                    fiducial_cl,
                    center,
                    units,
                    law,
                    as_name="",
                    tau_name="",
                    as_ref=None,
                    tau_ref=None):
    """Build the diagonal geometry from an analytic fiducial C_ell.

    The cosmic-variance precision is the diagonal (the covinv RULING,
    Motloch & Hu eq 3 at zero noise)

        cinv_l = (2l+1) / (2 * C_fid_l^2),   l = 2..ellmax

    and the whitening scale is its inverse root sigma_l = 1/sqrt(cinv_l)
    = C_fid_l * sqrt(2/(2l+1)), so whiten and the reported cinv are
    inverse-consistent (the plain sum-of-squares chi2 equals the
    cosmic-variance chi2). center is passed in (the training-mean
    target), not derived from the fiducial, because the target is the
    amplitude-rescaled C_ell whose mean only the training set knows.

    A non-positive fiducial C_ell entry is a loud error naming the
    multipole: sigma would be non-finite and decode would divide by it.
    (TE can legitimately cross zero; a TE run therefore needs a
    non-vanishing fiducial convention or the "none" law over |C_ell|,
    a recorded choice for the generator, not a silent divide.)

    Arguments:
      device      = device for the built tensors.
      spectrum    = spectrum name ("tt" / "te" / "ee" / "pp").
      ell         = (n_ell,) integer multipole grid l = 2..ellmax.
      fiducial_cl = (n_ell,) fiducial C_ell over ell (strictly
                    positive), the cosmic-variance diagonal's source.
      center      = (n_ell,) per-multipole training-mean target.
      units       = C_ell units string ("muK2" / "dimensionless").
      law         = the amplitude-law name ("none" / "as_exp2tau_ref";
                    an artifact fact, never defaulted by a consumer).
      as_name     = the amplitude column the law reads ("" for
                    "none").
      tau_name    = the tau column the law reads ("" for "none").
      as_ref      = the fiducial linear amplitude A_s_ref the
                    "as_exp2tau_ref" law measures A_s against (a resolved
                    float; None for "none").
      tau_ref     = the fiducial optical depth tau_ref (None for "none").

    Returns:
      a CmbDiagonalGeometry whose whiten scales by the cosmic variance.
    """
    ell_f = np.asarray(ell, dtype="float64")
    cl    = np.asarray(fiducial_cl, dtype="float64")
    if ell_f.shape != cl.shape:
      raise ValueError(
        "CmbDiagonalGeometry.from_fiducial: ell and fiducial_cl must "
        "have the same shape, got " + repr(ell_f.shape) + " and "
        + repr(cl.shape))
    bad = np.nonzero(~(cl > 0.0))[0]
    if bad.size > 0:
      show = ell_f[bad][:8].astype("int64").tolist()
      raise ValueError(
        "CmbDiagonalGeometry.from_fiducial: fiducial_cl must be "
        "strictly positive (the cosmic-variance scale sigma_l = "
        "C_fid_l * sqrt(2/(2l+1)) divides decode); non-positive at "
        "multipole(s) " + repr(show) + " (first 8). A spectrum that "
        "crosses zero (e.g. TE) needs a non-vanishing fiducial "
        "convention or the 'none' amplitude law, set in the generator.")
    # cosmic-variance diagonal precision (Motloch & Hu
    # 1709.03599 eq 3, the Gaussian part at zero noise; see the covinv
    # passage in families-scalar-cmb.md): the variance is 2/(2l+1) * C_fid_l^2,
    # so the precision is its inverse,
    #   cinv_l = (2l+1) / (2 * C_fid_l^2),
    # and sigma_l = 1/sqrt(cinv_l) = C_fid_l * sqrt(2/(2l+1)) — the
    # per-l error bar, DECREASING with l as more modes average down.
    # (The real training path takes
    # sigma from the compute_cmb_covariance.py .npz — WITH the eq-4
    # noise — through __init__ directly, so this classmethod is the
    # noise-free fixture / synthetic-gate form.)
    cinv  = (2.0 * ell_f + 1.0) / (2.0 * cl ** 2)
    sigma = 1.0 / np.sqrt(cinv)
    return cls(device=device,
               spectrum=spectrum,
               ell=ell_f.astype("int64"),
               center=center,
               sigma=sigma,
               fiducial_cl=cl,
               units=units,
               law=law,
               as_name=as_name,
               tau_name=tau_name,
               as_ref=as_ref,
               tau_ref=tau_ref)

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference / h5-rebuild path).

    state's keys match __init__ (spectrum / ell / center / sigma /
    fiducial_cl / units / law / as_name / tau_name, plus as_ref / tau_ref
    for the order-one law), so cls(device, **state) reconstructs the
    transform with no fiducial reread. cls (not the class name) keeps a
    subclass correct.

    This is the artifact READ boundary, so it enforces the amplitude-law
    rules: a persisted retired law is refused with its retrain
    instruction, and an "as_exp2tau_ref" artifact missing either persisted
    reference is refused (the never-trust-defaults proof — the loss reads
    the reference with no code fallback, so a file lacking it must not
    rebuild).

    Arguments:
      device = device to place the rebuilt tensors on.
      state  = dict from state(), splatted into __init__.

    Returns:
      a CmbDiagonalGeometry (or subclass, via cls).

    Raises:
      ValueError on a persisted retired law, or an "as_exp2tau_ref" state
      missing as_ref / tau_ref.
    """
    kwargs = dict(state)
    # strings ride some h5 round-trips as 1-element lists; normalize each
    # back (the sibling geometries' scalar-field pattern).
    for key in ("spectrum", "units", "law", "as_name", "tau_name"):
      val = kwargs.get(key)
      if isinstance(val, (list, tuple)):
        kwargs[key] = val[0]
    # the fiducial reference pair rides as 0-d float64 tensors (state()
    # writes them only for the order-one law); read each back to a Python
    # float so __init__ stores a scalar, exactly the grid offset pattern.
    for key in ("as_ref", "tau_ref"):
      val = kwargs.get(key)
      if isinstance(val, torch.Tensor):
        kwargs[key] = float(val.reshape(-1)[0])
    law = str(kwargs.get("law", ""))
    # the retired raw-factor law is refused at rebuild with its retrain
    # instruction: one message source, the loss registry's adjudicator
    # (imported lazily so this module stays importable torch-light).
    from ..losses.cmb import reject_retired_amplitude_law
    reject_retired_amplitude_law(law)
    # the order-one law reads its persisted reference with NO fallback, so a
    # rebuilt state missing either half is refused rather than silently
    # defaulted (the artifact must record the numbers).
    if law == "as_exp2tau_ref":
      for key in ("as_ref", "tau_ref"):
        if kwargs.get(key) is None:
          raise ValueError(
            "CmbDiagonalGeometry.from_state: the 'as_exp2tau_ref' law needs "
            "the persisted fiducial " + repr(key) + ", but the rebuilt state "
            "lacks it. The amplitude factor reads the reference with no code "
            "default, so a file missing it is refused — re-save the run (or "
            "retrain) so the artifact records the number.")
    return cls(device, **kwargs)

  def state(self):
    """Collect the persistable transform, keys matching __init__.

    dest_idx / total_size are derived from ell, so they are not
    persisted. ell rides as a long tensor, spectrum / units as strings;
    the h5 writer handles each. The fiducial reference pair rides as 0-d
    float64 tensors, written ONLY for the order-one law that carries one
    (the "none" law has no reference, so its artifact records none --
    resolved values, never a placeholder).

    Returns:
      the mapping from_state(device, state()) rebuilds the identical
      geometry from.
    """
    st = {"spectrum":    self.spectrum,
          "ell":         self.ell.cpu(),
          "center":      self.center.cpu(),
          "sigma":       self.sigma.cpu(),
          "fiducial_cl": self.fiducial_cl.cpu(),
          "units":       self.units,
          "law":         self.law,
          "as_name":     self.as_name,
          "tau_name":    self.tau_name}
    if self.as_ref is not None:
      st["as_ref"] = torch.tensor(self.as_ref, dtype=torch.float64)
    if self.tau_ref is not None:
      st["tau_ref"] = torch.tensor(self.tau_ref, dtype=torch.float64)
    return st

  def attach_head_coords(self):
    """Attach the conv/TRF heads' channel/token split.

    The correction heads (designs/plain.py ResCNN / ResTRF) read
    geom.bin_sizes for their channel/token layout — the cosmolike
    geometry gets it from build_shear_angle_map; here it is a pure
    derivation from the geometry's own ell grid: ONE bin covering the
    whole spectrum, coordinate = ell (the conv slides along ell; the
    TRF re-segments via model.trf.n_tokens so attention has windows
    to attend across). There is no permutation and no basis change:
    the whitening is per multipole IN ell order, so the heads' W_fd /
    W_df maps stay None. Idempotent; no files, no torch build — safe
    at training (build_geometry) and at rebuild (rebuild_emulator).
    """
    width = int(self.ell.numel())
    self.bin_sizes = [width]
    self.head_pad_idx = torch.arange(
      width, dtype=torch.long, device=self.ell.device)
    self.head_valid_mask = torch.ones(
      (1, width), dtype=torch.bool, device=self.ell.device)

  # --- low-level transforms (all multipoles kept, so squeeze /
  #     unsqueeze are the identity gather / scatter) ---
  def squeeze(self, dv):
    """Keep every multipole (identity gather, a copy).

    Present for interface parity with the data-vector geometries; with
    dest_idx = arange(n_ell) this returns dv unchanged (a copy), so the
    amplitude-law chi2 wrapper can compose squeeze / whiten / center
    exactly as it does for a masked geometry.

    Arguments:
      dv = (B, n_ell) one spectrum's C_ell rows.

    Returns:
      (B, n_ell) the same entries, in ell order.
    """
    return dv[:, self.dest_idx]

  def unsqueeze(self, sq):
    """Scatter to the full length (identity, every multipole kept).

    Inverse of squeeze; with total_size = n_ell and dest_idx the full
    arange this places every entry back and masks nothing.

    Arguments:
      sq = (B, n_ell) kept entries (squeeze output).

    Returns:
      (B, n_ell) full vector (identical, no masked slots).
    """
    full = torch.zeros(sq.shape[0],
                       self.total_size,
                       dtype=sq.dtype,
                       device=sq.device)
    full[:, self.dest_idx] = sq
    return full

  def whiten(self, centered_sq):
    """Centered target -> whitened target: divide by sigma_l.

    Arguments:
      centered_sq = (B, n_ell) target with the center already subtracted.

    Returns:
      (B, n_ell) per-multipole unit-cosmic-variance target, float32.
    """
    return (centered_sq.to(torch.float32) / self.sigma)

  def unwhiten(self, whitened_sq):
    """Exact inverse of whiten: multiply each multipole by sigma_l.

    Arguments:
      whitened_sq = (B, n_ell) whitened target.

    Returns:
      (B, n_ell) un-whitened (centered) target, float32.
    """
    return whitened_sq.to(torch.float32) * self.sigma

  # --- high-level: target <-> network space ---
  def encode(self, dv):
    """Raw target -> network target: squeeze, center, whiten.

    The plain (amplitude-law "none") encode; the amplitude-law chi2
    wrapper overrides its own encode to multiply the law's per-row
    factor into squeeze(dv) before this centering.

    Arguments:
      dv = (B, n_ell) one spectrum's target rows.

    Returns:
      (B, n_ell) whitened target the network predicts.
    """
    return self.whiten(self.squeeze(dv) - self.center)

  def decode(self, whitened_sq):
    """Network target -> physical target: unwhiten, add the center.

    Arguments:
      whitened_sq = (B, n_ell) whitened network output.

    Returns:
      (B, n_ell) physical (kept-entry) target, before any amplitude law
      is multiplied back (the chi2 wrapper's decode does that).
    """
    return self.unwhiten(whitened_sq) + self.center
