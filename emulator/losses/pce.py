"""NPCE losses: a frozen PCE base plus a neural refiner.

The polynomial-chaos member of the emulator/losses/ family (the former
emulator PCE subpackage), paired with the base in designs/pce.py.
All classes wrap a fitted, frozen PCEEmulator as the base prediction
under a trainable refiner network. On the cosmolike (dense-covariance)
family the two combine forms are PCEResidualChi2, additive in the
whitened basis (the refiner learns truth minus base), and PCERatioChi2,
multiplicative in the physical basis (the refiner learns a fractional
correction). On the elementwise-whitened families (cmb / grid / grid2d
/ scalar) the residual form is PCEResidualDiagChi2, the same additive
algebra under the diagonal metric (2026-07-12 ruling: the PCE trunk
rides every family — arXiv 2404.12344 runs an NPCE on the MPS boost,
and EuclidEmulator2 is a PCE). Either way the chi2 stays the family's
own metric; the base moves only the target's zero point or scale,
never the metric.

All declare needs_params = True (encode / decode evaluate the base
from the whitened parameters), the capability flag the loaders
(batching.py), the training loop's compiled forward+loss and eval
twins (training.py), and the diagnostics already branch on; the
trim / focus / focus_scale reduction is inherited from CosmolikeChi2
(_reduce), so these losses ride every loss-side update for free.

PS: frozen = evaluated under no_grad, never trained; base = the
closed-form PCE prediction; refiner = the SGD-trained network (any
model spec) correcting it; Mahalanobis distance = r^T Cinv r, the
covariance-weighted squared residual; whitened = rotated into the
covariance eigenbasis and scaled to unit variance (decorrelated);
encoded = a dv put through the geometry's encode (kept entries,
centered, whitened); squeeze = keep only the unmasked dv entries; a
loader is a closure load(rows) -> a ready-to-train batch on the
device.
"""

import torch

from .cmb import CmbDiagonalChi2
from .core import CosmolikeChi2


class PCEResidualChi2(CosmolikeChi2):
  """
  NPCE integration: the refiner model learns the residual of the
  full whitened dv after a frozen PCE base, the chi2 staying
  plain. Mirrors ResidualBaseChi2, with the PCE base in place of
  the analytic center/R baseline.

  Encode (target construction, at load time):

      dv  (B, total_size)        raw data vector
         │  geom.encode          squeeze -> center -> whiten
         ▼
      t  (B, n_keep)             whitened truth
         │  - PCE(theta)         the frozen base (no_grad)
         ▼
      target (B, n_keep)         the refiner's residual target

  Loss: plain chi2 on (pred - target), the base cancels in the
  residual, (base + pred) - truth == pred - target, so the metric
  is exact. Decode inverts: geom.decode(y + PCE(theta)).

      (legend: B = batch rows; total_size = full 3x2pt dv length;
       n_keep = kept entries the model emulates; theta = the
       whitened parameters, params_whitened; PCE = the frozen
       PCEEmulator, evaluated in the whitened basis.)

  The refiner is any model spec (ResMLP, ResCNN, ...) trained by
  run_emulator with the robust chi2 loss. It outputs the full dv
  correction, so it is not confined to the PCE's K-mode subspace,
  a too-small K only costs a smaller head start, never caps
  accuracy (the conservative high-T property). For a ResCNN
  refiner, pass a DiagonalGeometry geom (theta order), as in the
  standalone ResCNN run.

  needs_params = True: encode/decode take the whitened params
  (the model inputs) to evaluate the frozen PCE base.
  """
  needs_params = True

  def __init__(self, geom, pce):
    """Hold the output geometry and the frozen PCE base.

    Arguments:
      geom = the output DataVectorGeometry (whitening + chi2).
      pce  = the fitted, frozen PCEEmulator; evaluated in the
             whitened basis as the base prediction.
    """
    super().__init__(geom)
    self.pce = pce          # frozen PCE base (whitened dv)

  def _base(self, params_whitened):
    """Evaluate the frozen PCE base under no_grad.

    Arguments:
      params_whitened = (B, n_param) whitened model inputs.

    Returns:
      (B, n_keep) whitened base prediction; no grad flows into it.
    """
    with torch.no_grad():
      return self.pce(params_whitened)

  def encode(self, dv, params_whitened):
    """Raw dv -> the refiner's residual target.

    Arguments:
      dv              = (B, total_size) raw data vectors.
      params_whitened = (B, n_param) whitened inputs (for the base).

    Returns:
      (B, n_keep) whitened truth minus the PCE base = the residual
      the refiner learns.
    """
    return self.geom.encode(dv) - self._base(params_whitened)

  def decode(self, y, params_whitened):
    """Refiner output -> physical dv (inverse of encode).

    Arguments:
      y               = (B, n_keep) refiner output (the residual).
      params_whitened = (B, n_param) whitened inputs (for the base).

    Returns:
      (B, n_keep) physical (kept-entry) dv: geom.decode(y + base).
    """
    return self.geom.decode(y + self._base(params_whitened))

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    """Per-sample chi2 of a residual prediction against its target.

    The base is baked into target at encode time, so pred - target
    equals full_pred - truth and the plain chi2 is exact.

    Arguments:
      pred            = (B, n_keep) refiner output.
      target          = (B, n_keep) residual target from encode.
      params_whitened = accepted for the needs_params signature,
                        unused here (the base is already in target).
      full            = if True, contract the full-length precision
                        (diagnostics); else the kept sub-block.

    Returns:
      (B,) per-sample chi2.
    """
    return CosmolikeChi2.chi2(self, pred=pred, target=target, full=full)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    """Training loss: the base CosmolikeChi2 reduction on the residual.

    Arguments:
      pred            = (B, n_keep) refiner output.
      target          = (B, n_keep) residual target.
      params_whitened = the needs_params argument, unused (the plain
                        chi2 needs no params).
      *args, **kwargs = the mode / trim / focus reduction controls,
                        forwarded positionally (never keyword pred /
                        target before the *args forwarder).

    Returns:
      a scalar loss tensor.
    """
    return CosmolikeChi2.loss(
      self, pred, target, *args, **kwargs)


class PCERatioChi2(CosmolikeChi2):
  """
  Multiplicative ("1 + delta") NPCE: pred = b * (1 + delta) in
  physical (squeezed) dv space, b = geom.decode(PCE(theta)) the
  frozen base, delta the model output (fractional correction).
  Division-free; the chi2 is on (pred - truth) directly.

  Speed design: the frozen base is precomputed once at load time
  and packed with the truth into the encoded target, so the chi2
  never re-runs the PCE in the training loop, it just unpacks
  and forms b * (1 + delta). The loader stages the wider target
  via the target_dim attribute (batching.py reads it).

  Encode (at load, once per row):

      dv  (B, total_size)
         │  geom.squeeze         physical kept entries
         ▼
      xi  (B, n_keep)        b  (B, n_keep) = geom.decode(
         │                      │              PCE(theta))
         │                      │  the frozen base, physical
         └───────── cat ────────┘
         ▼
      target = [b ; xi]  (B, 2*n_keep)

  chi2 (hot path, per batch):

      target (B, 2*n_keep)
         │  unpack              b = target[:, :n_keep]
         │                      xi = target[:, n_keep:]
         ▼
      r = b*(1 + pred) - xi     pred = the fractional correction
         │  einsum r^T Cinv_sq r
         ▼
      c  (B,)                   per-sample chi2, no PCE recompute

      (legend: B = batch rows; total_size = full 3x2pt dv length;
       n_keep = kept entries; theta = the whitened parameters,
       params_whitened; Cinv_sq = the kept x kept sub-block of the
       inverse covariance.)

  Trade-offs vs additive: target is not whitened; where b ~ 0
  (xi+/- zero crossings) the refiner has little leverage. Use a
  smooth, low-order PCE base.

  needs_params = True (encode/decode evaluate the base).
  """
  needs_params = True

  def __init__(self, geom, pce):
    """Hold the output geometry and the frozen PCE base.

    Arguments:
      geom = the output DataVectorGeometry (squeeze / decode + chi2).
      pce  = the fitted, frozen PCEEmulator; its whitened output is
             decoded to the physical base b.
    """
    super().__init__(geom)
    self.pce = pce

  @property
  def target_dim(self):
    """Staged-target width: encode packs [base ; truth].

    Returns:
      2 * n_keep, so batching.py stages a target twice the kept
      vector width (the base cached beside the truth).
    """
    return 2 * self.geom.dest_idx.numel()

  def _base_phys(self, params_whitened):
    """Frozen physical base b = geom.decode(PCE(theta)).

    Arguments:
      params_whitened = (B, n_param) whitened model inputs.

    Returns:
      (B, n_keep) physical base, under no_grad.
    """
    with torch.no_grad():
      return self.geom.decode(self.pce(params_whitened))

  def encode(self, dv, params_whitened):
    """Raw dv -> the packed [base ; physical truth] target.

    Precomputes the frozen base once per row at load, so the hot
    chi2 never re-runs the PCE (see Speed in the class docstring).

    Arguments:
      dv              = (B, total_size) raw data vectors.
      params_whitened = (B, n_param) whitened inputs (for the base).

    Returns:
      (B, 2*n_keep) target = [b ; xi] (physical base, physical
      kept truth).
    """
    b  = self._base_phys(params_whitened)
    xi = self.geom.squeeze(dv).float()
    return torch.cat([b, xi], dim=1)         # (B, 2*n_keep)

  def decode(self, pred, params_whitened):
    """Fractional correction -> physical dv (diagnostics only).

    Arguments:
      pred            = (B, n_keep) fractional correction delta.
      params_whitened = (B, n_param) whitened inputs (for the base).

    Returns:
      (B, n_keep) physical prediction b * (1 + delta). Not hot, so
      recomputing the base here is fine.
    """
    b = self._base_phys(params_whitened)
    return b * (1.0 + pred)

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    """Per-sample chi2 of b*(1 + pred) against the truth.

    Unpacks the base and truth cached in target, so the PCE is
    never recomputed in the loop.

    Arguments:
      pred            = (B, n_keep) fractional correction delta.
      target          = (B, 2*n_keep) packed [b ; xi] from encode.
      params_whitened = accepted for the needs_params signature,
                        unused (the base is already in target).
      full            = if True, contract the full-length precision
                        (diagnostics); else the kept sub-block.

    Returns:
      (B,) per-sample chi2 of r = b*(1 + pred) - xi.
    """
    nk  = self.geom.dest_idx.numel()
    b   = target[:, :nk]
    xi  = target[:, nk:]
    geo = self.geom
    r = b * (1.0 + pred) - xi
    # masked Mahalanobis (as in the base chi2) on the residual
    # r = b*(1+pred)-xi; unsqueeze to full length when full.
    if full:
      rf = geo.unsqueeze(r)
      return torch.einsum("bi,ij,bj->b", rf, geo.Cinv, rf)
    return torch.einsum("bi,ij,bj->b", r, geo.Cinv_sq, r)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    """Training loss: the base CosmolikeChi2 reduction.

    Arguments:
      pred            = (B, n_keep) fractional correction.
      target          = (B, 2*n_keep) packed [b ; xi].
      params_whitened = the needs_params argument, unused (the base
                        is already in target).
      *args, **kwargs = the mode / trim / focus reduction controls,
                        forwarded positionally (never keyword pred /
                        target before the *args forwarder).

    Returns:
      a scalar loss tensor.
    """
    return CosmolikeChi2.loss(self, pred, target,
                              *args, **kwargs)


class PCEResidualDiagChi2(CmbDiagonalChi2):
  """
  Residual NPCE under the diagonal metric: the family-wide form
  (2026-07-12 ruling) for every elementwise-whitened geometry —
  CmbDiagonalGeometry (law "none"), GridGeometry, Grid2DGeometry,
  ScalarGeometry. The refiner model learns the residual of the
  whitened target after a frozen PCE base, exactly as
  PCEResidualChi2 does on the cosmolike family; only the metric
  differs, and it is inherited: these families whiten per element,
  so the per-sample chi2 is CmbDiagonalChi2's plain sum of squared
  whitened residuals, no covariance to contract.

  Encode (target construction, at load time):

      dv  (B, n_out)             raw target row (C_ell / grid rows /
         │  geom.encode           law-space rows / named outputs)
         ▼
      t  (B, n_out)              whitened truth
         │  - PCE(theta)         the frozen base (no_grad)
         ▼
      target (B, n_out)          the refiner's residual target

  Loss: the diagonal chi2 on (pred - target); the base cancels in
  the residual, (base + pred) - truth == pred - target, so the
  metric is exact. Decode inverts: geom.decode(y + PCE(theta)), so
  a grid law (log space) or a D-MP9 constant pin applies to the
  COMBINED prediction, one definition.

      (legend: B = batch rows; n_out = the family's output length,
       n_ell / nz / nz*nk / n_named; theta = the whitened
       parameters, params_whitened; PCE = the frozen PCEEmulator,
       fitted and evaluated in the whitened basis.)

  Two deliberate boundaries of the family-wide form:
    - Residual only. The ratio form is a dense-covariance concept
      (a fractional correction where whitening mixes elements); on
      an elementwise whitening the residual form already gives the
      refiner per-element leverage, and on the log-law grids a
      whitened residual IS a multiplicative correction in linear
      space. validate_pce rejects form "ratio" for these families.
    - CMB amplitude law "none" only. The as_exp2tau law's loss owns
      the target construction (CmbFactoredChi2), the same
      one-at-a-time exclusivity as pce vs rescale / model.ia;
      validate_cmb rejects the combination.

  The D-CM8 roughness penalty composes unchanged (inherited
  configure_roughness / loss): pred - target here equals the FULL
  whitened residual (base + net - truth), the exact quantity the
  penalty is defined on.

  needs_params = True: encode/decode take the whitened params
  (the model inputs) to evaluate the frozen PCE base.
  """
  needs_params = True

  def __init__(self, geom, pce):
    """Hold the output geometry and the frozen PCE base.

    Arguments:
      geom = the family's elementwise-whitened output geometry
             (encode / decode; the chi2 is the plain whitened L2).
      pce  = the fitted, frozen PCEEmulator; evaluated in the
             whitened basis as the base prediction.
    """
    super().__init__(geom)
    self.pce = pce          # frozen PCE base (whitened target)

  def _base(self, params_whitened):
    """Evaluate the frozen PCE base under no_grad.

    Arguments:
      params_whitened = (B, n_param) whitened model inputs.

    Returns:
      (B, n_out) whitened base prediction; no grad flows into it.
    """
    with torch.no_grad():
      return self.pce(params_whitened)

  def encode(self, dv, params_whitened):
    """Raw target row -> the refiner's residual target.

    Arguments:
      dv              = (B, n_out) raw target rows.
      params_whitened = (B, n_param) whitened inputs (for the base).

    Returns:
      (B, n_out) whitened truth minus the PCE base = the residual
      the refiner learns.
    """
    return self.geom.encode(dv) - self._base(params_whitened)

  def decode(self, y, params_whitened):
    """Refiner output -> physical target row (inverse of encode).

    Arguments:
      y               = (B, n_out) refiner output (the residual).
      params_whitened = (B, n_param) whitened inputs (for the base).

    Returns:
      (B, n_out) physical row: geom.decode(y + base), so the
      geometry's law inverse / constant pins act on the combined
      prediction.
    """
    return self.geom.decode(y + self._base(params_whitened))

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    """Per-sample chi2 of a residual prediction against its target.

    The base is baked into target at encode time, so pred - target
    equals full_pred - truth and the diagonal chi2 is exact.

    Arguments:
      pred            = (B, n_out) refiner output.
      target          = (B, n_out) residual target from encode.
      params_whitened = accepted for the needs_params signature,
                        unused here (the base is already in target).
      full            = accepted for interface parity; ignored (no
                        mask / covariance on the diagonal families).

    Returns:
      (B,) per-sample chi2.
    """
    return CmbDiagonalChi2.chi2(self, pred=pred, target=target)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    """Training loss: the diagonal reduction (+ roughness when set).

    Arguments:
      pred            = (B, n_out) refiner output.
      target          = (B, n_out) residual target.
      params_whitened = the needs_params argument, unused (the
                        diagonal chi2 needs no params).
      *args, **kwargs = the mode / trim / focus reduction controls,
                        forwarded positionally (never keyword pred /
                        target before the *args forwarder).

    Returns:
      a scalar loss tensor.
    """
    return CmbDiagonalChi2.loss(
      self, pred, target, *args, **kwargs)
