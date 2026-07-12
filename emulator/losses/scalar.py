"""Standardized mean-square loss for scalar (derived-parameter) emulators.

The scalar sibling of losses/core.py's CosmolikeChi2: it wraps a
ScalarGeometry and exposes the exact loop interface the data-vector
losses do (encode / decode / a per-sample chi2 / loss), so the shared
training machinery (trimming, the focal hardness weight, the sqrt / berhu
transform ladder, EMA, the L2-SP anchor) composes unchanged. The only
difference from the data-vector chi2 is the metric: the outputs are
standardized and treated as independent, so the per-sample chi2 is the
plain sum of squared standardized residuals (a diagonal, unit-variance
Mahalanobis distance), not a covariance contraction.

PS: a standardized residual is (pred - target) after each output has been
put in units of its training standard deviation (see geometries.scalar's
PS); its square summed over the outputs is the per-sample chi2 this loss
reduces. "The loop interface" is the small set of methods the training
loop calls on the loss object: encode (raw target -> network space),
decode (its inverse), chi2 (per-sample metric), and loss (the scalar
training objective). Reusing CosmolikeChi2's reduction keeps every knob
identical.
"""

import torch

from .core import CosmolikeChi2


class ScalarChi2(CosmolikeChi2):
  """
  CosmolikeChi2 for standardized scalar outputs.

  Holds a ScalarGeometry (self.geom) instead of a DataVectorGeometry and
  overrides only the metric: the per-sample chi2 is the sum of squared
  standardized residuals, sum_i (pred_i - target_i)^2 over the n_out
  outputs. Everything else is inherited from CosmolikeChi2:

    - encode / decode forward to self.geom (standardize / destandardize);
    - dest_idx / total_size forward to self.geom (arange(n_out) / n_out),
      so the training loop sizes the model to n_out with no scalar branch;
    - loss(...) and _reduce(...) are the shared trim / mode transform /
      focal reduction, so trim / focus / berhu / EMA / anchor all apply
      to the scalar per-sample chi2 with no change.

  Because the outputs are standardized to unit variance, the squared
  residual already is the chi2 (a diagonal, unit-variance covariance), so
  there is no covariance to contract and no mask to unsqueeze through: the
  full argument is accepted for interface parity but makes no difference
  (there are no masked-out entries).
  """

  def chi2(self, pred, target, full=False):
    """Per-sample chi2 = sum of squared standardized residuals.

    The outputs live in the standardized (unit-variance, independent)
    space the geometry's encode produced, so the Mahalanobis distance is
    the plain squared L2 residual summed over the outputs. full is
    accepted so the signature matches CosmolikeChi2.chi2 (the eval path
    passes it on the data-vector loss) but has no effect here: a scalar
    geometry has no masked entries to scatter through.

    Arguments:
      pred   = (B, n_out) network outputs in the standardized space.
      target = (B, n_out) standardized targets, same shape.
      full   = accepted for interface parity with CosmolikeChi2.chi2;
               ignored (no mask / covariance on the scalar path).

    Returns:
      (B,) per-sample chi2.
    """
    r = pred - target
    return (r * r).sum(dim=1)


def make_scalar_chi2(geom):
  """
  Build the scalar chi2fn (loss + geometry wrapper) for run_emulator.

  The scalar analogue of make_chi2: there is no rescaling / ia variant
  (those are data-vector concepts), so this simply wraps the geometry in
  a ScalarChi2. The NPCE variant DOES exist family-wide (the 2026-07-12
  ruling) but is not built here: when a pce: block is present the
  experiment fits the base and wraps PCEResidualDiagChi2 instead of
  calling this factory (experiment._fit_diag_pce). Kept as a factory so
  the driver and experiment build the loss by a call, mirroring
  make_chi2.

  Arguments:
    geom = ScalarGeometry for the emulated outputs (its .state() is what
           save_emulator persists; pass the geometry, not a chi2fn).

  Returns:
    a ScalarChi2 to pass to run_emulator as chi2fn.
  """
  return ScalarChi2(geom=geom)
