"""Output geometry for scalar (derived-parameter) emulators.

The output side of a scalar emulator: maps a small set of named derived
parameters (H0, omegam, rdrag, ...) to the standardized targets the
network predicts (encode) and back (decode). ScalarGeometry standardizes
each output independently: subtract the training mean, divide by the
training standard deviation, so every output is zero-mean, unit-variance
and equally weighted in the loss. Unlike the data-vector geometries there
is no mask, no covariance, and no probe; a derived parameter is one
number, kept in full.

PS: to standardize is to subtract a quantity's mean and divide by its
standard deviation, so it becomes zero-mean and unit-variance (comparable
in scale to the others). encode = standardize each output column; decode
is its exact inverse. There is no rotation (the outputs are treated as
independent), so standardize is the diagonal analogue of the data-vector
geometries' whiten.

    y  (B, n_out)            raw derived parameters (H0, omegam, ...)
       │  - center           subtract each output's training mean
       ▼
       │  / scale            divide by each output's training spread
       ▼
    t  (B, n_out)            standardized targets the network predicts

(legend: B = batch rows; n_out = number of derived-parameter outputs,
the width the network emits; center / scale = per-output training mean
and standard deviation. decode runs the arrows bottom-up, exactly
inverting each step. dest_idx = arange(n_out) and total_size = n_out
give the training loop the same output-width surface the data-vector
geometries expose through their mask, so the loop sizes the model to
n_out with no scalar-specific branch.)
"""

import numpy as np
import torch


class ScalarGeometry:
  """
  Per-output standardization for a scalar (derived-parameter) emulator.

  Standardizing (see module PS) is applied to the derived-parameter
  targets, so the network sees zero-mean, unit-variance outputs instead
  of quantities on wildly different physical scales (H0 near 70,
  omegam near 0.3). One instance owns the encode/decode between raw
  derived parameters and the standardized target vector. It holds:

    - names: the output parameter names, in column order.
    - center: each output's training mean (the zero-point).
    - scale: each output's training standard deviation (the unit).

  Build from the training targets at training time (from_targets) or
  from saved tensors at inference time (from_state); the transform
  travels with the weights, exactly like the parameter and data-vector
  geometries. There is no mask/covariance/probe: dest_idx and
  total_size are the trivial identity (arange(n_out), n_out), present
  only so the training loop's output-width surface is unchanged.
  """

  def __init__(self,
               device,
               names,
               center,
               scale):
    """Place the standardization tensors on the device.

    Plain constructor: stores fields only; the classmethods below build
    them. as_tensor accepts numpy (from training targets) or cpu tensors
    (from a saved state), so both paths share this code. dest_idx and
    total_size are derived from the output count (an identity, not a
    stored knob), so from_state need not persist them.

    Arguments:
      device = device the tensors live on.
      names  = output parameter column order (e.g. ["H0", "omegam"]);
               the loop and the cobaya theory class read the emulated
               names off this.
      center = per-output training mean, the zero-point subtracted
               before scaling.
      scale  = per-output training standard deviation, the unit each
               output is divided by (strictly positive).
    """
    self.names  = list(names)
    self.center = torch.as_tensor(center,
                                  dtype=torch.float32,
                                  device=device)
    self.scale  = torch.as_tensor(scale,
                                  dtype=torch.float32,
                                  device=device)
    # output-width surface for the training loop: with no mask every
    # output is kept, so dest_idx is the identity arange and total_size
    # the output count. Derived from names (not persisted): the loop
    # sizes the model by dest_idx.numel(), so this reuses the
    # data-vector sizing path with no scalar branch.
    n_out = len(self.names)
    self.total_size = n_out
    self.dest_idx   = torch.arange(n_out, device=device)

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference path).

    state's keys match __init__ (names / center / scale), so
    cls(device, **state) reconstructs the transform with no training
    targets reread. cls (not the class name) keeps a subclass correct.

    Arguments:
      device = device to place the rebuilt tensors on.
      state  = dict from state() (names / center / scale), splatted
               into __init__.

    Returns:
      a ScalarGeometry (or subclass, via cls).
    """
    return cls(device, **state)

  @classmethod
  def from_targets(cls, device, targets, names):
    """Build the standardization from the training targets.

    center is each output's training mean, scale its training standard
    deviation (population, ddof 0, deterministic). An un-standardizable
    output is a loud error naming the column(s): a constant column
    (whose std is only mean-rounding noise) or one whose spread is below
    float32 resolution at its magnitude (scale <= 8 * float32-eps *
    |center|), either of which would make decode divide by near-zero.

    Arguments:
      device  = device for the built tensors.
      targets = (N, n_out) raw derived-parameter training targets, one
                row per cosmology, columns in `names` order.
      names   = output parameter column names.

    Returns:
      a ScalarGeometry whose encode standardizes each output.
    """
    Y = np.asarray(targets, dtype="float64")
    center = Y.mean(0)
    scale  = Y.std(0)                         # population std (ddof 0)
    # a constant column's std is pure mean-rounding noise
    # (~eps64 * |mean|), and a spread below float32 resolution at the
    # column's magnitude cannot survive the float32 cast; both are
    # un-standardizable, so both are the loud error.
    tiny = 8.0 * np.finfo("float32").eps * np.abs(center)
    zero = np.nonzero(scale <= tiny)[0]
    if zero.size > 0:
      bad = []
      for j in zero.tolist():
        bad.append(names[j])
      raise ValueError(
        "ScalarGeometry.from_targets: un-standardizable output "
        "column(s) " + repr(bad) + "; each is either a constant column "
        "(its std is only mean-rounding noise) or has a spread below "
        "float32 resolution at its magnitude (scale <= 8 * float32-eps "
        "* |center|), so it cannot be standardized. Drop it from "
        "outputs.")
    return cls(device=device, names=names, center=center, scale=scale)

  def state(self):
    """Collect the persistable transform, keys matching __init__.

    dest_idx / total_size are derived from names, so they are not
    persisted.

    Returns:
      the mapping of output names and per-output center / scale;
      from_state(device, state()) rebuilds the identical geometry.
    """
    return {"names":  self.names,
            "center": self.center.cpu(),
            "scale":  self.scale.cpu()}

  def encode(self, y):
    """Raw derived parameters -> standardized target.

    Subtract the per-output center, divide by the per-output scale.

    Arguments:
      y = (B, n_out) raw derived parameters, columns in names order.

    Returns:
      (B, n_out) standardized targets the network predicts.
    """
    return (y - self.center) / self.scale

  def decode(self, t):
    """Standardized target -> raw derived parameters (inverse of encode).

    Multiply by the per-output scale, add the per-output center back.

    Arguments:
      t = (B, n_out) standardized network outputs.

    Returns:
      (B, n_out) raw derived parameters, columns in names order.
    """
    return t * self.scale + self.center
