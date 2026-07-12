"""Grid geometry: a background function on a persisted redshift grid.

The BSN output geometry (D-BSN1): the emulated quantity is a FUNCTION
of redshift — H(z) on the SN-range grid, or the comoving distance
D_M(z) on the recombination window — stored as a vector over a grid
that lives IN the artifact (never a ZLIN sidecar, the
never-trust-defaults rule). Standardization is the ScalarGeometry math
at grid width, applied AFTER the target law:

    raw target row (e.g. H(z_grid), km/s/Mpc)
       │  the law's forward map      "log_offset": t = log(raw + offset)
       │                             "none":       t = raw
       ▼
    law space
       │  (t - center) / scale       per-grid-point standardization,
       ▼                             center/scale from the TRAINING set
    encoded target (what the network learns)

decode inverts both steps (destandardize, then exp(y) - offset for the
log_offset law — the legacy emulbaosn convention, its offset persisted
here). TARGET_LAWS is the small registry (the D-CM2 pattern: persisted
by name in the artifact, never a code default). The loss is ScalarChi2
reused unchanged — it reads only encode / decode / dest_idx off the
geometry, and with the law inside encode/decode the chi2 lives in the
standardized law space.

PS: standardized = shifted to zero mean and scaled to unit variance per
grid point; the law = the invertible closed-form transform applied to
the raw quantity BEFORE standardizing (log(H + offset) keeps the
network target gentle across decades of H); quantity tag = the string
naming what the grid holds ("Hubble" / "D_M"), which the adapter checks
before serving getters from it.
"""

import numpy as np
import torch


# The target-law registry: law name -> the extra state keys it needs
# (persisted by name in the artifact, resolved values, never a code
# default). "none" learns the raw quantity; "log_offset" learns
# log(raw + offset) — the legacy emulbaosn H(z) convention.
TARGET_LAWS = {
  "none":       (),
  "log_offset": ("offset",),
}


class GridGeometry:
  """
  Standardization + target law for a function on a stored z grid.

  Beside ScalarGeometry (per-component center/scale over named
  outputs), this stores the grid itself plus the quantity/units/law
  facts a consumer needs to serve the function: z (the grid), quantity
  ("Hubble" / "D_M"), units ("km/s/Mpc" / "Mpc"), law (a TARGET_LAWS
  key) and offset (the log_offset law's additive constant; 0.0 for
  "none"). dest_idx / total_size are the trivial identity over the
  grid width, so the training loop sizes the model with no new branch.
  """

  def __init__(self,
               device,
               quantity,
               units,
               law,
               offset,
               z,
               center,
               scale):
    """Place the grid-geometry tensors on the device.

    Plain constructor: stores fields only; from_targets builds them
    from training rows and from_state from a saved dict.

    Arguments:
      device   = device the tensors live on.
      quantity = the quantity tag ("Hubble" / "D_M"); the adapter
                 dispatches its getters on it.
      units    = the quantity's units string ("km/s/Mpc" / "Mpc").
      law      = the target-law name (a TARGET_LAWS key).
      offset   = the log_offset law's additive constant (a float;
                 0.0 under "none" — stored resolved either way).
      z        = (NZ,) ascending redshift grid (persisted; the
                 function's domain).
      center   = (NZ,) per-grid-point training mean IN LAW SPACE.
      scale    = (NZ,) per-grid-point training std IN LAW SPACE.
    """
    if law not in TARGET_LAWS:
      raise ValueError(
        "GridGeometry: unknown target law " + repr(law) + "; the "
        "registry has " + repr(sorted(TARGET_LAWS)) + " (persisted by "
        "name, never a default).")
    self.quantity = str(quantity)
    self.units    = str(units)
    self.law      = str(law)
    self.offset   = float(offset)
    self.z = torch.as_tensor(z, dtype=torch.float64, device=device)
    self.center = torch.as_tensor(center,
                                  dtype=torch.float32,
                                  device=device)
    self.scale = torch.as_tensor(scale,
                                 dtype=torch.float32,
                                 device=device)
    # output-width surface for the training loop: every grid point is
    # kept, so dest_idx is the identity arange and total_size the grid
    # width (derived, not persisted).
    n_z = int(self.z.numel())
    self.total_size = n_z
    self.dest_idx   = torch.arange(n_z, device=device)

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference path).

    state's keys match __init__, so cls(device, **state) reconstructs
    the transform with no training targets reread.

    Arguments:
      device = device to place the rebuilt tensors on.
      state  = dict from state(), splatted into __init__.

    Returns:
      a GridGeometry (or subclass, via cls).
    """
    kwargs = dict(state)
    # the string/scalar fields ride h5 round-trips as 1-element lists /
    # 0-dim tensors; normalize both back (the ScalarGeometry names
    # pattern applied to scalars).
    for key in ("quantity", "units", "law"):
      val = kwargs[key]
      if isinstance(val, (list, tuple)):
        kwargs[key] = val[0]
    off = kwargs["offset"]
    if isinstance(off, torch.Tensor):
      kwargs["offset"] = float(off.reshape(-1)[0])
    return cls(device, **kwargs)

  @classmethod
  def from_targets(cls, device, targets, z, quantity, units, law,
                   offset=0.0):
    """Build the standardization from training rows (law applied first).

    center/scale are the per-grid-point mean/std of the LAW-TRANSFORMED
    training targets (population std, ddof 0, deterministic). The
    ScalarGeometry un-standardizable guard applies per grid point; a
    log_offset target that is non-positive after the offset is a loud
    error naming the first bad grid points (log would be non-finite).

    Arguments:
      device   = device for the built tensors.
      targets  = (N, NZ) raw training rows (one per cosmology).
      z        = (NZ,) the ascending redshift grid the columns live on.
      quantity = the quantity tag ("Hubble" / "D_M").
      units    = the quantity's units string.
      law      = the target-law name (a TARGET_LAWS key).
      offset   = the log_offset law's constant (ignored under "none",
                 stored as 0.0 there — resolved values only).

    Returns:
      a GridGeometry whose encode law-transforms then standardizes.
    """
    if law not in TARGET_LAWS:
      raise ValueError(
        "GridGeometry.from_targets: unknown target law " + repr(law)
        + "; the registry has " + repr(sorted(TARGET_LAWS)))
    z = np.asarray(z, dtype="float64")
    Y = np.asarray(targets, dtype="float64")
    if Y.ndim != 2 or Y.shape[1] != z.shape[0]:
      raise ValueError(
        "GridGeometry.from_targets: targets must be (N, NZ) with NZ = "
        "the grid width; got " + repr(Y.shape) + " against a grid of "
        + repr(int(z.shape[0])) + " points")
    if law == "log_offset":
      shifted = Y + float(offset)
      bad = np.nonzero(~(shifted > 0.0).all(axis=0))[0]
      if bad.size > 0:
        show = z[bad][:8].tolist()
        raise ValueError(
          "GridGeometry.from_targets: the log_offset law needs "
          "target + offset > 0 everywhere; non-positive at grid "
          "redshift(s) " + repr(show) + " (first 8). Raise the offset "
          "or use law 'none'.")
      T = np.log(shifted)
      off_store = float(offset)
    else:
      T = Y
      off_store = 0.0
    center = T.mean(0)
    scale  = T.std(0)                         # population std (ddof 0)
    tiny = 8.0 * np.finfo("float32").eps * np.abs(center)
    zero = np.nonzero(scale <= tiny)[0]
    if zero.size > 0:
      show = z[zero][:8].tolist()
      raise ValueError(
        "GridGeometry.from_targets: un-standardizable grid point(s) at "
        "redshift(s) " + repr(show) + " (first 8): the training spread "
        "there is below float32 resolution at its magnitude, so decode "
        "would divide by near-zero. The dump is degenerate at those "
        "points (all cosmologies agree) — check the generator.")
    return cls(device=device, quantity=quantity, units=units, law=law,
               offset=off_store, z=z, center=center, scale=scale)

  def state(self):
    """Tensors/strings to save; keys match __init__ (dest_idx /
    total_size are derived from z, so they are not persisted)."""
    return {"quantity": self.quantity,
            "units":    self.units,
            "law":      self.law,
            "offset":   torch.tensor(self.offset, dtype=torch.float64),
            "z":        self.z.cpu(),
            "center":   self.center.cpu(),
            "scale":    self.scale.cpu()}

  def attach_head_coords(self):
    """Attach the conv/TRF heads' channel/token split (D-CM13).

    The correction heads (designs/plain.py ResCNN / ResTRF) read
    geom.bin_sizes for their channel/token layout; here it is a pure
    derivation from the geometry's own z grid: ONE bin covering the
    whole function, coordinate = z (the conv slides along z; the TRF
    re-segments via model.trf.n_tokens so attention has windows to
    attend across). No permutation, no basis change: the whitening
    is per grid point IN z order, so the heads' W_fd / W_df maps
    stay None. Idempotent; no files, no torch build — safe at
    training (build_geometry) and at rebuild (rebuild_emulator).
    """
    self.bin_sizes = [int(self.z.numel())]

  def encode(self, y):
    """Raw quantity rows -> standardized law-space target.

    Arguments:
      y = (B, NZ) raw rows (e.g. H(z_grid)).

    Returns:
      (B, NZ) the standardized law-space target.
    """
    if self.law == "log_offset":
      t = torch.log(y + self.offset)
    else:
      t = y
    return (t - self.center) / self.scale

  def decode(self, t):
    """Standardized law-space output -> the raw physical quantity.

    Arguments:
      t = (B, NZ) network output in the standardized law space.

    Returns:
      (B, NZ) the raw quantity (exp(y) - offset under log_offset — the
      legacy emulbaosn convention).
    """
    y = t * self.scale + self.center
    if self.law == "log_offset":
      return torch.exp(y) - self.offset
    return y
