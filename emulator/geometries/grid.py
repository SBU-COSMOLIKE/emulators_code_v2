"""Grid geometry: a background function on a persisted redshift grid.

The BSN output geometry: the emulated quantity is a FUNCTION
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
here). TARGET_LAWS is the small registry (persisted
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

from ..validation import _is_finite_real


# The target-law registry: law name -> the extra state keys it needs
# (persisted by name in the artifact, resolved values, never a code
# default). "none" learns the raw quantity; "log_offset" learns
# log(raw + offset) — the legacy emulbaosn H(z) convention.
TARGET_LAWS = {
  "none":       (),
  "log_offset": ("offset",),
}


# One producer/consumer registry owns both the quantity name and its unit.
# A training run must not publish a pair the Cobaya adapter cannot serve.
BACKGROUND_QUANTITY_UNITS = {
  "Hubble": "km/s/Mpc",
  "D_M": "Mpc",
}


def validate_background_quantity_units(quantity, units, where):
  """Require one background quantity and its matching physical unit.

  Arguments:
    quantity = candidate background function name.
    units    = candidate unit string for that function.
    where    = short name of the configuration or artifact being checked.

  Returns:
    None. A valid pair needs no conversion.

  Raises:
    ValueError when either value is not a string or the pair is not listed in
    ``BACKGROUND_QUANTITY_UNITS``. The message lists every accepted pair so a
    user can correct the configuration or regenerate the artifact.
  """
  allowed_pairs = tuple(BACKGROUND_QUANTITY_UNITS.items())
  if not isinstance(quantity, str) or not isinstance(units, str):
    raise ValueError(
      where + ": background quantity and units must both be strings; got "
      "quantity " + repr(quantity) + " (" + type(quantity).__name__ + ") "
      "and units " + repr(units) + " (" + type(units).__name__ + "). Use "
      "one of the accepted pairs " + repr(allowed_pairs) + ".")

  expected_units = BACKGROUND_QUANTITY_UNITS.get(quantity)
  if expected_units is None or units != expected_units:
    raise ValueError(
      where + ": background quantity/units pair "
      + repr((quantity, units))
      + " is not accepted. Use one of the accepted pairs "
      + repr(allowed_pairs) + "; change the configuration or regenerate "
      "the artifact with that matching pair.")


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

    The constructor validates the quantity, units, law, offset, center, and
    scale before storing them. ``from_targets`` builds those values from
    training rows, while ``from_state`` reads them from a saved dictionary.

    Arguments:
      device   = device the tensors live on.
      quantity = the quantity tag ("Hubble" / "D_M"); the adapter
                 dispatches its getters on it.
      units    = the quantity's units string ("km/s/Mpc" / "Mpc").
      law      = the target-law name (a TARGET_LAWS key).
      offset   = the finite non-Boolean real used by log_offset; 0.0 under
                 "none". The resolved value is always saved.
      z        = (NZ,) ascending redshift grid (persisted; the
                 function's domain).
      center   = (NZ,) finite per-grid-point training mean in law space.
      scale    = (NZ,) finite per-grid-point training std in law space.
    """
    validate_background_quantity_units(
      quantity=quantity,
      units=units,
      where="GridGeometry",
    )
    if law not in TARGET_LAWS:
      raise ValueError(
        "GridGeometry: unknown target law " + repr(law) + "; the "
        "registry has " + repr(sorted(TARGET_LAWS)) + " (persisted by "
        "name, never a default).")
    if not _is_finite_real(offset):
      raise ValueError(
        "GridGeometry.offset must be a finite real number, not a Boolean "
        "or string; got " + repr(offset) + ". Regenerate the geometry with "
        "a finite numeric offset.")

    offset_value = float(offset)
    z_tensor = torch.as_tensor(z, dtype=torch.float64, device=device)
    grid_is_valid = (
      z_tensor.ndim == 1
      and z_tensor.numel() >= 2
      and bool(torch.isfinite(z_tensor).all())
      and bool((z_tensor >= 0.0).all())
      and bool((z_tensor[1:] > z_tensor[:-1]).all())
    )
    if not grid_is_valid or (quantity == "Hubble" and z_tensor[0] != 0.0):
      raise ValueError(
        "GridGeometry redshift grid must be finite, nonnegative, strictly "
        "increasing and one-dimensional; "
        "a Hubble grid must start exactly at z = 0")
    center_tensor = torch.as_tensor(
      center,
      dtype=torch.float32,
      device=device,
    )
    scale_tensor = torch.as_tensor(
      scale,
      dtype=torch.float32,
      device=device,
    )
    bad_center_count = int((~torch.isfinite(center_tensor)).sum().item())
    if bad_center_count > 0:
      raise ValueError(
        "GridGeometry.center contains " + repr(bad_center_count)
        + " non-finite value(s). The saved law-space center must be finite; "
        "check the training rows and rebuild the artifact.")
    bad_scale_count = int((~torch.isfinite(scale_tensor)).sum().item())
    if bad_scale_count > 0:
      raise ValueError(
        "GridGeometry.scale contains " + repr(bad_scale_count)
        + " non-finite value(s). The saved law-space scale must be finite; "
        "check the training rows and rebuild the artifact.")

    self.quantity = quantity
    self.units = units
    self.law = law
    self.offset = offset_value
    self.z = z_tensor
    self.center = center_tensor
    self.scale = scale_tensor
    # output-width surface for the training loop: every grid point is
    # kept, so dest_idx is the identity arange and total_size the grid
    # width (derived, not persisted).
    n_z = int(self.z.numel())
    self.total_size = n_z
    self.dest_idx   = torch.arange(n_z, device=device)

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference path).

    The saved keys match ``__init__``. This method normalizes the HDF5 scalar
    forms, and the constructor checks the scientific values before the
    transform is returned. No training targets are read.

    Arguments:
      device = device to place the rebuilt tensors on.
      state  = dict from state(), splatted into __init__.

    Returns:
      a GridGeometry (or subclass, via cls).

    Raises:
      ValueError when the saved offset is not one finite numeric scalar or
      when the constructor rejects saved metadata, center, or scale values.
    """
    kwargs = dict(state)
    # the string/scalar fields ride h5 round-trips as 1-element lists /
    # 0-dim tensors; normalize both back (the ScalarGeometry names
    # pattern applied to scalars).
    for key in ("quantity", "units", "law"):
      val = kwargs[key]
      if isinstance(val, (list, tuple)):
        kwargs[key] = val[0]
    saved_offset = kwargs["offset"]
    if isinstance(saved_offset, torch.Tensor):
      if saved_offset.numel() != 1 or saved_offset.dtype == torch.bool:
        raise ValueError(
          "GridGeometry.from_state: saved offset must contain one numeric "
          "value, not a Boolean; got shape "
          + repr(tuple(saved_offset.shape)) + " and dtype "
          + repr(saved_offset.dtype) + ". Regenerate the artifact.")
      saved_offset_is_finite = bool(torch.isfinite(saved_offset).all().item())
      if not saved_offset_is_finite:
        raise ValueError(
          "GridGeometry.from_state: saved offset must be finite; got "
          + repr(saved_offset.detach().cpu().reshape(-1).tolist())
          + ". Regenerate the artifact with a finite offset.")
      kwargs["offset"] = saved_offset.reshape(-1)[0].item()
    return cls(
      device=device,
      quantity=kwargs["quantity"],
      units=kwargs["units"],
      law=kwargs["law"],
      offset=kwargs["offset"],
      z=kwargs["z"],
      center=kwargs["center"],
      scale=kwargs["scale"],
    )

  @classmethod
  def from_targets(cls, device, targets, z, quantity, units, law,
                   offset=0.0):
    """Build the standardization from training rows (law applied first).

    center/scale are the per-grid-point mean/std of the LAW-TRANSFORMED
    training targets (population std, ddof 0, deterministic). The
    ScalarGeometry's un-standardizable guard applies per grid point. Raw,
    shifted, transformed, center, and scale values are checked for finiteness
    before comparative guards run. A log_offset target that is non-positive
    after the offset is an error naming the first bad grid points.

    Arguments:
      device   = device for the built tensors.
      targets  = (N, NZ) raw training rows (one per cosmology).
      z        = (NZ,) the ascending redshift grid the columns live on.
      quantity = the quantity tag ("Hubble" / "D_M").
      units    = the quantity's units string.
      law      = the target-law name (a TARGET_LAWS key).
      offset   = a finite non-Boolean real. The log_offset law uses it; the
                 "none" law stores 0.0 instead.

    Returns:
      a GridGeometry whose encode law-transforms then standardizes.
    """
    validate_background_quantity_units(
      quantity=quantity,
      units=units,
      where="GridGeometry.from_targets",
    )
    if law not in TARGET_LAWS:
      raise ValueError(
        "GridGeometry.from_targets: unknown target law " + repr(law)
        + "; the registry has " + repr(sorted(TARGET_LAWS)))
    if not _is_finite_real(offset):
      raise ValueError(
        "GridGeometry.from_targets: offset must be a finite real number, "
        "not a Boolean or string; got " + repr(offset) + ". Set "
        "data.grid.offset to a finite number.")
    offset_value = float(offset)
    z = np.asarray(z, dtype="float64")
    Y = np.asarray(targets, dtype="float64")
    if Y.ndim != 2 or Y.shape[1] != z.shape[0]:
      raise ValueError(
        "GridGeometry.from_targets: targets must be (N, NZ) with NZ = "
        "the grid width; got " + repr(Y.shape) + " against a grid of "
        + repr(int(z.shape[0])) + " points")
    bad_raw = np.argwhere(~np.isfinite(Y))
    if bad_raw.size > 0:
      bad_columns = np.unique(bad_raw[:, 1])
      show = z[bad_columns][:8].tolist()
      raise ValueError(
        "GridGeometry.from_targets: raw target rows contain non-finite "
        "values at grid redshift(s) " + repr(show) + " (first 8). Repair "
        "or regenerate the training dump before building the geometry.")
    if law == "log_offset":
      with np.errstate(over="ignore", invalid="ignore"):
        shifted = Y + offset_value
      bad_shifted = np.argwhere(~np.isfinite(shifted))
      if bad_shifted.size > 0:
        bad_columns = np.unique(bad_shifted[:, 1])
        show = z[bad_columns][:8].tolist()
        raise ValueError(
          "GridGeometry.from_targets: target + offset is non-finite at "
          "grid redshift(s) " + repr(show) + " (first 8). Use a finite "
          "offset that does not overflow the shifted training rows.")
      bad = np.nonzero(~(shifted > 0.0).all(axis=0))[0]
      if bad.size > 0:
        show = z[bad][:8].tolist()
        raise ValueError(
          "GridGeometry.from_targets: the log_offset law needs "
          "target + offset > 0 everywhere; non-positive at grid "
          "redshift(s) " + repr(show) + " (first 8). Raise the offset "
          "or use law 'none'.")
      T = np.log(shifted)
      bad_transformed = np.argwhere(~np.isfinite(T))
      if bad_transformed.size > 0:
        bad_columns = np.unique(bad_transformed[:, 1])
        show = z[bad_columns][:8].tolist()
        raise ValueError(
          "GridGeometry.from_targets: log(target + offset) is non-finite "
          "at grid redshift(s) " + repr(show) + " (first 8). Repair the "
          "training rows or choose a finite offset with a positive sum.")
      off_store = offset_value
    else:
      T = Y
      off_store = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
      center = T.mean(0)
      scale = T.std(0)                         # population std (ddof 0)
    bad_center = np.nonzero(~np.isfinite(center))[0]
    if bad_center.size > 0:
      show = z[bad_center][:8].tolist()
      raise ValueError(
        "GridGeometry.from_targets: the law-space center is non-finite at "
        "grid redshift(s) " + repr(show) + " (first 8). Check the training "
        "rows before standardization.")
    bad_scale = np.nonzero(~np.isfinite(scale))[0]
    if bad_scale.size > 0:
      show = z[bad_scale][:8].tolist()
      raise ValueError(
        "GridGeometry.from_targets: the law-space scale is non-finite at "
        "grid redshift(s) " + repr(show) + " (first 8). Check the training "
        "rows before standardization.")
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
    return cls(
      device=device,
      quantity=quantity,
      units=units,
      law=law,
      offset=off_store,
      z=z,
      center=center,
      scale=scale,
    )

  def state(self):
    """Collect the persistable transform, keys matching __init__.

    dest_idx / total_size are derived from z, so they are not persisted.

    Returns:
      the mapping of quantity / units / law / offset, the z axis, and
      the per-element statistics; from_state(device, state()) rebuilds
      the identical geometry.
    """
    return {"quantity": self.quantity,
            "units":    self.units,
            "law":      self.law,
            "offset":   torch.tensor(self.offset, dtype=torch.float64),
            "z":        self.z.cpu(),
            "center":   self.center.cpu(),
            "scale":    self.scale.cpu()}

  def attach_head_coords(self):
    """Attach the conv/TRF heads' channel/token split.

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
    width = int(self.z.numel())
    self.bin_sizes = [width]
    self.head_pad_idx = torch.arange(
      width, dtype=torch.long, device=self.z.device)
    self.head_valid_mask = torch.ones(
      (1, width), dtype=torch.bool, device=self.z.device)

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
