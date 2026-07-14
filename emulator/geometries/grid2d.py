"""2D grid geometry: a function on a persisted (z, k) grid.

The MPS output geometry: the emulated quantity is a function of BOTH
redshift and wavenumber — the linear P(k, z) or the nonlinear boost
B(k, z) = P_nl/P_lin — stored as one flattened row per cosmology:

    row layout (row-major, z outer):   [f(z_0, k_0) ... f(z_0, k_last),
                                        f(z_1, k_0) ... f(z_last, k_last)]

Standardization is the GridGeometry math at width nz*nk, applied to the
LAW-SPACE rows: for the syren laws the staging already formed
log(P / P_base) (the base is cosmology-dependent, so it lives with the
generator + emul_mps through emulator/syren_base.py; the geometry's
encode/decode are pure standardize /
destandardize, torch-only). The law NAME persists here as the artifact
fact that tells every consumer which base to multiply back; the stored
k grid is the (possibly downsampled) grid the run actually trained on —
resolved values, never a stride knob to re-apply.

PS: law space = the rows after the registry transform (log(P/P_base)
for "syren_linear", log(B/B_base) for "syren_halofit", the raw rows for
"none"); flattened = the (nz, nk) surface unrolled z-outer into one
vector the network predicts.
"""

import numpy as np
import torch


# The 2D target-law registry: persisted by name in the artifact.
# The syren laws' base functions live in emulator/syren_base.py; "none"
# learns the raw rows. No extra state keys — the base is recomputed from
# the sampled parameters by the consumer, never stored per-row here.
TARGET_LAWS_2D = {
  "none":          (),
  "syren_linear":  (),
  "syren_halofit": (),
}


def _normalize_const_mask(
    const_mask,
    n_out,
    device):
  """Return one explicit boolean mask covering the flattened grid.

  The in-memory representation is boolean.  This class writes the same
  values as explicit uint8 zeros and ones in state().  A direct constructor
  may pass None explicitly. The helper converts that value into an all-false
  mask immediately. Saved-state readback requires the key before it calls
  this helper.
  """
  if const_mask is None:
    return torch.zeros(
      n_out,
      dtype=torch.bool,
      device=device)

  try:
    mask = torch.as_tensor(const_mask)
  except (TypeError, ValueError) as exc:
    raise TypeError(
      "Grid2DGeometry: const_mask must be a one-dimensional boolean or "
      "uint8 array") from exc
  if mask.ndim != 1:
    raise ValueError(
      "Grid2DGeometry: const_mask must be a one-dimensional array over "
      "the flattened (z, k) grid. Got shape " + repr(tuple(mask.shape)))
  if int(mask.numel()) != n_out:
    raise ValueError(
      "Grid2DGeometry: const_mask has " + str(int(mask.numel()))
      + " points but the (z, k) grid has " + str(n_out)
      + ". The mask must cover the flattened surface")
  if mask.dtype not in (torch.bool, torch.uint8):
    raise TypeError(
      "Grid2DGeometry: const_mask must contain booleans or persisted "
      "uint8 zeros and ones. Got dtype " + str(mask.dtype))
  if mask.dtype == torch.uint8:
    is_zero = mask == 0
    is_one = mask == 1
    invalid = torch.logical_not(torch.logical_or(is_zero, is_one))
    if bool(invalid.any().item()):
      raise ValueError(
        "Grid2DGeometry: persisted uint8 const_mask values must be 0 or "
        "1")
  return mask.to(
    device=device,
    dtype=torch.bool)


class Grid2DGeometry:
  """
  Standardization + law tag for a function on a stored (z, k) grid.

  Beside GridGeometry (one axis) this stores two axes plus the
  quantity/units/law facts: z (nz,), k (nk, 1/Mpc), quantity ("pklin" /
  "boost"), units ("Mpc3" / "dimensionless"), law (a TARGET_LAWS_2D
  key). center/scale are per flattened grid point over the law-space
  training rows. dest_idx / total_size are the identity over nz*nk.
  const_mask marks the pinned points. These are law-space
  columns constant across the training cosmologies (the boost's
  low-k region, where B = 1 for every cosmology under ANY law):
  decode returns the training constant there, and the mask persists
  with every current artifact. An all-false mask states explicitly
  that no points are pinned. A WHOLLY constant surface is still a
  loud error because it identifies a dead dump, never physics.
  """

  def __init__(self,
               device,
               quantity,
               units,
               law,
               z,
               k,
               center,
               scale,
               const_mask):
    """Place the 2D grid-geometry tensors on the device.

    Arguments:
      device   = device the tensors live on.
      quantity = "pklin" (linear P) or "boost" (P_nl/P_lin).
      units    = the raw quantity's units ("Mpc3" / "dimensionless").
      law      = the target-law name (a TARGET_LAWS_2D key).
      z        = (nz,) ascending redshift axis.
      k        = (nk,) ascending wavenumber axis, 1/Mpc (the grid the
                 run trained on — already downsampled if it was).
      center   = (nz*nk,) per-point training mean IN LAW SPACE.
      scale    = (nz*nk,) per-point training std IN LAW SPACE (1.0 at
                 the pinned points below).
      const_mask = (nz*nk,) booleans. True marks a constant law-space
                   training column. False leaves the network output active.
                   Explicit None is accepted as a direct-construction
                   convenience and is converted immediately to an all-false
                   mask. Omitting the argument is an error, and saved-state
                   readback requires the key.
    """
    if law not in TARGET_LAWS_2D:
      raise ValueError(
        "Grid2DGeometry: unknown target law " + repr(law) + "; the "
        "registry has " + repr(sorted(TARGET_LAWS_2D)) + " (persisted "
        "by name, never a default).")
    self.quantity = str(quantity)
    self.units    = str(units)
    self.law      = str(law)
    self.z = torch.as_tensor(z, dtype=torch.float64, device=device)
    self.k = torch.as_tensor(k, dtype=torch.float64, device=device)
    self.center = torch.as_tensor(center,
                                  dtype=torch.float32,
                                  device=device)
    self.scale = torch.as_tensor(scale,
                                 dtype=torch.float32,
                                 device=device)
    n_out = int(self.z.numel()) * int(self.k.numel())
    if int(self.center.numel()) != n_out:
      raise ValueError(
        "Grid2DGeometry: center has " + str(int(self.center.numel()))
        + " points but the (z, k) grid is " + str(int(self.z.numel()))
        + " x " + str(int(self.k.numel())) + " = " + str(n_out)
        + "; the rows must be the flattened (z-outer) surface")
    self.total_size = n_out
    self.dest_idx   = torch.arange(n_out, device=device)
    self.const_mask = _normalize_const_mask(
      const_mask=const_mask,
      n_out=n_out,
      device=device)

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference path).

    Arguments:
      device = device to place the rebuilt tensors on.
      state  = dict from state(), splatted into __init__. Current grid2d
               state requires const_mask even when every value is false.

    Returns:
      a Grid2DGeometry (or subclass, via cls).
    """
    if "const_mask" not in state:
      raise KeyError(
        "Grid2DGeometry.from_state: the saved geometry is missing required "
        "const_mask state. Key absence cannot choose pinned or unpinned "
        "science. Re-save the emulator with the current grid2d geometry "
        "schema before rebuilding it.")
    kwargs = dict(state)
    for key in ("quantity", "units", "law"):
      val = kwargs[key]
      if isinstance(val, (list, tuple)):
        kwargs[key] = val[0]
    return cls(device, **kwargs)

  @classmethod
  def from_stats(cls, device, center, scale, z, k, quantity, units,
                 law):
    """Build the standardization from PRECOMPUTED law-space moments.

    from_targets standardizes a materialized (N, nz*nk) matrix; the
    bounded MPS staging (experiment._grid2d_law_rows) instead STREAMS
    the same two per-point moments over row chunks, because the
    122 x 2,000 production surface is tens of GiB and never fits in
    memory whole. Both routes end here: from_stats is the single home
    of the constant-pin and dead-dump rules, so the pinning is
    identical whether the moments were materialized or streamed.

    Arguments:
      device   = device for the built tensors.
      center   = (nz*nk,) per-point law-space mean.
      scale    = (nz*nk,) per-point law-space population std (ddof 0),
                 BEFORE pinning (pinned points are set to 1 here).
      z        = (nz,) the redshift axis.
      k        = (nk,) the wavenumber axis (1/Mpc).
      quantity = "pklin" / "boost".
      units    = the raw quantity's units string.
      law      = the target-law name (a TARGET_LAWS_2D key).

    Returns:
      a Grid2DGeometry whose encode standardizes the law-space rows.

    Raises:
      ValueError on an unknown law, a center/scale length that is not
      nz*nk, or a WHOLLY constant surface (a dead dump under any law).
    """
    if law not in TARGET_LAWS_2D:
      raise ValueError(
        "Grid2DGeometry.from_stats: unknown target law " + repr(law)
        + "; the registry has " + repr(sorted(TARGET_LAWS_2D)))
    z = np.asarray(z, dtype="float64")
    k = np.asarray(k, dtype="float64")
    center = np.asarray(center, dtype="float64").reshape(-1)
    scale  = np.asarray(scale,  dtype="float64").reshape(-1)
    n_out  = int(z.shape[0]) * int(k.shape[0])
    if center.shape[0] != n_out or scale.shape[0] != n_out:
      raise ValueError(
        "Grid2DGeometry.from_stats: center/scale must be length "
        "nz*nk = " + str(int(n_out)) + "; got center "
        + str(int(center.shape[0])) + ", scale "
        + str(int(scale.shape[0])))
    # the un-standardizable points: a law-space column with no spread.
    tiny = 8.0 * np.finfo("float32").eps * np.abs(center)
    zero = np.nonzero(scale <= tiny)[0]
    # The mask is always explicit.  All false means that the geometry has no
    # pins. state() persists that fact instead of making key presence select
    # a scientific branch during readback.
    const_mask = np.zeros(n_out, dtype=bool)
    if zero.size > 0:
      # a WHOLLY constant surface is a dead dump under any law (a
      # stale generator writing one cosmology's rows everywhere — the
      # bsn-smoke failure class), never a physical region.
      if zero.size == n_out:
        raise ValueError(
          "Grid2DGeometry: EVERY grid point is constant across the "
          "training rows — the dump is degenerate (a stale generator "
          "writing one cosmology everywhere); check the generator, "
          "this is never a physical surface.")
      # the constant pin (LAW-AGNOSTIC): a constant law-space column
      # that is not the whole surface is PHYSICS, not a generator bug —
      # the boost is 1 below the nonlinear scale for every cosmology,
      # so its low-k columns are constant under ANY law: under a syren
      # law log(B/B_base) = 0 identically (the base is exact there);
      # under law "none" the raw value 1 itself is the constant. PIN
      # those points — scale 1 (nothing to whiten by), and decode
      # returns the training constant exactly. The mask persists in
      # the artifact (state), so serving matches training bit for bit.
      # The dead-dump protection is the WHOLE-surface guard above,
      # which fires for every law.
      const_mask[zero] = True
      scale = scale.copy()
      scale[zero] = 1.0
    return cls(device=device, quantity=quantity, units=units, law=law,
               z=z, k=k, center=center, scale=scale,
               const_mask=const_mask)

  @classmethod
  def from_targets(cls, device, targets, z, k, quantity, units, law):
    """Build the standardization from LAW-SPACE training rows.

    The rows arrive already law-transformed (the staging formed
    log(P/P_base) from the generator's base files), so this only
    computes the two per-point moments (population mean/std, ddof 0)
    and hands them to from_stats, which owns the pin / dead-dump rules
    (the bounded staging streams the same two moments — one code path
    standardizes both).

    Arguments:
      device   = device for the built tensors.
      targets  = (N, nz*nk) law-space training rows (flattened
                 z-outer).
      z        = (nz,) the redshift axis.
      k        = (nk,) the wavenumber axis (1/Mpc).
      quantity = "pklin" / "boost".
      units    = the raw quantity's units string.
      law      = the target-law name (a TARGET_LAWS_2D key).

    Returns:
      a Grid2DGeometry whose encode standardizes the law-space rows.
    """
    if law not in TARGET_LAWS_2D:
      raise ValueError(
        "Grid2DGeometry.from_targets: unknown target law " + repr(law)
        + "; the registry has " + repr(sorted(TARGET_LAWS_2D)))
    z = np.asarray(z, dtype="float64")
    k = np.asarray(k, dtype="float64")
    Y = np.asarray(targets, dtype="float64")
    n_out = z.shape[0] * k.shape[0]
    if Y.ndim != 2 or Y.shape[1] != n_out:
      raise ValueError(
        "Grid2DGeometry.from_targets: targets must be (N, nz*nk) with "
        "nz*nk = " + str(int(n_out)) + "; got " + repr(Y.shape))
    center = Y.mean(0)
    scale  = Y.std(0)                          # population std (ddof 0)
    return cls.from_stats(device=device, center=center, scale=scale,
                          z=z, k=k, quantity=quantity, units=units,
                          law=law)

  def state(self):
    """Tensors/strings to save; keys match __init__ (dest_idx /
    total_size are derived from the axes, so they are not persisted).
    const_mask is always present. An all-false mask records an explicitly
    unpinned geometry, while true values record the pinned columns. This
    class writes uint8 zeros and ones so the stored representation is
    explicit, and __init__ normalizes it back to boolean."""
    st = {"quantity": self.quantity,
          "units":    self.units,
          "law":      self.law,
          "z":        self.z.cpu(),
          "k":        self.k.cpu(),
          "center":   self.center.cpu(),
          "scale":    self.scale.cpu(),
          "const_mask": self.const_mask.cpu().to(torch.uint8)}
    return st

  def attach_head_coords(self):
    """Attach the conv/TRF heads' channel/token split.

    The correction heads (designs/plain.py ResCNN / ResTRF) read
    geom.bin_sizes for their channel/token layout; here it is a pure
    derivation from the geometry's own axes. The flattening is
    z-outer (row = z0 all k, z1 all k, ...), so the natural split is
    one bin PER Z SLICE, each of length nk: the conv gets the z
    slices as channels and slides along k (channel mixing couples
    redshifts at like k), the TRF gets one token per z slice
    (attention shares information across redshifts, each slice's
    private MLP specializes along k). model.trf.n_tokens is rejected
    here — the z slices ARE the tokens. No permutation, no basis
    change: the whitening is per (z, k) point in grid order, so the
    heads' W_fd / W_df maps stay None. Idempotent; no files, no
    torch build — safe at training and at rebuild.
    """
    sizes = []
    for _ in range(int(self.z.numel())):
      sizes.append(int(self.k.numel()))
    self.bin_sizes = sizes

  def encode(self, y):
    """Law-space rows -> standardized target.

    Arguments:
      y = (B, nz*nk) law-space rows (the staging already applied the
          registry transform).

    Returns:
      (B, nz*nk) the standardized target.
    """
    return (y - self.center) / self.scale

  def decode(self, t):
    """Standardized output -> law-space rows (the consumer multiplies
    the base back through emulator/syren_base.py).

    At the pinned points (law-space training columns constant
    across cosmologies — the boost's low-k region under any law) the
    network's output is REPLACED by the training constant: that
    constant IS the physics there (the base under a syren law, the
    raw value itself under law "none"), and letting network noise
    through at scale 1.0 would leak a spurious percent-level wiggle
    into the served spectrum's low-k tail.

    Arguments:
      t = (B, nz*nk) network output.

    Returns:
      (B, nz*nk) law-space rows (log(P/P_base) for the syren laws).
    """
    out = t * self.scale + self.center
    return torch.where(
      self.const_mask,
      self.center,
      out)
