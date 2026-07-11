"""2D grid geometry: a function on a persisted (z, k) grid (D-MP1).

The MPS output geometry: the emulated quantity is a function of BOTH
redshift and wavenumber — the linear P(k, z) or the nonlinear boost
B(k, z) = P_nl/P_lin — stored as one flattened row per cosmology:

    row layout (row-major, z outer):   [f(z_0, k_0) ... f(z_0, k_last),
                                        f(z_1, k_0) ... f(z_last, k_last)]

Standardization is the GridGeometry math at width nz*nk, applied to the
LAW-SPACE rows: for the syren laws the staging already formed
log(P / P_base) (the base is cosmology-dependent, so it lives with the
generator + emul_mps through emulator/syren_base.py — the D-MP2-A
ruling; the geometry's encode/decode are pure standardize /
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


# The 2D target-law registry (D-MP2): persisted by name in the artifact.
# The syren laws' base functions live in emulator/syren_base.py; "none"
# learns the raw rows. No extra state keys — the base is recomputed from
# the sampled parameters by the consumer, never stored per-row here.
TARGET_LAWS_2D = {
  "none":          (),
  "syren_linear":  (),
  "syren_halofit": (),
}


class Grid2DGeometry:
  """
  Standardization + law tag for a function on a stored (z, k) grid.

  Beside GridGeometry (one axis) this stores two axes plus the
  quantity/units/law facts: z (nz,), k (nk, 1/Mpc), quantity ("pklin" /
  "boost"), units ("Mpc3" / "dimensionless"), law (a TARGET_LAWS_2D
  key). center/scale are per flattened grid point over the law-space
  training rows. dest_idx / total_size are the identity over nz*nk.
  """

  def __init__(self,
               device,
               quantity,
               units,
               law,
               z,
               k,
               center,
               scale):
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
      scale    = (nz*nk,) per-point training std IN LAW SPACE.
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

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference path).

    Arguments:
      device = device to place the rebuilt tensors on.
      state  = dict from state(), splatted into __init__.

    Returns:
      a Grid2DGeometry (or subclass, via cls).
    """
    kwargs = dict(state)
    for key in ("quantity", "units", "law"):
      val = kwargs[key]
      if isinstance(val, (list, tuple)):
        kwargs[key] = val[0]
    return cls(device, **kwargs)

  @classmethod
  def from_targets(cls, device, targets, z, k, quantity, units, law):
    """Build the standardization from LAW-SPACE training rows.

    The rows arrive already law-transformed (the staging formed
    log(P/P_base) from the generator's base files — D-MP2-A(2)), so
    this only standardizes: per-point mean/std (population, ddof 0)
    with the un-standardizable guard naming the first bad (z, k)
    points.

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
    tiny = 8.0 * np.finfo("float32").eps * np.abs(center)
    zero = np.nonzero(scale <= tiny)[0]
    if zero.size > 0:
      show = []
      for j in zero[:8].tolist():
        show.append((float(z[j // k.shape[0]]),
                     float(k[j % k.shape[0]])))
      raise ValueError(
        "Grid2DGeometry.from_targets: un-standardizable grid point(s) "
        "at (z, k) " + repr(show) + " (first 8): the training spread "
        "there is below float32 resolution at its magnitude. The dump "
        "is degenerate at those points — check the generator (a "
        "constant law-space column usually means the base already "
        "equals the truth there).")
    return cls(device=device, quantity=quantity, units=units, law=law,
               z=z, k=k, center=center, scale=scale)

  def state(self):
    """Tensors/strings to save; keys match __init__ (dest_idx /
    total_size are derived from the axes, so they are not persisted)."""
    return {"quantity": self.quantity,
            "units":    self.units,
            "law":      self.law,
            "z":        self.z.cpu(),
            "k":        self.k.cpu(),
            "center":   self.center.cpu(),
            "scale":    self.scale.cpu()}

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
    the base back through emulator/syren_base.py, D-MP2-A(4)).

    Arguments:
      t = (B, nz*nk) network output.

    Returns:
      (B, nz*nk) law-space rows (log(P/P_base) for the syren laws).
    """
    return t * self.scale + self.center
