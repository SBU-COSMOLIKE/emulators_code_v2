"""Analytic cosmic-shear rescaling R: the As / shape preprocessor.

Cosmic shear is the correlated distortion of galaxy shapes produced by
gravitational lensing from large-scale structure.  Its data vector is
built from the two correlation functions xi+ and xi-, measured between
tomographic bins (redshift slices of the galaxy sample) as functions
of the angular separation theta.  This module evaluates a fast
closed-form reference for that signal — the Eisenstein-Hu zero-baryon
transfer function, linear theory, and a single-plane Limber mapping
from angle to wavenumber — and divides it out, so the network emulates
a flatter target: R removes the broadband dependence on the primordial
amplitude As and on the spectrum's shape, leaving the network the
residual physics the closed form misses.

_analytic_R holds the formula once (numpy or torch, picked by input
type).  analytic_shape_ratio wraps it over the masked data vector (the
emulator path); rescale_xi wraps it over the (theta, xip, xim) matrix
layout (for plotting and visual checks).  The RescaledChi2 and
ResidualBaseChi2 losses call _analytic_R on-device, with tensors
already resident on the training GPU.

Vocabulary used throughout: the geometry's squeeze keeps only the
unmasked data-vector entries, so the squeezed dv has one column per
kept element, and unsqueeze scatters them back into the full-length
vector; "resident" means held in GPU memory the whole run, not
re-loaded each batch.
"""

import numpy as np
import torch


def _analytic_R(theta_arcmin,
                z_eff,
                cosmo,
                cosmo_mid,
                names,
                u_star=0.5,
                include_amp=False):
  """Core analytic cosmic-shear rescaling R: the one place the formula lives.

    R = (As_mid/As) * q_mid^ns_mid T(q_mid)^2 / (q^ns T(q)^2),
    q = K / (theta_rad * z_eff * Omega_m h),
    K = 100 Theta^2 / (c[km/s] * u*),

  T is the Eisenstein-Hu zero-baryon transfer function: the closed-form
  fit for how matter physics filters the shape of the primordial power
  spectrum when baryon features are neglected.  The q map is the
  single-plane Limber step — it converts an angular separation, at one
  effective lens plane set by z_eff and u_star, into the wavenumber
  ratio the transfer function is evaluated at.  As is the primordial
  amplitude and ns the spectral tilt; each cosmology uses its own ns,
  and R = 1 for a row equal to the reference cosmology.  Theta = the
  CMB temperature in units of 2.7 K, the EH fit's convention.

  One formula supports two array libraries. NumPy is used by the
  analysis and plotting wrappers. Torch is used by the training loop
  with tensors already placed on the selected device. The arithmetic
  is shared, while the logarithm and input conversion differ.

  theta_arcmin and z_eff broadcast to the element shape S.
  Broadcasting is the numpy/torch rule that stretches axes of length 1
  to match the other operand, so both can be (n_keep,) for the masked
  data vector, or (ntheta,1,1) and (1,nt,nt) for the full xi matrix —
  the two spellings expand to the same grid without copying data.

  Arguments:
    theta_arcmin = per-element angular scale(s) [arcmin].
    z_eff        = per-element effective source redshift(s), e.g.
                   min(z_i, z_j) for a tomographic pair.
    cosmo        = (N, n_param) rows to rescale. A Torch tensor keeps
                   its dtype and device. A NumPy input is converted to
                   float64 before the calculation.
    cosmo_mid    = (n_param,) reference ("mid") cosmology; R=1
                   for a row equal to it.
    names        = parameter column names (pgeom.names order);
                   locate As_1e9 / ns / H0 / omegam.
    u_star       = lens position (kernel peak) in the theta -> q
                   map; ~0.5.
    include_amp  = if True, also multiply R by the surviving
                   geometric-amplitude factor N ~ (Omega_m
                   h^2)^ns / h (a second amplitude direction
                   beyond A_s). The standard run sets it True;
                   off by default.

  Returns:
    R = (N, *S). Torch input returns a Torch tensor with the input
        dtype and device. NumPy input returns a float64 NumPy array.
  """
  # pick the array library once: torch when cosmo is a tensor
  # (on-device path), numpy otherwise. log is the only math call
  # that differs; coerce casts the geometry arrays into cosmo's
  # library/dtype/device to broadcast.
  is_torch = torch.is_tensor(cosmo)
  if is_torch:
    log = torch.log
    def coerce(a):
      """Match one constant to the input's tensor dtype and device."""
      return torch.as_tensor(a, dtype=cosmo.dtype, device=cosmo.device)
    # a lone 1D row -> (1, n_param): the tensor np.atleast_2d, so
    # the [:, col] indexing below works.
    if cosmo.ndim == 1:
      cosmo = cosmo[None, :]
  else:
    log = np.log
    def coerce(a):
      """Match one constant to the numpy float64 the branch computes in."""
      return np.asarray(a, dtype="float64")
    cosmo = np.atleast_2d(
      np.asarray(cosmo, dtype="float64"))
  mid = coerce(cosmo_mid)

  iA = names.index("As_1e9")
  iN = names.index("ns")
  iH = names.index("H0")
  iO = names.index("omegam")
  As   = cosmo[:, iA]
  ns   = cosmo[:, iN]
  Gam  = cosmo[:, iO] * (cosmo[:, iH] / 100.0)
  As_m, ns_m = mid[iA], mid[iN]
  Gam_m = mid[iO] * (mid[iH] / 100.0)

  Theta2 = (2.725 / 2.7) ** 2
  C_KMS  = 2.99792458e5
  K      = 100.0 * Theta2 / (C_KMS * u_star)
  th_rad = coerce(theta_arcmin) * (np.pi / (180.0 * 60.0))
  base   = K / (th_rad * coerce(z_eff))    # element shape S

  # flatten the elements to one axis so cosmo (N) broadcasts
  # cleanly, then restore the element shape at the end. S is a
  # plain tuple so (N,) + S works for both libraries.
  S     = tuple(base.shape)
  flat  = base.reshape(-1)                   # (n_elem,)
  # flat[None, :] -> (1, n_elem); Gam[:, None] -> (N, 1). Dividing
  # broadcasts to the full (N, n_elem) grid: every cosmology's
  # Gamma against every element's base wavenumber. ([None, :] and
  # [:, None] are the numpy/torch spelling of unsqueeze.)
  q     = flat[None, :] / Gam[:, None]       # (N, n_elem)
  q_mid = flat[None, :] / Gam_m              # (1, n_elem)

  def T(qq):
    """Eisenstein-Hu zero-baryon transfer function T(q) (1998).

    Arguments:
      qq = the shape variable q = k / Gamma, per row and element.

    Returns:
      T(q) = L / (L + C q^2), with L = ln(2e + 1.8 q) and
      C = 14.2 + 731 / (1 + 62.5 q).
    """
    L = log(2.0 * np.e + 1.8 * qq)
    C = 14.2 + 731.0 / (1.0 + 62.5 * qq)
    return L / (L + C * qq * qq)

  shape     = q ** ns[:, None] * T(q) ** 2
  shape_mid = q_mid ** ns_m * T(q_mid) ** 2
  R = (As_m / As)[:, None] * shape_mid / shape  # (N, n_elem)

  if include_amp:
    # surviving geometric amplitude N ~ (Om h^2)^ns / h: the z_s
    # and theta parts cancelled in the ratio, leaving a
    # per-cosmology scalar (second amplitude direction).
    wm   = cosmo[:, iO] * (cosmo[:, iH] / 100.0) ** 2
    wm_m = mid[iO] * (mid[iH] / 100.0) ** 2
    h    = cosmo[:, iH] / 100.0
    h_m  = mid[iH] / 100.0
    amp_mid = wm_m ** ns_m / h_m
    amp     = wm ** ns / h
    R = R * (amp_mid / amp)[:, None]

  return R.reshape((cosmo.shape[0],) + S)        # (N, *S)


def analytic_shape_ratio(cosmo,
                         cosmo_mid,
                         names,
                         theta_kept,
                         zsrc_i,
                         zsrc_j,
                         u_star=0.5,
                         include_amp=False):
  """Cosmic-shear rescaling R over the masked (kept) data vector.

  The emulator-pipeline wrapper: column k of R aligns with kept
  element dest_idx[k] of the squeezed data vector (the geometry's
  squeeze keeps only the unmasked entries).  Multiply the squeezed dv
  by R to preprocess; divide it back out before the chi2.  The
  effective redshift of each kept element is min(zsrc_i, zsrc_j),
  because the shear signal of a tomographic bin pair is dominated by
  the nearer of the two source planes.  Calls _analytic_R with the
  kept-element geometry.

  Arguments:
    cosmo       = (N, n_param) rows to rescale.
    cosmo_mid   = (n_param,) reference cosmology.
    names       = parameter names (pgeom.names).
    theta_kept  = (n_keep,) angular scale per kept element
                  [arcmin] (geom.theta_kept).
    zsrc_i      = (n_keep,) first source redshift per kept pair
                  (geom.zsrc_i).
    zsrc_j      = (n_keep,) second source redshift (geom.zsrc_j);
                  pair = min(zsrc_i, zsrc_j).
    u_star      = kernel peak ~0.5.
    include_amp = forwarded to _analytic_R (the (Omega_m
                  h^2)^ns / h amplitude factor).
  Returns:
    R = (N, n_keep) float64.
  """
  z_eff = np.minimum(zsrc_i, zsrc_j)
  return _analytic_R(theta_arcmin=theta_kept,
                     z_eff=z_eff,
                     cosmo=cosmo,
                     cosmo_mid=cosmo_mid,
                     names=names,
                     u_star=u_star,
                     include_amp=include_amp)


def rescale_xi(xi,
               cosmo,
               cosmo_mid,
               names,
               z_src,
               u_star=0.5,
               include_amp=True):
  """Rescale a list of xi curves by R in the (theta, xip, xim) layout.

  The plotting/visual-check wrapper: it calls _analytic_R with the
  full-block matrix geometry — every tomographic bin pair at every
  theta, no mask applied — so a whole xi curve can be flattened for
  inspection.  R is strictly positive, so xi- keeps its sign; R = 1
  for a curve equal to cosmo_mid.  The effective redshift of a cross
  pair is min(z_i, z_j), the nearer source plane.

  Arguments:
    xi          = list of (theta, xip, xim); xip/xim are
                  (ntheta, ntomo, ntomo), theta [arcmin].
    cosmo       = (len(xi), n_param) params, one row per curve.
    cosmo_mid   = (n_param,) reference cosmology.
    names       = parameter names (pgeom.names).
    z_src       = (ntomo,) source-bin peak redshifts (geom.z_src).
    u_star      = kernel peak ~0.5.
    include_amp = forwarded to _analytic_R (the (Omega_m
                  h^2)^ns / h amplitude factor).
  Returns:
    a new list of (theta, xip*R, xim*R).
  """
  z     = np.asarray(z_src)
  z_eff = np.minimum(z[:, None], z[None, :])     # (nt, nt)
  theta = np.asarray(xi[0][0])                   # (ntheta,)
  # R[k] for curve k: (ntheta, ntomo, ntomo).
  R = _analytic_R(theta_arcmin=theta[:, None, None],
                  z_eff=z_eff[None, :, :],
                  cosmo=cosmo,
                  cosmo_mid=cosmo_mid,
                  names=names,
                  u_star=u_star,
                  include_amp=include_amp)
  out = []
  for k, (th, xip, xim) in enumerate(xi):
    out.append((th, xip * R[k], xim * R[k]))
  return out
