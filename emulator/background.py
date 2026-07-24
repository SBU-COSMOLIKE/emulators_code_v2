"""Imposed background physics: H(z) -> the cosmological distances.

H(z) is the Hubble rate: the universe's expansion rate at redshift z,
in km/s/Mpc.  The background family keeps the design rule of the
legacy emulbaosn code (the ancestor implementation whose conventions
this module preserves verbatim): only H(z) comes from a neural
network; every distance is known physics computed from it by the
pipeline below.  This module owns that pipeline once, and both
consumers read it here — the cobaya adapter (cobaya is the sampling
framework; the emul_baosn theory class presents this emulator to it)
and direct scripting on an EmulatorPredictor (the class that wraps a
trained model for evaluation).  With one owner the integration
convention can never fork between the two doors.

    H(z) on the stored grid (the emulator output, km/s/Mpc)
       │  c/H cubic-interpolated onto the DOUBLED grid
       │  z_step = linspace(0, z_max, 2*NZ + 1)   (odd point count)
       ▼
    chi(z) = integral_0^z c/H dz'      cumulative Simpson (composite
       │ even + one-interval odd)
       ▼
    flat conversions:  D_C = chi           comoving distance
                       D_A = chi / (1+z)   angular diameter
                       D_L = chi * (1+z)   luminosity

(legend: NZ = the stored grid's point count; z_max = its last redshift;
c = the speed of light in km/s, so every distance is in Mpc when H is
in km/s/Mpc.  The comoving distance chi is the distance measured on
today's spatial grid, unaffected by expansion; the angular-diameter
distance relates an object's physical size to the angle it spans on
the sky; the luminosity distance relates its intrinsic brightness to
the flux we receive.)

The pipeline is valid STRICTLY inside the grid window [0, z_max].  The
legacy analytic extension toward z = 1200 (H_ext = H0*sqrt(om(1+z)^3 +
omegar(1+z)^4), self-labeled "this is an approximation") is
deliberately not ported: the recombination window — redshifts near
1100, where the cosmic microwave background formed — has its own
trained distance emulator, so no bridging integration through the
unsampled redshift range between the two windows exists anywhere.
Curvature support is flat-only: the legacy curved-space branch is
dimensionally wrong (sinh(chi*K)/K where the correct form needs
sqrt(K); it agreed with flat space only because sinh(x) ~ x for small
x), and the corrected form is a recorded future item behind a
CAMB-comparison gate.  The window and flatness guards live in the
adapter, which knows the query and the windows; this module is the
pure math.
"""

import numpy as np
from scipy import interpolate

# the speed of light in km/s — the legacy constant, verbatim, so chi is
# in Mpc when H is in km/s/Mpc.
C_KMS = 2.99792458e5


def cumulative_simpson(z, y):
  """Cumulative Simpson integral of y over the uniform grid z.

  Simpson's rule integrates a sampled function by fitting a parabola
  through three consecutive samples — two grid intervals — and
  integrating that parabola exactly; summing over successive pairs of
  intervals gives the composite rule, which is why the grid must have
  an even number of intervals, meaning an odd number of points.  This
  routine returns the RUNNING integral: C[i] approximates
  integral_{z[0]}^{z[i]} y dz at every grid point.  At the even points
  it is the composite Simpson sum; at each odd point it adds the
  ONE-interval integral of the parabola through the three surrounding
  samples, h/12 * (5*y[i-1] + 8*y[i] - y[i+1]).  The result is exact
  on cubic polynomials at the even nodes and on quadratics at the odd
  nodes.

  A tempting wrong form for the odd node is dz/6 * (y[i-1] + 4*y[i] +
  y[i+1]): that is HALF the two-interval Simpson total (the integral over
  the whole [z[i-1], z[i+1]] chunk), not the one-interval integral, and
  it carries a first-order h^2/2 error at every odd node. The one-interval
  rule above is the correct form; see families-background-mps.md.

  Arguments:
    z = (n,) UNIFORM ascending grid; n must be odd (an even number of
        intervals), n >= 3.
    y = (n,) integrand samples on z.

  Returns:
    (n,) the cumulative integral, C[0] = 0.

  Raises:
    ValueError on an even point count (Simpson needs interval pairs).
  """
  n = len(z)
  if n < 3 or (n - 1) % 2 != 0:
    raise ValueError("Need an odd number of points (even number of intervals).")
  dz = z[1] - z[0]

  # Simpson contributions on each pair of intervals [z[2m], z[2m+2]]
  # there are (n-1)/2 such chunks
  f0 = y[:-2:2]    # y[0], y[2], y[4], …
  f1 = y[1:-1:2]   # y[1], y[3], y[5], …
  f2 = y[2::2]     # y[2], y[4], y[6], …
  chunks = dz/3 * (f0 + 4*f1 + f2)

  # cumulative sum of the full-chunk integrals at even indices
  cum_even = np.concatenate(([0.0], np.cumsum(chunks)))

  # build the full cumulative array
  C = np.empty_like(y)
  C[0] = 0.0
  C[2::2] = cum_even[1:]               # at z[2], z[4], …
  # odd node i: the ONE-interval integral integral_{z[i-1]}^{z[i]} of the
  # quadratic through the three samples (y[i-1], y[i], y[i+1]) — exact on
  # quadratics. h/12 * (5*y[i-1] + 8*y[i] - y[i+1]). NOT dz/6 * (1,4,1),
  # which is HALF the two-interval Simpson total (integral over the whole
  # [z[i-1], z[i+1]] chunk) and a first-order h^2/2 error; see
  # families-background-mps.md.
  for i in range(1, n, 2):
    C[i] = C[i - 1] + dz / 12 * (5 * y[i - 1] + 8 * y[i] - y[i + 1])
  return C


def comoving_distance_grid(z_grid, h_grid):
  """chi(z) over the doubled grid from H(z) on the stored grid (flat).

  In a flat universe the comoving distance is the integral
  chi(z) = integral_0^z c/H(z') dz'.  Following the legacy convention
  verbatim, c/H is first cubic-interpolated — approximated by piecewise
  third-degree polynomials through the samples — onto the finer grid
  z_step = linspace(0, z_max, 2*NZ + 1).  That interpolation step
  exists for three reasons: the stored grid is not required to be
  uniform while Simpson's rule assumes one step size, linspace
  guarantees the odd point count Simpson needs, and doubling refines
  the integration step.  The grid starts at z = 0 so chi(0) = 0, and
  the interpolated integrand is then cumulative-Simpson integrated.

  Arguments:
    z_grid = (NZ,) the stored ascending redshift grid. It starts exactly at
             zero and z_grid[-1] is the window edge z_max.
    h_grid = (NZ,) H(z_grid) in km/s/Mpc.

  Returns:
    (z_step, chi): the doubled grid and the comoving distance on it,
    both (2*NZ + 1,), chi in Mpc.
  """
  z_grid = np.asarray(z_grid, dtype="float64")
  h_grid = np.asarray(h_grid, dtype="float64")
  grid_is_valid = (
    z_grid.ndim == 1 and len(z_grid) >= 4
    and np.isfinite(z_grid).all() and z_grid[0] == 0.0
    and (z_grid[1:] > z_grid[:-1]).all())
  if not grid_is_valid:
    raise ValueError(
      "the Hubble redshift grid must be finite, strictly increasing, "
      "one-dimensional, start exactly at z = 0, and have at least 4 "
      "points (the cubic interpolation needs them)")
  if h_grid.shape != z_grid.shape or not np.isfinite(h_grid).all() \
      or not (h_grid > 0.0).all():
    raise ValueError(
      "H(z) must be one finite positive value for every redshift")

  func = interpolate.interp1d(z_grid, C_KMS / h_grid,
                              kind='cubic',
                              assume_sorted=True,
                              fill_value="extrapolate")
  zstep = np.linspace(0.0, z_grid[-1], 2 * len(z_grid) + 1)
  chi = cumulative_simpson(zstep, func(zstep))
  return zstep, chi


def distance_interpolators(z_grid, h_grid):
  """Build the SN-window background interpolators from one H(z) row.

  SN is supernova: this serves the low-redshift window where the
  supernova and baryon-acoustic-oscillation data live.  One call per
  sampled cosmology: it integrates chi (comoving_distance_grid) and
  wraps H and the three flat distances in cubic interpolators —
  functions built from the grid samples that return a value at any
  redshift between them.  Every consumer (the cobaya getters, a
  profile script, the diagnostics pages) reads these, so the
  convention exists once.

  Arguments:
    z_grid = (NZ,) the stored ascending redshift grid.
    h_grid = (NZ,) H(z_grid) in km/s/Mpc.

  Returns:
    a dict with:
      H     = cubic interpolator z -> H(z)  (km/s/Mpc)
      chi   = cubic interpolator z -> comoving distance D_C (Mpc)
      da    = cubic interpolator z -> angular-diameter distance (Mpc)
      dl    = cubic interpolator z -> luminosity distance (Mpc)
      z_max = the window edge (queries beyond it are the ADAPTER's
              loud-error job, not silent extrapolation).
  """
  zstep, chi = comoving_distance_grid(z_grid=z_grid, h_grid=h_grid)
  # flat conversions (legacy verbatim): dl = chi*(1+z), da = dl/(1+z)^2.
  dl = chi * (1.0 + zstep)
  da = dl / (1.0 + zstep) ** 2

  def cubic(x, y):
    """Build one cubic interpolant over the grid (extrapolation allowed).

    ``fill_value="extrapolate"`` makes scipy continue the outermost
    polynomial pieces beyond the grid instead of raising.  That is safe
    here because refusing an out-of-window query is deliberately the
    adapter's job: only the adapter knows where a query came from and
    which window should answer it.

    Arguments:
      x = the strictly increasing coordinate grid.
      y = the values on that grid.

    Returns:
      a scipy interp1d callable y(x).
    """
    return interpolate.interp1d(x, y,
                                kind='cubic',
                                assume_sorted=True,
                                fill_value="extrapolate")

  return {"H":     cubic(z_grid, np.asarray(h_grid)),
          "chi":   cubic(zstep, chi),
          "da":    cubic(zstep, da),
          "dl":    cubic(zstep, dl),
          "z_max": float(z_grid[-1])}
