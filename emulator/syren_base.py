"""One home for the syren analytic P(k) base.

The MPS emulators CORRECT an approximate formula: the network target is
log(P / P_base), where P_base comes from the syren (symbolic_pofk)
formulas, VENDORED in-repo under syren/. This module is the base's ONLY
definition — the dump generator (which writes the base beside the raw
dump), the emul_mps adapter (which multiplies it back at inference),
and the gates all call these two functions, so the formula the emulator
corrects can never fork between them.

The math is the legacy emulmps_w0wa.py verbatim (the porting
discipline; the calls' unit conventions are load-bearing):

    base_pklin:  k [1/Mpc] -> k/h [h/Mpc] for syren; plin_emulated at
                 z = 0 in (Mpc/h)^3; rescaled to each z by the
                 approximate growth (D_z/D_0)^2 * (R_z/R_0) at
                 k_ref = 1e-4 (mnu fixed at 0.06 — the legacy
                 convention); divided by h^3 -> Mpc^3.
    base_boost:  sigma8 from As_to_sigma8, then syren-halofit's
                 run_halofit_vec with return_boost=True and the LINEAR
                 P handed back in (Mpc/h)^3 (Plin_in = P_lin * h^3).

PS: As_1e9 = the primordial amplitude in units of 1e-9 (A_s = 2.1e-9
-> As_1e9 = 2.1), the convention the syren formulas take; the boost =
P_nonlinear / P_linear, the ratio the second MPS artifact emulates.
"""

import numbers

import numpy as np

# the formulas are VENDORED in-repo (syren/, numpy-only — provenance
# and the import-only deviations in syren/README.md), so the imports
# are unconditional: no pip install, no version drift against the
# base the artifacts were trained on.
from syren.linear import (plin_emulated, get_approximate_D,
                          growth_correction_R, As_to_sigma8)
from syren.syrenhalofit import run_halofit_vec


DARK_ENERGY_COORDINATE_ATOL = float(4.0 * np.finfo(np.float32).eps)
_DARK_ENERGY_LAWS = (
  "w0wa-cpl",
  "constant-w",
  "cosmological-constant",
)


def _dark_energy_scalar(params, name, where):
  """Read one finite real scalar without Boolean or array coercion."""
  value = params[name]
  if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
    raise TypeError(
      where + ": " + repr(name)
      + " must be a finite real scalar and not a Boolean; got "
      + repr(value) + " (type " + type(value).__name__ + ")")
  try:
    value = float(value)
  except (OverflowError, TypeError, ValueError) as error:
    raise ValueError(
      where + ": " + repr(name)
      + " cannot be represented as a finite float; got " + repr(value)) \
      from error
  if not np.isfinite(value):
    raise ValueError(
      where + ": " + repr(name) + " must be finite; got " + repr(value))
  return value


def _dark_energy_close(left, right):
  """Compare two coordinates under the one float32 storage tolerance."""
  return bool(np.isclose(
    left, right, rtol=0.0, atol=DARK_ENERGY_COORDINATE_ATOL))


def resolve_dark_energy_coordinates(
    params, *, dark_energy_law=None, where="dark-energy coordinates"):
  """Resolve the canonical ``(w0, wa)`` pair used by the Syren base.

  Two complete coordinate forms are accepted without an additional law:
  ``(w or w0, wa)`` and ``(w or w0, w0pwa)``.  In the second form,
  ``wa = w0pwa - w0``.  Every redundant value is still checked.  In
  particular, ``w`` and ``w0`` must agree before any missing-coordinate
  refusal, and a supplied ``w0pwa`` must equal ``w0 + wa``.

  An incomplete form can be completed only by an explicit persisted law.
  ``constant-w`` supplies ``wa = 0`` but still needs the present-day value
  through ``w``, ``w0``, or ``w0pwa``.  ``cosmological-constant`` supplies
  ``(w0, wa) = (-1, 0)`` and checks any redundant coordinates against that
  pair.  ``w0wa-cpl`` supplies neither coordinate.  Absence and ``None`` never
  silently select a law.

  Coordinates are stored in float32 data products.  Redundant forms therefore
  use zero relative tolerance and the single public absolute tolerance
  ``DARK_ENERGY_COORDINATE_ATOL``.  Each individual value must nevertheless be
  a finite Python or NumPy real scalar. Boolean, string, array, complex, NaN,
  and infinity inputs are refused rather than coerced.

  Arguments:
    params = mapping that may contain ``w``, ``w0``, ``wa``, and ``w0pwa``.
    dark_energy_law = optional explicit law: ``w0wa-cpl``, ``constant-w``, or
                      ``cosmological-constant``.
    where = context included in refusal messages.

  Returns:
    ``(w0, wa)`` as two native floats.

  Raises:
    TypeError for a non-scalar coordinate or non-native law value. ValueError
    for an unknown law, nonfinite value, inconsistent redundant form, a
    coordinate incompatible with the explicit law, or insufficient information.
  """
  if type(where) is not str or not where:
    raise TypeError("dark-energy resolver where must be nonempty native text")
  if dark_energy_law is not None:
    if type(dark_energy_law) is not str:
      raise TypeError(
        where + ": dark_energy_law must be native text or None; got "
        + repr(dark_energy_law))
    if dark_energy_law not in _DARK_ENERGY_LAWS:
      raise ValueError(
        where + ": dark_energy_law must be one of "
        + repr(_DARK_ENERGY_LAWS) + "; got " + repr(dark_energy_law))

  supplied = {}
  for name in ("w", "w0", "wa", "w0pwa"):
    if name in params:
      supplied[name] = _dark_energy_scalar(params, name, where)

  # Alias agreement is the first coordinate relationship checked.  A caller
  # with conflicting aliases receives that precise refusal even when wa is
  # also missing.
  if "w" in supplied and "w0" in supplied:
    if not _dark_energy_close(supplied["w"], supplied["w0"]):
      raise ValueError(
        where + ": aliases 'w'=" + repr(supplied["w"])
        + " and 'w0'=" + repr(supplied["w0"])
        + " disagree beyond absolute tolerance "
        + repr(DARK_ENERGY_COORDINATE_ATOL) + " (relative tolerance is zero)")

  w0 = supplied.get("w", supplied.get("w0"))
  wa = supplied.get("wa")
  w0pwa = supplied.get("w0pwa")

  if dark_energy_law == "cosmological-constant":
    expected_w0 = -1.0
    expected_wa = 0.0
    if w0 is not None and not _dark_energy_close(w0, expected_w0):
      raise ValueError(
        where + ": cosmological-constant requires w0=-1, but 'w'/'w0' is "
        + repr(w0))
    if wa is not None and not _dark_energy_close(wa, expected_wa):
      raise ValueError(
        where + ": cosmological-constant requires wa=0, but 'wa' is "
        + repr(wa))
    if w0pwa is not None and not _dark_energy_close(
        w0pwa, expected_w0 + expected_wa):
      raise ValueError(
        where + ": cosmological-constant requires w0pwa=-1, but 'w0pwa' is "
        + repr(w0pwa))
    return expected_w0, expected_wa

  if dark_energy_law == "constant-w":
    if wa is not None and not _dark_energy_close(wa, 0.0):
      raise ValueError(
        where + ": constant-w requires wa=0, but 'wa' is " + repr(wa))
    if w0 is None:
      if w0pwa is None:
        raise ValueError(
          where + ": constant-w needs 'w', 'w0', or 'w0pwa' to state its "
          "present-day equation of state")
      w0 = w0pwa
    wa = 0.0

  if w0 is None:
    raise ValueError(
      where + ": cannot resolve w0; provide 'w' or 'w0' together with 'wa' "
      "or 'w0pwa', or provide the explicit cosmological-constant law")

  if wa is None:
    if w0pwa is not None:
      wa = w0pwa - w0
    elif dark_energy_law != "constant-w":
      raise ValueError(
        where + ": cannot resolve wa from 'w'/'w0' alone. Provide 'wa' or "
        "'w0pwa', or explicitly declare dark_energy_law='constant-w'")

  if w0pwa is not None and not _dark_energy_close(w0pwa, w0 + wa):
    raise ValueError(
      where + ": 'w0pwa'=" + repr(w0pwa) + " disagrees with w0 + wa="
      + repr(w0 + wa) + " beyond absolute tolerance "
      + repr(DARK_ENERGY_COORDINATE_ATOL) + " (relative tolerance is zero)")

  return float(w0), float(wa)


def syren_params_from(params, *, dark_energy_law=None):
  """Read the seven Syren-base arguments from resolved parameters.

  The base formulas need ``As_1e9``, ``ns``, ``H0``, ``Ob``, ``Om``,
  ``w0``, and ``wa``. The dump generator and the matter-power adapter
  use this one mapping from named parameters to the formula arguments.

    As_1e9 = params["As_1e9"] if present, else params["As"] * 1e9
             (the linear amplitude either way; a missing amplitude is
             loud).
    ns / H0 / omegab / omegam = read by exactly those names, loud when
             absent (the base cannot be formed without them).
    w0 / wa = resolved together by ``resolve_dark_energy_coordinates``.
              Complete CPL coordinates need no additional law. Incomplete
              constant-w or cosmological-constant coordinates require the
              explicit persisted ``dark_energy_law``; absence never silently
              selects ``wa = 0``.

  Arguments:
    params = a mapping of resolved parameter values. During generation
             this is the provider's input dictionary. During inference
             it is the sampled cosmological point.
    dark_energy_law = optional persisted law passed to the shared coordinate
             resolver. Complete coordinate forms remain accepted when it is
             omitted; incomplete forms do not.

  Returns:
    ``(As_1e9, ns, H0, Ob, Om, w0, wa)`` as seven floats. ``H0`` is in
    km s^-1 Mpc^-1. ``Ob`` and ``Om`` are dimensionless density
    fractions. The remaining entries are dimensionless.

  Raises:
    KeyError naming missing non-dark-energy inputs. TypeError or ValueError
    from the shared dark-energy resolver for an invalid, inconsistent, or
    incomplete coordinate description.
  """
  w0, wa = resolve_dark_energy_coordinates(
    params, dark_energy_law=dark_energy_law,
    where="syren_params_from dark-energy coordinates")

  missing = []
  if "As_1e9" in params:
    as_1e9 = float(params["As_1e9"])
  elif "As" in params:
    as_1e9 = float(params["As"]) * 1e9
  else:
    missing.append("As_1e9 (or As)")
  for name in ("ns", "H0", "omegab", "omegam"):
    if name not in params:
      missing.append(name)
  if missing:
    raise KeyError(
      "syren_params_from: the syren base needs parameter(s) "
      + repr(missing) + " among the resolved inputs; the run's params "
      "block must provide them by these names (materialize omegab / "
      "omegam as inputs if the YAML samples densities another way)")
  return (as_1e9, float(params["ns"]), float(params["H0"]),
          float(params["omegab"]), float(params["omegam"]), w0, wa)


def base_pklin(k_mpc, z, As_1e9, ns, H0, Ob, Om, w0=-1.0, wa=0.0,
               mnu=0.06):
  """The syren linear P(k, z) base, in Mpc^3 (legacy math verbatim).

  Arguments:
    k_mpc  = (nk,) wavenumbers in 1/Mpc.
    z      = (nz,) redshifts (the growth rescaling's targets).
    As_1e9 = the primordial amplitude in 1e-9 units (2.1 for LCDM).
    ns     = the spectral index.
    H0     = the Hubble constant, km/s/Mpc.
    Ob     = Omega_b.
    Om     = Omega_m.
    w0, wa = the dark-energy equation of state (defaults LCDM).
    mnu    = the neutrino mass the growth formulas take (the legacy
             fixes 0.06; exposed, never silently different).

  Returns:
    (nz, nk) the syren linear P(k, z) in Mpc^3.
  """
  k_mpc = np.asarray(k_mpc, dtype="float64")
  z     = np.asarray(z, dtype="float64")
  h   = float(H0) / 100.0
  k_h = k_mpc / h
  a_array = 1.0 / (1.0 + z)

  pk_fid = plin_emulated(k_h, Om, Ob, h, ns, As=As_1e9, w0=w0, wa=wa)

  kref = 1e-4
  D0 = get_approximate_D(k=kref, As=As_1e9, Om=Om, Ob=Ob, h=h, ns=ns,
                         mnu=mnu, w0=w0, wa=wa, a=1)
  Dz = get_approximate_D(k=kref, As=As_1e9, Om=Om, Ob=Ob, h=h, ns=ns,
                         mnu=mnu, w0=w0, wa=wa, a=a_array)
  R0 = growth_correction_R(As=As_1e9, Om=Om, Ob=Ob, h=h, ns=ns,
                           mnu=mnu, w0=w0, wa=wa, a=1)
  Rz = growth_correction_R(As=As_1e9, Om=Om, Ob=Ob, h=h, ns=ns,
                           mnu=mnu, w0=w0, wa=wa, a=a_array)

  growth = (Dz / D0) ** 2 * (Rz / R0)
  return ((pk_fid * growth[:, None]) / h ** 3).astype("float64")


def base_boost(k_mpc, z, pk_lin_mpc, As_1e9, ns, H0, Ob, Om,
               w0=-1.0, wa=0.0, mnu=0.06):
  """The syren-halofit nonlinear boost base B(k, z) (legacy verbatim).

  Arguments:
    k_mpc      = (nk,) wavenumbers in 1/Mpc.
    z          = (nz,) redshifts.
    pk_lin_mpc = (nz, nk) the LINEAR P(k, z) in Mpc^3 the boost
                 multiplies (handed to syren-halofit in (Mpc/h)^3, the
                 legacy Plin_in convention).
    As_1e9 / ns / H0 / Ob / Om / w0 / wa / mnu = as base_pklin.

  Returns:
    (nz, nk) the syren-halofit boost (dimensionless).
  """
  k_mpc = np.asarray(k_mpc, dtype="float64")
  z     = np.asarray(z, dtype="float64")
  h   = float(H0) / 100.0
  k_h = k_mpc / h
  a_array = 1.0 / (1.0 + z)

  sigma8 = As_to_sigma8(As_1e9, Om, Ob, h, ns, mnu, w0, wa)
  boost = run_halofit_vec(k_h, sigma8, Om, Ob, h, ns, a_array,
                          return_boost=True,
                          Plin_in=np.asarray(pk_lin_mpc,
                                             dtype="float64") * h ** 3)
  return np.asarray(boost, dtype="float64")
