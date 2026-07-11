"""One home for the syren analytic P(k) base (D-MP2-A(3)).

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

import numpy as np

# the formulas are VENDORED in-repo (syren/, numpy-only — provenance
# and the import-only deviations in syren/README.md), so the imports
# are unconditional: no pip install, no version drift against the
# base the artifacts were trained on.
from syren.linear import (plin_emulated, get_approximate_D,
                          growth_correction_R, As_to_sigma8)
from syren.syrenhalofit import run_halofit_vec


def syren_params_from(params):
  """Read the six syren-base arguments from a resolved parameter dict.

  The base formulas need (As_1e9, ns, H0, Ob, Om, w0, wa). One mapping
  rule, shared by the dump generator and the emul_mps adapter (so the
  two can never disagree about which columns feed the base):

    As_1e9 = params["As_1e9"] if present, else params["As"] * 1e9
             (the linear amplitude either way; a missing amplitude is
             loud).
    ns / H0 / omegab / omegam = read by exactly those names, loud when
             absent (the base cannot be formed without them).
    w0     = params["w"] if present, else params["w0"], else -1.0 —
             an ABSENT equation-of-state parameter means the sampled
             model IS LCDM (a model fact, not a config default).
    wa     = params["wa"] if present, else 0.0 (same reasoning).

  Arguments:
    params = a mapping of resolved parameter values (the generator's
             to_input dict / the adapter's sampled point).

  Returns:
    (As_1e9, ns, H0, Ob, Om, w0, wa) as floats.

  Raises:
    KeyError naming the missing required name(s).
  """
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
  if "w" in params:
    w0 = float(params["w"])
  elif "w0" in params:
    w0 = float(params["w0"])
  else:
    w0 = -1.0
  wa = float(params["wa"]) if "wa" in params else 0.0
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
