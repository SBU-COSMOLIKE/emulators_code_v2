"""Thin cobaya Theory adapter: matter power spectra from saved grid2d
emulators (the MPS family — the EMUL2 hybrid-inference provider).

This adapter contains no trainable network. ``EmulatorPredictor`` owns the
learned prediction, while this file owns the deterministic reconstruction
that turns two saved grid2d predictions into the matter-power products used
by CosmoLike's hybrid mode. It evaluates the stored Syren base, forms
``P_lin = exp(r_lin) P_base``, forms the nonlinear boost
``B = exp(r_boost) B_base``, applies the low-k boost rule, computes
``P_nl = B P_lin``, and provides sigma8. The two artifacts are:

    the "pklin" artifact              the "boost" artifact
    log(P_lin / P_syren) on the       log(B / B_syren-halofit), same
    stored (z, k) grid                grid; B = P_nl / P_lin
       │                                 │
       │  x syren base (emulator/        │  x syren-halofit base (needs
       ▼  syren_base.py)                 ▼  the emulated P_lin) + the
    P_lin(k, z)  [Mpc^3]             low-k blend -> B(k, z)
                        \\               /
                         P_nl = B * P_lin

get_Pk_grid / get_Pk_interpolator serve both (nonlinear True/False)
through the CAMB-compatible PowerSpectrumInterpolator below (ported
verbatim from the legacy adapter, itself adapted from CAMB by Antony
Lewis), so a likelihood written against CAMB's provider reads the
emulated spectra unchanged. An artifact trained with law "none" skips
its base multiply (the artifact says so itself; nothing is re-declared
in the YAML).

PS: law space = what the network learned (log of the ratio to the syren
base under a syren law, the raw surface under "none"); the low-k blend
= the legacy convention boost -> 1 + (boost - 1) * (1 - exp(-(k/k_t)^n))
with k_t = 0.005 1/Mpc, n = 2, pinning the boost to exactly 1 on
linear scales (applied on the syren path, where the base construction
needs it — a law-none boost learned the raw low-k boost directly).
"""

import math
import numbers
import os
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline
from cobaya.theory import Theory
from cobaya.log import LoggedError, get_logger

# The adapter lives in <root>/cobaya_theory/; put <root> on sys.path so
# the training package `emulator` imports (the emul_cosmic_shear prepend).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.inference import (                               # noqa: E402
    EmulatorPredictor,
    check_artifacts_fixed_values,
    check_artifacts_pair_up,
)
from cobaya_theory._adapter_contract import (                   # noqa: E402
    exact_bool,
    pick_device,
    resolve_emulator_roots,
    validate_extra_args,
)
from emulator import syren_base                           # noqa: E402

# The only extra_args the schema-v2 convention accepts (the legacy
# model_file / metadata_file / nl_* / use_syren / param_order keys are
# retired: the h5 recipe + the stored grids/laws replace them).
_ALLOWED_EXTRA_ARGS = (
    "device", "emulators", "compile", "allow_k_extrapolation")

# the legacy low-k blend constants (verbatim): below k_t the boost is
# pinned to 1 (linear scales), with a sharpness-n exponential turn-on.
_BLEND_K_T = 0.005   # [1/Mpc]
_BLEND_N   = 2.0     # 1 = pure exponential, larger = sharper transition

# sigma8 uses a spherical top-hat whose conventional radius is 8 Mpc/h.  The
# saved matter-power axis is in 1/Mpc, so calculate passes h=H0/100 and the
# helper converts the radius to Mpc before forming kR.
_SIGMA8_RADIUS_MPC_OVER_H = 8.0

# A finite integral is not enough: a short or poorly sampled k grid can miss
# most of sigma8 while still returning a smooth positive number.  The helper
# checks the positive contribution per unit log(k).  Two adjacent half-decade
# bands at each edge estimate the missing tails, and two interlaced
# every-other-point integrals check the stored resolution.  These bounds limit
# the estimated
# omitted variance to 1e-5 and the observed quadrature change to 1e-3.
_SIGMA8_TAIL_VARIANCE_RTOL = 1.0e-5
_SIGMA8_QUADRATURE_RTOL = 1.0e-3
_SIGMA8_MAX_PANEL_FRACTION = 0.1
_SIGMA8_LOG_DECADE = math.log(10.0)

_DARK_ENERGY_NAMES = ("w", "w0", "wa", "w0pwa")
_DARK_ENERGY_LAWS = {
    "w0wa-cpl": ("w", "wa"),
    "constant-w": ("w",),
    "cosmological-constant": (),
}


def _require_finite_syren_parameter(params, name):
    """Return one finite real Syren input without Boolean coercion."""
    if name not in params:
        raise KeyError("emul_mps: missing Syren parameter " + repr(name))
    value = params[name]
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real) or not finite):
        raise ValueError(
            "emul_mps: Syren parameter " + repr(name)
            + " must be a finite real scalar and not a Boolean; got "
            + repr(value))
    return value


def _require_positive_sigma8_h0(value):
    """Return a finite positive H0 for the sigma-eight radius conversion."""
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real) or not finite
            or float(value) <= 0.0):
        raise ValueError(
            "emul_mps: H0 must be a finite, strictly positive real scalar "
            "and not a Boolean when sigma8 is requested; got " + repr(value))
    return float(value)


def _require_mps_surface(value, *, name, shape, positive=False):
    """Require one exact finite matter-power surface before composition."""
    array = np.asarray(value)
    expected = tuple(int(x) for x in shape)
    if array.shape != expected:
        raise ValueError(
            "emul_mps: " + name + " must have exact shape "
            + repr(expected) + ", got " + repr(array.shape)
            + "; broadcasting a row or column is not a valid grid")
    if not np.isfinite(array).all():
        return None
    if positive and not (array > 0.0).all():
        return None
    return array


def _valid_mps_axis(axis, *, positive=False):
    """Check a spline axis."""
    return (axis.ndim == 1 and len(axis) >= 4 and np.isfinite(axis).all()
            and np.all(axis[1:] > axis[:-1])
            and (not positive or (axis > 0).all()))


def _saved_dark_energy_value(fixed, name):
    """Return one numeric saved dark-energy fact, or ``None`` for ``n/a``."""
    value = fixed.get(name)
    if value in (None, "n/a"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "emul_mps: saved dark-energy fact " + repr(name)
            + " is not a finite number: " + repr(value)) from error
    if not math.isfinite(number):
        raise ValueError(
            "emul_mps: saved dark-energy fact " + repr(name)
            + " is not finite: " + repr(value))
    return number


def _dark_energy_contract(predictor, predictor_names, req, *, need_base):
    """Validate one artifact's coordinate law and adjust Cobaya requirements.

    A dropped ``w0pwa`` is a sampling coordinate used to define ``wa``.  It is
    not a value that a Cobaya Theory may request directly.  The adapter asks
    for the present-day value and ``wa``, then reconstructs every spelling the
    saved input geometry needs inside :meth:`calculate`.
    """
    facts = predictor.fixed_facts
    law = facts.get("dark_energy_law")
    declared_inputs = facts.get("dark_energy_inputs")
    dark_names = set(predictor_names).intersection(_DARK_ENERGY_NAMES)
    if not need_base and not dark_names:
        return law, {}, False
    if law not in _DARK_ENERGY_LAWS:
        raise ValueError(
            "emul_mps: the saved matter-power artifact has dark-energy law "
            + repr(law) + "; serving requires one explicit current law from "
            + repr(sorted(_DARK_ENERGY_LAWS))
            + ". Re-generate the dataset and retrain the emulator.")
    expected_inputs = list(_DARK_ENERGY_LAWS[law])
    if declared_inputs != expected_inputs:
        raise ValueError(
            "emul_mps: dark_energy_inputs " + repr(declared_inputs)
            + " disagree with saved law " + repr(law) + "; expected "
            + repr(expected_inputs)
            + ". Re-generate the dataset and retrain the emulator.")
    if "w0pwa" in dark_names and law != "w0wa-cpl":
        raise ValueError(
            "emul_mps: an artifact sampled w0pwa but records " + repr(law)
            + "; older transformed-coordinate records can silently erase "
            "nonzero wa. Re-generate the dataset and retrain the emulator.")
    if "wa" in dark_names and law != "w0wa-cpl":
        raise ValueError(
            "emul_mps: an artifact samples wa but records " + repr(law)
            + "; re-generate the dataset and retrain the emulator.")
    if law == "cosmological-constant" and dark_names:
        raise ValueError(
            "emul_mps: a cosmological-constant artifact cannot have sampled "
            "dark-energy coordinates " + repr(sorted(dark_names))
            + "; re-generate the dataset and retrain the emulator.")

    fixed_block = facts.get("cosmology_fixed")
    if type(fixed_block) is not dict:
        raise ValueError(
            "emul_mps: the artifact's cosmology_fixed record must be a "
            "mapping before dark-energy coordinates can be resolved")
    fixed = {}
    for name in ("w", "wa"):
        value = _saved_dark_energy_value(fixed_block, name)
        if value is not None:
            fixed[name] = value

    req.pop("w0pwa", None)
    if law in ("w0wa-cpl", "constant-w") and "w" not in fixed:
        alias = "w0" if "w0" in dark_names and "w" not in dark_names else "w"
        req[alias] = None
    if law == "w0wa-cpl" and "wa" not in fixed:
        req["wa"] = None
    return law, fixed, True


def _resolved_dark_energy_point(params, *, law, fixed, needed):
    """Return parameters with all four equivalent coordinate names filled."""
    point = dict(params)
    for name, value in fixed.items():
        if name in point:
            # The shared resolver will compare an explicit sampled value with
            # every redundant representation.  Do not overwrite that evidence.
            continue
        point[name] = value
    if not needed:
        return point
    w0, wa = syren_base.resolve_dark_energy_coordinates(
        point, dark_energy_law=law, where="emul_mps input")
    point.update({"w": w0, "w0": w0, "wa": wa, "w0pwa": w0 + wa})
    return point


########## Taken from cobaya/theories/cosmo/boltzmannbase.py ############
class PowerSpectrumInterpolator(RectBivariateSpline):
    r"""
    2D spline interpolation object (scipy.interpolate.RectBivariateSpline)
    to evaluate matter power spectrum as function of z and k.

    *This class is adapted from CAMB's own P(k) interpolator, by Antony Lewis;
    it's mostly interface-compatible with the original.*

    :param z: values of z for which the power spectrum was evaluated.
    :param k: values of k for which the power spectrum was evaluated.
    :param P_or_logP: Values of the power spectrum (or log-values, if logP=True).
    :param logP: if True (default: False), log of power spectrum are given and used
        for the underlying interpolator.
    :param logsign: if logP is True, P_or_logP is log(logsign*Pk)
    :param extrap_kmax: if set, use power law extrapolation beyond kmax up to
        extrap_kmax; useful for tails of integrals.
    """

    def __init__(self, z, k, P_or_logP, extrap_kmin=None, extrap_kmax=None, logP=False,
                 logsign=1):
        self.islog = logP
        z, k, P_or_logP = np.asarray(z), np.asarray(k), np.asarray(P_or_logP)
        if (z.ndim != 1 or k.ndim != 1
                or P_or_logP.shape != (len(z), len(k))):
            raise ValueError("Pk needs 1-D axes and a matching surface")
        self.logsign = logsign
        if (not _valid_mps_axis(z)
                or not _valid_mps_axis(k, positive=True)
                or not np.isfinite(P_or_logP).all()):
            raise ValueError(
                "Pk needs finite ordered axes, finite surface, and positive k")
        self.z, self.k = z, k
        for name, bound in (("extrap_kmin", extrap_kmin),
                            ("extrap_kmax", extrap_kmax)):
            if bound is not None and (
                    isinstance(bound, (bool, np.bool_))
                    or not isinstance(bound, numbers.Real)
                    or not math.isfinite(float(bound)) or bound <= 0):
                raise ValueError(name + " must be a finite positive number")
        if extrap_kmin is not None and extrap_kmin >= self.k[0]:
            raise ValueError("extrap_kmin must be below saved k range")
        if extrap_kmax is not None and extrap_kmax <= self.k[-1]:
            raise ValueError("extrap_kmax must be above saved k range")
        self.zmin, self.zmax = self.z[0], self.z[-1]
        self.extrap_kmin, self.extrap_kmax = extrap_kmin, extrap_kmax
        logk = np.log(self.k)
        if extrap_kmin is not None:
            if not logP:
                raise ValueError('extrap_kmin must use logP')
            logk = np.hstack(
                [np.log(extrap_kmin),
                 np.log(self.input_kmin) * 0.1 + np.log(extrap_kmin) * 0.9, logk])
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, 2:] = P_or_logP
            diff = (logPnew[:, 3] - logPnew[:, 2]) / (logk[3] - logk[2])
            delta = diff * (logk[2] - logk[0])
            logPnew[:, 0] = logPnew[:, 2] - delta
            logPnew[:, 1] = logPnew[:, 2] - delta * 0.9
            P_or_logP = logPnew
        if extrap_kmax is not None:
            if not logP:
                raise ValueError('extrap_kmax must use logP')
            logk = np.hstack(
                [logk, np.log(self.input_kmax) * 0.1 + np.log(extrap_kmax) * 0.9,
                 np.log(extrap_kmax)])
            logPnew = np.empty((P_or_logP.shape[0], P_or_logP.shape[1] + 2))
            logPnew[:, :-2] = P_or_logP
            diff = (logPnew[:, -3] - logPnew[:, -4]) / (logk[-3] - logk[-4])
            delta = diff * (logk[-1] - logk[-3])
            logPnew[:, -1] = logPnew[:, -3] + delta
            logPnew[:, -2] = logPnew[:, -3] + delta * 0.9
            P_or_logP = logPnew
        super().__init__(self.z, logk, P_or_logP)

    @property
    def input_kmin(self):
        return self.k[0]

    @property
    def input_kmax(self):
        return self.k[-1]

    @property
    def kmin(self):
        if self.extrap_kmin is None:
            return self.input_kmin
        return self.extrap_kmin

    @property
    def kmax(self):
        if self.extrap_kmax is None:
            return self.input_kmax
        return self.extrap_kmax

    def check_ranges(self, z, k):
        z = np.atleast_1d(z).flatten()
        if not z.size or not np.isfinite(z).all():
            raise LoggedError(get_logger(self.__class__.__name__),
                              "z query must be nonempty and finite")
        min_z, max_z = min(z), max(z)
        if min_z < self.zmin:
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to z={min(z)} "
                              f"(minimum z computed is {self.zmin}).")
        if max_z > self.zmax:
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to z={max(z)} "
                              f"(maximum z computed is {self.zmax}).")
        k = np.atleast_1d(k).flatten()
        if not k.size or not np.isfinite(k).all() or not (k > 0).all():
            raise LoggedError(get_logger(self.__class__.__name__),
                              "k query must be nonempty, finite, and positive")
        min_k, max_k = min(k), max(k)
        if min_k < self.kmin:
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to k={min(k)} 1/Mpc "
                              f"(minimum k possible is {self.kmin} 1/Mpc).")
        if max_k > self.kmax:
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to k={max(k)} 1/Mpc "
                              f"(maximum k possible is {self.kmax} 1/Mpc).")

    def P(self, z, k, grid=None):
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self.logsign * np.exp(self(z, np.log(k), grid=grid, warn=False))
        else:
            return self(z, np.log(k), grid=grid, warn=False)

    def logP(self, z, k, grid=None):
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self(z, np.log(k), grid=grid, warn=False)
        else:
            return np.log(self(z, np.log(k), grid=grid, warn=False))

    def __call__(self, *args, warn=True, **kwargs):
        if warn:
            get_logger(self.__class__.__name__).warning(
                "Do not call the instance directly. Use instead methods P(z, k) or "
                "logP(z, k) to get the (log)power spectrum. (If you know what you are "
                "doing, pass warn=False)")
        return super().__call__(*args, **kwargs)


class emul_mps(Theory):
    """Cobaya Theory serving P(k, z) from grid2d emulators (EMUL2).

    extra_args:
      device    = 'cpu' / 'cuda' / 'mps' (default 'cpu').
      emulators = list of exactly TWO saved grid2d-emulator path roots
                  (ROOTDIR-relative unless absolute): one whose stored
                  quantity is "pklin" and one whose quantity is
                  "boost". Order does not matter — each artifact
                  declares itself (quantity, grids, law).
      compile   = optional bool, torch.compile on CUDA (default False).
    """

    renames = {}
    extra_args = {}
    _allow_k_extrapolation = True

    def initialize(self):
        """Build the two predictors; check grids, laws, requirements."""
        super().initialize()
        self._sigma8_requested = False
        self._check_extra_args()
        self.device = pick_device(self.extra_args, adapter="emul_mps")
        roots = resolve_emulator_roots(
            self.extra_args, adapter="emul_mps", exact_count=2)
        compile_model = exact_bool(
            self.extra_args, "compile", adapter="emul_mps")
        self._allow_k_extrapolation = exact_bool(
            self.extra_args, "allow_k_extrapolation", adapter="emul_mps",
            default=True)

        by_quantity = {}
        req = {}
        predictor_names = set()
        for root in roots:
            predictor = EmulatorPredictor(root, self.device,
                                          compile_model=compile_model)
            # wrong-kind guard: grid2d only.
            if not predictor._grid2d:
                if predictor._scalar:
                    kind, where = "scalar", "emul_scalars"
                elif predictor._cmb:
                    kind, where = "CMB spectrum", "emul_cmb"
                elif predictor._grid:
                    kind, where = "background grid", "emul_baosn"
                else:
                    kind, where = "data-vector", "emul_cosmic_shear"
                raise ValueError(
                    "emul_mps: " + repr(root) + " is not a matter-power-"
                    "spectrum emulator (its h5 rebuilds a " + kind
                    + " geometry); this theory serves grid2d artifacts "
                    "only; that emulator belongs in " + where + "'s "
                    "emulators list")
            if predictor.quantity in by_quantity:
                raise ValueError(
                    "emul_mps: two artifacts both declare quantity "
                    + repr(predictor.quantity) + "; the pair must be "
                    "one 'pklin' + one 'boost'")
            by_quantity[predictor.quantity] = predictor
            for name in predictor.names:
                req[name] = None
                predictor_names.add(name)

        for quantity in ("pklin", "boost"):
            if quantity not in by_quantity:
                raise ValueError(
                    "emul_mps: no loaded artifact declares quantity "
                    + repr(quantity) + " (loaded: "
                    + repr(sorted(by_quantity)) + "); the pair must be "
                    "one 'pklin' + one 'boost'")
        self.p_lin   = by_quantity["pklin"]
        self.p_boost = by_quantity["boost"]

        # A rebuilt Grid2D geometry validates that each field is registered,
        # but the serving adapter must also validate their physical pairing.
        # Otherwise a hand-built artifact can label a nonlinear correction as
        # linear power (or attach dimensional power units to a ratio) and still
        # produce a smooth, plausible array.
        allowed_tuples = {
            "pklin": {
                ("Mpc3", "none"),
                ("Mpc3", "syren_linear"),
            },
            "boost": {
                ("dimensionless", "none"),
                ("dimensionless", "syren_halofit"),
            },
        }
        for quantity, predictor in (("pklin", self.p_lin),
                                    ("boost", self.p_boost)):
            observed = (predictor.units, predictor.law)
            if observed not in allowed_tuples[quantity]:
                raise ValueError(
                    "emul_mps: artifact quantity " + repr(quantity)
                    + " has unsupported (units, law) " + repr(observed)
                    + "; accepted pairs are "
                    + repr(sorted(allowed_tuples[quantity])))

        # the two artifacts must share one (z, k) grid exactly (they
        # come from one generator run) — the boost multiplies the
        # linear P point-for-point.
        z1 = self.p_lin.z.detach().cpu().numpy()
        z2 = self.p_boost.z.detach().cpu().numpy()
        k1 = self.p_lin.k.detach().cpu().numpy()
        k2 = self.p_boost.k.detach().cpu().numpy()
        if not (np.array_equal(z1, z2) and np.array_equal(k1, k2)):
            raise ValueError(
                "emul_mps: the pklin and boost artifacts were trained "
                "on different (z, k) grids (pklin: "
                + repr((z1.shape, k1.shape)) + ", boost: "
                + repr((z2.shape, k2.shape)) + " or differing values); "
                "the pair must come from one generator run + one "
                "k_stride")
        for name, axis in (("z", z1), ("k", k1)):
            if not _valid_mps_axis(axis, positive=(name == "k")):
                raise ValueError(
                    "emul_mps: invalid saved " + name + " axis")
        self._z = z1
        self._k = k1

        # A Syren law reads named cosmology values beyond the artifact inputs,
        # so require them explicitly.  Dark energy is handled separately:
        # the saved physical law decides which values Cobaya must supply.  A
        # dropped sampling coordinate such as w0pwa is never requested from a
        # Theory component; Cobaya supplies its calculated wa instead.
        need_syren_base = (
            (self.p_lin.law != "none") or (self.p_boost.law != "none"))
        if need_syren_base:
            if "As" not in req and "As_1e9" not in req:
                req["As"] = None
            for name in ("ns", "H0", "omegab", "omegam"):
                req[name] = None
        (self._dark_energy_law,
         self._fixed_dark_energy,
         self._dark_energy_needed) = _dark_energy_contract(
             self.p_lin, predictor_names, req, need_base=need_syren_base)
        self._req = req

        # horizontal law, LAST: the pair is one 'pklin' + one 'boost' on one
        # shared (z, k) grid -- every configuration law above has passed. Only
        # now is it worth asking whether the two artifacts are ONE dataset.
        check_artifacts_pair_up(predictors=[self.p_lin, self.p_boost])

    def initialize_with_provider(self, provider):
        """Register the provider and compare directly named fixed values."""
        super().initialize_with_provider(provider)
        check_artifacts_fixed_values(
            predictors=[self.p_lin, self.p_boost],
            provider=provider)

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly."""
        validate_extra_args(
            self.extra_args, adapter="emul_mps", allowed=_ALLOWED_EXTRA_ARGS,
            retired=("the legacy model_file / metadata_file / nl_* / "
                     "use_syren / param_order keys are retired because the "
                     "artifacts store those facts"))

    def get_requirements(self):
        """The sampled parameters the emulators + bases need."""
        return dict(self._req)

    def get_can_support_params(self):
        """Do not claim derived products as sampled input parameters.

        Cobaya discovers ``Pk_grid`` and ``Pk_interpolator`` from their public
        getter methods.  Sigma-eight is declared as a derived result below.
        ``must_provide`` requests H0 only when sigma8 is assigned, so a
        law-free power-grid request does not acquire an unnecessary H0
        dependency or cache input.
        """
        return []

    def get_can_provide_params(self):
        """Name the derived scalar this adapter can calculate."""
        return ["sigma8"]

    def must_provide(self, **requirements):
        """Request H0 only when Cobaya assigns sigma8 to this adapter."""
        super().must_provide(**requirements)
        if "sigma8" in requirements:
            self._sigma8_requested = True
            return {"H0": None}
        return None

    def calculate(self, state, want_derived=True, **params):
        """Assemble P_lin and P_nl on the stored grid for this point.

        The two artifacts decode to LAW SPACE; each base is multiplied
        back per its own stored law (the consumer's one multiply-back
        step): the syren linear base for pklin, syren-halofit (fed the
        emulated P_lin, the legacy flow) + the low-k blend for boost.
        Non-finite or non-positive spectra reject the point (return
        False, the legacy semantics) rather than crash the chain.

        Arguments:
          state  = the cobaya state dict to populate.
          want_derived = compute sigma8 when it is a requested derived.
          params = the sampled parameter values.

        Returns:
          True, with the legacy state keys populated (("Pk_grid",
          nonlinear, "delta_tot", "delta_tot") pairs + the "Pk_grid"
          dict); False to reject the point on a bad spectrum.
        """
        z, k = self._z, self._k
        surface_shape = (len(z), len(k))
        resolved_params = _resolved_dark_energy_point(
            params, law=self._dark_energy_law,
            fixed=self._fixed_dark_energy,
            needed=self._dark_energy_needed)
        out_lin = _require_mps_surface(
            self.p_lin.predict(resolved_params)["pklin"],
            name="the linear-emulator output", shape=surface_shape)
        if out_lin is None:
            self.log.debug("non-finite linear-emulator output at "
                           f"params={params} — rejecting point.")
            return False
        need_base = (self.p_lin.law != "none"
                     or self.p_boost.law != "none")
        if need_base:
            amplitude_name = (
                "As_1e9" if "As_1e9" in resolved_params else "As")
            for name in (amplitude_name, "ns", "H0", "omegab", "omegam"):
                _require_finite_syren_parameter(resolved_params, name)
            (as_1e9, ns, H0, Ob, Om,
             w0, wa) = syren_base.syren_params_from(
                 resolved_params, dark_energy_law=self._dark_energy_law)
        if self.p_lin.law == "syren_linear":
            base = _require_mps_surface(
                syren_base.base_pklin(k_mpc=k, z=z, As_1e9=as_1e9,
                                      ns=ns, H0=H0, Ob=Ob, Om=Om,
                                      w0=w0, wa=wa),
                name="the Syren linear base", shape=surface_shape,
                positive=True)
            if base is None:
                self.log.debug("non-finite or non-positive Syren linear base "
                               f"at params={params} — rejecting point.")
                return False
            with np.errstate(over="ignore", invalid="ignore"):
                pk_lin = np.exp(out_lin) * base
        else:
            pk_lin = out_lin
        pk_lin = _require_mps_surface(
            pk_lin, name="the assembled linear spectrum",
            shape=surface_shape, positive=True)
        if pk_lin is None:
            self.log.debug("non-finite or non-positive P_lin at "
                           f"params={params} — rejecting point.")
            return False

        out_b = _require_mps_surface(
            self.p_boost.predict(resolved_params)["boost"],
            name="the boost-emulator output", shape=surface_shape)
        if out_b is None:
            self.log.debug("non-finite boost-emulator output at "
                           f"params={params} — rejecting point.")
            return False
        if self.p_boost.law == "syren_halofit":
            b_base = _require_mps_surface(
                syren_base.base_boost(k_mpc=k, z=z,
                                      pk_lin_mpc=pk_lin,
                                      As_1e9=as_1e9, ns=ns, H0=H0,
                                      Ob=Ob, Om=Om, w0=w0, wa=wa),
                name="the Syren nonlinear-boost base", shape=surface_shape,
                positive=True)
            if b_base is None:
                self.log.debug("non-finite or non-positive Syren boost base "
                               f"at params={params} — rejecting point.")
                return False
            with np.errstate(over="ignore", invalid="ignore"):
                boost = np.exp(out_b) * b_base
            # the legacy low-k blend (verbatim constants): pin the
            # boost to exactly 1 on linear scales.
            weight = 1.0 - np.exp(-(k / _BLEND_K_T) ** _BLEND_N)
            boost = 1.0 + (boost - 1.0) * weight[None, :]
        else:
            boost = out_b
        boost = _require_mps_surface(
            boost, name="the assembled nonlinear boost",
            shape=surface_shape, positive=True)
        if boost is None:
            self.log.debug("non-finite or non-positive boost at "
                           f"params={params} — rejecting point.")
            return False

        with np.errstate(over="ignore", invalid="ignore"):
            pk_nl_raw = boost * pk_lin
        pk_nl = _require_mps_surface(
            pk_nl_raw, name="the assembled nonlinear spectrum",
            shape=surface_shape, positive=True)
        if pk_nl is None:
            self.log.debug("non-finite or non-positive P_nl at "
                           f"params={params} — rejecting point.")
            return False

        derived = None
        if want_derived:
            derived = {}
            if ("sigma8" in self.output_params
                    or getattr(self, "_sigma8_requested", False)):
                if "H0" in params:
                    raw_h0 = params["H0"]
                else:
                    try:
                        raw_h0 = self.provider.get_param("H0")
                    except (AttributeError, KeyError, LoggedError) as error:
                        raise ValueError(
                            "emul_mps: sigma8 requires H0 from the sampled "
                            "parameters or the assigned Cobaya provider") from error
                H0_for_sigma8 = _require_positive_sigma8_h0(raw_h0)
                sigma8 = self._compute_sigma8(
                    pk_lin, k, z, h=H0_for_sigma8 / 100.0)
                if (isinstance(sigma8, (bool, np.bool_))
                        or not isinstance(sigma8, numbers.Real)
                        or not math.isfinite(float(sigma8))
                        or float(sigma8) <= 0.0):
                    self.log.debug("non-finite or non-positive sigma8 at "
                                   f"params={params} — rejecting point.")
                    return False
                derived["sigma8"] = float(sigma8)

        # Nothing reaches Cobaya's state until both emulators, both analytic
        # bases, the composed spectra, and the optional derived scalar pass.
        state[("Pk_grid", True, "delta_tot", "delta_tot")] = (k, z, pk_nl)
        state[("Pk_grid", False, "delta_tot", "delta_tot")] = (k, z,
                                                               pk_lin)
        state["Pk_grid"] = {"k": k, "z": z, "Pk": pk_lin}
        if derived is not None:
            state["derived"] = derived
        return True

    def get_Pk_grid(self, var_pair=("delta_tot", "delta_tot"),
                    nonlinear=True):
        r"""P(k, z) grid in cobaya units: k in 1/Mpc, Pk in Mpc^3.

        nonlinear follows Cobaya's BoltzmannBase default: an omitted
        argument returns the NONLINEAR grid. A caller that wants the
        linear spectrum requests it explicitly with nonlinear=False.

        The three returned arrays own their storage. Editing them cannot
        change the provider cache used by another likelihood.
        """
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emul_mps only supports delta_tot power spectra, "
                f"not {var_pair}")
        key = ("Pk_grid", nonlinear) + tuple(sorted(var_pair))
        if key in self.current_state:
            k, z, power = self.current_state[key]
            return (np.array(k, copy=True), np.array(z, copy=True),
                    np.array(power, copy=True))
        raise LoggedError(
            self.log,
            f"Matter power spectrum (nonlinear={nonlinear}) not computed.")

    def get_Pk_interpolator(self, var_pair=("delta_tot", "delta_tot"),
                            nonlinear=True, extrap_kmin=None,
                            extrap_kmax=None):
        r"""Get a P(z, k) bicubic interpolation object (the legacy flow:
        log-P spline over the stored grid, power-law tails on request).

        nonlinear follows Cobaya's BoltzmannBase default: an omitted
        argument interpolates the NONLINEAR grid. A caller that wants the
        linear spectrum requests it explicitly with nonlinear=False.
        """
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emul_mps only supports delta_tot power spectra, "
                f"not {var_pair}")
        k, z, pk = self.get_Pk_grid(var_pair=var_pair,
                                    nonlinear=nonlinear)
        log_p = True
        sign = 1
        if np.any(pk < 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_p = False
                self.log.debug(
                    "Power spectrum has both positive and negative "
                    "values; using linear interpolation.")
        extrapolating = ((extrap_kmax and extrap_kmax > k[-1])
                         or (extrap_kmin and extrap_kmin < k[0]))
        if extrapolating and not self._allow_k_extrapolation:
            raise LoggedError(
                self.log, "allow_k_extrapolation is false")
        if log_p:
            pk_for_interp = np.log(sign * pk)
        elif extrapolating:
            raise LoggedError(
                self.log,
                f"Cannot do log extrapolation with zero-crossing Pk "
                f"for {var_pair}")
        else:
            pk_for_interp = pk
        result = PowerSpectrumInterpolator(
            z, k, pk_for_interp,
            logP=log_p,
            logsign=sign,
            extrap_kmin=extrap_kmin,
            extrap_kmax=extrap_kmax,
        )
        return result

    @staticmethod
    def _sigma8_log_band(log_k, contribution, lower, upper):
        """Integrate one exact log-k band, inserting its two boundaries."""
        inside = (log_k > lower) & (log_k < upper)
        band_log_k = np.concatenate((
            np.array([lower]), log_k[inside], np.array([upper])))
        band_values = np.concatenate((
            np.array([np.interp(lower, log_k, contribution)]),
            contribution[inside],
            np.array([np.interp(upper, log_k, contribution)])))
        return float(np.trapz(band_values, band_log_k))

    @classmethod
    def _check_sigma8_completeness(cls, log_k, contribution, variance):
        """Refuse missing tails or a grid too coarse for the integral.

        ``contribution`` is the nonnegative contribution to sigma squared per
        unit log(k).  Adjacent half-decade integrals must decay toward both
        missing tails.  Their observed ratios give a geometric estimate of
        the omitted variance.  A second integral using every other stored
        point independently checks the numerical resolution.
        """
        required_span = _SIGMA8_LOG_DECADE
        span = float(log_k[-1] - log_k[0])
        if span <= required_span:
            raise ValueError(
                "emul_mps: the sigma8 k grid is incomplete: it spans "
                + repr(span / _SIGMA8_LOG_DECADE)
                + " decades; more than one decade is required so two "
                "half-decade evidence bands can be measured at each tail")

        lo = float(log_k[0])
        hi = float(log_k[-1])
        d = 0.5 * _SIGMA8_LOG_DECADE
        low_outer = cls._sigma8_log_band(
            log_k, contribution, lo, lo + d)
        low_inner = cls._sigma8_log_band(
            log_k, contribution, lo + d, lo + 2.0 * d)
        high_inner = cls._sigma8_log_band(
            log_k, contribution, hi - 2.0 * d, hi - d)
        high_outer = cls._sigma8_log_band(
            log_k, contribution, hi - d, hi)

        bands = (low_outer, low_inner, high_inner, high_outer)
        if not all(math.isfinite(value) and value > 0.0 for value in bands):
            raise ValueError(
                "emul_mps: sigma8 tail evidence must contain four finite, "
                "strictly positive half-decade integrals; got "
                + repr(bands))

        low_ratio = low_outer / low_inner
        high_ratio = high_outer / high_inner
        if not 0.0 <= low_ratio < 1.0:
            raise ValueError(
                "emul_mps: the sigma8 k grid has an incomplete low-k tail: "
                "the contribution does not decay toward the lower edge "
                "(outer/inner half-decade ratio "
                + repr(low_ratio) + ")")
        if not 0.0 <= high_ratio < 1.0:
            raise ValueError(
                "emul_mps: the sigma8 k grid has an incomplete high-k tail: "
                "the contribution does not decay toward the upper edge "
                "(outer/inner half-decade ratio "
                + repr(high_ratio) + ")")

        omitted_low = low_outer * low_ratio / (1.0 - low_ratio)
        omitted_high = high_outer * high_ratio / (1.0 - high_ratio)
        omitted_fraction = (omitted_low + omitted_high) / variance
        if (not math.isfinite(omitted_fraction)
                or omitted_fraction > _SIGMA8_TAIL_VARIANCE_RTOL):
            raise ValueError(
                "emul_mps: the sigma8 k grid leaves an estimated variance "
                "fraction " + repr(omitted_fraction)
                + " outside its low-k and high-k edges; the limit is "
                + repr(_SIGMA8_TAIL_VARIANCE_RTOL))

        even_indices = np.arange(0, log_k.size, 2)
        if even_indices[-1] != log_k.size - 1:
            even_indices = np.append(even_indices, log_k.size - 1)
        odd_indices = np.arange(1, log_k.size, 2)
        odd_indices = np.unique(np.concatenate((
            np.array([0]), odd_indices, np.array([log_k.size - 1]))))
        coarse_variances = (
            float(np.trapz(
                contribution[even_indices], log_k[even_indices])),
            float(np.trapz(
                contribution[odd_indices], log_k[odd_indices])))
        quadrature_change = max(
            abs(value - variance) / variance for value in coarse_variances)
        if (not all(math.isfinite(value) and value > 0.0
                    for value in coarse_variances)
                or not math.isfinite(quadrature_change)
                or quadrature_change > _SIGMA8_QUADRATURE_RTOL):
            raise ValueError(
                "emul_mps: the sigma8 k grid is too coarse: the two "
                "interlaced every-other-point integrals change the variance "
                "by as much as "
                + repr(quadrature_change) + ", above the limit "
                + repr(_SIGMA8_QUADRATURE_RTOL))

        panel_areas = (0.5 * (contribution[:-1] + contribution[1:])
                       * np.diff(log_k))
        largest_panel_fraction = float(np.max(panel_areas) / variance)
        if (not np.isfinite(panel_areas).all()
                or not (panel_areas >= 0.0).all()
                or not math.isfinite(largest_panel_fraction)
                or largest_panel_fraction > _SIGMA8_MAX_PANEL_FRACTION):
            raise ValueError(
                "emul_mps: the sigma8 k grid is locally under-resolved: one "
                "trapezoid carries variance fraction "
                + repr(largest_panel_fraction) + ", above the limit "
                + repr(_SIGMA8_MAX_PANEL_FRACTION))

    def _compute_sigma8(self, Pk_2d, k_array, z_array, *, h):
        """Return conventional sigma8 from a checked linear-power surface.

        The saved wavenumbers are in 1/Mpc and the power values are in Mpc^3.
        Conventional sigma8 uses a top-hat radius of 8/h Mpc, where
        ``h = H0/100``.  The calculation is available only when the stored
        surface contains an exact z=0 row and its positive log-k integrand
        demonstrates adequate tails and numerical resolution.

        Arguments:
          Pk_2d   = linear P(k,z) with exact shape (nz,nk), in Mpc^3.
          k_array = strictly increasing positive wavenumbers in 1/Mpc.
          z_array = strictly increasing nonnegative redshifts including 0.
          h       = positive finite H0/100, supplied by ``calculate``.

        Returns:
          Conventional sigma8 as a finite positive Python float.
        """
        if (isinstance(h, (bool, np.bool_))
                or not isinstance(h, numbers.Real)):
            raise ValueError(
                "emul_mps: sigma8 h=H0/100 must be a finite, strictly "
                "positive real scalar and not a Boolean; got " + repr(h))
        try:
            h_value = float(h)
        except (OverflowError, TypeError, ValueError):
            h_value = float("nan")
        if not math.isfinite(h_value) or h_value <= 0.0:
            raise ValueError(
                "emul_mps: sigma8 h=H0/100 must be finite and strictly "
                "positive; got " + repr(h))

        k_raw = np.asarray(k_array)
        z_raw = np.asarray(z_array)
        power_raw = np.asarray(Pk_2d)
        for name, value in (("k axis", k_raw), ("z axis", z_raw),
                            ("linear-power surface", power_raw)):
            if (not np.isrealobj(value)
                    or value.dtype.kind not in "fiu"):
                raise ValueError(
                    "emul_mps: sigma8 " + name
                    + " must contain real numerical values, not "
                    + repr(value.dtype))
        if k_raw.ndim != 1 or k_raw.size < 3:
            raise ValueError(
                "emul_mps: sigma8 k axis must be one-dimensional with at "
                "least three values; got shape " + repr(k_raw.shape))
        if z_raw.ndim != 1 or z_raw.size < 1:
            raise ValueError(
                "emul_mps: sigma8 z axis must be a nonempty one-dimensional "
                "array; got shape " + repr(z_raw.shape))
        expected_shape = (z_raw.size, k_raw.size)
        if power_raw.shape != expected_shape:
            raise ValueError(
                "emul_mps: sigma8 linear-power surface must have exact "
                "shape " + repr(expected_shape) + ", got "
                + repr(power_raw.shape))

        k = np.asarray(k_raw, dtype=np.float64)
        z = np.asarray(z_raw, dtype=np.float64)
        power = np.asarray(power_raw, dtype=np.float64)
        if (not np.isfinite(k).all() or not (k > 0.0).all()
                or not (np.diff(k) > 0.0).all()):
            raise ValueError(
                "emul_mps: sigma8 k axis must be finite, strictly positive, "
                "and strictly increasing")
        if (not np.isfinite(z).all() or not (z >= 0.0).all()
                or not (np.diff(z) > 0.0).all()):
            raise ValueError(
                "emul_mps: sigma8 z axis must be finite, nonnegative, and "
                "strictly increasing")
        zero_rows = np.flatnonzero(z == 0.0)
        if zero_rows.size != 1:
            raise ValueError(
                "emul_mps: sigma8 requires exactly one stored row at exact "
                "z=0; a nearby redshift is not zero")
        if not np.isfinite(power).all() or not (power > 0.0).all():
            raise ValueError(
                "emul_mps: sigma8 linear-power surface must contain only "
                "finite, strictly positive values")
        power_at_zero = power[int(zero_rows[0]), :]

        radius_mpc = _SIGMA8_RADIUS_MPC_OVER_H / h_value
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x = k * radius_mpc
        if not np.isfinite(x).all() or not (x > 0.0).all():
            raise ValueError(
                "emul_mps: sigma8 kR values must be finite and strictly "
                "positive")

        window = np.empty_like(x, dtype=np.float64)
        small = x <= 1.0e-3
        x2 = x[small] ** 2
        window[small] = (1.0 - x2 / 10.0 + x2 ** 2 / 280.0
                         - x2 ** 3 / 15120.0)
        regular = ~small
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            window[regular] = (
                3.0 * (np.sin(x[regular])
                       - x[regular] * np.cos(x[regular]))
                / x[regular] ** 3)
            contribution = (
                k ** 3 * power_at_zero * window ** 2
                / (2.0 * np.pi ** 2))
            log_k = np.log(k)
        if (not np.isfinite(window).all()
                or not np.isfinite(contribution).all()
                or not (contribution >= 0.0).all()
                or not np.isfinite(log_k).all()):
            raise ValueError(
                "emul_mps: sigma8 window and log-k integrand must remain "
                "finite and nonnegative")

        variance = float(np.trapz(contribution, log_k))
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError(
                "emul_mps: sigma8 variance must be finite and strictly "
                "positive before taking its square root; got "
                + repr(variance))
        self._check_sigma8_completeness(log_k, contribution, variance)
        sigma8 = math.sqrt(variance)
        if not math.isfinite(sigma8) or sigma8 <= 0.0:
            raise ValueError(
                "emul_mps: sigma8 result must be finite and strictly "
                "positive; got " + repr(sigma8))
        return sigma8

    def get_sigma8(self):
        return self.current_state.get("derived", {}).get("sigma8")

    def get_param(self, param_name):
        return self.current_state.get("derived", {}).get(param_name)
