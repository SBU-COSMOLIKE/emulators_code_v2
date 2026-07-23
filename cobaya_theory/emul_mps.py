"""Thin cobaya Theory adapter: matter power spectra from saved grid2d
emulators (the MPS family, serving CosmoLike's hybrid mode).

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
import os
import sys

import numpy as np
import torch
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

# A finite integral is not enough: a k grid that stops before the sigma8
# integrand has decayed misses part of the variance while still returning a
# smooth positive number.  The helper refuses a grid whose edge contribution
# per unit log(k) is more than this fraction of the integrand's peak.
_SIGMA8_EDGE_FRACTION = 1.0e-3

_DARK_ENERGY_NAMES = ("w", "w0", "wa", "w0pwa")
_DARK_ENERGY_LAWS = {
    "w0wa-cpl": ("w", "wa"),
    "constant-w": ("w",),
    "cosmological-constant": (),
}


def _valid_mps_axis(axis, *, positive=False):
    """Is this a usable spline axis (1-D, finite, strictly increasing)?

    A bicubic spline needs at least 4 points per axis, strictly
    increasing coordinates, and finite values; the k axis must also be
    positive because the interpolation runs in log(k).

    Arguments:
      axis     = the candidate axis array.
      positive = additionally require every entry > 0 (the k axis).

    Returns:
      True when the axis satisfies every rule; False otherwise.
    """
    return (axis.ndim == 1 and len(axis) >= 4 and np.isfinite(axis).all()
            and np.all(axis[1:] > axis[:-1])
            and (not positive or (axis > 0).all()))


def _saved_dark_energy_value(fixed, name):
    """Read one saved dark-energy fact from the artifact's fixed block.

    The record writes "n/a" for a fact that does not apply (never omits
    the key), so both an absent key and the "n/a" marker mean "no pinned
    value here"; a present value must parse as a finite number.

    Arguments:
      fixed = the artifact's cosmology_fixed mapping.
      name  = the coordinate to read ("w" or "wa").

    Returns:
      the pinned value as a float, or None when the fact does not apply.

    Raises:
      ValueError when a present value is not a finite number.
    """
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

    A dropped ``w0pwa`` is a sampling coordinate used to define ``wa``.  It
    is not a value that a Cobaya Theory may request directly.  The adapter
    asks for the present-day value and ``wa``, then reconstructs every
    spelling the saved input geometry needs inside :meth:`calculate`.

    Arguments:
      predictor       = the pklin predictor whose record carries the law.
      predictor_names = the union of both artifacts' sampled input names.
      req             = the Cobaya requirement mapping under construction;
                        edited in place (w0pwa removed, the law's own
                        coordinates added when not pinned).
      need_base       = whether a Syren base will be evaluated (a base
                        needs the law even when no dark-energy coordinate
                        is sampled).

    Returns:
      the triple (law, fixed, needed): the artifact's dark-energy law
      name, the mapping of pinned coordinate values, and whether
      calculate must resolve the coordinates at all.

    Raises:
      ValueError when the record's law is missing or unknown, its
      declared inputs disagree with the law, or a sampled coordinate
      contradicts the law (each refusal names the regeneration path).
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
    """Fill every dark-energy spelling of one sampled point.

    The saved input geometry may name any of the four equivalent
    coordinates (w, w0, wa, w0pwa), and the Syren base needs (w0, wa).
    The shared resolver derives the canonical pair from whatever the
    point carries -- pinned values fill the gaps, but never overwrite an
    explicitly sampled value, so a contradiction is still caught -- and
    every spelling is then written back.

    Arguments:
      params = the sampled point as Cobaya handed it over.
      law    = the artifact's dark-energy law name.
      fixed  = the pinned coordinate values from the record.
      needed = whether resolution is needed at all (False returns the
               merged point unchanged).

    Returns:
      a new mapping with w / w0 / wa / w0pwa all present and consistent.
    """
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
        """Build the bicubic spline, adding power-law tails on request.

        A bicubic spline interpolates a surface with piecewise cubic
        polynomials that are smooth in both axes; here it is fitted on
        (z, log k), the axes the spectrum varies gently over. The class
        docstring documents every parameter. Extrapolation works by
        extending the log(k) axis with two synthetic columns on the
        requested side whose values continue the edge slope of
        log(P) -- a straight line in log-log space is a power law in
        k -- before the spline is fitted.
        """
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
                    not math.isfinite(float(bound)) or bound <= 0):
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
        """The stored grid's smallest wavenumber, in 1/Mpc."""
        return self.k[0]

    @property
    def input_kmax(self):
        """The stored grid's largest wavenumber, in 1/Mpc."""
        return self.k[-1]

    @property
    def kmin(self):
        """The smallest servable wavenumber: the stored edge, or the
        requested extrapolation bound when one was given (1/Mpc)."""
        if self.extrap_kmin is None:
            return self.input_kmin
        return self.extrap_kmin

    @property
    def kmax(self):
        """The largest servable wavenumber: the stored edge, or the
        requested extrapolation bound when one was given (1/Mpc)."""
        if self.extrap_kmax is None:
            return self.input_kmax
        return self.extrap_kmax

    def check_ranges(self, z, k):
        """Refuse a query outside the servable (z, k) region, loudly.

        Arguments:
          z = scalar or array of query redshifts; must lie inside the
              stored z range (z is never extrapolated).
          k = scalar or array of query wavenumbers in 1/Mpc; must be
              positive, finite, and inside [kmin, kmax].
        """
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
        """Evaluate P(z, k) in Mpc^3 at the query points.

        Arguments:
          z    = scalar or array of redshifts (inside the stored range).
          k    = scalar or array of wavenumbers in 1/Mpc (inside
                 [kmin, kmax]).
          grid = evaluate on the outer product of z and k (True), or
                 pointwise (False); None decides from the inputs (both
                 arrays -> grid).

        Returns:
          the power spectrum values (undoing the internal log when the
          spline stores log P).
        """
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self.logsign * np.exp(self(z, np.log(k), grid=grid, warn=False))
        else:
            return self(z, np.log(k), grid=grid, warn=False)

    def logP(self, z, k, grid=None):
        """Evaluate log(logsign * P(z, k)) at the query points.

        Arguments:
          z, k, grid = as in :meth:`P`.

        Returns:
          the natural log of the (sign-corrected) power spectrum.
        """
        self.check_ranges(z, k)
        if grid is None:
            grid = not np.isscalar(z) and not np.isscalar(k)
        if self.islog:
            return self(z, np.log(k), grid=grid, warn=False)
        else:
            return np.log(self(z, np.log(k), grid=grid, warn=False))

    def __call__(self, *args, warn=True, **kwargs):
        """Raw spline evaluation over (z, log k); prefer P / logP.

        A direct call skips the range checks and the log convention, so
        it warns unless the caller passes warn=False knowingly.

        Returns:
          the underlying RectBivariateSpline evaluation.
        """
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
      allow_k_extrapolation = optional bool (default True): permit the
                  power-law k tails when a consumer requests an
                  interpolator wider than the stored grid; False refuses
                  such a request loudly.
    """

    renames = {}
    extra_args = {}
    _allow_k_extrapolation = True

    def initialize(self):
        """Build the two predictors; check grids, laws, requirements.

        Cobaya constructs one instance of this class from the sampling
        YAML's theory block and calls initialize() once, before any
        sampling; extra_args is that block's option mapping, handed
        over by Cobaya as an attribute. This method validates the
        options, picks the device, rebuilds the linear-spectrum and
        boost predictors, checks that their (z, k) grids and analytic
        laws agree, and derives the required sampled inputs from the
        artifacts' own stored names.

        Raises:
          ValueError for an unknown option, a malformed emulators
          list, an artifact of the wrong family, or predictors whose
          grids or laws disagree.
        """
        super().initialize()
        self._sigma8_requested = False
        self._check_extra_args()
        self.device = self._pick_device(self.extra_args.get("device", "cpu"))

        roots = self.extra_args.get("emulators")
        if not roots or len(roots) != 2:
            raise ValueError(
                "emul_mps: extra_args needs an 'emulators' list of exactly "
                "TWO saved grid2d-emulator path roots (one 'pklin' + one "
                "'boost'; each root -> <root>.h5 + <root>.emul), got "
                + repr(roots))
        compile_model = bool(self.extra_args.get("compile", False))
        self._allow_k_extrapolation = bool(
            self.extra_args.get("allow_k_extrapolation", True))
        rootdir = os.environ.get("ROOTDIR", "")

        by_quantity = {}
        req = {}
        predictor_names = set()
        for root in roots:
            path = root if os.path.isabs(root) else os.path.join(rootdir, root)
            predictor = EmulatorPredictor(path, self.device,
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

        # cross-artifact law, LAST: the pair is one 'pklin' + one 'boost' on
        # one shared (z, k) grid -- every configuration law above has passed.
        # Only now is it worth asking whether the two artifacts describe one
        # cosmology over one region.
        check_artifacts_pair_up(predictors=[self.p_lin, self.p_boost])

    def initialize_with_provider(self, provider):
        """Register the provider and compare directly named fixed values.

        Cobaya calls this once, when the full model exists. Both
        artifacts' recorded fixed values are compared against the
        model's directly named constants; a disagreement refuses at
        startup.

        Arguments:
          provider = the Cobaya provider carrying the resolved model.
        """
        super().initialize_with_provider(provider)
        check_artifacts_fixed_values(
            predictors=[self.p_lin, self.p_boost],
            provider=provider)

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly.

        Cobaya passes the theory block's options through without
        checking them, so a typo'd key would otherwise be silently
        ignored and its intended setting silently defaulted; the
        closed list turns that into a named error.
        """
        unknown = []
        for key in self.extra_args:
            if key not in _ALLOWED_EXTRA_ARGS:
                unknown.append(key)
        if unknown:
            raise ValueError(
                "emul_mps: unrecognized extra_args key(s) "
                f"{sorted(unknown)}. The schema-v2 convention accepts only "
                f"{list(_ALLOWED_EXTRA_ARGS)}; the legacy model_file / "
                "metadata_file / nl_* / use_syren / param_order keys are "
                "retired (the artifacts store those facts).")

    @staticmethod
    def _pick_device(requested):
        """Resolve the requested device to cpu / cuda / mps (TPU dropped).

        Arguments:
          requested = the extra_args 'device' string (cpu / cuda / mps).

        Returns:
          a torch.device, falling back to cpu when the requested accelerator
          is unavailable (cuda -> mps -> cpu, matching emul_cosmic_shear).
        """
        req = str(requested).lower()
        if req == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if (req in ("cuda", "mps")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")

    def get_requirements(self):
        """The sampled parameters the emulators and Syren bases need.

        Returns:
          a fresh {name: None} mapping (Cobaya's requirement form): the
          artifacts' stored input names, the Syren base inputs when a
          law is active, and the dark-energy coordinates the saved law
          requires.
        """
        return dict(self._req)

    def get_can_support_params(self):
        """Do not claim derived products as sampled input parameters.

        Cobaya discovers ``Pk_grid`` and ``Pk_interpolator`` from their
        public getter methods.  Sigma-eight is declared as a derived
        result below.  ``must_provide`` requests H0 only when sigma8 is
        assigned, so a law-free power-grid request does not acquire an
        unnecessary H0 dependency or cache input.

        Returns:
          the empty list.
        """
        return []

    def get_can_provide_params(self):
        """Name the derived scalar this adapter can calculate.

        Returns:
          ["sigma8"]: the one derived parameter, computed on request
          from the linear spectrum's z=0 row.
        """
        return ["sigma8"]

    def _validate_matter_power_request(self, product, options):
        """Refuse an unservable matter-power request while setup can stop.

        Cobaya forwards each likelihood's requirement options before
        sampling starts. Checking them here turns a configuration
        mistake into an immediate startup error the user can fix in the
        YAML, instead of a failure deep inside a running chain.

        Arguments:
          product = the requirement name, "Pk_grid" or
                    "Pk_interpolator" (used in every refusal message).
          options = that requirement's option mapping as Cobaya passed
                    it; None and an empty mapping mean "defaults" and
                    pass. The mapping is only read, never edited.

        Raises:
          LoggedError when the options are not a mapping, request a
          density pair other than ("delta_tot", "delta_tot"), carry a
          non-boolean nonlinear choice, ask for a redshift outside the
          stored z grid, or ask for wavenumbers the stored grid cannot
          serve (the fixed grid for Pk_grid; beyond the power-law
          tails' reach for Pk_interpolator when extrapolation is
          disabled).
        """
        if options is None:
            return
        if not isinstance(options, dict):
            raise LoggedError(
                self.log,
                f"emul_mps: {product} requirement options must be a "
                f"mapping of option names to values, got {options!r}; "
                "see Cobaya's Pk_grid / Pk_interpolator requirement "
                "format")

        pairs = options.get("vars_pairs")
        if pairs:
            for pair in pairs:
                if tuple(pair) != ("delta_tot", "delta_tot"):
                    raise LoggedError(
                        self.log,
                        f"emul_mps: {product} requests the density pair "
                        f"{tuple(pair)!r}; this adapter serves only "
                        "('delta_tot', 'delta_tot'). Correct vars_pairs "
                        "in the likelihood's requirements")

        nonlinear = options.get("nonlinear")
        if nonlinear is not None:
            if isinstance(nonlinear, (list, tuple, set)):
                nonlinear_values = list(nonlinear)
            else:
                nonlinear_values = [nonlinear]
            for value in nonlinear_values:
                if not isinstance(value, bool):
                    raise LoggedError(
                        self.log,
                        f"emul_mps: {product} nonlinear must be boolean "
                        "(True for P_nl, False for P_lin), got "
                        f"{value!r}")

        requested_z = options.get("z")
        if requested_z is not None:
            z_values = np.atleast_1d(np.asarray(requested_z,
                                                dtype=np.float64))
            if z_values.size == 0 or not np.isfinite(z_values).all():
                raise LoggedError(
                    self.log,
                    f"emul_mps: {product} z request must be nonempty "
                    f"and finite, got {requested_z!r}")
            stored_z_min = float(self._z[0])
            stored_z_max = float(self._z[-1])
            requested_min = float(z_values.min())
            requested_max = float(z_values.max())
            if requested_min < stored_z_min or requested_max > stored_z_max:
                raise LoggedError(
                    self.log,
                    f"emul_mps: {product} requests z in "
                    f"[{requested_min:g}, {requested_max:g}] outside "
                    f"the stored redshift range [{stored_z_min:g}, "
                    f"{stored_z_max:g}]; z is never extrapolated, so "
                    "lower the likelihood's z request or retrain the "
                    "emulators on a wider range")

        k_max = options.get("k_max")
        if k_max is not None:
            k_max = float(k_max)
            if not math.isfinite(k_max) or k_max <= 0.0:
                raise LoggedError(
                    self.log,
                    f"emul_mps: {product} k_max must be a finite "
                    f"positive wavenumber in 1/Mpc, got {k_max!r}")
            stored_k_max = float(self._k[-1])
            if product == "Pk_grid" and k_max > stored_k_max:
                raise LoggedError(
                    self.log,
                    f"emul_mps: Pk_grid requests k_max={k_max:g} 1/Mpc "
                    f"above the stored grid edge {stored_k_max:g} "
                    "1/Mpc; the fixed stored grid cannot extend, so "
                    "lower k_max or retrain with a wider k range")
            if (product == "Pk_interpolator" and k_max > stored_k_max
                    and not self._allow_k_extrapolation):
                raise LoggedError(
                    self.log,
                    f"emul_mps: Pk_interpolator requests "
                    f"k_max={k_max:g} 1/Mpc above the stored grid edge "
                    f"{stored_k_max:g} 1/Mpc while allow_k_extrapolation "
                    "is false; enable the power-law tails or retrain "
                    "with a wider k range")

    def must_provide(self, **requirements):
        """Check matter-power requests early; request H0 for sigma8.

        Every "Pk_grid" and "Pk_interpolator" requirement is validated
        against the loaded artifacts before sampling starts (see
        :meth:`_validate_matter_power_request`), so an unsupported
        density pair, redshift, or wavenumber request stops while the
        user can still correct the YAML.

        Arguments:
          requirements = the requirement mapping Cobaya forwards; the
                         matter-power products are validated, and a
                         "sigma8" key adds the H0 dependency.

        Returns:
          {"H0": None} when sigma8 was assigned (the top-hat radius is
          8/h Mpc, so the calculation needs H0); None otherwise, so a
          power-grid-only run acquires no extra dependency.

        Raises:
          LoggedError from the validation helper for an unservable
          matter-power request.
        """
        for name in ("Pk_grid", "Pk_interpolator"):
            if name in requirements:
                self._validate_matter_power_request(
                    name, requirements[name])
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
          state  = Cobaya's results dictionary for the current sampled
                   point; this method fills the matter-power entries.
          want_derived = True when Cobaya asks for derived parameters;
                   sigma8 is computed only then, and only if requested.
          params = the sampled parameter values, by name.

        Returns:
          True, with the legacy state keys populated (("Pk_grid",
          nonlinear, "delta_tot", "delta_tot") pairs + the "Pk_grid"
          dict); False to reject the point on a bad spectrum.
        """
        z, k = self._z, self._k
        resolved_params = _resolved_dark_energy_point(
            params, law=self._dark_energy_law,
            fixed=self._fixed_dark_energy,
            needed=self._dark_energy_needed)
        out_lin = np.asarray(
            self.p_lin.predict(resolved_params)["pklin"])   # (nz, nk)
        need_base = (self.p_lin.law != "none"
                     or self.p_boost.law != "none")
        if need_base:
            (as_1e9, ns, H0, Ob, Om,
             w0, wa) = syren_base.syren_params_from(
                 resolved_params, dark_energy_law=self._dark_energy_law)
        if self.p_lin.law == "syren_linear":
            base = syren_base.base_pklin(k_mpc=k, z=z, As_1e9=as_1e9,
                                         ns=ns, H0=H0, Ob=Ob, Om=Om,
                                         w0=w0, wa=wa)
            with np.errstate(over="ignore", invalid="ignore"):
                pk_lin = np.exp(out_lin) * base
        else:
            pk_lin = out_lin
        if not (np.isfinite(pk_lin).all() and (pk_lin > 0).all()):
            self.log.debug("non-finite or non-positive P_lin at "
                           f"params={params} — rejecting point.")
            return False

        out_b = np.asarray(
            self.p_boost.predict(resolved_params)["boost"])  # (nz, nk)
        if self.p_boost.law == "syren_halofit":
            b_base = syren_base.base_boost(k_mpc=k, z=z,
                                           pk_lin_mpc=pk_lin,
                                           As_1e9=as_1e9, ns=ns, H0=H0,
                                           Ob=Ob, Om=Om, w0=w0, wa=wa)
            with np.errstate(over="ignore", invalid="ignore"):
                boost = np.exp(out_b) * b_base
            # the legacy low-k blend (verbatim constants): pin the
            # boost to exactly 1 on linear scales.
            weight = 1.0 - np.exp(-(k / _BLEND_K_T) ** _BLEND_N)
            boost = 1.0 + (boost - 1.0) * weight[None, :]
        else:
            boost = out_b
        if not (np.isfinite(boost).all() and (boost > 0).all()):
            self.log.debug("non-finite or non-positive boost at "
                           f"params={params} — rejecting point.")
            return False

        with np.errstate(over="ignore", invalid="ignore"):
            pk_nl = boost * pk_lin
        if not (np.isfinite(pk_nl).all() and (pk_nl > 0).all()):
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
                h0 = float(raw_h0)
                if not math.isfinite(h0) or h0 <= 0.0:
                    raise ValueError(
                        "emul_mps: sigma8 needs a finite positive H0, got "
                        + repr(raw_h0))
                sigma8 = self._compute_sigma8(pk_lin, k, z, h=h0 / 100.0)
                if not math.isfinite(sigma8) or sigma8 <= 0.0:
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

        Arguments:
          var_pair  = the density pair; only ("delta_tot", "delta_tot")
                      is served.
          nonlinear = True (default) for P_nl, False for P_lin.

        Returns:
          the triple (k, z, P): k in 1/Mpc, z, and the (nz, nk) power
          surface in Mpc^3. The three arrays own their storage, so
          editing them cannot change the provider cache used by another
          likelihood.

        Raises:
          LoggedError for an unsupported pair or a point whose spectra
          were not computed.
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

        Arguments:
          var_pair    = the density pair; only ("delta_tot",
                        "delta_tot") is served.
          nonlinear   = True (default) for P_nl, False for P_lin.
          extrap_kmin = optional bound below the stored grid; the
                        interpolator adds a power-law tail down to it.
          extrap_kmax = optional bound above the stored grid, same
                        convention.

        Returns:
          a fresh PowerSpectrumInterpolator over the current point's
          grid (each caller gets its own object).

        Raises:
          LoggedError for an unsupported pair, a missing point, tails
          requested with allow_k_extrapolation false, or a
          zero-crossing spectrum that cannot be log-extrapolated.
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
    def _compute_sigma8(Pk_2d, k_array, z_array, *, h):
        """Return conventional sigma8 from the linear-power surface at z=0.

        The saved wavenumbers are in 1/Mpc and the power values are in Mpc^3.
        Conventional sigma8 uses a top-hat radius of 8/h Mpc, where
        ``h = H0/100`` (the legacy adapter used R = 8 Mpc with k in 1/Mpc,
        which is sigma(8 Mpc) rather than sigma8; the corrected radius is a
        real fix and is kept). The stored surface must contain an exact z=0
        row, and the integrand must have decayed at both ends of the stored
        k range -- a grid that stops too early would return a smooth number
        that is silently missing part of the variance.

        Arguments:
          Pk_2d   = (nz, nk) linear P(k, z) in Mpc^3, already checked finite
                    and positive by ``calculate``.
          k_array = (nk,) strictly increasing wavenumbers in 1/Mpc.
          z_array = (nz,) the stored redshift grid.
          h       = H0/100, checked finite and positive by ``calculate``.

        Returns:
          sigma8 as a Python float.

        Raises:
          ValueError when no exact z=0 row is stored, or when the k grid
          truncates the sigma8 integrand.
        """
        z = np.asarray(z_array, dtype=np.float64)
        zero_rows = np.flatnonzero(z == 0.0)
        if zero_rows.size != 1:
            raise ValueError(
                "emul_mps: sigma8 requires exactly one stored row at exact "
                "z=0; a nearby redshift is not zero")
        power_at_zero = np.asarray(Pk_2d, dtype=np.float64)[int(zero_rows[0])]
        k = np.asarray(k_array, dtype=np.float64)

        x = k * (_SIGMA8_RADIUS_MPC_OVER_H / h)
        window = np.empty_like(x)
        small = x <= 1.0e-3
        x2 = x[small] ** 2
        window[small] = (1.0 - x2 / 10.0 + x2 ** 2 / 280.0
                         - x2 ** 3 / 15120.0)
        big = ~small
        window[big] = (3.0 * (np.sin(x[big]) - x[big] * np.cos(x[big]))
                       / x[big] ** 3)
        contribution = (k ** 3 * power_at_zero * window ** 2
                        / (2.0 * np.pi ** 2))
        variance = float(np.trapz(contribution, np.log(k)))
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError(
                "emul_mps: the sigma8 variance integral must be finite and "
                "strictly positive; got " + repr(variance))
        edge = float(max(contribution[0], contribution[-1]))
        peak = float(contribution.max())
        if edge > _SIGMA8_EDGE_FRACTION * peak:
            raise ValueError(
                "emul_mps: the stored k grid truncates the sigma8 "
                "integrand: the contribution per unit log(k) at a grid edge "
                "is " + repr(edge) + ", more than "
                + repr(_SIGMA8_EDGE_FRACTION) + " of its peak " + repr(peak)
                + "; the integrand must have decayed at both edges, so "
                "retrain with a wider k range")
        return math.sqrt(variance)

    def get_sigma8(self):
        """Serve the current point's sigma8.

        Returns:
          the derived sigma8 cached by calculate, or None when the
          current point did not compute it.
        """
        return self.current_state.get("derived", {}).get("sigma8")

    def get_param(self, param_name):
        """Serve one derived parameter from the current point's cache.

        Arguments:
          param_name = the derived parameter's name ("sigma8" is the
                       only one this adapter calculates).

        Returns:
          its cached value at the current point, or None when absent.
        """
        return self.current_state.get("derived", {}).get(param_name)
