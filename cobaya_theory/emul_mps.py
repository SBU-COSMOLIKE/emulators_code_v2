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
import torch
from scipy.interpolate import RectBivariateSpline
from cobaya.theory import Theory
from cobaya.log import LoggedError, get_logger

# The adapter lives in <root>/cobaya_theory/; put <root> on sys.path so
# the training package `emulator` imports (the emul_cosmic_shear prepend).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.inference import EmulatorPredictor          # noqa: E402
from emulator.inference import (check_artifacts_belong_to,      # noqa: E402
                                check_artifacts_pair_up)
from emulator import syren_base                           # noqa: E402

# The only extra_args the schema-v2 convention accepts (the legacy
# model_file / metadata_file / nl_* / use_syren / param_order keys are
# retired: the h5 recipe + the stored grids/laws replace them).
_ALLOWED_EXTRA_ARGS = ("device", "emulators", "compile")

# the legacy low-k blend constants (verbatim): below k_t the boost is
# pinned to 1 (linear scales), with a sharpness-n exponential turn-on.
_BLEND_K_T = 0.005   # [1/Mpc]
_BLEND_N   = 2.0     # 1 = pure exponential, larger = sharper transition


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
        z, k = (np.atleast_1d(x) for x in [z, k])
        if len(z) < 4:
            raise ValueError('Require at least four redshifts for Pk interpolation.'
                             'Consider using Pk_grid if you just need a small number'
                             'of specific redshifts (doing 1D splines in k yourself).')
        z, k, P_or_logP = np.array(z), np.array(k), np.array(P_or_logP)
        i_z = np.argsort(z)
        i_k = np.argsort(k)
        self.logsign = logsign
        self.z, self.k, P_or_logP = z[i_z], k[i_k], P_or_logP[i_z, :][:, i_k]
        self.zmin, self.zmax = self.z[0], self.z[-1]
        self.extrap_kmin, self.extrap_kmax = extrap_kmin, extrap_kmax
        logk = np.log(self.k)
        if extrap_kmin and extrap_kmin < self.input_kmin:
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
        if extrap_kmax and extrap_kmax > self.input_kmax:
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
        min_z, max_z = min(z), max(z)
        if min_z < self.zmin and not np.allclose(min_z, self.zmin):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to z={min(z)} "
                              f"(minimum z computed is {self.zmin}).")
        if max_z > self.zmax and not np.allclose(max_z, self.zmax):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to z={max(z)} "
                              f"(maximum z computed is {self.zmax}).")
        k = np.atleast_1d(k).flatten()
        min_k, max_k = min(k), max(k)
        if min_k < self.kmin and not np.allclose(min_k, self.kmin):
            raise LoggedError(get_logger(self.__class__.__name__),
                              f"Not possible to extrapolate to k={min(k)} 1/Mpc "
                              f"(minimum k possible is {self.kmin} 1/Mpc).")
        if max_k > self.kmax and not np.allclose(max_k, self.kmax):
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

    def initialize(self):
        """Build the two predictors; check grids, laws, requirements."""
        super().initialize()
        self._check_extra_args()
        self.device = self._pick_device(self.extra_args.get("device", "cpu"))

        roots = self.extra_args.get("emulators")
        if not roots or len(roots) != 2:
            raise ValueError(
                "emul_mps: extra_args needs an 'emulators' list of "
                "exactly TWO saved grid2d-emulator path roots — one "
                "'pklin' and one 'boost'; got " + repr(roots))
        compile_model = bool(self.extra_args.get("compile", False))
        rootdir = os.environ.get("ROOTDIR", "")

        by_quantity = {}
        req = {}
        for root in roots:
            path = root if os.path.isabs(root) else os.path.join(rootdir,
                                                                 root)
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

        for quantity in ("pklin", "boost"):
            if quantity not in by_quantity:
                raise ValueError(
                    "emul_mps: no loaded artifact declares quantity "
                    + repr(quantity) + " (loaded: "
                    + repr(sorted(by_quantity)) + "); the pair must be "
                    "one 'pklin' + one 'boost'")
        self.p_lin   = by_quantity["pklin"]
        self.p_boost = by_quantity["boost"]

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
        self._z = z1
        self._k = k1

        # a syren law reads named cosmology values (syren_params_from's
        # one rule) beyond the artifact inputs; require them so cobaya
        # hands them to calculate. w / wa ride the artifact inputs when
        # the training sampled them (an absent EoS = LCDM, the same
        # rule the generator applied — the two sides cannot disagree).
        if (self.p_lin.law != "none") or (self.p_boost.law != "none"):
            for name in ("As", "ns", "H0", "omegab", "omegam"):
                req[name] = None
        self._req = req

        # horizontal law, LAST: the pair is one 'pklin' + one 'boost' on one
        # shared (z, k) grid -- every configuration law above has passed. Only
        # now is it worth asking whether the two artifacts are ONE dataset.
        check_artifacts_pair_up(predictors=[self.p_lin, self.p_boost])

    def initialize_with_provider(self, provider):
        """Prove both served emulators were generated for THIS cosmology.

        Cobaya hands a theory its provider exactly once, when the chain is set
        up, and the provider carries the resolved model -- the same object the
        dataset generator read when it wrote the record these two emulators
        were fitted to. So the question is asked once, here, before the first
        sampled point, and never again per point: the facts a chain holds fixed
        (the physics it is not sampling) cannot change while it runs.

        It matters because an emulator generated under a different cosmology
        does not fail. It answers every point, confidently, and every answer is
        wrong. Here the wrongness is doubly hidden: the boost multiplies the
        linear spectrum point for point, so a P(k, z) built under fixed physics
        this chain is not assuming (a different neutrino mass, say, which
        suppresses small-scale power) comes out perfectly smooth and positive,
        passes every finiteness check in calculate, and lands in CosmoLike as a
        clean spectrum with the wrong amplitude -- which is precisely the
        quantity the survey is measuring.

        Arguments:
          provider = the cobaya Provider, carrying the resolved global model.

        Returns:
          None. The method is called for its refusal.

        Raises:
          ValueError when the provider cannot hand over the model, or when a
          served artifact was generated under a cosmology this chain is not
          sampling.
        """
        super().initialize_with_provider(provider)
        check_artifacts_belong_to(predictors=[self.p_lin, self.p_boost],
                                  provider=provider,
                                  adapter="emul_mps")

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly."""
        unknown = []
        for key in self.extra_args:
            if key not in _ALLOWED_EXTRA_ARGS:
                unknown.append(key)
        if unknown:
            raise ValueError(
                "emul_mps: unrecognized extra_args key(s) "
                f"{sorted(unknown)}. The schema-v2 convention accepts "
                f"only {list(_ALLOWED_EXTRA_ARGS)}; the legacy "
                "model_file / metadata_file / nl_* / use_syren / "
                "param_order keys are retired (the h5 recipe + the "
                "stored grids and laws replace them).")

    @staticmethod
    def _pick_device(requested):
        """Resolve the requested device to cpu / cuda / mps.

        Arguments:
          requested = the extra_args 'device' string (cpu / cuda / mps).

        Returns:
          a torch.device, falling back to cpu when the requested
          accelerator is unavailable (cuda -> mps -> cpu).
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
        """The sampled parameters the emulators + bases need."""
        return self._req

    def get_can_support_params(self):
        return ['Pk_grid', 'Pk_interpolator', 'sigma8']

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
        out_lin = _require_mps_surface(
            self.p_lin.predict(params)["pklin"],
            name="the linear-emulator output", shape=surface_shape)
        if out_lin is None:
            self.log.debug("non-finite linear-emulator output at "
                           f"params={params} — rejecting point.")
            return False
        need_base = (self.p_lin.law != "none"
                     or self.p_boost.law != "none")
        if need_base:
            amplitude_name = "As_1e9" if "As_1e9" in params else "As"
            for name in (amplitude_name, "ns", "H0", "omegab", "omegam"):
                _require_finite_syren_parameter(params, name)
            if "w" in params:
                _require_finite_syren_parameter(params, "w")
            elif "w0" in params:
                _require_finite_syren_parameter(params, "w0")
            if "wa" in params:
                _require_finite_syren_parameter(params, "wa")
            (as_1e9, ns, H0, Ob, Om,
             w0, wa) = syren_base.syren_params_from(params)
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
            self.p_boost.predict(params)["boost"],
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
            if "sigma8" in self.output_params:
                sigma8 = self._compute_sigma8(
                    pk_lin, k, z, z_eval=0.0)
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
        """
        if var_pair != ("delta_tot", "delta_tot"):
            raise LoggedError(
                self.log,
                f"emul_mps only supports delta_tot power spectra, "
                f"not {var_pair}")
        key = ("Pk_grid", nonlinear) + tuple(sorted(var_pair))
        if key in self.current_state:
            return self.current_state[key]
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
        key = (("Pk_interpolator", nonlinear, extrap_kmin, extrap_kmax)
               + tuple(sorted(var_pair)))
        if key in self.current_state:
            return self.current_state[key]
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
        self.current_state[key] = result
        return result

    def _compute_sigma8(self, Pk_2d, k_array, z_array, z_eval=0.0):
        """sigma8 from the linear P(k) at redshift z_eval (legacy math:
        the R = 8 Mpc/h... note the legacy integrates in k [1/Mpc] with
        R = 8 — ported verbatim, its convention).

        Arguments:
          Pk_2d   = (nz, nk) the linear P(k, z), Mpc^3.
          k_array = (nk,) wavenumbers, 1/Mpc.
          z_array = (nz,) the grid redshifts.
          z_eval  = the redshift sigma8 is evaluated at (0 by default).

        Returns:
          sigma8 at z_eval (a float).
        """
        z_idx = np.argmin(np.abs(z_array - z_eval))
        if np.abs(z_array[z_idx] - z_eval) > 0.01:
            from scipy.interpolate import interp1d
            Pk_interp = interp1d(z_array, Pk_2d, axis=0, kind='cubic')
            Pk_z = Pk_interp(z_eval)
        else:
            Pk_z = Pk_2d[z_idx, :]

        R = 8.0
        x = k_array * R
        W = np.zeros_like(x)
        mask = x > 1e-6
        W[mask] = 3.0 * (np.sin(x[mask])
                         - x[mask] * np.cos(x[mask])) / x[mask] ** 3
        W[~mask] = 1.0

        integrand = k_array ** 2 * Pk_z * W ** 2 / (2.0 * np.pi ** 2)
        log_k = np.log(k_array)
        return np.sqrt(np.trapz(integrand * k_array, log_k))

    def get_sigma8(self):
        return self.current_state.get("derived", {}).get("sigma8")

    def get_param(self, param_name):
        return self.current_state.get("derived", {}).get(param_name)
