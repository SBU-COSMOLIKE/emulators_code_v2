"""Thin cobaya Theory adapter: background H(z) + distances from saved
grid emulators (the BAOSN family).

This adapter contains no trainable network. ``EmulatorPredictor`` owns the
learned Hubble-rate or distance prediction. This file owns the deterministic
background calculation applied afterward: it forms ``c/H(z)``, integrates
the comoving radial distance, and uses the flat-universe relations for
angular-diameter and luminosity distances. Two saved grid artifacts provide
the two redshift regimes:

    the "Hubble" artifact          H(z) on the SN-range grid [~0, 3]
       │  emulator/background.py   c/H cubic onto the doubled grid,
       │                           cumulative Simpson (the ONE pipeline,
       ▼                           shared with direct scripting)
    chi / D_A / D_L in the SN window

    the "D_M" artifact             the comoving distance, trained
                                   DIRECTLY on the recombination window
                                   (z ~ [1000, 1200]); the network
                                   learned the full integral, so NO
                                   bridging integration through the
                                   query desert exists anywhere.

The getters serve PIECEWISE by query redshift: the SN window integrates
from H; the recombination window interpolates the D_M artifact; a query
in the desert between them is a LOUD error naming both covered windows
(never a silent bridge — the legacy analytic z->1200 extension is not
ported). Flat-only in V1: an omk among the sampled inputs is a loud
error (the legacy curvature formula was dimensionally wrong and is
not reproduced).

PS: path root = a saved emulator's path without extension, resolving to
<root>.h5 + <root>.emul; window = a closed redshift interval an
artifact's stored grid covers; the desert = the interval between the
two windows, where nothing is emulated and no likelihood queries.
"""

import os
import sys

import numpy as np
import torch
from cobaya.theory import Theory
from scipy import interpolate

# The adapter lives in <root>/cobaya_theory/; put <root> on sys.path so
# the training package `emulator` imports (the emul_cosmic_shear prepend).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.inference import (                               # noqa: E402
    EmulatorPredictor,
    check_artifacts_fixed_values,
    check_artifacts_pair_up,
)
from emulator.background import distance_interpolators, C_KMS  # noqa: E402
from emulator.geometries.grid import (                     # noqa: E402
    BACKGROUND_QUANTITY_UNITS,
    validate_background_quantity_units,
)

# The only extra_args the schema-v2 convention accepts (the legacy ord /
# extrapar / extra / file / TMAT / ZLIN keys are retired: the h5 recipe +
# the stored grid/quantity/units/law replace them).
_ALLOWED_EXTRA_ARGS = ("device", "emulators", "compile")

# the products this theory can serve (the cobaya provider dispatches
# get_<product> to it for each).
_PROVIDES = ("Hubble",
             "comoving_radial_distance",
             "angular_diameter_distance",
             "luminosity_distance",
             "angular_diameter_distance_2")


def _require_finite_background_vector(value, *, name, length, positive=False):
    """Validate one background vector before it is cached for Cobaya.

    The reciprocal c/H and the cumulative distance integral are new
    arithmetic performed by this adapter, so a finite emulator output
    alone cannot guarantee a finite result; every derived vector is
    checked before the sampled point's interpolators are cached.

    Arguments:
      value    = the vector to check (array-like).
      name     = what the vector is, named in the refusal.
      length   = the exact required length.
      positive = additionally require every entry > 0 (an H(z) row must
                 be positive before the c/H reciprocal).

    Returns:
      the vector as a numpy array of shape (length,).

    Raises:
      ValueError naming the vector for a wrong shape, a NaN / infinity,
      or a non-positive entry when positive is required.
    """
    array = np.asarray(value)
    expected = (int(length),)
    if array.shape != expected:
        raise ValueError(
            "emul_baosn: " + name + " must have exact shape "
            + repr(expected) + ", got " + repr(array.shape))
    if not np.isfinite(array).all():
        raise ValueError(
            "emul_baosn: " + name + " contains NaN or infinity; no "
            "background result was cached")
    if positive and not (array > 0.0).all():
        raise ValueError(
            "emul_baosn: " + name
            + " must be strictly positive before the c/H calculation; "
            "no background result was cached")
    return array


class emul_baosn(Theory):
    """Cobaya Theory serving the expansion history from grid emulators.

    extra_args:
      device    = 'cpu' / 'cuda' / 'mps' (default 'cpu').
      emulators = list of exactly TWO saved grid-emulator path roots
                  (ROOTDIR-relative unless absolute): one whose stored
                  quantity is "Hubble" (the SN-range H(z)) and one whose
                  quantity is "D_M" (the recombination-window comoving
                  distance). Order does not matter — each artifact
                  declares itself.
      compile   = optional bool, torch.compile on CUDA (default False).
    """

    renames = {}
    extra_args = {}

    def initialize(self):
        """Build the two predictors and assemble the window layout."""
        super().initialize()
        self._check_extra_args()
        self.device = self._pick_device(self.extra_args.get("device", "cpu"))

        roots = self.extra_args.get("emulators")
        if not roots or len(roots) != 2:
            raise ValueError(
                "emul_baosn: extra_args needs an 'emulators' list of exactly "
                "TWO saved grid-emulator path roots (one 'Hubble' + one "
                "'D_M'; each root -> <root>.h5 + <root>.emul), got "
                + repr(roots))
        compile_model = bool(self.extra_args.get("compile", False))
        rootdir = os.environ.get("ROOTDIR", "")

        by_quantity = {}
        req = {}
        for root in roots:
            path = root if os.path.isabs(root) else os.path.join(rootdir, root)
            predictor = EmulatorPredictor(path, self.device,
                                          compile_model=compile_model)
            # wrong-kind guard: grid artifacts only.
            if not predictor._grid:
                if predictor._scalar:
                    kind, where = "scalar", "emul_scalars"
                elif predictor._cmb:
                    kind, where = "CMB spectrum", "emul_cmb"
                elif predictor._grid2d:
                    kind, where = ("matter-power-spectrum grid",
                                   "emul_mps")
                else:
                    kind, where = "data-vector", "emul_cosmic_shear"
                raise ValueError(
                    "emul_baosn: " + repr(root) + " is not a background "
                    "grid emulator (its h5 rebuilds a " + kind
                    + " geometry); this theory serves grid artifacts "
                    "only; that emulator belongs in " + where + "'s "
                    "emulators list")
            quantity = predictor.quantity
            units = predictor.units
            validate_background_quantity_units(
                quantity=quantity,
                units=units,
                where="emul_baosn artifact " + repr(root),
            )
            if predictor.fixed_facts["flat_only"] is not True:
                raise ValueError(
                    "emul_baosn: background artifacts must be flat-only; "
                    "regenerate this artifact with omk fixed to zero")
            if quantity in by_quantity:
                raise ValueError(
                    "emul_baosn: two artifacts both declare quantity "
                    + repr(quantity) + "; the pair must be one "
                    "'Hubble' + one 'D_M'")
            by_quantity[quantity] = predictor
            for name in predictor.names:
                req[name] = None

        for quantity, units in BACKGROUND_QUANTITY_UNITS.items():
            if quantity not in by_quantity:
                raise ValueError(
                    "emul_baosn: no loaded artifact declares quantity "
                    + repr(quantity) + " (loaded: "
                    + repr(sorted(by_quantity)) + "); the pair must be "
                    "one 'Hubble' + one 'D_M'")
            stored_units = by_quantity[quantity].units
            if stored_units != units:
                raise ValueError(
                    "emul_baosn: the " + quantity + " artifact stores "
                    "units " + repr(stored_units) + " but this adapter serves "
                    + repr(units) + "; regenerate the dump with the v2 "
                    "generator (it writes " + repr(units) + ")")
        self.p_h  = by_quantity["Hubble"]
        self.p_dm = by_quantity["D_M"]

        # the two covered windows; everything between them is the desert.
        z_sn  = self.p_h.z.detach().cpu().numpy()
        z_rec = self.p_dm.z.detach().cpu().numpy()
        if z_sn[-1] >= z_rec[0]:
            raise ValueError(
                "emul_baosn: the SN-range grid ends at z = "
                + repr(float(z_sn[-1])) + " but the recombination window "
                "starts at z = " + repr(float(z_rec[0])) + "; the two "
                "windows must be disjoint (SN below recombination)")
        self._sn_max  = float(z_sn[-1])
        self._rec_min = float(z_rec[0])
        self._rec_max = float(z_rec[-1])

        # flat-only: a sampled curvature would silently be
        # ignored by the flat conversions, so it is refused loudly.
        if "omk" in req:
            raise ValueError(
                "emul_baosn: 'omk' is among the emulator inputs; V1 is "
                "FLAT-ONLY. A correct curvature branch does not exist "
                "yet, and the legacy curvature formula was dimensionally "
                "wrong, so it is not reproduced; sample without omk")

        self._req = req

        # cross-artifact law, LAST: the pair is one 'Hubble' + one 'D_M', in
        # the right units, on disjoint windows -- every configuration law
        # above has passed. Only now is it worth asking whether the two
        # artifacts describe one cosmology over one region.
        check_artifacts_pair_up(predictors=[self.p_h, self.p_dm])

    def initialize_with_provider(self, provider):
        """Register the provider and compare directly named fixed values.

        Cobaya calls this once, when the full model exists. The artifacts'
        recorded fixed values are compared against the model's directly
        named constants, and the flat-only rule is enforced against the
        live parameterization: a sampled omk, or one pinned away from
        zero, refuses at startup.

        Arguments:
          provider = the Cobaya provider carrying the resolved model.
        """
        super().initialize_with_provider(provider)
        check_artifacts_fixed_values(
            predictors=[self.p_h, self.p_dm],
            provider=provider)
        parameterization = getattr(
            getattr(provider, "model", None), "parameterization", None)
        if parameterization is None:
            return
        sampled = parameterization.sampled_params()
        constants = parameterization.constant_params()
        if "omk" in sampled or constants.get("omk", 0.0) != 0.0:
            raise ValueError(
                "emul_baosn is flat-only; omk must be fixed to zero")

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly."""
        unknown = []
        for key in self.extra_args:
            if key not in _ALLOWED_EXTRA_ARGS:
                unknown.append(key)
        if unknown:
            raise ValueError(
                "emul_baosn: unrecognized extra_args key(s) "
                f"{sorted(unknown)}. The schema-v2 convention accepts only "
                f"{list(_ALLOWED_EXTRA_ARGS)}; the legacy ord / extrapar / "
                "extra / file / TMAT / ZLIN keys are retired (the h5 recipe "
                "+ stored grid/quantity/units replace them).")

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
        """The sampled parameters the emulators need.

        Returns:
          a fresh {name: None} mapping (Cobaya's requirement form) over
          the union of both artifacts' stored input names.
        """
        return dict(self._req)

    def get_can_provide(self):
        """The background products this theory provides.

        Returns:
          a fresh list of the product names in _PROVIDES (Hubble and the
          four distance getters).
        """
        return list(_PROVIDES)

    def must_provide(self, **requirements):
        """Validate every requested redshift against the two windows.

        A query in the desert between the SN range and the recombination
        window (or beyond either edge) cannot be served (nothing is
        emulated there, and no bridge is ever integrated), so it fails
        HERE, at startup, naming both covered windows.

        Arguments:
          requirements = the requirement dict cobaya forwards; each of
                         this theory's products may carry a {"z": array}.
        """
        for product in _PROVIDES:
            spec = requirements.get(product)
            if not isinstance(spec, dict):
                continue
            if product == "angular_diameter_distance_2":
                z = spec.get("z_pairs")
                if z is not None:
                    self._redshift_pairs(
                        z, who="the " + product + " requirement")
                continue
            z = spec.get("z")
            if z is None:
                continue
            z = np.atleast_1d(np.asarray(z, dtype="float64")).reshape(-1)
            self._check_windows(z, who="the " + product + " requirement")

    def _redshift_pairs(self, value, who):
        """Validate one (z1, z2) pair array for the two-redshift distance.

        Arguments:
          value = the requested pairs, an (N, 2) array-like with
                  z1 <= z2 in every row.
          who   = the requester, named in the refusal.

        Returns:
          the pairs as a float64 numpy array of shape (N, 2), every
          redshift inside the two emulated windows.

        Raises:
          ValueError for a wrong shape, a reversed pair, or a redshift
          in the desert between the windows.
        """
        pairs = np.asarray(value, dtype="float64")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(
                "emul_baosn: " + who + " must have exact shape (N, 2); got "
                + repr(pairs.shape))
        if (pairs[:, 0] > pairs[:, 1]).any():
            raise ValueError(
                "emul_baosn: every redshift pair must satisfy z1 <= z2")
        self._check_windows(pairs.reshape(-1), who=who)
        return pairs

    def _check_windows(self, z, who):
        """Loudly reject any redshift outside the two covered windows.

        Arguments:
          z   = 1-D array of query redshifts.
          who = the requester named in the error message.

        Raises:
          ValueError naming the offending redshifts and both windows.
        """
        z = np.asarray(z, dtype="float64")
        in_sn  = (z >= 0.0) & (z <= self._sn_max)
        in_rec = (z >= self._rec_min) & (z <= self._rec_max)
        bad = z[~(in_sn | in_rec)]
        if bad.size > 0:
            raise ValueError(
                "emul_baosn: " + who + " asks for redshift(s) "
                + repr(bad[:8].tolist()) + " (first 8) outside the two "
                "emulated windows [0, " + repr(self._sn_max) + "] (SN "
                "range, integrated from H) and [" + repr(self._rec_min)
                + ", " + repr(self._rec_max) + "] (recombination D_M); "
                "the desert between them is never emulated and never "
                "bridged — the two-window design leaves the gap loud. "
                "Retrain with wider grids if the likelihood really needs "
                "those redshifts.")

    def calculate(self, state, want_derived=True, **params):
        """Predict both background functions at the current sampled point.

        Arguments:
          state  = Cobaya's results dictionary for the current sampled
                   point; this method fills its "baosn" entry.
          want_derived = Cobaya's flag; this theory derives no scalar
                   parameters, so it is accepted and unused.
          params = the sampled parameter values, by name.

        Returns:
          True; state["baosn"] holds the SN-window interpolators (the
          background.py pipeline output) and the recombination-window
          D_M interpolator.
        """
        out_h = self.p_h.predict(params)      # {"z": grid, "Hubble": row}
        z_h = _require_finite_background_vector(
            out_h["z"], name="the Hubble redshift grid",
            length=len(out_h["z"]))
        h = _require_finite_background_vector(
            out_h["Hubble"], name="the predicted Hubble row",
            length=len(z_h), positive=True)
        itp = distance_interpolators(z_grid=z_h, h_grid=h)
        # c/H and the cumulative integral are new arithmetic performed by this
        # adapter.  Validate them before caching the interpolators; finite model
        # output alone cannot prevent overflow after a reciprocal or integral.
        for name in ("H", "chi", "da", "dl"):
            axis = np.asarray(itp[name].x)
            _require_finite_background_vector(
                itp[name](axis), name="the derived " + name + " grid",
                length=len(axis))
        out_dm = self.p_dm.predict(params)    # {"z": grid, "D_M": row}
        z_dm = _require_finite_background_vector(
            out_dm["z"], name="the D_M redshift grid",
            length=len(out_dm["z"]))
        d_m = _require_finite_background_vector(
            out_dm["D_M"], name="the predicted D_M row",
            length=len(z_dm))
        dm_itp = interpolate.interp1d(z_dm, d_m,
                                      kind='cubic',
                                      assume_sorted=True,
                                      fill_value="extrapolate")
        state["baosn"] = {"itp": itp, "dm_itp": dm_itp}
        return True

    def _chi(self, z):
        """Piecewise comoving distance at the query redshifts (Mpc).

        The SN window reads the integrated pipeline; the recombination
        window the D_M artifact; anything else is the loud desert error.

        Arguments:
          z = scalar or array of query redshifts.

        Returns:
          (n,) comoving distances (flat: D_M = chi).
        """
        z = np.atleast_1d(np.asarray(z, dtype="float64"))
        self._check_windows(z, who="a getter")
        cached = self.current_state["baosn"]
        chi = np.empty_like(z)
        in_sn = z <= self._sn_max
        if in_sn.any():
            chi[in_sn] = cached["itp"]["chi"](z[in_sn])
        if (~in_sn).any():
            chi[~in_sn] = cached["dm_itp"](z[~in_sn])
        return chi

    def get_Hubble(self, z, units="km/s/Mpc"):
        """H at the query redshifts (SN window only — H is not emulated
        around recombination; a D_M-window H query is a loud error).

        Arguments:
          z     = scalar or array of query redshifts.
          units = "km/s/Mpc" (the artifact convention) or "1/Mpc"
                  (H/c, the CAMB alternative); anything else is loud.

        Returns:
          (n,) H(z) in the requested units.
        """
        z = np.atleast_1d(np.asarray(z, dtype="float64"))
        beyond = z[(z < 0.0) | (z > self._sn_max)]
        if beyond.size > 0:
            raise ValueError(
                "emul_baosn.get_Hubble: redshift(s) "
                + repr(beyond[:8].tolist()) + " (first 8) are outside "
                "the SN-range window [0, " + repr(self._sn_max) + "]; "
                "H(z) is emulated only there (the recombination window "
                "carries D_M, not H)")
        h = np.array(
            self.current_state["baosn"]["itp"]["H"](z), copy=True)
        if units == "km/s/Mpc":
            return h
        if units == "1/Mpc":
            return h / C_KMS
        raise ValueError(
            "emul_baosn.get_Hubble: units must be 'km/s/Mpc' or "
            "'1/Mpc', got " + repr(units))

    def get_comoving_radial_distance(self, z):
        """The comoving distance chi(z) in Mpc (piecewise, flat).

        Arguments:
          z = scalar or array of query redshifts, inside the two
              emulated windows.

        Returns:
          (n,) comoving distances in Mpc.
        """
        return self._chi(z)

    def get_angular_diameter_distance(self, z):
        """The angular-diameter distance D_A = chi/(1+z) in Mpc (flat).

        Arguments:
          z = scalar or array of query redshifts, inside the two
              emulated windows.

        Returns:
          (n,) angular-diameter distances in Mpc.
        """
        z = np.atleast_1d(np.asarray(z, dtype="float64"))
        return self._chi(z) / (1.0 + z)

    def get_luminosity_distance(self, z):
        """The luminosity distance D_L = chi*(1+z) in Mpc (flat).

        Arguments:
          z = scalar or array of query redshifts, inside the two
              emulated windows.

        Returns:
          (n,) luminosity distances in Mpc.
        """
        z = np.atleast_1d(np.asarray(z, dtype="float64"))
        return self._chi(z) * (1.0 + z)

    def get_angular_diameter_distance_2(self, z_pairs):
        """D_A between redshift pairs (flat: (chi2 - chi1)/(1+z2)).

        Arguments:
          z_pairs = (n, 2) array-like of (z1, z2) pairs, z1 <= z2.

        Returns:
          (n,) the angular-diameter distance between each pair, Mpc.
        """
        pairs = self._redshift_pairs(
            z_pairs, who="get_angular_diameter_distance_2")
        chi1 = self._chi(pairs[:, 0])
        chi2 = self._chi(pairs[:, 1])
        return (chi2 - chi1) / (1.0 + pairs[:, 1])
