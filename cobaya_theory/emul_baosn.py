"""Thin cobaya Theory adapter: background H(z) + distances from saved
grid emulators (the BAOSN family, D-BSN4).

This is a shell over emulator.inference.EmulatorPredictor and
emulator.background — it defines no nn.Module and re-derives no physics.
Two saved grid artifacts serve everything (the two-regime design,
D-BSN3-A):

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
error citing D-BSN3 (the legacy curvature formula was dimensionally
wrong and is not reproduced).

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
from emulator.inference import EmulatorPredictor      # noqa: E402
from emulator.background import distance_interpolators, C_KMS  # noqa: E402

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
                "emul_baosn: extra_args needs an 'emulators' list of "
                "exactly TWO saved grid-emulator path roots — one "
                "'Hubble' (SN range) and one 'D_M' (recombination "
                "window); got " + repr(roots))
        compile_model = bool(self.extra_args.get("compile", False))
        rootdir = os.environ.get("ROOTDIR", "")

        by_quantity = {}
        req = {}
        for root in roots:
            path = root if os.path.isabs(root) else os.path.join(rootdir,
                                                                 root)
            predictor = EmulatorPredictor(path, self.device,
                                          compile_model=compile_model)
            # wrong-kind guard (the D-SPE2-4 lesson): grid artifacts only.
            if not predictor._grid:
                if predictor._scalar:
                    kind, where = "scalar", "emul_scalars"
                elif predictor._cmb:
                    kind, where = "CMB spectrum", "emul_cmb"
                else:
                    kind, where = "data-vector", "emul_cosmic_shear"
                raise ValueError(
                    "emul_baosn: " + repr(root) + " is not a background "
                    "grid emulator (its h5 rebuilds a " + kind
                    + " geometry); this theory serves grid artifacts "
                    "only; that emulator belongs in " + where + "'s "
                    "emulators list")
            if predictor.quantity in by_quantity:
                raise ValueError(
                    "emul_baosn: two artifacts both declare quantity "
                    + repr(predictor.quantity) + "; the pair must be one "
                    "'Hubble' + one 'D_M'")
            by_quantity[predictor.quantity] = predictor
            for name in predictor.names:
                req[name] = None

        for quantity, units in (("Hubble", "km/s/Mpc"), ("D_M", "Mpc")):
            if quantity not in by_quantity:
                raise ValueError(
                    "emul_baosn: no loaded artifact declares quantity "
                    + repr(quantity) + " (loaded: "
                    + repr(sorted(by_quantity)) + "); the pair must be "
                    "one 'Hubble' + one 'D_M'")
            got = by_quantity[quantity].units
            if got != units:
                raise ValueError(
                    "emul_baosn: the " + quantity + " artifact stores "
                    "units " + repr(got) + " but this adapter serves "
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

        # flat-only (D-BSN3): a sampled curvature would silently be
        # ignored by the flat conversions, so it is refused loudly.
        if "omk" in req:
            raise ValueError(
                "emul_baosn: 'omk' is among the emulator inputs; V1 is "
                "FLAT-ONLY (D-BSN3 — the corrected curvature branch is a "
                "recorded future item, and the legacy formula was "
                "dimensionally wrong, so it is not reproduced)")

        self._req = req

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly."""
        unknown = []
        for key in self.extra_args:
            if key not in _ALLOWED_EXTRA_ARGS:
                unknown.append(key)
        if unknown:
            raise ValueError(
                "emul_baosn: unrecognized extra_args key(s) "
                f"{sorted(unknown)}. The schema-v2 convention accepts "
                f"only {list(_ALLOWED_EXTRA_ARGS)}; the legacy ord / "
                "extrapar / extra / file / TMAT / ZLIN keys are retired "
                "(the h5 recipe + the stored grid replace them).")

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
        """The sampled parameters the emulators need (a cobaya dict)."""
        return self._req

    def get_can_provide(self):
        """The background products this theory provides."""
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
            z = spec.get("z")
            if z is None:
                continue
            z = np.atleast_1d(np.asarray(z, dtype="float64")).reshape(-1)
            if product == "angular_diameter_distance_2":
                z = z.reshape(-1)
            self._check_windows(z, who="the " + product + " requirement")

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
                "bridged (D-BSN3-A). Retrain with wider grids if the "
                "likelihood really needs those redshifts.")

    def calculate(self, state, want_derived=True, **params):
        """Run both predictors and cache the piecewise pipeline.

        Arguments:
          state  = the cobaya state dict to populate.
          want_derived = cobaya's flag; this theory derives no scalar
                   parameters, so it is accepted and unused.
          params = the sampled parameter values.

        Returns:
          True; state["baosn"] holds the SN-window interpolators (the
          background.py pipeline output) and the recombination-window
          D_M interpolator.
        """
        out_h = self.p_h.predict(params)      # {"z": grid, "Hubble": row}
        itp = distance_interpolators(z_grid=out_h["z"],
                                     h_grid=out_h["Hubble"])
        out_dm = self.p_dm.predict(params)    # {"z": grid, "D_M": row}
        dm_itp = interpolate.interp1d(out_dm["z"], out_dm["D_M"],
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
        h = self.current_state["baosn"]["itp"]["H"](z)
        if units == "km/s/Mpc":
            return h
        if units == "1/Mpc":
            return h / C_KMS
        raise ValueError(
            "emul_baosn.get_Hubble: units must be 'km/s/Mpc' or "
            "'1/Mpc', got " + repr(units))

    def get_comoving_radial_distance(self, z):
        """The comoving distance chi(z) in Mpc (piecewise, flat)."""
        return self._chi(z)

    def get_angular_diameter_distance(self, z):
        """The angular-diameter distance D_A = chi/(1+z) in Mpc (flat)."""
        z = np.atleast_1d(np.asarray(z, dtype="float64"))
        return self._chi(z) / (1.0 + z)

    def get_luminosity_distance(self, z):
        """The luminosity distance D_L = chi*(1+z) in Mpc (flat)."""
        z = np.atleast_1d(np.asarray(z, dtype="float64"))
        return self._chi(z) * (1.0 + z)

    def get_angular_diameter_distance_2(self, z_pairs):
        """D_A between redshift pairs (flat: (chi2 - chi1)/(1+z2)).

        Arguments:
          z_pairs = (n, 2) array-like of (z1, z2) pairs, z1 <= z2.

        Returns:
          (n,) the angular-diameter distance between each pair, Mpc.
        """
        pairs = np.atleast_2d(np.asarray(z_pairs, dtype="float64"))
        chi1 = self._chi(pairs[:, 0])
        chi2 = self._chi(pairs[:, 1])
        return (chi2 - chi1) / (1.0 + pairs[:, 1])
