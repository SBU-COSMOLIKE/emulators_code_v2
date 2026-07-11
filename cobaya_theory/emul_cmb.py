"""Thin cobaya Theory adapter: serve CMB spectra from saved emulators.

This is a shell over emulator.inference.EmulatorPredictor -- it defines no
nn.Module and holds no prediction physics. Its whole job is the cobaya
contract: pick a device, build one predictor per saved CMB-emulator path
root, declare the sampled parameters the predictors need (read from the
h5s' stored geometry names) and the Cl product they provide, and on each
step assemble the cobaya Cl dict from batch-1 decodes (the imposed
amplitude law is multiplied back inside the predictor's decode, so the
served spectra are physical C_ell).

WHICH spectra exist and each one's multipole range are artifact facts: the
saved CmbDiagonalGeometry stores its spectrum name ("tt" / "te" / "ee" /
"pp"), its multipole grid, and its units, so the YAML lists only path
roots (D-CM5 -- the legacy eval: mask, ord, file, extra, extrapar all
die). Two artifacts claiming the same spectrum is a loud error; a
likelihood requesting a spectrum no artifact provides, or multipoles
beyond an artifact's stored range, is a loud error at must_provide naming
what IS loaded.

PS: path root = a saved emulator's path without extension, resolving to
<root>.h5 (the recipe + geometries) and <root>.emul (the weights); the
cobaya Cl dict = the mapping cobaya likelihoods consume from get_Cl: an
"ell" integer array 0..lmax plus one array per spectrum, aligned with
"ell" and zero at the multipoles below the artifact's range (l = 0, 1
carry no CMB information); raw C_ell = the spectrum without the
l(l+1)/2pi plotting factor.
"""

import os
import sys

import numpy as np
import torch
from cobaya.theory import Theory

# The adapter lives in <root>/cobaya_theory/; put <root> on sys.path so the
# training package `emulator` imports (the same prepend emul_cosmic_shear
# uses).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.inference import EmulatorPredictor  # noqa: E402

# The only extra_args the schema-v2 convention accepts. The legacy ord /
# extrapar / extra / file keys are retired (the h5 recipe + stored names
# replace them); an unknown key errors loudly rather than being ignored.
_ALLOWED_EXTRA_ARGS = ("device", "emulators", "compile")


class emul_cmb(Theory):
    """Cobaya Theory that emulates CMB spectra (the Cl provider).

    extra_args:
      device    = 'cpu' / 'cuda' / 'mps' (default 'cpu'; TPU dropped).
      emulators = list of saved CMB-emulator path roots, one per spectrum
                  (ROOTDIR-relative unless absolute). Each root declares
                  its own spectrum + multipole range + units.
      compile   = optional bool, torch.compile each module on CUDA
                  (default False; batch-1 MCMC latency rarely pays off
                  the compile).
    """

    renames = {}
    extra_args = {}

    def initialize(self):
        """Build the predictors and assemble requirements + the Cl layout."""
        super().initialize()
        self._check_extra_args()
        self.device = self._pick_device(self.extra_args.get("device", "cpu"))

        roots = self.extra_args.get("emulators")
        if not roots:
            raise ValueError(
                "emul_cmb: extra_args needs a non-empty 'emulators' list of "
                "saved CMB-emulator path roots (each root -> <root>.h5 + "
                "<root>.emul).")
        compile_model = bool(self.extra_args.get("compile", False))
        rootdir = os.environ.get("ROOTDIR", "")

        # one predictor per root. The requirements are the union of the
        # predictors' stored input names; the served spectra and their
        # multipole ranges are the artifacts' stored facts (never YAML).
        self.predictors = []
        provided_by = {}       # spectrum name -> the root that provides it
        req = {}               # required input names (a cobaya dict)
        for root in roots:
            path = root if os.path.isabs(root) else os.path.join(rootdir, root)
            predictor = EmulatorPredictor(path, self.device,
                                          compile_model=compile_model)
            # wrong-kind guard (the D-SPE2-4 failure class, one layer up):
            # this theory serves CMB spectrum artifacts only.
            if not predictor._cmb:
                if predictor._scalar:
                    kind, where = "scalar", "emul_scalars"
                elif predictor._grid:
                    kind, where = "background grid", "emul_baosn"
                else:
                    kind, where = "data-vector", "emul_cosmic_shear"
                raise ValueError(
                    "emul_cmb: " + repr(root) + " is not a CMB spectrum "
                    "emulator (its h5 rebuilds a " + kind + " geometry); "
                    "this theory serves CMB artifacts only; that "
                    "emulator belongs in " + where + "'s emulators list")
            # duplicate spectrum across two artifacts = loud: each C_ell
            # must be produced by exactly one emulator.
            if predictor.spectrum in provided_by:
                raise ValueError(
                    "emul_cmb: two emulators provide the spectrum "
                    + repr(predictor.spectrum) + " ("
                    + repr(provided_by[predictor.spectrum]) + " and "
                    + repr(root) + "); each spectrum must come from exactly "
                    "one emulator")
            provided_by[predictor.spectrum] = root
            self.predictors.append(predictor)
            for name in predictor.names:
                req[name] = None

        # units sanity: the temperature/polarization spectra (tt/te/ee)
        # must share one units convention; pp is dimensionless by
        # construction. A mismatch means the artifacts were generated with
        # different dump conventions -- refuse loudly rather than serve a
        # mixed-unit Cl dict.
        units = {}
        for predictor in self.predictors:
            if predictor.spectrum != "pp":
                units[predictor.spectrum] = predictor.units
        if len(set(units.values())) > 1:
            raise ValueError(
                "emul_cmb: the loaded tt/te/ee artifacts disagree on units "
                + repr(units) + "; regenerate the dumps with one convention "
                "(the v2 generator always writes muK2)")
        self._cl_units = next(iter(set(units.values())), "muK2")

        # the Cl-dict layout: one shared 0..lmax_global "ell" axis; each
        # spectrum's array is zero outside its artifact's stored range
        # (below l=2 always; above its own lmax never served, enforced at
        # must_provide).
        self._ell_arrays = {}   # spectrum -> stored multipole grid (ints)
        self._lmax_of = {}      # spectrum -> its artifact's max multipole
        lmax_global = 0
        for predictor in self.predictors:
            ells = predictor.ell.detach().cpu().numpy().astype(int)
            self._ell_arrays[predictor.spectrum] = ells
            self._lmax_of[predictor.spectrum] = int(ells[-1])
            lmax_global = max(lmax_global, int(ells[-1]))
        self._lmax_global = lmax_global

        self._req = req

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly."""
        unknown = []
        for key in self.extra_args:
            if key not in _ALLOWED_EXTRA_ARGS:
                unknown.append(key)
        if unknown:
            raise ValueError(
                "emul_cmb: unrecognized extra_args key(s) "
                f"{sorted(unknown)}. The schema-v2 convention accepts only "
                f"{list(_ALLOWED_EXTRA_ARGS)}; the legacy ord / extrapar / "
                "extra / file keys are retired (the h5 recipe + stored names "
                "replace them, the emulator file IS the emulator).")

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
        """The sampled parameters the emulators need (a cobaya dict)."""
        return self._req

    def get_can_provide(self):
        """The products this theory provides: the Cl dict."""
        return ["Cl"]

    def must_provide(self, **requirements):
        """Validate every Cl request against the loaded artifacts, loudly.

        cobaya calls this with each downstream component's requirements;
        the "Cl" entry maps spectrum name -> requested lmax. A spectrum no
        artifact provides, or an lmax beyond the artifact's stored range,
        cannot be served (an emulator has no accuracy beyond its training
        grid), so both are errors naming what IS loaded -- never a
        silently truncated or zero-padded answer.

        Arguments:
          requirements = the requirement dict cobaya forwards; only the
                         "Cl" key concerns this theory.
        """
        cl_req = requirements.get("Cl")
        if cl_req is None:
            return
        for spectrum, lmax in cl_req.items():
            spec = str(spectrum).lower()
            if spec not in self._lmax_of:
                raise ValueError(
                    "emul_cmb: a likelihood requests the spectrum "
                    + repr(spectrum) + " but no loaded artifact provides "
                    "it; the loaded spectra are "
                    + repr(sorted(self._lmax_of)) + " (one emulator path "
                    "root per spectrum in extra_args.emulators)")
            if int(lmax) > self._lmax_of[spec]:
                raise ValueError(
                    "emul_cmb: a likelihood requests " + spec + " to lmax="
                    + repr(int(lmax)) + " but the loaded artifact stops at "
                    "lmax=" + repr(self._lmax_of[spec]) + "; an emulator "
                    "has no accuracy beyond its training grid, so retrain "
                    "with a wider train_args.lrange or lower the request")

    def calculate(self, state, want_derived=True, **params):
        """Run every predictor and assemble the Cl dict on the state.

        Arguments:
          state  = the cobaya state dict to populate.
          want_derived = cobaya's flag; this theory derives no scalar
                   parameters, so it is accepted and unused.
          params = the sampled parameter values (each predictor reads its
                   own input names in order).

        Returns:
          True; state["Cl"] holds {"ell": (lmax+1,) ints, <spectrum>:
          (lmax+1,) float64 physical C_ell, zero below the artifact's
          multipole range} for every loaded spectrum.
        """
        cl = {"ell": np.arange(self._lmax_global + 1)}
        for predictor in self.predictors:
            spec = predictor.spectrum
            row = np.zeros(self._lmax_global + 1, dtype=np.float64)
            row[self._ell_arrays[spec]] = predictor.predict(params)
            cl[spec] = row
        state["Cl"] = cl
        return True

    def get_Cl(self, ell_factor=False, units="muK2"):
        """Serve the assembled Cl dict for the current sampled point.

        The served convention is the artifacts' own: raw C_ell (no
        l(l+1)/2pi factor), tt/te/ee in the units the dumps carried
        (muK2 from the v2 generator), pp dimensionless. Any OTHER
        convention is refused loudly rather than converted: a silent
        unit conversion here would be a second definition of the dump
        convention (the never-trust-defaults rule, serve side).

        Arguments:
          ell_factor = must be False (raw C_ell); True (the l(l+1)/2pi
                       plotting convention) is not served.
          units      = must equal the artifacts' stored units for
                       tt/te/ee (the v2 dumps write "muK2").

        Returns:
          the {"ell", <spectrum>...} dict calculate assembled.
        """
        if ell_factor:
            raise ValueError(
                "emul_cmb.get_Cl serves raw C_ell only (ell_factor=False), "
                "matching the training dumps and the covariance script; "
                "apply the l(l+1)/2pi factor in the consumer if needed")
        if units != self._cl_units:
            raise ValueError(
                "emul_cmb.get_Cl: requested units " + repr(units) + " but "
                "the loaded artifacts store " + repr(self._cl_units) + "; "
                "the adapter serves the artifact convention and never "
                "converts silently")
        return self.current_state["Cl"]
