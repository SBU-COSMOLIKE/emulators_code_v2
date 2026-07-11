"""Thin cobaya Theory adapter: run a saved cosmic-shear emulator in an MCMC.

This is a shell over emulator.inference.EmulatorPredictor -- it defines no
nn.Module and holds no prediction physics. Its whole job is the cobaya
contract: pick a device, build one predictor per saved-emulator path root,
declare the sampled parameters the predictors need (read from the h5's stored
geometry names), and on each step hand the params to the predictor and stash
the returned data vector as state["cosmic_shear"].

What the legacy Theory hand-typed in the sampling YAML is gone: ord (the
input ordering -- the saved ParamGeometry stores the names in training
order), extrapar (the architecture dims -- the h5 model_recipe rebuilds the
module), and the copied architecture code (rebuild_emulator instantiates the
one training package). Each was a definition/default-drift channel; the
schema-v2 artifact retires them (the never-trust-defaults rule applied to the
MCMC config). What is KEPT: the class / file name emul_cosmic_shear, the
likelihood contract (calculate -> state["cosmic_shear"], get_cosmic_shear),
the list-shaped extra_args (multi-emulator ready, one emulator today), and
the cpu / cuda / mps device pick (TPU dropped -- a device-string re-add).

PS: path root = a saved emulator's path without extension, resolving to
<root>.h5 (the recipe + geometries + optional NPCE base) and <root>.emul (the
weights); fast params = sampler parameters this theory only passes through to
get_requirements (e.g. shear calibration), never entering any network input.
"""

import os
import sys

import numpy as np
import torch
from cobaya.theory import Theory

# The adapter lives in <root>/cobaya_theory/; put <root> on sys.path so the
# training package `emulator` imports. The drivers rely on launch-by-path
# putting their own dir on sys.path[0]; this file sits one folder deeper, so
# it prepends its parent (the repo root beside emulator/) explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.inference import EmulatorPredictor  # noqa: E402

# The only extra_args the schema-v2 convention accepts. The legacy ord /
# extrapar / extra / file keys are retired (see the module docstring); an
# unknown key errors loudly rather than being silently ignored.
_ALLOWED_EXTRA_ARGS = ("device", "emulators", "fast_params", "compile",
                       "dv_return")


class emul_cosmic_shear(Theory):
    """Cobaya Theory that emulates the cosmic-shear data vector.

    extra_args:
      device      = 'cpu' / 'cuda' / 'mps' (default 'cpu'; TPU dropped).
      emulators   = list of saved-emulator path roots, one per emulator
                    (list-shaped for future probes; one cosmic-shear emulator
                    today). ROOTDIR-relative unless absolute.
      fast_params = optional list (per emulator) of sampled parameters to pass
                    through to get_requirements without feeding the network
                    (shear calibration and the like); v1 semantics are
                    requirements-passthrough only.
      compile     = optional bool, torch.compile each module on CUDA (default
                    False; batch-1 MCMC latency rarely pays off the compile).
      dv_return   = optional 'section' (default) | '3x2pt', the shape each
                    predictor returns: 'section' = the emulator's own probe
                    block (the per-probe vector the likelihood glues), '3x2pt'
                    = the full scattered vector. Passed to every predictor.
    """

    renames = {}
    extra_args = {}

    def initialize(self):
        """Build the predictors and assemble the requirements from the h5s."""
        super().initialize()
        self._check_extra_args()
        self.device = self._pick_device(self.extra_args.get("device", "cpu"))

        roots = self.extra_args.get("emulators")
        if not roots:
            raise ValueError(
                "emul_cosmic_shear: extra_args needs a non-empty 'emulators' "
                "list of saved-emulator path roots (each root -> <root>.h5 + "
                "<root>.emul).")
        compile_model = bool(self.extra_args.get("compile", False))
        # the returned shape, passed to every predictor (default 'section':
        # the per-probe block the likelihood glues; '3x2pt' for the full
        # scattered vector). The predictor validates the value.
        dv_return = str(self.extra_args.get("dv_return", "section"))
        rootdir = os.environ.get("ROOTDIR", "")

        # one predictor per root; the requirements are the union of the
        # predictors' stored geometry names -- the YAML never re-declares them.
        self.predictors = []
        req = {}
        for root in roots:
            path = root if os.path.isabs(root) else os.path.join(rootdir, root)
            predictor = EmulatorPredictor(path, self.device,
                                          compile_model=compile_model,
                                          dv_return=dv_return)
            # wrong-kind guard: this theory serves data-vector emulators
            # only; a scalar or CMB artifact has its own adapter, and
            # letting it through would return the wrong object shape
            # silently (the D-SPE2-4 failure class).
            if predictor._scalar:
                raise ValueError(
                    "emul_cosmic_shear: " + repr(root) + " is a scalar "
                    "(derived-parameter) emulator; it belongs in "
                    "emul_scalars' emulators list, not here")
            if predictor._cmb:
                raise ValueError(
                    "emul_cosmic_shear: " + repr(root) + " is a CMB "
                    "spectrum emulator; it belongs in emul_cmb's "
                    "emulators list, not here")
            if predictor._grid:
                raise ValueError(
                    "emul_cosmic_shear: " + repr(root) + " is a "
                    "background (grid) emulator; it belongs in "
                    "emul_baosn's emulators list, not here")
            if predictor._grid2d:
                raise ValueError(
                    "emul_cosmic_shear: " + repr(root) + " is a matter-"
                    "power-spectrum (grid2d) emulator; it belongs in "
                    "emul_mps's emulators list, not here")
            self.predictors.append(predictor)
            for name in predictor.names:
                req[name] = None

        # fast_params (list per emulator): requirements passthrough only. The
        # sampler provides and blocks them as fast (this theory is cheap), but
        # they never enter a network input -- an in-theory analytic use (e.g.
        # applying shear calibration to the dv) is a flagged future step.
        for group in (self.extra_args.get("fast_params") or []):
            names = [group] if isinstance(group, str) else list(group)
            for name in names:
                req[name] = None
        self._req = req

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly."""
        unknown = []
        for key in self.extra_args:
            if key not in _ALLOWED_EXTRA_ARGS:
                unknown.append(key)
        if unknown:
            raise ValueError(
                "emul_cosmic_shear: unrecognized extra_args key(s) "
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
          is unavailable (cuda -> mps -> cpu, matching the legacy Theory).
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

    def calculate(self, state, want_derived=True, **params):
        """Predict cosmic_shear from the sampled params.

        Arguments:
          state  = the cobaya state dict to populate.
          params = the sampled parameter values (each predictor reads its own
                   names in order; fast params are ignored here).

        Returns:
          True; state["cosmic_shear"] holds the physical data vector (one
          emulator today; multiple concatenate in the emulators-list order).
        """
        dvs = []
        for predictor in self.predictors:
            dvs.append(predictor.predict(params))
        state["cosmic_shear"] = (dvs[0] if len(dvs) == 1
                                 else np.concatenate(dvs))
        return True

    def get_cosmic_shear(self):
        """The likelihood contract kept from the legacy Theory."""
        return self.current_state["cosmic_shear"]
