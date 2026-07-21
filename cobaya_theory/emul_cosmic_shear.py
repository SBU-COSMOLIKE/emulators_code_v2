"""Thin cobaya Theory adapter: run a saved cosmic-shear emulator in an MCMC.

This is a shell over emulator.inference.EmulatorPredictor -- it defines no
nn.Module and holds no prediction physics. Its whole job is what Cobaya
expects of a theory component: pick a device, load one predictor (one
rebuilt saved emulator) per configured path root, declare the sampled
parameters those emulators need (read from the saved files' stored names),
and at each sampled point hand the parameter values to the predictor and
store the returned data vector as state["cosmic_shear"].

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
from emulator.inference import (                               # noqa: E402
    EmulatorPredictor,
    check_artifacts_fixed_values,
    check_artifacts_pair_up,
)

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
            # silently (the wrong-kind-artifact failure class).
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

        # cross-artifact law, LAST: every configuration law above has passed,
        # so the served set is a well-formed one (data-vector artifacts,
        # nothing from another family). Only now is it worth asking whether
        # those artifacts describe one cosmology over one parameter region.
        check_artifacts_pair_up(predictors=self.predictors)
        self._composition = self._build_composition(dv_return)

    def _build_composition(self, dv_return):
        """Plan how the loaded predictors assemble one data vector.

        In 'section' mode every predictor serves its stored probe's
        global blocks; the plan orders the predictors by their first
        block (physical order, independent of the YAML root order) and
        refuses overlaps or inconsistent stored layouts. In '3x2pt' mode
        one predictor serves the full scattered vector.

        Arguments:
          dv_return = the configured return shape, 'section' or '3x2pt'.

        Returns:
          a list of (predictor, width) pairs in assembly order; widths
          are the exact segment lengths calculate later checks against.

        Raises:
          ValueError for a missing stored layout, an inconsistent
          global layout, overlapping blocks, invalid block indices, or
          more than one full-vector emulator.
        """
        if dv_return == "3x2pt":
            if len(self.predictors) != 1:
                raise ValueError(
                    "emul_cosmic_shear: dv_return='3x2pt' serves one full "
                    "global vector, so it cannot combine multiple emulators")
            predictor = self.predictors[0]
            return [(predictor, int(predictor.total_size))]

        first_sizes = None
        first_total = None
        occupied = {}
        plan = []
        for predictor in self.predictors:
            if predictor.section_sizes is None or predictor.probe is None:
                raise ValueError(
                    "emul_cosmic_shear: dv_return='section' requires every "
                    "artifact to store section_sizes and probe")
            sizes = tuple(int(value) for value in predictor.section_sizes)
            total = int(predictor.total_size)
            if sum(sizes) != total:
                raise ValueError(
                    "emul_cosmic_shear: section_sizes sum to "
                    + repr(sum(sizes)) + " but total_size is " + repr(total))
            if first_sizes is None:
                first_sizes, first_total = sizes, total
            elif sizes != first_sizes or total != first_total:
                raise ValueError(
                    "emul_cosmic_shear: section emulators must describe the "
                    "same global layout; got section_sizes/total_size "
                    + repr((first_sizes, first_total)) + " and "
                    + repr((sizes, total)))
            try:
                blocks = tuple(predictor.geom.PROBE_BLOCKS[predictor.probe])
            except (AttributeError, KeyError) as error:
                raise ValueError(
                    "emul_cosmic_shear: artifact probe "
                    + repr(predictor.probe)
                    + " is not registered in its stored layout") from error
            if not blocks or min(blocks) < 0 or max(blocks) >= len(sizes):
                raise ValueError(
                    "emul_cosmic_shear: probe " + repr(predictor.probe)
                    + " names invalid global blocks " + repr(blocks)
                    + " for " + repr(len(sizes)) + " sections")
            for block in blocks:
                if block in occupied:
                    raise ValueError(
                        "emul_cosmic_shear: probes "
                        + repr(occupied[block]) + " and "
                        + repr(predictor.probe) + " both serve global block "
                        + repr(block) + "; section blocks must be disjoint")
                occupied[block] = predictor.probe
            width = sum(sizes[block] for block in blocks)
            plan.append((min(blocks), predictor, width))
        plan.sort(key=lambda item: item[0])
        return [(predictor, width) for _, predictor, width in plan]

    def initialize_with_provider(self, provider):
        """Register the provider and compare directly named fixed values.

        Cobaya calls this once, when the full model exists. Each served
        artifact's recorded fixed values are compared against the model's
        directly named constants; a disagreement refuses at startup.

        Arguments:
          provider = the Cobaya provider carrying the resolved model.
        """
        super().initialize_with_provider(provider)
        check_artifacts_fixed_values(
            predictors=self.predictors,
            provider=provider)

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
        """The sampled parameters the emulators need.

        Returns:
          a fresh {name: None} mapping (Cobaya's requirement form) over
          the stored geometry names plus any fast_params passthrough.
        """
        return dict(self._req)

    def calculate(self, state, want_derived=True, **params):
        """Predict cosmic_shear from the sampled params.

        Arguments:
          state  = Cobaya's results dictionary for the current sampled
                   point; this method fills its "cosmic_shear" entry.
          params = the sampled parameter values, by name (each predictor
                   reads the names it needs, in its stored order; fast
                   parameters are ignored here).

        Returns:
          True; state["cosmic_shear"] holds the physical data vector (one
          emulator today; disjoint sections follow their stored physical
          block order, independent of the YAML root order).
        """
        segments = []
        for predictor, expected_width in self._composition:
            segment = np.asarray(predictor.predict(params))
            if segment.shape != (expected_width,):
                raise ValueError(
                    "emul_cosmic_shear: predictor for probe "
                    + repr(predictor.probe) + " returned shape "
                    + repr(segment.shape) + "; the stored layout requires "
                    + repr((expected_width,)))
            segments.append(segment)
        result = (segments[0] if len(segments) == 1
                  else np.concatenate(segments))
        expected_total = sum(width for _, width in self._composition)
        if result.shape != (expected_total,):
            raise ValueError(
                "emul_cosmic_shear: assembled data vector has shape "
                + repr(result.shape) + "; the composition plan requires "
                + repr((expected_total,)))
        state["cosmic_shear"] = result
        return True

    def get_cosmic_shear(self):
        """Serve the current point's data vector as an owned copy.

        Returns:
          a fresh numpy copy of state["cosmic_shear"], so a likelihood
          that edits its copy cannot alter the cached provider result.
        """
        return np.array(self.current_state["cosmic_shear"], copy=True)
