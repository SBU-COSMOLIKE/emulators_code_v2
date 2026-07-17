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
from cobaya.theory import Theory

# The adapter lives in <root>/cobaya_theory/; put <root> on sys.path so the
# training package `emulator` imports. The drivers rely on launch-by-path
# putting their own dir on sys.path[0]; this file sits one folder deeper, so
# it prepends its parent (the repo root beside emulator/) explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.inference import EmulatorPredictor  # noqa: E402
from emulator.inference import (check_artifacts_belong_to,      # noqa: E402
                                check_artifacts_pair_up)
from cobaya_theory._adapter_contract import (                   # noqa: E402
    exact_bool,
    exact_choice,
    fast_parameter_groups,
    pick_device,
    resolve_emulator_roots,
    validate_extra_args,
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
        self.device = pick_device(
            self.extra_args, adapter="emul_cosmic_shear")
        roots = resolve_emulator_roots(
            self.extra_args, adapter="emul_cosmic_shear")
        compile_model = exact_bool(
            self.extra_args, "compile", adapter="emul_cosmic_shear")
        # the returned shape, passed to every predictor (default 'section':
        # the per-probe block the likelihood glues; '3x2pt' for the full
        # scattered vector). The predictor validates the value.
        dv_return = exact_choice(
            self.extra_args, "dv_return", adapter="emul_cosmic_shear",
            choices=("section", "3x2pt"), default="section")
        if "fast_params" in self.extra_args:
            fast_groups = fast_parameter_groups(
                self.extra_args["fast_params"],
                adapter="emul_cosmic_shear", emulator_count=len(roots))
        else:
            fast_groups = [[] for _ in roots]

        # one predictor per root; the requirements are the union of the
        # predictors' stored geometry names -- the YAML never re-declares them.
        self.predictors = []
        req = {}
        for root in roots:
            predictor = EmulatorPredictor(root, self.device,
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
        for group in fast_groups:
            for name in group:
                req[name] = None
        self._req = req

        # horizontal law, LAST: every configuration law above has passed, so
        # the served set is a well-formed one (data-vector artifacts, nothing
        # from another family). Only now is it worth asking whether those
        # artifacts are ONE dataset.
        check_artifacts_pair_up(predictors=self.predictors)
        self._composition = self._build_composition(dv_return)

    def _build_composition(self, dv_return):
        """Return predictors in physical block order with checked widths."""
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
            if (not predictor.section_sizes
                    or any(type(value) is not int or value <= 0
                           for value in predictor.section_sizes)):
                raise ValueError(
                    "emul_cosmic_shear: section_sizes must be nonempty "
                    "positive native integers, got "
                    + repr(predictor.section_sizes))
            sizes = tuple(predictor.section_sizes)
            if (type(predictor.total_size) is not int
                    or predictor.total_size <= 0):
                raise ValueError(
                    "emul_cosmic_shear: total_size must be a positive "
                    "native integer, got " + repr(predictor.total_size))
            total = predictor.total_size
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
            if not blocks:
                raise ValueError(
                    "emul_cosmic_shear: artifact probe "
                    + repr(predictor.probe) + " serves no global blocks")
            if (any(type(block) is not int for block in blocks)
                    or min(blocks) < 0 or max(blocks) >= len(sizes)):
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
        """Prove every served emulator was generated for THIS cosmology.

        Cobaya hands a theory its provider exactly once, when the chain is set
        up, and the provider carries the resolved model -- the same object the
        dataset generator read when it wrote the record these emulators were
        fitted to. So the question is asked once, here, before the first
        sampled point, and never again per point: the facts a chain holds fixed
        (the physics it is not sampling) cannot change while it runs.

        It matters because an emulator generated under a different cosmology
        does not fail. It answers every point, confidently, and every answer is
        wrong. A data vector is the worst place for it: the served vector goes
        straight into a chi2 against the measured one, so an emulator trained
        under fixed physics this chain is not assuming produces a smooth,
        entirely reasonable-looking shear signal that is simply the wrong
        universe's -- and the sampler absorbs the difference into the
        cosmological parameters it is allowed to move, which are the answer the
        analysis is for.

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
        check_artifacts_belong_to(predictors=self.predictors,
                                  provider=provider,
                                  adapter="emul_cosmic_shear")

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly."""
        validate_extra_args(
            self.extra_args, adapter="emul_cosmic_shear",
            allowed=_ALLOWED_EXTRA_ARGS,
            retired=("the legacy ord / extrapar / extra / file keys are "
                     "retired because the artifact stores those facts"))

    def get_requirements(self):
        """The sampled parameters the emulators need (a cobaya dict)."""
        return dict(self._req)

    def calculate(self, state, want_derived=True, **params):
        """Predict cosmic_shear from the sampled params.

        Arguments:
          state  = the cobaya state dict to populate.
          params = the sampled parameter values (each predictor reads its own
                   names in order; fast params are ignored here).

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
        """Return an owned vector; callers cannot alter the provider cache."""
        return np.array(self.current_state["cosmic_shear"], copy=True)
