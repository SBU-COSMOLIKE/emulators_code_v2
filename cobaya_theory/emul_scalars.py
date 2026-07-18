"""Thin cobaya Theory adapter: run saved scalar (derived-parameter) emulators.

This is a shell over emulator.inference.EmulatorPredictor -- it defines no
nn.Module and holds no prediction physics. Its whole job is the cobaya
contract: pick a device, build one predictor per saved scalar-emulator path
root, declare the sampled parameters the predictors need and the derived
parameters they provide (both read from the h5s' stored geometry names), and
on each step publish the artifact outputs in Cobaya's derived mapping.

One generic class replaces the legacy per-emulator classes (emultheta,
emulrdrag, ...): each of those hard-coded one getter method per output
(get_H0 / get_omegam / ...) and required a hand-typed provides list in the
sampling YAML. Here the provided names come FROM the artifacts
(get_can_provide_params reads each predictor's stored output names) and one
generic get_param serves any of them, so a new scalar map is a new path root
in the emulators list, not a new class. The never-trust-defaults rule applied
to `provides`: a schema-v2 artifact records both its input and its output
names, so the YAML restates neither.

PS: path root = a saved emulator's path without extension, resolving to
<root>.h5 (the recipe + geometries) and <root>.emul (the weights); provided
parameter = a derived parameter this theory computes and hands to the rest of
the pipeline (H0, omegam, rdrag), so a downstream likelihood or theory can
consume it; required parameter = a sampled input a predictor feeds to its
network.
"""

import os
import sys

from cobaya.theory import Theory

# The adapter lives in <root>/cobaya_theory/; put <root> on sys.path so the
# training package `emulator` imports (the same prepend emul_cosmic_shear uses).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.inference import (                               # noqa: E402
    EmulatorPredictor,
    check_artifacts_fixed_values,
    check_artifacts_pair_up,
)
from cobaya_theory._adapter_contract import (                   # noqa: E402
    exact_bool,
    name_sequence,
    pick_device,
    resolve_emulator_roots,
    validate_extra_args,
)

# The only extra_args the schema-v2 convention accepts. The legacy ord /
# extrapar / extra / file keys are retired (the h5 recipe + stored names
# replace them); an unknown key errors loudly rather than being ignored.
_ALLOWED_EXTRA_ARGS = ("device", "emulators", "provides", "compile")


class emul_scalars(Theory):
    """Cobaya Theory that emulates named derived (scalar) parameters.

    extra_args:
      device    = 'cpu' / 'cuda' / 'mps' (default 'cpu'; TPU dropped).
      emulators = list of saved scalar-emulator path roots, one per emulator
                  (ROOTDIR-relative unless absolute). Each root provides its
                  own stored output names.
      provides  = optional check-only list of derived-parameter names. Every
                  name must belong to the artifact union, but the list does
                  not hide other artifact outputs or invent a new one.
      compile   = optional bool, torch.compile each module on CUDA (default
                  False; batch-1 MCMC latency rarely pays off the compile).
    """

    renames = {}
    extra_args = {}

    def initialize(self):
        """Build the predictors and assemble the requirements + provides."""
        super().initialize()
        self._check_extra_args()
        self.device = pick_device(self.extra_args, adapter="emul_scalars")
        roots = resolve_emulator_roots(
            self.extra_args, adapter="emul_scalars")
        compile_model = exact_bool(
            self.extra_args, "compile", adapter="emul_scalars")
        if "provides" in self.extra_args:
            declared = name_sequence(
                self.extra_args["provides"], adapter="emul_scalars",
                option="provides")
        else:
            declared = None

        # one predictor per root. The requirements are the union of the
        # predictors' stored input names; the provides the union of their
        # stored output names (both artifact facts, never restated in YAML).
        self.predictors = []
        provides = []          # provided output names, in load order
        provided_by = {}       # output name -> the root that provides it
        req = {}               # required input names (a cobaya dict)
        for root in roots:
            predictor = EmulatorPredictor(root, self.device,
                                          compile_model=compile_model)
            # wrong-kind guard: a data-vector artifact rebuilds without
            # output_names, so reject it loudly here rather than dying
            # on the attribute access below.
            if not predictor._scalar:
                if predictor._cmb:
                    kind, where = "CMB spectrum", "emul_cmb"
                elif predictor._grid:
                    kind, where = "background grid", "emul_baosn"
                elif predictor._grid2d:
                    kind, where = "matter-power-spectrum grid", "emul_mps"
                else:
                    kind, where = "data-vector", "emul_cosmic_shear"
                raise ValueError(
                    "emul_scalars: " + repr(root) + " is not a scalar "
                    "emulator (its h5 rebuilds a " + kind + " geometry); "
                    "this theory serves scalar artifacts only; that "
                    "emulator belongs in " + where + "'s emulators list")
            self.predictors.append(predictor)
            # duplicate output across two artifacts = loud: a derived
            # parameter must be produced by exactly one emulator.
            output_names = name_sequence(
                predictor.output_names,
                adapter="emul_scalars",
                option="output_names",
                allow_empty=False,
                label="artifact " + repr(root) + " output_names",
            )
            for name in output_names:
                if name in provided_by:
                    raise ValueError(
                        "emul_scalars: two emulators provide the output "
                        + repr(name) + " (" + repr(provided_by[name]) + " and "
                        + repr(root) + "); each derived parameter must come "
                        "from exactly one emulator")
                provided_by[name] = root
                provides.append(name)
            for name in predictor.names:
                req[name] = None

        # forbid-overlap (ruled): a name that is both a required input of
        # one emulator and a provided output of another would need a chained
        # calculate order (out of scope), so refuse it loudly rather
        # than silently mis-order.
        overlap = []
        for name in req:
            if name in provided_by:
                overlap.append(name)
        if overlap:
            raise ValueError(
                "emul_scalars: parameter(s) " + repr(sorted(overlap)) + " are "
                "both an emulator input and an emulator output; chaining "
                "scalar emulators is out of scope (each output must be a fresh "
                "derived parameter, not another emulator's input)")

        # Optional provides checks a subset of the artifact union. It is not a
        # selector: existing configurations may name only the outputs another
        # component asks for while the adapter still advertises every output
        # recorded by its artifacts.
        if declared is not None:
            missing = []
            for name in declared:
                if name not in provided_by:
                    missing.append(name)
            if missing:
                raise ValueError(
                    "emul_scalars: extra_args provides " + repr(list(declared))
                    + " lists name(s) " + repr(sorted(missing)) + " that no "
                    "loaded emulator provides (the provided set is "
                    + repr(sorted(provided_by)) + "); provides is a subset "
                    "check, never a source of the provided names")

        self._req = req
        self._provides = list(provides)
        reserved = {"derived", "params", "dependency_params"}
        collisions = sorted(reserved.intersection(self._provides))
        if collisions:
            raise ValueError(
                "emul_scalars: output name(s) " + repr(collisions)
                + " are reserved by Cobaya state and cannot be published")

        # horizontal law, LAST: every configuration law above has passed, so
        # the served set is a well-formed one. Only now is it worth asking
        # whether those artifacts are ONE dataset.
        check_artifacts_pair_up(predictors=self.predictors)

    def initialize_with_provider(self, provider):
        """Register the provider and compare directly named fixed values."""
        super().initialize_with_provider(provider)
        check_artifacts_fixed_values(
            predictors=self.predictors,
            provider=provider)

    def _check_extra_args(self):
        """Reject any extra_args key outside the v2 convention, loudly."""
        validate_extra_args(
            self.extra_args, adapter="emul_scalars",
            allowed=_ALLOWED_EXTRA_ARGS,
            retired=("the legacy ord / extrapar / extra / file keys are "
                     "retired because the artifact stores those facts"))

    def get_requirements(self):
        """The sampled parameters the emulators need (a cobaya dict)."""
        return dict(self._req)

    def get_can_provide_params(self):
        """The derived parameters this theory provides (the artifact union).

        Read from the predictors' stored output names, so cobaya learns the
        provided set from the files, not a hand-typed YAML list.

        Returns:
          the list of provided derived-parameter names.
        """
        return list(self._provides)

    def calculate(self, state, want_derived=True, **params):
        """Run every predictor, then publish selected derived outputs once.

        Arguments:
          state  = the cobaya state dict to populate.
          want_derived = whether cobaya wants the derived-parameter outputs
                   this step (they are also written to state["derived"]).
          params = the sampled parameter values (each predictor reads its own
                   input names in order).

        Returns:
          True. When want_derived is true, state["derived"] is created when
          absent and receives the artifact output names. No arbitrary output
          is written at the top level of state.
        """
        pending = {}
        for predictor in self.predictors:
            outputs = predictor.predict(params)   # {name: value}
            for name, value in outputs.items():
                pending[name] = value
        # Publish only after every predictor has passed its finite/type/shape
        # checks.  If a later artifact refuses, this sampled point must not
        # leave earlier scalar outputs behind in Cobaya's state.
        if want_derived:
            current = state.get("derived", {})
            if not isinstance(current, dict):
                raise ValueError(
                    "emul_scalars: state['derived'] must be a mapping when "
                    "want_derived is true")
            derived = dict(current)
            for name in self._provides:
                derived[name] = pending[name]
            state["derived"] = derived
        return True

    def get_param(self, param):
        """Serve one provided derived parameter from the current point's cache.

        One generic getter for every output, replacing the legacy per-output
        get_H0 / get_omegam / ... methods. The calculate step stores the value
        in Cobaya's state["derived"] mapping.

        Arguments:
          param = a provided derived-parameter name.

        Returns:
          its value at the current sampled point.
        """
        if param not in self._provides:
            raise KeyError("emul_scalars does not provide " + repr(param))
        return self.current_state["derived"][param]
