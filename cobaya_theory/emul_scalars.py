"""Thin cobaya Theory adapter: run saved scalar (derived-parameter) emulators.

This is a shell over emulator.inference.EmulatorPredictor -- it defines no
nn.Module and holds no prediction physics. Its whole job is what Cobaya
expects of a theory component: pick a device, load one predictor (one
rebuilt saved emulator) per configured path root, declare the sampled
parameters those emulators need and the derived parameters they provide
(both read from the saved files' stored names), and at each sampled point
hand the outputs to Cobaya through its "derived" results mapping.

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

import torch
from cobaya.theory import Theory

# The adapter lives in <root>/cobaya_theory/; put <root> on sys.path so the
# training package `emulator` imports (the same prepend emul_cosmic_shear uses).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulator.inference import (                               # noqa: E402
    EmulatorPredictor,
    check_artifacts_fixed_values,
    check_artifacts_pair_up,
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
        """Build the predictors and assemble the requirements + provides.

        Cobaya constructs one instance of this class from the sampling
        YAML's theory block and calls initialize() once, before any
        sampling; extra_args is that block's option mapping, handed
        over by Cobaya as an attribute. This method validates the
        options, picks the device, rebuilds one EmulatorPredictor per
        configured path root, and derives the required inputs and
        provided outputs from the artifacts' own stored names.

        Raises:
          ValueError for an unknown option, an empty emulators list, a
          non-scalar artifact, or two artifacts providing one output.
        """
        super().initialize()
        self._check_extra_args()
        self.device = self._pick_device(self.extra_args.get("device", "cpu"))

        roots = self.extra_args.get("emulators")
        if not roots:
            raise ValueError(
                "emul_scalars: extra_args needs a non-empty 'emulators' list "
                "of saved scalar-emulator path roots (each root -> <root>.h5 "
                "+ <root>.emul).")
        compile_model = bool(self.extra_args.get("compile", False))
        declared = self.extra_args.get("provides")
        rootdir = os.environ.get("ROOTDIR", "")

        # one predictor per root. The requirements are the union of the
        # predictors' stored input names; the provides the union of their
        # stored output names (both artifact facts, never restated in YAML).
        self.predictors = []
        provides = []          # provided output names, in load order
        provided_by = {}       # output name -> the root that provides it
        req = {}               # required input names (a cobaya dict)
        for root in roots:
            path = root if os.path.isabs(root) else os.path.join(rootdir, root)
            predictor = EmulatorPredictor(path, self.device,
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
            for name in predictor.output_names:
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

        # forbid-overlap: a name that is both a required input of
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
                + " are reserved names in Cobaya's results dictionary and "
                "cannot be served as derived parameters")

        # cross-artifact law, LAST: every configuration law above has passed,
        # so the served set is a well-formed one. Only now is it worth asking
        # whether those artifacts describe one cosmology over one region.
        check_artifacts_pair_up(predictors=self.predictors)

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
                "emul_scalars: unrecognized extra_args key(s) "
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
        """The sampled parameters the emulators need.

        Returns:
          a fresh {name: None} mapping (Cobaya's requirement form) over
          the union of the loaded artifacts' stored input names.
        """
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
        """Compute every derived parameter at the current sampled point.

        Cobaya calls this once per sampled point. Every loaded emulator is
        evaluated first and its outputs collected; only after all of them
        succeed are the values written into ``state["derived"]``, so a
        refusal from a later emulator cannot leave a half-filled result.

        Arguments:
          state  = Cobaya's results dictionary for the current sampled
                   point; this method fills its "derived" entry.
          want_derived = True when Cobaya asks for the derived-parameter
                   values this step; when False nothing is written.
          params = the sampled parameter values, by name (each emulator
                   reads the names it needs, in its stored order).

        Returns:
          True. When want_derived is true, state["derived"] is created when
          absent and receives one value per stored output name. Nothing
          else in ``state`` is touched.
        """
        pending = {}
        for predictor in self.predictors:
            outputs = predictor.predict(params)   # {name: value}
            for name, value in outputs.items():
                pending[name] = value
        # Write into state only after every emulator has passed its
        # finite/type/shape checks.  If a later one refuses, this sampled
        # point must not leave earlier outputs behind in Cobaya's results.
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
