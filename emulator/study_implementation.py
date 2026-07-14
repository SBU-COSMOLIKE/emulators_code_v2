"""Versioned semantic implementation identities for tuning studies.

Source and package-version bytes are provenance, not compatibility: comments,
formatting, checkout location, and operationally equivalent runtime builds do
not change the science a tuner executes.  This registry instead gives the
shared tuning machinery and each family implementation an explicit semantic
version.  Any repo or dependency change that alters a tuner's scientific
objective or training semantics must add a registry entry and advance the
corresponding current version before that code may be used.

PS: a semantic version here is an integer owned by this registry, not a
package-release promise.  Its only job is to say whether two journals may
compare their trials as one experiment.
"""


REGISTRY_SCHEMA = 1

_COMPONENT_REGISTRY = {
  "tuner.study_protocol": {
    1: (
      "canonical manifest authentication, strict legacy refusal, worker "
      "re-authentication, and one manifest-owned default control"
    ),
  },
  "tuner.experiment_resolution": {
    1: "configuration validation and resolved experiment construction",
  },
  "tuner.data_staging_target_law": {
    1: (
      "training and validation row selection, cuts, materialization, and "
      "target-law preparation"
    ),
  },
  "tuner.training_optimizer_scheduler": {
    1: "optimizer, scheduler, numerical policy, and model selection",
  },
  "tuner.model_design": {
    1: "network recipes, blocks, heads, and family design dispatch",
  },
  "tuner.activation_normalization": {
    1: "activation and normalization registries and their execution",
  },
  "tuner.parameter_output_geometry_decoder": {
    1: "parameter encoding plus output whitening and physical decoding",
  },
  "tuner.loss_composition": {
    1: "objective loss composition and selection statistics",
  },
  "tuner.warmstart_transfer": {
    1: "fine-tune and transfer source validation and composition",
  },
  "tuner.analytic_base": {
    1: (
      "analytic rescaling/base selection and composition protocol; concrete "
      "family base semantics live in the selected family component"
    ),
  },
  "tuner.runtime_numeric_contract": {
    1: (
      "scientifically relevant NumPy, Torch, and HDF5 numerical behavior; "
      "package releases alone remain provenance"
    ),
  },
  "tuner.runtime_dataset_parser_contract": {
    1: (
      "scientifically relevant GetDist and CosmoLike dataset/interface "
      "semantics, including environment expansion"
    ),
  },
  "tuner.cosmolike": {
    1: "dense-covariance CosmoLike objective and data-vector geometry",
  },
  "tuner.outputs": {
    1: "named scalar-output objective and parameter geometry",
  },
  "tuner.cmb": {
    1: "diagonal CMB objective with its persisted amplitude law",
  },
  "tuner.grid": {
    1: "background-grid objective with its persisted target law",
  },
  "tuner.grid2d": {
    1: "matter-power-grid objective with its persisted target and base laws",
  },
}

_CURRENT_VERSIONS = {
  "tuner.study_protocol": 1,
  "tuner.experiment_resolution": 1,
  "tuner.data_staging_target_law": 1,
  "tuner.training_optimizer_scheduler": 1,
  "tuner.model_design": 1,
  "tuner.activation_normalization": 1,
  "tuner.parameter_output_geometry_decoder": 1,
  "tuner.loss_composition": 1,
  "tuner.warmstart_transfer": 1,
  "tuner.analytic_base": 1,
  "tuner.runtime_numeric_contract": 1,
  "tuner.runtime_dataset_parser_contract": 1,
  "tuner.cosmolike": 1,
  "tuner.outputs": 1,
  "tuner.cmb": 1,
  "tuner.grid": 1,
  "tuner.grid2d": 1,
}

_FAMILY_COMPONENTS = {
  "cosmolike": "tuner.cosmolike",
  "outputs": "tuner.outputs",
  "cmb": "tuner.cmb",
  "grid": "tuner.grid",
  "grid2d": "tuner.grid2d",
}

_SHARED_COMPONENTS = (
  "tuner.study_protocol",
  "tuner.experiment_resolution",
  "tuner.data_staging_target_law",
  "tuner.training_optimizer_scheduler",
  "tuner.model_design",
  "tuner.activation_normalization",
  "tuner.parameter_output_geometry_decoder",
  "tuner.loss_composition",
  "tuner.warmstart_transfer",
  "tuner.analytic_base",
  "tuner.runtime_numeric_contract",
  "tuner.runtime_dataset_parser_contract",
)


def study_implementation_identity(family):
  """Return the current versioned semantic identity for one tuner family.

  Arguments:
    family = explicit tuner family name.

  Returns:
    a fresh JSON-native mapping containing the registry schema plus the
    shared and family-specific component names and semantic versions.
  """
  if family not in _FAMILY_COMPONENTS:
    supported = ", ".join(sorted(_FAMILY_COMPONENTS))
    raise ValueError(
      "unknown tuner implementation family " + repr(family)
      + "; supported families: " + supported)

  names = list(_SHARED_COMPONENTS)
  names.append(_FAMILY_COMPONENTS[family])
  components = []
  for name in names:
    version = _CURRENT_VERSIONS.get(name)
    versions = _COMPONENT_REGISTRY.get(name)
    if versions is None or version not in versions:
      raise RuntimeError(
        "tuner implementation registry is inconsistent for " + repr(name)
        + "; register the current semantic version before tuning")
    components.append({
      "name": name,
      "version": version,
    })
  return {
    "registry_schema": REGISTRY_SCHEMA,
    "components": components,
  }
