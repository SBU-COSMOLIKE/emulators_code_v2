"""Validate saved model recipes without importing model implementations.

An artifact recipe is data, not executable Python.  This module therefore
contains only plain registries and value checks.  It does not import Torch,
the model designs, geometry classes, activation factories, or normalization
factories.  A reader can validate the entire recipe before a saved module path
is allowed to reach ``importlib``.

The registries are deliberately closed.  Adding a constructor argument or a
new implementation requires updating this file and its census test.  That
turns a new default into an explicit artifact-format decision instead of a
silent fallback during reconstruction.
"""

import math


MODEL_RECIPE_TOP_LEVEL_KEYS = (
  "cls",
  "name",
  "ia",
  "input_dim",
  "output_dim",
  "compile_mode",
  "needs_geom",
  "kwargs",
)

COMPILE_MODES = ("reduce-overhead", "default", None)
ACTIVATION_NAMES = (
  "H", "power", "multigate", "gated_power", "relu", "tanh")
NORMALIZATION_NAMES = ("affine", "per_feature", "none")
COMPOSITION_MODES = ("plain", "npce", "transfer")
IA_DESIGNS = ("none", "nla", "tatt")
ANALYTIC_TARGET_LAWS = (
  "none", "as_exp2tau_ref", "log_offset", "syren_linear",
  "syren_halofit",
)


def _model_spec(*, name, ia, needs_geom, constructor_parameters):
  """Build one immutable-by-convention model registry entry."""
  injected = {"input_dim", "output_dim"}
  if needs_geom:
    injected.add("geom")
  kwargs = tuple(
    field for field in constructor_parameters if field not in injected)
  return {
    "name": name,
    "ia": tuple(ia),
    "needs_geom": needs_geom,
    "constructor_parameters": tuple(constructor_parameters),
    "kwargs": kwargs,
  }


_MODEL_SPECS = {
  "emulator.designs.plain.ResMLP": _model_spec(
    name="resmlp", ia=(None,), needs_geom=False,
    constructor_parameters=(
      "input_dim", "output_dim", "int_dim_res", "n_blocks", "block_opts")),
  "emulator.designs.plain.ResCNN": _model_spec(
    name="rescnn", ia=(None,), needs_geom=True,
    constructor_parameters=(
      "input_dim", "output_dim", "int_dim_res", "geom", "kernel_size",
      "rescale_kernel", "groups", "separable", "film", "n_blocks",
      "n_blocks_cnn", "gate_init", "head_act", "block_opts")),
  "emulator.designs.plain.ResTRF": _model_spec(
    name="restrf", ia=(None,), needs_geom=True,
    constructor_parameters=(
      "input_dim", "output_dim", "int_dim_res", "geom", "n_heads",
      "n_blocks", "n_blocks_trf", "n_mlp_blocks", "n_tokens",
      "gate_init", "shared_mlp", "film", "head_act", "block_opts")),
  "emulator.designs.ia.TemplateMLP": _model_spec(
    name="resmlp", ia=("nla", "tatt"), needs_geom=False,
    constructor_parameters=(
      "input_dim", "output_dim", "n_amps", "n_templates", "int_dim_res",
      "n_blocks", "block_opts")),
  "emulator.designs.ia.TemplateResCNN": _model_spec(
    name="rescnn", ia=("nla", "tatt"), needs_geom=True,
    constructor_parameters=(
      "input_dim", "output_dim", "n_amps", "n_templates", "int_dim_res",
      "geom", "kernel_size", "rescale_kernel", "groups", "separable",
      "film", "n_blocks", "n_blocks_cnn", "gate_init", "head_act",
      "block_opts")),
  "emulator.designs.ia.TemplateResTRF": _model_spec(
    name="restrf", ia=("nla", "tatt"), needs_geom=True,
    constructor_parameters=(
      "input_dim", "output_dim", "n_amps", "n_templates", "int_dim_res",
      "geom", "n_heads", "n_blocks", "n_blocks_trf", "n_mlp_blocks",
      "gate_init", "shared_mlp", "film", "head_act", "block_opts")),
}

MODEL_RECIPE_CLASSES = tuple(_MODEL_SPECS)


_MODEL_IMPLEMENTATION_IDS = {
  path: "model:" + path + ":v1" for path in MODEL_RECIPE_CLASSES
}
_ACTIVATION_IMPLEMENTATION_IDS = {
  name: "activation:" + name + ":v1" for name in ACTIVATION_NAMES
}
_NORMALIZATION_IMPLEMENTATION_IDS = {
  name: "normalization:" + name + ":v1" for name in NORMALIZATION_NAMES
}
_COMPOSITION_IMPLEMENTATION_IDS = {
  mode: "composition-decoder:" + mode + ":v1"
  for mode in COMPOSITION_MODES
}
_IA_IMPLEMENTATION_IDS = {
  design: "ia-coefficients:" + design + ":v1"
  for design in IA_DESIGNS
}
_ANALYTIC_TARGET_IMPLEMENTATION_IDS = {
  law: "analytic-target:" + law + ":v1"
  for law in ANALYTIC_TARGET_LAWS
}

PARAMETER_GEOMETRY_CLASSES = (
  "emulator.geometries.parameter.ParamGeometry",
  "emulator.geometries.parameter.LogParamGeometry",
  "emulator.geometries.parameter.AmplitudeFactorGeometry",
)
OUTPUT_GEOMETRY_CLASSES = (
  "emulator.geometries.output.DataVectorGeometry",
  "emulator.geometries.output.DiagonalGeometry",
  "emulator.geometries.output.BlockDiagonalGeometry",
  "emulator.geometries.scalar.ScalarGeometry",
  "emulator.geometries.cmb.CmbDiagonalGeometry",
  "emulator.geometries.grid.GridGeometry",
  "emulator.geometries.grid2d.Grid2DGeometry",
)
_GEOMETRY_IMPLEMENTATION_IDS = {
  path: "geometry:" + path + ":v1"
  for path in PARAMETER_GEOMETRY_CLASSES + OUTPUT_GEOMETRY_CLASSES
}
_ANALYTIC_LAWS_BY_OUTPUT_GEOMETRY = {
  "emulator.geometries.output.DataVectorGeometry": ("none",),
  "emulator.geometries.output.DiagonalGeometry": ("none",),
  "emulator.geometries.output.BlockDiagonalGeometry": ("none",),
  "emulator.geometries.scalar.ScalarGeometry": ("none",),
  "emulator.geometries.cmb.CmbDiagonalGeometry": (
    "none", "as_exp2tau_ref"),
  "emulator.geometries.grid.GridGeometry": ("none", "log_offset"),
  "emulator.geometries.grid2d.Grid2DGeometry": (
    "none", "syren_linear", "syren_halofit"),
}

COMPATIBILITY_MANIFEST_SCHEMA = 1
_MANIFEST_KEYS = (
  "schema",
  "model",
  "trunk_activation",
  "head_activation",
  "normalization",
  "composition_decoder",
  "ia_coefficients",
  "analytic_target",
  "parameter_geometry",
  "output_geometry",
)


def _plain_mapping(value, where):
  """Require a native mapping so custom mapping behavior cannot execute."""
  if type(value) is not dict:
    raise TypeError(where + " must be a plain mapping")
  return value


def _exact_keys(value, expected, where):
  """Require one exact closed mapping schema and name every difference."""
  mapping = _plain_mapping(value, where)
  expected_set = set(expected)
  missing = sorted(expected_set - set(mapping))
  unknown = sorted(set(mapping) - expected_set)
  if missing or unknown:
    details = []
    if missing:
      details.append("missing " + repr(missing))
    if unknown:
      details.append("unknown " + repr(unknown))
    raise ValueError(where + " has " + " and ".join(details))
  return mapping


def _native_text(value, where):
  if type(value) is not str or not value:
    raise TypeError(where + " must be nonempty native text")
  return value


def _native_bool(value, where):
  if type(value) is not bool:
    raise TypeError(where + " must be a native boolean")
  return value


def _native_int(value, where, minimum):
  if type(value) is not int or value < minimum:
    raise TypeError(
      where + " must be a native integer >= " + str(minimum))
  return value


def _finite_nonzero_number(value, where):
  if type(value) not in (int, float) or not math.isfinite(float(value)):
    raise TypeError(where + " must be a finite native number")
  if float(value) == 0.0:
    raise ValueError(where + " must be nonzero")
  return value


def _validate_activation(spec, where):
  """Validate one complete activation-factory description."""
  spec = _exact_keys(spec, ("type", "n_gates"), where)
  name = _native_text(spec["type"], where + ".type")
  if name not in ACTIVATION_NAMES:
    raise ValueError(
      where + ".type must be one of " + repr(ACTIVATION_NAMES))
  _native_int(spec["n_gates"], where + ".n_gates", 1)
  return spec


def _validate_block_options(block, where):
  """Validate the complete residual-block constructor description."""
  block = _exact_keys(block, ("n_layers", "act", "norm"), where)
  _native_int(block["n_layers"], where + ".n_layers", 1)
  _validate_activation(block["act"], where + ".act")
  norm = _native_text(block["norm"], where + ".norm")
  if norm not in NORMALIZATION_NAMES:
    raise ValueError(
      where + ".norm must be one of " + repr(NORMALIZATION_NAMES))
  return block


def expected_constructor_parameters(class_path):
  """Return the exact constructor census for one supported model class."""
  if class_path not in _MODEL_SPECS:
    raise ValueError(
      "unknown model class " + repr(class_path) + "; supported classes are "
      + repr(MODEL_RECIPE_CLASSES))
  return _MODEL_SPECS[class_path]["constructor_parameters"]


def expected_recipe_kwargs(class_path):
  """Return the exact serialized constructor fields for one model class."""
  if class_path not in _MODEL_SPECS:
    raise ValueError(
      "unknown model class " + repr(class_path) + "; supported classes are "
      + repr(MODEL_RECIPE_CLASSES))
  return _MODEL_SPECS[class_path]["kwargs"]


def validate_model_recipe(recipe, where="model_recipe"):
  """Validate one complete inert model recipe.

  The function performs no imports and constructs no objects.  It requires
  every supported constructor field, including fields whose Python
  constructor currently has a default.  A missing value is therefore never
  reinterpreted through a future default.  For a structured head,
  ``head_act: None`` is a valid instruction to inherit the trunk activation;
  an absent ``head_act`` key is corruption.
  """
  recipe = _exact_keys(recipe, MODEL_RECIPE_TOP_LEVEL_KEYS, where)
  class_path = _native_text(recipe["cls"], where + ".cls")
  if class_path not in _MODEL_SPECS:
    raise ValueError(
      where + ".cls is not a supported model class: " + repr(class_path))
  spec = _MODEL_SPECS[class_path]

  name = _native_text(recipe["name"], where + ".name")
  if name != spec["name"]:
    raise ValueError(
      where + ".name=" + repr(name) + " disagrees with "
      + repr(class_path) + ", which requires " + repr(spec["name"]))
  ia = recipe["ia"]
  if ia is not None and type(ia) is not str:
    raise TypeError(where + ".ia must be native text or null")
  if ia not in spec["ia"]:
    raise ValueError(
      where + ".ia=" + repr(ia) + " is incompatible with "
      + repr(class_path) + "; allowed " + repr(spec["ia"]))
  _native_int(recipe["input_dim"], where + ".input_dim", 1)
  _native_int(recipe["output_dim"], where + ".output_dim", 1)
  compile_mode = recipe["compile_mode"]
  if compile_mode not in COMPILE_MODES or (
      compile_mode is not None and type(compile_mode) is not str):
    raise ValueError(
      where + ".compile_mode must be one of " + repr(COMPILE_MODES))
  needs_geom = _native_bool(recipe["needs_geom"], where + ".needs_geom")
  if needs_geom is not spec["needs_geom"]:
    raise ValueError(
      where + ".needs_geom=" + repr(needs_geom) + " disagrees with "
      + repr(class_path))

  kwargs = _exact_keys(
    recipe["kwargs"], spec["kwargs"], where + ".kwargs")
  positive_ints = {
    "int_dim_res", "kernel_size", "groups", "n_blocks_cnn", "n_heads",
    "n_blocks_trf", "n_mlp_blocks", "n_amps", "n_templates",
  }
  native_bools = {
    "rescale_kernel", "separable", "film", "shared_mlp",
  }
  for key in positive_ints.intersection(kwargs):
    _native_int(kwargs[key], where + ".kwargs." + key, 1)
  if "n_blocks" in kwargs:
    _native_int(kwargs["n_blocks"], where + ".kwargs.n_blocks", 0)
  for key in native_bools.intersection(kwargs):
    _native_bool(kwargs[key], where + ".kwargs." + key)
  if "kernel_size" in kwargs and kwargs["kernel_size"] % 2 == 0:
    raise ValueError(where + ".kwargs.kernel_size must be odd")
  if "n_tokens" in kwargs:
    if kwargs["n_tokens"] is not None:
      _native_int(kwargs["n_tokens"], where + ".kwargs.n_tokens", 2)
  if "gate_init" in kwargs:
    _finite_nonzero_number(kwargs["gate_init"], where + ".kwargs.gate_init")

  _validate_block_options(
    kwargs["block_opts"], where + ".kwargs.block_opts")
  if "head_act" in kwargs and kwargs["head_act"] is not None:
    _validate_activation(kwargs["head_act"], where + ".kwargs.head_act")

  if class_path == "emulator.designs.plain.ResCNN" \
      and kwargs["groups"] not in (1, 2):
    raise ValueError(where + ".kwargs.groups must be 1 or 2 for ResCNN")
  if ia is not None:
    expected_shape = {"nla": (1, 3), "tatt": (3, 10)}[ia]
    observed_shape = (kwargs["n_amps"], kwargs["n_templates"])
    if observed_shape != expected_shape:
      raise ValueError(
        where + " factored shape " + repr(observed_shape)
        + " disagrees with ia=" + repr(ia) + ", which requires "
        + repr(expected_shape))
    if class_path == "emulator.designs.ia.TemplateResCNN":
      allowed_groups = (1, kwargs["n_templates"], 2 * kwargs["n_templates"])
      if kwargs["groups"] not in allowed_groups:
        raise ValueError(
          where + ".kwargs.groups must be one of " + repr(allowed_groups)
          + " for " + repr(class_path))
  return recipe


def build_runtime_model_recipe(
    *, class_path, name, ia, input_dim, output_dim, needs_geom, kwargs,
    compile_mode=None):
  """Build and validate the inert recipe attached to one live model.

  Model constructors call this after they have materialized every Python
  default.  The result contains only inert data and is therefore safe to
  compare with the independently assembled experiment recipe.  Construction
  itself still permits deliberately custom low-level factories and template
  shapes used by numerical tests; the save boundary applies the closed recipe
  validator and refuses those unregistered choices.
  """
  recipe = {
    "cls": class_path,
    "name": name,
    "ia": ia,
    "input_dim": int(input_dim),
    "output_dim": int(output_dim),
    "compile_mode": compile_mode,
    "needs_geom": bool(needs_geom),
    "kwargs": kwargs,
  }
  return recipe


def _unwrapped_model(model):
  """Return the eager module beneath a torch.compile wrapper, if present."""
  return getattr(model, "_orig_mod", model)


def set_runtime_compile_mode(model, compile_mode):
  """Record the compile choice consumed by ``make_model`` or rebuild."""
  eager = _unwrapped_model(model)
  if not hasattr(eager, "emul_runtime_recipe"):
    raise ValueError(
      "live model has no canonical runtime recipe; use a registered model "
      "constructor")
  recipe = dict(eager.emul_runtime_recipe)
  recipe["compile_mode"] = compile_mode
  validate_model_recipe(recipe, where="live model recipe")
  eager.emul_runtime_recipe = recipe
  return model


def require_runtime_model_recipe(model, expected, where="live model"):
  """Require the live constructor facts to equal a supplied saved recipe."""
  validate_model_recipe(expected, where=where + " expected recipe")
  eager = _unwrapped_model(model)
  if not hasattr(eager, "emul_runtime_recipe"):
    raise ValueError(
      where + " has no canonical runtime recipe; save refuses caller claims "
      "that are not bound to the constructed model")
  observed = eager.emul_runtime_recipe
  validate_model_recipe(observed, where=where + " runtime recipe")
  if observed != expected:
    raise ValueError(
      where + " runtime recipe does not exactly match resolved_model; "
      "the caller described " + repr(expected) + " but constructed "
      + repr(observed))
  return observed


def _component(*, key, value, semantic_id):
  return {key: value, "semantic_id": semantic_id}


def _geometry_component(path, *, kind, where):
  allowed = (PARAMETER_GEOMETRY_CLASSES if kind == "parameter"
             else OUTPUT_GEOMETRY_CLASSES)
  _native_text(path, where)
  if path not in allowed:
    raise ValueError(
      where + " must name a registered " + kind + " geometry; got "
      + repr(path))
  return _component(
    key="path", value=path, semantic_id=_GEOMETRY_IMPLEMENTATION_IDS[path])


def _named_component(value, *, key, allowed, semantic_ids, where):
  """Build one component only after validating its explicit native name."""
  _native_text(value, where)
  if value not in allowed:
    raise ValueError(where + " must be one of " + repr(allowed))
  return _component(key=key, value=value, semantic_id=semantic_ids[value])


def build_compatibility_manifest(
    recipe, *, parameter_geometry_cls, output_geometry_cls,
    composition_mode, ia_design, analytic_law):
  """Build semantic identities for the implementations a recipe selects.

  These identifiers are an explicit compatibility registry, not a Git commit
  or a Python import path treated as proof.  A behavior-changing edit to one
  registered implementation must bump its semantic identifier.
  """
  validate_model_recipe(recipe)
  block = recipe["kwargs"]["block_opts"]
  trunk_name = block["act"]["type"]
  model = _component(
    key="path", value=recipe["cls"],
    semantic_id=_MODEL_IMPLEMENTATION_IDS[recipe["cls"]])
  trunk = _component(
    key="name", value=trunk_name,
    semantic_id=_ACTIVATION_IMPLEMENTATION_IDS[trunk_name])
  if "head_act" not in recipe["kwargs"]:
    head = {"mode": "none"}
  elif recipe["kwargs"]["head_act"] is None:
    head = {
      "mode": "inherit",
      "name": trunk_name,
      "semantic_id": _ACTIVATION_IMPLEMENTATION_IDS[trunk_name],
    }
  else:
    head_name = recipe["kwargs"]["head_act"]["type"]
    head = {
      "mode": "explicit",
      "name": head_name,
      "semantic_id": _ACTIVATION_IMPLEMENTATION_IDS[head_name],
    }
  norm_name = block["norm"]
  normalization = _component(
    key="name", value=norm_name,
    semantic_id=_NORMALIZATION_IMPLEMENTATION_IDS[norm_name])
  composition = _named_component(
    composition_mode, key="mode", allowed=COMPOSITION_MODES,
    semantic_ids=_COMPOSITION_IMPLEMENTATION_IDS,
    where="composition_mode")
  ia_coefficients = _named_component(
    ia_design, key="design", allowed=IA_DESIGNS,
    semantic_ids=_IA_IMPLEMENTATION_IDS, where="ia_design")
  expected_ia = "none" if recipe["ia"] is None else recipe["ia"]
  if ia_design != expected_ia:
    raise ValueError(
      "ia_design=" + repr(ia_design) + " disagrees with model_recipe.ia="
      + repr(recipe["ia"]) + ", which requires " + repr(expected_ia))
  analytic_target = _named_component(
    analytic_law, key="law", allowed=ANALYTIC_TARGET_LAWS,
    semantic_ids=_ANALYTIC_TARGET_IMPLEMENTATION_IDS,
    where="analytic_law")
  allowed_laws = _ANALYTIC_LAWS_BY_OUTPUT_GEOMETRY.get(output_geometry_cls)
  if allowed_laws is not None and analytic_law not in allowed_laws:
    raise ValueError(
      "analytic_law=" + repr(analytic_law) + " is incompatible with "
      + repr(output_geometry_cls) + "; allowed " + repr(allowed_laws))
  return {
    "schema": COMPATIBILITY_MANIFEST_SCHEMA,
    "model": model,
    "trunk_activation": trunk,
    "head_activation": head,
    "normalization": normalization,
    "composition_decoder": composition,
    "ia_coefficients": ia_coefficients,
    "analytic_target": analytic_target,
    "parameter_geometry": _geometry_component(
      parameter_geometry_cls, kind="parameter",
      where="parameter_geometry_cls"),
    "output_geometry": _geometry_component(
      output_geometry_cls, kind="output", where="output_geometry_cls"),
  }


def _compare_manifest(observed, expected, where):
  """Compare a manifest recursively while naming the first mismatch."""
  if type(expected) is dict:
    observed = _exact_keys(observed, tuple(expected), where)
    for key in expected:
      _compare_manifest(observed[key], expected[key], where + "." + key)
    return
  if type(observed) is not type(expected) or observed != expected:
    raise ValueError(
      where + "=" + repr(observed) + " does not match the registered value "
      + repr(expected))


def validate_compatibility_manifest(
    manifest, recipe, *, parameter_geometry_cls, output_geometry_cls,
    composition_mode, ia_design, analytic_law,
    where="compatibility_manifest"):
  """Validate a saved manifest against the closed semantic registries."""
  expected = build_compatibility_manifest(
    recipe,
    parameter_geometry_cls=parameter_geometry_cls,
    output_geometry_cls=output_geometry_cls,
    composition_mode=composition_mode,
    ia_design=ia_design,
    analytic_law=analytic_law)
  _exact_keys(manifest, _MANIFEST_KEYS, where)
  _compare_manifest(manifest, expected, where)
  return manifest
