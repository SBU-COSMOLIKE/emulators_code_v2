"""Check the saved instructions used to rebuild one neural network.

The recipe is data, not executable Python.  This module therefore
contains only plain registries and value checks.  It does not import Torch,
the model designs, geometry classes, activation factories, or normalization
factories.  A reader can validate the entire recipe before a saved module path
is allowed to reach ``importlib``.

The class list and saved keyword lists are closed. Adding a constructor
argument requires updating this file and its census test, so rebuilding never
silently adopts a new Python default.
"""

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


_MODEL_SPECS = {
  "emulator.designs.plain.ResMLP": {
    "name": "resmlp", "ia": (None,), "needs_geom": False,
    "kwargs": ("int_dim_res", "n_blocks", "block_opts"),
  },
  "emulator.designs.plain.ResCNN": {
    "name": "rescnn", "ia": (None,), "needs_geom": True,
    "kwargs": (
      "int_dim_res", "kernel_size",
      "rescale_kernel", "groups", "separable", "film", "n_blocks",
      "n_blocks_cnn", "gate_init", "head_act", "block_opts"),
  },
  "emulator.designs.plain.ResTRF": {
    "name": "restrf", "ia": (None,), "needs_geom": True,
    "kwargs": (
      "int_dim_res", "n_heads",
      "n_blocks", "n_blocks_trf", "n_mlp_blocks", "n_tokens",
      "gate_init", "shared_mlp", "film", "head_act", "block_opts"),
  },
  "emulator.designs.ia.TemplateMLP": {
    "name": "resmlp", "ia": ("nla", "tatt"), "needs_geom": False,
    "kwargs": (
      "n_amps", "n_templates", "int_dim_res", "n_blocks", "block_opts"),
  },
  "emulator.designs.ia.TemplateResCNN": {
    "name": "rescnn", "ia": ("nla", "tatt"), "needs_geom": True,
    "kwargs": (
      "n_amps", "n_templates", "int_dim_res", "kernel_size",
      "rescale_kernel", "groups", "separable",
      "film", "n_blocks", "n_blocks_cnn", "gate_init", "head_act",
      "block_opts"),
  },
  "emulator.designs.ia.TemplateResTRF": {
    "name": "restrf", "ia": ("nla", "tatt"), "needs_geom": True,
    "kwargs": (
      "n_amps", "n_templates", "int_dim_res", "n_heads", "n_blocks",
      "n_blocks_trf", "n_mlp_blocks", "gate_init", "shared_mlp", "film",
      "head_act", "block_opts"),
  },
}

MODEL_RECIPE_CLASSES = tuple(_MODEL_SPECS)


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


def _validate_activation(spec, where):
  """Require a complete description of one known activation."""
  spec = _exact_keys(spec, ("type", "n_gates"), where)
  name = _native_text(spec["type"], where + ".type")
  if name not in ACTIVATION_NAMES:
    raise ValueError(
      where + ".type must be one of " + repr(ACTIVATION_NAMES))
  return spec


def _validate_block_options(block, where):
  """Require every saved residual-block field and known factory name."""
  block = _exact_keys(block, ("n_layers", "act", "norm"), where)
  _validate_activation(block["act"], where + ".act")
  norm = _native_text(block["norm"], where + ".norm")
  if norm not in NORMALIZATION_NAMES:
    raise ValueError(
      where + ".norm must be one of " + repr(NORMALIZATION_NAMES))
  return block


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

  kwargs = _exact_keys(recipe["kwargs"], spec["kwargs"], where + ".kwargs")
  _validate_block_options(
    kwargs["block_opts"], where + ".kwargs.block_opts")
  if "head_act" in kwargs and kwargs["head_act"] is not None:
    _validate_activation(kwargs["head_act"], where + ".kwargs.head_act")
  return recipe


def record_model_recipe(
    *, class_path, name, ia, input_dim, output_dim, needs_geom, kwargs,
    compile_mode=None):
  """Record the constructor values used by one live model."""
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
  """Add the selected compile mode to a live model's recipe."""
  eager = _unwrapped_model(model)
  if not hasattr(eager, "model_recipe"):
    raise ValueError(
      "live model has no constructor recipe; use a registered model "
      "constructor")
  recipe = dict(eager.model_recipe)
  recipe["compile_mode"] = compile_mode
  eager.model_recipe = recipe
  return model


def check_model_matches_recipe(model, expected, where="live model"):
  """Require a live model to match the recipe supplied by its caller."""
  eager = _unwrapped_model(model)
  if not hasattr(eager, "model_recipe"):
    raise ValueError(
      where + " has no constructor recipe; save refuses caller claims "
      "that are not bound to the constructed model")
  observed = eager.model_recipe
  if observed != expected:
    raise ValueError(
      where + " constructor recipe does not exactly match resolved_model; "
      "the caller described " + repr(expected) + " but constructed "
      + repr(observed))
  return observed
