"""Check the saved instructions used to rebuild one neural network.

A saved emulator carries a "model recipe": a plain mapping that names the
network class, its input and output sizes, and every constructor option
needed to rebuild it.  The constructor options travel under the key
"kwargs", Python's shorthand for keyword arguments — the named options a
constructor accepts.  The recipe is data, not executable Python.  This
module therefore contains only plain lists of accepted values and value
checks.  It does not import Torch, the model designs, the geometry classes,
or the helpers that build activations and normalizations from their saved
names.  A reader can validate the entire recipe before a saved module path
is allowed to reach ``importlib``, Python's import machinery.

The class list and saved keyword lists are closed: a recipe that names a
class or keyword outside them is refused.  Adding a constructor argument
requires updating this file and the test that checks these lists, so
rebuilding never silently adopts a new Python default.
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

# torch.compile rewrites a model's Python into fused GPU kernels for
# speed; the mode names its optimization profile, and None means the
# model runs eagerly (its Python executed directly, uncompiled).
COMPILE_MODES = ("reduce-overhead", "default", None)
# the registered activation-family names (the nonlinear functions
# between network layers) and the per-block normalization choices a
# recipe may name.
ACTIVATION_NAMES = (
  "H", "power", "multigate", "gated_power", "relu", "tanh")
NORMALIZATION_NAMES = ("affine", "per_feature", "none")


# One entry per rebuildable class. "name" is the public architecture
# name (resmlp / rescnn / restrf: residual multilayer perceptron,
# convolutional, and transformer designs); "ia" lists the intrinsic-
# alignment designs the class supports (None = not an IA model; see
# record_model_recipe for what IA is); "needs_geom" says whether the
# constructor consumes the output geometry; "kwargs" is the closed
# list of constructor fields the recipe must carry.
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
  """Require one plain dict, so custom mapping code cannot execute.

  The recipe is read from a saved file; a mapping SUBCLASS could run its
  own __getitem__ during validation, which would be executing code from
  the saved file. Only the exact built-in dict type is accepted.

  Arguments:
    value = the candidate mapping read from the recipe.
    where = the recipe path being validated, named in the refusal.

  Returns:
    the mapping unchanged, when its type is exactly dict.

  Raises:
    TypeError naming the path for any other type.
  """
  if type(value) is not dict:
    raise TypeError(where + " must be a plain mapping")
  return value


def _exact_keys(value, expected, where):
  """Require a mapping to carry exactly the expected keys, no more, no less.

  The recipe schema is closed: a missing key would be silently replaced by
  a future code default (the drift this module exists to prevent), and an
  unknown key means the writer and reader disagree about the schema. Both
  directions are collected and reported together, so one refusal shows the
  complete difference.

  Arguments:
    value    = the candidate mapping.
    expected = the exact key set the schema requires.
    where    = the recipe path being validated, named in the refusal.

  Returns:
    the mapping unchanged, when its keys equal the expected set.

  Raises:
    TypeError from the plain-dict check; ValueError listing the missing
    and unknown keys.
  """
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
  """Require one nonempty plain string (no str subclasses, no bytes).

  Arguments:
    value = the candidate value read from the recipe.
    where = the recipe path being validated, named in the refusal.

  Returns:
    the string unchanged.

  Raises:
    TypeError when the value is empty or not exactly a str.
  """
  if type(value) is not str or not value:
    raise TypeError(where + " must be nonempty native text")
  return value


def _native_bool(value, where):
  """Require one plain Boolean (True == 1 in Python, so ints are refused).

  Arguments:
    value = the candidate value read from the recipe.
    where = the recipe path being validated, named in the refusal.

  Returns:
    the Boolean unchanged.

  Raises:
    TypeError when the value is not exactly a bool.
  """
  if type(value) is not bool:
    raise TypeError(where + " must be a native boolean")
  return value


def _native_int(value, where, minimum):
  """Require one plain integer at or above ``minimum``.

  Arguments:
    value   = the candidate value read from the recipe.
    where   = the recipe path being validated, named in the refusal.
    minimum = the smallest accepted value (inclusive).

  Returns:
    the integer unchanged.

  Raises:
    TypeError when the value is not exactly an int, or is below minimum.
  """
  if type(value) is not int or value < minimum:
    raise TypeError(
      where + " must be a native integer >= " + str(minimum))
  return value


def _validate_activation(spec, where):
  """Validate one saved activation description ({"type", "n_gates"}).

  Arguments:
    spec  = the activation mapping stored by the recipe writer
            (activation_factory_recipe).
    where = the recipe path being validated, named in every refusal.

  Returns:
    the mapping unchanged, when its type names a registered activation.

  Raises:
    TypeError / ValueError from the schema checks; ValueError when the
    type is not one of ACTIVATION_NAMES.
  """
  spec = _exact_keys(spec, ("type", "n_gates"), where)
  name = _native_text(spec["type"], where + ".type")
  if name not in ACTIVATION_NAMES:
    raise ValueError(
      where + ".type must be one of " + repr(ACTIVATION_NAMES))
  return spec


def _validate_block_options(block, where):
  """Validate one saved ResBlock description ({"n_layers", "act", "norm"}).

  Arguments:
    block = the block-options mapping stored by the recipe writer
            (materialized_block_recipe).
    where = the recipe path being validated, named in every refusal.

  Returns:
    the mapping unchanged, when the activation is registered and the
    norm names one of NORMALIZATION_NAMES.

  Raises:
    TypeError / ValueError from the schema and activation checks;
    ValueError for an unregistered norm name.
  """
  block = _exact_keys(block, ("n_layers", "act", "norm"), where)
  _validate_activation(block["act"], where + ".act")
  norm = _native_text(block["norm"], where + ".norm")
  if norm not in NORMALIZATION_NAMES:
    raise ValueError(
      where + ".norm must be one of " + repr(NORMALIZATION_NAMES))
  return block


def expected_recipe_kwargs(class_path):
  """Look up the exact constructor fields one model class serializes.

  Arguments:
    class_path = the model's import path, e.g.
                 "emulator.designs.plain.ResMLP".

  Returns:
    the tuple of kwargs key names the recipe must carry for that class
    (a closed list; a new constructor argument must be added here and in
    the test that checks these lists before it can be saved).

  Raises:
    ValueError listing the supported classes when the path is unknown.
  """
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
  reinterpreted through a future default.  For a structured head (the
  model's final output stage; the "trunk" is the shared body beneath it),
  ``head_act: None`` is a valid instruction to inherit the trunk's
  activation; an absent ``head_act`` key is corruption.

  Arguments:
    recipe = the complete recipe mapping read from the artifact (keys
             MODEL_RECIPE_TOP_LEVEL_KEYS).
    where  = the label used in every refusal (default "model_recipe").

  Returns:
    the recipe unchanged, once every field passes.

  Raises:
    TypeError / ValueError naming the exact recipe path that failed: an
    unsupported class, a name / ia / needs_geom value that disagrees with
    the class registry, a bad dimension, an unknown compile mode, or a
    malformed kwargs block.
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
  """Assemble the recipe mapping a model constructor stores on itself.

  Every model constructor calls this with the values it ACTUALLY used
  (defaults materialized), and keeps the result as its ``model_recipe``
  attribute; save_emulator later writes that attribute to the .h5, and
  check_model_matches_recipe compares it against the caller's claim.

  Arguments:
    class_path   = the model's import path (a MODEL_RECIPE_CLASSES entry).
    name         = the public architecture name ("resmlp" / "rescnn" /
                   "restrf").
    ia           = the factored-IA design name ("nla" / "tatt") or None.
                   IA is intrinsic alignment: galaxy shapes correlate
                   with their local environment, not only with lensing,
                   and the factored designs emulate that contribution
                   as separate templates; "nla" and "tatt" are the two
                   supported alignment models.
    input_dim    = number of model inputs: the width of one parameter
                   vector after the input geometry encodes it.
    output_dim   = number of model outputs: the width of one data vector
                   in the recentered, rescaled units the network learns in.
    needs_geom   = whether the constructor consumed the output geometry
                   (the structured heads do; the plain trunks do not).
    kwargs       = the class-specific constructor fields, defaults
                   materialized (the expected_recipe_kwargs list).
    compile_mode = the torch.compile mode the run selected, or None for
                   an eager model.

  Returns:
    the recipe mapping, in the exact top-level schema
    MODEL_RECIPE_TOP_LEVEL_KEYS.
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
  """Reach the eager module beneath a torch.compile wrapper.

  torch.compile returns a wrapper holding the real module as
  ``_orig_mod``; attributes such as ``model_recipe`` live on the real
  module. An uncompiled model (an "eager" model, in Torch's words: one
  that runs its Python directly) is returned unchanged.

  Arguments:
    model = a live model, compiled or eager.

  Returns:
    the eager nn.Module that owns the parameters and the recipe.
  """
  return getattr(model, "_orig_mod", model)


def set_runtime_compile_mode(model, compile_mode):
  """Record the run's selected compile mode on a live model's recipe.

  The constructor cannot know the compile mode (compilation happens
  afterward), so the driver stamps it here once the choice is made; the
  saved recipe then rebuilds the model under the same mode.

  Arguments:
    model        = the live model (compiled or eager); its underlying
                   eager module must carry a ``model_recipe``.
    compile_mode = the selected mode, one of COMPILE_MODES.

  Returns:
    the same model object, with the recipe's compile_mode replaced.

  Raises:
    ValueError when the model carries no constructor recipe (it did not
    come from a registered constructor), or when compile_mode is not one
    of COMPILE_MODES.
  """
  if compile_mode not in COMPILE_MODES:
    raise ValueError(
      "set_runtime_compile_mode: compile_mode must be one of "
      + repr(COMPILE_MODES) + "; got " + repr(compile_mode)
      + ". Stamping an unknown mode here would only surface later at "
      "torch.compile or at save; refuse it at the stamping site.")
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
  """Require a live model's own recipe to equal the caller's claim.

  save_emulator receives both the model and a resolved_model mapping from
  the driver. The model's constructor stamped its true recipe on itself,
  so a driver that constructed one architecture while describing another
  is caught here, before the wrong instructions are persisted.

  Arguments:
    model    = the live model about to be saved.
    expected = the caller's resolved_model recipe mapping.
    where    = the label used in the refusal (default "live model").

  Returns:
    the model's own recipe mapping, once it equals the claim.

  Raises:
    ValueError showing both mappings when they differ, or when the model
    carries no recipe at all.
  """
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
