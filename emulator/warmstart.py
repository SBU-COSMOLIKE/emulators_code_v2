"""Fine-tune warm start: continue a trained emulator on a new training set.

Training normally starts from random weights. This module lets a run start
from a saved emulator instead (say one trained on LCDM) and keep training on
a new set of cosmologies (say w0waCDM), with a lower initial learning rate.
The usual case is a nested one: the new parameter space is the old one plus a
few extra parameters (w0, wa).

Loading the weights is one line; the real work is the whitening bases. The
input encoding is a full rotation into the parameter covariance eigenbasis
(ParamGeometry, geometries.parameter.py), and a new cosmology's covariance
remixes every parameter, so a naively rebuilt geometry would point the loaded
input weights at scrambled coordinates and the run would spend its first
epochs relearning the basis. This module builds the new input geometry by
block extension instead: the shared parameters keep the source's exact
center, rotation and scale, and the extra parameters are whitened on their
own marginal covariance and appended as the last encoded coordinates. Paired
with a state-dict transfer that pads the extra input columns with exact zeros,
this gives two separate checks. First, the widened model must reproduce the
source prediction within ``_PARITY_TOL``. Changing matrix width can change
floating-point reduction order, so this is a numerical comparison rather
than a bitwise comparison. Second, changing only the zero-connected extra
inputs on that same widened model must leave its output bit-identical under
``torch.equal``. Fine-tuning therefore begins from a checked source function
rather than a scrambled coordinate system.

The functions here are called by experiment.py at three points: config
validation (validate_finetune_config, resolve_source_root, load_source),
geometry building (extend_input_geometry, pin_output_geometry), and training
(recipe_to_model_opts, build_warm_start, which runs the state-dict transfer
and the pre-train parity check). training.py's make_model / run_emulator load
the transferred weights; nothing else in the training loop changes.

PS: whiten = rotate into the covariance eigenbasis and scale each coordinate
to unit variance under the covariance used to define the transform. This
gives comparable numerical scales, while learning difficulty can still differ.
encode = center then whiten a raw parameter vector; state_dict = PyTorch's
name -> tensor mapping of registered parameters, including frozen parameters,
and persistent registered buffers; recipe = the model_recipe record
a schema-v2 .h5 stores, the class + constructor kwargs rebuild_emulator reads
to reconstruct the network; the extras = the parameters present in the new
covmat but not the source's (the new physics being fine-tuned in).
"""

import os

import numpy as np
import torch

from .geometries.parameter import ParamGeometry, AmplitudeFactorGeometry
from .results import rebuild_emulator, read_artifact_schema
from .training import make_model, _report_nonfinite
from .activations import make_activation
from .designs.blocks import make_norm


# Keys recognized by the train_args.finetune parser. "from" is the required
# source-artifact path root. "compile_mode" is the optional torch.compile mode
# for the current machine. "anchor" has a reserved name but is refused by the
# validator below; omit it to run unanchored fine-tuning. The lower learning
# rate belongs in the existing lr block through a smaller lr_base.
_FINETUNE_KEYS = ("from", "compile_mode", "anchor")

# the torch.compile modes a finetune.compile_mode override may name (None =
# plain eager, no compile). Mirrors make_model's compile_mode parameter.
_COMPILE_MODES = ("reduce-overhead", "default", None)

# the parity check scores the epoch-0 warm-started model against the source on
# this many staged training rows. float32 whitened-dv units.
_PARITY_ROWS = 256

# the largest whitened-dv deviation the parity check tolerates between the
# warm-started model and the source on the shared parameters. Not bit-equality:
# the two models run different-width input matmuls (n_s vs n_n wide), so the
# floating-point reduction order differs.
_PARITY_TOL = 1.0e-5


class FinetuneSource:
  """The saved emulator used to begin fine-tuning or transfer learning.

  ``load_source`` constructs one ``FinetuneSource`` object per experiment.
  Later geometry and training steps reuse that object. A successful
  construction performs two separate reads of ``<root>.h5``. The first read
  occurs inside ``rebuild_emulator`` and reconstructs the network and both
  geometries. That function also loads the weights from ``<root>.emul``. The
  second HDF5 read occurs inside ``load_source`` and retrieves run metadata
  that ``rebuild_emulator`` does not return, including the saved model recipe
  and resolved data configuration. Thus one in-memory source object comes
  from two HDF5 file opens and one weight-file load.

  Attributes:
    root       = resolved absolute source path root (<root>.h5 + <root>.emul).
    model      = the source network, rebuilt eager (never torch.compile'd),
                 in eval() with the source weights loaded; the parity check
                 scores against it.
    model_cls  = the source network's class (resolved from the recipe), which
                 the new run inherits (the architecture is never restated).
    pgeom      = the source input ParamGeometry (names / center / evecs /
                 sqrt_ev), block-extended for the new run.
    geom       = the source output DataVectorGeometry, pinned (reused) for the
                 new run.
    recipe     = the model_recipe dict from the source .h5 (class qualname,
                 dims, every constructor kwarg, the act / norm / head
                 factories by name).
    compile_mode = the torch.compile mode the source recipe stored (the
                 finetune default when the YAML does not override it).
    data_dir   = the source run's cosmolike_data_dir (checked equal to the new
                 run's).
    dataset    = the source run's cosmolike_dataset (checked equal too).
    ia         = the source intrinsic-alignment design. The value is ``nla``
                 for the nonlinear-alignment design, ``tatt`` for the tidal
                 alignment and tidal torquing design, or ``None`` for a
                 source without a factored intrinsic-alignment model.
  """

  def __init__(
               self,
               root,
               model,
               model_cls,
               pgeom,
               geom,
               recipe,
               compile_mode,
               data_dir,
               dataset,
               ia=None):
    self.root         = root
    self.model        = model
    self.model_cls    = model_cls
    self.pgeom        = pgeom
    self.geom         = geom
    self.recipe       = recipe
    self.compile_mode = compile_mode
    self.data_dir     = data_dir
    self.dataset      = dataset
    # the source's factored intrinsic-alignment design (nla / tatt) or None
    # for a plain source. None on the fine-tune path (which allows only a
    # plain source); set on the transfer path, where a factored base is the
    # headline use.
    self.ia           = ia


def finetune_provenance_attrs(
    *,
    source,
    extra_names):
  """Build the two saved attributes that identify a fine-tune source.

  Both training drivers call this function before saving an emulator. A
  plain run has no source and receives no fine-tune attributes. A fine-tune
  run records the resolved source path and the ordered names of parameters
  added by the new run.

  Arguments:
    source      = the ``FinetuneSource`` used by the run, or ``None`` for a
                  plain run.
    extra_names = the ordered extra parameter names. This may be empty when
                  the new run keeps the source parameter space unchanged.

  Returns:
    an empty dict for a plain run, or the exact two root attributes required
    for a fine-tuned artifact.

  Raises:
    ValueError if a fine-tune source has no usable path or an extra parameter
    name is empty or is not text.
    TypeError if ``extra_names`` is not a sequence of names.
  """
  if source is None:
    return {}

  root = getattr(source, "root", None)
  if not isinstance(root, str) or not root:
    raise ValueError(
      "fine-tune provenance needs a nonempty resolved source path")
  if extra_names is None or isinstance(extra_names, (str, bytes)):
    raise TypeError(
      "fine-tune provenance extra_names must be a sequence of names")

  checked_names = []
  for name in extra_names:
    if not isinstance(name, str) or not name:
      raise ValueError(
        "fine-tune provenance contains an empty or non-text parameter name")
    checked_names.append(name)

  return {
    "finetuned_from": root,
    "finetune_extra_names": " ".join(checked_names),
  }


def validate_finetune_config(cfg, train_args, rescale, activation_flag):
  """Check the finetune YAML surface, raising a loud error on any violation.

  Runs the config-time checks before anything is loaded: the
  finetune block's own key whitelist, and the keys a warm start forbids
  because the architecture, loss form, and schedule are all inherited or out
  of scope. Every failure names what to remove.

  Arguments:
    cfg             = the full parsed config mapping (its top-level pce: block
                      is checked here).
    train_args      = the resolved train_args mapping (holds the finetune
                      block and the forbidden siblings).
    rescale         = the driver's --rescale value (must be "none").
    activation_flag = the driver's --activation value (must be None: absent).

  Raises:
    KeyError / ValueError naming the offending key and the fix.
  """
  ft = train_args["finetune"]
  if not isinstance(ft, dict):
    raise ValueError(
      "train_args.finetune must be a block with a 'from' key, got "
      + type(ft).__name__)
  # The anchor name is reserved but not currently accepted. Reject it with
  # the user action instead of letting the generic unknown-key error hide it.
  if "anchor" in ft:
    raise NotImplementedError(
      "train_args.finetune.anchor is not available. Remove the anchor key "
      "to run unanchored fine-tuning")
  # the finetune block's own whitelist.
  unknown = set(ft) - set(_FINETUNE_KEYS)
  if unknown:
    raise KeyError(
      "unknown train_args.finetune key(s): " + str(sorted(unknown))
      + "; allowed: " + str(list(_FINETUNE_KEYS)))
  if not ft.get("from"):
    raise KeyError(
      "train_args.finetune needs a 'from' key: the source artifact path "
      "root (<root>.h5 + <root>.emul, as written by save_emulator)")
  if "compile_mode" in ft and ft["compile_mode"] not in _COMPILE_MODES:
    raise ValueError(
      "train_args.finetune.compile_mode must be one of "
      + str(list(_COMPILE_MODES)) + " (None = plain eager), got "
      + repr(ft["compile_mode"]))
  # anchor: the optional L2-SP strength lambda (>= 0; 0.0 states free
  # fine-tuning deliberately). A bool is not a number here (True/False would
  # be a config typo).
  if "anchor" in ft:
    a = ft["anchor"]
    if isinstance(a, bool) or not isinstance(a, (int, float)) or a < 0.0:
      raise ValueError(
        "train_args.finetune.anchor must be a number >= 0 (the L2-SP anchor "
        "strength; 0.0 = free fine-tuning), got " + repr(a))

  # the architecture is inherited from the source artifact, so a model: block
  # would silently contradict it.
  if "model" in train_args:
    raise KeyError(
      "a finetune run inherits its architecture from the source artifact; "
      "delete the train_args.model block (the source .h5 model_recipe is "
      "the only architecture source)")
  # --activation would try to re-pin an activation the source already fixed.
  if activation_flag is not None:
    raise ValueError(
      "a finetune run inherits its activation from the source artifact; "
      "drop the --activation flag")
  # NPCE composition across bases is out of scope.
  if cfg.get("pce") is not None:
    raise KeyError(
      "a finetune run does not compose with a pce: block (warm-starting a "
      "PCE refiner across bases needs its own design); remove it")
  # transfer x finetune: two different reuse tools, one at a time (the
  # same rule validate_transfer states from its side; checked here too so
  # neither branch order can silently ignore the other block).
  if cfg.get("transfer") is not None:
    raise ValueError(
      "train_args.finetune and a transfer: block are exclusive (a warm "
      "start adapts every weight; a transfer freezes the base under a "
      "parallel correction); use one at a time")
  # the loss form is inherited too; a rescale would restate the target.
  if rescale != "none":
    raise ValueError(
      "a finetune run requires --rescale none (the source's target "
      "construction is inherited); got --rescale " + repr(rescale))
  # a warm start is past the trunk-warming era: V1 is single-phase only.
  for key in ("trunk", "head"):
    if key in train_args:
      raise KeyError(
        "a finetune run is single-phase (V1): remove the train_args." + key
        + " block (the two-phase schedule does not apply to a warm start)")
  if int(train_args.get("trunk_epochs", 0) or 0) > 0:
    raise ValueError(
      "a finetune run is single-phase (V1): set trunk_epochs 0 or remove it")
  if "freeze_trunk" in train_args:
    raise KeyError(
      "a finetune run is single-phase (V1): remove train_args.freeze_trunk")


def resolve_source_root(finetune_cfg):
  """Resolve finetune.from to an absolute path root, both files present.

  Expands a leading ~ and any environment variables, then anchors a relative
  path under $ROOTDIR (the cobaya adapter's convention). A relative path with
  $ROOTDIR unset is a loud error, as is either source file missing.

  Arguments:
    finetune_cfg = the train_args.finetune block (reads its "from" key).

  Returns:
    the resolved absolute path root; <root>.h5 and <root>.emul both exist.

  Raises:
    RuntimeError if the path is relative and $ROOTDIR is unset;
    FileNotFoundError naming either missing file.
  """
  raw  = str(finetune_cfg["from"])
  path = os.path.expanduser(os.path.expandvars(raw))
  if not os.path.isabs(path):
    rootdir = os.environ.get("ROOTDIR")
    if not rootdir:
      raise RuntimeError(
        "finetune.from is a relative path (" + raw + ") and $ROOTDIR is "
        "unset; set $ROOTDIR or give an absolute source path")
    path = os.path.join(rootdir, path)
  # both artifact files must exist; name the exact resolved path on a miss.
  for ext in (".h5", ".emul"):
    if not os.path.isfile(path + ext):
      raise FileNotFoundError(
        "finetune source file not found: " + path + ext
        + " (finetune.from resolves to the path root " + path + ")")
  return path


def load_source(
    root,
    device,
    allow_factored=False):
  """Build one validated source object from a saved emulator pair.

  The first HDF5 read is owned by ``rebuild_emulator``. It reconstructs the
  eager source network and both source geometries, then loads the weights from
  ``<root>.emul``. Eager means that ``torch.compile`` has not wrapped the
  network, so state-dict keys carry no compile prefix. A second HDF5 read in
  this function runs the shared schema reader again and retrieves the recipe,
  the saved rescale value and the resolved cosmolike data block. These metadata
  values are needed by the warm-start validator and are not part of
  ``rebuild_emulator``'s return value.

  The function then enforces the source-artifact constraints: a plain
  ParamGeometry input geometry, no PCE base, no embedded transfer base and a
  rescale value of ``none``. The file's schema version and its scientific
  record (the cosmology it was trained under, and the parameter region it was
  sampled over) are enforced by ``read_artifact_schema`` (results.py), the one
  shared reader that ``rebuild_emulator`` also calls, so this path cannot
  accept a file that path would refuse. Fine-tuning narrows the region an
  emulator serves, which makes it the path that most needs that record to be
  present rather than assumed.

  Arguments:
    root           = resolved absolute source path root (from
                     resolve_source_root).
    device         = device to rebuild the source network and geometries on.
    allow_factored = the transfer path sets this to accept a factored
                     intrinsic-alignment base. Its ``ia`` value is ``nla`` or
                     ``tatt`` and its input geometry is an
                     AmplitudeFactorGeometry. The fine-tune path leaves this
                     value False, which accepts only a plain source.
                     LogParamGeometry is outside this version of both paths.

  Returns:
    a FinetuneSource carrying the parity-reference model, both geometries,
    the recipe, and the values later steps validate against (its .ia is the
    factored design or None).

  Raises:
    ValueError / KeyError naming any constraint the source artifact breaks.
  """
  # h5py lives only here (the training machines ship it); import lazily so the
  # config paths stay importable without it.
  import h5py
  import yaml

  # First HDF5 open: rebuild the source network and both geometries from the
  # saved recipe and geometry records. rebuild_emulator also loads the weight
  # file once. compile_model=False keeps the module eager, so its state_dict
  # keys carry no "_orig_mod." compile prefix.
  model, pgeom, geom, info = rebuild_emulator(
    path_root=root,
    device=device,
    compile_model=False)

  # the input geometry: a plain ParamGeometry always works; the transfer path
  # also accepts an AmplitudeFactorGeometry (factored base). LogParamGeometry
  # stays out of V1 either way (its whitening is not block-extended yet).
  geom_name = type(pgeom).__name__
  allowed   = ("ParamGeometry", "AmplitudeFactorGeometry") if allow_factored \
      else ("ParamGeometry",)
  if geom_name not in allowed:
    raise ValueError(
      "source input geometry is " + geom_name + "; this path supports only "
      + " / ".join(allowed) + " (V1)")
  # no factored intrinsic-alignment source unless the caller allows it.
  if info.get("ia") is not None and not allow_factored:
    raise ValueError(
      "source is a factored intrinsic-alignment emulator (ia="
      + repr(info["ia"]) + "); this path supports only a plain source")
  if info["composition_mode"] == "npce":
    raise ValueError(
      "source carries an NPCE base; V1 warm start / transfer does not compose "
      "with PCE")

  # Second HDF5 open: read metadata that rebuild_emulator consumes internally
  # or does not return. The warm-start path needs these values for validation
  # and for the new run's resolved record. The already validated authoritative
  # composition mode, not transfer_base presence, marks a transfer output,
  # which this version cannot chain.
  with h5py.File(root + ".h5", "r") as f:
    # the schema version and both blocks of the scientific record, through the
    # one shared reader (results.read_artifact_schema), the same reader
    # rebuild_emulator used on the first open. Every read of a saved emulator
    # goes through it, so a file cannot be refused on one path and defaulted
    # away on the other. It is called here for its refusals; the blocks it
    # returns are already enforced and nothing below needs them.
    read_artifact_schema(f=f, where=root + ".h5")
    recipe   = yaml.safe_load(f["model_recipe"][()])
    src_resc = f.attrs.get("rescale")
    resolved = yaml.safe_load(f["config_resolved_yaml"][()])
    if info["composition_mode"] == "transfer":
      raise ValueError(
        "source " + root + ".h5 is itself a transfer artifact (it embeds a "
        "transfer_base group); chaining a transfer over a transfer is out of "
        "scope (no chaining)")
  if src_resc is None:
    # a missing attr is not the same failure as a wrong value: the training
    # drivers stamp rescale in the run-identity attrs, but an artifact saved
    # by another path (e.g. a check script) may predate the stamp. No
    # fallback to "none" here. An artifact that does not record its rescale
    # is ambiguous (the never-trust-defaults rule).
    raise ValueError(
      "finetune source records no 'rescale' root attr in " + root + ".h5; "
      "the artifact was saved by a path that skipped the run-identity "
      "attrs. Re-save the source with attrs including rescale='none' (the "
      "training drivers stamp it; for the board's gates_emul_evaluate "
      "artifact, --force-rerun save-rebuild-drift re-persists it)")
  if src_resc != "none":
    raise ValueError(
      "finetune source was trained with rescale=" + repr(src_resc)
      + "; V1 warm start requires a rescale-none source (the loss form is "
      "inherited)")
  data_block = resolved.get("data", {})

  # resolve the source network's class from the recipe (the class the new run
  # inherits), by the same module.qualname the recipe stores.
  import importlib
  cls_path        = recipe["cls"]
  mod_name, _, qn = cls_path.rpartition(".")
  model_cls       = getattr(importlib.import_module(mod_name), qn)

  return FinetuneSource(
    root=root,
    model=model,
    model_cls=model_cls,
    pgeom=pgeom,
    geom=geom,
    recipe=recipe,
    compile_mode=recipe.get("compile_mode"),
    data_dir=data_block.get("cosmolike_data_dir"),
    dataset=data_block.get("cosmolike_dataset"),
    ia=info.get("ia"))


def recipe_to_model_opts(recipe, geom=None, compile_mode="__inherit__"):
  """Turn a stored model recipe into a live run_emulator model_opts dict.

  The finetune run inherits the source architecture rather than restating it
  in YAML, so its make_model spec is rebuilt from the recipe here. This
  mirrors rebuild_emulator's recipe-to-constructor step (results.py), but
  produces the {cls, compile_mode, **kwargs} dict run_emulator consumes,
  never a built model. input_dim / output_dim are absent: make_model injects
  them (the new run's widths, not the source's).

  Arguments:
    recipe       = the model_recipe dict (from FinetuneSource.recipe).
    geom         = the pinned output geometry, injected as the geom kwarg when
                   the recipe records needs_geom (the conv / TRF heads); None
                   otherwise.
    compile_mode = the resolved torch.compile mode for the new run. The
                   sentinel "__inherit__" keeps the recipe's own mode; any
                   other value (including None for eager) overrides it.

  Returns:
    a model_opts dict: "cls" the resolved class, "compile_mode" the resolved
    mode, plus every constructor kwarg with its act / norm / head factories
    rebuilt into live callables.
  """
  import importlib
  cls_path        = recipe["cls"]
  mod_name, _, qn = cls_path.rpartition(".")
  cls             = getattr(importlib.import_module(mod_name), qn)

  kwargs = dict(recipe["kwargs"])
  # rebuild the ResBlock factory dict: the serialized {type, n_gates} act and
  # the make_norm name become the live callables the constructor expects.
  if "block_opts" in kwargs:
    bo  = kwargs["block_opts"]
    act = bo["act"]
    kwargs["block_opts"] = {
      "act":  make_activation(act["type"], n_gates=act["n_gates"]),
      "norm": make_norm(bo["norm"]),
    }
  # rebuild the per-head activation factory (head models only; None otherwise).
  if kwargs.get("head_act") is not None:
    ha = kwargs["head_act"]
    kwargs["head_act"] = make_activation(ha["type"], n_gates=ha["n_gates"])
  # geometry-consuming heads take the (pinned) geometry; plain trunks take none.
  if recipe.get("needs_geom"):
    kwargs["geom"] = geom

  cm = recipe.get("compile_mode") if compile_mode == "__inherit__" \
      else compile_mode
  model_opts = {"cls": cls, "compile_mode": cm}
  for k, v in kwargs.items():
    model_opts[k] = v
  return model_opts


def _extend_param_geometry(src_pgeom, names_n, cov, train_mean, device):
  """The block extension of one plain ParamGeometry (in memory).

  Given a source ParamGeometry and the new (superset) parameter names, the
  new covariance matrix, and the staged-train mean, build a plain
  ParamGeometry that copies the source parameters' center, rotation, and
  scale into the first encoded block. The extra parameters are whitened on
  their marginal covariance and appended as the final encoded coordinates.
  Floating-point equality of a model prediction is checked later by
  ``build_warm_start``. This helper is also reused for a factored base's
  inner ``pg_keep``.

    source geometry (n_s params)          new names (n_n params)
       │  center_s / evecs_s=V / sqrt_ev_s=s     │  names_n, cov (n_n x n_n)
       ▼                                         ▼
       │  place each source name's row at its position r(i) in names_n,
       │  copy V into columns 0..n_s-1 there (source rotation, verbatim);
       │  whiten the extras on Sigma_xx = cov's extras-only block:
       │      lam_x, W = eigh(Sigma_xx), placed in columns n_s..n_n-1
       ▼
    extended geometry: encode = [source coords (n_s) ; extras (n_x)]
       (legend: n_s / n_n / n_x = source / new / extra parameter counts;
        r(i) = the row of source name i in names_n; V, s = the source
        eigenvectors and sqrt-eigenvalues; Sigma_xx, lam_x, W = the extras'
        marginal covariance and its eigen-pairs.)

  Arguments:
    src_pgeom  = the source ParamGeometry (names / center / evecs / sqrt_ev).
    names_n    = the new parameter names (a superset of the source's), the
                 order the encoded shared/extra blocks follow.
    cov        = (n_n, n_n) new covariance matrix, names_n order.
    train_mean = (n_n,) staged-train mean, names_n order (centers the extras).
    device     = device for the built tensors.

  Returns:
    (pgeom, extra_names): the extended ParamGeometry and the extra names in
    names_n order ([] when there are none).

  Raises:
    ValueError if names_n is not a superset of the source names.
  """
  # source tensors, pulled to numpy (float32 values, kept exact through the
  # float64 build and the ParamGeometry float32 recast below).
  names_s  = list(src_pgeom.names)
  V        = src_pgeom.evecs.detach().cpu().numpy()       # (n_s, n_s)
  s        = src_pgeom.sqrt_ev.detach().cpu().numpy()     # (n_s,)
  center_s = src_pgeom.center.detach().cpu().numpy()      # (n_s,)
  names_n  = list(names_n)
  cov      = np.asarray(cov, dtype="float64")
  n_s = len(names_s)
  n_n = len(names_n)

  # every source parameter must survive in the new space (a superset); the
  # extras are the new-only parameters, kept in names_n order.
  new_set = set(names_n)
  missing = []
  for nm in names_s:
    if nm not in new_set:
      missing.append(nm)
  if missing:
    raise ValueError(
      "the new covmat names must be a superset of the source's. Missing "
      "source parameter(s): " + str(missing)
      + "\n  source names: " + str(names_s)
      + "\n  new names:    " + str(names_n))
  source_set = set(names_s)
  extra_names = []
  for nm in names_n:
    if nm not in source_set:
      extra_names.append(nm)
  n_x = len(extra_names)

  cmean = np.asarray(train_mean, dtype="float64")

  # build the extended center and the extended rotation E, block by block.
  center_e = np.zeros(n_n, dtype="float64")
  E        = np.zeros((n_n, n_n), dtype="float64")
  # shared block: each source name i sits at row r = names_n.index(name) in
  # the new order; its center and its rotation column carry over verbatim.
  for i in range(n_s):
    r = names_n.index(names_s[i])
    center_e[r] = center_s[i]
    for j in range(n_s):
      E[r, j] = V[i, j]
  # extra block: the extras' own rows, centered on the staged-train mean and
  # whitened on their marginal covariance, filling the last n_x columns.
  x_rows = []
  for nm in extra_names:
    x_rows.append(names_n.index(nm))
  for k in range(n_x):
    center_e[x_rows[k]] = cmean[x_rows[k]]
  if n_x > 0:
    # Sigma_xx = the extras-only block of the new covmat; eigh gives the
    # extras' own whitening (W columns orthonormal, lam_x > 0).
    sigma_xx = cov[np.ix_(x_rows, x_rows)]                 # (n_x, n_x)
    lam_x, W = np.linalg.eigh(sigma_xx)
    for k in range(n_x):
      for kk in range(n_x):
        E[x_rows[k], n_s + kk] = W[k, kk]
    sqrt_ev_e = np.concatenate([s, np.sqrt(lam_x)])
  else:
    # no extras: E is the source rotation with rows keyed by name, and the
    # scales are the source's. Same-order names make this byte-identical to
    # the source geometry (one code path covers both flavors).
    sqrt_ev_e = s

  pgeom = ParamGeometry(device=device,
                        names=names_n,
                        center=center_e,
                        evecs=E,
                        sqrt_ev=sqrt_ev_e)
  return pgeom, extra_names


def extend_input_geometry(source, covmat_path, train_mean, device):
  """Build the new run's input geometry by block extension.

  Dispatches on the source geometry: a plain ParamGeometry is block-extended
  directly (the fine-tune case); an AmplitudeFactorGeometry (a factored NLA /
  TATT base, the transfer case) has its inner pg_keep block extended by the
  same math, while the raw amplitude columns stay last, so the encoded layout
  is [shared-whitened (n_s') ; extras-whitened (n_x) ; raw amps (n_amp)] and
  the base's own encoding is the column slice cat(enc[:, :n_s'], enc[:, -n_amp:]).

  Arguments:
    source      = the loaded source (reads source.pgeom).
    covmat_path = the new run's parameter covmat file; its first line is a
                  "#"-prefixed list of column names (names_n).
    train_mean  = (n_n,) staged-train mean of the new run's parameters
                  (train_set["C_mean"]); centers the extra columns.
    device      = device for the built tensors.

  Returns:
    (pgeom, extra_names): the extended geometry (same class as the source's
    input geometry), and the extra (non-amplitude) parameter names in names_n
    order ([] when there are none).

  Raises:
    ValueError if names_n is not a superset of the source names, or (factored)
    an amplitude column is missing from the new covmat.
  """
  with open(covmat_path) as f:
    names_n = f.readline().lstrip("#").split()
  cov   = np.loadtxt(covmat_path)                          # (n_n, n_n)
  cmean = np.asarray(train_mean, dtype="float64")

  pgeom_src = source.pgeom
  if isinstance(pgeom_src, AmplitudeFactorGeometry):
    # the base's amplitude names, in coeff_fn order (raw, appended last).
    amp_ids   = pgeom_src.amp_idx.detach().cpu().tolist()
    amp_names = []
    for i in amp_ids:
      amp_names.append(pgeom_src.names[int(i)])
    new_set = set(names_n)
    for a in amp_names:
      if a not in new_set:
        raise ValueError(
          "the new covmat is missing the base's amplitude column " + repr(a)
          + "; the factored amplitudes must survive in the new space")
    # the kept (non-amplitude) new names, in names_n order; extend pg_keep on
    # their sub-covmat / sub-mean.
    amp_set      = set(amp_names)
    kept_names_n = []
    for nm in names_n:
      if nm not in amp_set:
        kept_names_n.append(nm)
    kept_rows = []
    for nm in kept_names_n:
      kept_rows.append(names_n.index(nm))
    kept_cov  = cov[np.ix_(kept_rows, kept_rows)]
    kept_mean = cmean[kept_rows]
    ext_pg_keep, extra_names = _extend_param_geometry(
      src_pgeom=pgeom_src.pg_keep,
      names_n=kept_names_n,
      cov=kept_cov,
      train_mean=kept_mean,
      device=device)
    new_amp_idx = []
    for a in amp_names:
      new_amp_idx.append(names_n.index(a))
    pgeom = AmplitudeFactorGeometry(device=device,
                                    pg_keep=ext_pg_keep,
                                    amp_idx=new_amp_idx,
                                    n_param=len(names_n),
                                    names=names_n)
    return pgeom, extra_names

  # plain base / fine-tune: extend the whole geometry directly.
  return _extend_param_geometry(src_pgeom=pgeom_src,
                                names_n=names_n,
                                cov=cov,
                                train_mean=cmean,
                                device=device)


def pin_output_geometry(source, run_data, run_probe, new_dv_width):
  """Reuse the source output geometry for the new run, after checks.

  The output whitening basis and inverse covariance come from the dataset
  covariance and mask, which are cosmology-independent, so the source dv
  geometry is reused wholesale (and persisted again in the new artifact,
  self-consistent under the persist-resolved-values rule). Before pinning it,
  the fine-tune prerequisites are checked: the same dataset, mask, and scale
  cut (same cosmolike data dir + dataset), the same probe, and a new dv width
  that matches the pinned geometry.

  Arguments:
    source       = the loaded FinetuneSource (reads source.geom / data_dir /
                   dataset).
    run_data     = the new run's data block (its cosmolike_data_dir /
                   cosmolike_dataset).
    run_probe    = the new run's probe string.
    new_dv_width = the new dv dump's width (columns of the raw data vector).

  Returns:
    the source DataVectorGeometry, unchanged, to wrap in the run's chi2.

  Raises:
    ValueError naming any mismatch (data dir, dataset, probe, width), or a
    source of the wrong family (a scalar / CMB artifact pins on its own
    family's path, never here).
  """
  # wrong-kind guard: this pin is the cosmolike (data-vector) one; a scalar
  # or CMB source has no probe/dataset to check and pins on its own path
  # (experiment.build_geometry's scalar / cmb branches).
  geom_kind = type(source.geom).__name__
  if geom_kind != "DataVectorGeometry":
    raise ValueError(
      "finetune source rebuilds a " + geom_kind + " output geometry; the "
      "cosmolike warm-start pin serves data-vector sources only (a scalar "
      "or CMB source fine-tunes through its own family's run config)")
  run_dir = run_data.get("cosmolike_data_dir")
  run_set = run_data.get("cosmolike_dataset")
  if run_dir != source.data_dir or run_set != source.dataset:
    raise ValueError(
      "finetune requires the same dataset + mask + scale cut as the source. "
      "source cosmolike (" + repr(source.data_dir) + ", "
      + repr(source.dataset) + ") != new (" + repr(run_dir) + ", "
      + repr(run_set) + ")")
  if source.geom.probe != run_probe:
    raise ValueError(
      "finetune probe mismatch: source geometry probe "
      + repr(source.geom.probe) + " != new run probe " + repr(run_probe))
  if int(new_dv_width) != int(source.geom.total_size):
    raise ValueError(
      "finetune dv width mismatch: new dump width " + str(int(new_dv_width))
      + " != pinned geometry total_size " + str(int(source.geom.total_size)))
  return source.geom


def transfer_state_dict(source_state, template_state, n_extra):
  """Transfer the source weights into the new model's shape.

  Generic and shape-driven, with no per-design enumeration: the new model's
  own state_dict is the shape template. For every key, a matching shape copies
  the source tensor verbatim; a tensor that differs only by +n_extra columns
  in dim 1 (the input Linear of every design, and the FiLM generators of the
  conv / TRF heads, all sized by input_dim) takes the source columns first
  and exact-zero columns for the extras. Any other shape difference, a missing
  key, or a leftover source key is a loud error, so the same-architecture
  guarantee is checked, not assumed.

  The zero columns plus the block-extended encoding are the exactness proof:
  the extras' encoded coordinates are the only ones that see the extra
  parameters, and they meet zero weights, so epoch 0 computes the source
  function. With n_extra = 0 every shape matches and the transfer is a
  verbatim strict load.

  Per-tensor shape flow, the two cases one key can fall into:

    matched key (bias, norm buffer, any hidden-layer weight):
       source T_s, shape S            template T_t, shape S (equal)
          │                              │
          ▼                              ▼
          └──────── T_new = T_s.clone() ────────► shape S (verbatim copy)

    grown key (an input-consumer weight, dim 1 = input width):
       source W_s               template W_t
        shape (m, n_s)           shape (m, n_s + n_x)
       ┌────────────┐           ┌────────────┬──────┐
       │  n_s cols  │  m rows   │  n_s cols  │ n_x  │  m rows
       └────────────┘           └────────────┴──────┘
          │                        │              │
          │  W_new = zeros(m, n_s + n_x)          │
          ▼                        ▼              ▼
       copy W_s into           source columns   extra columns
       cols 0..n_s-1  ───────► kept verbatim    stay exact zero
       (all m rows copied; dim 0 never grows)

  (legend: T_s / T_t / T_new = the source tensor, the template tensor, and
   the built tensor for one state_dict key; S = a shape the source and
   template share exactly (any rank); W_s / W_t / W_new = the same for an
   input-consumer weight whose dim 1 is the encoded input width; m = that
   weight's dim-0 size (output rows, unchanged by the transfer); n_s / n_x =
   the source input width and the number of appended extra columns
   (n_x = n_extra); n_s + n_x = the new input width; cols = columns along
   dim 1; a rank-3 FiLM generator grows the same way, its trailing axes
   copied position for position.)

  Arguments:
    source_state   = the source model's state_dict (name -> tensor).
    template_state = the new model's state_dict (the shape template).
    n_extra        = the number of extra input columns (n_x); the exact dim-1
                     growth a padded tensor must show.

  Returns:
    (new_state, padded_keys): a state_dict matching template_state's shapes,
    and the sorted list of keys that received zero-padded extra columns.

  Raises:
    KeyError on a missing or leftover key; ValueError on any other shape
    mismatch (the key and both shapes named).
  """
  new_state  = {}
  padded_keys = []
  for key, tmpl in template_state.items():
    if key not in source_state:
      raise KeyError(
        "finetune transfer: new model key " + repr(key) + " is absent from "
        "the source state dict (same-architecture guarantee broken)")
    src = source_state[key]
    if tuple(tmpl.shape) == tuple(src.shape):
      new_state[key] = src.clone()
      continue
    # the only accepted difference: same rank, same dims except dim 1, which
    # grows by exactly n_extra (the input-consumer tensors).
    grows_dim1 = (n_extra > 0
                  and tmpl.dim() >= 2
                  and src.dim() == tmpl.dim()
                  and tmpl.shape[1] == src.shape[1] + n_extra)
    if grows_dim1:
      other_ok = (tmpl.shape[0] == src.shape[0])
      for d in range(2, tmpl.dim()):
        if tmpl.shape[d] != src.shape[d]:
          other_ok = False
      if other_ok:
        padded = torch.zeros_like(tmpl)
        # source columns first (dim 1), the extras' columns stay zero.
        padded[:, :src.shape[1]] = src
        new_state[key] = padded
        padded_keys.append(key)
        continue
    raise ValueError(
      "finetune transfer: key " + repr(key) + " has an unexpected shape "
      "difference. source " + str(tuple(src.shape)) + " vs new "
      + str(tuple(tmpl.shape)) + " (only a +" + str(n_extra)
      + " growth in dim 1 is a padded input tensor)")
  # no source key may be left behind (again, the architecture is identical).
  for key in source_state:
    if key not in template_state:
      raise KeyError(
        "finetune transfer: source key " + repr(key) + " has no counterpart "
        "in the new model (same-architecture guarantee broken)")
  padded_keys.sort()
  return new_state, padded_keys


def _shared_columns(source_pgeom, new_pgeom, device):
  """Index tensors mapping the new parameter order to source / extra columns.

  Arguments:
    source_pgeom = the source input geometry (names_s order).
    new_pgeom    = the extended input geometry (names_n order).
    device       = device for the returned index tensors.

  Returns:
    (shared_cols, extra_cols): long tensors of column indices into the new
    (names_n) parameter vector. shared_cols[i] is where source name i lives
    in names_n (so theta[:, shared_cols] is in source order); extra_cols are
    the new-only columns.
  """
  names_n = list(new_pgeom.names)
  names_s = list(source_pgeom.names)
  source_set = set(names_s)
  shared = []
  for nm in names_s:
    shared.append(names_n.index(nm))
  extra = []
  for j in range(len(names_n)):
    if names_n[j] not in source_set:
      extra.append(j)
  shared_cols = torch.tensor(shared, dtype=torch.long, device=device)
  extra_cols  = torch.tensor(extra, dtype=torch.long, device=device)
  return shared_cols, extra_cols


def _require_parity_finite(
    side,
    quantity,
    values,
    rows):
  """Abort the pre-train parity check if any per-row value is non-finite.

  The finite contract on the warm-start parity gates (finetune and
  transfer): the "[ok] parity" verdict must be IMPOSSIBLE unless every
  compared tensor is finite. Otherwise a diverged epoch-0 model makes the
  compared difference non-finite, and "max|dv| = nan" compares False to
  the tolerance (and torch.equal reads a NaN as an ordinary mismatch) —
  so a broken warm start prints as if parity held, or fails with a
  misleading "extras leaked" / "not the frozen base" reason. This raises
  the one shared finite-contract message instead (never a sentinel),
  naming the offending staged rows.

  Arguments:
    side     = "finetune parity" or "transfer parity" (the pipeline side,
               threaded into the shared _report_nonfinite message).
    quantity = what was checked (a noun phrase, e.g. "epoch-0 new-model
               outputs"); reads as "N of M <quantity> are non-finite".
    values   = the tensor to test; its first axis is the staged rows, any
               remaining axes are the per-row payload (flattened here).
    rows     = the staged row indices aligned to values' first axis.

  Raises:
    ValueError (the shared finite-contract message) on any non-finite
    entry; returns None otherwise.
  """
  # collapse every per-row payload axis so a row counts as bad if any of
  # its entries is non-finite (works for a (R, W) output and a factored
  # (R, T, W) surface alike).
  per_row = torch.isfinite(values).reshape(values.shape[0], -1).all(dim=1)
  if bool(per_row.all()):
    return
  bad_rows = np.asarray(rows)[(~per_row).cpu().numpy()]
  _report_nonfinite(
    side=side,
    quantity=quantity,
    n_bad=int(bad_rows.size),
    n_total=int(values.shape[0]),
    positions=bad_rows[:8].tolist())


def build_warm_start(source,
                     new_pgeom,
                     pinned_geom,
                     model_opts,
                     train_set,
                     extra_names,
                     device):
  """Transfer the weights and run the pre-train parity check.

  Builds a fresh eager model at the new input width from the inherited recipe,
  transfers the source weights into it (verbatim plus zero-padded extra input
  columns), and checks, on staged training rows, that epoch 0 reproduces the
  source emulator's function: the shared-parameter output matches the source
  within a whitened-dv tolerance, and changing only the extra parameters
  leaves the output bit-identical. Prints exactly one verdict line (the
  essential-only terminal rule). The transferred state dict is what the
  training model then loads strict, so the checked function is the trained one.

  Arguments:
    source      = the loaded FinetuneSource (the parity reference + weights).
    new_pgeom   = the block-extended input geometry (n_n params).
    pinned_geom = the pinned output geometry (sizes the model output).
    model_opts  = the live model_opts (recipe_to_model_opts), used to build
                  the template / parity model at the new input width.
    train_set   = the staged training source (reads "C" and "idx").
    extra_names = the extra parameter names (its length is n_x).
    device      = device the models, geometry, and rows live on.

  Returns:
    (init_state, verdict, padded_keys): the transferred state dict to hand
    run_emulator as init_state, the one-line parity verdict string, and the
    list of padded input-consumer keys (the finetune.anchor mask excludes
    their extra columns).

  Raises:
    ValueError if the parity tolerance is exceeded, if the extra
    parameters leak into the epoch-0 output, or (the finite contract) if
    the encoded inputs or either model output are non-finite — the [ok]
    verdict is impossible unless every compared tensor is finite.
  """
  n_extra = len(extra_names)
  in_dim  = len(new_pgeom.names)
  out_dim = int(pinned_geom.dest_idx.numel())

  # a fresh model at the new input width, forced eager (compile_mode None) so
  # its state-dict keys carry no compile prefix and the transfer / parity run
  # on the plain module. make_model injects the dims.
  template_opts = dict(model_opts)
  template_opts["compile_mode"] = None
  model = make_model(model_opts=template_opts,
                     input_dim=in_dim,
                     output_dim=out_dim,
                     device=device)

  init_state, padded_keys = transfer_state_dict(
    source_state=source.model.state_dict(),
    template_state=model.state_dict(),
    n_extra=n_extra)
  model.load_state_dict(init_state, strict=True)
  model.eval()

  # staged rows for the parity check (it wants at least _PARITY_ROWS; use
  # what is available if a tiny synthetic set has fewer).
  idx  = train_set["idx"]
  take = min(_PARITY_ROWS, int(len(idx)))
  rows = idx[:take]
  C_rows = np.asarray(train_set["C"][rows])                # (R, n_n)
  theta  = torch.from_numpy(C_rows).float().to(device)

  shared_cols, extra_cols = _shared_columns(
    source_pgeom=source.pgeom, new_pgeom=new_pgeom, device=device)

  with torch.no_grad():
    # new model on the full (n_n) input; source model on the shared params in
    # source order (theta[:, shared_cols] is (R, n_s), source-ordered).
    enc_new = new_pgeom.encode(theta)                      # model(x): x=input
    enc_src = source.pgeom.encode(theta[:, shared_cols])
    out_new = model(enc_new)
    out_src = source.model(enc_src)
    # finite contract (before any comparison below): guard the encoded
    # inputs and BOTH model outputs, naming the offending staged rows and
    # which arm diverged. Their difference and the scalar max are then
    # finite by construction, so the tolerance comparison at the foot of
    # this function and the extras-independence torch.equal just below can
    # never read a NaN as an ordinary mismatch — which would print parity
    # as HELD, or raise the wrong reason, on a broken warm start.
    _require_parity_finite("finetune parity", "encoded new-run inputs",
                           enc_new, rows)
    _require_parity_finite("finetune parity", "encoded source inputs",
                           enc_src, rows)
    _require_parity_finite("finetune parity", "epoch-0 new-model outputs",
                           out_new, rows)
    _require_parity_finite("finetune parity", "epoch-0 source-model outputs",
                           out_src, rows)
    max_dv = (out_new - out_src).abs().max().item()

    # Extras-independence: perturbing only the extra columns must not move the
    # epoch-0 output. Keep the perturbed encoding and output as separate named
    # tensors. Each tensor is checked before it becomes input to the next step
    # or to torch.equal, so an invalid transform is distinguished from an
    # invalid model output.
    if n_extra > 0:
      theta_pert = theta.clone()
      theta_pert[:, extra_cols] = theta_pert[:, extra_cols] + 1.0
      enc_pert = new_pgeom.encode(theta_pert)
      _require_parity_finite(
        side="finetune parity",
        quantity="perturbed encoded new-run inputs",
        values=enc_pert,
        rows=rows)
      out_pert = model(enc_pert)
      _require_parity_finite(
        side="finetune parity",
        quantity="perturbed epoch-0 new-model outputs",
        values=out_pert,
        rows=rows)
      if not torch.equal(out_new, out_pert):
        raise ValueError(
          "finetune parity: the extra parameters leaked into the epoch-0 "
          "output (perturbing only the extras changed the model output); the "
          "padded input columns are not exact zeros")

  if max_dv > _PARITY_TOL:
    raise ValueError(
      "finetune parity failed: max|dv| = " + format(max_dv, ".3e")
      + " exceeds " + format(_PARITY_TOL, ".0e") + " on " + str(take)
      + " rows (epoch 0 does not reproduce the source function)")
  verdict = ("[ok] finetune parity: max|dv| = " + format(max_dv, ".3e")
             + " on " + str(take) + " rows")
  return init_state, verdict, padded_keys


def anchor_masks(init_state, padded_keys, n_extra, device):
  """Per-parameter anchor masks for the finetune warm start.

  Every anchored parameter is pulled toward the transferred init_state, EXCEPT
  the padded extra input columns: they are exact zeros by design and the
  designated carriers of the new-physics dependence, so anchoring them to zero
  would fight the warm start. This returns one mask per padded key (ones on the
  source columns, zeros on the last n_extra columns); a non-padded parameter
  gets no entry (build_anchor then pulls all of it).

  Arguments:
    init_state  = the transferred state dict (name -> tensor).
    padded_keys = the input-consumer keys that gained n_extra zero columns.
    n_extra     = the number of appended extra columns (n_x).
    device      = device for the mask tensors.

  Returns:
    a dict name -> mask (same shape as the parameter, 1 on the kept columns,
    0 on the extra columns); empty when n_extra == 0.
  """
  masks = {}
  if n_extra <= 0:
    return masks
  for key in padded_keys:
    t = init_state[key]
    mask = torch.ones_like(t, device=device)
    # the extras are the last n_extra columns along dim 1 (transfer_state_dict
    # appends them there); zero the pull on exactly those.
    mask[:, t.shape[1] - n_extra:] = 0.0
    masks[key] = mask
  return masks


def _zero_final_linear(model):
  """Zero the last nn.Linear of a correction net (weight and bias).

  The transfer correction nets (ResMLP, TemplateMLP) end in an output Linear
  followed by an identity-initialized Affine (gain 1, bias 0). Zeroing the
  Linear makes the whole net output exactly zero at init, while the Affine
  passes gradients straight through, so the net still trains away from the
  zero start (its gain sees a nonzero input after the first step). Epoch 0 is
  then exactly the frozen base. Every other tensor is untouched.

  Arguments:
    model = the correction network (eager module).

  Returns:
    the zeroed nn.Linear (for the gate to inspect).

  Raises:
    ValueError if the model has no nn.Linear.
  """
  last = None
  for m in model.modules():
    if isinstance(m, torch.nn.Linear):
      last = m
  if last is None:
    raise ValueError(
      "the transfer correction model has no nn.Linear to zero-init")
  with torch.no_grad():
    last.weight.zero_()
    if last.bias is not None:
      last.bias.zero_()
  return last


def build_transfer_start(chi2fn,
                         model_opts,
                         new_pgeom,
                         train_set,
                         extra_names,
                         device):
  """Build the zero-init correction net and run the bitwise parity gate.

  Builds a fresh eager correction network at the run's input width, applies the
  zero-init surgery (the final output Linear is zeroed), and checks, on
  staged rows, that epoch 0 reproduces the frozen base EXACTLY: the composed
  prediction (base plus a zero correction) is torch.equal the frozen base's own
  decode, and perturbing only the extra parameters leaves it bit-identical
  (their coordinates never reach the base slice). One verdict line, essential
  only. The returned state dict is loaded strict by the training model, so the
  checked function is the trained one's epoch-0 state.

  Arguments:
    chi2fn      = the built TransferChi2 (its decode composes base + correction,
                  its base_decode gives the frozen base in the loss's space).
    model_opts  = the correction model spec (make_model consumes it).
    new_pgeom   = the (block-extended) input geometry; encodes the staged rows.
    train_set   = the staged training source (reads "C" and "idx").
    extra_names = the new (non-amplitude) parameter names (its length is n_x).
    device      = device the models, geometry, and rows live on.

  Returns:
    (init_state, verdict): the zero-init correction state dict to hand
    run_emulator as init_state, and the one-line parity verdict.

  Raises:
    ValueError if epoch 0 is not the frozen base bitwise, if the extras
    move it, or (the finite contract) if the encoded inputs, the composed
    prediction, or the frozen base decode are non-finite — the [ok]
    verdict is impossible unless every compared tensor is finite.
  """
  in_dim  = getattr(new_pgeom, "encoded_dim", len(new_pgeom.names))
  out_dim = int(chi2fn.dest_idx.numel())
  # a fresh correction net, forced eager (compile_mode None), then zero-init.
  template_opts = dict(model_opts)
  template_opts["compile_mode"] = None
  corr = make_model(model_opts=template_opts,
                    input_dim=in_dim,
                    output_dim=out_dim,
                    device=device)
  _zero_final_linear(corr)
  init_state = corr.state_dict()
  corr.eval()

  idx  = train_set["idx"]
  take = min(_PARITY_ROWS, int(len(idx)))
  rows = idx[:take]
  C_rows = np.asarray(train_set["C"][rows])
  theta  = torch.from_numpy(C_rows).float().to(device)

  names_n = list(new_pgeom.names)
  extra_cols = []
  for nm in extra_names:
    extra_cols.append(names_n.index(nm))

  with torch.no_grad():
    enc      = new_pgeom.encode(theta)
    composed = chi2fn.decode(corr(enc), enc)       # base + zero correction
    base     = chi2fn.base_decode(enc)             # the frozen base, same space
    # finite contract (before the bitwise / extras comparisons below): a
    # non-finite composed or base makes torch.equal read False and the
    # error below print "max|dv| = nan" as if a real parity failure
    # occurred. Guard the encoded inputs and both surfaces here, naming
    # the offending staged rows, so the [ok] verdict is impossible unless
    # epoch 0 really is the frozen base.
    _require_parity_finite("transfer parity", "encoded run inputs",
                           enc, rows)
    _require_parity_finite("transfer parity", "epoch-0 composed prediction",
                           composed, rows)
    _require_parity_finite("transfer parity", "frozen base decode",
                           base, rows)
    if not torch.equal(composed, base):
      max_dv = (composed - base).abs().max().item()
      raise ValueError(
        "transfer parity failed: epoch 0 is not the frozen base bitwise "
        "(max|dv| = " + format(max_dv, ".3e") + " on " + str(take) + " rows); "
        "the zero-init surgery or the composition space handling is wrong")
    if len(extra_cols) > 0:
      idx_cols = torch.tensor(extra_cols, dtype=torch.long, device=device)
      theta_pert = theta.clone()
      theta_pert[:, idx_cols] = theta_pert[:, idx_cols] + 1.0
      enc_pert = new_pgeom.encode(theta_pert)
      _require_parity_finite(
        side="transfer parity",
        quantity="perturbed encoded run inputs",
        values=enc_pert,
        rows=rows)
      correction_pert = corr(enc_pert)
      composed_pert = chi2fn.decode(correction_pert, enc_pert)
      _require_parity_finite(
        side="transfer parity",
        quantity="perturbed epoch-0 composed prediction",
        values=composed_pert,
        rows=rows)
      if not torch.equal(composed, composed_pert):
        raise ValueError(
          "transfer parity: the extra parameters moved the epoch-0 prediction "
          "(they must never reach the frozen base's input slice)")

  verdict = ("[ok] transfer parity: epoch 0 == frozen base (bitwise) on "
             + str(take) + " rows")
  return init_state, verdict
