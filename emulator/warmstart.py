"""Fine-tune warm start: continue a trained emulator on a new training set.

Training normally starts from random weights. This module lets a run start
from a saved emulator instead (say one trained on LCDM) and keep training on
a new set of cosmologies (say w0waCDM), with a lower initial learning rate.
The usual case is a nested one: the new parameter space is the old one plus a
few extra parameters (w0, wa).

Loading the weights is one line; the real work is the whitening bases. The
input encoding is a full rotation into the parameter covariance eigenbasis
(ParamGeometry, geometries_parameter.py), and a new cosmology's covariance
remixes every parameter, so a naively rebuilt geometry would point the loaded
input weights at scrambled coordinates and the run would spend its first
epochs relearning the basis. This module builds the new input geometry by
block extension instead: the shared parameters keep the source's exact
center, rotation and scale, and the extra parameters are whitened on their
own marginal covariance and appended as the last encoded coordinates. Paired
with a state-dict transfer that copies the source weights verbatim and pads
the extra input columns with exact zeros, this makes epoch 0 compute the
source emulator's own function, bit for bit, independent of the extra
parameters' values. Fine-tuning then moves away from a proven starting point
instead of a scrambled one.

The functions here are called by experiment.py at three points: config
validation (validate_finetune_config, resolve_source_root, load_source),
geometry building (extend_input_geometry, pin_output_geometry), and training
(recipe_to_model_opts, build_warm_start, which runs the state-dict transfer
and the pre-train parity check). training.py's make_model / run_emulator load
the transferred weights; nothing else in the training loop changes.

PS: whiten = rotate into the covariance eigenbasis and scale each direction
to unit variance (decorrelated, equally hard to fit); encode = center then
whiten a raw parameter vector; state_dict = torch's name -> tensor mapping of
a model's learnable parameters and buffers; recipe = the model_recipe record
a schema-v2 .h5 stores, the class + constructor kwargs rebuild_emulator reads
to reconstruct the network; the extras = the parameters present in the new
covmat but not the source's (the new physics being fine-tuned in).
"""

import os

import numpy as np
import torch

from .geometries_parameter import ParamGeometry
from .results import rebuild_emulator
from .training import make_model
from .activations import make_activation
from .designs.blocks import make_norm


# the only keys the train_args.finetune block accepts. "from" is the source
# artifact path root (required); "compile_mode" is the one machine knob allowed
# beside it (torch.compile mode override for this machine). The lower learning
# rate is not here: it rides the existing lr: block (a smaller lr_base).
_FINETUNE_KEYS = ("from", "compile_mode")

# the torch.compile modes a finetune.compile_mode override may name (None =
# plain eager, no compile). Mirrors make_model's compile_mode parameter.
_COMPILE_MODES = ("reduce-overhead", "default", None)

# the parity check scores the epoch-0 warm-started model against the source on
# this many staged training rows (D-FT7). float32 whitened-dv units.
_PARITY_ROWS = 256

# the largest whitened-dv deviation the parity check tolerates between the
# warm-started model and the source on the shared parameters. Not bit-equality:
# the two models run different-width input matmuls (n_s vs n_n wide), so the
# floating-point reduction order differs.
_PARITY_TOL = 1.0e-5


class FinetuneSource:
  """The source emulator a fine-tune run continues from, loaded once.

  load_source builds this once per experiment and every later step reads it,
  so the source .h5 / .emul is opened a single time. It carries the source's
  reconstructed network (the parity reference), both source geometries, the
  model rebuild recipe, and the few root-attr / resolved-config values the
  validation and provenance steps need.

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
                 run's, D-FT4).
    dataset    = the source run's cosmolike_dataset (checked equal too).
  """

  def __init__(self,
               root,
               model,
               model_cls,
               pgeom,
               geom,
               recipe,
               compile_mode,
               data_dir,
               dataset):
    self.root         = root
    self.model        = model
    self.model_cls    = model_cls
    self.pgeom        = pgeom
    self.geom         = geom
    self.recipe       = recipe
    self.compile_mode = compile_mode
    self.data_dir     = data_dir
    self.dataset      = dataset


def validate_finetune_config(cfg, train_args, rescale, activation_flag):
  """Check the finetune YAML surface, raising a loud error on any violation.

  Runs the D-FT1 / D-FT2 config-time checks before anything is loaded: the
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
  # NPCE composition across bases is out of scope (D-FT10).
  if cfg.get("pce") is not None:
    raise KeyError(
      "a finetune run does not compose with a pce: block (warm-starting a "
      "PCE refiner across bases needs its own design; D-FT10); remove it")
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


def load_source(root, device):
  """Load and validate the source emulator once (wraps rebuild_emulator).

  Reconstructs the source network eager (never torch.compile'd, so its
  state-dict keys are unprefixed) plus both source geometries, then reads the
  recipe, the source rescale root attr, and the source's cosmolike data
  block from the .h5. Enforces the D-FT2 source-artifact constraints: a plain
  ParamGeometry input geometry (not the log or factored variants), no
  intrinsic-alignment factoring, no PCE base, and a rescale of "none". Schema
  v2 is enforced by rebuild_emulator itself.

  Arguments:
    root   = resolved absolute source path root (from resolve_source_root).
    device = device to rebuild the source network and geometries on.

  Returns:
    a FinetuneSource carrying the parity-reference model, both geometries,
    the recipe, and the values later steps validate against.

  Raises:
    ValueError / KeyError naming any constraint the source artifact breaks.
  """
  # h5py lives only here (the training machines ship it); import lazily so the
  # config paths stay importable without it.
  import h5py
  import yaml

  # rebuild_emulator (results.py): reconstruct the source network + both
  # geometries from the .h5 + .emul, using only the file (schema v2 enforced
  # there). compile_model=False keeps the module eager, so its state_dict keys
  # carry no "_orig_mod." compile prefix.
  model, pgeom, geom, info = rebuild_emulator(
    path_root=root, device=device, compile_model=False)

  # the input geometry must be a plain ParamGeometry: block extension is
  # defined on its center / rotation / scale, not on the log or factored
  # variants (those are V2 candidates, D-FT10).
  if type(pgeom).__name__ != "ParamGeometry":
    raise ValueError(
      "finetune source input geometry is " + type(pgeom).__name__
      + "; V1 warm start supports only a plain ParamGeometry source (not "
      "LogParamGeometry / AmplitudeFactorGeometry)")
  # no factored intrinsic-alignment source, no NPCE base.
  if info.get("ia") is not None:
    raise ValueError(
      "finetune source is a factored intrinsic-alignment emulator (ia="
      + repr(info["ia"]) + "); V1 warm start supports only a plain source")
  if info.get("pce_base") is not None:
    raise ValueError(
      "finetune source carries an NPCE base; V1 warm start does not compose "
      "with PCE (D-FT10)")

  # read the recipe, the source rescale, and the source data block from the
  # .h5 (rebuild consumes the recipe but does not return it).
  with h5py.File(root + ".h5", "r") as f:
    recipe   = yaml.safe_load(f["model_recipe"][()])
    src_resc = f.attrs.get("rescale")
    resolved = yaml.safe_load(f["config_resolved_yaml"][()])
  if src_resc is None:
    # a missing attr is not the same failure as a wrong value: the training
    # drivers stamp rescale in the run-identity attrs, but an artifact saved
    # by another path (e.g. a check script) may predate the stamp. No
    # fallback to "none" here -- an artifact that does not record its rescale
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
    dataset=data_block.get("cosmolike_dataset"))


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


def extend_input_geometry(source, covmat_path, train_mean, device):
  """Build the new run's input geometry by block extension (D-FT3).

  The shared parameters keep the source's exact center, rotation, and scale;
  the extra parameters are whitened on their own marginal block of the new
  covariance and appended as the last encoded coordinates. The result is a
  plain ParamGeometry (no new class, no new save/load code) whose encoding of
  the shared parameters is bit-identical to the source's, and whose extra
  coordinates depend only on the extra parameters.

    source geometry (n_s params)          new covmat header (n_n params)
       │  center_s / evecs_s=V / sqrt_ev_s=s     │  names_n, cov (n_n x n_n)
       ▼                                         ▼
       │  place each source name's row at its position r(i) in names_n,
       │  copy V into columns 0..n_s-1 there (source rotation, verbatim);
       │  whiten the extras on Sigma_xx = cov's extras-only block:
       │      lam_x, W = eigh(Sigma_xx), placed in columns n_s..n_n-1
       ▼
    extended geometry (n_n params): encode = [source coords (n_s) ; extras (n_x)]
       (legend: n_s / n_n / n_x = source / new / extra parameter counts;
        r(i) = the row of source name i in the new covmat header order;
        V, s = the source eigenvectors and sqrt-eigenvalues; Sigma_xx,
        lam_x, W = the extras' marginal covariance and its eigen-pairs;
        the extras' encoded coordinates are the only ones that see the extra
        parameters, and the state-dict transfer meets them with zero weights.)

  Arguments:
    source      = the loaded FinetuneSource (reads source.pgeom).
    covmat_path = the new run's parameter covmat file; its first line is a
                  "#"-prefixed list of column names (names_n).
    train_mean  = (n_n,) staged-train mean of the new run's parameters
                  (train_set["C_mean"]); centers the extra columns.
    device      = device for the built tensors.

  Returns:
    (pgeom, extra_names): the extended ParamGeometry, and the extra parameter
    names in names_n order ([] when n_x = 0).

  Raises:
    ValueError if names_n is not a superset of the source names (both lists
    printed).
  """
  # source tensors, pulled to numpy (float32 values, kept exact through the
  # float64 build and the ParamGeometry float32 recast below).
  names_s  = list(source.pgeom.names)
  V        = source.pgeom.evecs.detach().cpu().numpy()     # (n_s, n_s)
  s        = source.pgeom.sqrt_ev.detach().cpu().numpy()   # (n_s,)
  center_s = source.pgeom.center.detach().cpu().numpy()    # (n_s,)

  # the new run's covmat: header names and the full matrix.
  with open(covmat_path) as f:
    names_n = f.readline().lstrip("#").split()
  cov = np.loadtxt(covmat_path)                            # (n_n, n_n)
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
      "finetune requires the new covmat names to be a superset of the "
      "source's. Missing source parameter(s): " + str(missing)
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
    # the source geometry (one code path covers both fine-tune flavors).
    sqrt_ev_e = s

  pgeom = ParamGeometry(device=device,
                        names=names_n,
                        center=center_e,
                        evecs=E,
                        sqrt_ev=sqrt_ev_e)
  return pgeom, extra_names


def pin_output_geometry(source, run_data, run_probe, new_dv_width):
  """Reuse the source output geometry for the new run, after checks (D-FT4).

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
    ValueError naming any mismatch (data dir, dataset, probe, width).
  """
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
  """Transfer the source weights into the new model's shape (D-FT5).

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


def build_warm_start(source,
                     new_pgeom,
                     pinned_geom,
                     model_opts,
                     train_set,
                     extra_names,
                     device):
  """Transfer the weights and run the pre-train parity check (D-FT5 + D-FT7).

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
    (init_state, verdict): the transferred state dict to hand run_emulator as
    init_state, and the one-line parity verdict string.

  Raises:
    ValueError if the parity tolerance is exceeded, or if the extra
    parameters leak into the epoch-0 output.
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

  init_state, _padded = transfer_state_dict(
    source_state=source.model.state_dict(),
    template_state=model.state_dict(),
    n_extra=n_extra)
  model.load_state_dict(init_state, strict=True)
  model.eval()

  # staged rows for the parity check (D-FT7 wants at least _PARITY_ROWS; use
  # what is available if a tiny synthetic set has fewer).
  idx  = train_set["idx"]
  take = min(_PARITY_ROWS, int(len(idx)))
  rows = idx[:take]
  theta = torch.from_numpy(
    np.asarray(train_set["C"][rows])).float().to(device)   # (R, n_n)

  shared_cols, extra_cols = _shared_columns(
    source_pgeom=source.pgeom, new_pgeom=new_pgeom, device=device)

  with torch.no_grad():
    # new model on the full (n_n) input; source model on the shared params in
    # source order (theta[:, shared_cols] is (R, n_s), source-ordered).
    out_new = model(new_pgeom.encode(theta))               # model(x): x=input
    out_src = source.model(source.pgeom.encode(theta[:, shared_cols]))
    max_dv = (out_new - out_src).abs().max().item()

    # extras-independence: perturbing only the extra columns must not move the
    # epoch-0 output at all (their encoded coordinates meet zero weights).
    if n_extra > 0:
      theta_pert = theta.clone()
      theta_pert[:, extra_cols] = theta_pert[:, extra_cols] + 1.0
      out_pert = model(new_pgeom.encode(theta_pert))
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
  return init_state, verdict
