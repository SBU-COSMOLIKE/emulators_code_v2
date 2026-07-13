"""Run-output I/O: learning-curve tables and trained-emulator files.

save_learning_curves writes a whitespace-delimited table (row per N_train,
column per curve, "#"-comment header carrying the config) that np.loadtxt
reads back, the format the sweep and bake-off drivers save, so several
runs can be overlaid later. save_emulator persists a trained run as two
files: <root>.emul, the model weights (a torch state_dict, cpu tensors),
and <root>.h5, everything inference or a paper trail needs (both
whitening geometries, the training histories, and the full config).

PS: state_dict = torch's name -> tensor mapping of a model's learnable
parameters and buffers; whitening = the center/rotate/scale transform the
geometries apply to parameters (input) and data vectors (output).
"""

import os
import subprocess
import time

import numpy as np
import torch
import yaml


def save_learning_curves(path, sizes, curves, meta=None):
  """
  Write learning curve(s) as a whitespace-delimited text table.

  A single config writes a one-entry `curves`; a bake-off writes all its
  curves to one file. Header lines are "#" comments np.loadtxt skips.
  Layout:

    # learning curve: f(delta-chi2 > threshold) vs N_train
    # model=ResMLP  rescale=none  threshold=0.2  pool=82000
    # columns: N_train, H, power, multigate, gated_power
    2000     0.401234  0.410512  0.395001  0.402310
    4203     ...

  Arguments:
    path   = output text-file path.
    sizes  = the N_train values, one per row (cast to int).
    curves = mapping label -> per-size fractions aligned with `sizes`
             (curves[label][i] is the value at sizes[i]). Labels become the
             data columns (dict order), documented on the "# columns:" line.
    meta   = optional mapping written as a "# key=val  key=val" line
             (model / rescale / threshold / pool); None to omit.
  """
  sizes  = list(sizes)
  labels = list(curves)
  lines  = ["# learning curve: f(delta-chi2 > threshold) vs N_train"]
  if meta:
    # one "# key=val  key=val ..." line (insertion order kept).
    pairs = []
    for k, v in meta.items():
      pairs.append(f"{k}={v}")
    lines.append("# " + "  ".join(pairs))
  # column header is a comment too (skipped on load); labels are
  # comma-separated to keep a label with spaces unambiguous.
  header = ["N_train"]
  for l in labels:
    header.append(str(l))
  lines.append("# columns: " + ", ".join(header))
  for i, n in enumerate(sizes):
    row = [f"{int(n):d}"]
    for l in labels:
      row.append(f"{curves[l][i]:.6f}")
    lines.append("  ".join(row))
  with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")


def save_sweep_table(path, param, values, fracs, meta=None):
  """
  Write a one-hyperparameter sweep as a whitespace-delimited table.

  The generic twin of save_learning_curves for an arbitrary swept
  knob. Numeric values become the first data column; categorical
  values (strings, or booleans, a film on/off sweep) become an
  integer index column with the label map on a "# values:" comment
  line, so the body stays np.loadtxt-loadable either way. Layouts:

      # sweep: f(delta-chi2 > threshold) vs lr.lr_base
      # model=rescnn  threshold=0.2  n_train=250000
      # columns: lr.lr_base, frac
      0.001  0.401234
      ...

      # sweep: f(delta-chi2 > threshold) vs model.activation
      # values: 0=H, 1=power, 2=multigate
      # columns: index, frac
      0  0.401234
      ...

  Arguments:
    path   = output text-file path.
    param  = the swept hyperparameter's dotted YAML path (names the
             x column).
    values = the swept values, one per row (numbers, strings, or
             booleans).
    fracs  = per-value fractions aligned with `values`.
    meta   = optional mapping written as a "# key=val" line; None
             to omit.
  """
  # booleans pass isinstance(v, int) in Python; check them first so
  # a True/False sweep is labeled, not silently cast to 1/0.
  numeric = True
  for v in values:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
      numeric = False
  lines = [f"# sweep: f(delta-chi2 > threshold) vs {param}"]
  if meta:
    pairs = []
    for k, v in meta.items():
      pairs.append(f"{k}={v}")
    lines.append("# " + "  ".join(pairs))
  if numeric:
    lines.append(f"# columns: {param}, frac")
    for v, f in zip(values, fracs):
      lines.append(f"{float(v):.8g}  {f:.6f}")
  else:
    labels = []
    for i, v in enumerate(values):
      labels.append(f"{i}={v}")
    lines.append("# values: " + ", ".join(labels))
    lines.append("# columns: index, frac")
    for i, f in enumerate(fracs):
      lines.append(f"{i:d}  {f:.6f}")
  with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")


def save_emulator(path_root,
                  model,
                  param_geometry,
                  geometry,
                  config,
                  histories,
                  train_args=None,
                  attrs=None,
                  pce=None,
                  pce_form=None,
                  resolved_train=None,
                  resolved_model=None,
                  transfer_base=None):
  """
  Persist a trained emulator as <path_root>.emul + <path_root>.h5.

  The .emul holds only the model weights: torch.save of the
  state_dict with every tensor moved to cpu, so it loads on any
  machine (a cuda-saved state needs the saving GPU visible). A
  torch.compile'd model wraps the real one and prefixes every
  state_dict key with "_orig_mod."; the prefix is stripped so the
  saved keys always match the plain architecture.

  The .h5 holds everything else, grouped:
    param_geometry/  the input-whitening state, keys exactly
                     ParamGeometry.state() (names, center, evecs,
                     sqrt_ev; a factored run nests pg_keep + amp_idx),
                     plus a "cls" attr = the geometry's class, so
                     rebuild dispatches to the right from_state with
                     no covmat reread.
    dv_geometry/     the output-geometry state, keys exactly
                     DataVectorGeometry.state() (total_size,
                     dest_idx, evecs, sqrt_ev, Cinv, center, dtype),
                     plus a "cls" attr, so from_state rebuilds the
                     exact geometry class with no cosmolike.
    pce/             (NPCE runs only) the frozen PCEEmulator base's
                     buffers (PCEEmulator.state(): lo / hi /
                     multi_index / C / Vk / Ybar) plus a "form" attr, so
                     PCEEmulator.from_state rebuilds the base with no
                     refit and no cosmolike (the refiner .emul unchanged).
    transfer_base/   (transfer runs only) the frozen base emulator embedded
                     whole: its own model_recipe (yaml), a state/ subgroup of
                     the base weights (name -> tensor), param_geometry/ and
                     dv_geometry/ states with cls markers, and form / space
                     attrs. The main model above is then the correction net and
                     the main geometries are the run's; rebuild composes the
                     two. Self-contained: no reference to the base file.
    history/         per-epoch training curves: train_losses,
                     val_medians, val_means, val_fracs (one row per
                     epoch, one column per threshold), thresholds.
    config_yaml      the driver's resolved config (data + train_args
                     blocks), as YAML text.
    config_resolved_yaml  schema v2: the CONSUMED config, defaults
                     materialized (resolved_train + the data block), so a
                     saved run reconstructs even if code defaults drift.
    model_recipe     schema v2: the serializable model rebuild recipe
                     (class qualname, dims, every constructor kwarg, the
                     act / norm / head factories by name), read by
                     rebuild_emulator (h5-only, a missing key loud, never a
                     code default). Root attrs schema_version = 2 +
                     git_commit mark a v2 file (rebuild refuses one without).
    train_args_yaml  the collapsed train_args actually used (search
                     ranges resolved to their defaults), as YAML.
  plus one root attribute per entry of `attrs` (run identity:
  model name, activation, rescale, N_train, best epoch, ...), a
  "created" timestamp, and the torch version.

  Arguments:
    path_root      = output path without extension; writes
                     <path_root>.emul and <path_root>.h5.
    model          = the trained network (best-epoch weights already
                     restored by the training loop).
    param_geometry = the input ParamGeometry (its .state() is saved).
    geometry       = the output DataVectorGeometry (its .state() is
                     saved). Pass chi2fn.geom, not the chi2fn.
    config         = the resolved config mapping (data + train_args),
                     stored verbatim as YAML text.
    histories      = mapping with the per-epoch lists the training
                     loop returned: "train_losses", "val_medians",
                     "val_means", "val_fracs" (list of per-threshold
                     tensors), "thresholds".
    train_args     = the collapsed train_args the run actually used
                     (search ranges resolved), or None to omit.
    attrs          = optional mapping of scalar run metadata, each
                     written as one h5 root attribute.

  Returns:
    (emul_path, h5_path), the two files written.
  """
  # h5py lives only here: the training machines (cocoa) ship it, the
  # plotting/train paths never need it.
  import h5py

  # --- <root>.emul: the weights, cpu, unprefixed ---
  sd = {}
  for k, v in model.state_dict().items():
    # a torch.compile wrapper (OptimizedModule) stores the real
    # model as ._orig_mod, so its keys arrive prefixed; strip it.
    sd[k.removeprefix("_orig_mod.")] = v.detach().cpu()
  emul_path = path_root + ".emul"
  torch.save(sd, emul_path)

  # --- <root>.h5: geometries + histories + config + identity ---
  h5_path = path_root + ".h5"
  str_dt  = h5py.string_dtype(encoding="utf-8")

  def write_state(group, state):
    # Write one geometry state() dict recursively, so every geometry
    # saves without per-class code: tensors -> datasets, name lists
    # -> string datasets, nested dicts (a composed geometry, e.g.
    # AmplitudeFactorGeometry's pg_keep) -> subgroups, scalars and
    # dtypes -> attributes. Keys stay exactly state()'s, so the
    # matching from_state rebuilds from a read-back dict.
    for k, v in state.items():
      if isinstance(v, dict):
        write_state(group.create_group(k), v)
      elif torch.is_tensor(v):
        group.create_dataset(k, data=v.cpu().numpy())
      elif isinstance(v, list):
        group.create_dataset(k,
                             data=np.asarray(v, dtype=object),
                             dtype=str_dt)
      elif isinstance(v, (int, float)):
        group.attrs[k] = v
      else:
        group.attrs[k] = str(v)   # torch.dtype and friends

  with h5py.File(h5_path, "w") as f:
    # input whitening. param_geometry.state() (geometries.parameter.py):
    # the input-whitening tensors keyed exactly as from_state expects. The
    # group also records its own CLASS (materialized from the object's type
    # at write time) so rebuild dispatches to the right from_state -- a
    # factored run's AmplitudeFactorGeometry, a log run's LogParamGeometry,
    # a plain run's ParamGeometry -- rather than hardcoding the base class
    # (the never-trust-defaults rule applied to class identity).
    pg_group = f.create_group("param_geometry")
    write_state(pg_group, param_geometry.state())
    pg_group.attrs["cls"] = (type(param_geometry).__module__ + "."
                             + type(param_geometry).__qualname__)

    # output geometry. geometry.state() (geometries.output.py): the
    # output-geometry tensors keyed exactly as from_state expects; the same
    # class marker (a Diagonal / BlockDiagonal geometry shares the base
    # state keys but a different whitening, so the marker prevents a silent
    # wrong-transform decode).
    dv_group = f.create_group("dv_geometry")
    write_state(dv_group, geometry.state())
    dv_group.attrs["cls"] = (type(geometry).__module__ + "."
                             + type(geometry).__qualname__)

    # NPCE base (present only when the run used a pce: block): the frozen
    # PCEEmulator buffers keyed exactly as from_state expects, plus the
    # combine form, so inference rebuilds base + refiner with no refit and
    # no cosmolike (the refiner .emul is unchanged).
    if pce is not None:
      g = f.create_group("pce")
      write_state(g, pce.state())
      g.attrs["form"] = pce_form

    # transfer_base (present only when the run used a transfer: block): the
    # frozen base emulator embedded whole, so the transfer artifact is
    # self-contained and survives the base file moving (never a path
    # reference). Its own model_recipe + state_dict + both geometry states
    # (with cls markers) let rebuild reconstruct the base exactly, and the
    # form / space attrs tell the predictor how to compose. The main model /
    # geometries / recipe above are the correction net and the run geometries.
    if transfer_base is not None:
      tb = f.create_group("transfer_base")
      tb.create_dataset("model_recipe",
                        data=yaml.safe_dump(transfer_base["recipe"],
                                            sort_keys=False),
                        dtype=str_dt)
      # the base weights as a name -> tensor subgroup (state dict keys carry
      # dots, not h5 "/" separators, so each is one dataset).
      state_grp = tb.create_group("state")
      for k, v in transfer_base["state"].items():
        state_grp.create_dataset(k, data=v.detach().cpu().numpy())
      # both base geometry states + their class markers (the same
      # write_state + cls pattern the main geometries use, so a factored
      # AmplitudeFactorGeometry base rebuilds as itself).
      base_pg = transfer_base["param_geometry"]
      pg_grp  = tb.create_group("param_geometry")
      write_state(pg_grp, base_pg.state())
      pg_grp.attrs["cls"] = (type(base_pg).__module__ + "."
                             + type(base_pg).__qualname__)
      base_dv = transfer_base["dv_geometry"]
      dv_grp  = tb.create_group("dv_geometry")
      write_state(dv_grp, base_dv.state())
      dv_grp.attrs["cls"] = (type(base_dv).__module__ + "."
                             + type(base_dv).__qualname__)
      tb.attrs["form"]  = transfer_base["form"]
      tb.attrs["space"] = transfer_base["space"]
      # a refined run (transfer.refine): the DRIFTED base weights, kept
      # separate from the pretrained state above (which stays the anchor
      # reference + provenance). The predictor
      # loads these; the transfer_refined root attr marks their presence,
      # two-way consistent with the group (rebuild refuses either half alone).
      drifted = transfer_base.get("drifted_state")
      if drifted is not None:
        drift_grp = tb.create_group("drifted_state")
        for k, v in drifted.items():
          drift_grp.create_dataset(k, data=v.detach().cpu().numpy())
        f.attrs["transfer_refined"] = True

    # per-epoch histories; fracs stack to (nepochs, n_thresholds).
    hg = f.create_group("history")
    hg.create_dataset("train_losses",
                      data=np.asarray(histories["train_losses"]))
    hg.create_dataset("val_medians",
                      data=np.asarray(histories["val_medians"]))
    hg.create_dataset("val_means",
                      data=np.asarray(histories["val_means"]))
    rows = []
    for fr in histories["val_fracs"]:
      rows.append(np.asarray(fr.cpu() if torch.is_tensor(fr) else fr))
    hg.create_dataset("val_fracs", data=np.stack(rows))
    thr = histories["thresholds"]
    if torch.is_tensor(thr):
      thr = thr.cpu().numpy()
    hg.create_dataset("thresholds", data=np.asarray(thr))

    # the full configs, verbatim, as YAML text.
    f.create_dataset("config_yaml",
                     data=yaml.safe_dump(config, sort_keys=False),
                     dtype=str_dt)
    if train_args is not None:
      f.create_dataset("train_args_yaml",
                       data=yaml.safe_dump(train_args,
                                           sort_keys=False),
                       dtype=str_dt)

    # schema v2: the CONSUMED view, defaults materialized, so a saved run
    # reconstructs even if code defaults drift (the standing rule). The raw
    # config_yaml / train_args_yaml above stay as the provenance of what was
    # WRITTEN; these record what the run RESOLVED.
    if resolved_train is not None:
      f.create_dataset(
        "config_resolved_yaml",
        data=yaml.safe_dump({"train_args": resolved_train,
                             "data": config.get("data", {})},
                            sort_keys=False),
        dtype=str_dt)
    if resolved_model is not None:
      # the model rebuild recipe as a YAML string (plain dict, not tensors),
      # read back by rebuild_emulator; a missing recipe key there is loud,
      # never a code default.
      f.create_dataset(
        "model_recipe",
        data=yaml.safe_dump(resolved_model, sort_keys=False),
        dtype=str_dt)

    # run identity + provenance as root attributes. str() guards the
    # str subclasses h5py rejects (torch.__version__ is one: numpy
    # coerces a str subclass to a fixed-width unicode dtype h5py has
    # no conversion for; a plain str stores as variable-length utf8).
    if attrs is not None:
      for k, v in attrs.items():
        f.attrs[k] = str(v) if isinstance(v, str) else v
    f.attrs["created"]       = time.strftime("%Y-%m-%d %H:%M:%S")
    f.attrs["torch_version"] = str(torch.__version__)
    # schema_version marks a file that carries the resolved recipe (both v2
    # payloads present); rebuild_emulator refuses any file without it. The
    # code commit is best-effort provenance (rev-parse, else "unknown").
    if resolved_train is not None and resolved_model is not None:
      f.attrs["schema_version"] = 2
      try:
        f.attrs["git_commit"] = subprocess.check_output(
          ["git", "rev-parse", "HEAD"],
          cwd=os.path.dirname(os.path.abspath(__file__)),
          stderr=subprocess.DEVNULL).decode().strip()
      except Exception:
        f.attrs["git_commit"] = "unknown"

  return emul_path, h5_path


def _read_native_bool(attrs, key, *, default, where):
  """Read an HDF5 boolean attribute with NO truthiness coercion.

  A native boolean is required. An absent key returns the default; a Python
  or numpy boolean returns its value; anything else (a string, an integer)
  raises. HDF5 attributes carry no strong type, so a mistyped or forged
  attribute like the string "False" would be TRUE under Python truthiness
  (every nonempty string is true) and silently flip a feature-selection bit.
  For transfer_refined that would load a file's drifted prediction weights
  even though its marker literally reads false. The value is parsed by TYPE
  first, so any type check comparing it downstream sees a real boolean.

  Arguments:
    attrs   = an HDF5 attribute mapping (h5py AttributeManager or a dict).
    key     = the attribute name.
    default = the boolean returned when the key is absent (a native bool).
    where   = a location string for the error message (the file identity).

  Returns:
    a Python bool.

  Raises:
    ValueError when the attribute is present but not a native boolean.
  """
  if key not in attrs:
    return default
  value = attrs[key]
  if isinstance(value, (bool, np.bool_)):
    return bool(value)
  raise ValueError(
    where + ": the attribute " + repr(key) + " must be a native boolean, "
    "got " + repr(value) + " (type " + type(value).__name__ + "). HDF5 "
    "attributes are weakly typed, so a string or integer is refused rather "
    "than coerced by truthiness -- the string 'False' would read as True and "
    "select the wrong branch. Re-save the artifact with a real boolean.")


def rebuild_emulator(path_root, device, compile_model=True):
  """
  Reconstruct a saved emulator from <path_root>.h5 + .emul, using ONLY the
  file (save schema v2). The in-package inference entry point and the proof
  of the schema-v2 guarantee: every knob comes from the resolved recipe in
  the h5, so a run rebuilds bit-exactly even if code defaults later drift. A
  missing recipe key is a loud error, NEVER a fallback to a code default; a
  v1 file (no schema_version) is refused (it predates the guarantee).

  Arguments:
    path_root     = the output path without extension (as passed to
                    save_emulator); reads <path_root>.h5 and <path_root>.emul.
    device        = device to rebuild the module and geometries on.
    compile_model = torch.compile the rebuilt module on CUDA when the recipe
                    stored a compile_mode (default True; the predictor passes
                    False for batch-1 inference, where the compile latency
                    rarely pays off).

  Returns:
    (model, param_geometry, geometry, info): the inference-ready quad; the
    model in eval() with the best-epoch weights loaded (strict). info is a
    plain dict of the physics-branch metadata the predictor needs to build
    the decoder, read straight from the file so nothing is re-declared
    downstream:
      "ia"       = the factored-IA design name (nla / tatt) or None;
      "pce_base" = the reconstructed frozen PCEEmulator when the run used an
                   NPCE base (h5 pce group), else None;
      "pce_form" = the NPCE recombination form (residual / ratio) stored on
                   the pce group, else None.

  Raises:
    ValueError if the .h5 is not schema v2; KeyError naming any missing
    recipe / geometry / pce key (never a code-default fallback).
  """
  import importlib
  # h5py lives only here: the training machines (cocoa) ship it, the
  # plotting/train paths never need it.
  import h5py

  from .activations import make_activation
  from .designs.blocks import make_norm
  from .geometries.scalar import ScalarGeometry
  from .geometries.cmb import CmbDiagonalGeometry
  from .geometries.grid import GridGeometry
  from .geometries.grid2d import Grid2DGeometry

  def _read_group(g):
    # inverse of save's write_state: numeric datasets -> tensors, string
    # datasets (names) -> str lists, attrs -> scalars (a "torch.<dtype>"
    # string restored to the torch.dtype), subgroups -> nested dicts.
    state = {}
    for k in g:
      item = g[k]
      if isinstance(item, h5py.Group):
        state[k] = _read_group(item)
      elif h5py.check_string_dtype(item.dtype) is not None:
        vals = np.atleast_1d(item[()])
        state[k] = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in vals]
      else:
        state[k] = torch.as_tensor(np.asarray(item[()])).to(device)
    for k, v in g.attrs.items():
      if isinstance(v, str) and v.startswith("torch."):
        state[k] = getattr(torch, v.split(".", 1)[1])
      else:
        state[k] = v
    return state

  def _need(d, k, where):
    if k not in d:
      raise KeyError(
        f"{path_root}.h5 {where} is missing {k!r}; rebuild_emulator reads "
        "only the file and never falls back to a code default")
    return d[k]

  def _rebuild_geometry(group, where):
    # dispatch on the persisted class marker: read the group, resolve its
    # own "cls" (importlib, the model-recipe pattern), and call THAT class's
    # from_state, so a factored AmplitudeFactorGeometry / a LogParamGeometry
    # rebuilds as itself. A missing marker is loud and names a re-save --
    # never a silent fallback to the base geometry (the read-side rule).
    st = _read_group(group)
    if "cls" not in st:
      raise KeyError(
        f"{path_root}.h5 {where} is missing the 'cls' class marker; it was "
        "saved before the geometry-class fix. Re-save the emulator (retrain, "
        "or re-run save_emulator on the run) to add it -- rebuild_emulator "
        "never falls back to a base geometry class.")
    cls_path = st.pop("cls")
    mod, _, qual = cls_path.rpartition(".")
    return getattr(importlib.import_module(mod), qual).from_state(device, st)

  with h5py.File(path_root + ".h5", "r") as f:
    sv = f.attrs.get("schema_version")
    if sv != 2:
      raise ValueError(
        f"{path_root}.h5 is not a schema-v2 emulator (schema_version={sv!r}): "
        "rebuild_emulator needs the resolved model recipe a v2 save writes. "
        "v1 files predate the reconstruction guarantee; retrain + save to "
        "upgrade.")
    if "model_recipe" not in f:
      raise KeyError(f"{path_root}.h5 is missing the model_recipe")
    recipe = yaml.safe_load(f["model_recipe"][()])
    pgeom = _rebuild_geometry(f["param_geometry"], "param_geometry group")
    geom  = _rebuild_geometry(f["dv_geometry"], "dv_geometry group")
    pce_base = None
    pce_form = None
    if "pce" in f:
      from .designs.pce import PCEEmulator
      pce_grp  = _read_group(f["pce"])
      pce_form = _need(pce_grp, "form", "pce group")
      pce_base = PCEEmulator.from_state(pce_grp, device)

    # transfer_base (a transfer artifact only): read the embedded frozen base's
    # recipe, weights, and both geometries; the model is reconstructed below,
    # after the main model, through the shared helper. form / space tell the
    # predictor how to compose base + correction.
    tb_recipe = None
    tb_state  = None
    tb_pgeom  = None
    tb_geom   = None
    tb_form   = None
    tb_space  = None
    if "transfer_base" in f:
      tb        = f["transfer_base"]
      tb_recipe = yaml.safe_load(tb["model_recipe"][()])
      tb_state  = _read_group(tb["state"])
      tb_pgeom  = _rebuild_geometry(tb["param_geometry"],
                                    "transfer_base param_geometry group")
      tb_geom   = _rebuild_geometry(tb["dv_geometry"],
                                    "transfer_base dv_geometry group")
      tb_form   = tb.attrs.get("form")
      tb_space  = tb.attrs.get("space")
      # refined artifact (transfer.refine): the drifted base weights + the
      # transfer_refined root attr are two-way consistent (either half alone is
      # a corrupt file). When present the predictor composes with the DRIFTED
      # base (silently, no flag); the pretrained tb_state stays the provenance.
      # The marker is read as a NATIVE boolean, never truthiness-coerced: a
      # forged string "False" would otherwise read True and load the drifted
      # weights from a file whose marker says false.
      refined  = _read_native_bool(f.attrs, "transfer_refined",
                                   default=False, where=path_root + ".h5")
      have_dr  = "drifted_state" in tb
      if refined != have_dr:
        raise KeyError(
          path_root + ".h5 transfer_base is inconsistent: transfer_refined="
          + repr(refined) + " but drifted_state "
          + ("present" if have_dr else "absent")
          + "; a refined artifact must carry both, a frozen-only run neither")
      if have_dr:
        tb_state = _read_group(tb["drifted_state"])

  def _rebuild_model(rc, geom_for_needs, state, want_compile):
    # Reconstruct one module from its recipe + state dict (h5-only, a missing
    # key loud). Shared by the main model (the correction net on a transfer
    # run, or the plain emulator otherwise) and the embedded transfer base, so
    # the recipe -> constructor logic lives once. state is already the plain
    # (compile-prefix-stripped) name -> tensor mapping to load strict.
    cls_path = _need(rc, "cls", "model_recipe")
    mn, _, qn = cls_path.rpartition(".")
    cls = getattr(importlib.import_module(mn), qn)
    kwargs = dict(_need(rc, "kwargs", "model_recipe"))
    # re-make the factory objects from their serialized names.
    if "block_opts" in kwargs:
      bo  = kwargs["block_opts"]
      act = _need(bo, "act", "model_recipe.kwargs.block_opts")
      kwargs["block_opts"] = {
        "act": make_activation(_need(act, "type", "block_opts.act"),
                               n_gates=_need(act, "n_gates", "block_opts.act")),
        "norm": make_norm(_need(bo, "norm", "model_recipe.kwargs.block_opts")),
      }
    if kwargs.get("head_act") is not None:
      ha = kwargs["head_act"]
      kwargs["head_act"] = make_activation(
        _need(ha, "type", "head_act"),
        n_gates=_need(ha, "n_gates", "head_act"))
    if _need(rc, "needs_geom", "model_recipe"):
      # conv/TRF heads read geom.bin_sizes at construction. A diagonal
      # family geometry (cmb / grid / grid2d) derives the split
      # from its own saved grid — attach it here so a head artifact
      # rebuilds from the files alone. The cosmolike DataVectorGeometry
      # instead PERSISTS the split (bin_sizes / pm_kept in its state,
      # attached at training by build_shear_angle_map, which needs the
      # dataset ini — files rebuild must never require); from_state has
      # already restored it, so only its absence is checked: an older
      # head artifact that predates the persistence is refused loudly,
      # never guessed at.
      if hasattr(geom_for_needs, "attach_head_coords"):
        geom_for_needs.attach_head_coords()
      elif not hasattr(geom_for_needs, "bin_sizes"):
        raise KeyError(
          f"{path_root}.h5 dv_geometry has no bin_sizes, but the model "
          "recipe needs the geometry (a conv/TRF head): this artifact "
          "predates the bin-split persistence (the split was attached "
          "only at training time and never saved). Retrain, or re-run "
          "save_emulator on a live run, to write it -- rebuild_emulator "
          "never re-derives it (that would need ROOTDIR data files).")
      kwargs["geom"] = geom_for_needs
    m = cls(input_dim=_need(rc, "input_dim", "model_recipe"),
            output_dim=_need(rc, "output_dim", "model_recipe"),
            **kwargs).to(device)
    m.load_state_dict(state, strict=True)
    m.eval()
    cm = rc.get("compile_mode")
    if want_compile and device.type == "cuda" and cm is not None:
      m = torch.compile(m, mode=cm)
    return m

  # the main model: the correction net (transfer run) or the plain emulator.
  # the .emul holds its plain state dict; re-compile per the recipe on CUDA.
  sd    = torch.load(path_root + ".emul", map_location=device)
  model = _rebuild_model(rc=recipe, geom_for_needs=geom, state=sd,
                         want_compile=compile_model)

  # the frozen transfer base, rebuilt from the embedded recipe + weights +
  # geometries (never compiled: a batch-1 no_grad component of the predictor).
  transfer_base = None
  if tb_recipe is not None:
    base_model = _rebuild_model(rc=tb_recipe, geom_for_needs=tb_geom,
                                state=tb_state, want_compile=False)
    transfer_base = {"model": base_model,
                     "pgeom": tb_pgeom,
                     "geom":  tb_geom}
  return model, pgeom, geom, {
    "ia":             _need(recipe, "ia", "model_recipe"),
    "pce_base":       pce_base,
    "pce_form":       pce_form,
    "transfer_base":  transfer_base,
    "transfer_form":  tb_form,
    "transfer_space": tb_space,
    # scalar (derived-parameter) emulator: the output geometry rebuilt as
    # a ScalarGeometry, so the predictor takes the scalar branch.
    # Dispatched on the rebuilt class, not a stored attr, so an
    # older non-scalar artifact simply reports False.
    "scalar":         isinstance(geom, ScalarGeometry),
    # CMB-spectrum emulator: dispatched on the rebuilt class the
    # same way; the amplitude law + its column names are ARTIFACT FACTS
    # persisted in the geometry state, surfaced here so the
    # predictor / cobaya adapter rebuild the law-aware decode
    # without rereading the config. None / absent on non-CMB artifacts.
    # the law keys are guarded by the class check (a GridGeometry also
    # carries a .law attr — its TARGET law, a different registry), so a
    # bare getattr would smear one family's fact onto another.
    "cmb":            isinstance(geom, CmbDiagonalGeometry),
    "amplitude_law":  (geom.law
                       if isinstance(geom, CmbDiagonalGeometry) else None),
    "as_name":        (geom.as_name
                       if isinstance(geom, CmbDiagonalGeometry) else None),
    "tau_name":       (geom.tau_name
                       if isinstance(geom, CmbDiagonalGeometry) else None),
    # the order-one law's fiducial reference pair, surfaced so the
    # predictor rebuilds the same law-aware decode without rereading the
    # config; None for the "none" law (and for non-CMB artifacts).
    "as_ref":         (geom.as_ref
                       if isinstance(geom, CmbDiagonalGeometry) else None),
    "tau_ref":        (geom.tau_ref
                       if isinstance(geom, CmbDiagonalGeometry) else None),
    # grid (background-function) emulator: dispatched on the
    # rebuilt class; the quantity / units / law / offset are ARTIFACT
    # FACTS persisted in the geometry state, surfaced for the predictor
    # and the emul_baosn adapter. None / absent on non-grid artifacts.
    "grid":           isinstance(geom, GridGeometry),
    "grid_quantity":  (geom.quantity
                       if isinstance(geom, GridGeometry) else None),
    "grid_units":     (geom.units
                       if isinstance(geom, GridGeometry) else None),
    "grid_law":       (geom.law
                       if isinstance(geom, GridGeometry) else None),
    "grid_offset":    (geom.offset
                       if isinstance(geom, GridGeometry) else None),
    # grid2d (matter-power-spectrum) emulator: the same
    # class-guarded dispatch; the law names the syren base the artifact
    # corrects (the consumer multiplies it back).
    "grid2d":           isinstance(geom, Grid2DGeometry),
    "grid2d_quantity":  (geom.quantity
                         if isinstance(geom, Grid2DGeometry) else None),
    "grid2d_units":     (geom.units
                         if isinstance(geom, Grid2DGeometry) else None),
    "grid2d_law":       (geom.law
                         if isinstance(geom, Grid2DGeometry) else None),
  }
