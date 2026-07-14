"""Run-output I/O: learning-curve tables and trained-emulator files.

save_learning_curves writes a whitespace-delimited table (row per N_train,
column per curve, "#"-comment header carrying the config) that np.loadtxt
reads back, the format the sweep and bake-off drivers save, so several
runs can be overlaid later. save_emulator persists a trained run as two
files: <root>.emul, the model weights (a torch state_dict, cpu tensors),
and <root>.h5, everything inference or a paper trail needs (both
whitening geometries, the training histories, the full config, and the
scientific record the dataset was born under, copied verbatim from the
generator's sidecar).

read_artifact_schema is the one reader of a saved file's schema version and
of that record. Every path that opens a saved emulator goes through it: the
rebuild path here, and the warm-start loader that fine-tunes one
(emulator/warmstart.py). One reader is the whole point. Two readers of one
schema is how a file gets refused on one path and quietly accepted on the
other, and the accepted copy is the one that answers a likelihood.

PS: ``state_dict`` is PyTorch's name-to-tensor mapping for every registered
parameter, including frozen parameters, and every persistent registered
buffer. It is not the list of tensors that an optimizer updates. Whitening is
the center, rotation, and scaling transform applied by the geometries. The
scientific record is the pair of blocks defined in emulator/fixed_facts.py:
the cosmology held fixed while the dataset was generated, and the parameter
region it was sampled over. A sidecar is a small companion file written next
to a data file and sharing its name stem.
"""

import os
import subprocess
import time

import numpy as np
import torch
import yaml

from . import fixed_facts


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
                  transfer_base=None,
                  facts_yaml=None):
  """
  Persist a trained emulator as <path_root>.emul + <path_root>.h5.

  The .emul holds only the model weights. Before ``torch.save``,
  ``save_emulator`` detaches every state_dict tensor and moves it to
  the CPU. Loading therefore does not require the accelerator used
  during training. ``rebuild_emulator`` passes ``map_location=device``
  to ``torch.load``; ``map_location`` selects the restored tensors'
  destination device. A torch.compile'd model wraps the real one and
  prefixes every state_dict key with "_orig_mod."; the prefix is stripped so the
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
    config_resolved_yaml  the consumed config, defaults materialized
                     (resolved_train + the data block), so a saved run
                     reconstructs even if code defaults drift.
    model_recipe     the serializable model rebuild recipe (class qualname,
                     dims, every constructor kwarg, the act / norm / head
                     factories by name), read by rebuild_emulator (h5-only,
                     a missing key loud, never a code default).
    fixed_facts/     (facts_yaml runs only) the cosmology the dataset was
                     generated under: what was held fixed and at what value,
                     the neutrino convention, the dark-energy law, the units
                     the spectra are measured in.
    input_domain/    (facts_yaml runs only) the parameters the generator
                     sampled, in the canonical order it declared, with the
                     interval each was drawn from.
    facts_sidecar_yaml  (facts_yaml runs only) the producer's sidecar text
                     itself, stored verbatim beside those two groups, so the
                     record can be checked against the words it was copied
                     from. fixed_facts.write_h5 writes all three: it parses
                     the text and writes that parse, so the blocks in the
                     file are by construction the reading of the text in the
                     file, and this function never authors a scientific fact.
    train_args_yaml  the collapsed train_args actually used (search
                     ranges resolved to their defaults), as YAML.
  plus one root attribute per entry of `attrs` (run identity: model name,
  activation, rescale, N_train, best epoch, ...), a "created" timestamp, the
  torch version, and the schema version with the git commit beside it. Those
  last two are written exactly when the run resolved both recipes: version 3
  when it also carried the scientific record, version 2 when it did not.

  Reversible map (write here -> read in rebuild_emulator):
    This table pairs everything this function writes with the exact
    place rebuild_emulator reads it back, so the round trip can be
    checked line by line. The house rule is that a saved run
    reconstructs from the file alone, so the read side never falls
    back to a code default: every key it needs is fetched through the
    _need / _read_native_bool helpers, which raise a named error when
    the key is absent instead of substituting a value. The rows marked
    "not read (provenance)" are written on purpose as a paper trail
    (plots, audits, git history); rebuild_emulator does not consume
    them, and that is the intended asymmetry, not a dropped key.

      written by save_emulator      | read back in rebuild_emulator
      ------------------------------|------------------------------------
      <root>.emul (state_dict,      | torch.load(<root>.emul), loaded
        cpu, compile-prefix         |   strict into the main model by
        stripped)                   |   _rebuild_model
      param_geometry/ group +       | _rebuild_geometry(f["param_geometry"])
        its "cls" attr              |   -> <cls>.from_state; "cls" is
                                    |   required (missing = loud re-save)
      dv_geometry/ group +          | _rebuild_geometry(f["dv_geometry"])
        its "cls" attr              |   -> <cls>.from_state; the info-dict
                                    |   family flags and the CMB / grid /
                                    |   grid2d facts below are read off
                                    |   this rebuilt geometry object, not
                                    |   from separate h5 keys
      pce/ group (NPCE runs) +      | PCEEmulator.from_state(pce_grp) and
        its "form" attr             |   _need(pce_grp, "form") -> pce_base,
                                    |   pce_form
      transfer_base/ group          | read whole when "transfer_base" in f:
        (transfer runs):            |   tb_recipe / tb_state / tb_pgeom /
        model_recipe, state/,       |   tb_geom rebuilt, then _rebuild_model
        param_geometry/,            |   builds the frozen base; form / space
        dv_geometry/,               |   -> transfer_form / transfer_space
        "form" + "space" attrs      |
      transfer_base/drifted_state/  | _read_group(tb["drifted_state"])
        (refined runs only) +       |   replaces tb_state; the root attr is
        root attr transfer_refined  |   read as a native bool by
                                    |   _read_native_bool, then cross-checked
                                    |   two-way against the group's presence
      model_recipe (YAML)           | yaml.safe_load(f["model_recipe"]) ->
                                    |   recipe; drives _rebuild_model (cls,
                                    |   dims, kwargs, act / norm / head
                                    |   factories, compile_mode) and supplies
                                    |   info["ia"], each via _need
      schema_version (root attr)    | read_artifact_schema(f, <root>.h5): a
                                    |   version this code does not know is
                                    |   refused before any other read, and so
                                    |   is an absent one
      fixed_facts/ + input_domain/  | the same call hands both groups to
        groups + facts_sidecar_yaml |   fixed_facts.read_h5, which re-parses
        (facts_yaml runs only)      |   the stored text and checks it against
                                    |   the stored blocks in both directions;
                                    |   the two blocks reach the caller as
                                    |   info["fixed_facts"] and
                                    |   info["input_domain"]
      history/ group (train_losses, | not read (provenance): per-epoch
        val_medians, val_means,     |   training curves for plots and audit
        val_fracs, thresholds)      |
      config_yaml,                  | not read (provenance): the verbatim and
        train_args_yaml,            |   resolved config text, kept so a saved
        config_resolved_yaml        |   run documents what it consumed
      attrs entries, created,       | not read (provenance): run identity,
        torch_version, git_commit   |   timestamp, and build marks
    (legend: "<root>" = path_root; "cls" = a "module.QualName" string
     naming the class to reconstruct; state_dict = PyTorch's name -> tensor
     map of registered parameters, including frozen parameters, and
     persistent registered buffers; _need / _read_group
     / _read_native_bool / _rebuild_geometry / _rebuild_model = the reader
     helpers defined inside rebuild_emulator; read_artifact_schema = the one
     shared reader of the schema version and the scientific record, defined
     below at module level because the warm-start loader calls it too;
     "->" = "feeds into".)

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
    facts_yaml     = the producer sidecar's text, the generator's own
                     <paramsf>.facts.yaml, carried here verbatim by the
                     training loader (data_staging.read_facts_sidecar). Given,
                     the file records the science it was born under and
                     announces schema version 3; None (a dataset generated
                     before the record existed), it stays a version 2 file and
                     records no science at all. It is never re-derived here:
                     two authors of one scientific fact is how the two copies
                     of that fact drift apart.

  Returns:
    (emul_path, h5_path), the two files written.

  Raises:
    ValueError when facts_yaml arrives without the resolved recipes (the file
    would say which cosmology it was trained under but not how to rebuild the
    network), when the sidecar breaks any law in fixed_facts.validate, or when
    the whitening geometry and the sidecar disagree on the sampled parameters.
    All three are checked before either file is written, so a refused save
    leaves no half-written pair on disk.
  """
  # h5py lives only here: the training machines (cocoa) ship it, the
  # plotting/train paths never need it.
  import h5py

  # --- the version this file will announce: one decision, before any write ---
  # The version is a statement about the payload, so it is decided once, from
  # the payload, and every version-dependent write below reads this one value.
  # A run that resolved both recipes rebuilds from the file alone (version 2).
  # A run that also carries the producer's scientific record additionally says
  # which cosmology it was trained under and which region it may be asked about
  # (version 3). Deciding the number in two places is how a file ends up
  # announcing a payload it does not carry.
  have_recipe = resolved_train is not None and resolved_model is not None
  if facts_yaml is not None and not have_recipe:
    raise ValueError(
      path_root + ": the scientific record reached save_emulator without the "
      "resolved training and model recipes, so the file would say which "
      "cosmology it was trained under but not how to rebuild the network from "
      "it. Pass resolved_train and resolved_model as well, or pass no record.")
  if not have_recipe:
    schema_version = None
  elif facts_yaml is None:
    schema_version = 2
  else:
    schema_version = fixed_facts.SCHEMA_VERSION

  # The parameters the generator declared it sampled must be the parameters the
  # whitening geometry holds, in the same order, and the check runs before
  # either file is written. Every law the record must satisfy belongs to
  # fixed_facts and is enforced there, on the way in and on the way back out,
  # so that a record cannot be refused on one path and accepted on another.
  # This function is a caller of those laws, never a second author of them.
  # check_names_match / parse_sidecar (fixed_facts.py): the record's own laws.
  if facts_yaml is not None:
    record = fixed_facts.parse_sidecar(
      text=facts_yaml,
      where="the producer sidecar being saved into " + path_root + ".h5")
    fixed_facts.check_names_match(
      geometry_names=param_geometry.state()["names"],
      blocks=record,
      where=path_root + ".h5")

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
    # the scientific record, when the dataset published one: the producer's own
    # sidecar text, stored verbatim, and the two blocks that text parses to.
    # write_h5 (fixed_facts.py) does both, and it writes the parse of the text
    # it stores, so the file's blocks and the file's text can be held against
    # each other when it is read back. Training copies the record; it never
    # writes one. The groups appear exactly when the version decided above
    # promises them, so the number the file announces and the groups the file
    # carries are one decision, not two.
    if schema_version == fixed_facts.SCHEMA_VERSION:
      fixed_facts.write_h5(f=f, sidecar_text=facts_yaml)

    # input whitening. param_geometry.state() (geometries.parameter.py):
    # the input-whitening tensors keyed exactly as from_state expects. The
    # group also records its own CLASS (materialized from the object's type
    # at write time) so rebuild dispatches to the right from_state. A
    # factored run's AmplitudeFactorGeometry, a log run's LogParamGeometry,
    # a plain run's ParamGeometry, rather than hardcoding the base class
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
    # the version decided at the top of this function, written once, here. It
    # marks a file that carries the resolved recipe, and at version 3 the
    # scientific record too; rebuild_emulator refuses any file without it. The
    # code commit is best-effort provenance (rev-parse, else "unknown").
    if schema_version is not None:
      f.attrs["schema_version"] = schema_version
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
    "than coerced by truthiness. The string 'False' would read as True and "
    "select the wrong branch. Re-save the artifact with a real boolean.")


def read_artifact_schema(f, where):
  """Read a saved emulator's schema version and the science it was born under.

  The one reader of both. Every path that opens a saved emulator comes through
  here: rebuild_emulator below, and the warm-start loader that fine-tunes one
  (warmstart.load_source). A second reader of the same file would be free to
  accept what this one refuses, and fine-tuning is exactly the path that
  narrows the parameter region an emulator serves, so it is the path that most
  needs the record present rather than assumed.

  The version is read first, because it says which grammar the rest of the file
  is written in. A file that announces no version is refused rather than
  guessed at: a file written before the version existed and a file whose writer
  forgot to stamp one read exactly alike, and only one of the two would be safe
  to open. A version this code does not know is refused for the same reason.
  Both refusals carry the migration instruction, because a message that says a
  file is incompatible and then stops is not a refusal, it is a shrug.

  The two groups are then handed to fixed_facts.read_h5, which owns every law
  they must satisfy and which checks the blocks stored in the file against the
  producer's own text stored beside them. None of that is restated here.

  Arguments:
    f     = the open h5py File to read. The caller opens it and closes it; this
            function only reads.
    where = the file's identity, named in every refusal (its path, usually).

  Returns:
    the two blocks of the scientific record, in plain Python types, keyed
    "fixed_facts" (the cosmology the run was trained under) and "input_domain"
    (the parameters the generator sampled, in the canonical order it declared,
    with the interval each was drawn from). They are plain values, so they
    outlive the open file the caller is about to close.

  Raises:
    ValueError when the file announces no schema version, when it announces a
    version this code does not know (every emulator saved before the record
    existed), or when fixed_facts.read_h5 refuses either block.
  """
  version = f.attrs.get("schema_version")
  if version is None:
    raise ValueError(
      where + " announces no schema version, so it does not say which grammar "
      "it was written in, and an emulator whose grammar is unknown is refused "
      "rather than read. " + fixed_facts.MIGRATION)
  # HDF5 attributes carry no Python types: an integer written here comes back
  # as a numpy integer, and the schema laws compare the version against plain
  # integers. It is made plain in this one place, the one place that reads it.
  # A value that is not an integer at all matches no version this code knows
  # and is refused just below, which is the right outcome for the same reason.
  if isinstance(version, np.integer):
    version = int(version)
  if version != fixed_facts.SCHEMA_VERSION:
    raise ValueError(
      where + " is written in schema version " + repr(version) + ", and this "
      "code reads version " + repr(fixed_facts.SCHEMA_VERSION) + " only. An "
      "emulator saved under an earlier version records neither the cosmology "
      "it was trained under nor the parameter region it may be asked about, so "
      "serving it would mean guessing at both. " + fixed_facts.MIGRATION)
  return fixed_facts.read_h5(f=f,
                             schema_version=version,
                             where=where)


def rebuild_emulator(path_root, device, compile_model=True):
  """
  Reconstruct a saved emulator from <path_root>.h5 + .emul, using only the
  file. The in-package inference entry point and the proof of the
  reconstruction guarantee: every knob comes from the resolved recipe in the
  h5, so a run rebuilds bit-exactly even if code defaults later drift. A
  missing recipe key is a loud error, never a fallback to a code default.
  read_artifact_schema (above) reads the file's schema version and its
  scientific record before anything else is read; a file saved under a version
  this code does not know is refused there, because it cannot say which
  cosmology it was trained under or which parameter region it may be asked
  about.

  For the full write-to-read crosswalk (every value save_emulator writes
  paired with the line here that reads it, and the entries written only as
  provenance that this function does not consume), see the "Reversible map"
  table in save_emulator's docstring.

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
      "fixed_facts"  = the cosmology the run was trained under, exactly as the
                       generator published it: what was held fixed and at what
                       value, the neutrino convention, the dark-energy law;
      "input_domain" = the parameters the generator sampled, in the canonical
                       order it declared, with the interval each was drawn
                       from, which is the region this emulator may be asked
                       about;
      "ia"           = the factored-IA design name (nla / tatt) or None;
      "pce_base"     = the reconstructed frozen PCEEmulator when the run used
                       an NPCE base (h5 pce group), else None;
      "pce_form"     = the NPCE recombination form (residual / ratio) stored
                       on the pce group, else None.

  Raises:
    ValueError from read_artifact_schema when the .h5 announces no schema
    version, announces one this code does not know, or breaks any law of the
    scientific record; ValueError when the rebuilt input geometry disagrees
    with that record's sampled-parameter order; KeyError naming any missing
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
        "or re-run save_emulator on the run) to add it. rebuild_emulator "
        "never falls back to a base geometry class.")
    cls_path = st.pop("cls")
    mod, _, qual = cls_path.rpartition(".")
    return getattr(importlib.import_module(mod), qual).from_state(device, st)

  with h5py.File(path_root + ".h5", "r") as f:
    # the schema version and the scientific record, through the one shared
    # reader. It refuses a version this code does not know, and a file that
    # announces none, before any other value is read. The blocks it returns are
    # plain Python, so they survive this file being closed and travel out in
    # the info dict below.
    record = read_artifact_schema(f=f, where=path_root + ".h5")
    if "model_recipe" not in f:
      raise KeyError(f"{path_root}.h5 is missing the model_recipe")
    recipe = yaml.safe_load(f["model_recipe"][()])
    pgeom = _rebuild_geometry(f["param_geometry"], "param_geometry group")
    # The HDF5 blocks and their copied sidecar text prove that the scientific
    # record is internally consistent, but both copies can be rewritten
    # together.  The rebuilt whitening geometry is the independent copy of the
    # sampled-coordinate order, so compare it again on the read path before a
    # model can be constructed or handed values in the wrong column order.
    fixed_facts.check_names_match(
      geometry_names=pgeom.names,
      blocks=record,
      where=path_root + ".h5")
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
          "save_emulator on a live run, to write it. rebuild_emulator "
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
    # the science the run was born under, read by the shared reader above and
    # handed on whole: the cosmology held fixed while the dataset was generated
    # (fixed_facts) and the region the generator sampled (input_domain). The
    # consumer's comparison laws execute on these, so they travel with the
    # rebuilt emulator instead of sending the consumer back to the file.
    "fixed_facts":    record[fixed_facts.FIXED_FACTS_GROUP],
    "input_domain":   record[fixed_facts.INPUT_DOMAIN_GROUP],
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
