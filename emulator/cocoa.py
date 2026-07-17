"""Cocoa-framework path resolution shared by the CLI drivers.

The drivers run inside the cocoa framework, launched from $ROOTDIR rather than
from the data folder, so every path is resolved against the project layout
instead of the current directory. A --root names the project folder under
$ROOTDIR (e.g. projects/lsst_y1), a --fileroot a subfolder of it holding this
emulator's YAML and outputs (e.g. emulators/training_scripts), and the training
/ validation data files sit under the project's chains/ folder (where
dataset_generator_lensing.py writes them). add_cocoa_path_args registers
the three shared flags (--root, --fileroot, --yaml); resolve_cocoa_config reads
$ROOTDIR, builds the two roots, and loads the YAML from the fileroot (or an
absolute --yaml as-is). The YAML keeps short logical dataset filenames. Each
parameter filename selects an immutable published generation, and every
related path is rewritten to a verified member of that same generation. It
returns the config plus the fileroot (configs) and chains (published data + run
products) folders; cocoa_output places a run output under one of them.

PS: ROOTDIR is the cocoa install root, an environment variable cocoa exports;
every project path is taken relative to it.
"""

import os
from pathlib import Path

import yaml

from compute_data_vectors.dataset_publication import load_dataset_locator
from compute_data_vectors.dataset_publication import load_located_generation


_DATASET_SOURCES_KEY = "_dataset_sources"
_RESOLVER_OWNED_KEYS = (
  _DATASET_SOURCES_KEY,
  "train_failure_mask",
  "val_failure_mask",
)


def _logical_basename(data, key):
  """Return one YAML filename that names a publication member.

  Dataset YAMLs retain their familiar filenames, but those names are lookup
  keys rather than permission to open a mutable flat file.  Requiring one
  basename also prevents a config from selecting another project's locator.
  """
  if key not in data:
    raise KeyError(
      "data is missing " + repr(key) + "; a published dataset is selected "
      "through that logical filename")
  value = data[key]
  if type(value) is not str or not value:
    raise ValueError(
      "data." + key + " must be one nonempty logical filename; got "
      + repr(value))
  path = Path(value)
  if path.is_absolute() or path.parent != Path(".") or path.name != value:
    raise ValueError(
      "data." + key + " must be one filename from the project chains "
      "folder, not a path; got " + repr(value))
  return value


def _member(generation, role, *, split):
  """Return one required authenticated member with split-aware context."""
  try:
    return generation.member(role)
  except KeyError as exc:
    raise ValueError(
      split + " published dataset has no required member role "
      + repr(role)) from exc


def _rewrite_logical_member(container, key, generation, role, *, split):
  """Check one logical name and replace it with its immutable member path."""
  logical = _logical_basename(container, key)
  member = _member(generation, role, split=split)
  if logical != member.relative_path:
    raise ValueError(
      "data." + key + " names " + repr(logical) + " but the " + split
      + " published dataset assigns role " + repr(role) + " to "
      + repr(member.relative_path))
  container[key] = str(member.path)
  return member


def _generation_pin(locator, generation, chains):
  """Return a YAML-safe record of the exact generation used by this run."""
  try:
    locator_path = str(locator.path.relative_to(Path(chains)))
  except ValueError:
    # A compliant locator is always inside chains.  Keep a useful diagnostic
    # if a custom test double or future backend violates that boundary.
    locator_path = str(locator.path)
  return {
    "schema": 1,
    "logical_parameter": locator.logical_parameter,
    "locator_path": locator_path,
    "slot_id": generation.slot.slot_id,
    "slot": generation.slot.descriptor,
    "generation": generation.generation,
    "active_sha256": generation.active_sha256,
    "manifest_sha256": generation.manifest_sha256,
    "identity": generation.identity,
    "members": {
      member.role: {
        "relative_path": member.relative_path,
        "size": member.size,
        "sha256": member.sha256,
      }
      for member in generation.members
    },
  }


def _load_split(chains, data, key, *, split):
  """Resolve one logical parameter file to one pinned active generation."""
  logical = _logical_basename(data, key)
  locator = load_dataset_locator(chains, logical_parameter=logical)
  if locator.logical_parameter != logical:
    raise ValueError(
      split + " dataset locator returned logical parameter "
      + repr(locator.logical_parameter) + " for requested " + repr(logical))
  generation = load_located_generation(locator)
  parameter = _member(generation, "parameters.chain", split=split)
  if parameter.relative_path != logical:
    raise ValueError(
      split + " dataset locator names logical parameter " + repr(logical)
      + " but its authenticated generation assigns parameters.chain to "
      + repr(parameter.relative_path))
  return locator, generation


def _config_family(data):
  """Return the publication family and mode selected by the YAML shape."""
  special = [name for name in ("outputs", "cmb", "grid", "grid2d")
             if name in data]
  if len(special) > 1:
    raise ValueError(
      "data selects more than one emulator family: " + repr(special))
  if special == ["outputs"]:
    # Scalar targets are derived columns in a chain-only parameter table.  The
    # underlying generator family remains part of the publication identity.
    return None, "chain-only"
  if special == ["cmb"]:
    return "cmb", "full"
  if special == ["grid"]:
    return "grid", "full"
  if special == ["grid2d"]:
    return "grid2d", "full"
  return "cosmolike", "full"


def _require_route(train, validation, *, family, mode):
  """Require both splits to describe the same scientific payload.

  Train and validation may use different random seeds, sampling modes, and
  resolved sampling controls. They may not change what one row means: the
  probe, generator, ordered parameter columns, and invariant scientific
  contract must agree exactly.
  """
  train_identity = train.identity
  val_identity = validation.identity
  for split, identity in (("training", train_identity),
                          ("validation", val_identity)):
    if identity.get("dataset_mode") != mode:
      raise ValueError(
        split + " dataset mode is " + repr(identity.get("dataset_mode"))
        + " but this config requires " + repr(mode))
    if family is not None and identity.get("family") != family:
      raise ValueError(
        split + " dataset family is " + repr(identity.get("family"))
        + " but this config requires " + repr(family))
  if train_identity.get("family") != val_identity.get("family"):
    raise ValueError(
      "training and validation datasets belong to different families: "
      + repr(train_identity.get("family")) + " and "
      + repr(val_identity.get("family")))
  if train_identity.get("family_variant") \
      != val_identity.get("family_variant"):
    raise ValueError(
      "training and validation datasets use different family variants: "
      + repr(train_identity.get("family_variant")) + " and "
      + repr(val_identity.get("family_variant")))
  for field, label in (("probe", "probes"),
                       ("generator", "generators"),
                       ("scientific_contract_sha256",
                        "scientific contracts")):
    if train_identity.get(field) != val_identity.get(field):
      raise ValueError(
        "training and validation datasets use different " + label + ": "
        + repr(train_identity.get(field)) + " and "
        + repr(val_identity.get(field)))
  train_parameters = train_identity.get("parameters", {})
  val_parameters = val_identity.get("parameters", {})
  if train_parameters.get("names") != val_parameters.get("names"):
    raise ValueError(
      "training and validation datasets use different ordered parameter "
      "names: " + repr(train_parameters.get("names")) + " and "
      + repr(val_parameters.get("names")))


def _require_same_axis(train, validation, role):
  """Require train and validation to publish byte-identical coordinate axes."""
  train_axis = _member(train, role, split="training")
  val_axis = _member(validation, role, split="validation")
  if (train_axis.size, train_axis.sha256) != (val_axis.size, val_axis.sha256):
    raise ValueError(
      "training and validation datasets use different " + role
      + " axes; regenerate one split on the same coordinates")
  return train_axis


def _resolve_family_paths(data, train, validation, *, family, mode):
  """Rewrite every consumer path from two already pinned generations."""
  _rewrite_logical_member(
    data, "train_params", train, "parameters.chain", split="training")
  _rewrite_logical_member(
    data, "val_params", validation, "parameters.chain", split="validation")
  _rewrite_logical_member(
    data, "train_covmat", train, "parameters.covariance", split="training")

  if mode == "chain-only":
    if "train_dv" in data or "val_dv" in data:
      raise ValueError(
        "a chain-only scalar dataset has no train_dv or val_dv payload; "
        "remove those keys")
    return

  if family == "cosmolike":
    payload_role = "payload.cosmolike.vector"
  elif family == "cmb":
    cmb = data.get("cmb")
    if not isinstance(cmb, dict):
      raise ValueError("data.cmb must be a mapping")
    spectrum = str(cmb.get("spectrum", "")).lower()
    if spectrum not in ("tt", "te", "ee", "pp"):
      raise ValueError(
        "data.cmb.spectrum must be one of tt / te / ee / pp before its "
        "published payload can be selected; got " + repr(cmb.get("spectrum")))
    payload_role = "payload.cmb." + spectrum
    _require_same_axis(train, validation, "axis.cmb.multipole")
  elif family == "grid":
    grid = data.get("grid")
    if not isinstance(grid, dict):
      raise ValueError("data.grid must be a mapping")
    quantity_roles = {"Hubble": "h", "D_M": "dm"}
    quantity = grid.get("quantity")
    if quantity not in quantity_roles:
      raise ValueError(
        "data.grid.quantity must be 'Hubble' or 'D_M' before its published "
        "payload can be selected; got " + repr(quantity))
    suffix = quantity_roles[quantity]
    payload_role = "payload.grid." + suffix
    axis_role = "axis.grid." + suffix + ".redshift"
    _require_same_axis(train, validation, axis_role)
    _rewrite_logical_member(
      grid, "z_file", train, axis_role, split="training")
  elif family == "grid2d":
    grid = data.get("grid2d")
    if not isinstance(grid, dict):
      raise ValueError("data.grid2d must be a mapping")
    quantity = str(grid.get("quantity", ""))
    if quantity not in ("pklin", "boost"):
      raise ValueError(
        "data.grid2d.quantity must be 'pklin' or 'boost' before its "
        "published payload can be selected; got "
        + repr(grid.get("quantity")))
    payload_role = "payload.grid2d." + quantity
    for key, role in (("z_file", "axis.grid2d.redshift"),
                      ("k_file", "axis.grid2d.wavenumber")):
      _require_same_axis(train, validation, role)
      _rewrite_logical_member(
        grid, key, train, role, split="training")
    variant = train.identity.get("family_variant")
    if variant == "syren-base":
      base_role = "base.grid2d." + quantity
      _rewrite_logical_member(
        grid, "train_base", train, base_role, split="training")
      _rewrite_logical_member(
        grid, "val_base", validation, base_role, split="validation")
    elif "train_base" in grid or "val_base" in grid:
      raise ValueError(
        "a native grid2d publication has no train_base or val_base; remove "
        "those keys or generate the syren-base variant")
  else:
    raise AssertionError("unhandled full dataset family " + repr(family))

  _rewrite_logical_member(
    data, "train_dv", train, payload_role, split="training")
  _rewrite_logical_member(
    data, "val_dv", validation, payload_role, split="validation")
  data["train_failure_mask"] = str(
    _member(train, "rows.failure-mask", split="training").path)
  data["val_failure_mask"] = str(
    _member(validation, "rows.failure-mask", split="validation").path)


def _resolve_published_datasets(data, chains):
  """Pin train and validation once, then rewrite all related config paths."""
  for key in _RESOLVER_OWNED_KEYS:
    if key in data:
      raise ValueError(
        "data." + key + " is written by resolve_cocoa_config and must not "
        "appear in source YAML")
  train_locator, train = _load_split(
    chains, data, "train_params", split="training")
  val_locator, validation = _load_split(
    chains, data, "val_params", split="validation")
  family, mode = _config_family(data)
  _require_route(train, validation, family=family, mode=mode)
  _resolve_family_paths(
    data, train, validation, family=family, mode=mode)
  data[_DATASET_SOURCES_KEY] = {
    "schema": 1,
    "train": _generation_pin(train_locator, train, chains),
    "validation": _generation_pin(val_locator, validation, chains),
  }


def add_cocoa_path_args(parser):
  """
  Register the shared cocoa path flags on an argument parser.

  Adds the three flags every driver shares: --root and --fileroot (the
  ROOTDIR-relative project layout) and --yaml (the config file under
  fileroot). Call once per driver before parse_known_args.

  Arguments:
    parser = the argparse.ArgumentParser to add the flags to.
  """
  parser.add_argument("--root",
                      dest="root",
                      help="project folder under $ROOTDIR holding the "
                           "data and this emulator (e.g. "
                           "projects/lsst_y1)",
                      type=str,
                      required=True)
  parser.add_argument("--fileroot",
                      dest="fileroot",
                      help="subfolder of --root holding this emulator's "
                           "YAML and outputs (e.g. "
                           "emulators/training_scripts)",
                      type=str,
                      required=True)
  parser.add_argument("--yaml",
                      dest="yaml",
                      help="config YAML under --fileroot, or an "
                           "absolute path used as-is (data + train_args "
                           "blocks); default test.yaml",
                      type=str,
                      default=None)


def resolve_cocoa_config(args):
  """
  Resolve the cocoa project layout and load the path-resolved config.

  Reads $ROOTDIR, joins --root and --fileroot under it, ensures the project
  chains/ folder exists, and loads the YAML from the fileroot (test.yaml when
  --yaml is unset; an absolute --yaml is read as-is). The logical training and
  validation parameter filenames each select one immutable published
  generation. Every parameter, payload, covariance, coordinate, base, and
  failure-mask path is taken from those two authenticated generations. There
  is no fallback to mutable flat files in chains/.

  Arguments:
    args = the parsed CLI namespace; reads args.root, args.fileroot,
           and args.yaml (the flags add_cocoa_path_args registered).

  Returns:
    cfg      = the parsed config mapping, its logical data names replaced by
               immutable published-member paths and source identity records.
    fileroot = absolute emulator folder (<root>/<fileroot>), holding the
               YAML configs.
    chains   = absolute project chains/ folder (<root>/chains), holding
               the data files and this driver's run products.
  """
  # $ROOTDIR/<root>/<fileroot>, mirroring dataset_generator_lensing.py:
  # root holds the data and a chains/ folder; fileroot holds this
  # emulator's YAML configs.
  root_env = os.environ.get("ROOTDIR")
  if not root_env:
    raise RuntimeError("ROOTDIR environment variable is not set")
  root = root_env.rstrip("/")
  root = f"{root}/{args.root.rstrip('/')}"
  fileroot = f"{root}/{args.fileroot.rstrip('/')}"
  chains = f"{root}/chains"
  Path(chains).mkdir(parents=True, exist_ok=True)

  # The YAML lives under the emulator's fileroot; default test.yaml. An
  # absolute --yaml is read as-is so a caller such as the gates harness can
  # pass a fully resolved config path. Dataset filenames inside that config
  # remain logical basenames and are resolved separately below.
  if args.yaml is None:
    yaml_path = f"{fileroot}/test.yaml"
  elif os.path.isabs(args.yaml):
    yaml_path = args.yaml
  else:
    yaml_path = f"{fileroot}/{args.yaml}"
  if not os.path.isfile(yaml_path):
    raise FileNotFoundError(f"YAML file not found: {yaml_path}")
  with open(yaml_path) as f:
    cfg = yaml.safe_load(f)
  if not isinstance(cfg, dict):
    raise ValueError(f"config did not parse to a mapping: {yaml_path}")

  # Each familiar YAML filename is now a logical name.  Its immutable locator
  # selects one slot, and that slot's active record is read exactly once for
  # train and once for validation.  The returned config contains only paths
  # inside those two authenticated generations.  A missing locator is an
  # explicit migration/regeneration error; mutable legacy flat files are never
  # a fallback.
  data = cfg.get("data")
  if isinstance(data, dict):
    _resolve_published_datasets(data, chains)

  return cfg, fileroot, chains


def cocoa_output(base, path):
  """
  Place a run-output path under a base folder.

  Joins a relative output path under `base` (the chains folder for run
  products like the diagnostics PDF, or the fileroot for configs); an
  absolute path passes through unchanged (os.path.join drops the
  earlier parts on an absolute tail).

  Arguments:
    base = absolute folder to place the output in (fileroot or chains,
           both from resolve_cocoa_config).
    path = the output path, relative to base or absolute.

  Returns:
    the resolved output path.
  """
  return os.path.join(base, path)
