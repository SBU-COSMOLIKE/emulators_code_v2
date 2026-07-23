"""Cocoa-framework path resolution shared by the command-line drivers.

The drivers are the train/tune/sweep programs at the repository root.
They run inside cocoa, the wider analysis framework this library is one
arm of.  Cocoa exports the environment variable ROOTDIR, its install
root, and launches every program from there rather than from the data
folder, so a path written relative to the current directory would point
at the wrong place.  Every path is therefore resolved against the
project layout instead:

- ``--root`` names the project folder under $ROOTDIR, for example
  ``projects/lsst_y1``;
- ``--fileroot`` names a subfolder of the project holding this
  emulator's YAML configuration and outputs, for example
  ``emulators/training_scripts``;
- the training and validation data files sit under the project's
  ``chains/`` folder, where the dataset generators (for example
  dataset_generator_lensing.py) write them.  The folder keeps the name
  cocoa gives sampler outputs, because the generators write their files
  beside those chains.

``add_cocoa_path_args`` registers the three shared flags (--root,
--fileroot, --yaml).  ``resolve_cocoa_config`` reads $ROOTDIR, builds
the two roots, ensures the project ``chains/`` folder exists, loads the
YAML from the fileroot (an absolute --yaml is read as-is), and rewrites
the configuration's data paths to absolute so the experiment reads the
same files regardless of the launch directory; it returns the
configuration plus the fileroot (configs) and chains (data and run
products) folders.  ``cocoa_output`` places one run output under one of
those two folders.
"""

import os
from pathlib import Path

import yaml


# data-block keys naming input files on disk; each is resolved against the
# project root. "dv" is the data vector (the stacked observables the
# emulator predicts for one parameter row), "params" the sampled parameter
# table, "covmat" the covariance matrix that weights the training loss,
# and the failure masks carry one 0/1 flag per row marking rows whose
# data-vector calculation failed. The cosmolike_* keys are absent here:
# they resolve against $ROOTDIR/external_modules/data inside the output
# geometry, not the project.
_DATA_PATH_KEYS = (
  "train_dv",
  "train_params",
  "train_covmat",
  "val_dv",
  "val_params",
  "train_failure_mask",
  "val_failure_mask",
)

# file-naming keys nested inside a family sub-block of data; each is also a
# file the generator wrote into the project chains/ folder: the grid axis
# files (z_file and k_file list the redshift and wavenumber points the
# grids are evaluated on), the syren baseline dumps (train_base and
# val_base hold the analytic syren-formula values the emulator corrects),
# and the CMB covariance from compute_cmb_covariance.py.
_NESTED_DATA_PATH_KEYS = (
  ("grid",   ("z_file",)),
  ("grid2d", ("z_file", "k_file", "train_base", "val_base")),
  ("cmb",    ("covariance",)),
)


def add_cocoa_path_args(parser):
  """Register the shared cocoa path flags on an argument parser.

  argparse is Python's standard command-line parser: a driver creates
  one ArgumentParser, declares the flags it accepts, and the parser
  turns ``--root projects/lsst_y1`` on the command line into an
  attribute ``args.root``.  This helper declares the three flags every
  driver shares: --root and --fileroot (the ROOTDIR-relative project
  layout described in the module docstring) and --yaml (the
  configuration file under the fileroot).  Call it once per driver,
  before the driver asks the parser to read the command line.

  Arguments:
    parser = the argparse.ArgumentParser to add the flags to.

  Returns:
    None; the parser is modified in place.
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
  """Resolve the cocoa project layout and load the path-resolved config.

  Reads the $ROOTDIR environment variable, joins --root and --fileroot
  under it, and creates the project ``chains/`` folder when missing
  (``Path.mkdir`` with ``parents=True`` also creates missing parent
  folders, and ``exist_ok=True`` makes an already-existing folder a
  no-op instead of an error).  The YAML is then loaded from the
  fileroot -- ``test.yaml`` when --yaml is unset, an absolute --yaml
  read as-is -- with ``yaml.safe_load``, the YAML parser variant that
  builds only plain Python values (mappings, lists, strings, numbers)
  and refuses the tags that would let a config file construct arbitrary
  Python objects.

  Finally every data-block file path is rewritten to an absolute path
  under the project's ``chains/`` folder: the flat keys (train / val
  data vector, parameter table, covariance matrix, failure masks) and
  the family sub-block files (grid axis files, syren baseline dumps,
  the CMB covariance), all defined next to the two key tuples at the
  top of this module.  The YAML lists bare filenames; an
  already-absolute path passes through unchanged, because
  ``os.path.join`` discards the earlier parts when a later part is
  absolute.  Resolving here rather than in the YAML lets the driver run
  from $ROOTDIR (the cocoa launch directory) without a
  current-directory-relative path breaking.  Whether a required block
  is present at all is checked later by the experiment construction,
  which gives a clearer error.

  Arguments:
    args = the parsed command-line namespace; reads args.root,
           args.fileroot, and args.yaml (the flags that
           add_cocoa_path_args registered).

  Returns:
    cfg      = the parsed config mapping, its data paths made absolute.
    fileroot = absolute emulator folder (<root>/<fileroot>), holding the
               YAML configs.
    chains   = absolute project chains/ folder (<root>/chains), holding
               the data files and this driver's run products.

  Raises:
    RuntimeError when $ROOTDIR is not set; FileNotFoundError when the
    resolved YAML path is not a file; ValueError when the YAML does not
    parse to a mapping.
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

  # the YAML lives under the emulator's fileroot; default test.yaml. An
  # absolute --yaml is read as-is (os.path.isabs), mirroring the data-path
  # rewrite below, where os.path.join lets an absolute path through
  # unchanged; this lets a caller (e.g. the gates harness) pass a
  # fully-resolved path from outside the fileroot.
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

  # rewrite each input data path to absolute, under the project chains/
  # folder, where dataset_generator_lensing.py writes the dvs / params /
  # covmat. The YAML lists bare filenames; os.path.join puts each under
  # root/chains (and passes an absolute path through unchanged). The
  # block-presence check is left to EmulatorExperiment.from_config, which
  # gives a clearer error.
  data = cfg.get("data")
  if isinstance(data, dict):
    for key in _DATA_PATH_KEYS:
      if key in data:
        data[key] = os.path.join(root, "chains", data[key])
    for block_name, keys in _NESTED_DATA_PATH_KEYS:
      block = data.get(block_name)
      if isinstance(block, dict):
        for key in keys:
          if key in block:
            block[key] = os.path.join(root, "chains", block[key])

  return cfg, fileroot, chains


def cocoa_output(base, path):
  """Place a run-output path under a base folder.

  Joins a relative output path under ``base`` (the chains folder for
  run products such as the diagnostics PDF, or the fileroot for
  configs).  An absolute path passes through unchanged, because
  ``os.path.join`` discards every earlier part when a later part is
  absolute: ``os.path.join("/a", "/b/c")`` is ``"/b/c"``.  A user can
  therefore redirect any single output by writing an absolute path in
  the configuration.

  Arguments:
    base = absolute folder to place the output in (fileroot or chains,
           both from resolve_cocoa_config).
    path = the output path, relative to base or absolute.

  Returns:
    the resolved output path.
  """
  return os.path.join(base, path)
