"""Save a finished training run to disk, and rebuild it later from disk alone.

A trained emulator outlives the process that trained it: a likelihood
run, a plotting script, or a warm-started retraining may reopen it
months later, on a different machine, after the code's defaults have
drifted. This module owns that round trip. The saved form is called an
artifact: a pair of files under one name holding everything needed to
rebuild the trained model exactly, so nothing about the run has to be
remembered, re-derived, or guessed.

save_emulator writes the pair. <root>.emul holds the model weights (a
torch state_dict of cpu tensors) beside one fresh random pair token;
<root>.h5 holds everything else -- both whitening geometries, the
training histories, the full config, and the scientific record the
dataset was born under, copied unchanged from the generator's sidecar
-- and carries the same token, so the two files can prove they were
saved together. The .h5 is an HDF5 file: HDF5 is a portable binary
container that stores named arrays ("datasets") inside nested folders
("groups"), with small named values ("attributes") attached to either;
h5py is its Python interface. rebuild_emulator reads the pair back into
a ready model, under one rule: every fact comes from the file, and a
missing fact is a loud error, never a fallback to a code default.

save_learning_curves and save_sweep_table write the small text tables
the sweep and bake-off drivers save (one row per point, "#"-comment
header carrying the config), in a form np.loadtxt reads back, so
several runs can be overlaid on one figure later.

read_artifact_schema is the one reader of a saved file's schema version
-- the integer stamped in the file that names its exact layout, which
groups and keys exist and what they mean -- and of the scientific
record. Every path that opens a saved emulator goes through it: the
rebuild path here, and the warm-start loader that fine-tunes one
(emulator/warmstart.py). One reader is the whole point. Two readers of
one schema is how a file gets refused on one path and quietly accepted
on the other, and the accepted copy is the one that answers a
likelihood.

PS: ``state_dict`` is PyTorch's name-to-tensor mapping for every registered
parameter, including frozen parameters, and every persistent registered
buffer. It is not the list of tensors that an optimizer updates. Whitening is
the center, rotation, and scaling transform applied by the geometries. The
scientific record is the pair of blocks defined in emulator/fixed_facts.py:
the cosmology held fixed while the dataset was generated, and the parameter
region it was sampled over. A sidecar is a small companion file written next
to a data file and sharing its name stem. Values called provenance are a
paper trail: stored so a person can reconstruct how a run was made, never
read back by the code.
"""

import contextlib
import os
import stat
import subprocess
import tempfile
import time
import uuid
import zipfile

import numpy as np
import torch
import yaml

from . import fixed_facts
from .model_recipe import check_model_matches_recipe
from .model_recipe import set_runtime_compile_mode
from .model_recipe import validate_model_recipe


_GRID2D_CLASS = "emulator.geometries.grid2d.Grid2DGeometry"
_COMPOSITION_MODE_ATTR = "composition_mode"
_TRANSFER_REFINED_ATTR = "transfer_refined"
_COMPOSITION_MODES = ("plain", "npce", "transfer")
_PCE_FORMS = ("residual", "ratio")
_TRANSFER_FORMS = ("gain", "sum")
_TRANSFER_SPACES = ("physical", "whitened")


def _validate_live_recipe_geometry_widths(
    recipe, param_geometry, output_geometry, where):
  """Bind a live recipe's two network widths to its live geometries.

  A recipe that claims one width while the geometries carry another would
  save an artifact whose rebuilt model cannot consume its own encode() or
  produce its own decode() shape. Both widths are read off the live
  geometry objects (encoded_dim / names for the input, dest_idx for the
  output) and compared against the recipe before anything is written.

  Arguments:
    recipe          = the validated model recipe about to be saved.
    param_geometry  = the live input geometry (ParamGeometry family).
    output_geometry = the live output geometry (its dest_idx count is
                      the model's output width).
    where           = the save location, named in every refusal.

  Returns:
    None. The function is called for its refusals.

  Raises:
    ValueError when either recipe width disagrees with its geometry, or
    a geometry exposes no width at all.
  """
  if hasattr(param_geometry, "encoded_dim"):
    input_width = int(param_geometry.encoded_dim)
  elif hasattr(param_geometry, "names"):
    input_width = len(param_geometry.names)
  else:
    raise ValueError(where + " parameter geometry exposes no encoded width")
  if not hasattr(output_geometry, "dest_idx"):
    raise ValueError(where + " output geometry exposes no destination width")
  output_width = int(output_geometry.dest_idx.numel())
  if recipe["input_dim"] != input_width:
    raise ValueError(
      where + " model_recipe.input_dim=" + str(recipe["input_dim"])
      + " disagrees with the parameter geometry encoded width "
      + str(input_width))
  if recipe["output_dim"] != output_width:
    raise ValueError(
      where + " model_recipe.output_dim=" + str(recipe["output_dim"])
      + " disagrees with the output geometry destination width "
      + str(output_width))


def _validate_saved_recipe_geometry_widths(
    recipe, param_group, output_group, where):
  """Check saved geometry widths while both HDF5 groups remain inert data.

  Neither side has one universal persisted width field.  A factored input
  geometry derives its encoded width from a nested whitening basis plus its
  amplitude indices.  An output geometry may derive its width from names,
  kept positions, or grid axes.

  Scalar geometries derive it from ``names``; masked data-vector geometries
  derive it from ``dest_idx``; the grid families derive it from their axes.
  Check those class-specific facts and every same-width transform array
  before importing a geometry class.  Using ``center`` alone would miss
  corruption such as two scalar names beside one center, or three kept
  data-vector positions beside a four-output model.

  Arguments:
    recipe       = the validated model recipe (its input_dim / output_dim
                   are the widths every saved array must agree with).
    param_group  = the artifact's open parameter-geometry HDF5 group.
    output_group = the artifact's open output-geometry HDF5 group.
    where        = the artifact's identity, named in every refusal.

  Returns:
    None. The function is called for its refusals.

  Raises:
    KeyError / TypeError / ValueError naming the dataset whose shape,
    type, or cross-field consistency disagrees with the recipe.
  """
  def saved_shape(group, name, label):
    """Read one dataset's shape from an HDF5 group, as plain ints.

    Arguments:
      group = the open HDF5 group.
      name  = the dataset name inside the group.
      label = which geometry the dataset belongs to, for refusals.

    Returns:
      the dataset's shape tuple.
    """
    if name not in group:
      raise KeyError(where + " " + label + " is missing " + repr(name))
    item = group[name]
    if not hasattr(item, "shape"):
      raise TypeError(
        where + " " + label + " " + repr(name) + " must be a dataset")
    return tuple(int(size) for size in item.shape)

  def require_parameter_vector(group, name, width, label):
    """Require one saved parameter-side dataset to be a (width,) vector.

    Arguments:
      group = the open HDF5 group.
      name  = the dataset name inside the group.
      width = the required length (the recipe's input width).
      label = which geometry the dataset belongs to, for refusals.
    """
    observed = saved_shape(group, name, label)
    expected = (width,)
    if observed != expected:
      raise ValueError(
        where + " " + label + " " + repr(name) + " must have shape "
        + repr(expected) + ", got " + repr(observed)
        + "; it must agree with model_recipe.input_dim="
        + str(recipe["input_dim"]))

  def require_parameter_matrix(group, name, width, label):
    """Require one saved whitening basis to be a (width, width) square.

    Arguments:
      group = the open HDF5 group.
      name  = the dataset name inside the group.
      width = the required side length (the recipe's input width).
      label = which geometry the dataset belongs to, for refusals.
    """
    observed = saved_shape(group, name, label)
    expected = (width, width)
    if observed != expected:
      raise ValueError(
        where + " " + label + " " + repr(name) + " must have shape "
        + repr(expected) + ", got " + repr(observed)
        + "; it must be a square whitening basis of width " + str(width))

  parameter_class = _saved_geometry_class(
    param_group, where + " parameter geometry")
  input_width = recipe["input_dim"]
  ordinary_parameters = {
    "emulator.geometries.parameter.ParamGeometry",
    "emulator.geometries.parameter.LogParamGeometry",
  }
  amplitude_parameters = (
    "emulator.geometries.parameter.AmplitudeFactorGeometry")
  if parameter_class in ordinary_parameters:
    for name in ("names", "center", "sqrt_ev"):
      require_parameter_vector(
        param_group, name, input_width, "parameter geometry")
    require_parameter_matrix(
      param_group, "evecs", input_width, "parameter geometry")
    if parameter_class.endswith(".LogParamGeometry"):
      require_parameter_vector(
        param_group, "log_mask", input_width, "parameter geometry")
  elif parameter_class == amplitude_parameters:
    if "n_param" not in param_group.attrs:
      raise KeyError(
        where + " parameter geometry is missing required 'n_param'")
    n_param_value = param_group.attrs["n_param"]
    if isinstance(n_param_value, (bool, np.bool_)) \
        or not isinstance(n_param_value, (int, np.integer)) \
        or int(n_param_value) < 1:
      raise TypeError(
        where + " parameter geometry 'n_param' must be a positive integer")
    n_param = int(n_param_value)
    require_parameter_vector(
      param_group, "names", n_param, "parameter geometry")
    amp_shape = saved_shape(param_group, "amp_idx", "parameter geometry")
    if len(amp_shape) != 1:
      raise ValueError(
        where + " parameter geometry 'amp_idx' must be one-dimensional; got "
        + repr(amp_shape))
    amp_values = np.asarray(param_group["amp_idx"][()])
    if amp_values.dtype.kind not in "iu":
      raise TypeError(
        where + " parameter geometry 'amp_idx' must contain integers")
    amp_indices = [int(value) for value in amp_values.tolist()]
    declared_amps = recipe["kwargs"].get("n_amps")
    if declared_amps is None:
      raise ValueError(
        where + " AmplitudeFactorGeometry requires a factored model recipe "
        "with explicit n_amps")
    if len(amp_indices) != declared_amps:
      raise ValueError(
        where + " parameter geometry 'amp_idx' has "
        + str(len(amp_indices)) + " column(s), but model_recipe declares "
        + "n_amps=" + str(declared_amps))
    if len(set(amp_indices)) != len(amp_indices):
      raise ValueError(
        where + " parameter geometry 'amp_idx' contains duplicate columns")
    if any(index < 0 or index >= n_param for index in amp_indices):
      raise ValueError(
        where + " parameter geometry 'amp_idx' contains an index outside "
        "0.." + str(n_param - 1))
    if "pg_keep" not in param_group \
        or not hasattr(param_group["pg_keep"], "keys"):
      raise TypeError(
        where + " parameter geometry 'pg_keep' must be a saved group")
    kept = param_group["pg_keep"]
    kept_width = n_param - len(amp_indices)
    for name in ("names", "center", "sqrt_ev"):
      require_parameter_vector(
        kept, name, kept_width, "parameter geometry pg_keep")
    require_parameter_matrix(
      kept, "evecs", kept_width, "parameter geometry pg_keep")
    encoded_width = kept_width + len(amp_indices)
    if input_width != n_param or input_width != encoded_width:
      raise ValueError(
        where + " model_recipe.input_dim=" + str(input_width)
        + " disagrees with AmplitudeFactorGeometry encoded width "
        + str(encoded_width) + " and n_param=" + str(n_param))
  else:
    raise ValueError(
      where + " parameter geometry class " + repr(parameter_class)
      + " is not a class with known saved-width rules, so its shapes "
      "cannot be checked before the class is imported")

  def dataset_shape(name):
    """Read one output-geometry dataset's shape as plain ints.

    A closure over saved_shape with the group and label fixed, so the
    output-side checks below stay one short line each.

    Arguments:
      name = the dataset name inside the output-geometry group.

    Returns:
      the dataset's shape tuple.
    """
    return saved_shape(output_group, name, "output geometry")

  def require_vector(name, width):
    """Require one output-side dataset to be a (width,) vector.

    Arguments:
      name  = the dataset name inside the output-geometry group.
      width = the required length (the recipe's output width).
    """
    observed = dataset_shape(name)
    expected = (width,)
    if observed != expected:
      raise ValueError(
        where + " output geometry " + repr(name) + " must have shape "
        + repr(expected) + ", got " + repr(observed)
        + "; it must agree with model_recipe.output_dim="
        + str(recipe["output_dim"]))

  def require_matrix(name, width):
    """Require one output-side dataset to be a (width, width) square.

    Arguments:
      name  = the dataset name inside the output-geometry group.
      width = the required side length (the recipe's output width).
    """
    observed = dataset_shape(name)
    expected = (width, width)
    if observed != expected:
      raise ValueError(
        where + " output geometry " + repr(name) + " must have shape "
        + repr(expected) + ", got " + repr(observed)
        + "; it must agree with model_recipe.output_dim="
        + str(recipe["output_dim"]))

  class_path = _saved_geometry_class(
    output_group, where + " output geometry")
  output_width = recipe["output_dim"]
  scalar = "emulator.geometries.scalar.ScalarGeometry"
  data_vectors = {
    "emulator.geometries.output.DataVectorGeometry",
    "emulator.geometries.output.DiagonalGeometry",
    "emulator.geometries.output.BlockDiagonalGeometry",
  }
  cmb = "emulator.geometries.cmb.CmbDiagonalGeometry"
  grid = "emulator.geometries.grid.GridGeometry"
  grid2d = "emulator.geometries.grid2d.Grid2DGeometry"

  if class_path == scalar:
    for name in ("names", "center", "scale"):
      require_vector(name, output_width)
  elif class_path in data_vectors:
    for name in ("dest_idx", "center", "sqrt_ev"):
      require_vector(name, output_width)
    require_matrix("evecs", output_width)
  elif class_path == cmb:
    for name in ("ell", "center", "sigma", "fiducial_cl"):
      require_vector(name, output_width)
  elif class_path == grid:
    for name in ("z", "center", "scale"):
      require_vector(name, output_width)
  elif class_path == grid2d:
    z_shape = dataset_shape("z")
    k_shape = dataset_shape("k")
    if len(z_shape) != 1 or len(k_shape) != 1:
      raise ValueError(
        where + " Grid2DGeometry axes 'z' and 'k' must be one-dimensional; "
        "got " + repr(z_shape) + " and " + repr(k_shape))
    axis_width = z_shape[0] * k_shape[0]
    if axis_width != output_width:
      raise ValueError(
        where + " model_recipe.output_dim=" + str(output_width)
        + " disagrees with the saved Grid2DGeometry z*k width "
        + str(z_shape[0]) + "*" + str(k_shape[0]) + "="
        + str(axis_width))
    for name in ("center", "scale", "const_mask"):
      require_vector(name, output_width)
  else:
    raise ValueError(
      where + " output geometry class " + repr(class_path)
      + " is not a class with known saved-width rules, so its shapes "
      "cannot be checked before the class is imported")


def _load_tensor_state_dict(checkpoint, *, device, where, expected_token):
  """Load one checkpoint's state dict and prove it belongs to its record.

  The read runs under ``weights_only=True``, PyTorch's restricted-pickle
  mode: a checkpoint whose bytes were rewritten to smuggle executable
  pickle payloads fails to load instead of running them. The loaded
  object must be the container save_emulator writes -- a plain dict with
  exactly the keys ``pair_token`` and ``state_dict`` -- and its token
  must equal the one the ``.h5`` record carries, so a ``.emul`` mixed in
  from a different run refuses before any model is constructed. The
  inner state dict must then LOOK like a model state: exactly a plain
  dict, nonempty, string keys, tensor values.

  Arguments:
    checkpoint     = the open binary file object of ``<root>.emul``; its
                     position is rewound before and after the read.
    device         = the torch device the restored tensors are mapped
                     onto (the ``map_location``), so a GPU-trained model
                     loads on a CPU-only machine.
    where          = the checkpoint's identity, named in every refusal.
    expected_token = the ``pair_token`` string read off the ``.h5``
                     record this checkpoint claims to belong to.

  Returns:
    the state dict: a plain dict mapping parameter names to tensors on
    ``device``.

  Raises:
    ValueError when the bytes are not a tensor-only checkpoint, the
    container or its token is missing or mismatched, the mapping is
    empty or not a plain dict, a key is not a nonempty string, or a
    value is not a tensor.
  """
  checkpoint.seek(0)
  try:
    state = torch.load(
      checkpoint, map_location=device, weights_only=True)
  except Exception as exc:
    raise ValueError(
      where + " cannot be read as a tensor-only PyTorch state dict") from exc
  finally:
    checkpoint.seek(0)
  if type(state) is not dict or set(state) != {"pair_token", "state_dict"}:
    raise ValueError(
      where + " does not hold the {'pair_token', 'state_dict'} container "
      "save_emulator writes. A checkpoint saved before the pair token "
      "existed must be re-saved (retrain, or re-run save_emulator on the "
      "run) before it can be rebuilt.")
  found_token = state["pair_token"]
  if type(found_token) is not str or not found_token:
    raise ValueError(
      where + " pair_token must be a nonempty string; it read as "
      + repr(found_token))
  if found_token != expected_token:
    raise ValueError(
      where + " and its .h5 record are not from the same save: the .h5 "
      "carries pair_token " + repr(expected_token) + " but the checkpoint "
      "carries " + repr(found_token) + ". The two files were mixed from "
      "different runs; restore the matching pair, or retrain.")
  state = state["state_dict"]
  if type(state) is not dict or not state:
    raise ValueError(
      where + " must contain one nonempty plain state-dict mapping")
  bad_keys = [key for key in state if type(key) is not str or not key]
  if bad_keys:
    raise ValueError(
      where + " state-dict keys must be nonempty native strings; got "
      + repr(bad_keys[:3]))
  bad_values = [key for key, value in state.items()
                if not torch.is_tensor(value)]
  if bad_values:
    raise ValueError(
      where + " state-dict values must all be tensors; non-tensor value(s) "
      "appear at " + repr(bad_values[:3]))
  return state


def _refuse_existing_artifact_root(path_root):
  """Refuse to save over any file already using this artifact name.

  A completed emulator name is immutable: overwriting even ONE member of
  the ``.emul`` / ``.h5`` pair would leave a mixed artifact whose halves
  came from different runs. ``os.path.lexists`` also counts a dangling
  symbolic link as occupation, so a link cannot redirect the write.

  Arguments:
    path_root = the artifact path root (the shared path before the two
                extensions).

  Returns:
    None when both final names are free.

  Raises:
    FileExistsError listing the occupied path(s) and telling the user to
    choose another --save name or move the earlier result first.
  """
  root = os.fspath(path_root)
  occupied = [path for path in (root + ".emul", root + ".h5")
              if os.path.lexists(path)]
  if occupied:
    raise FileExistsError(
      os.fspath(path_root) + " is already occupied by " + repr(occupied)
      + ". A training run never replaces an existing emulator. Choose a "
      "different --save name, or move the complete earlier result first.")


def _unlink_if_present(path):
  """Remove one temporary file, treating an already-absent file as done.

  Cleanup after a failed save must not raise a second error that masks
  the first: a temporary that was never created, or was already removed,
  is simply skipped.

  Arguments:
    path = the temporary file path to remove.
  """
  try:
    os.unlink(path)
  except FileNotFoundError:
    pass


@contextlib.contextmanager
def _tmp_h5_file(h5py, h5_path, cleanup_paths):
  """Open a temporary HDF5 file whose failure removes every temporary.

  The save writes both members to temporary names first. If anything
  inside the HDF5 write raises, every temporary listed is deleted before
  the error propagates, so a failed save leaves no partial file behind
  -- neither public names nor stray ``.tmp`` files.

  Arguments:
    h5py          = the imported h5py module (imported lazily by the
                    caller, so this module never imports it at top level).
    h5_path       = the temporary HDF5 file path to create and write.
    cleanup_paths = every temporary path to delete on failure (both the
                    HDF5 temporary and the already-written checkpoint
                    temporary).

  Returns:
    a context manager yielding the open h5py.File handle.
  """
  try:
    with h5py.File(h5_path, "w") as handle:
      yield handle
  except BaseException:
    for path in cleanup_paths:
      _unlink_if_present(path)
    raise


def _open_regular_checkpoint(path):
  """Open one checkpoint file, refusing links and special files.

  The file is opened ONCE and every later read uses the returned handle,
  so the bytes that were validated are the bytes that are loaded --
  nothing can swap the pathname between two opens. ``O_NOFOLLOW`` makes a
  symbolic link at the final path component fail the open, and the
  ``S_ISREG`` check refuses directories, pipes, and device files.

  Arguments:
    path = the ``<root>.emul`` checkpoint path.

  Returns:
    an open binary file object positioned at the start.

  Raises:
    ValueError when the path cannot be opened or is not a regular file.
  """
  flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags)
  except OSError as exc:
    raise ValueError(path + " cannot be opened as a saved checkpoint") from exc
  try:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
      raise ValueError(path + " is not a regular checkpoint file")
    return os.fdopen(descriptor, "rb")
  except BaseException:
    os.close(descriptor)
    raise


def save_learning_curves(path, sizes, curves, meta=None):
  """
  Write learning curve(s) as a whitespace-delimited text table.

  A learning curve is the validation score as a function of the
  training-set size N_train: it shows whether more data would still
  help. The score is f(delta-chi2 > threshold), the fraction of
  validation points whose emulated data vector misses the exact one
  by more than the chi2 threshold -- smaller is better.

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

  The generic twin of save_learning_curves for a hyperparameter
  sweep: the same emulator trained once per candidate value of one
  training setting, every run scored the same way. Numeric values
  become the first data column; categorical values (strings, or a
  True/False switch) become an integer index column with the label
  map on a "# values:" comment line, so the body stays
  np.loadtxt-loadable either way. Layouts:

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


def executed_composition(pce, transfer_base):
  """Derive the composition declaration from what one run actually built.

  The composition names how the served prediction is assembled from
  parts. A saved emulator is one of three compositions: "plain" (the
  network alone), "npce" (a frozen polynomial-chaos base plus a refiner
  network; NPCE = Neural PCE, see emulator/designs/pce.py),
  or "transfer" (a frozen source emulator plus a correction network,
  "refined" when a drifted copy of the base weights rides along). The
  writer declares the mode from the LIVE objects the trainer is about to
  persist -- never from what groups happen to exist in a file -- and
  readers later trust that declaration alone.

  Arguments:
    pce           = the fitted PCEEmulator base of an NPCE run, else None.
    transfer_base = the embedded-base payload mapping of a transfer run
                    (its "drifted_state" key marks a refined run), else
                    None.

  Returns:
    the pair (composition_mode, transfer_refined): one of
    ("plain", False), ("npce", False), ("transfer", <bool>).

  Raises:
    ValueError when both bases are supplied (no run can be both);
    TypeError when transfer_base is not a mapping.
  """
  if pce is not None and transfer_base is not None:
    raise ValueError(
      "an executed run cannot carry both an NPCE base and a transfer base")
  if transfer_base is not None:
    if not isinstance(transfer_base, dict):
      raise TypeError("transfer_base must be a mapping when transfer is on")
    refined = transfer_base.get("drifted_state") is not None
    return "transfer", refined
  if pce is not None:
    return "npce", False
  return "plain", False


def _validate_executed_composition(
    *, composition_mode, transfer_refined, pce, pce_form, transfer_base,
    resolved_pce, resolved_transfer, where):
  """Cross-check the composition declaration against every payload.

  The declaration (mode + refined flag) and the payloads (the pce base,
  the transfer base, the resolved records) all describe the same run, so
  every pair must corroborate: a "plain" declaration with a pce payload,
  an "npce" declaration whose resolved record names a different form, or
  a refined flag without a drifted state are each a writer bug that would
  persist a self-contradictory artifact. Validation runs before either
  output file is touched, so a contradiction costs nothing on disk.

  Arguments:
    composition_mode  = the declared mode ("plain" / "npce" / "transfer").
    transfer_refined  = the declared refined flag (native bool).
    pce               = the live PCEEmulator base, else None.
    pce_form          = the NPCE recombination form string, else None.
    transfer_base     = the embedded-base payload mapping, else None.
    resolved_pce      = the run's resolved NPCE record mapping, else None.
    resolved_transfer = the run's resolved transfer record, else None.
    where             = the save location, named in every refusal.

  Returns:
    the corroborated (composition_mode, transfer_refined) pair.

  Raises:
    ValueError naming the disagreeing pair of surfaces.
  """
  if type(composition_mode) is not str or composition_mode not in \
      _COMPOSITION_MODES:
    raise ValueError(
      where + ": composition_mode must be one native string in "
      + repr(_COMPOSITION_MODES) + ", got " + repr(composition_mode))
  if type(transfer_refined) is not bool:
    raise ValueError(
      where + ": transfer_refined must be a native boolean, got "
      + repr(transfer_refined))

  derived_mode, derived_refined = executed_composition(
    pce=pce, transfer_base=transfer_base)
  if composition_mode != derived_mode:
    raise ValueError(
      where + ": composition_mode=" + repr(composition_mode)
      + " disagrees with the executed payload, which is "
      + repr(derived_mode))
  if transfer_refined != derived_refined:
    raise ValueError(
      where + ": transfer_refined=" + repr(transfer_refined)
      + " disagrees with transfer_base/drifted_state="
      + repr(derived_refined))

  if composition_mode == "plain":
    if pce_form is not None:
      raise ValueError(where + ": plain composition forbids pce_form")
    if resolved_pce is not None or resolved_transfer is not None:
      raise ValueError(
        where + ": plain composition forbids resolved pce/transfer records")
  elif composition_mode == "npce":
    if type(pce_form) is not str or pce_form not in _PCE_FORMS:
      raise ValueError(
        where + ": npce composition requires one native pce_form in "
        + repr(_PCE_FORMS))
    if not isinstance(resolved_pce, dict) or resolved_transfer is not None:
      raise ValueError(
        where + ": npce composition requires resolved_pce and forbids "
        "resolved_transfer")
    resolved_pce_form = resolved_pce.get("form")
    if type(resolved_pce_form) is not str \
        or resolved_pce_form not in _PCE_FORMS:
      raise ValueError(
        where + ": resolved_pce.form must be one native string in "
        + repr(_PCE_FORMS))
    if resolved_pce_form != pce_form:
      raise ValueError(
        where + ": resolved_pce.form disagrees with the persisted pce form")
  else:
    if pce_form is not None:
      raise ValueError(where + ": transfer composition forbids pce_form")
    if resolved_pce is not None or not isinstance(resolved_transfer, dict):
      raise ValueError(
        where + ": transfer composition requires resolved_transfer and "
        "forbids resolved_pce")
    transfer_form = transfer_base.get("form")
    if type(transfer_form) is not str \
        or transfer_form not in _TRANSFER_FORMS:
      raise ValueError(
        where + ": transfer_base.form must be one native string in "
        + repr(_TRANSFER_FORMS))
    transfer_space = transfer_base.get("space")
    if type(transfer_space) is not str \
        or transfer_space not in _TRANSFER_SPACES:
      raise ValueError(
        where + ": transfer_base.space must be one native string in "
        + repr(_TRANSFER_SPACES))
    resolved_transfer_form = resolved_transfer.get("form")
    if (type(resolved_transfer_form) is not str
        or resolved_transfer_form not in _TRANSFER_FORMS):
      raise ValueError(
        where + ": resolved_transfer.form must be one native string in "
        + repr(_TRANSFER_FORMS))
    resolved_transfer_space = resolved_transfer.get("space")
    if (type(resolved_transfer_space) is not str
        or resolved_transfer_space not in _TRANSFER_SPACES):
      raise ValueError(
        where + ": resolved_transfer.space must be one native string in "
        + repr(_TRANSFER_SPACES))
    if resolved_transfer_form != transfer_form:
      raise ValueError(
        where + ": resolved_transfer.form disagrees with transfer_base.form")
    if resolved_transfer_space != transfer_space:
      raise ValueError(
        where + ": resolved_transfer.space disagrees with "
        "transfer_base.space")
    resolved_refine = resolved_transfer.get("refine")
    if resolved_refine is not None and type(resolved_refine) is not dict:
      raise ValueError(
        where + ": resolved_transfer.refine must be a mapping when present")
    resolved_refined = resolved_refine is not None
    if resolved_refined != transfer_refined:
      raise ValueError(
        where + ": resolved_transfer.refine presence disagrees with "
        "transfer_refined=" + repr(transfer_refined))

  return composition_mode, transfer_refined


_HISTORY_KEYS = (
  "train_losses", "val_medians", "val_means", "val_fracs", "thresholds",
)


def _history_arrays_for_save(histories, where):
  """Convert the five training-history curves to compatible saved arrays.

  The history block records the run's per-epoch curves: train_losses,
  val_medians, val_means (each one value per epoch), thresholds (the
  delta-chi2 thresholds the run scored against), and val_fracs (one row
  per epoch, one column per threshold). The five are converted to plain
  numpy arrays, checked finite, and checked mutually consistent -- equal
  epoch counts, and a val_fracs column count equal to the threshold
  count -- before anything is written.

  Arguments:
    histories = the history mapping with exactly the five keys above
                (tensors or arrays).
    where     = the save location, named in every refusal.

  Returns:
    a mapping of the five names to plain numpy arrays, shapes verified.

  Raises:
    TypeError / ValueError naming the missing, unknown, nonfinite, or
    shape-inconsistent member.
  """
  if type(histories) is not dict:
    raise TypeError(where + " must be a plain mapping")
  missing = sorted(set(_HISTORY_KEYS) - set(histories))
  unknown = sorted(set(histories) - set(_HISTORY_KEYS))
  if missing or unknown:
    details = []
    if missing:
      details.append("missing " + repr(missing))
    if unknown:
      details.append("unknown " + repr(unknown))
    raise ValueError(where + " has " + " and ".join(details))

  def as_finite_array(value, label):
    """Convert one curve to a plain numpy array, refusing nonfinite values.

    Arguments:
      value = the curve (a tensor, array, or sequence of numbers).
      label = the history key, named in the refusal.

    Returns:
      the curve as a numpy array.
    """
    if torch.is_tensor(value):
      value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) \
        or not bool(np.all(np.isfinite(array))):
      raise ValueError(where + "." + label + " must contain finite numbers")
    return array

  arrays = {
    name: as_finite_array(histories[name], name)
    for name in ("train_losses", "val_medians", "val_means", "thresholds")
  }
  curves = ("train_losses", "val_medians", "val_means")
  for name in curves:
    if arrays[name].ndim != 1:
      raise ValueError(where + "." + name + " must be one-dimensional")
  epochs = arrays["train_losses"].size
  if epochs < 1 or any(arrays[name].size != epochs for name in curves):
    raise ValueError(where + " curves must have one common nonzero length")
  if arrays["thresholds"].ndim != 1 or arrays["thresholds"].size < 1:
    raise ValueError(where + ".thresholds must be one-dimensional and nonempty")

  rows = histories["val_fracs"]
  if not isinstance(rows, (list, tuple)) or not rows:
    raise ValueError(where + ".val_fracs must contain one row per epoch")
  fraction_rows = [as_finite_array(row, "val_fracs") for row in rows]
  try:
    arrays["val_fracs"] = np.stack(fraction_rows)
  except ValueError as exc:
    raise ValueError(
      where + ".val_fracs rows must have one common threshold width") from exc
  expected = (epochs, arrays["thresholds"].size)
  if arrays["val_fracs"].shape != expected:
    raise ValueError(
      where + ".val_fracs must have shape " + repr(expected)
      + ", got " + repr(arrays["val_fracs"].shape))
  return arrays


def _read_yaml_mapping_dataset(container, name, where):
  """Read one stored YAML text dataset and parse it into a plain mapping.

  The artifact stores its configuration blocks as YAML TEXT (never
  pickled objects), so reading one is: fetch the scalar dataset, decode
  the bytes as UTF-8, parse with the safe YAML loader (which constructs
  only plain Python values, never classes), and require a mapping at the
  top level.

  Arguments:
    container = the open HDF5 file or group holding the dataset.
    name      = the dataset name (for example "config_yaml").
    where     = the artifact's identity, named in every refusal.

  Returns:
    the parsed plain dict.

  Raises:
    KeyError when the dataset is absent; ValueError for invalid UTF-8 or
    invalid YAML; TypeError when the stored value is not scalar text or
    does not parse to a plain mapping.
  """
  if name not in container:
    raise KeyError(where + " is missing required " + repr(name))
  value = container[name][()]
  if isinstance(value, bytes):
    try:
      value = value.decode("utf-8")
    except UnicodeDecodeError as exc:
      raise ValueError(where + " " + name + " is not valid UTF-8") from exc
  if type(value) is not str:
    raise TypeError(where + " " + name + " must be scalar native text")
  try:
    parsed = yaml.safe_load(value)
  except yaml.YAMLError as exc:
    raise ValueError(where + " " + name + " is invalid YAML") from exc
  if type(parsed) is not dict:
    raise TypeError(where + " " + name + " must decode to a plain mapping")
  return parsed


def _saved_geometry_class(group, where):
  """Read a saved geometry's class marker as plain text.

  Every persisted geometry group carries a "cls" attribute naming the
  class that wrote it (for example
  "emulator.geometries.parameter.ParamGeometry"). Reading the marker is
  deliberately separated from importing the class, so shape validation
  can run while the file is still nothing but inert data.

  Arguments:
    group = the open HDF5 geometry group.
    where = which geometry is being read, named in every refusal.

  Returns:
    the class path string.

  Raises:
    KeyError when the marker is absent; TypeError when it is empty or
    not native text.
  """
  if not hasattr(group, "attrs") or "cls" not in group.attrs:
    raise KeyError(where + " is missing the required 'cls' class marker")
  value = group.attrs["cls"]
  if type(value) is not str or not value:
    raise TypeError(where + " 'cls' must be nonempty native text")
  return value


def _saved_head_layout(geometry, recipe, where):
  """Derive the fixed structured-head buffers promised by one recipe.

  A structured head (conv or transformer) carries two fixed buffers that
  map the trunk's flat output into its padded rectangle: ``pad_idx`` (the
  slot of each physical value) and ``pad_valid`` (the mask of physical
  slots). This helper recomputes both from the geometry and recipe alone,
  so the saver can compare them against the live model's buffers.

  Most heads use the geometry's physical rectangle directly. Template
  heads repeat its validity mask once per template. ``ResTRF.n_tokens``
  is the one supported transformation: it divides a complete
  one-dimensional grid into contiguous, near-equal tokens. Keeping that
  transformation here prevents a valid segmented Transformer from being
  mistaken for a geometry mismatch at save time.

  Arguments:
    geometry = the live output geometry (attach_head_coords is invoked
               when the class provides it).
    recipe   = the validated model recipe (its cls / kwargs select the
               transformation, its output_dim fixes the value count).
    where    = the save location, named in every refusal.

  Returns:
    the pair (pad_idx, pad_valid) as CPU tensors: a (output_dim,) long
    index map and a (1, tokens-or-bins, width) Boolean mask.

  Raises:
    ValueError when n_tokens is out of range, re-segments anything but a
    complete one-dimensional grid, or a template head lacks a positive
    n_templates.
  """
  from .designs.blocks import resolve_padded_head_layout

  if hasattr(geometry, "attach_head_coords"):
    geometry.attach_head_coords()
  output_dim = recipe["output_dim"]
  _, pad_idx, pad_valid = resolve_padded_head_layout(
    geom=geometry, output_dim=output_dim, where=where)
  cls_path = recipe["cls"]
  kwargs = recipe["kwargs"]

  if cls_path == "emulator.designs.plain.ResTRF" \
      and kwargs.get("n_tokens") is not None:
    n_tokens = kwargs["n_tokens"]
    if type(n_tokens) is not int or n_tokens < 2 or n_tokens > output_dim:
      raise ValueError(
        where + ": resolved ResTRF.n_tokens must be in 2.."
        + str(output_dim) + "; got " + repr(n_tokens))
    identity = torch.arange(
      output_dim, dtype=torch.long, device=pad_idx.device)
    if pad_valid.shape[1] != 1 or not bool(torch.all(pad_valid).item()) \
        or not torch.equal(pad_idx, identity):
      raise ValueError(
        where + ": resolved ResTRF.n_tokens can re-segment only one complete "
        "one-dimensional geometry")
    quotient, remainder = divmod(output_dim, n_tokens)
    sizes = []
    for token_index in range(n_tokens):
      sizes.append(quotient + 1 if token_index < remainder else quotient)
    width = max(sizes)
    positions = []
    for token_index, size in enumerate(sizes):
      for coordinate_index in range(size):
        positions.append(token_index * width + coordinate_index)
    pad_idx = torch.tensor(
      positions, dtype=torch.long, device=pad_idx.device)
    pad_valid = torch.zeros(
      (1, n_tokens, width), dtype=torch.bool, device=pad_valid.device)
    pad_valid.reshape(-1)[pad_idx] = True

  template_classes = (
    "emulator.designs.ia.TemplateResCNN",
    "emulator.designs.ia.TemplateResTRF",
  )
  if cls_path in template_classes:
    n_templates = kwargs.get("n_templates")
    if type(n_templates) is not int or n_templates < 1:
      raise ValueError(
        where + ": resolved template head needs a positive native "
        "n_templates value")
    pad_valid = pad_valid.repeat(1, n_templates, 1)
  return pad_idx.detach().cpu(), pad_valid.detach().cpu()


def _validate_saved_head_layout(model_state, geometry, recipe, where):
  """Refuse a model/geometry pair whose head buffers disagree.

  The model's state dict carries its own ``pad_idx`` / ``pad_valid``
  buffers, and the geometry+recipe derive an independent expectation of
  both. If the two disagree, the artifact would save cleanly and then
  fail (or worse, scatter values into wrong slots) on rebuild -- so the
  disagreement is refused BEFORE anything is written.

  Arguments:
    model_state = the model's state dict about to be saved.
    geometry    = the live output geometry.
    recipe      = the validated model recipe; a trunk-only recipe
                  (needs_geom False) has no head buffers and passes.
    where       = the save location, named in every refusal.

  Returns:
    None. The function is called for its refusals.

  Raises:
    KeyError when a head buffer is missing from the state; ValueError
    when a buffer's dtype, shape, or values differ from the expectation.
  """
  if not recipe["needs_geom"]:
    return
  expected_idx, expected_valid = _saved_head_layout(
    geometry=geometry, recipe=recipe, where=where)
  expected = {"pad_idx": expected_idx, "pad_valid": expected_valid}
  for name in ("pad_idx", "pad_valid"):
    if name not in model_state:
      raise KeyError(
        where + ": structured-head model state has no " + name
        + " buffer, so save_emulator would publish an artifact that cannot "
        "be reopened")
    recorded = model_state[name]
    if recorded.dtype != expected[name].dtype \
        or recorded.shape != expected[name].shape \
        or not torch.equal(recorded, expected[name]):
      raise ValueError(
        where + ": structured-head " + name
        + " disagrees between the model state and output geometry; "
        "save_emulator refuses the pair before changing any files")


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
                  facts_yaml=None,
                  *,
                  composition_mode,
                  transfer_refined,
                  resolved_pce,
                  resolved_transfer,
                  resolved_rescale=None):
  """
  Persist a trained emulator as <path_root>.emul + <path_root>.h5.

  One finished run becomes one artifact under one name: the weights
  travel in the torch-native .emul, every other fact travels in the
  open HDF5 .h5, and the pair can then be reopened months later, on
  another machine, after code defaults have drifted -- from the files
  alone. Everything below serves that one goal.

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
                     no covariance-matrix (covmat) reread.
    dv_geometry/     the output-geometry state, keys exactly
                     DataVectorGeometry.state() (total_size,
                     dest_idx, evecs, sqrt_ev, Cinv, center, dtype),
                     plus a "cls" attr, so from_state rebuilds the
                     exact geometry class with no cosmolike run
                     (cosmolike is the C likelihood code that computed
                     the training data). A Grid2D state
                     carries its low-k constant mask inside the geometry
                     state itself (const_mask).
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
    config_resolved_yaml  the consumed config, defaults materialized: native
                     composition facts, resolved pce / transfer records,
                     resolved_train, and data. A saved run therefore
                     reconstructs even if code defaults drift.
    model_recipe     the serializable model rebuild recipe (class qualname,
                     dims, every constructor kwarg, the act / norm / head
                     factories by name), read by rebuild_emulator (h5-only,
                     a missing key loud, never a code default).
    fixed_facts/     the cosmology the dataset was
                     generated under: what was held fixed and at what value,
                     the neutrino convention, the dark-energy law, the units
                     the spectra are measured in.
    input_domain/    the parameters the generator
                     sampled, in the canonical order it declared, with the
                     interval each was drawn from.
    facts_sidecar_yaml  the producer's sidecar text
                     itself, stored verbatim beside those two groups, so the
                     record can be checked against the words it was copied
                     from. fixed_facts.write_h5 writes all three: it parses
                     the text and writes that parse, so the blocks in the
                     file are by construction the reading of the text in the
                     file, and this function never authors a scientific fact.
    train_args_yaml  the collapsed train_args actually used (search
                     ranges resolved to their defaults), as YAML.
  plus one root attribute per entry of `attrs` (run identity: model name,
  activation, rescale, N_train, best epoch, ...), the pair token (one fresh
  random string per save, also stored inside the .emul, so a rebuild can
  prove the two files were saved together), a "created" timestamp, the
  torch version, the git commit, and the schema version. Every successful
  save carries both resolved recipes and the scientific record, and it
  always writes schema 3.
  save_emulator has no option for writing an older format.

  Reversible map (write here -> read in rebuild_emulator):
    This table pairs everything this function writes with the exact
    place rebuild_emulator reads it back, so the round trip can be
    checked line by line. The house rule is that a saved run
    reconstructs from the file alone, so the read side never falls
    back to a code default: every key it needs is fetched through the
    _need / _read_artifact_composition helpers, which raise a named error when
    the key is absent instead of substituting a value. The few rows marked
    "not read (provenance)" are written on purpose as a paper trail. The
    model recipe and composition record are checked before executable
    reconstruction begins. Training plans and histories describe how the run
    was made; they do not control how its learned model is reopened.

      written by save_emulator      | read back in rebuild_emulator
      ------------------------------|------------------------------------
      <root>.emul ({"pair_token",   | weights-only torch.load through the
        "state_dict"}: cpu tensors, |   handle opened at the start; the
        compile-prefix stripped)    |   token must equal the .h5 attr,
                                    |   then strict _rebuild_model
      param_geometry/ group +       | _rebuild_geometry(f["param_geometry"])
        its "cls" attr              |   -> <cls>.from_state; "cls" is
                                    |   required (missing = loud re-save)
      dv_geometry/ group +          | _rebuild_geometry(f["dv_geometry"])
        its "cls" attr              |   -> <cls>.from_state; the info-dict
                                    |   family flags and the CMB / grid /
                                    |   grid2d facts below are read off
                                    |   this rebuilt geometry object, not
                                    |   from separate h5 keys
      root composition_mode +      | _read_artifact_composition validates
        transfer_refined attrs;    |   the root facts, exact group matrix,
        composition fields in      |   resolved record, and declared raw
        config_resolved_yaml;       |   config before construction; returns
        config_yaml declarations   |   info composition_mode / refined
      pce/ group (NPCE runs) +      | mode == "npce" selects the group;
        its "form" attr             |   _need(pce_grp, "form") -> pce_base,
                                    |   pce_form
      transfer_base/ group          | mode == "transfer" reads the whole:
        (transfer runs):            |   tb_recipe / tb_state / tb_pgeom /
        model_recipe, state/,       |   tb_geom rebuilt, then _rebuild_model
        param_geometry/,            |   builds the frozen base; form / space
        dv_geometry/,               |   -> transfer_form / transfer_space;
        "form" + "space" attrs      |   -> transfer_form / transfer_space
      transfer_base/drifted_state/  | validated transfer_refined == True
        (refined runs only)         |   selects _read_group(drifted_state);
                                    |   membership never selects the mode
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
                                    |   the stored text and checks it against
                                    |   the stored blocks in both directions;
                                    |   the two blocks reach the caller as
                                    |   info["fixed_facts"] and
                                    |   info["input_domain"]
      history/ group (train_losses, | not read (provenance); the writer checks
        val_medians, val_means,     |   that the five finite arrays have
        val_fracs, thresholds)      |   compatible shapes before the save
      config_yaml,                  | composition declarations are checked;
        config_resolved_yaml        |   data, model, rescale, composition, and
                                    |   the opaque resolved training record
                                    |   must reproduce the output identity
      train_args_yaml               | not read (provenance)
      root rescale attr             | _read_public_rescale requires native
                                    |   "none" before geometry/model rebuild;
                                    |   transformed artifacts lack an inverse
      other attrs entries, created, | not read (provenance): run identity,
        torch_version, git_commit   |   timestamp, and build marks
    (legend: "<root>" = path_root; "cls" = a "module.QualName" string
     naming the class to reconstruct; state_dict = PyTorch's name -> tensor
     map of registered parameters, including frozen parameters, and
     persistent registered buffers; _need / _read_group
     / _read_artifact_composition / _read_native_bool / _rebuild_geometry
     / _rebuild_model = the reader
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
                     written as one h5 root attribute. It may not replace the
                     writer-owned composition declarations.
    pce            = fitted PCEEmulator base for an NPCE run, else None.
    pce_form       = NPCE recombination form, required with pce, else None.
    resolved_train = the training settings the run actually consumed,
                     with every default filled in (materialized).
    resolved_model = the model-construction settings, defaults filled
                     in the same way.
    transfer_base  = embedded transfer payload mapping, else None.
    facts_yaml     = the required producer sidecar's text, the generator's own
                     <paramsf>.facts.yaml, carried here verbatim by the
                     training loader (data_staging.read_facts_sidecar). The
                     file records the science it was born under and announces
                     the current schema version. It is never re-derived here:
                     two authors of one scientific fact is how the two copies
                     of that fact drift apart. None is refused; migration of an
                     older dataset is explicit regeneration, never a new
                     older-format emulator.
    composition_mode = required writer-derived native string: plain, npce, or
                     transfer. It must match pce / transfer_base exactly.
    transfer_refined = required native bool; true exactly when transfer_base
                     carries drifted_state.
    resolved_pce   = consumed pce mapping for NPCE, else None.
    resolved_transfer = consumed transfer mapping for transfer, else None.
    resolved_rescale = the executed analytic-rescaling mode. Schema 3 can
                     publish only ``none`` because it stores no inverse for
                     ``rescaled`` or ``residual`` targets. Production callers
                     supply this fact explicitly. A low-level caller may omit
                     it only when attrs explicitly records native ``none``;
                     that saved fact is then used rather than a code default.

  Returns:
    (emul_path, h5_path), the two files written.

  Both files are first written whole under temporary names beside the
  destination and renamed into place only when both are complete, so an
  ordinary failure mid-save leaves no partial file under the public names.
  The two renames are separate filesystem operations; a process kill exactly
  between them can leave one member newer than the other, and the
  occupied-root refusal then makes the next save loud instead of silent.

  Raises:
    KeyError when attrs tries to replace a writer-owned declaration.
    TypeError when facts_yaml is not text or a resolved recipe is not a plain
    mapping.
    ValueError when the composition facts, payload, and consumed records
    disagree; when facts_yaml or either resolved recipe is missing; when the
    sidecar breaks a law in fixed_facts.validate; when the whitening geometry
    and the sidecar disagree on the sampled parameters; when the destination
    root already exists; or when a structured model's fixed padding layout
    disagrees with its output geometry.
    These required-input checks run before model.state_dict and before any
    temporary or final output is created.
  """
  # Writer-owned declarations are checked first because this is a pure mapping
  # check: it touches neither the model nor the filesystem. A forged caller
  # value therefore keeps its precise diagnostic even when another required
  # input is also absent.
  reserved_attrs = {
    _COMPOSITION_MODE_ATTR,
    _TRANSFER_REFINED_ATTR,
  }
  if attrs is not None and reserved_attrs.intersection(attrs):
    forged = sorted(reserved_attrs.intersection(attrs))
    raise KeyError(
      "save_emulator: attrs cannot set reserved key(s) " + repr(forged)
      + ". Remove them; save_emulator derives these declarations from the "
      "executed run.")

  # A completed name is immutable.  This check runs before serializer
  # imports, model-state inspection, or temporary-file allocation.
  _refuse_existing_artifact_root(path_root)

  # A public reader accepts only the current schema, so a new training run must
  # never emit a schema-less or older saved emulator. Refuse missing facts or
  # model-building instructions before importing the serializer, inspecting
  # model.state_dict, or reserving either temporary output. This ordering keeps
  # a failed long-run save from leaving a marker or a misleading partial file.
  if facts_yaml is None:
    raise ValueError(
      path_root + ": save_emulator requires the producer's "
      + fixed_facts.SIDECAR_SUFFIX + " scientific record. "
      + fixed_facts.MIGRATION)
  if type(facts_yaml) is not str:
    raise TypeError(
      path_root + ": facts_yaml must be the producer sidecar text, not "
      + type(facts_yaml).__name__
      + ". Read the " + fixed_facts.SIDECAR_SUFFIX
      + " file as UTF-8 text; do not pass parsed data or raw bytes.")
  missing_recipes = []
  if resolved_train is None:
    missing_recipes.append("resolved_train")
  if resolved_model is None:
    missing_recipes.append("resolved_model")
  if missing_recipes:
    raise ValueError(
      path_root + ": save_emulator requires "
      + " and ".join(missing_recipes)
      + " before it can write a schema-3 emulator that can be reopened. "
      "Run training through the current configuration reader and pass the "
      "exact training settings and model-building instructions it used; do "
      "not create an older-format emulator.")
  for recipe_name, recipe in (
      ("resolved_train", resolved_train),
      ("resolved_model", resolved_model),
  ):
    if type(recipe) is not dict:
      raise TypeError(
        path_root + ": " + recipe_name
        + " must be a plain mapping of the settings the run used, not "
        + type(recipe).__name__
        + ". Pass the resolved configuration mapping without converting it "
        "to a list, tuple, or YAML string.")
  validate_model_recipe(
    resolved_model, where=path_root + ": resolved_model")
  _validate_live_recipe_geometry_widths(
    resolved_model, param_geometry, geometry, path_root)

  # h5py lives only here: the training machines (cocoa) ship it, the
  # plotting/train paths never need it.
  import h5py

  composition_mode, transfer_refined = _validate_executed_composition(
    composition_mode=composition_mode,
    transfer_refined=transfer_refined,
    pce=pce,
    pce_form=pce_form,
    transfer_base=transfer_base,
    resolved_pce=resolved_pce,
    resolved_transfer=resolved_transfer,
    where=path_root)

  # Validate the embedded base recipe before reading its model state.  The
  # outer transfer decoder belongs only to the main correction recipe.
  if composition_mode == "transfer":
    for key in ("recipe", "param_geometry", "dv_geometry"):
      if key not in transfer_base:
        raise KeyError(
          path_root + ": transfer_base is missing " + repr(key)
          + "; save refuses an embedded model that cannot be rebuilt")
    transfer_recipe = transfer_base["recipe"]
    validate_model_recipe(
      transfer_recipe, where=path_root + ": transfer_base.model_recipe")
    _validate_live_recipe_geometry_widths(
      transfer_recipe, transfer_base["param_geometry"],
      transfer_base["dv_geometry"], path_root + ": transfer_base")

  history_arrays = _history_arrays_for_save(
    histories, where=path_root + ": histories")

  if type(attrs) is not dict or "rescale" not in attrs:
    raise ValueError(
      path_root + ": schema 3 requires attrs to record the explicit native "
      "string rescale='none'; missing metadata is not a default")
  recorded_rescale = attrs["rescale"]
  if type(recorded_rescale) is not str or recorded_rescale != "none":
    raise ValueError(
      path_root + ": schema 3 can publish only the explicit native string "
      "rescale='none'; it stores no inverse for rescaled or residual targets")
  if resolved_rescale is not None:
    if type(resolved_rescale) is not str \
        or resolved_rescale != recorded_rescale:
      raise ValueError(
        path_root + ": resolved_rescale disagrees with attrs.rescale; both "
        "must record the exact native string 'none'")

  # New artifacts have one schema.  Legacy files remain a reader/migration
  # concern; there is deliberately no writer flag that can create another.
  schema_version = fixed_facts.SCHEMA_VERSION

  # The parameters the generator declared it sampled must be the parameters the
  # whitening geometry holds, in the same order, and the check runs before
  # either file is written. Every law the record must satisfy belongs to
  # fixed_facts and is enforced there, on the way in and on the way back out,
  # so that a record cannot be refused on one path and accepted on another.
  # This function is a caller of those laws, never a second author of them.
  # check_names_match / parse_sidecar (fixed_facts.py): the record's own laws.
  record = fixed_facts.parse_sidecar(
    text=facts_yaml,
    where="the producer sidecar being saved into " + path_root + ".h5")
  fixed_facts.check_names_match(
    geometry_names=param_geometry.state()["names"],
    blocks=record,
    where=path_root + ".h5")

  # The caller-built recipe is not evidence about the module it hands us.
  # Compare the constructor-owned runtime record after all cheaper inert facts
  # have received their own diagnostics, but before weight inspection or any
  # staging-file allocation.  A transfer base carries its live source module
  # for the same independent binding; the module itself is not serialized.
  check_model_matches_recipe(
    model, resolved_model, where=path_root + ": live model")
  if composition_mode == "transfer":
    if "model" not in transfer_base:
      raise KeyError(
        path_root + ": transfer_base is missing 'model'; pass the live "
        "source model so its constructor recipe can be checked")
    check_model_matches_recipe(
      transfer_base["model"], transfer_recipe,
      where=path_root + ": live transfer base")

  # --- stage <root>.emul: the weights, cpu, unprefixed ---
  sd = {}
  for k, v in model.state_dict().items():
    # a torch.compile wrapper (OptimizedModule) stores the real
    # model as ._orig_mod, so its keys arrive prefixed; strip it.
    sd[k.removeprefix("_orig_mod.")] = v.detach().cpu()
  _validate_saved_head_layout(
    model_state=sd,
    geometry=geometry,
    recipe=resolved_model,
    where=path_root)
  emul_path = path_root + ".emul"
  h5_path = path_root + ".h5"
  # both members are written to temporary names and renamed into place at
  # the very end, so a crash mid-save leaves no partial artifact under the
  # public names.
  tmp_emul = emul_path + ".tmp"
  tmp_h5 = h5_path + ".tmp"
  # One fresh random pair token per save, stored in BOTH members. Rebuild
  # compares the two copies, so a .emul placed beside another run's .h5
  # (or the reverse) refuses instead of pairing weights with the wrong
  # scientific record. Two saves never share a token, even from identical
  # settings, so any cross-run mix is caught.
  pair_token = uuid.uuid4().hex
  try:
    torch.save({"pair_token": pair_token, "state_dict": sd}, tmp_emul)
  except BaseException:
    _unlink_if_present(tmp_emul)
    raise

  # --- write <root>.h5: geometries + histories + config ---
  str_dt  = h5py.string_dtype(encoding="utf-8")

  def write_state(group, state):
    """Write one geometry state() dict into an HDF5 group, recursively.

    Every geometry saves through this one function, without per-class
    code: tensors become datasets, name lists become string datasets,
    nested dicts (a composed geometry, e.g. AmplitudeFactorGeometry's
    pg_keep) become subgroups, and scalars become attributes.

    Arguments:
      group = the open HDF5 group to fill.
      state = the geometry's state() mapping.
    """
    # tensors -> datasets, name lists -> string datasets, nested dicts
    # -> subgroups, scalars and
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

  with _tmp_h5_file(h5py, tmp_h5, cleanup_paths=(tmp_emul, tmp_h5)) as f:
    # the scientific record, when the dataset published one: the producer's own
    # sidecar text, stored verbatim, and the two blocks that text parses to.
    # write_h5 (fixed_facts.py) does both, and it writes the parse of the text
    # it stores, so the file's blocks and the file's text can be held against
    # each other when it is read back. Training copies the record; it never
    # writes one. The groups appear exactly when the version decided above
    # promises them, so the number the file announces and the groups the file
    # carries are one decision, not two.
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
    dv_state = geometry.state()
    write_state(dv_group, dv_state)
    dv_cls = (type(geometry).__module__ + "."
              + type(geometry).__qualname__)
    dv_group.attrs["cls"] = dv_cls
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
      base_dv_state = base_dv.state()
      write_state(dv_grp, base_dv_state)
      base_dv_cls = (type(base_dv).__module__ + "."
                     + type(base_dv).__qualname__)
      dv_grp.attrs["cls"] = base_dv_cls
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

    # per-epoch histories; fracs stack to (nepochs, n_thresholds).
    hg = f.create_group("history")
    hg.create_dataset("train_losses",
                      data=history_arrays["train_losses"])
    hg.create_dataset("val_medians",
                      data=history_arrays["val_medians"])
    hg.create_dataset("val_means",
                      data=history_arrays["val_means"])
    hg.create_dataset("val_fracs", data=history_arrays["val_fracs"])
    hg.create_dataset("thresholds", data=history_arrays["thresholds"])

    # the full configs, verbatim, as YAML text.
    f.create_dataset("config_yaml",
                     data=yaml.safe_dump(config, sort_keys=False),
                     dtype=str_dt)
    if train_args is not None:
      f.create_dataset("train_args_yaml",
                       data=yaml.safe_dump(train_args,
                                           sort_keys=False),
                       dtype=str_dt)

    # The CONSUMED view, defaults materialized, so a saved run
    # reconstructs even if code defaults drift (the standing rule). The raw
    # config_yaml / train_args_yaml above stay as the provenance of what was
    # WRITTEN; these record what the run RESOLVED.
    if resolved_train is not None:
      f.create_dataset(
        "config_resolved_yaml",
        data=yaml.safe_dump({
          "composition_mode": composition_mode,
          "transfer_refined": transfer_refined,
          "pce": resolved_pce,
          "transfer": resolved_transfer,
          "train_args": resolved_train,
          "data": config.get("data", {}),
        },
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
    f.attrs[_COMPOSITION_MODE_ATTR] = composition_mode
    f.attrs[_TRANSFER_REFINED_ATTR] = transfer_refined
    # the same random string written into the .emul container above; the
    # rebuild compares the two copies to prove the pair was saved together.
    f.attrs["pair_token"]    = pair_token
    f.attrs["created"]       = time.strftime("%Y-%m-%d %H:%M:%S")
    f.attrs["torch_version"] = str(torch.__version__)
    # the version decided at the top of this function, written once, here. It
    # marks a file that carries the resolved recipe, and at version 3 the
    # scientific record too; rebuild_emulator refuses any file without it. The
    # code commit is best-effort provenance (rev-parse, else "unknown").
    f.attrs["schema_version"] = schema_version
    try:
      f.attrs["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
      f.attrs["git_commit"] = "unknown"

  # both members are complete: give them their public names. The early
  # _refuse_existing_artifact_root call proved the root was free.
  os.replace(tmp_emul, emul_path)
  os.replace(tmp_h5, h5_path)

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


def _read_native_enum(attrs, key, *, allowed, where):
  """Read one required HDF5 string attribute from a closed value set.

  h5py decodes both native UTF-8 and byte-string attributes to Python
  strings on some versions, so the STORED dtype is checked too: a byte
  payload cannot impersonate a writer-owned native declaration.

  Arguments:
    attrs   = the HDF5 attribute mapping (a group's or file's .attrs).
    key     = the attribute name.
    allowed = the closed set of accepted values.
    where   = the artifact's identity, named in every refusal.

  Returns:
    the attribute value, one of ``allowed``.

  Raises:
    KeyError when the attribute is absent; ValueError when its storage
    type is not a native string or its value is outside ``allowed``.
  """
  if key not in attrs:
    raise KeyError(
      where + " is missing required native attribute " + repr(key))
  value = attrs[key]
  dtype = attrs.get_id(key).dtype
  storage = (dtype.metadata or {}).get("vlen")
  if storage is not str or type(value) is not str or value not in allowed:
    raise ValueError(
      where + ": the attribute " + repr(key)
      + " must be one native HDF5 string in " + repr(allowed)
      + ", got " + repr(value) + " (type " + type(value).__name__ + ").")
  return value


def _read_public_rescale(attrs, *, where):
  """Read the only target transform schema 3 can invert for prediction.

  Training may save ``rescaled`` or ``residual`` products for diagnostics,
  but schema 3 does not store the parameter-dependent information needed to
  reverse those transforms when serving a new point.  Public rebuild therefore
  requires an explicit native HDF5 string ``rescale='none'``.  Missing metadata
  is not interpreted as the default.

  Arguments:
    attrs = root HDF5 attribute mapping.
    where = artifact path used in the refusal.

  Returns:
    the native string ``"none"``.

  Raises:
    KeyError when the attribute is absent; ValueError when its storage type or
    value is unsupported.  Both explain why the inverse is unavailable.
  """
  reason = (
    " Public prediction supports only the explicit native string "
    "rescale='none'. Schema 3 does not store the parameter-dependent inverse "
    "transform required by 'rescaled' or 'residual' artifacts. Re-save or "
    "retrain the deployable artifact with rescale='none'.")
  try:
    return _read_native_enum(
      attrs, "rescale", allowed=("none",), where=where)
  except KeyError as error:
    raise KeyError(str(error.args[0]) + reason) from error
  except ValueError as error:
    raise ValueError(str(error) + reason) from error


def _read_artifact_composition(f, where):
  """Read and validate the artifact's authoritative composition facts.

  The main network has the same state-dict shape for a plain emulator, an
  NPCE refiner, and a transfer correction.  Group absence therefore cannot
  select the plain decoder.  The writer-owned root enum and native refined
  boolean select the mode; the payload groups and consumed YAML record must
  corroborate those facts in both directions before any geometry or model is
  constructed.

  ``config_yaml`` is provenance.  When it declares a non-null ``pce`` or
  ``transfer`` block, that declaration must agree too, but it can never
  replace either required root fact.

  Arguments:
    f     = the open HDF5 file of the artifact.
    where = the artifact's identity, named in every refusal.

  Returns:
    ``(composition_mode, transfer_refined)`` as plain Python values.

  Raises:
    KeyError / ValueError for a missing, mistyped, unknown, or contradictory
    fact, group, or resolved record.  Presence-only legacy artifacts receive
    an explicit re-save/migration instruction rather than defaulting to plain.
  """
  if _COMPOSITION_MODE_ATTR not in f.attrs:
    raise KeyError(
      where + " is a presence-only artifact with no required "
      + repr(_COMPOSITION_MODE_ATTR) + " root attribute. Absence never means "
      "plain. Re-save the artifact with the current save_emulator, or migrate "
      "it by reconstructing the executed plain/npce/transfer mode explicitly.")
  composition_mode = _read_native_enum(
    f.attrs,
    _COMPOSITION_MODE_ATTR,
    allowed=_COMPOSITION_MODES,
    where=where)

  if _TRANSFER_REFINED_ATTR not in f.attrs:
    raise KeyError(
      where + " is a presence-only artifact with no required "
      + repr(_TRANSFER_REFINED_ATTR) + " root attribute. Re-save the artifact "
      "with the current save_emulator; refined state is never inferred from "
      "a nested group.")
  transfer_refined = _read_native_bool(
    f.attrs,
    _TRANSFER_REFINED_ATTR,
    default=False,
    where=where)

  have_pce = "pce" in f
  have_transfer = "transfer_base" in f
  expected_pce = composition_mode == "npce"
  expected_transfer = composition_mode == "transfer"
  if have_pce != expected_pce or have_transfer != expected_transfer:
    raise KeyError(
      where + ": composition_mode=" + repr(composition_mode)
      + " requires pce=" + repr(expected_pce)
      + " and transfer_base=" + repr(expected_transfer)
      + ", but the artifact carries pce=" + repr(have_pce)
      + " and transfer_base=" + repr(have_transfer)
      + ". Required and forbidden composition groups are checked in both "
      "directions; re-save an internally consistent artifact.")
  payload_pce_form = None
  if have_pce:
    pce_group = f["pce"]
    if not hasattr(pce_group, "keys"):
      raise ValueError(where + ": pce must be an HDF5 group")
    payload_pce_form = _read_native_enum(
      pce_group.attrs,
      "form",
      allowed=_PCE_FORMS,
      where=where + " pce group")
  have_drifted = False
  payload_transfer_form = None
  payload_transfer_space = None
  if have_transfer:
    transfer_group = f["transfer_base"]
    if not hasattr(transfer_group, "keys"):
      raise ValueError(where + ": transfer_base must be an HDF5 group")
    payload_transfer_form = _read_native_enum(
      transfer_group.attrs,
      "form",
      allowed=_TRANSFER_FORMS,
      where=where + " transfer_base group")
    payload_transfer_space = _read_native_enum(
      transfer_group.attrs,
      "space",
      allowed=_TRANSFER_SPACES,
      where=where + " transfer_base group")
    have_drifted = "drifted_state" in transfer_group
    if have_drifted and not hasattr(transfer_group["drifted_state"], "keys"):
      raise ValueError(
        where + ": transfer_base/drifted_state must be an HDF5 group")
  expected_drifted = expected_transfer and transfer_refined
  if have_drifted != expected_drifted:
    raise KeyError(
      where + ": composition_mode=" + repr(composition_mode)
      + " and transfer_refined=" + repr(transfer_refined)
      + " require transfer_base/drifted_state=" + repr(expected_drifted)
      + ", but it is " + ("present" if have_drifted else "absent") + ".")
  if composition_mode != "transfer" and transfer_refined:
    raise ValueError(
      where + ": transfer_refined=True is forbidden for composition_mode="
      + repr(composition_mode))

  def _read_yaml_mapping(dataset_name, *, required):
    """Read one stored YAML text dataset as a plain mapping, or None.

    Arguments:
      dataset_name = the dataset holding the YAML text.
      required     = when True, an absent dataset is a KeyError naming a
                     re-save; when False, absence returns None.

    Returns:
      the parsed plain dict, or None for an absent optional dataset.
    """
    if dataset_name not in f:
      if required:
        raise KeyError(
          where + " is missing required " + dataset_name
          + "; re-save the artifact with its consumed composition record")
      return None
    value = f[dataset_name][()]
    if isinstance(value, bytes):
      try:
        value = value.decode("utf-8")
      except UnicodeDecodeError as exc:
        raise ValueError(
          where + ": " + dataset_name + " is not valid UTF-8") from exc
    if type(value) is not str:
      raise ValueError(
        where + ": " + dataset_name + " must be scalar UTF-8 YAML text, got "
        + type(value).__name__)
    try:
      parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
      raise ValueError(
        where + ": " + dataset_name + " is invalid YAML") from exc
    if type(parsed) is not dict:
      raise ValueError(
        where + ": " + dataset_name + " must decode to a mapping")
    return parsed

  resolved = _read_yaml_mapping("config_resolved_yaml", required=True)
  for key in ("composition_mode", "transfer_refined", "pce", "transfer"):
    if key not in resolved:
      raise KeyError(
        where + ": config_resolved_yaml is missing required " + repr(key))
  resolved_mode = resolved["composition_mode"]
  if type(resolved_mode) is not str or resolved_mode not in _COMPOSITION_MODES:
    raise ValueError(
      where + ": config_resolved_yaml composition_mode must be one native "
      "string in " + repr(_COMPOSITION_MODES))
  if resolved_mode != composition_mode:
    raise ValueError(
      where + ": config_resolved_yaml composition_mode="
      + repr(resolved_mode) + " disagrees with the authoritative root "
      + repr(composition_mode))
  resolved_refined = resolved["transfer_refined"]
  if type(resolved_refined) is not bool:
    raise ValueError(
      where + ": config_resolved_yaml transfer_refined must be a native "
      "boolean")
  if resolved_refined != transfer_refined:
    raise ValueError(
      where + ": config_resolved_yaml transfer_refined="
      + repr(resolved_refined) + " disagrees with the authoritative root "
      + repr(transfer_refined))

  resolved_pce = resolved["pce"]
  resolved_transfer = resolved["transfer"]
  if composition_mode == "plain":
    if resolved_pce is not None:
      raise ValueError(
        where + ": plain config_resolved_yaml requires pce: null")
    if resolved_transfer is not None:
      raise ValueError(
        where + ": plain config_resolved_yaml requires transfer: null")
  elif composition_mode == "npce":
    if type(resolved_pce) is not dict:
      raise ValueError(
        where + ": npce config_resolved_yaml requires a pce mapping")
    if resolved_transfer is not None:
      raise ValueError(
        where + ": npce config_resolved_yaml requires transfer: null")
    resolved_pce_form = resolved_pce.get("form")
    if type(resolved_pce_form) is not str \
        or resolved_pce_form not in _PCE_FORMS:
      raise ValueError(
        where + ": config_resolved_yaml pce.form must be one native string in "
        + repr(_PCE_FORMS))
    if resolved_pce_form != payload_pce_form:
      raise ValueError(
        where + ": config_resolved_yaml pce.form="
        + repr(resolved_pce_form) + " disagrees with pce/form="
        + repr(payload_pce_form))
  else:
    if resolved_pce is not None:
      raise ValueError(
        where + ": transfer config_resolved_yaml requires pce: null")
    if type(resolved_transfer) is not dict:
      raise ValueError(
        where + ": transfer config_resolved_yaml requires a transfer mapping")
    resolved_transfer_form = resolved_transfer.get("form")
    if type(resolved_transfer_form) is not str \
        or resolved_transfer_form not in _TRANSFER_FORMS:
      raise ValueError(
        where + ": config_resolved_yaml transfer.form must be one native "
        "string in " + repr(_TRANSFER_FORMS))
    resolved_transfer_space = resolved_transfer.get("space")
    if type(resolved_transfer_space) is not str \
        or resolved_transfer_space not in _TRANSFER_SPACES:
      raise ValueError(
        where + ": config_resolved_yaml transfer.space must be one native "
        "string in " + repr(_TRANSFER_SPACES))
    if resolved_transfer_form != payload_transfer_form:
      raise ValueError(
        where + ": config_resolved_yaml transfer.form="
        + repr(resolved_transfer_form) + " disagrees with "
        "transfer_base/form=" + repr(payload_transfer_form))
    if resolved_transfer_space != payload_transfer_space:
      raise ValueError(
        where + ": config_resolved_yaml transfer.space="
        + repr(resolved_transfer_space) + " disagrees with "
        "transfer_base/space=" + repr(payload_transfer_space))
    resolved_refine = resolved_transfer.get("refine")
    if resolved_refine is not None and type(resolved_refine) is not dict:
      raise ValueError(
        where + ": config_resolved_yaml transfer.refine must be a mapping "
        "when present")
    resolved_has_refine = resolved_refine is not None
    if resolved_has_refine != transfer_refined:
      raise ValueError(
        where + ": config_resolved_yaml transfer.refine presence disagrees "
        "with transfer_refined=" + repr(transfer_refined))

  raw = _read_yaml_mapping("config_yaml", required=False)
  if raw is not None:
    raw_pce = raw.get("pce")
    raw_transfer = raw.get("transfer")
    if raw_pce is not None and raw_transfer is not None:
      raise ValueError(
        where + ": config_yaml declares both pce and transfer")
    if raw_pce is not None and composition_mode != "npce":
      raise ValueError(
        where + ": config_yaml pce declaration disagrees with "
        "composition_mode=" + repr(composition_mode))
    if raw_transfer is not None and composition_mode != "transfer":
      raise ValueError(
        where + ": config_yaml transfer declaration disagrees with "
        "composition_mode=" + repr(composition_mode))
    if raw_pce is not None:
      if type(raw_pce) is not dict:
        raise ValueError(where + ": config_yaml pce must be a mapping")
      raw_pce_form = raw_pce.get("form")
      if type(raw_pce_form) is not str or raw_pce_form not in _PCE_FORMS:
        raise ValueError(
          where + ": config_yaml pce.form must be one native string in "
          + repr(_PCE_FORMS))
      if raw_pce_form != payload_pce_form:
        raise ValueError(
          where + ": config_yaml pce.form=" + repr(raw_pce_form)
          + " disagrees with the executed pce/form="
          + repr(payload_pce_form))
    if raw_transfer is not None:
      if type(raw_transfer) is not dict:
        raise ValueError(where + ": config_yaml transfer must be a mapping")
      raw_transfer_form = raw_transfer.get("form")
      if (type(raw_transfer_form) is not str
          or raw_transfer_form not in _TRANSFER_FORMS):
        raise ValueError(
          where + ": config_yaml transfer.form must be one native string in "
          + repr(_TRANSFER_FORMS))
      if raw_transfer_form != payload_transfer_form:
        raise ValueError(
          where + ": config_yaml transfer.form=" + repr(raw_transfer_form)
          + " disagrees with the executed transfer_base/form="
          + repr(payload_transfer_form))
      raw_transfer_space = raw_transfer.get("space")
      if raw_transfer_space is not None:
        if (type(raw_transfer_space) is not str
            or raw_transfer_space not in _TRANSFER_SPACES):
          raise ValueError(
            where + ": config_yaml transfer.space must be null or one "
            "native string in " + repr(_TRANSFER_SPACES))
        if raw_transfer_space != payload_transfer_space:
          raise ValueError(
            where + ": config_yaml transfer.space="
            + repr(raw_transfer_space)
            + " disagrees with the executed transfer_base/space="
            + repr(payload_transfer_space))
      raw_refine = raw_transfer.get("refine")
      if raw_refine is not None and type(raw_refine) is not dict:
        raise ValueError(
          where + ": config_yaml transfer.refine must be a mapping when "
          "present")
      raw_refined = raw_refine is not None
      if raw_refined != transfer_refined:
        raise ValueError(
          where + ": config_yaml transfer.refine presence disagrees with "
          "transfer_refined=" + repr(transfer_refined))

  return composition_mode, transfer_refined


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
  file.

  This is the in-package inference entry point, and it is written as
  the working proof of the save's central promise: a run rebuilds
  bit-exactly even if code defaults later drift, because every setting
  comes from the resolved recipe in the h5. A missing recipe key is a
  loud error, never a fallback to a code default.
  read_artifact_schema (above) reads the file's schema version and its
  scientific record before anything else is read; a file saved under a version
  this code does not know is refused there, because it cannot say which
  cosmology it was trained under or which parameter region it may be asked
  about.

  For the full write-to-read pairing (every value save_emulator writes
  beside the place here that reads it, and the entries written only as
  provenance -- a paper trail this function does not consume), see the
  "Reversible map" table in save_emulator's docstring.

  Arguments:
    path_root     = the output path without extension (as passed to
                    save_emulator); reads <path_root>.h5 and <path_root>.emul.
    device        = device to rebuild the module and geometries on.
    compile_model = torch.compile the rebuilt module on CUDA when the recipe
                    stored a compile_mode (default True; the predictor passes
                    False for batch-1 inference, where the compile latency
                    rarely pays off).

  Returns:
    (model, param_geometry, geometry, info): the four objects inference
    needs; the model in eval() with the best-epoch weights loaded
    (strict). info is a plain dict of the per-family facts the predictor
    needs to build the decoder, read straight from the file so nothing
    is re-declared downstream:
      "fixed_facts"  = the cosmology the run was trained under, exactly as the
                       generator published it: what was held fixed and at what
                       value, the neutrino convention, the dark-energy law;
      "input_domain" = the parameters the generator sampled, in the canonical
                       order it declared, with the interval each was drawn
                       from, which is the region this emulator may be asked
                       about;
      "composition_mode" = required plain / npce / transfer artifact mode;
      "transfer_refined" = required native refined-transfer fact;
      "ia"           = the factored-IA design name (nla / tatt) or None;
      "pce_base"     = the reconstructed frozen PCEEmulator when the run used
                       an NPCE base (h5 pce group), else None;
      "pce_form"     = the NPCE recombination form (residual / ratio) stored
                       on the pce group, else None;
      "model_recipe" and "config_resolved" = warm-start metadata copied from
                       the same open HDF5 handle, so a caller never has to
                       reopen the pathname;
      "rescale"      = the required native string "none". Public rebuild
                       refuses every transformed form because schema 3 lacks
                       the parameter-dependent inverse.

  Raises:
    ValueError when the checkpoint is not a plain tensor-only state dict,
    refused before model construction. ValueError from
    read_artifact_schema when the .h5 announces no schema version, announces
    one this code does not know, or breaks any law of the scientific record;
    ValueError when the rebuilt input geometry disagrees with that record's
    sampled-parameter order;
    KeyError naming a missing rescale fact or recipe / geometry / pce key
    (never a code-default fallback); ValueError when rescale is not native
    "none" because public inference cannot reconstruct a parameter-dependent
    inverse transform.
  """
  import importlib
  # h5py lives only here: the training machines (cocoa) ship it, the
  # plotting/train paths never need it.
  import h5py

  def _read_group(g):
    """Read one HDF5 group back into a state dict, recursively.

    The inverse of save's write_state: numeric datasets -> tensors on
    ``device``, string datasets (names) -> str lists, attributes ->
    scalars (a "torch.<dtype>" string restored to the torch.dtype),
    subgroups -> nested dicts.

    Arguments:
      g = the open HDF5 group.

    Returns:
      the state mapping, ready for a geometry's from_state.
    """
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
        tensor = torch.as_tensor(np.asarray(item[()]))
        # Apple Silicon's MPS backend cannot hold float64 at all, and
        # some saved datasets (the z / k coordinate grids, the CMB
        # fiducial-reference scalars) are float64. Moving one to MPS
        # would raise; narrow it to float32 first (a coordinate axis
        # needs no more), so a `device: mps` rebuild works instead of
        # crashing. cpu / cuda keep the saved float64 exactly.
        if (tensor.dtype == torch.float64
            and torch.device(device).type == "mps"):
          tensor = tensor.to(torch.float32)
        state[k] = tensor.to(device)
    for k, v in g.attrs.items():
      if isinstance(v, str) and v.startswith("torch."):
        state[k] = getattr(torch, v.split(".", 1)[1])
      else:
        state[k] = v
    return state

  def _need(d, k, where):
    """Read one required key from a saved mapping, never a code default.

    Arguments:
      d     = the saved mapping (recipe, block options, group state).
      k     = the required key.
      where = which mapping is being read, named in the refusal.

    Returns:
      the stored value.
    """
    if k not in d:
      raise KeyError(
        f"{path_root}.h5 {where} is missing {k!r}; rebuild_emulator reads "
        "only the file and never falls back to a code default")
    return d[k]

  def _rebuild_geometry(group, where):
    """Rebuild one geometry object from its saved group.

    Dispatches on the persisted class marker: read the group, resolve
    its own "cls" path through importlib (the model-recipe pattern), and
    call THAT class's from_state, so a factored AmplitudeFactorGeometry
    or a LogParamGeometry rebuilds as itself. A missing marker is loud
    and names a re-save -- never a silent fallback to the base geometry
    class (the read-side rule).

    Arguments:
      group = the open HDF5 geometry group.
      where = which geometry is being rebuilt, named in the refusal.

    Returns:
      the rebuilt geometry object, on ``device``.
    """
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

  with contextlib.ExitStack() as pair_stack:
    checkpoint = pair_stack.enter_context(
      _open_regular_checkpoint(path_root + ".emul"))
    f = pair_stack.enter_context(h5py.File(path_root + ".h5", "r"))
    # the schema version and the scientific record, through the one shared
    # reader. It refuses a version this code does not know, and a file that
    # announces none, before any other value is read. The blocks it returns are
    # plain Python, so they survive this file being closed and travel out in
    # the info dict below.
    record = read_artifact_schema(f=f, where=path_root + ".h5")
    # The authoritative composition facts are the next read.  This validates
    # the complete required/forbidden group matrix and its consumed record
    # before recipe parsing, geometry construction, model construction, or
    # torch.load can reinterpret a damaged NPCE/transfer artifact as plain.
    composition_mode, transfer_refined = _read_artifact_composition(
      f=f, where=path_root + ".h5")
    saved_rescale = _read_public_rescale(
      f.attrs, where=path_root + ".h5")
    # Everything below this line is still inert metadata. The model recipe and
    # geometry dimensions must agree before a saved class can be imported.
    # Training histories remain provenance and are not needed for prediction.
    recipe = _read_yaml_mapping_dataset(
      f, "model_recipe", path_root + ".h5")
    validate_model_recipe(recipe, where=path_root + ".h5 model_recipe")
    _validate_saved_recipe_geometry_widths(
      recipe, f["param_geometry"], f["dv_geometry"], path_root + ".h5")
    resolved_config = _read_yaml_mapping_dataset(
      f, "config_resolved_yaml", path_root + ".h5")
    if "train_args" not in resolved_config:
      raise KeyError(
        path_root + ".h5 config_resolved_yaml is missing 'train_args'")
    tb = None
    tb_recipe = None
    if composition_mode == "transfer":
      tb = f["transfer_base"]
      tb_recipe = _read_yaml_mapping_dataset(
        tb, "model_recipe", path_root + ".h5 transfer_base")
      validate_model_recipe(
        tb_recipe, where=path_root + ".h5 transfer_base model_recipe")
      _validate_saved_recipe_geometry_widths(
        tb_recipe, tb["param_geometry"], tb["dv_geometry"],
        path_root + ".h5 transfer_base")
    # The trusted local factories and registered geometry implementations are
    # imported only after all saved strings above have passed their closed
    # schemas.  Dynamic geometry/model paths cannot execute during preflight.
    from .activations import make_activation
    from .designs.blocks import make_norm
    from .geometries.scalar import ScalarGeometry
    from .geometries.cmb import CmbDiagonalGeometry
    from .geometries.grid import GridGeometry
    from .geometries.grid2d import Grid2DGeometry

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
    if composition_mode == "npce":
      from .designs.pce import PCEEmulator
      pce_grp  = _read_group(f["pce"])
      pce_form = _need(pce_grp, "form", "pce group")
      pce_base = PCEEmulator.from_state(pce_grp, device)

    # transfer_base (a transfer artifact only): read the embedded frozen base's
    # recipe, weights, and both geometries; the model is reconstructed below,
    # after the main model, through the shared helper. form / space tell the
    # predictor how to compose base + correction.
    tb_state  = None
    tb_pgeom  = None
    tb_geom   = None
    tb_form   = None
    tb_space  = None
    if composition_mode == "transfer":
      tb_state  = _read_group(tb["state"])
      tb_pgeom  = _rebuild_geometry(tb["param_geometry"],
                                    "transfer_base param_geometry group")
      tb_geom   = _rebuild_geometry(tb["dv_geometry"],
                                    "transfer_base dv_geometry group")
      tb_form   = tb.attrs.get("form")
      tb_space  = tb.attrs.get("space")
      # The root fact and nested-group presence were already validated before
      # any geometry construction.  Route solely by that validated fact; group
      # membership is never a second mode-selection algorithm.
      if transfer_refined:
        tb_state = _read_group(tb["drifted_state"])

    # Composition and geometry validation deliberately precede deserialization,
    # preserving their more useful refusal messages.  The checkpoint handle is
    # still the same one opened before those checks began, so no pathname swap
    # can occur between the validation and this explicit tensor-only load.
    # The pair token is the same random string save_emulator wrote into both
    # members of this pair; comparing the two copies proves these exact files
    # came from one save. A file without it predates the token and is refused
    # with re-save guidance, never guessed about.
    stored_token = f.attrs.get("pair_token")
    if type(stored_token) is not str or not stored_token:
      raise ValueError(
        f"{path_root}.h5 carries no pair_token root attribute, so the "
        ".h5/.emul pair cannot be checked as coming from one save. It was "
        "saved before the pair token existed; re-save the emulator "
        "(retrain, or re-run save_emulator on the run) to add it.")
    sd = _load_tensor_state_dict(
      checkpoint, device=device, where=path_root + ".emul",
      expected_token=stored_token)

  def _rebuild_model(rc, geom_for_needs, state, want_compile):
    """Reconstruct one module from its saved recipe and state dict.

    Shared by the main model (the correction net on a transfer run, or
    the plain emulator otherwise) and the embedded transfer base, so the
    recipe -> constructor logic lives once. Factory objects (activation,
    norm) are re-made from their serialized names; every value comes
    from the file, a missing key is loud, never a code default.

    Arguments:
      rc             = the saved model-recipe mapping.
      geom_for_needs = the rebuilt output geometry, handed to the
                       constructor when the recipe declares needs_geom.
      state          = the plain (compile-prefix-stripped) name -> tensor
                       mapping to load strict.
      want_compile   = whether to wrap the module in torch.compile with
                       the recipe's stored mode.

    Returns:
      the module in eval() with the weights loaded (compiled when
      requested and a mode is stored).
    """
    cls_path = _need(rc, "cls", "model_recipe")
    mn, _, qn = cls_path.rpartition(".")
    cls = getattr(importlib.import_module(mn), qn)
    kwargs = dict(_need(rc, "kwargs", "model_recipe"))
    # re-make the factory objects from their serialized names.
    if "block_opts" in kwargs:
      bo  = kwargs["block_opts"]
      act = _need(bo, "act", "model_recipe.kwargs.block_opts")
      kwargs["block_opts"] = {
        "n_layers": _need(
          bo, "n_layers", "model_recipe.kwargs.block_opts"),
        "act": make_activation(_need(act, "type", "block_opts.act"),
                               n_gates=_need(act, "n_gates", "block_opts.act")),
        "norm": make_norm(_need(bo, "norm", "model_recipe.kwargs.block_opts")),
      }
    if kwargs.get("head_act") is not None:
      ha = kwargs["head_act"]
      kwargs["head_act"] = make_activation(
        _need(ha, "type", "head_act"),
        n_gates=_need(ha, "n_gates", "head_act"))
    needs_geom = _need(rc, "needs_geom", "model_recipe")
    if needs_geom:
      # Structured heads need bin counts plus the exact physical slot map and
      # validity mask. A diagonal family (cmb / grid / grid2d) derives its
      # complete rectangular layout from the axes already saved in the
      # geometry. A cosmic-shear geometry instead persists all three facts,
      # because recreating its angular mask would require external dataset
      # files. An older artifact without the complete layout is refused; bin
      # counts are never used to guess survivor coordinates.
      if hasattr(geom_for_needs, "attach_head_coords"):
        geom_for_needs.attach_head_coords()
      layout_complete = True
      for name in ("bin_sizes", "head_pad_idx", "head_valid_mask"):
        if not hasattr(geom_for_needs, name):
          layout_complete = False
      if not layout_complete:
        raise KeyError(
          f"{path_root}.h5 dv_geometry lacks the complete physical padded-"
          "head layout (bin_sizes, head_pad_idx, and head_valid_mask). "
          "Bin counts cannot recover masked angular coordinates. Retrain "
          "the structured-head model with the current geometry writer")
      for name in ("pad_idx", "pad_valid"):
        if name not in state:
          raise KeyError(
            f"{path_root}.emul has no {name} buffer. This structured-head "
            "artifact predates physical padding-map persistence; retrain it")
      kwargs["geom"] = geom_for_needs
    m = cls(input_dim=_need(rc, "input_dim", "model_recipe"),
            output_dim=_need(rc, "output_dim", "model_recipe"),
            **kwargs).to(device)
    cm = _need(rc, "compile_mode", "model_recipe")
    set_runtime_compile_mode(m, cm)
    check_model_matches_recipe(
      m, rc, where=path_root + ": rebuilt live model")
    if needs_geom:
      for name in ("pad_idx", "pad_valid"):
        expected = getattr(m, name).detach().cpu()
        recorded = state[name].detach().cpu()
        if expected.dtype != recorded.dtype \
            or expected.shape != recorded.shape \
            or not torch.equal(expected, recorded):
          raise ValueError(
            f"{path_root} structured-head {name} disagrees between the "
            "saved physical geometry and model checkpoint")
    m.load_state_dict(state, strict=True)
    m.eval()
    if want_compile and device.type == "cuda" and cm is not None:
      m = torch.compile(m, mode=cm)
    return m

  # the main model: the correction net (transfer run) or the plain emulator.
  # the .emul holds its plain state dict; re-compile per the recipe on CUDA.
  model = _rebuild_model(rc=recipe, geom_for_needs=geom, state=sd,
                         want_compile=compile_model)

  # the frozen transfer base, rebuilt from the embedded recipe + weights +
  # geometries (never compiled: a batch-1 no_grad component of the predictor).
  transfer_base = None
  if composition_mode == "transfer":
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
    "composition_mode": composition_mode,
    "transfer_refined": transfer_refined,
    "ia":             _need(recipe, "ia", "model_recipe"),
    "pce_base":       pce_base,
    "pce_form":       pce_form,
    "transfer_base":  transfer_base,
    "transfer_form":  tb_form,
    "transfer_space": tb_space,
    # Warm-start needs these values from the same open HDF5 handle, so a
    # caller never has to reopen the pathname.
    "model_recipe":   recipe,
    "config_resolved": resolved_config,
    "rescale":        saved_rescale,
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
