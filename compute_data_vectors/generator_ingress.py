"""Validate generator inputs before a generator creates output files.

Dataset generators receive values from command-line options, YAML files, and
plain-text covariance files.  Python can silently turn values such as
``True`` into ``1`` or truncate ``3.5`` to ``3``.  Those conversions are not
appropriate for scientific settings.  The small functions in this module
therefore check the original values and report a useful error before a caller
opens an output file.

This module deliberately has no dependency on Cobaya, MPI, emcee, or GetDist.
Tests can exercise the input boundary on an ordinary CPU installation.
"""

from collections.abc import Mapping
import math
from pathlib import Path
from pathlib import PureWindowsPath
import unicodedata

import numpy as np


def native_integer(value, label, minimum=None, allowed=None):
  """Return an integer only when the caller supplied a Python ``int``.

  ``bool`` is a subclass of ``int`` in Python, so an ``isinstance`` check
  would incorrectly accept ``True`` as the number one.  Exact type checking
  also prevents a floating value from being truncated without the user's
  knowledge.
  """
  if type(label) is not str or not label.strip():
    raise ValueError("integer setting label must be a nonempty string")
  if type(value) is not int:
    raise ValueError(
      label + " must be a native Python integer; got " + repr(value))
  if minimum is not None:
    if type(minimum) is not int:
      raise ValueError("minimum for " + label + " must be a native integer")
    if value < minimum:
      raise ValueError(
        label + " must be at least " + str(minimum) + "; got " + repr(value))
  if allowed is not None:
    try:
      accepted = tuple(allowed)
    except TypeError as error:
      raise ValueError("allowed values for " + label + " must be iterable") \
        from error
    if not accepted or any(type(item) is not int for item in accepted):
      raise ValueError(
        "allowed values for " + label + " must be native integers")
    if value not in accepted:
      raise ValueError(
        label + " must be one of " + repr(accepted) + "; got " + repr(value))
  return value


def finite_number(value, label):
  """Return a finite float made from one native Python number.

  Scientific YAML settings may use either an integer or a decimal value.
  Booleans, strings, NumPy scalar objects, infinities, and NaNs are refused so
  that their meaning cannot change during an implicit conversion.
  """
  if type(label) is not str or not label.strip():
    raise ValueError("number setting label must be a nonempty string")
  if type(value) not in (int, float):
    raise ValueError(
      label + " must be a native Python int or float; got " + repr(value))
  try:
    result = float(value)
  except (OverflowError, ValueError) as error:
    raise ValueError(
      label + " must be finite; the supplied integer is too large to "
      "represent as a float") from error
  if not math.isfinite(result):
    raise ValueError(label + " must be finite; got " + repr(value))
  return result


def native_boolean(value, label):
  """Return a switch only when the caller supplied ``True`` or ``False``."""
  if type(label) is not str or not label.strip():
    raise ValueError("boolean setting label must be a nonempty string")
  if type(value) is not bool:
    raise ValueError(
      label + " must be a native Python boolean; got " + repr(value))
  return value


def direct_child_filename(value, label):
  """Return one portable filename that cannot leave its stated folder.

  A supporting file named in YAML belongs beside that YAML.  Absolute paths,
  parent traversal, and either platform's path separator would let the same
  configuration read a different file depending on the machine or working
  directory.  Refusing those spellings keeps the configuration self-contained
  and makes the file included in the request unambiguous.
  """
  if type(label) is not str or not label.strip():
    raise ValueError("filename setting label must be a nonempty string")
  if type(value) is not str or not value.strip() or value != value.strip():
    raise ValueError(label + " must be one nonempty filename; got " + repr(value))
  if "/" in value or "\\" in value or value in (".", ".."):
    raise ValueError(
      label + " must name a direct child of the YAML folder; got "
      + repr(value))
  if any(unicodedata.category(character).startswith("C")
         for character in value):
    raise ValueError(label + " must not contain control characters")
  path = Path(value)
  if path.is_absolute() or path.parent != Path(".") or path.name != value:
    raise ValueError(
      label + " must name a direct child of the YAML folder; got "
      + repr(value))
  windows_path = PureWindowsPath(value)
  forbidden = '<>:"/\\|?*'
  reserved = {"CON", "PRN", "AUX", "NUL"}
  reserved.update("COM" + str(index) for index in range(1, 10))
  reserved.update("LPT" + str(index) for index in range(1, 10))
  windows_stem = value.split(".", 1)[0].upper()
  if windows_path.drive or any(character in forbidden for character in value) \
      or value.endswith((".", " ")) or windows_stem in reserved:
    raise ValueError(
      label + " must be one portable filename without a drive, reserved "
      "name, or platform-specific character; got " + repr(value))
  return value


def validate_train_args(train_args, extra_keys, uniform):
  """Return the ordered parameter names from one complete ``train_args``.

  Every generator family has ``probe`` and ``ord`` settings.  A family may
  add fields such as a multipole range through ``extra_keys``.  Gaussian MCMC
  sampling additionally needs a fiducial point and a parameter covariance
  filename.  Uniform sampling does not accept those Gaussian-only fields.

  Requiring the exact set of keys catches misspellings instead of silently
  ignoring them.  Requiring one inner list for ``ord`` also preserves the
  parameter order used by the saved table.
  """
  native_boolean(uniform, "uniform sampling switch")
  if type(train_args) is not dict:
    raise ValueError("train_args must be a YAML mapping")

  if type(extra_keys) not in (list, tuple):
    raise ValueError("family train_args keys must be a native list or tuple")
  family_keys = []
  for key in extra_keys:
    if type(key) is not str or not key.strip():
      raise ValueError("family train_args key must be a nonempty string")
    if key in family_keys or key in ("probe", "ord", "fiducial",
                                     "params_covmat_file"):
      raise ValueError("family train_args key is repeated or reserved: "
                       + repr(key))
    family_keys.append(key)

  expected = {"probe", "ord", *family_keys}
  if not uniform:
    expected.update(("fiducial", "params_covmat_file"))
  observed = set(train_args)
  if observed != expected or any(type(key) is not str for key in train_args):
    missing = sorted(expected - observed)
    unknown = sorted(repr(key) for key in observed - expected)
    details = []
    if missing:
      details.append("missing " + repr(missing))
    if unknown:
      details.append("unknown " + repr(unknown))
    raise ValueError(
      "train_args must contain exactly the fields for this generator ("
      + "; ".join(details) + ")")

  probe = train_args["probe"]
  if type(probe) is not str or not probe.strip():
    raise ValueError("train_args.probe must be a nonempty string")

  names = _ordered_names(train_args["ord"], "train_args.ord", wrapped=True)
  if not uniform:
    if type(train_args["fiducial"]) is not dict:
      raise ValueError("train_args.fiducial must be a YAML mapping")
    covariance_path = train_args["params_covmat_file"]
    direct_child_filename(
      covariance_path, "train_args.params_covmat_file")
  return names


def load_parameter_covariance(path, sampled_names):
  """Read and select a named parameter covariance matrix.

  The first line must have the form ``# name1 name2``.  The remaining rows
  must form one finite, symmetric, positive-diagonal square matrix with one
  row and column per header name.  The file may contain parameters that are
  not sampled in this run; the returned ``float64`` matrix follows the exact
  order in ``sampled_names``.
  """
  names = _ordered_names(sampled_names, "sampled parameter names")
  try:
    covariance_path = Path(path)
  except TypeError as error:
    raise ValueError("parameter covariance path must be a path or string") \
      from error
  try:
    with covariance_path.open("r", encoding="utf-8") as stream:
      first_line = stream.readline()
  except (OSError, TypeError) as error:
    raise ValueError(
      "cannot read parameter covariance file " + repr(str(path))) from error

  header = first_line.strip().split()
  if len(header) < 2 or header[0] != "#":
    raise ValueError(
      "parameter covariance first line must be '# name ...'")
  covariance_names = header[1:]
  if any(not name for name in covariance_names):
    raise ValueError("parameter covariance header contains an empty name")
  repeated = _first_repeat(covariance_names)
  if repeated is not None:
    raise ValueError(
      "parameter covariance header repeats name " + repr(repeated))

  try:
    matrix = np.loadtxt(covariance_path, dtype=np.float64, ndmin=2)
  except (OSError, TypeError, ValueError) as error:
    raise ValueError(
      "parameter covariance body must contain a rectangular numeric table") \
      from error
  if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError(
      "parameter covariance body must be a square two-dimensional matrix; "
      "got shape " + repr(matrix.shape))
  if matrix.shape[0] != len(covariance_names):
    raise ValueError(
      "parameter covariance header names and matrix dimensions disagree: "
      + str(len(covariance_names)) + " names for shape " + repr(matrix.shape))
  if not np.isfinite(matrix).all():
    raise ValueError("parameter covariance matrix must contain only finite values")
  # Text writers can represent the same computed covariance with a few final
  # binary digits of roundoff on opposite sides of the diagonal.  Accept only
  # that scale-aware floating-point noise, then make the accepted matrix
  # exactly symmetric.  A scientifically different upper/lower entry still
  # refuses instead of being silently repaired.
  symmetry_tolerance = 64.0 * np.finfo(np.float64).eps
  if not np.allclose(
      matrix, matrix.T,
      rtol=symmetry_tolerance,
      atol=0.0):
    raise ValueError(
      "parameter covariance matrix must be symmetric within floating-point "
      "roundoff")
  # Compute each off-diagonal midpoint once and assign that identical bit
  # pattern to both positions. ``0.5 * (a + b)`` can overflow when two valid
  # values are close to the largest float64 value, while computing the two
  # sides separately can round them differently near the smallest subnormal
  # values.
  upper_rows, upper_columns = np.triu_indices(matrix.shape[0], k=1)
  upper_values = matrix[upper_rows, upper_columns]
  lower_values = matrix[upper_columns, upper_rows]
  midpoints = upper_values + 0.5 * (lower_values - upper_values)
  matrix[upper_rows, upper_columns] = midpoints
  matrix[upper_columns, upper_rows] = midpoints
  if not np.isfinite(matrix).all():
    raise ValueError(
      "parameter covariance became nonfinite while normalizing roundoff")
  if np.any(np.diag(matrix) <= 0.0):
    raise ValueError("parameter covariance diagonal entries must be positive")

  positions = {name: index for index, name in enumerate(covariance_names)}
  missing = [name for name in names if name not in positions]
  if missing:
    raise ValueError(
      "parameter covariance header is missing sampled names " + repr(missing))
  indices = [positions[name] for name in names]
  return np.asarray(matrix[np.ix_(indices, indices)], dtype=np.float64)


def convert_prior_bounds(hard_bounds, finite_bounds, dtype=np.float32):
  """Convert Cobaya prior bounds without turning a finite edge into infinity.

  Cobaya may intentionally use infinity for an open hard-prior endpoint.  A
  very large *finite* endpoint has a different meaning and must not become an
  apparent open endpoint merely because the generator stores parameters in a
  narrower floating dtype.  The finite-confidence interval must remain fully
  finite after the same conversion.
  """
  try:
    hard = np.asarray(hard_bounds, dtype=np.float64)
    finite = np.asarray(finite_bounds, dtype=np.float64)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("Cobaya prior bounds must be numeric arrays") from error
  if hard.ndim != 2 or hard.shape[1:] != (2,) or finite.shape != hard.shape:
    raise ValueError(
      "Cobaya hard and finite prior bounds must have matching (rows, 2) "
      "shapes; got " + repr(hard.shape) + " and " + repr(finite.shape))
  if np.isnan(hard).any() or not (hard[:, 0] < hard[:, 1]).all():
    raise ValueError(
      "Cobaya hard prior bounds must be ordered and may contain infinity "
      "only as an open endpoint")
  if not np.isfinite(finite).all() \
      or not (finite[:, 0] < finite[:, 1]).all():
    raise ValueError(
      "Cobaya finite prior bounds must contain one finite increasing "
      "interval per sampled parameter")

  try:
    with np.errstate(over="ignore", invalid="ignore"):
      converted_hard = np.array(hard, copy=True, dtype=dtype)
      converted_finite = np.array(finite, copy=True, dtype=dtype)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("generator parameter dtype must be a real floating dtype") \
      from error
  if not np.issubdtype(converted_hard.dtype, np.floating) \
      or not np.issubdtype(converted_finite.dtype, np.floating):
    raise ValueError("generator parameter dtype must be a real floating dtype")

  finite_hard = np.isfinite(hard)
  if not np.isfinite(converted_hard[finite_hard]).all():
    raise ValueError(
      "a finite Cobaya hard-prior endpoint became nonfinite in the "
      "generator parameter dtype")
  if not np.array_equal(np.isposinf(converted_hard), np.isposinf(hard)) \
      or not np.array_equal(np.isneginf(converted_hard), np.isneginf(hard)):
    raise ValueError(
      "an intentional open Cobaya hard-prior endpoint changed during "
      "conversion")
  if not np.isfinite(converted_finite).all() \
      or not (converted_finite[:, 0] < converted_finite[:, 1]).all():
    raise ValueError(
      "Cobaya finite prior bounds collapse or become nonfinite in the "
      "generator parameter dtype")
  return converted_hard, converted_finite


def validate_fiducial(mapping, names, dtype=np.float32):
  """Return the finite fiducial vector in the saved parameter order.

  The values are checked before and after conversion to the requested storage
  dtype.  The second check catches a very large finite Python number that
  would become infinity in ``float32``.
  """
  ordered = _ordered_names(names, "sampled parameter names")
  if type(mapping) is not dict:
    raise ValueError("fiducial settings must be a YAML mapping")
  missing = [name for name in ordered if name not in mapping]
  if missing:
    raise ValueError("fiducial settings are missing parameters " + repr(missing))
  raw = [finite_number(mapping[name], "fiducial value for " + name)
         for name in ordered]
  try:
    with np.errstate(over="ignore", invalid="ignore"):
      converted = np.asarray(raw, dtype=dtype)
  except (TypeError, ValueError) as error:
    raise ValueError("fiducial storage dtype must be a numeric NumPy dtype") \
      from error
  if converted.ndim != 1 or converted.shape[0] != len(ordered):
    raise ValueError("fiducial conversion did not produce one value per parameter")
  if not np.issubdtype(converted.dtype, np.floating):
    raise ValueError("fiducial storage dtype must be a real floating dtype")
  if not np.isfinite(converted).all():
    raise ValueError(
      "fiducial values must remain finite after conversion to "
      + str(converted.dtype))
  return converted


def parameter_labels(info, names):
  """Return display labels, using the parameter name when LaTeX is absent.

  Cobaya does not require a ``latex`` field.  A missing, ``None``, or blank
  field therefore has the honest and readable fallback of the parameter name.
  A non-string value is refused because passing it to a table writer would
  create an ambiguous or malformed sidecar later in the run.
  """
  ordered = _ordered_names(names, "sampled parameter names")
  if not isinstance(info, Mapping):
    raise ValueError("parameter information must be a mapping")
  labels = []
  for name in ordered:
    if name not in info:
      raise ValueError("parameter information is missing " + repr(name))
    record = info[name]
    if not isinstance(record, Mapping):
      raise ValueError(
        "parameter information for " + repr(name) + " must be a mapping")
    label = record.get("latex")
    if label is None or (type(label) is str and not label.strip()):
      labels.append(name)
    elif type(label) is not str:
      raise ValueError(
        "latex label for " + repr(name) + " must be a string, null, or absent")
    else:
      cleaned = label.strip()
      if any(unicodedata.category(character).startswith("C")
             for character in cleaned):
        raise ValueError(
          "latex label for " + repr(name)
          + " must not contain control characters or line breaks")
      labels.append(cleaned)
  return labels


def select_unique_rows(samples, log_prob, requested, ndim, rng,
                       storage_dtype=np.float32):
  """Choose exactly ``requested`` distinct finite sample rows.

  MCMC chains can revisit a point.  Two raw float64 rows can also become the
  same row when the generator stores parameters as float32.  Publishing fewer
  stored rows than requested would make the dataset size depend on chance and
  would break a user's explicit request.  This function finds duplicates at
  the actual storage precision and refuses the operation when too few remain.
  The supplied NumPy-style random generator chooses among the remaining rows
  without changing their original values or probabilities.
  """
  requested = native_integer(requested, "requested sample count", minimum=1)
  ndim = native_integer(ndim, "sample dimension", minimum=1)
  try:
    sample_array = np.asarray(samples, dtype=np.float64)
    probability_array = np.asarray(log_prob, dtype=np.float64)
  except (TypeError, ValueError) as error:
    raise ValueError("samples and log probabilities must be numeric arrays") \
      from error
  if sample_array.ndim != 2 or sample_array.shape[1] != ndim:
    raise ValueError(
      "samples must have shape (rows, " + str(ndim) + "); got "
      + repr(sample_array.shape))
  if probability_array.ndim != 1 \
      or probability_array.shape[0] != sample_array.shape[0]:
    raise ValueError(
      "log probabilities must be a one-dimensional array with one value per "
      "sample row")
  if not np.isfinite(sample_array).all():
    raise ValueError("sample rows must contain only finite values")
  if not np.isfinite(probability_array).all():
    raise ValueError("log probabilities must contain only finite values")

  try:
    with np.errstate(over="ignore", invalid="ignore"):
      stored_samples = np.asarray(sample_array, dtype=storage_dtype)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("sample storage dtype must be a real floating dtype") \
      from error
  if not np.issubdtype(stored_samples.dtype, np.floating):
    raise ValueError("sample storage dtype must be a real floating dtype")
  if not np.isfinite(stored_samples).all():
    raise ValueError(
      "sample rows must remain finite at the published storage precision")

  _, first_indices = np.unique(
    stored_samples, axis=0, return_index=True)
  # Preserve the earlier generator's raw row values and first matching log
  # probability while using stored rows to decide whether they are distinct
  # in the dataset that readers receive.
  unique_samples = sample_array[first_indices]
  unique_probabilities = probability_array[first_indices]
  if unique_samples.shape[0] < requested:
    raise ValueError(
      "MCMC produced only " + str(unique_samples.shape[0])
      + " unique rows, fewer than the requested " + str(requested)
      + "; no dataset was published")
  choice = getattr(rng, "choice", None)
  if not callable(choice):
    raise ValueError("sample selector requires a random generator with choice")
  try:
    selected = np.asarray(choice(
      np.arange(unique_samples.shape[0]),
      size=requested,
      replace=False))
  except Exception as error:
    raise ValueError("random generator could not select the unique rows") \
      from error
  if selected.shape != (requested,) \
      or not np.issubdtype(selected.dtype, np.integer) \
      or len(set(selected.tolist())) != requested \
      or np.any(selected < 0) \
      or np.any(selected >= unique_samples.shape[0]):
    raise ValueError("random generator returned an invalid unique-row selection")
  return (unique_samples[selected].copy(),
          unique_probabilities[selected].copy())


def _ordered_names(value, label, wrapped=False):
  """Return a checked native list while preserving the written order."""
  if wrapped:
    if type(value) is not list or len(value) != 1:
      raise ValueError(label + " must be a list containing one parameter list")
    value = value[0]
  if type(value) is not list or not value:
    raise ValueError(label + " must be a nonempty native list")
  seen = set()
  result = []
  for name in value:
    if type(name) is not str or not name.strip():
      raise ValueError(label + " entries must be nonempty native strings")
    if any(character.isspace() or
           unicodedata.category(character).startswith("C")
           for character in name):
      raise ValueError(
        label + " entries must be one visible token without whitespace or "
        "control characters; got " + repr(name))
    if name in seen:
      raise ValueError(label + " repeats parameter " + repr(name))
    seen.add(name)
    result.append(name)
  return result


def _first_repeat(values):
  """Return the first repeated value, or ``None`` when all are unique."""
  seen = set()
  for value in values:
    if value in seen:
      return value
    seen.add(value)
  return None
