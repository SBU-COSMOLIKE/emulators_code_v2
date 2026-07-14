"""Canonical study manifests keep unlike tuning experiments apart.

An Optuna journal can contain trials from several invocations.  Before a
worker trains, this module records the scientific inputs and fixed choices
that make those trials comparable.  A resumed study must carry the identical
record.  Operational choices such as the worker count are deliberately absent
because they change how work is scheduled, not which experiment is run.

PS: a manifest is a complete, machine-readable list of the facts that identify
one tuning study.  A digest is the SHA-256 summary of that list.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

from .manifest_digest import canonical_json, file_digest, manifest_digest


MANIFEST_VERSION = 1
STUDY_MANIFEST_ATTR = "study_manifest"
STUDY_MANIFEST_DIGEST_ATTR = "study_manifest_sha256"

_OPERATIONAL_KEYS = {
  "gpu_count",
  "n_gpus",
  "n_trials",
  "quiet",
  "ram_frac",
  "timeout",
  "worker_count",
}

_MIGRATION = (
  "Use a new journal path for this experiment.  To migrate comparable old "
  "trials, create a new journal and copy them only after verifying their "
  "scientific manifest; a legacy journal cannot be blessed from the current "
  "configuration."
)


def _plain_value(value, field):
  """Convert one value to the stable JSON types used by a manifest.

  Arguments:
    value = value to materialize.  Mappings, sequences, paths, NumPy scalars,
            and tensor-like values with ``tolist`` are accepted.
    field = dotted location used in a specific validation error.

  Returns:
    the value represented with dictionaries, lists, strings, booleans,
    integers, floating-point numbers, and null values only.
  """
  if isinstance(value, Mapping):
    result = {}
    keys = list(value.keys())
    keys.sort(key=str)
    for key in keys:
      if not isinstance(key, str):
        raise TypeError(field + " has a non-string mapping key " + repr(key))
      child_field = field + "." + key
      result[key] = _plain_value(
        value=value[key],
        field=child_field)
    return result

  if isinstance(value, (list, tuple)):
    result = []
    for index, item in enumerate(value):
      child_field = field + "[" + str(index) + "]"
      result.append(_plain_value(
        value=item,
        field=child_field))
    return result

  if isinstance(value, Path):
    return str(value)
  if value is None or isinstance(value, (str, bool, int)):
    return value
  if isinstance(value, float):
    if not math.isfinite(value):
      raise ValueError(field + " must be finite; got " + repr(value))
    return value

  list_method = getattr(value, "tolist", None)
  if callable(list_method):
    plain = list_method()
    if plain is not value:
      return _plain_value(
        value=plain,
        field=field)

  item_method = getattr(value, "item", None)
  if callable(item_method):
    plain = item_method()
    if plain is not value:
      return _plain_value(
        value=plain,
        field=field)

  raise TypeError(
    field + " has unsupported manifest value " + repr(type(value).__name__))


def _without_operational_values(value):
  """Copy a configuration while removing scheduling-only keys recursively.

  Arguments:
    value = resolved configuration value to copy.

  Returns:
    a JSON-compatible copy without operational keys.
  """
  if isinstance(value, dict):
    result = {}
    for key in value:
      if key in _OPERATIONAL_KEYS:
        continue
      result[key] = _without_operational_values(value=value[key])
    return result
  if isinstance(value, list):
    result = []
    for item in value:
      result.append(_without_operational_values(value=item))
    return result
  return value


def _data_roots(data):
  """Find directories that can resolve nested data-file names.

  Arguments:
    data = the resolved data configuration.

  Returns:
    existing directories named directly or implied by absolute file values.
  """
  roots = set()

  def visit(value):
    if isinstance(value, dict):
      for child in value.values():
        visit(value=child)
      return
    if isinstance(value, list):
      for child in value:
        visit(value=child)
      return
    if not isinstance(value, str):
      return
    path = Path(value).expanduser()
    if path.is_dir():
      roots.add(path.resolve())
    elif path.is_absolute() and path.is_file():
      roots.add(path.resolve().parent)

  visit(value=data)
  ordered = list(roots)
  ordered.sort(key=str)
  return ordered


def _resolved_file(value, roots):
  """Resolve one data-config string when it names an existing file.

  Arguments:
    value = candidate string from the data configuration.
    roots = directories inferred from already resolved data paths.

  Returns:
    the absolute file path, or None when the value does not name a file.
  """
  if not isinstance(value, str):
    return None
  candidate = Path(value).expanduser()
  if candidate.is_file():
    return candidate.resolve()
  if candidate.is_absolute():
    return None
  for root in roots:
    rooted = root / candidate
    if rooted.is_file():
      return rooted.resolve()
  return None


def _sidecar_candidates(path):
  """Return possible scientific sidecars adjacent to one input file.

  Arguments:
    path = absolute scientific input path.

  Returns:
    possible ``.paramnames`` and ``.facts.yaml`` paths.  Cobaya chain names
    such as ``sample.1.txt`` also try the shared ``sample`` stem.
  """
  stems = [path.with_suffix("")]
  first_stem = stems[0]
  if first_stem.suffix[1:].isdigit():
    stems.append(first_stem.with_suffix(""))

  candidates = []
  for stem in stems:
    candidates.append(Path(str(stem) + ".paramnames"))
    candidates.append(Path(str(stem) + ".facts.yaml"))
  return candidates


def _scientific_input_files(data):
  """Collect existing files named anywhere in the data configuration.

  Arguments:
    data = fully resolved data block.

  Returns:
    sorted absolute paths, including existing adjacent scientific sidecars.
  """
  roots = _data_roots(data=data)
  found = set()

  def visit(value):
    if isinstance(value, dict):
      for child in value.values():
        visit(value=child)
      return
    if isinstance(value, list):
      for child in value:
        visit(value=child)
      return
    path = _resolved_file(
      value=value,
      roots=roots)
    if path is not None:
      found.add(path)

  visit(value=data)
  primary = list(found)
  for path in primary:
    candidates = _sidecar_candidates(path=path)
    for candidate in candidates:
      if candidate.is_file():
        found.add(candidate.resolve())

  ordered = list(found)
  ordered.sort(key=str)
  return ordered


def _file_identities(paths):
  """Materialize path and byte digest for a sequence of files.

  Arguments:
    paths = file paths whose identity belongs in the study.

  Returns:
    sorted identity dictionaries with ``path`` and ``digest`` keys.
  """
  unique = set()
  for path_value in paths:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
      raise FileNotFoundError("manifest identity file not found: " + str(path))
    unique.add(path)

  ordered = list(unique)
  ordered.sort(key=str)
  identities = []
  for path in ordered:
    identity = {
      "path": str(path),
      "digest": file_digest(path),
    }
    identities.append(identity)
  return identities


def build_study_manifest(*,
                         family,
                         probe,
                         study_name,
                         thresholds,
                         fixed_config,
                         search_space,
                         default_trial,
                         rescale,
                         activation,
                         implementation_identity,
                         additional_scientific_files=(),
                         resolved_scientific_values=None):
  """Build the one canonical scientific identity for an Optuna study.

  Arguments:
    family              = explicit emulator family identity.
    probe               = observable scored by the study.
    study_name          = stable name selected by the family resolver.
    thresholds          = ordered delta-chi2 report thresholds.  The first
                          threshold defines the minimized objective.
    fixed_config        = fully resolved configuration with range leaves
                          replaced by the fixed values the experiment uses.
    search_space        = exact nested range schema from the raw train block.
    default_trial       = known-default search point enqueued as the control.
    rescale             = resolved command-line analytic rescaling choice.
    activation          = resolved command-line activation choice.
    implementation_identity = versioned semantic compatibility record for
                              the shared tuner and selected family.
    additional_scientific_files = resolved files consumed outside the data
                                  block, such as fine-tune and transfer
                                  artifact pairs.
    resolved_scientific_values = runtime-resolved scientific facts whose
                                 source text alone is ambiguous, or None.

  Returns:
    the canonical manifest dictionary.  The caller computes its digest once.
  """
  plain_thresholds = _plain_value(
    value=thresholds,
    field="objective.thresholds")
  if not isinstance(plain_thresholds, list) or len(plain_thresholds) == 0:
    raise ValueError("objective.thresholds must contain at least one value")

  plain_config = _plain_value(
    value=fixed_config,
    field="fixed_config")
  plain_config = _without_operational_values(value=plain_config)
  data = plain_config.get("data", {})
  scientific_paths = _scientific_input_files(data=data)
  scientific_paths.extend(additional_scientific_files)

  manifest = {
    "manifest_version": MANIFEST_VERSION,
    "study": {
      "family": _plain_value(value=family, field="study.family"),
      "probe": _plain_value(value=probe, field="study.probe"),
      "name": _plain_value(value=study_name, field="study.name"),
    },
    "objective": {
      "direction": "minimize",
      "metric": "fraction_delta_chi2_above_threshold",
      "thresholds": plain_thresholds,
      "selection_rule": {
        "primary": "lowest_fraction_above_first_threshold",
        "tie_break": "lowest_median_delta_chi2",
      },
    },
    "fixed_config": plain_config,
    "search_space": _plain_value(
      value=search_space,
      field="search_space"),
    "default_trial": _plain_value(
      value=default_trial,
      field="default_trial"),
    "cli_fixed": {
      "rescale": _plain_value(value=rescale, field="cli_fixed.rescale"),
      "activation": _plain_value(
        value=activation,
        field="cli_fixed.activation"),
    },
    "scientific_inputs": _file_identities(paths=scientific_paths),
    "resolved_scientific_values": _plain_value(
      value={} if resolved_scientific_values is None
      else resolved_scientific_values,
      field="resolved_scientific_values"),
    "implementation": _plain_value(
      value=implementation_identity,
      field="implementation"),
  }
  return _plain_value(
    value=manifest,
    field="manifest")


def _different_fields(left, right, field="manifest"):
  """Name every dotted manifest field whose stored and current values differ.

  Arguments:
    left  = stored manifest value.
    right = current manifest value.
    field = dotted name of the values being compared.

  Returns:
    sorted dotted field names.  List positions use square brackets.
  """
  differences = []
  if type(left) is not type(right):
    differences.append(field)
    return differences
  if isinstance(left, dict):
    keys = set(left.keys()) | set(right.keys())
    ordered = list(keys)
    ordered.sort()
    for key in ordered:
      child_field = field + "." + key
      if key not in left or key not in right:
        differences.append(child_field)
        continue
      child_differences = _different_fields(
        left=left[key],
        right=right[key],
        field=child_field)
      differences.extend(child_differences)
    return differences
  if isinstance(left, list):
    if len(left) != len(right):
      differences.append(field + ".length")
    common = min(len(left), len(right))
    for index in range(common):
      child_field = field + "[" + str(index) + "]"
      child_differences = _different_fields(
        left=left[index],
        right=right[index],
        field=child_field)
      differences.extend(child_differences)
    return differences
  if left != right:
    differences.append(field)
  return differences


def _canonical_json(manifest):
  """Serialize one manifest with the exact digest ordering.

  Arguments:
    manifest = canonical manifest dictionary.

  Returns:
    compact JSON text with sorted keys.
  """
  return canonical_json(value=manifest)


def bind_study_manifest(*, study, manifest, digest, initialize):
  """Create or authenticate the manifest attributes on one Optuna study.

  Arguments:
    study    = opened Optuna study with user_attrs, trials, and set_user_attr.
    manifest = current canonical manifest dictionary.
    digest   = digest already computed for the current manifest.
    initialize = true only when the caller just created this exact study.

  Returns:
    None.  An explicitly new empty study receives both attributes; a matching
    resumed study is unchanged. Loaded legacy studies are never inferred new.
  """
  if type(initialize) is not bool:
    raise TypeError("initialize must be an explicit boolean creation fact")
  current = _plain_value(
    value=manifest,
    field="manifest")
  expected_digest = manifest_digest(current)
  if digest != expected_digest:
    raise ValueError(
      "current study manifest digest does not match its contents: got "
      + repr(digest) + ", expected " + repr(expected_digest))

  attrs = study.user_attrs
  has_manifest = STUDY_MANIFEST_ATTR in attrs
  has_digest = STUDY_MANIFEST_DIGEST_ATTR in attrs
  if not has_manifest and not has_digest:
    if initialize and len(attrs) == 0 and len(study.trials) == 0:
      study.set_user_attr(
        STUDY_MANIFEST_ATTR,
        _canonical_json(manifest=current))
      study.set_user_attr(
        STUDY_MANIFEST_DIGEST_ATTR,
        digest)
      return None
    raise RuntimeError(
      "the journal study has no scientific manifest and is a legacy study. "
      + _MIGRATION)

  if not has_manifest or not has_digest:
    missing = STUDY_MANIFEST_ATTR
    if has_manifest:
      missing = STUDY_MANIFEST_DIGEST_ATTR
    raise RuntimeError(
      "the journal study has an incomplete scientific manifest; missing "
      + repr(missing) + ". " + _MIGRATION)

  stored_text = attrs[STUDY_MANIFEST_ATTR]
  try:
    stored = json.loads(stored_text)
  except (TypeError, ValueError) as error:
    raise RuntimeError(
      "the journal study_manifest attribute is not canonical JSON. "
      + _MIGRATION) from error
  stored = _plain_value(
    value=stored,
    field="stored_manifest")
  if stored_text != _canonical_json(manifest=stored):
    raise RuntimeError(
      "the journal study_manifest attribute is not canonical JSON. "
      + _MIGRATION)
  stored_digest = attrs[STUDY_MANIFEST_DIGEST_ATTR]
  recorded_digest = manifest_digest(stored)
  if stored_digest != recorded_digest:
    raise RuntimeError(
      "the journal study manifest digest does not match its recorded "
      "contents. " + _MIGRATION)

  differences = _different_fields(
    left=stored,
    right=current)
  if differences or stored_digest != digest:
    if not differences:
      differences.append("manifest_digest")
    raise RuntimeError(
      "the journal belongs to a different scientific study; differing "
      "fields: " + ", ".join(differences) + ". " + _MIGRATION)
  return None
