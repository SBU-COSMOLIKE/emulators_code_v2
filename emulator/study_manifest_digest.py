"""Study manifests become stable bytes and digests through one strict writer.

Canonical JSON means UTF-8 JSON with object keys sorted, no optional spaces,
and the ordinary JSON spellings for booleans and null. Only JSON-native data
is admitted. In particular, a tuple is not silently changed into an array and
a non-finite float is not written as an implementation-specific token.

PS: A digest is a short identity computed from all bytes in a value or file.
Changing any byte changes the identity with overwhelming probability.
"""

import hashlib
import json
import math


def _validate_json_value(value, location, active_containers):
  """Refuse values that the JSON writer would otherwise coerce or invent.

  Arguments:
    value = one value in the manifest tree.
    location = human-readable location of the value in that tree.
    active_containers = identities of containers currently being visited.

  Returns:
    None after the value and all of its children are valid JSON data.
  """
  value_type = type(value)
  if value is None or value_type is bool or value_type is int:
    return
  if value_type is float:
    if not math.isfinite(value):
      raise ValueError(
        "study manifest contains a non-finite number at " + location
        + "; replace it with a finite JSON number or an explicit string")
    return
  if value_type is str:
    try:
      value.encode(
        encoding="utf-8",
        errors="strict")
    except UnicodeEncodeError as error:
      raise ValueError(
        "study manifest contains text that is not valid UTF-8 at " + location
        + "; replace the unpaired surrogate before writing the manifest"
      ) from error
    return

  if value_type is not list and value_type is not dict:
    raise TypeError(
      "study manifest contains non-JSON data at " + location + ": "
      + value_type.__name__
      + "; use only null, booleans, numbers, strings, lists, and objects")

  container_id = id(value)
  if container_id in active_containers:
    raise ValueError(
      "study manifest contains a container cycle at " + location
      + "; replace the cycle with an ordinary JSON value")
  active_containers.add(container_id)
  try:
    if value_type is list:
      for index in range(len(value)):
        child_location = location + "[" + str(index) + "]"
        _validate_json_value(
          value=value[index],
          location=child_location,
          active_containers=active_containers)
      return

    for key in value:
      if type(key) is not str:
        raise TypeError(
          "study manifest contains a non-string object key at " + location
          + ": " + repr(key)
          + "; JSON object keys must be strings")
      _validate_json_value(
        value=key,
        location=location + " key " + repr(key),
        active_containers=active_containers)
      _validate_json_value(
        value=value[key],
        location=location + "[" + repr(key) + "]",
        active_containers=active_containers)
  finally:
    active_containers.remove(container_id)


def canonical_json(value):
  """Serialize JSON data once so equal manifests have equal bytes.

  Object insertion order and optional whitespace do not affect the result.
  The returned text always has a strict UTF-8 encoding. Values that would
  require JSON's coercions or non-standard number tokens are refused.

  Arguments:
    value = JSON-native value to serialize.

  Returns:
    compact canonical JSON text with object keys in sorted order.
  """
  _validate_json_value(
    value=value,
    location="$",
    active_containers=set())
  # value is the JSON subject; json.dumps has no keyword for that argument.
  return json.dumps(
    value,
    allow_nan=False,
    ensure_ascii=False,
    separators=(
      ",",
      ":",
    ),
    sort_keys=True)


def manifest_digest(manifest):
  """Digest the canonical scientific identity of one tuning study.

  Arguments:
    manifest = JSON-native canonical study manifest.

  Returns:
    the digest as the string ``sha256:<hex>``.
  """
  text = canonical_json(value=manifest)
  digest = hashlib.sha256()
  digest.update(text.encode(
    encoding="utf-8",
    errors="strict"))
  return "sha256:" + digest.hexdigest()


def file_digest(path):
  """Digest every byte of one scientific input without loading it at once.

  Arguments:
    path = path to the input file whose exact bytes identify it.

  Returns:
    the digest as the string ``sha256:<hex>``.
  """
  digest = hashlib.sha256()
  bytes_per_chunk = 1 << 20
  with open(path, mode="rb") as input_file:
    while True:
      chunk = input_file.read(bytes_per_chunk)
      if not chunk:
        break
      digest.update(chunk)
  return "sha256:" + digest.hexdigest()
