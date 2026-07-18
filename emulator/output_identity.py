"""Build the stable filename identity shared by training and diagnostics.

A readable filename such as ``resmlp_ntrain50000`` is not enough to identify
one scientific result.  Two runs can use the same model and row count while
learning different spectra, background quantities, scalar outputs, datasets,
or compositions.  This module reduces the complete executed description to a
short stable suffix while retaining a small family-and-product prefix for a
human reader.

The digest never depends on a checkout path, a timestamp, a random artifact
identifier, or dictionary insertion order.  Production callers must provide
the two immutable dataset records after staging has added the exact selected
row order.  Low-level fixture saves may omit those records; that allowance is
for isolated tests that do not represent a production training run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np


OUTPUT_IDENTITY_SCHEMA = 1
OUTPUT_IDENTITY_DIGEST_BYTES = 16
_DIGEST_DOMAIN = b"emulators-code-v2-output-identity-v1\x00"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_PART_RE = re.compile(r"[^a-z0-9]+")
_STATE_DIGEST_DOMAIN = b"emulators-code-v2-tensor-state-v1\x00"


def _plain_value(value, *, where):
  """Return one deterministic JSON value or reject an ambiguous object."""
  if value is None or type(value) in (bool, int, str):
    return value
  if type(value) is float:
    if not math.isfinite(value):
      raise ValueError(where + " contains a nonfinite number")
    return value
  if isinstance(value, (list, tuple)):
    return [_plain_value(item, where=where + "[]") for item in value]
  if type(value) is dict:
    out = {}
    for key, item in value.items():
      if type(key) is not str or not key:
        raise TypeError(where + " keys must be nonempty native strings")
      out[key] = _plain_value(item, where=where + "." + key)
    return out
  # NumPy scalar values appear in a few resolved test fixtures.  Convert only
  # scalar objects with a native ``item`` result; arrays remain a loud error.
  item_method = getattr(value, "item", None)
  if callable(item_method):
    native = item_method()
    if native is not value:
      return _plain_value(native, where=where)
  raise TypeError(
    where + " must contain only YAML/JSON scalar values, lists, and mappings; "
    "got " + type(value).__name__)


def _canonical_json(value):
  """Encode one checked value with stable key order and no whitespace."""
  return json.dumps(
    _plain_value(value, where="output identity"),
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True)


def digest_canonical_output_identity(canonical_json):
  """Hash one canonical identity subject with the format's version marker."""
  if type(canonical_json) is not str:
    raise TypeError("canonical output identity must be native text")
  try:
    encoded = canonical_json.encode("ascii")
  except UnicodeEncodeError as exc:
    raise ValueError("canonical output identity must use ASCII JSON") from exc
  return hashlib.sha256(_DIGEST_DOMAIN + encoded).hexdigest()


def validate_saved_output_identity(canonical_json, digest):
  """Validate a saved canonical subject and return its decoded mapping."""
  _require_digest(digest, where="saved output identity")
  if type(canonical_json) is not str:
    raise TypeError("saved output identity JSON must be native text")
  try:
    subject = json.loads(canonical_json)
  except (TypeError, ValueError) as exc:
    raise ValueError("saved output identity is not valid JSON") from exc
  if type(subject) is not dict or subject.get("schema") != OUTPUT_IDENTITY_SCHEMA:
    raise ValueError("saved output identity does not use schema 1")
  if _canonical_json(subject) != canonical_json:
    raise ValueError("saved output identity JSON is not in canonical form")
  observed = digest_canonical_output_identity(canonical_json)
  if observed != digest:
    raise ValueError(
      "saved output identity digest does not match its canonical record")
  return subject


def _require_digest(value, *, where):
  if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
    raise ValueError(where + " must be one lowercase SHA-256 digest")
  return value


def _slug(value):
  """Turn one scientific label into a short portable filename component."""
  if type(value) is not str or not value.strip():
    raise ValueError("output identity needs a nonempty family/product label")
  slug = _SLUG_PART_RE.sub("-", value.strip().lower()).strip("-")
  if not slug:
    raise ValueError("output identity label has no portable letters or digits")
  return slug[:40].rstrip("-")


def _required_text(value, *, where):
  if type(value) is not str or not value.strip():
    raise ValueError(where + " must be nonempty native text")
  return value


def _source_identity(data):
  """Return the train source identity when Cocoa published one."""
  sources = data.get("_dataset_sources")
  if type(sources) is not dict:
    return None
  train = sources.get("train")
  if type(train) is not dict:
    return None
  identity = train.get("identity")
  return identity if type(identity) is dict else None


def digest_cmb_covariance_inputs(ell, sigma, fiducial_cl):
  """Fingerprint the exact covariance arrays used by one CMB experiment.

  The covariance filename is local bookkeeping and may change when a project
  moves.  These three arrays are the scientific values read from that file:
  the multipole axis, whitening scale, and fiducial spectrum.  Normalizing
  their byte order makes the digest stable across machines.
  """
  arrays = (
    ("ell", np.asarray(ell, dtype="<i8")),
    ("sigma", np.asarray(sigma, dtype="<f8")),
    ("fiducial_cl", np.asarray(fiducial_cl, dtype="<f8")),
  )
  digest = hashlib.sha256(b"emulators-code-v2-cmb-covariance-input-v1\x00")
  for name, values in arrays:
    if values.ndim != 1:
      raise ValueError(
        "CMB covariance input " + name + " must be one-dimensional; got "
        + repr(values.shape))
    if not np.all(np.isfinite(values)):
      raise ValueError("CMB covariance input " + name + " must be finite")
    values = np.ascontiguousarray(values)
    encoded_name = name.encode("ascii")
    digest.update(len(encoded_name).to_bytes(2, "big"))
    digest.update(encoded_name)
    digest.update(int(values.size).to_bytes(8, "big"))
    digest.update(values.tobytes(order="C"))
  return digest.hexdigest()


def digest_tensor_state(state, *, where="tensor state"):
  """Fingerprint every named tensor in one embedded model state.

  The digest includes each name, dtype, shape, and byte sequence in sorted
  name order.  It accepts ordinary NumPy arrays as well as PyTorch tensors so
  the writer can hash live weights and the reader can independently hash the
  same arrays directly from HDF5 before importing a model implementation.
  """
  if type(state) is not dict:
    raise TypeError(where + " must be a plain name-to-tensor mapping")
  digest = hashlib.sha256(_STATE_DIGEST_DOMAIN)
  for name in sorted(state):
    if type(name) is not str or not name:
      raise TypeError(where + " keys must be nonempty native strings")
    value = state[name]
    detach = getattr(value, "detach", None)
    if callable(detach):
      value = detach()
      cpu = getattr(value, "cpu", None)
      if not callable(cpu):
        raise TypeError(where + "." + name + " cannot be moved to the CPU")
      value = cpu()
      numpy_method = getattr(value, "numpy", None)
      if not callable(numpy_method):
        raise TypeError(where + "." + name + " is not a tensor array")
      value = numpy_method()
    try:
      array = np.asarray(value)
    except Exception as exc:
      raise TypeError(where + "." + name + " is not a tensor array") from exc
    if array.dtype.hasobject:
      raise TypeError(where + "." + name + " has an object dtype")
    # HDF5 returns native-endian arrays.  Converting multibyte values to
    # little endian makes the same state portable across host byte orders.
    dtype = array.dtype
    if dtype.itemsize > 1:
      dtype = dtype.newbyteorder("<")
      array = array.astype(dtype, copy=False)
    array = np.ascontiguousarray(array)
    record = json.dumps(
      {"dtype": array.dtype.str, "name": name, "shape": list(array.shape)},
      ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    raw = array.tobytes(order="C")
    digest.update(len(record).to_bytes(8, "big"))
    digest.update(record)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
  return digest.hexdigest()


def _family_product(data, *, require_executed_inputs):
  """Return a readable family, product, and path-free scientific descriptor."""
  source_identity = _source_identity(data) or {}
  if "outputs" in data:
    outputs = data["outputs"]
    if not isinstance(outputs, (list, tuple)) or not outputs:
      raise ValueError("data.outputs must contain at least one scalar name")
    checked = []
    for name in outputs:
      if type(name) is not str or not name:
        raise ValueError("data.outputs must contain nonempty native strings")
      checked.append(name)
    product = "-".join(checked)
    return "scalar", product, {"outputs": checked}

  if "cmb" in data:
    block = data["cmb"]
    if type(block) is not dict:
      raise ValueError("data.cmb must be a mapping")
    spectrum = _required_text(
      block.get("spectrum"), where="data.cmb.spectrum")
    covariance_digest = block.get("_covariance_input_sha256")
    if covariance_digest is None and require_executed_inputs:
      raise ValueError(
        "production CMB output identity needs the executed covariance "
        "fingerprint created after its ell, sigma, and fiducial spectrum "
        "were loaded")
    if covariance_digest is not None:
      covariance_digest = _require_digest(
        covariance_digest, where="executed CMB covariance")
    descriptor = {
      key: block[key]
      for key in ("spectrum", "amplitude_law", "as_name", "tau_name",
                  "as_ref", "tau_ref")
      if key in block
    }
    descriptor["covariance_input_sha256"] = covariance_digest
    return "cmb", spectrum, descriptor

  if "grid" in data:
    block = data["grid"]
    if type(block) is not dict:
      raise ValueError("data.grid must be a mapping")
    quantity = _required_text(
      block.get("quantity"), where="data.grid.quantity")
    descriptor = {
      key: block[key]
      for key in ("quantity", "units", "law", "offset")
      if key in block
    }
    return "baosn", quantity, descriptor

  if "grid2d" in data:
    block = data["grid2d"]
    if type(block) is not dict:
      raise ValueError("data.grid2d must be a mapping")
    quantity = _required_text(
      block.get("quantity"), where="data.grid2d.quantity")
    descriptor = {
      key: block[key]
      for key in ("quantity", "units", "law", "k_stride")
      if key in block
    }
    return "mps", quantity, descriptor

  probe = source_identity.get("probe")
  if type(probe) is not str or not probe:
    # A direct-library fixture has no published dataset.  Its generic product
    # remains distinct through its recipes and fixed scientific record; a
    # production driver asks for published selection records below and cannot
    # take this fallback.
    probe = "data-vector"
  descriptor = {
    "probe": probe,
    "cosmolike_data_dir": data.get("cosmolike_data_dir"),
    "cosmolike_dataset": data.get("cosmolike_dataset"),
  }
  return "cosmolike", probe, descriptor


def _stable_generation_pin(pin, *, split):
  """Remove path spelling while retaining every authenticated source fact."""
  if type(pin) is not dict or pin.get("schema") != 1:
    raise ValueError(split + " dataset source pin must use schema 1")
  selection = pin.get("selection")
  if type(selection) is not dict or selection.get("schema") != 1:
    raise ValueError(
      split + " dataset source pin has no schema-1 staged row selection")
  _require_digest(
    selection.get("row_order_sha256"),
    where=split + " staged row order")
  _require_digest(pin.get("active_sha256"), where=split + " active record")
  _require_digest(
    pin.get("manifest_sha256"), where=split + " dataset manifest")
  members = pin.get("members")
  if type(members) is not dict or not members:
    raise ValueError(split + " dataset source pin has no authenticated members")
  stable_members = {}
  for role, member in members.items():
    if type(role) is not str or not role or type(member) is not dict:
      raise ValueError(split + " dataset members must be named mappings")
    size = member.get("size")
    if type(size) is not int or size < 0:
      raise ValueError(split + " dataset member " + role + " has invalid size")
    stable_members[role] = {
      "size": size,
      "sha256": _require_digest(
        member.get("sha256"), where=split + " dataset member " + role),
    }
  return {
    "schema": 1,
    "slot_id": pin.get("slot_id"),
    "slot": pin.get("slot"),
    "generation": pin.get("generation"),
    "active_sha256": pin["active_sha256"],
    "manifest_sha256": pin["manifest_sha256"],
    "identity": pin.get("identity"),
    "members": stable_members,
    "selection": selection,
  }


def _staged_selection(data, *, require_published_selection):
  sources = data.get("_dataset_sources")
  if sources is None:
    if require_published_selection:
      raise ValueError(
        "production output identity needs data._dataset_sources after both "
        "training and validation rows have been staged")
    return {"fixture_without_published_dataset": True}
  if type(sources) is not dict or sources.get("schema") != 1:
    raise ValueError("data._dataset_sources must use schema 1")
  return {
    "schema": 1,
    "train": _stable_generation_pin(sources.get("train"), split="training"),
    "validation": _stable_generation_pin(
      sources.get("validation"), split="validation"),
  }


def _path_free_reuse_block(block, *, label):
  """Replace a reusable source path with its authenticated pair binding."""
  if type(block) is not dict:
    raise ValueError(label + " resolved record must be a mapping")
  out = dict(block)
  out.pop("from", None)
  artifact_id = out.get("source_artifact_id")
  checkpoint_sha256 = out.get("source_checkpoint_sha256")
  if type(artifact_id) is not str or not re.fullmatch(
      r"[0-9a-f]{32}", artifact_id):
    raise ValueError(
      label + " source needs its authenticated source_artifact_id")
  _require_digest(
    checkpoint_sha256, where=label + " source checkpoint")
  return out


def _path_free_training_recipe(resolved_train):
  if type(resolved_train) is not dict:
    raise TypeError("resolved_train must be a plain mapping")
  out = dict(resolved_train)
  for label in ("finetune", "transfer"):
    if label in out:
      out[label] = _path_free_reuse_block(out[label], label=label)
  return out


def build_output_identity(
    *,
    data,
    resolved_train,
    resolved_model,
    resolved_rescale,
    composition_mode,
    transfer_refined,
    resolved_pce,
    resolved_transfer,
    require_published_selection=False):
  """Return the shared, versioned identity for one completed training run.

  The returned mapping contains the full digest, the readable filename tag,
  and the canonical JSON subject that produced the digest.  A caller may save
  that JSON as provenance, but must not edit it and retain the old digest.
  """
  if type(data) is not dict:
    raise TypeError("output identity data must be a plain mapping")
  if type(resolved_model) is not dict:
    raise TypeError("resolved_model must be a plain mapping")
  if resolved_rescale not in ("none", "rescaled", "residual"):
    raise ValueError(
      "output identity resolved_rescale must be none, rescaled, or residual")
  if composition_mode not in ("plain", "npce", "transfer"):
    raise ValueError(
      "output identity composition_mode must be plain, npce, or transfer")
  if type(transfer_refined) is not bool:
    raise TypeError("output identity transfer_refined must be a native bool")

  family, product, product_record = _family_product(
    data, require_executed_inputs=require_published_selection)
  path_free_transfer = None
  if resolved_transfer is not None:
    path_free_transfer = _path_free_reuse_block(
      resolved_transfer, label="transfer")
  subject = {
    "schema": OUTPUT_IDENTITY_SCHEMA,
    "family": family,
    "product": product,
    "product_record": product_record,
    "model_recipe": resolved_model,
    "training_recipe": _path_free_training_recipe(resolved_train),
    "loss_recipe": {"rescale": resolved_rescale},
    "staged_selection": _staged_selection(
      data, require_published_selection=require_published_selection),
    "composition": {
      "mode": composition_mode,
      "transfer_refined": transfer_refined,
      "pce": resolved_pce,
      "transfer": path_free_transfer,
    },
  }
  canonical = _canonical_json(subject)
  digest = digest_canonical_output_identity(canonical)
  prefix = _slug(family) + "-" + _slug(product)
  return {
    "schema": OUTPUT_IDENTITY_SCHEMA,
    "family": family,
    "product": product,
    "sha256": digest,
    "tag": prefix + "-" + digest[:2 * OUTPUT_IDENTITY_DIGEST_BYTES],
    "canonical_json": canonical,
  }


def build_experiment_output_identity(experiment):
  """Build the production identity after one experiment finishes staging.

  ``EmulatorExperiment.run`` materializes the model and training recipes and
  adds the selected-row fingerprints to ``experiment.data``.  Calling earlier
  would describe the request rather than the run that actually completed.
  """
  resolved_train = getattr(experiment, "resolved_train", None)
  resolved_model = getattr(experiment, "resolved_model", None)
  pce_opts = getattr(experiment, "pce_opts", None)
  transfer_base = getattr(experiment, "_transfer_base", None)
  if pce_opts is not None and transfer_base is not None:
    raise ValueError("one run cannot be both NPCE and transfer")
  if pce_opts is not None:
    composition_mode = "npce"
  elif transfer_base is not None:
    composition_mode = "transfer"
  else:
    composition_mode = "plain"
  transfer_refined = (
    getattr(experiment, "_transfer_pretrained_base", None) is not None)
  return build_output_identity(
    data=getattr(experiment, "data", None),
    resolved_train=resolved_train,
    resolved_model=resolved_model,
    resolved_rescale=getattr(experiment, "rescale", None),
    composition_mode=composition_mode,
    transfer_refined=transfer_refined,
    resolved_pce=(dict(pce_opts) if pce_opts is not None else None),
    resolved_transfer=(resolved_train.get("transfer")
                       if transfer_base is not None
                       and type(resolved_train) is dict else None),
    require_published_selection=True)


def require_same_output_identity(expected, observed):
  """Refuse when a caller's filename identity differs from the saved run."""
  if type(expected) is not dict:
    raise TypeError("output_identity must be the mapping from its builder")
  for key in ("schema", "family", "product", "sha256", "tag",
              "canonical_json"):
    if expected.get(key) != observed.get(key):
      raise ValueError(
        "output_identity disagrees with the executed run at " + repr(key)
        + "; rebuild the filename from the final staged experiment")
