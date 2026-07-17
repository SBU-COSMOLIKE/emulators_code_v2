"""Crash-safe filesystem publication for immutable dataset generations.

The numerical generator still needs a later integration step before it writes
through this module.  This file provides the small, CPU-only transaction that
that integration will use:

* a logical dataset maps to one stable slot under ``chains/.datasets``;
* writers build ordinary files in a private, mutable draft directory;
* sealing copies an exact member census into new exclusive inodes, hashes the
  copies, and makes the installed generation read-only;
* one canonical ``active.json`` record is replaced last; and
* readers pin that record once, then verify and use only its immutable files;
  and
* continuation makes new private writable copies while retaining the original
  active-record SHA-256 value for a later lost-update check.

There are deliberately no per-file "current" links.  Switching several links
one by one would let a reader combine members from different generations.
Published files are never reopened writable; resume and append must copy a
verified generation into a new draft before changing any byte.

Publication is a barrier, not a way to snapshot writers that are still active.
The caller must first stop and close every writer (including MPI ranks and
memmaps).  As a second line of defence, sealing keeps every source file open
before copying the first byte.  Once all members have been acquired, it refuses
the draft if any source token or the census changes through the final copy.
That proves the accepted bytes match one on-disk state after the last open;
the external writer barrier is what makes that state one scientific run.

The trust boundary is the repository owner and compliant concurrent writers.
This module rejects malformed entries, links, unsafe POSIX modes, and lost
updates between compliant publishers.  It does not defend against the same
account changing ancestors or ACLs while an operation is running.  Normal
``fsync`` durability is used, with Darwin ``F_FULLFSYNC`` for regular files
when available. Before the final slot-directory sync returns, portable power
recovery may show the old active record, the new active record, or no active
record. Any surviving authenticated record still names one complete
generation.
"""

from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import warnings

from compute_data_vectors.dataset_manifest import build_dataset_member_map


PUBLICATION_SCHEMA = 1
ACTIVE_NAME = "active.json"
MANIFEST_NAME = "manifest.json"
FILES_NAME = "files"
DATASETS_NAME = ".datasets"
LOCATORS_NAME = "locators"

_MAX_ACTIVE_BYTES = 64 * 1024
_MAX_LOCATOR_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_JSON_INTEGER_DIGITS = 1024
_MAX_JSON_INTEGER_BITS = 3402
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_RE = re.compile(r"^gen-[0-9a-f]{32}$")
_SLOT_RE = re.compile(r"^slot-[0-9a-f]{64}$")
_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._/-][a-z0-9]+)*$")
_PORTABLE_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DatasetPublicationError(RuntimeError):
  """A dataset generation cannot be trusted or safely published."""


@dataclass(frozen=True)
class DatasetSlot:
  """Stable location and identity for one logical dataset output family."""

  chains_dir: Path
  slot_id: str
  descriptor_json: bytes

  @property
  def descriptor(self):
    """Return a fresh copy of the canonical slot descriptor."""
    return json.loads(self.descriptor_json.decode("utf-8"))

  @property
  def path(self):
    return self.chains_dir / DATASETS_NAME / self.slot_id

  @property
  def active_path(self):
    return self.path / ACTIVE_NAME

  @property
  def work_path(self):
    return self.path / "work"

  @property
  def generations_path(self):
    return self.path / "generations"


@dataclass(frozen=True)
class DatasetLocator:
  """Stable request information for one logical parameter-chain filename.

  A locator says which dataset slot and scientific request own a familiar
  chain filename such as ``params.1.txt``.  It deliberately does not say which
  immutable generation is active.  ``load_located_generation`` consults the
  slot's active record at the moment a consumer starts and returns that one
  verified generation.
  """

  slot: DatasetSlot
  logical_parameter: str
  path: Path
  identity_json: bytes
  members_json: bytes

  @property
  def identity(self):
    """Return a fresh copy of the canonical scientific request."""
    return json.loads(self.identity_json.decode("utf-8"))

  @property
  def members(self):
    """Return a fresh copy of the canonical role-to-filename map."""
    return json.loads(self.members_json.decode("utf-8"))


@dataclass(frozen=True)
class DraftGeneration:
  """One private mutable generation that has not been made active."""

  slot: DatasetSlot
  generation: str
  path: Path

  @property
  def files_path(self):
    return self.path / FILES_NAME

  def member_path(self, relative_path):
    """Return a safe path inside this draft's files directory.

    Parent directories are created privately when a future family needs a
    nested member.  Current generator families use one directory and retain
    their exact historical basenames.
    """
    relative = _member_relative_path(relative_path)
    result = self.files_path.joinpath(*relative.parts)
    _make_private_parents(result.parent, stop=self.files_path)
    return result


@dataclass(frozen=True)
class PublishedMember:
  """One authenticated regular file in a published generation."""

  role: str
  relative_path: str
  path: Path
  size: int
  sha256: str


@dataclass(frozen=True)
class ActiveGeneration:
  """A reader's pinned, verified view of one active generation."""

  slot: DatasetSlot
  generation: str
  active_sha256: str
  manifest_sha256: str
  identity_json: bytes
  members: tuple

  @property
  def identity(self):
    """Return a fresh copy of the canonical run identity."""
    return json.loads(self.identity_json.decode("utf-8"))

  def member(self, role):
    """Return the authenticated member carrying ``role``."""
    matches = [member for member in self.members if member.role == role]
    if len(matches) != 1:
      raise KeyError(role)
    return matches[0]


@dataclass(frozen=True)
class ContinuationDraft:
  """A verified immutable source and its independent mutable copy."""

  source: ActiveGeneration
  draft: DraftGeneration


def canonical_json_bytes(value):
  """Return the one accepted UTF-8 JSON representation of ``value``.

  Keys are sorted, insignificant whitespace is absent, non-ASCII text is
  escaped, non-finite floats are refused, and one trailing LF terminates the
  record.  Type checks are recursive so unsupported objects cannot acquire an
  accidental string representation.
  """
  _validate_json_value(value, path="$", seen=set())
  try:
    text = json.dumps(
      value,
      allow_nan=False,
      ensure_ascii=True,
      separators=(",", ":"),
      sort_keys=True)
  except (TypeError, ValueError) as exc:
    raise DatasetPublicationError(
      "dataset record is not canonical JSON: " + str(exc)) from exc
  return (text + "\n").encode("ascii")


def derive_dataset_slot(chains_dir, *, params_stem, dvs_stem, fail_stem,
                        dataset_mode, family):
  """Derive one stable slot from the logical output stems and family.

  The descriptor stores portable basenames rather than an absolute checkout
  path, so moving a complete ``chains`` directory does not change its identity.
  All three stems must be direct children of that directory; silently dropping
  a caller-supplied parent would otherwise make two requested outputs collide.
  """
  chains = Path(os.path.abspath(os.fspath(chains_dir)))
  if dataset_mode not in ("full", "chain-only"):
    raise DatasetPublicationError(
      "dataset mode must be 'full' or 'chain-only'; got "
      + repr(dataset_mode))
  if type(family) is not str or not _FAMILY_RE.fullmatch(family):
    raise DatasetPublicationError(
      "dataset family must be a portable lower-case name; got " + repr(family))

  stems = {}
  for label, supplied in (("params", params_stem), ("dvs", dvs_stem),
                          ("fail", fail_stem)):
    if type(supplied) is not str or not supplied:
      raise DatasetPublicationError(
        label + " stem must be a nonempty string; got " + repr(supplied))
    candidate = Path(os.path.abspath(supplied))
    if candidate.parent != chains:
      raise DatasetPublicationError(
        label + " stem must be a direct child of " + str(chains)
        + "; got " + str(candidate))
    _portable_part(candidate.name, label + " stem")
    stems[label] = candidate.name
  if len({value.casefold() for value in stems.values()}) != len(stems):
    raise DatasetPublicationError(
      "parameter, data-vector, and failure stems must remain distinct on "
      "case-insensitive filesystems")

  descriptor = {
    "dataset_mode": dataset_mode,
    "family": family,
    "logical_stems": stems,
    "schema": PUBLICATION_SCHEMA,
  }
  descriptor_json = canonical_json_bytes(descriptor)
  slot_id = "slot-" + hashlib.sha256(descriptor_json).hexdigest()
  return DatasetSlot(
    chains_dir=chains,
    slot_id=slot_id,
    descriptor_json=descriptor_json)


def install_dataset_locator(slot, *, identity, members):
  """Install the stable lookup record for one logical parameter chain.

  The record is selected by the exact ``parameters.chain`` basename derived
  from the request.  A retry with the same slot, identity, and member map is a
  no-op.  A different request may not take over an existing logical filename.
  The record contains no generation name or active-record digest; consumers
  resolve the slot's current generation separately when they start.
  """
  _require_slot(slot)
  identity_json, members_json, member_map = _validated_locator_request(
    slot, identity, members)
  logical_parameter = member_map["parameters.chain"]
  locator_dir = _ensure_locator_root(slot.chains_dir)
  path = locator_dir / _locator_filename(logical_parameter)
  record = {
    "identity": json.loads(identity_json.decode("ascii")),
    "members": json.loads(members_json.decode("ascii")),
    "schema": PUBLICATION_SCHEMA,
    "slot": slot.descriptor,
    "slot_id": slot.slot_id,
  }
  payload = canonical_json_bytes(record)
  if len(payload) > _MAX_LOCATOR_BYTES:
    raise DatasetPublicationError(
      "dataset locator exceeds its " + str(_MAX_LOCATOR_BYTES)
      + " byte limit")

  lock_fd = _acquire_locator_lock(locator_dir, logical_parameter)
  temporary = None
  try:
    if os.path.lexists(path):
      located = _read_dataset_locator(
        slot.chains_dir, logical_parameter, path)
      _require_same_locator(located, slot, identity_json, members_json)
      return located

    temporary = locator_dir / (
      ".locator-" + secrets.token_hex(16) + ".tmp")
    _write_new_regular_file(temporary, payload, final_mode=0o444)
    if os.path.lexists(path):
      located = _read_dataset_locator(
        slot.chains_dir, logical_parameter, path)
      _require_same_locator(located, slot, identity_json, members_json)
      _best_effort_remove_locator_temporary(temporary, locator_dir)
      temporary = None
      return located
    try:
      os.replace(temporary, path)
    except OSError as exc:
      raise DatasetPublicationError(
        "could not install dataset locator " + str(path) + ": "
        + str(exc)) from exc
    temporary = None
    _fsync_directory(locator_dir)
    return _read_dataset_locator(slot.chains_dir, logical_parameter, path)
  finally:
    _best_effort_remove_locator_temporary(temporary, locator_dir)
    _release_publish_lock(lock_fd)


def load_dataset_locator(chains_dir, *, logical_parameter):
  """Load one locator by a familiar parameter-chain path or basename.

  ``logical_parameter`` may be a visible basename such as ``params.1.txt`` or
  an absolute path directly below ``chains_dir``.  Parent traversal and paths
  from another chains directory are refused rather than silently reduced to a
  basename.
  """
  chains = _absolute_directory_path(chains_dir, "chains directory")
  logical_name = _logical_parameter_name(chains, logical_parameter)
  locator_dir = _require_locator_root(chains)
  path = locator_dir / _locator_filename(logical_name)
  return _read_dataset_locator(chains, logical_name, path)


def load_located_generation(locator):
  """Resolve and authenticate the active generation named by ``locator``."""
  if type(locator) is not DatasetLocator:
    raise DatasetPublicationError("expected a DatasetLocator")
  return load_active_generation(
    locator.slot,
    expected_identity=locator.identity,
    expected_members=locator.members)


def begin_dataset_generation(slot, generation=None):
  """Create a private draft for a fresh, resumed, or appended generation."""
  _require_slot(slot)
  generation_id = (
    "gen-" + secrets.token_hex(16) if generation is None else generation)
  _require_generation_id(generation_id)

  _ensure_slot_directories(slot)
  final_path = slot.generations_path / generation_id
  if os.path.lexists(final_path):
    raise DatasetPublicationError(
      "generation already exists and will not be overwritten: "
      + str(final_path))

  draft_name = ("draft-" + generation_id + "-" + secrets.token_hex(8))
  draft_path = slot.work_path / draft_name
  try:
    os.mkdir(draft_path, 0o700)
    os.mkdir(draft_path / FILES_NAME, 0o700)
    _fsync_directory(draft_path / FILES_NAME)
    _fsync_directory(draft_path)
    _fsync_directory(slot.work_path)
  except (OSError, DatasetPublicationError) as exc:
    _best_effort_remove_new_tree(draft_path, slot.work_path)
    raise DatasetPublicationError(
      "could not create private dataset draft " + str(draft_path)
      + ": " + str(exc)) from exc
  return DraftGeneration(slot=slot, generation=generation_id, path=draft_path)


def discard_dataset_draft(draft):
  """Remove one abandoned private draft without touching any generation.

  Failure/crash recovery may leave an unlisted source or sealed draft below
  ``slot/work``.  Callers may discard a known draft after deciding it cannot be
  resumed.  The function accepts only a verified private directory immediately
  below that slot's work directory and never follows symlinks.
  """
  _require_draft(draft)
  try:
    shutil.rmtree(draft.path)
    _fsync_directory(draft.slot.work_path)
  except OSError as exc:
    raise DatasetPublicationError(
      "could not discard dataset draft " + str(draft.path)
      + ": " + str(exc)) from exc


def _best_effort_discard_sealed(sealed):
  """Remove an uninstalled sealed tree without masking the primary failure."""
  if not os.path.lexists(sealed.path):
    return
  try:
    discard_dataset_draft(sealed)
  except Exception:
    pass


def _best_effort_remove_new_tree(path, parent):
  """Remove a newly-created partial draft tree without following a link."""
  if path.parent != parent or not os.path.lexists(path):
    return
  try:
    status = os.lstat(path)
    if stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode):
      shutil.rmtree(path)
      _fsync_directory(parent)
  except Exception:
    pass


def _best_effort_remove_temporary_active(path, slot):
  """Remove one uncommitted active-record temporary, if it still exists."""
  if path is None or path.parent != slot.path \
      or not path.name.startswith(".active-") \
      or not path.name.endswith(".tmp") or not os.path.lexists(path):
    return
  try:
    status = os.lstat(path)
    if stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode) \
        and status.st_nlink == 1:
      os.unlink(path)
      _fsync_directory(slot.path)
  except Exception:
    pass


def _best_effort_remove_locator_temporary(path, locator_dir):
  """Remove only the uninstalled locator temporary created by this call."""
  if path is None or path.parent != locator_dir \
      or not path.name.startswith(".locator-") \
      or not path.name.endswith(".tmp") or not os.path.lexists(path):
    return
  try:
    status = os.lstat(path)
    if stat.S_ISREG(status.st_mode) and not stat.S_ISLNK(status.st_mode) \
        and status.st_nlink == 1:
      os.unlink(path)
      _fsync_directory(locator_dir)
  except Exception:
    pass


def publish_dataset_generation(draft, *, identity, members,
                               expected_active_sha256,
                               checkpoint=None):
  """Seal ``draft`` and atomically make it the slot's active generation.

  ``members`` maps stable semantic roles to paths relative to ``draft/files``.
  Every regular file below that directory must be named exactly once; an extra,
  missing, linked, or special file refuses publication.  ``identity`` is the
  caller-owned scientific/run record and must be a JSON object.  The later
  generator integration will give that object its exact domain schema.  A
  draft inode is never installed directly: copying first prevents an old
  writable descriptor or memmap from changing the published bytes.

  ``expected_active_sha256`` is the SHA-256 of the complete canonical active
  record and acts as a compare-and-swap token.  ``None`` means the
  slot must have no active generation.  An append/resume publisher passes the
  digest returned by ``load_active_generation``.  A per-slot advisory lock keeps
  two compliant writers from both validating the same token and losing one
  update.

  The optional ``checkpoint`` callback is a crash-test boundary.  Production
  callers omit it.  It receives, in order, ``draft-durable``,
  ``generation-installed``, ``active-temp-durable``, ``active-replaced``, and
  ``active-directory-durable``.
  """
  _require_draft(draft)
  identity_json = canonical_json_bytes(identity)
  if type(identity) is not dict:
    raise DatasetPublicationError("dataset identity must be a JSON object")
  identity_snapshot = json.loads(identity_json.decode("ascii"))
  member_paths = _validate_member_map(members)
  if expected_active_sha256 is not None:
    _require_hex_digest(expected_active_sha256, "expected active record")
  if checkpoint is not None and not callable(checkpoint):
    raise DatasetPublicationError("publication checkpoint must be callable")

  _ensure_slot_directories(draft.slot)
  lock_fd = _acquire_publish_lock(draft.slot)
  sealed = None
  sealed_installed = False
  temp_active = None
  try:
    try:
      observed = _read_active_pointer_if_present(draft.slot)
      observed_revision = None if observed is None else observed[1]
      if observed_revision != expected_active_sha256:
        raise DatasetPublicationError(
          "active generation changed before publication: expected active record "
          + repr(expected_active_sha256) + ", observed "
          + repr(observed_revision))

      _require_exact_source_draft_census(draft, member_paths)
      sealed = _begin_sealed_generation(draft)
      records = _copy_member_files(draft, sealed, member_paths)
      manifest = {
        "generation": draft.generation,
        "identity": identity_snapshot,
        "members": records,
        "schema": PUBLICATION_SCHEMA,
        "slot": draft.slot.descriptor,
        "slot_id": draft.slot.slot_id,
      }
      manifest_bytes = canonical_json_bytes(manifest)
      if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise DatasetPublicationError(
          "dataset manifest exceeds its " + str(_MAX_MANIFEST_BYTES)
          + " byte limit")
      manifest_path = sealed.path / MANIFEST_NAME
      _write_new_regular_file(manifest_path, manifest_bytes, final_mode=0o444)
      manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

      _require_exact_sealed_census(sealed, member_paths)
      _fsync_tree_directories(sealed.path)
      _fsync_directory(draft.slot.work_path)
      _call_checkpoint(checkpoint, "draft-durable")

      final_path = draft.slot.generations_path / draft.generation
      if os.path.lexists(final_path):
        raise DatasetPublicationError(
          "generation appeared during publication and will not be overwritten: "
          + str(final_path))
      try:
        os.rename(sealed.path, final_path)
      except OSError as exc:
        raise DatasetPublicationError(
          "could not install immutable generation " + str(final_path)
          + ": " + str(exc)) from exc
      sealed_installed = True
      # macOS refuses to rename a directory after its owner-write bit is
      # removed.  Install the still-unreachable draft first, then make the
      # installed tree read-only and durable before the active record names it.
      _make_tree_read_only(final_path)
      _fsync_tree_directories(final_path)
      _fsync_directory(draft.slot.work_path)
      _fsync_directory(draft.slot.generations_path)
      _call_checkpoint(checkpoint, "generation-installed")

      active = {
        "generation": draft.generation,
        "manifest_sha256": manifest_sha256,
        "schema": PUBLICATION_SCHEMA,
        "slot_id": draft.slot.slot_id,
      }
      active_bytes = canonical_json_bytes(active)
      active_sha256 = hashlib.sha256(active_bytes).hexdigest()
      temp_active = draft.slot.path / (
        ".active-" + secrets.token_hex(16) + ".tmp")
      _write_new_regular_file(temp_active, active_bytes, final_mode=0o444)
      _call_checkpoint(checkpoint, "active-temp-durable")
      try:
        os.replace(temp_active, draft.slot.active_path)
      except OSError as exc:
        raise DatasetPublicationError(
          "could not switch the active dataset generation: "
          + str(exc)) from exc
      temp_active = None
      _call_checkpoint(checkpoint, "active-replaced")
      _fsync_directory(draft.slot.path)
      _call_checkpoint(checkpoint, "active-directory-durable")
    except BaseException:
      # Pre-install failures must not strand a full sealed duplicate in work.
      # Installed-but-inactive generations are retained for explicit recovery.
      _best_effort_remove_temporary_active(temp_active, draft.slot)
      if sealed is not None and not sealed_installed:
        _best_effort_discard_sealed(sealed)
      raise

    try:
      discard_dataset_draft(draft)
    except DatasetPublicationError as exc:
      # The active record is already durable.  Reporting publication failure
      # here would invite a retry with a stale CAS token, so retain the orphan
      # for explicit recovery and return the committed generation truthfully.
      try:
        warnings.warn(
          "dataset generation was published, but its mutable source draft "
          "could not be removed: " + str(exc), RuntimeWarning)
      except Exception:
        # Warning filters may promote warnings to exceptions.  Cleanup
        # diagnostics must never turn a committed publication into a reported
        # transaction failure.
        pass
  finally:
    try:
      _release_publish_lock(lock_fd)
    except Exception:
      # Lock release is cleanup, not part of the scientific transaction.  The
      # concrete helper already makes unlock/close best-effort; this outer
      # guard also prevents an injected/custom cleanup hook from hiding either
      # the primary refusal or a durably committed generation.
      pass

  published_members = tuple(
    PublishedMember(
      role=role,
      relative_path=record["path"][len(FILES_NAME) + 1:],
      path=(draft.slot.generations_path / draft.generation
            / record["path"]),
      size=record["size"],
      sha256=record["sha256"])
    for role, record in sorted(records.items()))
  return ActiveGeneration(
    slot=draft.slot,
    generation=draft.generation,
    active_sha256=active_sha256,
    manifest_sha256=manifest_sha256,
    identity_json=identity_json,
    members=published_members)


def load_active_generation(slot, *, expected_identity, expected_members):
  """Pin and authenticate one slot's current immutable generation.

  The active record is read exactly once.  A later publisher may switch the
  slot, but this reader continues using the already named immutable directory.
  Expected identity is compared by canonical bytes (so ``True`` never equals
  integer ``1`` by Python's loose equality).  The role-to-relative-path map
  must also match exactly: a correct digest under the wrong familiar basename
  is not the requested dataset.  The returned Paths pin a generation name, not
  open file descriptors.  Callers and future garbage collection must keep that
  immutable generation directory present for the whole consuming operation.
  """
  _require_slot(slot)
  _require_existing_slot_layout(slot)
  expected_identity_json = canonical_json_bytes(expected_identity)
  if type(expected_identity) is not dict:
    raise DatasetPublicationError("expected dataset identity must be an object")
  expected_member_map = _validate_member_map(expected_members)

  active, active_sha256 = _read_active_pointer(slot)
  generation_path = slot.generations_path / active["generation"]
  _require_read_only_directory(generation_path, "generation directory")
  manifest_path = generation_path / MANIFEST_NAME
  manifest_bytes = _read_regular_bytes(
    manifest_path, _MAX_MANIFEST_BYTES, "dataset manifest",
    require_read_only=True)
  observed_manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
  if observed_manifest_digest != active["manifest_sha256"]:
    raise DatasetPublicationError(
      "active record names manifest digest " + active["manifest_sha256"]
      + " but the immutable manifest has digest "
      + observed_manifest_digest)
  manifest = _parse_canonical_record(
    manifest_bytes, "dataset manifest", _MANIFEST_KEYS)
  _validate_manifest_header(slot, active, manifest)

  identity_json = canonical_json_bytes(manifest["identity"])
  if identity_json != expected_identity_json:
    raise DatasetPublicationError(
      "active dataset identity does not match the requested run identity")
  member_records = manifest["members"]
  if type(member_records) is not dict or not member_records:
    raise DatasetPublicationError(
      "dataset manifest members must be a nonempty object")
  for role in member_records:
    _member_role(role)
  if set(member_records) != set(expected_member_map):
    raise DatasetPublicationError(
      "active dataset member roles differ from the requested census: expected "
      + repr(sorted(expected_member_map)) + ", observed "
      + repr(sorted(member_records)))

  files_path = generation_path / FILES_NAME
  _require_read_only_directory(files_path, "generation files directory")
  parsed_records = []
  seen_paths = set()
  for role in sorted(member_records):
    record = member_records[role]
    _require_exact_keys(record, _MEMBER_KEYS, "member " + role)
    relative = _manifest_member_path(record["path"])
    if relative.as_posix() in seen_paths:
      raise DatasetPublicationError(
        "dataset manifest assigns one file to more than one role: "
        + relative.as_posix())
    seen_paths.add(relative.as_posix())
    if relative.as_posix() != expected_member_map[role]:
      raise DatasetPublicationError(
        "member " + role + " is stored under " + relative.as_posix()
        + " but the requested basename/path is " + expected_member_map[role])
    if type(record["size"]) is not int or type(record["size"]) is bool \
        or record["size"] < 0:
      raise DatasetPublicationError(
        "member " + role + " size must be a nonnegative native integer")
    _require_hex_digest(record["sha256"], "member " + role)
    parsed_records.append((role, record, relative))

  _require_exact_generation_census(generation_path, seen_paths)
  published_members = []
  for role, record, relative in parsed_records:
    path = files_path.joinpath(*relative.parts)
    size, digest = _hash_regular_file(
      path, label="member " + role, durable=False, require_read_only=True)
    if size != record["size"] or digest != record["sha256"]:
      raise DatasetPublicationError(
        "member " + role + " does not match its manifest size/digest")
    published_members.append(PublishedMember(
      role=role,
      relative_path=relative.as_posix(),
      path=path,
      size=size,
      sha256=digest))
  return ActiveGeneration(
    slot=slot,
    generation=active["generation"],
    active_sha256=active_sha256,
    manifest_sha256=active["manifest_sha256"],
    identity_json=identity_json,
    members=tuple(published_members))


def begin_dataset_continuation(slot, *, expected_identity, expected_members,
                               generation=None,
                               expected_active_sha256=None):
  """Copy one authenticated active generation into a private mutable draft.

  Arguments:
    slot: Location and fixed description of the saved dataset.
    expected_identity: Exact scientific settings and run settings expected in
      the saved manifest.
    expected_members: Exact mapping from each member's purpose to its relative
      path inside the generation.
    generation: Optional name for the new draft. Omission makes a fresh random
      name.
    expected_active_sha256: Optional digest of the exact active record a caller
      already inspected. When supplied, a newer active record refuses before a
      draft is created.

  The active generation is fully checked before a draft is created. Every
  source member then remains open until all copies and directory checks are
  complete. A source-member replacement, link, mode change, size change, byte
  change, or added/missing member refuses the operation. The named manifest
  bytes, digest, and full directory contents are read and checked again after
  the copies; the manifest file itself is not held open during the copy.

  Returns:
    A ``ContinuationDraft`` containing the checked read-only source and its
    separate writable draft. ``source.active_sha256`` is the SHA-256 value
    calculated from the exact saved ``active.json`` bytes. A later publication
    passes this value back so it can refuse if another writer selected a newer
    active generation in the meantime.

  Raises:
    DatasetPublicationError: The active generation, a source member, the final
      source recheck, or the private copy does not match the required state.

  Side effects:
    Success creates private mode-0700 directories and mode-0600 member copies.
    A refusal while this function prepares the continuation asks for
    best-effort removal of only that new draft. If removal itself fails, the
    partial draft may remain in the work folder for inspection. The active
    record and the published source generation are never changed by this
    function. A refusal during a later publication is outside this cleanup
    step and keeps the completed continuation draft.
  """
  source = load_active_generation(
    slot,
    expected_identity=expected_identity,
    expected_members=expected_members)
  if expected_active_sha256 is not None:
    _require_hex_digest(
      expected_active_sha256, "expected active dataset record")
    if source.active_sha256 != expected_active_sha256:
      raise DatasetPublicationError(
        "active dataset changed after the caller's read-only validation; "
        "refuse to create a continuation draft")
  draft = None
  try:
    draft = begin_dataset_generation(slot, generation=generation)
    _copy_active_members_to_draft(source, draft)
    return ContinuationDraft(source=source, draft=draft)
  except BaseException:
    if draft is not None:
      _best_effort_remove_new_tree(draft.path, draft.slot.work_path)
    raise


_ACTIVE_KEYS = frozenset((
  "generation", "manifest_sha256", "schema", "slot_id"))
_LOCATOR_KEYS = frozenset((
  "identity", "members", "schema", "slot", "slot_id"))
_MANIFEST_KEYS = frozenset((
  "generation", "identity", "members", "schema", "slot", "slot_id"))
_MEMBER_KEYS = frozenset(("path", "sha256", "size"))


def _validate_json_value(value, *, path, seen):
  if value is None or type(value) in (bool, str):
    return
  if type(value) is int:
    # Bound the binary value before decimal conversion so Python 3.9 cannot be
    # forced into an unbounded big-integer-to-string conversion.
    if value.bit_length() > _MAX_JSON_INTEGER_BITS \
        or len(str(abs(value))) > _MAX_JSON_INTEGER_DIGITS:
      raise DatasetPublicationError(
        path + " contains a JSON integer longer than "
        + str(_MAX_JSON_INTEGER_DIGITS) + " decimal digits")
    return
  if type(value) is float:
    if not math.isfinite(value):
      raise DatasetPublicationError(
        path + " contains a non-finite JSON number")
    return
  if type(value) is list:
    identity = id(value)
    if identity in seen:
      raise DatasetPublicationError(path + " contains a recursive sequence")
    seen.add(identity)
    try:
      for index, item in enumerate(value):
        _validate_json_value(item, path=path + "[" + str(index) + "]",
                             seen=seen)
    finally:
      seen.remove(identity)
    return
  if type(value) is dict:
    identity = id(value)
    if identity in seen:
      raise DatasetPublicationError(path + " contains a recursive object")
    seen.add(identity)
    try:
      for key, item in value.items():
        if type(key) is not str:
          raise DatasetPublicationError(
            path + " has a non-string object key " + repr(key))
        _validate_json_value(item, path=path + "." + key, seen=seen)
    finally:
      seen.remove(identity)
    return
  raise DatasetPublicationError(
    path + " contains unsupported JSON value " + repr(type(value).__name__))


def _parse_canonical_record(payload, label, expected_keys):
  duplicates = []

  def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
      if key in result:
        duplicates.append(key)
      result[key] = value
    return result

  def reject_constant(token):
    raise ValueError("non-finite JSON token " + token)

  def bounded_integer(token):
    digits = token[1:] if token.startswith("-") else token
    if len(digits) > _MAX_JSON_INTEGER_DIGITS:
      raise ValueError(
        "JSON integer exceeds " + str(_MAX_JSON_INTEGER_DIGITS)
        + " decimal digits")
    return int(token)

  try:
    value = json.loads(
      payload.decode("utf-8"),
      object_pairs_hook=reject_duplicate_pairs,
      parse_constant=reject_constant,
      parse_int=bounded_integer)
  except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
    raise DatasetPublicationError(label + " is not strict UTF-8 JSON: "
                                  + str(exc)) from exc
  if duplicates:
    raise DatasetPublicationError(
      label + " contains duplicate object keys: " + repr(duplicates))
  if type(value) is not dict:
    raise DatasetPublicationError(label + " must be a JSON object")
  _require_exact_keys(value, expected_keys, label)
  if canonical_json_bytes(value) != payload:
    raise DatasetPublicationError(
      label + " is not in the canonical JSON encoding")
  return value


def _require_exact_keys(value, expected, label):
  if type(value) is not dict:
    raise DatasetPublicationError(label + " must be a JSON object")
  observed = set(value)
  if observed != set(expected):
    raise DatasetPublicationError(
      label + " fields must be exactly " + repr(sorted(expected))
      + "; got " + repr(sorted(observed)))


def _portable_part(value, label):
  if type(value) is not str or not _PORTABLE_PART_RE.fullmatch(value):
    raise DatasetPublicationError(
      label + " must be one portable filename component; got " + repr(value))
  if value in (".", "..") or value.startswith("."):
    raise DatasetPublicationError(
      label + " may not be hidden or a traversal component; got " + repr(value))
  return value


def _member_relative_path(value):
  if type(value) is not str or not value or "\\" in value or "\x00" in value:
    raise DatasetPublicationError(
      "member path must be a nonempty canonical POSIX relative path; got "
      + repr(value))
  path = PurePosixPath(value)
  if path.is_absolute() or path.as_posix() != value:
    raise DatasetPublicationError(
      "member path must be canonical and relative; got " + repr(value))
  if not path.parts:
    raise DatasetPublicationError("member path may not be empty")
  for part in path.parts:
    _portable_part(part, "member path component")
  return path


def _manifest_member_path(value):
  if type(value) is not str or not value.startswith(FILES_NAME + "/"):
    raise DatasetPublicationError(
      "manifest member path must begin with '" + FILES_NAME + "/'; got "
      + repr(value))
  return _member_relative_path(value[len(FILES_NAME) + 1:])


def _member_role(value):
  if type(value) is not str or not _ROLE_RE.fullmatch(value):
    raise DatasetPublicationError(
      "member role must be a portable semantic name; got " + repr(value))
  if ".." in value.split("/"):
    raise DatasetPublicationError("member role may not traverse directories")
  return value


def _validate_member_map(members):
  if type(members) is not dict or not members:
    raise DatasetPublicationError(
      "publication members must be a nonempty role-to-path object")
  result = {}
  seen = set()
  for role, value in members.items():
    _member_role(role)
    relative = _member_relative_path(value).as_posix()
    if relative in seen:
      raise DatasetPublicationError(
        "one draft file cannot carry more than one member role: " + relative)
    seen.add(relative)
    result[role] = relative
  return result


def _require_slot(slot):
  if type(slot) is not DatasetSlot:
    raise DatasetPublicationError("expected a DatasetSlot")
  if type(slot.slot_id) is not str or not _SLOT_RE.fullmatch(slot.slot_id):
    raise DatasetPublicationError("invalid dataset slot id " + repr(slot.slot_id))
  descriptor = _parse_canonical_record(
    slot.descriptor_json, "slot descriptor",
    frozenset(("dataset_mode", "family", "logical_stems", "schema")))
  expected = "slot-" + hashlib.sha256(slot.descriptor_json).hexdigest()
  if expected != slot.slot_id:
    raise DatasetPublicationError(
      "slot id does not match its canonical descriptor")
  _require_schema_value(descriptor["schema"], "slot descriptor")
  if descriptor["schema"] != PUBLICATION_SCHEMA:
    raise DatasetPublicationError("unsupported slot descriptor schema")
  if descriptor["dataset_mode"] not in ("full", "chain-only"):
    raise DatasetPublicationError("slot descriptor has an invalid dataset mode")
  family = descriptor["family"]
  if type(family) is not str or not _FAMILY_RE.fullmatch(family):
    raise DatasetPublicationError("slot descriptor has an invalid family")
  stems = descriptor["logical_stems"]
  _require_exact_keys(stems, frozenset(("params", "dvs", "fail")),
                      "slot logical stems")
  for label, value in stems.items():
    _portable_part(value, "slot " + label + " stem")
  if len({value.casefold() for value in stems.values()}) != len(stems):
    raise DatasetPublicationError(
      "slot parameter, data-vector, and failure stems must remain distinct "
      "on case-insensitive filesystems")
  if not isinstance(slot.chains_dir, Path) or not slot.chains_dir.is_absolute():
    raise DatasetPublicationError("slot chains directory must be absolute")


def _absolute_directory_path(value, label):
  try:
    supplied = os.fspath(value)
  except TypeError as exc:
    raise DatasetPublicationError(
      label + " must be a filesystem path") from exc
  if type(supplied) is not str or not supplied:
    raise DatasetPublicationError(
      label + " must be a nonempty text filesystem path")
  return Path(os.path.abspath(supplied))


def _logical_parameter_name(chains, value):
  try:
    supplied = os.fspath(value)
  except TypeError as exc:
    raise DatasetPublicationError(
      "logical parameter chain must be a filesystem path") from exc
  if type(supplied) is not str or not supplied:
    raise DatasetPublicationError(
      "logical parameter chain must be a nonempty text filesystem path")
  path = Path(supplied)
  if path.is_absolute():
    if path.parent != chains:
      raise DatasetPublicationError(
        "logical parameter chain must be a direct child of " + str(chains)
        + "; got " + str(path))
  elif path.parent != Path("."):
    raise DatasetPublicationError(
      "relative logical parameter chain must be one basename; got "
      + repr(supplied))
  return _portable_part(path.name, "logical parameter chain")


def _locator_filename(logical_parameter):
  _portable_part(logical_parameter, "logical parameter chain")
  return logical_parameter + ".json"


def _validated_locator_request(slot, identity, members):
  if type(identity) is not dict:
    raise DatasetPublicationError("dataset locator identity must be an object")
  if type(members) is not dict:
    raise DatasetPublicationError(
      "dataset locator members must be a native role-to-path object")
  identity_json = canonical_json_bytes(identity)
  member_map = _validate_member_map(members)
  descriptor = slot.descriptor
  stems = descriptor["logical_stems"]
  try:
    expected = build_dataset_member_map(
      identity,
      params_stem=stems["params"],
      dvs_stem=stems["dvs"],
      fail_stem=stems["fail"])
  except ValueError as exc:
    raise DatasetPublicationError(
      "dataset locator has an invalid scientific request: " + str(exc)) from exc
  if identity["dataset_mode"] != descriptor["dataset_mode"] \
      or identity["family"] != descriptor["family"]:
    raise DatasetPublicationError(
      "dataset locator request does not match its slot mode and family")
  expected_map = _validate_member_map(expected)
  members_json = canonical_json_bytes(member_map)
  if members_json != canonical_json_bytes(expected_map):
    raise DatasetPublicationError(
      "dataset locator member map is not the exact map derived from its "
      "request and slot stems")
  return identity_json, members_json, member_map


def _read_dataset_locator(chains, logical_parameter, path):
  payload = _read_regular_bytes(
    path, _MAX_LOCATOR_BYTES, "dataset locator", require_read_only=True)
  record = _parse_canonical_record(
    payload, "dataset locator", _LOCATOR_KEYS)
  _require_schema_value(record["schema"], "dataset locator")
  if record["schema"] != PUBLICATION_SCHEMA:
    raise DatasetPublicationError("unsupported dataset locator schema")
  descriptor_json = canonical_json_bytes(record["slot"])
  slot = DatasetSlot(
    chains_dir=chains,
    slot_id=record["slot_id"],
    descriptor_json=descriptor_json)
  _require_slot(slot)
  identity_json, members_json, member_map = _validated_locator_request(
    slot, record["identity"], record["members"])
  if member_map["parameters.chain"] != logical_parameter:
    raise DatasetPublicationError(
      "dataset locator file name does not match its logical parameter chain")
  return DatasetLocator(
    slot=slot,
    logical_parameter=logical_parameter,
    path=path,
    identity_json=identity_json,
    members_json=members_json)


def _require_same_locator(locator, slot, identity_json, members_json):
  if locator.slot.slot_id != slot.slot_id \
      or locator.slot.descriptor_json != slot.descriptor_json \
      or locator.identity_json != identity_json \
      or locator.members_json != members_json:
    raise DatasetPublicationError(
      "logical parameter chain already has a different dataset locator: "
      + locator.logical_parameter)


def _require_draft(draft):
  if type(draft) is not DraftGeneration:
    raise DatasetPublicationError("expected a DraftGeneration")
  _require_slot(draft.slot)
  _require_generation_id(draft.generation)
  if draft.path.parent != draft.slot.work_path:
    raise DatasetPublicationError("draft is outside its slot work directory")
  _require_private_directory(draft.path, "draft directory")
  _require_private_directory(draft.files_path, "draft files directory")


def _require_generation_id(value):
  if type(value) is not str or not _GENERATION_RE.fullmatch(value):
    raise DatasetPublicationError(
      "generation id must be 'gen-' plus 32 lower-case hex digits; got "
      + repr(value))


def _require_hex_digest(value, label):
  if type(value) is not str or not _HEX64_RE.fullmatch(value):
    raise DatasetPublicationError(
      label + " digest must be 64 lower-case hex digits; got " + repr(value))


def _make_private_parents(path, *, stop):
  if path == stop:
    return
  pending = []
  current = path
  while current != stop:
    if stop not in current.parents:
      raise DatasetPublicationError("member parent escapes draft files directory")
    pending.append(current)
    current = current.parent
  for directory in reversed(pending):
    if os.path.lexists(directory):
      _require_private_directory(directory, "draft member directory")
    else:
      _ensure_private_directory(directory)


def _ensure_slot_directories(slot):
  if not slot.chains_dir.exists():
    raise DatasetPublicationError(
      "chains directory does not exist: " + str(slot.chains_dir))
  _require_private_directory(slot.chains_dir, "chains directory")
  datasets = slot.chains_dir / DATASETS_NAME
  _ensure_private_directory(datasets)
  _ensure_private_directory(slot.path)
  _ensure_private_directory(slot.work_path)
  _ensure_private_directory(slot.generations_path)


def _ensure_locator_root(chains):
  _require_private_directory(chains, "chains directory")
  datasets = chains / DATASETS_NAME
  _ensure_private_directory(datasets)
  locators = datasets / LOCATORS_NAME
  _ensure_private_directory(locators)
  return locators


def _require_locator_root(chains):
  _require_private_directory(chains, "chains directory")
  datasets = chains / DATASETS_NAME
  _require_private_directory(datasets, "dataset locator root")
  locators = datasets / LOCATORS_NAME
  _require_private_directory(locators, "dataset locator directory")
  return locators


def _require_existing_slot_layout(slot):
  _require_private_directory(slot.chains_dir, "chains directory")
  _require_private_directory(
    slot.chains_dir / DATASETS_NAME, "dataset publication root")
  _require_private_directory(slot.path, "dataset slot directory")
  _require_private_directory(
    slot.generations_path, "dataset generations directory")


def _ensure_private_directory(path):
  try:
    os.mkdir(path, 0o700)
  except FileExistsError:
    pass
  except OSError as exc:
    raise DatasetPublicationError(
      "could not create dataset publication directory " + str(path)
      + ": " + str(exc)) from exc
  _require_private_directory(path, "dataset publication directory")
  # Repeat both syncs even when the directory already exists.  A previous
  # attempt may have completed mkdir but failed while syncing its parent; the
  # retry must repair that incomplete durability proof.
  _fsync_directory(path)
  _fsync_directory(path.parent)


def _require_real_directory(path, label):
  try:
    status = os.lstat(path)
  except OSError as exc:
    raise DatasetPublicationError(label + " is unavailable: " + str(exc)) from exc
  if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
    raise DatasetPublicationError(label + " is not a real directory: " + str(path))
  return status


def _require_private_directory(path, label):
  status = _require_real_directory(path, label)
  if status.st_mode & 0o022:
    raise DatasetPublicationError(
      label + " is writable by group or other users: " + str(path))
  if (status.st_mode & 0o700) != 0o700:
    raise DatasetPublicationError(
      label + " must give its owner read, write, and search access: "
      + str(path))
  if hasattr(os, "getuid") and status.st_uid != os.getuid():
    raise DatasetPublicationError(
      label + " is not owned by the current user: " + str(path))
  return status


def _require_read_only_directory(path, label):
  status = _require_real_directory(path, label)
  if status.st_mode & 0o222:
    raise DatasetPublicationError(
      label + " is writable and therefore not an immutable generation: "
      + str(path))
  if (status.st_mode & 0o500) != 0o500:
    raise DatasetPublicationError(
      label + " must give its owner read and search access: " + str(path))
  return status


def _acquire_publish_lock(slot):
  path = slot.path / "publish.lock"
  flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags, 0o600)
  except OSError as exc:
    raise DatasetPublicationError(
      "could not open dataset publication lock: " + str(exc)) from exc
  try:
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
      raise DatasetPublicationError(
        "dataset publication lock is not one private regular file")
    if status.st_mode & 0o077:
      raise DatasetPublicationError(
        "dataset publication lock is accessible to another user")
    if hasattr(os, "getuid") and status.st_uid != os.getuid():
      raise DatasetPublicationError(
        "dataset publication lock is not owned by the current user")
    fcntl.flock(descriptor, fcntl.LOCK_EX)
  except Exception:
    os.close(descriptor)
    raise
  return descriptor


def _acquire_locator_lock(locator_dir, logical_parameter):
  digest = hashlib.sha256(logical_parameter.encode("ascii")).hexdigest()
  path = locator_dir / (".locator-" + digest + ".lock")
  flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags, 0o600)
  except OSError as exc:
    raise DatasetPublicationError(
      "could not open dataset locator lock: " + str(exc)) from exc
  try:
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
      raise DatasetPublicationError(
        "dataset locator lock is not one private regular file")
    if status.st_mode & 0o077:
      raise DatasetPublicationError(
        "dataset locator lock is accessible to another user")
    if hasattr(os, "getuid") and status.st_uid != os.getuid():
      raise DatasetPublicationError(
        "dataset locator lock is not owned by the current user")
    fcntl.flock(descriptor, fcntl.LOCK_EX)
  except Exception:
    os.close(descriptor)
    raise
  return descriptor


def _release_publish_lock(descriptor):
  try:
    try:
      fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
      # close() releases an advisory flock too.  A release diagnostic must not
      # make a durably committed publication look like it failed.
      pass
  finally:
    try:
      os.close(descriptor)
    except OSError:
      pass


def _read_active_pointer_if_present(slot):
  if not os.path.lexists(slot.active_path):
    return None
  return _read_active_pointer(slot)


def _read_active_pointer(slot):
  payload = _read_regular_bytes(
    slot.active_path, _MAX_ACTIVE_BYTES, "active dataset record",
    require_read_only=True, require_path_stable=False)
  active = _parse_canonical_record(
    payload, "active dataset record", _ACTIVE_KEYS)
  _require_schema_value(active["schema"], "active dataset record")
  if active["schema"] != PUBLICATION_SCHEMA:
    raise DatasetPublicationError("unsupported active dataset schema")
  if active["slot_id"] != slot.slot_id:
    raise DatasetPublicationError(
      "active dataset record belongs to a different slot")
  _require_generation_id(active["generation"])
  _require_hex_digest(active["manifest_sha256"], "active manifest")
  return active, hashlib.sha256(payload).hexdigest()


def _begin_sealed_generation(draft):
  path = draft.slot.work_path / (
    "sealed-" + draft.generation + "-" + secrets.token_hex(8))
  try:
    os.mkdir(path, 0o700)
    os.mkdir(path / FILES_NAME, 0o700)
    _fsync_directory(path / FILES_NAME)
    _fsync_directory(path)
    _fsync_directory(draft.slot.work_path)
  except (OSError, DatasetPublicationError) as exc:
    _best_effort_remove_new_tree(path, draft.slot.work_path)
    raise DatasetPublicationError(
      "could not create private sealed generation " + str(path)
      + ": " + str(exc)) from exc
  return DraftGeneration(
    slot=draft.slot, generation=draft.generation, path=path)


def _copy_member_files(source, sealed, members):
  # Open and fingerprint every source before copying the first byte.  Keeping
  # all descriptors open and checking every token again after the final copy
  # proves one on-disk state after the last open.  The caller's quiescence
  # barrier owns scientific-run coherence; per-file checks alone would still
  # allow an A-old/B-new hybrid when A changes between the two copies.
  opened = {}
  records = {}
  try:
    for role in sorted(members):
      relative = members[role]
      source_path = source.files_path.joinpath(*PurePosixPath(relative).parts)
      descriptor, token = _open_source_member(
        source_path, label="member " + role)
      opened[role] = (descriptor, source_path, token)

    for role in sorted(members):
      relative = members[role]
      descriptor, _, _ = opened[role]
      destination_path = sealed.member_path(relative)
      size, digest = _copy_open_regular_file(
        descriptor, destination_path, label="member " + role)
      records[role] = {
        "path": FILES_NAME + "/" + relative,
        "sha256": digest,
        "size": size,
      }

    # Refuse renamed/replaced/added entries as well as changes through an
    # already-open descriptor.  This check remains inside the same all-open
    # window as every copy.
    _require_exact_source_draft_census(source, members)
    for role in sorted(opened):
      descriptor, source_path, before_token = opened[role]
      _require_unchanged_source_member(
        descriptor, source_path, before_token, label="member " + role)
    return records
  finally:
    for descriptor, _, _ in opened.values():
      os.close(descriptor)


def _copy_active_members_to_draft(source, draft):
  """Copy one authenticated generation without aliasing any source inode."""
  opened = {}
  copy_completed = False
  member_map = {
    member.role: member.relative_path for member in source.members}
  try:
    # Acquire every member before copying the first.  This makes the later
    # all-descriptor recheck one complete set of held source members rather
    # than a sequence of individually plausible files.
    for member in source.members:
      descriptor, token = _open_authenticated_member(member)
      opened[member.role] = (descriptor, member, token)

    for role in sorted(opened):
      descriptor, member, _ = opened[role]
      destination = draft.member_path(member.relative_path)
      size, digest = _copy_open_member_to_mutable_draft(
        descriptor, destination, label="member " + role)
      if size != member.size or digest != member.sha256:
        raise DatasetPublicationError(
          "member " + role
          + " changed after active-generation authentication")

    _require_unchanged_active_generation(source)
    for role in sorted(opened):
      descriptor, member, token = opened[role]
      _require_unchanged_authenticated_member(descriptor, member, token)
    _require_exact_source_draft_census(draft, member_map)
    _require_private_continuation_tree(draft, member_map)
    _fsync_tree_directories(draft.path)
    _fsync_directory(draft.slot.work_path)
    copy_completed = True
  finally:
    try:
      _close_active_member_descriptors(opened)
    except BaseException:
      # Keep the original validation/copy refusal when both work and cleanup
      # fail. If all work succeeded, a close failure still refuses the public
      # operation so the caller never receives a possibly leaked source.
      if copy_completed:
        raise


def _close_active_member_descriptors(opened):
  """Attempt every held close and report the first close failure afterward."""
  first_failure = None
  for descriptor, _, _ in opened.values():
    try:
      os.close(descriptor)
    except OSError as exc:
      if first_failure is None:
        first_failure = exc
  if first_failure is not None:
    raise DatasetPublicationError(
      "could not close every immutable continuation source: "
      + str(first_failure)) from first_failure


def _open_authenticated_member(member):
  """Open one published member and bind its path, inode, mode, and size."""
  flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
  descriptor = None
  label = "member " + member.role
  try:
    descriptor = os.open(member.path, flags)
    status = os.fstat(descriptor)
    named = os.lstat(member.path)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 \
        or stat.S_ISLNK(named.st_mode) \
        or (status.st_dev, status.st_ino) != (named.st_dev, named.st_ino):
      raise DatasetPublicationError(
        label + " is not one stable immutable source file")
    if hasattr(os, "getuid") and status.st_uid != os.getuid():
      raise DatasetPublicationError(
        label + " immutable source is not owned by the current user")
    if status.st_mode & 0o222:
      raise DatasetPublicationError(
        label + " immutable source is writable")
    if status.st_size != member.size:
      raise DatasetPublicationError(
        label + " immutable source size changed before continuation")
    return descriptor, _source_epoch_token(status)
  except DatasetPublicationError:
    if descriptor is not None:
      os.close(descriptor)
    raise
  except OSError as exc:
    if descriptor is not None:
      os.close(descriptor)
    raise DatasetPublicationError(
      "could not open " + label + " for continuation: " + str(exc)) from exc


def _require_unchanged_authenticated_member(descriptor, member, before_token):
  """Recheck one held published descriptor after the final member copy."""
  _require_unchanged_source_member(
    descriptor, member.path, before_token, label="member " + member.role)
  status = os.fstat(descriptor)
  if status.st_mode & 0o222 or status.st_size != member.size \
      or (hasattr(os, "getuid") and status.st_uid != os.getuid()):
    raise DatasetPublicationError(
      "member " + member.role
      + " no longer matches its immutable continuation source")


def _copy_open_member_to_mutable_draft(source_fd, destination, *, label):
  """Copy one open immutable member into a new independent mode-0600 file."""
  destination_flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                       | getattr(os, "O_NOFOLLOW", 0))
  destination_fd = None
  try:
    os.lseek(source_fd, 0, os.SEEK_SET)
    source_status = os.fstat(source_fd)
    destination_fd = os.open(destination, destination_flags, 0o600)
    digest = hashlib.sha256()
    copied = 0
    while True:
      block = os.read(source_fd, 1024 * 1024)
      if not block:
        break
      digest.update(block)
      view = memoryview(block)
      written = 0
      while written < len(view):
        count = os.write(destination_fd, view[written:])
        if count <= 0:
          raise OSError("short continuation write")
        written += count
      copied += len(block)
    os.fchmod(destination_fd, 0o600)
    _fsync_regular_file(destination_fd)
    destination_status = os.fstat(destination_fd)
    if not stat.S_ISREG(destination_status.st_mode) \
        or destination_status.st_nlink != 1 \
        or destination_status.st_size != copied \
        or stat.S_IMODE(destination_status.st_mode) != 0o600 \
        or (source_status.st_dev, source_status.st_ino) == (
          destination_status.st_dev, destination_status.st_ino):
      raise DatasetPublicationError(
        label + " continuation copy is not one independent private file")
    return copied, digest.hexdigest()
  except DatasetPublicationError:
    raise
  except OSError as exc:
    raise DatasetPublicationError(
      "could not copy " + label + " for continuation: " + str(exc)) from exc
  finally:
    if destination_fd is not None:
      os.close(destination_fd)


def _require_unchanged_active_generation(source):
  """Re-read the named manifest, immutable directories, and exact census."""
  generation_path = source.slot.generations_path / source.generation
  _require_read_only_directory(generation_path, "generation directory")
  _require_read_only_directory(
    generation_path / FILES_NAME, "generation files directory")
  manifest_bytes = _read_regular_bytes(
    generation_path / MANIFEST_NAME, _MAX_MANIFEST_BYTES,
    "dataset manifest", require_read_only=True)
  if hashlib.sha256(manifest_bytes).hexdigest() != source.manifest_sha256:
    raise DatasetPublicationError(
      "immutable source manifest changed across the continuation copy")
  _require_exact_generation_census(
    generation_path,
    {member.relative_path for member in source.members})


def _require_private_continuation_tree(draft, members):
  """Require exact owner-only modes for a returned mutable continuation."""
  directory_paths = [draft.path, draft.files_path]
  for relative in sorted(_expected_member_directories(members.values())):
    directory_paths.append(
      draft.files_path.joinpath(*PurePosixPath(relative).parts))
  for path in directory_paths:
    status = _require_private_directory(path, "continuation draft directory")
    if stat.S_IMODE(status.st_mode) != 0o700:
      raise DatasetPublicationError(
        "continuation draft directory must have mode 0700: " + str(path))
  for relative in members.values():
    path = draft.files_path.joinpath(*PurePosixPath(relative).parts)
    status = os.lstat(path)
    if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode) \
        or status.st_nlink != 1 or stat.S_IMODE(status.st_mode) != 0o600 \
        or (hasattr(os, "getuid") and status.st_uid != os.getuid()):
      raise DatasetPublicationError(
        "continuation member must be one owner-only mode-0600 file: "
        + str(path))


def _open_source_member(path, *, label):
  flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
  descriptor = None
  try:
    descriptor = os.open(path, flags)
    status = os.fstat(descriptor)
    named = os.lstat(path)
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 \
        or stat.S_ISLNK(named.st_mode) \
        or (status.st_dev, status.st_ino) != (named.st_dev, named.st_ino):
      raise DatasetPublicationError(
        label + " source is not one stable regular file")
    if hasattr(os, "getuid") and status.st_uid != os.getuid():
      raise DatasetPublicationError(
        label + " source is not owned by the current user")
    return descriptor, _source_epoch_token(status)
  except DatasetPublicationError:
    if descriptor is not None:
      os.close(descriptor)
    raise
  except OSError as exc:
    if descriptor is not None:
      os.close(descriptor)
    raise DatasetPublicationError(
      "could not open " + label + " for sealing: " + str(exc)) from exc


def _source_epoch_token(status):
  return (
    status.st_size, status.st_mtime_ns, status.st_ctime_ns,
    status.st_dev, status.st_ino)


def _require_unchanged_source_member(descriptor, path, before_token, *, label):
  try:
    after = os.fstat(descriptor)
    named = os.lstat(path)
  except OSError as exc:
    raise DatasetPublicationError(
      label + " source changed across the dataset copy: " + str(exc)) from exc
  if not stat.S_ISREG(after.st_mode) or after.st_nlink != 1 \
      or stat.S_ISLNK(named.st_mode) \
      or _source_epoch_token(after) != before_token \
      or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino):
    raise DatasetPublicationError(
      label + " source changed across the complete dataset copy")


def _copy_open_regular_file(source_fd, destination, *, label):
  destination_flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                       | getattr(os, "O_NOFOLLOW", 0))
  destination_fd = None
  try:
    os.lseek(source_fd, 0, os.SEEK_SET)
    destination_fd = os.open(destination, destination_flags, 0o600)
    digest = hashlib.sha256()
    copied = 0
    while True:
      block = os.read(source_fd, 1024 * 1024)
      if not block:
        break
      digest.update(block)
      view = memoryview(block)
      written = 0
      while written < len(view):
        count = os.write(destination_fd, view[written:])
        if count <= 0:
          raise OSError("short member write")
        written += count
      copied += len(block)
    _fsync_regular_file(destination_fd)
    os.fchmod(destination_fd, 0o444)
    _fsync_regular_file(destination_fd)
    destination_status = os.fstat(destination_fd)
    if not stat.S_ISREG(destination_status.st_mode) \
        or destination_status.st_nlink != 1 \
        or destination_status.st_size != copied \
        or destination_status.st_mode & 0o222:
      raise DatasetPublicationError(
        label + " sealed copy is not one read-only regular file")
    return copied, digest.hexdigest()
  except DatasetPublicationError:
    raise
  except OSError as exc:
    raise DatasetPublicationError(
      "could not seal " + label + ": " + str(exc)) from exc
  finally:
    if destination_fd is not None:
      os.close(destination_fd)


def _relative_regular_census(root, *, mutable):
  file_result = set()
  directory_result = set()

  def walk_error(error):
    raise DatasetPublicationError(
      "could not enumerate dataset member tree: " + str(error))

  for directory, names, files in os.walk(
      root, topdown=True, onerror=walk_error, followlinks=False):
    directory_path = Path(directory)
    if mutable:
      _require_private_directory(directory_path, "draft member directory")
    else:
      _require_read_only_directory(
        directory_path, "published member directory")
    for name in names:
      child = directory_path / name
      _portable_part(name, "member directory name")
      _require_real_directory(child, "member directory")
      directory_result.add(child.relative_to(root).as_posix())
    for name in files:
      _portable_part(name, "member filename")
      child = directory_path / name
      try:
        child_status = os.lstat(child)
      except OSError as exc:
        raise DatasetPublicationError(
          "could not inspect member " + str(child) + ": " + str(exc)) from exc
      if not stat.S_ISREG(child_status.st_mode) \
          or stat.S_ISLNK(child_status.st_mode) or child_status.st_nlink != 1:
        raise DatasetPublicationError(
          "dataset member is not one unlinked regular file: " + str(child))
      file_result.add(child.relative_to(root).as_posix())
  return file_result, directory_result


def _hash_regular_file(path, *, label, durable, require_read_only):
  flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags)
  except OSError as exc:
    raise DatasetPublicationError(label + " cannot be opened safely: "
                                  + str(exc)) from exc
  try:
    before = os.fstat(descriptor)
    try:
      named = os.lstat(path)
    except OSError as exc:
      raise DatasetPublicationError(
        label + " changed during validation: " + str(exc)) from exc
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
        or stat.S_ISLNK(named.st_mode) \
        or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
      raise DatasetPublicationError(
        label + " is not one stable regular file")
    if require_read_only and before.st_mode & 0o222:
      raise DatasetPublicationError(label + " is still writable")
    digest = hashlib.sha256()
    while True:
      block = os.read(descriptor, 1024 * 1024)
      if not block:
        break
      digest.update(block)
    if durable:
      _fsync_regular_file(descriptor)
    after = os.fstat(descriptor)
    before_token = (
      before.st_size, before.st_mtime_ns, before.st_ctime_ns,
      before.st_dev, before.st_ino)
    after_token = (
      after.st_size, after.st_mtime_ns, after.st_ctime_ns,
      after.st_dev, after.st_ino)
    if before_token != after_token:
      raise DatasetPublicationError(label + " changed while it was hashed")
    try:
      named_after = os.lstat(path)
    except OSError as exc:
      raise DatasetPublicationError(
        label + " path changed while it was hashed: " + str(exc)) from exc
    if stat.S_ISLNK(named_after.st_mode) \
        or (after.st_dev, after.st_ino) != (
          named_after.st_dev, named_after.st_ino):
      raise DatasetPublicationError(label + " path changed while it was hashed")
    return after.st_size, digest.hexdigest()
  finally:
    os.close(descriptor)


def _read_regular_bytes(path, maximum, label, *, require_read_only,
                        require_path_stable=True):
  flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags)
  except OSError as exc:
    raise DatasetPublicationError(label + " cannot be opened safely: "
                                  + str(exc)) from exc
  try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
      raise DatasetPublicationError(label + " is not a regular file")
    if require_path_stable:
      try:
        named = os.lstat(path)
      except OSError as exc:
        raise DatasetPublicationError(
          label + " changed during validation: " + str(exc)) from exc
      if before.st_nlink != 1 or stat.S_ISLNK(named.st_mode) \
          or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
        raise DatasetPublicationError(label + " is not one stable regular file")
    elif before.st_nlink not in (0, 1):
      raise DatasetPublicationError(
        label + " has an unsafe hard-link count")
    if require_read_only and before.st_mode & 0o222:
      raise DatasetPublicationError(label + " is still writable")
    if before.st_size > maximum:
      raise DatasetPublicationError(
        label + " exceeds its " + str(maximum) + " byte limit")
    chunks = []
    remaining = maximum + 1
    while remaining:
      block = os.read(descriptor, min(1024 * 1024, remaining))
      if not block:
        break
      chunks.append(block)
      remaining -= len(block)
    if remaining == 0:
      raise DatasetPublicationError(
        label + " exceeds its " + str(maximum) + " byte limit")
    after = os.fstat(descriptor)
    before_token = (
      before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
    after_token = (
      after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino)
    if require_path_stable or after.st_nlink == before.st_nlink:
      before_token += (before.st_ctime_ns,)
      after_token += (after.st_ctime_ns,)
    if before_token != after_token:
      raise DatasetPublicationError(label + " changed while it was read")
    if require_path_stable:
      try:
        named_after = os.lstat(path)
      except OSError as exc:
        raise DatasetPublicationError(
          label + " path changed while it was read: " + str(exc)) from exc
      if stat.S_ISLNK(named_after.st_mode) \
          or (after.st_dev, after.st_ino) != (
            named_after.st_dev, named_after.st_ino):
        raise DatasetPublicationError(label + " path changed while it was read")
    return b"".join(chunks)
  finally:
    os.close(descriptor)


def _write_new_regular_file(path, payload, *, final_mode):
  flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
           | getattr(os, "O_NOFOLLOW", 0))
  descriptor = None
  try:
    descriptor = os.open(path, flags, 0o600)
    view = memoryview(payload)
    written = 0
    while written < len(view):
      count = os.write(descriptor, view[written:])
      if count <= 0:
        raise OSError("short write")
      written += count
    _fsync_regular_file(descriptor)
    os.fchmod(descriptor, final_mode)
    _fsync_regular_file(descriptor)
  except OSError as exc:
    raise DatasetPublicationError(
      "could not durably write publication record " + str(path)
      + ": " + str(exc)) from exc
  finally:
    if descriptor is not None:
      os.close(descriptor)


def _expected_member_directories(relative_paths):
  result = set()
  for value in relative_paths:
    path = PurePosixPath(value)
    parts = path.parts[:-1]
    for end in range(1, len(parts) + 1):
      result.add(PurePosixPath(*parts[:end]).as_posix())
  return result


def _require_exact_source_draft_census(draft, members):
  _require_root_entries(draft.path, {FILES_NAME}, "source draft")
  expected_files = set(members.values())
  expected_directories = _expected_member_directories(expected_files)
  observed_files, observed_directories = _relative_regular_census(
    draft.files_path, mutable=True)
  if observed_files != expected_files or observed_directories != expected_directories:
    raise DatasetPublicationError(
      "source draft census differs from the declared publication: missing files "
      + repr(sorted(expected_files - observed_files)) + ", extra files "
      + repr(sorted(observed_files - expected_files)) + ", missing directories "
      + repr(sorted(expected_directories - observed_directories))
      + ", extra directories "
      + repr(sorted(observed_directories - expected_directories)))


def _require_exact_sealed_census(sealed, members):
  _require_root_entries(
    sealed.path, {FILES_NAME, MANIFEST_NAME}, "sealed generation")
  expected_files = set(members.values())
  expected_directories = _expected_member_directories(expected_files)
  observed_files, observed_directories = _relative_regular_census(
    sealed.files_path, mutable=True)
  if observed_files != expected_files or observed_directories != expected_directories:
    raise DatasetPublicationError(
      "sealed generation changed while it was authenticated")


def _require_exact_generation_census(generation_path, expected_relative):
  _require_root_entries(
    generation_path, {FILES_NAME, MANIFEST_NAME}, "immutable generation")
  expected_files = set(expected_relative)
  expected_directories = _expected_member_directories(expected_files)
  observed_files, observed_directories = _relative_regular_census(
    generation_path / FILES_NAME, mutable=False)
  if observed_files != expected_files or observed_directories != expected_directories:
    raise DatasetPublicationError(
      "immutable generation file census differs from its manifest: missing "
      + repr(sorted(expected_files - observed_files)) + ", extra "
      + repr(sorted(observed_files - expected_files))
      + ", unexpected directories "
      + repr(sorted(observed_directories - expected_directories)))


def _require_root_entries(path, expected, label):
  try:
    observed = set(os.listdir(path))
  except OSError as exc:
    raise DatasetPublicationError(
      "could not enumerate " + label + " root: " + str(exc)) from exc
  if observed != set(expected):
    raise DatasetPublicationError(
      label + " root entries must be exactly " + repr(sorted(expected))
      + "; got " + repr(sorted(observed)))


def _make_tree_read_only(root):
  directories = []
  for directory, names, files in os.walk(
      root, topdown=True, onerror=_raise_walk_error, followlinks=False):
    directory_path = Path(directory)
    directories.append(directory_path)
    for name in names:
      _require_real_directory(directory_path / name, "generation directory")
    for name in files:
      path = directory_path / name
      status = os.lstat(path)
      if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode) \
          or status.st_nlink != 1:
        raise DatasetPublicationError(
          "generation contains a linked or special file: " + str(path))
      os.chmod(path, 0o444, follow_symlinks=False)
  for directory in reversed(directories):
    os.chmod(directory, 0o555, follow_symlinks=False)


def _fsync_tree_directories(root):
  directories = []
  for directory, names, files in os.walk(
      root, topdown=True, onerror=_raise_walk_error, followlinks=False):
    del names, files
    directories.append(Path(directory))
  for directory in reversed(directories):
    _fsync_directory(directory)


def _raise_walk_error(error):
  raise DatasetPublicationError(
    "could not enumerate dataset generation tree: " + str(error))


def _fsync_directory(path):
  flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
          | getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags)
  except OSError as exc:
    raise DatasetPublicationError(
      "could not open directory for durability " + str(path)
      + ": " + str(exc)) from exc
  try:
    os.fsync(descriptor)
  except OSError as exc:
    raise DatasetPublicationError(
      "could not make directory durable " + str(path)
      + ": " + str(exc)) from exc
  finally:
    os.close(descriptor)


def _fsync_regular_file(descriptor):
  """Flush file data/metadata, including Darwin hardware caches when possible."""
  os.fsync(descriptor)
  full_fsync = getattr(fcntl, "F_FULLFSYNC", None)
  if full_fsync is not None:
    fcntl.fcntl(descriptor, full_fsync)


def _validate_manifest_header(slot, active, manifest):
  _require_schema_value(manifest["schema"], "dataset manifest")
  if manifest["schema"] != PUBLICATION_SCHEMA:
    raise DatasetPublicationError("unsupported dataset manifest schema")
  if type(manifest["identity"]) is not dict:
    raise DatasetPublicationError("dataset manifest identity must be an object")
  if manifest["slot_id"] != slot.slot_id \
      or manifest["slot_id"] != active["slot_id"]:
    raise DatasetPublicationError("dataset manifest belongs to another slot")
  if canonical_json_bytes(manifest["slot"]) != slot.descriptor_json:
    raise DatasetPublicationError(
      "dataset manifest slot descriptor does not match the requested output")
  if manifest["generation"] != active["generation"]:
    raise DatasetPublicationError(
      "active record and manifest name different generations")
  _require_generation_id(manifest["generation"])


def _require_schema_value(value, label):
  if type(value) is not int or value < 1:
    raise DatasetPublicationError(
      label + " schema must be a positive native integer")


def _call_checkpoint(callback, name):
  if callback is not None:
    callback(name)
