#!/usr/bin/env python3
"""Create and safely read portable backlog handoff archives.

The archive is an email attachment, not a Git artifact.  It contains the
local execution ledger and local supporting evidence while anchoring the eleven
permanent notes to an exact Git commit.  Incoming archives are treated as
hostile: inspection validates the complete XZ stream, tar structure,
manifest, paths, sizes, and hashes before returning any content.
"""

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from urllib.parse import urlsplit


FORMAT_NAME = "cocoa-backlog-handoff"
FORMAT_VERSION = 1
ARCHIVE_ROOT = "backlog-handoff-v1"
MANIFEST_MEMBER = ARCHIVE_ROOT + "/manifest.json"
PAYLOAD_PREFIX = ARCHIVE_ROOT + "/payload/"
BACKLOG_PATH = "ai/notes/backlog.md"
SUPPORT_ROOT = "ai/notes/backlog-support"
DEFAULT_BUNDLE_ROOT = "ai/backlog-bundles"
DEFAULT_IMPORT_ROOT = "ai/backlog-imports"

# These are the only durable notes.  The bundle never substitutes worktree
# bytes for them: the recorded base commit is their source of truth.
PERMANENT_NOTES = frozenset({
    "ai/notes/MEMORY.md",
    "ai/notes/artifacts-inference-warmstart.md",
    "ai/notes/conventions-and-workflow.md",
    "ai/notes/data-generation-and-cuts.md",
    "ai/notes/families-background-mps.md",
    "ai/notes/families-scalar-cmb.md",
    "ai/notes/models-and-designs.md",
    "ai/notes/project-and-history.md",
    "ai/notes/readme-go-no-go.md",
    "ai/notes/training-stack.md",
    "ai/notes/user-didactics-and-python-voice.md",
})

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_TAR_BYTES = 256 * 1024 * 1024
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MEMBERS = 512
MAX_PATH_BYTES = 240
MAX_PATH_DEPTH = 16
MAX_XZ_MEMORY = 128 * 1024 * 1024
IO_CHUNK = 1024 * 1024

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY_RE = re.compile(r"^[a-z0-9.-]+/[A-Za-z0-9._/-]+$")
WINDOWS_RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + ["COM" + str(index) for index in range(1, 10)]
    + ["LPT" + str(index) for index in range(1, 10)]
)


class BundleError(Exception):
    """A user-facing refusal caused by an unsafe or inconsistent bundle."""


def _git(repo, *args, check=True):
    result = subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BundleError("git " + " ".join(args) + " failed: " + detail)
    return result


def repository_root():
    """Return the repository containing this installed tool."""
    candidate = Path(__file__).resolve().parents[2]
    root = _git(candidate, "rev-parse", "--show-toplevel").stdout.strip()
    root_path = Path(root).resolve()
    if root_path != candidate:
        raise BundleError("tool path and Git repository root disagree")
    return root_path


def canonical_repository_id(remote):
    """Reduce an origin URL to a credential-free host/path identity."""
    remote = remote.strip()
    scp = re.match(r"^[^/@:]+@([^:]+):(.+)$", remote)
    if scp:
        host, path = scp.group(1), scp.group(2)
    else:
        parsed = urlsplit(remote)
        if parsed.scheme not in ("http", "https", "ssh", "git"):
            raise BundleError("origin must be a network Git URL")
        if not parsed.hostname:
            raise BundleError("origin has no host")
        host, path = parsed.hostname, parsed.path
    host = host.lower().rstrip(".")
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if host == "github.com":
        path = path.lower()
    identity = host + "/" + path
    if not REPOSITORY_RE.fullmatch(identity) or ".." in PurePosixPath(path).parts:
        raise BundleError("origin cannot be reduced to a portable repository id")
    return identity


def repository_identity(repo):
    remote = _git(repo, "remote", "get-url", "origin").stdout.strip()
    identity = canonical_repository_id(remote)
    return {
        "id": identity,
        # Display metadata must not depend on the recipient's clone directory.
        "name": identity.rsplit("/", 1)[-1],
    }


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_repo_path(value):
    """Validate one portable, repository-relative POSIX path."""
    if not isinstance(value, str) or not value:
        raise BundleError("manifest path must be a nonempty string")
    try:
        normalized = unicodedata.normalize("NFC", value)
    except UnicodeError:
        raise BundleError("path contains invalid Unicode: " + repr(value))
    if value != normalized:
        raise BundleError("path is not NFC-normalized: " + repr(value))
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise BundleError("path contains control characters: " + repr(value))
    if (value.startswith("/") or "\\" in value or ":" in value or
            any(char in value for char in '<>"|?*')):
        raise BundleError("path contains nonportable characters: " + repr(value))
    try:
        encoded_value = value.encode("utf-8")
    except UnicodeEncodeError:
        raise BundleError("path is not valid UTF-8: " + repr(value))
    if len(encoded_value) > MAX_PATH_BYTES:
        raise BundleError("path is too long: " + repr(value))
    parts = value.split("/")
    if len(parts) > MAX_PATH_DEPTH or any(part in ("", ".", "..") for part in parts):
        raise BundleError("path has unsafe components: " + repr(value))
    for part in parts:
        if part.endswith((".", " ")):
            raise BundleError("path has a nonportable trailing character: " + repr(value))
        if part.split(".", 1)[0].upper() in WINDOWS_RESERVED:
            raise BundleError("path uses a reserved Windows name: " + repr(value))
        if len(part.encode("utf-8")) > 100:
            raise BundleError("path component is too long for USTAR: " + repr(value))
    return value


def _repo_relative(repo, path_text):
    validate_repo_path(path_text)
    candidate = repo.joinpath(*path_text.split("/"))
    current = repo
    for part in path_text.split("/"):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise BundleError(
                "cannot inspect supporting file " + path_text + ": " +
                str(error))
        if stat.S_ISLNK(mode):
            raise BundleError("symlink supporting files are forbidden: " + path_text)
    try:
        candidate.resolve(strict=True).relative_to(repo)
    except (OSError, ValueError):
        raise BundleError("supporting file escapes the repository: " + path_text)
    return candidate


def stable_read(repo, path_text):
    """Read one regular file without following its final symlink."""
    path = _repo_relative(repo, path_text)
    try:
        initial = path.lstat()
    except OSError as error:
        raise BundleError("cannot inspect " + path_text + ": " + str(error))
    if not stat.S_ISREG(initial.st_mode):
        raise BundleError(
            "supporting path is not a regular file: " + path_text)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    # A selected path can be replaced between lstat and open.  Nonblocking
    # mode makes that race fail boundedly if the replacement is a FIFO; the
    # fstat check below still requires the opened object to be regular.
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise BundleError("cannot safely open " + path_text + ": " + str(error))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BundleError("supporting path is not a regular file: " + path_text)
        if before.st_size > MAX_FILE_BYTES:
            raise BundleError(
                "supporting file exceeds the per-file limit: " + path_text)
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(IO_CHUNK, MAX_FILE_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise BundleError(
                    "supporting file exceeds the per-file limit: " + path_text)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        final = path.lstat()
    except OSError as error:
        raise BundleError(
            "supporting file changed while it was read: " + path_text +
            ": " + str(error))
    identity_initial = (initial.st_dev, initial.st_ino, initial.st_size,
                        getattr(initial, "st_mtime_ns",
                                int(initial.st_mtime * 1e9)))
    identity_before = (before.st_dev, before.st_ino, before.st_size,
                       getattr(before, "st_mtime_ns", int(before.st_mtime * 1e9)))
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)))
    identity_final = (final.st_dev, final.st_ino, final.st_size,
                      getattr(final, "st_mtime_ns", int(final.st_mtime * 1e9)))
    if (identity_initial != identity_before or
            identity_before != identity_after or
            identity_after != identity_final):
        raise BundleError("supporting file changed while it was read: " + path_text)
    return b"".join(chunks)


def _head_file_digest(repo, path_text):
    result = _git(repo, "rev-parse", "HEAD:" + path_text, check=False)
    if result.returncode != 0:
        raise BundleError("permanent note is absent from HEAD: " + path_text)
    return result.stdout.strip()


def require_clean_permanent_notes(repo):
    dirty = []
    for path_text in sorted(PERMANENT_NOTES):
        path = _repo_relative(repo, path_text)
        worktree = _git(repo, "hash-object", str(path)).stdout.strip()
        if worktree != _head_file_digest(repo, path_text):
            dirty.append(path_text)
    if dirty:
        raise BundleError(
            "permanent notes differ from HEAD; the Architect must resolve "
            "them before packing: "
            + ", ".join(dirty)
        )


def _top_level_local_notes(repo):
    notes_root = repo / "ai" / "notes"
    result = []
    for path in sorted(notes_root.glob("*.md")):
        relative = path.relative_to(repo).as_posix()
        if relative not in PERMANENT_NOTES:
            result.append(relative)
    return result


def _support_files(repo):
    root = repo / SUPPORT_ROOT
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise BundleError(SUPPORT_ROOT + " must be a real directory")
    result = []
    for directory, names, files in os.walk(str(root), followlinks=False):
        names.sort()
        files.sort()
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            if child.is_symlink():
                raise BundleError("symlink support directories are forbidden: " +
                                  child.relative_to(repo).as_posix())
        for name in files:
            result.append((directory_path / name).relative_to(repo).as_posix())
    return result


def selected_files(repo, explicit):
    """Return a sorted mapping of repository path to bundle role."""
    roles = {BACKLOG_PATH: "backlog"}
    for path_text in _top_level_local_notes(repo):
        roles.setdefault(path_text, "local-note")
    for path_text in _support_files(repo):
        roles.setdefault(path_text, "support")
    for path_text in explicit:
        validate_repo_path(path_text)
        if path_text in PERMANENT_NOTES:
            raise BundleError(
                "permanent notes must come from the recorded Git base: " +
                path_text)
        roles.setdefault(path_text, "explicit")
    folded = {}
    for path_text in sorted(roles):
        key = path_text.casefold()
        if key in folded and folded[key] != path_text:
            raise BundleError("case-colliding paths are not portable: " +
                              folded[key] + " and " + path_text)
        folded[key] = path_text
    return roles


def open_backlog_lines(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise BundleError(BACKLOG_PATH + " is not valid UTF-8")
    return [line for line in text.splitlines() if line.startswith("- OPEN")]


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("ascii")


def build_bundle(repo, explicit):
    require_clean_permanent_notes(repo)
    roles = selected_files(repo, explicit)
    payload = {}
    records = []
    total = 0
    for path_text in sorted(roles):
        data = stable_read(repo, path_text)
        total += len(data)
        if total > MAX_PAYLOAD_BYTES:
            raise BundleError("selected payload exceeds the bundle limit")
        payload[path_text] = data
        records.append({
            "path": path_text,
            "role": roles[path_text],
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        })
    manifest = {
        "backlog_path": BACKLOG_PATH,
        "base_commit": _git(repo, "rev-parse", "HEAD").stdout.strip(),
        "base_tree": _git(repo, "rev-parse", "HEAD^{tree}").stdout.strip(),
        "files": records,
        "format": FORMAT_NAME,
        "open_items": open_backlog_lines(payload[BACKLOG_PATH]),
        "repository": repository_identity(repo),
        "version": FORMAT_VERSION,
    }
    validate_manifest(manifest)
    manifest_bytes = canonical_json(manifest)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise BundleError("manifest exceeds its size limit")
    canonical_tar_header(MANIFEST_MEMBER, len(manifest_bytes))
    for record in records:
        canonical_tar_header(
            PAYLOAD_PREFIX + record["path"], record["size"])
    return manifest, manifest_bytes, payload


def _tar_info(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info


def canonical_tar_header(name, size):
    """Return the one accepted USTAR header for a member."""
    try:
        return _tar_info(name, size).tobuf(
            format=tarfile.USTAR_FORMAT, encoding="utf-8", errors="strict")
    except (UnicodeError, ValueError) as error:
        raise BundleError("member name cannot be represented in USTAR: " +
                          repr(name) + ": " + str(error))


def write_archive(output, manifest_bytes, payload):
    """Write a deterministic archive and install it without overwriting."""
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise BundleError("refusing to overwrite archive: " + str(output))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".backlog-bundle-", suffix=".tmp", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        raw = os.fdopen(descriptor, "w+b")
        descriptor = -1
        with raw:
            os.fchmod(raw.fileno(), 0o600)
            with lzma.LZMAFile(
                raw, "wb", format=lzma.FORMAT_XZ,
                check=lzma.CHECK_CRC64,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 6}],
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w",
                                  format=tarfile.USTAR_FORMAT,
                                  encoding="utf-8", errors="strict") as archive:
                    archive.addfile(
                        _tar_info(MANIFEST_MEMBER, len(manifest_bytes)),
                        fileobj=_BytesReader(manifest_bytes))
                    for path_text in sorted(payload):
                        data = payload[path_text]
                        name = PAYLOAD_PREFIX + path_text
                        archive.addfile(_tar_info(name, len(data)),
                                        fileobj=_BytesReader(data))
            raw.flush()
            archive_stat = os.fstat(raw.fileno())
            if archive_stat.st_size > MAX_ARCHIVE_BYTES:
                raise BundleError("compressed archive exceeds its size limit")
            raw.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: raw.read(IO_CHUNK), b""):
                digest.update(chunk)
            temporary_stat = temporary.lstat()
            if ((archive_stat.st_dev, archive_stat.st_ino) !=
                    (temporary_stat.st_dev, temporary_stat.st_ino)):
                raise BundleError("temporary archive name changed before publication")
            try:
                os.link(str(temporary), str(output))
            except FileExistsError:
                raise BundleError("refusing to overwrite archive: " + str(output))
            output_stat = output.lstat()
            if ((archive_stat.st_dev, archive_stat.st_ino) !=
                    (output_stat.st_dev, output_stat.st_ino)):
                raise BundleError("published archive identity is ambiguous")
            archive_digest = digest.hexdigest()
    finally:
        # fdopen owns descriptor after successful construction.  If fdopen
        # itself failed, close the original descriptor here.
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return archive_digest


class _BytesReader:
    """Small file-like reader accepted by TarFile.addfile on Python 3.9."""

    def __init__(self, data):
        self.data = data
        self.offset = 0

    def read(self, size=-1):
        if size < 0:
            size = len(self.data) - self.offset
        result = self.data[self.offset:self.offset + size]
        self.offset += len(result)
        return result


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(IO_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decompress_xz(archive_path, tar_path):
    """Decompress exactly one bounded XZ stream to a scratch tar file."""
    try:
        archive_stat = archive_path.lstat()
    except FileNotFoundError:
        raise BundleError("archive does not exist: " + str(archive_path))
    if archive_path.is_symlink() or not stat.S_ISREG(archive_stat.st_mode):
        raise BundleError("archive must be a regular, non-symlink file")
    if archive_stat.st_size > MAX_ARCHIVE_BYTES:
        raise BundleError("compressed archive exceeds its size limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(archive_path), flags)
    except OSError as error:
        raise BundleError("cannot safely open archive: " + str(error))
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ,
                                    memlimit=MAX_XZ_MEMORY)
    expanded = 0
    digest = hashlib.sha256()
    source = os.fdopen(descriptor, "rb")
    opened = os.fstat(source.fileno())
    identity_initial = (archive_stat.st_dev, archive_stat.st_ino,
                        archive_stat.st_size,
                        getattr(archive_stat, "st_mtime_ns",
                                int(archive_stat.st_mtime * 1e9)))
    identity_opened = (opened.st_dev, opened.st_ino, opened.st_size,
                       getattr(opened, "st_mtime_ns",
                               int(opened.st_mtime * 1e9)))
    if identity_initial != identity_opened or not stat.S_ISREG(opened.st_mode):
        source.close()
        raise BundleError("archive changed before it could be opened")
    try:
        with tar_path.open("wb") as target:
            while not decoder.eof:
                chunk = source.read(IO_CHUNK) if decoder.needs_input else b""
                if chunk:
                    digest.update(chunk)
                if decoder.needs_input and not chunk:
                    raise BundleError("truncated XZ stream")
                try:
                    data = decoder.decompress(
                        chunk, max_length=min(
                            IO_CHUNK, MAX_TAR_BYTES + 1 - expanded))
                except lzma.LZMAError as error:
                    raise BundleError("invalid XZ stream: " + str(error))
                expanded += len(data)
                if expanded > MAX_TAR_BYTES:
                    raise BundleError("expanded tar exceeds its size limit")
                target.write(data)
            trailing_byte = source.read(1)
            if trailing_byte:
                digest.update(trailing_byte)
            if decoder.unused_data or trailing_byte:
                raise BundleError("concatenated or trailing XZ data is forbidden")
        after = os.fstat(source.fileno())
    finally:
        source.close()
    try:
        final = archive_path.lstat()
    except FileNotFoundError:
        raise BundleError("archive disappeared while it was validated")
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)))
    identity_final = (final.st_dev, final.st_ino, final.st_size,
                      getattr(final, "st_mtime_ns", int(final.st_mtime * 1e9)))
    if identity_opened != identity_after or identity_after != identity_final:
        raise BundleError("archive changed while it was validated")
    return digest.hexdigest()


def _parse_octal(field, label):
    if field and field[0] & 0x80:
        raise BundleError("base-256 tar " + label + " is forbidden")
    raw = field.rstrip(b"\0 ").lstrip(b" ")
    if not raw:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in raw):
        raise BundleError("invalid tar " + label)
    return int(raw, 8)


def scan_raw_tar(tar_path):
    """Reject tar extensions and nonregular types before TarFile parsing."""
    names = []
    with tar_path.open("rb") as stream:
        while True:
            header = stream.read(512)
            if len(header) != 512:
                raise BundleError("tar ended before its two zero blocks")
            if header == b"\0" * 512:
                second = stream.read(512)
                if second != b"\0" * 512:
                    raise BundleError("tar has only one zero end block")
                # TarFile writes the shortest zero fill that completes the
                # current 10 KiB record.  Requiring that exact fill prevents
                # an otherwise valid bundle from carrying unmanifested bytes
                # (or an arbitrary number of extra zero records).
                remainder_size = (-stream.tell()) % tarfile.RECORDSIZE
                remainder = stream.read(remainder_size)
                if (len(remainder) != remainder_size or
                        remainder != b"\0" * remainder_size or
                        stream.read(1)):
                    raise BundleError(
                        "tar has noncanonical data after its end blocks")
                break
            if header[257:263] != b"ustar\0" or header[263:265] != b"00":
                raise BundleError("only canonical USTAR headers are accepted")
            if header[156:157] != tarfile.REGTYPE:
                raise BundleError(
                    "links, directories, devices, and tar extensions are "
                    "forbidden")
            try:
                name = header[0:100].split(b"\0", 1)[0].decode("utf-8")
                prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8")
            except UnicodeDecodeError:
                raise BundleError("tar member name is not valid UTF-8")
            full_name = prefix + "/" + name if prefix else name
            if not full_name:
                raise BundleError("tar member has an empty name")
            size = _parse_octal(header[124:136], "size")
            if header != canonical_tar_header(full_name, size):
                raise BundleError("tar member header is not canonical: " +
                                  repr(full_name))
            names.append(full_name)
            if len(names) > MAX_MEMBERS + 1:
                raise BundleError("tar contains too many members")
            stream.seek(size, os.SEEK_CUR)
            padding_size = (-size) % 512
            padding = stream.read(padding_size)
            if (len(padding) != padding_size or
                    padding != b"\0" * padding_size):
                raise BundleError(
                    "tar member padding is not canonical: " +
                    repr(full_name))
    return names


def _object_no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise BundleError("manifest contains duplicate key: " + repr(key))
        value[key] = item
    return value


def _exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise BundleError(label + " has an unexpected schema")


def validate_manifest(manifest):
    _exact_keys(manifest, [
        "backlog_path", "base_commit", "base_tree", "files", "format",
        "open_items", "repository", "version",
    ], "manifest")
    if (not isinstance(manifest["format"], str) or
            manifest["format"] != FORMAT_NAME or
            not _is_int(manifest["version"]) or
            manifest["version"] != FORMAT_VERSION):
        raise BundleError("unsupported bundle format or version")
    if manifest["backlog_path"] != BACKLOG_PATH:
        raise BundleError("bundle uses an unexpected backlog path")
    if (not isinstance(manifest["base_commit"], str) or
            not SHA_RE.fullmatch(manifest["base_commit"])):
        raise BundleError("manifest base_commit is invalid")
    if (not isinstance(manifest["base_tree"], str) or
            not SHA_RE.fullmatch(manifest["base_tree"])):
        raise BundleError("manifest base_tree is invalid")
    _exact_keys(manifest["repository"], ["id", "name"], "repository")
    repository_id = manifest["repository"]["id"]
    repository_name = manifest["repository"]["name"]
    if not isinstance(repository_id, str) or not REPOSITORY_RE.fullmatch(repository_id):
        raise BundleError("manifest repository id is invalid")
    if (not isinstance(repository_name, str) or not repository_name or
            "/" in repository_name or "\\" in repository_name):
        raise BundleError("manifest repository name is invalid")
    validate_repo_path(repository_name)
    if not isinstance(manifest["open_items"], list) or any(
            not isinstance(item, str) or not item.startswith("- OPEN")
            for item in manifest["open_items"]):
        raise BundleError("manifest open_items is invalid")
    files = manifest["files"]
    if not isinstance(files, list) or not files or len(files) > MAX_MEMBERS:
        raise BundleError("manifest files list is invalid")
    paths = []
    folded = set()
    backlog_count = 0
    total = 0
    allowed_roles = {"backlog", "local-note", "support", "explicit"}
    for record in files:
        _exact_keys(record, ["path", "role", "sha256", "size"], "file record")
        path_text = validate_repo_path(record["path"])
        if path_text in PERMANENT_NOTES:
            raise BundleError("bundle illegally embeds permanent note: " + path_text)
        if (not isinstance(record["role"], str) or
                record["role"] not in allowed_roles):
            raise BundleError("file record has an invalid role")
        if (not _is_int(record["size"]) or record["size"] < 0 or
                record["size"] > MAX_FILE_BYTES):
            raise BundleError("file record has an invalid size")
        if (not isinstance(record["sha256"], str) or
                not DIGEST_RE.fullmatch(record["sha256"])):
            raise BundleError("file record has an invalid digest")
        if record["role"] == "backlog":
            backlog_count += 1
            if path_text != BACKLOG_PATH:
                raise BundleError("backlog role is attached to the wrong path")
        if path_text == BACKLOG_PATH and record["role"] != "backlog":
            raise BundleError("backlog path is missing its backlog role")
        key = path_text.casefold()
        if key in folded:
            raise BundleError("manifest has duplicate or case-colliding paths")
        folded.add(key)
        paths.append(path_text)
        total += record["size"]
    if paths != sorted(paths) or backlog_count != 1:
        raise BundleError("manifest paths are not canonical or backlog is not unique")
    if total > MAX_PAYLOAD_BYTES:
        raise BundleError("manifest payload exceeds its size limit")


def validate_archive(archive_path):
    """Fully validate an archive and return manifest, payload, and id."""
    archive_path = Path(archive_path).expanduser().absolute()
    with tempfile.TemporaryDirectory(prefix="backlog-bundle-read-") as temporary:
        tar_path = Path(temporary) / "bundle.tar"
        archive_digest = decompress_xz(archive_path, tar_path)
        raw_names = scan_raw_tar(tar_path)
        if not raw_names or raw_names[0] != MANIFEST_MEMBER:
            raise BundleError("manifest must be the first tar member")
        if len(raw_names) != len(set(raw_names)):
            raise BundleError("tar contains duplicate member names")
        try:
            with tarfile.open(str(tar_path), mode="r:", format=tarfile.USTAR_FORMAT,
                              encoding="utf-8", errors="strict") as archive:
                members = archive.getmembers()
                if [member.name for member in members] != raw_names:
                    raise BundleError("raw and parsed tar member lists disagree")
                manifest_member = members[0]
                if manifest_member.size > MAX_MANIFEST_BYTES:
                    raise BundleError("manifest exceeds its size limit")
                manifest_stream = archive.extractfile(manifest_member)
                if manifest_stream is None:
                    raise BundleError("manifest is unreadable")
                manifest_bytes = manifest_stream.read()
                try:
                    manifest = json.loads(
                        manifest_bytes.decode("ascii"),
                        object_pairs_hook=_object_no_duplicates,
                    )
                except BundleError:
                    raise
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise BundleError("manifest is not canonical JSON: " + str(error))
                validate_manifest(manifest)
                if canonical_json(manifest) != manifest_bytes:
                    raise BundleError("manifest JSON is not canonical")
                expected_names = [MANIFEST_MEMBER] + [
                    PAYLOAD_PREFIX + record["path"] for record in manifest["files"]]
                if raw_names != expected_names:
                    raise BundleError("tar members do not exactly match the manifest")
                payload = {}
                for member, record in zip(members[1:], manifest["files"]):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise BundleError("payload member is unreadable")
                    data = stream.read(MAX_FILE_BYTES + 1)
                    if len(data) != record["size"] or member.size != record["size"]:
                        raise BundleError("payload size mismatch: " + record["path"])
                    if hashlib.sha256(data).hexdigest() != record["sha256"]:
                        raise BundleError("payload digest mismatch: " + record["path"])
                    payload[record["path"]] = data
        except tarfile.TarError as error:
            raise BundleError("invalid tar stream: " + str(error))
    if open_backlog_lines(payload[BACKLOG_PATH]) != manifest["open_items"]:
        raise BundleError("manifest open_items do not match the backlog bytes")
    bundle_id = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return manifest, payload, bundle_id, archive_digest


def _terminal_line(value):
    return "".join(
        char if char.isprintable() and char not in "\r\n" else
        "\\x" + format(ord(char), "02x")
        for char in value
    )


def print_inspection(manifest, bundle_id, archive_digest, show_backlog, payload):
    print("Bundle id:", bundle_id)
    print("Archive SHA-256:", archive_digest)
    print("Repository:", json.dumps(manifest["repository"]["id"]))
    print("Base commit:", manifest["base_commit"])
    print("Base tree:", manifest["base_tree"])
    print("Files:", len(manifest["files"]))
    for record in manifest["files"]:
        print("  " + record["role"] + " " + json.dumps(record["path"]) +
              " (" + str(record["size"]) + " bytes)")
    print("Open items:", len(manifest["open_items"]))
    for item in manifest["open_items"]:
        print("  " + _terminal_line(item))
    if show_backlog:
        print("Backlog contents:")
        text = payload[BACKLOG_PATH].decode("utf-8")
        for line in text.splitlines():
            print("  " + _terminal_line(line))


def verify_import_repository(repo, manifest):
    # A recipient is free to clone into a differently named directory.  The
    # credential-free remote identity and exact Git objects are authoritative;
    # repository.name is display-only metadata.
    if repository_identity(repo)["id"] != manifest["repository"]["id"]:
        raise BundleError("archive repository identity does not match this checkout")
    base = manifest["base_commit"]
    present = _git(repo, "cat-file", "-e", base + "^{commit}", check=False)
    if present.returncode != 0:
        raise BundleError("archive base commit is not present in this repository")
    actual_tree = _git(repo, "rev-parse", base + "^{tree}").stdout.strip()
    if actual_tree != manifest["base_tree"]:
        raise BundleError("archive base commit has an unexpected tree")
    ancestor = _git(repo, "merge-base", "--is-ancestor", base, "HEAD", check=False)
    if ancestor.returncode not in (0, 1):
        raise BundleError("could not compare archive base commit to HEAD")
    return ancestor.returncode == 0


def _import_destination(repo, requested, bundle_id):
    import_root = Path(os.path.abspath(str(repo / DEFAULT_IMPORT_ROOT)))
    if requested is None:
        destination = import_root / bundle_id[:16]
    else:
        candidate = Path(requested).expanduser()
        destination = candidate if candidate.is_absolute() else repo / candidate
        # abspath normalizes '.' and '..' without following a symlink ancestor.
        destination = Path(os.path.abspath(str(destination)))
    try:
        relative = destination.relative_to(import_root)
    except ValueError:
        raise BundleError("import output must stay under " + DEFAULT_IMPORT_ROOT)
    if len(relative.parts) != 1:
        raise BundleError("import output must be one fresh directory directly under " +
                          DEFAULT_IMPORT_ROOT)
    validate_repo_path(relative.as_posix())
    return destination


def _directory_open_flags():
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_directory_at(parent_fd, name):
    try:
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except OSError as error:
        raise BundleError("cannot safely open import directory " +
                          repr(name) + ": " + str(error))
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise BundleError("import path is not a directory: " + repr(name))
    return descriptor


def _open_import_root(repo):
    """Open ai/backlog-imports through no-follow directory descriptors."""
    try:
        current = os.open(str(repo), _directory_open_flags())
    except OSError as error:
        raise BundleError("cannot safely open repository root: " + str(error))
    try:
        child = _open_directory_at(current, "ai")
        os.close(current)
        current = child
        try:
            os.mkdir("backlog-imports", 0o700, dir_fd=current)
        except FileExistsError:
            pass
        child = _open_directory_at(current, "backlog-imports")
        os.close(current)
        return child
    except Exception:
        try:
            os.close(current)
        except OSError:
            pass
        raise


def _write_file_at(parent_fd, name, data):
    """Create one mode-0600 regular file without following a name."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
    except OSError as error:
        raise BundleError("cannot exclusively create import file " +
                          repr(name) + ": " + str(error))
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise BundleError("short write while creating import file " +
                                  repr(name))
            offset += written
        result = os.fstat(descriptor)
        if not stat.S_ISREG(result.st_mode) or result.st_size != len(data):
            raise BundleError("import file identity changed during creation: " +
                              repr(name))
    finally:
        os.close(descriptor)


def _open_or_create_directory_at(parent_fd, name):
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return _open_directory_at(parent_fd, name)


def _read_file_at(root_fd, parts, expected_size):
    current = os.dup(root_fd)
    descriptor = None
    try:
        for part in parts[:-1]:
            child = _open_directory_at(current, part)
            os.close(current)
            current = child
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(parts[-1], flags, dir_fd=current)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            return None
        chunks = []
        remaining = expected_size + 1
        while remaining:
            chunk = os.read(descriptor, min(IO_CHUNK, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino, before.st_size,
             getattr(before, "st_mtime_ns", int(before.st_mtime * 1e9))) !=
                (after.st_dev, after.st_ino, after.st_size,
                 getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)))):
            return None
        return b"".join(chunks)
    except (BundleError, OSError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(current)


def _collect_import_tree(root_fd):
    files = set()
    directories = set()
    budget = [0]

    def visit(directory_fd, prefix, depth):
        if depth > MAX_PATH_DEPTH + 2:
            return False
        try:
            names = os.listdir(directory_fd)
        except OSError:
            return False
        for name in names:
            budget[0] += 1
            if budget[0] > (MAX_MEMBERS * (MAX_PATH_DEPTH + 2)):
                return False
            relative = prefix + "/" + name if prefix else name
            try:
                item = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                return False
            if stat.S_ISDIR(item.st_mode):
                directories.add(relative)
                try:
                    child = _open_directory_at(directory_fd, name)
                except BundleError:
                    return False
                try:
                    if not visit(child, relative, depth + 1):
                        return False
                finally:
                    os.close(child)
            elif stat.S_ISREG(item.st_mode):
                files.add(relative)
            else:
                return False
        return True

    if not visit(root_fd, "", 0):
        return None
    return files, directories


def _existing_import_is_exact(destination_fd, manifest, payload, bundle_id):
    """Return true only for a complete, byte-identical prior import."""
    expected = {
        ".COMPLETE": (bundle_id + "\n").encode("ascii"),
        "manifest.json": canonical_json(manifest),
    }
    for path_text, data in payload.items():
        expected["payload/" + path_text] = data
    expected_directories = set()
    for relative in expected:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    found = _collect_import_tree(destination_fd)
    if found is None:
        return False
    found_files, found_directories = found
    if (found_files != set(expected) or
            found_directories != expected_directories):
        return False
    for relative, wanted in expected.items():
        observed = _read_file_at(
            destination_fd, relative.split("/"), len(wanted))
        if observed != wanted:
            return False
    return True


def unpack_archive(repo, archive_path, requested_output):
    manifest, payload, bundle_id, archive_digest = validate_archive(archive_path)
    base_is_ancestor = verify_import_repository(repo, manifest)
    destination = _import_destination(repo, requested_output, bundle_id)
    import_root_fd = _open_import_root(repo)
    destination_fd = None
    try:
        try:
            os.mkdir(destination.name, 0o700, dir_fd=import_root_fd)
            created = True
        except FileExistsError:
            created = False
        destination_fd = _open_directory_at(import_root_fd, destination.name)
        if not created:
            if _existing_import_is_exact(
                    destination_fd, manifest, payload, bundle_id):
                print("Already imported:", destination)
                return
            raise BundleError(
                "refusing to reuse existing import directory: " +
                str(destination))
        _write_file_at(
            destination_fd, ".INCOMPLETE",
            (bundle_id + "\n").encode("ascii"))
        _write_file_at(
            destination_fd, "manifest.json", canonical_json(manifest))
        payload_root_fd = _open_or_create_directory_at(
            destination_fd, "payload")
        try:
            for path_text in sorted(payload):
                parts = path_text.split("/")
                current = os.dup(payload_root_fd)
                try:
                    for part in parts[:-1]:
                        child = _open_or_create_directory_at(current, part)
                        os.close(current)
                        current = child
                    _write_file_at(current, parts[-1], payload[path_text])
                finally:
                    os.close(current)
        finally:
            os.close(payload_root_fd)
        _write_file_at(
            destination_fd, ".COMPLETE",
            (bundle_id + "\n").encode("ascii"))
        try:
            os.unlink(".INCOMPLETE", dir_fd=destination_fd)
        except OSError as error:
            raise BundleError("could not finalize import marker: " + str(error))
        if not _existing_import_is_exact(
                destination_fd, manifest, payload, bundle_id):
            raise BundleError("import changed before final verification")
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(import_root_fd)
    print("Imported to:", destination)
    print("Bundle id:", bundle_id)
    print("Archive SHA-256:", archive_digest)
    if not base_is_ancestor:
        print("Warning: archive base is present but is not an ancestor of HEAD.",
              file=sys.stderr)
    print("Review the staged copy; live notes were not modified.")


def command_pack(args):
    repo = repository_root()
    manifest, manifest_bytes, payload = build_bundle(repo, args.include)
    bundle_id = hashlib.sha256(manifest_bytes).hexdigest()
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = repo / output
    else:
        output = repo / DEFAULT_BUNDLE_ROOT / (
            "backlog-" + bundle_id[:16] + ".backlog-bundle.tar.xz")
    output = output.expanduser().absolute()
    try:
        output_relative = output.resolve(strict=False).relative_to(
            repo.resolve()).as_posix()
    except ValueError:
        output_relative = None
    if output_relative is not None:
        ignored = _git(
            repo, "check-ignore", "-q", "--no-index", "--",
            output_relative, check=False)
        if ignored.returncode != 0:
            raise BundleError(
                "repository-contained archive output is not ignored: " +
                output_relative)
    print("Bundle id:", bundle_id)
    print("Selected files:", len(payload))
    for record in manifest["files"]:
        print("  " + record["role"] + " " + json.dumps(record["path"]))
    print("Open items:", len(manifest["open_items"]))
    if args.dry_run:
        print("Dry run; archive not written:", output)
        return
    archive_digest = write_archive(output, manifest_bytes, payload)
    print("Wrote:", output)
    print("Archive SHA-256:", archive_digest)


def command_inspect(args):
    manifest, payload, bundle_id, archive_digest = validate_archive(args.archive)
    print_inspection(manifest, bundle_id, archive_digest,
                     args.show_backlog, payload)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Package and safely read an offline backlog handoff.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser(
        "pack", help="create a deterministic .tar.xz email attachment")
    pack.add_argument(
        "--include", action="append", default=[], metavar="REPO_PATH",
        help="also include one regular repository-relative file; repeatable")
    pack.add_argument("--output", metavar="ARCHIVE",
                      help="archive path; existing files are never replaced")
    pack.add_argument("--dry-run", action="store_true",
                      help="validate and list the selection without writing")
    pack.set_defaults(handler=command_pack)

    inspect = subparsers.add_parser(
        "inspect", aliases=["read"],
        help="fully validate and summarize an incoming archive")
    inspect.add_argument("archive", metavar="ARCHIVE")
    inspect.add_argument("--show-backlog", action="store_true",
                         help="print sanitized backlog contents after validation")
    inspect.set_defaults(handler=command_inspect)

    unpack = subparsers.add_parser(
        "unpack", aliases=["import"],
        help="validate and copy an archive into a fresh review directory")
    unpack.add_argument("archive", metavar="ARCHIVE")
    unpack.add_argument(
        "--output", metavar="DIR",
        help="fresh directory under ai/backlog-imports/ (default: bundle id)")
    unpack.set_defaults(
        handler=lambda args: unpack_archive(repository_root(), args.archive,
                                            args.output))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except BundleError as error:
        parser.exit(2, "backlog_bundle.py: refusal: " + str(error) + "\n")


if __name__ == "__main__":
    main()
