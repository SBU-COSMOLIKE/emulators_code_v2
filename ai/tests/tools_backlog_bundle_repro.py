#!/usr/bin/env python3
"""Scratch-only regression witness for portable backlog bundles.

Every arm copies and imports the shipped tool inside a temporary fake Git
repository.  The witness never packages, reads, or imports the live backlog.
It exercises the public command line as well as hostile archive inputs that an
emailed handoff must refuse.
"""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


AI_ROOT = Path(__file__).resolve().parents[1]
SOURCE = AI_ROOT / "tools" / "backlog_bundle.py"
SOURCE_ROLE_READER = AI_ROOT / "tools" / "role_contract.py"
SOURCE_ROLE_CONTRACT = AI_ROOT / "notes" / "role-contract.yaml"
ORIGIN = "git@github.com:ExampleOrg/ExampleRepo.git"
ROLE_CONTRACT = "ai/notes/role-contract.yaml"

PERMANENT_NOTES = (
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
    "ai/notes/python-changes-go-no-go.md",
)

BACKLOG = (
    b"# Scratch backlog\n\n"
    b"- OPEN transfer the unfinished repair and its evidence\n"
    b"- CLOSED already landed\n"
)
LOCAL_NOTE = b"# Local repair evidence\n\nThe unfinished observation.\n"
SUPPORT_BYTES = b"\x00\xffscratch-binary\n\x10"
EXPLICIT_BYTES = b"diff --git a/example b/example\n+attempt\n"
MAILBOX_BYTES = b"routing summary that must not enter the bundle\n"
RELAY_BYTES = b"raw transport log that must not enter the bundle\n"


def run_git(repo, *arguments):
    """Run one Git command in the fake repository and return its output."""
    result = subprocess.run(
        ["git", "-C", str(repo)] + list(arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def write_bytes(repo, path_text, data):
    """Write one fixture below the temporary repository."""
    path = repo.joinpath(*path_text.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def load_tool(path, name):
    """Import the copied production tool under a unique scratch name."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
        sys.modules.pop("role_contract", None)
    return module


@contextlib.contextmanager
def scratch_repository(label):
    """Yield a complete fake repository and its copied/imported tool."""
    with tempfile.TemporaryDirectory(prefix="backlog-bundle-" + label + "-") as tmp:
        repo = Path(tmp) / "repo"
        tool_path = repo / "ai" / "tools" / "backlog_bundle.py"
        tool_path.parent.mkdir(parents=True)
        shutil.copy2(str(SOURCE), str(tool_path))
        shutil.copy2(str(SOURCE_ROLE_READER),
                     str(tool_path.parent / "role_contract.py"))

        run_git(repo, "init", "-q", "-b", "main")
        run_git(repo, "config", "user.name", "Scratch Bundle Test")
        run_git(repo, "config", "user.email", "scratch@example.invalid")
        run_git(repo, "remote", "add", "origin", ORIGIN)

        ignore_lines = ["ai/notes/*.md"]
        for path_text in PERMANENT_NOTES:
            ignore_lines.append("!" + path_text)
        ignore_lines.extend([
            "ai/backlog-bundles/",
            "ai/backlog-imports/",
            "*.backlog-bundle.tar.xz",
        ])
        write_bytes(repo, ".gitignore",
                    ("\n".join(ignore_lines) + "\n").encode("utf-8"))
        for index, path_text in enumerate(PERMANENT_NOTES):
            content = ("# Permanent note " + str(index) + "\n").encode("utf-8")
            write_bytes(repo, path_text, content)
        write_bytes(repo, ROLE_CONTRACT, SOURCE_ROLE_CONTRACT.read_bytes())

        run_git(repo, "add", ".gitignore", "ai/tools/backlog_bundle.py",
                "ai/tools/role_contract.py",
                *PERMANENT_NOTES, ROLE_CONTRACT)
        run_git(repo, "commit", "-q", "-m", "scratch permanent base")

        write_bytes(repo, "ai/notes/backlog.md", BACKLOG)
        write_bytes(repo, "ai/notes/local-repair.md", LOCAL_NOTE)
        write_bytes(
            repo,
            "ai/notes/backlog-support/nested/evidence.bin",
            SUPPORT_BYTES,
        )
        write_bytes(repo, "evidence/attempt.patch", EXPLICIT_BYTES)
        write_bytes(repo, "ai/notes/mailbox/0001-to-opus.md", MAILBOX_BYTES)
        write_bytes(repo, "ai/notes/mailbox/done/0002-to-fable.md",
                    MAILBOX_BYTES)
        write_bytes(repo, "ai/notes/relay/dispatch-opus.log", RELAY_BYTES)

        module = load_tool(tool_path, "scratch_backlog_bundle_" + label)
        assert set(module.PERMANENT_NOTES) == set(PERMANENT_NOTES)
        yield repo, tool_path, module


def run_cli(repo, tool_path, *arguments, timeout=20):
    """Run the copied tool through its public command line."""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-B", str(tool_path)] + list(arguments),
        cwd=str(repo),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def assert_ok(result, label):
    """Require one successful CLI result with useful diagnostics on failure."""
    assert result.returncode == 0, (
        label + " failed\nstdout:\n" + result.stdout
        + "\nstderr:\n" + result.stderr)


def assert_refused(result, *fragments):
    """Require a fail-closed CLI result containing each diagnostic fragment."""
    combined = (result.stdout + "\n" + result.stderr).lower()
    assert result.returncode == 2, (
        "expected refusal, got " + str(result.returncode)
        + "\n" + combined)
    assert "traceback" not in combined, (
        "refusal leaked a Python traceback\n" + combined)
    for fragment in fragments:
        assert fragment.lower() in combined, (
            "missing refusal fragment " + repr(fragment) + "\n" + combined)


def tree_snapshot(root):
    """Return a byte-and-type snapshot of a scratch tree."""
    snapshot = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot.append((relative, "symlink", os.readlink(str(path))))
        elif path.is_file():
            snapshot.append((relative, "file", path.read_bytes()))
        elif path.is_dir():
            snapshot.append((relative, "directory", b""))
        else:
            snapshot.append((relative, "other", b""))
    return snapshot


def arm_roundtrip_and_exact_payload():
    """Prove deterministic pack/read/unpack and exact selection semantics."""
    with scratch_repository("roundtrip") as (repo, tool_path, _module):
        notes = repo / "ai" / "notes"
        before = tree_snapshot(notes)
        first = repo / "first.backlog-bundle.tar.xz"
        second = repo / "second.backlog-bundle.tar.xz"
        include = ["--include", "evidence/attempt.patch"]

        packed = run_cli(repo, tool_path, "pack", "--output", str(first),
                         *include)
        assert_ok(packed, "first pack")
        first_bytes = first.read_bytes()
        recovered = run_cli(repo, tool_path, "pack", "--output", str(first),
                            *include)
        assert_ok(recovered, "retry after published output")
        assert first.read_bytes() == first_bytes
        packed_again = run_cli(repo, tool_path, "pack", "--output", str(second),
                               *include)
        assert_ok(packed_again, "second pack")
        assert first.read_bytes() == second.read_bytes()

        inspected = run_cli(repo, tool_path, "read", str(first),
                            "--show-backlog")
        assert_ok(inspected, "read")
        assert "Files: 4" in inspected.stdout
        assert "Open items: 1" in inspected.stdout
        assert "transfer the unfinished repair" in inspected.stdout

        destination_text = "ai/backlog-imports/roundtrip-review"
        imported = run_cli(repo, tool_path, "unpack", str(first),
                           "--output", destination_text)
        assert_ok(imported, "unpack")
        destination = repo / destination_text
        payload_root = destination / "payload"
        payload_files = {
            path.relative_to(payload_root).as_posix(): path.read_bytes()
            for path in payload_root.rglob("*") if path.is_file()
        }
        expected = {
            "ai/notes/backlog-support/nested/evidence.bin": SUPPORT_BYTES,
            "ai/notes/backlog.md": BACKLOG,
            "ai/notes/local-repair.md": LOCAL_NOTE,
            "evidence/attempt.patch": EXPLICIT_BYTES,
        }
        assert payload_files == expected

        manifest = json.loads(
            (destination / "manifest.json").read_text(encoding="ascii"))
        roles = {
            record["path"]: record["role"] for record in manifest["files"]
        }
        assert roles == {
            "ai/notes/backlog-support/nested/evidence.bin": "support",
            "ai/notes/backlog.md": "backlog",
            "ai/notes/local-repair.md": "local-note",
            "evidence/attempt.patch": "explicit",
        }
        assert manifest["repository"]["id"] == (
            "github.com/exampleorg/examplerepo")
        assert repo.name == "repo"
        assert manifest["repository"]["name"] == "examplerepo"
        assert manifest["base_commit"] == run_git(repo, "rev-parse", "HEAD")
        assert not any(path in payload_files for path in PERMANENT_NOTES)
        assert not any("mailbox" in path or "relay" in path
                       for path in payload_files)
        assert (destination / ".COMPLETE").is_file()
        assert not (destination / ".INCOMPLETE").exists()

        repeated = run_cli(repo, tool_path, "unpack", str(first),
                           "--output", destination_text)
        assert_ok(repeated, "idempotent unpack")
        assert "Already imported:" in repeated.stdout
        assert tree_snapshot(notes) == before
    return True


def arm_existing_output_and_import_collisions():
    """Prove output and staging collisions preserve every existing byte."""
    with scratch_repository("collisions") as (repo, tool_path, _module):
        notes_before = tree_snapshot(repo / "ai" / "notes")
        occupied = repo / "occupied.backlog-bundle.tar.xz"
        occupied.write_bytes(b"do-not-overwrite\n")
        packed = run_cli(repo, tool_path, "pack", "--output", str(occupied))
        assert_refused(packed, "refusing to overwrite")
        assert occupied.read_bytes() == b"do-not-overwrite\n"

        different = repo / "different.backlog-bundle.tar.xz"
        with_extra = run_cli(
            repo, tool_path, "pack", "--output", str(different),
            "--include", "evidence/attempt.patch")
        assert_ok(with_extra, "different-valid fixture pack")
        different_bytes = different.read_bytes()
        without_extra = run_cli(
            repo, tool_path, "pack", "--output", str(different))
        assert_refused(without_extra, "refusing to overwrite")
        assert different.read_bytes() == different_bytes

        target = repo / "symlink-target.backlog-bundle.tar.xz"
        target.write_bytes(b"keep-symlink-target\n")
        linked_output = repo / "symlink-output.backlog-bundle.tar.xz"
        linked_output.symlink_to(target.name)
        linked_pack = run_cli(
            repo, tool_path, "pack", "--output", str(linked_output))
        assert_refused(linked_pack, "refusing to overwrite")
        assert linked_output.is_symlink()
        assert target.read_bytes() == b"keep-symlink-target\n"

        valid = repo / "valid.backlog-bundle.tar.xz"
        result = run_cli(repo, tool_path, "pack", "--output", str(valid))
        assert_ok(result, "collision fixture pack")
        destination = repo / "ai" / "backlog-imports" / "occupied"
        destination.mkdir(parents=True)
        sentinel = destination / "sentinel.txt"
        sentinel.write_bytes(b"keep-existing-import\n")
        before = tree_snapshot(destination)
        imported = run_cli(
            repo, tool_path, "unpack", str(valid),
            "--output", "ai/backlog-imports/occupied")
        assert_refused(imported, "refusing to reuse existing import directory")
        assert tree_snapshot(destination) == before

        tampered_text = "ai/backlog-imports/tampered"
        first_import = run_cli(
            repo, tool_path, "unpack", str(valid),
            "--output", tampered_text)
        assert_ok(first_import, "tampering fixture import")
        tampered = repo / tampered_text
        staged_backlog = tampered / "payload" / "ai" / "notes" / "backlog.md"
        staged_backlog.write_bytes(b"tampered completed import\n")
        tampered_before = tree_snapshot(tampered)
        repeated = run_cli(
            repo, tool_path, "unpack", str(valid),
            "--output", tampered_text)
        assert_refused(repeated, "refusing to reuse existing import directory")
        assert tree_snapshot(tampered) == tampered_before
        assert tree_snapshot(repo / "ai" / "notes") == notes_before
    return True


def arm_interrupted_import_recovery():
    """Resume only an exact partial import bearing this bundle's marker."""
    with scratch_repository("import-recovery") as (repo, tool_path, module):
        archive = repo / "valid.backlog-bundle.tar.xz"
        packed = run_cli(repo, tool_path, "pack", "--output", str(archive))
        assert_ok(packed, "import-recovery fixture pack")
        manifest, payload, bundle_id, _digest = module.validate_archive(
            archive)
        marker = (bundle_id + "\n").encode("ascii")
        manifest_bytes = module.canonical_json(manifest)

        partial_text = "ai/backlog-imports/partial"
        partial = repo / partial_text
        partial.mkdir(parents=True)
        (partial / ".INCOMPLETE").write_bytes(marker)
        (partial / "manifest.json").write_bytes(manifest_bytes)
        retained_name = sorted(payload)[0]
        retained = partial / "payload" / retained_name
        retained.parent.mkdir(parents=True)
        retained.write_bytes(payload[retained_name])
        resumed = run_cli(
            repo, tool_path, "unpack", str(archive),
            "--output", partial_text)
        assert_ok(resumed, "resume partial import")
        repeated = run_cli(
            repo, tool_path, "unpack", str(archive),
            "--output", partial_text)
        assert_ok(repeated, "repeat resumed import")
        assert "Already imported:" in repeated.stdout
        assert retained.read_bytes() == payload[retained_name]

        final_text = "ai/backlog-imports/final-marker"
        finalized = run_cli(
            repo, tool_path, "unpack", str(archive), "--output", final_text)
        assert_ok(finalized, "complete-marker fixture import")
        final = repo / final_text
        (final / ".INCOMPLETE").write_bytes(marker)
        recovered_final = run_cli(
            repo, tool_path, "unpack", str(archive), "--output", final_text)
        assert_ok(recovered_final, "recover complete plus marker")
        assert not (final / ".INCOMPLETE").exists()

        def partial_directory(name, marker_bytes=marker,
                              contents=manifest_bytes):
            destination = repo / "ai" / "backlog-imports" / name
            destination.mkdir(parents=True)
            if marker_bytes is not None:
                (destination / ".INCOMPLETE").write_bytes(marker_bytes)
            (destination / "manifest.json").write_bytes(contents)
            return destination

        unsafe = [
            partial_directory("wrong-marker", b"wrong-bundle\n"),
            partial_directory("unmarked", None),
            partial_directory("corrupt", contents=b"wrong\n"),
        ]
        extra = partial_directory("extra")
        (extra / "unexpected.txt").write_bytes(b"keep\n")
        unsafe.append(extra)
        linked = partial_directory("linked")
        (linked / "payload").symlink_to(repo / "ai" / "notes")
        unsafe.append(linked)
        fifo = partial_directory("fifo")
        os.mkfifo(str(fifo / "blocked.fifo"), 0o600)
        unsafe.append(fifo)

        for destination in unsafe:
            before = tree_snapshot(destination)
            refused = run_cli(
                repo, tool_path, "unpack", str(archive),
                "--output", destination.relative_to(repo).as_posix())
            assert_refused(
                refused, "refusing to reuse existing import directory")
            assert tree_snapshot(destination) == before

        original_write = module._write_file_at

        def change_after_complete(parent_fd, name, data):
            original_write(parent_fd, name, data)
            if name == ".COMPLETE":
                original_write(parent_fd, "unexpected.txt", b"changed\n")

        module._write_file_at = change_after_complete
        try:
            try:
                module.unpack_archive(
                    repo, archive,
                    "ai/backlog-imports/finalization-check")
            except module.BundleError as error:
                finalization_refused = (
                    "changed before finalization" in str(error))
            else:
                finalization_refused = False
        finally:
            module._write_file_at = original_write
        finalization = (
            repo / "ai" / "backlog-imports" / "finalization-check")
        assert finalization_refused
        assert (finalization / ".INCOMPLETE").is_file()
        assert (finalization / ".COMPLETE").is_file()
    return True


def arm_dirty_permanent_and_source_symlink_refusals():
    """Prove unruled permanent edits and linked support fail before output."""
    with scratch_repository("sources") as (repo, tool_path, _module):
        notes = repo / "ai" / "notes"
        original = (repo / PERMANENT_NOTES[0]).read_bytes()
        before = tree_snapshot(notes)
        dirty_output = repo / "dirty.backlog-bundle.tar.xz"
        (repo / PERMANENT_NOTES[0]).write_bytes(original + b"unruled edit\n")
        dirty = run_cli(repo, tool_path, "pack", "--output",
                        str(dirty_output))
        assert_refused(dirty, "permanent notes differ from head",
                       PERMANENT_NOTES[0], "architect")
        assert not dirty_output.exists()
        (repo / PERMANENT_NOTES[0]).write_bytes(original)

        support = notes / "backlog-support" / "nested"
        linked = support / "linked.bin"
        os.symlink("evidence.bin", str(linked))
        linked_output = repo / "linked.backlog-bundle.tar.xz"
        symlinked = run_cli(repo, tool_path, "pack", "--output",
                            str(linked_output))
        assert_refused(symlinked, "symlink", "linked.bin")
        assert not linked_output.exists()
        linked.unlink()

        fifo = support / "blocked.fifo"
        os.mkfifo(str(fifo), 0o600)
        fifo_output = repo / "fifo.backlog-bundle.tar.xz"
        try:
            fifo_result = run_cli(
                repo, tool_path, "pack", "--output", str(fifo_output),
                timeout=3)
        finally:
            fifo.unlink()
        assert_refused(fifo_result, "not a regular file", "blocked.fifo")
        assert not fifo_output.exists()
        assert tree_snapshot(notes) == before
    return True


def arm_pack_bounds_and_output_policy():
    """Prove USTAR/member bounds and in-repository output ignore policy."""
    with scratch_repository("pack-policy") as (repo, tool_path, module):
        unsafe_output_text = "exports/handoff-package.txt"
        unsafe_output = repo / unsafe_output_text
        ignored = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-q", "--no-index",
             "--", unsafe_output_text])
        assert ignored.returncode == 1
        unignored = run_cli(repo, tool_path, "pack", "--output",
                            unsafe_output_text)
        assert_refused(unignored, "repository-contained archive output",
                       "not ignored")
        assert not unsafe_output.exists()

        long_path = (
            "evidence/" + "a" * 90 + "/" + "b" * 90 + "/payload.bin")
        write_bytes(repo, long_path, b"long-name evidence\n")
        overflow_output = repo.parent / "overflow.backlog-bundle.tar.xz"
        overflow = run_cli(
            repo, tool_path, "pack", "--output", str(overflow_output),
            "--include", long_path)
        assert_refused(overflow, "cannot be represented in ustar")
        assert not overflow_output.exists()

        original_limit = module.MAX_MEMBERS
        module.MAX_MEMBERS = 2
        try:
            try:
                module.build_bundle(repo.resolve(), [])
            except module.BundleError as error:
                bounded = "files list is invalid" in str(error).lower()
            else:
                bounded = False
        finally:
            module.MAX_MEMBERS = original_limit
        assert bounded
    return True


def arm_symlinked_import_root_refusal():
    """Prove an import-root symlink cannot redirect received payload bytes."""
    with scratch_repository("import-symlink") as (repo, tool_path, _module):
        archive = repo / "valid.backlog-bundle.tar.xz"
        packed = run_cli(repo, tool_path, "pack", "--output", str(archive))
        assert_ok(packed, "symlink-root fixture pack")

        redirected = repo.parent / "redirected-imports"
        redirected.mkdir()
        import_root = repo / "ai" / "backlog-imports"
        os.symlink(str(redirected), str(import_root))
        before = tree_snapshot(redirected)
        imported = run_cli(repo, tool_path, "unpack", str(archive))
        assert_refused(imported, "cannot safely open import directory",
                       "backlog-imports")
        assert tree_snapshot(redirected) == before
        assert import_root.is_symlink()
    return True


def regular_info(name, data):
    """Return one deterministic regular USTAR member and its bytes."""
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.REGTYPE
    return info, data


def link_info(name, target):
    """Return one deliberately hostile symbolic-link USTAR member."""
    info = tarfile.TarInfo(name)
    info.size = 0
    info.mode = 0o777
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


def write_archive(path, members):
    """Write a canonical-XZ USTAR containing the supplied member tuples."""
    with lzma.LZMAFile(
            str(path), "wb", format=lzma.FORMAT_XZ,
            check=lzma.CHECK_CRC64,
            filters=[{"id": lzma.FILTER_LZMA2, "preset": 6}]) as compressed:
        with tarfile.open(fileobj=compressed, mode="w",
                          format=tarfile.USTAR_FORMAT,
                          encoding="utf-8", errors="strict") as archive:
            for info, data in members:
                stream = None if data is None else io.BytesIO(data)
                archive.addfile(info, fileobj=stream)


def write_xz_stream(path, data):
    """Compress exact raw tar bytes as one canonical XZ stream."""
    path.write_bytes(lzma.compress(
        data,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 6}],
    ))


def minimal_manifest(module, manifest, payload):
    """Reduce a valid manifest to its required backlog payload."""
    result = copy.deepcopy(manifest)
    backlog_record = None
    for record in manifest["files"]:
        if record["path"] == module.BACKLOG_PATH:
            backlog_record = copy.deepcopy(record)
            break
    assert backlog_record is not None
    result["files"] = [backlog_record]
    result["open_items"] = module.open_backlog_lines(
        payload[module.BACKLOG_PATH])
    return result


def inspect_refusal(repo, tool_path, path, *fragments):
    """Run public read validation and require a bounded refusal."""
    result = run_cli(repo, tool_path, "read", str(path))
    assert_refused(result, *fragments)


def arm_hostile_archives():
    """Reject traversal, links, duplicates, corruption, and XZ framing tricks."""
    with scratch_repository("hostile") as (repo, tool_path, module):
        valid = repo / "valid.backlog-bundle.tar.xz"
        packed = run_cli(repo, tool_path, "pack", "--output", str(valid))
        assert_ok(packed, "hostile baseline pack")
        manifest, payload, _bundle_id, _archive_digest = (
            module.validate_archive(valid))
        minimal = minimal_manifest(module, manifest, payload)
        backlog = payload[module.BACKLOG_PATH]
        manifest_name = module.MANIFEST_MEMBER
        backlog_name = module.PAYLOAD_PREFIX + module.BACKLOG_PATH
        notes_before = tree_snapshot(repo / "ai" / "notes")

        canonical = repo / "canonical-minimal.backlog-bundle.tar.xz"
        manifest_bytes = module.canonical_json(minimal)
        write_archive(canonical, [
            regular_info(manifest_name, manifest_bytes),
            regular_info(backlog_name, backlog),
        ])
        canonical_tar = lzma.decompress(
            canonical.read_bytes(), format=lzma.FORMAT_XZ)

        padding_start = 512 + len(manifest_bytes)
        padding_end = 512 + ((len(manifest_bytes) + 511) // 512) * 512
        assert padding_start < padding_end
        nonzero_padding_tar = bytearray(canonical_tar)
        nonzero_padding_tar[padding_start] = 1
        nonzero_padding = repo / "member-padding.backlog-bundle.tar.xz"
        write_xz_stream(nonzero_padding, bytes(nonzero_padding_tar))
        inspect_refusal(repo, tool_path, nonzero_padding, "padding")

        noncanonical_end = repo / "end-record.backlog-bundle.tar.xz"
        write_xz_stream(noncanonical_end, canonical_tar + b"\0" * 512)
        inspect_refusal(repo, tool_path, noncanonical_end, "end")

        noncanonical = repo / "mtime.backlog-bundle.tar.xz"
        mtime_member = regular_info(
            manifest_name, module.canonical_json(minimal))
        mtime_member[0].mtime = 1
        write_archive(noncanonical, [
            mtime_member,
            regular_info(backlog_name, backlog),
        ])
        inspect_refusal(repo, tool_path, noncanonical,
                        "header is not canonical")

        boolean_version = copy.deepcopy(minimal)
        boolean_version["version"] = True
        version_archive = repo / "boolean-version.backlog-bundle.tar.xz"
        write_archive(version_archive, [
            regular_info(manifest_name,
                         module.canonical_json(boolean_version)),
            regular_info(backlog_name, backlog),
        ])
        inspect_refusal(repo, tool_path, version_archive,
                        "unsupported bundle format or version")

        list_role = copy.deepcopy(minimal)
        list_role["files"][0]["role"] = []
        role_archive = repo / "list-role.backlog-bundle.tar.xz"
        write_archive(role_archive, [
            regular_info(manifest_name, module.canonical_json(list_role)),
            regular_info(backlog_name, backlog),
        ])
        inspect_refusal(repo, tool_path, role_archive, "invalid role")

        traversal_manifest = copy.deepcopy(minimal)
        traversal_data = b"escape\n"
        traversal_manifest["files"].append({
            "path": "../escape.txt",
            "role": "explicit",
            "sha256": hashlib.sha256(traversal_data).hexdigest(),
            "size": len(traversal_data),
        })
        traversal_manifest["files"].sort(key=lambda record: record["path"])
        traversal = repo / "traversal.backlog-bundle.tar.xz"
        traversal_members = [
            regular_info(manifest_name,
                         module.canonical_json(traversal_manifest)),
        ]
        for record in traversal_manifest["files"]:
            data = (traversal_data if record["path"] == "../escape.txt"
                    else backlog)
            traversal_members.append(regular_info(
                module.PAYLOAD_PREFIX + record["path"], data))
        write_archive(traversal, traversal_members)
        inspect_refusal(repo, tool_path, traversal, "unsafe components")

        link = repo / "link.backlog-bundle.tar.xz"
        write_archive(link, [
            regular_info(manifest_name, module.canonical_json(minimal)),
            link_info(backlog_name, "../../outside"),
        ])
        inspect_refusal(repo, tool_path, link, "links")

        duplicate = repo / "duplicate.backlog-bundle.tar.xz"
        write_archive(duplicate, [
            regular_info(manifest_name, module.canonical_json(minimal)),
            regular_info(backlog_name, backlog),
            regular_info(backlog_name, backlog),
        ])
        inspect_refusal(repo, tool_path, duplicate, "duplicate")

        corrupt = repo / "digest.backlog-bundle.tar.xz"
        corrupt_backlog = backlog[:-1] + bytes([backlog[-1] ^ 1])
        write_archive(corrupt, [
            regular_info(manifest_name, module.canonical_json(minimal)),
            regular_info(backlog_name, corrupt_backlog),
        ])
        inspect_refusal(repo, tool_path, corrupt, "digest mismatch")

        truncated = repo / "truncated.backlog-bundle.tar.xz"
        truncated.write_bytes(valid.read_bytes()[:-12])
        inspect_refusal(repo, tool_path, truncated, "xz")

        concatenated = repo / "concatenated.backlog-bundle.tar.xz"
        concatenated.write_bytes(
            valid.read_bytes() + lzma.compress(
                b"second stream", format=lzma.FORMAT_XZ))
        inspect_refusal(repo, tool_path, concatenated,
                        "concatenated or trailing xz")

        assert tree_snapshot(repo / "ai" / "notes") == notes_before
        assert not (repo / "ai" / "backlog-imports").exists()
    return True


def main():
    """Run every isolated bundle regression and return nonzero on failure."""
    arms = [
        ("deterministic roundtrip and exact payload",
         arm_roundtrip_and_exact_payload),
        ("existing output and import collisions",
         arm_existing_output_and_import_collisions),
        ("interrupted import recovery", arm_interrupted_import_recovery),
        ("dirty permanent note and source symlink refusals",
         arm_dirty_permanent_and_source_symlink_refusals),
        ("pack bounds and output policy",
         arm_pack_bounds_and_output_policy),
        ("symlinked import root refusal",
         arm_symlinked_import_root_refusal),
        ("hostile archive refusals", arm_hostile_archives),
    ]
    failures = []
    for label, arm in arms:
        try:
            passed = arm()
        except BaseException as error:
            passed = False
            print("ERROR " + label + ": " + type(error).__name__
                  + ": " + str(error))
        print(("PASS " if passed else "FAIL ") + label)
        if not passed:
            failures.append(label)
    print("runtime-summary passed=" + str(len(arms) - len(failures))
          + "/" + str(len(arms)) + " failures=" + repr(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
