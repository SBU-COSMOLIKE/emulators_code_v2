"""CPU checks for publishing and reopening one saved-emulator file pair.

A trained emulator is saved in two files.  ``<root>.emul`` contains the
learned tensors, while ``<root>.h5`` explains how those tensors must be
interpreted.  These tests check that the two files cannot be silently mixed.

The fixture is deliberately small: it is the same two-input, one-output model
used by the compile-recipe gate.  No training data, CosmoLike installation, or
GPU is required.
"""

import hashlib
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

import h5py
import numpy as np
import torch

from ai.gates.checks.compile_recipe import save_fixture
from emulator import results
from emulator import warmstart
from emulator.designs.plain import ResMLP


_ARTIFACT_ID_ATTR = "artifact_id"
_CHECKPOINT_SHA256_ATTR = "checkpoint_sha256"
_COMMENT_PREFIX = b"emulators_code_v2 artifact_id v1:"
_LOWER_HEX = frozenset("0123456789abcdef")


def _checkpoint_path(root):
  """Return the learned-tensor member of one fixture pair."""
  return Path(str(root) + ".emul")


def _record_path(root):
  """Return the scientific-record member of one fixture pair."""
  return Path(str(root) + ".h5")


def _pending_path(root):
  """Return the marker that says publication stopped between file renames."""
  return Path(str(root) + ".pair-pending")


def _sha256(path):
  """Hash the exact bytes stored at ``path`` without loading the checkpoint."""
  digest = hashlib.sha256()
  with open(path, "rb") as stream:
    while True:
      block = stream.read(1024 * 1024)
      if not block:
        break
      digest.update(block)
  return digest.hexdigest()


def _zip_artifact_id(path):
  """Read the publication identifier from the safe ZIP comment."""
  with zipfile.ZipFile(path, "r") as archive:
    comment = archive.comment
  if not comment.startswith(_COMMENT_PREFIX):
    raise AssertionError(
      "checkpoint ZIP comment does not start with the artifact-id prefix")
  return comment[len(_COMMENT_PREFIX):].strip().decode("ascii")


def _pair_bytes(root):
  """Return immutable snapshots of both final files."""
  return (_checkpoint_path(root).read_bytes(),
          _record_path(root).read_bytes())


def _save(root, variant="default"):
  """Save one complete current-schema fixture with deterministic weights."""
  save_fixture(
    path_root=Path(root),
    compile_mode=variant,
    case_label="artifact-pair-" + variant)


def _rebuild(root):
  """Rebuild one fixture on the CPU without asking PyTorch to compile it."""
  return results.rebuild_emulator(
    path_root=str(root),
    device=torch.device("cpu"),
    compile_model=False)


def _rewrite_binding(root, *, artifact_id=None, checkpoint_sha256=None):
  """Replace either binding field while leaving every other HDF5 fact alone."""
  with h5py.File(_record_path(root), "r+") as record:
    if artifact_id is not None:
      record.attrs[_ARTIFACT_ID_ATTR] = artifact_id
    if checkpoint_sha256 is not None:
      record.attrs[_CHECKPOINT_SHA256_ATTR] = checkpoint_sha256


class _UnsafePickleValue:
  """A checkpoint value that would create a file under unrestricted pickle."""

  def __init__(self, marker):
    self.marker = os.fspath(marker)

  def __reduce__(self):
    """Describe the side effect without running it during ``torch.save``."""
    return os.system, ("touch " + shlex.quote(self.marker),)


class ArtifactPairTests(unittest.TestCase):
  """Exercise successful publication, tampering, and interrupted publication."""

  def test_valid_pair_binds_exact_checkpoint_and_rebuilds(self):
    """A normal save publishes matching native identifiers and loads again."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-valid-") as temp:
      root = Path(temp) / "saved"
      _save(root)

      with h5py.File(_record_path(root), "r") as record:
        artifact_id = record.attrs[_ARTIFACT_ID_ATTR]
        checkpoint_sha256 = record.attrs[_CHECKPOINT_SHA256_ATTR]

      self.assertIs(type(artifact_id), str)
      self.assertEqual(len(artifact_id), 32)
      self.assertTrue(set(artifact_id) <= _LOWER_HEX)
      self.assertEqual(artifact_id, _zip_artifact_id(_checkpoint_path(root)))

      self.assertIs(type(checkpoint_sha256), str)
      self.assertEqual(len(checkpoint_sha256), 64)
      self.assertTrue(set(checkpoint_sha256) <= _LOWER_HEX)
      self.assertEqual(checkpoint_sha256, _sha256(_checkpoint_path(root)))

      direct_state = torch.load(_checkpoint_path(root), weights_only=True)
      self.assertIs(type(direct_state), dict)
      self.assertTrue(direct_state)
      self.assertTrue(all(torch.is_tensor(value)
                          for value in direct_state.values()))

      model, pgeom, geometry, info = _rebuild(root)
      self.assertIsInstance(model, ResMLP)
      self.assertEqual(pgeom.names, ["p0", "p1"])
      self.assertEqual(geometry.names, ["derived"])
      self.assertEqual(info["composition_mode"], "plain")

  def test_warmstart_uses_metadata_from_the_authenticated_open(self):
    """Warm-start does not reopen a pathname after rebuilding its model."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-warmstart-") as temp:
      root = Path(temp) / "saved"
      _save(root)
      real_h5_file = h5py.File
      opens = 0

      def allow_one_h5_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        if opens > 1:
          raise AssertionError(
            "warm-start reopened the HDF5 pathname after authentication")
        return real_h5_file(*args, **kwargs)

      with mock.patch.object(h5py, "File", allow_one_h5_open):
        source = warmstart.load_source(
          root=str(root), device=torch.device("cpu"))

      self.assertEqual(opens, 1)
      self.assertIsInstance(source.model, ResMLP)
      self.assertIsNone(source.data_dir)
      self.assertIsNone(source.dataset)

  def test_same_shape_checkpoint_swap_refuses_before_load_or_construction(self):
    """Weights from another valid model cannot borrow this record's meaning."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-swap-") as temp:
      root_a = Path(temp) / "first"
      root_b = Path(temp) / "second"
      _save(root_a, "default")
      _save(root_b, "reduce-overhead")
      shutil.copyfile(_checkpoint_path(root_b), _checkpoint_path(root_a))

      with mock.patch.object(
          results.torch, "load",
          side_effect=AssertionError("torch.load must not be reached")) \
          as load_checkpoint, mock.patch.object(
            ResMLP, "__init__",
            side_effect=AssertionError("model construction must not run")) \
          as construct_model:
        with self.assertRaisesRegex(
            (KeyError, ValueError, RuntimeError),
            "artifact|binding|checkpoint|digest|SHA|sha|mismatch"):
          _rebuild(root_a)

      load_checkpoint.assert_not_called()
      construct_model.assert_not_called()

  def test_unsafe_pickle_global_is_blocked_without_running_it(self):
    """A digest-matching checkpoint cannot execute an unsafe pickle value."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-pickle-") as temp:
      root = Path(temp) / "saved"
      side_effect = Path(temp) / "pickle-side-effect"
      _save(root)

      with h5py.File(_record_path(root), "r") as record:
        artifact_id = record.attrs[_ARTIFACT_ID_ATTR]

      checkpoint = _checkpoint_path(root)
      torch.save({"weight": _UnsafePickleValue(side_effect)}, checkpoint)
      with zipfile.ZipFile(checkpoint, "a") as archive:
        archive.comment = _COMMENT_PREFIX + artifact_id.encode("ascii")
      _rewrite_binding(root, checkpoint_sha256=_sha256(checkpoint))

      real_torch_load = torch.load
      with mock.patch.object(
          results.torch, "load", wraps=real_torch_load) as load_checkpoint, \
          mock.patch.object(
            ResMLP, "__init__",
            side_effect=AssertionError("model construction must not run")) \
          as construct_model:
        with self.assertRaisesRegex(
            (TypeError, ValueError, RuntimeError),
            "tensor|state|checkpoint|weight"):
          _rebuild(root)

      self.assertEqual(load_checkpoint.call_count, 1)
      self.assertIs(load_checkpoint.call_args.kwargs.get("weights_only"), True)
      self.assertFalse(side_effect.exists())
      construct_model.assert_not_called()

  def test_non_tensor_checkpoint_uses_weights_only_and_refuses(self):
    """A digest-matching but non-tensor mapping is not a model state dict."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-nontensor-") as temp:
      root = Path(temp) / "saved"
      _save(root)

      with h5py.File(_record_path(root), "r") as record:
        artifact_id = record.attrs[_ARTIFACT_ID_ATTR]

      checkpoint = _checkpoint_path(root)
      torch.save({"weight": "this is not a tensor"}, checkpoint)
      with zipfile.ZipFile(checkpoint, "a") as archive:
        archive.comment = _COMMENT_PREFIX + artifact_id.encode("ascii")
      _rewrite_binding(root, checkpoint_sha256=_sha256(checkpoint))

      real_torch_load = torch.load
      with mock.patch.object(
          results.torch, "load", wraps=real_torch_load) as load_checkpoint, \
          mock.patch.object(
            ResMLP, "__init__",
            side_effect=AssertionError("model construction must not run")) \
          as construct_model:
        with self.assertRaisesRegex(
            (TypeError, ValueError, RuntimeError),
            "tensor|state|checkpoint|weight"):
          _rebuild(root)

      self.assertEqual(load_checkpoint.call_count, 1)
      self.assertIs(load_checkpoint.call_args.kwargs.get("weights_only"), True)
      construct_model.assert_not_called()

  def test_missing_or_malformed_binding_refuses_before_load(self):
    """Both required native lowercase-hex fields are checked before PyTorch."""
    cases = (
      ("missing-id", _ARTIFACT_ID_ATTR, None),
      ("short-id", _ARTIFACT_ID_ATTR, "abc"),
      ("uppercase-id", _ARTIFACT_ID_ATTR, "A" * 32),
      ("bytes-id", _ARTIFACT_ID_ATTR, np.bytes_(b"a" * 32)),
      ("missing-sha", _CHECKPOINT_SHA256_ATTR, None),
      ("short-sha", _CHECKPOINT_SHA256_ATTR, "abc"),
      ("uppercase-sha", _CHECKPOINT_SHA256_ATTR, "B" * 64),
      ("bytes-sha", _CHECKPOINT_SHA256_ATTR, np.bytes_(b"b" * 64)),
    )
    with tempfile.TemporaryDirectory(prefix="artifact-pair-binding-") as temp:
      for label, key, value in cases:
        with self.subTest(label=label):
          root = Path(temp) / label
          _save(root)
          with h5py.File(_record_path(root), "r+") as record:
            if value is None:
              del record.attrs[key]
            else:
              record.attrs[key] = value

          with mock.patch.object(
              results.torch, "load",
              side_effect=AssertionError("torch.load must not be reached")) \
              as load_checkpoint, mock.patch.object(
                ResMLP, "__init__",
                side_effect=AssertionError("model construction must not run")) \
              as construct_model:
            with self.assertRaisesRegex(
                (KeyError, ValueError, RuntimeError),
                "artifact|binding|checkpoint|digest|sha256|hex|re-save"):
              _rebuild(root)

          load_checkpoint.assert_not_called()
          construct_model.assert_not_called()

  def test_hdf5_stage_failure_keeps_previous_pair(self):
    """A failed second save never writes either final destination directly."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-hdf5-") as temp:
      root = Path(temp) / "saved"
      _save(root, "default")
      before = _pair_bytes(root)

      with mock.patch.object(
          h5py, "File",
          side_effect=OSError("injected staged HDF5 failure")):
        with self.assertRaisesRegex(OSError, "staged HDF5 failure"):
          _save(root, "reduce-overhead")

      self.assertEqual(_pair_bytes(root), before)
      _rebuild(root)
      self.assertFalse(_pending_path(root).exists())

  def test_second_stage_allocation_failure_removes_first_stage(self):
    """Failure to reserve the HDF5 stage does not leak the weight stage."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-allocate-") as temp:
      root = Path(temp) / "saved"
      _save(root, "default")
      before = _pair_bytes(root)
      before_names = set(os.listdir(temp))
      real_new_staging_path = results._new_staging_path
      calls = 0

      def fail_second_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
          raise OSError("injected HDF5 stage-allocation failure")
        return real_new_staging_path(*args, **kwargs)

      with mock.patch.object(
          results, "_new_staging_path", fail_second_allocation):
        with self.assertRaisesRegex(OSError, "stage-allocation failure"):
          _save(root, "reduce-overhead")

      self.assertEqual(calls, 2)
      self.assertEqual(set(os.listdir(temp)), before_names)
      self.assertEqual(_pair_bytes(root), before)
      _rebuild(root)

  def test_post_commit_cleanup_failure_does_not_report_save_failure(self):
    """A hidden rollback-link cleanup error cannot undo a committed pair."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-cleanup-") as temp:
      root = Path(temp) / "saved"
      _save(root, "default")
      before = _pair_bytes(root)
      real_unlink = os.unlink
      refused_cleanup = False

      def fail_rollback_cleanup(path, *args, **kwargs):
        nonlocal refused_cleanup
        if os.fspath(path).endswith(".rollback"):
          refused_cleanup = True
          raise OSError("injected rollback-link cleanup failure")
        return real_unlink(path, *args, **kwargs)

      with mock.patch.object(results.os, "unlink", fail_rollback_cleanup):
        _save(root, "reduce-overhead")

      self.assertTrue(refused_cleanup)
      self.assertNotEqual(_pair_bytes(root), before)
      self.assertFalse(_pending_path(root).exists())
      _rebuild(root)

  def test_marker_sync_failure_keeps_the_committed_new_pair(self):
    """A failed post-commit sync does not roll back over a later writer."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-marker-sync-") as temp:
      root = Path(temp) / "saved"
      _save(root, "default")
      before = _pair_bytes(root)
      real_fsync_directory = results._fsync_directory
      injected = False
      later_marker_text = "marker owned by a later writer\n"

      def fail_after_marker_removal(directory):
        nonlocal injected
        if not injected and not _pending_path(root).exists():
          injected = True
          _pending_path(root).write_text(
            later_marker_text, encoding="utf-8")
          raise OSError("injected marker-removal sync failure")
        return real_fsync_directory(directory)

      with mock.patch.object(
          results, "_fsync_directory", fail_after_marker_removal):
        _save(root, "reduce-overhead")

      self.assertTrue(injected)
      self.assertNotEqual(_pair_bytes(root), before)
      self.assertEqual(
        _pending_path(root).read_text(encoding="utf-8"), later_marker_text)
      _pending_path(root).unlink()
      _rebuild(root)

  def test_empty_root_interleaving_restores_the_first_committed_pair(self):
    """A later writer snapshots the pair only after acquiring the marker."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-interleave-") as temp:
      root = Path(temp) / "saved"
      marker = os.fspath(_pending_path(root))
      final_record = os.fspath(_record_path(root))
      real_open = os.open
      real_replace = os.replace
      first_writer_committed = False
      second_rename_failed = False
      interleaved = False
      first_pair = None

      def commit_first_writer_before_marker(path, flags, *args, **kwargs):
        nonlocal interleaved, first_writer_committed, first_pair
        if os.fspath(path) == marker and not interleaved:
          interleaved = True
          _save(root, "reduce-overhead")
          first_pair = _pair_bytes(root)
          first_writer_committed = True
        return real_open(path, flags, *args, **kwargs)

      def fail_later_writer_record(source, destination, *args, **kwargs):
        nonlocal second_rename_failed
        if (first_writer_committed
            and os.fspath(destination) == final_record
            and not second_rename_failed):
          second_rename_failed = True
          raise OSError("injected later-writer HDF5 rename failure")
        return real_replace(source, destination, *args, **kwargs)

      with mock.patch.object(results.os, "open", commit_first_writer_before_marker), \
          mock.patch.object(results.os, "replace", fail_later_writer_record):
        with self.assertRaisesRegex(OSError, "later-writer HDF5 rename"):
          _save(root, "default")

      self.assertTrue(interleaved)
      self.assertTrue(first_writer_committed)
      self.assertTrue(second_rename_failed)
      self.assertEqual(_pair_bytes(root), first_pair)
      self.assertFalse(_pending_path(root).exists())
      _rebuild(root)

  def test_second_final_rename_failure_restores_previous_pair(self):
    """An ordinary HDF5 commit-rename error rolls both final files back."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-rename-") as temp:
      root = Path(temp) / "saved"
      _save(root, "default")
      before = _pair_bytes(root)
      final_record = os.fspath(_record_path(root))
      real_replace = os.replace
      injected = False

      def fail_record_commit(source, destination, *args, **kwargs):
        nonlocal injected
        if os.fspath(destination) == final_record and not injected:
          injected = True
          raise OSError("injected final HDF5 rename failure")
        return real_replace(source, destination, *args, **kwargs)

      with mock.patch.object(results.os, "replace", fail_record_commit):
        with self.assertRaisesRegex(OSError, "final HDF5 rename failure"):
          _save(root, "reduce-overhead")

      self.assertTrue(injected)
      self.assertEqual(_pair_bytes(root), before)
      _rebuild(root)
      self.assertFalse(_pending_path(root).exists())

  def test_keyboard_interrupt_between_renames_leaves_refusal_marker(self):
    """An abrupt stop is visible and the mixed root cannot be rebuilt."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-interrupt-") as temp:
      root = Path(temp) / "saved"
      _save(root, "default")
      final_record = os.fspath(_record_path(root))
      real_replace = os.replace
      interrupted = False

      def interrupt_record_commit(source, destination, *args, **kwargs):
        nonlocal interrupted
        if os.fspath(destination) == final_record and not interrupted:
          interrupted = True
          raise KeyboardInterrupt("injected stop between final renames")
        return real_replace(source, destination, *args, **kwargs)

      with mock.patch.object(results.os, "replace", interrupt_record_commit):
        with self.assertRaisesRegex(
            KeyboardInterrupt, "stop between final renames"):
          _save(root, "reduce-overhead")

      self.assertTrue(interrupted)
      self.assertTrue(_pending_path(root).is_file())
      with mock.patch.object(
          results.torch, "load",
          side_effect=AssertionError("torch.load must not be reached")) \
          as load_checkpoint, mock.patch.object(
            ResMLP, "__init__",
            side_effect=AssertionError("model construction must not run")) \
          as construct_model:
        with self.assertRaisesRegex(
            (KeyError, ValueError, RuntimeError),
            "pending|interrupted|incomplete|pair|publication"):
          _rebuild(root)

      load_checkpoint.assert_not_called()
      construct_model.assert_not_called()


if __name__ == "__main__":
  unittest.main()
