"""CPU checks for saving and reopening one saved-emulator file pair.

A trained emulator is saved in two files.  ``<root>.emul`` contains the
learned tensors, while ``<root>.h5`` explains how those tensors must be
interpreted.  These tests check that an occupied name is never replaced,
that a failed save leaves no partial file, and that a corrupted checkpoint
cannot execute code or masquerade as a model state.

The fixture is deliberately small: it is the same two-input, one-output model
used by the compile-recipe gate.  No training data, CosmoLike installation, or
GPU is required.
"""

import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock

import h5py
import torch

from ai.gates.checks.compile_recipe import save_fixture
from emulator import results
from emulator.designs.plain import ResMLP


def _checkpoint_path(root):
  """Return the learned-tensor member of one fixture pair."""
  return Path(str(root) + ".emul")


def _record_path(root):
  """Return the scientific-record member of one fixture pair."""
  return Path(str(root) + ".h5")


def _pair_bytes(root):
  """Return immutable snapshots of both final files."""
  return (_checkpoint_path(root).read_bytes(),
          _record_path(root).read_bytes())


def _directory_snapshot(directory):
  """Record every regular-file byte and symbolic-link target in a folder."""
  snapshot = {}
  for path in Path(directory).iterdir():
    if path.is_symlink():
      snapshot[path.name] = ("symlink", os.readlink(path))
    elif path.is_file():
      snapshot[path.name] = ("file", path.read_bytes())
    else:
      snapshot[path.name] = ("other",)
  return snapshot


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


class _UnsafePickleValue:
  """A checkpoint value that would create a file under unrestricted pickle."""

  def __init__(self, marker):
    self.marker = os.fspath(marker)

  def __reduce__(self):
    """Describe the side effect without running it during ``torch.save``."""
    return os.system, ("touch " + shlex.quote(self.marker),)


class ArtifactPairTests(unittest.TestCase):
  """Exercise a valid save, occupied-name refusals, and checkpoint safety."""

  def assert_save_refuses_before_staging(self, root):
    """Require an occupied name to survive unchanged without a temporary file."""
    before = _directory_snapshot(Path(root).parent)
    with mock.patch.object(
        results.torch, "save",
        side_effect=AssertionError("staging must not begin")) as first_write:
      with self.assertRaisesRegex(
          FileExistsError, "already occupied|never replaces"):
        _save(root, "reduce-overhead")
    first_write.assert_not_called()
    self.assertEqual(_directory_snapshot(Path(root).parent), before)

  def test_valid_save_rebuilds_and_leaves_no_temporaries(self):
    """A normal save publishes both members and loads again, nothing else."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-valid-") as temp:
      root = Path(temp) / "saved"
      _save(root)

      members = sorted(path.name for path in Path(temp).iterdir())
      self.assertEqual(members, ["saved.emul", "saved.h5"])

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

  def test_complete_root_refuses_before_staging_and_keeps_exact_bytes(self):
    """A public save cannot replace an earlier complete emulator pair."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-complete-") as temp:
      root = Path(temp) / "saved"
      _save(root, "default")
      before = _pair_bytes(root)

      self.assert_save_refuses_before_staging(root)

      self.assertEqual(_pair_bytes(root), before)
      _rebuild(root)

  def test_partial_roots_refuse_before_staging_and_keep_exact_bytes(self):
    """Either lone pair member reserves the name instead of being replaced."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-partial-") as temp:
      for suffix, payload in ((".emul", b"lone checkpoint\x00\xff"),
                              (".h5", b"lone record\x00\xfe")):
        with self.subTest(suffix=suffix):
          root = Path(temp) / ("saved-" + suffix[1:])
          Path(str(root) + suffix).write_bytes(payload)

          self.assert_save_refuses_before_staging(root)

          self.assertEqual(Path(str(root) + suffix).read_bytes(), payload)

  def test_symlink_roots_refuse_before_staging_and_keep_link_and_target(self):
    """A symbolic link reserves either final name, including its target bytes."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-symlink-") as temp:
      for suffix in (".emul", ".h5"):
        with self.subTest(suffix=suffix):
          root = Path(temp) / ("saved-" + suffix[1:])
          target = Path(temp) / ("target-" + suffix[1:])
          payload = ("target bytes for " + suffix).encode("ascii")
          target.write_bytes(payload)
          link = Path(str(root) + suffix)
          os.symlink(target.name, link)

          self.assert_save_refuses_before_staging(root)

          self.assertTrue(link.is_symlink())
          self.assertEqual(os.readlink(link), target.name)
          self.assertEqual(target.read_bytes(), payload)

  def test_fresh_root_hdf5_failure_removes_every_temporary(self):
    """A failed HDF5 write on a new name leaves no public or temporary file."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-hdf5-") as temp:
      root = Path(temp) / "saved"
      before = _directory_snapshot(temp)

      with mock.patch.object(
          h5py, "File",
          side_effect=OSError("injected HDF5 failure")):
        with self.assertRaisesRegex(OSError, "HDF5 failure"):
          _save(root, "default")

      self.assertEqual(_directory_snapshot(temp), before)

  def test_unsafe_pickle_global_is_blocked_without_running_it(self):
    """A rewritten checkpoint cannot execute an unsafe pickle value."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-pickle-") as temp:
      root = Path(temp) / "saved"
      side_effect = Path(temp) / "pickle-side-effect"
      _save(root)

      torch.save({"weight": _UnsafePickleValue(side_effect)},
                 _checkpoint_path(root))

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
    """A well-formed but non-tensor mapping is not a model state dict."""
    with tempfile.TemporaryDirectory(prefix="artifact-pair-nontensor-") as temp:
      root = Path(temp) / "saved"
      _save(root)

      torch.save({"weight": "this is not a tensor"}, _checkpoint_path(root))

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


if __name__ == "__main__":
  unittest.main()
