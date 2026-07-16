"""Check the exact CMB multipole axis saved beside generator checkpoints.

The production CMB module imports the full scientific runtime. These tests
compile only its checkpoint methods from the syntax tree, then exercise those
methods with small NumPy files on the CPU.
"""

import ast
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.lib.format import open_memmap

from compute_data_vectors.dataset_manifest import CheckpointLoadError
from compute_data_vectors.dataset_manifest import require_checkpoint_members


ROOT = Path(__file__).resolve().parents[2]
CMB_GENERATOR = (
  ROOT / "compute_data_vectors" / "dataset_generator_cmb.py")
SPECTRA = ("tt", "te", "ee", "pp")


class _NumpyBoundary:
  """Record checkpoint reads while delegating small array work to NumPy."""

  int64 = np.int64

  def __init__(self):
    self.loads = []

  def load(self, path, **options):
    self.loads.append(Path(path).name)
    return np.load(path, **options)

  @staticmethod
  def arange(start, stop, dtype):
    return np.arange(start, stop, dtype=dtype)

  @staticmethod
  def array_equal(left, right):
    return np.array_equal(left, right)

  @staticmethod
  def dtype(value):
    return np.dtype(value)

  @staticmethod
  def save(path, value):
    np.save(path, value)

  @staticmethod
  def zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)


def _compile_cmb_checkpoint_class(numpy_boundary):
  """Compile only the real CMB checkpoint methods needed by these tests."""
  source = CMB_GENERATOR.read_text(encoding="utf-8")
  tree = ast.parse(source, filename=str(CMB_GENERATOR))
  dataset_classes = []
  for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "dataset":
      dataset_classes.append(node)
  if len(dataset_classes) != 1:
    raise AssertionError("expected exactly one CMB dataset class")

  wanted = {
    "_dv_alloc",
    "_dv_chk_files",
    "_dv_load_chk",
    "_load_multipole_axis",
    "_multipole_axis",
  }
  methods = []
  for node in dataset_classes[0].body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
      methods.append(copy.deepcopy(node))
  if len(methods) != len(wanted):
    raise AssertionError("CMB checkpoint method census is incomplete")

  family = ast.ClassDef(
    name="CmbCheckpoint",
    bases=[],
    keywords=[],
    body=methods,
    decorator_list=[])
  module = ast.Module(body=[family], type_ignores=[])
  ast.fix_missing_locations(module)

  def available_memory():
    """Return a budget that keeps every small test array in memory."""
    return SimpleNamespace(available=10**12)

  namespace = {
    "SPECTRA": SPECTRA,
    "np": numpy_boundary,
    "open_memmap": open_memmap,
    "psutil": SimpleNamespace(virtual_memory=available_memory),
  }
  exec(compile(module, str(CMB_GENERATOR), "exec"), namespace)
  return namespace["CmbCheckpoint"]


def _new_instance(checkpoint_class, directory):
  """Build one small CMB checkpoint object without scientific setup."""
  instance = checkpoint_class()
  instance.dvsf = str(Path(directory) / "dv")
  instance.lrange = np.array([2, 5], dtype=np.int64)
  instance.samples = np.zeros((2, 2), dtype=np.float32)
  instance.failed = np.zeros((2,), dtype=bool)
  instance.dtype = np.float32
  return instance


def _write_spectrum_files(directory):
  """Write the four valid payload members used by load-refusal tests."""
  directory = Path(directory)
  for spectrum in SPECTRA:
    path = directory / ("dv_" + spectrum + ".npy")
    np.save(path, np.zeros((2, 4), dtype=np.float32))


def _snapshot(directory):
  """Return exact bytes and modification times for every checkpoint file."""
  result = {}
  paths = sorted(Path(directory).glob("dv_*.npy"))
  for path in paths:
    result[path.name] = (path.read_bytes(), path.stat().st_mtime_ns)
  return result


class CmbCheckpointAxisTests(unittest.TestCase):
  """Require a saved int64 axis before a CMB checkpoint can load."""

  def test_fresh_allocation_saves_the_exact_int64_axis(self):
    numpy_boundary = _NumpyBoundary()
    checkpoint_class = _compile_cmb_checkpoint_class(numpy_boundary)
    with tempfile.TemporaryDirectory() as directory:
      instance = _new_instance(checkpoint_class, directory)
      first_payload = np.zeros((4, 4), dtype=np.float32)

      instance._dv_alloc(nrows=2, first_dvs=first_payload)

      axis_path = Path(directory) / "dv_ell.npy"
      observed = np.load(axis_path, allow_pickle=False)
      expected = np.arange(2, 6, dtype=np.int64)
      self.assertEqual(observed.dtype, np.dtype(np.int64))
      np.testing.assert_array_equal(observed, expected)
      self.assertEqual(tuple(instance.datavectors), SPECTRA)
      for spectrum in SPECTRA:
        self.assertEqual(instance.datavectors[spectrum].shape, (2, 4))

  def test_missing_axis_is_a_requested_load_refusal(self):
    numpy_boundary = _NumpyBoundary()
    checkpoint_class = _compile_cmb_checkpoint_class(numpy_boundary)
    with tempfile.TemporaryDirectory() as directory:
      instance = _new_instance(checkpoint_class, directory)
      _write_spectrum_files(directory)
      members = instance._dv_chk_files()
      before = _snapshot(directory)

      with self.assertRaises(CheckpointLoadError) as refusal:
        require_checkpoint_members(
          operation="resume",
          members=members,
          is_file=os.path.isfile)

      self.assertIn("dv_ell.npy", str(refusal.exception))
      self.assertIn("No existing dataset file was changed",
                    str(refusal.exception))
      self.assertEqual(numpy_boundary.loads, [])
      self.assertEqual(_snapshot(directory), before)

  def test_corrupt_axes_refuse_before_any_spectrum_is_loaded(self):
    cases = (
      ("wrong dtype", np.arange(2, 6, dtype=np.int32), "dtype"),
      ("wrong dimension", np.array([[2, 3, 4, 5]], dtype=np.int64),
       "must be 1D"),
      ("wrong width", np.array([2, 3, 4], dtype=np.int64), "shape"),
      ("shifted", np.array([3, 4, 5, 6], dtype=np.int64),
       "every integer"),
      ("reversed", np.array([5, 4, 3, 2], dtype=np.int64),
       "every integer"),
      ("gapped", np.array([2, 3, 5, 6], dtype=np.int64),
       "every integer"),
      ("same width but wrong", np.array([2, 4, 3, 5], dtype=np.int64),
       "every integer"),
    )
    for label, axis, message in cases:
      with self.subTest(case=label):
        numpy_boundary = _NumpyBoundary()
        checkpoint_class = _compile_cmb_checkpoint_class(numpy_boundary)
        with tempfile.TemporaryDirectory() as directory:
          instance = _new_instance(checkpoint_class, directory)
          _write_spectrum_files(directory)
          np.save(Path(directory) / "dv_ell.npy", axis)
          before = _snapshot(directory)

          with self.assertRaisesRegex(ValueError, message):
            instance._dv_load_chk()

          self.assertEqual(numpy_boundary.loads, ["dv_ell.npy"])
          self.assertFalse(hasattr(instance, "datavectors"))
          self.assertEqual(_snapshot(directory), before)

  def test_valid_axis_loads_every_spectrum_without_changing_files(self):
    numpy_boundary = _NumpyBoundary()
    checkpoint_class = _compile_cmb_checkpoint_class(numpy_boundary)
    with tempfile.TemporaryDirectory() as directory:
      instance = _new_instance(checkpoint_class, directory)
      _write_spectrum_files(directory)
      np.save(
        Path(directory) / "dv_ell.npy",
        np.arange(2, 6, dtype=np.int64))
      before = _snapshot(directory)

      instance._dv_load_chk()

      self.assertEqual(numpy_boundary.loads[0], "dv_ell.npy")
      self.assertEqual(tuple(instance.datavectors), SPECTRA)
      self.assertEqual(_snapshot(directory), before)


if __name__ == "__main__":
  unittest.main()
