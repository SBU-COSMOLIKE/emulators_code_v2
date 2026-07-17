"""Focused CPU tests for the data-vector success-row contract.

The generator module imports MPI and several scientific packages that are not
needed for these checks.  The tests compile the relevant production methods
from their syntax trees, then exercise them with small NumPy arrays.
"""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from numpy.lib.format import open_memmap


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "compute_data_vectors" / "generator_core.py"
FAMILY_PATHS = {
  "cmb": ROOT / "compute_data_vectors" / "dataset_generator_cmb.py",
  "background": (
    ROOT / "compute_data_vectors" / "dataset_generator_background.py"),
  "mps": ROOT / "compute_data_vectors" / "dataset_generator_mps.py",
}
CORE_METHODS = {
  "_dv_payload_names",
  "_dv_payload_mapping",
  "_dv_expected_payload_shape",
  "_dv_payload_store",
  "_prepare_payload_mapping",
  "_stored_payload_mapping",
  "_accept_payload_row",
  "_validate_loaded_success_rows",
  "_dv_alloc",
  "_dv_write",
}
FAMILY_METHODS = {
  "cmb": {
    "_dv_payload_names",
    "_dv_payload_mapping",
    "_dv_expected_payload_shape",
    "_dv_payload_store",
    "_multipole_axis",
    "_dv_write",
  },
  "background": {
    "_grid_of",
    "_dv_payload_names",
    "_dv_payload_mapping",
    "_dv_expected_payload_shape",
    "_dv_payload_store",
    "_dv_write",
  },
  "mps": {
    "_quantities",
    "_dv_payload_names",
    "_dv_payload_mapping",
    "_dv_expected_payload_shape",
    "_dv_payload_store",
    "_dv_write",
  },
}


def _class_node(path, class_name):
  """Return one copied class syntax node from a production file."""
  tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == class_name
  ]
  if len(classes) != 1:
    raise AssertionError(
      f"expected one {class_name} class in {path}, got {len(classes)}")
  return copy.deepcopy(classes[0])


def _selected_methods(path, class_name, names):
  """Return the requested production methods and require every name once."""
  class_node = _class_node(path, class_name)
  methods = [
    copy.deepcopy(node) for node in class_node.body
    if isinstance(node, ast.FunctionDef) and node.name in names
  ]
  observed = {method.name for method in methods}
  if observed != names or len(methods) != len(names):
    raise AssertionError(
      f"method selection in {path} returned {observed!r}, expected {names!r}")
  return methods


def _large_available_memory():
  """Keep the allocation tests on their small in-memory path."""
  return SimpleNamespace(available=10**12)


def _compile_core():
  """Compile the shared production payload methods without heavy imports."""
  methods = _selected_methods(GENERATOR, "GeneratorCore", CORE_METHODS)
  class_node = ast.ClassDef(
    name="Core",
    bases=[],
    keywords=[],
    body=methods,
    decorator_list=[])
  module = ast.Module(body=[class_node], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "np": np,
    "open_memmap": open_memmap,
    "psutil": SimpleNamespace(virtual_memory=_large_available_memory),
  }
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["Core"]


def _compile_family(name, core_class):
  """Compile one family's real payload methods on the shared test core."""
  path = FAMILY_PATHS[name]
  methods = _selected_methods(path, "dataset", FAMILY_METHODS[name])
  class_node = ast.ClassDef(
    name="Family",
    bases=[ast.Name(id="Core", ctx=ast.Load())],
    keywords=[],
    body=methods,
    decorator_list=[])
  module = ast.Module(body=[class_node], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {"Core": core_class, "np": np}
  if name == "cmb":
    namespace["SPECTRA"] = ("tt", "te", "ee", "pp")
  elif name == "background":
    namespace["QUANTITIES"] = ("h", "dm")
  exec(compile(module, str(path), "exec"), namespace)
  return namespace["Family"]


def _flat_instance(core_class, nrows=2, width=3, allocate=True):
  """Build one flat-store object at the shared payload boundary."""
  instance = core_class()
  instance.failed = np.ones(nrows, dtype=bool)
  instance.dtype = np.float32
  instance.samples = np.zeros((nrows, 1), dtype=np.float32)
  instance.dvs_is_memmap = False
  instance.dvsf = "unused"
  if allocate:
    instance.datavectors = np.zeros((nrows, width), dtype=np.float32)
  return instance


def _snapshot_array(array):
  """Copy an array so refusal tests can prove that it did not change."""
  return np.array(array, copy=True)


def _function_node(path, class_name, function_name):
  """Return one production method syntax node for a source-coverage check."""
  class_node = _class_node(path, class_name)
  functions = [
    node for node in class_node.body
    if isinstance(node, ast.FunctionDef) and node.name == function_name
  ]
  if len(functions) != 1:
    raise AssertionError(
      f"expected one {function_name} method in {path}, got {len(functions)}")
  return functions[0]


def _attribute_calls(function_node, attribute_name):
  """Return calls to one ``self`` method in source-line order."""
  calls = []
  for node in ast.walk(function_node):
    if not isinstance(node, ast.Call):
      continue
    function = node.func
    if (isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "self"
        and function.attr == attribute_name):
      calls.append(node)

  def source_position(node):
    return node.lineno, node.col_offset

  return sorted(calls, key=source_position)


class GeneratorPayloadSuccessTests(unittest.TestCase):
  """Require exact storage before any row is recorded as successful."""

  @classmethod
  def setUpClass(cls):
    cls.core_class = _compile_core()
    cls.family_classes = {
      name: _compile_family(name, cls.core_class)
      for name in FAMILY_PATHS
    }

  def test_valid_flat_row_casts_once_and_clears_failure_after_readback(self):
    instance = _flat_instance(self.core_class)
    payload = np.array([1.25, -2.5, 3.75], dtype=np.float64)
    writes = []
    production_write = instance._dv_write

    def count_write(i, dvs):
      writes.append((i, {name: value.copy() for name, value in dvs.items()}))
      production_write(i=i, dvs=dvs)

    instance._dv_write = count_write
    instance._accept_payload_row(index=1, payload=payload, write_row=True)

    self.assertEqual(len(writes), 1)
    self.assertEqual(writes[0][0], 1)
    self.assertEqual(writes[0][1]["vector"].dtype, np.dtype(np.float32))
    np.testing.assert_array_equal(
      instance.datavectors[1], payload.astype(np.float32))
    self.assertFalse(instance.failed[1])

  def test_nonfinite_wrong_shape_and_post_cast_overflow_refuse_before_write(self):
    bad_payloads = (
      np.array([1.0, np.nan, 3.0]),
      np.array([1.0, np.inf, 3.0]),
      np.array([1.0, 2.0]),
      np.array([1.0e300, 2.0, 3.0], dtype=np.float64),
    )
    for payload in bad_payloads:
      with self.subTest(payload=repr(payload)):
        instance = _flat_instance(self.core_class)
        before = _snapshot_array(instance.datavectors)
        writes = []

        def reject_write(i, dvs):
          writes.append((i, dvs))
          raise AssertionError("invalid payload reached the storage method")

        instance._dv_write = reject_write
        with self.assertRaises(ValueError):
          instance._accept_payload_row(
            index=0,
            payload=payload,
            write_row=True)
        self.assertEqual(writes, [])
        self.assertTrue(instance.failed[0])
        np.testing.assert_array_equal(instance.datavectors, before)

  def test_readback_mismatch_keeps_row_failed_after_one_write(self):
    instance = _flat_instance(self.core_class)
    writes = []

    def corrupt_write(i, dvs):
      writes.append(i)
      instance.datavectors[i] = dvs["vector"]
      instance.datavectors[i, 0] += np.float32(1.0)

    instance._dv_write = corrupt_write
    payload = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    with self.assertRaisesRegex(ValueError, "differs from the exact"):
      instance._accept_payload_row(index=0, payload=payload, write_row=True)

    self.assertEqual(writes, [0])
    self.assertTrue(instance.failed[0])

  def test_signed_zero_readback_corruption_keeps_row_failed(self):
    instance = _flat_instance(self.core_class)
    writes = []

    def change_zero_sign(i, dvs):
      writes.append(i)
      instance.datavectors[i] = dvs["vector"]
      instance.datavectors[i, 0] = np.float32(-0.0)

    instance._dv_write = change_zero_sign
    payload = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "payload bytes"):
      instance._accept_payload_row(index=0, payload=payload, write_row=True)

    self.assertEqual(writes, [0])
    self.assertTrue(np.signbit(instance.datavectors[0, 0]))
    self.assertTrue(instance.failed[0])

  def test_first_row_is_validated_before_allocation_then_written_once(self):
    instance = _flat_instance(self.core_class, allocate=False)
    writes = []
    production_write = instance._dv_write

    def count_write(i, dvs):
      writes.append(i)
      production_write(i=i, dvs=dvs)

    instance._dv_write = count_write
    payload = np.array([7.0, 8.0, 9.0], dtype=np.float64)
    instance._accept_payload_row(
      index=0,
      payload=payload,
      write_row=True,
      allocation_rows=2)

    self.assertEqual(writes, [0])
    self.assertEqual(instance.datavectors.shape, (2, 3))
    self.assertEqual(instance.datavectors.dtype, np.dtype(np.float32))
    np.testing.assert_array_equal(
      instance.datavectors[0], payload.astype(np.float32))
    self.assertFalse(instance.failed[0])

    invalid = _flat_instance(self.core_class, allocate=False)
    overflow = np.array([1.0e300, 2.0, 3.0], dtype=np.float64)
    with self.assertRaises(ValueError):
      invalid._accept_payload_row(
        index=0,
        payload=overflow,
        write_row=True,
        allocation_rows=2)
    self.assertFalse(hasattr(invalid, "datavectors"))
    self.assertTrue(invalid.failed[0])

  def test_cmb_requires_four_exact_spectrum_rows(self):
    family_class = self.family_classes["cmb"]
    instance = family_class()
    instance.lrange = (2, 4)
    instance.dtype = np.float32
    instance.failed = np.ones(1, dtype=bool)
    instance.datavectors = {
      name: np.zeros((1, 3), dtype=np.float32)
      for name in ("tt", "te", "ee", "pp")
    }
    payload = np.arange(12, dtype=np.float64).reshape(4, 3)
    instance._accept_payload_row(index=0, payload=payload, write_row=True)
    for spectrum_index, spectrum in enumerate(("tt", "te", "ee", "pp")):
      np.testing.assert_array_equal(
        instance.datavectors[spectrum][0],
        payload[spectrum_index].astype(np.float32))
    self.assertFalse(instance.failed[0])

    invalid = family_class()
    invalid.lrange = (2, 4)
    invalid.dtype = np.float32
    invalid.failed = np.ones(1, dtype=bool)
    invalid.datavectors = {
      name: np.zeros((1, 3), dtype=np.float32)
      for name in ("tt", "te", "ee", "pp")
    }
    before = {
      name: _snapshot_array(array)
      for name, array in invalid.datavectors.items()
    }
    with self.assertRaises(ValueError):
      invalid._accept_payload_row(
        index=0,
        payload=np.zeros((4, 2), dtype=np.float64),
        write_row=True)
    for name in before:
      np.testing.assert_array_equal(invalid.datavectors[name], before[name])
    self.assertTrue(invalid.failed[0])

  def test_background_requires_exact_keys_and_grid_shapes(self):
    family_class = self.family_classes["background"]

    def make_instance():
      instance = family_class()
      instance.z_sn = np.array([0.1, 0.2])
      instance.z_rec = np.array([1000.0, 1100.0, 1200.0])
      instance.dtype = np.float32
      instance.failed = np.ones(1, dtype=bool)
      instance.datavectors = {
        "h": np.zeros((1, 2), dtype=np.float32),
        "dm": np.zeros((1, 3), dtype=np.float32),
      }
      return instance

    valid = make_instance()
    payload = {
      "h": np.array([70.0, 71.0], dtype=np.float64),
      "dm": np.array([1.0, 2.0, 3.0], dtype=np.float64),
    }
    valid._accept_payload_row(index=0, payload=payload, write_row=True)
    self.assertFalse(valid.failed[0])

    bad_payloads = (
      {"h": payload["h"]},
      {**payload, "extra": np.array([1.0])},
      {"h": np.array([70.0]), "dm": payload["dm"]},
    )
    for bad_payload in bad_payloads:
      with self.subTest(payload_keys=tuple(bad_payload)):
        instance = make_instance()
        before = {
          name: _snapshot_array(array)
          for name, array in instance.datavectors.items()
        }
        with self.assertRaises(ValueError):
          instance._accept_payload_row(
            index=0,
            payload=bad_payload,
            write_row=True)
        self.assertTrue(instance.failed[0])
        for name in before:
          np.testing.assert_array_equal(
            instance.datavectors[name], before[name])

  def test_matter_power_keys_follow_the_base_switch(self):
    family_class = self.family_classes["mps"]
    for write_base in (False, True):
      with self.subTest(write_base=write_base):
        instance = family_class()
        instance.write_base = write_base
        instance.z_mps = np.array([0.0, 1.0])
        instance.k_mps = np.array([0.1, 1.0, 10.0])
        instance.dtype = np.float32
        instance.failed = np.ones(1, dtype=bool)
        quantities = instance._quantities()
        instance.datavectors = {
          name: np.zeros((1, 6), dtype=np.float32)
          for name in quantities
        }
        payload = {
          name: np.arange(6, dtype=np.float64)
          for name in quantities
        }
        instance._accept_payload_row(
          index=0,
          payload=payload,
          write_row=True)
        self.assertFalse(instance.failed[0])

        invalid = family_class()
        invalid.write_base = write_base
        invalid.z_mps = instance.z_mps
        invalid.k_mps = instance.k_mps
        invalid.dtype = np.float32
        invalid.failed = np.ones(1, dtype=bool)
        invalid.datavectors = {
          name: np.zeros((1, 6), dtype=np.float32)
          for name in quantities
        }
        bad_payload = dict(payload)
        if write_base:
          del bad_payload["pklin_base"]
        else:
          bad_payload["pklin_base"] = np.arange(6, dtype=np.float64)
        with self.assertRaises(ValueError):
          invalid._accept_payload_row(
            index=0,
            payload=bad_payload,
            write_row=True)
        self.assertTrue(invalid.failed[0])

        wrong_shape = dict(payload)
        wrong_shape["pklin"] = np.arange(5, dtype=np.float64)
        with self.assertRaises(ValueError):
          instance._prepare_payload_mapping(
            payload_mapping=instance._dv_payload_mapping(wrong_shape),
            use_storage=True)

  def test_invalid_saved_success_row_refuses_without_changing_file(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "payload.npy"
      stored = open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(2, 3))
      stored[:] = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
      stored.flush()
      del stored
      stamp = 1_700_000_000_000_000_000
      os.utime(path, ns=(stamp, stamp))
      before_bytes = path.read_bytes()
      before_mtime = path.stat().st_mtime_ns

      instance = _flat_instance(self.core_class)
      instance.failed = np.array([False, True], dtype=bool)
      instance.datavectors = np.load(
        path,
        mmap_mode="r+",
        allow_pickle=False)
      with self.assertRaises(ValueError):
        instance._validate_loaded_success_rows()
      del instance.datavectors

      self.assertEqual(path.read_bytes(), before_bytes)
      self.assertEqual(path.stat().st_mtime_ns, before_mtime)
      np.testing.assert_array_equal(
        instance.failed, np.array([False, True], dtype=bool))

  def test_saved_success_store_must_use_the_configured_dtype(self):
    cases = (
      ("float64", np.float64, [1.0, 2.0, 3.0]),
      ("integer", np.int32, [1, 2, 3]),
      ("complex", np.complex64, [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
    )
    for label, dtype, values in cases:
      with self.subTest(dtype=label):
        with tempfile.TemporaryDirectory() as directory:
          path = Path(directory) / "payload.npy"
          stored = open_memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=(1, 3))
          stored[0] = np.asarray(values, dtype=dtype)
          stored.flush()
          del stored
          stamp = 1_700_000_000_000_000_000
          os.utime(path, ns=(stamp, stamp))
          before_bytes = path.read_bytes()
          before_mtime = path.stat().st_mtime_ns

          instance = _flat_instance(self.core_class, nrows=1)
          instance.failed = np.array([False], dtype=bool)
          instance.datavectors = np.load(
            path,
            mmap_mode="r+",
            allow_pickle=False)
          with self.assertRaisesRegex(ValueError, "configured dtype"):
            instance._validate_loaded_success_rows()
          del instance.datavectors

          self.assertEqual(path.read_bytes(), before_bytes)
          self.assertEqual(path.stat().st_mtime_ns, before_mtime)
          np.testing.assert_array_equal(
            instance.failed, np.array([False], dtype=bool))

  def test_saved_failure_rows_are_not_reclassified(self):
    instance = _flat_instance(self.core_class)
    instance.datavectors[:] = np.array(
      [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]],
      dtype=np.float32)
    instance.failed = np.array([False, True], dtype=bool)
    before = _snapshot_array(instance.datavectors)

    instance._validate_loaded_success_rows()

    np.testing.assert_array_equal(instance.datavectors, before)
    np.testing.assert_array_equal(
      instance.failed, np.array([False, True], dtype=bool))

  def test_checkpoint_validation_runs_before_loaded_report(self):
    function = _function_node(GENERATOR, "GeneratorCore", "__load_chk")
    load_calls = _attribute_calls(function, "_dv_load_chk")
    validation_calls = _attribute_calls(
      function, "_validate_loaded_success_rows")
    loaded_reports = [
      node for node in ast.walk(function)
      if isinstance(node, ast.Constant)
      and node.value == "Loaded models from chk"
    ]
    self.assertEqual(len(load_calls), 1)
    self.assertEqual(len(validation_calls), 1)
    self.assertEqual(len(loaded_reports), 1)
    self.assertLess(load_calls[0].lineno, validation_calls[0].lineno)
    self.assertLess(validation_calls[0].lineno, loaded_reports[0].lineno)

  def test_every_success_path_uses_the_shared_predicate(self):
    function = _function_node(
      GENERATOR, "GeneratorCore", "__generate_datavectors")
    accept_calls = _attribute_calls(function, "_accept_payload_row")
    consume_calls = _attribute_calls(
      function, "_consume_worker_result_message")
    consumer = _function_node(
      GENERATOR, "GeneratorCore", "_consume_worker_result_message")
    consumer_accept_calls = _attribute_calls(consumer, "_accept_payload_row")
    write_calls = _attribute_calls(function, "_dv_write")
    direct_success_assignments = []
    for node in ast.walk(function):
      if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        continue
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      if not isinstance(node.value, ast.Constant) or node.value.value is not False:
        continue
      for target in targets:
        if (isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "self"
            and target.value.attr == "failed"):
          direct_success_assignments.append(node)

    self.assertEqual(len(accept_calls), 3)
    self.assertEqual(len(consume_calls), 2)
    self.assertEqual(len(consumer_accept_calls), 1)
    self.assertEqual(write_calls, [])
    self.assertEqual(direct_success_assignments, [])

  def test_saved_success_loop_uses_read_only_predicate(self):
    function = _function_node(
      GENERATOR, "GeneratorCore", "_validate_loaded_success_rows")
    accept_calls = _attribute_calls(function, "_accept_payload_row")
    self.assertEqual(len(accept_calls), 1)
    keywords = {keyword.arg: keyword.value
                for keyword in accept_calls[0].keywords}
    self.assertIsInstance(keywords["write_row"], ast.Constant)
    self.assertIs(keywords["write_row"].value, False)


if __name__ == "__main__":
  unittest.main()
