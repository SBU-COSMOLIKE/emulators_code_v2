"""Focused CPU tests for fail-closed requested checkpoint loads.

The production generator cannot be imported in a minimal test process because
its module imports MPI, Cobaya, GetDist, emcee, and NumPy.  These tests instead
compile the real ``GeneratorCore`` class and family census/loader methods from
their syntax trees.  They drive only checkpoint decisions and geometry checks;
all expensive scientific boundaries remain unevaluated.
"""

import ast
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
import unittest

from compute_data_vectors import dataset_manifest
from compute_data_vectors.dataset_manifest import CheckpointLoadError
from compute_data_vectors.dataset_manifest import validate_run_control
from emulator import fixed_facts


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "compute_data_vectors" / "generator_core.py"
FAMILIES = {
  "cmb": (ROOT / "compute_data_vectors" / "dataset_generator_cmb.py",
          {"SPECTRA": ("tt", "te", "ee", "pp")},
          ("dv_tt.npy", "dv_te.npy", "dv_ee.npy", "dv_pp.npy")),
  "background": (
    ROOT / "compute_data_vectors" / "dataset_generator_background.py",
    {"QUANTITIES": ("h", "dm")},
    ("dv_h.npy", "dv_h_z.npy", "dv_dm.npy", "dv_dm_z.npy")),
  "mps": (ROOT / "compute_data_vectors" / "dataset_generator_mps.py",
          {},
          ("dv_pklin.npy", "dv_boost.npy", "dv_z.npy", "dv_k.npy")),
}
MPS_BASE_CHECKPOINTS = (
  "dv_pklin.npy", "dv_boost.npy", "dv_pklin_base.npy",
  "dv_boost_base.npy", "dv_z.npy", "dv_k.npy")


class _FakeSamples:
  """Two named sample columns returned by the resolver boundary."""

  ndim = 2
  shape = (1, 2)
  nbytes = 8


class _FakeArray:
  """Small array-shaped value used by the family checkpoint loaders."""

  def __init__(self, shape, values=None):
    self.shape = tuple(shape)
    self.ndim = len(self.shape)
    self.nbytes = max(1, 4 * self._size())
    self.values = values

  def _size(self):
    size = 1
    for dimension in self.shape:
      size *= dimension
    return size

  def __len__(self):
    return self.shape[0]

  def __repr__(self):
    return "FakeArray(shape=" + repr(self.shape) + ")"


class _FamilyNumpy:
  """Read-only NumPy boundary for family geometry validation."""

  def __init__(self, files):
    self.files = dict(files)
    self.loads = []

  def load(self, path, **kwargs):
    self.loads.append((path, kwargs))
    return self.files[path]

  @staticmethod
  def asarray(value):
    if isinstance(value, _FakeArray):
      return value
    values = tuple(value)
    return _FakeArray((len(values),), values=values)

  @staticmethod
  def array_equal(left, right):
    return left.shape == right.shape and left.values == right.values


class _FakeFailed:
  """One valid failure-flag row returned by the NumPy boundary."""

  ndim = 1
  shape = (1,)
  nbytes = 1

  def astype(self, dtype):
    del dtype
    return self


class _FakeNumpy:
  """Tiny NumPy boundary for the failure-flag part of checkpoint loading."""

  uint8 = object()

  def __init__(self, error=None):
    self.error = error
    self.loadtxt_calls = 0

  def loadtxt(self, *args, **kwargs):
    del args, kwargs
    self.loadtxt_calls += 1
    if self.error is not None:
      raise self.error
    return _FakeFailed()

  @staticmethod
  def atleast_1d(value):
    return value

  @staticmethod
  def asarray(value, dtype=None):
    del dtype
    if isinstance(value, list):
      return _FakeFailed()
    return value

class _ResolverBoundary:
  """Named-table resolver spy with an optional validation failure."""

  def __init__(self, error=None, declarations=None, sidecar_path=None):
    self.error = error
    self.declarations = declarations
    self.sidecar_path = sidecar_path
    self.calls = []

  def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    if self.error is not None:
      raise self.error
    input_names = tuple(kwargs["input_names"])
    declarations = self.declarations
    if declarations is None:
      declarations = tuple(
        (name, False, 2 + index)
        for index, name in enumerate(input_names)
      ) + (("chi2", True, 2 + len(input_names)),)
    sidecar_path = self.sidecar_path
    if sidecar_path is None:
      base = os.path.splitext(os.fspath(kwargs["params_path"]))[0]
      root, chain_suffix = os.path.splitext(base)
      sidecar_path = (root if chain_suffix[1:].isdigit() else base)
      sidecar_path += ".paramnames"
    return SimpleNamespace(inputs=_FakeSamples(), declarations=declarations,
                           sidecar_path=sidecar_path)


class _CheckpointFailure(RuntimeError):
  """Sentinel raised by the requested-load boundary."""


class _FreshPathTouched(RuntimeError):
  """Sentinel proving that a failed requested load reached fresh work."""


class _FreshMarker:
  """The sampling branch compares its configured sampler mode to one."""

  def __init__(self):
    self.touched = False

  def __eq__(self, other):
    del other
    self.touched = True
    raise _FreshPathTouched("fresh generation was reached")


def _generator_class_node():
  """Return an isolated copy of the one production GeneratorCore class."""
  tree = ast.parse(GENERATOR.read_text(encoding="utf-8"),
                   filename=str(GENERATOR))
  classes = [node for node in tree.body
             if isinstance(node, ast.ClassDef)
             and node.name == "GeneratorCore"]
  if len(classes) != 1:
    raise AssertionError("expected exactly one GeneratorCore class")
  return copy.deepcopy(classes[0])


def _compile_generator(np_boundary=None, resolver=None):
  """Compile the production class without importing its heavy module."""
  node = _generator_class_node()
  module = ast.Module(body=[node], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "np": _FakeNumpy() if np_boundary is None else np_boundary,
    "os": os,
    "Path": Path,
    "sys": sys,
    "traceback": traceback,
    "fixed_facts": fixed_facts,
    "resolve_parameter_table": (
      _ResolverBoundary() if resolver is None else resolver),
  }
  # A narrow production repair may put a pure checkpoint helper beside the
  # existing RunControl in dataset_manifest.py.  Make those bindings available
  # to the extracted class without importing generator_core.py.
  namespace.update({name: value for name, value in vars(dataset_manifest).items()
                    if not name.startswith("__")})
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["GeneratorCore"]


def _checkpoint_instance(generator_class, directory, operation,
                         complete=False):
  """Build one object at the checkpoint boundary, without running setup."""
  if operation == "fresh":
    loadchk, append = 0, 0
  elif operation == "resume":
    loadchk, append = 1, 0
  elif operation == "append":
    loadchk, append = 1, 1
  else:
    raise AssertionError("unknown operation " + repr(operation))

  root = Path(directory)
  paramsf = root / "params"
  failf = root / "fail"
  dv = root / "dv.npy"
  members = {
    "data vector": dv,
    "failure flags": Path(str(failf) + ".txt"),
    "covariance": Path(str(paramsf) + ".covmat"),
    "parameter names": Path(str(paramsf) + ".paramnames"),
    "ranges": Path(str(paramsf) + ".ranges"),
    "chain": Path(str(paramsf) + ".1.txt"),
    "scientific facts": Path(str(paramsf) + ".facts.yaml"),
  }
  if complete:
    for index, (name, path) in enumerate(members.items()):
      payload = "0\n" if name == "failure flags" else name + " sentinel\n"
      path.write_bytes(payload.encode("utf-8"))
      stamp = 1_700_000_000_000_000_000 + index
      os.utime(path, ns=(stamp, stamp))

  instance = object.__new__(generator_class)
  instance.loadchk = loadchk
  instance.append = append
  instance.run_control = validate_run_control(loadchk, append, 0)
  instance.paramsf = str(paramsf)
  instance.failf = str(failf)
  instance.dvsf = str(root / "dv")
  instance.dtype = object()
  instance.sampled_params = ("p0", "p1")
  instance.loadedsamples = False
  instance.loadedfromchk = False
  instance._dv_chk_files = lambda: [str(dv)]
  instance._dv_load_chk = lambda: None
  return instance, members


def _snapshot(members):
  """Return exact bytes and modification times for existing members."""
  return {name: (path.read_bytes(), path.stat().st_mtime_ns)
          for name, path in members.items() if path.exists()}


def _family_checkpoint_files(name, write_base=False):
  """Execute one production family's real checkpoint-census method."""
  path, globals_, _ = FAMILIES[name]
  tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  classes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name == "dataset"]
  if len(classes) != 1:
    raise AssertionError("expected one dataset class in " + str(path))
  method_names = {"_dv_chk_files"}
  if name == "mps":
    method_names.add("_quantities")
  methods = [copy.deepcopy(node) for node in classes[0].body
             if isinstance(node, ast.FunctionDef)
             and node.name in method_names]
  if len(methods) != len(method_names):
    raise AssertionError("expected one _dv_chk_files in " + str(path))
  module = ast.Module(body=[ast.ClassDef(
    name="Family", bases=[], keywords=[], body=methods,
    decorator_list=[])], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = dict(globals_)
  exec(compile(module, str(path), "exec"), namespace)
  instance = namespace["Family"]()
  instance.dvsf = "dv"
  if name == "mps":
    instance.write_base = write_base
  return tuple(instance._dv_chk_files())


def _compile_family_loader(name, numpy_boundary):
  """Compile one production family loader plus the shared axis validator."""
  path, globals_, _ = FAMILIES[name]
  tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  classes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name == "dataset"]
  if len(classes) != 1:
    raise AssertionError("expected one dataset class in " + str(path))
  wanted = {"_dv_load_chk"}
  if name == "background":
    wanted.add("_grid_of")
  if name == "mps":
    wanted.add("_quantities")
  methods = [copy.deepcopy(node) for node in classes[0].body
             if isinstance(node, ast.FunctionDef) and node.name in wanted]
  if len(methods) != len(wanted):
    raise AssertionError("missing family loader method in " + str(path))
  if name in ("background", "mps"):
    axis_methods = [copy.deepcopy(node) for node in
                    _generator_class_node().body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_load_axis_checkpoint"]
    if len(axis_methods) != 1:
      raise AssertionError("expected one shared axis validator")
    methods.extend(axis_methods)
  module = ast.Module(body=[ast.ClassDef(
    name="Family", bases=[], keywords=[], body=methods,
    decorator_list=[])], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = dict(globals_)
  namespace.update({
    "np": numpy_boundary,
    "psutil": SimpleNamespace(virtual_memory=lambda: SimpleNamespace(
      available=10**12)),
  })
  exec(compile(module, str(path), "exec"), namespace)
  return namespace["Family"]


def _family_geometry_fixture(name, corrupt=False):
  """Return a family loader and one valid or deliberately corrupt store."""
  row = _FakeArray((1, 2))
  if name == "background":
    z_sn = _FakeArray((2,), values=(0.1, 0.2))
    z_rec = _FakeArray((2,), values=(1000.0, 1100.0))
    files = {
      "dv_h_z.npy": (_FakeArray((2,), values=(0.1, 9.9))
                     if corrupt else z_sn),
      "dv_dm_z.npy": z_rec,
      "dv_h.npy": row,
      "dv_dm.npy": row,
    }
  elif name == "mps":
    z_mps = _FakeArray((2,), values=(0.0, 1.0))
    k_mps = _FakeArray((3,), values=(0.1, 1.0, 10.0))
    files = {
      "dv_z.npy": z_mps,
      "dv_k.npy": (_FakeArray((2,), values=(0.1, 1.0))
                    if corrupt else k_mps),
      "dv_pklin.npy": _FakeArray((1, 6)),
      "dv_boost.npy": _FakeArray((1, 6)),
    }
  elif name == "cmb":
    files = {"dv_" + spectrum + ".npy": _FakeArray((1, 3))
             for spectrum in ("tt", "te", "ee", "pp")}
    if corrupt:
      files["dv_tt.npy"] = _FakeArray((1, 2))
  else:
    raise AssertionError("unknown family " + name)

  numpy_boundary = _FamilyNumpy(files)
  family_class = _compile_family_loader(name, numpy_boundary)
  instance = family_class()
  instance.dvsf = "dv"
  instance.samples = _FakeArray((1, 2))
  instance.failed = _FakeArray((1,))
  if name == "background":
    instance.z_sn = z_sn
    instance.z_rec = z_rec
  elif name == "mps":
    instance.z_mps = z_mps
    instance.k_mps = k_mps
    instance.write_base = False
  else:
    instance.lrange = (2, 4)
  return instance, numpy_boundary


class GeneratorCheckpointRefusalTests(unittest.TestCase):
  """Pin the distinction between fresh absence and requested-load failure."""

  def test_fresh_loader_may_report_no_checkpoint(self):
    generator_class = _compile_generator()
    with tempfile.TemporaryDirectory() as directory:
      instance, _ = _checkpoint_instance(
        generator_class, directory, operation="fresh")
      self.assertIs(instance._GeneratorCore__load_chk(), False)

  def test_requested_resume_and_append_refuse_a_missing_member(self):
    for operation in ("resume", "append"):
      generator_class = _compile_generator()
      with tempfile.TemporaryDirectory() as directory:
        _, complete_members = _checkpoint_instance(
          generator_class, directory, operation=operation, complete=True)
        member_names = tuple(complete_members)
      for missing_name in member_names:
        with self.subTest(operation=operation, missing=missing_name):
          with tempfile.TemporaryDirectory() as directory:
            resolver = _ResolverBoundary()
            numpy_boundary = _FakeNumpy()
            generator_class = _compile_generator(
              np_boundary=numpy_boundary, resolver=resolver)
            instance, members = _checkpoint_instance(
              generator_class, directory, operation=operation, complete=True)
            missing = members[missing_name]
            missing.unlink()
            before = _snapshot(members)
            with self.assertRaises(CheckpointLoadError) as refusal:
              instance._GeneratorCore__load_chk()
            self.assertIn(operation, str(refusal.exception))
            self.assertIn(missing.name, str(refusal.exception))
            self.assertIn("No existing dataset file was changed",
                          str(refusal.exception))
            self.assertEqual(resolver.calls, [])
            self.assertEqual(numpy_boundary.loadtxt_calls, 0)
            self.assertFalse(missing.exists())
            self.assertEqual(_snapshot(members), before)

  def test_corrupt_checkpoint_load_exception_propagates(self):
    corrupt = ValueError("synthetic named-table width mismatch")
    resolver = _ResolverBoundary(error=corrupt)
    generator_class = _compile_generator(resolver=resolver)
    with tempfile.TemporaryDirectory() as directory:
      instance, members = _checkpoint_instance(
        generator_class, directory, operation="resume", complete=True)
      before = _snapshot(members)
      with self.assertRaises(CheckpointLoadError) as refusal:
        instance._GeneratorCore__run_mcmc()
      self.assertIn("Cannot resume", str(refusal.exception))
      self.assertIs(refusal.exception.__cause__, corrupt)
      self.assertEqual(_snapshot(members), before)
    self.assertEqual(len(resolver.calls), 1)

  def test_requested_load_refuses_semantically_corrupt_sidecars(self):
    cases = (
      ("wrong derived declaration",
       _ResolverBoundary(declarations=(
         ("p0", False, 2), ("p1", False, 3),
         ("not_chi2", True, 4))),
       _FakeNumpy(),
       "followed by chi2*"),
      ("unexpected exact-stem shadow sidecar",
       _ResolverBoundary(sidecar_path="params.1.paramnames"),
       _FakeNumpy(),
       "unexpected shadow path"),
    )
    for label, resolver, numpy_boundary, expected_text in cases:
      with self.subTest(case=label):
        generator_class = _compile_generator(
          np_boundary=numpy_boundary, resolver=resolver)
        with tempfile.TemporaryDirectory() as directory:
          instance, members = _checkpoint_instance(
            generator_class, directory, operation="append", complete=True)
          before = _snapshot(members)
          with self.assertRaises(CheckpointLoadError) as refusal:
            instance._GeneratorCore__run_mcmc()
          self.assertIn("Cannot append", str(refusal.exception))
          self.assertIsInstance(refusal.exception.__cause__, ValueError)
          self.assertIn(expected_text, str(refusal.exception.__cause__))
          self.assertEqual(_snapshot(members), before)

  def test_failure_flags_require_literal_producer_tokens(self):
    invalid_tokens = (
      "2", "1e-400", "1.00000000000000000000001",
      "0.99999999999999999999999", "-0", "0\v1", "0\f1",
      "0\x1c1", "0\x1d1", "0\x1e1")
    for token in invalid_tokens:
      with self.subTest(token=token):
        generator_class = _compile_generator(resolver=_ResolverBoundary())
        with tempfile.TemporaryDirectory() as directory:
          instance, members = _checkpoint_instance(
            generator_class, directory, operation="append", complete=True)
          members["failure flags"].write_text(token + "\n", encoding="ascii")
          before = _snapshot(members)
          with self.assertRaises(CheckpointLoadError) as refusal:
            instance._GeneratorCore__run_mcmc()
          self.assertIsInstance(refusal.exception.__cause__, ValueError)
          self.assertIn("literal producer tokens", str(
            refusal.exception.__cause__))
          self.assertEqual(_snapshot(members), before)

    for label, newline in (("LF", b"\n"), ("CRLF", b"\r\n"),
                           ("CR", b"\r")):
      with self.subTest(valid_newline=label):
        generator_class = _compile_generator(resolver=_ResolverBoundary())
        with tempfile.TemporaryDirectory() as directory:
          instance, members = _checkpoint_instance(
            generator_class, directory, operation="resume", complete=True)
          members["failure flags"].write_bytes(b"0" + newline)
          before = _snapshot(members)
          self.assertIs(instance._GeneratorCore__load_chk(), True)
          self.assertEqual(_snapshot(members), before)

  def test_run_mcmc_never_turns_requested_load_failure_into_fresh(self):
    generator_class = _compile_generator()

    def fail_load(instance):
      del instance
      raise _CheckpointFailure("requested checkpoint is unusable")

    setattr(generator_class, "_GeneratorCore__load_chk", fail_load)
    for operation, append in (("resume", 0), ("append", 1)):
      with self.subTest(operation=operation):
        instance = object.__new__(generator_class)
        instance.loadchk = 1
        instance.append = append
        instance.run_control = validate_run_control(1, append, 0)
        marker = _FreshMarker()
        instance.sampled_params = ("p0", "p1")
        instance.unif = marker
        caught = None
        with contextlib.redirect_stderr(io.StringIO()):
          try:
            instance._GeneratorCore__run_mcmc()
          except Exception as error:
            caught = error
        self.assertIsNotNone(caught)
        chain = []
        seen = set()
        while caught is not None and id(caught) not in seen:
          seen.add(id(caught))
          chain.append(caught)
          caught = caught.__cause__ or caught.__context__
        self.assertTrue(any(type(error) is _CheckpointFailure
                            for error in chain))
        self.assertFalse(marker.touched)

  def test_valid_resume_loads_without_entering_sampling(self):
    resolver = _ResolverBoundary()
    generator_class = _compile_generator(resolver=resolver)
    with tempfile.TemporaryDirectory() as directory:
      instance, members = _checkpoint_instance(
        generator_class, directory, operation="resume", complete=True)
      marker = _FreshMarker()
      instance.sampled_params = ("p0", "p1")
      instance.unif = marker
      before = _snapshot(members)
      instance._GeneratorCore__run_mcmc()
      self.assertFalse(marker.touched)
      self.assertTrue(instance.loadedfromchk)
      self.assertTrue(instance.loadedsamples)
      self.assertEqual(_snapshot(members), before)
    self.assertEqual(len(resolver.calls), 1)

  def test_family_checkpoint_census_includes_all_current_members(self):
    for name, (_, _, expected) in FAMILIES.items():
      with self.subTest(family=name):
        self.assertEqual(_family_checkpoint_files(name), expected)
    self.assertEqual(
      _family_checkpoint_files("mps", write_base=True),
      MPS_BASE_CHECKPOINTS)

  def test_family_checkpoint_geometry_matches_configured_axes(self):
    for name in ("background", "mps", "cmb"):
      with self.subTest(family=name, store="valid"):
        instance, _ = _family_geometry_fixture(name, corrupt=False)
        instance._dv_load_chk()
      with self.subTest(family=name, store="corrupt"):
        instance, _ = _family_geometry_fixture(name, corrupt=True)
        with self.assertRaises(ValueError):
          instance._dv_load_chk()


if __name__ == "__main__":
  unittest.main()
