#!/usr/bin/env python3
"""Prove requested checkpoint loads fail closed before fresh generation.

The generator module imports MPI, Cobaya, GetDist, emcee, and NumPy.  This CPU
child therefore AST-extracts the real ``GeneratorCore`` class plus each family
census/loader and drives only checkpoint decisions and geometry checks with
tiny filesystem and NumPy boundaries.  Fresh mode may report that no
checkpoint was loaded; requested resume and append may not use that result for
an absent or corrupt checkpoint.

Isolated in-memory mutations delete the member preflight, semantic sidecar
checks, family geometry checks, exception propagation, and correct operation
routing.  Each affected AID must turn red.
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
import warnings

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from compute_data_vectors import dataset_manifest
from compute_data_vectors.dataset_manifest import CheckpointLoadError
from compute_data_vectors.dataset_manifest import validate_run_control
from emulator import fixed_facts


GENERATOR = ROOT / "compute_data_vectors" / "generator_core.py"
PARAMETER_TABLE = ROOT / "emulator" / "parameter_table.py"
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
LEG_AIDS = (
  "checkpoint-refusal.missing-member",
  "checkpoint-refusal.corrupt-load",
  "checkpoint-refusal.no-fresh-fallback",
  "checkpoint-refusal.family-geometry",
)


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


class _FakeNumericTable:
  """One parameter row with a configurable all-finite verdict."""

  ndim = 2
  shape = (1, 5)
  size = 5

  def __init__(self, finite, label):
    self.finite = finite
    self.label = label


class _ParameterNumpy:
  """NumPy boundary for the production numeric-table loader."""

  float32 = object()

  def __init__(self, table):
    self.table = table

  def loadtxt(self, *args, **kwargs):
    del args, kwargs
    return self.table

  @staticmethod
  def isfinite(table):
    return SimpleNamespace(all=lambda: table.finite)


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
  """Sentinel raised by a requested checkpoint load."""


class _FreshPathTouched(RuntimeError):
  """Sentinel proving a failed requested load reached fresh work."""


class _FreshMarker:
  """The sampling branch compares its configured sampler mode to one."""

  def __init__(self):
    self.touched = False

  def __eq__(self, other):
    del other
    self.touched = True
    raise _FreshPathTouched("fresh generation was reached")


def _generator_class_node():
  """Return an isolated copy of the unique production GeneratorCore class."""
  tree = ast.parse(GENERATOR.read_text(encoding="utf-8"),
                   filename=str(GENERATOR))
  classes = [node for node in tree.body
             if isinstance(node, ast.ClassDef)
             and node.name == "GeneratorCore"]
  if len(classes) != 1:
    raise AssertionError("expected exactly one GeneratorCore class")
  return copy.deepcopy(classes[0])


def _method(class_node, name):
  """Return one uniquely named method from the extracted production class."""
  methods = [node for node in class_node.body
             if isinstance(node, ast.FunctionDef) and node.name == name]
  if len(methods) != 1:
    raise AssertionError("expected exactly one GeneratorCore." + name)
  return methods[0]


def _mutation_remove_member_preflight(class_node):
  """Mutation: delete the required-member preflight entirely."""
  method = _method(class_node, "__load_chk")
  kept = []
  removed = 0
  for statement in method.body:
    calls = [node for node in ast.walk(statement)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "require_checkpoint_members"]
    if calls:
      removed += len(calls)
    else:
      kept.append(statement)
  if removed != 1:
    raise AssertionError("member-preflight mutation needs exactly one call")
  method.body = kept


def _mutation_corruption_returns_false(class_node):
  """Mutation: restore exception-to-False conversion at the loader boundary."""
  method = _method(class_node, "__load_chk")
  method.body = [ast.Try(
    body=method.body,
    handlers=[ast.ExceptHandler(
      type=ast.Name(id="Exception", ctx=ast.Load()),
      name=None,
      body=[ast.Return(value=ast.Constant(value=False))])],
    orelse=[],
    finalbody=[])]


def _mutation_remove_semantic_sidecar_checks(class_node):
  """Mutation: accept shadow/schema sidecars and nonliteral fail tokens."""
  method = _method(class_node, "__load_chk")

  class RemoveChecks(ast.NodeTransformer):
    def __init__(self):
      self.removed = 0

    def visit_If(self, node):
      names = {item.id for item in ast.walk(node.test)
               if isinstance(item, ast.Name)}
      is_declaration_check = (
        "expected_declarations" in names
        and any(isinstance(item, ast.Attribute)
                and item.attr == "declarations"
                for item in ast.walk(node.test)))
      is_authoritative_sidecar_check = (
        "expected_sidecar" in names and "observed_sidecar" in names)
      is_failure_domain_check = "invalid_failure_tokens" in names
      if (is_declaration_check or is_authoritative_sidecar_check
          or is_failure_domain_check):
        self.removed += 1
        return ast.copy_location(ast.Pass(), node)
      return self.generic_visit(node)

  remover = RemoveChecks()
  remover.visit(method)
  if remover.removed != 3:
    raise AssertionError(
      "semantic-sidecar mutation needs exactly three validation checks")


def _mutation_requested_failure_reaches_fresh(class_node):
  """Mutation: swallow a run failure and touch the fresh-generation branch."""
  method = _method(class_node, "__run_mcmc")
  fresh_touch = ast.Expr(value=ast.Compare(
    left=ast.Attribute(
      value=ast.Name(id="self", ctx=ast.Load()),
      attr="unif",
      ctx=ast.Load()),
    ops=[ast.Eq()],
    comparators=[ast.Constant(value=1)]))
  method.body = [ast.Try(
    body=method.body,
    handlers=[ast.ExceptHandler(
      type=ast.Name(id="Exception", ctx=ast.Load()),
      name=None,
      body=[fresh_touch, ast.Return(value=ast.Constant(value=None))])],
    orelse=[],
    finalbody=[])]


def _mutation_resume_reaches_sampling(class_node):
  """Mutation: route a successful resume into the sampling branch."""
  method = _method(class_node, "__run_mcmc")
  changed = 0
  for node in ast.walk(method):
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
      continue
    if not isinstance(node.ops[0], ast.In) or len(node.comparators) != 1:
      continue
    comparator = node.comparators[0]
    if not isinstance(comparator, ast.Tuple):
      continue
    values = [item.value for item in comparator.elts
              if isinstance(item, ast.Constant)]
    if values != ["fresh", "append"]:
      continue
    comparator.elts.append(ast.Constant(value="resume"))
    changed += 1
  if changed != 1:
    raise AssertionError("resume-routing mutation needs one operation tuple")


def _compile_generator(np_boundary=None, mutation=None, resolver=None):
  """Compile the production class with an optional in-memory regression."""
  node = _generator_class_node()
  if mutation is not None:
    mutation(node)
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
  namespace.update({name: value for name, value in vars(dataset_manifest).items()
                    if not name.startswith("__")})
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["GeneratorCore"]


def _compile_numeric_table_loader(numpy_boundary, mutation=False):
  """Compile the production numeric-table loader with an optional bypass."""
  tree = ast.parse(PARAMETER_TABLE.read_text(encoding="utf-8"),
                   filename=str(PARAMETER_TABLE))
  functions = [copy.deepcopy(node) for node in tree.body
               if isinstance(node, ast.FunctionDef)
               and node.name == "_load_numeric_table"]
  if len(functions) != 1:
    raise AssertionError("expected one _load_numeric_table function")
  function = functions[0]
  if mutation:
    removed = 0
    body = []
    for statement in function.body:
      is_finite_check = (
        isinstance(statement, ast.If)
        and any(isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == "isfinite"
                and isinstance(item.func.value, ast.Name)
                and item.func.value.id == "np"
                for item in ast.walk(statement.test)))
      if is_finite_check:
        removed += 1
      else:
        body.append(statement)
    if removed != 1:
      raise AssertionError("finite-table mutation needs exactly one check")
    function.body = body
  module = ast.Module(body=[function], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {"np": numpy_boundary, "os": os, "warnings": warnings}
  exec(compile(module, str(PARAMETER_TABLE), "exec"), namespace)
  return namespace["_load_numeric_table"]


def _numeric_finiteness_contract(mutation=False):
  """All bookkeeping, sampled, and derived numeric cells must be finite."""
  problems = []
  valid_table = _FakeNumericTable(True, "valid")
  valid_loader = _compile_numeric_table_loader(
    _ParameterNumpy(valid_table), mutation=mutation)
  try:
    observed = valid_loader("params.1.txt")
  except Exception as error:
    problems.append("valid raised " + type(error).__name__)
  else:
    if observed is not valid_table:
      problems.append("valid table identity changed")
  for label in ("weight", "minuslogpost", "sampled", "derived"):
    table = _FakeNumericTable(False, label)
    loader = _compile_numeric_table_loader(
      _ParameterNumpy(table), mutation=mutation)
    caught = None
    try:
      loader("params.1.txt")
    except Exception as error:
      caught = error
    if not isinstance(caught, ValueError) or "nonfinite" not in str(caught):
      problems.append(label + " raised "
                      + ("nothing" if caught is None
                         else type(caught).__name__ + ": " + str(caught)))
  if problems:
    return False, "; ".join(problems)
  return True, "finite control loads; four nonfinite column roles refuse"


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


def _family_census_contract():
  """Require every current family payload and persisted axis sidecar."""
  observed = {}
  problems = []
  for name, (_, _, expected) in FAMILIES.items():
    observed[name] = _family_checkpoint_files(name)
    if observed[name] != expected:
      problems.append(name + "=" + repr(observed[name]))
  observed["mps-with-base"] = _family_checkpoint_files(
    "mps", write_base=True)
  if observed["mps-with-base"] != MPS_BASE_CHECKPOINTS:
    problems.append("mps-with-base=" + repr(observed["mps-with-base"]))
  if problems:
    return False, "; ".join(problems)
  return True, repr(observed)


def _compile_family_loader(name, numpy_boundary, mutation=False):
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

  if mutation:
    removed = 0

    class RemoveGeometry(ast.NodeTransformer):
      def visit_Expr(self, node):
        nonlocal removed
        if (isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_load_axis_checkpoint"):
          removed += 1
          return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)

      def visit_If(self, node):
        nonlocal removed
        has_width_index = any(
          isinstance(item, ast.Subscript)
          and isinstance(item.value, ast.Attribute)
          and item.value.attr == "shape"
          for item in ast.walk(node.test))
        has_checkpoint_width_error = any(
          isinstance(item, ast.Constant)
          and isinstance(item.value, str)
          and "chk datavectors" in item.value
          for item in ast.walk(node))
        if has_width_index and has_checkpoint_width_error:
          removed += 1
          return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)

    transformer = RemoveGeometry()
    methods = [transformer.visit(method) for method in methods]
    expected_removed = {"background": 2, "mps": 3, "cmb": 1}[name]
    if removed != expected_removed:
      raise AssertionError(
        name + " geometry mutation removed " + str(removed)
        + " checks, expected " + str(expected_removed))

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


def _family_geometry_fixture(name, corrupt=False, mutation=False):
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
  family_class = _compile_family_loader(
    name, numpy_boundary, mutation=mutation)
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


def _family_geometry_contract(mutation=False):
  """Require configured axes and CMB width; valid family stores still load."""
  problems = []
  details = []
  for name in ("background", "mps", "cmb"):
    valid, valid_boundary = _family_geometry_fixture(
      name, corrupt=False, mutation=mutation)
    try:
      valid._dv_load_chk()
    except Exception as error:
      problems.append(name + " valid raised " + type(error).__name__)
    invalid, invalid_boundary = _family_geometry_fixture(
      name, corrupt=True, mutation=mutation)
    caught = None
    try:
      invalid._dv_load_chk()
    except Exception as error:
      caught = error
    if not isinstance(caught, ValueError):
      problems.append(name + " corrupt raised "
                      + ("nothing" if caught is None
                         else type(caught).__name__))
    details.append(name + "="
                   + ("none" if caught is None else str(caught))
                   + "; valid-loads=" + str(len(valid_boundary.loads))
                   + "; corrupt-loads=" + str(len(invalid_boundary.loads)))
  if problems:
    return False, "; ".join(problems) + "; " + "; ".join(details)
  return True, "; ".join(details)


def _exception_chain_text(error):
  """Collect diagnostic text from an exception and its explicit causes."""
  parts = []
  seen = set()
  while error is not None and id(error) not in seen:
    seen.add(id(error))
    parts.append(str(error))
    error = error.__cause__ or error.__context__
  return " | ".join(parts)


def _missing_member_contract(generator_class, resolver, numpy_boundary):
  """Fresh absence is ordinary; resume/append absence is an error."""
  problems = []
  with tempfile.TemporaryDirectory() as directory:
    fresh, _ = _checkpoint_instance(
      generator_class, directory, operation="fresh")
    try:
      result = fresh._GeneratorCore__load_chk()
    except Exception as error:
      problems.append("fresh raised " + type(error).__name__)
    else:
      if result is not False:
        problems.append("fresh returned " + repr(result))

  missing_arms = 0
  for operation in ("resume", "append"):
    with tempfile.TemporaryDirectory() as directory:
      _, complete_members = _checkpoint_instance(
        generator_class, directory, operation=operation, complete=True)
      member_names = tuple(complete_members)
    for missing_name in member_names:
      missing_arms += 1
      with tempfile.TemporaryDirectory() as directory:
        instance, members = _checkpoint_instance(
          generator_class, directory, operation=operation, complete=True)
        missing = members[missing_name]
        missing.unlink()
        before = _snapshot(members)
        resolver_before = len(resolver.calls)
        numpy_before = numpy_boundary.loadtxt_calls
        caught = None
        try:
          result = instance._GeneratorCore__load_chk()
        except Exception as error:
          caught = error
        else:
          problems.append(operation + " missing " + missing_name
                          + " returned " + repr(result))
        if type(caught) is not CheckpointLoadError:
          problems.append(operation + " missing " + missing_name
                          + " raised "
                          + ("nothing" if caught is None
                             else type(caught).__name__))
        else:
          text = str(caught)
          if (operation not in text or missing.name not in text
              or "No existing dataset file was changed" not in text):
            problems.append(operation + " missing " + missing_name
                            + " had untruthful message " + repr(text))
        if len(resolver.calls) != resolver_before:
          problems.append(operation + " parsed parameters before refusing "
                          + missing_name)
        if numpy_boundary.loadtxt_calls != numpy_before:
          problems.append(operation + " parsed failure flags before refusing "
                          + missing_name)
        if missing.exists():
          problems.append(operation + " recreated missing " + missing_name)
        if _snapshot(members) != before:
          problems.append(operation + " changed a surviving member while "
                          + missing_name + " was absent")
  if problems:
    return False, "; ".join(problems)
  family_ok, family_detail = _family_census_contract()
  if not family_ok:
    return False, "family census: " + family_detail
  return True, ("fresh=False; all " + str(missing_arms)
                + " missing-member arms raise before parsing; families="
                + family_detail)


def _corrupt_load_contract(generator_class, resolver, original_error):
  """A named-table validation error must retain its cause and preserve files."""
  caught = None
  with tempfile.TemporaryDirectory() as directory:
    instance, members = _checkpoint_instance(
      generator_class, directory, operation="resume", complete=True)
    marker = _FreshMarker()
    instance.sampled_params = ("p0", "p1")
    instance.unif = marker
    before = _snapshot(members)
    try:
      instance._GeneratorCore__run_mcmc()
    except Exception as error:
      caught = error
    preserved = _snapshot(members) == before

  text = "" if caught is None else _exception_chain_text(caught)
  ok = (type(caught) is CheckpointLoadError
        and caught.__cause__ is original_error
        and len(resolver.calls) == 1
        and "Cannot resume" in text
        and "synthetic named-table width mismatch" in text
        and not marker.touched
        and preserved)
  detail = ("resolver_calls=" + str(len(resolver.calls))
            + "; original-cause=" + str(
              caught is not None and caught.__cause__ is original_error)
            + "; fresh-touched=" + str(marker.touched)
            + "; bytes+mtime-preserved=" + str(preserved)
            + "; exception="
            + ("none" if caught is None
               else type(caught).__name__ + ": " + text))
  return ok, detail


def _semantic_sidecar_contract(generator_class, expected_text,
                               failure_text=None):
  """Semantic sidecar corruption must stop append before sampling or writes."""
  with tempfile.TemporaryDirectory() as directory:
    instance, members = _checkpoint_instance(
      generator_class, directory, operation="append", complete=True)
    if failure_text is not None:
      members["failure flags"].write_text(
        failure_text, encoding="ascii")
    marker = _FreshMarker()
    instance.sampled_params = ("p0", "p1")
    instance.unif = marker
    before = _snapshot(members)
    caught = None
    try:
      instance._GeneratorCore__run_mcmc()
    except Exception as error:
      caught = error
    preserved = _snapshot(members) == before
  cause = None if caught is None else caught.__cause__
  ok = (type(caught) is CheckpointLoadError
        and isinstance(cause, ValueError)
        and expected_text in str(cause)
        and not marker.touched
        and preserved)
  return ok, ("cause="
              + ("none" if cause is None
                 else type(cause).__name__ + ": " + str(cause))
              + "; sampling-touched=" + str(marker.touched)
              + "; bytes+mtime-preserved=" + str(preserved))


def _failure_token_contract(mutation=False):
  """Only the producer's literal one-token-per-line 0/1 form is accepted."""
  problems = []
  tokens = ("2", "1e-400", "1.00000000000000000000001",
            "0.99999999999999999999999", "-0", "0\v1", "0\f1",
            "0\x1c1", "0\x1d1", "0\x1e1")
  for token in tokens:
    generator_class = _compile_generator(
      resolver=_ResolverBoundary(),
      mutation=(_mutation_remove_semantic_sidecar_checks
                if mutation else None))
    ok, detail = _semantic_sidecar_contract(
      generator_class,
      "literal producer tokens",
      failure_text=token + "\n")
    if not ok:
      problems.append(token + " => " + detail)
  if problems:
    return False, "; ".join(problems)

  newline_problems = []
  for label, newline in (("LF", b"\n"), ("CRLF", b"\r\n"),
                         ("CR", b"\r")):
    generator_class = _compile_generator(resolver=_ResolverBoundary())
    with tempfile.TemporaryDirectory() as directory:
      instance, members = _checkpoint_instance(
        generator_class, directory, operation="resume", complete=True)
      members["failure flags"].write_bytes(b"0" + newline)
      before = _snapshot(members)
      try:
        loaded = instance._GeneratorCore__load_chk()
      except Exception as error:
        newline_problems.append(
          label + " raised " + type(error).__name__ + ": " + str(error))
        continue
      if loaded is not True or _snapshot(members) != before:
        newline_problems.append(
          label + " loaded=" + repr(loaded)
          + "; preserved=" + repr(_snapshot(members) == before))
  if newline_problems:
    return False, "; ".join(newline_problems)
  return True, ("ten nonliteral rounding, underflow, signed-zero, and "
                "control-separator tokens refuse exactly; LF/CRLF/CR "
                "physical lines load")


def _valid_resume_contract(generator_class, resolver):
  """A valid resume loads exactly once and never enters sampling."""
  with tempfile.TemporaryDirectory() as directory:
    instance, members = _checkpoint_instance(
      generator_class, directory, operation="resume", complete=True)
    marker = _FreshMarker()
    instance.sampled_params = ("p0", "p1")
    instance.unif = marker
    before = _snapshot(members)
    try:
      instance._GeneratorCore__run_mcmc()
    except Exception as error:
      return False, "raised " + type(error).__name__ + ": " + str(error)
    preserved = _snapshot(members) == before
  ok = (not marker.touched and instance.loadedfromchk
        and instance.loadedsamples and len(resolver.calls) == 1 and preserved)
  return ok, ("resolver_calls=" + str(len(resolver.calls))
              + "; flags=" + repr((instance.loadedfromchk,
                                    instance.loadedsamples))
              + "; fresh-touched=" + str(marker.touched)
              + "; bytes+mtime-preserved=" + str(preserved))


def _no_fresh_fallback_contract(generator_class):
  """A requested-load exception must escape before any fresh-branch work."""
  problems = []

  def fail_load(instance):
    del instance
    raise _CheckpointFailure("requested checkpoint is unusable")

  setattr(generator_class, "_GeneratorCore__load_chk", fail_load)
  for operation, append in (("resume", 0), ("append", 1)):
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
    chain = []
    seen = set()
    current = caught
    while current is not None and id(current) not in seen:
      seen.add(id(current))
      chain.append(current)
      current = current.__cause__ or current.__context__
    if not any(type(error) is _CheckpointFailure for error in chain):
      problems.append(
        operation + " raised "
        + ("nothing" if caught is None else type(caught).__name__))
    if marker.touched:
      problems.append(operation + " touched fresh generation")
  if problems:
    return False, "; ".join(problems)
  return True, "resume/append load errors escape; fresh marker untouched"


def _aid(aid, normal, mutation_red, detail):
  """Emit exactly one terminal for a board-declared evidence leg."""
  passed = normal and mutation_red
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + aid + "  (" + detail
        + "; mutation-red=" + str(mutation_red) + ")")
  print("##AID " + aid + " " + mark)
  return passed


def main():
  """Run the four contract arms and their isolated AST mutations."""
  missing_resolver = _ResolverBoundary()
  missing_numpy = _FakeNumpy()
  production = _compile_generator(
    np_boundary=missing_numpy, resolver=missing_resolver)
  missing_ok, missing_detail = _missing_member_contract(
    production, missing_resolver, missing_numpy)
  missing_mutant_resolver = _ResolverBoundary()
  missing_mutant_numpy = _FakeNumpy()
  missing_mutant = _compile_generator(
    np_boundary=missing_mutant_numpy,
    resolver=missing_mutant_resolver,
    mutation=_mutation_remove_member_preflight)
  missing_mutant_ok, _ = _missing_member_contract(
    missing_mutant, missing_mutant_resolver, missing_mutant_numpy)
  missing = _aid(
    LEG_AIDS[0],
    missing_ok,
    not missing_mutant_ok,
    missing_detail)

  corrupt_error = ValueError("synthetic named-table width mismatch")
  corrupt_resolver = _ResolverBoundary(error=corrupt_error)
  production_corrupt = _compile_generator(resolver=corrupt_resolver)
  corrupt_ok, corrupt_detail = _corrupt_load_contract(
    production_corrupt, corrupt_resolver, corrupt_error)
  corrupt_mutant_error = ValueError(
    "synthetic named-table width mismatch")
  corrupt_mutant_resolver = _ResolverBoundary(error=corrupt_mutant_error)
  corrupt_mutant = _compile_generator(
    resolver=corrupt_mutant_resolver,
    mutation=_mutation_corruption_returns_false)
  corrupt_mutant_ok, _ = _corrupt_load_contract(
    corrupt_mutant, corrupt_mutant_resolver, corrupt_mutant_error)

  wrong_schema = (("p0", False, 2), ("p1", False, 3),
                  ("not_chi2", True, 4))
  schema_production = _compile_generator(
    resolver=_ResolverBoundary(declarations=wrong_schema))
  schema_ok, schema_detail = _semantic_sidecar_contract(
    schema_production, "followed by chi2*")
  schema_mutant = _compile_generator(
    resolver=_ResolverBoundary(declarations=wrong_schema),
    mutation=_mutation_remove_semantic_sidecar_checks)
  schema_mutant_ok, _ = _semantic_sidecar_contract(
    schema_mutant, "followed by chi2*")

  shadow_production = _compile_generator(
    resolver=_ResolverBoundary(sidecar_path="params.1.paramnames"))
  shadow_ok, shadow_detail = _semantic_sidecar_contract(
    shadow_production, "unexpected shadow path")
  shadow_mutant = _compile_generator(
    resolver=_ResolverBoundary(sidecar_path="params.1.paramnames"),
    mutation=_mutation_remove_semantic_sidecar_checks)
  shadow_mutant_ok, _ = _semantic_sidecar_contract(
    shadow_mutant, "unexpected shadow path")

  failure_ok, failure_detail = _failure_token_contract(mutation=False)
  failure_mutant_ok, _ = _failure_token_contract(mutation=True)
  finite_ok, finite_detail = _numeric_finiteness_contract(mutation=False)
  finite_mutant_ok, _ = _numeric_finiteness_contract(mutation=True)
  corrupt_result = _aid(
    LEG_AIDS[1],
    corrupt_ok and schema_ok and shadow_ok and failure_ok and finite_ok,
    (not corrupt_mutant_ok
     and not schema_mutant_ok
     and not shadow_mutant_ok
     and not failure_mutant_ok
     and not finite_mutant_ok),
    corrupt_detail + "; schema: " + schema_detail
    + "; sidecar: " + shadow_detail
    + "; fail-flags: " + failure_detail
    + "; finiteness: " + finite_detail)

  run_production = _compile_generator()
  fallback_ok, fallback_detail = _no_fresh_fallback_contract(run_production)
  fallback_mutant = _compile_generator(
    mutation=_mutation_requested_failure_reaches_fresh)
  fallback_mutant_ok, _ = _no_fresh_fallback_contract(fallback_mutant)

  resume_resolver = _ResolverBoundary()
  resume_production = _compile_generator(resolver=resume_resolver)
  resume_ok, resume_detail = _valid_resume_contract(
    resume_production, resume_resolver)
  resume_mutant_resolver = _ResolverBoundary()
  resume_mutant = _compile_generator(
    resolver=resume_mutant_resolver,
    mutation=_mutation_resume_reaches_sampling)
  resume_mutant_ok, _ = _valid_resume_contract(
    resume_mutant, resume_mutant_resolver)
  fallback = _aid(
    LEG_AIDS[2],
    fallback_ok and resume_ok,
    not fallback_mutant_ok and not resume_mutant_ok,
    fallback_detail + "; valid-resume: " + resume_detail)

  geometry_ok, geometry_detail = _family_geometry_contract(mutation=False)
  geometry_mutant_ok, _ = _family_geometry_contract(mutation=True)
  geometry = _aid(
    LEG_AIDS[3],
    geometry_ok,
    not geometry_mutant_ok,
    geometry_detail)

  if not all((missing, corrupt_result, fallback, geometry)):
    print("generator-checkpoint-refusal: FAIL")
    return 1
  print("generator-checkpoint-refusal: ALL PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
