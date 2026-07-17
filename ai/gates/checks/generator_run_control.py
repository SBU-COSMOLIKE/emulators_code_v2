#!/usr/bin/env python3
"""Prove the generator refuses unsafe run controls before path mutation.

The production generator imports MPI, Cobaya, GetDist, and the family physics
stack.  This CPU child therefore imports only its pure run-control module and
executes the real ``GeneratorCore.__init__`` method from the production syntax
tree with tiny setup/MPI stubs.  The setup stub is deliberately a filesystem
mutation sentinel: an invalid state must refuse before that stub can run.

Seventeen in-memory mutations keep the evidence load-bearing. They coerce bools,
restore append-without-load, move setup or another statement before validation,
hide a preceding expression inside the validator assignment, shadow the imported
validator, route on raw chain mode, restore raw append/load setup assignments,
remove or misroute stem scoping, borrow data-vector members in chain-only mode,
weaken the full census, remove the chain-only loader's early return, bypass or
misplace the append helper, and remove the append helper's chain-only barrier.
Each mutation must make its acceptance arm red.
"""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

from compute_data_vectors.dataset_manifest import RunControl
from compute_data_vectors.dataset_manifest import build_dataset_member_census
from compute_data_vectors.dataset_manifest import require_checkpoint_members
from compute_data_vectors.dataset_manifest import scope_dataset_stem
from compute_data_vectors.dataset_manifest import validate_run_control


ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "compute_data_vectors" / "generator_core.py"


class _IntSubclass(int):
  """An integer subclass that the native-integer boundary must reject."""


class _FakeComm:
  """Rank-zero communicator sufficient for the extracted constructor."""

  @staticmethod
  def Get_rank():
    return 0

  @staticmethod
  def bcast(value, root=0):
    """Return rank zero's value, as a one-process communicator would."""
    if root != 0:
      raise AssertionError("the constructor must broadcast from rank zero")
    return value

  @staticmethod
  def Barrier():
    """Represent the final one-process synchronization boundary."""


class _FakeMPI:
  """Namespace matching the one MPI attribute read by the constructor."""

  COMM_WORLD = _FakeComm()


def _call_name(node):
  """Return the final name of one call expression, or ``None``."""
  if not isinstance(node, ast.Call):
    return None
  if isinstance(node.func, ast.Name):
    return node.func.id
  if isinstance(node.func, ast.Attribute):
    return node.func.attr
  return None


def _contains_call(node, name):
  """Tell whether one syntax-tree statement calls ``name``."""
  return any(_call_name(child) == name for child in ast.walk(node))


def _production_tree():
  """Parse the complete production module for binding and dataflow checks."""
  return ast.parse(GENERATOR.read_text(encoding="utf-8"),
                   filename=str(GENERATOR))


class _ModuleBindings(ast.NodeVisitor):
  """Collect names bound in module scope without descending into new scopes."""

  def __init__(self):
    self.names = []

  def visit_FunctionDef(self, node):
    self.names.append(node.name)

  def visit_AsyncFunctionDef(self, node):
    self.names.append(node.name)

  def visit_ClassDef(self, node):
    self.names.append(node.name)

  def visit_Lambda(self, node):
    return

  def visit_Import(self, node):
    for alias in node.names:
      self.names.append(alias.asname or alias.name.split(".")[0])

  def visit_ImportFrom(self, node):
    for alias in node.names:
      self.names.append(alias.asname or alias.name)

  def visit_Name(self, node):
    if isinstance(node.ctx, (ast.Store, ast.Del)):
      self.names.append(node.id)


def _generator_class(tree):
  """Return the unique production ``GeneratorCore`` class node."""
  classes = [node for node in tree.body
             if isinstance(node, ast.ClassDef)
             and node.name == "GeneratorCore"]
  if len(classes) != 1:
    raise AssertionError("expected exactly one GeneratorCore class")
  return classes[0]


def _generator_method(tree, name):
  """Return one uniquely named production ``GeneratorCore`` method."""
  methods = [node for node in _generator_class(tree).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
  if len(methods) != 1:
    raise AssertionError("expected exactly one GeneratorCore." + name)
  return methods[0]


def _attribute_chain(node):
  """Return a dotted attribute chain such as ``self.run_control.append``."""
  parts = []
  while isinstance(node, ast.Attribute):
    parts.append(node.attr)
    node = node.value
  if isinstance(node, ast.Name):
    parts.append(node.id)
    return ".".join(reversed(parts))
  return None


def _source_binding_and_order_contract(tree):
  """Bind the extracted validator to production and require it first."""
  intended_imports = []
  for node in tree.body:
    if not isinstance(node, ast.ImportFrom):
      continue
    for alias in node.names:
      local = alias.asname or alias.name
      if local == "validate_run_control":
        intended_imports.append((node.module, alias.name, alias.asname))
  expected_import = [(
    "compute_data_vectors.dataset_manifest",
    "validate_run_control",
    None)]
  if intended_imports != expected_import:
    return False, "validator imports=" + repr(intended_imports)

  bindings = _ModuleBindings()
  bindings.visit(tree)
  if bindings.names.count("validate_run_control") != 1:
    return False, "validator module bindings=" + repr(bindings.names.count(
      "validate_run_control"))

  constructor = _generator_method(tree, "__init__")
  if not constructor.body:
    return False, "constructor body is empty"
  statement = constructor.body[0]
  if (not isinstance(statement, ast.Assign)
      or len(statement.targets) != 1
      or not isinstance(statement.targets[0], ast.Name)
      or statement.targets[0].id != "run_control"):
    return False, "validator statement zero is not a sole run_control assignment"
  call = statement.value
  if (not isinstance(call, ast.Call)
      or not isinstance(call.func, ast.Name)
      or call.func.id != "validate_run_control"):
    return False, "run_control RHS is not the direct validator call"

  call_statements = [index for index, statement in enumerate(constructor.body)
                     if _contains_call(statement, "validate_run_control")]
  if call_statements != [0]:
    return False, "validator statement indices=" + repr(call_statements)
  if call.args:
    return False, "validator must use the three named arguments"
  observed = [(keyword.arg, _attribute_chain(keyword.value))
              for keyword in call.keywords]
  expected = [
    ("loadchk", "cli_args.loadchk"),
    ("append", "cli_args.append"),
    ("chain", "cli_args.chain"),
  ]
  if observed != expected:
    return False, "validator arguments=" + repr(observed)
  return True, ("sole dataset_manifest binding; direct validator assignment "
                "is statement zero")


def _setup_dataflow_contract(tree):
  """Require setup to consume normalized append/load values, never raw args."""
  setup = _generator_method(tree, "__setup_flags")
  assignments = {}
  for node in ast.walk(setup):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
      continue
    target = _attribute_chain(node.targets[0])
    if target in ("self.append", "self.loadchk"):
      assignments[target] = _attribute_chain(node.value)
  expected = {
    "self.append": "self.run_control.append",
    "self.loadchk": "self.run_control.loadchk",
  }
  if assignments != expected:
    return False, "setup assignments=" + repr(assignments)
  raw_reads = [_attribute_chain(node) for node in ast.walk(setup)
               if isinstance(node, ast.Attribute)]
  raw_reads = [name for name in raw_reads
               if name in ("self.args.append", "self.args.loadchk")]
  if raw_reads:
    return False, "raw setup reads=" + repr(raw_reads)
  return True, "setup consumes run_control.append/loadchk"


def _structural_mutations_red():
  """Prove order, binding, and setup-dataflow censuses reject regressions."""
  prefixed = copy.deepcopy(_production_tree())
  constructor = _generator_method(prefixed, "__init__")
  constructor.body.insert(0, ast.Pass())
  prefix_ok, _ = _source_binding_and_order_contract(prefixed)

  wrapped = copy.deepcopy(_production_tree())
  constructor = _generator_method(wrapped, "__init__")
  original_call = constructor.body[0].value
  constructor.body[0].value = ast.Subscript(
    value=ast.Tuple(
      elts=[ast.Constant(value="preceding expression"), original_call],
      ctx=ast.Load()),
    slice=ast.Constant(value=1),
    ctx=ast.Load())
  wrapped_ok, _ = _source_binding_and_order_contract(wrapped)

  shadowed = copy.deepcopy(_production_tree())
  shadowed.body.insert(0, ast.Assign(
    targets=[ast.Name(id="validate_run_control", ctx=ast.Store())],
    value=ast.Constant(value=None)))
  shadow_ok, _ = _source_binding_and_order_contract(shadowed)

  raw_setup = copy.deepcopy(_production_tree())
  setup = _generator_method(raw_setup, "__setup_flags")
  changes = 0
  for node in ast.walk(setup):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
      continue
    target = _attribute_chain(node.targets[0])
    if target not in ("self.append", "self.loadchk"):
      continue
    node.value = ast.Attribute(
      value=ast.Attribute(
        value=ast.Name(id="self", ctx=ast.Load()),
        attr="args",
        ctx=ast.Load()),
      attr=target.split(".")[-1],
      ctx=ast.Load())
    changes += 1
  raw_setup_ok, _ = _setup_dataflow_contract(raw_setup)
  return (not prefix_ok) and (not wrapped_ok) and (not shadow_ok) and (
    changes == 2) and (not raw_setup_ok)


class _RawModeRouting(ast.NodeTransformer):
  """Mutation: route data-vector work on raw ``cli_args.chain`` again."""

  def __init__(self):
    self.changes = 0

  def visit_Compare(self, node):
    node = self.generic_visit(node)
    if _attribute_chain(node.left) != "self.run_control.dataset_mode":
      return node
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
      return node
    if len(node.comparators) != 1:
      return node
    comparator = node.comparators[0]
    if not isinstance(comparator, ast.Constant) or comparator.value != "full":
      return node
    self.changes += 1
    replacement = ast.Compare(
      left=ast.Attribute(
        value=ast.Attribute(
          value=ast.Name(id="self", ctx=ast.Load()),
          attr="args",
          ctx=ast.Load()),
        attr="chain",
        ctx=ast.Load()),
      ops=[ast.NotEq()],
      comparators=[ast.Constant(value=1)])
    return ast.copy_location(replacement, node)


def _constructor_class(validator, setup_first=False, raw_mode_routing=False):
  """Compile the real constructor with optional setup/mode mutations.

  ``setup_first`` is an in-memory source mutation: it swaps the top-level
  validator and setup statements.  The extracted method otherwise remains the
  exact production syntax tree.
  """
  tree = _production_tree()
  constructor = copy.deepcopy(_generator_method(tree, "__init__"))
  if setup_first:
    validator_indices = [index for index, statement in enumerate(constructor.body)
                         if _contains_call(statement, "validate_run_control")]
    setup_indices = [index for index, statement in enumerate(constructor.body)
                     if _contains_call(statement, "__setup_flags")]
    if len(validator_indices) != 1 or len(setup_indices) != 1:
      raise AssertionError(
        "setup-order mutation needs one validator and one setup statement")
    validator_index = validator_indices[0]
    setup_index = setup_indices[0]
    constructor.body[validator_index], constructor.body[setup_index] = (
      constructor.body[setup_index], constructor.body[validator_index])
  if raw_mode_routing:
    mutation = _RawModeRouting()
    constructor = mutation.visit(constructor)
    if mutation.changes != 1:
      raise AssertionError(
        "raw-mode mutation needs exactly one normalized mode comparison")

  extracted = ast.ClassDef(
    name="GeneratorCore",
    bases=[],
    keywords=[],
    body=[constructor],
    decorator_list=[])
  module = ast.Module(body=[extracted], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "MPI": _FakeMPI,
    "validate_run_control": validator,
  }
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["GeneratorCore"]


def _drive_constructor(loadchk, append, chain, validator=validate_run_control,
                       setup_first=False, raw_mode_routing=False):
  """Run the extracted constructor and record every expensive boundary."""
  events = []
  with tempfile.TemporaryDirectory() as directory:
    touched = Path(directory) / "setup-touched-output"
    constructor_class = _constructor_class(
      validator=validator,
      setup_first=setup_first,
      raw_mode_routing=raw_mode_routing)

    def setup(instance):
      events.append("setup")
      touched.write_text("setup mutated the output root", encoding="utf-8")
      instance.setup = True

    def run_mcmc(instance):
      events.append("mcmc")

    def generate(instance):
      events.append("datavectors")

    setattr(constructor_class, "_GeneratorCore__setup_flags", setup)
    setattr(constructor_class, "_GeneratorCore__run_mcmc", run_mcmc)
    setattr(constructor_class, "_GeneratorCore__generate_datavectors", generate)
    setattr(constructor_class, "_prepare_dataset_publication", lambda instance: None)
    setattr(constructor_class, "_publish_dataset_generation", lambda instance: None)

    error = None
    try:
      constructor_class(SimpleNamespace(
        loadchk=loadchk, append=append, chain=chain))
    except Exception as exc:  # the caller checks exact class/text where needed
      error = exc
    return error, tuple(events), touched.exists()


def _binary_state_contract(validator):
  """Check legal operations, modes, immutability, and native integer types."""
  expected = {
    (0, 0): "fresh",
    (1, 0): "resume",
    (1, 1): "append",
  }
  observed = {}
  for loadchk, append in expected:
    control = validator(loadchk, append, 0)
    observed[(control.loadchk, control.append)] = control.operation
    if control.dataset_mode != "full":
      return False, "full mode was " + repr(control.dataset_mode)
    chain_control = validator(loadchk, append, 1)
    if chain_control.operation != control.operation:
      return False, "chain axis changed the operation"
    if chain_control.dataset_mode != "chain-only":
      return False, "chain-only mode was " + repr(chain_control.dataset_mode)
  if observed != expected:
    return False, "operation matrix=" + repr(observed)

  defaulted = validator(None, None, None)
  if (defaulted.loadchk, defaulted.append, defaulted.chain) != (0, 0, 0):
    return False, "None defaults=" + repr(defaulted)
  try:
    defaulted.append = 1
  except Exception:
    pass
  else:
    return False, "normalized record remained mutable"

  invalid = (False, True, -1, 2, 1.0, "1", _IntSubclass(1))
  for name in ("loadchk", "append", "chain"):
    for value in invalid:
      values = {"loadchk": 0, "append": 0, "chain": 0}
      values[name] = value
      try:
        validator(**values)
      except ValueError as exc:
        if "--" + name not in str(exc):
          return False, name + " error did not name its flag"
      else:
        return False, name + " accepted " + repr(value)
  return True, "3 legal operations; 2 modes; native int only"


def _append_requires_load_contract(validator):
  """Check the one illegal binary pair and its teaching diagnostic."""
  try:
    validator(0, 1, 0)
  except ValueError as exc:
    message = str(exc)
    required = (
      "--loadchk=0",
      "--append=1",
      "append extends a validated prior dataset",
      "never starts fresh generation",
    )
    missing = [phrase for phrase in required if phrase not in message]
    if missing:
      return False, "diagnostic missing " + repr(missing)
    return True, message
  return False, "append=1/loadchk=0 was accepted"


def _force_dataset_mode(dataset_mode):
  """Return a validator whose normalized mode can disagree with raw input."""
  def force(loadchk, append, chain):
    control = validate_run_control(loadchk, append, chain)
    return RunControl(
      loadchk=control.loadchk,
      append=control.append,
      chain=1 if dataset_mode == "chain-only" else 0,
      operation=control.operation,
      dataset_mode=dataset_mode)
  return force


def _pre_mutation_contract(validator, setup_first=False,
                           raw_mode_routing=False):
  """Check invalid rejection dominates setup and normalized mode is consumed."""
  source_ok, source_detail = _source_binding_and_order_contract(
    _production_tree())
  if not source_ok:
    return False, source_detail
  setup_ok, setup_detail = _setup_dataflow_contract(_production_tree())
  if not setup_ok:
    return False, setup_detail

  error, events, touched = _drive_constructor(
    loadchk=0, append=1, chain=0,
    validator=validator,
    setup_first=setup_first,
    raw_mode_routing=raw_mode_routing)
  if not isinstance(error, ValueError):
    return False, "invalid pair error=" + repr(error)
  if events or touched:
    return False, "invalid pair reached events=" + repr(events)

  error, full_events, _ = _drive_constructor(
    loadchk=0, append=0, chain=0,
    validator=validator,
    raw_mode_routing=raw_mode_routing)
  if error is not None or full_events != ("setup", "mcmc", "datavectors"):
    return False, "full events=" + repr(full_events) + "; error=" + repr(error)
  error, chain_events, _ = _drive_constructor(
    loadchk=0, append=0, chain=1,
    validator=validator,
    raw_mode_routing=raw_mode_routing)
  if error is not None or chain_events != ("setup", "mcmc"):
    return False, "chain events=" + repr(chain_events) + "; error=" + repr(error)
  # Deliberately disagree with the raw CLI axis. Production must route on the
  # immutable normalized record, not on a coincident raw value.
  error, forced_chain_events, _ = _drive_constructor(
    loadchk=0, append=0, chain=0,
    validator=_force_dataset_mode("chain-only"),
    raw_mode_routing=raw_mode_routing)
  if error is not None or forced_chain_events != ("setup", "mcmc"):
    return False, "forced-chain events=" + repr(forced_chain_events)
  error, forced_full_events, _ = _drive_constructor(
    loadchk=0, append=0, chain=1,
    validator=_force_dataset_mode("full"),
    raw_mode_routing=raw_mode_routing)
  if error is not None or forced_full_events != (
      "setup", "mcmc", "datavectors"):
    return False, "forced-full events=" + repr(forced_full_events)
  return True, ("invalid events=(); normalized mode overrides raw; "
                + source_detail + "; " + setup_detail)


def _is_mode_compare(node, mode):
  """Tell whether one expression compares the normalized mode to ``mode``."""
  return (
    isinstance(node, ast.Compare)
    and _attribute_chain(node.left) == "self.run_control.dataset_mode"
    and len(node.ops) == 1
    and isinstance(node.ops[0], ast.Eq)
    and len(node.comparators) == 1
    and isinstance(node.comparators[0], ast.Constant)
    and node.comparators[0].value == mode)


def _is_operation_compare(node, operation):
  """Tell whether one expression compares the operation to ``operation``."""
  return (
    isinstance(node, ast.Compare)
    and _attribute_chain(node.left) == "self.run_control.operation"
    and len(node.ops) == 1
    and isinstance(node.ops[0], ast.Eq)
    and len(node.comparators) == 1
    and isinstance(node.comparators[0], ast.Constant)
    and node.comparators[0].value == operation)


def _stem_scope_structure(tree):
  """Require all three output stems to consume one pure normalized scope."""
  intended_imports = []
  for node in tree.body:
    if not isinstance(node, ast.ImportFrom):
      continue
    for alias in node.names:
      local = alias.asname or alias.name
      if local == "scope_dataset_stem":
        intended_imports.append((node.module, alias.name, alias.asname))
  expected_import = [(
    "compute_data_vectors.dataset_manifest",
    "scope_dataset_stem",
    None)]
  if intended_imports != expected_import:
    return False, "stem-scoper imports=" + repr(intended_imports)

  bindings = _ModuleBindings()
  bindings.visit(tree)
  if bindings.names.count("scope_dataset_stem") != 1:
    return False, "stem-scoper module bindings=" + repr(
      bindings.names.count("scope_dataset_stem"))

  setup = _generator_method(tree, "__setup_flags")
  expected = {
    "self.dvsf": ("self.dvsf", "self.run_control.dataset_mode"),
    "self.paramsf": ("self.paramsf", "self.run_control.dataset_mode"),
    "self.failf": ("self.failf", "self.run_control.dataset_mode"),
  }
  observed = {}
  scoper_calls = 0
  for node in ast.walk(setup):
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
      continue
    call = node.value
    if (not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "scope_dataset_stem"):
      continue
    scoper_calls += 1
    target = _attribute_chain(node.targets[0])
    if call.keywords or len(call.args) != 2:
      observed[target] = ("invalid-call-shape",)
    else:
      observed[target] = (
        _attribute_chain(call.args[0]),
        _attribute_chain(call.args[1]))
  if scoper_calls != 3:
    return False, "stem-scoper call count=" + repr(scoper_calls)
  if observed != expected:
    return False, "stem-scoper assignments=" + repr(observed)
  return True, "dvsf/paramsf/failf each consume normalized dataset mode"


def _stem_scope_behavior(scoper):
  """Drive the pure scoper over all three stems and its refusal domain."""
  stems = (
    "/repo/chains/dv_probe_unifs",
    "/repo/chains/params_probe_unifs",
    "/repo/chains/fail_probe_unifs",
  )
  for stem in stems:
    if scoper(stem, "full") != stem:
      return False, "full scope changed " + repr(stem)
    expected = stem + "_chain_only"
    if scoper(stem, "chain-only") != expected:
      return False, "chain-only scope did not produce " + repr(expected)

  invalid_stems = (None, "", Path("not-a-string"))
  for stem in invalid_stems:
    try:
      scoper(stem, "full")
    except ValueError:
      pass
    else:
      return False, "invalid stem accepted: " + repr(stem)
  for mode in (None, 0, "FULL", "chain"):
    try:
      scoper(stems[0], mode)
    except ValueError:
      pass
    else:
      return False, "invalid normalized mode accepted: " + repr(mode)
  return True, "three stems use disjoint full/chain-only scopes"


class _ForbiddenNumpy:
  """Sentinel proving a chain-only load returned before fail/DV parsing."""

  def __getattr__(self, name):
    raise AssertionError("chain-only checkpoint touched numpy." + name)


def _checkpoint_class(tree, resolver, member_preflight):
  """Compile the real census and loader with small pure boundaries."""
  methods = [
    copy.deepcopy(_generator_method(tree, "_checkpoint_member_paths")),
    copy.deepcopy(_generator_method(tree, "__load_chk")),
  ]
  extracted = ast.ClassDef(
    name="GeneratorCore",
    bases=[],
    keywords=[],
    body=methods,
    decorator_list=[])
  module = ast.Module(body=[extracted], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "fixed_facts": SimpleNamespace(SIDECAR_SUFFIX=".facts.yaml"),
    "np": _ForbiddenNumpy(),
    "os": os,
    "require_checkpoint_members": member_preflight,
    "resolve_parameter_table": resolver,
  }
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["GeneratorCore"]


def _parameter_members(paramsf):
  """Return the exact five parameter members owned by chain-only mode."""
  return [
    paramsf + ".1.txt",
    paramsf + ".paramnames",
    paramsf + ".covmat",
    paramsf + ".ranges",
    paramsf + ".facts.yaml",
  ]


def _install_member_census(instance, directory, dataset_mode):
  """Give an extracted generator the canonical saved member census."""
  census = build_dataset_member_census(
    dataset_mode=dataset_mode,
    family="cosmolike",
    family_variant="standard",
    generator="dataset_generator_lensing",
    probe="cs",
    params_stem=Path(instance.paramsf).name,
    dvs_stem=Path(instance.dvsf).name,
    fail_stem=Path(instance.failf).name)
  instance.dataset_member_directory = Path(directory)
  instance.dataset_route = census.route
  instance.dataset_members = census.members
  return tuple(
    instance.dataset_member_directory / relative
    for relative in census.members.values())


def _checkpoint_census_contract(tree):
  """Require exact five-member chain-only and complete full censuses."""
  def unused_resolver(*args, **kwargs):
    del args, kwargs
    raise AssertionError("member census unexpectedly resolved a table")

  def unused_preflight(*args, **kwargs):
    del args, kwargs
    raise AssertionError("member census unexpectedly ran a preflight")

  checkpoint_class = _checkpoint_class(
    tree=tree,
    resolver=unused_resolver,
    member_preflight=unused_preflight)
  dv_calls = []

  def dv_files(instance):
    del instance
    dv_calls.append("dv-census")
    return ["/repo/chains/dv_a.npy", "/repo/chains/dv_axis.npy"]

  setattr(checkpoint_class, "_dv_chk_files", dv_files)
  instance = checkpoint_class()
  instance.paramsf = "/repo/chains/params"
  instance.failf = "/repo/chains/fail"
  instance.dvsf = "/repo/chains/dv"
  instance.run_control = RunControl(
    loadchk=1, append=0, chain=1,
    operation="resume", dataset_mode="chain-only")
  expected_parameters = _install_member_census(
    instance, "/repo/chains", "chain-only")
  chain_members = instance._checkpoint_member_paths()
  if tuple(chain_members) != expected_parameters:
    return False, "chain-only census=" + repr(chain_members)
  if dv_calls:
    return False, "chain-only census touched " + repr(dv_calls)

  instance.run_control = RunControl(
    loadchk=1, append=0, chain=0,
    operation="resume", dataset_mode="full")
  expected_full = _install_member_census(instance, "/repo/chains", "full")
  full_members = instance._checkpoint_member_paths()
  if tuple(full_members) != expected_full:
    return False, "full census=" + repr(full_members)
  if dv_calls:
    return False, "full census DV calls=" + repr(dv_calls)

  try:
    _install_member_census(instance, "/repo/chains", "unknown")
  except ValueError as exc:
    if "unknown" not in str(exc):
      return False, "unknown-mode refusal did not name the value"
  else:
    return False, "unknown normalized mode was accepted"
  return True, ("chain-only=5 parameter members; full=bound payload + fail "
                "+ 5; old DV census unused")


def _drive_chain_only_loader(tree, operation):
  """Run the exact chain-only loader with absent DV/failure sentinels."""
  events = []
  required_calls = []
  resolver_calls = []
  sample_marker = object()

  def resolver(*args, **kwargs):
    resolver_calls.append((args, dict(kwargs)))
    names = tuple(kwargs["input_names"])
    declarations = tuple(
      (name, False, 2 + index)
      for index, name in enumerate(names)
    ) + (("chi2", True, 2 + len(names)),)
    params_path = os.fspath(kwargs["params_path"])
    if not params_path.endswith(".1.txt"):
      raise AssertionError("unexpected parameter path " + params_path)
    return SimpleNamespace(
      inputs=sample_marker,
      declarations=declarations,
      sidecar_path=params_path[:-len(".1.txt")] + ".paramnames")

  def member_preflight(operation, members, is_file):
    required_calls.append((operation, tuple(members)))
    return require_checkpoint_members(operation, members, is_file)

  checkpoint_class = _checkpoint_class(
    tree=tree,
    resolver=resolver,
    member_preflight=member_preflight)

  def forbidden_dv_census(instance):
    del instance
    events.append("dv-census")
    raise AssertionError("chain-only loader requested DV members")

  def forbidden_dv_load(instance):
    del instance
    events.append("dv-load")
    raise AssertionError("chain-only loader opened DV members")

  setattr(checkpoint_class, "_dv_chk_files", forbidden_dv_census)
  setattr(checkpoint_class, "_dv_load_chk", forbidden_dv_load)

  with tempfile.TemporaryDirectory() as directory:
    paramsf = os.path.join(directory, "params_chain_only")
    for member in _parameter_members(paramsf):
      Path(member).write_text("sentinel\n", encoding="utf-8")

    instance = checkpoint_class()
    instance.paramsf = paramsf
    instance.failf = os.path.join(directory, "absent-fail")
    instance.dvsf = os.path.join(directory, "absent-dv")
    instance.sampled_params = ("alpha", "beta")
    instance.loadedsamples = False
    instance.loadedfromchk = False
    instance.run_control = RunControl(
      loadchk=1,
      append=1 if operation == "append" else 0,
      chain=1,
      operation=operation,
      dataset_mode="chain-only")
    expected_members = _install_member_census(
      instance, directory, "chain-only")
    error = None
    result = None
    try:
      result = instance._GeneratorCore__load_chk()
    except Exception as exc:
      error = exc

    expected_required = [(operation, expected_members)]
    if error is not None:
      return False, operation + " error=" + repr(error)
    if result is not True:
      return False, operation + " result=" + repr(result)
    if events:
      return False, operation + " touched events=" + repr(events)
    if required_calls != expected_required:
      return False, operation + " preflight=" + repr(required_calls)
    if len(resolver_calls) != 1 or resolver_calls[0][0]:
      return False, operation + " resolver calls=" + repr(resolver_calls)
    kwargs = resolver_calls[0][1]
    if (kwargs.get("params_path") != paramsf + ".1.txt"
        or tuple(kwargs.get("input_names", ())) != ("alpha", "beta")):
      return False, operation + " resolver kwargs=" + repr(kwargs)
    if instance.samples is not sample_marker:
      return False, operation + " did not retain resolved samples"
    expected_flags = (operation == "resume", operation == "resume")
    observed_flags = (instance.loadedsamples, instance.loadedfromchk)
    if observed_flags != expected_flags:
      return False, operation + " loaded flags=" + repr(observed_flags)
  return True, operation + " returned before fail/DV members"


def _append_callsite_structure(tree):
  """Require append to use the mode barrier instead of direct full-store I/O."""
  method = _generator_method(tree, "__run_mcmc")
  helper_calls = [
    node for node in ast.walk(method)
    if isinstance(node, ast.Call)
    and _attribute_chain(node.func) == "self._append_full_checkpoint_rows"
  ]
  if len(helper_calls) != 1:
    return False, "append-helper call count=" + repr(len(helper_calls))
  call = helper_calls[0]
  if (call.keywords or len(call.args) != 1
      or not isinstance(call.args[0], ast.Name)
      or call.args[0].id != "nparams"):
    return False, "append-helper call does not consume nparams directly"

  fresh_branches = [
    node for node in ast.walk(method)
    if isinstance(node, ast.If)
    and _is_operation_compare(node.test, "fresh")
  ]
  if len(fresh_branches) != 1:
    return False, "fresh/append branch count=" + repr(len(fresh_branches))
  fresh_branch = fresh_branches[0]

  def helper_calls_in(statements):
    return [
      node
      for statement in statements
      for node in ast.walk(statement)
      if isinstance(node, ast.Call)
      and _attribute_chain(node.func) == "self._append_full_checkpoint_rows"
    ]

  fresh_calls = helper_calls_in(fresh_branch.body)
  append_calls = helper_calls_in(fresh_branch.orelse)
  if fresh_calls or len(append_calls) != 1:
    return False, ("append-helper branch placement fresh="
                   + repr(len(fresh_calls)) + ", append="
                   + repr(len(append_calls)))

  direct_dv_calls = [
    node for node in ast.walk(method)
    if isinstance(node, ast.Call)
    and _attribute_chain(node.func) == "self._dv_append"
  ]
  if direct_dv_calls:
    return False, "__run_mcmc directly appends data vectors"
  direct_failure_paths = [
    node for node in ast.walk(method)
    if isinstance(node, ast.Attribute)
    and _attribute_chain(node) == "self.failf"
  ]
  if direct_failure_paths:
    return False, "__run_mcmc directly accesses the failure-mask stem"
  return True, ("append branch uses one mode barrier and no direct full I/O")


def _drive_chain_only_append_helper(tree):
  """Execute the real append helper with every full-store boundary forbidden."""
  method = copy.deepcopy(
    _generator_method(tree, "_append_full_checkpoint_rows"))
  extracted = ast.ClassDef(
    name="GeneratorCore",
    bases=[],
    keywords=[],
    body=[method],
    decorator_list=[])
  module = ast.Module(body=[extracted], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {"np": _ForbiddenNumpy()}
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  helper_class = namespace["GeneratorCore"]

  events = []
  with tempfile.TemporaryDirectory() as directory:
    failure_path = Path(directory) / "must-not-exist.txt"
    instance = helper_class()
    instance.run_control = RunControl(
      loadchk=1, append=1, chain=1,
      operation="append", dataset_mode="chain-only")
    instance.failf = str(failure_path.with_suffix(""))
    instance._dv_append = lambda count: events.append(("dv-append", count))
    error = None
    try:
      instance._append_full_checkpoint_rows(7)
    except Exception as exc:
      error = exc
    if error is not None:
      return False, "chain-only append helper error=" + repr(error)
    if events:
      return False, "chain-only append helper events=" + repr(events)
    if failure_path.exists():
      return False, "chain-only append helper created the failure mask"

  instance = helper_class()
  instance.run_control = SimpleNamespace(dataset_mode="unknown")
  try:
    instance._append_full_checkpoint_rows(1)
  except ValueError as exc:
    if "unknown" not in str(exc):
      return False, "append-helper unknown-mode refusal hid the value"
  else:
    return False, "append helper accepted an unknown normalized mode"
  return True, "append helper returned before failure/DV I/O"


def _dataset_mode_isolation_contract(tree, scoper=scope_dataset_stem):
  """Check disjoint stems, exact censuses, and early chain-only returns."""
  structure_ok, structure_detail = _stem_scope_structure(tree)
  if not structure_ok:
    return False, structure_detail
  behavior_ok, behavior_detail = _stem_scope_behavior(scoper)
  if not behavior_ok:
    return False, behavior_detail
  census_ok, census_detail = _checkpoint_census_contract(tree)
  if not census_ok:
    return False, census_detail
  append_ok, append_detail = _drive_chain_only_loader(tree, "append")
  if not append_ok:
    return False, append_detail
  resume_ok, resume_detail = _drive_chain_only_loader(tree, "resume")
  if not resume_ok:
    return False, resume_detail
  callsite_ok, callsite_detail = _append_callsite_structure(tree)
  if not callsite_ok:
    return False, callsite_detail
  helper_ok, helper_detail = _drive_chain_only_append_helper(tree)
  if not helper_ok:
    return False, helper_detail
  return True, (behavior_detail + "; " + census_detail + "; "
                + append_detail + "; " + resume_detail + "; "
                + callsite_detail + "; " + helper_detail + "; "
                + structure_detail)


def _unscoped_dataset_stem(stem, dataset_mode):
  """Mutation: let full and chain-only runs collide at one stem."""
  if dataset_mode not in ("full", "chain-only"):
    raise ValueError("unknown mode")
  return stem


def _mutation_drop_fail_scope(tree):
  """Mutation: leave the failure stem shared with full datasets."""
  setup = _generator_method(tree, "__setup_flags")
  changes = 0
  for node in ast.walk(setup):
    if (not isinstance(node, ast.Assign)
        or len(node.targets) != 1
        or _attribute_chain(node.targets[0]) != "self.failf"):
      continue
    call = node.value
    if (isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "scope_dataset_stem"):
      node.value = ast.Attribute(
        value=ast.Name(id="self", ctx=ast.Load()),
        attr="failf",
        ctx=ast.Load())
      changes += 1
  if changes != 1:
    raise AssertionError("fail-scope mutation needs exactly one assignment")


def _mutation_scope_on_raw_chain(tree):
  """Mutation: scope one stem from raw CLI state instead of normalized mode."""
  setup = _generator_method(tree, "__setup_flags")
  changes = 0
  for node in ast.walk(setup):
    if (not isinstance(node, ast.Assign)
        or len(node.targets) != 1
        or _attribute_chain(node.targets[0]) != "self.paramsf"):
      continue
    call = node.value
    if (isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "scope_dataset_stem"):
      call.args[1] = ast.Attribute(
        value=ast.Attribute(
          value=ast.Name(id="self", ctx=ast.Load()),
          attr="args",
          ctx=ast.Load()),
        attr="chain",
        ctx=ast.Load())
      changes += 1
  if changes != 1:
    raise AssertionError("raw-scope mutation needs exactly one assignment")


def _mutation_chain_census_borrows_dv(tree):
  """Mutation: make chain-only preflight inspect full-dataset DV members."""
  method = _generator_method(tree, "_checkpoint_member_paths")
  old_census_call = ast.Expr(value=ast.Call(
    func=ast.Attribute(
      value=ast.Name(id="self", ctx=ast.Load()),
      attr="_dv_chk_files",
      ctx=ast.Load()),
    args=[],
    keywords=[]))
  method.body.insert(1, old_census_call)


def _mutation_full_census_is_parameter_only(tree):
  """Mutation: drop the full dataset's payload and failure members."""
  method = _generator_method(tree, "_checkpoint_member_paths")
  loops = [node for node in method.body if isinstance(node, ast.For)]
  if len(loops) != 1:
    raise AssertionError("full census mutation needs one member loop")
  loops[0].iter = ast.Subscript(
    value=ast.Call(
      func=ast.Name(id="tuple", ctx=ast.Load()),
      args=[loops[0].iter],
      keywords=[]),
    slice=ast.Slice(
      lower=None,
      upper=ast.Constant(value=5),
      step=None),
    ctx=ast.Load())


def _mutation_remove_chain_early_return(tree):
  """Mutation: let chain-only loading fall through into fail/DV parsing."""
  method = _generator_method(tree, "__load_chk")
  changes = 0
  for node in method.body:
    if not isinstance(node, ast.If) or not _is_mode_compare(
        node.test, "chain-only"):
      continue
    for index, statement in enumerate(node.body):
      if (isinstance(statement, ast.Return)
          and isinstance(statement.value, ast.Constant)
          and statement.value.value is True):
        node.body[index] = ast.copy_location(ast.Pass(), statement)
        changes += 1
  if changes != 1:
    raise AssertionError("early-return mutation needs one return True")


def _mutation_bypass_append_helper(tree):
  """Mutation: restore direct data-vector append in the sampling method."""
  method = _generator_method(tree, "__run_mcmc")
  changes = 0
  for node in ast.walk(method):
    if (isinstance(node, ast.Call)
        and _attribute_chain(node.func) ==
            "self._append_full_checkpoint_rows"):
      node.func.attr = "_dv_append"
      changes += 1
  if changes != 1:
    raise AssertionError("append-callsite mutation needs one helper call")


def _mutation_remove_append_barrier_return(tree):
  """Mutation: let chain-only append enter failure/data-vector mutation."""
  method = _generator_method(tree, "_append_full_checkpoint_rows")
  changes = 0
  for node in method.body:
    if not isinstance(node, ast.If) or not _is_mode_compare(
        node.test, "chain-only"):
      continue
    for index, statement in enumerate(node.body):
      if (isinstance(statement, ast.Return)
          and statement.value is None):
        node.body[index] = ast.copy_location(ast.Pass(), statement)
        changes += 1
  if changes != 1:
    raise AssertionError("append-barrier mutation needs one bare return")


def _mutation_move_append_helper_to_fresh(tree):
  """Mutation: execute the helper for fresh generation, not append."""
  method = _generator_method(tree, "__run_mcmc")
  branches = [
    node for node in ast.walk(method)
    if isinstance(node, ast.If)
    and _is_operation_compare(node.test, "fresh")
  ]
  if len(branches) != 1:
    raise AssertionError("helper-placement mutation needs one fresh branch")
  branch = branches[0]
  indices = [
    index for index, statement in enumerate(branch.orelse)
    if any(
      isinstance(node, ast.Call)
      and _attribute_chain(node.func) == "self._append_full_checkpoint_rows"
      for node in ast.walk(statement))
  ]
  if len(indices) != 1:
    raise AssertionError("helper-placement mutation needs one append statement")
  statement = branch.orelse.pop(indices[0])
  branch.body.append(statement)


def _dataset_mode_mutations_red():
  """Require each isolated stem/census/return regression to turn red."""
  results = []
  behavior_ok, _ = _dataset_mode_isolation_contract(
    _production_tree(), scoper=_unscoped_dataset_stem)
  results.append(behavior_ok)
  for mutation in (
      _mutation_drop_fail_scope,
      _mutation_scope_on_raw_chain,
      _mutation_chain_census_borrows_dv,
      _mutation_full_census_is_parameter_only,
      _mutation_remove_chain_early_return,
      _mutation_bypass_append_helper,
      _mutation_remove_append_barrier_return,
      _mutation_move_append_helper_to_fresh):
    tree = copy.deepcopy(_production_tree())
    mutation(tree)
    mutation_ok, _ = _dataset_mode_isolation_contract(tree)
    results.append(mutation_ok)
  return len(results) == 9 and not any(results)


def _coerce_bool(loadchk, append, chain):
  """Mutation: restore truthy/bool-to-integer flag coercion."""
  values = [loadchk, append, chain]
  values = [int(value) if isinstance(value, bool) else value
            for value in values]
  return validate_run_control(*values)


def _allow_append_without_load(loadchk, append, chain):
  """Mutation: restore the destructive independent-flag behavior."""
  if loadchk == 0 and append == 1:
    return RunControl(
      loadchk=0,
      append=1,
      chain=chain,
      operation="append",
      dataset_mode="chain-only" if chain == 1 else "full")
  return validate_run_control(loadchk, append, chain)


def _aid(aid, normal, mutation_red, detail):
  """Print one exact child evidence terminal and human-readable result."""
  passed = normal and mutation_red
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + aid + "  (" + detail
        + "; mutation-red=" + str(mutation_red) + ")")
  print("##AID " + aid + " " + mark)
  return passed


def main():
  """Run the four acceptance arms and their targeted in-memory mutations."""
  binary_ok, binary_detail = _binary_state_contract(validate_run_control)
  bool_mutation_ok, _ = _binary_state_contract(_coerce_bool)
  binary = _aid(
    "generator-run-control.binary-state",
    binary_ok,
    not bool_mutation_ok,
    binary_detail)

  append_ok, append_detail = _append_requires_load_contract(
    validate_run_control)
  append_mutation_ok, _ = _append_requires_load_contract(
    _allow_append_without_load)
  append = _aid(
    "generator-run-control.append-requires-load",
    append_ok,
    not append_mutation_ok,
    append_detail)

  ordering_ok, ordering_detail = _pre_mutation_contract(
    validate_run_control)
  reordered_ok, _ = _pre_mutation_contract(
    validate_run_control, setup_first=True)
  raw_mode_ok, _ = _pre_mutation_contract(
    validate_run_control, raw_mode_routing=True)
  ordering_mutations_red = (
    not reordered_ok
    and not raw_mode_ok
    and _structural_mutations_red())
  ordering = _aid(
    "generator-run-control.pre-mutation-refusal",
    ordering_ok,
    ordering_mutations_red,
    ordering_detail)

  mode_ok, mode_detail = _dataset_mode_isolation_contract(
    _production_tree())
  mode = _aid(
    "generator-run-control.dataset-mode-isolation",
    mode_ok,
    _dataset_mode_mutations_red(),
    mode_detail)

  if not all((binary, append, ordering, mode)):
    print("generator-run-control: FAIL")
    return 1
  print("generator-run-control: ALL PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
