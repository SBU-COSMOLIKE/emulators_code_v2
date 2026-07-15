#!/usr/bin/env python3
"""Prove the generator refuses unsafe run controls before path mutation.

The production generator imports MPI, Cobaya, GetDist, and the family physics
stack.  This CPU child therefore imports only its pure run-control module and
executes the real ``GeneratorCore.__init__`` method from the production syntax
tree with tiny setup/MPI stubs.  The setup stub is deliberately a filesystem
mutation sentinel: an invalid state must refuse before that stub can run.

Eight in-memory mutations keep the evidence load-bearing. They coerce bools,
restore append-without-load, move setup or another statement before validation,
hide a preceding expression inside the validator assignment, shadow the imported
validator, route on raw chain mode, and restore raw append/load setup
assignments. Each mutation must make its acceptance arm red.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile

from compute_data_vectors.dataset_manifest import RunControl
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
  """Run the three acceptance arms and their targeted in-memory mutations."""
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

  if not all((binary, append, ordering)):
    print("generator-run-control: FAIL")
    return 1
  print("generator-run-control: ALL PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
