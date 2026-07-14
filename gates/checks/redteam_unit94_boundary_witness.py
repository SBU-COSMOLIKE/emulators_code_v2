#!/usr/bin/env python3
"""Check that uniform sampling uses a valid representable interval interior.

This executable check reads the production source without importing its heavy
sampling dependencies.  It executes the real boundary helper and the real
uniform branch after extracting both with Python's syntax-tree tools.  The
recording random-number generator shows whether an invalid interval can reach
sampling.

Each command-line mutation changes only an extracted in-memory copy.  These
mutations demonstrate that the endpoint rule, both validation layers, and the
call order are necessary for the check to pass.
"""

import argparse
import ast
import copy
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_CORE = ROOT / "compute_data_vectors" / "generator_core.py"
POLICY_NAME = "nextafter-toward-interval-interior-v1"
MUTATIONS = (
  "endpoint-times-constant",
  "request-validation-bypass",
  "resolved-validation-bypass",
  "sampling-before-resolution",
)

FAILURES = []


def report(label, ok, detail=""):
  """Print one result and retain a failed acceptance arm.

  Arguments:
    label = short plain-language name for the acceptance arm.
    ok = true when the acceptance arm passed.
    detail = compact measured result shown beside the verdict.

  Returns:
    None.
  """
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    # WARNING: reads module global FAILURES.
    FAILURES.append(label)


def parse_generator_core():
  """Parse the production generator source.

  Arguments:
    None.

  Returns:
    The syntax tree for ``compute_data_vectors/generator_core.py``.
  """
  source = GENERATOR_CORE.read_text(encoding="utf-8")
  return ast.parse(
    source,
    filename=str(GENERATOR_CORE))


def top_level_function(tree, name):
  """Return one uniquely named top-level production function.

  Arguments:
    tree = parsed production module.
    name = function name to find.

  Returns:
    The matching ``ast.FunctionDef`` node.
  """
  matches = []
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == name:
      matches.append(node)
  if len(matches) != 1:
    raise AssertionError(
      "Expected one production function named " + name + "; found "
      + str(len(matches)) + ".")
  return matches[0]


def generator_method(tree, name):
  """Return one uniquely named method from ``GeneratorCore``.

  Arguments:
    tree = parsed production module.
    name = method name to find.

  Returns:
    The matching ``ast.FunctionDef`` node.
  """
  classes = []
  for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "GeneratorCore":
      classes.append(node)
  if len(classes) != 1:
    raise AssertionError(
      "Expected one GeneratorCore class; found " + str(len(classes)) + ".")

  methods = []
  for node in classes[0].body:
    if isinstance(node, ast.FunctionDef) and node.name == name:
      methods.append(node)
  if len(methods) != 1:
    raise AssertionError(
      "Expected one GeneratorCore method named " + name + "; found "
      + str(len(methods)) + ".")
  return methods[0]


def call_name(node):
  """Return the final name used by a call expression.

  Arguments:
    node = candidate syntax-tree node.

  Returns:
    The called name, or ``None`` when the node is not a named call.
  """
  if not isinstance(node, ast.Call):
    return None
  if isinstance(node.func, ast.Name):
    return node.func.id
  if isinstance(node.func, ast.Attribute):
    return node.func.attr
  return None


def is_rng_uniform_call(node):
  """Tell whether one node is ``self.rng.uniform(...)``.

  Arguments:
    node = candidate syntax-tree node.

  Returns:
    True only for the production uniform-sampler call shape.
  """
  if not isinstance(node, ast.Call):
    return False
  if not isinstance(node.func, ast.Attribute):
    return False
  if node.func.attr != "uniform":
    return False
  rng = node.func.value
  if not isinstance(rng, ast.Attribute) or rng.attr != "rng":
    return False
  return isinstance(rng.value, ast.Name) and rng.value.id == "self"


def contains_named_call(node, name):
  """Tell whether a node contains a call with one final name.

  Arguments:
    node = syntax-tree node to inspect.
    name = final call name to match.

  Returns:
    True when the named call occurs under the node.
  """
  for child in ast.walk(node):
    if call_name(child) == name:
      return True
  return False


def contains_rng_uniform(node):
  """Tell whether a node contains the production sampler call.

  Arguments:
    node = syntax-tree node to inspect.

  Returns:
    True when ``self.rng.uniform`` occurs under the node.
  """
  for child in ast.walk(node):
    if is_rng_uniform_call(child):
      return True
  return False


def constant_assignment(tree, name):
  """Return one uniquely named top-level constant assignment.

  Arguments:
    tree = parsed production module.
    name = assigned constant name to find.

  Returns:
    The matching ``ast.Assign`` node.
  """
  matches = []
  for node in tree.body:
    if not isinstance(node, ast.Assign):
      continue
    for target in node.targets:
      if isinstance(target, ast.Name) and target.id == name:
        matches.append(node)
  if len(matches) != 1:
    raise AssertionError(
      "Expected one production assignment for " + name + "; found "
      + str(len(matches)) + ".")
  return matches[0]


def node_text(node):
  """Join string literals below one syntax-tree node.

  Arguments:
    node = syntax-tree node whose diagnostic text is needed.

  Returns:
    One space-separated string containing its literal text.
  """
  parts = []
  for child in ast.walk(node):
    if isinstance(child, ast.Constant) and isinstance(child.value, str):
      parts.append(child.value)
  return " ".join(parts)


class ValidationBypass(ast.NodeTransformer):
  """Remove one named validation layer from an extracted helper copy."""

  def __init__(self, phrase):
    """Configure the diagnostic phrase that identifies the layer.

    Arguments:
      phrase = text present in the layer's ``ValueError`` diagnostics.

    Returns:
      None.
    """
    super().__init__()
    self.phrase = phrase
    self.changes = 0

  def visit_If(self, node):
    """Replace a matching raising guard with a no-op statement.

    Arguments:
      node = candidate ``ast.If`` node.

    Returns:
      The original node or an ``ast.Pass`` replacement.
    """
    node = self.generic_visit(node)
    has_raise = False
    for child in node.body:
      for descendant in ast.walk(child):
        if isinstance(descendant, ast.Raise):
          has_raise = True
    if has_raise and self.phrase in node_text(node):
      self.changes += 1
      replacement = ast.Pass()
      return ast.copy_location(replacement, node)
    return node


class EndpointTimesConstant(ast.NodeTransformer):
  """Restore the retired absolute-endpoint shrink rule in memory."""

  def __init__(self):
    """Start with no transformed ``nextafter`` calls.

    Arguments:
      None.

    Returns:
      None.
    """
    super().__init__()
    self.changes = 0

  def visit_Call(self, node):
    """Replace the two production ``nextafter`` calls in source order.

    Arguments:
      node = candidate call expression.

    Returns:
      The original call or the retired endpoint-times-constant expression.
    """
    node = self.generic_visit(node)
    if call_name(node) != "nextafter":
      return node
    if len(node.args) != 2:
      return node
    if not isinstance(node.func, ast.Attribute):
      return node
    if not isinstance(node.func.value, ast.Name):
      return node
    if node.func.value.id != "np":
      return node

    self.changes += 1
    coordinate = copy.deepcopy(node.args[0])
    if self.changes == 1:
      positive_factor = 1.0001
      negative_factor = 0.9999
    elif self.changes == 2:
      positive_factor = 0.9999
      negative_factor = 1.0001
    else:
      raise AssertionError(
        "The endpoint mutation found more than two nextafter calls.")

    test = ast.Compare(
      left=copy.deepcopy(coordinate),
      ops=[ast.Gt()],
      comparators=[ast.Constant(value=0.0)])
    positive = ast.BinOp(
      left=ast.Constant(value=positive_factor),
      op=ast.Mult(),
      right=copy.deepcopy(coordinate))
    negative = ast.BinOp(
      left=ast.Constant(value=negative_factor),
      op=ast.Mult(),
      right=copy.deepcopy(coordinate))
    replacement = ast.IfExp(
      test=test,
      body=positive,
      orelse=negative)
    return ast.copy_location(replacement, node)


def mutate_helper(helper, mutation):
  """Apply one command-line mutation to an extracted helper copy.

  Arguments:
    helper = copied production helper syntax tree.
    mutation = selected mutation name, or ``None``.

  Returns:
    The possibly transformed helper syntax tree.
  """
  if mutation == "endpoint-times-constant":
    transformer = EndpointTimesConstant()
    helper = transformer.visit(helper)
    if transformer.changes != 2:
      raise AssertionError(
        "The endpoint mutation needs exactly two nextafter calls; found "
        + str(transformer.changes) + ".")
  elif mutation == "request-validation-bypass":
    transformer = ValidationBypass(phrase="requested")
    helper = transformer.visit(helper)
    if transformer.changes != 3:
      raise AssertionError(
        "The requested-interval mutation needs three guards; found "
        + str(transformer.changes) + ".")
  elif mutation == "resolved-validation-bypass":
    finite_transformer = ValidationBypass(
      phrase="resolved boundary interior")
    helper = finite_transformer.visit(helper)
    order_transformer = ValidationBypass(
      phrase="representable interior")
    helper = order_transformer.visit(helper)
    changes = finite_transformer.changes + order_transformer.changes
    if changes != 2:
      raise AssertionError(
        "The resolved-interior mutation needs two guards; found "
        + str(changes) + ".")
  return helper


def load_helper(tree, mutation):
  """Compile the production helper and policy in a NumPy-only namespace.

  Arguments:
    tree = parsed production module.
    mutation = selected mutation name, or ``None``.

  Returns:
    A pair containing the executable helper and its production policy name.
  """
  policy_node = constant_assignment(
    tree=tree,
    name="UNIFORM_BOUNDARY_INTERIOR_POLICY")
  helper_node = top_level_function(
    tree=tree,
    name="resolve_uniform_sampling_support")
  helper_copy = copy.deepcopy(helper_node)
  helper_copy = mutate_helper(
    helper=helper_copy,
    mutation=mutation)
  module = ast.Module(
    body=[copy.deepcopy(policy_node), helper_copy],
    type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {"np": np}
  exec(
    compile(
      module,
      str(GENERATOR_CORE),
      "exec"),
    namespace)
  helper = namespace["resolve_uniform_sampling_support"]
  policy = namespace["UNIFORM_BOUNDARY_INTERIOR_POLICY"]
  return helper, policy


def production_uniform_statements(tree):
  """Extract the real uniform branch through its sampling call.

  Arguments:
    tree = parsed production module.

  Returns:
    Copied statements from the start of the uniform branch through sampling.
  """
  method = generator_method(
    tree=tree,
    name="__run_mcmc")
  candidates = []
  for node in ast.walk(method):
    if not isinstance(node, ast.If):
      continue
    found = False
    for statement in node.orelse:
      if contains_rng_uniform(statement):
        found = True
    if found:
      candidates.append(node)
  if len(candidates) != 1:
    raise AssertionError(
      "Expected one production uniform branch; found "
      + str(len(candidates)) + ".")

  statements = []
  found_sample = False
  for statement in candidates[0].orelse:
    statements.append(copy.deepcopy(statement))
    if contains_rng_uniform(statement):
      found_sample = True
      break
  if not found_sample:
    raise AssertionError("The production uniform branch has no sampling call.")
  return statements


def statement_index_with_call(statements, name):
  """Return the unique statement index containing one named call.

  Arguments:
    statements = ordered syntax-tree statements.
    name = final call name to find.

  Returns:
    The zero-based index of the matching statement.
  """
  matches = []
  for index in range(len(statements)):
    if contains_named_call(statements[index], name):
      matches.append(index)
  if len(matches) != 1:
    raise AssertionError(
      "Expected one statement calling " + name + "; found "
      + str(len(matches)) + ".")
  return matches[0]


def statement_index_with_sampler(statements):
  """Return the unique statement index containing the sampler call.

  Arguments:
    statements = ordered syntax-tree statements.

  Returns:
    The zero-based index of ``self.rng.uniform``.
  """
  matches = []
  for index in range(len(statements)):
    if contains_rng_uniform(statements[index]):
      matches.append(index)
  if len(matches) != 1:
    raise AssertionError(
      "Expected one uniform-sampler statement; found "
      + str(len(matches)) + ".")
  return matches[0]


def statement_assigns_attribute(statement, attribute):
  """Tell whether a statement assigns one attribute on ``self``.

  Arguments:
    statement = syntax-tree statement to inspect.
    attribute = attribute name to match.

  Returns:
    True when the statement assigns ``self.<attribute>``.
  """
  for node in ast.walk(statement):
    if not isinstance(node, ast.Assign):
      continue
    for target in node.targets:
      if not isinstance(target, ast.Attribute):
        continue
      if not isinstance(target.value, ast.Name):
        continue
      if target.value.id == "self" and target.attr == attribute:
        return True
  return False


def statement_index_with_assignment(statements, attribute):
  """Return the unique statement index assigning one ``self`` attribute.

  Arguments:
    statements = ordered syntax-tree statements.
    attribute = attribute name to match.

  Returns:
    The zero-based index of the matching assignment.
  """
  matches = []
  for index in range(len(statements)):
    if statement_assigns_attribute(statements[index], attribute):
      matches.append(index)
  if len(matches) != 1:
    raise AssertionError(
      "Expected one assignment to self." + attribute + "; found "
      + str(len(matches)) + ".")
  return matches[0]


def mutate_uniform_statements(statements, mutation):
  """Apply the sampling-order mutation to an extracted branch copy.

  Arguments:
    statements = copied production uniform-branch statements.
    mutation = selected mutation name, or ``None``.

  Returns:
    The possibly reordered statement list.
  """
  if mutation != "sampling-before-resolution":
    return statements

  resolve_index = statement_index_with_call(
    statements=statements,
    name="resolve_uniform_sampling_support")
  sample_index = statement_index_with_sampler(statements=statements)
  sample_statement = statements.pop(sample_index)
  if sample_index < resolve_index:
    resolve_index -= 1
  statements.insert(resolve_index, sample_statement)
  return statements


def load_uniform_runner(tree, helper, mutation):
  """Compile the extracted uniform branch with the extracted helper.

  Arguments:
    tree = parsed production module.
    helper = executable extracted production helper.
    mutation = selected mutation name, or ``None``.

  Returns:
    A pair containing the executable branch and its statement syntax trees.
  """
  statements = production_uniform_statements(tree=tree)
  statements = mutate_uniform_statements(
    statements=statements,
    mutation=mutation)
  arguments = ast.arguments(
    posonlyargs=[],
    args=[
      ast.arg(arg="self"),
      ast.arg(arg="names"),
      ast.arg(arg="ndim"),
    ],
    vararg=None,
    kwonlyargs=[],
    kw_defaults=[],
    kwarg=None,
    defaults=[])
  function = ast.FunctionDef(
    name="run_extracted_uniform_branch",
    args=arguments,
    body=copy.deepcopy(statements),
    decorator_list=[])
  module = ast.Module(
    body=[function],
    type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "np": np,
    "resolve_uniform_sampling_support": helper,
  }
  exec(
    compile(
      module,
      str(GENERATOR_CORE),
      "exec"),
    namespace)
  runner = namespace["run_extracted_uniform_branch"]
  return runner, statements


def expected_interior(bounds):
  """Calculate the independent one-step interior for test bounds.

  Arguments:
    bounds = floating array with shape (N, 2).

  Returns:
    An array holding the two independently calculated interior endpoints.
  """
  expected = np.empty_like(bounds)
  for index in range(bounds.shape[0]):
    low = bounds[index, 0]
    high = bounds[index, 1]
    expected[index, 0] = np.nextafter(low, high)
    expected[index, 1] = np.nextafter(high, low)
  return expected


def check_policy_and_mapping(resolve, policy):
  """Check the named policy and exact requested/resolved support mappings.

  Arguments:
    resolve = executable extracted production helper.
    policy = extracted production policy constant.

  Returns:
    None.
  """
  names = ["H0", "negative_control"]
  bounds = np.array(
    [
      [70.0, 70.02],
      [-2.0, -1.0],
    ],
    dtype=np.float64)
  expected = expected_interior(bounds=bounds)
  support = resolve(
    names=names,
    bounds=bounds)
  expected_requested = {
    "H0": (70.0, 70.02),
    "negative_control": (-2.0, -1.0),
  }
  expected_resolved = {
    "H0": (float(expected[0, 0]), float(expected[0, 1])),
    "negative_control": (
      float(expected[1, 0]),
      float(expected[1, 1])),
  }
  exact_keys = ["policy", "requested", "resolved", "bounds"]
  ok = policy == POLICY_NAME
  ok = ok and list(support.keys()) == exact_keys
  ok = ok and support["policy"] == POLICY_NAME
  ok = ok and support["requested"] == expected_requested
  ok = ok and support["resolved"] == expected_resolved
  ok = ok and np.array_equal(support["bounds"], expected)
  report(
    "policy and per-name support are exact",
    ok,
    "policy=" + repr(support["policy"]) + "; keys="
    + repr(list(support.keys())))


def check_minting_intervals(resolve):
  """Check both narrow positive intervals from the original defect report.

  Arguments:
    resolve = executable extracted production helper.

  Returns:
    None.
  """
  h0_bounds = np.array(
    [[70.0, 70.02]],
    dtype=np.float32)
  expected = expected_interior(bounds=h0_bounds)
  support = resolve(
    names=["H0"],
    bounds=h0_bounds)
  resolved = support["bounds"]
  requested_width = h0_bounds[0, 1] - h0_bounds[0, 0]
  resolved_width = resolved[0, 1] - resolved[0, 0]
  retained = resolved_width / requested_width
  ok = np.array_equal(resolved, expected)
  ok = ok and retained > 0.99
  report(
    "the H0 interval keeps its one-step interior",
    ok,
    "retained fraction=" + repr(float(retained)))

  offset_bounds = np.array(
    [[1000.0, 1000.01]],
    dtype=np.float32)
  expected = expected_interior(bounds=offset_bounds)
  try:
    support = resolve(
      names=["offset_control"],
      bounds=offset_bounds)
    resolved = support["bounds"]
    ok = np.array_equal(resolved, expected)
    ok = ok and bool(resolved[0, 0] < resolved[0, 1])
    detail = "resolved=" + repr(resolved[0].tolist())
  except ValueError as exc:
    ok = False
    detail = "unexpected refusal=" + str(exc)
  report(
    "the narrow offset interval remains ordered",
    ok,
    detail)


def check_translation_and_negative_mirror(resolve):
  """Check translated equal-width intervals and the negative mirror.

  Arguments:
    resolve = executable extracted production helper.

  Returns:
    None.
  """
  names = ["zero", "positive", "negative"]
  bounds = np.array(
    [
      [0.0, 0.02],
      [70.0, 70.02],
      [-70.02, -70.0],
    ],
    dtype=np.float64)
  support = resolve(
    names=names,
    bounds=bounds)
  resolved = support["bounds"]
  fractions = np.empty(3, dtype=np.float64)
  for index in range(3):
    requested_width = bounds[index, 1] - bounds[index, 0]
    resolved_width = resolved[index, 1] - resolved[index, 0]
    fractions[index] = resolved_width / requested_width
  spread = float(np.max(fractions) - np.min(fractions))
  translated_ok = bool(np.min(fractions) > 0.999999999)
  translated_ok = translated_ok and spread < 1.0e-10
  report(
    "translated intervals retain the same fractional width",
    translated_ok,
    "fractions=" + repr(fractions.tolist()) + "; spread=" + repr(spread))

  positive = resolved[1]
  negative = resolved[2]
  expected_negative = np.array(
    [-positive[1], -positive[0]],
    dtype=np.float64)
  mirror_ok = np.array_equal(negative, expected_negative)
  report(
    "the negative interval is the positive interval's mirror",
    mirror_ok,
    "negative=" + repr(negative.tolist()))


class RecordingRng:
  """Record every attempted uniform sample without validating the bounds."""

  def __init__(self, events):
    """Retain the event list owned by one refusal probe.

    Arguments:
      events = mutable list receiving sampler-call records.

    Returns:
      None.
    """
    self.events = events

  def uniform(self, low, high, size):
    """Record a sampler call and return an inert array.

    Arguments:
      low = lower bounds received from the extracted production branch.
      high = upper bounds received from the extracted production branch.
      size = requested output shape.

    Returns:
      A zero array with the requested shape.
    """
    self.events.append(
      {
        "low": np.array(low, copy=True),
        "high": np.array(high, copy=True),
        "size": tuple(size),
      })
    return np.zeros(
      size,
      dtype=np.float64)


class UniformFixture:
  """Supply only the state read before the production uniform sample."""

  def __init__(self, bounds, events):
    """Construct one extracted-branch fixture.

    Arguments:
      bounds = requested bounds sent to the production helper.
      events = mutable list receiving sampler-call records.

    Returns:
      None.
    """
    self.bounds = np.array(bounds, copy=True)
    self.nparams = 2
    self.rng = RecordingRng(events=events)


def check_one_refusal(runner, label, bounds, expected_text):
  """Check that one bad request refuses before the sampler is called.

  Arguments:
    runner = executable extracted production uniform branch.
    label = short description of the invalid interval.
    bounds = one-row requested-bounds array.
    expected_text = diagnostic phrase naming the responsible validation layer.

  Returns:
    None.
  """
  events = []
  fixture = UniformFixture(
    bounds=bounds,
    events=events)
  refused = False
  message = "no ValueError"
  try:
    runner(
      fixture,
      ["control"],
      1)
  except ValueError as exc:
    refused = True
    message = str(exc)
  ok = refused and len(events) == 0
  ok = ok and expected_text in message
  report(
    label + " refuses before sampling",
    ok,
    "sampler calls=" + str(len(events)) + "; message=" + message)


def check_refusals(runner):
  """Exercise every ordered, finite, and representable refusal class.

  Arguments:
    runner = executable extracted production uniform branch.

  Returns:
    None.
  """
  adjacent_low = np.float32(1.0)
  adjacent_high = np.nextafter(
    adjacent_low,
    np.float32(2.0))
  adjacent = np.array(
    [[adjacent_low, adjacent_high]],
    dtype=np.float32)
  check_one_refusal(
    runner=runner,
    label="a float32-adjacent interval",
    bounds=adjacent,
    expected_text="representable interior")

  equal = np.array(
    [[2.0, 2.0]],
    dtype=np.float64)
  check_one_refusal(
    runner=runner,
    label="an equal-endpoint interval",
    bounds=equal,
    expected_text="requested interval")

  inverted = np.array(
    [[3.0, 2.0]],
    dtype=np.float64)
  check_one_refusal(
    runner=runner,
    label="an inverted interval",
    bounds=inverted,
    expected_text="requested interval")

  nan_lower = np.array(
    [[np.nan, 2.0]],
    dtype=np.float64)
  check_one_refusal(
    runner=runner,
    label="a NaN endpoint",
    bounds=nan_lower,
    expected_text="requested lower endpoint")

  positive_infinity = np.array(
    [[1.0, np.inf]],
    dtype=np.float64)
  check_one_refusal(
    runner=runner,
    label="a positive-infinite endpoint",
    bounds=positive_infinity,
    expected_text="requested upper endpoint")

  negative_infinity = np.array(
    [[-np.inf, -1.0]],
    dtype=np.float64)
  check_one_refusal(
    runner=runner,
    label="a negative-infinite endpoint",
    bounds=negative_infinity,
    expected_text="requested lower endpoint")


def check_uniform_order(statements):
  """Check that both named support assignments dominate sampling.

  Arguments:
    statements = extracted uniform-branch statements in execution order.

  Returns:
    None.
  """
  resolve_index = statement_index_with_call(
    statements=statements,
    name="resolve_uniform_sampling_support")
  support_index = statement_index_with_assignment(
    statements=statements,
    attribute="uniform_sampling_support")
  bounds_index = statement_index_with_assignment(
    statements=statements,
    attribute="bounds")
  sample_index = statement_index_with_sampler(statements=statements)
  ok = resolve_index == support_index
  ok = ok and support_index < bounds_index
  ok = ok and bounds_index < sample_index
  detail = (
    "resolve=" + str(resolve_index)
    + ", support=" + str(support_index)
    + ", bounds=" + str(bounds_index)
    + ", sample=" + str(sample_index))
  report(
    "support resolution and assignment dominate sampling",
    ok,
    detail)


def parse_args():
  """Parse the optional in-memory mutation selector.

  Arguments:
    None.

  Returns:
    Parsed command-line arguments.
  """
  parser = argparse.ArgumentParser(
    description="Check the production uniform boundary-interior rule.")
  parser.add_argument(
    "--mutation",
    choices=MUTATIONS,
    default=None,
    help="Apply one in-memory mutation to demonstrate catch power.")
  return parser.parse_args()


def main():
  """Run the NumPy-only acceptance and mutation checks.

  Arguments:
    None.

  Returns:
    Zero when every acceptance arm passes, otherwise one.
  """
  args = parse_args()
  tree = parse_generator_core()
  resolve, policy = load_helper(
    tree=tree,
    mutation=args.mutation)
  runner, statements = load_uniform_runner(
    tree=tree,
    helper=resolve,
    mutation=args.mutation)

  check_policy_and_mapping(
    resolve=resolve,
    policy=policy)
  check_minting_intervals(resolve=resolve)
  check_translation_and_negative_mirror(resolve=resolve)
  check_uniform_order(statements=statements)
  check_refusals(runner=runner)

  if len(FAILURES) != 0:
    print(
      "uniform-boundary-witness: FAIL ("
      + str(len(FAILURES)) + " failed arms)")
    return 1
  print("uniform-boundary-witness: ALL PASS")
  return 0


if __name__ == "__main__":
  sys.exit(main())
