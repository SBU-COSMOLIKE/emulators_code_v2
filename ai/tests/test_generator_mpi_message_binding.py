"""CPU tests for binding MPI replies to rank-zero row assignments.

The production generator imports Cobaya, MPI, GetDist, emcee, and NumPy. The
protocol validators and shared result consumer tested here need none of those
packages, so this file loads their real syntax-tree definitions directly. The
wiring check then proves that both result loops use the shared consumer and
that every reply is validated before its worker assignment is removed.
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "compute_data_vectors" / "generator_core.py"


def _production_tree():
  return ast.parse(GENERATOR.read_text(encoding="utf-8"), filename=str(GENERATOR))


def _load_protocol_functions():
  tree = _production_tree()
  names = {
    "validate_worker_result_message",
    "validate_worker_done_message",
  }
  definitions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in names
  ]
  if {node.name for node in definitions} != names:
    raise AssertionError("the two production MPI protocol validators are required")
  module = ast.Module(body=definitions, type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {}
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return tuple(namespace[name] for name in sorted(names))


validate_worker_done_message, validate_worker_result_message = (
  _load_protocol_functions())


def _method_node(class_name, method_name):
  classes = [
    node for node in _production_tree().body
    if isinstance(node, ast.ClassDef) and node.name == class_name
  ]
  if len(classes) != 1:
    raise AssertionError("expected one production class " + class_name)
  methods = [
    node for node in classes[0].body
    if isinstance(node, ast.FunctionDef) and node.name == method_name
  ]
  if len(methods) != 1:
    raise AssertionError("expected one production method " + method_name)
  return methods[0]


def _load_consumer_method():
  """Compile the real shared result consumer without heavy dependencies."""
  tree = _production_tree()
  validators = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "validate_worker_result_message"
  ]
  method = _method_node("GeneratorCore", "_consume_worker_result_message")
  selected_class = ast.ClassDef(
    name="SelectedGeneratorCore",
    bases=[],
    keywords=[],
    body=[method],
    decorator_list=[])
  module = ast.Module(body=validators + [selected_class], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {}
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["SelectedGeneratorCore"]._consume_worker_result_message


consume_worker_result_message = _load_consumer_method()


def _statement_lists(node):
  """Yield every direct list of statements nested below one syntax node."""
  for _field, value in ast.iter_fields(node):
    if isinstance(value, list):
      if value and all(isinstance(item, ast.stmt) for item in value):
        yield value
      for item in value:
        if isinstance(item, ast.AST):
          yield from _statement_lists(item)
    elif isinstance(value, ast.AST):
      yield from _statement_lists(value)


def _direct_call(statement):
  value = None
  if isinstance(statement, ast.Assign):
    value = statement.value
  elif isinstance(statement, ast.Expr):
    value = statement.value
  if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
    return value
  return None


def _deletes_active_source(statement):
  if not isinstance(statement, ast.Delete) or len(statement.targets) != 1:
    return False
  target = statement.targets[0]
  return (
    isinstance(target, ast.Subscript)
    and isinstance(target.value, ast.Name)
    and target.value.id == "active"
    and isinstance(target.slice, ast.Name)
    and target.slice.id == "src")


def _deletes_active_assignment(statement):
  if not isinstance(statement, ast.Delete) or len(statement.targets) != 1:
    return False
  target = statement.targets[0]
  return (
    isinstance(target, ast.Subscript)
    and isinstance(target.value, ast.Name)
    and target.value.id == "active"
    and isinstance(target.slice, ast.Name)
    and target.slice.id == "source")


class _ResultStore:

  def __init__(self, accept_error=None):
    self.failed = [True] * 20
    self.accept_error = accept_error
    self.accepted = []
    self.zeroed = []

  def _accept_payload_row(self, index, payload, write_row):
    self.accepted.append((index, payload, write_row))
    if self.accept_error is not None:
      raise self.accept_error
    self.failed[index] = False

  def _dv_zero(self, index):
    self.zeroed.append(index)


class WorkerResultProtocolTests(unittest.TestCase):

  def test_valid_result_uses_the_master_assignment(self):
    payload = object()
    active = {3: (17, 12.5)}
    self.assertEqual(
      validate_worker_result_message(("ok", 17, payload), 3, active),
      ("ok", 17, payload))
    self.assertEqual(active, {3: (17, 12.5)})

  def test_valid_error_keeps_traceback_text(self):
    active = {2: (9, 1.0)}
    self.assertEqual(
      validate_worker_result_message(("err", 9, "traceback"), 2, active),
      ("err", 9, "traceback"))

  def test_unknown_duplicate_or_wrong_row_refuses_before_state_changes(self):
    cases = (
      (("ok", 4, object()), 8, {2: (4, 0.0)}, "without a live"),
      (("ok", 5, object()), 2, {2: (4, 0.0)}, "assigned row 4"),
      (("ok", 4, object()), True, {1: (4, 0.0)}, "positive worker rank"),
    )
    for message, source, active, text in cases:
      with self.subTest(message=message, source=source):
        before = dict(active)
        with self.assertRaisesRegex(RuntimeError, text):
          validate_worker_result_message(message, source, active)
        self.assertEqual(active, before)

  def test_malformed_result_protocol_refuses(self):
    cases = (
      (["ok", 4, object()], "tuple"),
      (("ok", 4), "tuple"),
      (("future", 4, object()), "kind"),
      (("ok", True, object()), "nonnegative native integer"),
      (("ok", -1, object()), "nonnegative native integer"),
      (("err", 4, object()), "traceback text"),
    )
    for message, text in cases:
      with self.subTest(message=message):
        with self.assertRaisesRegex(RuntimeError, text):
          validate_worker_result_message(message, 2, {2: (4, 0.0)})

  def test_malformed_master_assignment_refuses(self):
    for assignment in ((4,), [4, 0.0], (True, 0.0), (-1, 0.0)):
      with self.subTest(assignment=assignment):
        with self.assertRaisesRegex(RuntimeError, "assignment.*malformed"):
          validate_worker_result_message(
            ("ok", 4, object()), 2, {2: assignment})


class WorkerDoneProtocolTests(unittest.TestCase):

  def test_valid_done_acknowledgement_names_its_sender(self):
    active = {4: (0, 3.5)}
    self.assertEqual(
      validate_worker_done_message(("worker done", 4), 4, active), 4)
    self.assertEqual(active, {4: (0, 3.5)})

  def test_stale_or_malformed_done_acknowledgement_refuses(self):
    cases = (
      (("worker done", 3), 3, {4: (0, 0.0)}, "not awaiting"),
      (("worker done", 4), 3, {3: (0, 0.0)}, "must be"),
      (("finished", 3), 3, {3: (0, 0.0)}, "must be"),
      (["worker done", 3], 3, {3: (0, 0.0)}, "must be"),
    )
    for message, source, active, text in cases:
      with self.subTest(message=message, source=source):
        before = dict(active)
        with self.assertRaisesRegex(RuntimeError, text):
          validate_worker_done_message(message, source, active)
        self.assertEqual(active, before)


class WorkerResultConsumerTests(unittest.TestCase):

  def test_valid_payload_changes_only_the_assigned_row(self):
    store = _ResultStore()
    payload = object()
    active = {3: (17, 12.5)}
    result = consume_worker_result_message(
      store, ("ok", 17, payload), 3, active)
    self.assertEqual(result, (17, "accepted", None))
    self.assertEqual(store.accepted, [(17, payload, True)])
    self.assertEqual(store.zeroed, [])
    self.assertFalse(store.failed[17])
    self.assertEqual(active, {})

  def test_two_workers_may_return_out_of_order_without_crossing_rows(self):
    store = _ResultStore()
    payload_4 = object()
    payload_17 = object()
    active = {2: (4, 10.0), 3: (17, 11.0)}

    result_17 = consume_worker_result_message(
      store, ("ok", 17, payload_17), 3, active)
    self.assertEqual(result_17, (17, "accepted", None))
    self.assertEqual(active, {2: (4, 10.0)})
    self.assertTrue(store.failed[4])
    self.assertFalse(store.failed[17])

    result_4 = consume_worker_result_message(
      store, ("ok", 4, payload_4), 2, active)
    self.assertEqual(result_4, (4, "accepted", None))
    self.assertEqual(
      store.accepted,
      [(17, payload_17, True), (4, payload_4, True)])
    self.assertFalse(store.failed[4])
    self.assertEqual(active, {})

  def test_replayed_result_refuses_without_a_second_write(self):
    store = _ResultStore()
    payload = object()
    active = {3: (17, 12.5)}
    consume_worker_result_message(
      store, ("ok", 17, payload), 3, active)
    accepted_once = list(store.accepted)

    with self.assertRaisesRegex(RuntimeError, "without a live"):
      consume_worker_result_message(
        store, ("ok", 17, payload), 3, active)
    self.assertEqual(store.accepted, accepted_once)
    self.assertEqual(store.zeroed, [])
    self.assertEqual(active, {})

  def test_wrong_row_refuses_before_storage_or_assignment_changes(self):
    store = _ResultStore()
    active = {3: (17, 12.5)}
    with self.assertRaisesRegex(RuntimeError, "assigned row 17"):
      consume_worker_result_message(
        store, ("ok", 5, object()), 3, active)
    self.assertEqual(store.accepted, [])
    self.assertEqual(store.zeroed, [])
    self.assertEqual(active, {3: (17, 12.5)})

  def test_invalid_payload_is_zeroed_and_remains_failed(self):
    error = ValueError("wrong shape")
    store = _ResultStore(accept_error=error)
    active = {2: (9, 1.0)}
    result = consume_worker_result_message(
      store, ("ok", 9, object()), 2, active)
    self.assertEqual(result, (9, "invalid-payload", error))
    self.assertEqual(store.zeroed, [9])
    self.assertTrue(store.failed[9])
    self.assertEqual(active, {})

  def test_worker_error_is_zeroed_and_keeps_traceback_text(self):
    store = _ResultStore()
    active = {2: (9, 1.0)}
    result = consume_worker_result_message(
      store, ("err", 9, "traceback"), 2, active)
    self.assertEqual(result, (9, "worker-error", "traceback"))
    self.assertEqual(store.accepted, [])
    self.assertEqual(store.zeroed, [9])
    self.assertTrue(store.failed[9])
    self.assertEqual(active, {})


class GeneratorWiringTests(unittest.TestCase):

  def test_each_reply_is_validated_before_assignment_removal(self):
    method = _method_node("GeneratorCore", "__generate_datavectors")
    done_pairs = []
    for statements in _statement_lists(method):
      for first, second in zip(statements, statements[1:]):
        call = _direct_call(first)
        called = call.func.id if call is not None else None
        if called == "validate_worker_done_message":
          arguments = {
            keyword.arg: keyword.value.id
            for keyword in call.keywords
            if keyword.arg is not None and isinstance(keyword.value, ast.Name)
          }
          done_pairs.append((
            called,
            arguments,
            _deletes_active_source(second)))
    self.assertEqual(
      done_pairs,
      [
        ("validate_worker_done_message",
         {"message": "message", "source": "src", "active": "active"},
         True),
      ])
    consumer_calls = [
      node for node in ast.walk(method)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and isinstance(node.func.value, ast.Name)
      and node.func.value.id == "self"
      and node.func.attr == "_consume_worker_result_message"
    ]
    self.assertEqual(len(consumer_calls), 2)
    for call in consumer_calls:
      arguments = {
        keyword.arg: keyword.value.id
        for keyword in call.keywords
        if keyword.arg is not None and isinstance(keyword.value, ast.Name)
      }
      self.assertEqual(
        arguments,
        {"message": "message", "source": "src", "active": "active"})

    result_branches = []
    for node in ast.walk(method):
      if not isinstance(node, ast.If):
        continue
      probes = [
        call for call in ast.walk(node.test)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "comm"
        and call.func.attr == "Iprobe"
        and any(
          keyword.arg == "tag"
          and isinstance(keyword.value, ast.Name)
          and keyword.value.id == "RTAG"
          for keyword in call.keywords)
      ]
      if probes:
        result_branches.append(node)
    self.assertEqual(len(result_branches), 2)
    for branch in result_branches:
      receives = [
        call for statement in branch.body for call in ast.walk(statement)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "comm"
        and call.func.attr == "recv"
        and any(
          keyword.arg == "tag"
          and isinstance(keyword.value, ast.Name)
          and keyword.value.id == "RTAG"
          for keyword in call.keywords)
      ]
      consumers = [
        call for statement in branch.body for call in ast.walk(statement)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == "_consume_worker_result_message"
      ]
      self.assertEqual(len(receives), 1)
      self.assertEqual(len(consumers), 1)
      self.assertLess(receives[0].lineno, consumers[0].lineno)
    all_assignment_removals = [
      node for node in ast.walk(method) if _deletes_active_source(node)
    ]
    self.assertEqual(
      len(all_assignment_removals), 1,
      "the result consumer owns result removals; the done validator owns one")

    consumer = _method_node(
      "GeneratorCore", "_consume_worker_result_message")
    validator_calls = [
      node for node in ast.walk(consumer)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id == "validate_worker_result_message"
    ]
    consumer_removals = [
      node for node in ast.walk(consumer)
      if _deletes_active_assignment(node)
    ]
    self.assertEqual(len(validator_calls), 1)
    self.assertEqual(len(consumer_removals), 1)
    self.assertLess(validator_calls[0].lineno, consumer_removals[0].lineno)


if __name__ == "__main__":
  unittest.main()
