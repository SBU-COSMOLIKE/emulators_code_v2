#!/usr/bin/env python3
"""Reproduce the unresolved unit-53 study-identity defects.

This is an adversarial-review witness, not an acceptance gate.  A PASS means
the named current defect was reproduced.  The fixture executes the real
``main`` syntax tree in an inert namespace: it reaches the production family
name selection, resume, default-control, spawn, join, and final-report code,
but no worker or training job runs and no journal is created.

Once production is repaired, the repair must replace these negative witnesses
with positive manifest acceptance arms plus mutations that restore the
observed failures.
"""

import argparse
import ast
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "cosmic_shear_tune_emulator.py"
WRAPPERS = (
  ROOT / "scalar_tune_emulator.py",
  ROOT / "cmb_tune_emulator.py",
  ROOT / "baosn_tune_emulator.py",
  ROOT / "mps_tune_emulator.py",
)

FAILURES = []
EVENTS = []
PROCESS_EXITCODE = 0


def report(
    label,
    ok,
    detail=""):
  """Print one witness result and retain an unsuccessful reproduction."""
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


def _function_node(tree, name):
  """Return one uniquely named top-level function."""
  matches = []
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == name:
      matches.append(node)
  if len(matches) != 1:
    raise AssertionError(
      "expected one function " + name + ", found " + str(len(matches)))
  return matches[0]


def _wrapper_routes():
  """Read the real wrapper arguments for every family route."""
  routes = {}
  for path in WRAPPERS:
    tree = ast.parse(
      path.read_text(encoding="utf-8"),
      filename=str(path))
    calls = []
    for node in ast.walk(tree):
      if not isinstance(node, ast.Call):
        continue
      if not isinstance(node.func, ast.Name) or node.func.id != "main":
        continue
      values = {}
      for keyword in node.keywords:
        values[keyword.arg] = ast.literal_eval(keyword.value)
      calls.append(values)
    if len(calls) != 1:
      raise AssertionError(
        path.name + " must call main exactly once; calls=" + repr(calls))
    routes[path.name] = calls[0]
  return routes


def _manifest_owner_files():
  """Return production Python files whose names claim study-manifest ownership."""
  found = []
  for path in ROOT.rglob("*.py"):
    if ".git" in path.parts or ".claude" in path.parts:
      continue
    if "study" in path.name and "manifest" in path.name:
      found.append(str(path.relative_to(ROOT)))
  return found


class OldTrial:
  """One incomparable COMPLETE trial from an older configuration."""

  state = "COMPLETE"
  user_attrs = {"median": 0.111}
  params = {"old_config": "manifest-A"}


class LegacyStudy:
  """A resumed journal study with no unit-53 identity attributes."""

  def __init__(self):
    self.trials = [OldTrial()]
    self.best_trial = self.trials[0]
    self.best_value = 0.001
    self.user_attrs = {}
    self.system_attrs = {}
    self.enqueued = []

  def enqueue_trial(self, values):
    """Record any attempted known-default enqueue."""
    self.enqueued.append(dict(values))


LEGACY_STUDY = LegacyStudy()


class FakeExperiment:
  """Inert parent experiment used only to reach the parallel path."""

  raw_train_args = {"lr": [0.01, 0.001, 0.1, "log"]}
  model_cls = object

  @classmethod
  def from_config(cls, cfg, **kwargs):
    """Return the inert experiment and record its resolved inputs."""
    EVENTS.append(("experiment", kwargs))
    return cls()

  def log(self, message):
    """Capture the production final report."""
    EVENTS.append(("log", message))

  def print_design(self):
    """Record the banner boundary without printing it."""
    EVENTS.append(("print_design",))

  def stage_train(self):
    """Refuse accidental parent-side staging."""
    raise AssertionError("parallel parent unexpectedly staged training")

  def stage_val(self):
    """Refuse accidental parent-side staging."""
    raise AssertionError("parallel parent unexpectedly staged validation")

  def build_geometry(self):
    """Refuse accidental parent-side geometry construction."""
    raise AssertionError("parallel parent unexpectedly built geometry")


class FakeProcess:
  """A process object that records spawn and join without running a child."""

  def __init__(self, target, args):
    self.target = target
    self.args = args
    self.exitcode = PROCESS_EXITCODE

  def start(self):
    """Record the exact study name forwarded to the worker."""
    EVENTS.append(("spawn", self.target.__name__, self.args[6]))

  def join(self):
    """Record the configured worker exit code."""
    EVENTS.append(("join", self.exitcode))


class FakeContext:
  """Return inert process objects from the production spawn loop."""

  def Process(self, target, args):
    """Construct one inert worker process."""
    return FakeProcess(
      target=target,
      args=args)


class FakeLogging:
  """Minimal Optuna logging namespace."""

  WARNING = "WARNING"

  @staticmethod
  def set_verbosity(value):
    """Record the logging boundary."""
    EVENTS.append(("verbosity", value))


class FakeSamplers:
  """Minimal Optuna sampler namespace."""

  @staticmethod
  def TPESampler(seed):
    """Return an inert sampler identity."""
    return ("TPE", seed)


class FakeTrialState:
  """Match the production COMPLETE comparison."""

  COMPLETE = "COMPLETE"


class FakeTrialNamespace:
  """Expose TrialState under the same Optuna path."""

  TrialState = FakeTrialState


def _create_study(**kwargs):
  """Return the manifest-free legacy study from the real creation boundary."""
  EVENTS.append(
    ("create_study", dict(kwargs), dict(LEGACY_STUDY.user_attrs)))
  return LEGACY_STUDY


def _load_study(**kwargs):
  """Return the same legacy study from the real reload boundary."""
  EVENTS.append(
    ("load_study", dict(kwargs), dict(LEGACY_STUDY.user_attrs)))
  return LEGACY_STUDY


def _require_family_block(data, family, prog):
  """Record the real driver's family-validation boundary."""
  EVENTS.append(("family", family, prog))


def _add_cocoa_path_args(parser):
  """Leave the inert parser with no filesystem requirements."""
  del parser


def _resolve_cocoa_config(args):
  """Return a small already-resolved configuration."""
  del args
  return {"data": {"ram_frac": 0.7}}, "/tmp", None


def _cocoa_output(fileroot, journal):
  """Resolve the inert journal name without creating it."""
  return str(Path(fileroot) / journal)


def _search_defaults(train_args):
  """Return one known-default search point."""
  del train_args
  return {"lr": 0.01}


def _validate_sweep_paths(paths, two_phase):
  """Accept the fixture's ordinary search path."""
  del paths, two_phase


def _journal_storage(path):
  """Return an inert journal-storage identity."""
  return ("journal", path)


def _tune_worker(*args):
  """Never run; FakeProcess records its forwarded arguments."""
  del args


def _device_count():
  """Select the production two-worker parallel path."""
  return 2


def _get_context(method):
  """Return the inert spawn context."""
  if method != "spawn":
    raise AssertionError("expected spawn context, got " + repr(method))
  return FakeContext()


def _build_namespace(main_node):
  """Compile the real main function into the inert execution namespace."""
  fake_train_module = types.ModuleType("cosmic_shear_train_emulator")
  fake_train_module.require_family_block = _require_family_block
  sys.modules["cosmic_shear_train_emulator"] = fake_train_module

  fake_optuna = types.SimpleNamespace(
    logging=FakeLogging,
    samplers=FakeSamplers,
    trial=FakeTrialNamespace,
    create_study=_create_study,
    load_study=_load_study)

  fake_torch = types.ModuleType("torch")
  fake_torch.__path__ = []
  fake_torch.cuda = types.SimpleNamespace(device_count=_device_count)
  mp_module = types.ModuleType("torch.multiprocessing")
  mp_module.get_context = _get_context
  fake_torch.multiprocessing = mp_module
  sys.modules["torch"] = fake_torch
  sys.modules["torch.multiprocessing"] = mp_module

  module = ast.Module(
    body=[main_node],
    type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "argparse": argparse,
    "STUDY_NAME": "cosmic_shear_tune",
    "add_cocoa_path_args": _add_cocoa_path_args,
    "resolve_cocoa_config": _resolve_cocoa_config,
    "cocoa_output": _cocoa_output,
    "EmulatorExperiment": FakeExperiment,
    "search_defaults": _search_defaults,
    "validate_sweep_paths": _validate_sweep_paths,
    "optuna": fake_optuna,
    "torch": fake_torch,
    "journal_storage": _journal_storage,
    "_tune_worker": _tune_worker,
  }
  exec(
    compile(
      module,
      str(DRIVER),
      "exec"),
    namespace)
  return namespace


def _events_named(name):
  """Return current events with one event tag."""
  matches = []
  for event in EVENTS:
    if event[0] == name:
      matches.append(event)
  return matches


def _run_route(namespace, prog, family, process_exitcode=0):
  """Execute one public family route through the real parallel parent."""
  global PROCESS_EXITCODE
  PROCESS_EXITCODE = process_exitcode
  EVENTS.clear()
  LEGACY_STUDY.enqueued.clear()
  old_argv = list(sys.argv)
  try:
    sys.argv = [prog]
    namespace["main"](
      prog=prog,
      family=family)
  finally:
    sys.argv = old_argv

  creates = _events_named("create_study")
  spawns = _events_named("spawn")
  joins = _events_named("join")
  logs = _events_named("log")
  if len(creates) != 1:
    raise AssertionError(
      "expected one create_study call, found " + str(len(creates)))
  joined = []
  for event in joins:
    joined.append(event[1])
  messages = []
  for event in logs:
    messages.append(str(event[1]))
  return {
    "prog": prog,
    "family": family,
    "selected_study_name": creates[0][1]["study_name"],
    "load_if_exists": creates[0][1]["load_if_exists"],
    "study_attrs_at_open": creates[0][2],
    "enqueued_defaults": list(LEGACY_STUDY.enqueued),
    "spawned_workers": len(spawns),
    "joined_exitcodes": joined,
    "reported_best_params": dict(LEGACY_STUDY.best_trial.params),
    "log_messages": messages,
  }


def _study_attr_calls(tree):
  """Return attribute writes visible anywhere in the production driver."""
  calls = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    function = node.func
    if not isinstance(function, ast.Attribute):
      continue
    if function.attr in ("set_user_attr", "set_system_attr"):
      calls.append((function.attr, node.lineno))
  return calls


def _worker_calls_before_load(worker_node):
  """Return worker calls that happen before its journal authentication point."""
  load_line = None
  calls = []
  for node in ast.walk(worker_node):
    if not isinstance(node, ast.Call):
      continue
    name = None
    if isinstance(node.func, ast.Name):
      name = node.func.id
    elif isinstance(node.func, ast.Attribute):
      name = node.func.attr
    if name == "load_study":
      load_line = node.lineno
    if name is not None:
      calls.append((name, node.lineno))
  if load_line is None:
    raise AssertionError("worker has no load_study call")
  before = []
  for call in calls:
    if call[1] < load_line:
      before.append(call)
  return before


def main():
  """Run every current unit-53 defect witness."""
  source = DRIVER.read_text(encoding="utf-8")
  tree = ast.parse(
    source,
    filename=str(DRIVER))
  main_node = _function_node(
    tree=tree,
    name="main")
  worker_node = _function_node(
    tree=tree,
    name="_tune_worker")
  routes_by_wrapper = _wrapper_routes()
  namespace = _build_namespace(main_node=main_node)

  routes = []
  routes.append(_run_route(
    namespace=namespace,
    prog="cosmic_shear_tune_emulator",
    family="cosmolike"))
  for values in routes_by_wrapper.values():
    routes.append(_run_route(
      namespace=namespace,
      prog=values["prog"],
      family=values["family"]))
  renamed = _run_route(
    namespace=namespace,
    prog="renamed_scalar_cli",
    family="outputs")
  failed_workers = _run_route(
    namespace=namespace,
    prog="cosmic_shear_tune_emulator",
    family="cosmolike",
    process_exitcode=1)

  owner_files = _manifest_owner_files()
  attribute_calls = _study_attr_calls(tree=tree)
  before_load = _worker_calls_before_load(worker_node=worker_node)
  report(
    "no canonical study-manifest owner exists",
    owner_files == [],
    "owner files=" + repr(owner_files))
  report(
    "the driver writes only per-trial median attributes",
    attribute_calls == [("set_user_attr", 192), ("set_user_attr", 370)],
    "attribute calls=" + repr(attribute_calls))
  report(
    "the direct cosmolike default forks the historic study name",
    routes[0]["selected_study_name"] == "cosmic_shear_tune_emulator"
    and routes[0]["selected_study_name"] != "cosmic_shear_tune",
    "selected=" + repr(routes[0]["selected_study_name"]))
  report(
    "wrapper naming depends on the mutable program label",
    renamed["selected_study_name"] == "renamed_scalar_cli"
    and renamed["selected_study_name"] != routes[1]["selected_study_name"],
    "stable=" + repr(routes[1]["selected_study_name"])
    + "; renamed=" + repr(renamed["selected_study_name"]))

  every_legacy_opened = True
  every_default_suppressed = True
  every_old_winner_reported = True
  for route in routes:
    every_legacy_opened = (
      every_legacy_opened
      and route["load_if_exists"]
      and route["study_attrs_at_open"] == {})
    every_default_suppressed = (
      every_default_suppressed
      and route["enqueued_defaults"] == [])
    every_old_winner_reported = (
      every_old_winner_reported
      and route["reported_best_params"] == {
        "old_config": "manifest-A",
      })
  report(
    "all five family routes accept a legacy no-manifest study",
    every_legacy_opened,
    "routes=" + str(len(routes)))
  report(
    "one old COMPLETE trial suppresses the manifest-owned default control",
    every_default_suppressed,
    "enqueued defaults=[] on every route")
  report(
    "an incomparable old trial is reported as every route's winner",
    every_old_winner_reported,
    "best params={'old_config': 'manifest-A'}")
  report(
    "workers stage scientific inputs before loading the journal",
    ("stage_train", 161) in before_load
    and ("stage_val", 162) in before_load
    and ("build_geometry", 163) in before_load,
    "calls before load=" + repr(before_load))

  final_text = "\n".join(failed_workers["log_messages"])
  report(
    "failed workers plus an old COMPLETE trial still report success",
    failed_workers["joined_exitcodes"] == [1, 1]
    and failed_workers["reported_best_params"] == {
      "old_config": "manifest-A",
    }
    and "search complete" in final_text,
    "exitcodes=" + repr(failed_workers["joined_exitcodes"]))
  report(
    "the final report names neither stable study name nor manifest digest",
    "study name:" not in final_text
    and "study manifest sha256:" not in final_text,
    "final log=" + repr(final_text))

  if FAILURES:
    print("FAILED to reproduce: " + ", ".join(FAILURES))
    raise SystemExit(1)
  print("ALL CURRENT DEFECTS REPRODUCED (review witness, not acceptance)")


if __name__ == "__main__":
  main()
