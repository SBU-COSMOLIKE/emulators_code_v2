#!/usr/bin/env python3
"""Reproduce the unresolved unit-41 persistence and sweep-product defects.

This is an adversarial-review witness, not an acceptance gate.  A PASS means
the named defect was reproduced against the current production source.  Once
production is repaired, the repair must replace each witness with the
corresponding positive acceptance arm and a mutation that restores the defect.

The script uses the Cocoa interpreter.  It executes the real artifact writer
and both real table writers.  Syntax-tree extraction supplies the metadata
expressions from the production sweep drivers, so a copied test-only metadata
mapping cannot conceal what the drivers actually publish.
"""

import ast
import os
import tempfile
from types import SimpleNamespace

import h5py
import torch
import yaml

from emulator.results import save_emulator
from emulator.results import save_learning_curves
from emulator.results import save_sweep_table


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_TRAINING = os.path.join(_REPO, "emulator", "training.py")
_NTRAIN = os.path.join(_REPO, "cosmic_shear_sweep_ntrain_emulator.py")
_HYPER = os.path.join(
  _REPO,
  "cosmic_shear_sweep_hyperparam_emulator.py")

FAILURES = []


def report(
    label,
    ok,
    detail=""):
  """Print one witness result and retain an unsuccessful reproduction."""
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


def _parse(path):
  """Parse one production Python file."""
  with open(
      path,
      encoding="utf-8") as source_file:
    return ast.parse(
      source_file.read(),
      filename=path)


def _function(tree, name):
  """Return the uniquely named top-level function."""
  matches = []
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      if node.name == name:
        matches.append(node)
  if len(matches) != 1:
    raise AssertionError(
      "expected one function " + name + ", found " + str(len(matches)))
  return matches[0]


def _call_name(call):
  """Return the final identifier of a call target."""
  if isinstance(call.func, ast.Name):
    return call.func.id
  if isinstance(call.func, ast.Attribute):
    return call.func.attr
  return None


def _call_keyword_expression(path, function_name, call_name, keyword_name):
  """Extract one keyword expression from one production call."""
  tree = _parse(path)
  owner = _function(
    tree=tree,
    name=function_name)
  matches = []
  for node in ast.walk(owner):
    if not isinstance(node, ast.Call):
      continue
    if _call_name(node) != call_name:
      continue
    for keyword in node.keywords:
      if keyword.arg == keyword_name:
        matches.append(keyword.value)
  if len(matches) != 1:
    raise AssertionError(
      "expected one " + call_name + " " + keyword_name
      + " expression, found " + str(len(matches)))
  expression = ast.Expression(body=matches[0])
  ast.fix_missing_locations(expression)
  return expression


def _call_keyword_names(path, function_name, call_name):
  """Return keyword names passed to one uniquely named production call."""
  tree = _parse(path)
  owner = _function(
    tree=tree,
    name=function_name)
  matches = []
  for node in ast.walk(owner):
    if isinstance(node, ast.Call) and _call_name(node) == call_name:
      matches.append(node)
  if len(matches) != 1:
    raise AssertionError(
      "expected one " + call_name + " call, found " + str(len(matches)))
  names = []
  for keyword in matches[0].keywords:
    names.append(keyword.arg)
  return names


def _assignment_expressions(path, function_name, target_name):
  """Return expressions assigned to one local name in a function."""
  tree = _parse(path)
  owner = _function(
    tree=tree,
    name=function_name)
  expressions = []
  for node in ast.walk(owner):
    if not isinstance(node, ast.Assign):
      continue
    for target in node.targets:
      if isinstance(target, ast.Name) and target.id == target_name:
        expression = ast.Expression(body=node.value)
        ast.fix_missing_locations(expression)
        expressions.append(expression)
  return expressions


def _evaluate(expression, path, namespace):
  """Evaluate an extracted production expression in a small fixture."""
  return eval(
    compile(
      expression,
      path,
      "eval"),
    {},
    namespace)


def _resolved_train_keys():
  """Return literal keys of the production ``resolved_train`` mapping."""
  tree = _parse(_TRAINING)
  owner = _function(
    tree=tree,
    name="run_emulator")
  candidates = []
  for node in ast.walk(owner):
    if not isinstance(node, ast.Assign):
      continue
    owns_record = False
    for target in node.targets:
      if isinstance(target, ast.Name) and target.id == "resolved_train":
        owns_record = True
    if owns_record and isinstance(node.value, ast.Dict):
      candidates.append(node.value)
  if len(candidates) != 1:
    raise AssertionError(
      "expected one literal resolved_train mapping, found "
      + str(len(candidates)))
  keys = []
  for key in candidates[0].keys:
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
      keys.append(key.value)
    else:
      raise AssertionError("resolved_train has a non-literal top-level key")
  return keys


def _amp_assignment_locations():
  """Return functions that locally assign the AMP dtype or scaler policy."""
  tree = _parse(_TRAINING)
  locations = []
  for owner in tree.body:
    if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
      continue
    for node in ast.walk(owner):
      names = []
      if isinstance(node, ast.Assign):
        for target in node.targets:
          if isinstance(target, ast.Name):
            names.append(target.id)
      elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name):
          names.append(node.target.id)
      for name in names:
        if name in ("amp_dtype", "scaler_policy"):
          locations.append((name, owner.name, node.lineno))
  return locations


class _EmptyGeometry:
  """Small state owner accepted by the real artifact writer."""

  def state(self):
    """Return an empty geometry state for this persistence-only witness."""
    return {}


def check_amp_artifact():
  """Write and read one artifact using the production record schema."""
  keys = _resolved_train_keys()
  resolved = {}
  for key in keys:
    resolved[key] = None
  resolved["use_amp"] = True
  resolved["device"] = "mps"

  with tempfile.TemporaryDirectory(prefix="unit41-policy-") as directory:
    root = os.path.join(directory, "artifact")
    histories = {
      "train_losses": [0.0],
      "val_medians": [0.0],
      "val_means": [0.0],
      "val_fracs": [torch.tensor([0.0])],
      "thresholds": torch.tensor([0.2]),
    }
    save_emulator(
      path_root=root,
      model=torch.nn.Linear(1, 1),
      param_geometry=_EmptyGeometry(),
      geometry=_EmptyGeometry(),
      config={"data": {}, "train_args": {}},
      histories=histories,
      resolved_train=resolved,
      resolved_model={"class": "red-team-fixture"})
    with h5py.File(root + ".h5", "r") as artifact:
      payload = artifact["config_resolved_yaml"][()]
    if isinstance(payload, bytes):
      payload = payload.decode("utf-8")
    readback = yaml.safe_load(payload)["train_args"]

  absent = []
  for key in ("amp_dtype", "scaler_policy"):
    if key not in readback:
      absent.append(key)
  report(
    "artifact omits the resolved AMP dtype and scaler policy",
    absent == ["amp_dtype", "scaler_policy"],
    "readback keys=" + repr(sorted(readback)))

  locations = _amp_assignment_locations()
  report(
    "AMP dtype is locally re-derived outside the record owner",
    locations == [("amp_dtype", "training_loop_batched", 1953)],
    "assignments=" + repr(locations)
    + "; record owner=run_emulator")


def _ntrain_metadata(activation_flag):
  """Evaluate the real N-train table metadata expression."""
  expression = _call_keyword_expression(
    path=_NTRAIN,
    function_name="main",
    call_name="save_learning_curves",
    keyword_name="meta")
  args = SimpleNamespace(
    activation=activation_flag,
    rescale="none",
    threshold=0.2)
  namespace = {
    "args": args,
    "family": "cosmolike",
    "model_name": "ResCNN",
    "pool": 20,
    "n_workers": 1,
  }
  return _evaluate(
    expression=expression,
    path=_NTRAIN,
    namespace=namespace)


def _hyper_metadata(activation_flag, act_mode):
  """Evaluate the real ordinary-sweep table metadata expression."""
  expression = _call_keyword_expression(
    path=_HYPER,
    function_name="main",
    call_name="save_sweep_table",
    keyword_name="meta")
  args = SimpleNamespace(
    activation=activation_flag,
    rescale="none",
    threshold=0.2)
  exp = SimpleNamespace(model_name="rescnn")
  namespace = {
    "act_mode": act_mode,
    "args": args,
    "cfg": {"data": {"n_train": 20}},
    "exp": exp,
    "family": "cosmolike",
    "n_workers": 2,
  }
  return _evaluate(
    expression=expression,
    path=_HYPER,
    namespace=namespace)


def check_sweep_products():
  """Execute both table writers with production metadata expressions."""
  default_meta = _ntrain_metadata(activation_flag=None)
  yaml_power_meta = _hyper_metadata(
    activation_flag=None,
    act_mode=False)
  explicit_meta = _ntrain_metadata(activation_flag="power")
  activation_sweep_meta = _hyper_metadata(
    activation_flag=None,
    act_mode=True)

  with tempfile.TemporaryDirectory(prefix="unit41-sweep-") as directory:
    ntrain_path = os.path.join(directory, "ntrain.txt")
    hyper_path = os.path.join(directory, "hyper.txt")
    activation_path = os.path.join(directory, "activation.txt")
    save_learning_curves(
      path=ntrain_path,
      sizes=[10, 20],
      curves={"frac": [0.2, 0.1]},
      meta=default_meta)
    save_sweep_table(
      path=hyper_path,
      param="lr.lr_base",
      values=[0.001, 0.002],
      fracs=[0.2, 0.1],
      meta=yaml_power_meta)
    save_sweep_table(
      path=activation_path,
      param="model.activation",
      values=["H", "power"],
      fracs=[0.2, 0.1],
      meta=activation_sweep_meta)
    with open(
        ntrain_path,
        encoding="utf-8") as table_file:
      ntrain_text = table_file.read()
    with open(
        hyper_path,
        encoding="utf-8") as table_file:
      hyper_text = table_file.read()
    with open(
        activation_path,
        encoding="utf-8") as table_file:
      activation_text = table_file.read()

  report(
    "default H is published as activation=None in the N-train table",
    default_meta.get("activation", "missing") is None
    and "activation=None" in ntrain_text,
    "metadata=" + repr(default_meta))
  report(
    "a YAML power selection is published as activation=None",
    yaml_power_meta.get("activation", "missing") is None
    and "activation=None" in hyper_text,
    "metadata=" + repr(yaml_power_meta))
  report(
    "the explicit CLI override is the control that happens to agree",
    explicit_meta.get("activation") == "power",
    "metadata=" + repr(explicit_meta))

  missing_head = []
  for key in ("head_activation", "head_activation_n_gates"):
    if key not in default_meta:
      missing_head.append(key)
  report(
    "sweep products omit the resolved head activation pin",
    missing_head == ["head_activation", "head_activation_n_gates"],
    "missing=" + repr(missing_head))

  missing_values = []
  for key in ("activation_values", "activation_n_gates"):
    if key not in activation_sweep_meta:
      missing_values.append(key)
  report(
    "activation-family value order survives as a categorical table control",
    "# values: 0=H, 1=power" in activation_text,
    "value header preserved")
  report(
    "activation-family metadata has no immutable resolved-value record",
    activation_sweep_meta.get("activation") == "swept"
    and missing_values == ["activation_values", "activation_n_gates"],
    "metadata=" + repr(activation_sweep_meta))

  payload_namespace = {
    "worker_cfg": {"data": {}},
    "args": SimpleNamespace(
      activation=None,
      rescale="none",
      threshold=0.2),
    "param": "lr.lr_base",
    "act_mode": False,
  }
  ntrain_payloads = _assignment_expressions(
    path=_NTRAIN,
    function_name="_run_parallel",
    target_name="extra")
  hyper_payloads = _assignment_expressions(
    path=_HYPER,
    function_name="main",
    target_name="extra")
  if len(ntrain_payloads) != 1:
    raise AssertionError(
      "expected one N-train worker payload, found "
      + str(len(ntrain_payloads)))
  resolved_hyper_payload = None
  for expression in hyper_payloads:
    value = _evaluate(
      expression=expression,
      path=_HYPER,
      namespace=payload_namespace)
    if isinstance(value, dict) and "activation" in value:
      resolved_hyper_payload = value
  if resolved_hyper_payload is None:
    raise AssertionError("hyperparameter pooled payload has no activation")
  ntrain_payload = _evaluate(
    expression=ntrain_payloads[0],
    path=_NTRAIN,
    namespace=payload_namespace)
  report(
    "both pooled paths transport the raw optional flag for re-resolution",
    ntrain_payload["activation"] is None
    and resolved_hyper_payload["activation"] is None,
    "N-train=None; hyper=None; executed YAML fixture=power")

  curves_expression = _call_keyword_expression(
    path=_NTRAIN,
    function_name="main",
    call_name="plot_learning_curves",
    keyword_name="curves")
  label_mapping = _evaluate(
    expression=curves_expression,
    path=_NTRAIN,
    namespace={
      "model_name": "ResCNN",
      "args": SimpleNamespace(rescale="none"),
      "sizes": [10, 20],
      "fracs": [0.2, 0.1],
    })
  labels = list(label_mapping)
  label_text = labels[0] if labels else ""
  report(
    "the figure label omits activation and the head pin",
    "H" not in label_text and "activation" not in label_text,
    "label=" + repr(label_text))
  hyper_plot_keywords = _call_keyword_names(
    path=_HYPER,
    function_name="main",
    call_name="plot_sweep_curve")
  report(
    "the ordinary-sweep figure receives no resolved design metadata",
    "design_label" not in hyper_plot_keywords
    and "metadata" not in hyper_plot_keywords,
    "keywords=" + repr(hyper_plot_keywords))

  class TemplateResCNN:
    pass

  identity_expressions = _assignment_expressions(
    path=_NTRAIN,
    function_name="main",
    target_name="model_name")
  if len(identity_expressions) != 1:
    raise AssertionError(
      "expected one N-train model_name assignment, found "
      + str(len(identity_expressions)))
  composed_exp = SimpleNamespace(
    model_cls=TemplateResCNN,
    model_name="rescnn_nla")
  selected_identity = _evaluate(
    expression=identity_expressions[0],
    path=_NTRAIN,
    namespace={"exp": composed_exp})
  report(
    "N-train drops the composed IA identity from its product name",
    selected_identity == "TemplateResCNN"
    and selected_identity != composed_exp.model_name,
    "selected=" + repr(selected_identity)
    + "; resolved=" + repr(composed_exp.model_name))


def main():
  """Run every current-defect witness."""
  print("unit 41 persisted-policy and sweep-product witnesses")
  check_amp_artifact()
  check_sweep_products()
  if FAILURES:
    print("FAILED to reproduce: " + ", ".join(FAILURES))
    raise SystemExit(1)
  print("ALL CURRENT DEFECTS REPRODUCED (review witness, not acceptance)")


if __name__ == "__main__":
  main()
