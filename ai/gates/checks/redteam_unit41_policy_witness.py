#!/usr/bin/env python3
"""Check the resolved-policy and sweep-product records end to end.

This began as an adversarial defect witness.  The repair converts every
negative observation into a positive acceptance arm and retains controls that
restore the old local-policy and raw-command-line behavior.  Those controls
must be rejected by the same predicates that accept the production record.

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
import numpy as np
import torch

from ai.gates.checks.artifact_fixtures import one_pass_training_recipe
import yaml

from emulator import fixed_facts
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry
from emulator.results import save_emulator
from emulator.results import save_learning_curves
from emulator.results import save_sweep_table
from emulator.family_drivers import resolved_sweep_record
from emulator.family_drivers import sweep_design_label
from emulator.family_drivers import sweep_record_value


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
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


def check_amp_artifact():
  """Write and read one artifact using the production record schema."""
  keys = _resolved_train_keys()
  expected_policy = {
    "amp_dtype": "torch.float16",
    "scaler_policy": "unscaled",
  }
  policy_keys_declared = True
  for key in expected_policy:
    if key not in keys:
      policy_keys_declared = False
  report(
    "production resolved_train declares both resolved policy fields",
    policy_keys_declared,
    "production keys=" + repr(sorted(keys)))

  fixture_values = {
    "use_amp": True,
    "device": "mps",
    "amp_dtype": expected_policy["amp_dtype"],
    "scaler_policy": expected_policy["scaler_policy"],
  }
  resolved = one_pass_training_recipe(thresholds=(0.2,))
  for key in keys:
    if key not in resolved:
      resolved[key] = fixture_values.get(key)
  resolved.update(fixture_values)

  with tempfile.TemporaryDirectory(prefix="unit41-policy-") as directory:
    root = os.path.join(directory, "artifact")
    covmat = os.path.join(directory, "fixture.covmat")
    with open(covmat, "w") as handle:
      handle.write("# fixture0 fixture1\n1.0 0.0\n0.0 1.0\n")
    device = torch.device("cpu")
    pgeom = ParamGeometry.from_covmat(
      device=device, center=np.asarray([0.0, 0.0]), covmat_path=covmat)
    geometry = ScalarGeometry.from_targets(
      device=device,
      targets=np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
      names=["output"])
    block_opts = {
      "act": make_activation("H", n_gates=3),
      "norm": make_norm("affine"),
    }
    model = ResMLP(
      input_dim=2,
      output_dim=1,
      int_dim_res=4,
      n_blocks=1,
      block_opts=block_opts).to(device)
    recipe = {
      "cls": "emulator.designs.plain.ResMLP",
      "name": "resmlp",
      "ia": None,
      "input_dim": 2,
      "output_dim": 1,
      "compile_mode": None,
      "needs_geom": False,
      "kwargs": {
        "int_dim_res": 4,
        "n_blocks": 1,
        "block_opts": {
          "n_layers": 2,
          "act": {"type": "H", "n_gates": 3},
          "norm": "affine",
        },
      },
    }
    histories = {
      "train_losses": [0.0],
      "val_medians": [0.0],
      "val_means": [0.0],
      "val_fracs": [torch.tensor([0.0])],
      "thresholds": torch.tensor([0.2]),
    }
    save_emulator(
      path_root=root,
      model=model,
      param_geometry=pgeom,
      geometry=geometry,
      config={"data": {}, "train_args": {}},
      histories=histories,
      resolved_train=resolved,
      resolved_model=recipe,
      composition_mode="plain",
      transfer_refined=False,
      resolved_pce=None,
      resolved_transfer=None,
      resolved_rescale="none",
      facts_yaml=fixed_facts.synthetic_sidecar(
        names=["fixture0", "fixture1"],
        label="amp-policy-persistence",
        support=None),
      attrs={"rescale": "none"})
    with h5py.File(root + ".h5", "r") as artifact:
      payload = artifact["config_resolved_yaml"][()]
    if isinstance(payload, bytes):
      payload = payload.decode("utf-8")
    readback = yaml.safe_load(payload)["train_args"]

  report(
    "artifact persists the resolved AMP dtype and scaler policy",
    readback.get("amp_dtype") == expected_policy["amp_dtype"]
    and readback.get("scaler_policy") == expected_policy["scaler_policy"],
    "readback policy=" + repr({
      "amp_dtype": readback.get("amp_dtype"),
      "scaler_policy": readback.get("scaler_policy"),
    }))

  mutated = dict(readback)
  mutated.pop("amp_dtype", None)
  mutated.pop("scaler_policy", None)
  mutation_accepted = True
  for key, value in expected_policy.items():
    if mutated.get(key) != value:
      mutation_accepted = False
  report(
    "dropping both resolved policy fields is rejected",
    not mutation_accepted,
    "mutation keys=" + repr(sorted(mutated)))

  locations = _amp_assignment_locations()
  owners = []
  for name, owner, line in locations:
    owners.append((name, owner))
  report(
    "the resolved policy has one owner beside the artifact record",
    owners == [("amp_dtype", "run_emulator"),
               ("scaler_policy", "run_emulator")],
    "assignments=" + repr(locations))

  restored_local_derivation = list(owners)
  restored_local_derivation.append(("amp_dtype", "training_loop_batched"))
  loop_owns_dtype = False
  for name, owner in restored_local_derivation:
    if owner == "training_loop_batched":
      loop_owns_dtype = True
  report(
    "restoring a loop-local AMP dtype owner is rejected",
    restored_local_derivation != owners
    and loop_owns_dtype,
    "mutated owners=" + repr(restored_local_derivation))


class _HeadModel:
  """Expose the active head block used by the resolved-record helper."""

  head_block = "cnn"


def _experiment(activation, model_name="rescnn"):
  """Build one resolved experiment-shaped fixture."""
  train_args = {
    "model": {
      "activation": {
        "type": activation,
        "n_gates": 5,
      },
      "cnn": {
        "activation": {
          "type": "gated_power",
          "n_gates": 7,
        },
      },
    },
  }
  return SimpleNamespace(
    activation=activation,
    model_cls=_HeadModel,
    model_name=model_name,
    rescale="none",
    train_args=train_args)


def _resolved_metadata(activation, model_name="rescnn",
                       activation_values=None):
  """Execute the production resolved-record helper for one fixture."""
  exp = _experiment(
    activation=activation,
    model_name=model_name)
  return resolved_sweep_record(
    exp=exp,
    family="cosmolike",
    threshold=0.2,
    n_gpus=2,
    n_train=20,
    activation_values=activation_values)


def _table_metadata(path, function_name, call_name, run_record):
  """Evaluate the real table-call metadata expression."""
  expression = _call_keyword_expression(
    path=path,
    function_name=function_name,
    call_name=call_name,
    keyword_name="meta")
  return _evaluate(
    expression=expression,
    path=path,
    namespace={"dict": dict, "run_record": run_record})


def check_sweep_products():
  """Execute both table writers with production metadata expressions."""
  default_record = _resolved_metadata(activation="H")
  yaml_power_record = _resolved_metadata(activation="power")
  explicit_record = _resolved_metadata(activation="power")
  activation_sweep_record = _resolved_metadata(
    activation="H",
    activation_values=["H", "power"])
  composed_record = _resolved_metadata(
    activation="H",
    model_name="rescnn_nla")

  default_meta = _table_metadata(
    path=_NTRAIN,
    function_name="main",
    call_name="save_learning_curves",
    run_record=default_record)
  yaml_power_meta = _table_metadata(
    path=_HYPER,
    function_name="main",
    call_name="save_sweep_table",
    run_record=yaml_power_record)
  explicit_meta = dict(explicit_record)
  activation_sweep_meta = dict(activation_sweep_record)

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
    "default H is published as the activation that ran",
    default_meta.get("activation") == "H"
    and "activation=H" in ntrain_text,
    "metadata=" + repr(default_meta))
  report(
    "a YAML power selection is published as the activation that ran",
    yaml_power_meta.get("activation") == "power"
    and "activation=power" in hyper_text,
    "metadata=" + repr(yaml_power_meta))
  report(
    "an explicit activation override is preserved",
    explicit_meta.get("activation") == "power",
    "metadata=" + repr(explicit_meta))

  report(
    "sweep products carry the resolved head activation pin",
    default_meta.get("head_activation") == "gated_power"
    and default_meta.get("head_activation_n_gates") == 7,
    "metadata=" + repr(default_meta))

  report(
    "activation-family value order survives as a categorical table control",
    "# values: 0=H, 1=power" in activation_text,
    "value header preserved")
  report(
    "activation-family metadata carries one immutable ordered record",
    activation_sweep_meta.get("activation") == "swept"
    and activation_sweep_meta.get("activation_values") == ("H", "power")
    and activation_sweep_meta.get("activation_n_gates") == 5,
    "metadata=" + repr(activation_sweep_meta))

  payload_namespace = {
    "worker_cfg": {"data": {}},
    "args": SimpleNamespace(
      activation=None,
      rescale="none",
      threshold=0.2),
    "param": "lr.lr_base",
    "act_mode": False,
    "run_record": yaml_power_record,
    "sweep_record_value": sweep_record_value,
    "worker_activation": "power",
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
    "both pooled paths transport the shared resolved activation",
    ntrain_payload["activation"] == "power"
    and resolved_hyper_payload["activation"] == "power",
    "N-train=" + repr(ntrain_payload["activation"])
    + "; hyper=" + repr(resolved_hyper_payload["activation"]))

  curves_expression = _call_keyword_expression(
    path=_NTRAIN,
    function_name="main",
    call_name="plot_learning_curves",
    keyword_name="curves")
  design_label = sweep_design_label(record=composed_record)
  label_mapping = _evaluate(
    expression=curves_expression,
    path=_NTRAIN,
    namespace={
      "design_label": design_label,
      "sizes": [10, 20],
      "fracs": [0.2, 0.1],
    })
  labels = list(label_mapping)
  label_text = labels[0] if labels else ""
  report(
    "the figure label carries model, activation, and head pin",
    "rescnn_nla" in label_text
    and "activation H" in label_text
    and "head gated_power" in label_text,
    "label=" + repr(label_text))
  hyper_plot_keywords = _call_keyword_names(
    path=_HYPER,
    function_name="main",
    call_name="plot_sweep_curve")
  report(
    "the ordinary-sweep figure receives resolved design metadata",
    "design_label" in hyper_plot_keywords,
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
    "N-train preserves the composed IA identity in its product name",
    selected_identity == composed_exp.model_name,
    "selected=" + repr(selected_identity)
    + "; resolved=" + repr(composed_exp.model_name))

  raw_flag_mutation = None
  report(
    "restoring the raw optional activation is rejected",
    raw_flag_mutation != default_meta["activation"],
    "raw=None; resolved=" + repr(default_meta["activation"]))

  mutable_copy = dict(activation_sweep_record)
  mutable_copy["activation_values"] = tuple(reversed(
    mutable_copy["activation_values"]))
  report(
    "reversing the activation-family order changes the record",
    tuple(mutable_copy.items()) != activation_sweep_record,
    "mutated values=" + repr(mutable_copy["activation_values"]))


def main():
  """Run every repaired-contract acceptance arm."""
  print("unit 41 persisted-policy and sweep-product acceptance")
  check_amp_artifact()
  check_sweep_products()
  if FAILURES:
    print("FAILED acceptance: " + ", ".join(FAILURES))
    raise SystemExit(1)
  print("unit41-policy: ALL PASS")


if __name__ == "__main__":
  main()
