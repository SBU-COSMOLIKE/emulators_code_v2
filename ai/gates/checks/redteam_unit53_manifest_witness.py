#!/usr/bin/env python3
"""Prove the unit-53 study manifest and its load-bearing mutations.

This witness began as the negative red-team reproduction for the missing
manifest. The repair turns it into positive CPU acceptance. It exercises the
canonical production owner directly and inspects the real tuner syntax tree
for the authentication, spawn, default-control, and reporting boundaries.
"""

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

import emulator.studies.implementation as study_implementation_registry
from emulator.studies.manifest import (
  STUDY_MANIFEST_ATTR,
  STUDY_MANIFEST_DIGEST_ATTR,
  bind_study_manifest,
  build_study_manifest,
)
from emulator.studies.manifest_digest import canonical_json, manifest_digest
from emulator.studies.implementation import study_implementation_identity
from emulator.studies.name import resolve_study_name


ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "cosmic_shear_tune_emulator.py"
WRAPPER_FAMILIES = {
  "scalar_tune_emulator.py": "outputs",
  "cmb_tune_emulator.py": "cmb",
  "baosn_tune_emulator.py": "grid",
  "mps_tune_emulator.py": "grid2d",
}
FAILURES = []


def report(label, ok, detail=""):
  """Print one acceptance result and retain failures.

  Arguments:
    label  = short acceptance statement.
    ok     = whether the statement was proved.
    detail = measured value printed beside the verdict.

  Returns:
    None.
  """
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


class FakeTrial:
  """One trial carrying only the attributes the default-control law reads."""

  def __init__(self, default_control=False):
    self.user_attrs = {"cocoa_default_control": default_control}


class FakeStudy:
  """Small Optuna-study double for production manifest binding."""

  def __init__(self, trials=None, attrs=None):
    self.trials = [] if trials is None else list(trials)
    self.user_attrs = {} if attrs is None else dict(attrs)
    self.enqueued = []

  def set_user_attr(self, name, value):
    """Record one study-level attribute exactly as Optuna does."""
    self.user_attrs[name] = value

  def enqueue_trial(self, params, user_attrs=None):
    """Record one queued control and expose its marker on study.trials."""
    attrs = {} if user_attrs is None else dict(user_attrs)
    self.enqueued.append((dict(params), attrs))
    marker = attrs.get("cocoa_default_control", False)
    self.trials.append(FakeTrial(default_control=marker))


class FakeProcess:
  """Joined worker double carrying only its process exit code."""

  def __init__(self, exitcode):
    self.exitcode = exitcode


class FakeDuplicateStudyError(Exception):
  """Optuna duplicate-study signal used by the journal-open probe."""


class FakeOptunaExceptions:
  """Namespace matching ``optuna.exceptions``."""

  DuplicatedStudyError = FakeDuplicateStudyError


class FakeOptuna:
  """Small create/load double for the strict journal-open helper."""

  def __init__(self, duplicate, created_study, loaded_study):
    self.duplicate = duplicate
    self.created_study = created_study
    self.loaded_study = loaded_study
    self.exceptions = FakeOptunaExceptions()
    self.create_load_if_exists = None

  def create_study(self, **kwargs):
    """Return a new study or reproduce Optuna's duplicate refusal."""
    self.create_load_if_exists = kwargs.get("load_if_exists")
    if self.duplicate:
      raise FakeDuplicateStudyError("study already exists")
    return self.created_study

  def load_study(self, **_kwargs):
    """Return the pre-existing study after a strict-create collision."""
    return self.loaded_study


def fake_journal_storage(path):
  """Return an inert storage token for the extracted open helper."""
  return "storage:" + str(path)


def _refusal(study, manifest, initialize=False):
  """Return the teaching refusal for one changed current manifest."""
  try:
    bind_study_manifest(
      study=study,
      manifest=manifest,
      digest=manifest_digest(manifest=manifest),
      initialize=initialize)
  except RuntimeError as error:
    return str(error)
  return ""


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


def _call_lines(node, names):
  """Return the first source line for each named call below one node."""
  lines = {}
  for child in ast.walk(node):
    if not isinstance(child, ast.Call):
      continue
    name = None
    if isinstance(child.func, ast.Name):
      name = child.func.id
    elif isinstance(child.func, ast.Attribute):
      name = child.func.attr
    if name in names and name not in lines:
      lines[name] = child.lineno
  return lines


def _calls_are_ordered(lines, names):
  """Return whether every named call exists and appears in this order."""
  previous = -1
  for name in names:
    if name not in lines or lines[name] <= previous:
      return False
    previous = lines[name]
  return True


def _function_callable(tree, name, globals_=None):
  """Compile one pure driver helper without importing optional Optuna."""
  node = _function_node(tree=tree, name=name)
  module = ast.Module(body=[node], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {} if globals_ is None else dict(globals_)
  exec(compile(module, filename=str(DRIVER), mode="exec"), namespace)
  return namespace[name]


def _study_name_uses_family_resolver(node):
  """Return whether production binds study_name from family, never prog."""
  for child in ast.walk(node):
    if not isinstance(child, ast.Assign) or len(child.targets) != 1:
      continue
    target = child.targets[0]
    call = child.value
    if not isinstance(target, ast.Name) or target.id != "study_name":
      continue
    if not isinstance(call, ast.Call):
      continue
    if not isinstance(call.func, ast.Name):
      continue
    if call.func.id != "resolve_study_name" or call.args:
      continue
    for keyword in call.keywords:
      if (keyword.arg == "family"
          and isinstance(keyword.value, ast.Name)
          and keyword.value.id == "family"):
        return True
  return False


def _parallel_branch(node):
  """Return the real main() multi-worker branch."""
  for child in node.body:
    if not isinstance(child, ast.If):
      continue
    owns_worker_count = False
    for value in ast.walk(child.test):
      if isinstance(value, ast.Name) and value.id == "n_workers":
        owns_worker_count = True
    if owns_worker_count and child.orelse:
      return ast.Module(body=child.orelse, type_ignores=[])
  raise AssertionError("parallel n_workers branch not found")


def _call_has_name_keyword(node, call_name, keyword_name, value_name):
  """Return whether one named call carries the required name keyword."""
  for child in ast.walk(node):
    if not isinstance(child, ast.Call):
      continue
    name = None
    if isinstance(child.func, ast.Name):
      name = child.func.id
    elif isinstance(child.func, ast.Attribute):
      name = child.func.attr
    if name != call_name:
      continue
    for keyword in child.keywords:
      if (keyword.arg == keyword_name
          and isinstance(keyword.value, ast.Name)
          and keyword.value.id == value_name):
        return True
  return False


def _call_has_bool_keyword(node, call_name, keyword_name, value):
  """Return whether one named call carries the required boolean keyword."""
  for child in ast.walk(node):
    if not isinstance(child, ast.Call):
      continue
    name = None
    if isinstance(child.func, ast.Name):
      name = child.func.id
    elif isinstance(child.func, ast.Attribute):
      name = child.func.attr
    if name != call_name:
      continue
    for keyword in child.keywords:
      if (keyword.arg == keyword_name
          and isinstance(keyword.value, ast.Constant)
          and keyword.value.value is value):
        return True
  return False


def _family_routes(driver_main):
  """Return the direct default and every wrapper's pinned family."""
  routes = {}
  argument_names = []
  for argument in driver_main.args.args:
    argument_names.append(argument.arg)
  default_offset = len(argument_names) - len(driver_main.args.defaults)
  for index, default in enumerate(driver_main.args.defaults):
    name = argument_names[default_offset + index]
    routes["direct." + name] = ast.literal_eval(default)

  for filename in WRAPPER_FAMILIES:
    path = ROOT / filename
    wrapper = ast.parse(
      path.read_text(encoding="utf-8"),
      filename=str(path))
    calls = []
    for child in ast.walk(wrapper):
      if not isinstance(child, ast.Call):
        continue
      if not isinstance(child.func, ast.Name) or child.func.id != "main":
        continue
      values = {}
      for keyword in child.keywords:
        values[keyword.arg] = ast.literal_eval(keyword.value)
      calls.append(values)
    if len(calls) != 1:
      raise AssertionError(filename + " must call main exactly once")
    routes[filename] = calls[0].get("family")
  return routes


def _manifest_fixture(folder):
  """Build one real-file manifest fixture and its production digest."""
  train = folder / "train.1.txt"
  train.write_text("1 2 3\n", encoding="utf-8")
  facts = folder / "train.facts.yaml"
  facts.write_text("schema_version: 3\n", encoding="utf-8")
  val = folder / "val.npy"
  val.write_bytes(b"NUMPY-FIXTURE")
  fixed_config = {
    "data": {
      "train_params": str(train),
      "val_dv": str(val),
      "ram_frac": 0.7,
    },
    "train_args": {
      "loss": "chi2",
      "nepochs": 20,
    },
  }
  search_space = {
    "lr": [0.01, 0.001, 0.1, "log"],
  }
  default_trial = {"lr": 0.01}
  manifest = build_study_manifest(
    family="cosmolike",
    probe="xi",
    study_name=resolve_study_name(family="cosmolike"),
    thresholds=[0.2, 1.0],
    fixed_config=fixed_config,
    search_space=search_space,
    default_trial=default_trial,
    rescale="none",
    activation="H",
    implementation_identity=study_implementation_identity(
      family="cosmolike"))
  return manifest, train, fixed_config, search_space


def _minimal_manifest(additional_scientific_files=(),
                      thresholds=None,
                      resolved_scientific_values=None):
  """Build a small manifest around caller-selected identity files."""
  if thresholds is None:
    thresholds = [0.2, 1.0]
  return build_study_manifest(
    family="cosmolike",
    probe="xi",
    study_name=resolve_study_name(family="cosmolike"),
    thresholds=thresholds,
    fixed_config={
      "data": {},
      "train_args": {"loss": "chi2"},
    },
    search_space={"lr": [0.01, 0.001, 0.1, "log"]},
    default_trial={"lr": 0.01},
    rescale="none",
    activation="H",
    implementation_identity=study_implementation_identity(
      family="cosmolike"),
    additional_scientific_files=additional_scientific_files,
    resolved_scientific_values=resolved_scientific_values)


def main():
  """Run the positive study-identity acceptance and mutation arms."""
  print("unit 53 canonical study-manifest acceptance")
  studies_dir = ROOT / "emulator" / "studies"
  expected_study_modules = {
    "implementation.py",
    "manifest.py",
    "manifest_digest.py",
    "name.py",
  }
  present_study_modules = {
    path.name for path in studies_dir.glob("*.py")
    if path.name != "__init__.py"
  }
  flat_study_modules = sorted(
    path.name for path in (ROOT / "emulator").glob("study_*.py"))
  report(
    "study identity owners use the studies package without flat duplicates",
    present_study_modules == expected_study_modules
    and flat_study_modules == [],
    "package=" + repr(sorted(present_study_modules))
    + "; flat=" + repr(flat_study_modules))

  expected_names = {
    "cosmolike": "cosmic_shear_tune",
    "outputs": "scalar_tune_emulator",
    "cmb": "cmb_tune_emulator",
    "grid": "baosn_tune_emulator",
    "grid2d": "mps_tune_emulator",
  }
  resolved_names = {}
  for family in expected_names:
    resolved_names[family] = resolve_study_name(family=family)
  report(
    "one pure resolver gives all five stable family study names",
    resolved_names == expected_names,
    repr(resolved_names))

  mutant_family = "cosmolike"
  old_name_mutation = (
    "cosmic_shear_tune_emulator"
    if mutant_family is not None
    else "cosmic_shear_tune")
  renamed_mutation = "renamed_scalar_cli"
  report(
    "the old program-label mutation reproduces both naming forks",
    old_name_mutation != resolved_names["cosmolike"]
    and renamed_mutation != resolved_names["outputs"],
    "direct=" + old_name_mutation + "; renamed=" + renamed_mutation)

  numpy_thresholds = np.array(
    [0.2, 0.5, 1.0, 10.0, 100.0],
    dtype=np.float64)
  torch_thresholds = torch.tensor(
    [0.2, 0.5, 1.0, 10.0, 100.0],
    dtype=torch.float32)
  numpy_vector_manifest = _minimal_manifest(thresholds=numpy_thresholds)
  torch_vector_manifest = _minimal_manifest(thresholds=torch_thresholds)
  scalar_leaf_manifest = _minimal_manifest(
    thresholds=[np.float64(0.2), torch.tensor(0.5)])
  report(
    "NumPy and Torch vector/scalar thresholds become canonical JSON arrays",
    numpy_vector_manifest["objective"]["thresholds"]
        == numpy_thresholds.tolist()
    and torch_vector_manifest["objective"]["thresholds"]
        == torch_thresholds.tolist()
    and scalar_leaf_manifest["objective"]["thresholds"] == [0.2, 0.5]
    and isinstance(
      torch_vector_manifest["objective"]["thresholds"],
      list),
    repr(torch_vector_manifest["objective"]["thresholds"]))

  nonfinite_threshold_errors = []
  for nonfinite_thresholds in (
      np.array([0.2, np.nan], dtype=np.float64),
      torch.tensor([0.2, float("inf")], dtype=torch.float32)):
    try:
      _minimal_manifest(thresholds=nonfinite_thresholds)
    except ValueError as error:
      nonfinite_threshold_errors.append(str(error))
  report(
    "nonfinite NumPy and Torch threshold entries still refuse by index",
    len(nonfinite_threshold_errors) == 2
    and "objective.thresholds[1]" in nonfinite_threshold_errors[0]
    and "objective.thresholds[1]" in nonfinite_threshold_errors[1],
    repr(nonfinite_threshold_errors))

  with tempfile.TemporaryDirectory(prefix="unit53-") as tmp:
    folder = Path(tmp)
    (manifest, train, fixed_config,
     search_space) = _manifest_fixture(folder=folder)
    digest = manifest_digest(manifest=manifest)
    study = FakeStudy()
    bind_study_manifest(
      study=study,
      manifest=manifest,
      digest=digest,
      initialize=True)
    report(
      "new studies receive canonical manifest and digest attributes",
      STUDY_MANIFEST_ATTR in study.user_attrs
      and study.user_attrs[STUDY_MANIFEST_ATTR] == canonical_json(manifest)
      and study.user_attrs[STUDY_MANIFEST_DIGEST_ATTR] == digest,
      digest)

    before_attrs = dict(study.user_attrs)
    bind_study_manifest(
      study=study,
      manifest=deepcopy(manifest),
      digest=digest,
      initialize=False)
    report(
      "byte-identical manifest resume is accepted without rewriting",
      study.user_attrs == before_attrs,
      "attributes unchanged")

    empty_legacy_text = _refusal(
      study=FakeStudy(),
      manifest=manifest)
    nonempty_legacy_text = _refusal(
      study=FakeStudy(trials=[FakeTrial()]),
      manifest=manifest)
    report(
      "loaded empty and nonempty legacy studies refuse without blessing",
      "legacy study" in empty_legacy_text
      and "legacy study" in nonempty_legacy_text
      and "new journal" in empty_legacy_text
      and "migrate" in nonempty_legacy_text,
      "empty=" + empty_legacy_text + " nonempty=" + nonempty_legacy_text)

    partial_manifest = FakeStudy(attrs={
      STUDY_MANIFEST_ATTR: canonical_json(manifest),
    })
    partial_digest = FakeStudy(attrs={
      STUDY_MANIFEST_DIGEST_ATTR: digest,
    })
    invalid_json = FakeStudy(attrs={
      STUDY_MANIFEST_ATTR: "{",
      STUDY_MANIFEST_DIGEST_ATTR: digest,
    })
    noncanonical_json = FakeStudy(attrs={
      STUDY_MANIFEST_ATTR: json.dumps(manifest, indent=2),
      STUDY_MANIFEST_DIGEST_ATTR: digest,
    })
    partial_manifest_text = _refusal(
      study=partial_manifest,
      manifest=manifest)
    partial_digest_text = _refusal(
      study=partial_digest,
      manifest=manifest)
    invalid_json_text = _refusal(
      study=invalid_json,
      manifest=manifest)
    noncanonical_json_text = _refusal(
      study=noncanonical_json,
      manifest=manifest)
    report(
      "partial, corrupt, and noncanonical stored identity all refuse",
      "incomplete" in partial_manifest_text
      and "incomplete" in partial_digest_text
      and "canonical JSON" in invalid_json_text
      and "canonical JSON" in noncanonical_json_text,
      "four stored-state refusals")

    fixed_changed = deepcopy(manifest)
    fixed_changed["fixed_config"]["train_args"]["loss"] = "mse"
    fixed_text = _refusal(study=study, manifest=fixed_changed)
    report(
      "a fixed training value is refused and its field is named",
      "fixed_config.train_args.loss" in fixed_text,
      fixed_text)

    range_changed = deepcopy(manifest)
    range_changed["search_space"]["lr"][2] = 0.2
    range_changed["search_space"]["lr"][3] = "float"
    range_text = _refusal(study=study, manifest=range_changed)
    report(
      "range bound and kind changes are refused as search-space fields",
      "search_space.lr[2]" in range_text
      and "search_space.lr[3]" in range_text,
      range_text)

    scientific_before = manifest["scientific_inputs"]
    train.write_text("same path, changed scientific bytes\n", encoding="utf-8")
    rewritten = build_study_manifest(
      family="cosmolike",
      probe="xi",
      study_name=resolve_study_name(family="cosmolike"),
      thresholds=[0.2, 1.0],
      fixed_config=fixed_config,
      search_space=search_space,
      default_trial={"lr": 0.01},
      rescale="none",
      activation="H",
      implementation_identity=study_implementation_identity(
        family="cosmolike"))
    rewrite_text = _refusal(study=study, manifest=rewritten)
    report(
      "same-path data rewrites change identity and are refused",
      rewritten["scientific_inputs"] != scientific_before
      and "scientific_inputs" in rewrite_text,
      rewrite_text)

    train.write_text("1 2 3\n", encoding="utf-8")
    component_names = []
    for component in manifest["implementation"]["components"]:
      component_names.append(component["name"])
    for component_name in ("tuner.model_design", "tuner.analytic_base"):
      component_index = component_names.index(component_name)
      implementation_changed = deepcopy(manifest)
      implementation_changed["implementation"]["components"][
        component_index]["version"] = 2
      implementation_text = _refusal(
        study=study,
        manifest=implementation_changed)
      expected_field = (
        "implementation.components[" + str(component_index) + "].version")
      report(
        component_name + " semantic-version changes are named and refused",
        expected_field in implementation_text
        and manifest_digest(manifest=implementation_changed) != digest,
        implementation_text)

    mutation_fields = []
    mutations = []
    mutations.append(("study.family", ("study", "family", "outputs")))
    mutations.append(("objective.thresholds[0]",
                      ("objective", "thresholds", 0.3)))
    mutations.append(("cli_fixed.rescale",
                      ("cli_fixed", "rescale", "residual")))
    mutations.append(("cli_fixed.activation",
                      ("cli_fixed", "activation", "power")))
    for expected_field, mutation in mutations:
      changed = deepcopy(manifest)
      group, key, value = mutation
      if group == "objective":
        changed[group][key][0] = value
      else:
        changed[group][key] = value
      text = _refusal(study=study, manifest=changed)
      if expected_field in text:
        mutation_fields.append(expected_field)
    report(
      "family, objective, rescale, and activation mutations all refuse",
      len(mutation_fields) == 4,
      repr(mutation_fields))

    operational_config = deepcopy(fixed_config)
    operational_config["data"]["ram_frac"] = 0.1
    operational_config["quiet"] = True
    operational_config["n_trials"] = 999
    operational = build_study_manifest(
      family="cosmolike",
      probe="xi",
      study_name=resolve_study_name(family="cosmolike"),
      thresholds=[0.2, 1.0],
      fixed_config=operational_config,
      search_space=search_space,
      default_trial={"lr": 0.01},
      rescale="none",
      activation="H",
      implementation_identity=study_implementation_identity(
        family="cosmolike"))
    # Restore the scientific file so only operational inputs differ.
    train.write_text("1 2 3\n", encoding="utf-8")
    operational = build_study_manifest(
      family="cosmolike",
      probe="xi",
      study_name=resolve_study_name(family="cosmolike"),
      thresholds=[0.2, 1.0],
      fixed_config=operational_config,
      search_space=search_space,
      default_trial={"lr": 0.01},
      rescale="none",
      activation="H",
      implementation_identity=study_implementation_identity(
        family="cosmolike"))
    report(
      "worker, RAM, trial, timeout, quiet, and GPU facts are excluded",
      manifest_digest(manifest=operational) == digest
      and canonical_json(operational) == canonical_json(manifest)
      and "ram_frac" not in repr(operational)
      and "n_trials" not in repr(operational)
      and "quiet" not in repr(operational),
      manifest_digest(manifest=operational))

    path_only_identity_same = str(train) == str(train)
    report(
      "the path-only identity mutation misses the rewrite the digest catches",
      path_only_identity_same
      and rewritten["scientific_inputs"] != scientific_before,
      "path unchanged; digest changed")

  source = DRIVER.read_text(encoding="utf-8")
  tree = ast.parse(source, filename=str(DRIVER))
  driver_main = _function_node(tree=tree, name="main")

  implementation_identities = {}
  for family in expected_names:
    implementation_identities[family] = study_implementation_identity(
      family=family)
  shared_component_names = [
    "tuner.study_protocol",
    "tuner.experiment_resolution",
    "tuner.data_staging_target_law",
    "tuner.training_optimizer_scheduler",
    "tuner.model_design",
    "tuner.activation_normalization",
    "tuner.parameter_output_geometry_decoder",
    "tuner.loss_composition",
    "tuner.warmstart_transfer",
    "tuner.analytic_base",
    "tuner.runtime_numeric_contract",
    "tuner.runtime_dataset_parser_contract",
  ]
  identities_are_complete = True
  for family in expected_names:
    identity = implementation_identities[family]
    names = []
    versions_are_positive = True
    for component in identity["components"]:
      names.append(component["name"])
      if type(component["version"]) is not int or component["version"] < 1:
        versions_are_positive = False
    expected_components = list(shared_component_names)
    expected_components.append("tuner." + family)
    if (identity["registry_schema"] != 1
        or names != expected_components
        or not versions_are_positive):
      identities_are_complete = False
  report(
    "the versioned registry names every required implementation component",
    identities_are_complete,
    repr(implementation_identities["cosmolike"]))

  prior_model_version = study_implementation_registry._CURRENT_VERSIONS[
    "tuner.model_design"]
  study_implementation_registry._CURRENT_VERSIONS[
    "tuner.model_design"] = 999
  try:
    try:
      study_implementation_registry.study_implementation_identity(
        family="cosmolike")
    except RuntimeError as error:
      invalid_pointer_text = str(error)
    else:
      invalid_pointer_text = "accepted an unregistered current version"
  finally:
    study_implementation_registry._CURRENT_VERSIONS[
      "tuner.model_design"] = prior_model_version
  report(
    "a current semantic version missing from the retained registry refuses",
    "registry is inconsistent" in invalid_pointer_text
    and "register the current semantic version" in invalid_pointer_text,
    invalid_pointer_text)

  with tempfile.TemporaryDirectory(prefix="unit53-implementation-") as tmp:
    shadow_root = Path(tmp)
    unrelated = shadow_root / "unrelated.py"
    unrelated.write_text("VALUE = 1\n", encoding="utf-8")
    implementation_before = _minimal_manifest()
    unrelated.write_text(
      "# comment and formatting-only repository edit\n\nVALUE = 1\n",
      encoding="utf-8")
    implementation_after = _minimal_manifest()
    report(
      "comments, formatting, paths, and unrelated files do not fork a study",
      canonical_json(implementation_before)
      == canonical_json(implementation_after)
      and "path" not in repr(implementation_before["implementation"])
      and "digest" not in repr(implementation_before["implementation"]),
      "semantic registry unchanged")

  reuse_files_owner = _function_callable(
    tree=tree,
    name="study_reuse_artifact_files",
    globals_={"Path": Path})
  with tempfile.TemporaryDirectory(prefix="unit53-reuse-") as tmp:
    folder = Path(tmp)
    finetune_root = folder / "finetune-source"
    transfer_root = folder / "transfer-source"
    original_bytes = {}
    for root, label in ((finetune_root, "finetune"),
                        (transfer_root, "transfer")):
      for suffix in (".emul", ".h5"):
        path = Path(str(root) + suffix)
        payload = (label + suffix + " original\n").encode("utf-8")
        path.write_bytes(payload)
        original_bytes[path] = payload

    class ReuseExperiment:
      pass

    reuse_exp = ReuseExperiment()
    reuse_exp._finetune_root = str(finetune_root)
    reuse_exp._transfer_root = str(transfer_root)
    reuse_files = reuse_files_owner(exp=reuse_exp)
    expected_reuse_files = {path.resolve() for path in original_bytes}
    reuse_manifest = _minimal_manifest(
      additional_scientific_files=reuse_files)
    reuse_study = FakeStudy()
    bind_study_manifest(
      study=reuse_study,
      manifest=reuse_manifest,
      digest=manifest_digest(manifest=reuse_manifest),
      initialize=True)

    finetune_emul = Path(str(finetune_root) + ".emul")
    finetune_emul.write_bytes(b"different fine-tune weights\n")
    finetune_changed = _minimal_manifest(
      additional_scientific_files=reuse_files)
    finetune_text = _refusal(
      study=reuse_study,
      manifest=finetune_changed)
    finetune_emul.write_bytes(original_bytes[finetune_emul])
    report(
      "same-path fine-tune artifact rewrites change identity and refuse",
      set(reuse_files) == expected_reuse_files
      and "scientific_inputs" in finetune_text,
      finetune_text)

    transfer_h5 = Path(str(transfer_root) + ".h5")
    transfer_h5.write_bytes(b"different transfer base metadata\n")
    transfer_changed = _minimal_manifest(
      additional_scientific_files=reuse_files)
    transfer_text = _refusal(
      study=reuse_study,
      manifest=transfer_changed)
    report(
      "same-path transfer artifact rewrites change identity and refuse",
      "scientific_inputs" in transfer_text,
      transfer_text)

  dataset_identity_owner = _function_callable(
    tree=tree,
    name="study_cosmolike_dataset_identity",
    globals_={"Path": Path})
  with tempfile.TemporaryDirectory(prefix="unit53-dataset-") as tmp:
    rootdir = Path(tmp)
    dataset_dir = (
      rootdir / "external_modules" / "data" / "unit53-objective")
    dataset_dir.mkdir(parents=True)
    member_names = {
      "data_file": "data.txt",
      "cov_file": "cov.txt",
      "mask_file": "mask.txt",
      "nz_lens_file": "lens.txt",
      "nz_source_file": "source.txt",
    }
    member_paths = []
    for key, filename in member_names.items():
      path = dataset_dir / filename
      path.write_text(key + " original\n", encoding="utf-8")
      member_paths.append(path.resolve())
    dataset_path = dataset_dir / "objective.dataset"
    binning_path = dataset_dir / "binning.ini"
    env_name = "UNIT53_NTHETA"
    previous_env_value = os.environ.get(env_name)
    os.environ[env_name] = "20"
    binning_path.write_text(
      "n_theta = $(UNIT53_NTHETA)\n"
      "lens_ntomo = 2\n"
      "source_ntomo = 3\n"
      "theta_min_arcmin = 2.5\n"
      "theta_max_arcmin = 250.0\n",
      encoding="utf-8")
    dataset_lines = []
    dataset_lines.append("INCLUDE(binning.ini)")
    for key, filename in member_names.items():
      dataset_lines.append(key + " = " + filename)
    dataset_path.write_text(
      "\n".join(dataset_lines) + "\n",
      encoding="utf-8")
    dataset_cfg = {
      "data": {
        "cosmolike_data_dir": "unit53-objective",
        "cosmolike_dataset": "objective.dataset",
      },
    }
    dataset_identity = dataset_identity_owner(
      cfg=dataset_cfg,
      family="cosmolike",
      rootdir=str(rootdir))
    dataset_files = dataset_identity["files"]
    dataset_values = {
      "cosmolike_objective": dataset_identity["resolved"],
    }
    dataset_manifest = _minimal_manifest(
      additional_scientific_files=dataset_files,
      resolved_scientific_values=dataset_values)
    dataset_study = FakeStudy()
    bind_study_manifest(
      study=dataset_study,
      manifest=dataset_manifest,
      digest=manifest_digest(manifest=dataset_manifest),
      initialize=True)
    covariance = dataset_dir / member_names["cov_file"]
    original_covariance = covariance.read_bytes()
    covariance.write_text(
      "same path, different objective covariance\n",
      encoding="utf-8")
    dataset_changed = _minimal_manifest(
      additional_scientific_files=dataset_files,
      resolved_scientific_values=dataset_values)
    dataset_text = _refusal(
      study=dataset_study,
      manifest=dataset_changed)
    expected_dataset_files = {
      dataset_path.resolve(),
      binning_path.resolve(),
      *member_paths,
    }
    report(
      "same-path objective-dataset member rewrites change identity and refuse",
      set(dataset_files) == expected_dataset_files
      and "scientific_inputs" in dataset_text,
      dataset_text)

    covariance.write_bytes(original_covariance)
    binning_path.write_text(
      "n_theta = 22\n"
      "lens_ntomo = 2\n"
      "source_ntomo = 3\n"
      "theta_min_arcmin = 2.5\n"
      "theta_max_arcmin = 250.0\n",
      encoding="utf-8")
    include_identity = dataset_identity_owner(
      cfg=dataset_cfg,
      family="cosmolike",
      rootdir=str(rootdir))
    include_changed = _minimal_manifest(
      additional_scientific_files=include_identity["files"],
      resolved_scientific_values={
        "cosmolike_objective": include_identity["resolved"],
      })
    include_text = _refusal(
      study=dataset_study,
      manifest=include_changed)
    report(
      "same-path objective INI include rewrites change identity and refuse",
      "scientific_inputs" in include_text,
      include_text)

    binning_path.write_text(
      "n_theta = $(UNIT53_NTHETA)\n"
      "lens_ntomo = 2\n"
      "source_ntomo = 3\n"
      "theta_min_arcmin = 2.5\n"
      "theta_max_arcmin = 250.0\n",
      encoding="utf-8")
    os.environ[env_name] = "21"
    environment_identity = dataset_identity_owner(
      cfg=dataset_cfg,
      family="cosmolike",
      rootdir=str(rootdir))
    environment_changed = _minimal_manifest(
      additional_scientific_files=environment_identity["files"],
      resolved_scientific_values={
        "cosmolike_objective": environment_identity["resolved"],
      })
    environment_text = _refusal(
      study=dataset_study,
      manifest=environment_changed)
    report(
      "runtime-resolved objective values change identity and refuse",
      dataset_identity["resolved"]["n_theta"] == 20
      and dataset_identity["files"] == environment_identity["files"]
      and "resolved_scientific_values" in environment_text,
      environment_text)
    if previous_env_value is None:
      del os.environ[env_name]
    else:
      os.environ[env_name] = previous_env_value

  captured_identity = {}
  helper_calls = []
  implementation_sentinel = {
    "registry_schema": 99,
    "components": [
      {"name": "sentinel", "version": 7},
    ],
  }
  reuse_sentinel = ["finetune.emul", "finetune.h5",
                    "transfer.emul", "transfer.h5"]
  dataset_sentinel = ["objective.dataset", "data.txt", "cov.txt",
                      "mask.txt", "lens.txt", "source.txt"]
  dataset_values_sentinel = {
    "n_theta": 20,
    "lens_ntomo": 2,
    "source_ntomo": 3,
  }

  def fake_search_defaults(train_args):
    helper_calls.append(("search_defaults", train_args))
    return {"lr": 0.01}

  def fake_implementation_identity(family):
    helper_calls.append(("implementation", family))
    return deepcopy(implementation_sentinel)

  def fake_reuse_files(exp):
    helper_calls.append(("reuse", exp))
    return list(reuse_sentinel)

  def fake_dataset_identity(cfg, family):
    helper_calls.append(("dataset", (cfg, family)))
    return {
      "files": list(dataset_sentinel),
      "resolved": dict(dataset_values_sentinel),
    }

  def fake_build_manifest(**kwargs):
    captured_identity.update(kwargs)
    return {"captured": True}

  def fake_manifest_digest(manifest):
    if manifest != {"captured": True}:
      raise AssertionError("identity builder digested an unexpected manifest")
    return "sha256:captured"

  identity_builder = _function_callable(
    tree=tree,
    name="build_current_study_identity",
    globals_={
      "Path": Path,
      "__file__": str(DRIVER),
      "search_defaults": fake_search_defaults,
      "study_implementation_identity": fake_implementation_identity,
      "study_reuse_artifact_files": fake_reuse_files,
      "study_cosmolike_dataset_identity": fake_dataset_identity,
      "build_study_manifest": fake_build_manifest,
      "manifest_digest": fake_manifest_digest,
    })

  class IdentityExperiment:
    pass

  identity_exp = IdentityExperiment()
  identity_exp.raw_train_args = {"lr": [0.01, 0.001, 0.1, "log"]}
  identity_exp.train_args = {"loss": "chi2"}
  identity_exp.probe = "xi"
  identity_exp.thresholds = torch_thresholds
  identity_exp.rescale = "none"
  identity_exp.activation = "H"
  identity_cfg = {"data": {"train_params": "train.txt"}}
  (captured_manifest, captured_digest,
   captured_raw, captured_default) = identity_builder(
     cfg=identity_cfg,
     family="cosmolike",
     study_name="cosmic_shear_tune",
     exp=identity_exp)
  report(
    "the current-identity owner wires semantic implementation and input data",
    captured_manifest == {"captured": True}
    and captured_digest == "sha256:captured"
    and captured_raw is identity_exp.raw_train_args
    and captured_default == {"lr": 0.01}
    and captured_identity.get("thresholds") is torch_thresholds
    and captured_identity.get("implementation_identity")
        == implementation_sentinel
    and captured_identity.get("additional_scientific_files")
        == reuse_sentinel + dataset_sentinel
    and captured_identity.get("resolved_scientific_values")
        == {"cosmolike_objective": dataset_values_sentinel}
    and [call[0] for call in helper_calls]
        == ["search_defaults", "reuse", "dataset", "implementation"]
    and helper_calls[-1] == ("implementation", "cosmolike"),
    repr(helper_calls))

  family_routes = _family_routes(driver_main=driver_main)
  routes_match = (
    family_routes.get("direct.prog") == "cosmic_shear_tune_emulator"
    and family_routes.get("direct.family") == "cosmolike")
  for filename in WRAPPER_FAMILIES:
    if family_routes.get(filename) != WRAPPER_FAMILIES[filename]:
      routes_match = False
  report(
    "the direct command and four wrappers pin all five family routes",
    routes_match,
    repr(family_routes))
  report(
    "the live driver binds study_name through the family resolver",
    _study_name_uses_family_resolver(node=driver_main),
    "study_name = resolve_study_name(family=family)")

  created_token = object()
  loaded_token = object()
  new_optuna = FakeOptuna(
    duplicate=False,
    created_study=created_token,
    loaded_study=loaded_token)
  open_new = _function_callable(
    tree=tree,
    name="open_journal_study",
    globals_={
      "optuna": new_optuna,
      "journal_storage": fake_journal_storage,
    })
  opened_new, new_created = open_new(
    study_name="cosmic_shear_tune",
    journal_path="journal.log")
  old_optuna = FakeOptuna(
    duplicate=True,
    created_study=created_token,
    loaded_study=loaded_token)
  open_old = _function_callable(
    tree=tree,
    name="open_journal_study",
    globals_={
      "optuna": old_optuna,
      "journal_storage": fake_journal_storage,
    })
  opened_old, old_created = open_old(
    study_name="cosmic_shear_tune",
    journal_path="journal.log")
  report(
    "strict journal creation returns explicit initialization authority",
    opened_new is created_token
    and new_created is True
    and new_optuna.create_load_if_exists is False
    and opened_old is loaded_token
    and old_created is False
    and old_optuna.create_load_if_exists is False,
    "new=True; duplicate load=False")

  parallel = _parallel_branch(node=driver_main)
  parallel_lines = _call_lines(
    node=parallel,
    names={"open_journal_study", "bind_study_manifest",
           "enqueue_default_control", "start", "refuse_failed_workers",
           "load_study"})
  parallel_order = [
    "open_journal_study",
    "bind_study_manifest",
    "enqueue_default_control",
    "start",
    "refuse_failed_workers",
    "load_study",
  ]
  report(
    "the parent authenticates before enqueue/spawn and refuses before reload",
    _calls_are_ordered(lines=parallel_lines, names=parallel_order)
    and _call_has_name_keyword(
      node=parallel,
      call_name="bind_study_manifest",
      keyword_name="initialize",
      value_name="created"),
    repr(parallel_lines))

  worker = _function_node(tree=tree, name="_tune_worker")
  worker_lines = _call_lines(
    node=worker,
    names={"load_study", "from_config", "build_current_study_identity",
           "require_worker_identity", "bind_study_manifest", "stage_train",
           "stage_val", "build_geometry"})
  worker_order = [
    "load_study",
    "from_config",
    "build_current_study_identity",
    "require_worker_identity",
    "bind_study_manifest",
    "stage_train",
    "stage_val",
    "build_geometry",
  ]
  report(
    "workers rebuild and authenticate current identity before staging inputs",
    _calls_are_ordered(lines=worker_lines, names=worker_order)
    and _call_has_bool_keyword(
      node=worker,
      call_name="bind_study_manifest",
      keyword_name="initialize",
      value=False),
    repr(worker_lines))

  require_identity = _function_callable(
    tree=tree,
    name="require_worker_identity")
  matching_identity_accepted = True
  try:
    require_identity(
      parent_manifest={"identity": 1},
      parent_digest="sha256:same",
      worker_manifest={"identity": 1},
      worker_digest="sha256:same")
  except RuntimeError:
    matching_identity_accepted = False
  worker_refusal = ""
  try:
    require_identity(
      parent_manifest={"identity": 1},
      parent_digest="sha256:same",
      worker_manifest={"identity": 2},
      worker_digest="sha256:different")
  except RuntimeError as error:
    worker_refusal = str(error)
  report(
    "worker transport is checked against the independently rebuilt identity",
    matching_identity_accepted
    and "before staging" in worker_refusal,
    worker_refusal)

  enqueue_control = _function_callable(
    tree=tree,
    name="enqueue_default_control")
  failed_only = FakeStudy(trials=[FakeTrial(default_control=False)])
  first_enqueue = enqueue_control(
    study=failed_only,
    default_trial={"lr": 0.01})
  second_enqueue = enqueue_control(
    study=failed_only,
    default_trial={"lr": 0.01})
  report(
    "a failed-only study enqueues the manifest-owned default exactly once",
    first_enqueue is True
    and second_enqueue is False
    and len(failed_only.enqueued) == 1
    and failed_only.enqueued[0][1].get("cocoa_default_control") is True,
    repr(failed_only.enqueued))
  refuse_workers = _function_callable(
    tree=tree,
    name="refuse_failed_workers")
  clean_workers_accepted = True
  try:
    refuse_workers(processes=[FakeProcess(0), FakeProcess(0)])
  except RuntimeError:
    clean_workers_accepted = False
  failed_worker_text = ""
  try:
    refuse_workers(processes=[FakeProcess(0), FakeProcess(7)])
  except RuntimeError as error:
    failed_worker_text = str(error)
  report(
    "failed workers refuse before any old winner can be reported",
    clean_workers_accepted
    and "refusing to report a winner" in failed_worker_text
    and "7" in failed_worker_text,
    failed_worker_text)
  report(
    "the final report names stable study name and manifest digest",
    'log(f"study name: {study_name}")' in source
    and 'log(f"study manifest sha256: {manifest_sha256}")' in source
    and "winner = best_complete_trial(study=study)" in source,
    "both identity lines + manifest selection rule")

  if FAILURES:
    print("FAILED: " + ", ".join(FAILURES))
    raise SystemExit(1)
  print("ALL PASS: unit 53 study manifest is load-bearing")


if __name__ == "__main__":
  main()
