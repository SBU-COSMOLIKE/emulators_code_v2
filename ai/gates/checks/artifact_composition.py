#!/usr/bin/env python3
"""artifact-composition: saved composition is declared, not inferred.

A rebuilt artifact must not decide that it is plain, NPCE, or transfer merely
because a payload group happens to be present.  Group presence is editable and
an absent group used to select the plain fallback.  The root declarations are
therefore authoritative, and the resolved record plus payload groups must
corroborate them in both directions.

This check owns the pure-HDF5 composition matrix.  It builds the four legal
rows and forges one contradiction class at a time: declarations, resolved
records, required groups, forbidden groups, and the refined-transfer marker.
It also verifies
that rebuild_emulator consults the composition reader before it reconstructs a
geometry or model, or loads weights.  No network is built and no torch file is
loaded by the fixtures in this check.
"""

import ast
import os
import sys
import tempfile


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)

FAILURES = []


def report(label, ok, detail=""):
  """Record one verdict while reserving one terminal for the whole contract."""
  print("  [" + ("PASS" if ok else "FAIL") + "] " + label
        + ("  (" + detail + ")" if detail else ""))
  if not ok:
    FAILURES.append(label)


def _resolved_record(mode, refined):
  """Return the authoritative resolved composition row for one fixture."""
  pce = None
  transfer = None
  if mode == "npce":
    pce = {"form": "residual"}
  elif mode == "transfer":
    transfer = {
      "form": "gain",
      "space": "physical",
      "refine": {"epochs": 1} if refined else None,
    }
  return {
    "composition_mode": mode,
    "transfer_refined": refined,
    "pce": pce,
    "transfer": transfer,
    "train_args": {},
    "data": {},
  }


def _raw_config(mode, refined):
  """Return realistic, corroborating raw config provenance for a fixture."""
  if mode == "npce":
    return {"pce": {"form": "residual"}}
  if mode == "transfer":
    block = {"form": "gain", "space": "physical"}
    if refined:
      block["refine"] = {"epochs": 1}
    return {"transfer": block}
  return {}


def _write_fixture(h5py, yaml, path, mode, refined):
  """Write only the HDF5 surfaces the composition reader owns."""
  str_dt = h5py.string_dtype(encoding="utf-8")
  with h5py.File(path, "w") as f:
    # Version 3 is intentional: an old v3 file may have all other schema facts
    # yet still lacks the new declarations, and must be refused rather than
    # treated as plain.
    f.attrs["schema_version"] = 3
    f.attrs["composition_mode"] = mode
    f.attrs["transfer_refined"] = refined
    f.create_dataset(
      "config_yaml",
      data=yaml.safe_dump(_raw_config(mode, refined), sort_keys=False),
      dtype=str_dt)
    f.create_dataset(
      "config_resolved_yaml",
      data=yaml.safe_dump(_resolved_record(mode, refined), sort_keys=False),
      dtype=str_dt)

    if mode == "npce":
      group = f.create_group("pce")
      group.attrs["form"] = "residual"
    elif mode == "transfer":
      group = f.create_group("transfer_base")
      group.attrs["form"] = "gain"
      group.attrs["space"] = "physical"
      if refined:
        group.create_group("drifted_state")


def _read_resolved(yaml, f):
  """Read the fixture's resolved mapping for one narrow in-place mutation."""
  return yaml.safe_load(f["config_resolved_yaml"][()])


def _replace_resolved(h5py, yaml, f, record):
  """Replace the resolved YAML dataset without touching another surface."""
  del f["config_resolved_yaml"]
  f.create_dataset(
    "config_resolved_yaml",
    data=yaml.safe_dump(record, sort_keys=False),
    dtype=h5py.string_dtype(encoding="utf-8"))


def _set_raw(key, value):
  """Return a mutator that adds one contradictory raw config declaration."""
  def mutate(h5py, yaml, f):
    record = yaml.safe_load(f["config_yaml"][()])
    record[key] = value
    del f["config_yaml"]
    f.create_dataset(
      "config_yaml",
      data=yaml.safe_dump(record, sort_keys=False),
      dtype=h5py.string_dtype(encoding="utf-8"))
  return mutate


def _set_payload_attr(group, key, value):
  """Return a mutator that changes one decoder-driving payload attribute."""
  def mutate(_h5py, _yaml, f):
    f[group].attrs[key] = value
  return mutate


def _set_raw_field(section, key, value):
  """Return a mutator that changes one nested raw provenance field."""
  def mutate(h5py, yaml, f):
    record = yaml.safe_load(f["config_yaml"][()])
    record[section][key] = value
    del f["config_yaml"]
    f.create_dataset(
      "config_yaml",
      data=yaml.safe_dump(record, sort_keys=False),
      dtype=h5py.string_dtype(encoding="utf-8"))
  return mutate


def _set_resolved_field(section, key, value):
  """Return a mutator that changes one nested resolved-record field."""
  def mutate(h5py, yaml, f):
    record = _read_resolved(yaml, f)
    record[section][key] = value
    _replace_resolved(h5py, yaml, f, record)
  return mutate


def _expect_accept(h5py, yaml, reader, directory, label, mode, refined):
  """Build one legal row and require the reader's exact tuple contract."""
  path = os.path.join(directory, "valid-" + label + ".h5")
  _write_fixture(h5py, yaml, path, mode, refined)
  try:
    with h5py.File(path, "r") as f:
      got = reader(f, path)
    ok = (type(got) is tuple and len(got) == 2
          and type(got[0]) is str and type(got[1]) is bool
          and got == (mode, refined))
    report(label + " accepted", ok,
           "returned " + repr(got) + ", expected "
           + repr((mode, refined)))
  except Exception as exc:
    report(label + " accepted", False,
           type(exc).__name__ + ": " + str(exc))


def _expect_refusal(
    h5py, yaml, reader, directory, index, label, mode, refined, mutate):
  """Forge one fixture fact and require a contextual load-boundary refusal."""
  path = os.path.join(directory, "forged-%02d.h5" % index)
  _write_fixture(h5py, yaml, path, mode, refined)
  with h5py.File(path, "r+") as f:
    mutate(h5py, yaml, f)

  refused = False
  detail = "reader accepted the forged artifact"
  try:
    with h5py.File(path, "r") as f:
      got = reader(f, path)
    detail += " and returned " + repr(got)
  except (KeyError, TypeError, ValueError) as exc:
    refused = path in str(exc)
    detail = type(exc).__name__ + ": " + str(exc)
    if not refused:
      detail += " [refusal omitted artifact location]"
  except Exception as exc:
    detail = ("unexpected " + type(exc).__name__ + ": " + str(exc))
  report(label + " refused", refused, detail)


def _delete(name):
  """Return an HDF5 mutator that deletes one root object."""
  def mutate(_h5py, _yaml, f):
    del f[name]
  return mutate


def _add_group(name):
  """Return an HDF5 mutator that adds one forbidden root group."""
  def mutate(_h5py, _yaml, f):
    f.create_group(name)
  return mutate


def _delete_attr(name):
  """Return an HDF5 mutator that deletes one root declaration."""
  def mutate(_h5py, _yaml, f):
    del f.attrs[name]
  return mutate


def _set_attr(name, value):
  """Return an HDF5 mutator that replaces one root declaration."""
  def mutate(_h5py, _yaml, f):
    f.attrs[name] = value
  return mutate


def _set_resolved(key, value):
  """Return a mutator that changes exactly one resolved top-level fact."""
  def mutate(h5py, yaml, f):
    record = _read_resolved(yaml, f)
    record[key] = value
    _replace_resolved(h5py, yaml, f, record)
  return mutate


def _delete_resolved(key):
  """Return a mutator that deletes one required resolved top-level fact."""
  def mutate(h5py, yaml, f):
    record = _read_resolved(yaml, f)
    del record[key]
    _replace_resolved(h5py, yaml, f, record)
  return mutate


def _old_v3(_h5py, yaml, f):
  """Forge a schema-v3 file from before composition facts existed."""
  del f.attrs["composition_mode"]
  del f.attrs["transfer_refined"]
  record = _read_resolved(yaml, f)
  for key in ("composition_mode", "transfer_refined", "pce", "transfer"):
    del record[key]
  # h5py is supplied to every mutator, but this one needs the dataset dtype
  # already present on the fixture before replacement.
  dtype = f["config_resolved_yaml"].dtype
  del f["config_resolved_yaml"]
  f.create_dataset(
    "config_resolved_yaml",
    data=yaml.safe_dump(record, sort_keys=False),
    dtype=dtype)


def _plain_marked_refined(h5py, yaml, f):
  """Make both declarations say refined while the mode remains plain."""
  f.attrs["transfer_refined"] = True
  record = _read_resolved(yaml, f)
  record["transfer_refined"] = True
  _replace_resolved(h5py, yaml, f, record)


def _refined_without_drifted(_h5py, _yaml, f):
  """Keep refined declarations but remove the required drifted payload."""
  del f["transfer_base/drifted_state"]


def _frozen_with_drifted(_h5py, _yaml, f):
  """Keep frozen declarations but add the forbidden drifted payload."""
  f["transfer_base"].create_group("drifted_state")


class _OuterCalls(ast.NodeVisitor):
  """Collect calls in rebuild_emulator while skipping its nested helpers."""

  def __init__(self, outer):
    self.outer = outer
    self.calls = []

  def visit_FunctionDef(self, node):
    if node is self.outer:
      self.generic_visit(node)

  def visit_AsyncFunctionDef(self, node):
    if node is self.outer:
      self.generic_visit(node)

  def visit_Call(self, node):
    name = None
    if isinstance(node.func, ast.Name):
      name = node.func.id
    elif isinstance(node.func, ast.Attribute):
      if isinstance(node.func.value, ast.Name):
        name = node.func.value.id + "." + node.func.attr
      else:
        name = node.func.attr
    if name is not None:
      self.calls.append((node.lineno, name))
    self.generic_visit(node)


def _check_preconstruction_order(results_path):
  """Prove composition validation precedes constructors and torch.load."""
  try:
    with open(results_path, encoding="utf-8") as source_file:
      tree = ast.parse(source_file.read(), filename=results_path)
    rebuild = next(
      node for node in tree.body
      if isinstance(node, ast.FunctionDef) and node.name == "rebuild_emulator")
    visitor = _OuterCalls(rebuild)
    visitor.visit(rebuild)
    composition_lines = [
      line for line, name in visitor.calls
      if name == "_read_artifact_composition"]
    danger_names = {
      "_rebuild_geometry",
      "_rebuild_model",
      "torch.load",
    }
    danger_lines = [
      line for line, name in visitor.calls if name in danger_names]
    ok = (len(composition_lines) == 1 and len(danger_lines) > 0
          and composition_lines[0] < min(danger_lines))
    report(
      "composition is validated before geometry/model construction or load",
      ok,
      "composition lines=" + repr(composition_lines)
      + "; constructor/load lines=" + repr(danger_lines))
  except Exception as exc:
    report(
      "composition is validated before geometry/model construction or load",
      False,
      type(exc).__name__ + ": " + str(exc))


def _check_downstream_mode_routing(results_path, inference_path,
                                   warmstart_path):
  """Census downstream selectors for a return to presence dispatch."""
  try:
    paths = (results_path, inference_path, warmstart_path)
    trees = {}
    sources = {}
    for path in paths:
      with open(path, encoding="utf-8") as source_file:
        sources[path] = source_file.read()
      trees[path] = ast.parse(sources[path], filename=path)

    def find_function(path, name):
      if "." in name:
        class_name, method_name = name.split(".", 1)
        owners = [
          node for node in ast.walk(trees[path])
          if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(owners) != 1:
          raise ValueError(
            path + " must define exactly one " + class_name + ", found "
            + str(len(owners)))
        matches = [
          node for node in owners[0].body
          if isinstance(node, ast.FunctionDef) and node.name == method_name]
      else:
        matches = [
          node for node in ast.walk(trees[path])
          if isinstance(node, ast.FunctionDef) and node.name == name]
      if len(matches) != 1:
        raise ValueError(
          path + " must define exactly one " + name + ", found "
          + str(len(matches)))
      return matches[0]

    def presence_dispatches(node):
      hits = []
      for item in ast.walk(node):
        if not isinstance(item, ast.Compare):
          continue
        names = []
        for part in (item.left,) + tuple(item.comparators):
          if isinstance(part, ast.Name):
            names.append(part.id)
        if ({"pce_base", "transfer_base"} & set(names)
            and any(isinstance(op, (ast.Is, ast.IsNot))
                    for op in item.ops)):
          hits.append(item.lineno)
        if (isinstance(item.left, ast.Constant)
            and item.left.value in ("pce", "transfer_base")
            and any(isinstance(op, (ast.In, ast.NotIn))
                    for op in item.ops)):
          hits.append(item.lineno)
      return hits

    checks = (
      (results_path, "rebuild_emulator"),
      (inference_path, "EmulatorPredictor.__init__"),
      (inference_path, "EmulatorPredictor._build_diag_decoder"),
      (inference_path, "EmulatorPredictor._build_decoder"),
      (warmstart_path, "load_source"),
    )
    bad = []
    missing_mode = []
    for path, name in checks:
      node = find_function(path, name)
      hits = presence_dispatches(node)
      if hits:
        bad.append(os.path.basename(path) + ":" + name + "@" + repr(hits))
      segment = ast.get_source_segment(sources[path], node) or ""
      if "composition_mode" not in segment:
        missing_mode.append(os.path.basename(path) + ":" + name)
    report(
      "rebuild, inference, and warm-start route on the validated mode",
      not bad and not missing_mode,
      "presence dispatches=" + repr(bad)
      + "; missing mode references=" + repr(missing_mode))
  except Exception as exc:
    report(
      "rebuild, inference, and warm-start route on the validated mode",
      False,
      type(exc).__name__ + ": " + str(exc))


def _run_matrix(h5py, yaml, reader):
  """Drive every legal row and the one-fact-at-a-time forgery matrix."""
  with tempfile.TemporaryDirectory(prefix="artifact-composition-") as tmp:
    print("\n-- legal composition rows --")
    _expect_accept(h5py, yaml, reader, tmp, "plain", "plain", False)
    _expect_accept(h5py, yaml, reader, tmp, "NPCE", "npce", False)
    _expect_accept(
      h5py, yaml, reader, tmp, "frozen transfer", "transfer", False)
    _expect_accept(
      h5py, yaml, reader, tmp, "refined transfer", "transfer", True)

    print("\n-- forged composition rows --")
    cases = [
      ("NPCE payload deletion", "npce", False, _delete("pce")),
      ("transfer payload deletion", "transfer", False,
       _delete("transfer_base")),
      ("plain artifact with a pce group", "plain", False,
       _add_group("pce")),
      ("NPCE artifact with a second transfer group", "npce", False,
       _add_group("transfer_base")),
      ("transfer artifact with a second pce group", "transfer", False,
       _add_group("pce")),
      ("missing composition_mode", "plain", False,
       _delete_attr("composition_mode")),
      ("non-string composition_mode", "plain", False,
       _set_attr("composition_mode", 1)),
      ("unknown composition_mode", "plain", False,
       _set_attr("composition_mode", "ensemble")),
      ("missing transfer_refined", "plain", False,
       _delete_attr("transfer_refined")),
      ("non-boolean transfer_refined", "plain", False,
       _set_attr("transfer_refined", "False")),
      ("plain artifact marked refined", "plain", False,
       _plain_marked_refined),
      ("refined transfer without drifted_state", "transfer", True,
       _refined_without_drifted),
      ("frozen transfer with drifted_state", "transfer", False,
       _frozen_with_drifted),
      ("resolved mode mismatch", "npce", False,
       _set_resolved("composition_mode", "plain")),
      ("resolved refined mismatch", "transfer", True,
       _set_resolved("transfer_refined", False)),
      ("resolved pce record mismatch", "npce", False,
       _set_resolved("pce", None)),
      ("resolved transfer record mismatch", "transfer", False,
       _set_resolved("transfer", None)),
      ("missing resolved composition key", "plain", False,
       _delete_resolved("composition_mode")),
      ("plain artifact with raw pce declaration", "plain", False,
       _set_raw("pce", {"form": "residual"})),
      ("plain artifact with raw transfer declaration", "plain", False,
       _set_raw("transfer", {"form": "gain", "space": "physical"})),
      ("old schema-v3 presence-only artifact", "plain", False, _old_v3),
      ("NPCE payload form mismatch", "npce", False,
       _set_payload_attr("pce", "form", "ratio")),
      ("transfer payload form mismatch", "transfer", False,
       _set_payload_attr("transfer_base", "form", "sum")),
      ("transfer payload space mismatch", "transfer", False,
       _set_payload_attr("transfer_base", "space", "whitened")),
      ("raw NPCE form mismatch", "npce", False,
       _set_raw_field("pce", "form", "ratio")),
      ("raw transfer form mismatch", "transfer", False,
       _set_raw_field("transfer", "form", "sum")),
      ("raw transfer space mismatch", "transfer", False,
       _set_raw_field("transfer", "space", "whitened")),
      ("raw refined-transfer marker deletion", "transfer", True,
       _set_raw_field("transfer", "refine", None)),
      ("resolved refined-transfer scalar marker", "transfer", True,
       _set_resolved_field("transfer", "refine", False)),
      ("raw refined-transfer scalar marker", "transfer", True,
       _set_raw_field("transfer", "refine", False)),
    ]
    for index, (label, mode, refined, mutate) in enumerate(cases):
      _expect_refusal(
        h5py, yaml, reader, tmp, index, label, mode, refined, mutate)


def _check_writer_submode_symmetry(results):
  """Require the writer to enforce the same native form/space grammar."""
  validate = results._validate_executed_composition

  valid = (
    ("plain", False, None, None, None, None, None),
    ("npce", False, object(), None, "residual",
     {"form": "residual"}, None),
    ("transfer", False, None,
     {"form": "gain", "space": "physical"}, None, None,
     {"form": "gain", "space": "physical"}),
    ("transfer", True, None,
     {"form": "sum", "space": "whitened", "drifted_state": object()},
     None, None,
     {"form": "sum", "space": "whitened", "refine": {"epochs": 1}}),
  )
  for mode, refined, pce, transfer, pce_form, resolved_pce, \
      resolved_transfer in valid:
    try:
      got = validate(
        composition_mode=mode,
        transfer_refined=refined,
        pce=pce,
        pce_form=pce_form,
        transfer_base=transfer,
        resolved_pce=resolved_pce,
        resolved_transfer=resolved_transfer,
        where="writer fixture")
      report("writer accepts " + mode + (" refined" if refined else ""),
             got == (mode, refined), "returned " + repr(got))
    except Exception as exc:
      report("writer accepts " + mode + (" refined" if refined else ""),
             False, type(exc).__name__ + ": " + str(exc))

  invalid = (
    ("unknown NPCE form", "npce", False, object(), None, "hybrid",
     {"form": "hybrid"}, None),
    ("byte NPCE form", "npce", False, object(), None, b"residual",
     {"form": b"residual"}, None),
    ("unknown transfer form/space", "transfer", False, None,
     {"form": "delta", "space": "law"}, None, None,
     {"form": "delta", "space": "law"}),
    ("numeric transfer form/space", "transfer", False, None,
     {"form": 1, "space": 2}, None, None,
     {"form": 1, "space": 2}),
    ("scalar transfer refine block", "transfer", True, None,
     {"form": "gain", "space": "physical", "drifted_state": object()},
     None, None,
     {"form": "gain", "space": "physical", "refine": False}),
  )
  for label, mode, refined, pce, transfer, pce_form, resolved_pce, \
      resolved_transfer in invalid:
    try:
      validate(
        composition_mode=mode,
        transfer_refined=refined,
        pce=pce,
        pce_form=pce_form,
        transfer_base=transfer,
        resolved_pce=resolved_pce,
        resolved_transfer=resolved_transfer,
        where="writer fixture")
    except (KeyError, TypeError, ValueError) as exc:
      report("writer refuses " + label, True,
             type(exc).__name__ + ": " + str(exc))
    except Exception as exc:
      report("writer refuses " + label, False,
             "unexpected " + type(exc).__name__ + ": " + str(exc))
    else:
      report("writer refuses " + label, False, "writer accepted it")


def main():
  print("artifact-composition (authoritative HDF5 composition contract)")
  try:
    import h5py
    import yaml
    from emulator import results

    reader = getattr(results, "_read_artifact_composition")
  except Exception as exc:
    report("composition dependencies and reader are available", False,
           type(exc).__name__ + ": " + str(exc))
  else:
    _run_matrix(h5py, yaml, reader)
    print("\n-- writer/read submode symmetry --")
    _check_writer_submode_symmetry(results)
    print("\n-- rebuild load-boundary order --")
    results_path = os.path.join(_REPO, "emulator", "results.py")
    _check_preconstruction_order(results_path)
    _check_downstream_mode_routing(
      results_path,
      os.path.join(_REPO, "emulator", "inference.py"),
      os.path.join(_REPO, "emulator", "warmstart.py"))

  print("")
  mark = "FAIL" if FAILURES else "PASS"
  print("##AID artifact-composition.contract " + mark)
  return 1 if FAILURES else 0


if __name__ == "__main__":
  sys.exit(main())
