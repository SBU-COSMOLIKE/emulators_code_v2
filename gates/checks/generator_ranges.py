#!/usr/bin/env python3
"""Check the generator's GetDist ``.ranges`` sidecar.

GetDist expects one row per sampled parameter. Each row contains the parameter
name, lower bound and upper bound. This child executes the production writer's
own syntax-tree statements, then asks GetDist to parse the resulting file.
Executing the production statements prevents a copied test writer from
staying green after the real writer changes.

The child requires the Cocoa Python environment because GetDist is a runtime
dependency of the generator and of this parser check.
"""

import ast
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

try:
  from getdist.parampriors import ParamBounds
except ImportError as error:
  raise SystemExit(
    "generator-ranges requires GetDist. Run this child with the Cocoa "
    "Python interpreter that runs the dataset generator.") from error


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)
_GENERATOR = os.path.join(
  _REPO,
  "compute_data_vectors",
  "generator_core.py")

# the production writer routes every bound through this module's one decimal
# policy, so the extracted statements need the module the production file
# imports (generator_core.py:73). The check hands it the real one: a private
# copy of the policy here would be a second implementation of the very thing
# this file exists to pin.
from emulator import fixed_facts

FAILURES = []

# The 25M-06 witnesses. Each pair is two bounds that are DISTINCT in the float32
# the generator owns and that the retired %.5e rounding collapsed into one
# string, so the sidecar declared a zero-width interval over a range the chain
# beside it kept apart. They are the acceptance bar for the decimal policy.
WITNESS_NAMES  = ["H0", "omegach2"]
WITNESS_BOUNDS = [
  [70.00001, 70.00002],
  [0.12345674, 0.12345676],
]


def report(
    label,
    ok,
    detail=""):
  """Print one result and retain the label when the result is false."""
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


def _fixed_fstring_text(node):
  """Return the literal text portions of one f-string syntax-tree node."""
  if not isinstance(node, ast.JoinedStr):
    return ""
  fixed_parts = []
  for part in node.values:
    if isinstance(part, ast.Constant) and isinstance(part.value, str):
      fixed_parts.append(part.value)
  return "".join(fixed_parts)


def _ranges_writer_candidates(tree):
  """Return every ``with open(...ranges)`` statement in the parsed file."""
  candidates = []
  for parent in ast.walk(tree):
    for _, child_value in ast.iter_fields(parent):
      if not isinstance(child_value, list):
        continue
      for statement_index, statement in enumerate(child_value):
        if not isinstance(statement, ast.With):
          continue
        if len(statement.items) != 1:
          continue
        context = statement.items[0].context_expr
        if not isinstance(context, ast.Call) or not context.args:
          continue
        path_text = _fixed_fstring_text(context.args[0])
        if ".ranges" in path_text:
          candidates.append(
            (child_value, statement_index, statement))
  return candidates


def _production_ranges_statements(source_path):
  """Extract the one active production writer and its required assignments.

  ``rows`` is always required because it owns the parameter-name and bound
  records. ``hd`` is included only when the extracted writer reads it. The
  latter condition lets the mutation arm execute the retired header while a
  later cleanup may remove the now-unused assignment without breaking the
  repaired control.
  """
  with open(
      source_path,
      encoding="utf-8") as source_file:
    tree = ast.parse(
      source_file.read(),
      filename=source_path)

  candidates = _ranges_writer_candidates(tree)
  if len(candidates) != 1:
    raise AssertionError(
      "expected exactly one production .ranges writer, found "
      + str(len(candidates)))

  statement_list, statement_index, writer = candidates[0]
  loaded_names = {
    node.id
    for node in ast.walk(writer)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
  }
  required_assignments = {"rows"}
  if "hd" in loaded_names:
    required_assignments.add("hd")

  assignments = []
  found_names = set()
  for earlier in statement_list[:statement_index]:
    if not isinstance(earlier, ast.Assign):
      continue
    assigned_names = {
      target.id
      for target in earlier.targets
      if isinstance(target, ast.Name)
    }
    owned_names = assigned_names & required_assignments
    if owned_names:
      assignments.append(earlier)
      found_names.update(owned_names)

  if found_names != required_assignments:
    raise AssertionError(
      "the production .ranges writer is missing assignment(s): "
      + repr(sorted(required_assignments - found_names)))
  return assignments + [writer]


def _run_production_ranges_writer(
    source_path,
    path_root,
    names,
    bounds):
  """Execute the extracted writer with one small parameter table.

  ``path_root`` omits the ``.ranges`` suffix, matching
  ``GeneratorCore.paramsf``. ``bounds`` has one row per name and two columns
  containing that parameter's lower and upper bounds.
  """
  statements = _production_ranges_statements(
    source_path=source_path)
  module = ast.Module(
    body=statements,
    type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "self": SimpleNamespace(paramsf=str(path_root)),
    "names": list(names),
    "bds": np.asarray(
      bounds,
      dtype=np.float64),
    "fixed_facts": fixed_facts,
  }
  exec(
    compile(
      module,
      source_path,
      "exec"),
    namespace)


def _parsed_bounds_match(
    path,
    expected):
  """Return GetDist's verdict and a short description of its parse."""
  try:
    parsed = ParamBounds(str(path))
  except Exception as error:
    return False, type(error).__name__ + ": " + str(error)

  names_match = parsed.names == list(expected)
  values_match = True
  for name, expected_bounds in expected.items():
    lower, upper = expected_bounds
    values_match = values_match and parsed.getLower(name) == lower
    values_match = values_match and parsed.getUpper(name) == upper
  return names_match and values_match, "names=" + repr(parsed.names)


def _write_and_parse(
    source_path,
    directory,
    stem,
    names,
    bounds,
    expected):
  """Write one sidecar with production code and parse it with GetDist."""
  path_root = os.path.join(
    directory,
    stem)
  _run_production_ranges_writer(
    source_path=source_path,
    path_root=path_root,
    names=names,
    bounds=bounds)
  return _parsed_bounds_match(
    path=path_root + ".ranges",
    expected=expected)


def check_repaired_writer(directory):
  """Check one-parameter behavior and a wider control."""
  one_ok, one_detail = _write_and_parse(
    source_path=_GENERATOR,
    directory=directory,
    stem="one_parameter",
    names=["H0"],
    bounds=[[60.0, 75.0]],
    expected={"H0": (60.0, 75.0)})
  report(
    "the production one-parameter .ranges file parses with GetDist",
    one_ok,
    one_detail)

  two_ok, two_detail = _write_and_parse(
    source_path=_GENERATOR,
    directory=directory,
    stem="two_parameters",
    names=["H0", "ombh2"],
    bounds=[
      [60.0, 75.0],
      [0.020, 0.024],
    ],
    expected={
      "H0": (60.0, 75.0),
      "ombh2": (0.020, 0.024),
    })
  report(
    "the production two-parameter .ranges control still parses",
    two_ok,
    two_detail)


def check_retired_header_mutation(directory):
  """Restore the retired header in a temporary source and prove it reds."""
  with open(
      _GENERATOR,
      encoding="utf-8") as source_file:
    repaired_source = source_file.read()
  writer_line = '        with open(f"{self.paramsf}.ranges", "w") as f:\n'
  temporary_header_names = '        hd = ["weights", "lnp"] + names\n'
  retired_header = '          f.write(f"# {\' \'.join(hd)}\\n")\n'
  if repaired_source.count(writer_line) != 1:
    raise AssertionError(
      "the mutation could not identify exactly one production writer")
  mutated_source = repaired_source.replace(
    writer_line,
    temporary_header_names + writer_line + retired_header,
    1)
  mutated_path = os.path.join(
    directory,
    "generator_core_with_retired_header.py")
  with open(
      mutated_path,
      "w",
      encoding="utf-8") as mutated_file:
    mutated_file.write(mutated_source)

  one_ok, one_detail = _write_and_parse(
    source_path=mutated_path,
    directory=directory,
    stem="mutated_one_parameter",
    names=["H0"],
    bounds=[[60.0, 75.0]],
    expected={"H0": (60.0, 75.0)})
  two_ok, two_detail = _write_and_parse(
    source_path=mutated_path,
    directory=directory,
    stem="mutated_two_parameters",
    names=["H0", "ombh2"],
    bounds=[
      [60.0, 75.0],
      [0.020, 0.024],
    ],
    expected={
      "H0": (60.0, 75.0),
      "ombh2": (0.020, 0.024),
    })
  mutation_ok = (
    not one_ok
    and "weights" in one_detail
    and two_ok)
  report(
    "the retired header breaks one parameter while the wider control hides it",
    mutation_ok,
    "one: " + one_detail + ". two: " + two_detail)


def _record_resolved_text(
    names,
    bounds):
  """Return the bound text the SCIENTIFIC RECORD publishes for these bounds.

  The record (``emulator/fixed_facts.py``) is the one author of a support, and
  the ``.ranges`` file is a GetDist view of it. This builds the record the
  generator would write for the same resolved bounds and hands back the text it
  stores, so the view can be compared against the thing it is a view of. If the
  two ever format a bound differently, they are two authors of one number.
  """
  requested = {}
  resolved = {}
  for index, name in enumerate(names):
    requested[name] = (bounds[index][0], bounds[index][1])
    resolved[name] = (bounds[index][0], bounds[index][1])

  held_fixed = {}
  for key in fixed_facts.COSMOLOGY_FIXED_KEYS:
    if key not in names:
      held_fixed[key] = fixed_facts.NOT_APPLICABLE

  text = fixed_facts.build_sidecar(
    dataset_id="sha256:" + "b" * 64,
    generator="generator-ranges-check",
    family=fixed_facts.NOT_APPLICABLE,
    cosmology_fixed=held_fixed,
    neutrino_convention=fixed_facts.NOT_APPLICABLE,
    flat_only=True,
    dark_energy_law=fixed_facts.NOT_APPLICABLE,
    dark_energy_inputs=[],
    cl_units=fixed_facts.NOT_APPLICABLE,
    base_identity=fixed_facts.NOT_APPLICABLE,
    names=list(names),
    requested=requested,
    resolved=resolved)
  blocks = fixed_facts.parse_sidecar(
    text=text,
    where="the record this check builds from the same bounds")
  return blocks[fixed_facts.INPUT_DOMAIN_GROUP]["resolved"]


def _ranges_text(path):
  """Return the literal ``{name: [low, high]}`` text the .ranges file holds."""
  rows = {}
  with open(
      path,
      encoding="utf-8") as handle:
    for line in handle:
      tokens = line.split()
      if len(tokens) == 3:
        rows[tokens[0]] = [tokens[1], tokens[2]]
  return rows


def check_decimal_policy(directory):
  """The 25M-06 witnesses survive, and the view says what the record says."""
  path_root = os.path.join(
    directory,
    "witness")
  _run_production_ranges_writer(
    source_path=_GENERATOR,
    path_root=path_root,
    names=WITNESS_NAMES,
    bounds=WITNESS_BOUNDS)
  parsed = ParamBounds(path_root + ".ranges")

  distinct_and_exact = True
  seen = []
  for index, name in enumerate(WITNESS_NAMES):
    lower = parsed.getLower(name)
    upper = parsed.getUpper(name)
    # distinct: the interval the sampler drew from is still an interval.
    # exact: each endpoint reads back as the very float32 the generator owns,
    # so the file records the bound rather than a rounding of it.
    if lower == upper:
      distinct_and_exact = False
    if np.float32(lower) != np.float32(WITNESS_BOUNDS[index][0]):
      distinct_and_exact = False
    if np.float32(upper) != np.float32(WITNESS_BOUNDS[index][1]):
      distinct_and_exact = False
    seen.append(name + " " + repr(lower) + " " + repr(upper))
  report(
    "both 25M-06 witness pairs round-trip distinct and float32-exact",
    distinct_and_exact,
    "; ".join(seen))

  record = _record_resolved_text(
    names=WITNESS_NAMES,
    bounds=WITNESS_BOUNDS)
  view = _ranges_text(path_root + ".ranges")
  agrees = True
  for name in WITNESS_NAMES:
    if view.get(name) != record.get(name):
      agrees = False
  report(
    "the .ranges view writes the bound text the record publishes",
    agrees,
    "view=" + repr(view) + " record=" + repr(record))


def check_collapsing_decimal_mutation(directory):
  """Restore the retired %.5e rounding and prove the witness legs go red.

  A check that cannot go red proves nothing. The mutation is the exact policy
  the writer used before this landing, so it reproduces 25M-06 itself: the two
  witness intervals collapse to zero width while the broad production bounds
  are unharmed. That second half matters — it is why the defect survived so
  long. A mutation that broke everything would prove only that the check runs.
  """
  with open(
      _GENERATOR,
      encoding="utf-8") as source_file:
    repaired_source = source_file.read()

  canonical = (
    '        rows = [(str(n), fixed_facts.format_value(float(l)),\n'
    '                         fixed_facts.format_value(float(h)))\n'
    '                for n, l, h in zip(names, bds[:, 0], bds[:, 1])]\n')
  collapsing = (
    '        rows = [(str(n), "%.5e" % float(l),\n'
    '                         "%.5e" % float(h))\n'
    '                for n, l, h in zip(names, bds[:, 0], bds[:, 1])]\n')
  if repaired_source.count(canonical) != 1:
    raise AssertionError(
      "the mutation could not identify the production writer's one formatter")
  mutated_source = repaired_source.replace(
    canonical,
    collapsing,
    1)
  mutated_path = os.path.join(
    directory,
    "generator_core_with_five_digits.py")
  with open(
      mutated_path,
      "w",
      encoding="utf-8") as mutated_file:
    mutated_file.write(mutated_source)

  path_root = os.path.join(
    directory,
    "mutated_witness")
  _run_production_ranges_writer(
    source_path=mutated_path,
    path_root=path_root,
    names=WITNESS_NAMES,
    bounds=WITNESS_BOUNDS)
  parsed = ParamBounds(path_root + ".ranges")

  collapsed = True
  seen = []
  for name in WITNESS_NAMES:
    lower = parsed.getLower(name)
    upper = parsed.getUpper(name)
    if lower != upper:
      collapsed = False
    seen.append(name + " " + repr(lower) + " " + repr(upper))

  broad_ok, broad_detail = _write_and_parse(
    source_path=mutated_path,
    directory=directory,
    stem="mutated_broad_bounds",
    names=["H0", "ombh2"],
    bounds=[
      [60.0, 75.0],
      [0.020, 0.024],
    ],
    expected={
      "H0": (60.0, 75.0),
      "ombh2": (0.020, 0.024),
    })
  report(
    "restoring %.5e collapses both witnesses while the broad bounds hide it",
    collapsed and broad_ok,
    "witnesses: " + "; ".join(seen) + ". broad: " + broad_detail)


def main():
  """Run the repaired controls, the decimal policy, and both mutation arms."""
  print("generator-ranges: GetDist sidecar format")
  with tempfile.TemporaryDirectory(
      prefix="generator-ranges-") as directory:
    check_repaired_writer(
      directory=directory)
    check_decimal_policy(
      directory=directory)
    check_collapsing_decimal_mutation(
      directory=directory)
    check_retired_header_mutation(
      directory=directory)
  print("")
  if FAILURES:
    print(
      "generator-ranges: " + str(len(FAILURES)) + " FAILURE(S): "
      + ", ".join(FAILURES))
    return 1
  print("generator-ranges: ALL PASS")
  return 0


if __name__ == "__main__":
  sys.exit(main())
