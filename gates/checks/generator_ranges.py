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
_GENERATOR = os.path.join(
  _REPO,
  "compute_data_vectors",
  "generator_core.py")

FAILURES = []


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


def main():
  """Run the repaired controls and the discriminating retired-header arm."""
  print("generator-ranges: GetDist sidecar format")
  with tempfile.TemporaryDirectory(
      prefix="generator-ranges-") as directory:
    check_repaired_writer(
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
