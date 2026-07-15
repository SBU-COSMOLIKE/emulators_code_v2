#!/usr/bin/env python3
"""Prove parameter consumers select producer-declared named columns.

This CPU/Torch gate owns three deliberately separate claims:

* the shared resolver validates the complete GetDist table schema and returns
  exact float32, two-dimensional named arrays;
* ordinary staging validates that schema before it opens a data-vector dump;
* ordinary/scalar staging and ``EmulatorExperiment.pool_size`` all consume the
  same named inputs (and the scalar paths validate their named outputs), while
  every optional-cut family treats an absent cut block as the full table.

The expected arrays below are gate-owned literals, never values re-derived by
the production resolver. Targeted in-memory regressions keep the claims
load-bearing: a positional resolver, a missing-sidecar compatibility guess, a
data-vector open or staging call moved ahead of resolution, a wrapped resolver
RHS, an ignored resolver result, a shadowed resolver import, and two isolated
positional-consumer mutations (staging only, then pool sizing only). The
consumer mutations are never enabled in the same run, so one correct path
cannot hide the other broken path behind parity.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import tempfile
import warnings
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from emulator import data_staging
from emulator import experiment
from emulator.experiment import EmulatorExperiment
from emulator.parameter_table import resolve_parameter_table


PARAMETER_TABLE = ROOT / "emulator" / "parameter_table.py"
DATA_STAGING = ROOT / "emulator" / "data_staging.py"
EXPERIMENT = ROOT / "emulator" / "experiment.py"


def _write_table(path, rows):
  """Write one gate-owned numeric fixture as float32 text."""
  np.savetxt(path, np.asarray(rows, dtype=np.float32), fmt="%.9e")


def _write_sidecar(path, tokens):
  """Write every declaration token, including its derived marker."""
  with open(path, "w") as handle:
    for token in tokens:
      handle.write(token + " label-" + token.rstrip("*") + "\n")


def _array_exact(observed, expected):
  """Require exact shape, float32 dtype, and gate-literal values."""
  expected = np.asarray(expected, dtype=np.float32)
  return (isinstance(observed, np.ndarray)
          and observed.dtype == np.float32
          and observed.shape == expected.shape
          and np.array_equal(observed, expected))


def _refuses(call, *fragments):
  """Return true only for a ValueError naming every required fragment."""
  try:
    call()
  except ValueError as exc:
    text = str(exc)
    return all(fragment in text for fragment in fragments)
  except Exception:
    return False
  return False


def _positional_resolver(params_path, input_names, output_names=()):
  """Mutation: restore the undocumented ``[:, 2:-1]`` layout guess."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    table = np.loadtxt(os.fspath(params_path), dtype=np.float32, ndmin=2)
  inputs = np.asarray(table[:, 2:-1], dtype=np.float32)
  # A positional compatibility shim has no trustworthy output map.  Returning
  # the final columns makes the mutation executable while keeping the defect
  # realistic: it guesses rather than reading the producer declaration.
  n_outputs = len(tuple(output_names))
  outputs = np.asarray(
    table[:, -n_outputs:] if n_outputs else table[:, 0:0],
    dtype=np.float32)
  return SimpleNamespace(inputs=inputs,
                         outputs=outputs,
                         declarations=(),
                         sidecar_path=None)


def _missing_compat_resolver(params_path, input_names, output_names=()):
  """Mutation: silently accept a legacy table when its sidecar is absent."""
  base = os.path.splitext(os.fspath(params_path))[0]
  candidates = [base + ".paramnames"]
  root, chain_ext = os.path.splitext(base)
  if chain_ext[1:].isdigit():
    candidates.append(root + ".paramnames")
  if not any(os.path.isfile(path) for path in candidates):
    table = np.loadtxt(os.fspath(params_path), dtype=np.float32, ndmin=2)
    n_inputs = len(tuple(input_names))
    n_outputs = len(tuple(output_names))
    return SimpleNamespace(
      inputs=np.asarray(table[:, 2:2 + n_inputs], dtype=np.float32),
      outputs=np.asarray(
        table[:, 2 + n_inputs:2 + n_inputs + n_outputs],
        dtype=np.float32),
      declarations=(),
      sidecar_path=None)
  return resolve_parameter_table(params_path=params_path,
                                 input_names=input_names,
                                 output_names=output_names)


def _schema_and_layout_contract(resolver):
  """Exercise accepted layouts and every strict producer-schema boundary."""
  try:
    with tempfile.TemporaryDirectory() as tmp:
      # Current generator layout: [weight, lnp, sampled..., chi2*].  The
      # expected arrays are literal and intentionally do not slice ``rows``.
      current = os.path.join(tmp, "current.txt")
      _write_table(current, [
        [1, -7, 0.02237, 0.1200, 67.4, 13.5],
        [2, -8, 0.02241, 0.1192, 68.1, 15.25],
      ])
      _write_sidecar(os.path.join(tmp, "current.paramnames"),
                     ["omegabh2", "omegach2", "H0", "chi2*"])
      found = resolver(current,
                       ["omegabh2", "omegach2", "H0"],
                       ["chi2"])
      current_ok = (
        _array_exact(found.inputs,
                     [[0.02237, 0.1200, 67.4],
                      [0.02241, 0.1192, 68.1]])
        and _array_exact(found.outputs, [[13.5], [15.25]])
        and found.declarations == (
          ("omegabh2", False, 2),
          ("omegach2", False, 3),
          ("H0", False, 4),
          ("chi2", True, 5)))

      # Zero derived columns and one row jointly pin exact-2D behavior.
      zero = os.path.join(tmp, "zero.txt")
      _write_table(zero, [[9, -1, 10, 11]])
      _write_sidecar(os.path.join(tmp, "zero.paramnames"), ["a", "b"])
      zero_found = resolver(zero, ["a", "b"])
      zero_ok = (_array_exact(zero_found.inputs, [[10, 11]])
                 and _array_exact(zero_found.outputs,
                                  np.empty((1, 0), dtype=np.float32)))

      # Multiple derived declarations are interleaved with the sampled block;
      # both returned arrays must follow requested NAME order, never position.
      interleaved = os.path.join(tmp, "interleaved.txt")
      _write_table(interleaved, [
        [1, 2, 30, 10, 31, 11, 32],
        [3, 4, 40, 20, 41, 21, 42],
      ])
      _write_sidecar(os.path.join(tmp, "interleaved.paramnames"),
                     ["d0*", "a", "d1*", "b", "d2*"])
      inter_found = resolver(interleaved, ["a", "b"],
                             ["d2", "d0", "d1"])
      interleaved_ok = (
        _array_exact(inter_found.inputs, [[10, 11], [20, 21]])
        and _array_exact(inter_found.outputs,
                         [[32, 30, 31], [42, 40, 41]])
        and inter_found.declarations == (
          ("d0", True, 2), ("a", False, 3),
          ("d1", True, 4), ("b", False, 5),
          ("d2", True, 6)))

      # Sidecar resolution: an exact numeric-chain stem wins, a numeric stem
      # falls back to its shared root, and a nonnumeric dotted stem is never
      # stripped.  Wrong candidate files carry declarations that would refuse
      # if production selected them.
      exact_numeric = os.path.join(tmp, "exact.7.txt")
      _write_table(exact_numeric, [[1, 2, 71]])
      exact_sidecar = os.path.join(tmp, "exact.7.paramnames")
      _write_sidecar(exact_sidecar, ["right"])
      _write_sidecar(os.path.join(tmp, "exact.paramnames"), ["wrong"])
      exact_found = resolver(exact_numeric, ["right"])

      root_numeric = os.path.join(tmp, "root.3.txt")
      _write_table(root_numeric, [[1, 2, 72]])
      root_sidecar = os.path.join(tmp, "root.paramnames")
      _write_sidecar(root_sidecar, ["right"])
      root_found = resolver(root_numeric, ["right"])

      dotted = os.path.join(tmp, "lcdm.v2.txt")
      _write_table(dotted, [[1, 2, 73]])
      dotted_sidecar = os.path.join(tmp, "lcdm.v2.paramnames")
      _write_sidecar(dotted_sidecar, ["right"])
      _write_sidecar(os.path.join(tmp, "lcdm.paramnames"), ["wrong"])
      dotted_found = resolver(dotted, ["right"])
      sidecars_ok = (
        exact_found.sidecar_path == exact_sidecar
        and root_found.sidecar_path == root_sidecar
        and dotted_found.sidecar_path == dotted_sidecar)

      # GetDist accepts UTF-8-with-BOM producer sidecars. The transport mark
      # is not part of the first logical name.
      bom = os.path.join(tmp, "bom.txt")
      _write_table(bom, [[1, 2, 74]])
      with open(os.path.join(tmp, "bom.paramnames"),
                "w", encoding="utf-8-sig") as handle:
        handle.write("right Right\n")
      bom_found = resolver(bom, ["right"])
      bom_ok = (bom_found.declarations == (("right", False, 2),)
                and _array_exact(bom_found.inputs, [[74]]))

      # Every refusal below changes only the producer declaration/schema; no
      # expected value is obtained from production.
      bad = os.path.join(tmp, "bad.txt")
      _write_table(bad, [[1, 2, 10, 11, 12]])
      bad_sidecar = os.path.join(tmp, "bad.paramnames")

      _write_sidecar(bad_sidecar, ["a", "a*", "d*"])
      duplicate_ok = _refuses(
        lambda: resolver(bad, ["a"], ["d"]),
        "duplicate", "a")

      marker_table = os.path.join(tmp, "markers.txt")
      marker_sidecar = os.path.join(tmp, "markers.paramnames")
      _write_table(marker_table, [[1, 2, 10]])
      marker_ok = True
      for token in ("a**", "a?"):
        _write_sidecar(marker_sidecar, [token])
        marker_ok = marker_ok and _refuses(
          lambda token=token: resolver(marker_table, ["a"]),
          "invalid GetDist", "allowed exactly once")

      _write_sidecar(bad_sidecar, ["a", "b", "d*"])
      requested_duplicate_ok = (
        _refuses(lambda: resolver(bad, ["a", "a"], ["d"]),
                 "input", "duplicates")
        and _refuses(lambda: resolver(bad, ["a", "b"], ["d", "d"]),
                     "output", "duplicates"))

      missing_ok = (
        _refuses(lambda: resolver(bad, ["a", "missing"], ["d"]),
                 "missing", "inputs")
        and _refuses(lambda: resolver(bad, ["a", "b"], ["missing"]),
                     "missing", "outputs"))

      _write_sidecar(bad_sidecar, ["b", "a", "d*"])
      order_ok = _refuses(
        lambda: resolver(bad, ["a", "b"], ["d"]),
        "order")

      _write_sidecar(bad_sidecar, ["a", "b", "extra", "d*"])
      _write_table(bad, [[1, 2, 10, 11, 99, 12]])
      extra_nonderived_ok = _refuses(
        lambda: resolver(bad, ["a", "b"], ["d"]),
        "declarations", "extra")

      _write_sidecar(bad_sidecar, ["a", "b", "d*"])
      _write_table(bad, [[1, 2, 10, 11, 12]])
      derived_ok = _refuses(
        lambda: resolver(bad, ["a", "b"], ["a"]),
        "not derived")

      _write_table(bad, [[1, 2, 10, 11]])
      width_minus_ok = _refuses(
        lambda: resolver(bad, ["a", "b"], ["d"]),
        "width", "expected 5")
      _write_table(bad, [[1, 2, 10, 11, 12, 13]])
      width_plus_ok = _refuses(
        lambda: resolver(bad, ["a", "b"], ["d"]),
        "width", "expected 5")

      empty = os.path.join(tmp, "empty.txt")
      Path(empty).write_text("", encoding="utf-8")
      _write_sidecar(os.path.join(tmp, "empty.paramnames"), ["a"])
      empty_ok = _refuses(lambda: resolver(empty, ["a"]), "empty")

      legacy = os.path.join(tmp, "legacy.4.txt")
      _write_table(legacy, [[1, 2, 99]])
      missing_sidecar_ok = _refuses(
        lambda: resolver(legacy, ["a"]),
        "legacy.4.paramnames", "legacy.paramnames", "cannot be mapped safely")

      checks = (current_ok, zero_ok, interleaved_ok, sidecars_ok, bom_ok,
                duplicate_ok, marker_ok, requested_duplicate_ok, missing_ok,
                order_ok, extra_nonderived_ok, derived_ok, width_minus_ok,
                width_plus_ok, empty_ok, missing_sidecar_ok)
      return all(checks), (
        "current/zero/interleaved="
        + repr((current_ok, zero_ok, interleaved_ok))
        + "; sidecars/BOM=" + repr((sidecars_ok, bom_ok))
        + "; strict refusals=" + repr(checks[5:]))
  except Exception as exc:
    return False, "schema probe raised " + type(exc).__name__ + ": " + str(exc)


def _attribute_chain(node):
  """Return a dotted attribute chain, or ``None`` for a dynamic base."""
  parts = []
  while isinstance(node, ast.Attribute):
    parts.append(node.attr)
    node = node.value
  if isinstance(node, ast.Name):
    parts.append(node.id)
    return ".".join(reversed(parts))
  return None


def _call_name(node):
  """Return a call's dotted function name, or ``None``."""
  if not isinstance(node, ast.Call):
    return None
  return _attribute_chain(node.func) if isinstance(node.func, ast.Attribute) \
    else (node.func.id if isinstance(node.func, ast.Name) else None)


def _tree(path):
  """Parse one complete production module."""
  return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class _ModuleBindings(ast.NodeVisitor):
  """Collect module-scope bindings without entering functions/classes."""

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


def _resolver_binding_contract(tree):
  """Require the sole resolver binding to be the sibling-module import."""
  imports = []
  for node in tree.body:
    if not isinstance(node, ast.ImportFrom):
      continue
    for alias in node.names:
      local = alias.asname or alias.name
      if local == "resolve_parameter_table":
        imports.append((node.level, node.module, alias.name, alias.asname))
  expected = [(1, "parameter_table", "resolve_parameter_table", None)]
  if imports != expected:
    return False, "resolver imports=" + repr(imports)
  bindings = _ModuleBindings()
  bindings.visit(tree)
  count = bindings.names.count("resolve_parameter_table")
  if count != 1:
    return False, "resolver module bindings=" + str(count)
  return True, "sole binding is .parameter_table import"


def _function(tree, name):
  """Find one uniquely named module-level function."""
  found = [node for node in tree.body
           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
           and node.name == name]
  if len(found) != 1:
    raise AssertionError("expected exactly one function " + name)
  return found[0]


def _method(tree, class_name, method_name):
  """Find one uniquely named method in one uniquely named class."""
  classes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name == class_name]
  if len(classes) != 1:
    raise AssertionError("expected exactly one class " + class_name)
  found = [node for node in classes[0].body
           if isinstance(node, ast.FunctionDef) and node.name == method_name]
  if len(found) != 1:
    raise AssertionError("expected exactly one method "
                         + class_name + "." + method_name)
  return found[0]


def _single_assign(function, target_name):
  """Return one top-level simple assignment to ``target_name``."""
  found = []
  for index, statement in enumerate(function.body):
    if (isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == target_name):
      found.append((index, statement))
  if len(found) != 1:
    raise AssertionError("expected one direct assignment to " + target_name)
  return found[0]


def _keyword_chains(call):
  """Return one call's keyword/value chains without evaluating them."""
  return [(keyword.arg, _attribute_chain(keyword.value))
          for keyword in call.keywords]


def _pre_dv_ast_contract(tree):
  """Require a direct resolver assignment before DV open and consume it."""
  try:
    binding_ok, binding_detail = _resolver_binding_contract(tree)
    if not binding_ok:
      return False, binding_detail
    function = _function(tree, "load_source")
    table_index, table_assign = _single_assign(function, "table")
    if _call_name(table_assign.value) != "resolve_parameter_table":
      return False, "table RHS is not a direct resolver call"
    if table_assign.value.args:
      return False, "resolver call must use explicit keywords"
    if _keyword_chains(table_assign.value) != [
        ("params_path", "params_path"), ("input_names", "names")]:
      return False, "load_source resolver keywords changed"

    dv_index, dv_assign = _single_assign(function, "dv")
    if _call_name(dv_assign.value) != "np.load":
      return False, "dv RHS is not a direct np.load call"
    if table_index >= dv_index:
      return False, "data-vector open does not follow schema resolution"

    np_loads = [node for node in ast.walk(function)
                if _call_name(node) == "np.load"]
    if len(np_loads) != 1 or np_loads[0] is not dv_assign.value:
      return False, "np.load call count/binding=" + str(len(np_loads))
    stage_calls = [node for node in ast.walk(function)
                   if _call_name(node) == "stage_source"]
    if len(stage_calls) != 1:
      return False, "stage_source call count=" + str(len(stage_calls))

    def containing_index(call):
      indices = [index for index, statement in enumerate(function.body)
                 if any(child is call for child in ast.walk(statement))]
      if len(indices) != 1:
        raise AssertionError("boundary call has no unique top-level owner")
      return indices[0]

    boundary_indices = [containing_index(node)
                        for node in np_loads + stage_calls]
    if any(index <= table_index for index in boundary_indices):
      return False, "DV/staging boundary precedes resolver=" + repr(
        boundary_indices)

    c_index, c_assign = _single_assign(function, "C")
    if _attribute_chain(c_assign.value) != "table.inputs":
      return False, "resolver result is ignored for C"
    if c_index <= table_index:
      return False, "C does not consume the resolved table after assignment"

    resolver_calls = [node for node in ast.walk(function)
                      if _call_name(node) == "resolve_parameter_table"]
    if len(resolver_calls) != 1:
      return False, "load_source resolver-call count=" + str(len(resolver_calls))
    positional_reads = [node for node in ast.walk(function)
                        if _call_name(node) == "np.loadtxt"]
    if positional_reads:
      return False, "load_source still reads the parameter table positionally"
    return True, (binding_detail + "; direct resolver assignment index "
                  + str(table_index)
                  + " precedes DV open index " + str(dv_index)
                  + " and staging index " + str(boundary_indices[1])
                  + "; C consumes table.inputs")
  except Exception as exc:
    return False, "AST contract raised " + type(exc).__name__ + ": " + str(exc)


def _pre_dv_structural_mutations_red():
  """Require moved/wrapped/ignored/shadowed regressions to be rejected."""
  moved = copy.deepcopy(_tree(DATA_STAGING))
  function = _function(moved, "load_source")
  table_index, _ = _single_assign(function, "table")
  dv_index, _ = _single_assign(function, "dv")
  function.body[table_index], function.body[dv_index] = (
    function.body[dv_index], function.body[table_index])
  moved_ok, _ = _pre_dv_ast_contract(moved)

  prefixed_load = copy.deepcopy(_tree(DATA_STAGING))
  function = _function(prefixed_load, "load_source")
  table_index, _ = _single_assign(function, "table")
  function.body.insert(table_index, ast.Expr(value=ast.Call(
    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                       attr="load", ctx=ast.Load()),
    args=[ast.Name(id="dv_path", ctx=ast.Load())], keywords=[])))
  prefixed_load_ok, _ = _pre_dv_ast_contract(prefixed_load)

  prefixed_stage = copy.deepcopy(_tree(DATA_STAGING))
  function = _function(prefixed_stage, "load_source")
  table_index, _ = _single_assign(function, "table")
  function.body.insert(table_index, ast.Expr(value=ast.Call(
    func=ast.Name(id="stage_source", ctx=ast.Load()),
    args=[], keywords=[])))
  prefixed_stage_ok, _ = _pre_dv_ast_contract(prefixed_stage)

  wrapped = copy.deepcopy(_tree(DATA_STAGING))
  function = _function(wrapped, "load_source")
  _, table_assign = _single_assign(function, "table")
  direct = table_assign.value
  table_assign.value = ast.Subscript(
    value=ast.Tuple(elts=[
      ast.Call(func=ast.Name(id="hidden_before", ctx=ast.Load()),
               args=[], keywords=[]),
      direct], ctx=ast.Load()),
    slice=ast.Constant(value=1),
    ctx=ast.Load())
  wrapped_ok, _ = _pre_dv_ast_contract(wrapped)

  ignored = copy.deepcopy(_tree(DATA_STAGING))
  function = _function(ignored, "load_source")
  _, c_assign = _single_assign(function, "C")
  c_assign.value = ast.Call(
    func=ast.Name(id="legacy_positional_parameters", ctx=ast.Load()),
    args=[], keywords=[])
  ignored_ok, _ = _pre_dv_ast_contract(ignored)

  shadowed = copy.deepcopy(_tree(DATA_STAGING))
  shadowed.body.append(ast.Assign(
    targets=[ast.Name(id="resolve_parameter_table", ctx=ast.Store())],
    value=ast.Name(id="permissive_resolver", ctx=ast.Load())))
  shadowed_ok, _ = _pre_dv_ast_contract(shadowed)
  return (not moved_ok) and (not prefixed_load_ok) \
    and (not prefixed_stage_ok) and (not wrapped_ok) \
    and (not ignored_ok) and (not shadowed_ok)


def _real_pre_dv_sentinel():
  """Drive real ``load_source`` and prove invalid schema opens no DV/stager."""
  events = []

  def dv_open(*args, **kwargs):
    events.append(("dv-open", args, kwargs))
    raise AssertionError("the DV sentinel must not be reached")

  def stage(*args, **kwargs):
    events.append(("stage", args, kwargs))
    raise AssertionError("the staging sentinel must not be reached")

  try:
    with tempfile.TemporaryDirectory() as tmp:
      missing = os.path.join(tmp, "missing.1.txt")
      _write_table(missing, [[1, 2, 0.02, 70]])
      bad = os.path.join(tmp, "bad.txt")
      _write_table(bad, [[1, 2, 0.02, 70]])
      _write_sidecar(os.path.join(tmp, "bad.paramnames"), ["H0", "omegab"])

      with mock.patch.object(data_staging.np, "load", side_effect=dv_open), \
           mock.patch.object(data_staging, "stage_source", side_effect=stage):
        missing_error = _refuses(
          lambda: data_staging.load_source(
            dv_path=os.path.join(tmp, "must-not-open-missing.npy"),
            params_path=missing,
            names=["omegab", "H0"],
            omegabh2_hi=None,
            n_keep=1,
            gen=torch.Generator()),
          ".paramnames")
        bad_error = _refuses(
          lambda: data_staging.load_source(
            dv_path=os.path.join(tmp, "must-not-open-invalid.npy"),
            params_path=bad,
            names=["omegab", "H0"],
            omegabh2_hi=None,
            n_keep=1,
            gen=torch.Generator()),
          "order")
      return missing_error and bad_error and events == [], (
        "missing/invalid refused=" + repr((missing_error, bad_error))
        + "; events=" + repr(events))
  except Exception as exc:
    return False, "pre-DV probe raised " + type(exc).__name__ + ": " + str(exc)


def _resolver_call_contract(function, expected_keywords, inputs_target,
                            outputs_target=None, stage_dv_chain=None):
  """Require one direct resolver assignment and direct named dataflow."""
  calls = [node for node in ast.walk(function)
           if _call_name(node) == "resolve_parameter_table"]
  if len(calls) != 1:
    return False, "resolver call count=" + str(len(calls))
  _, table_assign = _single_assign(function, "table")
  if table_assign.value is not calls[0]:
    return False, "resolver result is not the direct table assignment"
  if table_assign.value.args:
    return False, "resolver call uses positional arguments"
  if _keyword_chains(table_assign.value) != expected_keywords:
    return False, "resolver keywords=" + repr(_keyword_chains(table_assign.value))
  _, input_assign = _single_assign(function, inputs_target)
  if _attribute_chain(input_assign.value) != "table.inputs":
    return False, inputs_target + " does not consume table.inputs"
  input_stores = [node for node in ast.walk(function)
                  if isinstance(node, ast.Name)
                  and isinstance(node.ctx, ast.Store)
                  and node.id == inputs_target]
  if len(input_stores) != 1:
    return False, inputs_target + " assignment count=" + str(len(input_stores))
  if outputs_target is not None:
    _, output_assign = _single_assign(function, outputs_target)
    if _attribute_chain(output_assign.value) != "table.outputs":
      return False, outputs_target + " does not consume table.outputs"
    output_stores = [node for node in ast.walk(function)
                     if isinstance(node, ast.Name)
                     and isinstance(node.ctx, ast.Store)
                     and node.id == outputs_target]
    if len(output_stores) != 1:
      return False, outputs_target + " assignment count=" + str(
        len(output_stores))
  if any(_call_name(node) == "np.loadtxt" for node in ast.walk(function)):
    return False, "consumer still performs a positional np.loadtxt"
  stage_calls = [node for node in ast.walk(function)
                 if _call_name(node) == "stage_source"]
  if len(stage_calls) != 1:
    return False, "stage_source call count=" + str(len(stage_calls))
  if stage_dv_chain is not None:
    stage_call = stage_calls[0]
    if stage_call.args:
      return False, "stage_source call uses positional arguments"
    if _keyword_chains(stage_call) != [
        ("C", inputs_target), ("dv", stage_dv_chain),
        ("idx", "idx"), ("ram_frac", "ram_frac")]:
      return False, "stage_source dataflow=" + repr(_keyword_chains(stage_call))
  return True, "direct resolver assignment and named result dataflow"


def _string_subscript(node, base_name, key):
  """Tell whether ``node`` is exactly ``base_name[key]``."""
  return (isinstance(node, ast.Subscript)
          and isinstance(node.value, ast.Name)
          and node.value.id == base_name
          and isinstance(node.slice, ast.Constant)
          and node.slice.value == key)


def _optional_dict_get(node, base_name, key):
  """Tell whether ``node`` is exactly ``base_name.get(key)``."""
  return (isinstance(node, ast.Call)
          and isinstance(node.func, ast.Attribute)
          and isinstance(node.func.value, ast.Name)
          and node.func.value.id == base_name
          and node.func.attr == "get"
          and len(node.args) == 1
          and isinstance(node.args[0], ast.Constant)
          and node.args[0].value == key
          and not node.keywords)


def _pool_optional_cut_contract(pool):
  """Require the leading cut bound to be optional like ``stage_train``."""
  calls = [node for node in ast.walk(pool)
           if _call_name(node) == "phys_cut_idx"]
  if len(calls) != 1:
    return False, "phys_cut_idx call count=" + str(len(calls))
  values = [keyword.value for keyword in calls[0].keywords
            if keyword.arg == "omegabh2_hi"]
  if len(values) != 1:
    return False, "omegabh2_hi keyword count=" + str(len(values))
  if not _optional_dict_get(values[0], "pc", "omegabh2_hi"):
    return False, "pool_size requires pc['omegabh2_hi'] in a no-cut family"
  return True, "pool_size reads optional pc.get('omegabh2_hi')"


def _callsite_census():
  """Pin all three current consumers and scalar output validation."""
  try:
    staging_tree = _tree(DATA_STAGING)
    experiment_tree = _tree(EXPERIMENT)
    staging_binding_ok, staging_binding_detail = (
      _resolver_binding_contract(staging_tree))
    experiment_binding_ok, experiment_binding_detail = (
      _resolver_binding_contract(experiment_tree))
    ordinary_ok, ordinary_detail = _resolver_call_contract(
      _function(staging_tree, "load_source"),
      [("params_path", "params_path"), ("input_names", "names")],
      "C", stage_dv_chain="dv")
    scalar_ok, scalar_detail = _resolver_call_contract(
      _function(staging_tree, "load_scalar_source"),
      [("params_path", "params_path"), ("input_names", "in_names"),
       ("output_names", "out_names")],
      "C", "Y", stage_dv_chain="Y")

    pool = _method(experiment_tree, "EmulatorExperiment", "pool_size")
    calls = [node for node in ast.walk(pool)
             if _call_name(node) == "resolve_parameter_table"]
    _, table_assign = _single_assign(pool, "table")
    params_value = (table_assign.value.keywords[0].value
                    if table_assign.value.keywords else None)
    pool_ok = (len(calls) == 1 and table_assign.value is calls[0]
               and not table_assign.value.args
               and len(table_assign.value.keywords) == 3
               and table_assign.value.keywords[0].arg == "params_path"
               and _string_subscript(params_value, "d", "train_params")
               and table_assign.value.keywords[1].arg == "input_names"
               and _attribute_chain(
                 table_assign.value.keywords[1].value) == "self.names"
               and len(table_assign.value.keywords) == 3
               and table_assign.value.keywords[2].arg == "output_names")
    # The exact scalar conditional is load-bearing: pool sizing must validate
    # scalar outputs, while data-vector families request no derived outputs.
    output_value = (table_assign.value.keywords[2].value
                    if len(table_assign.value.keywords) == 3 else None)
    pool_ok = pool_ok and isinstance(output_value, ast.IfExp)
    if isinstance(output_value, ast.IfExp):
      pool_ok = (
        pool_ok
        and _attribute_chain(output_value.test) == "self._scalar"
        and _attribute_chain(output_value.body) == "self.outputs"
        and isinstance(output_value.orelse, ast.Tuple)
        and output_value.orelse.elts == [])
    _, c_assign = _single_assign(pool, "C")
    pool_ok = pool_ok and _attribute_chain(c_assign.value) == "table.inputs"
    pool_c_stores = [node for node in ast.walk(pool)
                     if isinstance(node, ast.Name)
                     and isinstance(node.ctx, ast.Store)
                     and node.id == "C"]
    pool_ok = pool_ok and len(pool_c_stores) == 1
    pool_ok = pool_ok and not any(
      _call_name(node) == "np.loadtxt" for node in ast.walk(pool))
    optional_cut_ok, optional_cut_detail = _pool_optional_cut_contract(pool)
    pool_ok = pool_ok and optional_cut_ok
    detail = ("bindings="
              + repr((staging_binding_ok, experiment_binding_ok))
              + " (" + staging_binding_detail + "; "
              + experiment_binding_detail + ")"
              + "; ordinary=" + str(ordinary_ok) + " (" + ordinary_detail + ")"
              + "; scalar=" + str(scalar_ok) + " (" + scalar_detail + ")"
              + "; pool=" + str(pool_ok)
              + " (" + optional_cut_detail + ")")
    return (staging_binding_ok and experiment_binding_ok
            and ordinary_ok and scalar_ok and pool_ok), detail
  except Exception as exc:
    return False, "callsite census raised " + type(exc).__name__ + ": " + str(exc)


def _binding_shadow_mutations_red():
  """Prove neither consumer module can shadow its reviewed resolver import."""
  results = []
  for path in (DATA_STAGING, EXPERIMENT):
    mutated = copy.deepcopy(_tree(path))
    mutated.body.append(ast.Assign(
      targets=[ast.Name(id="resolve_parameter_table", ctx=ast.Store())],
      value=ast.Name(id="permissive_resolver", ctx=ast.Load())))
    ok, _ = _resolver_binding_contract(mutated)
    results.append(not ok)
  return results == [True, True]


def _optional_cut_lookup_mutation_red():
  """Restore required-key lookup; the no-cut pool contract must reject it."""
  mutated = copy.deepcopy(_tree(EXPERIMENT))
  pool = _method(mutated, "EmulatorExperiment", "pool_size")
  calls = [node for node in ast.walk(pool)
           if _call_name(node) == "phys_cut_idx"]
  if len(calls) != 1:
    return False
  keywords = [keyword for keyword in calls[0].keywords
              if keyword.arg == "omegabh2_hi"]
  if len(keywords) != 1:
    return False
  keywords[0].value = ast.Subscript(
    value=ast.Name(id="pc", ctx=ast.Load()),
    slice=ast.Constant(value="omegabh2_hi"),
    ctx=ast.Load())
  ok, _ = _pool_optional_cut_contract(pool)
  return not ok


def _ordinary_stage_contract(resolver=resolve_parameter_table):
  """Drive accepted real ``load_source`` on zero/interleaved derived tables."""
  try:
    with tempfile.TemporaryDirectory() as tmp:
      fixtures = []

      zero_params = os.path.join(tmp, "zero.txt")
      zero_dv = os.path.join(tmp, "zero.npy")
      _write_table(zero_params, [
        [1, 0, 10, 11],
        [1, 0, 20, 21],
      ])
      _write_sidecar(os.path.join(tmp, "zero.paramnames"), ["a", "b"])
      np.save(zero_dv, np.asarray([[1001, 1002], [2001, 2002]],
                                 dtype=np.float32))
      fixtures.append((
        zero_params, zero_dv,
        np.asarray([[10, 11], [20, 21]], dtype=np.float32),
        np.asarray([[1001, 1002], [2001, 2002]], dtype=np.float32)))

      inter_params = os.path.join(tmp, "interleaved.txt")
      inter_dv = os.path.join(tmp, "interleaved.npy")
      _write_table(inter_params, [
        [1, 0, 30, 10, 31, 11, 32],
        [1, 0, 40, 20, 41, 21, 42],
      ])
      _write_sidecar(os.path.join(tmp, "interleaved.paramnames"),
                     ["d0*", "a", "d1*", "b", "d2*"])
      np.save(inter_dv, np.asarray([[3001], [4001]], dtype=np.float32))
      fixtures.append((
        inter_params, inter_dv,
        np.asarray([[10, 11], [20, 21]], dtype=np.float32),
        np.asarray([[3001], [4001]], dtype=np.float32)))

      results = []
      details = []
      with mock.patch.object(data_staging, "resolve_parameter_table", resolver):
        for number, (params_path, dv_path, expected_c, expected_dv) in enumerate(
            fixtures):
          staged = data_staging.load_source(
            dv_path=dv_path,
            params_path=params_path,
            names=["a", "b"],
            omegabh2_hi=None,
            n_keep=2,
            gen=torch.Generator().manual_seed(100 + number),
            ram_frac=1.0,
            verbose=False)
          observed_c = np.asarray(staged["C"])[np.asarray(staged["idx"])]
          observed_dv = np.asarray(staged["dv"])[np.asarray(staged["idx"])]
          order = np.argsort(observed_c[:, 0], kind="stable")
          observed_c = observed_c[order]
          observed_dv = observed_dv[order]
          ok = (_array_exact(observed_c, expected_c)
                and _array_exact(observed_dv, expected_dv))
          results.append(ok)
          details.append((observed_c.tolist(), observed_dv.tolist()))
      return results == [True, True], (
        "zero/interleaved ordinary=" + repr(tuple(results))
        + "; rows=" + repr(details))
  except Exception as exc:
    return False, "ordinary staging raised " + type(exc).__name__ + ": " + str(exc)


def _sorted_rows(inputs, outputs):
  """Sort staged rows by their second input for deterministic comparison."""
  order = np.argsort(np.asarray(inputs)[:, 1], kind="stable")
  return np.asarray(inputs)[order], np.asarray(outputs)[order]


def _stage_pool_contract(*, staging_resolver=resolve_parameter_table,
                         pool_resolver=resolve_parameter_table):
  """Compare real scalar staging/pool results on an interleaved decoy table."""
  try:
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "scalar.1.txt")
      _write_table(params, [
        [1, 0, 901, 0.040, 911, 50, 101, 921],
        [1, 0, 902, 0.050, 912, 80, 202, 922],
        [1, 0, 903, 0.030, 913, 60, 303, 923],
        [1, 0, 904, 0.060, 914, 90, 404, 924],
      ])
      _write_sidecar(os.path.join(tmp, "scalar.paramnames"),
                     ["decoy0*", "omegab", "decoy1*", "H0",
                      "target*", "decoy2*"])
      cuts = {"omegabh2_hi": 0.02}

      instance = EmulatorExperiment.__new__(EmulatorExperiment)
      instance.data = {"train_params": params, "param_cuts": cuts}
      instance.names = ["omegab", "H0"]
      instance.outputs = ["target"]
      instance._scalar = True
      instance._cmb = False
      instance._grid = False
      instance._grid2d = False

      # Mutations are isolated by binding: pool_size reads experiment's
      # resolver, while scalar staging reads data_staging's resolver.
      with mock.patch.object(experiment, "resolve_parameter_table",
                             pool_resolver):
        pool = instance.pool_size()
      try:
        with mock.patch.object(data_staging, "resolve_parameter_table",
                               staging_resolver):
          staged = data_staging.load_scalar_source(
            params_path=params,
            in_names=instance.names,
            out_names=instance.outputs,
            n_keep=2,
            gen=torch.Generator().manual_seed(17),
            ram_frac=1.0,
            verbose=False,
            omegabh2_hi=cuts["omegabh2_hi"])
      except (ValueError, IndexError):
        return False, "pool=" + str(pool) + "; staging refused"

      # Follow loader coordinates before comparison.  The expected retained
      # rows and output values are independent literals from the active cut:
      # 0.040*(.50)^2=.010 and 0.030*(.60)^2=.0108 are the only values < .02.
      observed_inputs = np.asarray(staged["C"])[np.asarray(staged["idx"])]
      observed_outputs = np.asarray(staged["dv"])[np.asarray(staged["idx"])]
      observed_inputs, observed_outputs = _sorted_rows(
        observed_inputs, observed_outputs)
      expected_inputs = np.asarray([[0.040, 50], [0.030, 60]],
                                   dtype=np.float32)
      expected_outputs = np.asarray([[101], [303]], dtype=np.float32)
      ok = (pool == 2
            and _array_exact(observed_inputs, expected_inputs)
            and _array_exact(observed_outputs, expected_outputs))
      return ok, ("pool=" + str(pool)
                  + "; inputs=" + repr(observed_inputs.tolist())
                  + "; outputs=" + repr(observed_outputs.tolist()))
  except Exception as exc:
    return False, "stage/pool probe raised " + type(exc).__name__ + ": " + str(exc)


def _optional_family_ceiling_contract():
  """Drive no-cut and active-cut pool/staging parity for four families."""
  try:
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "families.1.txt")
      dv_path = os.path.join(tmp, "families.npy")
      _write_table(params, [
        [1, 0, 0.040, 50, 101],
        [1, 0, 0.050, 80, 202],
        [1, 0, 0.030, 60, 303],
        [1, 0, 0.060, 90, 404],
      ])
      _write_sidecar(os.path.join(tmp, "families.paramnames"),
                     ["omegab", "H0", "target*"])
      np.save(dv_path, np.asarray([
        [1001, 1002],
        [2001, 2002],
        [3001, 3002],
        [4001, 4002],
      ], dtype=np.float32))

      # omega_b h^2 is .010, .032, .0108, .0486. The active window
      # therefore retains exactly two rows; no cuts retains all four.
      scenarios = (("no-cuts", None, 4),
                   ("active-cut", {"omegabh2_hi": 0.02}, 2))
      observed = []
      for family in ("scalar", "cmb", "grid", "grid2d"):
        for label, cuts, expected_pool in scenarios:
          instance = EmulatorExperiment.__new__(EmulatorExperiment)
          instance.data = {
            "train_params": params,
            "split_seed": 29,
            "n_train": expected_pool,
            "ram_frac": 1.0,
          }
          if family != "scalar":
            instance.data["train_dv"] = dv_path
          if cuts is not None:
            instance.data["param_cuts"] = cuts
          instance.names = ["omegab", "H0"]
          instance.outputs = ["target"] if family == "scalar" else []
          instance._scalar = family == "scalar"
          instance._cmb = family == "cmb"
          instance._grid = family == "grid"
          instance._grid2d = family == "grid2d"
          instance.quiet = True

          law_calls = []
          if instance._grid2d:
            instance._grid2d_train_tmp = None
            instance.grid2d = {"train_base": None}

            def law_stub(*, src, base_path, with_means):
              law_calls.append((len(src["idx"]),
                                isinstance(src["dv"], np.memmap),
                                base_path, with_means))
              return None

            instance._grid2d_law_rows = law_stub

          pool = instance.pool_size()
          staged = instance.stage_train(n_train=pool)
          refused = _refuses(
            lambda: instance.stage_train(n_train=pool + 1),
            "requested n_keep")
          grid2d_ok = (not instance._grid2d) or law_calls == [
            (pool, True, None, True)]
          ok = (pool == expected_pool
                and len(staged["idx"]) == pool
                and refused
                and grid2d_ok)
          observed.append((family, label, pool, ok))
      return all(row[3] for row in observed), "family ceilings=" + repr(observed)
  except Exception as exc:
    return False, "optional-family probe raised " + type(exc).__name__ + ": " + str(exc)


def _aid(aid, normal, mutation_red, detail):
  """Emit exactly one terminal for a board-declared evidence leg."""
  passed = normal and mutation_red
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + aid + "  (" + detail
        + "; mutation-red=" + str(mutation_red) + ")")
  print("##AID " + aid + " " + mark)
  return passed


def main():
  """Run normal production evidence and every isolated mutation."""
  schema_ok, schema_detail = _schema_and_layout_contract(
    resolve_parameter_table)
  positional_ok, _ = _schema_and_layout_contract(_positional_resolver)
  missing_compat_ok, _ = _schema_and_layout_contract(
    _missing_compat_resolver)
  schema = _aid(
    "parameter-table.schema-and-layout",
    schema_ok,
    (not positional_ok) and (not missing_compat_ok),
    schema_detail)

  sentinel_ok, sentinel_detail = _real_pre_dv_sentinel()
  ast_ok, ast_detail = _pre_dv_ast_contract(_tree(DATA_STAGING))
  pre_dv = _aid(
    "parameter-table.pre-dv-refusal",
    sentinel_ok and ast_ok,
    _pre_dv_structural_mutations_red(),
    sentinel_detail + "; " + ast_detail)

  census_ok, census_detail = _callsite_census()
  ordinary_ok, ordinary_detail = _ordinary_stage_contract()
  ordinary_mutant_ok, _ = _ordinary_stage_contract(_positional_resolver)
  parity_ok, parity_detail = _stage_pool_contract()
  optional_ok, optional_detail = _optional_family_ceiling_contract()
  staging_mutant_ok, _ = _stage_pool_contract(
    staging_resolver=_positional_resolver,
    pool_resolver=resolve_parameter_table)
  pool_mutant_ok, _ = _stage_pool_contract(
    staging_resolver=resolve_parameter_table,
    pool_resolver=_positional_resolver)
  parity = _aid(
    "parameter-table.stage-pool-parity",
    census_ok and ordinary_ok and parity_ok and optional_ok,
    (_binding_shadow_mutations_red()
     and _optional_cut_lookup_mutation_red()
     and (not ordinary_mutant_ok)
     and (not staging_mutant_ok)
     and (not pool_mutant_ok)),
    census_detail + "; " + ordinary_detail + "; " + parity_detail
    + "; " + optional_detail)

  if not (schema and pre_dv and parity):
    print("parameter-table: FAIL")
    return 1
  print("parameter-table: ALL PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
