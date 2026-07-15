"""Resolve named columns in one GetDist parameter table.

A parameter dump has two bookkeeping columns followed by the columns declared
by its ``.paramnames`` sidecar::

    weight  minuslogpost  <declaration 0>  <declaration 1>  ...

The table itself carries no trustworthy column names.  This module therefore
requires the producer sidecar, validates the whole declaration before slicing,
and returns inputs and derived outputs in the caller's requested order.  It is
deliberately independent of torch so the same resolver can be used by staging,
pool sizing, and generator checkpoint reads.
"""

from dataclasses import dataclass
import os
import warnings

import numpy as np


_MIGRATION = (
  "Re-generate or re-export this parameter table with its GetDist "
  ".paramnames sidecar.  The sidecar must declare every numeric column after "
  "weight and minuslogpost, in file order, and mark derived columns with one "
  "trailing '*'.  A legacy table without that producer-authored declaration "
  "cannot be mapped safely by position."
)


@dataclass(frozen=True)
class ResolvedParameterTable:
  """A validated parameter table sliced into named input/output arrays.

  ``declarations`` contains one immutable tuple per sidecar line:
  ``(logical_name, is_derived, numeric_column)``.  ``numeric_column`` is the
  zero-based column of the original numeric table, including its two leading
  bookkeeping columns.
  """

  inputs: np.ndarray
  outputs: np.ndarray
  declarations: tuple[tuple[str, bool, int], ...]
  sidecar_path: str


def _sidecar_candidates(params_path):
  """Return exact-stem then numeric-chain-root ``.paramnames`` candidates."""
  base = os.path.splitext(os.fspath(params_path))[0]
  candidates = [base + ".paramnames"]
  root, chain_ext = os.path.splitext(base)
  if chain_ext[1:].isdigit():
    candidates.append(root + ".paramnames")
  return tuple(candidates)


def _find_sidecar(params_path):
  """Resolve the producer sidecar, refusing positional legacy fallback."""
  candidates = _sidecar_candidates(params_path)
  for candidate in candidates:
    if os.path.isfile(candidate):
      return candidate
  rendered = "\n".join("  " + repr(path) for path in candidates)
  raise ValueError(
    "the parameter table has no .paramnames sidecar; tried:\n"
    + rendered + "\n" + _MIGRATION)


def _requested_names(names, role):
  """Freeze and validate one caller-supplied name sequence."""
  if isinstance(names, (str, bytes)):
    raise ValueError(f"requested {role} names must be a sequence, not a string")
  resolved = tuple(names)
  invalid = [name for name in resolved
             if not isinstance(name, str) or not name]
  if invalid:
    raise ValueError(
      f"requested {role} names must be non-empty strings; got {invalid!r}")
  counts = {}
  for name in resolved:
    counts[name] = counts.get(name, 0) + 1
  duplicates = sorted(name for name, count in counts.items() if count > 1)
  if duplicates:
    raise ValueError(
      f"requested {role} names contain duplicates {duplicates!r}; a named "
      "column may appear only once")
  return resolved


def _read_declarations(sidecar_path):
  """Read every nonblank sidecar declaration and attach numeric columns."""
  declarations = []
  # GetDist accepts producer sidecars written with a UTF-8 byte-order mark.
  # ``utf-8-sig`` removes it when present and is identical to ``utf-8`` when
  # absent, so the first logical name is never polluted by transport bytes.
  with open(sidecar_path, encoding="utf-8-sig") as handle:
    for line in handle:
      stripped = line.strip()
      if not stripped:
        continue
      token = stripped.split()[0]
      is_derived = token.endswith("*")
      logical_name = token[:-1] if is_derived else token
      if not logical_name:
        raise ValueError(
          f"the .paramnames sidecar {sidecar_path!r} contains the empty "
          "logical name '*'"
        )
      if "*" in logical_name or "?" in logical_name:
        raise ValueError(
          f"the .paramnames sidecar {sidecar_path!r} has invalid GetDist "
          f"name {token!r}; '*' is allowed exactly once as the trailing "
          "derived marker and '?' is not allowed in a logical name")
      declarations.append((logical_name, is_derived,
                           2 + len(declarations)))

  counts = {}
  for logical_name, _, _ in declarations:
    counts[logical_name] = counts.get(logical_name, 0) + 1
  duplicates = sorted(name for name, count in counts.items() if count > 1)
  if duplicates:
    raise ValueError(
      f"the .paramnames sidecar {sidecar_path!r} has duplicate normalized "
      f"declarations {duplicates!r}; a plain name and the same name with a "
      "trailing '*' are duplicates too")
  return tuple(declarations)


def _load_numeric_table(params_path):
  """Load a float32 table while preserving a one-row table's row axis."""
  with warnings.catch_warnings():
    # numpy warns before returning its useful empty-array sentinel.  Empty is
    # an ordinary validation refusal here, so keep that refusal deterministic.
    warnings.simplefilter("ignore", UserWarning)
    table = np.loadtxt(os.fspath(params_path), dtype=np.float32, ndmin=2)
  if table.ndim != 2:
    raise ValueError(
      f"the parameter table {os.fspath(params_path)!r} must be two-dimensional; "
      f"loaded shape {table.shape!r}")
  if table.size == 0 or table.shape[0] == 0:
    raise ValueError(
      f"the parameter table {os.fspath(params_path)!r} is empty; at least one "
      "sample row is required")
  if not np.isfinite(table).all():
    raise ValueError(
      f"the parameter table {os.fspath(params_path)!r} contains nonfinite "
      "numeric values; every bookkeeping, sampled, and derived cell must be "
      "finite")
  return table


def resolve_parameter_table(params_path, input_names, output_names=()):
  """Validate and slice a parameter table by producer-declared names.

  Arguments:
    params_path  = numeric parameter dump (``X.txt`` or ``X.1.txt``).
    input_names  = complete non-derived declaration sequence, order included.
    output_names = derived columns to return, in requested order.

  Returns:
    A frozen :class:`ResolvedParameterTable`.  Both arrays are float32 and
    always two-dimensional, including one-row tables and zero requested
    outputs.

  Raises:
    ValueError when names, declarations, or numeric shape cannot establish an
    exact mapping.  Missing sidecars are refused with migration instructions;
    there is no positional legacy fallback.
  """
  inputs_wanted = _requested_names(input_names, "input")
  outputs_wanted = _requested_names(output_names, "output")
  sidecar_path = _find_sidecar(params_path)
  declarations = _read_declarations(sidecar_path)
  by_name = {name: (is_derived, column)
             for name, is_derived, column in declarations}

  missing_inputs = [name for name in inputs_wanted if name not in by_name]
  missing_outputs = [name for name in outputs_wanted if name not in by_name]
  if missing_inputs or missing_outputs:
    declared = [name for name, _, _ in declarations]
    raise ValueError(
      f"requested names are missing from {sidecar_path!r}: "
      f"inputs={missing_inputs!r}, outputs={missing_outputs!r}; "
      f"declared names={declared!r}")

  non_derived = tuple(name for name, is_derived, _ in declarations
                      if not is_derived)
  if non_derived != inputs_wanted:
    raise ValueError(
      "the .paramnames non-derived declaration sequence must equal "
      "input_names (order included), or whitening would pair the wrong "
      f"columns: declarations={list(non_derived)!r}, "
      f"input_names={list(inputs_wanted)!r}")

  overlap = sorted(set(inputs_wanted).intersection(outputs_wanted))
  non_derived_outputs = [name for name in outputs_wanted
                         if not by_name[name][0]]
  if overlap or non_derived_outputs:
    parts = []
    if overlap:
      parts.append(f"input/output names overlap at {overlap!r}")
    if non_derived_outputs:
      parts.append(
        f"requested outputs {non_derived_outputs!r} are not derived columns")
    raise ValueError("; ".join(parts))

  table = _load_numeric_table(params_path)
  expected_width = 2 + len(declarations)
  if table.shape[1] != expected_width:
    raise ValueError(
      f"parameter table width {table.shape[1]} does not match the two "
      f"bookkeeping columns plus {len(declarations)} .paramnames "
      f"declarations (expected {expected_width})")

  input_columns = [by_name[name][1] for name in inputs_wanted]
  output_columns = [by_name[name][1] for name in outputs_wanted]
  inputs = np.asarray(table[:, input_columns], dtype=np.float32)
  outputs = np.asarray(table[:, output_columns], dtype=np.float32)
  return ResolvedParameterTable(inputs=inputs,
                                outputs=outputs,
                                declarations=declarations,
                                sidecar_path=sidecar_path)


__all__ = ["ResolvedParameterTable", "resolve_parameter_table"]
