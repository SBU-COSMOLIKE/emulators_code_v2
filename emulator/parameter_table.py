"""Resolve named columns in one GetDist parameter table.

GetDist is the Python package cosmologists use to analyze the output of a
Monte Carlo sampler.  The sampler explores cosmological parameter space and
records every visited point as one row of numbers in a plain-text file; that
file is called a chain, and this module calls it the parameter table.  Each
row starts with two bookkeeping columns and continues with the physics
columns::

    weight  minuslogpost  <declaration 0>  <declaration 1>  ...

``weight`` counts how many consecutive sampler steps stayed at that point,
and ``minuslogpost`` is the negative natural logarithm of the posterior
probability density there.  Neither is a physics input.

The table itself is numbers only, so it carries no trustworthy column
names.  The names live in a sidecar: a small companion file that sits
beside a data file and describes it.  GetDist's sidecar keeps the table's
name with the extension replaced by ``.paramnames``, and each of its lines
declares one physics column, in table order.  This module therefore
requires that sidecar, validates the whole declaration before slicing, and
returns inputs and derived outputs in the caller's requested order.  It is
deliberately independent of torch, so the same resolver serves training's
row staging, memory-budget sizing, and the generator's checkpoint reads.
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

  The ``@dataclass(frozen=True)`` line above the class asks Python to
  generate the constructor from the field list below and to raise on any
  assignment to a field after construction, so a resolved table cannot be
  edited in place.

  ``inputs`` holds the sampled physics columns and ``outputs`` the
  requested derived columns, both as float32 arrays of shape
  (rows, columns) with the columns in the caller's requested order.
  ``declarations`` contains one tuple per sidecar line:
  ``(logical_name, is_derived, numeric_column)``.  ``numeric_column`` is
  the zero-based column of the original numeric table, counting its two
  leading bookkeeping columns, so declaration k always sits at numeric
  column k + 2.  ``sidecar_path`` records which sidecar file supplied the
  names.
  """

  inputs: np.ndarray
  outputs: np.ndarray
  declarations: tuple[tuple[str, bool, int], ...]
  sidecar_path: str


def _sidecar_candidates(params_path):
  """List the paths where a table's ``.paramnames`` sidecar may live.

  A sidecar is a small companion file that sits beside a data file and
  describes it; here it is the text file naming the numeric table's
  columns, because the table itself holds only numbers.  GetDist writes
  one sidecar per sampling run, not per chain file: a sampler is usually
  launched as several parallel copies that write ``run.1.txt``,
  ``run.2.txt``, and so on, and the single ``run.paramnames`` describes
  them all, since every copy shares one column layout.

  The candidate list comes from ``os.path.splitext``, which splits a path
  at its last dot into a front part and an extension: ``run.1.txt``
  becomes ``("run.1", ".txt")``.  Dropping ``.txt`` and appending
  ``.paramnames`` gives the exact-name candidate ``run.1.paramnames``,
  tried first so a hand-exported single table with its own sidecar wins
  over the shared-run convention.  Splitting the front part once more
  exposes ``.1``; when the text after that second dot is all digits it is
  a chain number, and the shared ``run.paramnames`` becomes the second
  candidate.

  Arguments:
    params_path = path of the numeric parameter dump (``X.txt`` or
                  ``X.1.txt``).

  Returns:
    a tuple of one or two candidate sidecar paths, most specific first.
  """
  base = os.path.splitext(os.fspath(params_path))[0]
  candidates = [base + ".paramnames"]
  root, chain_ext = os.path.splitext(base)
  if chain_ext[1:].isdigit():
    candidates.append(root + ".paramnames")
  return tuple(candidates)


def _find_sidecar(params_path):
  """Locate the table's ``.paramnames`` sidecar, or refuse the table entirely.

  The numeric table is numbers only; the sidecar, the companion
  ``.paramnames`` file described in ``_sidecar_candidates``, is the only
  record of which column holds which physics parameter.  A missing
  sidecar is therefore not a degraded mode to work around: guessing the
  mapping from column position would silently feed the wrong physics to
  the wrong parameter, so the refusal instead names every path that was
  tried and how to regenerate the declaration.

  Arguments:
    params_path = path of the numeric parameter dump.

  Returns:
    the first candidate from ``_sidecar_candidates`` that exists as a
    file.

  Raises:
    ValueError listing the tried paths and the migration instruction when
    no candidate exists.
  """
  candidates = _sidecar_candidates(params_path)
  for candidate in candidates:
    if os.path.isfile(candidate):
      return candidate
  rendered = "\n".join("  " + repr(path) for path in candidates)
  raise ValueError(
    "the parameter table has no .paramnames sidecar; tried:\n"
    + rendered + "\n" + _MIGRATION)


def _requested_names(names, role):
  """Freeze and validate one caller-supplied column-name sequence.

  A bare string is refused before anything else because Python treats a
  string as a sequence of its own one-character strings: walking through
  the single name ``"H0"`` would silently turn it into a request for the
  two columns ``"H"`` and ``"0"``.  Every entry must then be a nonempty
  string, and a duplicate is refused because one named column can be
  returned only once.

  Arguments:
    names = the requested column names, in the order the caller wants
            the returned array's columns.
    role  = "input" or "output", named in every refusal.

  Returns:
    the names as a tuple, order preserved.  A tuple is Python's
    fixed-content sequence: entries cannot be added, removed, or
    replaced after creation, so the requested order cannot drift while
    the resolver works.

  Raises:
    ValueError for a string input, an empty or non-string entry, or a
    duplicated name.
  """
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
  """Read the sidecar's column declarations and attach numeric columns.

  Each nonblank sidecar line declares one table column.  A line looks
  like::

      omegabh2    \\Omega_\\mathrm{b} h^2

  The first whitespace-separated word is the logical name, the name the
  column is addressed by; anything after it is a plot label this module
  ignores.  A single trailing ``*`` on that word, as in ``sigma8*``,
  marks the column as derived: a quantity the sampler did not vary
  directly but computed from the sampled ones at each visited point.
  Declaration k describes numeric column k + 2, because the table opens
  with the two bookkeeping columns ``weight`` and ``minuslogpost`` that
  no sidecar line declares.

  The file is decoded as ``utf-8-sig`` because of the byte-order mark:
  some editors and Windows tools begin a UTF-8 text file with three
  invisible marker bytes, and plain ``utf-8`` decoding would glue those
  bytes onto the first logical name.  The ``-sig`` variant strips the
  marker when present and reads identically to ``utf-8`` when absent.

  Arguments:
    sidecar_path = path of the ``.paramnames`` sidecar.

  Returns:
    a tuple of ``(logical_name, is_derived, numeric_column)`` triples,
    one per declaration, in file order.

  Raises:
    ValueError for an empty logical name, a ``*`` or ``?`` inside a name
    (GetDist reserves both characters), or two declarations that
    normalize to one name: ``x`` and ``x*`` are duplicates too, because
    both would answer to the name ``x``.
  """
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
  """Load the numeric table as float32, keeping a one-row table 2-D.

  ``np.loadtxt`` parses the whitespace-separated text into an array of
  float32, the 32-bit floating-point type the emulators train in.  Its
  ``ndmin=2`` option stops numpy from collapsing a single-row file into
  a flat vector, so every caller can index rows and columns the same
  way regardless of the table's length.

  The load runs inside ``warnings.catch_warnings()``, a block that
  saves the process-wide warning settings on entry and restores them on
  exit; inside it, ``simplefilter("ignore", UserWarning)`` silences the
  console warning numpy emits for an empty file before returning its
  empty array.  Empty is an ordinary validation refusal here, so the
  refusal arrives as one clean error instead of an error plus a stray
  warning, and the setting change never leaks out of this function.

  Arguments:
    params_path = path of the numeric parameter dump.

  Returns:
    a float32 array of shape (rows, columns) with every value finite.

  Raises:
    ValueError when the file is empty, not two-dimensional, or contains
    a cell that is NaN (not-a-number) or infinite; ``np.isfinite`` is
    false for both.
  """
  with warnings.catch_warnings():
    # numpy prints a console warning before returning its empty array for an
    # empty file.  Empty is an ordinary validation refusal here, so the
    # refusal should arrive as one clean error without a stray warning.
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
  """Validate and slice a parameter table by the names its sidecar declares.

  This is the module's public entry.  It locates the ``.paramnames``
  sidecar (the companion file naming the table's columns), reads the
  declarations, checks them against the caller's request, loads the
  numbers, and returns the requested column slices.  The rules:

  - ``input_names`` must list every non-derived declaration, in the
    sidecar's order.  Inputs are the columns the sampler varied
    directly; they become the emulator's inputs.  The exact-sequence
    rule exists because of whitening: training rescales each input
    column to zero mean and unit spread using per-column statistics
    saved with the model, and a reordered or partial column set would
    silently pair a column with another column's statistics.
  - ``output_names`` may list any subset of the derived declarations, in
    any order.  A derived column is a quantity the sampler computed from
    the sampled ones at each point; these are what an emulator learns to
    predict.
  - A name may not appear on both sides, and a non-derived column may
    not be requested as an output.
  - The numeric table must be exactly two bookkeeping columns plus one
    column per declaration wide, so a sidecar edited out of step with
    its table is refused instead of sliced wrong.

  Arguments:
    params_path  = numeric parameter dump (``X.txt`` or ``X.1.txt``).
    input_names  = complete non-derived declaration sequence, order
                   included.
    output_names = derived columns to return, in requested order.

  Returns:
    A frozen :class:`ResolvedParameterTable`.  Both arrays are float32 and
    always two-dimensional, including one-row tables and zero requested
    outputs.

  Raises:
    ValueError when names, declarations, or numeric shape cannot establish an
    exact mapping.  Missing sidecars are refused with migration instructions;
    there is no fallback that maps columns by their position.
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
