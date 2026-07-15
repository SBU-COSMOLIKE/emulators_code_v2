"""Pure dataset-publication identity and run-control surfaces.

This module starts the dataset-manifest boundary with the command state that
decides whether a generator creates, resumes, or appends a dataset.  It imports
no generator, Cobaya, MPI, filesystem, or numerical code, so callers can reject
an invalid state before looking up paths or touching output.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunControl:
  """One normalized generator operation and dataset mode.

  Attributes:
    loadchk = 1 when an existing validated dataset is requested, else 0.
    append = 1 when new rows extend that dataset, else 0.
    chain = 1 for a chain-only dataset, else 0 for a full dataset.
    operation = ``fresh``, ``resume``, or ``append``.
    dataset_mode = ``full`` or ``chain-only``.
  """

  loadchk: int
  append: int
  chain: int
  operation: str
  dataset_mode: str


def _binary_flag(name, value, default):
  """Normalize one native-integer 0/1 flag.

  Arguments:
    name = command-line flag name without its leading dashes.
    value = value supplied by argparse or a programmatic caller.
    default = replacement for None, or None when absence is invalid.

  Returns:
    the native integer 0 or 1.

  Raises:
    ValueError when the value is absent without a default, is not a native
    integer, or is outside the two legal values.
  """
  if value is None:
    if default is None:
      raise ValueError(
        "--" + name + " must be a native integer 0 or 1 (not bool); got None")
    return default
  if type(value) is not int or value not in (0, 1):
    raise ValueError(
      "--" + name + " must be a native integer 0 or 1 (not bool); got "
      + repr(value))
  return value


def validate_run_control(loadchk, append, chain):
  """Validate and normalize the generator's run-control state.

  The three legal operations are fresh ``loadchk=0, append=0``, resume
  ``loadchk=1, append=0``, and append ``loadchk=1, append=1``.  Appending
  without loading is refused because append extends a validated prior dataset;
  it never means fresh generation at the same output path.  The independent
  chain axis records whether the operation targets a full or chain-only
  dataset.

  Arguments:
    loadchk = optional native integer 0/1; None means 0.
    append = optional native integer 0/1; None means 0.
    chain = optional native integer 0/1; None means 0.

  Returns:
    an immutable ``RunControl`` with normalized flags, operation, and dataset
    mode.

  Raises:
    ValueError when a flag is not a native integer 0/1, or append is requested
    without loading the prior dataset first.
  """
  normalized_loadchk = _binary_flag(
    name="loadchk", value=loadchk, default=0)
  normalized_append = _binary_flag(
    name="append", value=append, default=0)
  normalized_chain = _binary_flag(
    name="chain", value=chain, default=0)

  if normalized_append == 1 and normalized_loadchk != 1:
    raise ValueError(
      "--append=1 requires --loadchk=1; append extends a validated prior "
      "dataset and never starts fresh generation at the same path. Got "
      "--loadchk=" + str(normalized_loadchk)
      + " and --append=" + str(normalized_append) + ".")

  if normalized_loadchk == 0:
    operation = "fresh"
  elif normalized_append == 0:
    operation = "resume"
  else:
    operation = "append"
  dataset_mode = "chain-only" if normalized_chain == 1 else "full"

  return RunControl(
    loadchk=normalized_loadchk,
    append=normalized_append,
    chain=normalized_chain,
    operation=operation,
    dataset_mode=dataset_mode)
