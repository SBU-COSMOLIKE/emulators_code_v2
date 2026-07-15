"""Pure dataset-publication identity and run-control surfaces.

This module starts the dataset-manifest boundary with the command state that
decides whether a generator creates, resumes, or appends a dataset.  It imports
no generator, Cobaya, MPI, filesystem, or numerical code, so callers can reject
an invalid state before looking up paths or touching output.
"""

from dataclasses import dataclass


class CheckpointLoadError(RuntimeError):
  """A requested checkpoint cannot be loaded without risking old output."""


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


def require_checkpoint_members(operation, members, is_file):
  """Require every named member when resume or append was requested.

  ``fresh`` is the only operation allowed to proceed without an existing
  checkpoint.  The filesystem predicate is supplied by the caller so this
  module stays importable in small CPU-only checks without acquiring a path or
  generator dependency.

  Arguments:
    operation = ``fresh``, ``resume``, or ``append``.
    members = ordered paths that form the current checkpoint census.
    is_file = callable returning whether one path is an existing file.

  Returns:
    the ordered member tuple.  Returning the census makes it straightforward
    for later manifest work to consume the exact same list.

  Raises:
    CheckpointLoadError when a requested checkpoint member is missing.
    ValueError when the operation is not one of the normalized operations.
  """
  if operation not in ("fresh", "resume", "append"):
    raise ValueError("Unknown normalized generator operation: "
                     + repr(operation))

  checkpoint_members = tuple(members)
  if operation == "fresh":
    return checkpoint_members

  missing = [path for path in checkpoint_members if not is_file(path)]
  if missing:
    raise CheckpointLoadError(
      "Cannot " + operation + " the requested dataset because checkpoint "
      "members are missing: " + ", ".join(str(path) for path in missing)
      + ". No existing dataset file was changed.")
  return checkpoint_members


def load_checkpoint_or_refuse(operation, loader):
  """Run one checkpoint loader without converting failure into a fresh run.

  A historical broad exception handler treated every missing, truncated, or
  shape-incompatible checkpoint as if the user had requested fresh generation.
  This boundary keeps intent explicit: only ``fresh`` may produce the false
  ``not loaded`` result; resume and append either return ``True`` or raise.
  """
  try:
    loaded = loader()
  except Exception as exc:
    if operation == "fresh":
      raise
    raise CheckpointLoadError(
      "Cannot " + operation + " the requested dataset because its checkpoint "
      "could not be validated: " + str(exc)
      + ". No existing dataset file was changed.") from exc

  if operation == "fresh":
    if loaded:
      raise CheckpointLoadError(
        "Fresh generation unexpectedly loaded an existing checkpoint.")
    return False
  if loaded is not True:
    raise CheckpointLoadError(
      "Cannot " + operation + " the requested dataset because the checkpoint "
      "loader did not confirm a complete checkpoint. No existing dataset "
      "file was changed.")
  return True
