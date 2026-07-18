"""
Shared machinery for the dataset-generator drivers.

Every training-set generator in this package does the same job: draw
cosmological parameter samples, farm one expensive Boltzmann/cosmolike
call per sample over MPI, and write the results as the dv/params dumps
the training stack stages. This module holds that shared machinery ONCE;
each physics family ships a thin driver file beside it that subclasses
GeneratorCore and fills in only what differs.

    dataset_generator_lensing.py   cosmolike data vectors (cs/ggl/gc)
    dataset_generator_cmb.py       CMB spectra (tt/te/ee/pp)
              |                            |
              +------------+---------------+
                           |
                   generator_core.py (this file)
                   - CLI parser (identical flags for every driver)
                   - parameter sampling (emcee Gaussian / uniform)
                   - chain + .paramnames + .ranges + .covmat writers
                   - checkpoint save/load/append
                   - RAM-aware data-vector storage (default: one 2D array)
                   - MPI master/worker farm with timeouts

The subclass surface (everything a driver may override):

    VALID_PROBES              tuple of accepted train_args.probe names.
    EXTRA_TRAIN_KEYS          extra REQUIRED train_args keys (e.g. lrange).
    _read_train_args(ta)      read/validate the driver's own train_args
                              keys; called once during setup, after the
                              cobaya model exists (so a driver may also
                              add model requirements here).
    _compute_dvs_from_sample  one sample -> one data-vector payload.
    _dv_payload_names / _dv_payload_mapping /
    _dv_expected_payload_shape / _dv_payload_store
                              the exact payload members and their stores.
    _dv_alloc / _dv_write / _dv_zero / _dv_save / _dv_load_chk /
    _dv_append / _dv_chk_files
                              the data-vector STORE. The defaults below
                              implement the single 2D array the lensing
                              driver always used ({dvsf}.npy); a driver
                              whose payload is not one flat vector (the
                              CMB driver holds four per-spectrum arrays)
                              overrides all seven together.

Everything else (sampling, chain outputs, the MPI farm) is closed: the
.1.txt / .paramnames (with chi2*) / .ranges / .covmat sidecar
conventions come from here, so every family's dumps satisfy the same
getdist pairing rules automatically.

The bodies of the shared methods were MOVED VERBATIM from
dataset_generator_lensing.py (the porting discipline): the only
transformations are the module-global `args` becoming `self.args` and
the data-vector store lines becoming the _dv_* hook calls above.
"""
import numpy as np
import emcee, argparse, hashlib, os, sys, yaml, time, traceback
import psutil, gc, math, copy, tempfile, inspect
from cobaya.model import get_model
from mpi4py import MPI
from pathlib import Path
from getdist import loadMCSamples
from collections import deque
from contextlib import contextmanager
from numpy.lib.format import open_memmap
import contextlib, io

# The schema of the dataset's scientific record lives in the emulator package,
# one folder up from this file. The drivers are run by path, so the repo root is
# not on sys.path when this module is imported; dataset_generator_mps.py makes
# the same prepend before it imports the syren base. fixed_facts imports only
# hashlib at module scope (numpy, yaml and h5py are imported inside the
# functions that need them), so a generator run pays nothing to carry it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)
from emulator import fixed_facts
from emulator.parameter_table import resolve_parameter_table
from compute_data_vectors.dataset_manifest import (
  DATASET_PROBE_FAMILIES,
  DATASET_SAMPLING_POLICIES,
  UNIFORM_BOUNDARY_INTERIOR_POLICY as DATASET_UNIFORM_BOUNDARY_INTERIOR_POLICY,
  build_dataset_member_census,
  build_dataset_request_identity,
  load_checkpoint_or_refuse,
  require_checkpoint_members,
  scope_dataset_stem,
  validate_run_control,
)
from compute_data_vectors.dataset_publication import (
  begin_dataset_continuation,
  begin_dataset_generation,
  canonical_json_bytes,
  discard_dataset_draft,
  derive_dataset_slot,
  install_dataset_locator,
  load_active_generation,
  publish_dataset_generation,
)
from compute_data_vectors.generator_ingress import (
  convert_prior_bounds,
  direct_child_filename,
  load_parameter_covariance,
  native_integer,
  parameter_labels,
  select_unique_rows,
  validate_fiducial,
  validate_train_args,
)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Command line args (shared by every driver; the driver passes its prog name)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def make_cli_parser(prog):
  """
  Build the command-line parser shared by all dataset generators.

  Every driver exposes the SAME flags (documented in each driver's
  header comment); only the program name in the help text differs.

  Arguments:
    prog = the driver script name shown in --help (e.g.
           'dataset_generator_lensing').

  Returns:
    an argparse.ArgumentParser with the full shared flag set.
  """
  parser = argparse.ArgumentParser(prog=prog)

  parser.add_argument("--yaml",
                      dest="yaml",
                      help="The training YAML containing the training_args block",
                      type=str,
                      required=True)
  parser.add_argument("--root",
                      dest="root",
                      help="Project folder",
                      type=str,
                      required=True)
  parser.add_argument("--fileroot",
                      dest="fileroot",
                      help="Subfolder of Project folder where we find yaml and fisher",
                      type=str,
                      required=True)
  parser.add_argument("--datavsfile",
                      dest="datavsfile",
                      help="File to save data vectors",
                      type=str,
                      required=True)
  parser.add_argument("--paramfile",
                      dest="paramfile",
                      help="File to save parameters",
                      type=str,
                      required=True)
  parser.add_argument("--failfile",
                      dest="failfile",
                      help="File that tells which cosmo param fail to compute dvs",
                      type=str,
                      required=True)
  parser.add_argument("--chain",
                      dest="chain",
                      help="write only the parameter chain under isolated "
                           "*_chain_only filenames",
                      type=int,
                      choices=[0,1],
                      default=0)
  parser.add_argument("--nparams",
                      dest="nparams",
                      help="Requested Number of Parameters",
                      type=int,
                      required=True)
  parser.add_argument("--unif",
                      dest="unif",
                      help="Choose Between Uniform and Fisher based samples",
                      type=int,
                      choices=[0,1],
                      required=True)
  parser.add_argument("--temp",
                      dest="temp",
                      help="Number of Parameters to Generate",
                      type=int)
  parser.add_argument("--maxcorr",
                      dest="maxcorr",
                      help="Max correlation allowed",
                      type=float)
  parser.add_argument("--loadchk",
                      dest="loadchk",
                      help="Load from chk if exists",
                      type=int,
                      choices=[0,1])
  parser.add_argument("--freqchk",
                      dest="freqchk",
                      help="Load from chk if exists",
                      type=int)
  parser.add_argument("--append",
                      dest="append",
                      help="Append more models (only trye of loadchk == true)",
                      type=int,
                      choices=[0,1])
  parser.add_argument("--boundary",
                      dest="boundary",
                      help="Boundary setup: test/val requires boundaries to be cut",
                      type=float)
  parser.add_argument("--seed",
                      dest="seed",
                      help="Required integer sampling seed. Owns every random "
                           "draw (uniform sampling, the emcee walker init and "
                           "the sampler's own moves, and the thinning "
                           "subselection), so two runs with the same seed, "
                           "YAML, and code produce the same parameter table. "
                           "No default: an unrecorded seed cannot be replayed.",
                      type=int,
                      required=True)
  return parser
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Free Functions
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
UNIFORM_BOUNDARY_INTERIOR_POLICY = DATASET_UNIFORM_BOUNDARY_INTERIOR_POLICY


def resolve_uniform_sampling_support(names, bounds):
  """
  Move each requested uniform-sampling endpoint one representable value inward.

  The endpoint movement is defined in interval coordinates.  Its size therefore
  does not depend on the interval's distance from zero.

  Arguments:
    names = ordered parameter names, one name for each row of bounds.
    bounds = numeric array with shape (N, 2).  Each row is the requested
             [lower, upper] endpoint pair for one parameter.

  Returns:
    a dictionary containing the policy name, requested per-name endpoints,
    resolved per-name endpoints, and the resolved bounds array.

  Raises:
    ValueError = names and bounds do not have matching shapes, an endpoint is
                 nonfinite, a requested interval is not ordered, or no strictly
                 ordered representable interior remains.
  """
  try:
    parameter_names = list(names)
  except TypeError as exc:
    raise ValueError("Uniform sampling parameter names must be an ordered "
                     "collection.") from exc

  try:
    requested_bounds = np.asarray(bounds)
  except (TypeError, ValueError) as exc:
    raise ValueError("Uniform sampling bounds must be a numeric array with "
                     "shape (N, 2).") from exc

  if requested_bounds.ndim != 2 or requested_bounds.shape[1] != 2:
    raise ValueError("Uniform sampling bounds must have shape (N, 2); got "
                     f"{requested_bounds.shape}.")
  if requested_bounds.shape[0] != len(parameter_names):
    raise ValueError("Uniform sampling parameter names and bounds must have the "
                     "same row count; got "
                     f"{len(parameter_names)} names and "
                     f"{requested_bounds.shape[0]} bounds rows.")

  if not np.issubdtype(requested_bounds.dtype, np.floating):
    try:
      requested_bounds = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
      raise ValueError("Uniform sampling bounds must contain numeric endpoint "
                       "values.") from exc
  requested_bounds = np.array(requested_bounds, copy=True)
  resolved_bounds = np.empty_like(requested_bounds)
  requested_by_name = {}
  resolved_by_name = {}

  for index in range(len(parameter_names)):
    name = parameter_names[index]
    if name in requested_by_name:
      raise ValueError(f"Uniform sampling parameter name '{name}' is repeated; "
                       "each bounds row needs a unique parameter name.")

    low = requested_bounds[index, 0]
    high = requested_bounds[index, 1]
    if not np.isfinite(low):
      raise ValueError(f"Uniform sampling parameter '{name}' has a nonfinite "
                       f"requested lower endpoint: {low}.")
    if not np.isfinite(high):
      raise ValueError(f"Uniform sampling parameter '{name}' has a nonfinite "
                       f"requested upper endpoint: {high}.")
    if not low < high:
      raise ValueError(f"Uniform sampling parameter '{name}' needs a strictly "
                       "ordered requested interval; got "
                       f"[{low}, {high}].")

    resolved_low = np.nextafter(low, high)
    resolved_high = np.nextafter(high, low)
    if not np.isfinite(resolved_low) or not np.isfinite(resolved_high):
      raise ValueError(f"Uniform sampling parameter '{name}' has a nonfinite "
                       "resolved boundary interior.")
    if not resolved_low < resolved_high:
      raise ValueError(f"Uniform sampling parameter '{name}' has no strictly "
                       "ordered representable interior inside "
                       f"[{low}, {high}].")

    requested_by_name[name] = (float(low), float(high))
    resolved_by_name[name] = (float(resolved_low), float(resolved_high))
    resolved_bounds[index, 0] = resolved_low
    resolved_bounds[index, 1] = resolved_high

  support = {"policy":    UNIFORM_BOUNDARY_INTERIOR_POLICY,
             "requested": requested_by_name,
             "resolved":  resolved_by_name,
             "bounds":    resolved_bounds}
  return support


@contextmanager
def capture_native_output():
  """
  Redirect OS-level file descriptors to capture Fortran/C output.
  Unlike contextlib.redirect_stderr, this catches writes from any language
  (Fortran, C, etc.) that go directly to fd 1 (stdout) or fd 2 (stderr).
  Author: From Claude AI.
  """
  stdout_fd = sys.stdout.fileno()  # fd 1 — where stdout currently points
  stderr_fd = sys.stderr.fileno()  # fd 2 — where stderr currently points
  stdout_dup = os.dup(stdout_fd)   # bookmark original stdout destination
  stderr_dup = os.dup(stderr_fd)   # bookmark original stderr destination
  tmp = tempfile.TemporaryFile(mode='w+')
  try:
    os.dup2(tmp.fileno(), stdout_fd) # fd 1 now writes to tmp (not terminal)
    os.dup2(tmp.fileno(), stderr_fd) # fd 2 now writes to tmp (not terminal)
    yield tmp
  finally: # this block is always executed  regardless of exception generation
    os.dup2(stdout_dup, stdout_fd)   # restore fd 1 → terminal
    os.dup2(stderr_dup, stderr_fd)   # restore fd 2 → terminal
    os.close(stdout_dup)             # release the bookmarks
    os.close(stderr_dup)
    tmp.close()


def validate_worker_result_message(message, source, active):
  """Bind one MPI result to the row assigned to its sending worker.

  Rank zero records one ``worker -> (row, start time)`` assignment in
  ``active`` before it receives that worker's result. A result is safe to use
  only when it comes from a worker that still owns a task and reports that
  exact row. This check runs before the master deletes the assignment or
  changes a payload store, so a stale, duplicate, or malformed message cannot
  write a result into another parameter row.

  Arguments:
    message = the exact three-item tuple received from the worker.
    source = the positive integer rank reported by MPI for that message.
    active = rank zero's current mapping from worker rank to assigned row and
             task start time.

  Returns:
    ``(kind, row, payload)`` with ``row`` taken from the recorded active
    assignment. ``kind`` is either ``"ok"`` or ``"err"``.

  Raises:
    RuntimeError when the message does not follow the worker protocol or does
    not match the sender's live assignment.
  """
  if type(source) is not int or source < 1:
    raise RuntimeError(
      "MPI result source must be a positive worker rank; got "
      + repr(source))
  if source not in active:
    raise RuntimeError(
      "MPI result came from worker " + str(source)
      + " without a live row assignment; refuse a stale or duplicate result")
  assignment = active[source]
  if type(assignment) is not tuple or len(assignment) != 2 \
      or type(assignment[0]) is not int or assignment[0] < 0:
    raise RuntimeError(
      "MPI active assignment for worker " + str(source)
      + " is malformed: " + repr(assignment))
  if type(message) is not tuple or len(message) != 3:
    raise RuntimeError(
      "MPI worker result must be the tuple (kind, row, payload); got "
      + repr(message))
  kind, reported_row, payload = message
  if type(kind) is not str or kind not in ("ok", "err"):
    raise RuntimeError(
      "MPI worker result kind must be 'ok' or 'err'; got " + repr(kind))
  if type(reported_row) is not int or reported_row < 0:
    raise RuntimeError(
      "MPI worker result row must be a nonnegative native integer; got "
      + repr(reported_row))
  assigned_row = assignment[0]
  if reported_row != assigned_row:
    raise RuntimeError(
      "MPI worker " + str(source) + " reported row " + str(reported_row)
      + " but rank zero assigned row " + str(assigned_row)
      + "; refuse to write a payload under the wrong parameter row")
  if kind == "err" and type(payload) is not str:
    raise RuntimeError(
      "MPI worker error payload must be traceback text; got "
      + type(payload).__name__)
  return kind, assigned_row, payload


def validate_worker_done_message(message, source, active):
  """Validate one worker's acknowledgement of the MPI stop request.

  The acknowledgement must come from a worker still awaiting shutdown and
  must name the same rank.  Validation precedes removal from ``active`` so a
  stray message cannot make rank zero believe another worker stopped.

  Arguments:
    message = the exact two-item tuple received from the worker.
    source = the positive integer rank reported by MPI for that message.
    active = rank zero's mapping of workers still awaiting shutdown.

  Returns:
    The validated source rank.

  Raises:
    RuntimeError when the message is malformed, comes from an unexpected
    worker, or names a different worker.
  """
  if type(source) is not int or source < 1:
    raise RuntimeError(
      "MPI stop acknowledgement source must be a positive worker rank; got "
      + repr(source))
  if source not in active:
    raise RuntimeError(
      "MPI stop acknowledgement came from worker " + str(source)
      + " that is not awaiting shutdown")
  if type(message) is not tuple or len(message) != 2:
    raise RuntimeError(
      "MPI stop acknowledgement must be ('worker done', source rank); got "
      + repr(message) + " from worker " + str(source))
  label, reported_source = message
  if type(label) is not str or label != "worker done" \
      or type(reported_source) is not int or reported_source != source:
    raise RuntimeError(
      "MPI stop acknowledgement must be ('worker done', source rank); got "
      + repr(message) + " from worker " + str(source))
  return source
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# The dataset's scientific record (emulator/fixed_facts.py owns the schema)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Every probe a driver accepts belongs to exactly one output family, named here
# the way the training stack names it: cs / ggl / gc are the cosmolike data
# vectors, cmblensed / cmbunlensed the CMB spectra, mps the matter-power surface
# on its (z, k) grid, background the background functions on a z grid. A probe
# missing from this table stops the run, because a dataset that cannot say which
# family it feeds is a dataset no consumer can safely read.
# One pure request-contract module owns this mapping so the publication
# identity, family member census, and scientific sidecar cannot classify one
# probe three different ways.

# The units the CMB driver asks cobaya for at every sample
# (dataset_generator_cmb.py calls get_Cl(ell_factor=False, units="muK2")). The
# spectra are published in these units, so the record carries them. A family
# with no angular power spectrum has no such units and says so instead.
CMB_CL_UNITS = "muK2"

# The name a cobaya run gives the neutrino splitting when it states one. CAMB
# has its own default; the record does not supply it, because a convention the
# run never wrote down is not a fact about this run.
NEUTRINO_CONVENTION_KEY = "neutrino_hierarchy"


def _dark_energy_publication_facts(parameterization, pinned):
  """Describe the run's dark-energy coordinates in canonical ``w, wa`` form.

  Cobaya permits a run to sample coordinates that are convenient for the
  sampler and then calculate the coordinates consumed by a theory.  The
  shipped matter-power configuration uses exactly that feature: it samples
  ``w0pwa`` and ``w`` and calculates ``wa = w0pwa - w``.  Looking only at the
  sampled names therefore misses a varying ``wa`` and records the wrong
  physical law.

  This function reads Cobaya's public parameterization surfaces.  A calculated
  input varies when any of its declared dependencies is sampled.  The returned
  record always uses the physical names ``w`` and ``wa`` even when the YAML
  spells the present-day value ``w0`` or supplies the sum ``w0pwa = w0 + wa``.

  Arguments:
    parameterization = the resolved model's Cobaya Parameterization object.
    pinned = all constants resolved from the parameter and theory components.

  Returns:
    ``(fixed, varying, law, inputs)``. ``fixed`` maps canonical fixed
    coordinates to values, ``varying`` is the set of canonical coordinates
    that change across samples, and ``law`` / ``inputs`` are the values written
    to the scientific sidecar.

  Raises:
    TypeError or ValueError when a required public surface is unavailable or a
    fixed coordinate is non-numeric, non-finite, or internally inconsistent.
  """
  surfaces = {}
  for name in ("input_params", "constant_params", "sampled_params"):
    reader = getattr(parameterization, name, None)
    if not callable(reader):
      raise TypeError(
        "the resolved Cobaya parameterization must provide the public "
        + name + "() mapping before dark-energy facts can be published")
    values = reader()
    if hasattr(values, "items"):
      surfaces[name] = dict(values.items())
    elif name != "constant_params" and isinstance(
        values, (list, tuple, set, frozenset)):
      if any(type(item) is not str for item in values):
        raise TypeError(
          "Cobaya parameterization." + name
          + "() returned a name collection containing a non-string value")
      surfaces[name] = {item: None for item in values}
    else:
      raise TypeError(
        "Cobaya parameterization." + name
        + "() must return a name mapping"
        + (" or a name collection" if name != "constant_params" else "")
        + "; got " + type(values).__name__)

  dependencies = getattr(parameterization, "input_dependencies", None)
  if not hasattr(dependencies, "items"):
    raise TypeError(
      "the resolved Cobaya parameterization must provide the public "
      "input_dependencies mapping before dark-energy facts can be published")
  dependencies = dict(dependencies.items())

  input_names = set(surfaces["input_params"])
  sampled_names = set(surfaces["sampled_params"])
  varying_names = set(sampled_names)
  for name, required in dependencies.items():
    if name not in input_names:
      raise ValueError(
        "Cobaya input_dependencies names " + repr(name)
        + ", but input_params() does not contain that calculated input")
    if not isinstance(required, (set, frozenset, list, tuple)):
      raise TypeError(
        "Cobaya input_dependencies[" + repr(name)
        + "] must be a collection of parameter names; got "
        + type(required).__name__)
    if set(required).intersection(sampled_names):
      varying_names.add(name)

  # Parameter constants take precedence over theory-component defaults, just
  # as fixed_facts.resolved_constants specifies.  Ignore a fixed-looking value
  # for a coordinate Cobaya says varies; a theory default must not pin a
  # sampled parameter in the published record.
  resolved = dict(pinned)
  resolved.update(surfaces["constant_params"])

  # This is the same representation tolerance owned publicly by
  # emulator.syren_base.DARK_ENERGY_COORDINATE_ATOL.  Repeating the one-line
  # float32 calculation here avoids importing the MPS analytic formulas into
  # the lensing, CMB, and background generators merely to publish their facts.
  atol = 4.0 * float(np.finfo(np.float32).eps)

  def fixed_number(name):
    if name not in resolved or name in varying_names:
      return None
    value = resolved[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)):
      raise TypeError(
        "the fixed dark-energy coordinate " + repr(name)
        + " must be a real scalar; got " + repr(value))
    value = float(value)
    if not math.isfinite(value):
      raise ValueError(
        "the fixed dark-energy coordinate " + repr(name)
        + " must be finite; got " + repr(value))
    return value

  w = fixed_number("w")
  w0 = fixed_number("w0")
  if w is not None and w0 is not None and not np.isclose(
      w, w0, rtol=0.0, atol=atol):
    raise ValueError(
      "the resolved Cobaya model gives inconsistent fixed aliases w="
      + repr(w) + " and w0=" + repr(w0))
  present = w if w is not None else w0
  wa = fixed_number("wa")
  w0pwa = fixed_number("w0pwa")
  if present is not None and w0pwa is not None:
    derived_wa = w0pwa - present
    if wa is not None and not np.isclose(
        wa, derived_wa, rtol=0.0, atol=atol):
      raise ValueError(
        "the resolved Cobaya model gives inconsistent fixed dark-energy "
        "coordinates: w0pwa=" + repr(w0pwa) + " but w0 + wa="
        + repr(present + wa))
    wa = derived_wa
  elif w0pwa is not None and present is None:
    raise ValueError(
      "the resolved Cobaya model fixes w0pwa but does not fix w or w0, so "
      "the generator cannot determine the separate w and wa values to publish")

  varying = set()
  if varying_names.intersection(("w", "w0")):
    varying.add("w")
  if "wa" in varying_names or "w0pwa" in varying_names:
    varying.add("wa")

  if "w" not in varying and present is None:
    present = -1.0
  if "wa" not in varying and wa is None:
    wa = 0.0

  fixed = {}
  if "w" not in varying and present is not None:
    fixed["w"] = present
  if "wa" not in varying and wa is not None:
    fixed["wa"] = wa

  if "wa" in varying or (wa is not None and not np.isclose(
      wa, 0.0, rtol=0.0, atol=atol)):
    law = "w0wa-cpl"
    inputs = ["w", "wa"]
  elif "w" in varying or (present is not None and not np.isclose(
      present, -1.0, rtol=0.0, atol=atol)):
    law = "constant-w"
    inputs = ["w"]
  else:
    law = "cosmological-constant"
    inputs = []
  return fixed, varying, law, inputs


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Class Definition
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class GeneratorCore:
  """
  Base class for dataset-generator drivers (subclass surface: see the
  module docstring). A driver instantiates it with the parsed CLI args;
  construction runs the whole job (sampling on rank 0, then the MPI
  data-vector farm unless --chain 1).
  """
  VALID_PROBES = ()      # driver MUST override: accepted train_args.probe
  EXTRA_TRAIN_KEYS = ()  # driver MAY override: extra required train_args keys

  #-----------------------------------------------------------------------------
  # init
  #-----------------------------------------------------------------------------
  def __init__(self, cli_args):
    run_control = validate_run_control(
      loadchk=cli_args.loadchk,
      append=cli_args.append,
      chain=cli_args.chain)
    self.args = cli_args
    self.run_control = run_control
    self.setup = False
    self.__setup_flags()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_zero_error = None
    rank_zero_failure = None
    if rank == 0:
      try:
        # Resume and append first authenticate the completed generation they
        # name. A fresh run has no files to read, so it keeps the sampled rows
        # in memory and creates its private draft only after their exact count
        # and values have passed validation inside __run_mcmc.
        if self.run_control.operation != "fresh":
          self._prepare_dataset_publication()
        self.__run_mcmc()
      except Exception as error:
        # Other ranks have finished setup and are waiting to enter the MPI
        # data-vector farm. Tell them about a rank-zero refusal before this
        # rank re-raises it, so an expected append or resume refusal cannot
        # leave the workers waiting forever.
        rank_zero_error = error
        rank_zero_failure = type(error).__name__ + ": " + str(error)
    rank_zero_failure = comm.bcast(rank_zero_failure, root=0)
    if rank_zero_failure is not None:
      if rank == 0:
        raise rank_zero_error
      raise RuntimeError(
        "rank 0 stopped before MPI data-vector work: " + rank_zero_failure)
    if self.run_control.dataset_mode == "full":
      self.__generate_datavectors()
    # Full-dataset workers reach this point only after rank 0 has received every
    # explicit DTAG shutdown acknowledgement. Chain-only workers wait here while
    # rank 0 finishes the chain. No generation can become visible until every
    # rank has therefore stopped using the draft.
    comm.Barrier()
    if rank == 0:
      self._publish_dataset_generation()

  def __setup_flags(self):
    #---------------------------------------------------------------------------
    # Basic definitions
    #---------------------------------------------------------------------------
    root_env = os.environ.get("ROOTDIR")
    if not root_env:
      raise RuntimeError("ROOTDIR environment variable is not set")
    root = root_env.rstrip("/")
    root = f"{root}/{self.args.root.rstrip('/')}"
    fileroot = f"{root}/{self.args.fileroot.rstrip('/')}"

    self.append = self.run_control.append
    self.bounds = None           # the support the sampler drew from (resolved)
    self.bounds_requested = None # the support the prior declared (requested)
    boundary = 1.0 if self.args.boundary is None else self.args.boundary
    if type(boundary) is not float or not math.isfinite(boundary) \
        or not 0.0 < boundary <= 1.0:
      raise ValueError(
        "--boundary must be a finite native float in (0, 1]; got "
        + repr(boundary))
    self.bounds_adj = boundary
    self.covmat = None
    self.dtype = np.float32
    self.dvsf = None
    self.dataset_member_directory = None
    self.dataset_members = None
    self.dataset_route = None
    self.derived = True
    self.dvs_is_memmap = False
    self.freqchk = 5000 if self.args.freqchk is None else self.args.freqchk
    native_integer(self.freqchk, "--freqchk", minimum=1000)
    self.failed = None        # track which models failed to compute dv
    self.failf = None
    self.fiducial = None
    self.inv_covmat = None
    self.loadchk = self.run_control.loadchk
    self.loadedfromchk = False  # check if loaded from checkpoint sucessfully
    self.loadedsamples = False  # check loaded samples sucessfully
    self.maxcorr = 0.15 if self.args.maxcorr is None else self.args.maxcorr
    if type(self.maxcorr) is not float or not math.isfinite(self.maxcorr) \
        or not 0.01 < self.maxcorr <= 1.0:
      raise ValueError(
        "--maxcorr must be a finite native float in (0.01, 1]; got "
        + repr(self.maxcorr))
    # the sampling seed: a required non-bool integer, no default. Every random
    # draw goes through this owned Generator instead of the process-global
    # np.random, so two runs with the same seed, YAML and code produce the same
    # parameter table (bool is refused: argparse int never yields one, but a
    # programmatic caller might pass True, which would silently mean seed 1).
    self.seed = native_integer(self.args.seed, "--seed", minimum=0)
    self.rng = np.random.default_rng(self.seed)
    self.names = None
    self.model = None
    self.nparams = 10000 if self.args.nparams is None else self.args.nparams
    native_integer(self.nparams, "--nparams", minimum=200)
    self.paramsf = None
    self.probe = None
    self.sampled_params = None
    self.samples = None
    self.temp = 128 if self.args.temp is None else self.args.temp
    native_integer(self.temp, "--temp", minimum=1)
    self.unif = 0 if self.args.unif is None else self.args.unif
    native_integer(self.unif, "--unif", allowed=(0, 1))
    self.yaml = f"{fileroot}/test.yaml" if self.args.yaml is None else f"{fileroot}/{self.args.yaml}"
    if not os.path.isfile(f"{self.yaml}"):
      raise FileNotFoundError(f"YAML file not found: {self.yaml}")
    #---------------------------------------------------------------------------
    # Load yaml
    #---------------------------------------------------------------------------
    with open(self.yaml, 'r') as stream:
      info = yaml.safe_load(stream)

    if info is None:
      raise ValueError(f"YAML file is empty or invalid: {self.yaml}")

    if not isinstance(info, dict):
      raise ValueError(f"Cobaya YAML did not parse to a dict: {self.yaml}")
    # Bind the user's parsed configuration before Cobaya or a driver receives
    # the mapping. A downstream library is then free to normalize its private
    # copy without changing the identity of the request the user supplied.
    configuration_sha256 = hashlib.sha256(
      canonical_json_bytes(info)).hexdigest()

    missing = [k for k in ['params', 'likelihood', 'train_args'] if k not in info]
    if missing:
      raise KeyError(f"Cobaya YAML missing required blocks {missing}: {self.yaml}")

    train_args = info["train_args"]
    self.sampled_params = validate_train_args(
      train_args,
      extra_keys=self.EXTRA_TRAIN_KEYS,
      uniform=(self.unif == 1))

    #---------------------------------------------------------------------------
    # Load Cobaya model (needed for computing likelihood), cov matrix...
    #---------------------------------------------------------------------------
    try:
      self.model = get_model(info)
    except Exception as e:
      raise RuntimeError(f"get_model failed for {self.yaml}: {e}") from e

    self.probe = train_args["probe"]
    if type(self.probe) is not str or not self.probe:
      raise ValueError(
        "train_args.probe must be a nonempty native string; got "
        + repr(self.probe))
    if self.probe not in self.VALID_PROBES:
      raise ValueError(f"Invalid Probe: {self.probe}")

    # driver-specific train_args (and model requirements, if any) --------------
    self._read_train_args(train_args)

    if not self.unif == 1:
      self.fiducial = validate_fiducial(
        train_args["fiducial"], self.sampled_params, dtype=self.dtype)
      raw_covmat_file = direct_child_filename(
        train_args["params_covmat_file"],
        "train_args.params_covmat_file")
      covmat = load_parameter_covariance(
        Path(self.yaml).parent / raw_covmat_file,
        self.sampled_params)
      covmat = np.array(covmat, copy=True, dtype=self.dtype)
      if not np.isfinite(covmat).all():
        raise ValueError(
          "the selected parameter covariance is not finite after conversion "
          "to the generator's float32 dtype")

    # Reorder bounds -----------------------------------------------------------
    self.names = list(self.model.parameterization.sampled_params().keys())
    invalid_model_names = [name for name in self.names
                           if type(name) is not str or not name]
    if invalid_model_names:
      raise ValueError(
        "Cobaya sampled parameter names must be nonempty native strings; got "
        + repr(invalid_model_names))
    repeated_model_names = sorted(
      {name for name in self.names if self.names.count(name) > 1})
    missing_names = [name for name in self.names
                     if name not in self.sampled_params]
    extra_names = [name for name in self.sampled_params
                   if name not in self.names]
    if repeated_model_names or missing_names or extra_names \
        or len(self.sampled_params) != len(self.names):
      raise ValueError(
        "train_args.ord must be one unique permutation of Cobaya's sampled "
        "parameters; duplicate Cobaya names=" + repr(repeated_model_names)
        + ", missing from ord=" + repr(missing_names)
        + ", extra in ord=" + repr(extra_names)
        + ", ord count=" + str(len(self.sampled_params))
        + ", Cobaya count=" + str(len(self.names)))
    idx = self.reorder_idx_from_yaml_to_ord()

    model_info = self.model.info()
    if type(model_info) is not dict or type(model_info.get("params")) is not dict:
      raise ValueError(
        "Cobaya model information must contain a params mapping before output "
        "labels can be prepared")
    self.parameter_labels = parameter_labels(
      model_info["params"], self.sampled_params)

    # Cobaya may report infinite confidence=1 endpoints for a Gaussian prior.
    # The finite confidence interval supplies the width used to resolve those
    # endpoints. Both matrices must still have one ordered row per sampled
    # parameter before the generator reorders or converts them.
    hard_bounds = np.asarray(
      self.model.prior.bounds(confidence=1.0), dtype=np.float64)
    finite_bounds = np.asarray(
      self.model.prior.bounds(confidence=0.9999994), dtype=np.float64)
    expected_bounds_shape = (len(self.names), 2)
    if hard_bounds.shape != expected_bounds_shape:
      raise ValueError(
        "Cobaya confidence=1 prior bounds must have shape "
        + repr(expected_bounds_shape) + "; got " + repr(hard_bounds.shape))
    if finite_bounds.shape != expected_bounds_shape:
      raise ValueError(
        "Cobaya finite prior bounds must have shape "
        + repr(expected_bounds_shape) + "; got " + repr(finite_bounds.shape))
    tmp, self.bounds = convert_prior_bounds(
      hard_bounds[idx, :],
      finite_bounds[idx, :],
      dtype=self.dtype)

    # the support as the prior declared it, copied before the two mutations
    # below rewrite self.bounds in place: the infinite-endpoint stretch (a
    # Gaussian prior has no hard edge, so the sampler is handed one) and the
    # accuracy margin the --boundary flag trims off each edge for a test or
    # validation dump. self.bounds is then the support the sampler actually drew
    # from; self.bounds_requested is the one the prior asked for. Both are
    # published in the dataset's record, because they answer different questions
    # and they are different numbers whenever either mutation fires.
    self.bounds_requested = np.array(self.bounds,
                                     copy=True,
                                     dtype=self.dtype)

    for i in range(len(tmp)):
      if math.isinf(tmp[i,0]) or math.isinf(tmp[i,1]):
        width = (self.bounds[i, 1] - self.bounds[i, 0])
        if math.isinf(tmp[i,0]):
          self.bounds[i,0] -= self.temp*width/5.0
        if math.isinf(tmp[i,1]):
          self.bounds[i,1] += self.temp*width/5.0

    # near the bds: emulator accuracy degrades: (val/test) must reduce bds -----
    if self.bounds_adj < 1:
      margin = (1-self.bounds_adj) * 0.5*(self.bounds[:, 1]-self.bounds[:, 0])
      self.bounds[:, 0] += margin
      self.bounds[:, 1] -= margin
    if not np.isfinite(self.bounds).all() \
        or not (self.bounds[:, 0] < self.bounds[:, 1]).all():
      raise ValueError(
        "the resolved sampling bounds must remain finite and increasing after "
        "temperature stretching and the boundary margin")

    # adjust covmat- -----------------------------------------------------------
    if not self.unif == 1:
      #-------------------------------------------------------------------------
      # Reduce correlation on the covariance matrix to max = args.maxcorr
      #-------------------------------------------------------------------------
      if covmat.shape != (len(self.sampled_params), len(self.sampled_params)):
        raise ValueError(
          "the selected parameter covariance has the wrong shape: "
          + repr(covmat.shape))
      if not np.isfinite(covmat).all() \
          or not np.allclose(covmat, covmat.T, rtol=0.0, atol=0.0):
        raise ValueError(
          "the selected parameter covariance must remain finite and symmetric")
      diagonal = np.diag(covmat)
      if not (diagonal > 0.0).all():
        raise ValueError(
          "the selected parameter covariance must have a positive diagonal")
      sig   = np.sqrt(diagonal)
      n = len(sig)
      outer = np.outer(sig, sig)
      corr  = covmat / outer
      if not np.isfinite(corr).all():
        raise ValueError(
          "the parameter correlation matrix contains a nonfinite value")
      m = np.abs(corr - np.eye(n)).max()
      if m > self.maxcorr:
        corr /= max(1.0, m / self.maxcorr) if m > 0 else 1.0
        np.fill_diagonal(corr, 1.0)
      covmat = corr * outer
      if not np.isfinite(covmat).all() \
          or not np.allclose(covmat, covmat.T, rtol=0.0, atol=0.0):
        raise ValueError(
          "the adjusted parameter covariance must remain finite and symmetric")

      #-------------------------------------------------------------------------
      # Compute covmat inverse
      #-------------------------------------------------------------------------
      C = np.array(covmat, copy=True, dtype=self.dtype)
      jitt = 0.0
      for _ in range(10):
        try:
          L = np.linalg.cholesky(C + jitt*np.eye(C.shape[0]))
          break
        except np.linalg.LinAlgError:
          scale = np.mean(np.diag(C)) # scale jitt to matrix sz start tiny -> grow
          jitt = (1e-12 * scale if jitt == 0 else jitt*10)
      else:
        raise np.linalg.LinAlgError("could not stabilized cov to SPD w/ jitter")
      I = np.eye(C.shape[0])
      self.covmat = C + jitt*np.eye(C.shape[0])
      self.inv_covmat = np.linalg.solve(L.T, np.linalg.solve(L, I))
      if not np.isfinite(L).all() or not np.isfinite(self.covmat).all() \
          or not np.isfinite(self.inv_covmat).all():
        raise ValueError(
          "the covariance factor or inverse contains a nonfinite value")
      if not np.allclose(
          L @ L.T, self.covmat, rtol=2e-6, atol=2e-6):
        raise ValueError(
          "the covariance Cholesky factor does not reconstruct the stored "
          "covariance")
      if not np.allclose(
          self.covmat @ self.inv_covmat, I, rtol=2e-5, atol=2e-5):
        raise ValueError(
          "the parameter covariance and its inverse do not reconstruct the "
          "identity matrix")

    #---------------------------------------------------------------------------
    # Define output files
    #---------------------------------------------------------------------------
    datavsfile = Path(self.args.datavsfile).stem
    paramfile = Path(self.args.paramfile).stem
    failfile = Path(self.args.failfile).stem
    if not self.unif == 1:
      self.dvsf = f"{root}/chains/{datavsfile}_{self.probe}_{self.temp}"
      self.paramsf = f"{root}/chains/{paramfile}_{self.probe}_{self.temp}"
      self.failf = f"{root}/chains/{failfile}_{self.probe}_{self.temp}"
    else:
      self.dvsf = f"{root}/chains/{datavsfile}_{self.probe}_unifs"
      self.paramsf = f"{root}/chains/{paramfile}_{self.probe}_unifs"
      self.failf = f"{root}/chains/{failfile}_{self.probe}_unifs"
    self.dvsf = scope_dataset_stem(
      self.dvsf, self.run_control.dataset_mode)
    self.paramsf = scope_dataset_stem(
      self.paramsf, self.run_control.dataset_mode)
    self.failf = scope_dataset_stem(
      self.failf, self.run_control.dataset_mode)
    self._bind_dataset_member_census()
    self._bind_dataset_publication_request(configuration_sha256)

    #---------------------------------------------------------------------------
    # Validate fiducial is inside the sampling bounds (Gaussian sampling only)
    #---------------------------------------------------------------------------
    if not self.unif == 1:
      lp = self.__param_logpost(self.fiducial)
      if not math.isfinite(lp):
        oob = {p: (float(v), float(lo), float(hi))
               for p, v, lo, hi in zip(self.sampled_params, self.fiducial,
                                       self.bounds[:, 0], self.bounds[:, 1])
               if not (lo <= v <= hi)}
        raise ValueError(
          f"train_args.fiducial has a non-finite log-posterior ({lp}): it lies "
          f"outside the (temperature-stretched) sampling bounds. Fix the fiducial "
          f"or widen the corresponding prior.\n"
          f"Offending params [name: (value, low, high)]: {oob}")

    #---------------------------------------------------------------------------
    # Setup Done
    #---------------------------------------------------------------------------
    self.setup = True

  #-----------------------------------------------------------------------------
  # driver hooks (see the module docstring for the full subclass surface)
  #-----------------------------------------------------------------------------
  def _read_train_args(self, train_args):
    """
    Read/validate the driver's own train_args keys. Called once during
    setup, after self.model / self.probe / self.sampled_params exist.
    The base class needs nothing beyond the shared keys, so this is a
    no-op by default.

    Arguments:
      train_args = the YAML's train_args mapping.
    """
    return

  def _generator_program_name(self):
    """Return the filename stem that defines the concrete driver class.

    The registry states which driver a probe requires, but it cannot prove
    which Python file defined the object that is running. This method obtains
    that independent fact from the concrete class's loaded module.

    Raises:
      ValueError when the defining module or its Python file cannot be proved.
    """
    module_name = type(self).__module__
    module = sys.modules.get(module_name)
    defining_file = getattr(module, "__file__", None)
    if type(defining_file) is not str or not defining_file:
      raise ValueError(
        "the dataset generator's defining Python file cannot be proved for "
        "module " + repr(module_name))
    program_name = Path(defining_file).stem
    if not program_name:
      raise ValueError(
        "the dataset generator's defining Python file has no filename stem: "
        + repr(defining_file))
    return program_name

  def _family_variant(self):
    """Return the publication variant used by non-Grid2D drivers."""
    return "standard"

  def _bind_dataset_member_census(self):
    """Bind one validated route and checkpoint census without reading files.

    The three output stems must share one folder. Only their basenames enter
    the portable member census; the common folder is stored separately and is
    joined to those names when a requested checkpoint is inspected.

    Raises:
      ValueError when the stems use different folders, the concrete driver
      does not match its probe, or the route/member names are invalid.
    """
    parameter_path = Path(os.path.abspath(os.path.normpath(self.paramsf)))
    data_vector_path = Path(os.path.abspath(os.path.normpath(self.dvsf)))
    failure_path = Path(os.path.abspath(os.path.normpath(self.failf)))
    member_directory = parameter_path.parent
    if data_vector_path.parent != member_directory \
        or failure_path.parent != member_directory:
      raise ValueError(
        "dataset parameter, data-vector, and failure stems must share one "
        "folder; got " + repr((
          str(parameter_path.parent),
          str(data_vector_path.parent),
          str(failure_path.parent))))

    probe = self.probe
    if probe not in DATASET_PROBE_FAMILIES:
      raise ValueError(
        "the dataset probe has no registered output family: " + repr(probe))
    census = build_dataset_member_census(
      dataset_mode=self.run_control.dataset_mode,
      family=DATASET_PROBE_FAMILIES[probe],
      family_variant=self._family_variant(),
      generator=self._generator_program_name(),
      probe=probe,
      params_stem=parameter_path.name,
      dvs_stem=data_vector_path.name,
      fail_stem=failure_path.name)
    self.dataset_route = census.route
    self.dataset_members = census.members
    self.dataset_member_directory = member_directory

  def _facts_sidecar_text(self, names, dataset_id, resolved_bounds=None):
    """Build the scientific sidecar for one chain or provisional request.

    ``dataset_id`` names committed chain bytes when the sidecar is written. A
    provisional all-zero digest may instead be supplied while the immutable
    request identity is being prepared: the scientific-contract projection
    deliberately removes that generation-specific field before hashing.

    Uniform sampling moves each endpoint one representable value inward just
    before drawing. ``resolved_bounds`` lets request construction describe that
    future support without mutating ``self.bounds`` before the sampling gate.

    Arguments:
      names = sampled parameter names in their canonical column order.
      dataset_id = SHA-256 of committed chain bytes, or an all-zero provisional
                   value used only while the invariant request is built.
      resolved_bounds = optional final sampling bounds. None uses
                        ``self.bounds``.

    Returns:
      The validated YAML text owned by ``emulator.fixed_facts``.

    Raises:
      ValueError when the resolved facts or parameter support cannot form a
      valid scientific record. This method changes no generator state.
    """
    facts = self._resolve_fixed_facts()
    resolved_source = self.bounds if resolved_bounds is None else resolved_bounds
    requested = {}
    resolved = {}
    for index, name in enumerate(names):
      requested[name] = (self.bounds_requested[index, 0],
                         self.bounds_requested[index, 1])
      resolved[name] = (resolved_source[index, 0],
                        resolved_source[index, 1])
    return fixed_facts.build_sidecar(
      dataset_id=dataset_id,
      generator=facts["generator"],
      family=facts["family"],
      cosmology_fixed=facts["cosmology_fixed"],
      neutrino_convention=facts["neutrino_convention"],
      flat_only=facts["flat_only"],
      dark_energy_law=facts["dark_energy_law"],
      dark_energy_inputs=facts["dark_energy_inputs"],
      cl_units=facts["cl_units"],
      base_identity=facts["base_identity"],
      names=names,
      requested=requested,
      resolved=resolved)

  def _build_dataset_request_identity(self, configuration_sha256):
    """Build the exact request that a locator and generation authenticate.

    Arguments:
      configuration_sha256 = digest of the canonical parsed input YAML before
                              Cobaya or a driver can normalize its own copy.

    Returns:
      A validated native mapping containing the output route, parameter order,
      sampling controls, random-generator policy, and invariant science digest.

    Raises:
      ValueError when a route, sampling control, bound, or scientific fact is
      invalid. The method predicts uniform endpoint movement without drawing a
      random number or changing the stored bounds.
    """
    names = list(self.sampled_params)
    sampling_mode = "uniform" if self.unif == 1 else "gaussian-mcmc"
    policy = DATASET_SAMPLING_POLICIES[sampling_mode]

    resolved_bounds = self.bounds
    if sampling_mode == "uniform":
      provisional_support = resolve_uniform_sampling_support(
        names=names,
        bounds=self.bounds)
      resolved_bounds = provisional_support["bounds"]
    provisional_text = self._facts_sidecar_text(
      names=names,
      dataset_id="0" * 64,
      resolved_bounds=resolved_bounds)
    provisional_blocks = fixed_facts.parse_sidecar(
      text=provisional_text,
      where="the provisional dataset scientific record")
    scientific_digest = fixed_facts.scientific_contract_digest(
      blocks=provisional_blocks,
      where="the provisional dataset scientific record")
    return build_dataset_request_identity(
      dataset_mode=self.dataset_route["dataset_mode"],
      family=self.dataset_route["family"],
      family_variant=self.dataset_route["family_variant"],
      generator=self.dataset_route["generator"],
      probe=self.dataset_route["probe"],
      sampling_mode=sampling_mode,
      temperature=self.temp,
      boundary_factor=self.bounds_adj,
      max_correlation=(None if sampling_mode == "uniform"
                       else self.maxcorr),
      sampling_algorithm=policy["algorithm"],
      seed=self.seed,
      rng_bit_generator=policy["bit_generator"],
      rng_emcee_random=policy["emcee_random"],
      rng_policy=policy["policy"],
      boundary_interior_policy=(
        DATASET_UNIFORM_BOUNDARY_INTERIOR_POLICY
        if sampling_mode == "uniform" else None),
      ordered_names=names,
      configuration_sha256=configuration_sha256,
      scientific_contract_sha256=scientific_digest)

  def _bind_dataset_publication_request(self, configuration_sha256):
    """Preserve logical output names and bind one immutable request.

    Arguments:
      configuration_sha256 = canonical parsed-YAML digest passed to
                              ``_build_dataset_request_identity``.

    Returns:
      None. The method records the logical stems, stable dataset slot, request
      identity, and empty draft/publication state on this generator.

    Raises:
      ValueError when the output stems, route, or request identity is invalid.
      No file is created here.
    """
    self.logical_paramsf = self.paramsf
    self.logical_dvsf = self.dvsf
    self.logical_failf = self.failf
    self.dataset_slot = derive_dataset_slot(
      self.dataset_member_directory,
      params_stem=self.logical_paramsf,
      dvs_stem=self.logical_dvsf,
      fail_stem=self.logical_failf,
      dataset_mode=self.dataset_route["dataset_mode"],
      family=self.dataset_route["family"])
    self.dataset_identity = self._build_dataset_request_identity(
      configuration_sha256)
    self.dataset_locator = None
    self.dataset_draft = None
    self.dataset_expected_active_sha256 = None

  def _prepare_dataset_publication(self):
    """Create rank zero's private draft without exposing a member file.

    Returns:
      None. A fresh run receives an empty draft. A resume receives a private
      authenticated copy of the current completed generation. Output stems are
      rebound to that draft only on rank zero.

    Raises:
      RuntimeError for unsupported append or an existing fresh output.
      DatasetPublicationError when a locator or active generation is missing,
      corrupt, or belongs to another request. The active generation is never
      changed by a refusal.
    """
    operation = self.run_control.operation
    preflight_source = None
    if operation in ("resume", "append"):
      preflight_source = self._preflight_active_checkpoint()
    if operation == "append":
      raise RuntimeError(
        "--append=1 authenticated and semantically validated the active "
        "dataset, but exact append is not available yet. The generator does "
        "not persist the NumPy, emcee, walker, and row-selection state needed "
        "to continue without repeating or skipping samples. Use a fresh "
        "logical output; no locator, draft, or active dataset was changed.")

    if operation == "fresh":
      # Setup and in-memory sampling have already succeeded. This is the first
      # permitted output mutation for a fresh run; malformed configuration and
      # a unique-row shortfall therefore leave even the chains folder absent.
      self.dataset_slot.chains_dir.mkdir(
        mode=0o700, parents=True, exist_ok=True)
    members = dict(self.dataset_members)
    if operation == "fresh":
      locator = install_dataset_locator(
        self.dataset_slot,
        identity=self.dataset_identity,
        members=members)
      if os.path.lexists(locator.slot.active_path):
        load_active_generation(
          locator.slot,
          expected_identity=locator.identity,
          expected_members=locator.members)
        raise RuntimeError(
          "a complete active dataset already owns this logical output. "
          "Fresh generation will not replace it; choose new output stems or "
          "request an authenticated continuation.")
      draft = begin_dataset_generation(locator.slot)
      expected_active_sha256 = None
    elif operation == "resume":
      continuation = begin_dataset_continuation(
        self.dataset_slot,
        expected_identity=self.dataset_identity,
        expected_members=members,
        expected_active_sha256=preflight_source.active_sha256)
      draft = continuation.draft
      expected_active_sha256 = continuation.source.active_sha256
      try:
        locator = install_dataset_locator(
          self.dataset_slot,
          identity=self.dataset_identity,
          members=members)
      except BaseException:
        discard_dataset_draft(draft)
        raise
    else:
      raise ValueError(
        "unknown normalized generator operation: " + repr(operation))

    self.dataset_locator = locator
    self.dataset_draft = draft
    self.dataset_expected_active_sha256 = expected_active_sha256
    self.paramsf = str(draft.files_path / Path(self.logical_paramsf).name)
    self.dvsf = str(draft.files_path / Path(self.logical_dvsf).name)
    self.failf = str(draft.files_path / Path(self.logical_failf).name)
    self.dataset_member_directory = draft.files_path

  def _preflight_active_checkpoint(self):
    """Validate one active checkpoint without creating or opening writable state.

    The immutable publication layer first authenticates every member. The
    ordinary checkpoint loader then checks the producer sidecars, row counts,
    axes, payload shapes, and successful payload rows while every large array
    is mapped read-only. All temporary bindings are removed before this method
    returns. A resume may create its private copy only after this succeeds;
    append uses the same proof and then refuses because exact continuation state
    is not yet persisted.
    """
    operation = self.run_control.operation
    if operation not in ("resume", "append"):
      raise ValueError(
        "active checkpoint preflight requires resume or append; got "
        + repr(operation))
    if self.dataset_draft is not None or self.dataset_locator is not None:
      raise RuntimeError(
        "active checkpoint preflight must run before a locator or draft is "
        "bound to this generator")

    source = load_active_generation(
      self.dataset_slot,
      expected_identity=self.dataset_identity,
      expected_members=dict(self.dataset_members))
    source_directory = source.member("parameters.chain").path.parent
    original = (self.paramsf, self.dvsf, self.failf,
                self.dataset_member_directory)
    self.paramsf = str(source_directory / Path(self.logical_paramsf).name)
    self.dvsf = str(source_directory / Path(self.logical_dvsf).name)
    self.failf = str(source_directory / Path(self.logical_failf).name)
    self.dataset_member_directory = source_directory
    self._checkpoint_read_only = True
    try:
      if self.__load_chk() is not True:
        raise RuntimeError(
          "an active resume or append checkpoint was not loaded during its "
          "read-only semantic preflight")
    finally:
      try:
        self._close_dataset_memmaps()
      finally:
        if hasattr(self, "datavectors"):
          del self.datavectors
        self.samples = None
        self.failed = None
        self.loadedsamples = False
        self.loadedfromchk = False
        self.dvs_is_memmap = False
        self._checkpoint_read_only = False
        (self.paramsf, self.dvsf, self.failf,
         self.dataset_member_directory) = original
    return source

  def _close_dataset_memmaps(self):
    """Flush and close every retained payload memmap before publication.

    Returns:
      None. In-memory NumPy arrays are unchanged. A single memmap or every
      memmap in a family mapping is flushed and its file mapping is closed.

    Raises:
      An operating-system or NumPy error when dirty bytes cannot be flushed.
      Publication stops before the active generation can change.
    """
    stores = getattr(self, "datavectors", None)
    if isinstance(stores, dict):
      values = list(stores.values())
    elif stores is None:
      values = []
    else:
      values = [stores]

    closed = set()
    for value in values:
      if not isinstance(value, np.memmap) or id(value) in closed:
        continue
      if not getattr(self, "_checkpoint_read_only", False):
        value.flush()
      memory_map = getattr(value, "_mmap", None)
      if memory_map is not None and not memory_map.closed:
        memory_map.close()
      closed.add(id(value))

  def _require_publishable_failure_mask(self):
    """Refuse a full draft whose persisted mask names a failed physics row.

    Returns:
      None. Chain-only datasets have no payload calculations and return
      immediately. A full dataset returns only when every row token is ``0``
      and the mask length equals the parameter-chain row count.

    Raises:
      RuntimeError when the member census and mode disagree, a token is not a
      literal ``0`` or ``1``, the row count differs, or any row is marked
      failed. The draft remains private and the previous active generation is
      unchanged.
    """
    role = "rows.failure-mask"
    members = self.dataset_locator.members
    if role not in members:
      if self.run_control.dataset_mode != "chain-only":
        raise RuntimeError(
          "a full dataset has no failure-mask member in its publication census")
      return
    if self.run_control.dataset_mode != "full":
      raise RuntimeError(
        "a chain-only dataset unexpectedly owns a failure-mask member")

    path = self.dataset_draft.files_path / members[role]
    row_count = 0
    invalid = []
    failed_count = 0
    failed_preview = []
    with open(path, "r", encoding="ascii") as handle:
      for line_number, line in enumerate(handle, start=1):
        row_count += 1
        token = line[:-1] if line.endswith("\n") else line
        if token not in ("0", "1"):
          invalid.append((line_number, token))
          continue
        if token == "1":
          failed_count += 1
          if len(failed_preview) < 20:
            failed_preview.append(line_number - 1)
    if invalid:
      raise RuntimeError(
        "the draft failure mask contains invalid rows: " + repr(invalid))
    expected_rows = len(self.samples)
    if row_count != expected_rows:
      raise RuntimeError(
        "the draft failure mask has " + str(row_count) + " rows, but the "
        "parameter chain has " + str(expected_rows))
    if failed_count:
      suffix = "" if failed_count <= len(failed_preview) else " ..."
      raise RuntimeError(
        "dataset publication refused because " + str(failed_count)
        + " data-vector rows failed. Failed zero-based row indices begin "
        + repr(failed_preview) + suffix
        + ". The draft remains private; repair the "
        "underlying calculation and run the request again. Any earlier active "
        "generation remains unchanged.")

  def _publish_dataset_generation(self):
    """Close draft writers and atomically select the complete generation.

    Returns:
      None. Successful publication installs one immutable generation and then
      replaces the small active record as the last visible state change.

    Raises:
      RuntimeError when rank zero has no prepared draft or a full draft has a
      failed row. DatasetPublicationError reports a member, digest, mode,
      ownership, race, or filesystem failure. A refusal before active-record
      replacement leaves the previous generation selected. A later durability
      error is reported as an uncertain publication and must be inspected.
    """
    if self.dataset_draft is None or self.dataset_locator is None:
      raise RuntimeError("dataset publication was not prepared on rank 0")
    self._close_dataset_memmaps()
    self._require_publishable_failure_mask()
    publish_dataset_generation(
      self.dataset_draft,
      identity=self.dataset_locator.identity,
      members=self.dataset_locator.members,
      expected_active_sha256=self.dataset_expected_active_sha256)

  def _compute_dvs_from_sample(self, sample):
    """
    Compute one data-vector payload from one parameter sample (a 1D
    array ordered like train_args.ord). Every driver must implement it.
    """
    raise NotImplementedError(
      "the dataset-generator driver must implement _compute_dvs_from_sample")

  #-----------------------------------------------------------------------------
  # reorder indexes to match YAML original ordering
  #-----------------------------------------------------------------------------
  def reorder_idx_from_yaml_to_ord(self):
    pidx = {p : i for i, p in enumerate(self.names)}
    return np.array([pidx[p] for p in self.sampled_params],
                    copy=True,
                    dtype=int)

  def reorder_idx_from_ord_to_yaml(self):
    pidx = {p : i for i, p in enumerate(self.sampled_params)}
    return np.array([pidx[p] for p in self.names],
                    copy=True,
                    dtype=int)

  #-----------------------------------------------------------------------------
  # successful payload contract
  #-----------------------------------------------------------------------------
  def _dv_payload_names(self):
    """Return the exact member names for one computed payload."""
    return ("vector",)

  def _dv_payload_mapping(self, payload):
    """Map one flat lensing payload to the shared member representation."""
    if not isinstance(payload, np.ndarray):
      raise ValueError(
        "a flat data-vector payload must be a NumPy array; got "
        f"{type(payload).__name__}. Return one nonempty 1D array")
    return {"vector": payload}

  def _dv_expected_payload_shape(self, name):
    """Return a configured row shape, or None for an inferred flat width."""
    if name != "vector":
      raise ValueError(
        f"unknown flat data-vector payload member {name!r}; expected "
        "'vector'")
    return None

  def _dv_payload_store(self, name):
    """Return the 2D store that owns one flat payload member."""
    if name != "vector":
      raise ValueError(
        f"unknown flat data-vector payload member {name!r}; expected "
        "'vector'")
    return self.datavectors

  def _prepare_payload_mapping(self, payload_mapping, use_storage):
    """Cast and validate every member of one data-vector payload.

    Arguments:
      payload_mapping = raw member names and array-shaped values.
      use_storage = when true, take each target dtype and row shape from the
        allocated store. When false, use self.dtype and configured shapes
        before first-row allocation.

    Returns:
      A new mapping whose arrays have the target dtype and exact row shapes.

    Raises:
      ValueError when names, shapes, dtypes, or finite values cannot satisfy
      the family contract. The method does not change a store or failure flag.
    """
    if type(payload_mapping) is not dict:
      raise ValueError(
        "the internal data-vector payload mapping must be a dict; got "
        f"{type(payload_mapping).__name__}")

    expected_names = self._dv_payload_names()
    observed_names = tuple(payload_mapping.keys())
    if set(observed_names) != set(expected_names):
      raise ValueError(
        f"data-vector payload members are {observed_names!r}, expected "
        f"exactly {expected_names!r}. Return every required member once and "
        "remove extra members")

    prepared = {}
    for name in expected_names:
      configured_shape = self._dv_expected_payload_shape(name)
      if use_storage:
        storage = self._dv_payload_store(name)
        if storage.ndim != 2:
          raise ValueError(
            f"data-vector store {name!r} must be 2D, got {storage.shape}. "
            "Use a valid checkpoint or allocate the store again")
        target_dtype = np.dtype(storage.dtype)
        configured_dtype = np.dtype(self.dtype)
        if target_dtype != configured_dtype:
          raise ValueError(
            f"data-vector store {name!r} has dtype {target_dtype}, expected "
            f"the configured dtype {configured_dtype}. Use a checkpoint "
            "written with the current generator configuration")
        expected_shape = tuple(storage.shape[1:])
        if configured_shape is not None:
          configured_shape = tuple(configured_shape)
          if expected_shape != configured_shape:
            raise ValueError(
              f"data-vector store {name!r} has row shape {expected_shape}, "
              f"expected {configured_shape} from train_args. Use a matching "
              "checkpoint")
      else:
        target_dtype = np.dtype(self.dtype)
        expected_shape = configured_shape

      if not np.issubdtype(target_dtype, np.number):
        raise ValueError(
          f"data-vector store {name!r} has nonnumeric dtype {target_dtype}; "
          "use a checkpoint with numeric payload arrays")
      try:
        with np.errstate(over="ignore", invalid="ignore"):
          cast_value = np.array(
            payload_mapping[name],
            copy=True,
            dtype=target_dtype)
      except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
          f"data-vector payload member {name!r} cannot be cast to storage "
          f"dtype {target_dtype}; return a numeric array") from error

      if expected_shape is None:
        if cast_value.ndim != 1 or cast_value.shape[0] < 1:
          raise ValueError(
            f"data-vector payload member {name!r} must be a nonempty 1D "
            f"array, got {cast_value.shape}")
        expected_shape = cast_value.shape
      expected_shape = tuple(expected_shape)
      if cast_value.shape != expected_shape:
        raise ValueError(
          f"data-vector payload member {name!r} has shape "
          f"{cast_value.shape}, expected {expected_shape}. Return the exact "
          "configured row shape")
      if not np.isfinite(cast_value).all():
        raise ValueError(
          f"data-vector payload member {name!r} contains a nonfinite value "
          f"after casting to storage dtype {target_dtype}; correct the "
          "scientific calculation before retrying the row")
      prepared[name] = cast_value
    return prepared

  def _stored_payload_mapping(self, index):
    """Copy one row from every family store into the shared representation."""
    stored = {}
    for name in self._dv_payload_names():
      storage = self._dv_payload_store(name)
      stored[name] = np.array(storage[index], copy=True)
    return stored

  def _accept_payload_row(self, index, payload, write_row,
                          allocation_rows=None):
    """Validate one payload row before recording it as successful.

    Arguments:
      index = zero-based row in the failure mask and data-vector stores.
      payload = one raw family payload for a new result. Use None only when
        checking an existing successful checkpoint row.
      write_row = true for a newly computed row; false for read-only
        checkpoint validation.
      allocation_rows = total rows to allocate before the first write, or
        None when the stores already exist.

    Returns:
      None. A new row has failed[index] cleared only after exact readback.

    Raises:
      ValueError when the interface, payload, store, or readback is invalid.
      Validation of an existing checkpoint row does not write any file.
    """
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
      raise ValueError(
        f"data-vector row index must be an integer, got {index!r}")
    row_index = int(index)
    if self.failed.ndim != 1:
      raise ValueError(
        f"data-vector failure mask must be 1D, got {self.failed.shape}")
    if row_index < 0 or row_index >= self.failed.shape[0]:
      raise ValueError(
        f"data-vector row index {row_index} is outside the failure mask "
        f"with {self.failed.shape[0]} rows")
    if type(write_row) is not bool:
      raise ValueError(
        f"write_row must be true or false, got {write_row!r}")

    if write_row:
      if payload is None:
        raise ValueError("a new successful row requires a computed payload")
      payload_mapping = self._dv_payload_mapping(payload)
    else:
      if payload is not None or allocation_rows is not None:
        raise ValueError(
          "read-only checkpoint validation requires payload=None and "
          "allocation_rows=None")
      payload_mapping = self._stored_payload_mapping(row_index)

    if allocation_rows is not None:
      if isinstance(allocation_rows, bool) or not isinstance(
          allocation_rows, (int, np.integer)):
        raise ValueError(
          f"allocation_rows must be an integer, got {allocation_rows!r}")
      allocation_rows = int(allocation_rows)
      if allocation_rows != self.failed.shape[0]:
        raise ValueError(
          f"allocation_rows is {allocation_rows}, but the failure mask has "
          f"{self.failed.shape[0]} rows")
      self._prepare_payload_mapping(
        payload_mapping=payload_mapping,
        use_storage=False)
      self._dv_alloc(nrows=allocation_rows, first_dvs=payload)

    prepared = self._prepare_payload_mapping(
      payload_mapping=payload_mapping,
      use_storage=True)
    if write_row:
      self._dv_write(i=row_index, dvs=prepared)

    stored_mapping = self._stored_payload_mapping(row_index)
    stored = self._prepare_payload_mapping(
      payload_mapping=stored_mapping,
      use_storage=True)
    for name in self._dv_payload_names():
      if stored[name].dtype != prepared[name].dtype:
        raise ValueError(
          f"stored data-vector member {name!r} has dtype "
          f"{stored[name].dtype}, expected {prepared[name].dtype} after "
          "storing the row")
      stored_bytes = np.ascontiguousarray(stored[name]).tobytes(order="C")
      prepared_bytes = np.ascontiguousarray(
        prepared[name]).tobytes(order="C")
      if stored_bytes != prepared_bytes:
        raise ValueError(
          f"stored data-vector member {name!r} differs from the exact "
          "cast payload bytes after storing the row; keep the row marked "
          "failed and repair the store before retrying")

    if write_row:
      self.failed[row_index] = False

  def _validate_loaded_success_rows(self):
    """Validate every checkpoint row whose saved failure flag is false."""
    for index in range(self.failed.shape[0]):
      if self.failed[index]:
        continue
      self._accept_payload_row(
        index=index,
        payload=None,
        write_row=False)

  def _consume_worker_result_message(self, message, source, active):
    """Validate and apply one MPI worker result to its assigned row.

    Protocol validation happens before this method changes the assignment
    table, failure mask, or payload store. Both rank-zero result loops use this
    one method, so a normal result and a result received while finishing the
    last workers obey the same row-binding rule.

    Arguments:
      message = the exact worker-result tuple returned by MPI.
      source = the positive integer worker rank reported by MPI.
      active = rank zero's mutable worker-to-row assignment mapping.

    Returns:
      ``(row, outcome, detail)``. ``row`` is the row rank zero assigned.
      ``outcome`` is ``"accepted"``, ``"worker-error"``, or
      ``"invalid-payload"``. ``detail`` is ``None``, worker traceback text,
      or the payload-validation exception, respectively.

    Raises:
      RuntimeError when the worker message does not match the live assignment.
      A storage exception also propagates when a failed row cannot be cleared
      safely.
    """
    kind, index, payload = validate_worker_result_message(
      message=message,
      source=source,
      active=active)

    if kind == "err":
      self._dv_zero(index)
      self.failed[index] = True
      outcome = "worker-error"
      detail = payload
    else:
      try:
        self._accept_payload_row(
          index=index,
          payload=payload,
          write_row=True)
        outcome = "accepted"
        detail = None
      except Exception as error:
        self._dv_zero(index)
        self.failed[index] = True
        outcome = "invalid-payload"
        detail = error

    del active[source]
    return index, outcome, detail

  #-----------------------------------------------------------------------------
  # data-vector store (default: the single 2D array at {dvsf}.npy)
  #-----------------------------------------------------------------------------
  def _dv_chk_files(self):
    """Files the checkpoint loader must find before trusting a chk."""
    return [f"{self.dvsf}.npy"]

  def _load_axis_checkpoint(self, path, expected, label):
    """Load one read-only axis sidecar and match the configured coordinates."""
    observed = np.load(path, allow_pickle=False)
    expected = np.asarray(expected)
    if observed.ndim != 1:
      raise ValueError(
        f"checkpoint axis {label} must be 1D, got {observed.shape}")
    if observed.shape != expected.shape:
      raise ValueError(
        f"checkpoint axis {label} has shape {observed.shape}, expected "
        f"{expected.shape} from train_args")
    if not np.array_equal(observed, expected):
      raise ValueError(
        f"checkpoint axis {label} disagrees with the configured train_args "
        "coordinates")
    return observed

  def _dv_load_chk(self):
    """Load the data-vector store from its checkpoint files (RAM-aware)."""
    arr = np.load(f"{self.dvsf}.npy",
                  mmap_mode = "r",
                  allow_pickle = False)
    if getattr(self, "_checkpoint_read_only", False):
      self.datavectors = arr
      self.dvs_is_memmap = True
    else:
      RAMneed = (arr.nbytes +
                 self.samples.nbytes +
                 self.failed.nbytes)
      RAMavail = psutil.virtual_memory().available
      if RAMneed < 0.75 * RAMavail:
        self.datavectors = np.load(f"{self.dvsf}.npy", allow_pickle = False)
        self.dvs_is_memmap = False
      else:
        print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
              f"There is {RAMavail/1e9:.2f} GB of RAM available. "
              f"We will read dvs from HD (slow)")
        self.datavectors = np.load(f"{self.dvsf}.npy",
                                   mmap_mode = "r+",
                                   allow_pickle = False)
        self.dvs_is_memmap = True
      del arr

    if self.datavectors.ndim != 2:
      raise ValueError(f"datavectors must be 2D, got {self.datavectors.shape}")
    if self.datavectors.shape[0] != self.samples.shape[0]:
      raise ValueError(f"Incompatible samples/datavector chk files")

  def _dv_save(self):
    """Flush the data-vector store to disk (tmp file + atomic replace)."""
    if self.dvs_is_memmap == True:
      self.datavectors.flush()  # checkpoint dv in-place
    else:
      # save (flush) dvs to tmp file (safer) -----------------------------------
      np.save(f"{self.dvsf}.tmp.npy", self.datavectors)
      # save data vector file (from tmp) ---------------------------------------
      os.replace(f"{self.dvsf}.tmp.npy", f"{self.dvsf}.npy")

  def _dv_append(self, nparams):
    """Grow the store by nparams zero rows (append mode; RAM-aware)."""
    nrows = self.datavectors.shape[0]
    ncols = self.datavectors.shape[1]

    RAMneed = ( self.samples.nbytes +
                self.failed.nbytes +
                self.datavectors.nbytes +
                (nrows + nparams)*ncols*self.datavectors.dtype.itemsize)
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      # setup new datavector numpy array -----------------------------------
      self.datavectors = np.vstack((self.datavectors,
                                    np.zeros((nparams, ncols), dtype=self.dtype)))
      # save (flush) dvs to tmp file (safer) -------------------------------
      np.save(f"{self.dvsf}.tmp.npy", self.datavectors)
      # save data vector file (from tmp) -----------------------------------
      os.replace(f"{self.dvsf}.tmp.npy", f"{self.dvsf}.npy")
      self.dvs_is_memmap = False
    else:
      print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
            f"There is {RAMavail/1e9:.2f} GB of RAM available. "
            f"We will read dvs from HD (slow)")
      # setup new datavector numpy array -----------------------------------
      datavectors = open_memmap(f"{self.dvsf}.tmp.npy",
                                mode = "w+",
                                shape = (nrows + nparams, ncols),
                                dtype = self.datavectors.dtype)
      for s in range(0, nrows, 2500): # read dvs in chunks: avoid RAM spikes
        e = min(nrows, s + 2500)
        datavectors[s:e] = self.datavectors[s:e]
      for s in range(nrows, nrows + nparams, 2500):
        e = min(nrows + nparams, s + 2500)
        datavectors[s:e] = 0
      # save (flush) data vector (in-place) --------------------------------
      datavectors.flush()
      del datavectors
      # save data vector file (from tmp) -----------------------------------
      os.replace(f"{self.dvsf}.tmp.npy", f"{self.dvsf}.npy")
      # finally, load the new dv numpy array from file ---------------------
      self.datavectors = np.load(f"{self.dvsf}.npy",
                                 mmap_mode = "r+",
                                 allow_pickle = False)
      self.dvs_is_memmap = True

    # check final dimensions -----------------------------------------------
    if self.datavectors.shape[0] != self.samples.shape[0]:
      raise ValueError(f"Incompatible samples/datavector chk files")

  def _dv_alloc(self, nrows, first_dvs):
    """
    Allocate the store for nrows samples, sized from the first computed
    payload (RAM-aware: in-RAM zeros or an on-disk memmap).
    """
    ncols = len(first_dvs)
    RAMneed = ( self.samples.nbytes +
                self.failed.nbytes +
                nrows*ncols*np.dtype(self.dtype).itemsize)
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      self.datavectors = np.zeros((nrows, ncols), dtype=self.dtype)
      self.dvs_is_memmap = False
    else:
      print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
            f"There is {RAMavail/1e9:.2f} GB of RAM available. "
            f"We will read dvs from HD (slow)")
      self.datavectors = open_memmap(f"{self.dvsf}.npy",
                                     mode = "w+",
                                     shape = (nrows, ncols),
                                     dtype = self.dtype)
      self.datavectors[:] = 0.0
      self.datavectors.flush()
      self.dvs_is_memmap = True

  def _dv_write(self, i, dvs):
    """Write one computed payload at row i."""
    self.datavectors[i] = dvs["vector"]

  def _dv_zero(self, i):
    """Zero row i (a failed sample)."""
    self.datavectors[i, :] = 0.0

  #-----------------------------------------------------------------------------
  # save/load checkpoint
  #-----------------------------------------------------------------------------
  def _checkpoint_member_paths(self):
    """Return checkpoint paths from the census bound during setup."""
    checkpoint_paths = []
    for relative_name in self.dataset_members.values():
      checkpoint_paths.append(self.dataset_member_directory / relative_name)
    return checkpoint_paths

  def __load_chk(self):
    if self.run_control.operation == "fresh":
      return False

    require_checkpoint_members(
      operation=self.run_control.operation,
      members=self._checkpoint_member_paths(),
      is_file=os.path.isfile)

    # load sample file begins ------------------------------------------------
    # The producer-authored .paramnames sidecar owns the numeric layout. It
    # requires the complete non-derived sequence to match train_args.ord and
    # validates the two bookkeeping columns plus every declared column before
    # returning the sampled inputs.
    checkpoint_input_names = tuple(self.sampled_params)
    parameter_table = resolve_parameter_table(
      params_path=f"{self.paramsf}.1.txt",
      input_names=checkpoint_input_names)
    expected_sidecar = os.path.normcase(os.path.abspath(
      f"{self.paramsf}.paramnames"))
    observed_sidecar = os.path.normcase(os.path.abspath(
      parameter_table.sidecar_path))
    if observed_sidecar != expected_sidecar:
      raise ValueError(
        "generator checkpoint readback must use its producer-owned "
        f".paramnames sidecar {expected_sidecar!r}; the resolver selected "
        f"unexpected shadow path {observed_sidecar!r}")
    expected_declarations = tuple(
      (name, False, 2 + index)
      for index, name in enumerate(checkpoint_input_names)
    ) + (("chi2", True, 2 + len(checkpoint_input_names)),)
    if parameter_table.declarations != expected_declarations:
      raise ValueError(
        "checkpoint .paramnames must declare exactly the sampled parameters "
        "in train_args order followed by chi2*; got "
        f"{parameter_table.declarations!r}, expected "
        f"{expected_declarations!r}")
    self.samples = parameter_table.inputs
    # load sample file ends --------------------------------------------------

    if self.run_control.dataset_mode == "chain-only":
      print("Loaded chain-only models from chk")
      if self.run_control.operation == "resume":
        self.loadedsamples = True
        self.loadedfromchk = True
      return True

    # load fail file begins --------------------------------------------------
    with open(f"{self.failf}.txt", "r", encoding="ascii") as fail_handle:
      # The producer writes one ASCII token per physical line.  Text-file
      # iteration accepts the platform newline forms, but (unlike
      # ``str.splitlines``) does not reinterpret vertical tab, form feed, or
      # other control characters as extra producer records.
      failure_tokens = [
        line[:-1] if line.endswith("\n") else line
        for line in fail_handle
      ]
    invalid_failure_tokens = [
      (line_number, token)
      for line_number, token in enumerate(failure_tokens, start=1)
      if token not in ("0", "1")]
    if invalid_failure_tokens:
      raise ValueError(
        "failed checkpoint lines must be the literal producer tokens '0' or "
        f"'1'; invalid lines are {invalid_failure_tokens!r}")
    self.failed = np.asarray(
      [token == "1" for token in failure_tokens], dtype=bool)

    if self.samples.shape[0] != self.failed.shape[0]:
      raise ValueError(f"Incompatible samples/failed chk files")
    # load fail file ends ----------------------------------------------------

    # load datavectors (store-specific; row-count checks live inside) --------
    self._dv_load_chk()
    self._validate_loaded_success_rows()

    print("Loaded models from chk")
    if self.run_control.operation == "resume":
      self.loadedsamples = True
      self.loadedfromchk = True
    return True

  def __save_chk(self):
    # save data vector file (store-specific) ------------------------------------
    self._dv_save()
    # save fail file -----------------------------------------------------------
    # save (flush) dvs to tmp file (safer) -------------------------------------
    np.savetxt(f"{self.failf}.tmp.txt", self.failed.astype(np.uint8), fmt="%d")
    # save data vector file (from tmp) -----------------------------------------
    os.replace(f"{self.failf}.tmp.txt", f"{self.failf}.txt")

  def _append_full_checkpoint_rows(self, nparams):
    """Extend only a full dataset's failure mask and payload stores.

    Chain-only append owns parameter-side members only. Its output stems are
    already isolated, and this second mode check prevents a future refactor
    from creating or borrowing failure/data-vector members in that namespace.
    """
    if self.run_control.dataset_mode == "chain-only":
      return
    if self.run_control.dataset_mode != "full":
      raise ValueError(
        "Unknown normalized generator dataset mode: "
        + repr(self.run_control.dataset_mode))

    fname = f"{self.failf}.txt"
    failed = np.ones((nparams, 1), dtype=np.uint8)
    with open(fname, "a") as f:
      np.savetxt(f, failed.astype(np.uint8), fmt="%d")

    self.failed = np.atleast_1d(np.loadtxt(fname, dtype=np.uint8))
    self.failed = np.asarray(self.failed).astype(bool)
    if self.failed.ndim != 1:
      raise ValueError(f"failed must be 1D, got {self.failed.shape}")
    if self.samples.shape[0] != self.failed.shape[0]:
      raise ValueError(f"Incompatible samples/failed chk files")

    self._dv_append(nparams)

  #-----------------------------------------------------------------------------
  # likelihood
  #-----------------------------------------------------------------------------
  def __param_logprior(self, x):
    # because we shrink the prior, we need to do this check
    if np.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1])):
      return 0.0
    else:
      return -np.inf

  def __param_logpost(self,x):
    y = x - self.fiducial
    logprior = self.model.prior.logp(x[self.reorder_idx_from_ord_to_yaml()])
    if not math.isfinite(logprior):
      return -np.inf
    elif not math.isfinite(self.__param_logprior(x)):
      # this is important when --boundary command line option is < 1
      return -np.inf
    else:
      logp = (-0.5*(y @ self.inv_covmat @ y) + logprior)/self.temp
      return logp if math.isfinite(logp) else -np.inf

  #-----------------------------------------------------------------------------
  # the dataset's scientific record (schema: emulator/fixed_facts.py)
  #-----------------------------------------------------------------------------
  def _resolved_constants(self):
    """Read named constants and theory settings for the dataset record.

    The shared reader preserves the names exposed by Cobaya. Family-specific
    code interprets physical aliases when that interpretation is required for
    generation.

    Arguments:
      none.

    Returns:
      A mapping from each readable name to its plain value.
    """
    return fixed_facts.resolved_constants(model=self.model)

  def _syren_base_identity(self):
    """
    Name the frozen analytic base the matter-power dumps are built on top of.

    The matter-power emulators do not learn P(k, z) from nothing: they correct
    the syren analytic formulas, vendored in-repo under syren/ and owned by
    emulator/syren_base.py. Those formulas evaluate their growth factors at a
    neutrino mass pinned in the base's own signature, which does not follow the
    model's mnu, so an emulator trained against this base carries that pin
    whether or not the run sampled mnu. The record names the base and the pinned
    mass together, because the pair is what a consumer must match.

    The pinned mass is read from the formula's own default rather than written
    here as a number, so the record cannot drift from the base it names: the
    generator calls base_pklin without an mnu argument, so the default is the
    value the dumps were built with.

    Arguments:
      none.

    Returns:
      the base's identity, as one line of text.

    Raises:
      ValueError when the base's pinned neutrino mass cannot be read from the
      formula. A dataset whose base cannot be named must not be published as
      though it had no base.
    """
    from emulator.syren_base import base_pklin

    signature = inspect.signature(base_pklin)
    if "mnu" not in signature.parameters:
      raise ValueError(
        "the syren base formula base_pklin no longer takes an mnu argument, so "
        "the neutrino mass the matter-power base is pinned at cannot be read "
        "from it. The dataset's record must name the base it corrects together "
        "with the mass that base holds fixed; restore the argument in "
        "emulator/syren_base.py, or teach this record where the pin now lives.")
    pinned = signature.parameters["mnu"].default
    if pinned is inspect.Parameter.empty:
      raise ValueError(
        "the syren base formula base_pklin takes mnu without a default, so the "
        "neutrino mass the matter-power base is pinned at is now the caller's "
        "choice and this record no longer knows it. The generator calls the "
        "base without an mnu argument; give the argument its pinned default "
        "back in emulator/syren_base.py.")
    return ("syren analytic base, owned by emulator/syren_base.py (the "
            "symbolic_pofk formulas vendored in syren/), with mnu pinned at "
            + fixed_facts.format_value(value=pinned))

  def _resolve_fixed_facts(self):
    """
    Read this run's scientific facts off the resolved Cobaya model.

    The YAML is the request and the model is the fact, so every value below is
    read from self.model, never from the parsed YAML dictionary. The two differ
    exactly where it matters: a coordinate the YAML never mentioned has been
    given its value by the time the model exists, and it is that value the data
    vectors were computed at.

    A cosmology coordinate this run sampled is dropped from the roster rather
    than pinned: it is validated through the input domain instead, and a
    coordinate that was both sampled and pinned would let the two halves of the
    record answer one question two different ways (fixed_facts.build_sidecar
    refuses such a record, and this is the code that must not hand it one). A
    coordinate the model cannot resolve is reported "n/a", never omitted.

    Arguments:
      none.

    Returns:
      a mapping holding the nine facts fixed_facts.build_sidecar takes beside
      the two supports: generator, family, cosmology_fixed, neutrino_convention,
      flat_only, dark_energy_law, dark_energy_inputs, cl_units, base_identity.

    Raises:
      ValueError when the matter-power base cannot be named.
    """
    family = self.dataset_route["family"]

    sampled = set(self.sampled_params)
    pinned  = self._resolved_constants()
    (dark_energy_fixed,
     dark_energy_varying,
     dark_energy_law,
     dark_energy_inputs) = _dark_energy_publication_facts(
       self.model.parameterization, pinned)

    cosmology_fixed = {}
    for key in fixed_facts.COSMOLOGY_FIXED_KEYS:
      if key in ("w", "wa"):
        if key in dark_energy_varying:
          continue
        cosmology_fixed[key] = dark_energy_fixed.get(
          key, fixed_facts.NOT_APPLICABLE)
        continue
      if key in sampled:
        continue                                 # validated as a sampled input
      if key in pinned:
        cosmology_fixed[key] = pinned[key]
      else:
        cosmology_fixed[key] = fixed_facts.NOT_APPLICABLE

    # curvature. The run admits no spatial curvature unless omk is sampled, or
    # pinned away from zero. A model that carries no omk at all is flat: an
    # absent coordinate is the standard model's value for it, not a missing
    # fact, the same reading emulator/syren_base.py gives an absent w or wa.
    flat_only = True
    if "omk" in sampled:
      flat_only = False
    elif "omk" in pinned:
      flat_only = (pinned["omk"] == 0.0)
    if family == "grid" and not flat_only:
      raise ValueError(
        "background generation is flat-only; omk must be fixed to zero")

    # how the neutrino masses are split, in the model's own word for it.
    neutrino_convention = fixed_facts.NOT_APPLICABLE
    if NEUTRINO_CONVENTION_KEY in pinned:
      neutrino_convention = str(pinned[NEUTRINO_CONVENTION_KEY])

    cl_units = fixed_facts.NOT_APPLICABLE
    if family == "cmb":
      cl_units = CMB_CL_UNITS

    # the frozen base the dataset sits on top of. Only the matter-power dumps
    # have one, and only when the run asked for it: a run with write_syren_base
    # off writes no base file and its emulator corrects nothing, so it has no
    # base to name. Setup binds that choice as the validated family variant, so
    # later mutable driver fields cannot change the scientific record.
    base_identity = fixed_facts.NOT_APPLICABLE
    if self.dataset_route["family_variant"] == "syren-base":
      base_identity = self._syren_base_identity()

    # Setup has already proved the concrete driver file against the registry.
    # Reusing that immutable route prevents later mutable class or probe state
    # from changing the producer name written into the scientific sidecar.
    generator = self.dataset_route["generator"]

    return {"generator":           generator,
            "family":              family,
            "cosmology_fixed":     cosmology_fixed,
            "neutrino_convention": neutrino_convention,
            "flat_only":           flat_only,
            "dark_energy_law":     dark_energy_law,
            "dark_energy_inputs":  dark_energy_inputs,
            "cl_units":            cl_units,
            "base_identity":       base_identity}

  def _write_facts_sidecar(self, names):
    """
    Write the dataset's scientific record beside the chain it describes.

    The record is the pair of blocks emulator/fixed_facts.py defines: the
    cosmology this run held fixed, and the region of parameter space it sampled.
    It is written from the resolved Cobaya model and from the two supports, and
    it is keyed by the digest of the chain file as that file stands on disk, so
    the record names the exact draw it describes and no other. The chain must
    therefore already be written when this is called.

    Arguments:
      names = the sampled parameters in the canonical train_args.ord order, the
              same list the chain's columns and the .ranges rows are written in.
              Row i of both bound arrays is names[i], because both were reordered
              into that order during setup.

    Returns:
      None. The record is written to <paramsf>.facts.yaml.

    Raises:
      ValueError when the facts break one of the record's own laws (a coordinate
      both sampled and held fixed, a sampled parameter with no support).
      build_sidecar validates before it returns any text, so a record that would
      be refused on the way back in is never written out.
    """
    text = self._facts_sidecar_text(
      names=names,
      dataset_id=fixed_facts.chain_digest(
        chain_path=f"{self.paramsf}.1.txt"))
    with open(f"{self.paramsf}{fixed_facts.SIDECAR_SUFFIX}", "w") as f:
      f.write(text)

  #-----------------------------------------------------------------------------
  # run mcmc
  #-----------------------------------------------------------------------------
  def __run_mcmc(self):
    load_checkpoint_or_refuse(
      operation=self.run_control.operation,
      loader=self.__load_chk)

    if self.run_control.operation in ("fresh", "append"):
      ndim     = len(self.sampled_params)
      names    = list(self.sampled_params)

      if not self.unif == 1:
        # high number of samples: make sure we get unique samples
        nparam  = 40*self.nparams if self.bounds_adj < 1 else 20*self.nparams
        nparams  = max(nparam, 5000000)
        nwalkers = int(10*ndim)
        nsteps   = int(max(7500, nparams/nwalkers)) # (for safety we assume tau>100)
        burnin   = int(0.1*nsteps)                  # 10% burn-in

        sampler = emcee.EnsembleSampler(nwalkers = nwalkers,
                                        ndim = ndim,
                                        moves=[(emcee.moves.DEMove(), 0.9),
                                               (emcee.moves.DESnookerMove(), 0.1)],
                                        log_prob_fn = self.__param_logpost)
        # give emcee's own moves a seeded random state derived from the owned
        # Generator, so the walk (not just the starting point) is replayable.
        sampler._random = np.random.RandomState(
            int(self.rng.integers(0, 2**31 - 1)))
        sampler.run_mcmc(initial_state = self.fiducial[np.newaxis] +
                                         0.5*np.sqrt(np.diag(self.covmat))*
                                         self.rng.standard_normal(size=(nwalkers,ndim)),
                         nsteps=nsteps,
                         progress=False)
        raw_samples = sampler.get_chain(flat=True, discard=burnin, thin=1)
        raw_log_prob = sampler.get_log_prob(
          flat=True, discard=burnin, thin=1)
        xf, selected_log_prob = select_unique_rows(
          raw_samples,
          raw_log_prob,
          requested=self.nparams,
          ndim=ndim,
          rng=self.rng)
        lnp = selected_log_prob[:, None]
        nparams = self.nparams
        # Double check that prior is not -infty --------------------------------
        idx = self.reorder_idx_from_ord_to_yaml()
        for i, x in enumerate(xf):
          logprior = self.model.prior.logp(x[idx])
          if not math.isfinite(logprior):
              raise ValueError(
                f"Sample {i} has a nonfinite prior. Values: "
                f"{dict(zip(self.sampled_params, x))}")
      else:
        nparams  = self.nparams
        self.uniform_sampling_support = resolve_uniform_sampling_support(
            names=names,
            bounds=self.bounds)
        self.bounds = self.uniform_sampling_support["bounds"]
        xf  = self.rng.uniform(low  = self.bounds[:,0],
                               high = self.bounds[:,1],
                               size = (nparams,ndim))
        lnp = np.ones((nparams,1), dtype=self.dtype)
        # Double check that prior is not -infty --------------------------------
        idx = self.reorder_idx_from_ord_to_yaml()
        for i, x in enumerate(xf):
          logprior = self.model.prior.logp(x[idx])
          if not math.isfinite(logprior):
              raise ValueError(
                f"Sample {i} has a nonfinite prior. Values: "
                f"{dict(zip(self.sampled_params, x))}")
      if xf.shape != (self.nparams, ndim):
        raise ValueError(
          "the prepared parameter table must contain exactly "
          + str(self.nparams) + " rows and " + str(ndim)
          + " columns; got " + repr(xf.shape))
      if lnp.shape != (self.nparams, 1):
        raise ValueError(
          "the prepared log-probability column must have shape "
          + repr((self.nparams, 1)) + "; got " + repr(lnp.shape))
      if not np.isfinite(xf).all() or not np.isfinite(lnp).all():
        raise ValueError(
          "the prepared parameter or log-probability table contains a "
          "nonfinite value")
      w = np.ones((nparams,1), dtype=self.dtype)
      chi2 = -2*lnp
      modeled_columns = np.concatenate([w, lnp, xf, chi2], axis=1)
      if not np.isfinite(modeled_columns).all():
        raise ValueError(
          "the prepared chain columns contain a nonfinite value")
      if self.run_control.operation == "fresh":
        self._prepare_dataset_publication()
        # Output some debug messaging ------------------------------------------
        if not self.unif == 1:
          try:
            tau = np.array(sampler.get_autocorr_time(quiet=True, has_walkers=True),
                           copy=True,
                           dtype=self.dtype).max()
            print(f"Partial Result: tau = {tau}\n"
                  f"nwalkers={nwalkers}\n"
                  f"nsteps (per walker) = {nsteps}\n"
                  f"nsteps/tau = {nsteps/tau} (min should be ~50)\n"
                  f"nparams (after thin)={nparams}\n")
          except Exception as e:
            print(f"Partial Result: tau = N/A (emcee threw an exception)\n"
                  f"nwalkers={nwalkers}\n"
                  f"nsteps (per walker) = {nsteps}\n"
                  f"nparams (after thin)={nparams}\n")
            tau = 1 # make sure main MPI worker does not crash over trivial check

        # save a range files ---------------------------------------------------
        # GetDist's view of the support the sampler actually drew from. The
        # numbers are the resolved bounds the scientific record publishes, and
        # they go through the record's own formatter, so the file a cosmologist
        # opens and the file a consumer compares against say the same thing.
        # This used to round each bound to %.5e, which is coarser than the
        # float32 the generator owns: 70.00001 and 70.00002 both came out
        # 7.00000e+01, so the sidecar declared a zero-width interval over a
        # range the chain beside it kept apart. A view that rounds is a second
        # answer to a question the record has already answered.
        #
        # rows stays ONE statement on purpose: ai/gates/checks/generator_ranges.py
        # executes this writer by lifting these very statements out of the
        # syntax tree, and it lifts the single assignment that binds rows. A
        # loop here would be lifted as nothing and the check would write an
        # empty file and still pass.
        bds = self.bounds.copy()
        rows = [(str(n), fixed_facts.format_value(float(l)),
                         fixed_facts.format_value(float(h)))
                for n, l, h in zip(names, bds[:, 0], bds[:, 1])]
        with open(f"{self.paramsf}.ranges", "w") as f:
          for name, low, high in rows:
            f.write(f"{name} {low} {high}\n")

        # save chain begins ----------------------------------------------------
        fname = f"{self.paramsf}.1.txt";
        # record the sampling seed and RNG in the chain header so the parameter
        # table can be replayed from the recorded inputs alone.
        rng_tag = f"seed={self.seed} rng=numpy.default_rng"
        if not self.unif == 1:
          hd=f"nwalkers={nwalkers} {rng_tag}\n"
        else:
          hd=f"Uniform Sampling {rng_tag}\n"
        np.savetxt(fname,
                   modeled_columns,
                   fmt="%.9e",
                   header=hd + ' '.join(["weights", "lnp"] + names + ["chi2*"]),
                   comments="# ")

        # copy samples to self.samples  ----------------------------------------
        self.samples = np.array(xf, copy=True, dtype=self.dtype)

        # save paramname files -------------------------------------------------
        latex = list(self.parameter_labels)
        paramnames = copy.deepcopy(names)
        paramnames.append("chi2*")
        latex.append("\\chi^2")
        np.savetxt(f"{self.paramsf}.paramnames",
                   np.column_stack((paramnames,latex)),
                   fmt="%s")

        # save a cov matrix ------------------------------------------------------
        with contextlib.redirect_stdout(io.StringIO()): # so getdist dont write in terminal
          saved_covariance = np.array(
            loadMCSamples(
              f"{self.paramsf}",
              settings={'ignore_rows': u'0.'}).cov(pars=names),
            copy=True,
            dtype=self.dtype)
        expected_covariance_shape = (ndim, ndim)
        if saved_covariance.shape != expected_covariance_shape \
            or not np.isfinite(saved_covariance).all() \
            or not np.allclose(
              saved_covariance, saved_covariance.T, rtol=2e-6, atol=2e-6):
          raise ValueError(
            "GetDist returned an invalid saved parameter covariance: shape "
            + repr(saved_covariance.shape) + ", expected "
            + repr(expected_covariance_shape))
        np.savetxt(f"{self.paramsf}.covmat",
                   saved_covariance,
                   fmt="%.9e",
                   header=' '.join(names),
                   comments="# ")

        # save the scientific record -------------------------------------------
        # the cosmology this run held fixed and the region it sampled, written
        # once, beside the chain whose digest names it. It is written last: the
        # digest must be taken from the published chain file, and getdist then
        # reads this directory exactly as it did before the record existed.
        self._write_facts_sidecar(names=names)

        # delete arrays (save RAM) ---------------------------------------------
        del w         # save RAM memory
        del xf        # save RAM memory
        del lnp       # save RAM memory
        del chi2      # save RAM memory
        del modeled_columns
        gc.collect()  # save RAM memory
      else:
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # This branch is reachable only for a successfully loaded append.
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # append chain file begins ---------------------------------------------
        fname = f"{self.paramsf}.1.txt";
        with open(fname, "a") as f: # append mode
          hd = ' '.join(["weights","lnp"] + names + ["chi2*"])
          np.savetxt(f,
                     np.concatenate([w, lnp, xf, chi2], axis=1),
                     header = hd if (os.path.getsize(fname) == 0) else "",
                     fmt = "%.9e")
        del w         # save RAM memory
        del xf        # save RAM memory
        del lnp       # save RAM memory
        del chi2      # save RAM memory
        del modeled_columns
        gc.collect()  # save RAM memory

        self.samples = np.atleast_2d(np.loadtxt(fname, dtype=self.dtype))[:,2:-1]
        if self.samples.ndim != 2:
          raise ValueError(f"samples must be 2D, got {self.samples.shape}")
        # append chain file ends -----------------------------------------------

        # Full datasets extend their failure mask and payload stores. A
        # chain-only append owns only the parameter-side files and returns from
        # this helper before either full-dataset boundary can be touched.
        self._append_full_checkpoint_rows(nparams)

        # update a parameter cov matrix ----------------------------------------
        with contextlib.redirect_stdout(io.StringIO()): # so getdist dont write in terminal
          np.savetxt(f"{self.paramsf}.covmat",
                     np.array(loadMCSamples(f"{self.paramsf}",
                                            settings={'ignore_rows': u'0.'}).cov(pars=names),
                              copy=True,
                              dtype=self.dtype),
                     fmt="%.9e",
                     header=' '.join(names),
                     comments="# ")

        # update the scientific record -----------------------------------------
        # this branch appended rows to the chain, so the chain's bytes changed
        # and its digest with them. The record written when the dataset was first
        # published names bytes that no longer exist anywhere, and a record that
        # names a chain nobody has is worse than no record at all: it would let
        # two emulators trained on different draws claim to share one dump.
        # Rewrite it, so the record's dataset id is the digest of the file it
        # sits beside. The facts and the supports are unchanged (the same YAML
        # and the same priors produced the appended rows), but they are re-read
        # from the model rather than assumed, because the record has one author.
        self._write_facts_sidecar(names=names)

        # set self.loadedfromchk -----------------------------------------------
        self.loadedfromchk = True
    # set self.loadedsamples ---------------------------------------------------
    self.loadedsamples = True

  #-----------------------------------------------------------------------------
  # datavectors
  #-----------------------------------------------------------------------------
  def __generate_datavectors(self):
    if not self.setup:
      raise RuntimeError(f"Initial Setup not successful")
    TTAG = 1      # Task tag
    STAG = 0      # Stop tag
    RTAG = 2      # Result tag
    DTAG = 3      # Done (not crashed) tag
    TASK_TIMEOUT = 1800
    STOP_TIMEOUT = 300.0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nworkers = size - 1

    if size == 1:
      if not self.loadedsamples:
        raise RuntimeError(f"Model Samples not loaded/computed")

      nparams = len(self.samples)

      if not self.loadedfromchk:
        # Allocate failed array begins -----------------------------------------
        self.failed = np.ones(nparams, dtype = np.uint8) # start w/ all failed
        self.failed = np.asarray(self.failed).astype(bool)

        # Allocate data vectors begins -----------------------------------------
        try: # First run: get data vector size
          dvs = self._compute_dvs_from_sample(self.samples[0])
        except Exception:
          raise RuntimeError(f"Failed in _compute_dvs_from_sample\n"
                             f"Cannot determine datavector length.")
        self._accept_payload_row(
          index=0,
          payload=dvs,
          write_row=True,
          allocation_rows=nparams)
        # Allocate data vectors end --------------------------------------------

        idx = np.arange(1, nparams) # indexes to compute data vectors
      else:
        idx = np.where(self.failed == True)[0] # indexes to compute data vectors

      for i in idx:
        try:
          dvs = self._compute_dvs_from_sample(self.samples[i])
          self._accept_payload_row(
            index=i,
            payload=dvs,
            write_row=True)
        except Exception as e: # set datavector to zero and continue
          self.failed[i] = True
          self._dv_zero(i)
          sys.stderr.write(f"[Rank 0] Worker failed at idx={i}\n")
          sys.stderr.write(f"[Rank 0] Exception type: {type(e).__name__}\n")
          sys.stderr.write(f"[Rank 0] Exception message: {e}\n")
          sys.stderr.write(f"[Rank 0] Traceback: {traceback.format_exc(limit=8)}\n")
          sys.stderr.flush()
          continue
        if i % self.freqchk == 0 and i > 0:
          print(f"Model number: {i+1} (total: {nparams}) - checkpoint", flush=True)
          self.__save_chk()
      self.__save_chk()
    else:
      if rank == 0:
        if not self.loadedsamples:
          sys.stderr.write(f"Model Samples not loaded/computed\n")
          sys.stderr.flush()
          comm.Abort(1)
        status = MPI.Status()
        block = self.freqchk
        next_block = 1
        too_frequent = True
        nparams = len(self.samples)
        completed = np.zeros(nparams, dtype=bool)

        if not self.loadedfromchk:
          # Allocate failed array begins ---------------------------------------
          self.failed = np.ones(nparams, dtype=np.uint8) # start w/ all failed
          self.failed = np.asarray(self.failed).astype(bool)

          # Allocate data vectors begins ---------------------------------------
          try: # First run: get data vector size
            dvs = self._compute_dvs_from_sample(self.samples[0])
          except Exception:
            sys.stderr.write(f"Failed in _compute_dvs_from_sample for idx=0\n"
                             f"Cannot determine datavector length\n"
                             f"aborting MPI job\n")
            sys.stderr.write(traceback.format_exc())   # <-- the actual cause
            sys.stderr.flush()
            comm.Abort(1)
          self._accept_payload_row(
            index=0,
            payload=dvs,
            write_row=True,
            allocation_rows=nparams)
          # Allocate data vectors end ------------------------------------------

          completed[0] = True           # first data vector was already computed
          idx0 = np.arange(1, nparams)  # indexes to compute data vectors
        else:
          completed = ~self.failed
          idx0 = np.where(self.failed == True)[0] # indexes to compute data vectors

        tasks   = deque(idx0.tolist())
        nactive = min(nworkers, len(tasks))
        active  = {} # Dict: key = worker (src), value: (idx, t_start)

        # Start MPI workers ----------------------------------------------------
        for w in range(1, nactive+1):
          j = tasks.popleft()
          comm.send((j, self.samples[j]), dest = w, tag = TTAG)
          active[w] = (j, MPI.Wtime())

        # Send all tasks to active MPI workers ---------------------------------
        count  = 0
        while tasks:
          # comm.Iprobe = non-blocking operation used to check for an incoming
          #               message without actually receiving it
          # Why? protect the script against crashes (like CAMB/Class crash)
          if comm.Iprobe(source = MPI.ANY_SOURCE, tag = RTAG, status = status):
            message = comm.recv(source = MPI.ANY_SOURCE,
                                tag = RTAG,
                                status = status)
            count += 1
            src = status.Get_source()
            idx, outcome, detail = self._consume_worker_result_message(
              message=message,
              source=src,
              active=active)

            if outcome == "worker-error":
              sys.stderr.write(f"[Rank 0] Worker {src} failed at idx={idx}\n"
                               f"[Rank 0] Traceback: {detail}\n")
              sys.stderr.flush()
            elif outcome == "invalid-payload":
              sys.stderr.write(
                f"[Rank 0] Worker {src} returned an invalid data-vector "
                f"payload at idx={idx}\n"
                f"[Rank 0] Exception type: {type(detail).__name__}\n"
                f"[Rank 0] Exception message: {detail}\n")
              sys.stderr.flush()
            completed[idx] = True

            if not self.loadedfromchk:
              if count%block == 0:
                too_frequent = False

              if not too_frequent:
                start = (next_block - 1) * block
                end   = min(nparams, next_block * block)
                if completed[start:end].all():
                  self.__save_chk()
                  too_frequent = True
                  next_block += 1
            else:
              if count%block == 0:
                self.__save_chk()

            j = tasks.popleft()
            comm.send((j, self.samples[j]),
                      dest = src,
                      tag  = TTAG)
            active[src] = (j, MPI.Wtime())
          else:
            doabort = False
            for w, (idx, t0) in list(active.items()):
              if (MPI.Wtime()-t0) > TASK_TIMEOUT: # no task runtime > TIMEOUT
                sys.stderr.write(f"[Rank 0] Worker {w} at idx={idx} timed out (MPI RTAG)")
                sys.stderr.flush()
                doabort = True
            if doabort:
              for w, (idx, t0) in list(active.items()): # mark all running tasks as failed
                self._dv_zero(idx)
                self.failed[idx] = True
                completed[idx] = True
              self.__save_chk() # save before crashing
              comm.Abort(1)
            time.sleep(.005) # avoid 100% CPU usage
        # end of while loop

        # drain last tasks from active MPI workers -----------------------------
        while active:
          # comm.Iprobe = non-blocking operation used to check for an incoming
          #               message without actually receiving it
          # Why? protect the script against crashes (like CAMB/Class crash)
          if comm.Iprobe(source = MPI.ANY_SOURCE, tag = RTAG, status = status):
            message = comm.recv(source = MPI.ANY_SOURCE,
                                tag = RTAG,
                                status = status) # drain results
            src = status.Get_source()
            idx, outcome, detail = self._consume_worker_result_message(
              message=message,
              source=src,
              active=active)
            if outcome == "worker-error":
              sys.stderr.write(f"[Rank 0] Worker {src} failed at idx={idx}\n"
                               f"(MPI) Msg: {detail}\n")
              sys.stderr.flush()
            elif outcome == "invalid-payload":
              sys.stderr.write(
                f"[Rank 0] Worker {src} returned an invalid data-vector "
                f"payload at idx={idx}\n"
                f"[Rank 0] Exception type: {type(detail).__name__}\n"
                f"[Rank 0] Exception message: {detail}\n")
              sys.stderr.flush()
            completed[idx] = True
          else:
            doabort = False
            for w, (idx, t0) in list(active.items()):
              if (MPI.Wtime()-t0) > TASK_TIMEOUT: # no task runtime > TIMEOUT
                sys.stderr.write(f"[Rank 0] Worker {w} at idx={idx} timed out (MPI RTAG)")
                sys.stderr.flush()
                doabort = True
            if doabort:
              for w, (idx, t0) in list(active.items()): # mark all running tasks as failed
                self._dv_zero(idx)
                self.failed[idx] = True
                completed[idx] = True
              self.__save_chk() # save before crashing
              comm.Abort(1)
            time.sleep(.005) # avoid 100% CPU usage
        # end active workers

        # stop workers ---------------------------------------------------------
        self.__save_chk() # save before sending stop sign
        active = {}       # reinitialize active (extra safety)
        for w in range(1, nworkers + 1): # stop workers
          comm.send((0, None), dest=w, tag=STAG)
          active[w] = (0, MPI.Wtime())
        while active:
          # comm.Iprobe = non-blocking operation used to check for an incoming
          #               message without actually receiving it
          # Why? protect the script against crashes (like CAMB/Class crash)
          if comm.Iprobe(source=MPI.ANY_SOURCE, tag=DTAG, status=status):
            message = comm.recv(
              source=MPI.ANY_SOURCE,
              tag=DTAG,
              status=status)
            src = status.Get_source()
            validate_worker_done_message(
              message=message,
              source=src,
              active=active)
            del active[src]
          else:
            for w, (_, t0) in list(active.items()):
              if (MPI.Wtime()-t0) > STOP_TIMEOUT: # no task runtime > TIMEOUT
                sys.stderr.write(f"[Rank 0] Worker {w} timed out (MPI DTAG)")
                sys.stderr.flush()
                comm.Abort(1)
            time.sleep(.005) # avoid 100% CPU usage
        # end stop workers

      else:
        status = MPI.Status()
        while (True):
          # poll politely instead of busy-waiting in a blocking recv
          while not comm.Iprobe(source=0,
                                tag=MPI.ANY_TAG,
                                status=status):
            time.sleep(.005) # ~0% CPU while rank 0 runs the MCMC
          idx, sample = comm.recv(source = 0,
                                  tag = MPI.ANY_TAG,
                                  status = status) # try block on main b/c if
                                                   # rank zero throws an exception
                                                   # before send, MPI hangs
          if (status.Get_tag() == STAG):
            comm.send(("worker done", rank), dest = 0, tag = DTAG)
            break
          try:
            dvs = self._compute_dvs_from_sample(sample)
            comm.send(("ok", idx, dvs), dest = 0, tag = RTAG)
          except Exception as e:
            sys.stderr.write(f"[Rank {rank}] Exception type: {type(e).__name__}\n")
            sys.stderr.write(f"[Rank {rank}] Exception message: {e}\n")
            comm.send(("err", idx, traceback.format_exc(limit=8)),
                      dest = 0,
                      tag = RTAG)
            continue
    return

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# main helper shared by the drivers
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def run_generator(dataset_cls, prog):
  """
  Parse the shared CLI and run one dataset-generator driver under MPI.

  Every driver's __main__ block is this call: it keeps the crash
  behavior of the original script (print the traceback, MPI-abort so
  the other ranks do not hang, then finalize).

  Arguments:
    dataset_cls = the driver's GeneratorCore subclass.
    prog        = the driver script name for --help.
  """
  # strict parse: reject an unknown / misspelled flag (unit-17 ruling) rather
  # than silently ignoring it before staging or spawning workers.
  args = make_cli_parser(prog).parse_args()
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  try:
    generator = dataset_cls(args)
  except Exception:
    traceback.print_exc()
    comm.Abort(1)   # other ranks don’t hang in recv/barrier
  MPI.Finalize()
  exit(0)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
