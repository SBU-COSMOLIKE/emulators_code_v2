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
import emcee, argparse, os, sys, yaml, time, traceback
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
from compute_data_vectors.dataset_manifest import validate_run_control
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
                      help="only compute and output train/test/val chain",
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
UNIFORM_BOUNDARY_INTERIOR_POLICY = "nextafter-toward-interval-interior-v1"


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
PROBE_FAMILY = {"cs":          "cosmolike",
                "ggl":         "cosmolike",
                "gc":          "cosmolike",
                "cmblensed":   "cmb",
                "cmbunlensed": "cmb",
                "mps":         "grid2d",
                "background":  "grid"}

# The units the CMB driver asks cobaya for at every sample
# (dataset_generator_cmb.py calls get_Cl(ell_factor=False, units="muK2")). The
# spectra are published in these units, so the record carries them. A family
# with no angular power spectrum has no such units and says so instead.
CMB_CL_UNITS = "muK2"

# The name a cobaya run gives the neutrino splitting when it states one. CAMB
# has its own default; the record does not supply it, because a convention the
# run never wrote down is not a fact about this run.
NEUTRINO_CONVENTION_KEY = "neutrino_hierarchy"


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
    if rank == 0:
      self.__run_mcmc()
    if self.run_control.dataset_mode == "full":
      self.__generate_datavectors()

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
    Path(f"{root}/chains").mkdir(parents=True, exist_ok=True)

    self.append = self.run_control.append
    self.bounds = None           # the support the sampler drew from (resolved)
    self.bounds_requested = None # the support the prior declared (requested)
    self.bounds_adj = (
      self.args.boundary if self.args.boundary is not None and 0 < self.args.boundary < 1 else 1
    )
    self.covmat = None
    self.dtype = np.float32
    self.dvsf = None
    self.derived = True
    self.dvs_is_memmap = False
    self.freqchk = 5000 if self.args.freqchk is None else self.args.freqchk
    if self.freqchk < 1000:
      raise ValueError("--freqchk must be >= 1000") # avoid too much chk
    self.failed = None        # track which models failed to compute dv
    self.failf = None
    self.fiducial = None
    self.inv_covmat = None
    self.loadchk = self.run_control.loadchk
    self.loadedfromchk = False  # check if loaded from checkpoint sucessfully
    self.loadedsamples = False  # check loaded samples sucessfully
    self.maxcorr = 0.15 if self.args.maxcorr is None else self.args.maxcorr
    if not (0.01 < self.maxcorr <= 1):
      raise ValueError("--maxcorr must be between (0.01,1]")
    # the sampling seed: a required non-bool integer, no default. Every random
    # draw goes through this owned Generator instead of the process-global
    # np.random, so two runs with the same seed, YAML and code produce the same
    # parameter table (bool is refused: argparse int never yields one, but a
    # programmatic caller might pass True, which would silently mean seed 1).
    if isinstance(self.args.seed, bool) or not isinstance(self.args.seed, int):
      raise ValueError("--seed must be an integer (a non-bool int); got "
                       + repr(self.args.seed))
    self.seed = int(self.args.seed)
    self.rng = np.random.default_rng(self.seed)
    self.names = None
    self.model = None
    self.nparams = 10000 if self.args.nparams is None else self.args.nparams
    if self.nparams < 200:
      raise ValueError("--nparams must be >= 200")
    self.paramsf = None
    self.probe = None
    self.sampled_params = None
    self.samples = None
    self.temp = 128 if self.args.temp is None else self.args.temp
    if self.temp < 1:
      raise ValueError("--temp must be => 1")
    self.unif = 0 if self.args.unif is None else self.args.unif
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

    missing = [k for k in ['params', 'likelihood', 'train_args'] if k not in info]
    if missing:
      raise KeyError(f"Cobaya YAML missing required blocks {missing}: {self.yaml}")

    train_args = info["train_args"]
    required_keys = ['probe', 'ord'] + list(self.EXTRA_TRAIN_KEYS)
    if not self.unif == 1:
      required_keys += ['fiducial', 'params_covmat_file']

    missing = [k for k in required_keys if k not in train_args]
    if missing:
      raise KeyError(f"Cobaya YAML missing required keys {missing}: {self.yaml}")

    #---------------------------------------------------------------------------
    # Load Cobaya model (needed for computing likelihood), cov matrix...
    #---------------------------------------------------------------------------
    try:
      self.model = get_model(info)
    except Exception as e:
      raise RuntimeError(f"get_model failed for {self.yaml}: {e}") from e

    self.probe = train_args["probe"]
    if self.probe not in self.VALID_PROBES:
      raise ValueError(f"Invalid Probe: {self.probe}")

    self.sampled_params = train_args['ord'][0]  # preferred ordering of params

    # driver-specific train_args (and model requirements, if any) --------------
    self._read_train_args(train_args)

    if not self.unif == 1:
      fid = train_args["fiducial"] # load fiducial data vector

      # load cov param matrix --------------------------------------------------
      raw_covmat_file = train_args["params_covmat_file"]
      with open(f"{fileroot}/{raw_covmat_file}") as f:
        raw_covmat_params_names = np.array(f.readline().split()[1:])
        raw_covmat = np.loadtxt(f)

      #-------------------------------------------------------------------------
      # Reorder fiducial, bounds and covmat to follow ['train_args']['ord']
      #-------------------------------------------------------------------------
      # Reorder fiducial -------------------------------------------------------
      self.fiducial = np.array([fid[p] for p in self.sampled_params],
                               copy=True, dtype=self.dtype)

      # Reorder covmat ---------------------------------------------------------
      pidx = {p : i for i, p in enumerate(raw_covmat_params_names)}
      try:
        idx = np.array([pidx[p] for p in self.sampled_params],
                       copy=True,
                       dtype=int)
      except KeyError as e:
        raise ValueError(f"{e.args[0]!r} not found in cov header") from None
      covmat = raw_covmat[np.ix_(idx, idx)]

    # Reorder bounds -----------------------------------------------------------
    self.names = list(self.model.parameterization.sampled_params().keys())
    if set(self.sampled_params) != set(self.names):
      raise ValueError(f"train_args.ord {set(self.sampled_params)}"
                       f" != model sampled params {set(self.names)}")
    idx = self.reorder_idx_from_yaml_to_ord()

    # Here T (temp) stretch the hard bounds on parameters with Gaussian prior --
    tmp = np.array(self.model.prior.bounds(confidence=1.0),
                                           copy=True,
                                           dtype=self.dtype)[idx,:]
    self.bounds = np.array(self.model.prior.bounds(confidence=0.9999994),
                           copy=True,
                           dtype=self.dtype)[idx,:]

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

    # adjust covmat- -----------------------------------------------------------
    if not self.unif == 1:
      #-------------------------------------------------------------------------
      # Reduce correlation on the covariance matrix to max = args.maxcorr
      #-------------------------------------------------------------------------
      sig   = np.sqrt(np.diag(covmat))
      n = len(sig)
      outer = np.outer(sig, sig)
      corr  = covmat / outer
      m = np.abs(corr - np.eye(n)).max()
      if m > self.maxcorr:
        corr /= max(1.0, m / self.maxcorr) if m > 0 else 1.0
        np.fill_diagonal(corr, 1.0)
      covmat = corr * outer

      #-------------------------------------------------------------------------
      # Compute covmat inverse
      #-------------------------------------------------------------------------
      C = 0.5 * (covmat + covmat.T)  # enforce symmetry
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
  # data-vector store (default: the single 2D array at {dvsf}.npy)
  #-----------------------------------------------------------------------------
  def _dv_chk_files(self):
    """Files the checkpoint loader must find before trusting a chk."""
    return [f"{self.dvsf}.npy"]

  def _dv_load_chk(self):
    """Load the data-vector store from its checkpoint files (RAM-aware)."""
    arr = np.load(f"{self.dvsf}.npy",
                  mmap_mode = "r",
                  allow_pickle = False)
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
    self.datavectors[i] = dvs

  def _dv_zero(self, i):
    """Zero row i (a failed sample)."""
    self.datavectors[i, :] = 0.0

  #-----------------------------------------------------------------------------
  # save/load checkpoint
  #-----------------------------------------------------------------------------
  def __load_chk(self):
    rtnvar = False
    if self.loadchk == 1:
      loadchk = all([os.path.isfile(x) for x in self._dv_chk_files() +
                                                [f"{self.failf}.txt",
                                                 f"{self.paramsf}.covmat",
                                                 f"{self.paramsf}.ranges",
                                                 f"{self.paramsf}.1.txt"]])
      if loadchk:
        # load sample file begins ----------------------------------------------
        # row 0/1 rows are weights, lnp. Last row is chi2
        self.samples = np.atleast_2d(np.loadtxt(f"{self.paramsf}.1.txt",
                                                dtype=self.dtype))[:,2:-1]
        if self.samples.ndim != 2:
          raise ValueError(f"samples must be 2D, got {self.samples.shape}")
        # load sample file ends ------------------------------------------------

        # load fail file begins ------------------------------------------------
        self.failed = np.atleast_1d(np.loadtxt(f"{self.failf}.txt",
                                               dtype=np.uint8))
        self.failed = np.asarray(self.failed).astype(bool)
        if self.failed.ndim != 1:
          raise ValueError(f"failed must be 1D, got {self.failed.shape}")

        if self.samples.shape[0] != self.failed.shape[0]:
          raise ValueError(f"Incompatible samples/failed chk files")
        # load fail file ends --------------------------------------------------

        # load datavectors (store-specific; row-count checks live inside) ------
        self._dv_load_chk()

        print("Loaded models from chk")
        if self.append == 0:
          self.loadedsamples = True
          self.loadedfromchk = True
        rtnvar = True
    return rtnvar

  def __save_chk(self):
    # save data vector file (store-specific) ------------------------------------
    self._dv_save()
    # save fail file -----------------------------------------------------------
    # save (flush) dvs to tmp file (safer) -------------------------------------
    np.savetxt(f"{self.failf}.tmp.txt", self.failed.astype(np.uint8), fmt="%d")
    # save data vector file (from tmp) -----------------------------------------
    os.replace(f"{self.failf}.tmp.txt", f"{self.failf}.txt")

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
    if math.isinf(logprior):
      return -np.inf
    elif math.isinf(self.__param_logprior(x)):
      # this is important when --boundary command line option is < 1
      return -np.inf
    else:
      logp = (-0.5*(y @ self.inv_covmat @ y) + logprior)/self.temp
      return logp

  #-----------------------------------------------------------------------------
  # the dataset's scientific record (schema: emulator/fixed_facts.py)
  #-----------------------------------------------------------------------------
  def _resolved_constants(self):
    """
    Read every value the resolved Cobaya model pins to a constant.

    The reading itself belongs to emulator/fixed_facts.py, and this method is
    one of its callers. The generator is not the only code that has to read the
    model this way: each cobaya adapter reads the same model, at the start of a
    chain, to check a saved emulator's record against the cosmology now being
    sampled. The producer WRITES the fixed facts, the adapters CHECK them, and
    the two must read them identically — down to which block wins a name that
    both the params block and a theory component state. A copy of the reader
    living here would be a second author of that fact.

    Arguments:
      none.

    Returns:
      a mapping of name to plain value, holding every constant the model states.
      It is a superset of the coordinates the record reports on; the caller reads
      the names it needs.
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
      ValueError when the run's probe has no output family, or when the
      matter-power base cannot be named.
    """
    if self.probe not in PROBE_FAMILY:
      raise ValueError(
        "the probe " + repr(self.probe) + " has no output family, so the "
        "dataset cannot record which family it feeds. Add the probe to "
        "PROBE_FAMILY in compute_data_vectors/generator_core.py when a driver "
        "starts accepting it.")
    family = PROBE_FAMILY[self.probe]

    sampled = set(self.sampled_params)
    pinned  = self._resolved_constants()

    cosmology_fixed = {}
    for key in fixed_facts.COSMOLOGY_FIXED_KEYS:
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

    # the dark-energy law, read the way emulator/syren_base.py reads these two
    # names: an absent equation-of-state parameter means the model is a
    # cosmological constant (w = -1, wa = 0), not that a fact went missing. A
    # name the run sampled, or pinned away from that constant, is a name the law
    # consumes. dark_energy_inputs names what the law consumes, not what the run
    # happened to write down, so that a run pinning w at -1 and a run that never
    # mentions w describe the one universe they do describe.
    w_free  = ("w" in sampled) or ("w" in pinned and pinned["w"] != -1.0)
    wa_free = ("wa" in sampled) or ("wa" in pinned and pinned["wa"] != 0.0)
    if wa_free:
      dark_energy_law    = "w0wa-cpl"
      dark_energy_inputs = ["w", "wa"]
    elif w_free:
      dark_energy_law    = "constant-w"
      dark_energy_inputs = ["w"]
    else:
      dark_energy_law    = "cosmological-constant"
      dark_energy_inputs = []

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
    # base to name. write_base is the mps driver's own switch, so the base class
    # asks for it rather than assuming every driver has one.
    base_identity = fixed_facts.NOT_APPLICABLE
    if self.probe == "mps" and getattr(self, "write_base", False):
      base_identity = self._syren_base_identity()

    # the driver that produced the dataset, by the name a user types on the
    # command line. All four driver classes are named `dataset`, so the class
    # name says nothing about which one ran; the module it was defined in does.
    # A driver run by path is the __main__ module, whose file is that script.
    generator = type(self).__module__
    module = sys.modules.get(generator, None)
    path = getattr(module, "__file__", None)
    if path is not None:
      generator = Path(path).stem

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
    facts = self._resolve_fixed_facts()

    requested = {}
    resolved  = {}
    for i in range(len(names)):
      requested[names[i]] = (self.bounds_requested[i, 0],
                             self.bounds_requested[i, 1])
      resolved[names[i]]  = (self.bounds[i, 0],
                             self.bounds[i, 1])

    text = fixed_facts.build_sidecar(
             dataset_id          = fixed_facts.chain_digest(
                                     chain_path=f"{self.paramsf}.1.txt"),
             generator           = facts["generator"],
             family              = facts["family"],
             cosmology_fixed     = facts["cosmology_fixed"],
             neutrino_convention = facts["neutrino_convention"],
             flat_only           = facts["flat_only"],
             dark_energy_law     = facts["dark_energy_law"],
             dark_energy_inputs  = facts["dark_energy_inputs"],
             cl_units            = facts["cl_units"],
             base_identity       = facts["base_identity"],
             names               = names,
             requested           = requested,
             resolved            = resolved)
    with open(f"{self.paramsf}{fixed_facts.SIDECAR_SUFFIX}", "w") as f:
      f.write(text)

  #-----------------------------------------------------------------------------
  # run mcmc
  #-----------------------------------------------------------------------------
  def __run_mcmc(self):
    try:
      loadedfromchk = self.__load_chk()
    except Exception as e:
      sys.stderr.write(f"[load_chk] failed: {e}\n")
      traceback.print_exc(file=sys.stderr)
      loadedfromchk = False

    if (loadedfromchk == False) or (loadedfromchk == True and self.append == 1):
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
        xf  = sampler.get_chain(flat=True, discard=burnin, thin=1)
        xf, keep = np.unique(xf, axis=0, return_index=True)
        lnp = sampler.get_log_prob(flat=True, discard=burnin, thin=1)[keep, None]
        if len(xf) < self.nparams:
          print(f"Warning: only {len(xf)} unique rows, requested {self.nparams}")
        else:
          indices = self.rng.choice(np.arange(len(xf)), size=self.nparams, replace=False)
          xf  = xf[indices,:]
          lnp = lnp[indices,:]
        nparams = len(xf)
        # Double check that prior is not -infty --------------------------------
        idx = self.reorder_idx_from_ord_to_yaml()
        for i, x in enumerate(xf):
          logprior = self.model.prior.logp(x[idx])
          if math.isinf(logprior):
              raise ValueError(f"Sample {i} has -inf prior. (this should not happen)"
                               f"Values: {dict(zip(self.sampled_params, x))}")
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
          if math.isinf(logprior):
              raise ValueError(f"Sample {i} has -inf prior. (this should not happen)"
                               f"Values: {dict(zip(self.sampled_params, x))}")
      w = np.ones((nparams,1), dtype=self.dtype)
      chi2 = -2*lnp
      if not loadedfromchk:
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
                   np.concatenate([w, lnp, xf, chi2], axis=1),
                   fmt="%.9e",
                   header=hd + ' '.join(["weights", "lnp"] + names + ["chi2*"]),
                   comments="# ")

        # copy samples to self.samples  ----------------------------------------
        self.samples = np.array(xf, copy=True, dtype=self.dtype)

        # save paramname files -------------------------------------------------
        param_info = self.model.info()['params']
        latex  = [param_info[x]['latex'] for x in names]
        paramnames = copy.deepcopy(names)
        paramnames.append("chi2*")
        latex.append("\\chi^2")
        np.savetxt(f"{self.paramsf}.paramnames",
                   np.column_stack((paramnames,latex)),
                   fmt="%s")

        # save a cov matrix ------------------------------------------------------
        with contextlib.redirect_stdout(io.StringIO()): # so getdist dont write in terminal
          np.savetxt(f"{self.paramsf}.covmat",
                     np.array(loadMCSamples(f"{self.paramsf}",
                                            settings={'ignore_rows': u'0.'}).cov(pars=names),
                              copy=True,
                              dtype=self.dtype),
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
        gc.collect()  # save RAM memory
      else:
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # This option = loadedfromchk == True and self.append == 1
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
        gc.collect()  # save RAM memory

        self.samples = np.atleast_2d(np.loadtxt(fname, dtype=self.dtype))[:,2:-1]
        if self.samples.ndim != 2:
          raise ValueError(f"samples must be 2D, got {self.samples.shape}")
        # append chain file ends -----------------------------------------------

        # append fail file begins ----------------------------------------------
        fname = f"{self.failf}.txt";
        failed = np.ones((nparams, 1), dtype=np.uint8) # start w/ all failed
        with open(fname, "a") as f: # append mode
          np.savetxt(f, failed.astype(np.uint8), fmt="%d")

        self.failed = np.atleast_1d(np.loadtxt(fname, dtype=np.uint8))
        self.failed = np.asarray(self.failed).astype(bool)
        if self.failed.ndim != 1:
          raise ValueError(f"failed must be 1D, got {self.failed.shape}")

        if self.samples.shape[0] != self.failed.shape[0]:
          raise ValueError(f"Incompatible samples/failed chk files")
        # append fail file begins ----------------------------------------------

        # append dvs (with zeros; store-specific, row-count check inside) ------
        self._dv_append(nparams)

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
        self._dv_alloc(nrows=nparams, first_dvs=dvs)
        # Allocate data vectors end --------------------------------------------

        self._dv_write(0, dvs)      # first data vector was already computed
        self.failed[0] = False      # first data vector was already computed

        idx = np.arange(1, nparams) # indexes to compute data vectors
      else:
        idx = np.where(self.failed == True)[0] # indexes to compute data vectors

      for i in idx:
        try:
          dvs = self._compute_dvs_from_sample(self.samples[i])
          self.failed[i] = False
        except Exception as e: # set datavector to zero and continue
          self.failed[i] = True
          self._dv_zero(i)
          sys.stderr.write(f"[Rank 0] Worker failed at idx={i}\n")
          sys.stderr.write(f"[Rank 0] Exception type: {type(e).__name__}\n")
          sys.stderr.write(f"[Rank 0] Exception message: {e}\n")
          sys.stderr.write(f"[Rank 0] Traceback: {traceback.format_exc(limit=8)}\n")
          sys.stderr.flush()
          continue
        self._dv_write(i, dvs)
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
          self._dv_alloc(nrows=nparams, first_dvs=dvs)
          # Allocate data vectors end ------------------------------------------

          self._dv_write(0, dvs)        # first data vector was already computed
          self.failed[0] = False        # first data vector was already computed
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
            kind, idx, payload = comm.recv(source = MPI.ANY_SOURCE,
                                       tag = RTAG,
                                       status = status)
            count += 1
            src = status.Get_source()
            if src in active:
              del active[src]

            if kind == "err": # set datavector to zero and continue
              self._dv_zero(idx)
              self.failed[idx] = True
              sys.stderr.write(f"[Rank 0] Worker {src} failed at idx={idx}\n"
                               f"[Rank 0] Traceback: {payload}\n")
              sys.stderr.flush()
            else:
              self._dv_write(idx, payload)
              self.failed[idx] = False
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
            kind, idx, payload = comm.recv(source = MPI.ANY_SOURCE,
                                           tag = RTAG,
                                           status = status) # drain results
            src = status.Get_source()
            if kind == "err":
              self._dv_zero(idx)
              self.failed[idx] = True
              sys.stderr.write(f"[Rank 0] Worker {src} failed at idx={idx}\n"
                               f"(MPI) Msg: {payload}\n")
              sys.stderr.flush()
            else:
              self._dv_write(idx, payload)
              self.failed[idx] = False
            completed[idx] = True
            if src in active: # Remove worker from active list
              del active[src]
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
            _ = comm.recv(source=MPI.ANY_SOURCE, tag=DTAG, status=status)
            src = status.Get_source()
            if src in active:
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
