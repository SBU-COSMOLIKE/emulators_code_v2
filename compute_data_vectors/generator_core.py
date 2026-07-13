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
import psutil, gc, math, copy, tempfile
from cobaya.model import get_model
from mpi4py import MPI
from pathlib import Path
from getdist import loadMCSamples
from collections import deque
from contextlib import contextmanager
from numpy.lib.format import open_memmap
import contextlib, io
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
  return parser
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Free Functions
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
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
    self.args = cli_args
    self.setup = False
    self.__setup_flags()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
      self.__run_mcmc()
    if not self.args.chain == 1:
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

    self.append = 0 if self.args.append is None else self.args.append
    self.bounds = None
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
    self.loadchk = 0 if self.args.loadchk is None else self.args.loadchk
    self.loadedfromchk = False  # check if loaded from checkpoint sucessfully
    self.loadedsamples = False  # check loaded samples sucessfully
    self.maxcorr = 0.15 if self.args.maxcorr is None else self.args.maxcorr
    if not (0.01 < self.maxcorr <= 1):
      raise ValueError("--maxcorr must be between (0.01,1]")
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
        sampler.run_mcmc(initial_state = self.fiducial[np.newaxis] +
                                         0.5*np.sqrt(np.diag(self.covmat))*
                                         np.random.normal(size=(nwalkers,ndim)),
                         nsteps=nsteps,
                         progress=False)
        xf  = sampler.get_chain(flat=True, discard=burnin, thin=1)
        xf, keep = np.unique(xf, axis=0, return_index=True)
        lnp = sampler.get_log_prob(flat=True, discard=burnin, thin=1)[keep, None]
        if len(xf) < self.nparams:
          print(f"Warning: only {len(xf)} unique rows, requested {self.nparams}")
        else:
          indices = np.random.choice(np.arange(len(xf)), size=self.nparams, replace=False)
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
        bds = self.bounds.copy()
        # extra safety so logprior is not -infty --------------------
        bds[:,0] = np.where(bds[:,0] > 0, 1.0001*self.bounds[:,0],0.9999*self.bounds[:,0])
        bds[:,1] = np.where(bds[:,1] > 0, 0.9999*self.bounds[:,1],1.0001*self.bounds[:,1])
        xf  = np.random.uniform(low  = bds[:,0],
                                high = bds[:,1],
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
        bds = self.bounds.copy()
        hd  = ["weights","lnp"] + names
        rows = [(str(n),float(l),float(h)) for n,l,h in zip(names,bds[:,0],bds[:,1])]
        with open(f"{self.paramsf}.ranges", "w") as f:
          f.write(f"# {' '.join(hd)}\n")
          f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

        # save chain begins ----------------------------------------------------
        fname = f"{self.paramsf}.1.txt";
        if not self.unif == 1:
          hd=f"nwalkers={nwalkers}\n"
        else:
          hd=f"Uniform Sampling\n"
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
