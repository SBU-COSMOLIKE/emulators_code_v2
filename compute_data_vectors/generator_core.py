"""
Shared machinery for the dataset-generator drivers.

Every training-set generator in this package does the same job: draw
cosmological parameter samples, farm one expensive Boltzmann/cosmolike
call per sample over MPI, and write the results as the dv/params dumps
the training stack stages. This module holds that shared machinery ONCE;
each physics family ships a thin driver file beside it that subclasses
GeneratorCore and fills in only what differs.

    dataset_generator_lensing.py      cosmolike data vectors (cs/ggl/gc)
    dataset_generator_cmb.py          CMB spectra (tt/te/ee/pp)
    dataset_generator_background.py   H(z) and D_M(z) grids
    dataset_generator_mps.py          linear P(k,z) and nonlinear boost
              |               |               |               |
              +---------------+---------------+---------------+
                              |
                   generator_core.py (this file)
                   - CLI parser (identical flags for every driver)
                   - parameter sampling (emcee Gaussian / uniform),
                     seeded by the required --seed flag
                   - chain + .paramnames + .ranges + .covmat writers
                   - the .facts.yaml scientific record beside the chain
                   - checkpoint save/load/append
                   - RAM-aware data-vector storage (default: one 2D array)
                   - MPI master/worker farm with timeouts

The subclass surface (everything a driver may override):

    VALID_PROBES              tuple of accepted train_args.probe names.
    EXTRA_TRAIN_KEYS          extra REQUIRED train_args keys (e.g. lrange).
    FAMILY                    the scientific-record family name written to
                              the .facts.yaml sidecar: "cosmolike", "cmb",
                              "grid" (background), or "grid2d" (mps).
    PROGRAM                   the driver script name, recorded as the
                              producer in the .facts.yaml sidecar.
    _read_train_args(ta)      read/validate the driver's own train_args
                              keys; called once during setup, after the
                              cobaya model exists (so a driver may also
                              add model requirements here).
    _facts_base_identity()    one line naming the analytic base the dumps
                              sit on top of, or "n/a" (the default) when
                              the family has none. Only the mps driver
                              overrides it (the syren base).
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
getdist pairing rules automatically. The <paramfile>.facts.yaml sidecar
(format owned by emulator/fixed_facts.py) records the cosmology the run
held fixed and the parameter region it sampled; the training stack
refuses a dump without it, so the record is written whenever the chain
is written.
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

# The scientific-record format lives in the emulator package, one folder up
# from this file. The drivers run by path, so the repo root is not on
# sys.path when this module is imported; the prepend makes
# `from emulator import fixed_facts` resolve. fixed_facts imports only
# hashlib at module scope, so a generator run pays nothing extra to carry it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)
from emulator import fixed_facts

# units of the CMB spectra dumps (raw C_ell, muK^2), recorded in the
# scientific sidecar of the cmb family.
CMB_CL_UNITS = "muK2"
# the Cobaya/CAMB setting that names how the total neutrino mass is split
# into mass states; recorded in the sidecar when the model pins it.
NEUTRINO_CONVENTION_KEY = "neutrino_hierarchy"
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
                      help="1 = uniform sampling inside the prior bounds; "
                           "0 = Gaussian sampling from the covariance in "
                           "train_args.params_covmat_file",
                      type=int,
                      choices=[0,1],
                      required=True)
  parser.add_argument("--temp",
                      dest="temp",
                      help="temperature T: Gaussian sampling divides the "
                           "log-posterior curvature by T (wider cloud), and "
                           "a prior without hard bounds is given edges "
                           "stretched by T",
                      type=int)
  parser.add_argument("--maxcorr",
                      dest="maxcorr",
                      help="maximum |correlation| kept in the sampling "
                           "covariance; stronger correlations are scaled "
                           "down to this value",
                      type=float)
  parser.add_argument("--loadchk",
                      dest="loadchk",
                      help="1 = resume from the checkpoint files at the "
                           "output paths (recompute only rows marked failed)",
                      type=int,
                      choices=[0,1])
  parser.add_argument("--freqchk",
                      dest="freqchk",
                      help="how many computed rows between checkpoint saves",
                      type=int)
  parser.add_argument("--append",
                      dest="append",
                      help="1 = grow a loaded checkpoint by --nparams new "
                           "rows (requires --loadchk 1)",
                      type=int,
                      choices=[0,1])
  parser.add_argument("--boundary",
                      dest="boundary",
                      help="fraction of each prior interval kept, trimmed "
                           "symmetrically from both edges; a test/val dump "
                           "uses < 1 because emulator accuracy degrades at "
                           "the training boundary",
                      type=float)
  parser.add_argument("--seed",
                      dest="seed",
                      help="required integer sampling seed. Every random "
                           "draw (uniform sampling, the emcee walkers and "
                           "moves, the thinning subselection) comes from "
                           "this seed, so two runs with the same seed, YAML "
                           "and code produce the same parameter table. No "
                           "default: an unrecorded seed cannot be replayed.",
                      type=int,
                      required=True)
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


def dark_energy_facts(parameterization, pinned):
  """Describe the run's dark-energy coordinates in canonical (w, wa) form.

  Cobaya lets a run sample coordinates that are convenient for the sampler
  and then calculate the coordinates a theory consumes. The shipped
  matter-power configuration uses exactly that feature: it samples w0pwa
  and w and calculates wa = w0pwa - w. Looking only at the sampled names
  would miss a varying wa and record the wrong physical law, so this
  function also walks Cobaya's input_dependencies: a calculated input
  varies when any of its declared dependencies is sampled. The returned
  record always uses the physical names w and wa, even when the YAML
  spells the present-day value w0 or supplies the sum w0pwa = w0 + wa.

  Arguments:
    parameterization = the resolved model's Cobaya Parameterization object.
    pinned = mapping of constants resolved from the model (parameter
             constants and theory-component defaults).

  Returns:
    (fixed, varying, law, inputs):
      fixed   = mapping of canonical fixed coordinates ("w", "wa") to
                values, for the coordinates this run does NOT vary.
      varying = set of canonical coordinate names that change across
                samples (subset of {"w", "wa"}).
      law     = "cosmological-constant", "constant-w", or "w0wa-cpl".
      inputs  = list of canonical coordinates a consumer must supply to
                evaluate this law ([], ["w"], or ["w", "wa"]).

  Raises:
    ValueError when the model gives inconsistent fixed values (for
    example w0pwa that disagrees with w0 + wa), or fixes w0pwa without
    fixing w, so the separate values cannot be recovered.
  """
  sampled_names = set(parameterization.sampled_params())
  varying_names = set(sampled_names)
  for name, required in parameterization.input_dependencies.items():
    if set(required).intersection(sampled_names):
      varying_names.add(name)

  # Parameter constants take precedence over theory-component defaults.
  # A fixed-looking value for a coordinate Cobaya says varies is ignored:
  # a theory default must not pin a sampled parameter in the record.
  resolved = dict(pinned)
  resolved.update(parameterization.constant_params())

  # the same representation tolerance emulator.syren_base uses publicly for
  # dark-energy coordinates (float32 storage), repeated here so the three
  # non-mps generators need not import the analytic formulas.
  atol = 4.0 * float(np.finfo(np.float32).eps)

  def fixed_number(name):
    if name not in resolved or name in varying_names:
      return None
    value = float(resolved[name])
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
      "the generator cannot determine the separate w and wa values to record")

  varying = set()
  if varying_names.intersection(("w", "w0")):
    varying.add("w")
  if "wa" in varying_names or "w0pwa" in varying_names:
    varying.add("wa")

  # an absent coordinate is the standard model's value for it, not a
  # missing fact (the same reading emulator/syren_base.py applies).
  if "w" not in varying and present is None:
    present = -1.0
  if "wa" not in varying and wa is None:
    wa = 0.0

  fixed = {}
  if "w" not in varying:
    fixed["w"] = present
  if "wa" not in varying:
    fixed["wa"] = wa

  if "wa" in varying or not np.isclose(wa if wa is not None else 0.0,
                                       0.0, rtol=0.0, atol=atol):
    law = "w0wa-cpl"
    inputs = ["w", "wa"]
  elif "w" in varying or not np.isclose(present if present is not None
                                        else -1.0, -1.0,
                                        rtol=0.0, atol=atol):
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
  construction runs the whole job:

      __init__(cli_args)
         │  __setup_flags     resolve flags, YAML, cobaya model, bounds,
         │                    covariance, output stems (every rank)
         ▼
      rank 0: __run_mcmc      load a checkpoint or draw the parameter
         │                    table; write .1.txt/.paramnames/.ranges/
         │                    .covmat/.facts.yaml
         ▼
      all ranks: __generate_datavectors   one physics call per row,
         │                    farmed over MPI (skipped when --chain 1)
         ▼
      files in <root>/chains/   the complete dataset

  (legend: rank = one MPI process; rank 0 is the master that samples,
  assigns rows, and writes files; the other ranks only compute data
  vectors. cli_args = the argparse namespace from make_cli_parser.)

  PS: a "driver" is one of the thin dataset_generator_*.py scripts; it
  subclasses this core and fills in the family physics (which cobaya
  products to request, how to compute one row, how the row store is
  laid out on disk). Everything the drivers share — sampling, seeding,
  checkpointing, MPI, the sidecar files — lives here once.
  """
  VALID_PROBES = ()      # driver MUST override: accepted train_args.probe
  EXTRA_TRAIN_KEYS = ()  # driver MAY override: extra required train_args keys
  FAMILY = None          # driver MUST override: scientific-record family
  PROGRAM = None         # driver MUST override: driver script name

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
    """
    Resolve everything a run needs before any sampling or physics.

    Runs on every MPI rank, in this order:

      1. resolve $ROOTDIR + --root/--fileroot into the project paths;
      2. validate the command-line flags (ranges, legal flag pairs) and
         seed the owned random Generator from --seed;
      3. load the training YAML and require its params / likelihood /
         train_args blocks;
      4. build the cobaya model (get_model), check train_args.probe
         against the driver's whitelist, and hand the driver its own
         train_args keys (_read_train_args — this is also where a
         driver registers its cobaya requirements);
      5. Gaussian mode only: load the fiducial point and the parameter
         covariance, reduce its correlations to --maxcorr, and build a
         stabilized inverse for the tempered posterior;
      6. read the prior bounds off the model, reorder them into
         train_args.ord order, keep a copy of the requested support
         (bounds_requested), then stretch infinite endpoints by the
         temperature and trim the --boundary margin;
      7. build the three output stems (data vectors, parameters,
         failures) under <root>/chains/, suffixed _chain_only when
         --chain 1;
      8. Gaussian mode only: check the fiducial point sits inside the
         resolved sampling bounds.

    Everything later reads the attributes set here; nothing here writes
    an output file, so any refusal in this method leaves the disk
    untouched.

    Raises:
      ValueError / KeyError / FileNotFoundError / RuntimeError naming
      the flag, YAML block, file, or bound that is wrong — always
      before any output path is created.
    """
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
    boundary = 1.0 if self.args.boundary is None else self.args.boundary
    if not (math.isfinite(boundary) and 0.0 < boundary <= 1.0):
      # an out-of-range boundary must refuse, never silently become 1
      # (a test/val dump generated on the full interval would overlap the
      # training support the flag was supposed to trim away from).
      raise ValueError(f"--boundary must be in (0, 1], got {boundary}")
    self.bounds_adj = boundary
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
    if self.append == 1 and self.loadchk != 1:
      raise ValueError(
        "--append=1 requires --loadchk=1: append extends a validated prior "
        "dataset; it never means fresh generation at the same output path")
    self.loadedfromchk = False  # check if loaded from checkpoint sucessfully
    self.loadedsamples = False  # check loaded samples sucessfully
    self.maxcorr = 0.15 if self.args.maxcorr is None else self.args.maxcorr
    if not (0.01 < self.maxcorr <= 1):
      raise ValueError("--maxcorr must be between (0.01,1]")
    # every random draw goes through this one seeded Generator instead of the
    # process-global np.random, so a run is replayable from its recorded seed.
    self.seed = self.args.seed
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
    if self.FAMILY is None or self.PROGRAM is None:
      raise RuntimeError(
        "the driver class must set FAMILY and PROGRAM (the scientific-record "
        "family and producer names written to the .facts.yaml sidecar)")

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
    # below rewrite self.bounds in place (the infinite-endpoint stretch and
    # the --boundary trim). Both supports go into the .facts.yaml record:
    # they answer different questions and differ whenever a mutation fires.
    self.bounds_requested = np.array(self.bounds, copy=True, dtype=self.dtype)
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
        "the resolved sampling bounds must remain finite and increasing "
        "after temperature stretching and the boundary margin")

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
    # a --chain 1 run writes only the parameter-side files. It gets its own
    # output stem so it can never overwrite (or resume from) the failure and
    # data-vector files of a full run that used the same names.
    if self.args.chain == 1:
      self.dvsf = f"{self.dvsf}_chain_only"
      self.paramsf = f"{self.paramsf}_chain_only"
      self.failf = f"{self.failf}_chain_only"

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

  def _facts_base_identity(self):
    """
    Name the frozen analytic base the dumps sit on top of, or "n/a".

    Most families emulate the computed quantity directly and have no base.
    The mps driver overrides this: its dumps can carry the syren analytic
    base the emulator corrects, and the record must name that base.
    """
    return fixed_facts.NOT_APPLICABLE

  #-----------------------------------------------------------------------------
  # the scientific record (<paramfile>.facts.yaml)
  #-----------------------------------------------------------------------------
  def _resolved_constants(self):
    """
    Read every named constant the resolved model pins, as a plain mapping.

    Delegates to fixed_facts.resolved_constants (the one owner of how
    parameter constants and theory-component defaults combine).
    """
    return fixed_facts.resolved_constants(model=self.model)

  def _resolve_fixed_facts(self):
    """
    Read this run's scientific facts off the resolved Cobaya model.

    The YAML is the request and the model is the fact, so every value here
    is read from self.model, never from the parsed YAML mapping: a
    coordinate the YAML never mentioned has been given its value by the
    time the model exists, and it is that value the data vectors were
    computed at.

    A cosmology coordinate this run sampled is dropped from the fixed
    roster (it is described by the sampled-region half of the record
    instead); a coordinate the model cannot resolve is reported "n/a",
    never omitted.

    Returns:
      a mapping with the fields fixed_facts.build_sidecar takes beside the
      parameter supports: generator, family, cosmology_fixed,
      neutrino_convention, flat_only, dark_energy_law, dark_energy_inputs,
      cl_units, base_identity.

    Raises:
      ValueError when a background (grid-family) run is not flat, or the
      dark-energy coordinates are internally inconsistent.
    """
    sampled = set(self.sampled_params)
    pinned = self._resolved_constants()
    (de_fixed,
     de_varying,
     de_law,
     de_inputs) = dark_energy_facts(self.model.parameterization, pinned)

    cosmology_fixed = {}
    for key in fixed_facts.COSMOLOGY_FIXED_KEYS:
      if key in ("w", "wa"):
        if key in de_varying:
          continue                               # described as a sampled input
        cosmology_fixed[key] = de_fixed.get(key, fixed_facts.NOT_APPLICABLE)
        continue
      if key in sampled:
        continue                                 # described as a sampled input
      if key in pinned:
        cosmology_fixed[key] = pinned[key]
      else:
        cosmology_fixed[key] = fixed_facts.NOT_APPLICABLE

    # curvature: the run is flat unless omk is sampled or pinned away from
    # zero. A model with no omk at all is flat; an absent coordinate is the
    # standard model's value for it, not a missing fact.
    flat_only = True
    if "omk" in sampled:
      flat_only = False
    elif "omk" in pinned:
      flat_only = (pinned["omk"] == 0.0)
    if self.FAMILY == "grid" and not flat_only:
      raise ValueError(
        "background generation is flat-only (D_M = the comoving radial "
        "distance only when omk = 0); fix omk to zero")

    # how the total neutrino mass is split into states, in the model's own
    # word for it.
    neutrino_convention = fixed_facts.NOT_APPLICABLE
    if NEUTRINO_CONVENTION_KEY in pinned:
      neutrino_convention = str(pinned[NEUTRINO_CONVENTION_KEY])

    cl_units = fixed_facts.NOT_APPLICABLE
    if self.FAMILY == "cmb":
      cl_units = CMB_CL_UNITS

    return {"generator":           self.PROGRAM,
            "family":              self.FAMILY,
            "cosmology_fixed":     cosmology_fixed,
            "neutrino_convention": neutrino_convention,
            "flat_only":           flat_only,
            "dark_energy_law":     de_law,
            "dark_energy_inputs":  de_inputs,
            "cl_units":            cl_units,
            "base_identity":       self._facts_base_identity()}

  def _write_facts_sidecar(self, names):
    """
    Write the dataset's scientific record beside the chain it describes.

    The record is the pair of blocks emulator/fixed_facts.py defines: the
    cosmology this run held fixed, and the parameter region it sampled
    (both the support the prior requested and the support the sampler
    actually drew from). It is keyed by the digest of the chain file as
    that file stands on disk, so the chain must already be written when
    this is called; training staging pairs the two files and refuses a
    chain whose record is missing or stale.

    Arguments:
      names = the sampled parameters in train_args.ord order — the same
              order the chain columns, .ranges rows, and both bounds
              arrays use.

    Returns:
      None. The record is written to <paramsf>.facts.yaml.

    Raises:
      ValueError from fixed_facts.build_sidecar when the facts would form
      an invalid record (for example a coordinate both sampled and fixed);
      nothing is written in that case.
    """
    facts = self._resolve_fixed_facts()
    requested = {}
    resolved = {}
    for index, name in enumerate(names):
      requested[name] = (self.bounds_requested[index, 0],
                         self.bounds_requested[index, 1])
      resolved[name] = (self.bounds[index, 0],
                        self.bounds[index, 1])
    text = fixed_facts.build_sidecar(
      dataset_id=fixed_facts.chain_digest(chain_path=f"{self.paramsf}.1.txt"),
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
    with open(f"{self.paramsf}{fixed_facts.SIDECAR_SUFFIX}", "w") as f:
      f.write(text)

  #-----------------------------------------------------------------------------
  # reorder indexes to match YAML original ordering
  #-----------------------------------------------------------------------------
  # Two parameter orders coexist: cobaya's own sampled order (self.names,
  # the YAML's order) and the canonical column order the dumps use
  # (self.sampled_params = train_args.ord). The two helpers below build
  # the index arrays that translate a row between them; both orders hold
  # the same names, so each map is a permutation.

  def reorder_idx_from_yaml_to_ord(self):
    """
    Index array taking a cobaya-ordered row into train_args.ord order.

    Returns:
      an int array `idx` with row_ord = row_yaml[idx]; equivalently,
      idx[k] is where ord's k-th parameter sits in cobaya's order.
    """
    pidx = {p : i for i, p in enumerate(self.names)}
    return np.array([pidx[p] for p in self.sampled_params],
                    copy=True,
                    dtype=int)

  def reorder_idx_from_ord_to_yaml(self):
    """
    Index array taking a train_args.ord row into cobaya's order.

    The inverse permutation of reorder_idx_from_yaml_to_ord; used
    whenever a stored sample must be handed back to cobaya (prior
    evaluation, the physics call).

    Returns:
      an int array `idx` with row_yaml = row_ord[idx].
    """
    pidx = {p : i for i, p in enumerate(self.sampled_params)}
    return np.array([pidx[p] for p in self.names],
                    copy=True,
                    dtype=int)

  #-----------------------------------------------------------------------------
  # data-vector store (default: the single 2D array at {dvsf}.npy)
  #-----------------------------------------------------------------------------
  # The seven _dv_* hooks below are the STORE: how computed payload rows
  # live on disk and in memory. This default implements the simplest
  # layout — one 2D float32 array, one row per sample, saved whole at
  # {dvsf}.npy — which is exactly what the lensing driver needs (its
  # payload is one flat vector). A driver whose payload is richer (four
  # CMB spectra, two background quantities, the mps quantity set)
  # overrides all seven hooks together; the rest of the core never
  # looks inside the store, it only calls the hooks.
  #
  # Every implementation is RAM-aware in the same way: if the full store
  # fits comfortably in memory (< 75% of what is available) it lives in
  # RAM and is saved atomically at checkpoints; otherwise it is a memmap
  # (a NumPy array backed by the file on disk, read and written in
  # slices) and checkpoints just flush it.

  def _load_axis_checkpoint(self, path, expected, label):
    """
    Load one axis sidecar (a saved coordinate grid) and require an exact
    match with the coordinates train_args configures. Drivers that write
    axis sidecars call this on resume: a checkpoint written for one grid
    must never be continued on another (the row payloads would silently
    mean different coordinates).

    Arguments:
      path     = the sidecar .npy file (e.g. {dvsf}_z.npy).
      expected = the 1D coordinate array train_args resolves to.
      label    = short axis name for the error message (e.g. "mps redshift").

    Returns:
      the loaded axis array.

    Raises:
      ValueError when the saved axis and the configured axis differ.
    """
    observed = np.load(path, allow_pickle=False)
    expected = np.asarray(expected)
    if observed.ndim != 1:
      raise ValueError(
        f"checkpoint axis {label} must be 1D, got {observed.shape}")
    if observed.shape != expected.shape:
      raise ValueError(
        f"checkpoint axis {label} has shape {observed.shape}, expected "
        f"{expected.shape} from train_args; the chk and the YAML disagree")
    if not np.array_equal(observed, expected):
      raise ValueError(
        f"checkpoint axis {label} disagrees with the coordinates train_args "
        "configures; use a checkpoint written for this grid")
    return observed

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
    """
    Grow the store by nparams zero rows (append mode; RAM-aware).

    The old rows are preserved byte-for-byte; the new rows start zeroed
    with their failure flags set, so the physics farm treats them
    exactly like a resume's unfinished rows. When the grown store no
    longer fits in RAM, it is rebuilt as an on-disk memmap in bounded
    row chunks (never materializing old + new in memory at once).

    Arguments:
      nparams = the number of appended rows (the append run's --nparams).
    """
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
    """
    Load a saved run's files when --loadchk 1, refusing a partial set.

    The checkpoint is nothing more than the run's own output files read
    back: the chain (.1.txt) supplies the parameter rows, the failure
    file says which rows still need computing, and the family stores
    hold the finished payload rows. What counts as "the complete set"
    depends on the mode:

      full run        chain + .covmat + .ranges + failfile
                      + every family store and axis sidecar
                      (_dv_chk_files)
      chain-only run  chain + .covmat + .ranges only

    A missing file refuses (naming every missing file) rather than
    quietly starting fresh: a mistyped output name would otherwise
    regenerate — and overwrite — instead of resuming. The family
    loader (_dv_load_chk) additionally checks the loaded stores against
    the YAML: row counts against the chain, column counts and axis
    sidecars against the configured grids, so a checkpoint from one
    configuration can never continue under another.

    Returns:
      True when a checkpoint was loaded (self.samples / self.failed /
      the family stores are populated); False when --loadchk is 0 and
      the run is fresh.

    Raises:
      FileNotFoundError naming the missing checkpoint files, or
      ValueError when a loaded file disagrees with the chain or the
      YAML grids.
    """
    rtnvar = False
    if self.loadchk == 1:
      # a chain-only run owns only the parameter-side files; a full run also
      # owns the failure mask and the family data-vector stores.
      if self.args.chain == 1:
        required = [f"{self.paramsf}.1.txt",
                    f"{self.paramsf}.covmat",
                    f"{self.paramsf}.ranges"]
      else:
        required = self._dv_chk_files() + [f"{self.failf}.txt",
                                           f"{self.paramsf}.covmat",
                                           f"{self.paramsf}.ranges",
                                           f"{self.paramsf}.1.txt"]
      missing = [x for x in required if not os.path.isfile(x)]
      if missing:
        # a requested load must never silently become fresh generation: a
        # mistyped path would regenerate (and overwrite) instead of resuming.
        raise FileNotFoundError(
          "--loadchk 1 was requested but these checkpoint files are "
          "missing: " + repr(missing) + ". Fix the output names, or run "
          "fresh with --loadchk 0 at a new path.")
      # load sample file begins ----------------------------------------------
      # row 0/1 rows are weights, lnp. Last row is chi2
      self.samples = np.atleast_2d(np.loadtxt(f"{self.paramsf}.1.txt",
                                              dtype=self.dtype))[:,2:-1]
      if self.samples.ndim != 2:
        raise ValueError(f"samples must be 2D, got {self.samples.shape}")
      # load sample file ends ------------------------------------------------

      if self.args.chain == 0:
        # load fail file begins ----------------------------------------------
        self.failed = np.atleast_1d(np.loadtxt(f"{self.failf}.txt",
                                               dtype=np.uint8))
        self.failed = np.asarray(self.failed).astype(bool)
        if self.failed.ndim != 1:
          raise ValueError(f"failed must be 1D, got {self.failed.shape}")

        if self.samples.shape[0] != self.failed.shape[0]:
          raise ValueError(f"Incompatible samples/failed chk files")
        # load fail file ends ------------------------------------------------

        # load datavectors (store-specific; row-count checks live inside) ----
        self._dv_load_chk()

      print("Loaded models from chk")
      if self.append == 0:
        self.loadedsamples = True
        self.loadedfromchk = True
      rtnvar = True
    return rtnvar

  def __save_chk(self):
    """
    Flush the failure flags and the family stores to disk, atomically.

    Called every --freqchk computed rows and once at the end. The
    failure file and each in-RAM store are written to a temporary name
    first and renamed into place (os.replace is atomic on one
    filesystem), so a crash mid-save leaves the previous complete
    checkpoint, never a truncated file; a memory-mapped store is
    flushed in place instead (its bytes already live in the file).
    The flags and the stores are saved together on purpose: they
    describe the same rows and must never drift apart on disk.
    """
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
    """
    0 inside the RESOLVED sampling bounds, -inf outside.

    The resolved bounds differ from cobaya's own prior wherever setup
    stretched an infinite endpoint or trimmed the --boundary margin, so
    cobaya's prior alone cannot enforce them; this extra top-hat does.

    Arguments:
      x = one parameter row in train_args.ord order.

    Returns:
      0.0 when every coordinate sits inside [bounds_lo, bounds_hi],
      else -inf (the row is rejected).
    """
    # because we shrink the prior, we need to do this check
    if np.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1])):
      return 0.0
    else:
      return -np.inf

  def __param_logpost(self,x):
    """
    The tempered Gaussian log-posterior emcee samples in --unif 0 mode.

        log p(x) = [ -(x-fid)^T C^-1 (x-fid) / 2  +  log prior(x) ] / T

    (legend: fid = train_args.fiducial, C = the correlation-reduced
    parameter covariance from setup, T = --temp. Dividing by T flattens
    the distribution, so the training cloud extends well past the
    posterior it was built from — an emulator must stay accurate where
    a future chain explores, not only where it converges.)

    The row is rejected (-inf) when cobaya's own prior rejects it or
    when it leaves the resolved bounds (__param_logprior), which
    matters exactly when --boundary < 1 trimmed the support.

    Arguments:
      x = one parameter row in train_args.ord order.

    Returns:
      the tempered log-posterior (float), or -inf for a rejected row.
    """
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
    """
    Produce the parameter table and its sidecar files (rank 0 only).

    Despite the name, this method covers BOTH sampling modes; the flow
    for a fresh run is:

        --unif 1                          --unif 0
        uniform draws from the           emcee on the tempered Gaussian
        nextafter-resolved bounds        posterior (covmat / T), then
           │                             unique rows, seeded thinning
           ▼                                │
        xf (nparams, ndim)  ◄───────────────┘
           │  w = 1, chi2* = -2 lnp
           ▼
        .1.txt  .paramnames  .ranges  .covmat  .facts.yaml
        (chain)  (labels)    (bounds) (getdist  (scientific
                                       cov)      record)

    (legend: xf = the sampled parameter rows in train_args.ord order;
    lnp = the log-probability column (1 as a placeholder in uniform
    mode); chi2* = the derived getdist column -2 lnp; T = --temp.)

    A resume run (--loadchk 1, --append 0) only reads the files back.
    An append run draws nparams NEW rows — from a stream derived from
    the seed plus the existing row count, so the fresh rows are never
    repeated — extends the chain and the family stores, refreshes the
    .covmat, and rewrites the .facts.yaml (the chain's bytes changed,
    so the digest naming it changed too).

    Side effects: writes the five parameter-side files; sets
    self.samples (the float32 rows the physics farm consumes) and, for
    fresh full runs, leaves self.failed / the stores to
    __generate_datavectors.
    """
    # a load failure propagates: a requested resume or append must never
    # silently fall back to fresh generation over the same output paths.
    loadedfromchk = self.__load_chk()

    if (loadedfromchk == False) or (loadedfromchk == True and self.append == 1):
      ndim     = len(self.sampled_params)
      names    = list(self.sampled_params)

      if loadedfromchk == True and self.append == 1:
        # a fresh Generator from the bare seed would repeat the original
        # run's draws exactly (same seed, same stream) and duplicate every
        # existing row. Deriving the append stream from the seed plus the
        # existing row count keeps the run replayable from recorded inputs
        # while making the appended rows distinct.
        self.rng = np.random.default_rng((self.seed, self.samples.shape[0]))

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
        # give emcee's own moves a random state derived from the seeded
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
        # move each endpoint one representable float inward, so a draw can
        # never sit exactly on a hard prior edge (where logprior is -inf).
        # nextafter works in float-representation steps, so the movement is
        # correct for endpoints of any sign or magnitude, including zero
        # (a relative-factor shrink is wrong at and near zero). The resolved
        # bounds replace self.bounds so the .ranges file and the .facts.yaml
        # record describe the support the sampler actually drew from.
        self.bounds[:, 0] = np.nextafter(self.bounds[:, 0], self.bounds[:, 1])
        self.bounds[:, 1] = np.nextafter(self.bounds[:, 1], self.bounds[:, 0])
        if not (self.bounds[:, 0] < self.bounds[:, 1]).all():
          raise ValueError(
            "a requested uniform-sampling interval is too narrow to contain "
            "an interior point after moving both endpoints inward")
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
        # bounds go through the scientific record's own formatter: a rounded
        # view (the old %.5e) could print a zero-width interval for a pair of
        # nearby float32 endpoints the chain beside it keeps apart.
        bds = self.bounds.copy()
        rows = [(str(n), fixed_facts.format_value(float(l)),
                         fixed_facts.format_value(float(h)))
                for n, l, h in zip(names, bds[:, 0], bds[:, 1])]
        with open(f"{self.paramsf}.ranges", "w") as f:
          for name, low, high in rows:
            f.write(f"{name} {low} {high}\n")

        # save chain begins ----------------------------------------------------
        # the seed rides in the header, so the parameter table can be replayed
        # from the recorded inputs alone.
        fname = f"{self.paramsf}.1.txt";
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
        latex = []
        for x in names:
          entry = param_info[x]
          if isinstance(entry, dict) and 'latex' in entry:
            latex.append(entry['latex'])
          else:
            latex.append(x)     # a parameter without a latex label keeps its name
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
        # beside the chain whose digest names it. Written last: the digest is
        # taken from the finished chain file.
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

        # a chain-only append owns only the parameter-side files; the failure
        # mask and the data-vector stores exist only for a full dataset.
        if self.args.chain == 0:
          # append fail file begins --------------------------------------------
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
          # append fail file ends ----------------------------------------------

          # append dvs (with zeros; store-specific, row-count check inside) ----
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
        # appending changed the chain's bytes, so the digest in the old record
        # now names a file that no longer exists anywhere. Rewrite the record
        # against the appended chain; the facts themselves are re-read from
        # the model rather than assumed unchanged.
        self._write_facts_sidecar(names=names)

        # set self.loadedfromchk -----------------------------------------------
        self.loadedfromchk = True
    # set self.loadedsamples ---------------------------------------------------
    self.loadedsamples = True

  #-----------------------------------------------------------------------------
  # datavectors
  #-----------------------------------------------------------------------------
  def __generate_datavectors(self):
    """
    Compute one data-vector payload per parameter row, farmed over MPI.

    Serial (one process): rank 0 walks the rows itself. Parallel: rank 0
    is the master and every other rank a worker in a task farm —

        rank 0 (master)                      rank w (worker)
        send (row j, params[j]) ──TTAG──►    compute payload
        recv ("ok"/"err", j, payload) ◄─RTAG─ send result
           │  bind: the reply's j must be the j assigned to rank w
           │  write payload at row j, or zero the row + flag it failed
           │  checkpoint every --freqchk completed rows
           ▼
        when no tasks remain: send STAG stop, await DTAG acknowledgments

    (legend: TTAG/RTAG/STAG/DTAG = the four MPI message tags: task,
    result, stop, done. params[j] = row j of self.samples. A payload is
    whatever the driver's _compute_dvs_from_sample returns — one flat
    vector, a (4, nell) spectrum block, a per-quantity dict.)

    Failure handling: a worker exception zeroes the row and sets its
    failure flag (the run continues; resume recomputes flagged rows); a
    worker silent for TASK_TIMEOUT seconds aborts the whole job after a
    final checkpoint, because its row can no longer be trusted to
    arrive. The first row is always computed on rank 0 before the farm
    starts: its payload sizes the on-disk stores (_dv_alloc).

    Side effects: fills the family stores and self.failed; saves
    checkpoints along the way and once at the end.
    """
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
            # bind the result to the row this worker was assigned: a result
            # for any other row would silently overwrite another sample.
            if src not in active or active[src][0] != idx:
              sys.stderr.write(f"[Rank 0] Worker {src} returned idx={idx} "
                               f"but was assigned "
                               f"{active.get(src, ('nothing',))[0]}\n")
              sys.stderr.flush()
              self.__save_chk() # save before crashing
              comm.Abort(1)
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
            # bind the result to the row this worker was assigned: a result
            # for any other row would silently overwrite another sample.
            if src not in active or active[src][0] != idx:
              sys.stderr.write(f"[Rank 0] Worker {src} returned idx={idx} "
                               f"but was assigned "
                               f"{active.get(src, ('nothing',))[0]}\n")
              sys.stderr.flush()
              self.__save_chk() # save before crashing
              comm.Abort(1)
            del active[src]
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
  # strict parsing: a misspelled flag is a usage error, never silently
  # ignored (the cli-strict gate enforces this for every entry point).
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
