import numpy as np
import math, os, sys, traceback
import psutil
from numpy.lib.format import open_memmap
from generator_core import (GeneratorCore, capture_native_output,
                            run_generator)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# The script below computes CMB spectra (C_ell) training dumps: per sample,
# one CAMB call through cobaya produces the TT, TE, EE and phi-phi (CMB
# lensing potential) spectra on the multipole range train_args.lrange.
#
#     mpirun -n 10 --report-bindings \
#       python external_modules/code/emulators/emultrfv2/compute_data_vectors/dataset_generator_cmb.py \
#         --root projects/example/  \
#         --fileroot emulators/cmb/ \
#         --nparams 10000 \
#         --yaml 'lcdm_cmb.yaml' \
#         --datavsfile 'lcdm_dvs_train' \
#         --paramfile 'lcdm_params_train' \
#         --failfile  'lcdm_params_failed_train' \
#         --chain 0 \
#         --unif 1 \
#         --temp 64 \
#         --maxcorr 0.15 \
#         --freqchk 2000 \
#         --loadchk 0 \
#         --append 0 \
#         --boundary 1.0
#
# All command-line flags are the shared generator flags (documented in
# dataset_generator_lensing.py and parsed in generator_core.py).
#
#- The training YAML must carry, inside train_args:
#      probe:  cmblensed        # or cmbunlensed (TT/TE/EE without lensing)
#      lrange: [2, 5000]        # multipole range, 2 <= lmin < lmax
#  plus the shared keys (ord; fiducial/params_covmat_file when --unif 0).
#  The likelihood block may be the dummy `one` — this script adds the Cl
#  requirements to the cobaya model itself.
#
#- The output files are
#
#      # Distribution of training points ready to be plotted by GetDist
#      lcdm_params_train_cmblensed_unifs.1.txt
#      lcdm_params_train_cmblensed_unifs.covmat
#      lcdm_params_train_cmblensed_unifs.paramnames
#      lcdm_params_train_cmblensed_unifs.ranges
#
#      # Corresponding data vectors: FOUR per-spectrum 2D files, one row per
#      # sample, one column per multipole (raw C_ell, no l(l+1)/2pi factor,
#      # muK^2 for TT/TE/EE, dimensionless C_L^{phiphi} for pp — the same
#      # units/convention as compute_cmb_covariance.py, so dumps and
#      # covariance always match):
#      lcdm_dvs_train_cmblensed_unifs_tt.npy
#      lcdm_dvs_train_cmblensed_unifs_te.npy
#      lcdm_dvs_train_cmblensed_unifs_ee.npy
#      lcdm_dvs_train_cmblensed_unifs_pp.npy
#      # Exact int64 multipoles shared by the four spectrum files:
#      lcdm_dvs_train_cmblensed_unifs_ell.npy
#      # Training parameters in which the computation failed (or not computed)
#      lcdm_params_failed_train_cmblensed_unifs.txt
#
#  A training run then points data.train_dv / data.val_dv at ONE of the four
#  per-spectrum files (one emulator per spectrum, data.cmb.spectrum names it).
#
# Deviations from the legacy emultraining/dataset_generator_cmb.py, ruled in
# the shared-generator design (ai/notes/families-scalar-cmb.md):
#   1. Four per-spectrum 2D .npy files replace the legacy 3D (N, ell, 5)
#      array — the training stack stages 2D dv files.
#   2. phi-phi is FILLED from get_Cl (the legacy file zeroed that column and
#      never produced phiphi training data).
#   3. The legacy "EXTRA" derived-parameter column dies — derived scalars are
#      the scalar-emulator unit's job (scalar_train_emulator on the same
#      params dump).
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Class Definition
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
SPECTRA = ("tt", "te", "ee", "pp")

class dataset(GeneratorCore):
  """
  CMB spectra generator: one CAMB-through-cobaya call per sample yields
  all four spectra at once, stored as four per-spectrum 2D arrays
  ({dvsf}_tt.npy ... {dvsf}_pp.npy) — the core's dv-store hooks are all
  overridden together. The companion {dvsf}_ell.npy file stores the exact
  int64 multipole axis. The per-sample payload is a (4, nell) float32 array
  in SPECTRA row order.
  """
  VALID_PROBES = ("cmblensed", "cmbunlensed")
  EXTRA_TRAIN_KEYS = ("lrange",)

  def _read_train_args(self, train_args):
    """
    Read train_args.lrange and register the Cl requirements.

    lrange = [lmin, lmax], the multipole range every output row spans.
    lmin >= 2 is enforced: l = 0, 1 are not CMB observables (CAMB returns
    zeros there), the covariance file from compute_cmb_covariance.py
    starts at l = 2, and all-zero columns would poison the training
    whitening.

    The Cl requirements are added HERE, to the model itself, so the
    training YAML's likelihood block can be the dummy `one` — the script
    never depends on a likelihood having requested the spectra.
    """
    lrange = np.array(train_args["lrange"], dtype=int)
    if lrange.shape != (2,):
      raise ValueError(f"train_args.lrange must be [lmin, lmax], "
                       f"got {train_args['lrange']!r}")
    if lrange[0] < 2 or lrange[1] <= lrange[0]:
      raise ValueError(f"train_args.lrange must satisfy 2 <= lmin < lmax, "
                       f"got [{lrange[0]}, {lrange[1]}]")
    self.lrange = lrange

    lmax = int(self.lrange[1])
    if self.probe == "cmblensed":
      self.model.add_requirements(
        {"Cl": {"tt": lmax,
                "te": lmax,
                "ee": lmax,
                "pp": lmax}})
    else:
      # unlensed TT/TE/EE; phi-phi is not a lensed/unlensed quantity and
      # rides the plain Cl requirement on either probe.
      self.model.add_requirements(
        {"unlensed_Cl": {"tt": lmax,
                         "te": lmax,
                         "ee": lmax},
         "Cl": {"pp": lmax}})

  #-----------------------------------------------------------------------------
  # data-vector store: four per-spectrum 2D arrays -> {dvsf}_<spec>.npy
  #-----------------------------------------------------------------------------
  def _multipole_axis(self):
    """Return every configured CMB multipole as a 1D int64 array."""
    lmin = int(self.lrange[0])
    lmax = int(self.lrange[1])
    return np.arange(lmin, lmax + 1, dtype=np.int64)

  def _load_multipole_axis(self):
    """Load the CMB axis sidecar and require its exact saved coordinates."""
    axis_path = f"{self.dvsf}_ell.npy"
    observed = np.load(axis_path, allow_pickle=False)
    expected_dtype = np.dtype(np.int64)
    if observed.dtype != expected_dtype:
      raise ValueError(
        f"checkpoint CMB multipole axis has dtype {observed.dtype}, "
        f"expected {expected_dtype}; use a checkpoint written for this "
        "CMB train_args.lrange")
    if observed.ndim != 1:
      raise ValueError(
        f"checkpoint CMB multipole axis must be 1D, got {observed.shape}; "
        "use a checkpoint written for this CMB train_args.lrange")

    expected = self._multipole_axis()
    if observed.shape != expected.shape:
      raise ValueError(
        f"checkpoint CMB multipole axis has shape {observed.shape}, "
        f"expected {expected.shape} from train_args.lrange; use a matching "
        "checkpoint")
    if not np.array_equal(observed, expected):
      raise ValueError(
        "checkpoint CMB multipole axis must contain every integer from "
        f"{expected[0]} through {expected[-1]} in increasing order; use a "
        "checkpoint written for this CMB train_args.lrange")
    return observed

  def _dv_chk_files(self):
    """Files the checkpoint loader must find before trusting a chk."""
    files = []
    for spec in SPECTRA:
      files.append(f"{self.dvsf}_{spec}.npy")
    files.append(f"{self.dvsf}_ell.npy")
    return files

  def _dv_load_chk(self):
    """Load all four per-spectrum stores (RAM-aware, one shared policy)."""
    self._load_multipole_axis()

    RAMneed = self.samples.nbytes + self.failed.nbytes
    for spec in SPECTRA:
      arr = np.load(f"{self.dvsf}_{spec}.npy",
                    mmap_mode = "r",
                    allow_pickle = False)
      RAMneed += arr.nbytes
      del arr
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      self.dvs_is_memmap = False
    else:
      print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
            f"There is {RAMavail/1e9:.2f} GB of RAM available. "
            f"We will read dvs from HD (slow)")
      self.dvs_is_memmap = True

    self.datavectors = {}
    for spec in SPECTRA:
      if self.dvs_is_memmap:
        self.datavectors[spec] = np.load(f"{self.dvsf}_{spec}.npy",
                                         mmap_mode = "r+",
                                         allow_pickle = False)
      else:
        self.datavectors[spec] = np.load(f"{self.dvsf}_{spec}.npy",
                                         allow_pickle = False)
      if self.datavectors[spec].ndim != 2:
        raise ValueError(f"datavectors ({spec}) must be 2D, "
                         f"got {self.datavectors[spec].shape}")
      if self.datavectors[spec].shape[0] != self.samples.shape[0]:
        raise ValueError(f"Incompatible samples/datavector ({spec}) chk files")
      expected_width = int(self.lrange[1] - self.lrange[0] + 1)
      if self.datavectors[spec].shape[1] != expected_width:
        raise ValueError(
          f"chk datavectors ({spec}) have "
          f"{self.datavectors[spec].shape[1]} columns but train_args names "
          f"a {expected_width}-multipole range; the chk and the YAML disagree")

  def _dv_save(self):
    """Flush all four stores to disk (tmp file + atomic replace each)."""
    for spec in SPECTRA:
      if self.dvs_is_memmap == True:
        self.datavectors[spec].flush()  # checkpoint dv in-place
      else:
        # save (flush) dvs to tmp file (safer) ---------------------------------
        np.save(f"{self.dvsf}_{spec}.tmp.npy", self.datavectors[spec])
        # save data vector file (from tmp) -------------------------------------
        os.replace(f"{self.dvsf}_{spec}.tmp.npy", f"{self.dvsf}_{spec}.npy")

  def _dv_append(self, nparams):
    """Grow all four stores by nparams zero rows (append mode; RAM-aware)."""
    nrows = self.datavectors[SPECTRA[0]].shape[0]
    ncols = self.datavectors[SPECTRA[0]].shape[1]
    itemsize = self.datavectors[SPECTRA[0]].dtype.itemsize

    RAMneed = ( self.samples.nbytes +
                self.failed.nbytes +
                len(SPECTRA)*nrows*ncols*itemsize +
                len(SPECTRA)*(nrows + nparams)*ncols*itemsize)
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      for spec in SPECTRA:
        # setup new datavector numpy array -----------------------------------
        self.datavectors[spec] = np.vstack(
          (self.datavectors[spec],
           np.zeros((nparams, ncols), dtype=self.dtype)))
        # save (flush) dvs to tmp file (safer) -------------------------------
        np.save(f"{self.dvsf}_{spec}.tmp.npy", self.datavectors[spec])
        # save data vector file (from tmp) -----------------------------------
        os.replace(f"{self.dvsf}_{spec}.tmp.npy", f"{self.dvsf}_{spec}.npy")
      self.dvs_is_memmap = False
    else:
      print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
            f"There is {RAMavail/1e9:.2f} GB of RAM available. "
            f"We will read dvs from HD (slow)")
      for spec in SPECTRA:
        # setup new datavector numpy array -----------------------------------
        datavectors = open_memmap(f"{self.dvsf}_{spec}.tmp.npy",
                                  mode = "w+",
                                  shape = (nrows + nparams, ncols),
                                  dtype = self.datavectors[spec].dtype)
        for s in range(0, nrows, 2500): # read dvs in chunks: avoid RAM spikes
          e = min(nrows, s + 2500)
          datavectors[s:e] = self.datavectors[spec][s:e]
        for s in range(nrows, nrows + nparams, 2500):
          e = min(nrows + nparams, s + 2500)
          datavectors[s:e] = 0
        # save (flush) data vector (in-place) --------------------------------
        datavectors.flush()
        del datavectors
        # save data vector file (from tmp) -----------------------------------
        os.replace(f"{self.dvsf}_{spec}.tmp.npy", f"{self.dvsf}_{spec}.npy")
        # finally, load the new dv numpy array from file ---------------------
        self.datavectors[spec] = np.load(f"{self.dvsf}_{spec}.npy",
                                         mmap_mode = "r+",
                                         allow_pickle = False)
      self.dvs_is_memmap = True

    # check final dimensions -----------------------------------------------
    for spec in SPECTRA:
      if self.datavectors[spec].shape[0] != self.samples.shape[0]:
        raise ValueError(f"Incompatible samples/datavector ({spec}) chk files")

  def _dv_alloc(self, nrows, first_dvs):
    """
    Allocate all four stores for nrows samples, sized from the first
    computed (4, nell) payload (RAM-aware: in-RAM zeros or on-disk
    memmaps, one shared policy). Save the exact int64 multipole axis beside
    those stores.
    """
    multipoles = self._multipole_axis()
    expected_shape = (len(SPECTRA), multipoles.shape[0])
    if first_dvs.shape != expected_shape:
      raise ValueError(f"first computed payload has shape {first_dvs.shape}, "
                       f"expected {expected_shape} "
                       f"from train_args.lrange {self.lrange.tolist()}")
    ncols = first_dvs.shape[1]
    RAMneed = ( self.samples.nbytes +
                self.failed.nbytes +
                len(SPECTRA)*nrows*ncols*np.dtype(self.dtype).itemsize)
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      self.dvs_is_memmap = False
    else:
      print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
            f"There is {RAMavail/1e9:.2f} GB of RAM available. "
            f"We will read dvs from HD (slow)")
      self.dvs_is_memmap = True

    self.datavectors = {}
    for spec in SPECTRA:
      if self.dvs_is_memmap:
        self.datavectors[spec] = open_memmap(f"{self.dvsf}_{spec}.npy",
                                             mode = "w+",
                                             shape = (nrows, ncols),
                                             dtype = self.dtype)
        self.datavectors[spec][:] = 0.0
        self.datavectors[spec].flush()
      else:
        self.datavectors[spec] = np.zeros((nrows, ncols), dtype=self.dtype)

    np.save(f"{self.dvsf}_ell.npy", multipoles)

  def _dv_write(self, i, dvs):
    """Write one (4, nell) payload at row i of each per-spectrum store."""
    for j, spec in enumerate(SPECTRA):
      self.datavectors[spec][i] = dvs[j]

  def _dv_zero(self, i):
    """Zero row i of each per-spectrum store (a failed sample)."""
    for spec in SPECTRA:
      self.datavectors[spec][i, :] = 0.0

  #-----------------------------------------------------------------------------
  # per-sample computation
  #-----------------------------------------------------------------------------
  def _compute_dvs_from_sample(self, sample):
    # Define fortran errors we want to capture ---------------------------------
    camb_error_keywords = {"ERROR", "error", "Did not converge"}

    # Compute data vector (within using cobaya API) ----------------------------
    idx = self.reorder_idx_from_ord_to_yaml()
    param = dict(self.model.parameterization.to_input(
        sampled_params_values=dict(zip(self.names, sample[idx])))
    )
    self.model.provider.set_current_input_params(param)

    # Check prior before attempting computation --------------------------------
    if math.isinf(self.model.prior.logp(sample[idx])):
      raise RuntimeError(f"Prior is -inf (this should not happen). "
                         f"Values: {dict(zip(self.sampled_params, sample))}")

    captured = 0 # variable that will hold terminal output
    with capture_native_output() as tmp:
      for (x, _), z in zip(self.model._component_order.items(),
                           self.model._params_of_dependencies):
        x.check_cache_and_compute(
            params_values_dict = dict({p: param[p] for p in x.input_params}),
            want_derived = self.derived,
            dependency_params = list(param.keys()),
            cached = True
        )
      tmp.seek(0)
      captured = tmp.read() # copy terminal output -----------------------------

    # check for CAMB errors in the terminal output -----------------------------
    if any(kw in captured for kw in camb_error_keywords):
      raise RuntimeError(f"CAMB Fortran error: {captured.strip()}")

    # get results from Theory block (already computed) -------------------------
    nell = (self.lrange[1] - self.lrange[0]) + 1
    out  = np.zeros((len(SPECTRA), nell), dtype=self.dtype)
    lsel = slice(self.lrange[0], self.lrange[1] + 1)
    done = False
    for (x, _), z in zip(self.model._component_order.items(),
                         self.model._params_of_dependencies):
      if not (hasattr(x, 'get_Cl') and callable(getattr(x, 'get_Cl'))):
        continue
      # raw C_ell (ell_factor=False: no l(l+1)/2pi), muK^2 — the same call
      # compute_cmb_covariance.py makes, so dumps and covariance file share
      # units and conventions by construction.
      lensed = x.get_Cl(ell_factor=False, units="muK2")
      if self.probe == "cmblensed":
        cmb = lensed
      else:
        cmb = x.get_unlensed_Cl(ell_factor=False, units="muK2")
      out[0] = cmb["tt"][lsel]
      out[1] = cmb["te"][lsel]
      out[2] = cmb["ee"][lsel]
      out[3] = lensed["pp"][lsel] # phi-phi rides get_Cl on either probe
      done = True
    if not done:
      raise RuntimeError("no theory component in the YAML provides get_Cl; "
                         "the theory block must contain a Boltzmann code "
                         "(e.g. camb)")
    return out

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  run_generator(dataset, prog='dataset_generator_cmb')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
