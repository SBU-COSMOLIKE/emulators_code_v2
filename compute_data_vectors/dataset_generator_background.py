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
# The script below computes BACKGROUND training dumps for the BAOSN emulators:
# per sample, one background-only CAMB evaluation (cheap — no perturbations)
# yields BOTH targets of the two-regime design:
#   - H(z) on the SN-range grid  (train_args.z_sn),   km/s/Mpc
#   - the comoving distance D_M(z) on the recombination window
#     (train_args.z_rec), Mpc (flat: D_M = chi)
#
#     mpirun -n 10 --report-bindings \
#       python external_modules/code/emulators/emultrfv2/compute_data_vectors/dataset_generator_background.py \
#         --root projects/example/  \
#         --fileroot emulators/baosn/ \
#         --nparams 10000 \
#         --yaml 'lcdm_background.yaml' \
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
#      probe: background
#      z_sn:  [0.0, 3.0, 600]       # [zmin, zmax, nz] linspace, SN range
#      z_rec: [1000.0, 1200.0, 40]  # [zmin, zmax, nz], recombination window
#  plus the shared keys (ord; fiducial/params_covmat_file when --unif 0).
#  The likelihood block may be the dummy `one` — this script adds the
#  background requirements to the cobaya model itself.
#
#- The output files are
#
#      # Distribution of training points ready to be plotted by GetDist
#      lcdm_params_train_background_unifs.1.txt
#      lcdm_params_train_background_unifs.covmat
#      lcdm_params_train_background_unifs.paramnames
#      lcdm_params_train_background_unifs.ranges
#
#      # Data vectors: TWO 2D files (one row per sample) + their grids,
#      # written once at the first allocation (the training path reads the
#      # grid from the FILE — resolved values, never re-declared):
#      lcdm_dvs_train_background_unifs_h.npy      H(z_sn),  (N, nz)
#      lcdm_dvs_train_background_unifs_h_z.npy    the z_sn grid, (nz,)
#      lcdm_dvs_train_background_unifs_dm.npy     D_M(z_rec), (N, nz2)
#      lcdm_dvs_train_background_unifs_dm_z.npy   the z_rec grid, (nz2,)
#      # Training parameters in which the computation failed
#      lcdm_params_failed_train_background_unifs.txt
#
#  A training run points data.train_dv at ONE of the two dv files and
#  data.grid.z_file at its _z.npy sidecar (one emulator per quantity).
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Class Definition
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
QUANTITIES = ("h", "dm")

class dataset(GeneratorCore):
  """
  Background generator: one background-only CAMB evaluation per sample
  yields H on the SN grid and D_M on the recombination window at once
  (the one-pass rule). The store is two per-quantity 2D
  arrays ({dvsf}_h.npy / {dvsf}_dm.npy) with their grids beside them;
  the per-sample payload is a dict {"h": (nz,), "dm": (nz2,)} float32.
  """
  VALID_PROBES = ("background",)
  EXTRA_TRAIN_KEYS = ("z_sn", "z_rec")
  FAMILY = "grid"                            # scientific-record family name
  PROGRAM = "dataset_generator_background"   # producer name in the record

  def _read_train_args(self, train_args):
    """Read the two grids and register the background requirements.

    z_sn / z_rec = [zmin, zmax, nz] linspace specs. The SN grid starts
    exactly at zero, because the distance integral starts there and the
    adapter serves H(z) down to z = 0. The recombination grid starts
    above zero, and the two windows must not overlap (the desert between
    them is the adapter's loud-error region).
    """
    grids = {}
    for key in ("z_sn", "z_rec"):
      spec = train_args[key]
      if (not isinstance(spec, (list, tuple))) or len(spec) != 3:
        raise ValueError(f"train_args.{key} must be [zmin, zmax, nz], "
                         f"got {spec!r}")
      zmin, zmax, nz = float(spec[0]), float(spec[1]), int(spec[2])
      if key == "z_sn":
        limits_ok = (zmin == 0.0 < zmax)
        rule = "zmin = 0 < zmax"
      else:
        limits_ok = (0.0 < zmin < zmax)
        rule = "0 < zmin < zmax"
      if not limits_ok or nz < 8:
        raise ValueError(f"train_args.{key} needs {rule} and "
                         f"nz >= 8, got [{zmin}, {zmax}, {nz}]")
      grids[key] = np.linspace(zmin, zmax, nz)
    if grids["z_sn"][-1] >= grids["z_rec"][0]:
      raise ValueError(
        f"train_args.z_sn must end below train_args.z_rec (the desert "
        f"between the two windows is never emulated); got z_sn max "
        f"{grids['z_sn'][-1]} >= z_rec min {grids['z_rec'][0]}")
    self.z_sn  = grids["z_sn"]
    self.z_rec = grids["z_rec"]

    # explicit background requirements on the model itself (the training
    # YAML may carry only the dummy `one` likelihood). Hubble in
    # km/s/Mpc; comoving_radial_distance is served in Mpc. Background
    # products ONLY — no Cl requirement, so CAMB never computes
    # perturbations and one evaluation per sample stays cheap. With this
    # background-only requirement set, the hand-rolled
    # check_cache_and_compute(cached=True) component loop the other
    # generators use serves the SAME background for every sample (stale
    # cache); that is why _compute_dvs_from_sample evaluates through the
    # standard model.logposterior(point, cached=False) lifecycle instead.
    # The bsn-smoke gate's dump-variance tripwire guards this boundary.
    self.model.add_requirements(
      {"Hubble": {"z": self.z_sn, "units": "km/s/Mpc"},
       "comoving_radial_distance": {"z": self.z_rec}})

  #-----------------------------------------------------------------------------
  # data-vector store: two per-quantity 2D arrays -> {dvsf}_<q>.npy
  #-----------------------------------------------------------------------------
  def _grid_of(self, quantity):
    """The stored grid for one quantity tag ("h" -> z_sn, "dm" -> z_rec)."""
    return self.z_sn if quantity == "h" else self.z_rec

  def _dv_chk_files(self):
    """Files the checkpoint loader must find before trusting a chk."""
    files = []
    for q in QUANTITIES:
      files.append(f"{self.dvsf}_{q}.npy")
      files.append(f"{self.dvsf}_{q}_z.npy")
    return files

  def _dv_load_chk(self):
    """Load both per-quantity stores (RAM-aware, one shared policy)."""
    # a checkpoint written for one pair of redshift grids must never be
    # continued on another: the columns would silently change meaning.
    for q in QUANTITIES:
      self._load_axis_checkpoint(path=f"{self.dvsf}_{q}_z.npy",
                                 expected=self._grid_of(q),
                                 label=f"{q} redshift")
    RAMneed = self.samples.nbytes + self.failed.nbytes
    for q in QUANTITIES:
      arr = np.load(f"{self.dvsf}_{q}.npy",
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
    for q in QUANTITIES:
      if self.dvs_is_memmap:
        self.datavectors[q] = np.load(f"{self.dvsf}_{q}.npy",
                                      mmap_mode = "r+",
                                      allow_pickle = False)
      else:
        self.datavectors[q] = np.load(f"{self.dvsf}_{q}.npy",
                                      allow_pickle = False)
      if self.datavectors[q].ndim != 2:
        raise ValueError(f"datavectors ({q}) must be 2D, "
                         f"got {self.datavectors[q].shape}")
      if self.datavectors[q].shape[0] != self.samples.shape[0]:
        raise ValueError(f"Incompatible samples/datavector ({q}) chk files")
      if self.datavectors[q].shape[1] != len(self._grid_of(q)):
        raise ValueError(
          f"chk datavectors ({q}) have {self.datavectors[q].shape[1]} "
          f"columns but train_args names a {len(self._grid_of(q))}-point "
          f"grid; the chk and the YAML disagree")

  def _dv_save(self):
    """Flush both stores to disk (tmp file + atomic replace each)."""
    for q in QUANTITIES:
      if self.dvs_is_memmap == True:
        self.datavectors[q].flush()  # checkpoint dv in-place
      else:
        # save (flush) dvs to tmp file (safer) ---------------------------------
        np.save(f"{self.dvsf}_{q}.tmp.npy", self.datavectors[q])
        # save data vector file (from tmp) -------------------------------------
        os.replace(f"{self.dvsf}_{q}.tmp.npy", f"{self.dvsf}_{q}.npy")

  def _dv_append(self, nparams):
    """Grow both stores by nparams zero rows (append mode; RAM-aware)."""
    nrows = self.datavectors[QUANTITIES[0]].shape[0]
    RAMneed = self.samples.nbytes + self.failed.nbytes
    for q in QUANTITIES:
      ncols_q = self.datavectors[q].shape[1]
      itemsize = self.datavectors[q].dtype.itemsize
      RAMneed += (2 * nrows + nparams) * ncols_q * itemsize
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      for q in QUANTITIES:
        ncols_q = self.datavectors[q].shape[1]
        # setup new datavector numpy array -----------------------------------
        self.datavectors[q] = np.vstack(
          (self.datavectors[q],
           np.zeros((nparams, ncols_q), dtype=self.dtype)))
        # save (flush) dvs to tmp file (safer) -------------------------------
        np.save(f"{self.dvsf}_{q}.tmp.npy", self.datavectors[q])
        # save data vector file (from tmp) -----------------------------------
        os.replace(f"{self.dvsf}_{q}.tmp.npy", f"{self.dvsf}_{q}.npy")
      self.dvs_is_memmap = False
    else:
      print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
            f"There is {RAMavail/1e9:.2f} GB of RAM available. "
            f"We will read dvs from HD (slow)")
      for q in QUANTITIES:
        ncols_q = self.datavectors[q].shape[1]
        # setup new datavector numpy array -----------------------------------
        datavectors = open_memmap(f"{self.dvsf}_{q}.tmp.npy",
                                  mode = "w+",
                                  shape = (nrows + nparams, ncols_q),
                                  dtype = self.datavectors[q].dtype)
        for s in range(0, nrows, 2500): # read dvs in chunks: avoid RAM spikes
          e = min(nrows, s + 2500)
          datavectors[s:e] = self.datavectors[q][s:e]
        for s in range(nrows, nrows + nparams, 2500):
          e = min(nrows + nparams, s + 2500)
          datavectors[s:e] = 0
        # save (flush) data vector (in-place) --------------------------------
        datavectors.flush()
        del datavectors
        # save data vector file (from tmp) -----------------------------------
        os.replace(f"{self.dvsf}_{q}.tmp.npy", f"{self.dvsf}_{q}.npy")
        # finally, load the new dv numpy array from file ---------------------
        self.datavectors[q] = np.load(f"{self.dvsf}_{q}.npy",
                                      mmap_mode = "r+",
                                      allow_pickle = False)
      self.dvs_is_memmap = True

    # check final dimensions -----------------------------------------------
    for q in QUANTITIES:
      if self.datavectors[q].shape[0] != self.samples.shape[0]:
        raise ValueError(f"Incompatible samples/datavector ({q}) chk files")

  def _dv_alloc(self, nrows, first_dvs):
    """
    Allocate both stores for nrows samples (RAM-aware, one shared
    policy) and write the grid sidecars ({dvsf}_h_z.npy /
    {dvsf}_dm_z.npy) once — the training path reads the grid from the
    FILE (resolved values, never re-declared in a YAML).
    """
    RAMneed = self.samples.nbytes + self.failed.nbytes
    for q in QUANTITIES:
      want = (len(self._grid_of(q)),)
      if first_dvs[q].shape != want:
        raise ValueError(
          f"first computed payload ({q}) has shape {first_dvs[q].shape}, "
          f"expected {want} from the train_args grid")
      RAMneed += nrows * want[0] * np.dtype(self.dtype).itemsize
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      self.dvs_is_memmap = False
    else:
      print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
            f"There is {RAMavail/1e9:.2f} GB of RAM available. "
            f"We will read dvs from HD (slow)")
      self.dvs_is_memmap = True

    self.datavectors = {}
    for q in QUANTITIES:
      ncols_q = len(self._grid_of(q))
      if self.dvs_is_memmap:
        self.datavectors[q] = open_memmap(f"{self.dvsf}_{q}.npy",
                                          mode = "w+",
                                          shape = (nrows, ncols_q),
                                          dtype = self.dtype)
        self.datavectors[q][:] = 0.0
        self.datavectors[q].flush()
      else:
        self.datavectors[q] = np.zeros((nrows, ncols_q), dtype=self.dtype)
      # the grid sidecar, written once beside the store.
      np.save(f"{self.dvsf}_{q}_z.npy", self._grid_of(q))

  def _dv_write(self, i, dvs):
    """Write one payload dict at row i of each per-quantity store."""
    for q in QUANTITIES:
      self.datavectors[q][i] = dvs[q]

  def _dv_zero(self, i):
    """Zero row i of each per-quantity store (a failed sample)."""
    for q in QUANTITIES:
      self.datavectors[q][i, :] = 0.0

  #-----------------------------------------------------------------------------
  # per-sample computation
  #-----------------------------------------------------------------------------
  def _compute_dvs_from_sample(self, sample):
    """
    H(z) and D_M(z) for one parameter row, in one background solve.

    Arguments:
      sample = one parameter row (1D, train_args.ord order).

    Returns:
      {"h": (nz,), "dm": (nz2,)} float32 — H on the z_sn grid in
      km/s/Mpc, and the comoving radial distance on the z_rec grid in
      Mpc (equal to the transverse D_M because the run is flat).

    Raises:
      RuntimeError when the prior or the theory rejects the row (the
      finite log-posterior is the acceptance fact) or CAMB errors.
    """
    # Define fortran errors we want to capture ---------------------------------
    camb_error_keywords = {"ERROR", "error", "Did not converge"}

    # sample arrives in ord order; sample[idx] is the model's sampled
    # order (the same reordering the prior check uses).
    idx = self.reorder_idx_from_ord_to_yaml()

    # Check prior before attempting computation --------------------------------
    if math.isinf(self.model.prior.logp(sample[idx])):
      raise RuntimeError(f"Prior is -inf (this should not happen). "
                         f"Values: {dict(zip(self.sampled_params, sample))}")

    # Evaluate through the STANDARD cobaya lifecycle, not the hand-rolled
    # check_cache_and_compute component loop the other generators use.
    # With this generator's background-only requirement set, that loop
    # serves the SAME background for every sample (a stale component
    # cache: bitwise-constant H(z) columns in the dump).
    # logposterior(point, cached=False) recomputes every component with
    # cobaya's own parameter routing (derived inputs such as an omch2
    # lambda included), so it cannot serve stale physics; the dummy `one`
    # likelihood keeps it as cheap as the theory call itself.
    captured = 0 # variable that will hold terminal output
    with capture_native_output() as tmp:
      logpost = self.model.logposterior(sample[idx], cached=False)
      tmp.seek(0)
      captured = tmp.read() # copy terminal output -----------------------------

    # check for CAMB errors in the terminal output -----------------------------
    if any(kw in captured for kw in camb_error_keywords):
      raise RuntimeError(f"CAMB Fortran error: {captured.strip()}")
    # a rejected point returns -inf without raising or printing anything;
    # the finite log-posterior is the acceptance fact, never the terminal scan.
    if not math.isfinite(logpost.logpost):
      raise RuntimeError(
        f"cobaya rejected the point (log-posterior {logpost.logpost}); "
        f"values: {dict(zip(self.sampled_params, sample))}")

    # get results from the provider (already computed): H in km/s/Mpc on
    # the SN grid; the comoving distance (flat: D_M = chi) in Mpc on the
    # recombination window.
    h  = self.model.provider.get_Hubble(self.z_sn, units="km/s/Mpc")
    dm = self.model.provider.get_comoving_radial_distance(self.z_rec)
    return {"h":  np.array(h,  copy=True, dtype=self.dtype),
            "dm": np.array(dm, copy=True, dtype=self.dtype)}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  run_generator(dataset, prog='dataset_generator_background')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
