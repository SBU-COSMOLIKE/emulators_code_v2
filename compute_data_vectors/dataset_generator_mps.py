import numpy as np
import math, os, sys, traceback
import inspect
import psutil
from numpy.lib.format import open_memmap
from generator_core import (GeneratorCore, capture_native_output,
                            dark_energy_facts, run_generator)
# generator_core prepends the repo root to sys.path when it is imported, so
# the emulator package (fixed_facts, syren_base) resolves when this driver
# runs by path.
from emulator import fixed_facts
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# The script below computes MATTER POWER SPECTRUM training dumps for the MPS
# emulators (the fourth thin driver on generator_core): per sample,
# one CAMB call through cobaya's Pk_interpolator requirement yields BOTH
# targets on the fixed (z, k) grids:
#   - the linear P(k, z), Mpc^3
#   - the nonlinear BOOST B(k, z) = P_nl / P_lin (dimensionless)
# and, when train_args.write_syren_base is true, the syren analytic BASE for
# each (emulator/syren_base.py — the formula the emulator corrects, written
# beside the dump so the training's law transform reads resolved values from
# disk, never a re-run of a possibly-drifted package).
#
#     mpirun -n 10 --report-bindings \
#       python external_modules/code/emulators/emultrfv2/compute_data_vectors/dataset_generator_mps.py \
#         --root projects/example/  \
#         --fileroot emulators/mps/ \
#         --nparams 10000 \
#         --yaml 'w0wa_mps.yaml' \
#         --datavsfile 'w0wa_dvs_train' \
#         --paramfile 'w0wa_params_train' \
#         --failfile  'w0wa_params_failed_train' \
#         --chain 0 --unif 1 --temp 64 --maxcorr 0.15 \
#         --freqchk 2000 --loadchk 0 --append 0 --boundary 1.0
#
# All command-line flags are the shared generator flags (documented in
# dataset_generator_lensing.py and parsed in generator_core.py).
#
#- The training YAML must carry, inside train_args:
#      probe: mps
#      z_segments:                    # the z grid, a concat of linspaces
#        - [0.0, 2.0, 100, false]     #   [zmin, zmax, n, endpoint]
#        - [2.0, 10.0, 10, false]     #   (the legacy 122-point grid)
#        - [10.0, 50.0, 12, true]
#      k_log10: [-4.0, 2.0, 2000]     # k = logspace(min, max, n), 1/Mpc
#      extrap_kmax: 200.0             # the interpolator's power-law tail
#      write_syren_base: true         # write the *_base files (the syren
#                                     # formulas are vendored in syren/);
#                                     # false for a law-none run
#  plus the shared keys (ord; fiducial/params_covmat_file when --unif 0).
#  The theory block's halofit choice (e.g. halofit_version: mead2020) is the
#  YAML's, persisted with the run. The likelihood may be the dummy `one` —
#  this script adds the Pk requirements to the model itself.
#  When write_syren_base is true, the sampled/params block must resolve the
#  names syren_params_from reads (As or As_1e9, ns, H0, omegab, omegam).
#
#- The output files are
#
#      w0wa_params_train_mps_unifs.1.txt / .covmat / .paramnames / .ranges
#
#      # Data vectors: one row per sample, columns = the FLATTENED (z, k)
#      # surface (z outer), plus the grid sidecars written once:
#      w0wa_dvs_train_mps_unifs_pklin.npy         (N, nz*nk)  Mpc^3
#      w0wa_dvs_train_mps_unifs_boost.npy         (N, nz*nk)  dimensionless
#      w0wa_dvs_train_mps_unifs_pklin_base.npy    (when write_syren_base)
#      w0wa_dvs_train_mps_unifs_boost_base.npy    (when write_syren_base)
#      w0wa_dvs_train_mps_unifs_z.npy             (nz,)
#      w0wa_dvs_train_mps_unifs_k.npy             (nk,)  1/Mpc
#      w0wa_params_failed_train_mps_unifs.txt
#
#  A training run points data.train_dv at ONE quantity's file (and
#  data.grid2d.train_base at its _base sibling under a syren law).
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Class Definition
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class dataset(GeneratorCore):
  """
  MPS generator: one CAMB call per sample (through the Pk_interpolator
  requirement) yields the linear P and the boost on the fixed grids;
  the syren base rides along when write_syren_base is true. The store
  is two or four per-quantity 2D files + the grid sidecars; the
  per-sample payload is a dict of flattened float32 rows.
  """
  VALID_PROBES = ("mps",)
  EXTRA_TRAIN_KEYS = ("z_segments", "k_log10", "extrap_kmax",
                      "write_syren_base")
  FAMILY = "grid2d"                     # scientific-record family name
  PROGRAM = "dataset_generator_mps"     # producer name in the record

  def _read_train_args(self, train_args):
    """Build the grids, resolve the base switch, register requirements.

    z_segments = a list of [zmin, zmax, n, endpoint] linspace segments,
    concatenated ascending (the legacy 122-point grid is three of
    them); k_log10 = [log10 kmin, log10 kmax, nk]; extrap_kmax = the
    interpolator's power-law tail edge (must exceed the k grid's top);
    write_syren_base = whether the syren base files are computed and
    written (the formulas are vendored in-repo under syren/; stated
    explicitly, never a silent availability fallback).

    Arguments:
      train_args = the YAML's train_args mapping.
    """
    segs = train_args["z_segments"]
    if not isinstance(segs, (list, tuple)) or len(segs) < 1:
      raise ValueError("train_args.z_segments must be a non-empty list "
                       "of [zmin, zmax, n, endpoint] segments")
    pieces = []
    for seg in segs:
      if (not isinstance(seg, (list, tuple))) or len(seg) != 4:
        raise ValueError(f"z_segments entry must be [zmin, zmax, n, "
                         f"endpoint], got {seg!r}")
      zmin, zmax, n, endpoint = (float(seg[0]), float(seg[1]),
                                 int(seg[2]), bool(seg[3]))
      if not (zmin < zmax) or n < 2:
        raise ValueError(f"z_segments entry needs zmin < zmax and "
                         f"n >= 2, got {seg!r}")
      pieces.append(np.linspace(zmin, zmax, n, endpoint=endpoint))
    z = np.concatenate(pieces)
    if not (np.diff(z) > 0).all():
      raise ValueError("z_segments must concatenate to a strictly "
                       "ascending grid (segments overlap or repeat an "
                       "edge; use endpoint false on the inner pieces)")
    if len(z) < 4:
      raise ValueError("the z grid needs at least 4 points "
                       "(the Pk interpolation demands it)")
    kk = train_args["k_log10"]
    if (not isinstance(kk, (list, tuple))) or len(kk) != 3:
      raise ValueError(f"train_args.k_log10 must be [log10 kmin, "
                       f"log10 kmax, nk], got {kk!r}")
    lo, hi, nk = float(kk[0]), float(kk[1]), int(kk[2])
    if not (lo < hi) or nk < 8:
      raise ValueError(f"k_log10 needs log10 kmin < log10 kmax and "
                       f"nk >= 8, got {kk!r}")
    self.z_mps = z
    self.k_mps = np.logspace(lo, hi, nk)
    self.extrap_kmax = float(train_args["extrap_kmax"])
    if self.extrap_kmax < self.k_mps[-1]:
      raise ValueError(
        f"train_args.extrap_kmax ({self.extrap_kmax}) must reach the "
        f"k grid's top ({self.k_mps[-1]}); the interpolator cannot be "
        f"evaluated beyond its extrapolation edge")
    self.write_base = train_args["write_syren_base"]
    if type(self.write_base) is not bool:
      # bool(x) would silently accept the YAML strings "true"/"false"
      # (both nonempty, both True); the switch must be a YAML boolean.
      raise ValueError(
        "train_args.write_syren_base must be the YAML boolean true or "
        "false; got " + repr(self.write_base))
    if self.write_base:
      # fail at setup, not at sample 1 of an MPI farm: prove the base
      # formulas import on this rank (the syren package is vendored
      # in-repo and numpy-only, so this only fails on a broken tree).
      from emulator.syren_base import base_pklin, base_boost  # noqa: F401
    # the dark-energy law of this run in canonical (w, wa) form, resolved
    # once at setup. The per-sample syren base call needs it: a run that
    # samples w0pwa and derives wa must hand the base a varying wa, not
    # the LCDM default a name-only reading would give.
    (_, _, self.dark_energy_law, _) = dark_energy_facts(
      self.model.parameterization, self._resolved_constants())

    # explicit requirements on the model itself (the YAML likelihood
    # may be the dummy `one`). The Cl quirk is the legacy convention,
    # kept verbatim.
    #
    # The requirement's k_max is DERIVED from the resolved k grid:
    # 2 x the grid's top, floored at 20 (halofit's sigma integrals
    # need support past the grid edge). The legacy convention was the
    # verbatim constant 200 — which IS this formula on the legacy
    # production grid (k top 100 -> 200, byte-identical requirement),
    # so production behavior is unchanged; a small-grid smoke run
    # stops paying for transfers at k = 200 it never reads (the first
    # full mps-smoke run spent ~1 hour there: 400 CAMB calls at
    # k_max 200 against a k grid topping at 10). Every requested k is
    # still COMPUTED, never extrapolated: the dump only evaluates on
    # the grid itself (top < k_max); extrap_kmax governs the served
    # interpolator's power-law tail beyond that, as before.
    k_max_req = max(2.0 * float(self.k_mps[-1]), 20.0)
    self.model.add_requirements({
      "Pk_interpolator": {
        "z": self.z_mps,
        "k_max": k_max_req,
        "nonlinear": (True, False),
        "vars_pairs": ([("delta_tot", "delta_tot")]),
      }, "Cl": {  # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
        'tt': 0
      }})

  def _quantities(self):
    """The store's quantity tags for this run.

    Returns:
      ("pklin", "boost") for a plain run; the two *_base tags join when
      write_syren_base is on (the base files ride the switch).
    """
    if self.write_base:
      return ("pklin", "boost", "pklin_base", "boost_base")
    return ("pklin", "boost")

  def _facts_base_identity(self):
    """
    Name the syren analytic base these dumps sit on top of, or "n/a".

    Only a run with write_syren_base on has a base: it writes the *_base
    files the emulator corrects. The base formulas evaluate their growth
    factors at a neutrino mass pinned in base_pklin's own signature, so an
    emulator trained against this base carries that pin whether or not the
    run sampled mnu — the record names the base and the pinned mass
    together, because the pair is what a consumer must match. The pinned
    mass is read from the formula's own default (the generator calls
    base_pklin without an mnu argument), so the record cannot drift from
    the base it names.

    Returns:
      the base's description string (naming the vendored formulas and
      the pinned neutrino mass), or fixed_facts.NOT_APPLICABLE for a
      run without base files.
    """
    if not self.write_base:
      return fixed_facts.NOT_APPLICABLE
    from emulator.syren_base import base_pklin
    parameters = inspect.signature(base_pklin).parameters
    if "mnu" not in parameters \
        or parameters["mnu"].default is inspect.Parameter.empty:
      raise ValueError(
        "the neutrino mass the syren base is pinned at cannot be read from "
        "base_pklin's mnu default in emulator/syren_base.py; the dataset's "
        "record must name the base together with that pinned mass")
    return ("syren analytic base, owned by emulator/syren_base.py (the "
            "symbolic_pofk formulas vendored in syren/), with mnu pinned at "
            + fixed_facts.format_value(value=parameters["mnu"].default))

  #-----------------------------------------------------------------------------
  # data-vector store: per-quantity 2D files -> {dvsf}_<q>.npy
  #-----------------------------------------------------------------------------
  def _dv_chk_files(self):
    """Files a resume must find: every store plus the two axis sidecars.

    Returns:
      the list of required file paths.
    """
    files = []
    for q in self._quantities():
      files.append(f"{self.dvsf}_{q}.npy")
    files.extend((f"{self.dvsf}_z.npy", f"{self.dvsf}_k.npy"))
    return files

  def _dv_load_chk(self):
    """Load every per-quantity store (RAM-aware, one shared policy).

    This family holds one 2-D store per quantity, each row a flattened
    (z, k) surface, instead of the core's single array. Both axis
    sidecars (redshift and wavenumber) are compared against the YAML's
    grids before anything loads, so a checkpoint written for one
    (z, k) grid is never continued on another. The load-whole-or-
    memmap decision is then made ONCE over the combined bytes and
    applied to every store, because mixing a loaded store with a
    memmapped one would make row writes differ in speed and durability
    mid-run. The memmap modes and the fit rule are the core's (see
    GeneratorCore._dv_load_chk, which explains both).

    Raises:
      ValueError when an axis disagrees with the YAML, a store is not
      2-D, or a store's rows or columns disagree with the checkpoint.
    """
    # a checkpoint written for one (z, k) grid must never be continued on
    # another: the flattened columns would silently change meaning.
    self._load_axis_checkpoint(path=f"{self.dvsf}_z.npy",
                               expected=self.z_mps,
                               label="mps redshift")
    self._load_axis_checkpoint(path=f"{self.dvsf}_k.npy",
                               expected=self.k_mps,
                               label="mps wavenumber")
    RAMneed = self.samples.nbytes + self.failed.nbytes
    for q in self._quantities():
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

    width = len(self.z_mps) * len(self.k_mps)
    self.datavectors = {}
    for q in self._quantities():
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
      if self.datavectors[q].shape[1] != width:
        raise ValueError(
          f"chk datavectors ({q}) have {self.datavectors[q].shape[1]} "
          f"columns but train_args names a {width}-point (z, k) grid; "
          f"the chk and the YAML disagree")

  def _dv_save(self):
    """Flush every store to disk (tmp file + atomic replace each).

    Per store, the same two save paths as the core (see
    GeneratorCore._dv_save): a memmapped store checkpoints in place
    with flush(), and an in-RAM store writes a temporary sibling and
    swaps it in with os.replace, which the operating system performs
    atomically, so a crash mid-save leaves the previous complete file.
    """
    for q in self._quantities():
      if self.dvs_is_memmap == True:
        self.datavectors[q].flush()  # checkpoint dv in-place
      else:
        # save (flush) dvs to tmp file (safer) ---------------------------------
        np.save(f"{self.dvsf}_{q}.tmp.npy", self.datavectors[q])
        # save data vector file (from tmp) -------------------------------------
        os.replace(f"{self.dvsf}_{q}.tmp.npy", f"{self.dvsf}_{q}.npy")

  def _dv_append(self, nparams):
    """Grow every store by nparams zero rows (append mode; RAM-aware).

    Arguments:
      nparams = the number of new sample rows the append adds.
    """
    quantities = self._quantities()
    nrows = self.datavectors[quantities[0]].shape[0]
    RAMneed = self.samples.nbytes + self.failed.nbytes
    for q in quantities:
      ncols_q = self.datavectors[q].shape[1]
      itemsize = self.datavectors[q].dtype.itemsize
      RAMneed += (2 * nrows + nparams) * ncols_q * itemsize
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      for q in quantities:
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
      for q in quantities:
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
    for q in quantities:
      if self.datavectors[q].shape[0] != self.samples.shape[0]:
        raise ValueError(f"Incompatible samples/datavector ({q}) chk files")

  def _dv_alloc(self, nrows, first_dvs):
    """
    Allocate every store for nrows samples (RAM-aware, one shared
    policy) and write the grid sidecars ({dvsf}_z.npy / {dvsf}_k.npy)
    once — the training path reads the grids from the FILES.

    Arguments:
      nrows     = the total number of sample rows the run will fill.
      first_dvs = the first computed payload dict; every quantity's
                  width is checked against len(z) * len(k).
    """
    width = len(self.z_mps) * len(self.k_mps)
    RAMneed = self.samples.nbytes + self.failed.nbytes
    for q in self._quantities():
      if first_dvs[q].shape != (width,):
        raise ValueError(
          f"first computed payload ({q}) has shape "
          f"{first_dvs[q].shape}, expected ({width},) from the "
          f"train_args (z, k) grid")
      RAMneed += nrows * width * np.dtype(self.dtype).itemsize
    RAMavail = psutil.virtual_memory().available
    if RAMneed < 0.75 * RAMavail:
      self.dvs_is_memmap = False
    else:
      print(f"Warning: samples & dvs need {RAMneed/1e9:.2f} GB of RAM. "
            f"There is {RAMavail/1e9:.2f} GB of RAM available. "
            f"We will read dvs from HD (slow)")
      self.dvs_is_memmap = True

    self.datavectors = {}
    for q in self._quantities():
      if self.dvs_is_memmap:
        self.datavectors[q] = open_memmap(f"{self.dvsf}_{q}.npy",
                                          mode = "w+",
                                          shape = (nrows, width),
                                          dtype = self.dtype)
        self.datavectors[q][:] = 0.0
        self.datavectors[q].flush()
      else:
        self.datavectors[q] = np.zeros((nrows, width), dtype=self.dtype)
    # the grid sidecars, written once beside the stores.
    np.save(f"{self.dvsf}_z.npy", self.z_mps)
    np.save(f"{self.dvsf}_k.npy", self.k_mps)

  def _dv_write(self, i, dvs):
    """Write one payload dict into every per-quantity store.

    Arguments:
      i   = the sample's assigned row.
      dvs = the payload dict, one flattened (nz * nk) row per quantity.
    """
    for q in self._quantities():
      self.datavectors[q][i] = dvs[q]

  def _dv_zero(self, i):
    """Blank one failed sample's row in every per-quantity store.

    Arguments:
      i = the failed sample's row.
    """
    for q in self._quantities():
      self.datavectors[q][i, :] = 0.0

  #-----------------------------------------------------------------------------
  # per-sample computation
  #-----------------------------------------------------------------------------
  def _compute_dvs_from_sample(self, sample):
    """
    Linear P(k,z) and the nonlinear boost for one parameter row.

    One CAMB call (through the Pk_interpolator requirement) serves both
    interpolators; the fixed (z, k) grids are evaluated and flattened
    with z as the outer axis. When write_syren_base is on, the syren
    analytic base is evaluated at the same row — under the run's cached
    dark-energy law, so a varying wa reaches the base formulas — and
    rides along as the *_base members.

    Arguments:
      sample = one parameter row (1D, train_args.ord order).

    Returns:
      a dict of flat float32 rows, each of length nz*nk: pklin (Mpc^3)
      and boost (dimensionless), plus pklin_base / boost_base when
      write_syren_base is on.

    Raises:
      RuntimeError when the prior rejects the row, CAMB errors, or the
      linear power is nonfinite or nonpositive.
    """
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

    # get results from the provider (already computed): the linear and
    # nonlinear P(k, z) through the interpolators (the legacy flow:
    # extrap_kmax from train_args, k evaluated on the fixed grid).
    lin = self.model.provider.get_Pk_interpolator(
        ("delta_tot", "delta_tot"), nonlinear=False,
        extrap_kmax=self.extrap_kmax)
    nl = self.model.provider.get_Pk_interpolator(
        ("delta_tot", "delta_tot"), nonlinear=True,
        extrap_kmax=self.extrap_kmax)
    pk_lin = lin.P(self.z_mps, self.k_mps)       # (nz, nk), Mpc^3
    pk_nl  = nl.P(self.z_mps, self.k_mps)
    if not (np.isfinite(pk_lin).all() and (pk_lin > 0).all()):
      raise RuntimeError("non-finite or non-positive linear P(k, z)")
    boost = pk_nl / pk_lin

    out = {"pklin": np.asarray(pk_lin, dtype=self.dtype).reshape(-1),
           "boost": np.asarray(boost, dtype=self.dtype).reshape(-1)}
    if self.write_base:
      # the syren analytic base: the formula this emulator
      # family corrects, written beside the raw dump. One definition
      # (emulator/syren_base.py), the parameters read by one rule.
      from emulator.syren_base import (syren_params_from, base_pklin,
                                       base_boost)
      as_1e9, ns, H0, Ob, Om, w0, wa = syren_params_from(
        param, dark_energy_law=self.dark_energy_law)
      pb = base_pklin(k_mpc=self.k_mps, z=self.z_mps, As_1e9=as_1e9,
                      ns=ns, H0=H0, Ob=Ob, Om=Om, w0=w0, wa=wa)
      bb = base_boost(k_mpc=self.k_mps, z=self.z_mps, pk_lin_mpc=pk_lin,
                      As_1e9=as_1e9, ns=ns, H0=H0, Ob=Ob, Om=Om,
                      w0=w0, wa=wa)
      out["pklin_base"] = np.asarray(pb, dtype=self.dtype).reshape(-1)
      out["boost_base"] = np.asarray(bb, dtype=self.dtype).reshape(-1)
    return out

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  run_generator(dataset, prog='dataset_generator_mps')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
