"""Raw data loading, streaming statistics, and the physical cut.

This module is the bottom of the pipeline: it turns the on-disk
parameter (.txt) and data-vector (.npy) dumps into the in-memory
"source" dicts the rest of the package consumes, never loading the
(memmap-sized) dv file whole.
stream_chunks, stream_stats, and param_stats compute per-column
normalization stats over selected rows; stage_source materializes a row
subset in RAM if it fits (else keeps the memmap); phys_cut_idx applies the
physical cuts (omega_b h^2 below a bound, omegam^2 h^2 inside an optional
window); read_param_names reads parameter names off a covmat
header. load_source orchestrates: memmap, cut, size, stage one source into
a {C, dv, idx, (+means)} dict.

The staging pipeline, per source (load_source top to bottom):

    <params>.txt + .paramnames      <dv>.npy
       │  resolve by declared names    │  np.load(mmap_mode="r")
       ▼                               ▼
    C  (N, named inputs)            dv (N, total_size), on disk
       │  phys_cut_idx: the omegabh2 bound plus the optional
       │  omegam2h2 / omegamh2 / omegamh2*ns windows
       ▼
    pool = surviving row indices
       │  keep the first n_keep cut rows, one seeded shuffle
       ▼
    idx = this run's rows (seeded selection order)
       │  stage_source: do the used rows fit ram_frac of free RAM?
       │    yes -> copy C[rows] / dv[rows] (sorted, distinct), return
       │           local coordinates that walk the copy in selection order
       │    no  -> keep the memmap, idx stays the global selection order
       ▼
    source dict {C, dv, idx, C_mean, dv_mean}

    (legend: N = rows in the dump; the .txt begins with weight / lnp and
     then every column declared by .paramnames; derived count and placement
     come from that sidecar rather than a positional "last chi2" guess;
     total_size = full dv length;
     n_keep = the absolute number of cut rows to stage (enforced
     here, after the cuts; load_source raises if the pool is
     smaller);
     C_mean / dv_mean = training-subset means the geometries
     center on; ram_frac = the fraction of free RAM the staged
     subset may occupy (materialize only if it fits, else keep the
     memmap); local coordinates = searchsorted(rows, idx), returned by
     the resident branch instead of a plain arange so the compact copy
     is walked in the same seeded selection order the memmap path keeps,
     and the loader trains the same cosmology at the same step in either
     regime.)

Row coordinates (the same names are used in stage_source, load_source,
the loader closures in batching, and the grid2d base-row helper in
experiment): three coordinate systems name a row, and a bug hides
wherever they are confused.

  disk row     = a row's position in the original parameter / data-vector
                 files (0 is the first row on disk).
  storage row  = its position after the distinct selected rows are copied
                 into a fresh in-RAM array in sorted order (0 is the
                 smallest selected disk row; sorted so a memmap fills
                 sequentially). Also called the compact row.
  loader index = the per-position index the training loop hands a loader
                 each epoch (perm = idx_src[randperm]). The resident
                 branch sets idx_src = searchsorted(rows, idx), the
                 storage rows in selection order; the memmap branch sets
                 idx_src = idx, the global rows in selection order. Walking
                 either under the same permutation visits the same disk
                 rows in the same order.

The load-bearing invariant: dump_rows[j] is the DISK row that supplied
storage row j (dump_rows is sorted, distinct). The grid2d base-row helper
reads a separate file that stores another quantity for the same
cosmologies in the same original row order (the "base" dump) at
dump_rows, to line up cosmology for cosmology; it is the only consumer of
dump_rows.

A tiny example. Say a run's seeded selection order is disk rows [9, 2, 5]
(distinct, unsorted). The sorted distinct rows are [2, 5, 9], so
dump_rows = [2, 5, 9] and the compact copy's storage rows [0, 1, 2] hold
disk rows [2, 5, 9]. The resident branch returns the loader index
searchsorted([2, 5, 9], [9, 2, 5]) = [2, 0, 1], so loader positions
[0, 1, 2] point at storage rows [2, 0, 1] = disk rows [9, 2, 5], the
selection order. The memmap branch returns the loader index [9, 2, 5]
(the global selection order) and holds no compact copy. Under any epoch
permutation the two branches therefore walk the identical sequence of
disk rows; the RAM decision never reorders training.

PS: a dump is the full on-disk array from the data-generation run, every
simulated cosmology stored as one row (the data-vector dump is the .npy
file, the parameter dump the .txt); a training run draws its N_train
subset of rows from it. a memmap (memory-mapped array) is a NumPy array
backed by the file on disk and read in slices, so an array larger than RAM
is never loaded whole (advanced indexing a memmap still materializes the
requested rows; the pipeline stays disk-backed only because it reads
bounded blocks on demand). a copy is a fresh array with its own storage;
a view / memmap shares the file's storage. a loader is a closure
load(rows) -> a ready-to-train batch on the device, hiding where the data
lives (resident on the GPU, streamed from RAM, or read from the memmap).
to whiten is to rotate into the covariance eigenbasis and scale to unit
variance, so correlated quantities become decorrelated and equally scaled.
"""

import os

import numpy as np
import psutil
import torch

from . import fixed_facts
from .parameter_table import resolve_parameter_table


def stream_chunks(idx, chunk):
  """
  Yield the row indices in sorted blocks of `chunk` rows.

  A generator (it `yield`s, so blocks are produced lazily). Each
  block is sorted so that indexing a memmap with it walks the
  file in increasing order, sequential disk access, not random
  seeks.

  Arguments:
    idx   = 1D array of row indices (any order).
    chunk = number of indices per block.

  Yields:
    a sorted sub-array of up to `chunk` indices.
  """
  # step through idx in windows of `chunk` (the last may be
  # short); np.sort orders each window for sequential reads.
  for a in range(0, len(idx), chunk):
    yield np.sort(idx[a:a+chunk])


def stream_stats(mm, idx, method=1, CHUNK=10000):
  """
  Per-column normalization stats over a chosen subset of rows.

  Context: a run uses only the N_train subset of the dump. `mm`
  is a row-indexable view of the data vectors (the on-disk dump,
  a memmap larger than RAM, or its staged in-RAM subset), and
  `idx` the rows used. Stats accumulate over just those rows,
  streamed CHUNK at a time, so `mm` is never fully loaded.
  `method` picks the scheme:
    1 = z-score  -> returns (mean, std)
    2 = min-max  -> returns (min,  max - min)
  The caller then normalizes a row as (x - offset) / scale.

  Arguments:
    mm     = 2D array indexable by row (in-RAM or memmap);
             columns are the quantities to summarize.
    idx    = the rows of `mm` to include (the N_train subset).
    method = 1 for z-score, 2 for min-max.
    CHUNK  = rows read per streamed block.

  Returns:
    (offset, scale) as float32 torch tensors, one per column.
  """
  n = len(idx)               # total rows summarized
  ncols = mm.shape[1]        # one stat per column

  if method == 1:
    # one-pass mean/variance via running sums. float64
    # accumulators keep precision and avoid overflow over many
    # rows.
    s1 = np.zeros(ncols, dtype="float64")   # sum of x
    s2 = np.zeros(ncols, dtype="float64")   # sum of x^2
    for rows in stream_chunks(idx=idx, chunk=CHUNK):
      # read this block and upcast to float64.
      x = np.asarray(mm[rows], dtype="float64")
      s1 += x.sum(axis=0)            # accumulate sum
      s2 += (x * x).sum(axis=0)      # accumulate sum of sq

    mean = s1 / n
    # variance = (sum_sq - sum^2/n) / (n-1): the one-pass form of
    # the unbiased sample variance; sqrt gives the std.
    std  = np.sqrt((s2 - s1 * s1 / n) / (n - 1))
    offset, scale = mean, std
  elif method == 2:
    # running min/max, started at +inf / -inf so the first block
    # replaces them.
    mn = np.full(ncols,  np.inf, dtype="float64")
    mx = np.full(ncols, -np.inf, dtype="float64")
    for rows in stream_chunks(idx=idx, chunk=CHUNK):
      x = np.asarray(mm[rows], dtype="float64")
      mn = np.minimum(mn, x.min(axis=0))   # tighten the min
      mx = np.maximum(mx, x.max(axis=0))   # tighten the max
    offset, scale = mn, mx - mn

  # hand back float32 torch tensors (the model's dtype).
  off = torch.from_numpy(offset.astype("float32"))
  scl = torch.from_numpy(scale.astype("float32"))
  return off, scl


def param_stats(arr, idx, method=1):
  """
  Per-column normalization stats for the cosmo params.

  The caller normalizes as (x - offset) / scale. Sums run in
  float64 for accurate totals; the returned tensors are float32
  (the model's dtype).

  Arguments:
    arr    = the parameter array (or memmap), row per sample.
    idx    = row indices to compute the stats over (the training
             subset; never the validation rows).
    method = 1 -> z-score: returns (mean, std);
             2 -> min-max: returns (min, max - min).

  Returns:
    (offset, scale) float32 torch tensors, one value per column.
  """
  a = np.asarray(arr[idx], dtype="float64")
  if method == 1:
    offset = a.mean(axis=0)
    scale  = a.std(axis=0, ddof=1)
  elif method == 2:
    offset = a.min(axis=0)
    scale  = a.max(axis=0) - offset
  off = torch.from_numpy(offset.astype("float32"))
  scl = torch.from_numpy(scale.astype("float32"))
  return off, scl


def stage_source(C, dv, idx, ram_frac=0.7):
  """
  Stage a source's used rows in RAM if they fit, else leave them
  on disk.

  A run uses only the N_train subset of the dump, named by `idx`.
  That subset is far smaller than the dump and usually fits in
  RAM even when the dump does not. If the combined bytes of BOTH
  compact copies (the parameter table C[idx] and the target dv[idx],
  each at its own dtype and width) are below ram_frac of available
  RAM, materialize them and return local coordinates into that copy;
  otherwise return the inputs unchanged so the loaders stream dv from
  the memmap by global index.

  `idx` is the run's seeded selection order: a distinct, generally
  unsorted prefix of one shuffled permutation of the cut pool. Both
  branches must present the selected rows in that one canonical order,
  because the training loop applies the same epoch permutation to
  whichever index it receives (perm = idx_src[randperm]). If the
  resident branch renumbered to a plain arange over the sorted compact
  copy, the same seed would map to a different cosmology at every step
  than the disk-backed branch does -- host-memory availability would
  silently change training. The resident branch therefore returns
  searchsorted(rows, idx), the local coordinates that walk the compact
  copy in exactly the selection order the disk branch keeps.

  Arguments:
    C        = full parameter dump, (N, Ncosmo).
    dv       = full dv dump, (N, Ndv); ndarray or np.memmap.
    idx      = the rows this run uses, as global row indices, in the
               run's seeded selection order. Must be unique (a
               permutation prefix); a duplicate raises, since it would
               train one cosmology twice and skew the stats.
    ram_frac = fraction of available RAM the materialized subset
               may occupy (default 0.7).

  Returns:
    C_src, dv_src, idx_src. When the subset fits: the compact in-RAM
      copies C[rows] / dv[rows] over the distinct rows in sorted
      (sequential-read) order, and idx_src = searchsorted(rows, idx),
      the local coordinates that reproduce the seeded selection order
      (not a plain arange). When it does not fit: (C, dv, idx) unchanged (dv
      still the memmap, idx still the global selection order). Either
      way the loader walks the selected rows in one canonical order, so
      the RAM decision never changes which cosmology trains at a step.
  """
  idx_arr = np.asarray(idx)
  rows = np.sort(np.unique(idx_arr))    # sorted, distinct -> sequential
  # the training selection is a unique permutation prefix by construction:
  # load_source and the scalar loader both draw phys[:keep] from one shuffled
  # cut pool with no repeats. A duplicate reaching here is upstream corruption
  # (it would train one cosmology twice and weight the normalization stats by
  # the accident), so refuse loudly rather than paper over it with the
  # searchsorted map below, which would silently reproduce the repeat.
  if rows.size != idx_arr.size:
    raise ValueError(
      "stage_source got a selection of " + str(int(idx_arr.size))
      + " rows with only " + str(int(rows.size)) + " distinct; the training "
      "selection must be a unique set of rows (a permutation prefix). A repeat "
      "means the upstream shuffle or physical cut lost its uniqueness.")
  # advanced indexing (C[rows] and dv[rows]) materializes BOTH arrays as eager
  # copies, so the budget must count BOTH, each at its own dtype and width, plus
  # the reindex array. Counting only the dv copy underestimated by the whole
  # parameter table: a narrow-output scalar dump (many input columns, one
  # output column) then chose the resident branch even when the two copies
  # together did not fit -- the parameter bytes can dwarf a one-wide target.
  dv_bytes  = rows.size * dv.shape[1] * dv.dtype.itemsize
  par_bytes = rows.size * C.shape[1] * C.dtype.itemsize
  idx_bytes = rows.size * np.dtype(np.int64).itemsize
  need = dv_bytes + par_bytes + idx_bytes
  avail = psutil.virtual_memory().available
  budget = ram_frac * avail
  # strict less-than is a deliberate policy, not an accidental branch: the
  # eager copies need transient working room above their own bytes (the
  # allocator, the advanced-index temporaries), so an exact fill (need ==
  # budget) stays on disk rather than materializing to the last free byte.
  # The self-test pins need below, equal to, and above budget so this
  # boundary cannot drift to <= without a caught failure.
  fits = need < budget
  # one essential staging line: the three named byte terms, their sum, the
  # comparison that actually held, the budget, and the branch, so an
  # out-of-memory or a surprise disk-streaming run is visible up front. The
  # sum names idx_bytes explicitly -- an earlier line printed "params + dv =
  # total" while total already carried the reindex bytes, so the arithmetic
  # on screen did not add up.
  print("stage_source: %d rows, params %.1f MB + dv %.1f MB + idx %.1f MB "
        "= %.1f MB %s budget %.1f MB -> %s"
        % (rows.size, par_bytes / 1e6, dv_bytes / 1e6, idx_bytes / 1e6,
           need / 1e6, "<" if fits else ">=", budget / 1e6,
           "RAM (materialized)" if fits else "memmap (disk)"))
  if fits:
    # materialize the distinct rows into RAM in sorted (sequential-read)
    # order, then return LOCAL coordinates that walk that compact copy in the
    # run's seeded selection order -- not a plain arange. searchsorted sends
    # each selected row to its slot in the sorted copy, so C[rows][local] ==
    # C[idx] and dv[rows][local] == dv[idx]: the compact copy is presented in
    # exactly the order the disk-backed branch keeps its global index, and the
    # same epoch permutation trains the same cosmology at the same step in
    # either regime.
    local = np.searchsorted(rows, idx_arr)
    return (np.asarray(C[rows]),
            np.asarray(dv[rows]),
            local)
  # too big for RAM: keep full arrays + the global selection index, stream dv
  # from disk (the parameter table was already loaded eagerly upstream; only
  # the dv memmap stays disk-backed here). idx is the seeded selection order,
  # the same order the resident branch reproduces through its local map.
  return C, dv, idx


# --- derived physical densities the training-pool windows cut on ---
# Each formula takes col, a dict mapping a parameter name to that
# column of the candidate rows (col["H0"] = C[idx, names.index("H0")]),
# and returns the derived quantity per row. The window table in
# phys_cut_idx pairs one formula with its lo / hi bounds, so a new
# window (say omegamh2 * ns * As) is one helper plus one table row,
# not a new branch in the cut logic.
def _omega_b_h2(col):
  # omega_b h^2 = Omega_b * (H0/100)^2.
  return col["omegab"] * (col["H0"] / 100.0) ** 2


def _omega_m2_h2(col):
  # omegam^2 h^2 = (Omega_m * H0/100)^2 = Gamma^2, the transfer shape.
  return (col["omegam"] * col["H0"] / 100.0) ** 2


def _omega_m_h2(col):
  # omegam h^2 = Omega_m * (H0/100)^2 (Planck ~ 0.143).
  return col["omegam"] * (col["H0"] / 100.0) ** 2


def _omega_m_h2_ns(col):
  # omegam h^2 * n_s (Planck ~ 0.138).
  return _omega_m_h2(col) * col["ns"]


def phys_cut_idx(C, idx, names, omegabh2_hi,
                 omegabh2_lo=None,
                 omegam2h2_lo=None, omegam2h2_hi=None,
                 omegamh2_lo=None, omegamh2_hi=None,
                 omegamh2ns_lo=None, omegamh2ns_hi=None,
                 param_file="<param dump>"):
  """
  Keep only rows inside the physical-density windows.

  Cuts on derived products a per-parameter scan misses (all strict,
  lo < quantity < hi):

    omegabh2   = Omega_b (H0/100)^2      (omegabh2_lo, omegabh2_hi)
    omegam2h2  = (Omega_m H0/100)^2      (omegam2h2_lo, omegam2h2_hi)
    omegamh2   = Omega_m (H0/100)^2      (omegamh2_lo, omegamh2_hi)
    omegamh2ns = omegamh2 * n_s          (omegamh2ns_lo, omegamh2ns_hi)

  The high-omega_b h^2 cosmologies (a sparse, ~2x Planck corner) fail
  catastrophically and no real posterior visits them. omegam^2 h^2 is
  Gamma^2, the transfer-shape parameter squared and the coordinate
  along the sampling cloud's long axis; its window keeps the
  well-covered core around Planck's Gamma^2 ~ 0.045. omegamh2 (Planck
  ~ 0.143) and its n_s product omegamh2ns (~ 0.138) window the
  hardness direction the forensics flagged. Every bound may be None
  (that side is not cut) and the whole set defaults off, so a config
  without the window keys selects exactly the rows it did before.

  A window's quantity is computed only when the window is active, so a
  dump without an `ns` column is fine unless an omegamh2ns bound is
  set (then a loud error names the column and the file).

    window table                per row, over the candidate idx
       │  each active window: check its columns exist, compute its
       │  formula, keep lo < quantity < hi (strict, either side
       │  optional)
       ▼
    keep  (len(idx),) bool       rows clearing every active window
       │  idx[keep]
       ▼
    kept idx  +  report          one (name, tag, lo, hi, kept, total)
                                 per active window, for the banner

  (legend: idx = the candidate row indices; keep = the boolean mask,
  True for a row that clears every active window (the running
  intersection of the window masks); report = the per-window survivor
  counts, name = the window key, tag = its formula, lo / hi = its
  bounds, kept = rows that window alone keeps, total = len(idx).)

  Arguments:
    C          = full parameter dump, (N, n_param), physical units,
                 column order given by `names`.
    idx        = candidate row indices into C (e.g. a shuffle).
    names      = parameter column names in C's column order; the
                 windows locate their columns (omegab / omegam / H0 /
                 ns) by name.
    omegabh2_hi   = upper bound on omega_b h^2 (rows >= it dropped),
                    the one always-applied bound (renamed from `cut`).
    omegabh2_lo   = optional lower bound on omega_b h^2 (None = no
                    lower cut).
    omegam2h2_lo  = optional lower bound on omegam^2 h^2 = Gamma^2
                    (None = no lower cut).
    omegam2h2_hi  = optional upper bound on omegam^2 h^2 (None = no
                    upper cut).
    omegamh2_lo   = optional lower bound on omegam h^2 = Omega_m
                    (H0/100)^2 (None = no lower cut).
    omegamh2_hi   = optional upper bound on omegam h^2 (None = no
                    upper cut).
    omegamh2ns_lo = optional lower bound on omegam h^2 * n_s (needs
                    the ns column; None = no lower cut).
    omegamh2ns_hi = optional upper bound on omegam h^2 * n_s (None =
                    no upper cut).
    param_file    = the dump path, named in the missing-column error.

  Returns:
    (kept_idx, report): kept_idx = the subset of idx passing every
    active window, in idx's order; report = a list, one
    (name, tag, lo, hi, kept, total) tuple per active window, where
    kept = rows that window alone keeps (marginal) and total =
    len(idx).

  Raises:
    ValueError if a window is given lo >= hi, or an active window
    needs a parameter column absent from `names`.
  """
  # window table: name, banner tag (the formula), the columns the
  # formula needs, the formula, and its (lo, hi) bounds. A new window
  # is one row here (see the _omega_* helpers above), not a new
  # branch; obh2's hi is the always-applied `omegabh2_hi`.
  windows = [
    ("omegabh2",   "Om_b (H0/100)^2",  ("omegab", "H0"),
     _omega_b_h2,    omegabh2_lo,   omegabh2_hi),
    ("omegam2h2",  "(Om H0/100)^2",    ("omegam", "H0"),
     _omega_m2_h2,   omegam2h2_lo,  omegam2h2_hi),
    ("omegamh2",   "Om (H0/100)^2",    ("omegam", "H0"),
     _omega_m_h2,    omegamh2_lo,   omegamh2_hi),
    ("omegamh2ns", "Om (H0/100)^2 ns", ("omegam", "H0", "ns"),
     _omega_m_h2_ns, omegamh2ns_lo, omegamh2ns_hi),
  ]

  # keep = the running intersection of the active windows' masks (starts all-True);
  # report = one row per active window for the loading banner.
  keep   = np.ones(len(idx), dtype=bool)
  report = []
  for name, tag, need_cols, formula, lo, hi in windows:
    # a window with neither bound is inactive: skip it, so its
    # columns are never even required.
    if lo is None and hi is None:
      continue
    # both bounds given -> the window must be a real interval.
    if lo is not None and hi is not None and lo >= hi:
      raise ValueError(
        f"the {name} window needs lo < hi, got ({lo}, {hi})")
    # resolve the columns this window's formula needs; a missing one
    # (e.g. ns absent from the dump) is a loud error naming the
    # column and the file, not a silent no-cut.
    col = {}
    for cname in need_cols:
      if cname not in names:
        raise ValueError(
          f"the {name} window needs the {cname!r} parameter column, "
          f"but {param_file} has none (columns: {names})")
      col[cname] = C[idx, names.index(cname)]
    q = formula(col)
    # strict window: keep lo < q < hi (either side optional).
    wmask = np.ones(len(idx), dtype=bool)
    if lo is not None:
      wmask &= q > lo
    if hi is not None:
      wmask &= q < hi
    keep &= wmask
    report.append((name, tag, lo, hi, int(wmask.sum()), len(idx)))
  return idx[keep], report


def read_param_names(covmat_path, comment="#"):
  """
  Parameter column names from a covmat header line.

  Reads only the first line, strips the leading comment marker,
  splits on whitespace, the column order the parameter arrays
  (and ParamGeometry) use.

  Arguments:
    covmat_path = path to the covmat file; its first line lists
                  the column names, prefixed by `comment`.
    comment     = the leading marker to strip (default "#").

  Returns:
    a list of parameter-name strings, in column order.
  """
  with open(covmat_path) as f:
    return f.readline().lstrip(comment).split()


def _sidecar_candidates(params_path, suffix):
  """
  Return sidecar paths paired with one parameter dump, in lookup order.

  A generator dump ``X.txt`` owns sidecar ``X<suffix>``. A getdist/cobaya
  chain ``X.1.txt`` first tries its exact stem and then the shared chain-root
  sidecar ``X<suffix>``. Only an all-decimal final stem component is a chain
  number: a legitimate dotted dataset such as ``lcdm.v2.txt`` keeps ``.v2``.

  Arguments:
    params_path = parameter dump path.
    suffix      = complete sidecar suffix, including its leading dot.

  Returns:
    candidate path strings, exact stem first and numeric chain root second
    when one exists.
  """
  base = os.path.splitext(os.fspath(params_path))[0]
  candidates = [base + suffix]
  root, chain_ext = os.path.splitext(base)
  if chain_ext[1:].isdigit():
    candidates.append(root + suffix)
  return candidates


def _find_sidecar(params_path, suffix):
  """
  Find the first existing sidecar paired with one parameter dump.

  Arguments:
    params_path = parameter dump path.
    suffix      = complete sidecar suffix, including its leading dot.

  Returns:
    the first existing candidate from ``_sidecar_candidates``, or None when
    the dataset has no such sidecar.
  """
  for candidate in _sidecar_candidates(params_path=params_path,
                                        suffix=suffix):
    if os.path.exists(candidate):
      return candidate
  return None


def _read_facts_sidecar_with_path(params_path):
  """Resolve and read one required sidecar without a second path lookup."""
  sidecar = _find_sidecar(params_path=params_path,
                          suffix=fixed_facts.SIDECAR_SUFFIX)
  if sidecar is None:
    candidates = _sidecar_candidates(
      params_path=params_path, suffix=fixed_facts.SIDECAR_SUFFIX)
    rendered = "\n".join("  " + repr(path) for path in candidates)
    raise ValueError(
      "the parameter table has no " + fixed_facts.SIDECAR_SUFFIX
      + " scientific-record sidecar; tried:\n" + rendered + "\n"
      + fixed_facts.MIGRATION)
  # newline="" disables universal-newline rewriting. The exact producer text
  # is what the saved emulator carries, including its original line endings.
  with open(sidecar, encoding="utf-8", newline="") as fh:
    return sidecar, fh.read()


def read_facts_sidecar(params_path):
  """
  Read the generator's required scientific-record sidecar verbatim.

  The generator publishes one small companion file beside the chain it dumps:
  <paramsf>.facts.yaml, the record of the cosmology the dataset was generated
  under and the parameter region it was sampled over (emulator/fixed_facts.py
  defines the record and owns every law it must satisfy). Training copies that
  text into the saved emulator without ever regenerating it, so the emulator
  carries the producer's own words. This function is the first step of the
  copy, and it does exactly one thing: find the file and hand back its text. It
  parses nothing and derives nothing, because a second author of a scientific
  fact is how the two copies of that fact drift apart.

  Finding it is not a plain splitext, because two naming conventions are in use
  and only one of them is what splitext returns:

      a generator dump   params.txt     pairs with   params.facts.yaml
      a cobaya chain     params.1.txt   pairs with   params.facts.yaml

  A cobaya run writes one chain file per chain number and a single sidecar
  shared by all of them, so a pure-integer chain number has to come off before
  the sidecar can be found. Stripping the last extension unconditionally would
  be wrong the other way: a dataset legitimately named "lcdm.v2.txt" must keep
  its ".v2". The number therefore comes off only when the piece before the
  final extension is entirely digits, and the exact stem is tried first either
  way. This is the same pairing getdist uses to find a chain's .paramnames.

  Arguments:
    params_path = the parameter file the sidecar sits beside: params.txt from
                  a generator dump, or params.1.txt from a cobaya chain.

  Returns:
    the sidecar's text, exactly as it sits on disk. Parsing and the ordered-name
    comparison remain the loader's next steps, after the parameter table has
    established which names it actually contains.

  Raises:
    ValueError when neither candidate exists. Training does not create a new
    older-format emulator from a dataset whose scientific facts were never
    recorded; the refusal names every path tried and explains how to regenerate
    the dataset.
  """
  _, facts_yaml = _read_facts_sidecar_with_path(params_path=params_path)
  return facts_yaml


def validated_facts_sidecar(params_path, names, facts_yaml=None):
  """Read or reuse a producer record and compare its ordered names.

  ``from_config`` calls this before device or warm-start work and keeps the
  returned text. Staging passes that same text back, so changing the file later
  cannot replace the record already checked. A direct loader leaves
  ``facts_yaml`` at None and reads the required sidecar after its .paramnames
  file has been checked.
  """
  if facts_yaml is None:
    sidecar, facts_yaml = _read_facts_sidecar_with_path(
      params_path=params_path)
    where = repr(sidecar)
  elif type(facts_yaml) is not str:
    raise TypeError(
      "the saved scientific-record sidecar text must be text, got "
      + type(facts_yaml).__name__)
  else:
    where = ("the scientific record read during the early configuration check "
             "for " + repr(os.fspath(params_path)))
  blocks = fixed_facts.parse_sidecar(text=facts_yaml, where=where)
  fixed_facts.check_names_match(
    geometry_names=names,
    blocks=blocks,
    where=where)
  return facts_yaml


def _load_failure_mask(path, expected_rows):
  """Read the generator's row-failure file without accepting coercions.

  Each physical line is one producer-owned flag. ``0`` means that the row's
  data vector was calculated successfully, and ``1`` means that calculation
  failed. Training must not treat a failed row's zero-filled payload as
  scientific data.

  Arguments:
    path = the generator's failure-mask file (the failfile beside the dump,
           named by data.train_failure_mask / data.val_failure_mask).
           Chain-only scalar sources use ``load_scalar_source`` and never
           call this function.
    expected_rows = number of rows in the parameter and data-vector files.

  Returns:
    A one-dimensional Boolean NumPy array with ``True`` for failed rows.

  Raises:
    ValueError when the path is missing, expected_rows is invalid, a line is
    not exactly ``0`` or ``1``, or the mask has a different number of rows.
  """
  if type(expected_rows) is not int or expected_rows < 0:
    raise ValueError(
      "expected_rows must be a nonnegative native integer, got "
      + repr(expected_rows))
  if path is None:
    raise ValueError(
      "a data-vector source requires the generator's failure mask; name the "
      "failfile in data.train_failure_mask / data.val_failure_mask")

  tokens = []
  with open(path, "r", encoding="ascii") as failure_file:
    for line in failure_file:
      token = line[:-1] if line.endswith("\n") else line
      tokens.append(token)

  invalid = []
  for line_number, token in enumerate(tokens, start=1):
    if token not in ("0", "1"):
      invalid.append((line_number, token))
  if invalid:
    raise ValueError(
      "the generator failure mask must contain one literal '0' or '1' on "
      "each line; invalid lines are " + repr(invalid) + " in "
      + repr(os.fspath(path)))
  if len(tokens) != expected_rows:
    raise ValueError(
      "the generator failure mask has " + str(len(tokens)) + " rows, but "
      "the parameter and data-vector files have " + str(expected_rows)
      + " rows: " + repr(os.fspath(path)))
  return np.asarray([token == "1" for token in tokens], dtype=bool)


def load_source(dv_path, params_path, names, omegabh2_hi, n_keep,
                gen=None, ram_frac=0.7, with_means=False,
                stage_dv=True,
                verbose=True,
                omegabh2_lo=None,
                omegam2h2_lo=None, omegam2h2_hi=None,
                omegamh2_lo=None, omegamh2_hi=None,
                omegamh2ns_lo=None, omegamh2ns_hi=None,
                facts_yaml=None,
                failure_mask_path=None):
  """
  Load, physically cut, and stage one dv/param source.

  Memmaps the dv dump (never reading it whole), keeps the
  modeled param columns, applies the physical windows (the
  omega_b h^2 bound plus the optional omegam^2 h^2 / omegamh2 /
  omegamh2*ns windows, see phys_cut_idx), takes the first
  n_keep cut rows of a fixed shuffle, stages that subset,
  and, when with_means, computes the centering means. When
  verbose, prints one per-window kept/total line so a mistyped
  window value is visible at once. Wraps phys_cut_idx /
  stage_source / stream_stats / param_stats.

  Arguments:
    dv_path     = .npy data-vector dump (memmapped).
    params_path = parameter text file. Its required .paramnames sidecar
                  selects and orders the modeled columns by name.
    names       = parameter column names (covmat order, the kept
                  columns); phys_cut_idx finds the omegab / H0
                  columns by them.
    omegabh2_hi  = upper bound on omega_b h^2 (rows >= it dropped);
                   required, threaded to phys_cut_idx (renamed from
                   the former `cut`).
    n_keep      = absolute number of rows to stage (required, an int
                  >= 1); the first n_keep of the physically-cut,
                  seeded-shuffled pool. The cut pool must supply them,
                  else this raises (the post-cut enforcement point). A
                  row count here (dump rows staged), not the kept
                  data-vector length also called n_keep in
                  geometries.output / designs/plain.
    gen         = torch.Generator seeding the cut+shuffle (required).
    ram_frac    = fraction of available RAM stage_source may fill
                  (default 0.7).
    with_means  = if True, also compute C_mean / dv_mean (train
                  needs them; val does not). With stage_dv False the
                  dv_mean is left out (see stage_dv).
    stage_dv    = if True (default), stage the used dv rows in RAM
                  when they fit (stage_source) and, with with_means,
                  stream dv_mean over them. The grid2d MPS path passes
                  False: it keeps the raw dump a MEMMAP (the unthinned
                  50,000 x 244,000 selection would be tens of GiB) and
                  computes no dv_mean here, because _grid2d_law_rows
                  reads only the kept columns in bounded row chunks and
                  takes the mean over the THINNED law-space rows. C is
                  then left at full width with global idx too (the two
                  must share one row numbering); _grid2d_law_rows
                  compacts both after the transform.
    verbose     = if True (default), print a one-line summary
                  (shapes, rows, in-RAM).
    failure_mask_path = authenticated generator failure-mask file. Rows marked
                  ``1`` are removed before the physical cuts and shuffle
                  selection. A missing path is refused; chain-only data use
                  ``load_scalar_source`` instead.
    omegabh2_lo  = optional lower bound on omega_b h^2 (None = no
                   lower cut; omegabh2_hi stays the upper bound).
    omegam2h2_lo = optional lower bound on omegam^2 h^2 (the
                   Gamma^2 window, see phys_cut_idx; None = no
                   lower cut).
    omegam2h2_hi = optional upper bound on omegam^2 h^2 (None =
                   no upper cut).
    omegamh2_lo / omegamh2_hi     = optional window on omegam h^2 =
                   Omega_m (H0/100)^2 (see phys_cut_idx; None on a
                   side = not cut there).
    omegamh2ns_lo / omegamh2ns_hi = optional window on omegam h^2 *
                   n_s (needs the ns column; None on a side = not
                   cut there).

  Returns:
    a source dict ready for build_loaders / run_emulator: "C" the staged
    parameters, "dv" the staged data vectors, "idx" the loader index,
    "dump_rows" the disk rows the staged rows came from (see the comment at
    the return), "source_n_rows" the exact row count shared by the original
    parameter and data-vector dumps, and "selected_rows" the exact disk-row
    sequence chosen after the seeded shuffle, cuts, and failed-row removal.
    "facts_yaml" is the generator's required scientific record as its exact
    original text. "C_mean" and "dv_mean" are also present when with_means.
  """
  # require a generator (n_keep is a required positional arg, so a
  # missing size is a plain TypeError at the call site).
  if gen is None:
    raise ValueError("load_source needs a torch.Generator (gen=)")
  # Resolve the parameter table before opening the data-vector dump.  The
  # Check the required .paramnames file first. If it is missing, reordered, or
  # inconsistent with the numeric table, report that specific problem before
  # opening a potentially enormous data-vector file.
  table = resolve_parameter_table(params_path=params_path,
                                  input_names=names)
  # Check the producer's scientific record only after .paramnames succeeds. A
  # broken .paramnames file therefore keeps its direct explanation, while a
  # missing, malformed, or differently ordered facts record still stops before
  # the large data-vector file is opened. The parser checks the text; the exact
  # text is kept for the saved emulator instead of being rewritten here.
  facts_yaml = validated_facts_sidecar(
    params_path=params_path, names=names, facts_yaml=facts_yaml)
  dv = np.load(dv_path, mmap_mode="r", allow_pickle=False)
  C = table.inputs
  if C.shape[0] != dv.shape[0]:
    raise ValueError(
      f"incompatible files: {params_path} has {C.shape[0]} rows, "
      f"{dv_path} has {dv.shape[0]}")

  n = C.shape[0]
  failed = _load_failure_mask(
    path=failure_mask_path,
    expected_rows=n)
  order = torch.randperm(n, generator=gen).numpy()
  order = order[~failed[order]]
  # physical-window cuts are opt-in on the CMB path exactly as on the
  # scalar path: a CMB chain need not carry the
  # omegab / H0 / omegam / ns columns the windows read, so
  # omegabh2_hi = None skips the cuts entirely. The cosmolike path
  # always passes a value (validate_param_cuts requires it there), so
  # its behavior is unchanged.
  if omegabh2_hi is None:
    phys, cut_report = order, []
  else:
    phys, cut_report = phys_cut_idx(C=C, idx=order, names=names,
                                    omegabh2_hi=omegabh2_hi,
                                    omegabh2_lo=omegabh2_lo,
                                    omegam2h2_lo=omegam2h2_lo,
                                    omegam2h2_hi=omegam2h2_hi,
                                    omegamh2_lo=omegamh2_lo,
                                    omegamh2_hi=omegamh2_hi,
                                    omegamh2ns_lo=omegamh2ns_lo,
                                    omegamh2ns_hi=omegamh2ns_hi,
                                    param_file=params_path)
  if verbose:
    # per-window survivor counts: a value typed against the wrong
    # quantity (the omegamh2-vs-omegam2h2 one-character trap) shows up
    # here as an obviously wrong kept count.
    for name, tag, lo, hi, kept, total in cut_report:
      print(f"  cut {name} = {tag} in ({lo}, {hi}): "
            f"kept {kept}/{total}")
  # rows to keep: the absolute n_keep, enforced against the cut pool.
  keep  = int(n_keep)
  # guard the low end too: a negative keep would index phys[:keep] (pool
  # minus |keep| rows, a silent wrong-rows path) and keep = 0 stages an
  # empty source. Not YAML-reachable (validate_sizes guards that), but the
  # explicit stage_train(n_train=...) path (sweep_ntrain / bakeoff) is.
  if keep < 1:
    raise ValueError(
      f"n_keep must be >= 1 (absolute rows to stage), got {keep}")
  if len(phys) < keep:
    raise ValueError(
      f"physical pool too small after param_cuts: kept {len(phys)} of "
      f"{n} rows, requested n_keep = {keep} "
      f"({os.path.basename(dv_path)}); loosen the windows or enlarge "
      f"the dump")
  idx = phys[:keep]
  selected_rows = np.asarray(idx, dtype=np.int64).copy()

  # stage the cut rows in RAM if they fit, else keep the memmap. The
  # grid2d MPS path (stage_dv False) skips this: it keeps the raw dump
  # a memmap and the params at full width with the global idx, so
  # _grid2d_law_rows can read only the kept columns in bounded chunks
  # instead of materializing the unthinned selection (tens of GiB).
  if stage_dv:
    C_src, dv_src, idx_src = stage_source(
      C=C, dv=dv, idx=idx, ram_frac=ram_frac)
  else:
    C_src, dv_src, idx_src = C, dv, idx
  # dump_rows = the disk rows that supplied the staged rows, in ascending
  # distinct order (the invariant from the module docstring: dump_rows[j]
  # is the disk row behind compact row j). np.asarray gets an eager numpy
  # view of idx, np.unique returns its distinct values already ascending,
  # and the outer np.sort is redundant under current numpy (unique already
  # sorts) but states the intended order plainly. This is exactly the disk
  # order the resident copy was built over (stage_source copies at
  # np.sort(np.unique(idx))) and the order the memmap path keeps its global
  # index in, so it aligns either staging path. A consumer that must line
  # up a separate row-matched file cosmology for cosmology (the grid2d base
  # dump) reads it; every other consumer ignores the extra key.
  src = {"C": C_src,
         "dv": dv_src,
         "idx": idx_src,
         "dump_rows": np.sort(np.unique(np.asarray(idx))),
         "source_n_rows": int(n),
         "selected_rows": selected_rows,
         "facts_yaml": facts_yaml}
  if with_means:
    # param_stats returns (offset, scale); the "_" discards the sample
    # scale on purpose. Parameter CENTERING uses this returned mean, but
    # parameter WHITENING takes its scale and rotation from the covariance
    # geometry, not from this per-column sample standard deviation.
    c_mean, _ = param_stats(arr=C_src, idx=idx_src, method=1)
    src["C_mean"] = c_mean
    # grid2d (stage_dv False) recomputes dv_mean over the thinned law-space
    # rows in _grid2d_law_rows: it first selects the kept k columns, forms
    # the stored target law such as log(raw / base), materializes the
    # float32 training payload, and only then takes its mean. Computing a
    # mean here over the raw, unthinned dump would summarize the wrong
    # quantity (the pre-law surface over columns training never consumes),
    # so it is skipped, not merely deferred.
    if stage_dv:
      dv_mean, _ = stream_stats(mm=dv_src, idx=idx_src, method=1)
      src["dv_mean"] = dv_mean

  if verbose:
    in_ram = not isinstance(dv_src, np.memmap)
    failed_count = int(np.count_nonzero(failed))
    if failed_count:
      print("  ignored " + str(failed_count)
            + " rows whose generator calculations failed")
    # "used K of P cut rows": the staged count out of the post-cut pool,
    # so the absolute-count enforcement is visible on every run, not only
    # when the pool is too small.
    print(f"  {os.path.basename(dv_path)}: C {tuple(C_src.shape)} "
          f"dv {tuple(dv_src.shape)} used {keep} of {len(phys)} cut rows "
          f"| in RAM: {in_ram}")
  return src


def load_scalar_source(params_path, in_names, out_names, n_keep,
                       gen=None, ram_frac=0.7, with_means=False,
                       verbose=True, omegabh2_hi=None,
                       omegabh2_lo=None,
                       omegam2h2_lo=None, omegam2h2_hi=None,
                       omegamh2_lo=None, omegamh2_hi=None,
                       omegamh2ns_lo=None, omegamh2ns_hi=None,
                       facts_yaml=None):
  """
  Load, optionally cut, and stage one scalar (derived-parameter) source.

  The scalar sibling of load_source: inputs and outputs are both named
  columns of the ONE parameter .txt (no dv .npy, no cosmolike). The
  getdist .paramnames sidecar is required here (the only way to locate
  the output columns by name); the shared table resolver pins its
  non-derived names to in_names and selects derived outputs by name.
  Physical-window cuts are
  optional on this path (a scalar chain is already the target
  distribution, and the omega-windows reference params a scalar input set
  may not carry), so they run only when omegabh2_hi is given.

  Arguments:
    params_path = parameter text file (weight, minuslogpost, params...).
    in_names    = input parameter names (covmat order); C's columns, and
                  the names phys_cut_idx finds its window columns by.
    out_names   = output parameter names (data.outputs); the standardized
                  targets Y, staged in the "dv" slot.
    n_keep      = absolute rows to stage (an int >= 1), the first n_keep
                  of the seeded shuffle (after the optional cut).
    gen         = torch.Generator seeding the shuffle (required).
    ram_frac    = fraction of available RAM stage_source may fill.
    with_means  = if True, also compute C_mean (the ParamGeometry center)
                  and dv_mean (the output-column mean, kept for
                  source-dict parity; ScalarGeometry.from_targets
                  recomputes center / scale from the staged targets).
    verbose     = if True, print the per-window (when cut) and summary
                  lines.
    omegabh2_hi = optional upper bound on omega_b h^2. None (default) =
                  no physical-window cuts at all on the scalar path; a
                  value turns the same windows on as load_source (they
                  then require their columns to be in in_names, loud
                  otherwise).
    omegabh2_lo .. omegamh2ns_hi = the optional window bounds, applied
                  only when omegabh2_hi is given.

  Returns:
    a source dict ready for build_loaders / run_emulator: "C" the staged input
    parameters, "dv" the staged output targets, "idx" the loader index,
    "dump_rows" the sorted disk rows represented by compact storage,
    "source_n_rows" the number of rows in the original table, and
    "selected_rows" the exact disk-row sequence chosen by the seeded shuffle
    and optional cuts. "facts_yaml" is the generator's required scientific
    record as its exact original text. "C_mean" and "dv_mean" are also present
    when with_means.
  """
  if gen is None:
    raise ValueError("load_scalar_source needs a torch.Generator (gen=)")
  table = resolve_parameter_table(params_path=params_path,
                                  input_names=in_names,
                                  output_names=out_names)

  # As in load_source, check .paramnames first so a bad parameter declaration
  # receives its own explanation. Then parse the facts record and compare its
  # sampled-name order before copying any subset. Its original text, not a
  # training-authored rewrite, is what travels to save_emulator.
  facts_yaml = validated_facts_sidecar(
    params_path=params_path, names=in_names, facts_yaml=facts_yaml)

  C = table.inputs          # input parameters, in in_names order
  Y = table.outputs         # output targets, in out_names order

  n     = C.shape[0]
  order = torch.randperm(n, generator=gen).numpy()
  # physical-window cuts are opt-in on the scalar path (see docstring).
  if omegabh2_hi is None:
    phys, cut_report = order, []
  else:
    phys, cut_report = phys_cut_idx(C=C, idx=order, names=in_names,
                                    omegabh2_hi=omegabh2_hi,
                                    omegabh2_lo=omegabh2_lo,
                                    omegam2h2_lo=omegam2h2_lo,
                                    omegam2h2_hi=omegam2h2_hi,
                                    omegamh2_lo=omegamh2_lo,
                                    omegamh2_hi=omegamh2_hi,
                                    omegamh2ns_lo=omegamh2ns_lo,
                                    omegamh2ns_hi=omegamh2ns_hi,
                                    param_file=params_path)
  if verbose:
    for name, tag, lo, hi, kept, total in cut_report:
      print(f"  cut {name} = {tag} in ({lo}, {hi}): kept {kept}/{total}")
  keep = int(n_keep)
  if keep < 1:
    raise ValueError(
      f"n_keep must be >= 1 (absolute rows to stage), got {keep}")
  if len(phys) < keep:
    raise ValueError(
      f"scalar pool too small: {len(phys)} rows available "
      f"(of {n}{' after cuts' if omegabh2_hi is not None else ''}), "
      f"requested n_keep = {keep} ({os.path.basename(params_path)})")
  idx = phys[:keep]
  selected_rows = np.asarray(idx, dtype=np.int64).copy()

  # stage the used rows in RAM (the .txt is already in memory, so this
  # just compacts to the subset). dv = the output targets Y.
  C_src, Y_src, idx_src = stage_source(
    C=C, dv=Y, idx=idx, ram_frac=ram_frac)
  src = {"C": C_src,
         "dv": Y_src,
         "idx": idx_src,
         "dump_rows": np.sort(np.unique(np.asarray(idx))),
         "source_n_rows": int(n),
         "selected_rows": selected_rows,
         "facts_yaml": facts_yaml}
  if with_means:
    # param_stats returns (offset, scale); the "_" discards the sample scale
    # both times. The scalar geometry standardizes its outputs from Y_mean
    # here (center) and its own stored per-column scale; the input parameter
    # scale still comes from the covariance geometry, not this sample std.
    c_mean, _ = param_stats(arr=C_src, idx=idx_src, method=1)
    y_mean, _ = param_stats(arr=Y_src, idx=idx_src, method=1)
    src["C_mean"]  = c_mean
    src["dv_mean"] = y_mean

  if verbose:
    print(f"  {os.path.basename(params_path)}: C {tuple(C_src.shape)} "
          f"targets {tuple(Y_src.shape)} used {keep} of {len(phys)} rows")
  return src
