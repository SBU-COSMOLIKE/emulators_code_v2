#!/usr/bin/env python3
"""stage-ram: the host-RAM staging decision counts every array, keeps the
seeded row order, and shows honest arithmetic.

stage_source decides whether a run's selected rows are copied into RAM (the
resident branch) or streamed from the on-disk memmap (the disk-backed branch).
Three properties must hold, and each has its own legs below:

  Byte accounting. The resident branch materializes BOTH the compact parameter
  table C[rows] and the compact target dv[rows], plus the reindex array. The
  budget must count all three; counting only the dv bytes chose the resident
  branch for a narrow-output dump (many input columns, one output column) even
  when the two copies together exceeded the allowance.

  Seeded row order. `idx` is the run's seeded selection order (a distinct,
  generally unsorted prefix of a shuffled permutation). Both branches must
  present those rows in that one canonical order, because the training loop
  applies the same epoch permutation to whichever index stage_source returns
  (perm = idx_src[randperm]). If the resident branch returned a plain arange
  over the sorted compact copy, the same seed would train a different cosmology
  at each step than the disk branch does: host-memory availability would
  silently change training. The resident branch instead returns
  searchsorted(rows, idx), the local coordinates that walk the compact copy in
  selection order. The canonical-order legs drive the real per-source loader
  builder (_build_loaders_one, the function build_loaders calls) in both storage
  regimes and require the executed rows, their targets, their parameters, and
  the minibatch membership and order to be identical under one shared seed. A
  mutation arm that restores arange must fail.

  Honest banner. The staging line prints the three named byte terms, their sum,
  the comparison that actually held, the budget, and the branch. An earlier line
  printed "params + dv = total" while total already carried the reindex bytes,
  so the arithmetic on screen did not add up.

Importing emulator.data_staging and emulator.batching imports torch, so this
runs in the torch environment; every array here is tiny.
"""
import io
import os
import contextlib
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

import emulator.data_staging as ds
import emulator.batching as bt

FAILURES = []


def report(label, ok, detail=""):
    print("  [" + ("PASS" if ok else "FAIL") + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


class _FakeMem:
    """A stand-in for psutil.virtual_memory() with a fixed .available."""
    def __init__(self, available):
        self.available = available


def with_available(nbytes, fn):
    """Run fn() with stage_source's available-memory reading pinned to nbytes."""
    saved = ds.psutil.virtual_memory
    try:
        ds.psutil.virtual_memory = lambda: _FakeMem(nbytes)
        return fn()
    finally:
        ds.psutil.virtual_memory = saved


def took_resident(out, dv_in):
    """Whether stage_source took the resident branch.

    The resident branch returns a FRESH compact copy of dv; the disk-backed
    branch returns the same input dv object unchanged. Identity is the
    unambiguous test.
    """
    return out[1] is not dv_in


# ----------------------------------------------------------------------------
# Byte-accounting legs: the resident budget counts params + dv + idx.
# ----------------------------------------------------------------------------
def check_narrow_output():
    """The narrow-output scalar case: dv fits alone but not with C."""
    n, n_in, n_out = 1000, 12, 1        # many inputs, one output (scalar-like)
    C = np.arange(n * n_in, dtype="float32").reshape(n, n_in)
    dv = np.arange(n * n_out, dtype="float32").reshape(n, n_out)
    idx = np.arange(n)

    dv_bytes = n * n_out * 4
    par_bytes = n * n_in * 4
    idx_bytes = n * 8
    need = dv_bytes + par_bytes + idx_bytes

    # available set so ram_frac*avail sits ABOVE dv-alone but BELOW dv+C+idx.
    mid = (dv_bytes + need) / 2.0
    avail = mid / 0.7
    out = with_available(int(avail), lambda: ds.stage_source(C, dv, idx))
    report("narrow output: dv fits alone but dv+C does not -> disk-backed",
           not took_resident(out, dv),
           "0.7*avail=%.0f between dv=%d and need=%d" % (0.7 * avail,
                                                         dv_bytes, need))
    # control: plenty of memory -> resident branch, local reindex.
    out = with_available(int(need / 0.7 * 4), lambda: ds.stage_source(C, dv, idx))
    report("ample memory -> resident branch with local reindex",
           took_resident(out, dv) and np.array_equal(out[2], np.arange(n)),
           "materialized + local reindex (idx already sorted -> arange)")
    # control: almost no memory -> disk-backed branch, global index kept.
    out = with_available(int(need / 100), lambda: ds.stage_source(C, dv, idx))
    Cs, dvs, idxs = out
    report("tiny memory -> disk-backed branch (inputs unchanged)",
           idxs is idx and Cs is C and dvs is dv, "memmap kept")


def check_unequal_dtypes():
    """Unequal C / dv dtypes and widths are each counted at their own size."""
    n = 500
    C = np.zeros((n, 8), dtype="float64")           # 8 wide, 8 bytes each
    dv = np.zeros((n, 2), dtype="float32")          # 2 wide, 4 bytes each
    idx = np.arange(n)
    dv_bytes = n * 2 * 4
    par_bytes = n * 8 * 8
    need = dv_bytes + par_bytes + n * 8
    # budget above dv-only but below the real need: must go disk-backed.
    avail = ((dv_bytes + need) / 2.0) / 0.7
    out = with_available(int(avail), lambda: ds.stage_source(C, dv, idx))
    report("unequal dtypes/widths: each array counted at its own itemsize",
           not took_resident(out, dv),
           "float64 parameters dominate the two-column float32 target")


def check_byte_mutation():
    """The retired dv-only estimate would wrongly pick RAM; prove the delta."""
    n, n_in, n_out = 1000, 12, 1
    dv_bytes = n * n_out * 4
    par_bytes = n * n_in * 4
    need = dv_bytes + par_bytes + n * 8
    avail = ((dv_bytes + need) / 2.0) / 0.7
    budget = 0.7 * avail
    # the retired rule compared dv_bytes < budget (True here -> RAM); the
    # corrected rule compares need < budget (False here -> disk).
    old_picks_ram = dv_bytes < budget
    new_picks_disk = not (need < budget)
    report("mutation (dv-only estimate) picks RAM where the combined arrays "
           "do not fit; corrected estimate picks disk",
           old_picks_ram and new_picks_disk,
           "old RAM, new disk at the same budget")


# ----------------------------------------------------------------------------
# Uniqueness leg: a duplicated selection row is upstream corruption -> refuse.
# ----------------------------------------------------------------------------
def check_duplicate_refused():
    """A selection with a repeated row must raise, not silently reproduce it.

    The real selection is a unique permutation prefix; a duplicate reaching
    stage_source would train one cosmology twice and skew the normalization
    stats, so it is refused loudly. A unique control must still pass.
    """
    n = 20
    C = np.zeros((n, 3), dtype="float32")
    dv = np.zeros((n, 2), dtype="float32")
    dup = np.array([9, 2, 9, 5], dtype=np.int64)      # 9 appears twice
    raised = False
    try:
        with_available(10 ** 12, lambda: ds.stage_source(C, dv, dup))
    except ValueError:
        raised = True
    report("duplicate selection row is refused loudly", raised,
           "idx [9,2,9,5] raises ValueError")
    # unique control: the same shape without the repeat stages cleanly.
    ok_control = True
    try:
        with_available(10 ** 12,
                       lambda: ds.stage_source(C, dv,
                                               np.array([9, 2, 5], dtype=np.int64)))
    except ValueError:
        ok_control = False
    report("unique selection still stages (control)", ok_control,
           "idx [9,2,5] materializes without error")


# ----------------------------------------------------------------------------
# Fit-boundary legs: need < / == / > budget must map to RAM / disk / disk.
# ----------------------------------------------------------------------------
def check_fit_boundary():
    """The exact-fit boundary is a deliberate policy: strict less-than.

    ram_frac=1.0 with an integer available reading makes budget == avail
    exactly (float(int) is exact for these small sizes), so the three cases
    below/equal/above are pinned without floating-point slack.
    """
    n, n_in, n_out = 60, 5, 3
    rng = np.random.default_rng(1)
    C = rng.standard_normal((n, n_in)).astype("float32")
    dv = rng.standard_normal((n, n_out)).astype("float32")
    idx = np.arange(n)
    rows = np.sort(np.unique(idx))
    need = (rows.size * n_out * 4        # dv_bytes
            + rows.size * n_in * 4       # par_bytes
            + rows.size * 8)             # idx_bytes

    below = with_available(need + 1,
                           lambda: ds.stage_source(C, dv, idx, ram_frac=1.0))
    equal = with_available(need,
                           lambda: ds.stage_source(C, dv, idx, ram_frac=1.0))
    above = with_available(need - 1,
                           lambda: ds.stage_source(C, dv, idx, ram_frac=1.0))
    report("need < budget -> RAM (resident copy)",
           took_resident(below, dv), "avail = need + 1")
    report("need == budget -> disk (strict less-than keeps working headroom)",
           not took_resident(equal, dv), "avail = need exactly")
    report("need > budget -> disk (streamed)",
           not took_resident(above, dv), "avail = need - 1")


# ----------------------------------------------------------------------------
# Honest-banner legs: the printed line names params + dv + idx and adds up.
# ----------------------------------------------------------------------------
def _staging_line(C, dv, idx, avail, ram_frac=0.7):
    """Run stage_source with a pinned memory reading and return its stdout line."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with_available(avail,
                       lambda: ds.stage_source(C, dv, idx, ram_frac=ram_frac))
    for line in buf.getvalue().splitlines():
        if line.startswith("stage_source:"):
            return line
    return ""


def check_banner_terms():
    """The staging line prints three named terms whose sum equals the total,
    and the comparison operator agrees with the branch it selected."""
    n, n_in, n_out = 800, 10, 4
    C = np.zeros((n, n_in), dtype="float32")
    dv = np.zeros((n, n_out), dtype="float32")
    idx = np.arange(n)
    need = n * n_out * 4 + n * n_in * 4 + n * 8

    # resident line (ample memory): operator "<", branch "RAM".
    line_ram = _staging_line(C, dv, idx, int(need / 0.7 * 4))
    # disk line (tiny memory): operator ">=", branch "memmap".
    line_disk = _staging_line(C, dv, idx, int(need / 100))

    has_idx = "+ idx " in line_ram and "+ idx " in line_disk
    report("banner names the idx term explicitly (not folded into the total)",
           has_idx, repr(line_ram))

    # parse "params P MB + dv D MB + idx I MB = T MB <op> budget B MB -> ..."
    def three_terms_sum(line):
        try:
            p = float(line.split("params")[1].split("MB")[0])
            d = float(line.split("+ dv")[1].split("MB")[0])
            i = float(line.split("+ idx")[1].split("MB")[0])
            t = float(line.split("=")[1].split("MB")[0])
        except (IndexError, ValueError):
            return False, "unparseable"
        # each %.1f term rounds within 0.05 MB; the printed total is %.1f of the
        # same true need, so |t - (p+d+i)| stays well under 0.2 MB.
        return abs(t - (p + d + i)) < 0.2, "p+d+i=%.1f total=%.1f" % (p + d + i, t)

    ok_ram, det_ram = three_terms_sum(line_ram)
    ok_disk, det_disk = three_terms_sum(line_disk)
    report("resident banner: params + dv + idx sums to the printed total",
           ok_ram, det_ram)
    report("disk banner: params + dv + idx sums to the printed total",
           ok_disk, det_disk)
    report("operator agrees with the branch (< for RAM, >= for memmap)",
           ("< budget" in line_ram and "RAM (materialized)" in line_ram
            and ">= budget" in line_disk and "memmap (disk)" in line_disk),
           "RAM uses '<', disk uses '>='")


# ----------------------------------------------------------------------------
# Canonical-order legs: the real loader trains the same rows in the same order
# whichever branch stage_source took.
# ----------------------------------------------------------------------------
class _IdentityGeom:
    """A ParamGeometry stand-in whose encode is the identity.

    The row-order proof only needs load_C / load_dv to hand back the selected
    raw rows in the order the loader walks them; identity encode makes the
    returned tensor equal the raw dump row, so a row-for-row comparison reads
    off exactly which disk row trained at each step.
    """
    def encode(self, x):
        return x


class _IdentityChi2:
    """A loss stand-in whose encode is the identity over the full dv width."""
    def __init__(self, width):
        self.total_size = width
        self.dest_idx = torch.arange(width)
        self.needs_params = False

    def encode(self, dv):
        return dv


def _train_loaders(C_src, dv_src, idx_src, ncosmo, width, budget):
    """Build the real per-source train loaders for a staged source.

    _build_loaders_one is the exact function build_loaders calls per source; a
    large budget with a resident ndarray takes the pre-encoded gather regime, a
    small budget with a memmap takes the disk-stream regime, so the two storage
    regimes are exercised through the real code, not a re-derivation.
    """
    device = torch.device("cpu")
    model = torch.nn.Linear(ncosmo, width)
    load_C, load_dv, _load, _used = bt._build_loaders_one(
        device=device, C=C_src, dv=dv_src, idx=idx_src,
        param_geometry=_IdentityGeom(), chi2fn=_IdentityChi2(width),
        model=model, bs=2, budget=budget, dv_len=width)
    return load_C, load_dv


def _epoch_perm(idx_src, seed):
    """The training loop's own epoch permutation: perm = idx_src[randperm]."""
    g = torch.Generator().manual_seed(seed)
    ntrain = len(idx_src)
    return idx_src[torch.randperm(ntrain, generator=g).numpy()]


def check_canonical_order():
    """Resident and disk staging train identical rows in identical order.

    A unique, unsorted seeded selection (the real load_source path) is staged
    both ways: resident from an in-RAM ndarray, disk from a real .npy memmap.
    The real loaders are built for each, the training loop's epoch permutation
    is drawn once under a shared seed, and the executed parameters, targets, and
    minibatch order must match row-for-row. A mutation arm restoring arange in
    the resident branch must break the match.
    """
    n, ncosmo, width = 40, 5, 4
    rng = np.random.default_rng(7)
    C = rng.standard_normal((n, ncosmo)).astype("float32")
    dv = rng.standard_normal((n, width)).astype("float32")
    # seeded selection order: distinct and unsorted (a permutation prefix).
    # >= 10 rows (25M-15): the honest packed-target byte budget needs a
    # non-empty disk-stream window [resident + one complete batch, resident +
    # the full encoded set]. The full encoded set grows with the row count
    # while one batch (bs=2) does not, so a wide-enough selection makes a
    # streaming budget realisable; the old 8-row selection did not (one batch
    # exceeded the full encoded set, so the pre-repair 200-byte budget refused).
    idx = np.array([9, 2, 5, 7, 0, 3, 18, 11, 14, 6,
                    21, 30, 1, 25, 33, 8, 17, 12, 29, 4], dtype=np.int64)
    rows = np.sort(np.unique(idx))

    # a real on-disk memmap for the disk-stream regime.
    import tempfile
    fd, npy_path = tempfile.mkstemp(suffix=".stageram.npy")
    os.close(fd)
    try:
        np.save(npy_path, dv)
        dv_mm = np.load(npy_path, mmap_mode="r", allow_pickle=False)

        # resident staging (ample memory): compact copy + local coordinates.
        Cr, dvr, ir = with_available(10 ** 12,
                                     lambda: ds.stage_source(C, dv, idx))
        # disk staging (tiny memory, memmap dv): originals + global selection.
        Cd, dvd, idd = with_available(1,
                                      lambda: ds.stage_source(C, dv_mm, idx))
        both_branches = (dvr is not dv) and (dvd is dv_mm)
        report("staged both ways: resident copied, disk kept the memmap",
               both_branches, "resident fresh copy, disk memmap unchanged")

        # the real loaders for each regime (resident gather vs disk stream).
        # the disk-stream budget is chosen against the honest packed-target
        # planner (25M-15): its 0.8 allowance must sit INSIDE
        # [resident + one complete batch, resident + the full encoded set] --
        # large enough to stream one batch, too small to make the full encoded
        # set resident. A budget below that window (the pre-repair 200) refuses
        # with the named-terms MemoryError rather than silently mis-planning.
        loadC_r, loaddv_r = _train_loaders(Cr, dvr, ir, ncosmo, width, 10 ** 9)
        disk_budget = 1300                       # 0.8 * 1300 = 1040 allowance
        too_small = 700                          # below resident + one batch
        refused = False
        try:
            _train_loaders(Cd, dvd, idd, ncosmo, width, too_small)
        except MemoryError:
            refused = True
        streamed = True
        try:
            loadC_d, loaddv_d = _train_loaders(Cd, dvd, idd, ncosmo, width,
                                               disk_budget)
        except MemoryError:
            streamed = False
        report("disk path taken at the honest budget; a too-small budget refuses",
               streamed and refused,
               "budget=%d streams (0.8*budget=%d allowance, inside "
               "[resident+one batch, resident+full encoded set]); budget=%d "
               "refuses (below resident+one batch)"
               % (disk_budget, int(0.8 * disk_budget), too_small))

        # one shared epoch permutation, applied to each branch's own index.
        perm_r = _epoch_perm(ir, seed=1234)
        perm_d = _epoch_perm(idd, seed=1234)

        tgt_r = loaddv_r(perm_r).cpu().numpy()
        tgt_d = loaddv_d(perm_d).cpu().numpy()
        par_r = loadC_r(perm_r).cpu().numpy()
        par_d = loadC_d(perm_d).cpu().numpy()

        # the selection-order anchor: the global rows the seeded permutation
        # visits, computed straight from idx (no branch involved).
        g = torch.Generator().manual_seed(1234)
        order = idx[torch.randperm(len(idx), generator=g).numpy()]
        anchor_dv = dv[order]
        anchor_C = C[order]

        report("resident targets equal disk targets row-for-row (same seed)",
               np.array_equal(tgt_r, tgt_d), "identical executed dv sequence")
        report("resident params equal disk params row-for-row (same seed)",
               np.array_equal(par_r, par_d), "identical executed param sequence")
        report("executed sequence is the seeded selection order (anchor)",
               np.array_equal(tgt_r, anchor_dv) and np.array_equal(par_r, anchor_C),
               "matches dv[idx[randperm]] / C[idx[randperm]]")

        # minibatch membership and order at bs=2: consecutive slices of the
        # epoch permutation are the minibatches the loop draws, in order.
        bs = 2
        mb_r = [tuple(perm_r[a:a + bs]) for a in range(0, len(perm_r), bs)]
        mb_d = [tuple(perm_d[a:a + bs]) for a in range(0, len(perm_d), bs)]
        # map each branch's local index back to the global disk row it stands
        # for (resident: rows[local]; disk: already global) so membership is
        # compared in one coordinate system without sorting.
        mb_r_global = [tuple(rows[np.array(pair)]) for pair in mb_r]
        report("minibatch membership and order match across regimes (bs=2)",
               mb_r_global == mb_d, "%d minibatches, same pairs in same order"
               % len(mb_d))

        # sibling-dump alignment: mapping each resident loader position through
        # dump_rows recovers the global disk row, so a row-matched base dump
        # (grid2d) lines up cosmology for cosmology in either regime.
        dump_rows = rows                      # sort(unique(idx)) == rows
        report("dump_rows[idx_src] recovers the global selection order",
               np.array_equal(dump_rows[ir], idx) and np.array_equal(idd, idx),
               "resident maps through dump_rows; disk is already global")

        # mutation arm: restore the retired arange in the resident branch. The
        # loader then walks the sorted compact copy in randperm order, which is
        # NOT the selection order, so the cross-regime match must break.
        loadC_m, loaddv_m = _train_loaders(Cr, dvr, np.arange(rows.size),
                                           ncosmo, width, 10 ** 9)
        perm_m = _epoch_perm(np.arange(rows.size), seed=1234)
        tgt_m = loaddv_m(perm_m).cpu().numpy()
        report("mutation (resident arange) diverges from the disk order",
               not np.array_equal(tgt_m, tgt_d),
               "arange trains sorted-row order, caught against the disk branch")
    finally:
        os.unlink(npy_path)


def main():
    print("stage-ram (host-RAM staging: accounting, seeded order, honest banner)")
    check_narrow_output()
    check_unequal_dtypes()
    check_byte_mutation()
    check_duplicate_refused()
    check_fit_boundary()
    check_banner_terms()
    check_canonical_order()
    print("")
    if FAILURES:
        print("stage-ram: %d FAILURE(S): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("stage-ram: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
