#!/usr/bin/env python3
"""stage-ram: the host-RAM staging decision counts every array it materializes.

stage_source materializes BOTH the compact parameter table C[idx] and the
compact target dv[idx] when it takes the resident branch, but the budget check
formerly counted only the dv bytes. A narrow-output dump (many input columns,
one output column) then chose the resident branch even when the two copies
together exceeded the allowance. This check drives the real stage_source with a
mocked available-memory value set between "dv alone fits" and "dv plus C fits":
the corrected code must keep the disk-backed branch there, and the selected
values must be identical across both regimes.

Importing emulator.data_staging imports torch, so this runs in the torch
environment; the decision itself is pure NumPy and allocates only tiny arrays.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import emulator.data_staging as ds

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
    unambiguous test (an index equal to arange can occur in either branch when
    the caller's idx already is arange).
    """
    return out[1] is not dv_in


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
    # ram_frac default 0.7: pick avail so 0.7*avail is between dv_bytes and need.
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
           "materialized + local reindex")
    # control: almost no memory -> disk-backed branch, global index kept.
    out = with_available(int(need / 100), lambda: ds.stage_source(C, dv, idx))
    Cs, dvs, idxs = out
    report("tiny memory -> disk-backed branch (inputs unchanged)",
           idxs is idx and Cs is C and dvs is dv, "memmap kept")


def check_selected_values_match():
    """The staged rows are the same cosmologies in both regimes."""
    n, n_in, n_out = 40, 5, 3
    rng = np.random.default_rng(0)
    C = rng.standard_normal((n, n_in)).astype("float32")
    dv = rng.standard_normal((n, n_out)).astype("float32")
    idx = np.array([9, 2, 9, 5, 30, 5, 17])       # duplicates + unsorted
    want_rows = np.sort(np.unique(idx))

    res_C, res_dv, res_idx = with_available(
        10 ** 12, lambda: ds.stage_source(C, dv, idx))
    disk_C, disk_dv, disk_idx = with_available(
        1, lambda: ds.stage_source(C, dv, idx))
    # resident: compact copies + local arange; disk: the originals + global idx.
    resident_ok = (np.array_equal(res_C, C[want_rows])
                   and np.array_equal(res_dv, dv[want_rows])
                   and np.array_equal(res_idx, np.arange(want_rows.size)))
    disk_rows = np.sort(np.unique(disk_idx))
    disk_ok = (np.array_equal(disk_C[disk_rows], C[want_rows])
               and np.array_equal(disk_dv[disk_rows], dv[want_rows]))
    report("resident + disk regimes select the same distinct sorted rows",
           resident_ok and disk_ok, "byte-identical selected values")


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
           "float64 params dominate the one-wide float32 target")


def check_mutation():
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


def main():
    print("stage-ram (host-RAM staging accounting; tiny arrays, mocked memory)")
    check_narrow_output()
    check_selected_values_match()
    check_unequal_dtypes()
    check_mutation()
    print("")
    if FAILURES:
        print("stage-ram: %d FAILURE(S): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("stage-ram: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
