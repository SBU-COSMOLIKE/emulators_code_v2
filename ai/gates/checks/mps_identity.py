#!/usr/bin/env python3
"""mps-identity gate: the grid2d-emulator save/rebuild/predict
identity, the staging law transform, the emul_mps assembly math (base
stubbed: closed-form stub bases pin the assembly EXACTLY, independent
of the vendored syren/ formulas), the config loud errors, and the
finetune parity — torch + scipy, no CAMB.

Legs:
  - Grid2DGeometry: standardize round-trip to float32 round-off. Its state
    round-trip has eight byte-identical keys, including the mask.
    the width / unknown-law guards; the constant-column pinning (a
    constant law-space column that is not the whole surface is physics
    under ANY law — the boost's low-k B = 1 region: scale 1, decode
    returns the training constant, const_mask persists; a WHOLLY
    constant surface still raises, the dead-dump signature — the run-10
    pass deleted the pre-amendment partial-constant raise leg this
    section used to carry);
  - the STAGING law transform through the REAL load_source +
    _grid2d_law_rows: law rows == log(raw / base) with the base dump
    aligned by dump_rows through a real shuffled staging; resident and
    disk-backed results preserve the same nontrivial seeded row order;
    exact original raw/base row counts are required before target allocation;
    the k_stride thinning keeps the top edge; the positivity and width guards;
  - the BOUNDED staging on the production 122 x 2,000 grid (k_stride
    10, tiny memory budget, guarded memmap reads): every raw + base
    read is row-chunked and column-thinned (never the unthinned
    selection), the values and streamed mean equal an independent
    known-answer calculation, the low-RAM result is disk-backed, and
    the guard trips on the old whole-selection access — so the leg
    fails against the pre-fix implementation;
  - the STABLE streamed moments (check_stable_moments): the streamed
    geometry mean/std reproduce float64 np.std(ddof 0) of the
    materialized rows across uneven chunkings — a 50,000-row 1e8/1-ULP
    column keeps its true std 4.0 (the old one-pass form drifts to
    ~3.97 and can flip a varying column to a false constant pin), a
    constant column still pins, and the from_stats encode matches the
    materialized standardization;
  - the disk-backed staging LIFECYCLE (check_staging_lifecycle): the
    experiment owns its train / val temp file, supersedes it on restage
    (first file absent, second readable), keeps a three-point sweep's
    live temp count / bytes bounded to one point, unlinks a partial file
    on a mid-transform failure, releases a failed point's file through
    the lane-style cleanup, and the resident-RAM control makes no temp
    file and stays byte-identical;
  - save -> rebuild -> predict bitwise on the syren_linear and none
    laws (the predictor's grid2d branch returns {"z", "k", quantity}
    reshaped (nz, nk)); rebuild info flags class-guarded;
  - the correction-head leg: attach_head_coords (one bin per z slice),
    the identity basis (W_fd / W_df None), the ResCNN epoch-0
    identity start, the two-phase discipline (set_train_phase
    freezes the right groups per phase, the trunk phase bypasses
    the head bitwise, unknown phases raise — the 2026-07-12 ruling),
    the n_tokens rejection on real physical bins, and the head
    save -> rebuild -> predict bitwise round-trip (the rebuild-side
    attach in results._rebuild_model);
  - emul_mps (cobaya.theory stubbed, syren_base MONKEYPATCHED with
    synthetic closed forms): pair validation (missing quantity /
    duplicate / wrong-kind / grid mismatch loud); the calculate
    assembly EXACT against the stubs (P_lin = exp(net) * base; the
    low-k blend pins boost -> 1 below k_t; P_nl = B * P_lin); the
    legacy state keys; get_Pk_grid / get_Pk_interpolator round-trip at
    the grid nodes; conventional sigma8 has an independent analytic
    known answer, receives h=H0/100, requires exact z=0, and refuses
    incomplete or under-resolved k grids; a non-positive spectrum rejects
    the point (False);
  - the NPCE check_npce leg (the 2026-07-12 family-wide ruling): the
    residual base + refiner algebra bitwise under the diagonal metric,
    the base alive, the state round-trip, save -> rebuild -> predict
    composing base + net bitwise, the diagonal ratio-form rejection;
  - validate_grid2d legs (law-quantity pairing, base files both ways,
    k_stride, transfer ACCEPTED since the 2026-07-12 symmetry ruling);
  - finetune: epoch-0 parity from a grid2d source; the
    wrong-kind and metadata-mismatch from_config errors.
"""

import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import h5py
import numpy as np
import torch

from ai.gates.checks.artifact_fixtures import one_pass_training_recipe
import yaml

# Captured BEFORE any check stubs cobaya.log: the installed Cobaya base
# getter signatures, so the protocol-guard leg pins emul_mps's public
# nonlinear default against upstream (unit 69 / 20M-01).
try:
    from cobaya.theories.cosmo import BoltzmannBase as _BoltzmannBase
    _BASE_PK_SIGS = {name: inspect.signature(getattr(_BoltzmannBase, name))
                     for name in ("get_Pk_grid", "get_Pk_interpolator")}
except Exception:                                        # pragma: no cover
    _BASE_PK_SIGS = None

import emulator.designs.plain as plain_designs
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.grid2d import Grid2DGeometry, TARGET_LAWS_2D
from emulator.geometries.parameter import ParamGeometry
from emulator.data_staging import load_source
from emulator.experiment import EmulatorExperiment, validate_grid2d
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor
from emulator import fixed_facts
from emulator import warmstart

FAILURES = []
IN_NAMES = ["As", "H0", "omch2"]
N_IN = len(IN_NAMES)
Z4 = np.array([0.0, 0.5, 1.0, 2.0])
K6 = np.logspace(-4, 0.0, 6)
GRID2D_MASK_DECLARATION = "dv_geometry_const_mask_sha256"


def _write_paramnames(params_path, names=IN_NAMES):
    """Write the chain-root sidecar required by named-column staging."""
    stem = os.path.splitext(os.fspath(params_path))[0]
    root, chain = os.path.splitext(stem)
    if chain[1:].isdigit():
        stem = root
    with open(stem + ".paramnames", "w") as handle:
        for name in names:
            handle.write(name + " " + name + "\n")
        handle.write("chi2* chi2\n")


def _write_facts(params_path, label, names=IN_NAMES):
    """Write the scientific record required by an accepted staging fixture."""
    stem = os.path.splitext(os.fspath(params_path))[0]
    root, chain = os.path.splitext(stem)
    if chain[1:].isdigit():
        stem = root
    with open(stem + fixed_facts.SIDECAR_SUFFIX, "w") as handle:
        handle.write(fixed_facts.synthetic_sidecar(
            names=names, label=label, family="grid2d", support=None))


class MaskDeclarationModelConstructionReached(Exception):
    """The const-mask artifact reader reached model construction."""


class MaskDeclarationConstructionSentinel:
    """Constructor sentinel proving declaration refusal happens first."""

    def __init__(self, *args, **kwargs):
        raise MaskDeclarationModelConstructionReached(
            "changed mask reached model construction")


# The adapter legs serve a linear-power artifact and a boost artifact side by
# side, and multiply them into the one nonlinear spectrum: emul_mps refuses the
# set outright when either half is missing. They are the two halves of one
# matter-power dataset, so they carry one identity, which is what handing them
# the same label does.
ADAPTER_PAIR_LABEL = "mps-identity/adapter-power-pair"

# The region the doubles this gate PREDICTS THROUGH declare they stand for.
#
# An emulator may only be asked about a point inside the region it was trained
# over: outside it the network does not fail, it extrapolates, and it returns a
# power spectrum of the right shape and the right sign that is wrong. So a
# double the gate predicts through has to declare the box a real matter-power
# emulator of this shape would have been drawn from, and it must contain the
# points the gate asks about.
#
# There are TWO boxes because the two call sites hand the emulator its amplitude
# in two different units, and a record declares the region the emulator may be
# ASKED about. The round-trip / head / NPCE fixtures build their parameter
# geometry around As = 2.1 -- the 1e-9-scaled amplitude the syren formulas take
# -- and ask about As = 2.15. The adapter is driven the way cobaya drives it,
# with the cosmology's own As = 2.1e-9. Each box is therefore written in the
# units its own call site asks in; H0 and omch2 are the same physical interval
# in both, and both boxes contain the synthetic training rows.
#
# A double that is only saved, rebuilt, or refused -- never asked a question --
# declares NO support instead. That is the honest record for it, and
# check_domain_law proves such a double answers nothing.
GRID2D_SUPPORT = {"As":    (1.6, 2.6),
                  "H0":    (60.0, 76.0),
                  "omch2": (0.09, 0.15)}

ADAPTER_SUPPORT = {"As":    (1.6e-9, 2.6e-9),
                   "H0":    (60.0, 76.0),
                   "omch2": (0.09, 0.15)}


def supported_test_record(names, label, family, support):
    """Write this gate's fixed support bounds as literal decimal strings.

    CoCoA uses NumPy 1. A validation environment may contain NumPy 2, whose
    float32 representation includes ``np.float32(...)``. That text is not a
    decimal number. Keeping this conversion inside the synthetic gate avoids
    changing the production decimal policy for an unsupported environment.
    """
    blocks = yaml.safe_load(fixed_facts.synthetic_sidecar(
        names=names, label=label, family=family, support=None))
    domain = blocks[fixed_facts.INPUT_DOMAIN_GROUP]
    domain["constraint"] = "box"
    for key in ("requested", "resolved"):
        domain[key] = {
            name: [str(support[name][0]), str(support[name][1])]
            for name in names
        }
    fixed_facts.validate(blocks, where="the matter-power test record")
    return yaml.safe_dump(blocks, default_flow_style=False, sort_keys=False)


def report(label, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


def report_refusal(label, error, needle, law):
    """Report one refusal leg: the adapter raised AND named its own law.

    A bare `except ValueError` accepts ANY refusal. That is not enough here,
    because the adapter has several laws that refuse the same call, and one of
    them fires before the others: a pair of artifacts whose scientific records
    disagree is refused on IDENTITY, before the check this leg exists to prove
    is ever reached. A leg that only asks "did something raise?" would go green
    on that unrelated refusal and the law it names would go untested forever.

    So each leg demands a substring only its own law's message carries. A raise
    with any other message is a RED leg, and the detail line prints the message
    the adapter really produced, so the reader is not left guessing which law
    fired.

    Arguments:
      label  = the leg name, exactly as the board's evidence map carries it.
      error  = the ValueError the arm caught.
      needle = the substring only this law's refusal message contains.
      law    = the law's name, spelled the way the PASS line should read it.
    """
    text = str(error)
    if needle in text:
        report(label, True, "ValueError names " + law)
    else:
        report(label, False, "refused the WRONG law: " + text)


# (queue 2) the seven board-declared evidence legs this check emits, in the
# main() order. Each leg rolls up one contiguous group of check_* functions;
# a leg's single '##AID <aid> <PASS|FAIL>' terminal aggregates every report()
# its group made WITHOUT threading a per-report leg argument (the FAILURES
# snapshot in main() does the roll-up). One terminal per declared leg -- NOT
# one per probe. The child's exit status stays the single aggregate verdict.
# run_board folds these seven lines into the gate's executed set and
# reconciles them against gate_mps_a's declared evidence map. Note: the
# unit-63 const-mask real-artifact checks (check_const_mask_artifact) fold
# under geometry-laws-and-pins per the binding batch-5 seam ruling.
LEG_AIDS = [
    "mps-identity.geometry-laws-and-pins",
    "mps-identity.bounded-staging-values",
    "mps-identity.stable-streamed-moments",
    "mps-identity.staging-file-lifecycle",
    "mps-identity.saved-model-variants",
    "mps-identity.adapter-assembly-and-defaults",
    "mps-identity.config-and-finetune",
]


def emit_leg(aid, failures_before):
    """Print the one reserved '##AID <aid> <result>' line for a leg group.

    Called in main() right after a leg's group of check_* functions returns.
    The leg PASSes only when no report() in that group appended to FAILURES
    since the snapshot; any failing probe in the group reds this one terminal.

    Arguments:
      aid             = the board-declared assertion id for this leg (a member
                        of LEG_AIDS).
      failures_before = len(FAILURES) captured immediately before the leg's
                        group of check_* calls ran.
    """
    mark = "FAIL" if len(FAILURES) > failures_before else "PASS"
    print("##AID " + aid + " " + mark)


def write_covmat(path, names, seed):
    g = np.random.default_rng(seed)
    a = g.standard_normal((len(names), len(names)))
    cov = a @ a.T + len(names) * np.eye(len(names))
    with open(path, "w") as f:
        f.write("# " + " ".join(names) + "\n")
        for row in cov:
            f.write(" ".join(repr(float(x)) for x in row) + "\n")


def synth_rows(n, z=Z4, k=K6, seed=5):
    """Law-space-like synthetic rows over the (z, k) surface."""
    g = np.random.default_rng(seed)
    base = np.log(1.0 + (k[None, :] * (1 + z[:, None])))
    rows = base.reshape(1, -1) + 0.1 * g.standard_normal(
        (n, z.size * k.size))
    return rows.astype("float32")


def stored_float32_reference(law_rows):
    """Recreate the payload and mean that the staging code owns.

    ``law_rows`` is the independent calculation in float64.  The producer
    stores each law-space row as float32 before it computes normalization
    statistics.  Converting before the mean matters because the conversion
    rounds every element separately.  Taking a float64 mean first and
    converting only the final answer performs the operations in a different
    order and can produce different bits.

    Returns:
      stored_rows = the independent rows after the producer's float32
                    storage conversion.
      stored_mean = the column mean of those stored rows.  The accumulation
                    uses float64, then the result returns to the float32
                    dtype persisted in ``src["dv_mean"]``.
    """
    stored_rows = law_rows.astype("float32")
    rows_for_accumulation = stored_rows.astype("float64")
    mean_in_float64 = rows_for_accumulation.mean(axis=0)
    stored_mean = mean_in_float64.astype("float32")
    return stored_rows, stored_mean


def check_geometry(device):
    Y = synth_rows(400)
    geom = Grid2DGeometry.from_targets(device=device, targets=Y, z=Z4,
                                       k=K6, quantity="pklin",
                                       units="Mpc3", law="syren_linear")
    t = torch.randn(6, Z4.size * K6.size)
    back = geom.encode(geom.decode(t))
    rel = (back - t).abs().max().item()
    report("standardize round-trip to float32 round-off", rel < 1e-4,
           "max |d| %.1e" % rel)
    st0 = geom.state()
    geom2 = Grid2DGeometry.from_state(device=device, state=st0)
    st1 = geom2.state()
    ok = set(st0) == set(st1)
    for kk in st0:
        a, b = st0[kk], st1[kk]
        ok = ok and (torch.equal(a, b) if isinstance(a, torch.Tensor)
                     else a == b)
    report("grid2d state round-trip byte-identical", ok,
           "%d keys" % len(st0))
    try:
        Grid2DGeometry.from_targets(device=device, targets=Y[:, :-1],
                                    z=Z4, k=K6, quantity="pklin",
                                    units="Mpc3", law="none")
        report("width guard raises", False, "no raise")
    except ValueError:
        report("width guard raises", True, "ValueError")
    # (run-10 regression-pass catch: the old "un-standardizable guard
    # raises" leg lived here, feeding ONE constant column under law
    # "none" and expecting the pre-amendment raise. The constant-column
    # pin went law-agnostic in board runs 7-8 — a partial-constant
    # column is PINNED under any law, asserted by the pin legs below,
    # and the dead-dump raise now needs a WHOLLY constant surface,
    # asserted by the last leg. The stale expectation is deleted, not
    # reworded.)
    try:
        Grid2DGeometry.from_targets(device=device, targets=Y, z=Z4,
                                    k=K6, quantity="pklin",
                                    units="Mpc3", law="nope")
        report("unknown law raises", False, "no raise")
    except ValueError:
        report("unknown law raises", True, "ValueError")
    # The constant-column pin (amended law-agnostic after the gate's
    # law-none boost training hit it): a constant law-space column that
    # is not the whole surface is PHYSICS (boost = 1 below the
    # nonlinear scale for every cosmology, under ANY law) — pinned
    # (scale 1, decode returns the training constant, mask persisted),
    # never rejected; a WHOLLY constant surface still dies loudly for
    # every law.
    for law_c in ("syren_halofit", "none"):
        Yc = Y.copy()
        Yc[:, 3] = 7.0
        geom_c = Grid2DGeometry.from_targets(device=device, targets=Yc,
                                             z=Z4, k=K6,
                                             quantity="boost",
                                             units="dimensionless",
                                             law=law_c)
        t = torch.randn(5, Z4.size * K6.size)
        dec = geom_c.decode(t)
        st_c = geom_c.state()
        geom_c2 = Grid2DGeometry.from_state(device=device, state=st_c)
        dec2 = geom_c2.decode(t)
        ok = (geom_c.const_mask is not None
              and int(geom_c.const_mask.sum()) == 1
              and bool(geom_c.const_mask[3])
              and float(geom_c.scale[3]) == 1.0
              and bool((dec[:, 3] == geom_c.center[3]).all())
              and "const_mask" in st_c
              and torch.equal(dec, dec2))
        report("constant column pinned + round-trip (%s)" % law_c,
               ok, "1 pin, scale 1.0, decode = the constant, state rides")
    try:
        Yall = np.tile(Y[:1], (Y.shape[0], 1))
        Grid2DGeometry.from_targets(device=device, targets=Yall, z=Z4,
                                    k=K6, quantity="boost",
                                    units="dimensionless",
                                    law="none")
        report("wholly constant surface still raises", False,
               "no raise")
    except ValueError as e:
        report("wholly constant surface still raises",
               "EVERY grid point" in str(e), "names the dead dump")


def check_staging(tmp):
    """The real load_source + _grid2d_law_rows path, base-aligned."""
    nz, nk = Z4.size, K6.size
    n = 60
    g = np.random.default_rng(11)
    raw = np.exp(g.normal(0.0, 1.0, (n, nz * nk))).astype("float32")
    base = np.exp(g.normal(0.0, 1.0, (n, nz * nk))).astype("float32")
    np.save(os.path.join(tmp, "st_dv.npy"), raw)
    np.save(os.path.join(tmp, "st_base.npy"), base)
    np.save(os.path.join(tmp, "st_z.npy"), Z4)
    np.save(os.path.join(tmp, "st_k.npy"), K6)
    # a params .txt in the load_source layout (weight, lnp, 3 params,
    # trailing chi2 column).
    cols = np.column_stack([np.ones(n), np.zeros(n),
                            g.normal(2.1, 0.1, n),
                            g.normal(67.0, 2.0, n),
                            g.normal(0.12, 0.005, n),
                            np.zeros(n)])
    st_params = os.path.join(tmp, "st_params.1.txt")
    np.savetxt(st_params, cols)
    _write_paramnames(st_params)
    _write_facts(st_params, "mps-staging")
    st_failure = os.path.join(tmp, "st_fail.txt")
    with open(st_failure, "w", encoding="ascii") as handle:
        for _ in range(n):
            handle.write("0\n")
    gen = torch.Generator().manual_seed(3)
    src = load_source(dv_path=os.path.join(tmp, "st_dv.npy"),
                      params_path=os.path.join(tmp, "st_params.1.txt"),
                      names=IN_NAMES, omegabh2_hi=None, n_keep=40,
                      gen=gen, ram_frac=0.7, with_means=True,
                      verbose=False, failure_mask_path=st_failure)
    exp = EmulatorExperiment.__new__(EmulatorExperiment)
    exp.grid2d = {"quantity": "pklin",
                  "units": "Mpc3",
                  "law": "syren_linear",
                  "z_file": os.path.join(tmp, "st_z.npy"),
                  "k_file": os.path.join(tmp, "st_k.npy"),
                  "k_stride": 2}
    exp.data = {"ram_frac": 0.7}
    exp.log = lambda *a, **k: None
    exp._grid2d_z = None
    exp._grid2d_k = None
    exp._grid2d_center = None
    exp._grid2d_scale = None
    dump_rows = np.array(src["dump_rows"])
    seeded_disk_rows = dump_rows[np.asarray(src["idx"])]
    seeded_local_rows = np.asarray(src["idx"]).copy()
    exp._grid2d_law_rows(src=src, base_path=os.path.join(tmp,
                                                         "st_base.npy"),
                         with_means=True)
    kept_k = np.unique(np.concatenate([np.arange(0, nk, 2),
                                       np.array([nk - 1])]))
    cols_idx = (np.arange(nz)[:, None] * nk
                + kept_k[None, :]).reshape(-1)
    independent_law_rows = np.log(
        raw[dump_rows].astype("float64")
        / base[dump_rows].astype("float64")
    )[:, cols_idx]
    stored_reference_rows, stored_reference_mean = (
        stored_float32_reference(independent_law_rows)
    )
    ok = (np.array_equal(src["dv"], stored_reference_rows)
          and np.array_equal(exp._grid2d_k, K6[kept_k])
          and src["dv"].shape == (40, nz * kept_k.size)
          and np.array_equal(src["idx"], seeded_local_rows)
          and np.array_equal(dump_rows[src["idx"]], seeded_disk_rows)
          and not np.array_equal(src["idx"], np.arange(40))
          and np.array_equal(
              src["dv"][src["idx"]],
              stored_reference_rows[seeded_local_rows],
          )
          and (nk - 1) in kept_k)
    report("staging law transform: seeded order + base alignment + stride",
           ok, "shape %s; plain arange rejected" % (src["dv"].shape,))
    ok2 = np.array_equal(src["dv_mean"], stored_reference_mean)
    report("staging recomputes dv_mean over law rows", ok2, "")
    # positivity guard
    bad = raw.copy()
    bad[3, 5] = 0.0
    np.save(os.path.join(tmp, "st_bad.npy"), bad)
    gen = torch.Generator().manual_seed(3)
    src2 = load_source(dv_path=os.path.join(tmp, "st_bad.npy"),
                       params_path=os.path.join(tmp, "st_params.1.txt"),
                       names=IN_NAMES, omegabh2_hi=None, n_keep=40,
                       gen=gen, ram_frac=0.7, with_means=True,
                       verbose=False, failure_mask_path=st_failure)
    try:
        exp._grid2d_law_rows(src=src2,
                             base_path=os.path.join(tmp, "st_base.npy"),
                             with_means=True)
        report("staging positivity guard raises", False, "no raise")
    except ValueError:
        report("staging positivity guard raises", True, "ValueError")


class _GuardProxy:
    """A memmap-like read instrument for the bounded grid2d staging.

    Wraps a raw or base array and records every __getitem__, FAILING a
    read that (a) is not a paired (rows, columns) advanced index — the
    old whole-row-block access — (b) asks for more rows than the chunk
    bound, or (c) touches a column outside the kept set. So the leg
    fails against any implementation that materializes the unthinned
    selection or reads a whole row block before thinning.

    Arguments:
      arr         = the wrapped raw / base array (ndarray or memmap).
      kept_cols   = the flattened kept-column indices (the thinning).
      chunk_bound = the maximum rows one read may request.
      reads       = a list the proxy appends (n_rows, n_cols) to per
                    read, so the leg can assert every read was bounded
                    and thinned.
    """

    def __init__(self, arr, kept_cols, chunk_bound, reads):
        self._arr   = arr
        self._kept  = set(np.asarray(kept_cols).reshape(-1).tolist())
        self._bound = int(chunk_bound)
        self._reads = reads

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ndim(self):
        return self._arr.ndim

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise AssertionError(
                "guarded read: a whole-row-block access (key "
                + type(key).__name__ + ") — the bounded staging must "
                "index (rows, kept_cols) together, never a full-width "
                "row block")
        rows = np.asarray(key[0]).reshape(-1)
        cols = np.asarray(key[1]).reshape(-1)
        if int(rows.size) > self._bound:
            raise AssertionError(
                "guarded read: " + str(int(rows.size)) + " rows exceeds "
                "the chunk bound " + str(self._bound))
        touched = set(cols.tolist())
        if not touched.issubset(self._kept):
            raise AssertionError(
                "guarded read: touched unthinned columns "
                + str(sorted(touched - self._kept)[:6]))
        self._reads.append((int(rows.size), int(cols.size)))
        return np.asarray(self._arr[key])


def _reads_ok(reads, bound, width):
    """True when the recorded reads prove real chunking: more than one
    read, each within the row bound and exactly the thinned width."""
    if len(reads) < 2:
        return False
    for n_rows, n_cols in reads:
        if n_rows > bound or n_cols != width:
            return False
    return True


def check_bounded_staging(tmp):
    """The bounded grid2d law transform, the production shape.

    Runs _grid2d_law_rows on a 122 x 2,000 grid, k_stride 10, with a
    tiny memory budget and GUARDED reads: it proves every raw and base
    read is row-chunked and column-thinned (never the unthinned
    50,000 x 244,000 selection the old code cast to float64 twice),
    that the values and streamed mean equal an independent known-answer
    calculation, and that the low-RAM result is genuinely disk-backed.
    Part two runs the REAL load_source(stage_dv=False) so the genuine
    np.memmap source branch and the ram_frac resident/disk toggle are
    exercised end to end.
    """
    import emulator.experiment as expmod
    nz, nk = 122, 2000
    z = np.linspace(0.0, 3.0, nz)
    k = np.logspace(-4.0, 1.0, nk)
    np.save(os.path.join(tmp, "bs_z.npy"), z)
    np.save(os.path.join(tmp, "bs_k.npy"), k)
    stride = 10
    kept_k = np.unique(np.concatenate([np.arange(0, nk, stride),
                                       np.array([nk - 1])]))
    cols  = (np.arange(nz)[:, None] * nk + kept_k[None, :]).reshape(-1)
    width = int(cols.shape[0])

    # --- part 1: guarded reads + known answer + disk-backed result ---
    g = np.random.default_rng(707)
    n_raw, n_used = 60, 48
    raw_full = np.exp(
        g.normal(0.0, 1.0, (n_raw, nz * nk))).astype("float32")
    base_full = np.exp(
        g.normal(0.0, 1.0, (n_raw, nz * nk))).astype("float32")
    dump_rows = np.sort(g.choice(n_raw, size=n_used, replace=False))
    raw_compact = raw_full[dump_rows]              # the resident-form raw
    C_compact = g.normal(0.0, 1.0, (n_used, N_IN)).astype("float32")
    base_path = os.path.join(tmp, "bs_base.npy")
    np.save(base_path, base_full)

    # The independent law formula uses float64 so its arithmetic does not
    # borrow the producer's implementation.  The producer then stores the
    # result as float32 before it calculates the mean.  The helper below
    # repeats that representation order without calling the staging path.
    independent_law_rows = np.log(
        raw_compact[:, cols].astype("float64")
        / base_full[dump_rows][:, cols].astype("float64")
    )
    stored_reference_rows, stored_reference_mean = (
        stored_float32_reference(independent_law_rows)
    )

    # shrink the per-chunk budget so 48 rows split into several chunks,
    # and size the guard bound to the code's derived chunk height.
    saved_budget = expmod._GRID2D_CHUNK_BYTES
    expmod._GRID2D_CHUNK_BYTES = width * 8 * 16
    bound = max(1, expmod._GRID2D_CHUNK_BYTES // (width * 8))
    raw_reads, base_reads = [], []
    seeded_local = np.roll(np.arange(n_used, dtype="int64"), 7)
    src = {"C": C_compact,
           "dv": _GuardProxy(raw_compact, cols, bound, raw_reads),
           "idx": seeded_local.copy(),
           "dump_rows": dump_rows,
           "source_n_rows": n_raw}
    exp = EmulatorExperiment.__new__(EmulatorExperiment)
    exp.grid2d = {"quantity": "pklin", "units": "Mpc3",
                  "law": "syren_linear",
                  "z_file": os.path.join(tmp, "bs_z.npy"),
                  "k_file": os.path.join(tmp, "bs_k.npy"),
                  "k_stride": stride}
    exp.data = {"ram_frac": 1e-12}                 # force disk-backed
    exp.log = lambda *a, **k: None
    exp._grid2d_z = None
    exp._grid2d_k = None
    exp._grid2d_center = None
    exp._grid2d_scale = None

    real_load = np.load

    def guarded_load(path, *a, **kw):
        arr = real_load(path, *a, **kw)
        if os.path.abspath(path) == os.path.abspath(base_path):
            return _GuardProxy(arr, cols, bound, base_reads)
        return arr

    np.load = guarded_load
    try:
        exp._grid2d_law_rows(src=src, base_path=base_path,
                             with_means=True)
    finally:
        np.load = real_load
        expmod._GRID2D_CHUNK_BYTES = saved_budget

    got = np.asarray(src["dv"])
    report("bounded staging: 122 x 201 kept columns (24522 wide)",
           got.shape == (n_used, nz * kept_k.size) and kept_k.size == 201,
           "shape %s" % (got.shape,))
    report(
        "bounded staging: values equal the direct known answer",
        np.array_equal(got, stored_reference_rows),
        "max|d| %.1e"
        % float(np.abs(got - stored_reference_rows).max()),
    )
    report(
        "bounded staging: resident loader preserves seeded row order",
        np.array_equal(src["idx"], seeded_local)
        and not np.array_equal(src["idx"], np.arange(n_used))
        and np.array_equal(
            src["dump_rows"][src["idx"]],
            dump_rows[seeded_local],
        )
        and np.array_equal(
            np.asarray(src["dv"])[src["idx"]],
            stored_reference_rows[seeded_local],
        ),
        "nontrivial local permutation survives; plain arange rejected",
    )
    report(
        "bounded staging: streamed mean equals the known answer",
        np.array_equal(src["dv_mean"], stored_reference_mean),
        "stored-float32 payload, float64 accumulation",
    )
    # Mutation control: the former reference reversed the two operations.
    # It averaged the unrounded float64 formula rows first, then converted
    # only the final mean to float32.  The seeded fixture must distinguish
    # that wrong order from the mean of the rows the producer actually stores.
    pre_cast_mean = independent_law_rows.mean(axis=0).astype("float32")
    pre_cast_difference = np.abs(
        src["dv_mean"].astype("float64")
        - pre_cast_mean.astype("float64")
    )
    report(
        "bounded staging: mean-before-cast mutation is rejected",
        not np.array_equal(src["dv_mean"], pre_cast_mean),
        "max|d| %.9e" % float(pre_cast_difference.max()),
    )
    report("bounded staging: every raw + base read chunked and thinned",
           _reads_ok(raw_reads, bound, width)
           and _reads_ok(base_reads, bound, width),
           "raw %d reads, base %d reads, <= %d rows x %d cols"
           % (len(raw_reads), len(base_reads), bound, width))
    report("bounded staging: the tiny-budget result is disk-backed",
           isinstance(src["dv"], np.memmap), type(src["dv"]).__name__)
    tripped = False
    probe = _GuardProxy(raw_compact, cols, bound, [])
    try:
        _ = probe[np.arange(n_used)]               # the old mm[rows] read
    except AssertionError:
        tripped = True
    report("bounded staging: guard trips on a whole-selection read",
           tripped, "old mm[rows] pattern rejected")

    # --- part 2: the real load_source(stage_dv=False) memmap branch ---
    nz2, nk2 = Z4.size, K6.size
    np.save(os.path.join(tmp, "bs2_z.npy"), Z4)
    np.save(os.path.join(tmp, "bs2_k.npy"), K6)
    g2 = np.random.default_rng(808)
    n2 = 50
    raw2 = np.exp(g2.normal(0.0, 1.0, (n2, nz2 * nk2))).astype("float32")
    base2 = np.exp(g2.normal(0.0, 1.0, (n2, nz2 * nk2))).astype("float32")
    np.save(os.path.join(tmp, "bs2_dv.npy"), raw2)
    np.save(os.path.join(tmp, "bs2_base.npy"), base2)
    txt = np.column_stack([np.ones(n2), np.zeros(n2),
                           g2.normal(2.1, 0.1, n2),
                           g2.normal(67.0, 2.0, n2),
                           g2.normal(0.12, 0.005, n2),
                           np.zeros(n2)])
    bs2_params = os.path.join(tmp, "bs2_params.1.txt")
    np.savetxt(bs2_params, txt)
    _write_paramnames(bs2_params)
    _write_facts(bs2_params, "mps-bounded-staging")
    bs2_failure = os.path.join(tmp, "bs2_fail.txt")
    with open(bs2_failure, "w", encoding="ascii") as handle:
        for _ in range(n2):
            handle.write("0\n")

    def stage_and_transform(ram_frac):
        gen = torch.Generator().manual_seed(9)
        s = load_source(dv_path=os.path.join(tmp, "bs2_dv.npy"),
                        params_path=os.path.join(tmp, "bs2_params.1.txt"),
                        names=IN_NAMES, omegabh2_hi=None, n_keep=40,
                        gen=gen, ram_frac=ram_frac, with_means=True,
                        stage_dv=False, verbose=False,
                        failure_mask_path=bs2_failure)
        raw_is_memmap = isinstance(s["dv"], np.memmap)
        e = EmulatorExperiment.__new__(EmulatorExperiment)
        e.grid2d = {"quantity": "pklin", "units": "Mpc3",
                    "law": "syren_linear",
                    "z_file": os.path.join(tmp, "bs2_z.npy"),
                    "k_file": os.path.join(tmp, "bs2_k.npy"),
                    "k_stride": 1}
        e.data = {"ram_frac": ram_frac}
        e.log = lambda *a, **k: None
        e._grid2d_z = None
        e._grid2d_k = None
        e._grid2d_center = None
        e._grid2d_scale = None
        dump2 = np.array(s["dump_rows"])
        seeded2 = np.array(s["idx"])
        e._grid2d_law_rows(src=s,
                           base_path=os.path.join(tmp, "bs2_base.npy"),
                           with_means=True)
        want2 = np.log(raw2[dump2].astype("float64")
                       / base2[dump2].astype("float64"))
        want_seeded2 = np.log(raw2[seeded2].astype("float64")
                              / base2[seeded2].astype("float64"))
        return s, raw_is_memmap, want2, seeded2, want_seeded2

    src_lo, raw_mm, want2, seeded2, want_seeded2 = (
        stage_and_transform(1e-12)
    )
    report("bounded staging: stage_dv keeps the raw dump a memmap",
           raw_mm, "raw is np.memmap on load")
    report("bounded staging: memmap branch, tiny budget stays disk-backed",
           isinstance(src_lo["dv"], np.memmap)
           and np.allclose(np.asarray(src_lo["dv"]),
                           want2.astype("float32"), rtol=0, atol=0),
           "result %s" % type(src_lo["dv"]).__name__)
    report("bounded staging: memmap loader preserves seeded row order",
           not np.array_equal(src_lo["idx"], np.arange(src_lo["idx"].size))
           and np.array_equal(
               src_lo["dump_rows"][src_lo["idx"]], seeded2)
           and np.array_equal(
               np.asarray(src_lo["dv"])[src_lo["idx"]],
               want_seeded2.astype("float32")),
           "global selection becomes the matching local permutation")
    src_hi, _, _, seeded_hi, want_seeded_hi = stage_and_transform(0.7)
    report("bounded staging: memmap branch, ample budget is resident",
           isinstance(src_hi["dv"], np.ndarray)
           and not isinstance(src_hi["dv"], np.memmap),
           "result %s" % type(src_hi["dv"]).__name__)
    report("bounded staging: RAM/disk results expose identical loader order",
           np.array_equal(seeded_hi, seeded2)
           and np.array_equal(src_hi["idx"], src_lo["idx"])
           and np.array_equal(
               np.asarray(src_hi["dv"])[src_hi["idx"]],
               want_seeded_hi.astype("float32"))
           and np.array_equal(
               np.asarray(src_hi["dv"])[src_hi["idx"]],
               np.asarray(src_lo["dv"])[src_lo["idx"]]),
           "same seeded cosmology at every loader position")

    # Exact row equality is a scientific identity, not a lower-bound check.
    # N-1 and N+1 siblings can still contain every selected row, so merely
    # checking max(dump_rows) would accept them. This three-way fixture makes
    # only the exact N-row sibling legal.
    base_count_results = []
    for delta in (-1, 0, 1):
        probe_path = os.path.join(tmp, "bs2_base_count_%+d.npy" % delta)
        if delta < 0:
            probe_base = base2[:n2 + delta]
        elif delta > 0:
            probe_base = np.concatenate(
                [base2, np.ones((delta, base2.shape[1]), dtype="float32")],
                axis=0,
            )
        else:
            probe_base = base2
        np.save(probe_path, probe_base)
        probe = load_source(
            dv_path=os.path.join(tmp, "bs2_dv.npy"),
            params_path=os.path.join(tmp, "bs2_params.1.txt"),
            names=IN_NAMES,
            omegabh2_hi=None,
            n_keep=40,
            gen=torch.Generator().manual_seed(9),
            ram_frac=0.7,
            with_means=True,
            stage_dv=False,
            verbose=False,
            failure_mask_path=bs2_failure,
        )
        probe_exp = EmulatorExperiment.__new__(EmulatorExperiment)
        probe_exp.grid2d = {"quantity": "pklin", "units": "Mpc3",
                            "law": "syren_linear",
                            "z_file": os.path.join(tmp, "bs2_z.npy"),
                            "k_file": os.path.join(tmp, "bs2_k.npy"),
                            "k_stride": 1}
        probe_exp.data = {"ram_frac": 0.7}
        probe_exp.log = lambda *a, **k: None
        probe_exp._grid2d_z = probe_exp._grid2d_k = None
        probe_exp._grid2d_center = probe_exp._grid2d_scale = None
        try:
            probe_exp._grid2d_law_rows(
                src=probe,
                base_path=probe_path,
                with_means=False,
            )
            base_count_results.append(delta == 0)
        except ValueError as error:
            base_count_results.append(
                delta != 0 and "exactly the raw" in str(error)
            )
    report("bounded staging: base row count accepts only exact N",
           all(base_count_results), "N-1 refuses, N passes, N+1 refuses")


def _run_law_none(tmp, raw_compact, z, k, chunk_bytes=None):
    """Stage a resident-form law-none source through the REAL
    _grid2d_law_rows and return (exp, src) — the streamed moments land on
    exp._grid2d_center / _scale. chunk_bytes overrides the derived chunk
    height so uneven chunkings can be exercised."""
    import emulator.experiment as expmod
    n = int(raw_compact.shape[0])
    np.save(os.path.join(tmp, "sm_z.npy"), z)
    np.save(os.path.join(tmp, "sm_k.npy"), k)
    exp = EmulatorExperiment.__new__(EmulatorExperiment)
    exp.grid2d = {"quantity": "pklin", "units": "Mpc3", "law": "none",
                  "z_file": os.path.join(tmp, "sm_z.npy"),
                  "k_file": os.path.join(tmp, "sm_k.npy"), "k_stride": 1}
    exp.data = {"ram_frac": 0.9}
    exp.log = lambda *a, **k: None
    exp._grid2d_z = None
    exp._grid2d_k = None
    exp._grid2d_center = None
    exp._grid2d_scale = None
    src = {"C": np.zeros((n, N_IN), dtype="float32"),
           "dv": raw_compact.copy(),
           "idx": np.arange(n),
           "dump_rows": np.arange(n),
           "source_n_rows": n}
    saved = expmod._GRID2D_CHUNK_BYTES
    if chunk_bytes is not None:
        expmod._GRID2D_CHUNK_BYTES = int(chunk_bytes)
    try:
        exp._grid2d_law_rows(src=src, base_path=None, with_means=True)
    finally:
        expmod._GRID2D_CHUNK_BYTES = saved
    return exp, src


def check_stable_moments(tmp, device):
    """The streamed geometry moments must reproduce float64
    np.std(ddof 0) of the MATERIALIZED float32 rows. The old
    (s2 - s1^2/n)/n one-pass form drifts by percent on a high-offset
    small-spread column and, under some orderings, flips a genuinely
    varying column to an exact-zero (false constant) pin. These legs fail
    that form and pass the stable Chan/Welford accumulator."""
    # 1) the headline: 50,000 float32 rows alternating 1e8 / 1e8+ULP,
    # true population std exactly 4.0 per column, over several chunk
    # heights. The naive form returns ~3.97 (order-dependent); the stable
    # form returns 4.0 and never a false pin.
    nz, nk = 1, 4
    z = np.array([0.5])
    k = np.logspace(-3.0, 0.0, nk)
    n = 50000
    raw = np.empty((n, nz * nk), dtype="float32")
    for j in range(nz * nk):
        lo = np.float32(1e8 * (j + 1))
        hi = np.nextafter(lo, np.float32(2e8 * (j + 1)))
        raw[0::2, j] = lo
        raw[1::2, j] = hi
    scales = []
    for chunk_bytes in (None, nz * nk * 8 * 7, nz * nk * 8 * 337):
        exp, src = _run_law_none(tmp, raw, z, k, chunk_bytes=chunk_bytes)
        stored = np.asarray(src["dv"], dtype="float64")
        want = stored.std(axis=0, ddof=0)
        report("stable moments: 1e8/1-ULP scale = np.std(ddof 0), no "
               "false pin (chunk %s)" % (chunk_bytes,),
               np.allclose(exp._grid2d_scale, want, rtol=1e-9)
               and bool(np.all(exp._grid2d_scale > 0.0)),
               "scale %s" % np.round(exp._grid2d_scale, 3))
        scales.append(exp._grid2d_scale)
    report("stable moments: uneven chunkings agree (order-stable)",
           np.allclose(scales[0], scales[1], rtol=1e-9)
           and np.allclose(scales[0], scales[2], rtol=1e-9), "")

    # 2) the from_stats pin threshold is RELATIVE: tiny = 8 * eps32 *
    # |center|, about 95.4 at center 1e8. Three columns exercise it and
    # keep zero.size < n_out so the whole-surface dead-dump guard stays
    # out of reach: (col 0) exactly constant -> pins; (col 1) a 1-ULP
    # spread at 1e8, std 4 -- about 4e-8 relative, BELOW float32
    # precision -> pins BY THE RELATIVE RULE, which is correct
    # standardization (the model cannot resolve that spread in float32,
    # so decode should return the constant); (col 2) a std-1024 spread at
    # 1e8, about 10.7x tiny -> resolvable, must NOT pin. The stable
    # accumulator is what makes col 1 read 4 (not a cancellation
    # artifact) and col 2 read 1024 rather than a false zero.
    three = np.empty((4000, 3), dtype="float32")
    three[:, 0] = np.float32(1e8)                      # exactly constant
    lo1 = np.float32(1e8)
    hi1 = np.nextafter(lo1, np.float32(2e8))           # 1-ULP, std 4
    three[0::2, 1] = lo1
    three[1::2, 1] = hi1
    three[0::2, 2] = np.float32(1e8 - 1024.0)          # std 1024, above tiny
    three[1::2, 2] = np.float32(1e8 + 1024.0)
    exp2, _ = _run_law_none(tmp, three, np.array([0.5]),
                            np.logspace(-3.0, 0.0, 3))
    geom2 = Grid2DGeometry.from_stats(
        device=device, center=exp2._grid2d_center,
        scale=exp2._grid2d_scale, z=exp2._grid2d_z, k=exp2._grid2d_k,
        quantity="pklin", units="Mpc3", law="none")
    mask = (geom2.const_mask.cpu().numpy().tolist()
            if geom2.const_mask is not None else None)
    report("stable moments: relative pin threshold (constant + "
           "sub-eps32 spread pin, resolvable spread does not, no "
           "dead-dump crash)",
           geom2.const_mask is not None
           and bool(geom2.const_mask[0])
           and bool(geom2.const_mask[1])
           and not bool(geom2.const_mask[2]),
           "const_mask %s (tiny ~ 95.4 at 1e8)" % (mask,))

    # 3) the ordinary log-ratio fixture: streamed scale = np.std, and the
    # from_stats encode reproduces the materialized standardization.
    g = np.random.default_rng(21)
    nz3, nk3, n3 = 3, 5, 400
    z3 = np.linspace(0.0, 2.0, nz3)
    k3 = np.logspace(-3.0, 0.0, nk3)
    rawL = np.exp(g.normal(0.0, 1.0, (n3, nz3 * nk3))).astype("float32")
    baseL = np.exp(g.normal(0.0, 1.0, (n3, nz3 * nk3))).astype("float32")
    np.save(os.path.join(tmp, "sml_base.npy"), baseL)
    np.save(os.path.join(tmp, "sml_z.npy"), z3)
    np.save(os.path.join(tmp, "sml_k.npy"), k3)
    expL = EmulatorExperiment.__new__(EmulatorExperiment)
    expL.grid2d = {"quantity": "pklin", "units": "Mpc3",
                   "law": "syren_linear",
                   "z_file": os.path.join(tmp, "sml_z.npy"),
                   "k_file": os.path.join(tmp, "sml_k.npy"),
                   "k_stride": 1}
    expL.data = {"ram_frac": 0.9}
    expL.log = lambda *a, **k: None
    expL._grid2d_z = None
    expL._grid2d_k = None
    expL._grid2d_center = None
    expL._grid2d_scale = None
    dumpL = np.arange(n3)
    srcL = {"C": np.zeros((n3, N_IN), dtype="float32"),
            "dv": rawL.copy(), "idx": dumpL, "dump_rows": dumpL,
            "source_n_rows": n3}
    expL._grid2d_law_rows(src=srcL,
                          base_path=os.path.join(tmp, "sml_base.npy"),
                          with_means=True)
    storedL = np.asarray(srcL["dv"], dtype="float64")
    wantL = storedL.std(axis=0, ddof=0)
    report("stable moments: log-ratio scale = np.std(ddof 0)",
           np.allclose(expL._grid2d_scale, wantL, rtol=1e-9),
           "max rel %.1e"
           % float(np.max(np.abs(expL._grid2d_scale - wantL) / wantL)))
    geomL = Grid2DGeometry.from_stats(
        device=device, center=expL._grid2d_center,
        scale=expL._grid2d_scale, z=expL._grid2d_z, k=expL._grid2d_k,
        quantity="pklin", units="Mpc3", law="syren_linear")
    enc = geomL.encode(
        torch.from_numpy(storedL.astype("float32")).to(device)).cpu().numpy()
    want_enc = (storedL - expL._grid2d_center) / expL._grid2d_scale
    report("stable moments: from_stats encode matches the materialized "
           "standardization", np.allclose(enc, want_enc, rtol=1e-5,
                                           atol=1e-5),
           "max|d| %.1e" % float(np.abs(enc - want_enc).max()))


def _g2law_live(baseline):
    """The .g2law.dat temp files created since the `baseline` snapshot
    (a set of paths), and their total bytes — the live disk-backed
    staging footprint the sweep must keep bounded."""
    import glob
    now = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.g2law.dat")))
    new = now - baseline
    total = 0
    for p in new:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return new, total


def _lifecycle_files(tmp, law):
    """Write synthetic grid2d files (law 'none' or 'syren_linear') and
    return (data-block, grid2d-block, raw array) for a minimal
    experiment; a syren law also writes a positive base sibling."""
    nz, nk = 1, 4
    tag = "lc_" + law
    np.save(os.path.join(tmp, tag + "_z.npy"), np.array([0.5]))
    np.save(os.path.join(tmp, tag + "_k.npy"), np.logspace(-3.0, 0.0, nk))
    g = np.random.default_rng(303)
    n = 50
    raw = np.exp(g.normal(0.0, 1.0, (n, nz * nk))).astype("float32")
    np.save(os.path.join(tmp, tag + "_dv.npy"), raw)
    txt = np.column_stack([np.ones(n), np.zeros(n),
                           g.normal(2.1, 0.1, n), g.normal(67.0, 2.0, n),
                           g.normal(0.12, 0.005, n), np.zeros(n)])
    lifecycle_params = os.path.join(tmp, tag + "_params.1.txt")
    np.savetxt(lifecycle_params, txt)
    _write_paramnames(lifecycle_params)
    _write_facts(lifecycle_params, "mps-lifecycle-" + law)
    failure_mask = os.path.join(tmp, tag + "_failed.txt")
    Path(failure_mask).write_text("0\n" * n, encoding="ascii")
    g2 = {"quantity": "pklin", "units": "Mpc3", "law": law,
          "z_file": os.path.join(tmp, tag + "_z.npy"),
          "k_file": os.path.join(tmp, tag + "_k.npy"), "k_stride": 1}
    if law != "none":
        base = np.exp(g.normal(0.0, 1.0, (n, nz * nk))).astype("float32")
        np.save(os.path.join(tmp, tag + "_base.npy"), base)
        g2["train_base"] = os.path.join(tmp, tag + "_base.npy")
        g2["val_base"] = os.path.join(tmp, tag + "_base.npy")
    data = {"split_seed": 0, "n_train": 30, "n_val": 20,
            "train_params": os.path.join(tmp, tag + "_params.1.txt"),
            "train_dv": os.path.join(tmp, tag + "_dv.npy"),
            "train_failure_mask": failure_mask,
            "val_params": os.path.join(tmp, tag + "_params.1.txt"),
            "val_dv": os.path.join(tmp, tag + "_dv.npy"),
            "val_failure_mask": failure_mask}
    return data, g2, raw


def _lifecycle_exp(data, g2, ram_frac):
    """A minimal grid2d experiment whose stage_train / release_* paths
    run for real (no from_config, no GPU)."""
    exp = EmulatorExperiment.__new__(EmulatorExperiment)
    exp._scalar = exp._cmb = exp._grid = False
    exp._grid2d = True
    exp.names = IN_NAMES
    exp.quiet = True
    exp.log = lambda *a, **k: None
    exp._grid2d_z = exp._grid2d_k = None
    exp._grid2d_center = exp._grid2d_scale = None
    exp._grid2d_train_tmp = exp._grid2d_val_tmp = None
    exp.grid2d = dict(g2)
    d = dict(data)
    d["ram_frac"] = ram_frac
    exp.data = d
    return exp


def check_staging_lifecycle(tmp):
    """The disk-backed staging temp-file lifecycle (the reopened c03a084
    close): the experiment OWNS its train / val temp file, supersedes it
    on restage, releases it explicitly for the sweep lane, and never
    orphans a partial file on a failed transform. Runs the REAL
    stage_train / release_train_staging / _grid2d_law_rows paths."""
    import glob
    base0 = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                       "*.g2law.dat")))
    data, g2, raw = _lifecycle_files(tmp, "none")

    # (a) low-RAM staging twice on one experiment: the second staging
    # supersedes the first, so the first temp file is unlinked and the
    # second is live and readable.
    exp = _lifecycle_exp(data, g2, 1e-12)
    exp.stage_train(n_train=30)
    p1 = exp._grid2d_train_tmp
    exp.stage_train(n_train=30)
    p2 = exp._grid2d_train_tmp
    report("lifecycle: restage supersedes (1st file absent, 2nd readable)",
           p1 is not None and p2 is not None and p1 != p2
           and not os.path.exists(p1) and os.path.exists(p2)
           and isinstance(exp.train_set["dv"], np.memmap),
           "p1 gone=%s p2 live=%s" % (not os.path.exists(p1),
                                      os.path.exists(p2)))
    exp.release_train_staging()

    # (b) three N-train sweep points with the lane-style cleanup between
    # them: the live temp count + bytes stay bounded to ONE point, never
    # cumulative.
    exp = _lifecycle_exp(data, g2, 1e-12)
    peak = 0
    for N in (20, 25, 30):
        exp.stage_train(n_train=N)
        live, _ = _g2law_live(base0)
        peak = max(peak, len(live))
        exp.train_set = None
        exp.release_train_staging()          # the sweep lane's cleanup
    live_end, bytes_end = _g2law_live(base0)
    report("lifecycle: 3-point sweep bounded (<=1 live temp, 0 at end)",
           peak <= 1 and len(live_end) == 0 and bytes_end == 0,
           "peak %d live, end %d file(s)" % (peak, len(live_end)))

    # (c) a mid-transform failure (a non-positive value under a syren
    # law, caught inside the chunk loop) leaves NO temp file.
    datab, g2b, rawb = _lifecycle_files(tmp, "syren_linear")
    rawb[:, 2] = 0.0                          # positivity failure
    np.save(datab["train_dv"], rawb)
    exp = _lifecycle_exp(datab, g2b, 1e-12)
    before, _ = _g2law_live(base0)
    raised = False
    try:
        exp.stage_train(n_train=30)
    except ValueError:
        raised = True
    after, _ = _g2law_live(base0)
    report("lifecycle: mid-transform failure leaves no temp file",
           raised and len(after) == len(before)
           and exp._grid2d_train_tmp is None,
           "raised=%s new files=%d" % (raised, len(after) - len(before)))

    # (d) a failed TRAINING after a successful staging releases that
    # point's train file through the lane-style cleanup (the except path
    # reaches the same block).
    exp = _lifecycle_exp(data, g2, 1e-12)
    exp.stage_train(n_train=30)
    pth = exp._grid2d_train_tmp
    ok_setup = pth is not None and os.path.exists(pth)
    exp.train_set = None
    exp.release_train_staging()
    report("lifecycle: failed-point cleanup releases the train file",
           ok_setup and not os.path.exists(pth)
           and exp._grid2d_train_tmp is None,
           "staged=%s file gone=%s" % (ok_setup, not os.path.exists(pth)))

    # (e) resident-RAM control: no temp file, and the staged values are
    # byte-identical to the direct known answer (law none = raw
    # passthrough over the kept columns).
    exp = _lifecycle_exp(data, g2, 0.9)
    before_e, _ = _g2law_live(base0)
    exp.stage_train(n_train=30)
    after_e, _ = _g2law_live(base0)
    dump_rows = np.array(exp.train_set["dump_rows"])
    want = raw[dump_rows].astype("float32")     # stride 1 -> all columns
    report("lifecycle: resident control makes no temp file, byte-identical",
           exp._grid2d_train_tmp is None
           and len(after_e) == len(before_e)
           and not isinstance(exp.train_set["dv"], np.memmap)
           and np.array_equal(np.asarray(exp.train_set["dv"]), want),
           "no temp=%s bitwise=%s"
           % (exp._grid2d_train_tmp is None,
              np.array_equal(np.asarray(exp.train_set["dv"]), want)))
    exp.release_train_staging()


def grid2d_recipe(width):
    return {"cls": "emulator.designs.plain.ResMLP",
            "name": "resmlp",
            "ia": None,
            "input_dim": N_IN,
            "output_dim": width,
            "compile_mode": None,
            "needs_geom": False,
            "kwargs": {"int_dim_res": 16,
                       "n_blocks": 2,
                       "block_opts": {"n_layers": 2,
                                      "act": {"type": "H", "n_gates": 3},
                                      "norm": "affine"}}}


def save_synthetic_grid2d(
        root,
        device,
        tmp,
        label,
        quantity="pklin",
        units="Mpc3",
        law="syren_linear",
        z=Z4,
        k=K6,
        seed=0,
        pin_low_k=False,
        support=None,
        transfer_base=None):
    """Build, then save, a tiny synthetic grid2d emulator under `root`.

    `support` is the region the double stands for, as a mapping name -> (low,
    high). A double the gate PREDICTS through declares one -- the box a real
    emulator of this shape would have been drawn from -- because a prediction is
    refused unless the point lies inside the declared region. A double that is
    only saved, rebuilt or refused declares none: it is a test double, it is
    never asked a point, and the record says so.

    Arguments:
      root      = the path root the artifact is saved under.
      device    = the torch device the geometries and the model are built on.
      tmp       = the tempdir the fixture covmat is written into.
      label     = what this double is for; it fixes the record's identity.
      quantity  = the grid2d quantity the double serves ("pklin" / "boost").
      units     = the quantity's units, as the record carries them.
      law       = the target law the geometry was built under.
      z, k      = the (z, k) grid the surface lives on.
      seed      = the fixture seed (the covmat and the synthetic rows).
      pin_low_k = pin the first wavenumber of every redshift row to one (the
                  valid low-k boost identity).
      support   = the box the double declares, or None for no support at all.
      transfer_base = optional embedded base mapping passed to save_emulator.

    Returns:
      (pgeom, geom, model, covmat): the sources the caller compares against, and
      the covmat path the finetune legs re-read.
    """
    covmat = os.path.join(tmp, "g2_%d.covmat" % seed)
    write_covmat(covmat, IN_NAMES, seed=seed + 1)
    pgeom = ParamGeometry.from_covmat(
        device=device, center=np.array([2.1, 67.0, 0.12]),
        covmat_path=covmat)
    Y = synth_rows(400, z=z, k=k, seed=seed + 2)
    if pin_low_k:
        if quantity != "boost" or law != "none":
            raise ValueError(
                "the synthetic low-k pin belongs to a boost with law none")
        # Flattening is redshift outer and wavenumber inner. The first
        # wavenumber in every redshift row therefore has index iz * nk.
        # Setting those complete training columns to one creates the valid
        # low-k boost identity used by the artifact persistence check.
        first_k_indices = np.arange(z.size) * k.size
        Y[:, first_k_indices] = 1.0
    geom = Grid2DGeometry.from_targets(device=device, targets=Y, z=z,
                                       k=k, quantity=quantity,
                                       units=units, law=law)
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=z.size * k.size,
                   int_dim_res=16, n_blocks=2,
                   block_opts=block_opts).to(device)
    config = {"data": {"grid2d": {"quantity": quantity,
                                  "units": units,
                                  "law": law,
                                  "z_file": "z.npy",
                                  "k_file": "k.npy"},
                       "train_dv": "t.npy",
                       "val_dv": "v.npy",
                       "train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat)},
              "train_args": {"nepochs": 1}}
    if law != "none":
        config["data"]["grid2d"]["train_base"] = "tb.npy"
        config["data"]["grid2d"]["val_base"] = "vb.npy"
    histories = {"train_losses": [0.1],
                 "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    # A saved emulator now carries the science it was born under. This one was
    # born under nothing: no generator produced it, so it declares itself a
    # test double rather than carrying no record at all. The label says what
    # the double is for, and it fixes the identity the record holds: doubles
    # that belong to one dataset are handed the same label, doubles that must
    # be told apart are handed different ones.
    composition_mode = "transfer" if transfer_base is not None else "plain"
    transfer_refined = (
        transfer_base is not None
        and transfer_base.get("drifted_state") is not None)
    resolved_transfer = None
    if transfer_base is not None:
        resolved_transfer = {"form": transfer_base["form"],
                             "space": transfer_base["space"],
                             "source_artifact_id":
                                 transfer_base["source_artifact_id"],
                             "source_checkpoint_sha256":
                                 transfer_base["source_checkpoint_sha256"]}
        if transfer_refined:
            resolved_transfer["refine"] = {
                "fixture": "embedded-drifted-state"}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train=one_pass_training_recipe(
                    thresholds=(0.2, 1.0, 10.0, 100.0)),
                  resolved_model=grid2d_recipe(z.size * k.size),
                  transfer_base=transfer_base,
                  composition_mode=composition_mode,
                  transfer_refined=transfer_refined,
                  resolved_pce=None,
                  resolved_transfer=resolved_transfer,
                  facts_yaml=(fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label=label,
                      family="grid2d",
                      support=None)
                    if support is None else supported_test_record(
                      names=pgeom.state()["names"],
                      label=label,
                      family="grid2d",
                      support=support)),
                  attrs={"rescale": "none", "quantity": quantity})
    return pgeom, geom, model, covmat


def clone_emulator_artifact(source_root, destination_root):
    """Copy both files of one saved emulator for a mutation arm."""
    shutil.copy2(str(source_root) + ".h5", str(destination_root) + ".h5")
    shutil.copy2(str(source_root) + ".emul", str(destination_root) + ".emul")


def require_mask_declaration_refusal(root, device, label, expected_fragments):
    """Require targeted refusal before the artifact constructs its model."""
    original_class = plain_designs.ResMLP
    plain_designs.ResMLP = MaskDeclarationConstructionSentinel
    try:
        rebuild_emulator(str(root), device, compile_model=False)
        report(label, False, "changed artifact rebuilt without refusal")
    except MaskDeclarationModelConstructionReached as error:
        report(label, False, str(error))
    except (KeyError, ValueError) as error:
        message = str(error)
        ok = all(fragment in message for fragment in expected_fragments)
        report(label, ok,
               type(error).__name__ + " names the declaration and repair")
    except Exception as error:
        report(label, False,
               "unexpected " + type(error).__name__ + ": " + str(error))
    finally:
        plain_designs.ResMLP = original_class


def check_const_mask_artifact(tmp, device):
    """Prove the mask and its narrow integrity declaration are load-bearing.

    An all-false mask records that every output coordinate is trainable. A
    true entry records that decode must serve the stored training constant at
    that coordinate. The two states have different scientific behavior, so a
    missing dataset cannot select either one. The co-located unkeyed digest
    catches one-surface additions, toggles and moves; this gate does not call a
    coordinated rewrite of both fields whole-file authentication.
    """
    unpinned_root = os.path.join(tmp, "emul_g2_mask_unpinned")
    save_synthetic_grid2d(
        root=unpinned_root,
        device=device,
        tmp=tmp,
        label="mps-identity/const-mask-unpinned",
        quantity="pklin",
        units="Mpc3",
        law="none",
        seed=131)

    with h5py.File(
            unpinned_root + ".h5",
            "r") as artifact:
        stored_unpinned = artifact["dv_geometry/const_mask"][()]
        unpinned_declaration = artifact.attrs.get(
            GRID2D_MASK_DECLARATION)
    _, _, unpinned_geometry, _ = rebuild_emulator(
        unpinned_root,
        device,
        compile_model=False)
    unpinned_ok = (
        stored_unpinned.dtype == np.dtype("uint8")
        and stored_unpinned.shape == (Z4.size * K6.size,)
        and int(stored_unpinned.sum()) == 0
        and isinstance(unpinned_declaration, str)
        and len(unpinned_declaration) == 64
        and int(unpinned_geometry.const_mask.sum().item()) == 0)
    report(
        "const-mask artifact: unpinned state persists all-false",
        unpinned_ok,
        "stored dtype %s, true entries %d"
        % (stored_unpinned.dtype, int(stored_unpinned.sum())))

    added_root = os.path.join(tmp, "emul_g2_mask_added")
    clone_emulator_artifact(unpinned_root, added_root)
    with h5py.File(added_root + ".h5", "r+") as artifact:
        artifact["dv_geometry/const_mask"][0] = np.uint8(1)
    require_mask_declaration_refusal(
        added_root,
        device,
        "const-mask artifact: add against declared unmasked refuses early",
        ("const_mask", "declaration", "restore"))

    legacy_root = os.path.join(tmp, "emul_g2_mask_legacy_v3")
    clone_emulator_artifact(unpinned_root, legacy_root)
    with h5py.File(legacy_root + ".h5", "r+") as artifact:
        del artifact.attrs[GRID2D_MASK_DECLARATION]
    require_mask_declaration_refusal(
        legacy_root,
        device,
        "const-mask artifact: older schema-v3 declaration absence refuses",
        ("schema-v3", "re-save"))

    foreign_root = os.path.join(tmp, "emul_g2_mask_foreign_class")
    clone_emulator_artifact(unpinned_root, foreign_root)
    with h5py.File(foreign_root + ".h5", "r+") as artifact:
        artifact["dv_geometry"].attrs["cls"] = (
            "emulator.geometries.scalar.ScalarGeometry")
    require_mask_declaration_refusal(
        foreign_root,
        device,
        "const-mask artifact: paired fields on non-Grid2D refuse early",
        ("not a Grid2DGeometry", "restore"))

    pinned_root = os.path.join(tmp, "emul_g2_mask_pinned")
    save_synthetic_grid2d(
        root=pinned_root,
        device=device,
        tmp=tmp,
        label="mps-identity/const-mask-pinned",
        quantity="boost",
        units="dimensionless",
        law="none",
        seed=137,
        pin_low_k=True)

    with h5py.File(
            pinned_root + ".h5",
            "r") as artifact:
        stored_pinned = artifact["dv_geometry/const_mask"][()]
        pinned_declaration = artifact.attrs.get(GRID2D_MASK_DECLARATION)
    _, _, pinned_geometry, _ = rebuild_emulator(
        pinned_root,
        device,
        compile_model=False)
    expected_pins = np.zeros(Z4.size * K6.size, dtype=np.uint8)
    expected_pins[np.arange(Z4.size) * K6.size] = 1
    probe = torch.full(
        (1, Z4.size * K6.size),
        0.25,
        dtype=torch.float32,
        device=device)
    decoded = pinned_geometry.decode(probe).detach().cpu().numpy()[0]
    pinned_ok = (
        np.array_equal(stored_pinned, expected_pins)
        and isinstance(pinned_declaration, str)
        and len(pinned_declaration) == 64
        and np.array_equal(
            pinned_geometry.const_mask.cpu().numpy(),
            expected_pins.astype(bool))
        and np.array_equal(decoded[expected_pins.astype(bool)],
                           np.ones(Z4.size, dtype=np.float32)))
    report(
        "const-mask artifact: valid low-k pin survives save and rebuild",
        pinned_ok,
        "stored pins %s" % np.flatnonzero(stored_pinned).tolist())

    moved_root = os.path.join(tmp, "emul_g2_mask_moved_pin")
    clone_emulator_artifact(pinned_root, moved_root)
    with h5py.File(moved_root + ".h5", "r+") as artifact:
        dataset = artifact["dv_geometry/const_mask"]
        moved = dataset[()]
        old_index = int(np.flatnonzero(moved)[0])
        new_index = old_index + 1
        moved[old_index] = np.uint8(0)
        moved[new_index] = np.uint8(1)
        dataset[...] = moved
    require_mask_declaration_refusal(
        moved_root,
        device,
        "const-mask artifact: moved pin with equal true-count refuses early",
        ("const_mask", "declaration", "restore"))

    base_root = os.path.join(tmp, "emul_g2_mask_transfer_base_source")
    base_pgeom, base_geom, base_model, _ = save_synthetic_grid2d(
        root=base_root,
        device=device,
        tmp=tmp,
        label="mps-identity/const-mask-transfer-base-source",
        quantity="pklin",
        units="Mpc3",
        law="none",
        seed=139)
    with h5py.File(base_root + ".h5", "r") as source_artifact:
        source_artifact_id = source_artifact.attrs["artifact_id"]
        source_checkpoint_sha256 = source_artifact.attrs["checkpoint_sha256"]
    embedded_base = {
        "recipe": grid2d_recipe(Z4.size * K6.size),
        "model": base_model,
        "state": base_model.state_dict(),
        "param_geometry": base_pgeom,
        "dv_geometry": base_geom,
        "form": "gain",
        "space": "physical",
        "source_artifact_id": source_artifact_id,
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }
    transfer_root = os.path.join(tmp, "emul_g2_mask_transfer")
    save_synthetic_grid2d(
        root=transfer_root,
        device=device,
        tmp=tmp,
        label="mps-identity/const-mask-transfer",
        quantity="pklin",
        units="Mpc3",
        law="none",
        seed=141,
        transfer_base=embedded_base)
    with h5py.File(transfer_root + ".h5", "r") as artifact:
        transfer_group = artifact["transfer_base"]
        transfer_declaration = transfer_group.attrs.get(
            GRID2D_MASK_DECLARATION)
        transfer_mask = transfer_group["dv_geometry/const_mask"][()]
    _, _, _, transfer_info = rebuild_emulator(
        transfer_root, device, compile_model=False)
    rebuilt_base_mask = (
        transfer_info["transfer_base"]["geom"].const_mask.cpu().numpy())
    transfer_ok = (
        isinstance(transfer_declaration, str)
        and len(transfer_declaration) == 64
        and np.array_equal(transfer_mask, rebuilt_base_mask)
        and int(rebuilt_base_mask.sum()) == 0)
    report(
        "const-mask artifact: embedded transfer base writes and reads declaration",
        transfer_ok,
        "declaration=%s, true entries=%d"
        % (isinstance(transfer_declaration, str),
           int(rebuilt_base_mask.sum())))

    transfer_changed_root = os.path.join(
        tmp, "emul_g2_mask_transfer_changed")
    clone_emulator_artifact(transfer_root, transfer_changed_root)
    with h5py.File(transfer_changed_root + ".h5", "r+") as artifact:
        artifact["transfer_base/dv_geometry/const_mask"][0] = np.uint8(1)
    require_mask_declaration_refusal(
        transfer_changed_root,
        device,
        "const-mask artifact: embedded transfer mask toggle refuses early",
        ("transfer_base", "const_mask", "declaration", "restore"))

    with h5py.File(
            pinned_root + ".h5",
            "r+") as artifact:
        del artifact["dv_geometry/const_mask"]
    require_mask_declaration_refusal(
        pinned_root,
        device,
        "const-mask artifact: deleted required mask refuses early",
        ("const_mask", "declaration", "restore"))


def check_roundtrip(tmp, device, law):
    root = os.path.join(tmp, "emul_g2_" + law)
    pgeom, geom, model, _ = save_synthetic_grid2d(
        root, device, tmp, label="mps-identity/round-trip-" + law,
        law=law, seed=30, support=GRID2D_SUPPORT)
    theta = np.array([[2.15, 68.0, 0.121]])
    x = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        ref = geom.decode(model(pgeom.encode(x)))[0].cpu().numpy()
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    ok = (np.array_equal(got["pklin"].reshape(-1), ref)
          and got["pklin"].shape == (Z4.size, K6.size)
          and np.array_equal(got["z"], Z4)
          and np.array_equal(got["k"], K6)
          and getattr(pred, "_grid2d", False)
          and pred.law == law)
    report("predict round-trip bitwise (%s law)" % law, ok,
           "shape %s" % (got["pklin"].shape,))
    _, _, _, info = rebuild_emulator(root, device, compile_model=False)
    report("rebuild info: grid2d flags (%s)" % law,
           info["grid2d"] and info["grid2d_quantity"] == "pklin"
           and info["grid2d_law"] == law and not info["grid"]
           and info["grid_law"] is None,
           "law %s" % info["grid2d_law"])
    return root


def check_domain_law(tmp, device):
    """The domain law at the predictor's door, on a saved grid2d artifact.

    An emulator asked outside the region it was trained over does not fail. It
    extrapolates: a surface of the right shape, positive everywhere, and no
    warning of any kind. predict() therefore proves the point lies inside the
    region the artifact's own record declares before any number reaches the
    network, and these two arms drive both halves of that law through a real
    save + rebuild.

    Both arms read the WORDS of the refusal. They have to: float("n/a") raises
    the same ValueError class every refusal here raises, so an arm that only
    asked "did it raise?" would stay green while the law it names was broken.
    """
    # a double that declares no support: an emulator generated by nobody, valid
    # over nowhere. It saves, it rebuilds -- and it answers nothing.
    root_none = os.path.join(tmp, "emul_g2_undeclared")
    save_synthetic_grid2d(root_none, device, tmp,
                          label="mps-identity/undeclared-double",
                          law="none", seed=150)
    pred_none = EmulatorPredictor(root_none, device, compile_model=False)
    try:
        pred_none.predict({"As": 2.15, "H0": 68.0, "omch2": 0.121})
        report("an undeclared double refuses every prediction", False,
               "a test double answered a question")
    except ValueError as e:
        report_refusal("an undeclared double refuses every prediction", e,
                       needle="declares no support",
                       law="the domain law (no declared support)")

    # a double that declares its box: a point inside is served, a point outside
    # is refused. The served point proves the arm below is not vacuously green
    # (a predictor that refused everything would pass an outside-the-box arm
    # while serving nothing at all).
    root_box = os.path.join(tmp, "emul_g2_boxed")
    save_synthetic_grid2d(root_box, device, tmp,
                          label="mps-identity/boxed-double",
                          law="none", seed=160,
                          support=GRID2D_SUPPORT)
    pred_box = EmulatorPredictor(root_box, device, compile_model=False)
    inside = pred_box.predict({"As": 2.15, "H0": 68.0, "omch2": 0.121})
    report("a point inside the declared box is served",
           inside["pklin"].shape == (Z4.size, K6.size),
           "surface %s" % (inside["pklin"].shape,))
    # H0 = 100 is far outside the box the record declares. Nothing about the
    # power spectrum the network would return there would look wrong.
    try:
        pred_box.predict({"As": 2.15, "H0": 100.0, "omch2": 0.121})
        report("a point outside the declared box refuses", False,
               "extrapolated without a word")
    except ValueError as e:
        report_refusal("a point outside the declared box refuses", e,
                       needle="which is outside it",
                       law="the domain law (outside the declared box)")


def grid2d_head_recipe(width):
    """The model_recipe for the ResCNN head leg: needs_geom
    True (rebuild re-attaches the z-slice split via attach_head_coords),
    every constructor default materialized."""
    return {"cls": "emulator.designs.plain.ResCNN",
            "name": "rescnn",
            "ia": None,
            "input_dim": N_IN,
            "output_dim": width,
            "compile_mode": None,
            "needs_geom": True,
            "kwargs": {"int_dim_res": 16,
                       "n_blocks": 2,
                       "kernel_size": 3,
                       "rescale_kernel": False,
                       "groups": 1,
                       "separable": False,
                       "film": False,
                       "n_blocks_cnn": 1,
                       "gate_init": 0.1,
                       "head_act": None,
                       "block_opts": {"n_layers": 2,
                                      "act": {"type": "H", "n_gates": 3},
                                      "norm": "affine"}}}


def check_head(tmp, device):
    """The correction-head leg: the conv head on the grid2d geometry —
    z slices as channels, the identity basis, the epoch-0 identity
    start, the n_tokens rejection on real physical bins, and save ->
    rebuild -> predict bitwise (proving the rebuild-side
    attach_head_coords in results._rebuild_model)."""
    from emulator.designs.plain import ResCNN, ResTRF
    covmat = os.path.join(tmp, "g2h.covmat")
    write_covmat(covmat, IN_NAMES, seed=41)
    pgeom = ParamGeometry.from_covmat(
        device=device, center=np.array([2.1, 67.0, 0.12]),
        covmat_path=covmat)
    Y = synth_rows(400, seed=42)
    geom = Grid2DGeometry.from_targets(device=device, targets=Y, z=Z4,
                                       k=K6, quantity="pklin",
                                       units="Mpc3", law="syren_linear")
    geom.attach_head_coords()
    want_sizes = []
    for _ in range(Z4.size):
        want_sizes.append(int(K6.size))
    report("attach_head_coords: one bin per z slice, length nk",
           geom.bin_sizes == want_sizes,
           "bin_sizes %s" % (geom.bin_sizes,))
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    width = Z4.size * K6.size
    model = ResCNN(input_dim=N_IN, output_dim=width, int_dim_res=16,
                   geom=geom, kernel_size=3, n_blocks=2,
                   n_blocks_cnn=1, block_opts=block_opts).to(device)
    report("identity basis: W_fd / W_df stay None on the grid2d geometry",
           model.W_fd is None and model.W_df is None,
           "n_bins %d, max_bin %d" % (model.n_bins, model.max_bin))
    x = torch.randn(4, N_IN, device=device)
    with torch.no_grad():
        full = model(x)
        trunk = model.mlp(x)
    report("epoch-0 identity: the head model equals its trunk bitwise",
           torch.equal(full, trunk),
           "max|d| = %.2e" % (full - trunk).abs().max().item())
    # two-phase on the plain heads (user ruling 2026-07-12): the phase
    # switch freezes the right parameter groups, the trunk phase
    # bypasses the head (forward == the bare trunk), and an unknown
    # phase is loud. At the zero init every phase's output equals the
    # trunk, so the flags are the discriminating assertions.
    model.set_train_phase("trunk")
    trunk_on = all(p.requires_grad for p in model.mlp.parameters())
    head_off = (all(not p.requires_grad for p in model.convs.parameters())
                and all(not p.requires_grad
                        for p in model.acts.parameters()))
    with torch.no_grad():
        bypass = model(x)
    model.set_train_phase("head")
    trunk_off = all(not p.requires_grad for p in model.mlp.parameters())
    head_on = (all(p.requires_grad for p in model.convs.parameters())
               and model.gate.requires_grad)
    with torch.no_grad():
        head_out = model(x)
    model.set_train_phase("joint")
    joint_on = all(p.requires_grad for p in model.parameters())
    report("two-phase: freezes per phase + trunk-phase head bypass",
           trunk_on and head_off and trunk_off and head_on and joint_on
           and torch.equal(bypass, trunk)
           and torch.equal(head_out, trunk),
           "trunk/head/joint flags + bypass == trunk bitwise")
    try:
        model.set_train_phase("nope")
        report("unknown train phase raises", False, "no raise")
    except ValueError:
        report("unknown train phase raises", True, "ValueError")
    try:
        ResTRF(input_dim=N_IN, output_dim=width, int_dim_res=8,
               geom=geom, n_tokens=3, block_opts=block_opts)
        report("n_tokens on real physical bins raises", False, "no raise")
    except ValueError as e:
        report("n_tokens on real physical bins raises",
               "physical bins" in str(e), "ValueError names the bins")
    # save -> rebuild -> predict bitwise: the rebuilt Grid2DGeometry
    # carries no bin_sizes in its state (derived, never persisted), so
    # this leg fails unless _rebuild_model re-attaches before
    # constructing the head model.
    root = os.path.join(tmp, "emul_g2_head")
    config = {"data": {"grid2d": {"quantity": "pklin",
                                  "units": "Mpc3",
                                  "law": "syren_linear",
                                  "z_file": "z.npy",
                                  "k_file": "k.npy",
                                  "train_base": "tb.npy",
                                  "val_base": "vb.npy"},
                       "train_dv": "t.npy",
                       "val_dv": "v.npy",
                       "train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat)},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1],
                 "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train=one_pass_training_recipe(
                    thresholds=(0.2, 1.0, 10.0, 100.0)),
                  resolved_model=grid2d_head_recipe(width),
                  composition_mode="plain",
                  transfer_refined=False,
                  resolved_pce=None,
                  resolved_transfer=None,
                  facts_yaml=supported_test_record(
                      names=pgeom.state()["names"],
                      label="mps-identity/correction-head",
                      family="grid2d",
                      support=GRID2D_SUPPORT),
                  attrs={"rescale": "none", "quantity": "pklin"})
    theta = np.array([[2.15, 68.0, 0.121]])
    x1 = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        ref = geom.decode(model(pgeom.encode(x1)))[0].cpu().numpy()
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    report("head save -> rebuild -> predict bitwise (rebuild attach)",
           np.array_equal(got["pklin"].reshape(-1), ref),
           "max|d| = %.2e"
           % np.abs(got["pklin"].reshape(-1) - ref).max())


def check_npce(tmp, device):
    """NPCE on grid2d (the 2026-07-12 family-wide ruling): the residual
    base + refiner algebra is exact under the diagonal metric, the
    fitted base's state round-trips byte-identical, save -> rebuild ->
    predict composes base + net bitwise (_build_diag_decoder), and the
    ratio form is loudly rejected on a diagonal family (validate_pce)."""
    from emulator.designs.pce import PCEEmulator
    from emulator.losses.pce import PCEResidualDiagChi2
    from emulator.experiment import validate_pce
    covmat = os.path.join(tmp, "g2npce.covmat")
    write_covmat(covmat, IN_NAMES, seed=51)
    pgeom = ParamGeometry.from_covmat(
        device=device, center=np.array([2.1, 67.0, 0.12]),
        covmat_path=covmat)
    g = np.random.default_rng(53)
    C = np.column_stack([g.normal(2.1, 0.1, 400),
                         g.normal(67.0, 2.0, 400),
                         g.normal(0.12, 0.005, 400)]).astype("float32")
    # rows with a REAL smooth parameter dependence (an H0-linear shift),
    # so the LOO gate keeps a mode and the fitted base is alive — a leg
    # a dead base could pass proves nothing (the smoke-gate rule).
    Y = synth_rows(400, seed=52)
    Y = Y + 0.3 * ((C[:, 1] - 67.0) / 2.0)[:, None]
    geom = Grid2DGeometry.from_targets(device=device, targets=Y, z=Z4,
                                       k=K6, quantity="pklin",
                                       units="Mpc3", law="syren_linear")
    tC = torch.from_numpy(C).to(device)
    X_white = pgeom.encode(tC)
    dv = torch.from_numpy(Y.astype("float32")).to(device)
    pce = PCEEmulator.from_training(device, X_white, geom.encode(dv),
                                    p_max=2, r_max=2, q=0.5, k_max=4,
                                    loo_max=0.9, max_terms=8, silent=True)
    chi2fn = PCEResidualDiagChi2(geom=geom, pce=pce)
    report("NPCE wrapper: diagonal residual class + needs_params",
           type(chi2fn).__name__ == "PCEResidualDiagChi2"
           and chi2fn.needs_params, "")
    with torch.no_grad():
        base = pce(X_white[:8])
    report("NPCE base is alive (the fit kept a real mode)",
           base.abs().max().item() > 1e-4,
           "max|base| = %.2e" % base.abs().max().item())
    enc = chi2fn.encode(dv[:8], X_white[:8])
    report("NPCE encode: whitened truth minus the base, bitwise",
           torch.equal(enc, geom.encode(dv[:8]) - base), "")
    y = torch.randn(8, int(Z4.size) * int(K6.size), device=device)
    report("NPCE decode: geom.decode(net + base), bitwise",
           torch.equal(chi2fn.decode(y, X_white[:8]),
                       geom.decode(y + base)), "")
    pce2 = PCEEmulator.from_state(pce.state(), device)
    st_ok = True
    for key, val in pce.state().items():
        st_ok = st_ok and torch.equal(val, pce2.state()[key])
    report("NPCE base state round-trip byte-identical", st_ok,
           "%d buffers" % len(pce.state()))
    # save -> rebuild -> predict: the pce h5 group + form attr must
    # rebuild the base, and the family predictor must compose it
    # (a bare geom.decode here would be silently wrong).
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    width = int(Z4.size) * int(K6.size)
    model = ResMLP(input_dim=N_IN, output_dim=width, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    root = os.path.join(tmp, "emul_g2_npce")
    config = {"data": {"grid2d": {"quantity": "pklin",
                                  "units": "Mpc3",
                                  "law": "syren_linear",
                                  "z_file": "z.npy",
                                  "k_file": "k.npy",
                                  "train_base": "tb.npy",
                                  "val_base": "vb.npy"},
                       "train_dv": "t.npy",
                       "val_dv": "v.npy",
                       "train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat)},
              "pce": {"form": "residual"},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1],
                 "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train=one_pass_training_recipe(
                    thresholds=(0.2, 1.0, 10.0, 100.0)),
                  resolved_model=grid2d_recipe(width),
                  pce=pce, pce_form="residual",
                  composition_mode="npce",
                  transfer_refined=False,
                  resolved_pce={"form": "residual",
                                "p_max": 2,
                                "r_max": 2,
                                "q": 0.5,
                                "k_max": 4,
                                "loo_max": 0.9,
                                "max_terms": 8},
                  resolved_transfer=None,
                  facts_yaml=supported_test_record(
                      names=pgeom.state()["names"],
                      label="mps-identity/npce-power",
                      family="grid2d",
                      support=GRID2D_SUPPORT),
                  attrs={"rescale": "none", "quantity": "pklin"})
    theta = np.array([[2.15, 68.0, 0.121]])
    x1 = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        x1e = pgeom.encode(x1)
        ref = geom.decode(model(x1e) + pce(x1e))[0].cpu().numpy()
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    report("NPCE save -> rebuild -> predict composes base + net bitwise",
           np.array_equal(got["pklin"].reshape(-1), ref),
           "max|d| = %.2e"
           % np.abs(got["pklin"].reshape(-1) - ref).max())
    try:
        validate_pce({"form": "ratio"}, diagonal=True)
        report("diagonal family rejects pce form ratio", False, "no raise")
    except ValueError as e:
        report("diagonal family rejects pce form ratio",
               "residual" in str(e), "ValueError names the fix")


def _load_emul_mps_stubbed():
    if "cobaya" not in sys.modules:
        sys.modules["cobaya"] = types.ModuleType("cobaya")
    theory_mod = types.ModuleType("cobaya.theory")

    class _Theory:
        renames = {}
        extra_args = {}
        def initialize(self):
            pass

    theory_mod.Theory = _Theory
    sys.modules["cobaya.theory"] = theory_mod
    log_mod = types.ModuleType("cobaya.log")

    class _LoggedError(Exception):
        def __init__(self, logger, msg=""):
            super().__init__(msg)

    class _Logger:
        def warning(self, *a, **k):
            pass
        def debug(self, *a, **k):
            pass

    log_mod.LoggedError = _LoggedError
    log_mod.get_logger = lambda name: _Logger()
    sys.modules["cobaya.log"] = log_mod
    root = Path(__file__).resolve().parents[3]
    path = root / "cobaya_theory" / "emul_mps.py"
    spec = importlib.util.spec_from_file_location("emul_mps_shim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_adapter(tmp, device):
    mod = _load_emul_mps_stubbed()
    cls = mod.emul_mps
    root_p = os.path.join(tmp, "ad_pklin")
    save_synthetic_grid2d(root_p, device, tmp, label=ADAPTER_PAIR_LABEL,
                          quantity="pklin",
                          units="Mpc3", law="syren_linear", seed=40,
                          support=ADAPTER_SUPPORT)
    root_b = os.path.join(tmp, "ad_boost")
    save_synthetic_grid2d(root_b, device, tmp, label=ADAPTER_PAIR_LABEL,
                          quantity="boost",
                          units="dimensionless", law="syren_halofit",
                          seed=50,
                          support=ADAPTER_SUPPORT)

    # synthetic base stubs: closed forms the assembly must reproduce
    # EXACTLY (the real syren formulas are the workstation/EMUL2 story).
    calls = {}

    def stub_pklin(k_mpc, z, As_1e9, ns, H0, Ob, Om, w0=-1.0, wa=0.0,
                   mnu=0.06):
        calls["pklin"] = (As_1e9, ns, H0, Ob, Om, w0, wa)
        return (1.0 + z[:, None]) * (1.0 + k_mpc[None, :])

    def stub_boost(k_mpc, z, pk_lin_mpc, As_1e9, ns, H0, Ob, Om,
                   w0=-1.0, wa=0.0, mnu=0.06):
        calls["boost_plin"] = np.array(pk_lin_mpc)
        return 2.0 * np.ones((z.size, k_mpc.size))

    saved = (mod.syren_base.base_pklin, mod.syren_base.base_boost)
    mod.syren_base.base_pklin = stub_pklin
    mod.syren_base.base_boost = stub_boost
    try:
        t = cls()
        t.extra_args = {"device": "cpu", "emulators": [root_p, root_b]}
        t.initialize()
        report("requirements include the syren-base names",
               {"As", "ns", "H0", "omegab", "omegam"}
               <= set(t.get_requirements()), "")
        t.log = types.SimpleNamespace(debug=lambda *a, **k: None)
        t.output_params = []
        point = {"As": 2.1e-9,
                 "ns": 0.965,
                 "H0": 67.0,
                 "omegab": 0.049,
                 "omegam": 0.31,
                 "omch2": 0.12}
        state = {}
        ok_calc = t.calculate(state, want_derived=False, **point)
        # exact assembly against the stubs.
        out_lin = t.p_lin.predict(point)["pklin"]
        base = stub_pklin(K6, Z4, 2.1, 0.965, 67.0, 0.049, 0.31)
        want_plin = np.exp(out_lin) * base
        k_arr, z_arr, got_plin = state[("Pk_grid", False, "delta_tot",
                                        "delta_tot")]
        ok = ok_calc and np.allclose(got_plin, want_plin, rtol=0, atol=0)
        out_b = t.p_boost.predict(point)["boost"]
        weight = 1.0 - np.exp(-(K6 / mod._BLEND_K_T) ** mod._BLEND_N)
        want_boost = 1.0 + (np.exp(out_b) * 2.0 - 1.0) * weight[None, :]
        _, _, got_pnl = state[("Pk_grid", True, "delta_tot",
                               "delta_tot")]
        ok = ok and np.allclose(got_pnl, want_boost * want_plin,
                                rtol=1e-12)
        # the blend pins boost -> 1 at the lowest k (k = 1e-4 << k_t).
        low_boost = got_pnl[:, 0] / got_plin[:, 0]
        ok = ok and np.abs(low_boost - 1.0).max() < 1e-3
        # the As -> As_1e9 conversion reached the base.
        ok = ok and abs(calls["pklin"][0] - 2.1) < 1e-12
        # boost base received the EMULATED P_lin (the legacy flow).
        ok = ok and np.allclose(calls["boost_plin"], want_plin)
        report("calculate assembly exact vs base stubs + blend", ok, "")

        # Sigma-eight has an analytic reference independent of the adapter.
        # For P(k)=C/k with C=512*pi^2/9, sigma_R=8/R.  With h=.64 and
        # R=8/h Mpc, the infinite-domain answer is therefore .64; the finite
        # x=[1e-4,400] grid below has the exact value recorded here.
        sigma_helper = t._compute_sigma8
        h_sigma = 0.64
        radius = 8.0 / h_sigma
        k_sigma = np.geomspace(1.0e-4 / radius, 400.0 / radius, 4001)
        c_sigma = 512.0 * np.pi ** 2 / 9.0
        p_sigma = c_sigma / k_sigma
        z_sigma = np.array([0.0, 0.009, 0.5, 1.0])
        surface_sigma = np.stack(
            (p_sigma, 4.0 * p_sigma, 0.5 * p_sigma, 0.25 * p_sigma))
        got_sigma = sigma_helper(
            surface_sigma, k_sigma, z_sigma, h=h_sigma)
        report("sigma8 analytic known answer uses R=8/h Mpc",
               abs(got_sigma - 0.6399980037465730) < 2.0e-8,
               "got %.15g" % got_sigma)

        # The public calculate path must send the assembled linear spectrum
        # and H0 to sigma8.  A spy makes both facts observable without asking
        # the tiny six-point assembly grid to pass the integration certificate.
        seen_sigma = {}

        def sigma_spy(power, k_axis, z_axis, *, h):
            seen_sigma["power"] = np.array(power, copy=True)
            seen_sigma["shape"] = power.shape
            seen_sigma["h"] = h
            return 0.8

        t._compute_sigma8 = sigma_spy
        t.output_params = ["sigma8"]
        sigma_state = {}
        sigma_ok = t.calculate(sigma_state, want_derived=True, **point)
        report("calculate passes linear P(k,z) and h=H0/100 to sigma8",
               sigma_ok and abs(seen_sigma.get("h", -1.0) - 0.67) < 1e-15
               and np.array_equal(seen_sigma.get("power"), want_plin)
               and sigma_state.get("derived") == {"sigma8": 0.8}, "")
        t._compute_sigma8 = sigma_helper
        t.output_params = []

        try:
            sigma_helper(surface_sigma, k_sigma,
                         np.array([0.009, 0.1, 0.5, 1.0]), h=h_sigma)
            report("sigma8 requires an exact stored z=0 row", False,
                   "no raise")
        except ValueError as e:
            report("sigma8 requires an exact stored z=0 row",
                   "exact z=0" in str(e), type(e).__name__)

        try:
            k_short = np.geomspace(1.0, 10.0, 4001)
            p_short = c_sigma / k_short
            sigma_helper(np.stack((p_short, p_short, p_short, p_short)),
                         k_short, z_sigma, h=h_sigma)
            report("sigma8 refuses a short positive k interval", False,
                   "no raise")
        except ValueError as e:
            report("sigma8 refuses a short positive k interval",
                   "incomplete" in str(e), type(e).__name__)

        try:
            k_sparse = np.geomspace(1.0e-4 / radius,
                                    400.0 / radius, 8)
            p_sparse = c_sigma / k_sparse
            sigma_helper(
                np.stack((p_sparse, p_sparse, p_sparse, p_sparse)),
                k_sparse, z_sigma, h=h_sigma)
            report("sigma8 refuses a wide but under-resolved k grid", False,
                   "no raise")
        except ValueError as e:
            report("sigma8 refuses a wide but under-resolved k grid",
                   ("too coarse" in str(e)
                    or "under-resolved" in str(e)), type(e).__name__)
        # get_Pk_grid / interpolator at the nodes.
        t.current_state = state
        kk, zz, pk = t.get_Pk_grid(nonlinear=False)
        itp = t.get_Pk_interpolator(nonlinear=True)
        node = itp.P(float(Z4[2]), float(K6[3]))
        want_node = got_pnl[2, 3]
        ok = np.allclose(pk, want_plin) and abs(
            node / want_node - 1.0) < 1e-6
        report("get_Pk_grid + interpolator node round-trip", ok,
               "rel %.1e" % abs(node / want_node - 1.0))
        # unit 69 (20M-01): the getters serve Cobaya's public default --
        # an omitted nonlinear argument returns the NONLINEAR spectrum.
        # On the real computed state the omitted grid must equal the
        # explicit nonlinear grid and differ from the linear one.
        _, _, pk_omit = t.get_Pk_grid()
        _, _, pk_true = t.get_Pk_grid(nonlinear=True)
        _, _, pk_false = t.get_Pk_grid(nonlinear=False)
        ok = (np.array_equal(pk_omit, pk_true)
              and not np.array_equal(pk_omit, pk_false))
        report("get_Pk_grid() omitted == explicit nonlinear=True, != "
               "nonlinear=False (the public default is nonlinear)", ok, "")
        # the same three-arm comparison through the interpolator, at a
        # stored node and one interior (between-node) point.
        itp_omit = t.get_Pk_interpolator()
        itp_true = t.get_Pk_interpolator(nonlinear=True)
        itp_false = t.get_Pk_interpolator(nonlinear=False)
        z_mid = 0.5 * (float(Z4[1]) + float(Z4[2]))
        k_mid = 0.5 * (float(K6[3]) + float(K6[4]))
        ok_itp = True
        for zq, kq in ((float(Z4[2]), float(K6[3])), (z_mid, k_mid)):
            v_omit = itp_omit.P(zq, kq)
            v_true = itp_true.P(zq, kq)
            v_false = itp_false.P(zq, kq)
            ok_itp = (ok_itp
                      and abs(v_omit / v_true - 1.0) < 1e-12
                      and abs(v_omit / v_false - 1.0) > 1e-6)
        report("get_Pk_interpolator() omitted tracks nonlinear=True at "
               "node + interior, differs from nonlinear=False", ok_itp, "")
        # a sentinel state with deliberately separated linear/nonlinear
        # values proves the discrimination has catch power on its own,
        # independent of the assembly: the omitted call returns the
        # nonlinear sentinel (100), never the linear one (1).
        ks = np.array([1e-3, 1e-2, 1e-1])
        zs = np.array([0.0, 1.0])
        lin = np.ones((zs.size, ks.size))
        nl = 100.0 * np.ones((zs.size, ks.size))
        t.current_state = {
            ("Pk_grid", False, "delta_tot", "delta_tot"): (ks, zs, lin),
            ("Pk_grid", True, "delta_tot", "delta_tot"): (ks, zs, nl)}
        _, _, s_omit = t.get_Pk_grid()
        report("sentinel: omitted get_Pk_grid returns the nonlinear "
               "values, not the linear ones",
               np.array_equal(s_omit, nl) and not np.array_equal(s_omit, lin),
               "")
        t.current_state = state
        # mutation control: the two branches genuinely differ, so the
        # default VALUE is load-bearing -- a revert to nonlinear=False
        # would serve the linear spectrum by omission and red the legs
        # above and the signature guard below.
        report("mutation control: nonlinear=False and =True differ, so "
               "the default value decides the served spectrum",
               not np.array_equal(pk_false, pk_true), "")
        # protocol guard: pin emul_mps's public nonlinear default against
        # the INSTALLED Cobaya BoltzmannBase, so an upstream drift OR a
        # local revert reds for review instead of silently serving the
        # wrong spectrum.
        ok_sig = _BASE_PK_SIGS is not None
        detail = "BoltzmannBase not importable"
        if ok_sig:
            for name in ("get_Pk_grid", "get_Pk_interpolator"):
                base_def = _BASE_PK_SIGS[name].parameters["nonlinear"].default
                our_def = inspect.signature(
                    getattr(cls, name)).parameters["nonlinear"].default
                ok_sig = ok_sig and (our_def is True) and (our_def == base_def)
            detail = "nonlinear default True on both getters == BoltzmannBase"
        report("adapter getters pin Cobaya's nonlinear=True default "
               "(BoltzmannBase protocol guard)", ok_sig, detail)
        # a non-positive base -> calculate returns False (the legacy
        # rejection semantics).
        mod.syren_base.base_pklin = (
            lambda *a, **kw: -np.ones((Z4.size, K6.size)))
        state2 = {}
        report("non-positive spectrum rejects the point (False)",
               t.calculate(state2, want_derived=False, **point) is False,
               "")
        mod.syren_base.base_pklin = stub_pklin
        # pair-validation legs.
        try:
            t2 = cls()
            t2.extra_args = {"device": "cpu", "emulators": [root_p]}
            t2.initialize()
            report("pair-count guard raises", False, "no raise")
        except ValueError as e:
            report_refusal("pair-count guard raises", e,
                           needle="exactly 2",
                           law="the pair-count law")
        root_p2 = os.path.join(tmp, "ad_pklin2")
        # a second linear-power emulator, built to be refused beside the first.
        # It carries the PAIR's label on purpose: two artifacts declaring one
        # quantity, both trained off ONE generator dump, is exactly the
        # ambiguity the duplicate law exists to refuse -- one dataset, one
        # identity. Give this double an identity of its own instead and the
        # served pair stops being one dataset: the identity law refuses it
        # first, and the duplicate law -- the law this leg exists to prove --
        # is never reached.
        save_synthetic_grid2d(root_p2, device, tmp,
                              label=ADAPTER_PAIR_LABEL,
                              quantity="pklin",
                              units="Mpc3", law="none", seed=60)
        try:
            t3 = cls()
            t3.extra_args = {"device": "cpu",
                             "emulators": [root_p, root_p2]}
            t3.initialize()
            report("duplicate quantity raises", False, "no raise")
        except ValueError as e:
            report_refusal("duplicate quantity raises", e,
                           needle="two artifacts both declare quantity",
                           law="the duplicate-quantity law")
        root_bad = os.path.join(tmp, "ad_badgrid")
        # a boost emulator on a wavenumber grid the linear artifact does not
        # share, built to be refused: a different run, so its own identity. One
        # generator dump has ONE grid, so a double with a different grid could
        # not honestly carry the pair's label -- it would model a dataset that
        # cannot exist.
        save_synthetic_grid2d(root_bad, device, tmp,
                              label="mps-identity/adapter-mismatched-grid",
                              quantity="boost",
                              units="dimensionless", law="none",
                              k=np.logspace(-4, 0.0, 5), seed=70)
        try:
            t4 = cls()
            t4.extra_args = {"device": "cpu",
                             "emulators": [root_p, root_bad]}
            t4.initialize()
            report("grid mismatch raises", False, "no raise")
        except ValueError as e:
            report_refusal("grid mismatch raises", e,
                           needle="trained on different (z, k) grids",
                           law="the shared-grid law")
        # the dataset-identity law: the linear spectrum and the boost are
        # multiplied into one nonlinear spectrum, so they must come from ONE
        # generator dump. This pair is topologically PERFECT -- one 'pklin', one
        # 'boost', on the SAME (z, k) grid -- so every configuration law the
        # adapter runs first passes, and the refusal that fires is the identity
        # one. The only thing wrong with the pair is what the arm is about: the
        # boost double was saved under a label of its own, so its record carries
        # a different dataset identity, and a boost fitted to one draw does not
        # correct a linear spectrum fitted to another however well the grids and
        # the axes line up.
        root_b2 = os.path.join(tmp, "ad_boost_other")
        save_synthetic_grid2d(root_b2, device, tmp,
                              label="mps-identity/adapter-other-dataset",
                              quantity="boost",
                              units="dimensionless", law="syren_halofit",
                              seed=120)
        try:
            t5 = cls()
            t5.extra_args = {"device": "cpu",
                             "emulators": [root_p, root_b2]}
            t5.initialize()
            report("two datasets served together raise", False, "no raise")
        except ValueError as e:
            report_refusal("two datasets served together raise", e,
                           needle="different datasets",
                           law="the dataset-identity law")
    finally:
        mod.syren_base.base_pklin, mod.syren_base.base_boost = saved


def check_validate():
    def cfg(g2, extra=None):
        data = {"grid2d": g2,
                "train_dv": "a",
                "val_dv": "b",
                "train_params": "c",
                "val_params": "d",
                "train_covmat": "e"}
        if extra:
            data.update(extra)
        return {"data": data,
                "pce": None,
                "transfer": None}
    good = {"quantity": "boost",
            "units": "dimensionless",
            "law": "syren_halofit",
            "z_file": "z.npy",
            "k_file": "k.npy",
            "train_base": "tb.npy",
            "val_base": "vb.npy",
            "k_stride": 10}
    out = validate_grid2d(cfg(good), train_args={}, rescale="none")
    ok = out["quantity"] == "boost"
    bads = [
        {**good, "law": "syren_linear"},                # wrong pairing
        {**good, "units": "Mpc3"},                      # wrong units
        {k: v for k, v in good.items() if k != "train_base"},
        {**good, "law": "none"},                        # base under none
        {**good, "k_stride": 0},
    ]
    for b in bads:
        try:
            validate_grid2d(cfg(b), train_args={}, rescale="none")
            ok = False
            print("  validate_grid2d silent on:", b.get("law"),
                  b.get("units"), b.get("k_stride"))
        except ValueError:
            pass
    # transfer is ACCEPTED here since the 2026-07-12 symmetry ruling
    # (the block itself is vetted by validate_transfer on the
    # from_config branch); this leg pins the acceptance so a re-added
    # family forbid would be loud.
    c = cfg(good)
    c["transfer"] = {"from": "x"}
    try:
        validate_grid2d(c, train_args={}, rescale="none")
    except ValueError as e:
        ok = False
        print("  validate_grid2d rejected a transfer block:", str(e)[:70])
    report("validate_grid2d legs", ok, "")


def check_finetune(tmp, device):
    root = os.path.join(tmp, "ft_g2_src")
    pgeom, geom, model, covmat = save_synthetic_grid2d(
        root, device, tmp, label="mps-identity/finetune-source",
        law="syren_linear", seed=80)
    source = warmstart.load_source(root=root, device=device)
    report("load_source accepts a grid2d artifact",
           type(source.geom).__name__ == "Grid2DGeometry",
           "geom %s" % type(source.geom).__name__)
    g = np.random.default_rng(90)
    C = np.column_stack([g.normal(2.1, 0.1, 64),
                         g.normal(67.0, 2.0, 64),
                         g.normal(0.12, 0.005, 64)]).astype("float32")
    train_set = {"C": C,
                 "idx": np.arange(64),
                 "C_mean": C.mean(axis=0)}
    new_pgeom, extra = warmstart.extend_input_geometry(
        source=source, covmat_path=covmat,
        train_mean=train_set["C_mean"], device=device)
    model_opts = warmstart.recipe_to_model_opts(source.recipe)
    try:
        init_state, verdict, _ = warmstart.build_warm_start(
            source=source, new_pgeom=new_pgeom, pinned_geom=source.geom,
            model_opts=model_opts, train_set=train_set,
            extra_names=extra, device=device)
        report("grid2d warm start reproduces the source at epoch 0",
               init_state is not None, verdict.strip()[:60])
    except ValueError as e:
        report("grid2d warm start reproduces the source at epoch 0",
               False, str(e)[:80])
    ft_train = os.path.join(tmp, "ft_train.1.txt")
    ft_val = os.path.join(tmp, "ft_val.1.txt")
    for path, label in ((ft_train, "mps-finetune-train"),
                        (ft_val, "mps-finetune-val")):
        np.savetxt(path, [[1.0, 0.0, 2.1, 67.0, 0.12, 0.0]])
        _write_paramnames(path)
        _write_facts(path, label)

    def ft_cfg(g2, from_root):
        return {"data": {"grid2d": g2,
                         "train_dv": "t.npy",
                         "val_dv": "v.npy",
                         "train_params": ft_train,
                         "val_params": ft_val,
                         "train_covmat": covmat,
                         "n_train": 10,
                         "n_val": 5,
                         "split_seed": 0},
                "train_args": {"nepochs": 1,
                               "bs": 8,
                               "finetune": {"from": from_root}}}
    good = {"quantity": "pklin",
            "units": "Mpc3",
            "law": "syren_linear",
            "z_file": "z.npy",
            "k_file": "k.npy",
            "train_base": "tb.npy",
            "val_base": "vb.npy"}
    bad = dict(good)
    bad["quantity"] = "boost"
    bad["units"] = "dimensionless"
    bad["law"] = "syren_halofit"
    try:
        EmulatorExperiment.from_config(ft_cfg(bad, root),
                                       device=torch.device("cpu"))
        report("metadata mismatch raises", False, "no raise")
    except ValueError as e:
        report("metadata mismatch raises",
               "grid2d-metadata mismatch" in str(e), "ValueError")


def main():
    print("mps-identity: geometry + staging law + round-trip + "
          "assembly + finetune legs")
    # seed the GLOBAL torch RNG so the synthetic nets are the same
    # every run (a red must reproduce — the run-10 bsn lesson).
    torch.manual_seed(0)
    device = torch.device("cpu")
    # Emit one '##AID <leg> <PASS|FAIL>' per board-declared leg (LEG_AIDS), in
    # this order. Each leg wraps a contiguous group of check_* calls; the
    # FAILURES snapshot taken before a group and read after it rolls that
    # group's probes into the leg's single terminal (see emit_leg). The
    # const-mask real-artifact checks run adjacent to check_geometry so they
    # fold under geometry-laws-and-pins (batch-5 seam ruling), not as their
    # own leg.
    with tempfile.TemporaryDirectory() as tmp:
        before = len(FAILURES)
        check_geometry(device)
        check_const_mask_artifact(tmp, device)
        emit_leg("mps-identity.geometry-laws-and-pins", before)

        before = len(FAILURES)
        check_staging(tmp)
        check_bounded_staging(tmp)
        emit_leg("mps-identity.bounded-staging-values", before)

        before = len(FAILURES)
        check_stable_moments(tmp, device)
        emit_leg("mps-identity.stable-streamed-moments", before)

        before = len(FAILURES)
        check_staging_lifecycle(tmp)
        emit_leg("mps-identity.staging-file-lifecycle", before)

        before = len(FAILURES)
        check_roundtrip(tmp, device, law="syren_linear")
        check_roundtrip(tmp, device, law="none")
        # the domain arms ride this leg: they are the same saved artifact,
        # rebuilt and asked a question, and the question is the one predict()
        # refuses. They emit no aid of their own.
        check_domain_law(tmp, device)
        check_head(tmp, device)
        check_npce(tmp, device)
        emit_leg("mps-identity.saved-model-variants", before)

        before = len(FAILURES)
        check_adapter(tmp, device)
        emit_leg("mps-identity.adapter-assembly-and-defaults", before)

        before = len(FAILURES)
        check_validate()
        check_finetune(tmp, device)
        emit_leg("mps-identity.config-and-finetune", before)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: mps-identity all checks green")


if __name__ == "__main__":
    main()
