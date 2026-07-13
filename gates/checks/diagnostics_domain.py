#!/usr/bin/env python3
"""diagnostics-domain gate: the shared score-domain boundary at every chi2
CONSUMER — torch only (no cobaya, no CAMB, no cosmolike).

Increment (e) landed the finite-chi2 contract at the training reduction and
the two evaluation boundaries; increment (h) extends it to the diagnostics
producers, which called chi2fn.chi2 DIRECTLY and interpreted the unchecked
score (local_linear_floor's f_floor / median_floor, the CMB / grid / grid2d
residual bands). A geometry whose Cinv is not positive-definite (a
same-shaped h5 edit strict weight loading accepts) makes a score go
negative, and the old dchi2_floor > 0.2 test read a -1 floor as a PERFECT 0.

This gate drives the shared boundary screen_chi2 (losses/core.py) and the
REAL diagnostics producers on tiny synthetic CMB fixtures (a plain
CmbDiagonalChi2 over a synthetic fiducial C_ell + an identity ParamGeometry
+ a small model), and asserts:
  - screen_chi2 passes a valid positive score byte-identical, normalizes a
    within-band roundoff negative to exact 0, and RAISES naming the boundary,
    the offending rows, the minimum, and the band on a materially negative /
    NaN / +-Inf score; a loss without _chi2_n_terms falls back to the 1e-6
    band floor (still rejecting), and the term count widens the band with w;
  - the REAL local_linear_floor refuses a reachable negative floor BEFORE it
    computes f_floor (the floor guard fires ahead of the model arm), refuses
    a NaN floor, and returns finite scores on a valid run; the mutation arm
    (the guard bypassed) recreates the false f_floor = 0;
  - the REAL cmb_residual_diagnostic refuses a corrupt per-sample score and
    returns finite bands on a valid run;
  - a source census: grid / grid2d residual route through the same shared
    boundary (_screen_diag_chi2 -> screen_chi2), and no producer still
    interprets a raw .double() score.
"""
import ast
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from emulator.losses.core import screen_chi2
from emulator.losses.cmb import make_cmb_chi2, CmbDiagonalChi2
from emulator.geometries.cmb import CmbDiagonalGeometry
from emulator.geometries.parameter import ParamGeometry
from emulator.diagnostics import (local_linear_floor, cmb_residual_diagnostic,
                                  _screen_diag_chi2)
import emulator.diagnostics as diagnostics

FAILURES = []


def report(label, ok, detail=""):
    print("  [" + ("PASS" if ok else "FAIL") + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
N_PARAM = 3
N_ELL = 8


def make_pgeom(device):
    """An identity ParamGeometry over N_PARAM columns (encode/decode trivial)."""
    return ParamGeometry(device=device,
                         names=["As", "tau", "omegam"],
                         center=np.zeros(N_PARAM),
                         evecs=np.eye(N_PARAM),
                         sqrt_ev=np.ones(N_PARAM))


def make_cmb(device):
    """A plain (law none) CmbDiagonalChi2 over a synthetic fiducial C_ell."""
    ell = np.arange(2, 2 + N_ELL, dtype="int64")
    cl  = np.linspace(1000.0, 100.0, N_ELL)
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum="tt", ell=ell, fiducial_cl=cl,
        center=cl, units="muK2", law="none")
    return geom, make_cmb_chi2(geom=geom, law="none")


class _Net(nn.Module):
    """Tiny (N_PARAM -> N_ELL) map, seeded so the run is deterministic."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(N_PARAM, N_ELL)

    def forward(self, x):
        return self.lin(x)


def make_sets(device, geom, n_train=40, n_val=20):
    """Train / val source dicts ({C, dv, idx}) of synthetic rows near the
    fiducial, so the local linear fit and the residual bands are well posed."""
    rng = np.random.default_rng(1)
    cl = geom.fiducial_cl.cpu().numpy().astype("float64")
    n = n_train + n_val
    C  = rng.normal(0.0, 1.0, (n, N_PARAM))
    dv = cl[None, :] * (1.0 + 0.02 * rng.standard_normal((n, N_ELL)))
    idx = np.arange(n)
    train = {"C": C, "dv": dv, "idx": idx[:n_train]}
    val   = {"C": C, "dv": dv, "idx": idx[n_train:]}
    return train, val


class _NanChi2(CmbDiagonalChi2):
    """A producer whose per-sample chi2 is NaN (a diverged model stand-in)."""
    def chi2(self, pred, target, full=False):
        c = super().chi2(pred, target, full=full)
        return torch.full_like(c, float("nan"))


class _NegChi2(CmbDiagonalChi2):
    """A producer whose per-sample chi2 is materially negative (a non-PD
    geometry stand-in: strict weight loading accepts the shape)."""
    def chi2(self, pred, target, full=False):
        c = super().chi2(pred, target, full=full)
        return c - 1000.0


class _FloorOnlyNegChi2(CmbDiagonalChi2):
    """Negative on the FIRST chi2 call (the local-linear floor) and valid
    afterwards (the model arm), so the floor is corrupt beside a valid model
    arm — the reachable case the floor guard, not eval_source_chi2, catches."""
    def __init__(self, geom):
        super().__init__(geom=geom)
        self._ncall = 0

    def chi2(self, pred, target, full=False):
        c = super().chi2(pred, target, full=full)
        self._ncall += 1
        if self._ncall == 1:
            return c - 1000.0
        return c


# --------------------------------------------------------------------------
# A. screen_chi2 unit behavior
# --------------------------------------------------------------------------
def check_screen_chi2(device):
    _, chi2fn = make_cmb(device)                    # _chi2_n_terms == N_ELL
    good = torch.tensor([0.0, 1.0, 5.0, 123.0], dtype=torch.float32)
    out = screen_chi2(good.clone(), loss=chi2fn, label="unit")
    report("valid positive score passes byte-identical",
           torch.equal(out, good), "unchanged")

    # within-band roundoff negative -> exact 0 (band ~ 32*eps(f32)*8 ~ 3e-5)
    band_neg = torch.tensor([-1.0e-6, 2.0], dtype=torch.float32)
    out = screen_chi2(band_neg.clone(), loss=chi2fn, label="unit")
    report("within-band roundoff negative normalizes to exact 0",
           float(out[0]) == 0.0 and float(out[1]) == 2.0, "0.0 exact")

    for name, bad in (("materially negative", torch.tensor([1.0, -5.0, 2.0])),
                      ("NaN", torch.tensor([1.0, float("nan"), 2.0])),
                      ("+Inf", torch.tensor([1.0, float("inf"), 2.0])),
                      ("-Inf", torch.tensor([1.0, float("-inf"), 2.0]))):
        try:
            screen_chi2(bad.float(), loss=chi2fn, label="unit",
                        positions=np.array([100, 200, 300]))
            report("screen refuses a " + name + " score", False, "no raise")
        except ValueError as e:
            msg = str(e)
            report("screen refuses a " + name + " score",
                   "unit" in msg and "200" in msg and "band" in msg,
                   "names row 200 + band")

    # fallback-1 band: a bare object without _chi2_n_terms still rejects, at
    # the 1e-6 floor (a -5e-4 that a width-N_ELL band would also reject).
    class _Bare:
        pass
    try:
        screen_chi2(torch.tensor([1.0, -5.0e-4]), loss=_Bare(), label="bare")
        report("fallback-1 band still rejects out-of-domain", False, "no raise")
    except ValueError:
        report("fallback-1 band still rejects out-of-domain", True, "ValueError")

    # term-count census: the width sets the band, so a wider loss admits a
    # roundoff negative the 1e-6 floor would refuse (never a silent fallback).
    class _Wide(CmbDiagonalChi2):
        def _chi2_n_terms(self):
            return 100000
    _, wide = make_cmb(device)
    wide.__class__ = _Wide
    val = torch.tensor([-1.0e-4, 1.0], dtype=torch.float32)
    try:
        out = screen_chi2(val.clone(), loss=wide, label="wide")
        wide_ok = float(out[0]) == 0.0
    except ValueError:
        wide_ok = False
    try:
        screen_chi2(val.clone(), loss=_Bare(), label="floor")
        floor_ok = False
    except ValueError:
        floor_ok = True
    report("term count widens the band (no silent fallback-1)",
           wide_ok and floor_ok, "-1e-4 accepted at w=1e5, refused at floor")


# --------------------------------------------------------------------------
# B. the REAL local_linear_floor
# --------------------------------------------------------------------------
def check_local_linear_floor(device):
    geom, chi2fn = make_cmb(device)
    train, val = make_sets(device, geom)
    net = _Net().to(device)
    res = local_linear_floor(model=net, param_geometry=make_pgeom(device),
                             chi2fn=chi2fn, train_set=train, val_set=val,
                             device=device, k_nn=8, bs=256)
    report("valid local_linear_floor returns finite f_floor / median_floor",
           np.isfinite(res["f_floor"]) and np.isfinite(res["median_floor"]),
           "f_floor %.3f, median_floor %.3g"
           % (res["f_floor"], res["median_floor"]))

    # a reachable negative floor beside a VALID model arm: the floor guard
    # fires first, naming the local-linear floor (eval_source never runs).
    try:
        local_linear_floor(model=net, param_geometry=make_pgeom(device),
                           chi2fn=_FloorOnlyNegChi2(geom), train_set=train,
                           val_set=val, device=device, k_nn=8, bs=256)
        report("negative floor refused before f_floor", False, "no raise")
    except ValueError as e:
        report("negative floor refused before f_floor (the floor guard)",
               "local-linear floor" in str(e), "names the floor")
    try:
        local_linear_floor(model=net, param_geometry=make_pgeom(device),
                           chi2fn=_NanChi2(geom), train_set=train,
                           val_set=val, device=device, k_nn=8, bs=256)
        report("NaN floor refused", False, "no raise")
    except ValueError as e:
        report("NaN floor refused", "local-linear floor" in str(e), "ValueError")

    # mutation arm: bypass ONLY the diagnostics-side guard; the negative
    # floor then flows into f_floor = mean(dchi2_floor > 0.2) = 0.0 (the
    # impossible "data-only floor" reported PERFECT that the guard closes).
    saved = diagnostics.screen_chi2
    diagnostics.screen_chi2 = lambda c, loss, label, positions=None: c
    try:
        bad = local_linear_floor(model=net, param_geometry=make_pgeom(device),
                                 chi2fn=_FloorOnlyNegChi2(geom),
                                 train_set=train, val_set=val, device=device,
                                 k_nn=8, bs=256)
        report("mutation (guard bypassed) recreates the false f_floor = 0",
               bad["f_floor"] == 0.0 and bad["median_floor"] < 0.0,
               "f_floor %.1f, median_floor %.3g"
               % (bad["f_floor"], bad["median_floor"]))
    finally:
        diagnostics.screen_chi2 = saved


# --------------------------------------------------------------------------
# C. the REAL cmb_residual_diagnostic
# --------------------------------------------------------------------------
def check_cmb_residual(device):
    geom, chi2fn = make_cmb(device)
    _, val = make_sets(device, geom)
    net = _Net().to(device)
    out = cmb_residual_diagnostic(model=net, param_geometry=make_pgeom(device),
                                  chi2fn=chi2fn, val_set=val, device=device,
                                  bs=256)
    report("valid cmb_residual returns finite worst chi2",
           np.isfinite(out["worst"]["dchi2"]),
           "worst dchi2 %.3g" % out["worst"]["dchi2"])
    try:
        cmb_residual_diagnostic(model=net, param_geometry=make_pgeom(device),
                                chi2fn=_NegChi2(geom), val_set=val,
                                device=device, bs=256)
        report("cmb_residual refuses a corrupt score", False, "no raise")
    except ValueError as e:
        report("cmb_residual refuses a corrupt score (names the producer)",
               "cmb residual" in str(e), "names cmb residual")


# --------------------------------------------------------------------------
# D. source census: every producer routes through the shared boundary
# --------------------------------------------------------------------------
def check_producer_census():
    src = open(os.path.join(ROOT, "emulator", "diagnostics.py")).read()
    tree = ast.parse(src)
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    names.add(sub.func.id)
            calls[node.name] = names
    for fn in ("cmb_residual_diagnostic", "grid_residual_diagnostic",
               "grid2d_residual_diagnostic"):
        report(fn + " routes through the shared boundary",
               "_screen_diag_chi2" in calls.get(fn, set()),
               "calls _screen_diag_chi2")
    report("local_linear_floor screens its floor score",
           "screen_chi2" in calls.get("local_linear_floor", set()),
           "calls screen_chi2")
    # no producer still interprets a raw .double() chi2 (the (e) hole): the
    # per-chunk append is now the compute-dtype tensor, screened after concat.
    report("no residual producer keeps the raw .double() score path",
           ".double().cpu().numpy())" not in src.replace("cl.double", "X")
           or "chi2fn.chi2(pred=p, target=tw).double()" not in src,
           "compute-dtype accumulation")
    # _screen_diag_chi2 itself calls the ONE shared helper.
    report("_screen_diag_chi2 delegates to screen_chi2",
           "screen_chi2" in calls.get("_screen_diag_chi2", set()),
           "one shared helper")


def main():
    device = torch.device("cpu")
    print("diagnostics-domain gate (torch, CPU)")
    check_screen_chi2(device)
    check_local_linear_floor(device)
    check_cmb_residual(device)
    check_producer_census()
    print()
    if FAILURES:
        print("FAIL: %d check(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("PASS: diagnostics-domain all checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
