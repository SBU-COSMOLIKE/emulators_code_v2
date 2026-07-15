#!/usr/bin/env python3
"""cmb-identity gate: the CMB-emulator save/rebuild/predict identity,
the ruled cosmic-variance constants, the imposed amplitude law both ways, the
roughness legs, the finetune parity, and every CMB-path loud
error — torch only (no cobaya lifecycle, no CAMB).

It builds tiny synthetic CMB emulators by hand (a ParamGeometry over a
written covmat + a CmbDiagonalGeometry over a synthetic fiducial C_ell + a
small ResMLP), saves them with save_emulator, rebuilds, and asserts:
  - from_fiducial computes the RULED covinv (Motloch & Hu eq 3 at zero
    noise): sigma_l = C_fid_l * sqrt(2/(2l+1)), decreasing with l; a
    non-positive fiducial raises naming the multipole;
  - CmbDiagonalGeometry.state() round-trips byte-identical (all nine keys,
    the law strings included);
  - the as_exp2tau_ref law is exact: _factor equals the order-one
    (A_s_ref/A_s) exp(2 (tau - tau_ref)) bitwise, is exactly 1 at the
    fiducial reference and order-one over the box, and encode(decode(x))
    returns x to float32 tolerance; the metric divides the factor back out
 so the physical chi2 is invariant under (A_s, tau) at a fixed
    physical residual, the uncorrected plain sum misses by f^2, the
    roughness residual is factor-corrected, and chi2 without params raises;
    the registry / configure_law loud errors fire, the retired as_exp2tau
    law and the missing-reference case are refused, and a raw-factor
    mutation fails both the fiducial-unity and order-one legs;
  - save -> rebuild -> EmulatorPredictor.predict is bitwise vs the pre-save
    decode, on BOTH laws; the predictor takes the CMB branch and exposes
    spectrum / ell / units / amplitude_law;
  - the correction-head leg: attach_head_coords (one bin, coordinate = ell),
    the identity basis (W_fd / W_df None), the n_tokens segmentation
    (10 windows of 20 multipoles) and its loud range error, the ResTRF
    epoch-0 identity start, the two-phase discipline (set_train_phase
    freezes the right groups per phase, the trunk phase bypasses the
    head bitwise, unknown phases raise — the 2026-07-12 ruling: any
    trunk+head design trains in two phases), and the head save ->
    rebuild -> predict bitwise round-trip (the rebuild-side attach in
    results._rebuild_model);
  - the NPCE check_npce leg (the 2026-07-12 family-wide ruling): the
    residual base + refiner algebra bitwise under the diagonal metric,
    the roughness penalty composing on the full whitened residual,
    save -> rebuild -> predict composing base + net bitwise, and the
    pce x amplitude-law exclusivity loud;
  - the cobaya adapter emul_cmb assembles the Cl dict from two synthetic
    artifacts (shared ell axis, zero-padded below l=2 and outside each
    artifact's range) and raises on: a duplicate spectrum, a wrong-kind
    artifact, an unknown-spectrum or beyond-lmax must_provide, and both
    get_Cl convention violations;
  - the roughness band ratio (period 30 vs 300 at period_cut 50)
    exceeds 100; a zero residual scores exactly zero; the OFF identity (no
    roughness block -> loss bitwise-equal to the plain path); the
    composition (loss == the shared reduction of c_chi2 + lam * c_rough);
    the lensing guard (an acoustic-period residual is penalized at < 3% of
    its own plain chi2 at lam 0.1);
  - a warm start from a CMB artifact reproduces the source function
    at epoch 0 (build_warm_start's own parity check passes and the
    transferred model matches the source bitwise on shared inputs); the
    cosmolike pin refuses a CMB source loudly (wrong-kind); validate_cmb
    accepts a finetune block (the interim error is gone);
  - the eq-6 lens-induced covariance known-answer legs
    (compute_cmb_covariance.py,
    pure numpy): an affine fake CAMBdata makes the 5-point stencil exact,
    so the non-Gaussian contraction is checked against eq 6 built
    directly from the sensitivity matrix and the lensing-potential
    variance. The fake keeps the raw C^phiphi and CAMB's scaled
    [L(L+1)]^2 C/(2 pi) array genuinely distinct, so the pipeline reads
    the raw spectrum for its weight and the scaled one for the
    perturbation exactly as a real run does. The truth leg (the real
    contraction equals the direct eq 6 at round-off), the discrimination
    leg (the earlier band-summed-variance weights miss that truth by
    orders of magnitude, so a check those weights pass would be
    defective), a fixture-integrity leg (scaled and raw differ by an
    L-dependent factor; the raw_cl=True guard raises), and the band leg
    (a width-3 contraction with a response held constant across each band
    reproduces the per-multipole eq 6, and a zeroed last band carries a
    persisted weight of exactly 0).

The adapter legs stub cobaya.theory before loading the shipped emul_cmb.py
(this gate is torch-only; the real cobaya lifecycle is the cmb-smoke board
gate).
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.cmb import CmbDiagonalGeometry
from emulator.geometries.parameter import ParamGeometry
from emulator.losses.cmb import (AMPLITUDE_LAWS, ResidualRoughness,
                                 CmbDiagonalChi2, make_cmb_chi2)
from emulator.losses.core import CosmolikeChi2
from emulator.experiment import validate_cmb
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor
from emulator import fixed_facts
from emulator import warmstart

FAILURES = []

IN_NAMES = ["As", "tau", "omegam"]
N_IN = len(IN_NAMES)

# The adapter legs assemble a TT artifact and an EE artifact into one Cl dict
# on a shared ell axis: two spectra of one cosmology, served together as one
# theory block. They are one dataset, so they carry one identity, which is what
# handing them the same label does.
ADAPTER_PAIR_LABEL = "cmb-identity/adapter-spectra-pair"

# The region every double this gate PREDICTS THROUGH declares it stands for.
#
# An emulator may only be asked about a point inside the region it was trained
# over: outside it the network does not fail, it extrapolates, and it returns a
# confident number that is wrong. A double standing in for a real CMB emulator
# must therefore say which region that is, and these are the bounds a real LCDM
# run would have drawn from -- A_s near 2.1e-9, the optical depth near 0.055,
# omegam near 0.31. Every point the legs below ask about (round-trip, head,
# NPCE, adapter) sits within a few percent of that fiducial, well inside this
# box, and the synthetic training rows are drawn around it too.
#
# A double that is only saved, rebuilt, or refused -- never asked a question --
# declares NO support instead. That is the honest record for it, and
# check_domain_law proves such a double answers nothing.
CMB_SUPPORT = {"As":     (1.6e-9, 2.6e-9),
               "tau":    (0.02, 0.12),
               "omegam": (0.24, 0.40)}

# the fixture fiducial reference pair the order-one law measures against
#: the recommended values are the covariance's own fiducial, so
# these mirror a plausible LCDM (A_s ~ 2.1e-9, tau ~ 0.0544). The factor
# is exactly 1 at (As, tau) == (AS_REF_FIXTURE, TAU_REF_FIXTURE).
AS_REF_FIXTURE  = 2.1e-9
TAU_REF_FIXTURE = 0.0544


def report(label, ok, detail):
    """Print one PASS/FAIL line and remember any failure."""
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


def emit_aid(aid, n_before):
    """Emit ONE '##AID <aid> <PASS|FAIL>' line for a whole acceptance leg.

    (queue 2) The board's run_check folds these reserved lines into the gate's
    executed set: one per drafted leg, at the leg's aggregation point, not per
    sub-check. A leg groups several report() calls (one or more check_*
    functions); the leg's verdict is FAIL if that group appended any label to
    the module-level FAILURES list since it started. The child's exit status
    stays the single aggregate verdict; these lines add no new judgement.

    Arguments:
      aid      = the drafted board-unique leg id, "cmb-identity.<leg>".
      n_before = len(FAILURES) captured immediately before the leg's checks ran.
    """
    mark = "PASS" if len(FAILURES) == n_before else "FAIL"
    print("##AID " + aid + " " + mark)


def spd(n, seed):
    """A random symmetric positive-definite matrix (n x n)."""
    g = np.random.default_rng(seed)
    a = g.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def write_covmat(path, names, seed):
    """Write a covmat file (a "#"-prefixed header line + an SPD matrix)."""
    cov = spd(len(names), seed=seed)
    # keep the As / tau scales physical-ish so the law factor is benign.
    scale = np.ones(len(names))
    for i, nm in enumerate(names):
        if nm == "As":
            scale[i] = 1e-10
        elif nm == "tau":
            scale[i] = 0.01
    cov = cov * np.outer(scale, scale)
    with open(path, "w") as f:
        f.write("# " + " ".join(names) + "\n")
        for row in cov:
            f.write(" ".join(repr(float(x)) for x in row) + "\n")


def synth_ell_cl(n_ell=200, lmin=2):
    """A smooth, strictly positive synthetic fiducial C_ell."""
    ell = np.arange(lmin, lmin + n_ell)
    cl = 2500.0 * np.exp(-ell / 300.0) + 5.0
    return ell, cl


def make_pgeom(tmp, device, seed=0):
    """A ParamGeometry over a written covmat with the As/tau/omegam names."""
    path = os.path.join(tmp, "params_%d.covmat" % seed)
    write_covmat(path, IN_NAMES, seed=seed + 1)
    center = np.array([2.1e-9, 0.055, 0.31])
    return ParamGeometry.from_covmat(device=device, center=center,
                                     covmat_path=path), path


def cmb_recipe(n_ell):
    """The model_recipe a schema-v2 save stores for the CMB ResMLP."""
    return {
        "cls": "emulator.designs.plain.ResMLP",
        "name": "resmlp",
        "ia": None,
        "input_dim": N_IN,
        "output_dim": n_ell,
        "compile_mode": None,
        "needs_geom": False,
        "kwargs": {
            "int_dim_res": 16,
            "n_blocks": 2,
            "block_opts": {"act": {"type": "H", "n_gates": 3},
                           "norm": "affine"},
        },
    }


def save_synthetic_cmb(root, device, tmp, label, spectrum="tt",
                       law="as_exp2tau_ref", n_ell=200, seed=0, support=None):
    """Build, then save, a tiny synthetic CMB emulator under `root`.

    `label` is what this double is for. It fixes the identity of the scientific
    record the saved file carries; the comment at the save below says why the
    file carries one at all.

    `support` is the region the double stands for, as a mapping name -> (low,
    high). A double the gate PREDICTS through declares one -- the box a real
    emulator of this shape would have been drawn from -- because a prediction is
    refused unless the point lies inside the declared region. A double that is
    only saved, rebuilt or refused declares none: it is a test double, it is
    never asked a question, and the record says so.

    Returns:
      (pgeom, geom, model, chi2fn): the sources + the pre-save model and
      the law-dispatched chi2 (its decode is the round-trip reference).
    """
    pgeom, covmat_path = make_pgeom(tmp, device, seed=seed)
    ell, cl = synth_ell_cl(n_ell=n_ell)
    g = np.random.default_rng(seed + 3)
    center = cl * (1.0 + 0.01 * g.standard_normal(n_ell))
    units = "dimensionless" if spectrum == "pp" else "muK2"
    is_ref = (law == "as_exp2tau_ref")
    as_name  = "As" if is_ref else ""
    tau_name = "tau" if is_ref else ""
    as_ref   = AS_REF_FIXTURE if is_ref else None
    tau_ref  = TAU_REF_FIXTURE if is_ref else None
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum=spectrum, ell=ell, fiducial_cl=cl,
        center=center, units=units, law=law,
        as_name=as_name, tau_name=tau_name,
        as_ref=as_ref, tau_ref=tau_ref)
    if law == "none":
        chi2fn = make_cmb_chi2(geom=geom, law=law)
    else:
        chi2fn = make_cmb_chi2(geom=geom, law=law, param_geometry=pgeom,
                               as_name=as_name, tau_name=tau_name,
                               as_ref=as_ref, tau_ref=tau_ref)
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=n_ell, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    config = {"data": {"cmb": {"spectrum": spectrum,
                               "covariance": "cov.npz",
                               "amplitude_law": law},
                       "train_dv": "t.npy",
                       "val_dv": "v.npy",
                       "train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat_path)},
              "train_args": {"nepochs": 1}}
    if is_ref:
        config["data"]["cmb"]["as_name"] = "As"
        config["data"]["cmb"]["tau_name"] = "tau"
        config["data"]["cmb"]["as_ref"] = as_ref
        config["data"]["cmb"]["tau_ref"] = tau_ref
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
    save_emulator(path_root=str(root),
                  model=model,
                  param_geometry=pgeom,
                  geometry=geom,
                  config=config,
                  histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=cmb_recipe(n_ell),
                  facts_yaml=fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label=label,
                      family="cmb",
                      support=support),
                  attrs={"rescale": "none", "spectrum": spectrum})
    return pgeom, geom, model, chi2fn


def check_ruled_constants(device):
    """The covinv RULING: sigma_l = C_fid_l * sqrt(2/(2l+1)), decreasing."""
    ell, cl = synth_ell_cl()
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum="tt", ell=ell, fiducial_cl=cl,
        center=cl, units="muK2", law="none")
    want = (cl * np.sqrt(2.0 / (2.0 * ell + 1.0))).astype("float32")
    got = geom.sigma.cpu().numpy()
    report("sigma_l equals the ruled C_fid*sqrt(2/(2l+1))",
           np.array_equal(got, want),
           "max|d| = %.2e" % np.abs(got - want).max())
    report("sigma decreases with l on a decaying fiducial",
           got[0] > got[-1], "sigma[2] %.2f > sigma[max] %.4f"
           % (got[0], got[-1]))
    try:
        CmbDiagonalGeometry.from_fiducial(
            device=device, spectrum="te", ell=ell,
            fiducial_cl=np.where(ell == 50, 0.0, cl),
            center=cl, units="muK2", law="none")
        report("non-positive fiducial raises", False, "did not raise")
    except ValueError as e:
        report("non-positive fiducial raises naming the multipole",
               "50" in str(e), "ValueError names l=50")


def check_state_roundtrip(device):
    """CmbDiagonalGeometry.state() round-trips byte-identical (nine keys)."""
    ell, cl = synth_ell_cl()
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum="ee", ell=ell, fiducial_cl=cl,
        center=cl * 1.01, units="muK2", law="as_exp2tau_ref",
        as_name="As", tau_name="tau",
        as_ref=AS_REF_FIXTURE, tau_ref=TAU_REF_FIXTURE)
    st0 = geom.state()
    geom2 = CmbDiagonalGeometry.from_state(device=device, state=st0)
    st1 = geom2.state()
    ok = set(st0) == set(st1)
    for k in st0:
        a, b = st0[k], st1[k]
        if isinstance(a, torch.Tensor):
            ok = ok and torch.equal(a, b)
        else:
            ok = ok and (a == b)
    report("geometry state round-trip byte-identical",
           ok, "%d keys incl. law strings + fiducial refs" % len(st0))
    # the fiducial reference pair persisted as resolved float64 and rebuilt
    # byte-exact (the numbers the artifact records, not a code default).
    report("state persists as_ref / tau_ref as float64",
           st0["as_ref"].dtype == torch.float64
           and st0["tau_ref"].dtype == torch.float64,
           "0-d float64 tensors")
    report("as_ref / tau_ref round-trip byte-exact",
           geom2.as_ref == AS_REF_FIXTURE
           and geom2.tau_ref == TAU_REF_FIXTURE,
           "as_ref %.3e, tau_ref %.4f" % (geom2.as_ref, geom2.tau_ref))


def check_law(tmp, device):
    """The as_exp2tau_ref law: order-one _factor exact, encode/decode
    inverse, the fiducial-unity + order-one legs, the retired-law and
    missing-reference refusals, and the raw-factor mutation arm."""
    pgeom, _ = make_pgeom(tmp, device, seed=50)
    ell, cl = synth_ell_cl()
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum="tt", ell=ell, fiducial_cl=cl,
        center=cl, units="muK2", law="as_exp2tau_ref",
        as_name="As", tau_name="tau",
        as_ref=AS_REF_FIXTURE, tau_ref=TAU_REF_FIXTURE)
    chi2fn = make_cmb_chi2(geom=geom, law="as_exp2tau_ref",
                           param_geometry=pgeom,
                           as_name="As", tau_name="tau",
                           as_ref=AS_REF_FIXTURE, tau_ref=TAU_REF_FIXTURE)
    g = np.random.default_rng(60)
    theta = np.column_stack([g.normal(2.1e-9, 1e-10, 8),
                             g.normal(0.055, 0.005, 8),
                             g.normal(0.31, 0.01, 8)])
    x = torch.as_tensor(theta, dtype=torch.float32, device=device)
    x_enc = pgeom.encode(x)
    f = chi2fn._factor(x_enc)
    phys = pgeom.decode(x_enc)
    want = ((AS_REF_FIXTURE / phys[:, 0])
            * torch.exp(2.0 * (phys[:, 1] - TAU_REF_FIXTURE))).reshape(-1, 1)
    report("_factor equals the order-one (As_ref/A_s) exp(2(tau-tau_ref)) "
           "bitwise (same decoded params)",
           torch.equal(f, want), "max|d| = %.2e"
           % (f - want).abs().max().item())
    # the factor is exactly 1 at the fiducial reference and stays
    # order-one over the sampled box (the retired raw exp(2 tau)/A_s is
    # ~5e8 there). Build a row whose DECODED params are exactly the
    # fiducial pair, then read the factor.
    phys_fid = phys.clone()
    phys_fid[:, 0] = AS_REF_FIXTURE
    phys_fid[:, 1] = TAU_REF_FIXTURE
    x_fid = pgeom.encode(phys_fid)               # encode(decode) is identity
    f_fid = chi2fn._factor(x_fid)
    report("factor == 1 at the fiducial reference",
           torch.allclose(f_fid, torch.ones_like(f_fid), rtol=0, atol=1e-6),
           "max|f_fid - 1| = %.2e" % (f_fid - 1.0).abs().max().item())
    report("factor is order-one over the sampled box (not the raw ~5e8)",
           float(f.min()) > 0.5 and float(f.max()) < 2.0,
           "f in [%.3f, %.3f]" % (f.min().item(), f.max().item()))
    # encode(decode(pred)) returns pred to float32 round-off (the factor
    # multiplies and divides, so bitwise is not guaranteed; the bar is a
    # tight relative tolerance).
    pred = torch.randn(8, len(ell), device=device)
    back = chi2fn.encode(chi2fn.decode(pred, x_enc), x_enc)
    rel = ((back - pred).abs().max()
           / pred.abs().max()).item()
    report("encode(decode(x)) round-trips to float32 round-off",
           rel < 1e-4, "max rel %.2e" % rel)

    # the metric divides the per-row factor back out, so the physical
    # chi2 is invariant under (A_s, tau) at a FIXED physical residual, where a
    # plain sum of the whitened residual would carry f^2. Build pred/target
    # whose whitened residual is f * phys_r for one shared physical residual;
    # the corrected chi2 must be identical across the (varying-f) rows.
    phys_r = torch.randn(len(ell), device=device).reshape(1, -1).expand(
        8, -1).contiguous()
    target0 = torch.zeros(8, len(ell), device=device)
    pred0 = f * phys_r                              # whitened residual = f*phys_r
    phys_chi2 = (phys_r ** 2).sum(dim=1)
    c_corr = chi2fn.chi2(pred0, target0, params_whitened=x_enc)
    spread = (c_corr.max() - c_corr.min()).item()
    report("factored chi2 == physical chi2, invariant under (A_s, tau)",
           torch.allclose(c_corr, phys_chi2, rtol=1e-4, atol=1e-6)
           and spread < 1e-3 * float(c_corr.mean()),
           "row spread %.2e over f in [%.2e, %.2e]"
           % (spread, f.min().item(), f.max().item()))
    c_unc = ((pred0 - target0) ** 2).sum(dim=1)      # the uncorrected metric
    report("the uncorrected metric misses by exactly f^2 (catch-power)",
           torch.allclose(c_unc, (f.reshape(-1) ** 2) * phys_chi2, rtol=1e-4),
           "old / corrected == f^2")
    chi2fn._params = x_enc                            # roughness runs after loss stashes
    pr = chi2fn._penalty_residual(pred0, target0)
    report("roughness residual is factor-corrected (law-neutral)",
           torch.allclose(pr, phys_r, rtol=1e-4, atol=1e-6),
           "penalty residual == physical residual")
    chi2fn._params = None
    try:
        chi2fn.chi2(pred0, target0)
        report("factored chi2 without params raises", False, "no raise")
    except ValueError:
        report("factored chi2 without params raises", True, "ValueError")

    # unit 70 (20M-02): the stale-stash discrimination -- the analytic control
    # the diagnostic defect turns on. Two rows with amplitude factors f=[2, 0.5]
    # (tau at the reference so exp(2(tau-tau_ref))=1, A_s = as_ref/2 and
    # 2*as_ref), three multipoles, a zero prediction, and a whitened target set
    # to f per multipole (physical unit truth). Scoring with THIS row's params
    # gives the physical chi2 [3, 3]; reading a stale stash whose factor is
    # [1, 1] (the fiducial the LAST training batch left behind) gives the
    # shipped defect [12, 0.75].
    n_ell3 = 3
    phys_ctrl = phys[:2].clone()
    phys_ctrl[:, chi2fn.tau_idx] = TAU_REF_FIXTURE
    phys_ctrl[:, chi2fn.as_idx] = torch.tensor(
        [AS_REF_FIXTURE / 2.0, AS_REF_FIXTURE * 2.0], device=device)
    x_ctrl = pgeom.encode(phys_ctrl)
    f_ctrl = chi2fn._factor(x_ctrl).reshape(-1)
    report("control factors are exactly [2, 0.5]",
           torch.allclose(f_ctrl, torch.tensor([2.0, 0.5], device=device),
                          rtol=1e-4, atol=1e-5),
           "f = [%.4f, %.4f]" % (f_ctrl[0].item(), f_ctrl[1].item()))
    target_ctrl = f_ctrl.reshape(-1, 1) * torch.ones(2, n_ell3, device=device)
    pred_ctrl = torch.zeros(2, n_ell3, device=device)
    # a stash from a DIFFERENT (fiducial) batch: its factor is [1, 1].
    phys_stale = phys[:2].clone()
    phys_stale[:, chi2fn.as_idx] = AS_REF_FIXTURE
    phys_stale[:, chi2fn.tau_idx] = TAU_REF_FIXTURE
    x_stale = pgeom.encode(phys_stale)
    c_correct = chi2fn.chi2(pred_ctrl, target_ctrl, params_whitened=x_ctrl)
    report("params-passing chi2 is the physical [3, 3] (unit 70 caller fix)",
           torch.allclose(c_correct, torch.tensor([3.0, 3.0], device=device),
                          rtol=1e-4, atol=1e-4),
           "c = [%.4f, %.4f]" % (c_correct[0].item(), c_correct[1].item()))
    chi2fn._params = x_stale
    c_stale = chi2fn.chi2(pred_ctrl, target_ctrl)     # omitted -> reads stash
    report("the omitted-params path reads the stale stash: the [12, 0.75] "
           "defect (mutation arm)",
           torch.allclose(c_stale, torch.tensor([12.0, 0.75], device=device),
                          rtol=1e-4, atol=1e-4),
           "c = [%.4f, %.4f]" % (c_stale[0].item(), c_stale[1].item()))
    # the caller fix is invariant to whatever the stash holds: passing params
    # gives [3, 3] no matter what a prior loss stashed.
    chi2fn._params = x_stale
    c_inv = chi2fn.chi2(pred_ctrl, target_ctrl, params_whitened=x_ctrl)
    report("params-passing chi2 ignores the stash (stash-invariant)",
           torch.allclose(c_inv, c_correct, rtol=0, atol=0), "byte-identical")
    # a stale stash of a DIFFERENT batch length would broadcast-crash the
    # omitted path; passing this batch's params sizes the factor correctly.
    chi2fn._params = pgeom.encode(phys[:5].clone())   # 5 rows, not 2
    crashed = False
    try:
        chi2fn.chi2(pred_ctrl, target_ctrl, params_whitened=x_ctrl)
    except RuntimeError:
        crashed = True
    report("params-passing chi2 survives a wrong-length stale stash "
           "(no shape crash)", not crashed, "factor sized from this batch")
    chi2fn._params = None

    try:
        make_cmb_chi2(geom=geom, law="not_a_law")
        report("unknown law raises", False, "did not raise")
    except ValueError:
        report("unknown law raises", True, "ValueError")
    # the retired raw-factor law is refused with its retrain instruction
    #: an old convention name is never silently reinterpreted.
    try:
        make_cmb_chi2(geom=geom, law="as_exp2tau", param_geometry=pgeom,
                      as_name="As", tau_name="tau")
        report("retired as_exp2tau law refused with retrain error",
               False, "no raise")
    except ValueError as e:
        report("retired as_exp2tau law refused with retrain error",
               "retired" in str(e) and "retrain" in str(e), "ValueError")
    # the order-one law with its columns but WITHOUT the fiducial refs is
    # refused (the numbers are required, no code default).
    try:
        make_cmb_chi2(geom=geom, law="as_exp2tau_ref", param_geometry=pgeom,
                      as_name="As", tau_name="tau")
        report("as_exp2tau_ref without as_ref/tau_ref raises",
               False, "no raise")
    except ValueError as e:
        report("as_exp2tau_ref without as_ref/tau_ref raises",
               "as_ref" in str(e) and "tau_ref" in str(e), "ValueError")
    try:
        make_cmb_chi2(geom=geom, law="as_exp2tau_ref", param_geometry=pgeom,
                      as_name="NOPE", tau_name="tau",
                      as_ref=AS_REF_FIXTURE, tau_ref=TAU_REF_FIXTURE)
        report("configure_law bad column raises", False, "no raise")
    except ValueError:
        report("configure_law bad column raises", True, "ValueError")
    # a non-positive as_ref divides the factor by <= 0, so configure_law
    # refuses it (finite validation before the comparative law is applied).
    try:
        make_cmb_chi2(geom=geom, law="as_exp2tau_ref", param_geometry=pgeom,
                      as_name="As", tau_name="tau",
                      as_ref=-1.0, tau_ref=TAU_REF_FIXTURE)
        report("configure_law non-positive as_ref raises", False, "no raise")
    except ValueError as e:
        report("configure_law non-positive as_ref raises",
               "positive" in str(e), "ValueError")

    # --- the raw-factor mutation arm: a loss whose _factor
    # restores the RETIRED raw exp(2 tau)/A_s must FAIL both the
    # fiducial-unity leg (f_fid ~ 5e8, not 1) and the order-one leg. This
    # proves the two red legs actually discriminate the fixed law from the
    # regression they guard against.
    class _RawFactorChi2(type(chi2fn)):
        def _factor(self, params_whitened):
            phys = self.param_geometry.decode(params_whitened)
            a_s  = phys[:, self.as_idx]
            tau  = phys[:, self.tau_idx]
            return (torch.exp(2.0 * tau) / a_s).reshape(-1, 1)

    raw = _RawFactorChi2(geom=geom).configure_law(
        param_geometry=pgeom, as_name="As", tau_name="tau",
        as_ref=AS_REF_FIXTURE, tau_ref=TAU_REF_FIXTURE)
    raw_fid = raw._factor(x_fid)
    report("mutation (raw factor) FAILS the fiducial-unity leg",
           not torch.allclose(raw_fid, torch.ones_like(raw_fid),
                              rtol=0, atol=1e-6),
           "raw f_fid ~ %.2e (not 1)" % raw_fid.mean().item())
    report("mutation (raw factor) FAILS the order-one leg",
           float(raw._factor(x_enc).max()) > 1e6,
           "raw f max %.2e" % raw._factor(x_enc).max().item())


def check_roundtrip(tmp, device, law):
    """save -> rebuild -> predict bitwise vs the pre-save decode."""
    root = os.path.join(tmp, "emul_cmb_" + law)
    pgeom, geom, model, chi2fn = save_synthetic_cmb(
        root, device, tmp, label="cmb-identity/round-trip-" + law,
        spectrum="tt", law=law, seed=70, support=CMB_SUPPORT)
    g = np.random.default_rng(80)
    theta = np.array([[2.1e-9, 0.055, 0.31]]) * (1.0 + 0.01
                                                 * g.standard_normal((1, 3)))
    x = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    x_enc = pgeom.encode(x)
    with torch.no_grad():
        p = model(x_enc)
        if law == "none":
            ref = chi2fn.decode(p)[0]
        else:
            ref = chi2fn.decode(p, x_enc)[0]
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    ok = np.array_equal(got, ref.cpu().numpy())
    report("predict round-trip bitwise (%s law)" % law, ok,
           "max|d| = %.2e" % np.abs(got - ref.cpu().numpy()).max())
    _, _, _, info = rebuild_emulator(root, device, compile_model=False)
    report("rebuild info: cmb + law (%s)" % law,
           info["cmb"] and info["amplitude_law"] == law
           and not info["scalar"],
           "cmb %s law %s" % (info["cmb"], info["amplitude_law"]))
    report("predictor CMB branch exposes spectrum/ell/units (%s)" % law,
           getattr(pred, "_cmb", False) and pred.spectrum == "tt"
           and pred.units == "muK2"
           and int(pred.ell[0]) == 2,
           "spectrum %s units %s" % (pred.spectrum, pred.units))
    return root


def check_domain_law(tmp, device):
    """The domain law at the predictor's door, on a saved CMB artifact.

    An emulator asked outside the region it was trained over does not fail. It
    extrapolates: a spectrum of the right shape, with the right sign, and no
    warning. predict() therefore proves the point lies inside the region the
    artifact's own record declares before any number reaches the network, and
    these two arms drive both halves of that law through a real save + rebuild.

    Both arms read the WORDS of the refusal. They have to: float("n/a") raises
    the same ValueError class every refusal here raises, so an arm that only
    asked "did it raise?" would stay green while the law it names was broken.
    """
    # a double that declares no support: an emulator generated by nobody, valid
    # over nowhere. It saves, it rebuilds -- and it answers nothing.
    root_none = os.path.join(tmp, "emul_cmb_undeclared")
    save_synthetic_cmb(root_none, device, tmp,
                       label="cmb-identity/undeclared-double",
                       spectrum="tt", law="none", seed=160)
    pred_none = EmulatorPredictor(root_none, device, compile_model=False)
    try:
        pred_none.predict({"As": 2.1e-9, "tau": 0.055, "omegam": 0.31})
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
    root_box = os.path.join(tmp, "emul_cmb_boxed")
    save_synthetic_cmb(root_box, device, tmp,
                       label="cmb-identity/boxed-double",
                       spectrum="tt", law="none", seed=170,
                       support=CMB_SUPPORT)
    pred_box = EmulatorPredictor(root_box, device, compile_model=False)
    inside = pred_box.predict({"As": 2.1e-9, "tau": 0.055, "omegam": 0.31})
    report("a point inside the declared box is served",
           inside.shape == (200,),
           "spectrum of %d multipoles" % inside.shape[0])
    # omegam = 0.9 is far outside the box the record declares. Nothing about the
    # spectrum the network would return there would look wrong.
    try:
        pred_box.predict({"As": 2.1e-9, "tau": 0.055, "omegam": 0.9})
        report("a point outside the declared box refuses", False,
               "extrapolated without a word")
    except ValueError as e:
        report_refusal("a point outside the declared box refuses", e,
                       needle="which is outside it",
                       law="the domain law (outside the declared box)")


def _load_emul_cmb_stubbed():
    """Load the shipped emul_cmb.py with cobaya.theory stubbed."""
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

    root = Path(__file__).resolve().parents[3]
    path = root / "cobaya_theory" / "emul_cmb.py"
    spec = importlib.util.spec_from_file_location("emul_cmb_shim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.emul_cmb


def _build_adapter(cls, roots):
    """Instantiate the stubbed emul_cmb, set extra_args, initialize."""
    t = cls()
    t.extra_args = {"device": "cpu", "emulators": list(roots)}
    t.initialize()
    return t


def check_adapter(tmp, device):
    """emul_cmb: Cl assembly + every loud error, on synthetic artifacts."""
    cls = _load_emul_cmb_stubbed()
    root_tt = os.path.join(tmp, "ad_tt")
    save_synthetic_cmb(root_tt, device, tmp, label=ADAPTER_PAIR_LABEL,
                       spectrum="tt",
                       law="as_exp2tau_ref", n_ell=200, seed=90,
                       support=CMB_SUPPORT)
    root_ee = os.path.join(tmp, "ad_ee")
    save_synthetic_cmb(root_ee, device, tmp, label=ADAPTER_PAIR_LABEL,
                       spectrum="ee",
                       law="none", n_ell=100, seed=100,
                       support=CMB_SUPPORT)

    t = _build_adapter(cls, [root_tt, root_ee])
    report("requirements = the stored input-name union",
           set(t.get_requirements()) == set(IN_NAMES),
           "req %s" % (sorted(t.get_requirements()),))
    # calculate: assembly on the shared ell axis, zero-padded.
    state = {}
    t.calculate(state, want_derived=True,
                As=2.1e-9, tau=0.055, omegam=0.31)
    cl = state["Cl"]
    lmax_tt, lmax_ee = 201, 101
    ok = (np.array_equal(cl["ell"], np.arange(lmax_tt + 1))
          and cl["tt"].shape == (lmax_tt + 1,)
          and (cl["tt"][:2] == 0).all()
          and (cl["tt"][2:] != 0).any()
          and (cl["ee"][:2] == 0).all()
          and (cl["ee"][lmax_ee + 1:] == 0).all()
          and (cl["ee"][2:lmax_ee + 1] != 0).any())
    report("Cl dict assembly (shared axis, zero-padded)", ok,
           "ell 0..%d, ee zero beyond %d" % (lmax_tt, lmax_ee))
    # must_provide legs.
    t.must_provide(Cl={"tt": 201, "ee": 101})   # valid: no raise
    try:
        t.must_provide(Cl={"pp": 10})
        report("unknown-spectrum must_provide raises", False, "no raise")
    except ValueError as e:
        report_refusal("unknown-spectrum must_provide raises", e,
                       needle="no loaded artifact provides",
                       law="the unknown-spectrum law")
    try:
        t.must_provide(Cl={"ee": 150})
        report("beyond-lmax must_provide raises", False, "no raise")
    except ValueError as e:
        report_refusal("beyond-lmax must_provide raises", e,
                       needle="no accuracy beyond its training grid",
                       law="the beyond-lmax law")
    # get_Cl convention guards.
    t.current_state = {"Cl": cl}
    try:
        t.get_Cl(ell_factor=True)
        report("get_Cl ell_factor guard", False, "no raise")
    except ValueError as e:
        report_refusal("get_Cl ell_factor guard", e,
                       needle="serves raw C_ell only",
                       law="the raw-C_ell convention")
    try:
        t.get_Cl(units="FIRASmuK2")
        report("get_Cl units guard", False, "no raise")
    except ValueError as e:
        report_refusal("get_Cl units guard", e,
                       needle="never converts silently",
                       law="the artifact-units convention")
    # duplicate spectrum.
    root_tt2 = os.path.join(tmp, "ad_tt2")
    # a second TT emulator, built to be refused beside the first. It carries the
    # PAIR's label on purpose: two emulators of one spectrum, both trained off
    # ONE generator dump, is exactly the ambiguity the duplicate law exists to
    # refuse -- one dataset, one identity. Give this double an identity of its
    # own instead and the served pair stops being one dataset: the identity law
    # refuses it first, and the duplicate law -- the law this leg exists to
    # prove -- is never reached. The adapter never gets as far as a prediction
    # here, so the double declares no support: it is never asked a point.
    save_synthetic_cmb(root_tt2, device, tmp,
                       label=ADAPTER_PAIR_LABEL,
                       spectrum="tt",
                       law="none", n_ell=50, seed=110)
    try:
        _build_adapter(cls, [root_tt, root_tt2])
        report("duplicate spectrum raises", False, "no raise")
    except ValueError as e:
        report_refusal("duplicate spectrum raises", e,
                       needle="two emulators provide the spectrum",
                       law="the duplicate-spectrum law")

    # the dataset-identity law: everything the adapter serves is combined into
    # one theory block, so it must all come from ONE generator dump. This pair
    # is topologically PERFECT -- a TT emulator and an EE emulator, one spectrum
    # each, one units convention -- so every configuration law the adapter runs
    # first passes and the refusal that fires is the identity one. The only
    # thing wrong with the pair is what the arm is about: the EE double was
    # saved under a label of its own, so its record carries a different dataset
    # identity, and two emulators fitted to different datasets describe two
    # different universes however well their axes line up.
    root_ee2 = os.path.join(tmp, "ad_ee_other")
    save_synthetic_cmb(root_ee2, device, tmp,
                       label="cmb-identity/adapter-other-dataset",
                       spectrum="ee",
                       law="none", n_ell=100, seed=115)
    try:
        _build_adapter(cls, [root_tt, root_ee2])
        report("two datasets served together raise", False, "no raise")
    except ValueError as e:
        report_refusal("two datasets served together raise", e,
                       needle="different datasets",
                       law="the dataset-identity law")


def check_roughness(device):
    """Roughness: band ratio, zero, OFF identity, composition, and the
    lensing guard."""
    n_ell = 3000
    ell = torch.arange(n_ell, dtype=torch.float32, device=device)
    r30 = torch.sin(2 * np.pi * ell / 30.0).reshape(1, -1)
    r300 = torch.sin(2 * np.pi * ell / 300.0).reshape(1, -1)
    rough = ResidualRoughness(period_cut=50, device=device)
    p30 = rough.per_sample(r30).item()
    p300 = rough.per_sample(r300).item()
    report("band ratio (period 30 vs 300) > 100", p30 / p300 > 100.0,
           "ratio %.0f" % (p30 / p300))
    z = rough.per_sample(torch.zeros(2, n_ell, device=device))
    report("zero residual scores exactly zero", bool((z == 0).all()),
           "max %.1e" % z.max().item())

    # OFF identity: a CMB chi2fn with NO roughness must produce the loss the
    # plain path produces, bitwise.
    ell_np, cl_np = synth_ell_cl(n_ell=256)
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum="tt", ell=ell_np, fiducial_cl=cl_np,
        center=cl_np, units="muK2", law="none")
    chi2fn = make_cmb_chi2(geom=geom, law="none")
    g = torch.Generator().manual_seed(7)
    pred = torch.randn(32, 256, generator=g)
    targ = torch.randn(32, 256, generator=g)
    loss_off = chi2fn.loss(pred, targ, mode="sqrt", trim=0.05,
                           focus=0.0, focus_scale=1.0)
    loss_plain = CosmolikeChi2.loss(chi2fn, pred, targ, mode="sqrt",
                                    trim=0.05, focus=0.0, focus_scale=1.0)
    report("OFF identity: no roughness block == the plain loss, bitwise",
           torch.equal(loss_off, loss_plain),
           "d = %.2e" % (loss_off - loss_plain).abs().item())

    # composition: with roughness, loss == the shared reduction of
    # c_chi2 + lam * c_rough (one reduction path, no second ladder).
    chi2fn.configure_roughness(lam=0.1, period_cut=50)
    loss_on = chi2fn.loss(pred, targ, mode="sqrt", trim=0.05,
                          focus=0.0, focus_scale=1.0)
    c = chi2fn.chi2(pred=pred, target=targ)
    c_tot = c + 0.1 * chi2fn._rough.per_sample(pred - targ)
    want = chi2fn._reduce(c=c_tot, mode="sqrt", trim=0.05, focus=0.0,
                          focus_scale=1.0)
    report("composition: c_chi2 + lam*c_rough through one reduction",
           torch.equal(loss_on, want),
           "d = %.2e" % (loss_on - want).abs().item())

    # lensing guard: an acoustic-period residual (the lensed-minus-unlensed
    # peak-smoothing signature, period ~250) must be penalized at a few
    # percent of its own plain chi2 at most.
    ell_l = torch.arange(3000, dtype=torch.float32, device=device)
    r_ac = (torch.sin(2 * np.pi * ell_l / 250.0)
            * torch.exp(-((ell_l - 1500.0) / 900.0) ** 2)).reshape(1, -1)
    plain = (r_ac * r_ac).sum().item()
    pen = 0.1 * rough.per_sample(r_ac).item()
    report("lensing guard: acoustic-period penalty < 3% of its chi2",
           pen / plain < 0.03, "ratio %.2e" % (pen / plain))


def cmb_head_recipe(n_ell):
    """The model_recipe a schema-v2 save stores for the CMB ResTRF leg:
    needs_geom True (rebuild re-injects the geometry after
    attach_head_coords), every constructor default materialized, the
    n_tokens segmentation recorded."""
    return {
        "cls": "emulator.designs.plain.ResTRF",
        "name": "restrf",
        "ia": None,
        "input_dim": N_IN,
        "output_dim": n_ell,
        "compile_mode": None,
        "needs_geom": True,
        "kwargs": {
            "int_dim_res": 16,
            "n_blocks": 2,
            "n_heads": 2,
            "n_blocks_trf": 1,
            "n_mlp_blocks": 1,
            "n_tokens": 10,
            "gate_init": 0.1,
            "shared_mlp": False,
            "film": False,
            "head_act": None,
            "block_opts": {"act": {"type": "H", "n_gates": 3},
                           "norm": "affine"},
        },
    }


def check_head(tmp, device):
    """The correction-head leg: the TRF head on the CMB geometry — the
    attach, the identity basis, the epoch-0 identity start, the
    n_tokens loud error, and save -> rebuild -> predict bitwise
    (proving the rebuild-side attach_head_coords in
    results._rebuild_model)."""
    from emulator.designs.plain import ResTRF
    n_ell = 200
    pgeom, covmat_path = make_pgeom(tmp, device, seed=140)
    ell, cl = synth_ell_cl(n_ell=n_ell)
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum="tt", ell=ell, fiducial_cl=cl,
        center=cl, units="muK2", law="none")
    geom.attach_head_coords()
    report("attach_head_coords: one bin covering the spectrum",
           geom.bin_sizes == [n_ell],
           "bin_sizes %s" % (geom.bin_sizes,))
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResTRF(input_dim=N_IN, output_dim=n_ell, int_dim_res=16,
                   geom=geom, n_heads=2, n_blocks=2, n_blocks_trf=1,
                   n_mlp_blocks=1, n_tokens=10,
                   block_opts=block_opts).to(device)
    report("identity basis: W_fd / W_df stay None on the CMB geometry",
           model.W_fd is None and model.W_df is None,
           "n_bins %d, max_bin %d" % (model.n_bins, model.max_bin))
    report("n_tokens segmentation: 10 windows of 20 multipoles",
           model.n_bins == 10 and model.max_bin == 20,
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
    # phase is loud. At the identity init every phase's output equals
    # the trunk, so the flags are the discriminating assertions.
    model.set_train_phase("trunk")
    trunk_on = all(p.requires_grad for p in model.mlp.parameters())
    head_off = all(not p.requires_grad for p in model.trf.parameters())
    with torch.no_grad():
        bypass = model(x)
    model.set_train_phase("head")
    trunk_off = all(not p.requires_grad for p in model.mlp.parameters())
    head_on = (all(p.requires_grad for p in model.trf.parameters())
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
        ResTRF(input_dim=N_IN, output_dim=n_ell, int_dim_res=8,
               geom=geom, n_tokens=n_ell + 1, block_opts=block_opts)
        report("n_tokens beyond the spectrum raises", False, "no raise")
    except ValueError as e:
        report("n_tokens beyond the spectrum raises",
               "2.." in str(e), "ValueError names the valid range")
    # save -> rebuild -> predict, bitwise: the rebuilt geometry has NO
    # bin_sizes in its state (derived, never persisted), so this leg
    # fails unless _rebuild_model re-attaches before construction.
    chi2fn = make_cmb_chi2(geom=geom, law="none")
    root = os.path.join(tmp, "emul_cmb_head")
    config = {"data": {"cmb": {"spectrum": "tt",
                               "covariance": "cov.npz",
                               "amplitude_law": "none"},
                       "train_dv": "t.npy",
                       "val_dv": "v.npy",
                       "train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat_path)},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1],
                 "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    save_emulator(path_root=str(root),
                  model=model,
                  param_geometry=pgeom,
                  geometry=geom,
                  config=config,
                  histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=cmb_head_recipe(n_ell),
                  facts_yaml=fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label="cmb-identity/correction-head",
                      family="cmb",
                      support=CMB_SUPPORT),
                  attrs={"rescale": "none", "spectrum": "tt"})
    g = np.random.default_rng(150)
    theta = np.array([[2.1e-9, 0.055, 0.31]]) * (1.0 + 0.01
                                                 * g.standard_normal((1, 3)))
    x1 = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        ref = chi2fn.decode(model(pgeom.encode(x1)))[0]
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    report("head save -> rebuild -> predict bitwise (rebuild attach)",
           np.array_equal(got, ref.cpu().numpy()),
           "max|d| = %.2e" % np.abs(got - ref.cpu().numpy()).max())


def check_finetune(tmp, device):
    """Warm-start parity from a CMB source + the wrong-kind guard."""
    root = os.path.join(tmp, "emul_ft_src")
    pgeom, geom, model, chi2fn = save_synthetic_cmb(
        root, device, tmp, label="cmb-identity/finetune-source",
        spectrum="tt", law="as_exp2tau_ref", seed=120)
    source = warmstart.load_source(root=root, device=device)
    report("load_source accepts a CMB artifact",
           type(source.geom).__name__ == "CmbDiagonalGeometry",
           "geom %s" % type(source.geom).__name__)
    # the input geometry extended over the SAME covmat -> no extra names,
    # and build_warm_start's own parity check must pass (it raises when
    # epoch 0 does not reproduce the source function).
    covmat_path = os.path.join(tmp, "params_120.covmat")
    g = np.random.default_rng(130)
    C = np.column_stack([g.normal(2.1e-9, 1e-10, 64),
                         g.normal(0.055, 0.005, 64),
                         g.normal(0.31, 0.01, 64)]).astype("float32")
    train_set = {"C": C,
                 "idx": np.arange(64),
                 "C_mean": C.mean(axis=0)}
    new_pgeom, extra = warmstart.extend_input_geometry(
        source=source, covmat_path=covmat_path,
        train_mean=train_set["C_mean"], device=device)
    model_opts = warmstart.recipe_to_model_opts(source.recipe)
    try:
        init_state, verdict, padded = warmstart.build_warm_start(
            source=source, new_pgeom=new_pgeom, pinned_geom=source.geom,
            model_opts=model_opts, train_set=train_set,
            extra_names=extra, device=device)
        report("warm start reproduces the CMB source at epoch 0",
               init_state is not None, verdict.strip()[:60])
    except ValueError as e:
        report("warm start reproduces the CMB source at epoch 0",
               False, "parity raised: " + str(e)[:80])
    # wrong-kind: the cosmolike pin refuses a CMB source loudly.
    try:
        warmstart.pin_output_geometry(source=source,
                                      run_data={"cosmolike_data_dir": "d",
                                                "cosmolike_dataset": "s"},
                                      run_probe="cs", new_dv_width=10)
        report("cosmolike pin refuses a CMB source", False, "no raise")
    except ValueError as e:
        report("cosmolike pin refuses a CMB source",
               "CmbDiagonalGeometry" in str(e), "ValueError names the kind")
    # validate_cmb accepts a finetune block (the interim error died).
    cfg = {"data": {"cmb": {"spectrum": "tt",
                            "covariance": "c.npz",
                            "amplitude_law": "none"},
                    "train_dv": "a",
                    "val_dv": "b",
                    "train_params": "c",
                    "val_params": "d",
                    "train_covmat": "e"},
           "pce": None,
           "transfer": None}
    try:
        validate_cmb(cfg, train_args={"finetune": {"from": "x"}},
                     rescale="none")
        report("validate_cmb accepts train_args.finetune", True, "no raise")
    except ValueError as e:
        report("validate_cmb accepts train_args.finetune", False, str(e)[:80])


def check_npce(tmp, device):
    """NPCE on the CMB family (the 2026-07-12 family-wide ruling): the
    residual base + refiner algebra is exact under the diagonal metric,
    the roughness penalty composes on the FULL whitened residual,
    save -> rebuild -> predict composes base + net bitwise, and the
    pce x amplitude-law exclusivity is loud (validate_cmb)."""
    from emulator.designs.pce import PCEEmulator
    from emulator.losses.pce import PCEResidualDiagChi2
    pgeom, covmat_path = make_pgeom(tmp, device, seed=71)
    ell, cl = synth_ell_cl()
    n_ell = int(ell.size)
    g = np.random.default_rng(72)
    C = np.column_stack([g.normal(2.1e-9, 5e-11, 400),
                         g.normal(0.055, 0.003, 400),
                         g.normal(0.31, 0.01, 400)]).astype("float64")
    # C_ell rows that REALLY scale with the sampled amplitude, so the
    # LOO gate keeps a mode and the fitted base is alive (the
    # smoke-gate rule).
    Y = (cl[None, :] * (C[:, 0:1] / 2.1e-9)
         * (1.0 + 0.005 * g.standard_normal((400, n_ell))))
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum="tt", ell=ell, fiducial_cl=cl,
        center=Y.mean(axis=0), units="muK2", law="none")
    X_white = pgeom.encode(
        torch.from_numpy(C.astype("float32")).to(device))
    dv = torch.from_numpy(Y.astype("float32")).to(device)
    pce = PCEEmulator.from_training(device, X_white, geom.encode(dv),
                                    p_max=2, r_max=2, q=0.5, k_max=4,
                                    loo_max=0.9, max_terms=8, silent=True)
    chi2fn = PCEResidualDiagChi2(geom=geom, pce=pce)
    with torch.no_grad():
        base = pce(X_white[:8])
    report("NPCE base is alive (the fit kept a real mode)",
           base.abs().max().item() > 1e-4,
           "max|base| = %.2e" % base.abs().max().item())
    enc = chi2fn.encode(dv[:8], X_white[:8])
    report("NPCE encode: whitened truth minus the base, bitwise",
           torch.equal(enc, geom.encode(dv[:8]) - base), "")
    y = torch.randn(8, n_ell, device=device)
    report("NPCE decode: geom.decode(net + base), bitwise",
           torch.equal(chi2fn.decode(y, X_white[:8]),
                       geom.decode(y + base)), "")
    # Roughness composes: the penalty acts on pred - target, which
    # under the residual construction IS the full whitened residual
    # (base + net - truth); with lam > 0 the loss must move.
    plain = chi2fn.loss(y, enc, X_white[:8]).item()
    chi2fn.configure_roughness(lam=10.0, period_cut=40)
    rough = chi2fn.loss(y, enc, X_white[:8]).item()
    report("NPCE + roughness compose (lam moves the loss)",
           np.isfinite(rough) and rough > plain,
           "plain %.4f -> rough %.4f" % (plain, rough))
    # save -> rebuild -> predict: base + net composed by the predictor
    # (a bare law-none decode here would be silently wrong).
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=n_ell, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    root = os.path.join(tmp, "emul_cmb_npce")
    config = {"data": {"cmb": {"spectrum": "tt",
                               "covariance": "cov.npz",
                               "amplitude_law": "none"},
                       "train_dv": "t.npy",
                       "val_dv": "v.npy",
                       "train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat_path)},
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
                  resolved_train={"nepochs": 1},
                  resolved_model=cmb_recipe(n_ell),
                  pce=pce, pce_form="residual",
                  facts_yaml=fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label="cmb-identity/npce-spectrum",
                      family="cmb",
                      support=CMB_SUPPORT),
                  attrs={"rescale": "none", "spectrum": "tt"})
    theta = np.array([[2.15e-9, 0.056, 0.312]])
    x1 = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        x1e = pgeom.encode(x1)
        ref = geom.decode(model(x1e) + pce(x1e))[0].cpu().numpy()
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    report("NPCE save -> rebuild -> predict composes base + net bitwise",
           np.array_equal(got, ref),
           "max|d| = %.2e" % np.abs(got - ref).max())
    # the pce x amplitude-law exclusivity (one target construction at a
    # time): validate_cmb must be loud.
    cfg = {"data": {"cmb": {"spectrum": "tt",
                            "covariance": "c.npz",
                            "amplitude_law": "as_exp2tau_ref",
                            "as_name": "As",
                            "tau_name": "tau",
                            "as_ref": AS_REF_FIXTURE,
                            "tau_ref": TAU_REF_FIXTURE},
                    "train_dv": "a",
                    "val_dv": "b",
                    "train_params": "c",
                    "val_params": "d",
                    "train_covmat": "e"},
           "pce": {"form": "residual"},
           "transfer": None}
    try:
        validate_cmb(cfg, train_args={}, rescale="none")
        report("pce + amplitude_law exclusivity raises", False, "no raise")
    except ValueError as e:
        report("pce + amplitude_law exclusivity raises",
               "amplitude_law: none" in str(e),
               "ValueError names the fix")


# ---------------------------------------------------------------------------
# The eq-6 covariance known-answer legs (compute_cmb_covariance.py). Pure
# numpy: they drive the non-Gaussian contraction against a truth built
# independently of the pipeline's own algebra, so a normalization error in
# the contraction cannot hide behind symmetry / positive-definiteness checks
# (those are invariant under any positive diagonal reweighting). The
# derivation of the fractional-amplitude weight is in
# ai/notes/families-scalar-cmb.md. No torch, no CAMB.
# ---------------------------------------------------------------------------

def _load_cov_module():
    """Load compute_cmb_covariance.py by path (pure numpy; CAMB is lazy)."""
    repo = Path(__file__).resolve().parents[3]
    path = repo / "compute_data_vectors" / "compute_cmb_covariance.py"
    spec = importlib.util.spec_from_file_location("cmb_cov_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lensing_potential_scale(n):
    """The [L(L+1)]^2/(2 pi) factor CAMB folds into get_lens_potential_cls.

    Multiplying a raw lensing-potential spectrum C^phiphi_L by this gives
    the scaled array scaled_L = [L(L+1)]^2 C^phiphi_L / (2 pi) CAMB
    returns; dividing the scaled array by it recovers the raw spectrum.
    Zero at L < 2 (the monopole and dipole carry no lensing, and the
    scaled array is defined zero there).

    Arguments:
      n = array length (L = 0..n-1).

    Returns:
      (n,) the per-L scale factor, 0 at L = 0 and 1.
    """
    big_l = np.arange(n, dtype="float64")
    scale = (big_l * (big_l + 1.0)) ** 2 / (2.0 * np.pi)
    scale[:2] = 0.0
    return scale


class FakeCAMBData:
    """A CAMB results stand-in whose re-lensing is an exact affine map.

    get_lensed_cls_with_spectrum returns base_s + M_s @ C^raw, where the
    incoming argument is CAMB's SCALED lensing array and the fake converts
    it back to the raw C^phiphi internally, so dC_l/dC^raw_L = M_s[l, L]
    exactly and the 5-point stencil the covariance script runs is exact to
    round-off (an affine map has no truncation error). That lets the gate
    compare the script's eq-6 contraction against a truth built straight
    from M and the raw lensing-potential variance, independent of the
    script's own contraction algebra.

    The fixture keeps the two conventions genuinely distinct: the raw
    C^phiphi_L versus the scaled [L(L+1)]^2 C^phiphi_L / (2 pi) that CAMB
    returns. The covariance script reads the raw spectrum for its
    contraction weight (cls["pp"]) but the scaled spectrum for the
    perturbation array (get_lens_potential_cls), and the two differ by an
    L-dependent factor, so a bug that used one convention where the other
    belongs would change the result. See ai/notes/families-scalar-cmb.md.

    Arguments (constructor):
      clpp_raw = (lens_lmax+1,) the raw fiducial C^phiphi_L (0 at L < 2).
      M        = dict tt/te/ee of (lmax+1, lens_lmax+1) RAW sensitivities
                 (dC_l/dC^raw_L).
      base     = dict tt/te/ee of (lmax+1,) unperturbed lensed spectra.
    """

    def __init__(self, clpp_raw, M, base):
        self._clpp_raw = np.asarray(clpp_raw, dtype="float64")
        self._M = M
        self._base = base
        self._scale = _lensing_potential_scale(self._clpp_raw.shape[0])

    def get_lens_potential_cls(self, raw_cl=False):
        """The SCALED lensing array scaled_L = [L(L+1)]^2 C^phiphi_L/(2 pi).

        The covariance script reads column 0 with raw_cl=False, the
        convention CAMB returns. A raw_cl=True call raises: the script
        never makes one, and the fixture must not silently answer in the
        wrong convention.

        Arguments:
          raw_cl = must be False (the scaled convention the script reads);
                   True raises loudly.

        Returns:
          (n, 4) array, column 0 the scaled lensing spectrum, 0 at L < 2.
        """
        if raw_cl:
            raise ValueError(
                "FakeCAMBData.get_lens_potential_cls: raw_cl=True is not "
                "supported; the covariance script reads the scaled "
                "[L(L+1)]^2 C/(2 pi) array (raw_cl=False), so the fixture "
                "answers only in that convention")
        arr = np.zeros((self._clpp_raw.shape[0], 4), dtype="float64")
        arr[:, 0] = self._scale * self._clpp_raw
        return arr

    def get_lensed_cls_with_spectrum(self, clpp, lmax, CMB_unit="muK",
                                     raw_cl=True):
        """Affine re-lensing: convert the scaled argument to raw, then map.

        clpp arrives in CAMB's scaled convention (the script perturbs the
        array get_lens_potential_cls returned). The fake converts it back
        to the raw C^phiphi (raw_L = scaled_L / ([L(L+1)]^2/(2 pi)),
        0 at L < 2) and returns base_s + M_raw_s @ raw, so the derivative
        with respect to the RAW spectrum is exactly M_raw. Columns follow
        CAMB order (0 = TT, 1 = EE, 2 = BB, 3 = TE), the columns
        lensed_cls_with_clpp reads.

        Arguments:
          clpp     = (lens_lmax+1,) the scaled lensing array (perturbed).
          lmax     = top multipole of the returned spectra.
          CMB_unit = accepted for signature parity (unused).
          raw_cl   = accepted for signature parity (unused).

        Returns:
          (lmax+1, 4) the affine-re-lensed spectra.
        """
        scaled = np.asarray(clpp, dtype="float64")
        raw_vec = np.zeros_like(scaled)
        # invert the convention factor; L < 2 has scale 0, so it stays 0.
        raw_vec[2:] = scaled[2:] / self._scale[2:]
        n = int(lmax) + 1
        out = np.zeros((n, 4), dtype="float64")
        out[:, 0] = self._base["tt"][:n] + self._M["tt"][:n] @ raw_vec
        out[:, 1] = self._base["ee"][:n] + self._M["ee"][:n] @ raw_vec
        out[:, 3] = self._base["te"][:n] + self._M["te"][:n] @ raw_vec
        return out


def _oracle_truth(M, clpp_raw, fsky, ell, lens_lmax):
    """Eq 6 straight from M and the RAW lensing-potential variance.

    N^{ab}_{ll'} = sum_{L=2}^{lens_lmax} M_a[l, L] Var_L M_b[l', L],
    Var_L = 2 (C^raw_L)^2 / ((2L+1) fsky), the Gaussian variance of the
    raw lensing potential. M is dC_l/dC^raw_L (the fake maps in raw
    coordinates), so this contracts raw derivatives with the raw variance
    exactly as eq 6 reads, never through the pipeline's own contraction.
    The l index runs over the covariance grid `ell`, the rows sliced from M.

    Arguments:
      M         = dict tt/te/ee of (lmax+1, lens_lmax+1) raw sensitivities.
      clpp_raw  = (lens_lmax+1,) raw C^phiphi_L.
      fsky      = sky fraction (rescales the variance).
      ell       = (n_ell,) covariance multipole grid (l = 2..lmax).
      lens_lmax = top L of the phi sum.

    Returns:
      dict of (n_ell, n_ell) arrays keyed like assemble_lensing_blocks
      (cov_tt/cov_te/cov_ee and cov_tt_te/cov_tt_ee/cov_te_ee).
    """
    cl = np.asarray(clpp_raw, dtype="float64")
    big_l = np.arange(0, lens_lmax + 1, dtype="float64")
    var_l = np.zeros(lens_lmax + 1, dtype="float64")
    var_l[2:] = 2.0 * cl[2:lens_lmax + 1] ** 2 / ((2.0 * big_l[2:] + 1.0)
                                                  * fsky)
    lo = int(ell[0])
    hi = int(ell[-1]) + 1
    pairs = (("tt", "tt"), ("te", "te"), ("ee", "ee"),
             ("tt", "te"), ("tt", "ee"), ("te", "ee"))
    out = {}
    for a, b in pairs:
        m_a = M[a][lo:hi, :lens_lmax + 1]         # (n_ell, lens_lmax+1)
        m_b = M[b][lo:hi, :lens_lmax + 1]
        if a == b:
            key = "cov_" + a
        else:
            key = "cov_" + a + "_" + b
        out[key] = (m_a * var_l[None, :]) @ m_b.T
    return out


def _band_constant_M(rng, n_rows, lens_lmax, band_width):
    """A sensitivity matrix constant across each L-band.

    Columns L within one width-`band_width` band share a per-row value,
    so dC_l/dC^phiphi_L is exactly constant across the band and the
    smooth-response band projection is exact. Columns L < 2 stay zero
    (the bands start at L = 2, so they never contribute).

    Arguments:
      rng        = numpy Generator.
      n_rows     = lmax+1 (the covariance row count).
      lens_lmax  = top L of the phi sum.
      band_width = multipoles per band.

    Returns:
      (n_rows, lens_lmax+1) sensitivity matrix, constant within each band.
    """
    M = np.zeros((n_rows, lens_lmax + 1), dtype="float64")
    start = 2
    while start <= lens_lmax:
        stop = min(lens_lmax, start + band_width - 1)
        col = rng.standard_normal((n_rows, 1))
        M[:, start:stop + 1] = col              # same value across the band
        start = stop + 1
    return M


def _worst_rel(blocks, truth):
    """Max relative block difference over all six spectrum pairs."""
    worst = 0.0
    for key in truth:
        denom = np.abs(truth[key]).max() + 1e-300
        rel = np.abs(blocks[key] - truth[key]).max() / denom
        if rel > worst:
            worst = rel
    return worst


def check_covariance_oracle():
    """The eq-6 known-answer legs: truth, discrimination, fixture, band + zero.

    The fake keeps the raw C^phiphi and CAMB's scaled
    [L(L+1)]^2 C^phiphi/(2 pi) genuinely distinct, so the pipeline reads
    the raw spectrum for its weight and the scaled one for the
    perturbation (the real convention boundary), and the truth is built
    from the raw sensitivity and the raw variance throughout.
    """
    cov = _load_cov_module()
    rng = np.random.default_rng(2024)

    # ---- legs (a) + (b): band width 1, a general (per-L) sensitivity ----
    lens_lmax = 12
    lmax = 12
    ell = np.arange(2, lmax + 1, dtype="int64")     # l = 2..12, n_ell = 11
    fsky = 0.4
    # a positive RAW lensing-potential array, physically small so the old
    # extra-C_L^2 weights land many orders of magnitude below the truth.
    clpp_raw = np.zeros(lens_lmax + 1, dtype="float64")
    clpp_raw[2:] = 1e-8 * (1.0 + 0.5 * rng.random(lens_lmax - 1))
    M = {}
    base = {}
    for s in ("tt", "te", "ee"):
        M[s] = rng.standard_normal((lmax + 1, lens_lmax + 1))
        # a ZERO baseline: the derivative signal is O(M C^phiphi) ~ 1e-8,
        # so a large unperturbed spectrum would swamp it and the stencil
        # would lose it to cancellation (~1e-7). With no baseline the
        # stencil of the affine map is exact to round-off, isolating the
        # contraction weight these legs test.
        base[s] = np.zeros(lmax + 1, dtype="float64")
    fake = FakeCAMBData(clpp_raw=clpp_raw, M=M, base=base)
    # the pipeline reads the RAW spectrum for its weight (cls["pp"]) and
    # the SCALED spectrum for the perturbation (from the fake's getter).
    cls = {"pp": clpp_raw.copy(),
           "tt": np.zeros(lmax + 1, dtype="float64"),
           "te": np.zeros(lmax + 1, dtype="float64"),
           "ee": np.zeros(lmax + 1, dtype="float64")}
    ng1 = {"enabled": True,
           "lens_lmax": lens_lmax,
           "band_width": 1,
           "step_fracs": [0.01, 0.02],
           "converge_rtol": 0.05}
    blocks1, study1 = cov.nongaussian_blocks(cambdata=fake, cls=cls, ell=ell,
                                             ng_cfg=ng1, fsky=fsky,
                                             log=lambda *a: None)
    truth1 = _oracle_truth(M=M, clpp_raw=clpp_raw, fsky=fsky, ell=ell,
                           lens_lmax=lens_lmax)
    worst1 = _worst_rel(blocks1, truth1)
    report("eq-6 truth: real contraction matches eq 6 from M and Var(C_L)",
           worst1 < 1e-9, "max rel over six blocks %.2e" % worst1)

    # (b) discrimination: the OLD weights (band-summed C^phiphi variance,
    # the extra C_L^2) on the same derivatives must miss the truth by
    # orders of magnitude. deriv is the affine map's exact analytic
    # derivative in RAW coordinates, dCl/dA_b = sum_{L in b} M[l, L] C^raw_L.
    bands = cov.band_windows(lmin=2, lmax=lens_lmax, band_width=1)
    big_l = np.arange(0, lens_lmax + 1, dtype="float64")
    var_pp = np.zeros(lens_lmax + 1, dtype="float64")
    var_pp[2:] = 2.0 * clpp_raw[2:] ** 2 / ((2.0 * big_l[2:] + 1.0) * fsky)
    lo = int(ell[0])
    hi = int(ell[-1]) + 1
    n_ell = len(ell)
    deriv = {}
    for s in ("tt", "te", "ee"):
        deriv[s] = np.zeros((len(bands), n_ell), dtype="float64")
    s_old = np.zeros(len(bands), dtype="float64")
    for bi, (b_lo, b_hi) in enumerate(bands):
        s_old[bi] = var_pp[b_lo:b_hi + 1].sum()
        band_cl = clpp_raw[b_lo:b_hi + 1]
        for s in ("tt", "te", "ee"):
            deriv[s][bi] = (M[s][lo:hi, b_lo:b_hi + 1] * band_cl).sum(axis=1)
    old_blocks = cov.assemble_lensing_blocks(deriv=deriv, w=s_old)
    truth_max = max(np.abs(truth1[k]).max() for k in truth1)
    old_max = max(np.abs(old_blocks[k]).max() for k in old_blocks)
    orders = np.log10(truth_max / (old_max + 1e-300))
    report("eq-6 discrimination: the old extra-C_L^2 weights miss by orders",
           orders > 6.0, "truth / old magnitude ~ 1e%.0f" % orders)

    # fixture integrity: the scaled array the pipeline perturbs and the raw
    # array it weights with must genuinely differ for every L >= 2, and by
    # an L-DEPENDENT factor (a constant ratio would leave the wide-band
    # weight invariant and weaken the leg the same way raw == scaled did).
    scaled = fake.get_lens_potential_cls(raw_cl=False)[:, 0]
    differ = bool((scaled[2:] != clpp_raw[2:]).all())
    ratio = scaled[2:] / clpp_raw[2:]
    l_dependent = float(ratio.max()) > 1.5 * float(ratio.min())
    raw_raises = False
    try:
        fake.get_lens_potential_cls(raw_cl=True)
    except ValueError:
        raw_raises = True
    report("fixture integrity: scaled != raw for L>=2, ratio L-dependent, "
           "raw_cl=True raises",
           differ and l_dependent and raw_raises,
           "ratio %.2f..%.2f, raw_cl guard %s"
           % (float(ratio.min()), float(ratio.max()), raw_raises))

    # ---- leg (c): band width 3, a sensitivity CONSTANT across each band ----
    # the smooth-response projection is then exact, so the width-3
    # contraction must reproduce the per-L eq 6 truth. The last band's raw
    # spectrum is zeroed: its weight must be exactly 0 (the no-divide
    # guard) and the truth still matches (a zero band contributes nothing
    # on both sides).
    lens_lmax_c = 13
    lmax_c = 13
    ell_c = np.arange(2, lmax_c + 1, dtype="int64")   # l = 2..13
    band_width_c = 3
    clpp_raw_c = np.zeros(lens_lmax_c + 1, dtype="float64")
    clpp_raw_c[2:] = 1e-8 * (1.0 + 0.5 * rng.random(lens_lmax_c - 1))
    bands_c = cov.band_windows(lmin=2, lmax=lens_lmax_c,
                               band_width=band_width_c)
    last_lo, last_hi = bands_c[-1]
    clpp_raw_c[last_lo:last_hi + 1] = 0.0            # zero the last band
    M_c = {}
    base_c = {}
    for s in ("tt", "te", "ee"):
        M_c[s] = _band_constant_M(rng, lmax_c + 1, lens_lmax_c, band_width_c)
        # zero baseline (see the width-1 leg): the stencil stays exact.
        base_c[s] = np.zeros(lmax_c + 1, dtype="float64")
    fake_c = FakeCAMBData(clpp_raw=clpp_raw_c, M=M_c, base=base_c)
    cls_c = {"pp": clpp_raw_c.copy(),
             "tt": np.zeros(lmax_c + 1, dtype="float64"),
             "te": np.zeros(lmax_c + 1, dtype="float64"),
             "ee": np.zeros(lmax_c + 1, dtype="float64")}
    # the zeroed last band has an identically-zero derivative, but the
    # stencil formula f_m2 - 8 f_m1 + 8 f_p1 - f_p2 on the four
    # bit-identical re-lensings (its perturbation 0*(1+eps) is 0 for every
    # step) leaves a ~1e-22 rounding residue; that residue is the same at
    # both steps and the derivative scales as 1/h, so its relative spread
    # is exactly 1 - h_min/h_next = 0.5. These legs check the contraction
    # weight, not stencil convergence (the smoke gate covers that on real
    # CAMB), so it uses a loose convergence tolerance here; the real bands
    # still converge to ~3e-14 and the truth comparison below is what
    # validates the numbers.
    ng3 = {"enabled": True,
           "lens_lmax": lens_lmax_c,
           "band_width": band_width_c,
           "step_fracs": [0.01, 0.02],
           "converge_rtol": 1.0}
    blocks3, study3 = cov.nongaussian_blocks(cambdata=fake_c, cls=cls_c,
                                             ell=ell_c, ng_cfg=ng3, fsky=fsky,
                                             log=lambda *a: None)
    truth3 = _oracle_truth(M=M_c, clpp_raw=clpp_raw_c, fsky=fsky, ell=ell_c,
                           lens_lmax=lens_lmax_c)
    worst3 = _worst_rel(blocks3, truth3)
    last_weight = study3["per_band_weight"][-1]
    report("eq-6 band projection: width-3 constant-response matches truth",
           worst3 < 1e-9, "max rel %.2e (policy %s)"
           % (worst3, study3["band_weight_policy"]))
    report("eq-6 zero band: the zeroed last band's weight is exactly 0",
           last_weight == 0.0, "per_band_weight[-1] = %r" % last_weight)


def check_cmb_ref_schema():
    """validate_cmb admits only a finite non-boolean real as_ref / tau_ref.

    The fiducial reference values feed the amplitude factor
    (As_ref / A_s) exp(2(tau - tau_ref)); a boolean or a numeric string that a
    float() coercion would silently accept becomes a wrong-magnitude fiducial
    and a scaled target. validate_cmb now validates them by type, the same
    finite non-boolean real predicate the optimizer schema and the other public
    scientific controls use. Control: a valid reference config passes. Red
    legs: as_ref / tau_ref as True, False, a numeric string, NaN, infinity, a
    zero or negative as_ref (the divide-by boundary) each raise.
    """
    def cfg():
        return {"data": {"cmb": {"spectrum": "tt", "covariance": "cov.npz",
                                 "amplitude_law": "as_exp2tau_ref",
                                 "as_name": "As", "tau_name": "tau",
                                 "as_ref": AS_REF_FIXTURE,
                                 "tau_ref": TAU_REF_FIXTURE},
                         "train_dv": "t.npy", "val_dv": "v.npy",
                         "train_params": "t.txt", "val_params": "v.txt",
                         "train_covmat": "cm.npz"},
                "train_args": {"nepochs": 1}}

    try:
        validate_cmb(cfg(), {"nepochs": 1})
        report("CMB ref schema: a valid finite as_ref / tau_ref passes",
               True, "accepted")
    except Exception as exc:
        report("CMB ref schema: a valid finite as_ref / tau_ref passes",
               False, repr(exc))

    cases = [("as_ref", True, "boolean as_ref (True)"),
             ("as_ref", False, "boolean as_ref (False)"),
             ("as_ref", "2.1e-9", "string as_ref"),
             ("as_ref", float("nan"), "NaN as_ref"),
             ("as_ref", float("inf"), "infinite as_ref"),
             ("as_ref", 0.0, "zero as_ref (divide-by boundary)"),
             ("as_ref", -1.0, "negative as_ref"),
             ("tau_ref", True, "boolean tau_ref (True)"),
             ("tau_ref", "0.05", "string tau_ref"),
             ("tau_ref", float("nan"), "NaN tau_ref")]
    for key, value, label in cases:
        config = cfg()
        config["data"]["cmb"][key] = value
        refused = False
        try:
            validate_cmb(config, {"nepochs": 1})
        except ValueError:
            refused = True
        report("CMB ref schema: " + label + " is refused", refused,
               "ValueError")


def main():
    """Run the cmb-identity checks in a tempdir; exit non-zero on failure."""
    print("cmb-identity: geometry + law + round-trip + roughness "
          "+ finetune + adapter + covariance known-answer legs")
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        # Each drafted leg emits ONE ##AID line at its aggregation point (a
        # contiguous group of check_* functions). n0 = the FAILURES count just
        # before the leg, so emit_aid reads the leg's own verdict, not the
        # board-wide one; the seven aids match the note's drafted anchor block.
        n0 = len(FAILURES)
        check_ruled_constants(device)
        check_state_roundtrip(device)
        check_cmb_ref_schema()
        emit_aid("cmb-identity.geometry-and-reference-schema", n0)

        n0 = len(FAILURES)
        check_law(tmp, device)
        emit_aid("cmb-identity.amplitude-law-and-score", n0)

        n0 = len(FAILURES)
        check_roundtrip(tmp, device, law="none")
        check_roundtrip(tmp, device, law="as_exp2tau_ref")
        # the domain arms ride this leg: they are the same saved artifact,
        # rebuilt and asked a question, and the question is the one predict()
        # refuses. They emit no aid of their own.
        check_domain_law(tmp, device)
        check_adapter(tmp, device)
        emit_aid("cmb-identity.artifact-and-adapter-round-trip", n0)

        n0 = len(FAILURES)
        check_roughness(device)
        emit_aid("cmb-identity.roughness-contract", n0)

        n0 = len(FAILURES)
        check_head(tmp, device)
        check_npce(tmp, device)
        emit_aid("cmb-identity.model-variant-composition", n0)

        n0 = len(FAILURES)
        check_finetune(tmp, device)
        emit_aid("cmb-identity.finetune-parity", n0)

        n0 = len(FAILURES)
        check_covariance_oracle()
        emit_aid("cmb-identity.covariance-known-answer", n0)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: cmb-identity all checks green")


if __name__ == "__main__":
    main()
