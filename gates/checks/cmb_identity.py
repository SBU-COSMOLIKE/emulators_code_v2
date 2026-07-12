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
  - the as_exp2tau law is exact: _factor equals exp(2 tau)/A_s computed the
    same way (bitwise) and encode(decode(x)) returns x to float32 tolerance;
    the registry / configure_law loud errors fire;
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
    accepts a finetune block (the interim error is gone).

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
from emulator import warmstart

FAILURES = []

IN_NAMES = ["As", "tau", "omegam"]
N_IN = len(IN_NAMES)


def report(label, ok, detail):
    """Print one PASS/FAIL line and remember any failure."""
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


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


def save_synthetic_cmb(root, device, tmp, spectrum="tt", law="as_exp2tau",
                       n_ell=200, seed=0):
    """Build, then save, a tiny synthetic CMB emulator under `root`.

    Returns:
      (pgeom, geom, model, chi2fn): the sources + the pre-save model and
      the law-dispatched chi2 (its decode is the round-trip reference).
    """
    pgeom, covmat_path = make_pgeom(tmp, device, seed=seed)
    ell, cl = synth_ell_cl(n_ell=n_ell)
    g = np.random.default_rng(seed + 3)
    center = cl * (1.0 + 0.01 * g.standard_normal(n_ell))
    units = "dimensionless" if spectrum == "pp" else "muK2"
    as_name = "As" if law == "as_exp2tau" else ""
    tau_name = "tau" if law == "as_exp2tau" else ""
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum=spectrum, ell=ell, fiducial_cl=cl,
        center=center, units=units, law=law,
        as_name=as_name, tau_name=tau_name)
    if law == "none":
        chi2fn = make_cmb_chi2(geom=geom, law=law)
    else:
        chi2fn = make_cmb_chi2(geom=geom, law=law, param_geometry=pgeom,
                               as_name=as_name, tau_name=tau_name)
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
    if law == "as_exp2tau":
        config["data"]["cmb"]["as_name"] = "As"
        config["data"]["cmb"]["tau_name"] = "tau"
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
                  resolved_model=cmb_recipe(n_ell),
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
        center=cl * 1.01, units="muK2", law="as_exp2tau",
        as_name="As", tau_name="tau")
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
           ok, "%d keys incl. law strings" % len(st0))


def check_law(tmp, device):
    """The as_exp2tau law: _factor exact, encode/decode inverse, loud errors."""
    pgeom, _ = make_pgeom(tmp, device, seed=50)
    ell, cl = synth_ell_cl()
    geom = CmbDiagonalGeometry.from_fiducial(
        device=device, spectrum="tt", ell=ell, fiducial_cl=cl,
        center=cl, units="muK2", law="as_exp2tau",
        as_name="As", tau_name="tau")
    chi2fn = make_cmb_chi2(geom=geom, law="as_exp2tau",
                           param_geometry=pgeom,
                           as_name="As", tau_name="tau")
    g = np.random.default_rng(60)
    theta = np.column_stack([g.normal(2.1e-9, 1e-10, 8),
                             g.normal(0.055, 0.005, 8),
                             g.normal(0.31, 0.01, 8)])
    x = torch.as_tensor(theta, dtype=torch.float32, device=device)
    x_enc = pgeom.encode(x)
    f = chi2fn._factor(x_enc)
    phys = pgeom.decode(x_enc)
    want = (torch.exp(2.0 * phys[:, 1]) / phys[:, 0]).reshape(-1, 1)
    report("_factor equals exp(2 tau)/A_s bitwise (same decoded params)",
           torch.equal(f, want), "max|d| = %.2e"
           % (f - want).abs().max().item())
    # encode(decode(pred)) returns pred to float32 round-off (the factor
    # multiplies and divides, so bitwise is not guaranteed; the bar is a
    # tight relative tolerance).
    pred = torch.randn(8, len(ell), device=device)
    back = chi2fn.encode(chi2fn.decode(pred, x_enc), x_enc)
    rel = ((back - pred).abs().max()
           / pred.abs().max()).item()
    report("encode(decode(x)) round-trips to float32 round-off",
           rel < 1e-4, "max rel %.2e" % rel)
    try:
        make_cmb_chi2(geom=geom, law="not_a_law")
        report("unknown law raises", False, "did not raise")
    except ValueError:
        report("unknown law raises", True, "ValueError")
    try:
        make_cmb_chi2(geom=geom, law="as_exp2tau")
        report("as_exp2tau without its columns raises", False, "no raise")
    except ValueError:
        report("as_exp2tau without its columns raises", True, "ValueError")
    try:
        make_cmb_chi2(geom=geom, law="as_exp2tau", param_geometry=pgeom,
                      as_name="NOPE", tau_name="tau")
        report("configure_law bad column raises", False, "no raise")
    except ValueError:
        report("configure_law bad column raises", True, "ValueError")


def check_roundtrip(tmp, device, law):
    """save -> rebuild -> predict bitwise vs the pre-save decode."""
    root = os.path.join(tmp, "emul_cmb_" + law)
    pgeom, geom, model, chi2fn = save_synthetic_cmb(
        root, device, tmp, spectrum="tt", law=law, seed=70)
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

    root = Path(__file__).resolve().parents[2]
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
    save_synthetic_cmb(root_tt, device, tmp, spectrum="tt",
                       law="as_exp2tau", n_ell=200, seed=90)
    root_ee = os.path.join(tmp, "ad_ee")
    save_synthetic_cmb(root_ee, device, tmp, spectrum="ee",
                       law="none", n_ell=100, seed=100)

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
    except ValueError:
        report("unknown-spectrum must_provide raises", True, "ValueError")
    try:
        t.must_provide(Cl={"ee": 150})
        report("beyond-lmax must_provide raises", False, "no raise")
    except ValueError:
        report("beyond-lmax must_provide raises", True, "ValueError")
    # get_Cl convention guards.
    t.current_state = {"Cl": cl}
    try:
        t.get_Cl(ell_factor=True)
        report("get_Cl ell_factor guard", False, "no raise")
    except ValueError:
        report("get_Cl ell_factor guard", True, "ValueError")
    try:
        t.get_Cl(units="FIRASmuK2")
        report("get_Cl units guard", False, "no raise")
    except ValueError:
        report("get_Cl units guard", True, "ValueError")
    # duplicate spectrum.
    root_tt2 = os.path.join(tmp, "ad_tt2")
    save_synthetic_cmb(root_tt2, device, tmp, spectrum="tt",
                       law="none", n_ell=50, seed=110)
    try:
        _build_adapter(cls, [root_tt, root_tt2])
        report("duplicate spectrum raises", False, "no raise")
    except ValueError:
        report("duplicate spectrum raises", True, "ValueError")


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
        root, device, tmp, spectrum="tt", law="as_exp2tau", seed=120)
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
                            "amplitude_law": "as_exp2tau",
                            "as_name": "As",
                            "tau_name": "tau"},
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


def main():
    """Run the cmb-identity checks in a tempdir; exit non-zero on failure."""
    print("cmb-identity: geometry + law + round-trip + roughness "
          "+ finetune + adapter legs")
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        check_ruled_constants(device)
        check_state_roundtrip(device)
        check_law(tmp, device)
        check_roundtrip(tmp, device, law="none")
        check_roundtrip(tmp, device, law="as_exp2tau")
        check_head(tmp, device)
        check_npce(tmp, device)
        check_adapter(tmp, device)
        check_roughness(device)
        check_finetune(tmp, device)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: cmb-identity all checks green")


if __name__ == "__main__":
    main()
