#!/usr/bin/env python3
"""mps-identity gate (MPS-A): the grid2d-emulator save/rebuild/predict
identity, the staging law transform, the emul_mps assembly math (base
stubbed: closed-form stub bases pin the assembly EXACTLY, independent
of the vendored syren/ formulas), the config loud errors, and the
D-MP7 finetune parity — torch + scipy, no CAMB.

Legs:
  - Grid2DGeometry: standardize round-trip to float32 round-off; state
    round-trip byte-identical (seven keys incl. quantity/units/law);
    the width / un-standardizable / unknown-law guards (errors name
    (z, k) points); the D-MP9 pinning (a constant law-space column
    that is not the whole surface is physics under ANY law — the
    boost's low-k B = 1 region: scale 1, decode returns the training
    constant, const_mask persists; a wholly constant surface still
    raises, that is the dead-dump signature);
  - the STAGING law transform through the REAL load_source +
    _grid2d_law_rows: law rows == log(raw / base) with the base dump
    aligned by dump_rows through a real shuffled staging; the k_stride
    thinning keeps the top edge; the positivity and width guards;
  - save -> rebuild -> predict bitwise on the syren_linear and none
    laws (the predictor's grid2d branch returns {"z", "k", quantity}
    reshaped (nz, nk)); rebuild info flags class-guarded;
  - the D-CM13 head leg: attach_head_coords (one bin per z slice),
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
    the grid nodes; a non-positive spectrum rejects the point (False);
  - validate_grid2d legs (law-quantity pairing, base files both ways,
    k_stride, transfer PERMANENT wording);
  - D-MP7 finetune: epoch-0 parity from a grid2d source; the
    wrong-kind and metadata-mismatch from_config errors.
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
from emulator.geometries.grid2d import Grid2DGeometry, TARGET_LAWS_2D
from emulator.geometries.parameter import ParamGeometry
from emulator.data_staging import load_source
from emulator.experiment import EmulatorExperiment, validate_grid2d
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor
from emulator import warmstart

FAILURES = []
IN_NAMES = ["As", "H0", "omch2"]
N_IN = len(IN_NAMES)
Z4 = np.array([0.0, 0.5, 1.0, 2.0])
K6 = np.logspace(-4, 0.0, 6)


def report(label, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


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
    try:
        Yc = Y.copy()
        Yc[:, 3] = 7.0
        Grid2DGeometry.from_targets(device=device, targets=Yc, z=Z4,
                                    k=K6, quantity="pklin",
                                    units="Mpc3", law="none")
        report("un-standardizable guard raises", False, "no raise")
    except ValueError as e:
        report("un-standardizable guard raises", "(z, k)" in str(e),
               "names (z, k) points")
    try:
        Grid2DGeometry.from_targets(device=device, targets=Y, z=Z4,
                                    k=K6, quantity="pklin",
                                    units="Mpc3", law="nope")
        report("unknown law raises", False, "no raise")
    except ValueError:
        report("unknown law raises", True, "ValueError")
    # D-MP9 (amended law-agnostic after the gate's law-none boost
    # training hit it): a constant law-space column that is not the
    # whole surface is PHYSICS (boost = 1 below the nonlinear scale
    # for every cosmology, under ANY law) — pinned (scale 1, decode
    # returns the training constant, mask persisted), never rejected;
    # a WHOLLY constant surface still dies loudly for every law.
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
        report("D-MP9: constant column pinned + round-trip (%s)" % law_c,
               ok, "1 pin, scale 1.0, decode = the constant, state rides")
    try:
        Yall = np.tile(Y[:1], (Y.shape[0], 1))
        Grid2DGeometry.from_targets(device=device, targets=Yall, z=Z4,
                                    k=K6, quantity="boost",
                                    units="dimensionless",
                                    law="none")
        report("D-MP9: wholly constant surface still raises", False,
               "no raise")
    except ValueError as e:
        report("D-MP9: wholly constant surface still raises",
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
    np.savetxt(os.path.join(tmp, "st_params.1.txt"), cols)
    gen = torch.Generator().manual_seed(3)
    src = load_source(dv_path=os.path.join(tmp, "st_dv.npy"),
                      params_path=os.path.join(tmp, "st_params.1.txt"),
                      names=IN_NAMES, omegabh2_hi=None, n_keep=40,
                      gen=gen, ram_frac=0.7, with_means=True,
                      verbose=False)
    exp = EmulatorExperiment.__new__(EmulatorExperiment)
    exp.grid2d = {"quantity": "pklin", "units": "Mpc3",
                  "law": "syren_linear",
                  "z_file": os.path.join(tmp, "st_z.npy"),
                  "k_file": os.path.join(tmp, "st_k.npy"),
                  "k_stride": 2}
    exp._grid2d_z = None
    exp._grid2d_k = None
    dump_rows = np.array(src["dump_rows"])
    exp._grid2d_law_rows(src=src, base_path=os.path.join(tmp,
                                                         "st_base.npy"))
    kept_k = np.unique(np.concatenate([np.arange(0, nk, 2),
                                       np.array([nk - 1])]))
    cols_idx = (np.arange(nz)[:, None] * nk
                + kept_k[None, :]).reshape(-1)
    want = np.log(raw[dump_rows].astype("float64")
                  / base[dump_rows].astype("float64"))[:, cols_idx]
    ok = (np.allclose(src["dv"], want.astype("float32"), rtol=0, atol=0)
          and np.array_equal(exp._grid2d_k, K6[kept_k])
          and src["dv"].shape == (40, nz * kept_k.size)
          and np.array_equal(src["idx"], np.arange(40))
          and (nk - 1) in kept_k)
    report("staging law transform: base-aligned + strided + top edge",
           ok, "shape %s" % (src["dv"].shape,))
    ok2 = np.allclose(src["dv_mean"],
                      want.mean(axis=0).astype("float32"))
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
                       verbose=False)
    try:
        exp._grid2d_law_rows(src=src2,
                             base_path=os.path.join(tmp, "st_base.npy"))
        report("staging positivity guard raises", False, "no raise")
    except ValueError:
        report("staging positivity guard raises", True, "ValueError")


def grid2d_recipe(width):
    return {"cls": "emulator.designs.plain.ResMLP", "name": "resmlp",
            "ia": None, "input_dim": N_IN, "output_dim": width,
            "compile_mode": None, "needs_geom": False,
            "kwargs": {"int_dim_res": 16, "n_blocks": 2,
                       "block_opts": {"act": {"type": "H", "n_gates": 3},
                                      "norm": "affine"}}}


def save_synthetic_grid2d(root, device, tmp, quantity="pklin",
                          units="Mpc3", law="syren_linear",
                          z=Z4, k=K6, seed=0):
    covmat = os.path.join(tmp, "g2_%d.covmat" % seed)
    write_covmat(covmat, IN_NAMES, seed=seed + 1)
    pgeom = ParamGeometry.from_covmat(
        device=device, center=np.array([2.1, 67.0, 0.12]),
        covmat_path=covmat)
    Y = synth_rows(400, z=z, k=k, seed=seed + 2)
    geom = Grid2DGeometry.from_targets(device=device, targets=Y, z=z,
                                       k=k, quantity=quantity,
                                       units=units, law=law)
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=z.size * k.size,
                   int_dim_res=16, n_blocks=2,
                   block_opts=block_opts).to(device)
    config = {"data": {"grid2d": {"quantity": quantity, "units": units,
                                  "law": law, "z_file": "z.npy",
                                  "k_file": "k.npy"},
                       "train_dv": "t.npy", "val_dv": "v.npy",
                       "train_params": "t.1.txt", "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat)},
              "train_args": {"nepochs": 1}}
    if law != "none":
        config["data"]["grid2d"]["train_base"] = "tb.npy"
        config["data"]["grid2d"]["val_base"] = "vb.npy"
    histories = {"train_losses": [0.1], "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=grid2d_recipe(z.size * k.size),
                  attrs={"rescale": "none", "quantity": quantity})
    return pgeom, geom, model, covmat


def check_roundtrip(tmp, device, law):
    root = os.path.join(tmp, "emul_g2_" + law)
    pgeom, geom, model, _ = save_synthetic_grid2d(
        root, device, tmp, law=law, seed=30)
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


def grid2d_head_recipe(width):
    """The model_recipe for the ResCNN head leg (D-CM13): needs_geom
    True (rebuild re-attaches the z-slice split via attach_head_coords),
    every constructor default materialized."""
    return {"cls": "emulator.designs.plain.ResCNN", "name": "rescnn",
            "ia": None, "input_dim": N_IN, "output_dim": width,
            "compile_mode": None, "needs_geom": True,
            "kwargs": {"int_dim_res": 16, "n_blocks": 2,
                       "kernel_size": 3, "rescale_kernel": False,
                       "groups": 1, "separable": False, "film": False,
                       "n_blocks_cnn": 1, "gate_init": 0.1,
                       "head_act": None,
                       "block_opts": {"act": {"type": "H", "n_gates": 3},
                                      "norm": "affine"}}}


def check_head(tmp, device):
    """D-CM13: the conv head on the grid2d geometry — z slices as
    channels, the identity basis, the epoch-0 identity start, the
    n_tokens rejection on real physical bins, and save -> rebuild ->
    predict bitwise (proving the rebuild-side attach_head_coords in
    results._rebuild_model)."""
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
    config = {"data": {"grid2d": {"quantity": "pklin", "units": "Mpc3",
                                  "law": "syren_linear",
                                  "z_file": "z.npy", "k_file": "k.npy",
                                  "train_base": "tb.npy",
                                  "val_base": "vb.npy"},
                       "train_dv": "t.npy", "val_dv": "v.npy",
                       "train_params": "t.1.txt", "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat)},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1], "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=grid2d_head_recipe(width),
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
    X_white = pgeom.encode(torch.from_numpy(C).to(device))
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
    config = {"data": {"grid2d": {"quantity": "pklin", "units": "Mpc3",
                                  "law": "syren_linear",
                                  "z_file": "z.npy", "k_file": "k.npy",
                                  "train_base": "tb.npy",
                                  "val_base": "vb.npy"},
                       "train_dv": "t.npy", "val_dv": "v.npy",
                       "train_params": "t.1.txt", "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat)},
              "pce": {"form": "residual"},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1], "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=grid2d_recipe(width),
                  pce=pce, pce_form="residual",
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
    root = Path(__file__).resolve().parents[2]
    path = root / "cobaya_theory" / "emul_mps.py"
    spec = importlib.util.spec_from_file_location("emul_mps_shim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_adapter(tmp, device):
    mod = _load_emul_mps_stubbed()
    cls = mod.emul_mps
    root_p = os.path.join(tmp, "ad_pklin")
    save_synthetic_grid2d(root_p, device, tmp, quantity="pklin",
                          units="Mpc3", law="syren_linear", seed=40)
    root_b = os.path.join(tmp, "ad_boost")
    save_synthetic_grid2d(root_b, device, tmp, quantity="boost",
                          units="dimensionless", law="syren_halofit",
                          seed=50)

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
        point = {"As": 2.1e-9, "ns": 0.965, "H0": 67.0, "omegab": 0.049,
                 "omegam": 0.31, "omch2": 0.12}
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
        except ValueError:
            report("pair-count guard raises", True, "ValueError")
        root_p2 = os.path.join(tmp, "ad_pklin2")
        save_synthetic_grid2d(root_p2, device, tmp, quantity="pklin",
                              units="Mpc3", law="none", seed=60)
        try:
            t3 = cls()
            t3.extra_args = {"device": "cpu",
                             "emulators": [root_p, root_p2]}
            t3.initialize()
            report("duplicate quantity raises", False, "no raise")
        except ValueError:
            report("duplicate quantity raises", True, "ValueError")
        root_bad = os.path.join(tmp, "ad_badgrid")
        save_synthetic_grid2d(root_bad, device, tmp, quantity="boost",
                              units="dimensionless", law="none",
                              k=np.logspace(-4, 0.0, 5), seed=70)
        try:
            t4 = cls()
            t4.extra_args = {"device": "cpu",
                             "emulators": [root_p, root_bad]}
            t4.initialize()
            report("grid mismatch raises", False, "no raise")
        except ValueError:
            report("grid mismatch raises", True, "ValueError")
    finally:
        mod.syren_base.base_pklin, mod.syren_base.base_boost = saved


def check_validate():
    def cfg(g2, extra=None):
        data = {"grid2d": g2, "train_dv": "a", "val_dv": "b",
                "train_params": "c", "val_params": "d",
                "train_covmat": "e"}
        if extra:
            data.update(extra)
        return {"data": data, "pce": None, "transfer": None}
    good = {"quantity": "boost", "units": "dimensionless",
            "law": "syren_halofit", "z_file": "z.npy", "k_file": "k.npy",
            "train_base": "tb.npy", "val_base": "vb.npy",
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
    c = cfg(good)
    c["transfer"] = {"from": "x"}
    try:
        validate_grid2d(c, train_args={}, rescale="none")
        ok = False
    except ValueError as e:
        ok = ok and "PERMANENTLY" in str(e)
    report("validate_grid2d legs", ok, "")


def check_finetune(tmp, device):
    root = os.path.join(tmp, "ft_g2_src")
    pgeom, geom, model, covmat = save_synthetic_grid2d(
        root, device, tmp, law="syren_linear", seed=80)
    source = warmstart.load_source(root=root, device=device)
    report("load_source accepts a grid2d artifact",
           type(source.geom).__name__ == "Grid2DGeometry",
           "geom %s" % type(source.geom).__name__)
    g = np.random.default_rng(90)
    C = np.column_stack([g.normal(2.1, 0.1, 64),
                         g.normal(67.0, 2.0, 64),
                         g.normal(0.12, 0.005, 64)]).astype("float32")
    train_set = {"C": C, "idx": np.arange(64), "C_mean": C.mean(axis=0)}
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
    def ft_cfg(g2, from_root):
        return {"data": {"grid2d": g2,
                         "train_dv": "t.npy", "val_dv": "v.npy",
                         "train_params": "t.1.txt",
                         "val_params": "v.1.txt", "train_covmat": covmat,
                         "n_train": 10, "n_val": 5, "split_seed": 0},
                "train_args": {"nepochs": 1, "bs": 8,
                               "finetune": {"from": from_root}}}
    good = {"quantity": "pklin", "units": "Mpc3", "law": "syren_linear",
            "z_file": "z.npy", "k_file": "k.npy",
            "train_base": "tb.npy", "val_base": "vb.npy"}
    bad = dict(good)
    bad["quantity"] = "boost"
    bad["units"] = "dimensionless"
    bad["law"] = "syren_halofit"
    try:
        EmulatorExperiment.from_config(ft_cfg(bad, root),
                                       device=torch.device("cpu"))
        report("D-MP7 metadata mismatch raises", False, "no raise")
    except ValueError as e:
        report("D-MP7 metadata mismatch raises",
               "grid2d-metadata mismatch" in str(e), "ValueError")


def main():
    print("mps-identity (MPS-A): geometry + staging law + round-trip + "
          "assembly + finetune legs")
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        check_geometry(device)
        check_staging(tmp)
        check_roundtrip(tmp, device, law="syren_linear")
        check_roundtrip(tmp, device, law="none")
        check_head(tmp, device)
        check_npce(tmp, device)
        check_adapter(tmp, device)
        check_validate()
        check_finetune(tmp, device)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: mps-identity all checks green")


if __name__ == "__main__":
    main()
