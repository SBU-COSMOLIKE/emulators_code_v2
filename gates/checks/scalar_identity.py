#!/usr/bin/env python3
"""scalar-identity gate (SPE-A): the scalar-emulator save/rebuild/predict
identity and every scalar-path loud error, torch only (no cosmolike).

It builds a tiny synthetic scalar emulator by hand (a ParamGeometry over a
written covmat + a ScalarGeometry over synthetic targets + a small ResMLP),
saves it with save_emulator, rebuilds it, and asserts:
  - predict returns a {name: value} dict that reproduces the pre-save model
    bitwise (save/rebuild preserves the weights + the standardization);
  - ScalarGeometry.state() round-trips byte-identical;
  - the scalar-path loud errors fire: D-SPE1-1 (a constant output column),
    D-SPE2-1 (a duplicated .paramnames name), D-SPE2-3 (a head architecture
    on a scalar run);
  - the cobaya adapter emul_scalars derives its provides from the artifacts
    and raises on a duplicate output, an input/provide overlap, a bad
    `provides` subset, and a wrong-kind (data-vector) artifact (D-SPE2-4).

The adapter legs stub cobaya.theory before loading the shipped
emul_scalars.py (this gate is torch-only; the real cobaya lifecycle is the
scalar-smoke board gate). The real training / cobaya evaluate is scalar-smoke.
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
from emulator.geometries.output import DataVectorGeometry
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry
from emulator.data_staging import _scalar_columns
from emulator.experiment import EmulatorExperiment
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor

FAILURES = []

N_IN    = 3                          # input parameters
IN_NAMES  = ["omegabh2", "omegach2", "thetastar"]
OUT_NAMES = ["H0", "omegam"]         # emulated derived parameters
N_OUT   = len(OUT_NAMES)


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
    with open(path, "w") as f:
        f.write("# " + " ".join(names) + "\n")
        for row in cov:
            f.write(" ".join(repr(float(x)) for x in row) + "\n")


def scalar_recipe():
    """The model_recipe a schema-v2 save stores for the scalar ResMLP.

    Mirrors EmulatorExperiment.build_specs on a scalar run: ia None,
    output_dim = the emulated-output count, needs_geom False.
    """
    return {
        "cls": "emulator.designs.plain.ResMLP",
        "name": "resmlp",
        "ia": None,
        "input_dim": N_IN,
        "output_dim": N_OUT,
        "compile_mode": None,
        "needs_geom": False,
        "kwargs": {
            "int_dim_res": 16,
            "n_blocks": 2,
            "block_opts": {"act": {"type": "H", "n_gates": 3},
                           "norm": "affine"},
        },
    }


def save_synthetic_scalar(root, device, covmat_path, seed=0,
                          in_names=None, out_names=None):
    """Build, then save, a tiny synthetic scalar emulator under `root`.

    A ParamGeometry over the written covmat (inputs), a ScalarGeometry over
    synthetic targets (outputs), and a freshly initialized ResMLP. No
    training runs, the identity path only needs consistent weights.

    Returns:
      (pgeom, geom, model): the source geometries + the pre-save model, for
      the bitwise round-trip reference.
    """
    in_names  = IN_NAMES  if in_names  is None else in_names
    out_names = OUT_NAMES if out_names is None else out_names
    write_covmat(covmat_path, in_names, seed=seed + 1)
    center = np.random.default_rng(seed + 2).standard_normal(len(in_names))
    pgeom = ParamGeometry.from_covmat(device=device, center=center,
                                      covmat_path=covmat_path)
    # synthetic targets on physical scales, so standardization is non-trivial.
    g = np.random.default_rng(seed + 3)
    targets = g.normal(loc=[70.0, 0.3][:len(out_names)],
                       scale=[3.0, 0.02][:len(out_names)],
                       size=(2000, len(out_names))).astype("float32")
    geom = ScalarGeometry.from_targets(device=device, targets=targets,
                                       names=out_names)
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=len(in_names),
                   output_dim=len(out_names),
                   int_dim_res=16,
                   n_blocks=2,
                   block_opts=block_opts).to(device)
    config = {"data": {"train_params": "src_train.1.txt",
                       "val_params": "src_val.1.txt",
                       "train_covmat": os.path.basename(covmat_path),
                       "outputs": list(out_names)},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1],
                 "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    recipe = scalar_recipe()
    recipe["input_dim"] = len(in_names)
    recipe["output_dim"] = len(out_names)
    save_emulator(path_root=str(root),
                  model=model,
                  param_geometry=pgeom,
                  geometry=geom,
                  config=config,
                  histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=recipe,
                  # rescale rides the run-identity attrs so the artifact is
                  # a valid finetune source (load_source refuses an
                  # ambiguous one — the never-trust-defaults rule).
                  attrs={"outputs": " ".join(out_names),
                         "rescale": "none"})
    return pgeom, geom, model


def check_roundtrip(root, device, pgeom, geom, model):
    """Item 1: rebuild -> predict {name: value} reproduces the pre-save model.

    The reference is the ORIGINAL model + geometry (before save); predict
    (after rebuild) must match it bitwise, proving save/rebuild preserved the
    weights and the standardization.
    """
    theta = {}
    row = np.random.default_rng(9).standard_normal(len(pgeom.names))
    for i, nm in enumerate(pgeom.names):
        theta[nm] = float(row[i])
    x = torch.as_tensor([row], dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        ref = geom.decode(model(pgeom.encode(x)))[0]
    ref_d = {}
    for i, nm in enumerate(geom.names):
        ref_d[nm] = float(ref[i])

    pred = EmulatorPredictor(str(root), device, compile_model=False)
    got = pred.predict(theta)
    ok = (set(got) == set(ref_d)
          and all(got[k] == ref_d[k] for k in ref_d))
    dmax = max((abs(got[k] - ref_d[k]) for k in ref_d), default=0.0)
    report("predict round-trip bitwise vs pre-save model",
           ok, "max|d| = %.2e, names %s" % (dmax, list(got)))
    report("predictor reports scalar branch + output_names",
           getattr(pred, "_scalar", False) and pred.output_names == geom.names,
           "output_names %s" % (getattr(pred, "output_names", None),))


def check_state(root, device, geom):
    """Item 2: the rebuilt ScalarGeometry.state() is byte-identical."""
    _, _, geom_rb, info = rebuild_emulator(str(root), device,
                                           compile_model=False)
    st0, st1 = geom.state(), geom_rb.state()
    ok = (st0["names"] == st1["names"]
          and torch.equal(st0["center"], st1["center"])
          and torch.equal(st0["scale"], st1["scale"]))
    report("ScalarGeometry state round-trip byte-identical", ok,
           "names/center/scale equal")
    report("rebuild info['scalar'] is True", bool(info.get("scalar")),
           "info scalar = %s" % (info.get("scalar"),))


def check_from_targets_errors(device):
    """D-SPE1-1: from_targets raises on an un-standardizable output column,
    and does NOT raise on a legitimate tiny-magnitude one."""
    try:
        ScalarGeometry.from_targets(
            device=device,
            targets=np.full((100, 1), 0.31, dtype="float32"),
            names=["const"])
        report("D-SPE1-1 constant column raises", False, "did not raise")
    except ValueError:
        report("D-SPE1-1 constant column raises (0.31)", True, "ValueError")
    # must-NOT-raise: a real tiny-magnitude output (center 1e-9, 10% spread ->
    # std ~ 1e-10, threshold 8*eps32*|center| ~ 1e-15) builds and standardizes
    # to unit variance; the relative guard passes it with orders to spare.
    tiny = np.random.default_rng(7).normal(
        1e-9, 1e-10, size=(4000, 1)).astype("float32")
    try:
        g = ScalarGeometry.from_targets(device=device, targets=tiny,
                                        names=["ok"])
        std = float(g.encode(torch.as_tensor(tiny, device=device)).std())
        report("D-SPE1-1 tiny-magnitude column builds (std ~ 1)",
               abs(std - 1.0) < 0.05,
               "did not raise; standardized std = %.4f" % std)
    except ValueError as e:
        report("D-SPE1-1 tiny-magnitude column builds", False,
               "unexpectedly raised: " + str(e))


def check_sidecar_errors(tmp):
    """D-SPE2-1: _scalar_columns raises on a duplicated .paramnames name."""
    dup = os.path.join(tmp, "dup.paramnames")
    with open(dup, "w") as f:
        f.write("omegabh2 x\nH0* a\nH0* b\n")
    try:
        _scalar_columns(dup, ["omegabh2"], ["H0"])
        report("D-SPE2-1 duplicate sidecar name raises", False, "no raise")
    except ValueError:
        report("D-SPE2-1 duplicate sidecar name raises", True, "ValueError")


def check_head_architecture():
    """D-SPE2-3: from_config raises on a head architecture (scalar is
    trunk-only). The head_block guard fires before the experiment is built,
    so dummy file names in the data block are enough."""
    cfg = {"data": {"train_params": "t.1.txt", "val_params": "v.1.txt",
                    "train_covmat": "t.covmat", "outputs": ["H0"],
                    "n_train": 10, "n_val": 5, "split_seed": 0},
           "train_args": {"nepochs": 1, "bs": 8, "model": {"name": "rescnn"}}}
    try:
        EmulatorExperiment.from_config(cfg, device=torch.device("cpu"))
        report("D-SPE2-3 rescnn scalar run raises", False, "did not raise")
    except ValueError as e:
        report("D-SPE2-3 rescnn scalar run raises", "trunk-only" in str(e),
               "ValueError names trunk-only")


def _load_emul_scalars_stubbed():
    """Load the shipped emul_scalars.py with cobaya.theory stubbed.

    scalar-identity is torch-only: stub cobaya + cobaya.theory (a trivial
    Theory base with a no-op initialize) before executing the module, so the
    adapter's provides / requirements / error logic runs without a real
    cobaya. EmulatorPredictor stays real (torch).
    """
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
    path = root / "cobaya_theory" / "emul_scalars.py"
    spec = importlib.util.spec_from_file_location("emul_scalars_shim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.emul_scalars


def _build(cls, roots, provides=None):
    """Instantiate the stubbed emul_scalars, set extra_args, initialize."""
    t = cls()
    t.extra_args = {"device": "cpu", "emulators": list(roots)}
    if provides is not None:
        t.extra_args["provides"] = list(provides)
    t.initialize()
    return t


def check_adapter(tmp, device):
    """emul_scalars: artifact-derived provides + every D-SP4 / D-SPE2-4 error."""
    cls = _load_emul_scalars_stubbed()
    # two disjoint scalar emulators: A provides H0/omegam from one input set,
    # B provides rdrag from another.
    root_a = os.path.join(tmp, "emul_a")
    save_synthetic_scalar(root_a, device, os.path.join(tmp, "a.covmat"),
                          seed=10, in_names=["omegabh2", "omegach2"],
                          out_names=["H0", "omegam"])
    root_b = os.path.join(tmp, "emul_b")
    save_synthetic_scalar(root_b, device, os.path.join(tmp, "b.covmat"),
                          seed=20, in_names=["omegabh2", "omegach2"],
                          out_names=["rdrag"])

    t = _build(cls, [root_a, root_b])
    report("auto-provides == union of stored output names",
           t.get_can_provide_params() == ["H0", "omegam", "rdrag"],
           "provides %s" % (t.get_can_provide_params(),))
    report("get_requirements is the input union (a dict)",
           set(t.get_requirements()) == {"omegabh2", "omegach2"},
           "req %s" % (sorted(t.get_requirements()),))

    # duplicate output across two artifacts -> loud.
    root_c = os.path.join(tmp, "emul_c")
    save_synthetic_scalar(root_c, device, os.path.join(tmp, "c.covmat"),
                          seed=30, in_names=["omegabh2", "omegach2"],
                          out_names=["H0"])
    try:
        _build(cls, [root_a, root_c])
        report("duplicate output name raises", False, "no raise")
    except ValueError:
        report("duplicate output name raises", True, "ValueError")

    # input/provide overlap -> loud: an emulator whose input is another's
    # output (H0 in -> would-be chain).
    root_d = os.path.join(tmp, "emul_d")
    save_synthetic_scalar(root_d, device, os.path.join(tmp, "d.covmat"),
                          seed=40, in_names=["H0", "omegach2"],
                          out_names=["sigma8"])
    try:
        _build(cls, [root_a, root_d])
        report("input/provide overlap raises", False, "no raise")
    except ValueError:
        report("input/provide overlap raises", True, "ValueError")

    # provides: subset ok, superset raises.
    ok_sub = True
    try:
        _build(cls, [root_a], provides=["H0"])
    except ValueError:
        ok_sub = False
    report("provides subset accepted", ok_sub, "H0 subset of {H0,omegam}")
    try:
        _build(cls, [root_a], provides=["H0", "not_provided"])
        report("provides superset raises", False, "no raise")
    except ValueError:
        report("provides superset raises", True, "ValueError")

    # D-SPE2-4 wrong-kind: a data-vector artifact -> loud.
    root_dv = os.path.join(tmp, "emul_dv")
    _save_tiny_dv(root_dv, device)
    try:
        _build(cls, [root_dv])
        report("D-SPE2-4 wrong-kind (dv artifact) raises", False, "no raise")
    except ValueError as e:
        report("D-SPE2-4 wrong-kind (dv artifact) raises",
               "not a scalar" in str(e), "ValueError names non-scalar")


def _save_tiny_dv(root, device):
    """Save a minimal data-vector emulator (a DataVectorGeometry artifact)."""
    n_in, out_dim, total = 3, 4, 6
    names = ["p0", "p1", "p2"]
    cov = spd(n_in, seed=101)
    lam, V = np.linalg.eigh(cov)
    pgeom = ParamGeometry(device=device, names=names,
                          center=np.zeros(n_in), evecs=V, sqrt_ev=np.sqrt(lam))
    cov_k = spd(out_dim, seed=102)
    lam_k, V_k = np.linalg.eigh(cov_k)
    geom = DataVectorGeometry(device=device, total_size=total,
                              dest_idx=list(range(out_dim)), evecs=V_k,
                              sqrt_ev=np.sqrt(lam_k), Cinv=spd(total, seed=103),
                              center=np.zeros(out_dim),
                              section_sizes=[total], probe="xi")
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=n_in, output_dim=out_dim, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    recipe = {"cls": "emulator.designs.plain.ResMLP", "name": "resmlp",
              "ia": None, "input_dim": n_in, "output_dim": out_dim,
              "compile_mode": None, "needs_geom": False,
              "kwargs": {"int_dim_res": 16, "n_blocks": 2,
                         "block_opts": {"act": {"type": "H", "n_gates": 3},
                                        "norm": "affine"}}}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom,
                  config={"data": {"cosmolike_data_dir": "lsst_y1",
                                   "cosmolike_dataset": "d.dataset",
                                   "train_dv": "t.npy", "val_dv": "v.npy"},
                          "train_args": {"nepochs": 1}},
                  histories={"train_losses": [0.1], "val_medians": [0.1],
                             "val_means": [0.1],
                             "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                             "thresholds": torch.tensor([0.2, 1.0, 10.0])},
                  train_args={"nepochs": 1}, resolved_train={"nepochs": 1},
                  resolved_model=recipe, attrs={"rescale": "none"})


def check_finetune(tmp, device):
    """SPE-FT (D-SF1/2/4): epoch-0 parity, outputs/wrong-kind legs, the
    anchor mask over padded extra columns."""
    from emulator import warmstart

    # a scalar source over two inputs; the run's covmat adds a third.
    root = os.path.join(tmp, "ft_src")
    src_cov = os.path.join(tmp, "ft_src.covmat")
    pgeom, geom, model = save_synthetic_scalar(
        root, device, src_cov, seed=200,
        in_names=["omegabh2", "omegach2"], out_names=["H0", "omegam"])
    source = warmstart.load_source(root=root, device=device)
    report("load_source accepts a scalar artifact",
           type(source.geom).__name__ == "ScalarGeometry",
           "geom %s" % type(source.geom).__name__)

    # parity on the SAME covmat (no extras): build_warm_start raises on a
    # parity violation, so returning is the pass; assert bitwise anyway.
    g = np.random.default_rng(210)
    C2 = g.standard_normal((64, 2)).astype("float32")
    train_set = {"C": C2, "idx": np.arange(64), "C_mean": C2.mean(axis=0)}
    new_pgeom, extra = warmstart.extend_input_geometry(
        source=source, covmat_path=src_cov,
        train_mean=train_set["C_mean"], device=device)
    model_opts = warmstart.recipe_to_model_opts(source.recipe)
    try:
        init_state, verdict, padded = warmstart.build_warm_start(
            source=source, new_pgeom=new_pgeom, pinned_geom=source.geom,
            model_opts=model_opts, train_set=train_set,
            extra_names=extra, device=device)
        report("scalar warm start reproduces the source at epoch 0",
               init_state is not None, verdict.strip()[:60])
    except ValueError as e:
        report("scalar warm start reproduces the source at epoch 0",
               False, "parity raised: " + str(e)[:80])

    # extended covmat (a third input): the padded keys' anchor mask zeros
    # exactly the appended column (D-SF2 = D-FT3 unchanged).
    ext_cov = os.path.join(tmp, "ft_ext.covmat")
    write_covmat(ext_cov, ["omegabh2", "omegach2", "thetastar"], seed=220)
    C3 = g.standard_normal((64, 3)).astype("float32")
    train_set3 = {"C": C3, "idx": np.arange(64), "C_mean": C3.mean(axis=0)}
    new_pgeom3, extra3 = warmstart.extend_input_geometry(
        source=source, covmat_path=ext_cov,
        train_mean=train_set3["C_mean"], device=device)
    init3, _, padded3 = warmstart.build_warm_start(
        source=source, new_pgeom=new_pgeom3, pinned_geom=source.geom,
        model_opts=model_opts, train_set=train_set3,
        extra_names=extra3, device=device)
    masks = warmstart.anchor_masks(init_state=init3, padded_keys=padded3,
                                   n_extra=len(extra3), device=device)
    ok = (extra3 == ["thetastar"] and len(masks) > 0)
    for key, m in masks.items():
        ok = ok and bool((m[:, -1] == 0).all()) \
                and bool((m[:, :-1] == 1).all())
    report("anchor mask zeros exactly the padded extra column",
           ok, "extras %s, %d masked key(s)" % (extra3, len(masks)))

    # the combined from_config path: outputs-mismatch and wrong-kind are
    # loud BEFORE any staging (dummy data file names suffice), and the
    # finetune YAML carries no model: block (the FTW model-block lesson).
    def ft_cfg(outputs, from_root):
        return {"data": {"train_params": "t.1.txt", "val_params": "v.1.txt",
                         "train_covmat": ext_cov, "outputs": list(outputs),
                         "n_train": 10, "n_val": 5, "split_seed": 0},
                "train_args": {"nepochs": 1, "bs": 8,
                               "finetune": {"from": from_root}}}
    try:
        EmulatorExperiment.from_config(ft_cfg(["H0"], root),
                                       device=torch.device("cpu"))
        report("D-SF1 outputs mismatch raises", False, "did not raise")
    except ValueError as e:
        report("D-SF1 outputs mismatch raises",
               "H0" in str(e) and "omegam" in str(e),
               "ValueError names both lists")
    dv_root = os.path.join(tmp, "ft_dv")
    _save_tiny_dv(dv_root, device)
    try:
        EmulatorExperiment.from_config(ft_cfg(["H0", "omegam"], dv_root),
                                       device=torch.device("cpu"))
        report("D-SF1 wrong-kind source raises", False, "did not raise")
    except ValueError as e:
        report("D-SF1 wrong-kind source raises",
               "scalar source" in str(e), "ValueError names the family")


def check_npce(tmp, device):
    """NPCE on the scalar family (the 2026-07-12 family-wide ruling): the
    residual base + refiner algebra is exact under the standardized
    metric, and save -> rebuild -> predict composes base + net exactly
    in the {name: value} dict (_build_diag_decoder on ScalarGeometry)."""
    from emulator.designs.pce import PCEEmulator
    from emulator.losses.pce import PCEResidualDiagChi2
    covmat_path = os.path.join(tmp, "npce.covmat")
    write_covmat(covmat_path, IN_NAMES, seed=81)
    center = np.array([0.0224, 0.120, 1.041])
    pgeom = ParamGeometry.from_covmat(device=device, center=center,
                                      covmat_path=covmat_path)
    g = np.random.default_rng(82)
    C = np.column_stack([g.normal(0.0224, 0.0002, 400),
                         g.normal(0.120, 0.002, 400),
                         g.normal(1.041, 0.001, 400)]).astype("float32")
    # outputs that REALLY move with the inputs (H0-like and omegam-like
    # linear responses), so the LOO gate keeps a mode and the fitted
    # base is alive (the smoke-gate rule).
    Y = np.column_stack([
        67.0 + 3000.0 * (C[:, 1] - 0.120) + 0.1 * g.standard_normal(400),
        0.31 + 5.0 * (C[:, 1] - 0.120)
        + 0.001 * g.standard_normal(400)]).astype("float32")
    geom = ScalarGeometry.from_targets(device=device, targets=Y,
                                       names=OUT_NAMES)
    X_white = pgeom.encode(torch.from_numpy(C).to(device))
    dv = torch.from_numpy(Y).to(device)
    pce = PCEEmulator.from_training(device, X_white, geom.encode(dv),
                                    p_max=2, r_max=2, q=0.5, k_max=2,
                                    loo_max=0.9, max_terms=8, silent=True)
    chi2fn = PCEResidualDiagChi2(geom=geom, pce=pce)
    with torch.no_grad():
        base = pce(X_white[:8])
    report("NPCE base is alive (the fit kept a real mode)",
           base.abs().max().item() > 1e-4,
           "max|base| = %.2e" % base.abs().max().item())
    enc = chi2fn.encode(dv[:8], X_white[:8])
    report("NPCE encode: standardized truth minus the base, bitwise",
           torch.equal(enc, geom.encode(dv[:8]) - base), "")
    y = torch.randn(8, N_OUT, device=device)
    report("NPCE decode: geom.decode(net + base), bitwise",
           torch.equal(chi2fn.decode(y, X_white[:8]),
                       geom.decode(y + base)), "")
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=N_OUT, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    root = os.path.join(tmp, "emul_npce")
    config = {"data": {"train_params": "t.1.txt", "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat_path),
                       "outputs": list(OUT_NAMES)},
              "pce": {"form": "residual"},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1], "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    recipe = scalar_recipe()
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=recipe,
                  pce=pce, pce_form="residual",
                  attrs={"outputs": " ".join(OUT_NAMES),
                         "rescale": "none"})
    theta = np.array([[0.0225, 0.121, 1.0412]])
    x1 = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        x1e = pgeom.encode(x1)
        ref = geom.decode(model(x1e) + pce(x1e))[0]
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    ok = True
    for i, nm in enumerate(OUT_NAMES):
        ok = ok and got[nm] == float(ref[i])
    report("NPCE save -> rebuild -> predict composes base + net exactly",
           ok, "H0 %.6f, omegam %.6f" % (got["H0"], got["omegam"]))


def main():
    """Run the scalar-identity checks in a tempdir and exit non-zero on any
    failure."""
    print("scalar-identity (SPE-A): save/rebuild/predict + loud errors")
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "emul")
        pgeom, geom, model = save_synthetic_scalar(
            root, device, os.path.join(tmp, "src.covmat"), seed=0)
        check_roundtrip(root, device, pgeom, geom, model)
        check_state(root, device, geom)
        check_from_targets_errors(device)
        check_sidecar_errors(tmp)
        check_head_architecture()
        check_npce(tmp, device)
        check_adapter(tmp, device)
        check_finetune(tmp, device)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): " + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: scalar-identity all checks green")


if __name__ == "__main__":
    main()
