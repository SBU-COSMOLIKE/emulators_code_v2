#!/usr/bin/env python3
"""bsn-smoke gate: the BAOSN emulators end to end on real CAMB.

The strongest smoke of the program — TRUTH IS AVAILABLE: CAMB's own
background at the test point is exact, so the served H / D_A / D_M are
checked against it, not against a proxy.

  1  dataset_generator_background.py writes two tiny dumps (200 rows,
     uniform over omegam / H0 / w): H(z) on the SN grid + D_M on the
     recombination window, one background CAMB evaluation per sample
     (fast — no perturbations), plus the _z.npy grid sidecars.
  2  two data.grid training runs (the "Hubble" artifact, log_offset
     law; the "D_M" artifact, none law), each with the
     dead-network-RELATIVE collapse bar (best val median < 0.5x the
     staged mean predictor, the dead-network rule).
  3  the real cobaya lifecycle through emul_baosn (get_model +
     add_requirements + the getters) at an off-center point, checked
     against CAMB's OWN background at that point: H and D_A in the SN
     window, D_M in the recombination window, each within 2%.
  4  the desert query is loud through the real lifecycle.
  5  the diagnostics leg: the grid pages build (2 pages for the
     Hubble artifact — bands + derived distances) and the PDF lands.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from emulator.experiment import EmulatorExperiment
from emulator.training import ordinary_median
from emulator.results import save_emulator

FAILURES = []
REPO = Path(__file__).resolve().parents[2]
NROWS = 200


def report(label, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


def gen_yaml():
    """The generator YAML: three sampled background params, camb theory."""
    return (
        "likelihood:\n"
        "  one: null\n"
        "theory:\n"
        "  camb:\n"
        "    path: ./external_modules/code/CAMB\n"
        "params:\n"
        "  omegam:\n"
        "    prior:\n"
        "      min: 0.24\n"
        "      max: 0.40\n"
        "    ref: 0.31\n"
        "    proposal: 0.01\n"
        "    latex: \\Omega_m\n"
        "    drop: true\n"
        "  H0:\n"
        "    prior:\n"
        "      min: 60.0\n"
        "      max: 75.0\n"
        "    ref: 67.36\n"
        "    proposal: 0.5\n"
        "    latex: H_0\n"
        "  w:\n"
        "    prior:\n"
        "      min: -1.3\n"
        "      max: -0.7\n"
        "    ref: -1.0\n"
        "    proposal: 0.02\n"
        "    latex: w\n"
        "  ombh2:\n"
        "    value: 0.02237\n"
        "  omch2:\n"
        "    value: 'lambda omegam, H0, ombh2: "
        "omegam*(H0/100)**2 - ombh2 - 0.06*(3.046/3)**0.75/94.0708'\n"
        "  mnu:\n"
        "    value: 0.06\n"
        "  As:\n"
        "    value: 2.1e-9\n"
        "  ns:\n"
        "    value: 0.965\n"
        "  tau:\n"
        "    value: 0.055\n"
        "train_args:\n"
        "  probe: background\n"
        "  ord: [['omegam', 'H0', 'w']]\n"
        "  z_sn:  [0.01, 3.0, 120]\n"
        "  z_rec: [1000.0, 1200.0, 24]\n")


def run_generator(rootdir, rel_root, tag):
    cmd = [sys.executable,
           str(REPO / "compute_data_vectors"
               / "dataset_generator_background.py"),
           "--root", rel_root, "--fileroot", "emul", "--yaml", "gen.yaml",
           "--datavsfile", "dvs_" + tag, "--paramfile", "params_" + tag,
           "--failfile", "failed_" + tag, "--chain", "0",
           "--nparams", str(NROWS), "--unif", "1", "--temp", "2", "--seed", "1234"]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=rootdir)


def check_generate(rootdir, rel_root):
    emul_dir = os.path.join(rootdir, rel_root, "emul")
    os.makedirs(emul_dir, exist_ok=True)
    with open(os.path.join(emul_dir, "gen.yaml"), "w") as f:
        f.write(gen_yaml())
    chains = os.path.join(rootdir, rel_root, "chains")
    out = {}
    for tag in ("train", "val"):
        proc = run_generator(rootdir, rel_root, tag)
        stem = os.path.join(chains, "params_%s_background_unifs" % tag)
        dv = os.path.join(chains, "dvs_%s_background_unifs" % tag)
        files_ok = all(os.path.isfile(p) for p in (
            stem + ".1.txt", stem + ".covmat",
            dv + "_h.npy", dv + "_h_z.npy",
            dv + "_dm.npy", dv + "_dm_z.npy"))
        detail = "rc=%d" % proc.returncode
        if not files_ok:
            detail += " missing; stderr tail: " + proc.stderr.strip()[-200:]
        report("background dump (%s): both quantities + grids" % tag,
               proc.returncode == 0 and files_ok, detail)
        out[tag] = {"stem": stem, "dv": dv}
    return out


def build_cfg(paths, quantity):
    """The data.grid training config for one quantity ('h' or 'dm')."""
    q = "h" if quantity == "Hubble" else "dm"
    grid = {"quantity": quantity,
            "units": "km/s/Mpc" if quantity == "Hubble" else "Mpc",
            "law": "log_offset" if quantity == "Hubble" else "none",
            "z_file": paths["train"]["dv"] + "_%s_z.npy" % q}
    if quantity == "Hubble":
        grid["offset"] = 0.0
    return {
        "data": {
            "grid": grid,
            "train_dv":     paths["train"]["dv"] + "_%s.npy" % q,
            "val_dv":       paths["val"]["dv"] + "_%s.npy" % q,
            "train_params": paths["train"]["stem"] + ".1.txt",
            "val_params":   paths["val"]["stem"] + ".1.txt",
            "train_covmat": paths["train"]["stem"] + ".covmat",
            "n_train":      180,
            "n_val":        180,
            "split_seed":   0,
        },
        "train_args": {
            "nepochs": 40,
            "bs": 64,
            "model": {"name": "resmlp",
                      "mlp": {"width": 32, "n_blocks": 2}},
            "loss": {"mode": "sqrt"},
            "optimizer": {"weight_decay": 0.0},
            "lr": {"lr_base": 0.01,
                   "bs_base": 64.0,
                   "warmup_epochs": 0},
            "scheduler": {"mode": "min",
                          "patience": 10,
                          "factor": 0.8},
            "trim": {"start": 0.0,
                     "end": 0.0,
                     "hold_epochs": 0,
                     "anneal_epochs": 1,
                     "shape": "cosine"},
            "focus": {"start": 0.0,
                      "end": 0.0,
                      "hold_epochs": 0,
                      "anneal_epochs": 1,
                      "shape": "linear",
                      "kappa": 0.15},
        },
    }


def check_train(paths, tmp, device, quantity):
    cfg = build_cfg(paths, quantity)
    exp = EmulatorExperiment.from_config(cfg, device=device, quiet=True)
    model, train_losses, medians, means, fracs = exp.run()
    idx = np.sort(exp.val_set["idx"])
    dv = torch.from_numpy(np.asarray(exp.val_set["dv"][idx])).float()
    tw = exp.chi2fn.encode(dv.to(device))
    c_mean = exp.chi2fn.chi2(pred=torch.zeros_like(tw), target=tw)
    mean_median = ordinary_median(c_mean)  # unit 60: the ordinary median
    best_median = min(float(m) for m in medians)
    report("val collapses below the mean predictor (%s)" % quantity,
           best_median < 0.5 * mean_median,
           "best %.3g vs mean-predictor %.3g" % (best_median, mean_median))
    root = os.path.join(tmp, "emul_bsn_" + quantity.lower())
    save_emulator(path_root=root, model=model, param_geometry=exp.pgeom,
                  geometry=exp.geom, config=cfg,
                  histories={"train_losses": train_losses,
                             "val_medians": medians,
                             "val_means": means,
                             "val_fracs": fracs,
                             "thresholds": exp.thresholds},
                  train_args=exp.train_args, pce=None, pce_form=None,
                  resolved_train=exp.resolved_train,
                  resolved_model=exp.resolved_model, transfer_base=None,
                  attrs={"rescale": "none", "quantity": quantity})
    return exp, model, root


def camb_truth(point, z_sn, z_rec):
    """CAMB's own background at the test point (the exact reference).

    The CAMB path is resolved ABSOLUTELY from $ROOTDIR: cocoa's
    conventional "./external_modules/code/CAMB" only works when the
    process runs from $ROOTDIR, and this check runs in-process from
    the repo directory (board run 5 caught the relative form failing
    with "camblib.so not found"). The generator legs never hit this —
    their subprocesses run with cwd=rootdir.
    """
    from cobaya.model import get_model
    camb_path = os.path.join(os.environ["ROOTDIR"],
                             "external_modules/code/CAMB")
    info = {
        "likelihood": {"one": None},
        "theory": {"camb": {"path": camb_path}},
        "params": {
            "omegam": {"value": point["omegam"], "drop": True},
            "H0":     {"value": point["H0"]},
            "w":      {"value": point["w"]},
            "ombh2":  {"value": 0.02237},
            "omch2":  {"value": "lambda omegam, H0, ombh2: "
                                "omegam*(H0/100)**2 - ombh2 "
                                "- 0.06*(3.046/3)**0.75/94.0708"},
            "mnu":    {"value": 0.06},
            "As":     {"value": 2.1e-9},
            "ns":     {"value": 0.965},
            "tau":    {"value": 0.055},
        },
    }
    model = get_model(info)
    model.add_requirements(
        {"Hubble": {"z": z_sn, "units": "km/s/Mpc"},
         "angular_diameter_distance": {"z": z_sn},
         "comoving_radial_distance": {"z": np.concatenate([z_sn, z_rec])}})
    model.logposterior({})
    prov = model.provider
    return {"H": prov.get_Hubble(z_sn, units="km/s/Mpc"),
            "da": prov.get_angular_diameter_distance(z_sn),
            "dm_rec": prov.get_comoving_radial_distance(z_rec)}


def check_cobaya(root_h, root_dm, tmp):
    try:
        from cobaya.model import get_model
    except Exception as e:
        report("cobaya lifecycle through emul_baosn", False,
               "cobaya not importable: " + str(e))
        return
    point = {"omegam": 0.33,
             "H0": 69.5,
             "w": -0.95}
    z_sn = np.array([0.2, 0.8, 1.5, 2.5])
    z_rec = np.array([1060.0, 1090.0, 1150.0])
    info = {
        "likelihood": {"one": None},
        "theory": {"emul_baosn": {
            "python_path": str(REPO / "cobaya_theory"),
            "extra_args": {"device": "cpu",
                           "emulators": [root_h, root_dm]}}},
        "params": {
            "omegam": {"prior": {"min": 0.24, "max": 0.40},
                       "ref": 0.31,
                       "proposal": 0.01},
            "H0":     {"prior": {"min": 60.0, "max": 75.0},
                       "ref": 67.36,
                       "proposal": 0.5},
            "w":      {"prior": {"min": -1.3, "max": -0.7},
                       "ref": -1.0,
                       "proposal": 0.02},
        },
    }
    try:
        model = get_model(info)
        model.add_requirements(
            {"Hubble": {"z": z_sn},
             "angular_diameter_distance": {"z": z_sn},
             "comoving_radial_distance": {"z": z_rec}})
        model.logposterior(point)
        theory = list(model.theory.values())[0]
        got_h = theory.get_Hubble(z_sn)
        got_da = theory.get_angular_diameter_distance(z_sn)
        got_dm = theory.get_comoving_radial_distance(z_rec)
    except Exception as e:
        report("cobaya lifecycle through emul_baosn", False,
               type(e).__name__ + ": " + str(e)[:200])
        return
    truth = camb_truth(point, z_sn, z_rec)
    rel_h = np.abs(got_h / truth["H"] - 1).max()
    rel_da = np.abs(got_da / truth["da"] - 1).max()
    rel_dm = np.abs(got_dm / truth["dm_rec"] - 1).max()
    report("H / D_A / D_M vs CAMB's own background (2%)",
           max(rel_h, rel_da, rel_dm) < 0.02,
           "rel H %.3g, D_A %.3g, D_M %.3g" % (rel_h, rel_da, rel_dm))
    # the desert stays loud through the real lifecycle.
    try:
        theory.get_comoving_radial_distance(np.array([500.0]))
        report("desert query loud through the real lifecycle",
               False, "no raise")
    except ValueError:
        report("desert query loud through the real lifecycle",
               True, "ValueError")


def check_diagnostics(exp, model, tmp):
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from emulator.diagnostics import grid_residual_diagnostic
        from emulator.plotting import _grid_pages, plot_diagnostics
        import matplotlib.pyplot as plt
        gd = grid_residual_diagnostic(model=model,
                                      param_geometry=exp.pgeom,
                                      chi2fn=exp.chi2fn,
                                      val_set=exp.val_set,
                                      device=exp.device)
        figs = _grid_pages(gd)
        n_pages = len(figs)
        for f in figs:
            plt.close(f)
        pdf = os.path.join(tmp, "bsn_diag.pdf")
        # the hand-built fracs row must MATCH the run's threshold count
        # (DEFAULT_THRESHOLDS has five entries; a fixed 4-wide row made
        # the history panel index column 4 out of bounds — the first
        # execution of scalar-smoke's leg caught it in all four smoke
        # gates).
        plot_diagnostics(train_losses=[0.1], medians=[0.1], means=[0.1],
                         fracs=[0.5 * torch.ones(int(exp.thresholds.numel()))],
                         thresholds=exp.thresholds,
                         coverage={"knn_dist": np.ones(4),
                                   "dchi2": np.ones(4),
                                   "k_nn": 2},
                         grid=gd, savepath=pdf)
        ok = (n_pages == 2 and os.path.isfile(pdf)
              and os.path.getsize(pdf) > 10000)
        report("diagnostics: 2 grid pages + the PDF lands", ok,
               "%d pages, %d bytes" % (n_pages,
                                       os.path.getsize(pdf)
                                       if os.path.isfile(pdf) else 0))
    except Exception as e:
        report("diagnostics: 2 grid pages + the PDF lands", False,
               type(e).__name__ + ": " + str(e)[:200])


def check_dump_variance(paths):
    """The stale-cache tripwire: the H dump must VARY across cosmologies.

    Board run 1 caught the background generator writing the SAME
    background for every sample (the legacy hand-rolled
    check_cache_and_compute component loop served stale physics on a
    background-only requirement set); board run 3 falsified the
    wants-Cl-quirk hypothesis (the quirk was added and this leg still
    measured spread exactly 0.0). The fix that stands is the standard
    model.logposterior(point, cached=False) lifecycle inside
    _compute_dvs_from_sample. The geometry guard catches staleness too,
    but deep inside training with a message blaming the dump; this leg
    fails AT THE DUMP, naming the generator. The bar is loose on
    purpose: the H0 prior width alone gives a relative spread of ~6e-2
    at low z, while a stale dump gives ~0 — anything within four
    decades of the healthy value passes.
    """
    h = np.load(paths["train"]["dv"] + "_h.npy")
    rel = h.std(axis=0) / np.abs(h.mean(axis=0))
    worst = float(rel.min())
    report("H dump varies across cosmologies (stale-cache tripwire)",
           worst > 1e-5,
           "min relative spread %.2e (~0 means the logposterior("
           "cached=False) lifecycle in dataset_generator_background.py "
           "regressed to a cached path)" % worst)


def main():
    print("bsn-smoke: generator + two trainings + cobaya vs CAMB "
          "truth + diagnostics")
    rootdir = os.environ.get("ROOTDIR")
    if not rootdir:
        print("FAIL: ROOTDIR is not set")
        sys.exit(1)
    device = torch.device("cpu")
    rel_root = "tmp_gate_bsn_smoke_%d" % os.getpid()
    work = os.path.join(rootdir, rel_root)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            paths = check_generate(rootdir, rel_root)
            if not FAILURES:
                check_dump_variance(paths)
            if not FAILURES:
                exp_h, model_h, root_h = check_train(paths, tmp, device,
                                                     "Hubble")
                _, _, root_dm = check_train(paths, tmp, device, "D_M")
                check_cobaya(root_h, root_dm, tmp)
                check_diagnostics(exp_h, model_h, tmp)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: bsn-smoke all checks green")


if __name__ == "__main__":
    main()
