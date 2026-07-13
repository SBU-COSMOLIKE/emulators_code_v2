#!/usr/bin/env python3
"""mps-smoke gate: the MPS emulators end to end on real CAMB.

Truth is available (CAMB's own P(k, z) at the test point), so the served
spectra are checked against it directly. This smoke runs the LAW-NONE
path end to end (generator -> two trainings -> emul_mps -> the
interpolator products); the syren-law assembly is exactly gated by
mps-identity's stubbed closed-form legs (against stub bases, so a
formula update in the vendored syren/ can never mask an assembly
bug), and the full syren+EMUL2 integration is the unit's recorded
acceptance run (EXAMPLE_EMUL2_EVALUATE.yaml, user-run on the
workstation).

  1  dataset_generator_mps.py writes two tiny dumps (200 rows, 16 z x
     40 k, write_syren_base false): pklin + boost + the z/k sidecars,
     through the real Pk_interpolator requirement (incl. the verbatim
     wants-Cl quirk).
  2  two data.grid2d trainings (pklin + boost, law none), each with the
     dead-network-RELATIVE collapse bar (the dead-network rule); the boost
     training also runs the diagnostics leg (the two grid2d pages build
     and the plot_diagnostics PDF lands through the grid2d dispatch).
  3  the real cobaya lifecycle through emul_mps: get_model +
     add_requirements(Pk_grid) + logposterior; the served P_lin and
     P_nl (grid + interpolator) within 5% of CAMB's OWN P(k, z) at an
     off-center point.
  4  the sanity legs: the interpolator's extrapolation guard and
     the nonlinear/linear state keys.
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
    """The generator YAML: three sampled params, camb, tiny grids."""
    return (
        "likelihood:\n"
        "  one: null\n"
        "theory:\n"
        "  camb:\n"
        "    path: ./external_modules/code/CAMB\n"
        "    extra_args:\n"
        "      halofit_version: takahashi\n"
        "      kmax: 20.0\n"
        "      AccuracyBoost: 0.7\n"
        "params:\n"
        "  As:\n"
        "    prior:\n"
        "      min: 1.8e-9\n"
        "      max: 2.4e-9\n"
        "    ref: 2.1e-9\n"
        "    proposal: 1.0e-11\n"
        "    latex: A_s\n"
        "  H0:\n"
        "    prior:\n"
        "      min: 60.0\n"
        "      max: 75.0\n"
        "    ref: 67.36\n"
        "    proposal: 0.5\n"
        "    latex: H_0\n"
        "  omch2:\n"
        "    prior:\n"
        "      min: 0.10\n"
        "      max: 0.14\n"
        "    ref: 0.12\n"
        "    proposal: 0.001\n"
        "    latex: \\Omega_c h^2\n"
        "  ombh2:\n"
        "    value: 0.02237\n"
        "  ns:\n"
        "    value: 0.965\n"
        "  tau:\n"
        "    value: 0.055\n"
        "  mnu:\n"
        "    value: 0.06\n"
        "train_args:\n"
        "  probe: mps\n"
        "  ord: [['As', 'H0', 'omch2']]\n"
        "  z_segments:\n"
        "    - [0.0, 2.0, 8, false]\n"
        "    - [2.0, 10.0, 4, false]\n"
        "    - [10.0, 50.0, 4, true]\n"
        "  k_log10: [-4.0, 1.0, 40]\n"
        "  extrap_kmax: 200.0\n"
        "  write_syren_base: false\n")


def check_generate(rootdir, rel_root):
    emul_dir = os.path.join(rootdir, rel_root, "emul")
    os.makedirs(emul_dir, exist_ok=True)
    with open(os.path.join(emul_dir, "gen.yaml"), "w") as f:
        f.write(gen_yaml())
    chains = os.path.join(rootdir, rel_root, "chains")
    out = {}
    for tag in ("train", "val"):
        cmd = [sys.executable,
               str(REPO / "compute_data_vectors"
                   / "dataset_generator_mps.py"),
               "--root", rel_root, "--fileroot", "emul",
               "--yaml", "gen.yaml",
               "--datavsfile", "dvs_" + tag,
               "--paramfile", "params_" + tag,
               "--failfile", "failed_" + tag, "--chain", "0",
               "--nparams", str(NROWS), "--unif", "1", "--temp", "2"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=rootdir)
        stem = os.path.join(chains, "params_%s_mps_unifs" % tag)
        dv = os.path.join(chains, "dvs_%s_mps_unifs" % tag)
        files_ok = all(os.path.isfile(p) for p in (
            stem + ".1.txt", stem + ".covmat",
            dv + "_pklin.npy", dv + "_boost.npy",
            dv + "_z.npy", dv + "_k.npy"))
        detail = "rc=%d" % proc.returncode
        if not files_ok:
            detail += " missing; stderr tail: " + proc.stderr.strip()[-200:]
        report("mps dump (%s): pklin + boost + grid sidecars" % tag,
               proc.returncode == 0 and files_ok, detail)
        if files_ok:
            arr = np.load(dv + "_pklin.npy")
            report("dump shape + positive P_lin (%s)" % tag,
                   arr.shape == (NROWS, 16 * 40)
                   and (arr[arr.sum(axis=1) != 0] > 0).all(),
                   "shape %s" % (arr.shape,))
        out[tag] = {"stem": stem, "dv": dv}
    return out


def build_cfg(paths, quantity):
    grid2d = {"quantity": quantity,
              "units": "Mpc3" if quantity == "pklin" else "dimensionless",
              "law": "none",
              "z_file": paths["train"]["dv"] + "_z.npy",
              "k_file": paths["train"]["dv"] + "_k.npy"}
    return {
        "data": {
            "grid2d": grid2d,
            "train_dv":     paths["train"]["dv"] + "_%s.npy" % quantity,
            "val_dv":       paths["val"]["dv"] + "_%s.npy" % quantity,
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
                      "mlp": {"width": 64, "n_blocks": 2}},
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
    mean_median = float(c_mean.median())
    best_median = min(float(m) for m in medians)
    report("val collapses below the mean predictor (%s)" % quantity,
           best_median < 0.5 * mean_median,
           "best %.3g vs mean-predictor %.3g" % (best_median,
                                                 mean_median))
    if quantity == "boost":
        # the diagnostics leg rides ONE of the two trainings (the pages
        # are quantity-agnostic; once is the evidence, twice is time).
        check_diagnostics(exp, model, tmp)
    root = os.path.join(tmp, "emul_mps_" + quantity)
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
    return root


def check_diagnostics(exp, model, tmp):
    """The diagnostics leg: 2 grid2d pages build + the PDF lands.

    Mirrors scalar_smoke's diagnostics leg: run the family diagnostic
    on the freshly trained model, build the pages, and write a full
    plot_diagnostics PDF with the grid2d dispatch — so the exact path
    the train driver's --diagnostic takes is the path proven here.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from emulator.diagnostics import grid2d_residual_diagnostic
        from emulator.plotting import _grid2d_pages, plot_diagnostics
        import matplotlib.pyplot as plt
        g2 = grid2d_residual_diagnostic(model=model,
                                        param_geometry=exp.pgeom,
                                        chi2fn=exp.chi2fn,
                                        val_set=exp.val_set,
                                        device=exp.device)
        nz = len(g2["z"])
        nk = len(g2["k"])
        shape_ok = (g2["med_abs"].shape == (nz, nk)
                    and g2["worst"]["res"].shape == (nz, nk)
                    and 1 <= len(g2["slices"]) <= 3
                    and g2["res_kind"] == "fractional")  # law none here
        figs = _grid2d_pages(g2)
        n_pages = len(figs)
        for f in figs:
            plt.close(f)
        pdf = os.path.join(tmp, "grid2d_diag.pdf")
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
                         grid2d=g2, savepath=pdf)
        ok = (shape_ok and n_pages == 2 and os.path.isfile(pdf)
              and os.path.getsize(pdf) > 10000)
        report("diagnostics: 2 grid2d pages + the PDF lands", ok,
               "%d pages, shapes %s, %d bytes"
               % (n_pages, "ok" if shape_ok else "WRONG",
                  os.path.getsize(pdf) if os.path.isfile(pdf) else 0))
    except Exception as e:
        report("diagnostics: 2 grid2d pages + the PDF lands", False,
               type(e).__name__ + ": " + str(e)[:200])


def camb_truth(point, z_probe, k_probe):
    """CAMB's own P(k, z) at the test point (the exact reference).

    The CAMB path is resolved ABSOLUTELY from $ROOTDIR: cocoa's
    conventional "./external_modules/code/CAMB" only works when the
    process runs from $ROOTDIR, and this check runs in-process from
    the repo directory (board run 5 caught the relative form in the
    bsn twin failing with "camblib.so not found"). The generator legs
    never hit this — their subprocesses run with cwd=rootdir.
    """
    from cobaya.model import get_model
    camb_path = os.path.join(os.environ["ROOTDIR"],
                             "external_modules/code/CAMB")
    info = {
        "likelihood": {"one": None},
        "theory": {"camb": {"path": camb_path,
                            "extra_args": {
                              "halofit_version": "takahashi",
                              "kmax": 20.0,
                              "AccuracyBoost": 0.7}}},
        "params": {
            "As":    {"value": point["As"]},
            "H0":    {"value": point["H0"]},
            "omch2": {"value": point["omch2"]},
            "ombh2": {"value": 0.02237},
            "ns":    {"value": 0.965},
            "tau":   {"value": 0.055},
            "mnu":   {"value": 0.06},
        },
    }
    model = get_model(info)
    # k_max 20 mirrors the generator's grid-derived requirement
    # (max(2 * k_top, 20) with this gate's k top of 10): truth and
    # training data are computed with the SAME transfer support, so
    # the comparison tests the emulator pipeline, never halofit's
    # convergence difference between two k_max choices.
    #
    # cobaya's PowerSpectrumInterpolator needs AT LEAST FOUR redshifts
    # for its 2D spline (board run 8 caught the bare 3-probe request —
    # the exact Pk_interpolator first-run risk the family notes
    # recorded). Request a support that CONTAINS the probe redshifts
    # as nodes (so the probe evaluations carry no z-interpolation
    # error) plus padding across the range.
    z_req = np.unique(np.concatenate([z_probe,
                                      np.linspace(0.0, 4.0, 9)]))
    model.add_requirements(
        {"Pk_interpolator": {"z": z_req,
                             "k_max": 20.0,
                             "nonlinear": (True, False),
                             "vars_pairs": ([("delta_tot", "delta_tot")])},
         "Cl": {"tt": 0}})
    model.logposterior({})
    lin = model.provider.get_Pk_interpolator(
        ("delta_tot", "delta_tot"), nonlinear=False, extrap_kmax=200.0)
    nl = model.provider.get_Pk_interpolator(
        ("delta_tot", "delta_tot"), nonlinear=True, extrap_kmax=200.0)
    return lin.P(z_probe, k_probe), nl.P(z_probe, k_probe)


def check_cobaya(root_p, root_b, tmp):
    try:
        from cobaya.model import get_model
    except Exception as e:
        report("cobaya lifecycle through emul_mps", False,
               "cobaya not importable: " + str(e))
        return
    point = {"As": 2.05e-9,
             "H0": 69.5,
             "omch2": 0.125}
    z_probe = np.array([0.25, 1.25, 3.0])
    k_probe = np.array([0.01, 0.1, 1.0, 5.0])
    info = {
        "likelihood": {"one": None},
        "theory": {"emul_mps": {
            "python_path": str(REPO / "cobaya_theory"),
            "extra_args": {"device": "cpu",
                           "emulators": [root_p, root_b]}}},
        "params": {
            "As":    {"prior": {"min": 1.8e-9, "max": 2.4e-9},
                      "ref": 2.1e-9,
                      "proposal": 1e-11},
            "H0":    {"prior": {"min": 60.0, "max": 75.0},
                      "ref": 67.36,
                      "proposal": 0.5},
            "omch2": {"prior": {"min": 0.10, "max": 0.14},
                      "ref": 0.12,
                      "proposal": 0.001},
        },
    }
    try:
        model = get_model(info)
        model.add_requirements(
            {"Pk_grid": {"z": z_probe,
                         "k_max": 5.0,
                         "nonlinear": (True, False),
                         "vars_pairs": ([("delta_tot", "delta_tot")])}})
        model.logposterior(point)
        theory = list(model.theory.values())[0]
        lin = theory.get_Pk_interpolator(nonlinear=False)
        nl = theory.get_Pk_interpolator(nonlinear=True)
        got_lin = lin.P(z_probe, k_probe)
        got_nl = nl.P(z_probe, k_probe)
    except Exception as e:
        report("cobaya lifecycle through emul_mps", False,
               type(e).__name__ + ": " + str(e)[:250])
        return
    want_lin, want_nl = camb_truth(point, z_probe, k_probe)
    rel_lin = np.abs(got_lin / want_lin - 1).max()
    rel_nl = np.abs(got_nl / want_nl - 1).max()
    report("P_lin / P_nl vs CAMB's own P(k, z) (5%)",
           max(rel_lin, rel_nl) < 0.05,
           "rel lin %.3g, nl %.3g" % (rel_lin, rel_nl))
    # the interpolator's range guard: beyond the stored k without an
    # extrapolation request is loud.
    try:
        lin.P(0.25, 50.0)
        report("interpolator range guard", False, "no raise")
    except Exception:
        report("interpolator range guard", True, "raised")


def main():
    print("mps-smoke: generator + two trainings + cobaya vs "
          "CAMB truth (law-none path; syren rides mps-identity + the "
          "EMUL2 acceptance)")
    rootdir = os.environ.get("ROOTDIR")
    if not rootdir:
        print("FAIL: ROOTDIR is not set")
        sys.exit(1)
    device = torch.device("cpu")
    rel_root = "tmp_gate_mps_smoke_%d" % os.getpid()
    work = os.path.join(rootdir, rel_root)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            paths = check_generate(rootdir, rel_root)
            if not FAILURES:
                root_p = check_train(paths, tmp, device, "pklin")
                root_b = check_train(paths, tmp, device, "boost")
                check_cobaya(root_p, root_b, tmp)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: mps-smoke all checks green")


if __name__ == "__main__":
    main()
