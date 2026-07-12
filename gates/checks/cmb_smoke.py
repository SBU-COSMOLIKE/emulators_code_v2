#!/usr/bin/env python3
"""cmb-smoke gate (CME-B): the CMB emulator end to end, on real CAMB.

The full pipeline the unit ships, in one run (board only; needs torch,
cobaya, and a compiled CAMB under $ROOTDIR — budget several minutes, the
two tiny dumps are ~400 serial CAMB calls at low accuracy):

  1  dataset_generator_cmb.py writes a TINY training dump (200 rows,
     l = 2..350, probe cmblensed, uniform sampling over As / tau / omch2 —
     As sampled LINEARLY, so the as_exp2tau law reads a raw amplitude
     column) and a second dump for validation. This leg also gates the
     generator itself (D-CM3-A): four per-spectrum dv files + the chain
     sidecars must land with the documented names.
  2  compute_cmb_covariance.py (D-CM11) writes the Gaussian covariance
     .npz on the same fiducial LCDM (zero noise, fsky 1) — the training
     path consumes a REAL script-produced file, never from_fiducial.
  3  a data.cmb training run (spectrum tt, amplitude_law as_exp2tau)
     trains a small ResMLP; the collapse bar is RELATIVE to the staged
     mean predictor (best val median < 0.5x its median chi2), so a dead
     network that only learns the training mean fails the gate
     (the D-SPE2-5 rule, applied off-fiducial by construction: uniform
     sampling has no fiducial row).
  4  the saved artifact serves Cl through the REAL cobaya lifecycle:
     get_model over theory emul_cmb + a Cl-requiring add_requirements,
     logposterior at a test point, provider.get_Cl == the predictor's own
     output at that point (the adapter adds nothing to the physics).
  5  the D-CM9 diagnostics leg: cmb_residual_diagnostic + the CMB pages
     build without exception and the PDF lands non-trivially sized.
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
from emulator.inference import EmulatorPredictor

FAILURES = []
REPO = Path(__file__).resolve().parents[2]

LMAX = 350          # lrange upper edge = the covariance lmax
NROWS = 200         # generator floor (--nparams >= 200)


def report(label, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


def gen_yaml():
    """The generator YAML: low-accuracy CAMB, three sampled params.

    As is sampled LINEARLY (the endorsed decision 5 ruling: the
    as_exp2tau law reads a raw amplitude column, so the dump carries As
    itself, never logA).
    """
    return (
        "likelihood:\n"
        "  one: null\n"
        "theory:\n"
        "  camb:\n"
        "    path: ./external_modules/code/CAMB\n"
        "    extra_args:\n"
        "      lmax: 500\n"
        "      AccuracyBoost: 0.7\n"
        "      lens_potential_accuracy: 1\n"
        "params:\n"
        "  As:\n"
        "    prior:\n"
        "      min: 1.8e-9\n"
        "      max: 2.4e-9\n"
        "    ref: 2.1e-9\n"
        "    proposal: 1.0e-11\n"
        "    latex: A_s\n"
        "  tau:\n"
        "    prior:\n"
        "      min: 0.03\n"
        "      max: 0.09\n"
        "    ref: 0.055\n"
        "    proposal: 0.002\n"
        "    latex: \\tau\n"
        "  omch2:\n"
        "    prior:\n"
        "      min: 0.11\n"
        "      max: 0.13\n"
        "    ref: 0.12\n"
        "    proposal: 0.001\n"
        "    latex: \\Omega_c h^2\n"
        "  ns:\n"
        "    value: 0.965\n"
        "  H0:\n"
        "    value: 67.36\n"
        "  ombh2:\n"
        "    value: 0.02237\n"
        "  mnu:\n"
        "    value: 0.06\n"
        "train_args:\n"
        "  probe: cmblensed\n"
        "  ord: [['As', 'tau', 'omch2']]\n"
        "  lrange: [2, " + str(LMAX) + "]\n")


def cov_yaml():
    """The D-CM11 covariance YAML: fixed fiducial LCDM, zero noise.

    The params block follows example_yamls/cmb_covariance_lcdm.yaml
    EXACTLY in both conventions the script validates loudly: PLAIN
    NUMBERS (never cobaya {value: X} mappings — board run 1 caught
    that) and the script's OWN parameter names, omegabh2 / omegach2,
    not CAMB's ombh2 / omch2 (board run 3 caught that). The lesson:
    mirror the shipped example, never re-type its keys from memory.
    """
    return (
        "theory:\n"
        "  camb:\n"
        "    path: ./external_modules/code/CAMB\n"
        "    extra_args:\n"
        "      lmax: 500\n"
        "      AccuracyBoost: 0.7\n"
        "      lens_potential_accuracy: 1\n"
        "params:\n"
        "  As:       2.1e-9\n"
        "  ns:       0.965\n"
        "  H0:       67.36\n"
        "  omegabh2: 0.02237\n"
        "  omegach2: 0.12\n"
        "  tau:      0.055\n"
        "  mnu:      0.06\n"
        "cov_args:\n"
        "  lmax: " + str(LMAX) + "\n"
        "  fsky: 1.0\n"
        "  noise:\n"
        "    delta_tt: 0.0\n"
        "    delta_ee: 0.0\n"
        "    delta_te: 0.0\n"
        "    beam_fwhm: 1.0\n")


def run_tool(script, arglist, rootdir):
    """Run one compute_data_vectors tool as a subprocess under $ROOTDIR."""
    cmd = [sys.executable, str(REPO / "compute_data_vectors" / script)]
    cmd.extend(arglist)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=rootdir)
    return proc


def check_generate(rootdir, rel_root):
    """Legs 1 + 2: the CMB dump generator + the covariance script."""
    emul_dir = os.path.join(rootdir, rel_root, "emul")
    os.makedirs(emul_dir, exist_ok=True)
    with open(os.path.join(emul_dir, "gen.yaml"), "w") as f:
        f.write(gen_yaml())
    with open(os.path.join(emul_dir, "cov.yaml"), "w") as f:
        f.write(cov_yaml())

    chains = os.path.join(rootdir, rel_root, "chains")
    out = {}
    for tag in ("train", "val"):
        proc = run_tool(
            "dataset_generator_cmb.py",
            ["--root", rel_root, "--fileroot", "emul", "--yaml", "gen.yaml",
             "--datavsfile", "dvs_" + tag, "--paramfile", "params_" + tag,
             "--failfile", "failed_" + tag, "--chain", "0",
             "--nparams", str(NROWS), "--unif", "1", "--temp", "2"],
            rootdir)
        stem = os.path.join(chains, "params_%s_cmblensed_unifs" % tag)
        dv = os.path.join(chains, "dvs_%s_cmblensed_unifs_tt.npy" % tag)
        files_ok = (os.path.isfile(stem + ".1.txt")
                    and os.path.isfile(stem + ".covmat")
                    and os.path.isfile(stem + ".paramnames")
                    and all(os.path.isfile(os.path.join(
                        chains, "dvs_%s_cmblensed_unifs_%s.npy" % (tag, s)))
                        for s in ("tt", "te", "ee", "pp")))
        detail = "rc=%d" % proc.returncode
        if not files_ok:
            detail += " missing outputs; stderr tail: " \
                      + proc.stderr.strip()[-200:]
        report("generator dump (%s): four dv files + sidecars" % tag,
               proc.returncode == 0 and files_ok, detail)
        if files_ok:
            arr = np.load(dv)
            pp = np.load(dv.replace("_tt", "_pp"))
            report("dump shape + filled phiphi (%s)" % tag,
                   arr.shape == (NROWS, LMAX - 1)
                   and float(np.abs(pp).max()) > 0.0,
                   "tt %s, |pp|max %.2e" % (arr.shape, np.abs(pp).max()))
        out[tag] = {"params": stem, "dv": dv}

    proc = run_tool(
        "compute_cmb_covariance.py",
        ["--root", rel_root, "--fileroot", "emul", "--yaml", "cov.yaml",
         "--output", "cmbcov"], rootdir)
    npz = os.path.join(chains, "cmbcov.npz")
    ok = proc.returncode == 0 and os.path.isfile(npz)
    detail = "rc=%d" % proc.returncode
    if not ok:
        detail += " stderr tail: " + proc.stderr.strip()[-300:]
    else:
        cov = np.load(npz, allow_pickle=False)
        ok = (np.array_equal(cov["ell"], np.arange(2, LMAX + 1))
              and (cov["sigma_tt"] > 0).all())
        detail = "ell 2..%d, sigma_tt > 0" % LMAX
    report("D-CM11 covariance .npz (Gaussian, zero noise)", ok, detail)
    out["cov"] = npz
    return out


def build_cfg(paths):
    """The data.cmb training config over the generated fixture files."""
    return {
        "data": {
            "cmb": {"spectrum": "tt",
                    "covariance": paths["cov"],
                    "amplitude_law": "as_exp2tau",
                    "as_name": "As",
                    "tau_name": "tau"},
            "train_dv":     paths["train"]["dv"],
            "val_dv":       paths["val"]["dv"],
            "train_params": paths["train"]["params"] + ".1.txt",
            "val_params":   paths["val"]["params"] + ".1.txt",
            "train_covmat": paths["train"]["params"] + ".covmat",
            "n_train":      180,
            "n_val":        180,
            "split_seed":   0,
        },
        # the full block set build_run_specs requires (the D-SPE2-7
        # subscript census); shape mirrors the proven scalar-smoke config.
        "train_args": {
            "nepochs": 40,
            "bs": 64,
            "model": {"name": "resmlp",
                      "mlp": {"width": 32, "n_blocks": 2}},
            "loss": {"mode": "sqrt",
                     "roughness": {"lam": 0.1, "period_cut": 50}},
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


def check_train(paths, tmp, device):
    """Leg 3: train on the fixture; the dead-network-relative bar."""
    cfg = build_cfg(paths)
    exp = EmulatorExperiment.from_config(cfg, device=device, quiet=True)
    model, train_losses, medians, means, fracs = exp.run()

    # the mean predictor's median chi2 on the staged val rows: whitened
    # pred = 0 (the geometry's center IS the training-mean target), so a
    # network that learned nothing beyond the mean scores this. The bar
    # is relative: a real 40-epoch train must at least halve it.
    idx = np.sort(exp.val_set["idx"])
    dv_rows = np.asarray(exp.val_set["dv"][idx])
    C_rows  = np.asarray(exp.val_set["C"][idx])
    dv = torch.from_numpy(dv_rows).float().to(device)
    C  = torch.from_numpy(C_rows).float().to(device)
    x_enc = exp.pgeom.encode(C)
    if getattr(exp.chi2fn, "needs_params", False):
        tw = exp.chi2fn.encode(dv, x_enc)
    else:
        tw = exp.chi2fn.encode(dv)
    c_mean = exp.chi2fn.chi2(pred=torch.zeros_like(tw), target=tw)
    mean_median = float(c_mean.median())
    best_median = min(float(m) for m in medians)
    report("val collapses below the mean predictor (relative bar)",
           best_median < 0.5 * mean_median,
           "best %.3g vs mean-predictor %.3g" % (best_median, mean_median))

    root = os.path.join(tmp, "emul_cmb_smoke")
    save_emulator(path_root=root, model=model,
                  param_geometry=exp.pgeom, geometry=exp.geom, config=cfg,
                  histories={"train_losses": train_losses,
                             "val_medians": medians,
                             "val_means": means,
                             "val_fracs": fracs,
                             "thresholds": exp.thresholds},
                  train_args=exp.train_args, pce=None, pce_form=None,
                  resolved_train=exp.resolved_train,
                  resolved_model=exp.resolved_model, transfer_base=None,
                  attrs={"rescale": "none", "spectrum": "tt"})
    return exp, model, root


def check_cobaya(root, device):
    """Leg 4: the real cobaya lifecycle serves the emulator's own Cl."""
    try:
        from cobaya.model import get_model
    except Exception as e:
        report("cobaya lifecycle through emul_cmb", False,
               "cobaya not importable: " + str(e))
        return
    info = {
        "likelihood": {"one": None},
        "theory": {"emul_cmb": {
            "python_path": str(REPO / "cobaya_theory"),
            "extra_args": {"device": "cpu", "emulators": [root]}}},
        "params": {
            "As":    {"prior": {"min": 1.8e-9, "max": 2.4e-9},
                      "ref": 2.1e-9,
                      "proposal": 1e-11},
            "tau":   {"prior": {"min": 0.03, "max": 0.09},
                      "ref": 0.055,
                      "proposal": 0.002},
            "omch2": {"prior": {"min": 0.11, "max": 0.13},
                      "ref": 0.12,
                      "proposal": 0.001},
        },
    }
    try:
        model = get_model(info)
        model.add_requirements({"Cl": {"tt": LMAX}})
        point = {"As": 2.05e-9,
                 "tau": 0.06,
                 "omch2": 0.121}
        model.logposterior(point)
        cl = model.provider.get_Cl(ell_factor=False, units="muK2")
    except Exception as e:
        report("cobaya lifecycle through emul_cmb", False,
               type(e).__name__ + ": " + str(e)[:200])
        return
    pred = EmulatorPredictor(root, device, compile_model=False)
    own = pred.predict(point)
    served = np.asarray(cl["tt"], dtype="float64")
    ok = (served.shape[0] >= LMAX + 1
          and (served[:2] == 0).all()
          and np.allclose(served[2:LMAX + 1], own, rtol=1e-6, atol=0.0)
          and np.isfinite(served).all())
    report("provider.get_Cl equals the predictor's own C_ell",
           ok, "max rel d = %.2e" % (np.abs(
               served[2:LMAX + 1] - own) / np.abs(own)).max())


def check_diagnostics(exp, model, tmp):
    """Leg 5 (D-CM9): the CMB pages build and the PDF lands."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from emulator.diagnostics import cmb_residual_diagnostic
        from emulator.plotting import _cmb_pages, plot_diagnostics
        import matplotlib.pyplot as plt
        cmb = cmb_residual_diagnostic(model=model,
                                      param_geometry=exp.pgeom,
                                      chi2fn=exp.chi2fn,
                                      val_set=exp.val_set,
                                      device=exp.device)
        figs = _cmb_pages(cmb)
        n_pages = len(figs)
        for f in figs:
            plt.close(f)
        pdf = os.path.join(tmp, "cmb_diag.pdf")
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
                         cmb=cmb, savepath=pdf)
        ok = (n_pages == 2 and os.path.isfile(pdf)
              and os.path.getsize(pdf) > 10000)
        report("D-CM9 diagnostics: 2 CMB pages + the PDF lands", ok,
               "%d pages, %d bytes" % (n_pages,
                                       os.path.getsize(pdf)
                                       if os.path.isfile(pdf) else 0))
    except Exception as e:
        report("D-CM9 diagnostics: 2 CMB pages + the PDF lands", False,
               type(e).__name__ + ": " + str(e)[:200])


def main():
    """Run the cmb-smoke pipeline; exit non-zero on any failure."""
    print("cmb-smoke (CME-B): generator + covariance + train + cobaya "
          "+ diagnostics (real CAMB; several minutes)")
    rootdir = os.environ.get("ROOTDIR")
    if not rootdir:
        print("FAIL: ROOTDIR is not set (the generator and the covariance "
              "script resolve their paths under it)")
        sys.exit(1)
    device = torch.device("cpu")
    rel_root = "tmp_gate_cmb_smoke_%d" % os.getpid()
    work = os.path.join(rootdir, rel_root)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            paths = check_generate(rootdir, rel_root)
            if not FAILURES:
                exp, model, root = check_train(paths, tmp, device)
                check_cobaya(root, device)
                check_diagnostics(exp, model, tmp)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: cmb-smoke all checks green")


if __name__ == "__main__":
    main()
