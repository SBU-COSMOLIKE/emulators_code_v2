#!/usr/bin/env python3
"""scalar-smoke gate (SPE-B): a real scalar emulator end to end, on a fixture.

It writes a tiny fixture parameter chain (a .txt + its getdist .paramnames
sidecar) whose only output column is an EXACTLY-derivable target,
omegamh2 = omegam * (H0/100)^2, computed from each row's own H0 / omegam. A
2-input (H0, omegam) -> 1-output (omegamh2) map is deterministic and smooth,
so a 2-epoch train must already collapse the validation error. Then it saves
the emulator, rebuilds it, and checks that predict reproduces the analytic
omegamh2 at a test point; finally it runs a cobaya `evaluate` through
emul_scalars and confirms the same derived value comes back.

Needs torch (the train) and, for the evaluate leg, cobaya + a real ROOTDIR,
so it is a board gate. The fixture generation is pure numpy (Mac-checkable).
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from emulator.experiment import EmulatorExperiment
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor

FAILURES = []
IN_NAMES = ["H0", "omegam"]
OUT_NAME = "omegamh2"


def report(label, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


def omegamh2(h0, omegam):
    """The exactly-derivable target: omega_m h^2 = omegam * (H0/100)^2."""
    return omegam * (h0 / 100.0) ** 2


def write_fixture(stem, n_rows, seed):
    """Write <stem>.1.txt + <stem>.paramnames for a scalar training chain.

    Columns: weight, minuslogpost, H0, omegam, omegamh2, minusloglike. The
    getdist .paramnames names the parameter columns (H0, omegam sampled;
    omegamh2 derived, marked with a trailing '*'), so _scalar_columns finds
    the output column by name and check_paramnames pins the sampled block.

    Returns:
      the number of rows written.
    """
    g = np.random.default_rng(seed)
    h0 = g.normal(70.0, 3.0, size=n_rows)
    om = g.normal(0.3, 0.02, size=n_rows)
    target = omegamh2(h0, om)
    weight = np.ones(n_rows)
    zero = np.zeros(n_rows)
    cols = np.column_stack([weight, zero, h0, om, target, zero])
    np.savetxt(stem + ".1.txt", cols)
    with open(stem + ".paramnames", "w") as f:
        f.write("H0\t H_0\n")
        f.write("omegam\t \\Omega_m\n")
        f.write("omegamh2*\t \\Omega_m h^2\n")
    return n_rows


def write_covmat(path, seed):
    """Write the input (H0, omegam) covmat (header + a diagonal-ish SPD)."""
    g = np.random.default_rng(seed)
    a = g.standard_normal((2, 2))
    cov = a @ a.T + 2.0 * np.eye(2)
    with open(path, "w") as f:
        f.write("# " + " ".join(IN_NAMES) + "\n")
        for row in cov:
            f.write(" ".join(repr(float(x)) for x in row) + "\n")


def build_cfg(tmp, n_train, n_val):
    """The scalar training config pointing at the fixture files."""
    return {
        "data": {
            "train_params": os.path.join(tmp, "train.1.txt"),
            "val_params":   os.path.join(tmp, "val.1.txt"),
            "train_covmat": os.path.join(tmp, "params.covmat"),
            "outputs":      [OUT_NAME],
            "n_train":      n_train,
            "n_val":        n_val,
            "split_seed":   0,
        },
        # the full block set build_run_specs requires (model / optimizer /
        # lr / scheduler / trim / focus are plain subscripts there, no
        # code defaults — D-SPE2-7); the shape mirrors the proven-green
        # transfer-smoke-config.yaml, trim / focus zeroed.
        "train_args": {
            "nepochs": 2,
            "bs": 128,
            "model": {"name": "resmlp",
                      "mlp": {"width": 32, "n_blocks": 2}},
            "loss": {"mode": "sqrt"},
            "optimizer": {"weight_decay": 0.0},
            "lr": {"lr_base": 0.01, "bs_base": 128.0, "warmup_epochs": 0},
            "scheduler": {"mode": "min", "patience": 10, "factor": 0.8},
            "trim": {"start": 0.0, "end": 0.0, "hold_epochs": 0,
                     "anneal_epochs": 1, "shape": "cosine"},
            "focus": {"start": 0.0, "end": 0.0, "hold_epochs": 0,
                      "anneal_epochs": 1, "shape": "linear",
                      "kappa": 0.15},
        },
    }


def check_train_and_predict(tmp, device):
    """Train 2 epochs, save, rebuild, and check the analytic target.

    Returns the saved path root (for the cobaya evaluate leg).
    """
    write_fixture(os.path.join(tmp, "train"), 4000, seed=1)
    write_fixture(os.path.join(tmp, "val"), 1000, seed=2)
    write_covmat(os.path.join(tmp, "params.covmat"), seed=3)
    cfg = build_cfg(tmp, n_train=4000, n_val=1000)

    exp = EmulatorExperiment.from_config(cfg, device=device, quiet=True)
    model, train_losses, medians, means, fracs = exp.run()
    # a deterministic smooth 2->1 map: the validation median chi2 must be
    # small after 2 epochs (val collapses). D-SPE2-5: the bar is 0.3, below
    # the mean-predictor's median standardized chi2 (0.455, the median of a
    # chi-square-1), so a dead network that only learns the target mean fails.
    best_median = min(float(m) for m in medians)
    report("val collapses on the deterministic map (median chi2 small)",
           best_median < 0.3, "best val median = %.3g" % best_median)

    root = os.path.join(tmp, "emul_scalar_smoke")
    save_emulator(path_root=root, model=model,
                  param_geometry=exp.pgeom, geometry=exp.geom, config=cfg,
                  histories={"train_losses": train_losses,
                             "val_medians": medians, "val_means": means,
                             "val_fracs": fracs, "thresholds": exp.thresholds},
                  train_args=exp.train_args, pce=None, pce_form=None,
                  resolved_train=exp.resolved_train,
                  resolved_model=exp.resolved_model, transfer_base=None,
                  attrs={"outputs": OUT_NAME})

    # rebuild + predict at a test point; the emulated omegamh2 must track the
    # analytic value (the map is exact, so a trained emulator is close).
    # D-SPE2-5: OFF the fixture mean (one sigma out in each input, still
    # in-distribution), so a network that only learned the target mean is
    # 13.7% off and fails the 5% bar, while a trained one passes.
    pred = EmulatorPredictor(root, device, compile_model=False)
    h0_t, om_t = 73.0, 0.32
    got = pred.predict({"H0": h0_t, "omegam": om_t})[OUT_NAME]
    want = omegamh2(h0_t, om_t)
    rel = abs(got - want) / want
    report("predict reproduces the analytic omegamh2 at a test point",
           rel < 0.05, "got %.5f want %.5f (rel %.3g)" % (got, want, rel))
    check_diagnostics(exp, model, tmp)
    return root


def check_diagnostics(exp, model, tmp):
    """The D-CM9 scalar diagnostics leg: 3 pages build + the PDF lands."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from emulator.diagnostics import scalar_output_diagnostic
        from emulator.plotting import _scalar_pages, plot_diagnostics
        import matplotlib.pyplot as plt
        sc = scalar_output_diagnostic(model=model,
                                      param_geometry=exp.pgeom,
                                      chi2fn=exp.chi2fn,
                                      val_set=exp.val_set,
                                      device=exp.device)
        figs = _scalar_pages(sc)
        n_pages = len(figs)
        for f in figs:
            plt.close(f)
        pdf = os.path.join(tmp, "scalar_diag.pdf")
        plot_diagnostics(train_losses=[0.1], medians=[0.1], means=[0.1],
                         fracs=[torch.tensor([0.5, 0.4, 0.3, 0.2])],
                         thresholds=exp.thresholds,
                         coverage={"knn_dist": np.ones(4),
                                   "dchi2": np.ones(4), "k_nn": 2},
                         scalar=sc, savepath=pdf)
        ok = (n_pages == 3 and os.path.isfile(pdf)
              and os.path.getsize(pdf) > 10000)
        report("D-CM9 diagnostics: 3 scalar pages + the PDF lands", ok,
               "%d pages, %d bytes" % (n_pages,
                                       os.path.getsize(pdf)
                                       if os.path.isfile(pdf) else 0))
    except Exception as e:
        report("D-CM9 diagnostics: 3 scalar pages + the PDF lands", False,
               type(e).__name__ + ": " + str(e)[:200])


def check_cobaya_evaluate(tmp, root):
    """Run a cobaya `evaluate` through emul_scalars and read back omegamh2.

    Writes a minimal evaluate YAML (theory: emul_scalars over the saved root;
    an external-lambda likelihood that consumes omegamh2, so cobaya must ask
    the theory for it; omegamh2 declared derived) and runs cobaya-run. The
    derived omegamh2 lands in the run's <root>.1.txt; check it against the
    analytic value at the evaluated point. Board only (needs cobaya).
    """
    try:
        import cobaya  # noqa: F401
    except Exception as e:
        report("cobaya evaluate through emul_scalars", False,
               "cobaya not importable: " + str(e))
        return
    cobaya_dir = str(Path(__file__).resolve().parents[2] / "cobaya_theory")
    out_root = os.path.join(tmp, "evaluate", "scalar_eval")
    os.makedirs(os.path.dirname(out_root), exist_ok=True)
    # D-SPE2-5: off the fixture mean (same point as the predict leg), so the
    # evaluate leg also fails a mean-only network.
    h0_t, om_t = 73.0, 0.32
    want = omegamh2(h0_t, om_t)
    # D-SPE2-8: mirror the PROVEN cobaya-adapter-evaluate.yaml shape —
    # sampled params with priors, the point pinned by the evaluate
    # sampler's override (never value:-fixed params: with zero sampled
    # dimensions the run left no readable chain, board run 3's got-None).
    yaml_text = (
        "stop_at_error: True\n"
        "force: True\n"
        "theory:\n"
        "  emul_scalars:\n"
        "    python_path: " + cobaya_dir + "\n"
        "    stop_at_error: True\n"
        "    extra_args:\n"
        "      device: cpu\n"
        "      emulators:\n"
        "        - " + root + "\n"
        # the lambda's argument name is how a cobaya external likelihood
        # declares its input params; omegamh2 then resolves from the
        # theory's provides. No separate requires key (D-SPE2-6b): the
        # signature is the documented mechanism.
        "likelihood:\n"
        "  test_like:\n"
        "    external: 'lambda omegamh2: 0.0'\n"
        "params:\n"
        "  H0:\n"
        "    prior:\n"
        "      min: 55.0\n"
        "      max: 91.0\n"
        "    ref: 70.0\n"
        "    proposal: 1.0\n"
        "  omegam:\n"
        "    prior:\n"
        "      min: 0.1\n"
        "      max: 0.9\n"
        "    ref: 0.3\n"
        "    proposal: 0.01\n"
        "  omegamh2:\n"
        "    derived: True\n"
        "sampler:\n"
        "  evaluate:\n"
        "    N: 1\n"
        "    override:\n"
        "      H0: " + repr(h0_t) + "\n"
        "      omegam: " + repr(om_t) + "\n"
        "output: " + out_root + "\n")
    yaml_path = os.path.join(tmp, "scalar_evaluate.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_text)
    proc = subprocess.run([sys.executable, "-m", "cobaya", "run",
                           yaml_path, "-f"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        report("cobaya evaluate through emul_scalars", False,
               "cobaya-run rc=%d: %s" % (proc.returncode,
                                         proc.stderr.strip()[-300:]))
        return
    # D-SPE2-9: an evaluate run writes NO .paramnames sidecar (board run 4's
    # diag: only the .1.txt + input/updated yamls land), so read the value
    # from what the run provably produces. Primary: the evaluate sampler's
    # own "Derived params:" stdout block (format in evidence from run 4).
    # Secondary: the chain's header row names its columns directly (no
    # +2 offset — weight / minuslogpost are named there too).
    got, cols = None, []
    tail_at = proc.stdout.find("Derived params:")
    if tail_at >= 0:
        m = re.search(r"\b" + re.escape(OUT_NAME) + r"\s*=\s*([0-9eE+.-]+)",
                      proc.stdout[tail_at:])
        if m:
            got = float(m.group(1))
    txt = out_root + ".1.txt"
    if got is None and os.path.exists(txt):
        with open(txt) as fh:
            head = fh.readline()
        if head.startswith("#"):
            cols = head[1:].split()
            row = np.loadtxt(txt).reshape(-1)
            if OUT_NAME in cols:
                got = float(row[cols.index(OUT_NAME)])
    okval = got is not None and abs(got - want) / want < 0.05
    report("cobaya evaluate through emul_scalars returns omegamh2",
           okval, "got %s want %.5f" % (got, want))
    if got is None:
        # self-diagnosis (D-SPE2-8): a got-None red must name its own cause
        # in the log — which files cobaya wrote, which columns the chain
        # carries, and the run's stdout tail — so the next delta needs no
        # extra board round trip.
        out_dir = os.path.dirname(out_root)
        print("  [diag] output dir listing:", sorted(os.listdir(out_dir))
              if os.path.isdir(out_dir) else "MISSING")
        print("  [diag] chain columns:", cols or "no .paramnames")
        print("  [diag] cobaya stdout tail:",
              proc.stdout.strip()[-400:] or "(empty)")


def main():
    """Run the scalar-smoke checks and exit non-zero on any failure."""
    print("scalar-smoke (SPE-B): fixture train + predict + cobaya evaluate")
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        root = check_train_and_predict(tmp, device)
        check_cobaya_evaluate(tmp, root)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: scalar-smoke all checks green")


if __name__ == "__main__":
    main()
