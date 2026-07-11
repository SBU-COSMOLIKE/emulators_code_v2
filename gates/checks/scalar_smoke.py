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
        "train_args": {
            "nepochs": 2,
            "bs": 128,
            "model": {"name": "resmlp",
                      "mlp": {"width": 32, "n_blocks": 2}},
            "lr": {"lr_base": 0.01, "bs_base": 128.0, "warmup_epochs": 0},
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
    return root


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
    yaml_text = (
        "stop_at_error: True\n"
        "theory:\n"
        "  emul_scalars:\n"
        "    python_path: " + cobaya_dir + "\n"
        "    extra_args:\n"
        "      device: cpu\n"
        "      emulators:\n"
        "        - " + root + "\n"
        "likelihood:\n"
        "  test_like:\n"
        "    external: 'lambda omegamh2: 0.0'\n"
        "    requires: [omegamh2]\n"
        "params:\n"
        "  H0:\n    value: " + repr(h0_t) + "\n"
        "  omegam:\n    value: " + repr(om_t) + "\n"
        "  omegamh2:\n    derived: True\n"
        "output: " + out_root + "\n"
        "sampler:\n  evaluate:\n")
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
    # the evaluate writes a one-row <out_root>.1.txt; the derived omegamh2 is
    # among its columns (named by <out_root>.paramnames).
    txt = out_root + ".1.txt"
    names_file = out_root + ".paramnames"
    ok = os.path.exists(txt) and os.path.exists(names_file)
    got = None
    if ok:
        cols = [ln.split()[0].rstrip("*") for ln in open(names_file)
                if ln.strip()]
        row = np.loadtxt(txt).reshape(-1)
        if OUT_NAME in cols:
            got = float(row[2 + cols.index(OUT_NAME)])
    okval = got is not None and abs(got - want) / want < 0.05
    report("cobaya evaluate through emul_scalars returns omegamh2",
           okval, "got %s want %.5f" % (got, want))


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
