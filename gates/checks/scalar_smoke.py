#!/usr/bin/env python3
"""scalar-smoke gate: a real scalar emulator end to end, on a fixture.

It writes a tiny fixture parameter chain (a .txt + its getdist .paramnames
sidecar) whose only output column is an EXACTLY-derivable target,
omegamh2 = omegam * (H0/100)^2, computed from each row's own H0 / omegam.
Training has a fixed two-epoch smoke budget.  The gate measures the trained
validation error against the mean-predictor error on the same disjoint rows,
so the budget does not imply a claim about why the network converges.  It then
saves the emulator, rebuilds it, and checks that predict reproduces the
analytic omegamh2 at a test point.  Finally it runs a cobaya `evaluate`
through emul_scalars and confirms the same derived value comes back.

Needs torch (the train) and, for the evaluate leg, cobaya + a real ROOTDIR,
so it is a board gate. The fixture generation is pure numpy (Mac-checkable).
"""

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from emulator.experiment import EmulatorExperiment
from emulator.data_staging import load_source
from emulator.training import ordinary_median
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor

FAILURES = []
IN_NAMES = ["H0", "omegam"]
OUT_NAME = "omegamh2"
TRAIN_GENERATOR_SEED = 1234
VAL_GENERATOR_SEED = 5678
SPLIT_SEED = 0

# The bars are calibrated on the exact disjoint fixture above, with the model,
# epoch count, and training settings below left unchanged.  The collapse bar
# is recomputed from the fixture's independent mean predictor on every run.
# The accuracy bar uses the measured deterministic two-epoch error recorded in
# the durable note, multiplied by the stated 1.5 safety margin.  Keeping the
# measured error separate from the current run prevents a regression from
# loosening its own acceptance bar.
COLLAPSE_BAR_FRACTION = 0.5
CALIBRATED_TWO_EPOCH_REL_ERROR = 0.07459584140804423
ACCURACY_BAR_MARGIN = 1.5


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
    GetDist .paramnames file names the parameter columns. H0 and omegam are
    sampled. omegamh2 is derived and marked with a trailing '*', so
    _scalar_columns finds the output column by name and check_paramnames pins
    the sampled block.

    Returns:
      the number of rows written.
    """
    g = np.random.default_rng(seed)
    # One generator call produces one pair per row.  The first random draw
    # becomes that row's H0 and the second becomes its Omega_m.  This order is
    # load-bearing for the negative control below: two files made with the same
    # seed have identical prefix rows, even when the files have different row
    # counts.  Separate one-dimensional calls would consume all H0 draws first
    # and would hide the same-seed error by pairing them with different
    # Omega_m draws.
    physical_rows = g.normal(
        loc=np.asarray([70.0, 0.3]),
        scale=np.asarray([3.0, 0.02]),
        size=(n_rows, 2),
    )
    h0 = physical_rows[:, 0]
    om = physical_rows[:, 1]
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
            row_text = []
            for value in row:
                row_text.append(repr(float(value)))
            f.write(" ".join(row_text) + "\n")


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
            "split_seed":   SPLIT_SEED,
        },
        # the full block set build_run_specs requires (model / optimizer /
        # lr / scheduler / trim / focus are plain subscripts there, no
        # code defaults). The shape mirrors the proven-green
        # transfer-smoke-config.yaml, trim / focus zeroed.
        "train_args": {
            "nepochs": 2,
            "bs": 128,
            "model": {"name": "resmlp",
                      "mlp": {"width": 32, "n_blocks": 2}},
            "loss": {"mode": "sqrt"},
            "optimizer": {"weight_decay": 0.0},
            "lr": {"lr_base": 0.01,
                   "bs_base": 128.0,
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


def _selected_parameter_and_target_rows(source):
    """Return staged rows in the order consumed by the first loader epoch.

    A staged source can use two coordinate systems.  ``source["C"]`` and
    ``source["dv"]`` hold the storage arrays.  ``source["idx"]`` holds the
    row numbers that walk those arrays in the seeded selection order.  NumPy
    indexing with that integer array makes an eager copy in loader order.

    Returns:
      ``(parameter_rows, target_rows)`` as two NumPy arrays with matching
      first dimensions.
    """
    storage_indices = np.asarray(source["idx"], dtype=np.int64)
    parameter_rows = np.asarray(source["C"])[storage_indices]
    target_rows = np.asarray(source["dv"])[storage_indices]
    return parameter_rows, target_rows


def _canonical_parameter_keys(parameter_rows):
    """Represent each physical parameter row as an exact float32 tuple.

    Training reads the text fixtures as float32.  The disjointness comparison
    therefore uses those consumed values rather than the higher-precision
    decimal text.  A Python tuple is hashable, which means it can be inserted
    into a set and compared with rows from the other file.  The explicit loop
    keeps the conversion visible to readers who are new to NumPy.
    """
    keys = []
    rows_float32 = np.asarray(parameter_rows, dtype=np.float32)
    for row in rows_float32:
        row_values = []
        for value in row:
            row_values.append(float(value))
        keys.append(tuple(row_values))
    return keys


def _count_aligned_scalar_rows(parameter_rows, target_rows):
    """Count rows whose stored target matches its own physical parameters.

    This reference spells out ``Omega_m * (H0 / 100)^2`` independently from
    ``write_fixture``.  The inputs and target were rounded to float32 when the
    staging code loaded the text file.  Five float32 machine epsilons allow
    only that representation-level rounding.  A target moved to another row
    differs by much more and fails the check.
    """
    if target_rows.ndim != 2 or target_rows.shape[1] != 1:
        return 0
    alignment_rtol = 5.0 * float(np.finfo(np.float32).eps)
    aligned = 0
    for row_number in range(parameter_rows.shape[0]):
        h0 = float(parameter_rows[row_number, 0])
        omega_m = float(parameter_rows[row_number, 1])
        expected = omega_m * (h0 / 100.0) ** 2
        observed = float(target_rows[row_number, 0])
        allowed_error = alignment_rtol * abs(expected)
        if abs(observed - expected) <= allowed_error:
            aligned += 1
    return aligned


def _require_disjoint_aligned_fixtures(
    train_source,
    val_source,
    train_generator_seed,
    val_generator_seed,
    split_seed,
):
    """Validate fixture identity before a model or optimizer is constructed.

    The generator seeds choose the physical cosmologies written to each file.
    ``split_seed`` only permutes rows inside each existing file.  It cannot
    turn two identical files into independent samples.  This function checks
    the physical float32 rows directly and raises before training when even
    one validation cosmology is present in the training source.
    """
    train_parameters, train_targets = _selected_parameter_and_target_rows(
        train_source
    )
    val_parameters, val_targets = _selected_parameter_and_target_rows(
        val_source
    )

    train_keys = _canonical_parameter_keys(train_parameters)
    val_keys = _canonical_parameter_keys(val_parameters)
    train_key_set = set(train_keys)
    val_key_set = set(val_keys)
    overlap_count = len(train_key_set.intersection(val_key_set))
    train_aligned = _count_aligned_scalar_rows(train_parameters, train_targets)
    val_aligned = _count_aligned_scalar_rows(val_parameters, val_targets)

    print(
        "fixture split: train generator seed %d, val generator seed %d, "
        "split seed %d, staged rows train %d / val %d, overlap %d"
        % (
            train_generator_seed,
            val_generator_seed,
            split_seed,
            len(train_keys),
            len(val_keys),
            overlap_count,
        )
    )
    print(
        "fixture row alignment: train %d/%d, val %d/%d"
        % (train_aligned, len(train_keys), val_aligned, len(val_keys))
    )

    if len(train_key_set) != len(train_keys):
        raise ValueError(
            "the scalar training fixture repeats a physical parameter row"
        )
    if len(val_key_set) != len(val_keys):
        raise ValueError(
            "the scalar validation fixture repeats a physical parameter row"
        )
    if train_aligned != len(train_keys) or val_aligned != len(val_keys):
        raise ValueError(
            "a scalar fixture target is not aligned with the H0 and omegam "
            "values on its own row"
        )
    if overlap_count != 0:
        raise ValueError(
            "the scalar smoke fixtures overlap on %d physical parameter "
            "row(s). Generator seeds choose cosmologies, while split_seed "
            "only reorders rows inside each file" % overlap_count
        )


def check_same_seed_refusal(tmp, device):
    """Prove that restoring the training seed to validation is refused."""
    same_seed_stem = os.path.join(tmp, "same_seed_val")
    write_fixture(
        same_seed_stem,
        n_rows=1000,
        seed=TRAIN_GENERATOR_SEED,
    )
    mutation_cfg = build_cfg(tmp, n_train=4000, n_val=1000)
    mutation_cfg["data"]["val_params"] = same_seed_stem + ".1.txt"
    mutation_exp = EmulatorExperiment.from_config(
        mutation_cfg,
        device=device,
        quiet=True,
    )
    mutation_exp.stage_train()
    mutation_exp.stage_val()
    try:
        _require_disjoint_aligned_fixtures(
            train_source=mutation_exp.train_set,
            val_source=mutation_exp.val_set,
            train_generator_seed=TRAIN_GENERATOR_SEED,
            val_generator_seed=TRAIN_GENERATOR_SEED,
            split_seed=SPLIT_SEED,
        )
    except ValueError as exc:
        detail = str(exc)
        report(
            "same-seed validation fixture is refused before training",
            "overlap" in detail,
            detail,
        )
        return
    report(
        "same-seed validation fixture is refused before training",
        False,
        "the overlap validator accepted the same generator seed",
    )


def _independent_window_selection(physical_rows, split_seed, n_keep):
    """Compute the tight-window survivor order without production helpers."""
    eligible = []
    for row_number, row in enumerate(physical_rows):
        h0 = float(row[0])
        omega_b = float(row[1])
        omega_m = float(row[2])
        omega_b_h2 = omega_b * (h0 / 100.0) ** 2
        omega_m_h2 = omega_m * (h0 / 100.0) ** 2
        inside_baryon_window = 0.020 < omega_b_h2 < 0.024
        inside_matter_window = 0.135 < omega_m_h2 < 0.155
        if inside_baryon_window and inside_matter_window:
            eligible.append(row_number)

    generator = torch.Generator().manual_seed(int(split_seed))
    shuffled_rows = torch.randperm(
        len(physical_rows),
        generator=generator,
    ).tolist()
    eligible_set = set(eligible)
    selected = []
    for row_number in shuffled_rows:
        if row_number in eligible_set:
            selected.append(row_number)
        if len(selected) == n_keep:
            break
    return eligible, selected


def _validate_window_banner(
    text,
    observed_rows,
    expected_selected_rows,
    eligible_count,
    raw_count,
    requested_count,
):
    """Compare a production banner and staged row identities with references."""
    matches = re.findall(r"used\s+(\d+)\s+of\s+(\d+)\s+cut rows", text)
    if len(matches) != 1:
        return False, "expected one cut-row banner, found %d" % len(matches)
    used_count = int(matches[0][0])
    banner_pool_count = int(matches[0][1])
    observed = []
    for value in observed_rows:
        observed.append(int(value))
    expected = []
    for value in expected_selected_rows:
        expected.append(int(value))

    count_order_is_valid = 0 <= used_count <= banner_pool_count
    count_values_match = (
        used_count == requested_count
        and used_count == len(observed)
        and banner_pool_count == eligible_count
    )
    fixture_has_real_shrinkage = used_count < banner_pool_count < raw_count
    row_identities_match = observed == expected
    ok = (
        count_order_is_valid
        and count_values_match
        and fixture_has_real_shrinkage
        and row_identities_match
    )
    detail = (
        "banner used %d of %d, expected %d of %d. Staged rows %s, "
        "expected rows %s"
        % (
            used_count,
            banner_pool_count,
            requested_count,
            eligible_count,
            observed,
            expected,
        )
    )
    return ok, detail


def check_parameter_window_banner(tmp):
    """Check cut counts and staged row identities against an independent mask."""
    physical_rows = np.asarray(
        [
            [70.0, 0.045, 0.300, 0.965],
            [70.0, 0.046, 0.310, 0.965],
            [70.0, 0.043, 0.280, 0.965],
            [70.0, 0.048, 0.290, 0.965],
            [70.0, 0.042, 0.315, 0.965],
            [70.0, 0.060, 0.300, 0.965],
            [70.0, 0.030, 0.300, 0.965],
            [70.0, 0.045, 0.400, 0.965],
        ],
        dtype=np.float32,
    )
    raw_count = physical_rows.shape[0]
    requested_count = 3
    window_split_seed = 19
    eligible, expected_selected = _independent_window_selection(
        physical_rows=physical_rows,
        split_seed=window_split_seed,
        n_keep=requested_count,
    )

    # load_source expects the GetDist layout: two bookkeeping columns, the
    # physical parameter block, then one trailing diagnostic column.  The
    # data-vector file stores the original row number, so its staged values
    # reveal exactly which disk rows the production selection chose.
    ones = np.ones((raw_count, 1), dtype=np.float32)
    zeros = np.zeros((raw_count, 1), dtype=np.float32)
    parameter_table = np.concatenate(
        [ones, zeros, physical_rows, zeros],
        axis=1,
    )
    params_path = os.path.join(tmp, "window_params.txt")
    dv_path = os.path.join(tmp, "window_dv.npy")
    np.savetxt(params_path, parameter_table)
    np.save(
        dv_path,
        np.arange(raw_count, dtype=np.float32).reshape(raw_count, 1),
    )

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        source = load_source(
            dv_path=dv_path,
            params_path=params_path,
            names=["H0", "omegab", "omegam", "ns"],
            omegabh2_hi=0.024,
            n_keep=requested_count,
            gen=torch.Generator().manual_seed(window_split_seed),
            ram_frac=1.0,
            with_means=False,
            verbose=True,
            omegabh2_lo=0.020,
            omegamh2_lo=0.135,
            omegamh2_hi=0.155,
        )
    banner_text = captured.getvalue()
    for line in banner_text.splitlines():
        if " cut rows" in line:
            print("parameter-window production banner:" + line)

    _, staged_targets = _selected_parameter_and_target_rows(source)
    observed_rows = staged_targets[:, 0]
    ok, detail = _validate_window_banner(
        text=banner_text,
        observed_rows=observed_rows,
        expected_selected_rows=expected_selected,
        eligible_count=len(eligible),
        raw_count=raw_count,
        requested_count=requested_count,
    )
    report(
        "parameter-window banner and staged row identities match an independent mask",
        ok,
        detail,
    )

    # This negative control changes both evidence channels.  The banner claims
    # a plausible one-of-one selection, while the staged row list names three
    # rows that the independent mask rejected.  A validator that checks only
    # the presence of the words "used N of P" would accept it.
    mutation_ok, mutation_detail = _validate_window_banner(
        text="used 1 of 1 cut rows",
        observed_rows=[7, 6, 5],
        expected_selected_rows=expected_selected,
        eligible_count=len(eligible),
        raw_count=raw_count,
        requested_count=requested_count,
    )
    report(
        "hardcoded cut banner with wrong staged rows is rejected",
        not mutation_ok,
        mutation_detail,
    )


def _mean_predictor_median(exp):
    """Measure the learned-nothing baseline on the staged validation rows.

    Scalar targets are standardized with the training mean as zero.  A model
    that always returns zero in network coordinates therefore predicts that
    mean for every cosmology.  This function sends the exact staged validation
    targets through the production geometry and scalar chi-squared metric,
    then takes the same ordinary median used by production evaluation.
    """
    _, target_rows = _selected_parameter_and_target_rows(exp.val_set)
    physical_targets = torch.from_numpy(np.asarray(target_rows)).float()
    physical_targets = physical_targets.to(exp.device)
    encoded_targets = exp.chi2fn.encode(physical_targets)
    encoded_mean_prediction = torch.zeros_like(encoded_targets)
    baseline_scores = exp.chi2fn.chi2(
        pred=encoded_mean_prediction,
        target=encoded_targets,
    )
    return ordinary_median(baseline_scores)


def _calibration_bars(mean_predictor_median):
    """Derive both acceptance bars from recorded, visible measurements."""
    collapse_bar = COLLAPSE_BAR_FRACTION * mean_predictor_median
    accuracy_bar = ACCURACY_BAR_MARGIN * CALIBRATED_TWO_EPOCH_REL_ERROR
    if not accuracy_bar < collapse_bar:
        raise ValueError(
            "the scalar-smoke accuracy bar must stay stricter than the "
            "collapse bar, got accuracy %.12g and collapse %.12g"
            % (accuracy_bar, collapse_bar)
        )
    return collapse_bar, accuracy_bar


def _check_dead_network_mutation(
    exp,
    mean_predictor_median,
    collapse_bar,
    accuracy_bar,
    physical_truth,
):
    """Prove that a network returning the training mean fails both bars."""
    mean_prediction = float(exp.geom.center[0].detach().cpu())
    dead_accuracy_error = abs(mean_prediction - physical_truth) / physical_truth
    dead_passes_collapse = mean_predictor_median < collapse_bar
    dead_passes_accuracy = dead_accuracy_error < accuracy_bar
    report(
        "mean-predictor mutation fails the collapse and accuracy bars",
        not dead_passes_collapse and not dead_passes_accuracy,
        "median %.12g vs collapse bar %.12g. Relative error %.12g vs "
        "accuracy bar %.12g"
        % (
            mean_predictor_median,
            collapse_bar,
            dead_accuracy_error,
            accuracy_bar,
        ),
    )


def check_train_and_predict(tmp, device):
    """Train 2 epochs, save, rebuild, and check the analytic target.

    Returns:
      ``(saved_path_root, accuracy_bar)`` for the Cobaya evaluate leg.  A
      fixture-integrity refusal returns ``(None, None)`` and skips training.
    """
    write_fixture(
        os.path.join(tmp, "train"),
        4000,
        seed=TRAIN_GENERATOR_SEED,
    )
    write_fixture(
        os.path.join(tmp, "val"),
        1000,
        seed=VAL_GENERATOR_SEED,
    )
    write_covmat(os.path.join(tmp, "params.covmat"), seed=3)
    cfg = build_cfg(tmp, n_train=4000, n_val=1000)

    exp = EmulatorExperiment.from_config(cfg, device=device, quiet=True)
    # Stage both files first, then authenticate their rows before geometry
    # construction or training.  This is the expanded form of exp.run():
    # run() normally performs stage_train, stage_val, build_geometry, and
    # train in that order.  Keeping those calls separate makes the refusal
    # boundary executable and visible in this gate.
    exp.stage_train()
    exp.stage_val()
    try:
        _require_disjoint_aligned_fixtures(
            train_source=exp.train_set,
            val_source=exp.val_set,
            train_generator_seed=TRAIN_GENERATOR_SEED,
            val_generator_seed=VAL_GENERATOR_SEED,
            split_seed=SPLIT_SEED,
        )
    except ValueError as exc:
        report(
            "training and validation fixture rows are aligned and disjoint",
            False,
            str(exc),
        )
        return None, None
    report(
        "training and validation fixture rows are aligned and disjoint",
        True,
        "4000 train rows, 1000 validation rows, zero physical-row overlap",
    )
    check_same_seed_refusal(tmp=tmp, device=device)
    check_parameter_window_banner(tmp=tmp)

    exp.build_geometry(train_set=exp.train_set)
    mean_predictor_median = _mean_predictor_median(exp)
    try:
        collapse_bar, accuracy_bar = _calibration_bars(mean_predictor_median)
    except ValueError as exc:
        report(
            "scalar-smoke calibration bars are properly ordered",
            False,
            str(exc),
        )
        return None, None
    model, train_losses, medians, means, fracs = exp.train()
    best_median = float("inf")
    for median in medians:
        median_value = float(median)
        if median_value < best_median:
            best_median = median_value
    report(
        "two-epoch validation median beats the measured mean predictor",
        best_median < collapse_bar,
        "trained median %.12g vs collapse bar %.12g"
        % (best_median, collapse_bar),
    )

    root = os.path.join(tmp, "emul_scalar_smoke")
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
                  attrs={"outputs": OUT_NAME})

    # Rebuild and predict at a test point.  The emulated omegamh2 must track the
    # analytic value (the map is exact, so a trained emulator is close).
    # The point is away from the fixture mean while staying inside the sampled
    # distribution.  A network that returns only the target mean misses it by
    # about 13.7 percent, above the independently calibrated accuracy bar.
    pred = EmulatorPredictor(root, device, compile_model=False)
    h0_t, om_t = 73.0, 0.32
    got = pred.predict({"H0": h0_t, "omegam": om_t})[OUT_NAME]
    want = omegamh2(h0_t, om_t)
    rel = abs(got - want) / want
    validation_rows = len(exp.val_set["idx"])
    collapse_derivation_margin = mean_predictor_median / collapse_bar
    trained_collapse_headroom = (
        float("inf") if best_median == 0.0 else collapse_bar / best_median
    )
    accuracy_derivation_margin = (
        accuracy_bar / CALIBRATED_TWO_EPOCH_REL_ERROR
    )
    current_accuracy_headroom = (
        float("inf") if rel == 0.0 else accuracy_bar / rel
    )
    print(
        "scalar-smoke calibration measurements: validation rows %d, "
        "mean-predictor median %.12g, trained median %.12g, "
        "two-epoch prediction relative error %.12g"
        % (validation_rows, mean_predictor_median, best_median, rel)
    )
    print(
        "scalar-smoke calibration bars: collapse %.12g = %.3gx the "
        "mean-predictor median. Accuracy %.12g = %.3gx the recorded "
        "honest error %.12g"
        % (
            collapse_bar,
            COLLAPSE_BAR_FRACTION,
            accuracy_bar,
            accuracy_derivation_margin,
            CALIBRATED_TWO_EPOCH_REL_ERROR,
        )
    )
    print(
        "scalar-smoke calibration margins: mean predictor / collapse bar "
        "%.6gx, collapse bar / trained median %.6gx, accuracy bar / "
        "current error %.6gx"
        % (
            collapse_derivation_margin,
            trained_collapse_headroom,
            current_accuracy_headroom,
        )
    )
    report(
        "predict reproduces the analytic omegamh2 within the measured bar",
        rel < accuracy_bar,
        "got %.8f want %.8f, relative error %.12g vs bar %.12g"
        % (got, want, rel, accuracy_bar),
    )
    _check_dead_network_mutation(
        exp=exp,
        mean_predictor_median=mean_predictor_median,
        collapse_bar=collapse_bar,
        accuracy_bar=accuracy_bar,
        physical_truth=want,
    )
    check_diagnostics(exp, model, tmp)
    return root, accuracy_bar


def check_diagnostics(exp, model, tmp):
    """The scalar diagnostics leg: 3 pages build + the PDF lands."""
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
        # the hand-built fracs row must MATCH the run's threshold count
        # (DEFAULT_THRESHOLDS has five entries. A fixed 4-wide row made
        # the history panel index column 4 out of bounds — the first
        # execution of this leg caught it in all four smoke gates).
        plot_diagnostics(train_losses=[0.1], medians=[0.1], means=[0.1],
                         fracs=[0.5 * torch.ones(int(exp.thresholds.numel()))],
                         thresholds=exp.thresholds,
                         coverage={"knn_dist": np.ones(4),
                                   "dchi2": np.ones(4),
                                   "k_nn": 2},
                         scalar=sc, savepath=pdf)
        ok = (n_pages == 3 and os.path.isfile(pdf)
              and os.path.getsize(pdf) > 10000)
        report("diagnostics: 3 scalar pages + the PDF lands", ok,
               "%d pages, %d bytes" % (n_pages,
                                       os.path.getsize(pdf)
                                       if os.path.isfile(pdf) else 0))
    except Exception as e:
        report("diagnostics: 3 scalar pages + the PDF lands", False,
               type(e).__name__ + ": " + str(e)[:200])


def check_cobaya_evaluate(tmp, root, accuracy_bar):
    """Run a cobaya `evaluate` through emul_scalars and read back omegamh2.

    Writes a minimal evaluate YAML. Its theory block loads emul_scalars over
    the saved root. An external-lambda likelihood consumes omegamh2, so Cobaya
    must ask the theory for it. The YAML declares omegamh2 as derived and runs
    cobaya-run. The derived value lands in the run's <root>.1.txt. The check
    compares it with the analytic value at the evaluated point with the same
    measured accuracy bar used by the direct predictor. Board only (needs
    cobaya).
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
    # Dead-network rule: off the fixture mean (same point as the predict
    # leg), so the evaluate leg also fails a mean-only network.
    h0_t, om_t = 73.0, 0.32
    want = omegamh2(h0_t, om_t)
    # Mirror the PROVEN cobaya-adapter-evaluate.yaml shape —
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
        # declares its input params. omegamh2 then resolves from the
        # theory's provides. No separate requires key: the
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
    # An evaluate run writes NO .paramnames sidecar (board run 4's
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
    readback_rel = None if got is None else abs(got - want) / want
    okval = readback_rel is not None and readback_rel < accuracy_bar
    report(
        "cobaya evaluate returns omegamh2 within the measured bar",
        okval,
        "got %s want %.8f, relative error %s vs bar %.12g"
        % (got, want, readback_rel, accuracy_bar),
    )
    if got is None:
        # self-diagnosis: a got-None red must name its own cause
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
    print("scalar-smoke: fixture train + predict + cobaya evaluate")
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        root, accuracy_bar = check_train_and_predict(tmp, device)
        if root is not None:
            check_cobaya_evaluate(tmp, root, accuracy_bar)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: scalar-smoke all checks green")


if __name__ == "__main__":
    main()
