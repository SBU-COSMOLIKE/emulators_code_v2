#!/usr/bin/env python3
"""bsn-identity gate: the grid-emulator save/rebuild/predict
identity, the imposed-physics distance pipeline, the target law both
ways, the emul_baosn window/piecewise legs, the finetune parity,
and every grid-path loud error — torch + scipy, no CAMB.

Legs:
  - cumulative_simpson: EVEN doubled-grid points exact on cubics (the
    original z grid sits there), the odd node the correct one-interval
    integral (exact on quadratics; the old half-chunk form is the
    mutation control that must fail the linear / quadratic legs), the
    even-point-count guard;
  - the distance pipeline (real scipy cubic) against an independent
    adaptive-quadrature reference for flat LCDM; the same-integrator fine-grid
    comparison retained as a resolution-only control; mutations proving that
    a shared reference misses scaled Simpson weights and that a nonfinite
    distance makes the scientific comparison fail;
  - GridGeometry: the log_offset law exact both ways (encode(decode) to
    float32 round-off), state round-trip byte-identical (strings and
    offset included), the un-standardizable / log-positivity /
    unknown-law guards;
  - save -> rebuild -> predict bitwise on BOTH laws (the predictor's
    grid branch returns {"z", quantity}); rebuild info flags;
  - emul_baosn (cobaya.theory stubbed): pair validation (missing D_M /
    duplicate quantity / wrong-kind / wrong units loud), the window
    layout, the DESERT query loud at must_provide AND at the getters,
    get_Hubble outside the SN window loud + the 1/Mpc convention, the
    piecewise chi equal to the pipeline / the D_M artifact in their own
    windows, D_A_2 = (chi2 - chi1)/(1+z2);
  - finetune: warm-start epoch-0 parity from a grid source; the
    wrong-kind and grid-metadata-mismatch from_config legs;
  - the NPCE check_npce leg (the 2026-07-12 family-wide ruling): the
    residual base + refiner algebra bitwise, decode composing base +
    net THROUGH the log law, save -> rebuild -> predict bitwise.
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from scipy import integrate, interpolate

from emulator import background as background_math
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.grid import GridGeometry, TARGET_LAWS
from emulator.geometries.parameter import ParamGeometry
from emulator.background import (cumulative_simpson,
                                 distance_interpolators, C_KMS)
from emulator.experiment import EmulatorExperiment
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor
from emulator import fixed_facts
from emulator import warmstart

FAILURES = []
IN_NAMES = ["omegam", "H0", "w"]
N_IN = len(IN_NAMES)

# The adapter legs serve a Hubble artifact and a comoving-distance artifact
# side by side, as the two windows of one background model: emul_baosn refuses
# the set outright when either half is missing. They are one dataset, so they
# carry one identity, which is what handing them the same label does.
ADAPTER_PAIR_LABEL = "bsn-identity/adapter-background-pair"

# The missing-quantity leg's pair: two grid artifacts declaring neither of the
# quantities this adapter serves. They are one fixture, built for one purpose,
# so they carry one identity — and a pair with one identity leaves the
# dataset-identity law nothing to say, so the guard this leg targets is the one
# that speaks.
MISSING_QUANTITY_LABEL = "bsn-identity/missing-quantity-pair"

# The region every grid double this gate PREDICTS THROUGH declares. An emulator
# now refuses any point outside the interval its record was drawn over, so a
# double the gate asks a question of has to stand in for a real emulator's
# region; a double that is only saved, rebuilt, or refused by the adapter
# declares none, and refuses every point, which is the honest record for it.
#
# The box is the design these doubles are built around: the whitening center is
# (omegam, H0, w) = (0.31, 67, -1), and the NPCE leg draws its training columns
# as normals about that center with sigmas (0.01, 2.0, 0.05). Five sigma each
# way is the interval a prior over these coordinates would have declared, and it
# is what a real background emulator of this shape would have been drawn from.
# It contains every point this gate asks about: the round-trip and NPCE point
# (0.32, 68.0, -0.98) and the adapter's point (0.31, 67.0, -1.0).
GRID_SUPPORT = {"omegam": (0.26, 0.36),
                "H0":     (57.0, 77.0),
                "w":      (-1.25, -0.75)}

# A point that box does NOT contain, for the arm proving predict refuses
# outside it. It leaves the box on ONE coordinate: a point outside on every
# coordinate would also be refused by a box law that only ever looked at the
# first one.
OUTSIDE_GRID_BOX = {"omegam": 0.31,
                    "H0":     90.0,
                    "w":      -1.0}

# A point well inside the box, for the arms that must reach the network (and
# for the undeclared double, whose refusal must be about its missing record and
# not about where the point sits).
INSIDE_GRID_BOX = {"omegam": 0.31,
                   "H0":     67.0,
                   "w":      -1.0}

# The production distance pipeline has historically promised relative
# agreement at 1e-6.  scipy.integrate.quad also returns an absolute error
# estimate for its independent reference.  The comparison band adds ten
# times that estimate to the production allowance, so uncertainty in the
# reference cannot create a false failure.
PIPELINE_RELATIVE_TOLERANCE = 1.0e-6
QUADRATURE_ERROR_MULTIPLIER = 10.0
QUADRATURE_ABSOLUTE_TOLERANCE = 1.0e-10
QUADRATURE_RELATIVE_TOLERANCE = 1.0e-12
QUADRATURE_MAX_SUBDIVISIONS = 100

PIPELINE_MIN_REDSHIFT = 0.001
PIPELINE_MAX_REDSHIFT = 3.0
PIPELINE_GRID_POINT_COUNT = 600
PIPELINE_QUERY_REDSHIFTS = (0.1, 0.5, 1.0, 2.0, 2.9)
FINE_REFERENCE_POINT_COUNT = 120001

# Scaling the returned integral by this factor is algebraically identical to
# scaling every Simpson weight by the same factor.  The 1% error is much wider
# than the 1e-6 production allowance while remaining small enough that a
# shared-function reference follows it almost exactly.
SIMPSON_WEIGHT_MUTATION_SCALE = 0.99


def report(label, ok, detail):
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
    # (queue 2) the board folds one reserved '##AID <aid> <result>' line per
    # DECLARED acceptance leg into this gate's executed set. A leg here bundles
    # several human-readable report() sub-checks, so its aggregate verdict is
    # PASS iff no sub-check appended to FAILURES since n_before (the failure
    # count snapshotted before the leg's block ran). The child's exit status
    # stays the single aggregate verdict; these lines carry the per-leg map.
    mark = "PASS" if len(FAILURES) == n_before else "FAIL"
    print("##AID " + aid + " " + mark)


def write_covmat(path, names, seed):
    g = np.random.default_rng(seed)
    a = g.standard_normal((len(names), len(names)))
    cov = a @ a.T + len(names) * np.eye(len(names))
    with open(path, "w") as f:
        f.write("# " + " ".join(names) + "\n")
        for row in cov:
            f.write(" ".join(repr(float(x)) for x in row) + "\n")


def lcdm_h(z, H0=67.36, om=0.315):
    return H0 * np.sqrt(om * (1 + z) ** 3 + (1 - om))


def _lcdm_distance_integrand(redshift):
    """Return c/H(z), the quantity whose integral is comoving distance."""
    return C_KMS / lcdm_h(redshift)


def _old_odd_simpson(z, y):
    """The pre- odd-node rule (dz/6 * (y[i-1] + 4*y[i] + y[i+1])).

    A mutation control: this is HALF the two-interval Simpson total, so it
    must fail the linear / quadratic known-answer legs by a wide margin (a
    first-order h^2/2 error). Even nodes are the same composite Simpson.
    """
    n = len(z)
    dz = z[1] - z[0]
    f0, f1, f2 = y[:-2:2], y[1:-1:2], y[2::2]
    cum_even = np.concatenate(([0.0], np.cumsum(dz / 3 * (f0 + 4 * f1 + f2))))
    C = np.empty_like(y)
    C[0] = 0.0
    C[2::2] = cum_even[1:]
    for i in range(1, n, 2):
        C[i] = C[i - 1] + dz / 6 * (y[i - 1] + 4 * y[i] + y[i + 1])
    return C


def _scaled_simpson_weights(z, y):
    """Mutation control that scales every production Simpson weight."""
    return SIMPSON_WEIGHT_MUTATION_SCALE * cumulative_simpson(z, y)


def _distance_pipeline_with_simpson(z_grid, h_grid, simpson_fn):
    """Build the real distance pipeline with one temporary integrator.

    comoving_distance_grid looks up cumulative_simpson in the
    emulator.background module each time it runs.  Replacing that module
    attribute therefore exercises the real pipeline with the mutation.  The
    finally block restores the production function even if construction
    raises, so later gate legs cannot inherit the deliberate defect.
    """
    production_simpson = background_math.cumulative_simpson
    background_math.cumulative_simpson = simpson_fn
    try:
        return background_math.distance_interpolators(
            z_grid=z_grid,
            h_grid=h_grid,
        )
    finally:
        background_math.cumulative_simpson = production_simpson


def _independent_lcdm_distance(redshift):
    """Integrate c/H(z) without any production Simpson code.

    scipy.integrate.quad adaptively chooses its own evaluation points.  It
    returns both the integral and an estimate of the absolute numerical
    error.  The gate uses that estimate when it constructs its acceptance
    band.
    """
    distance, error_estimate = integrate.quad(
        _lcdm_distance_integrand,
        0.0,
        float(redshift),
        epsabs=QUADRATURE_ABSOLUTE_TOLERANCE,
        epsrel=QUADRATURE_RELATIVE_TOLERANCE,
        limit=QUADRATURE_MAX_SUBDIVISIONS,
    )
    return distance, error_estimate


def _comparison_band(reference, reference_error):
    """Combine the production allowance with quad's error estimate."""
    production_allowance = PIPELINE_RELATIVE_TOLERANCE * abs(reference)
    reference_allowance = QUADRATURE_ERROR_MULTIPLIER * reference_error
    return production_allowance + reference_allowance


def _flat_distance_reference(comoving_distance, comoving_band, redshift):
    """Return flat-distance reference values and their matching bands."""
    one_plus_redshift = 1.0 + redshift
    return {
        "chi": (comoving_distance, comoving_band),
        "da": (
            comoving_distance / one_plus_redshift,
            comoving_band / one_plus_redshift,
        ),
        "dl": (
            comoving_distance * one_plus_redshift,
            comoving_band * one_plus_redshift,
        ),
    }


def _finite_difference_over_band(observed, reference, band):
    """Return |observed-reference|/band, or infinity for invalid inputs.

    The gate's decision is a comparison with a positive acceptance band.
    NaN and infinity have no ordering that can establish agreement.  A zero
    or negative band is not an acceptance interval.  Mapping every invalid
    case to positive infinity makes the ordinary ``ratio < 1`` decision fail
    without relying on Python's ordering behavior for NaN.
    """
    if not np.isfinite(observed):
        return float("inf")
    if not np.isfinite(reference):
        return float("inf")
    if not np.isfinite(band) or band <= 0.0:
        return float("inf")
    return abs(observed - reference) / band


def _pipeline_ratios(pipeline, reference_table):
    """Evaluate one pipeline against all named reference values and bands."""
    ratios = []
    for redshift in PIPELINE_QUERY_REDSHIFTS:
        references = reference_table[redshift]
        for quantity, reference_pair in references.items():
            reference, band = reference_pair
            observed = float(pipeline[quantity](redshift))
            ratio = _finite_difference_over_band(
                observed=observed,
                reference=reference,
                band=band,
            )
            ratios.append(ratio)
    return ratios


def _ratios_pass(ratios):
    """Return True only when every finite comparison lies inside its band."""
    if not ratios:
        return False
    for ratio in ratios:
        if not np.isfinite(ratio) or ratio >= 1.0:
            return False
    return True


def _nonfinite_distance(_redshift):
    """Mutation control: one pipeline quantity returns no numeric distance."""
    return float("nan")


def check_simpson():
    # known-answer integrals at EVERY node. The one-interval odd
    # rule h/12*(5,8,-1) is exact on quadratics, so even AND odd nodes hit
    # machine precision on constant / linear / quadratic; the cubic keeps
    # even exact (composite Simpson) with a small bounded odd error. The
    # retired e_odd < 1e-3 tolerance encoded the old first-order bug.
    z = np.linspace(0.0, 3.0, 601)
    cases = [
        ("constant", np.ones_like(z), z),
        ("linear",   z,               0.5 * z ** 2),
        ("quadratic", z ** 2,         z ** 3 / 3.0),
        ("cubic",     2.0 * z ** 3 - z ** 2 + 4.0 * z + 1.0,
         0.5 * z ** 4 - z ** 3 / 3.0 + 2.0 * z ** 2 + z),
    ]
    for label, y, truth in cases:
        got = cumulative_simpson(z, y)
        e_even = np.abs(got[::2] - truth[::2]).max()
        e_odd = np.abs(got[1::2] - truth[1::2]).max()
        odd_bar = 1e-4 if label == "cubic" else 1e-9   # cubic: bounded, not exact
        report("Simpson: " + label + " -- even exact, odd "
               + ("bounded" if label == "cubic" else "exact"),
               e_even < 1e-9 and e_odd < odd_bar and got[0] == 0.0,
               "even %.1e odd %.1e" % (e_even, e_odd))

    # mutation catch-power: the OLD (1,4,1)/6 odd form must be wide of the
    # machine-precision answer on the linear and quadratic legs.
    for label, y, truth in cases[1:3]:
        e_odd_old = np.abs(_old_odd_simpson(z, y)[1::2] - truth[1::2]).max()
        report("Simpson: the OLD (1,4,1)/6 odd form FAILS " + label
               + " (mutation catch)", e_odd_old > 1e-6,
               "old odd error %.1e (wide of machine precision)" % e_odd_old)

    # odd-point guard: an even point count raises.
    guarded = False
    try:
        cumulative_simpson(z[:-1], (z ** 2)[:-1])
    except ValueError:
        guarded = True
    report("Simpson: even point count raises (odd-point guard)",
           guarded, "guarded")


def check_pipeline():
    z_grid = np.linspace(
        PIPELINE_MIN_REDSHIFT,
        PIPELINE_MAX_REDSHIFT,
        PIPELINE_GRID_POINT_COUNT,
    )
    h_grid = lcdm_h(z_grid)
    pipeline = distance_interpolators(
        z_grid=z_grid,
        h_grid=h_grid,
    )

    # This reference shares neither the production integration rule nor its
    # evaluation grid.  For each redshift, quad integrates the analytic LCDM
    # c/H(z) function directly from zero to that redshift.
    independent_reference_table = {}
    for redshift in PIPELINE_QUERY_REDSHIFTS:
        chi_reference, chi_reference_error = _independent_lcdm_distance(
            redshift,
        )
        chi_band = _comparison_band(
            reference=chi_reference,
            reference_error=chi_reference_error,
        )
        independent_reference_table[redshift] = _flat_distance_reference(
            comoving_distance=chi_reference,
            comoving_band=chi_band,
            redshift=redshift,
        )
    independent_ratios = _pipeline_ratios(
        pipeline=pipeline,
        reference_table=independent_reference_table,
    )
    report(
        "pipeline vs independent adaptive quadrature",
        _ratios_pass(independent_ratios),
        "largest difference/band %.3e" % max(independent_ratios),
    )

    # The same-integrator fine-grid calculation remains useful as a resolution
    # control.  It checks that the 600-point production grid resolves the same
    # curve as a 120001-point grid.  Because both sides call cumulative_simpson,
    # this assertion makes no claim about the scientific normalization of the
    # integration weights.
    fine_redshift_grid = np.linspace(
        0.0,
        float(z_grid[-1]),
        FINE_REFERENCE_POINT_COUNT,
    )
    fine_chi_reference = cumulative_simpson(
        fine_redshift_grid,
        C_KMS / lcdm_h(fine_redshift_grid),
    )
    fine_reference_table = {}
    for redshift in PIPELINE_QUERY_REDSHIFTS:
        chi_reference = float(np.interp(
            redshift,
            fine_redshift_grid,
            fine_chi_reference,
        ))
        chi_band = PIPELINE_RELATIVE_TOLERANCE * abs(chi_reference)
        fine_reference_table[redshift] = _flat_distance_reference(
            comoving_distance=chi_reference,
            comoving_band=chi_band,
            redshift=redshift,
        )
    fine_ratios = _pipeline_ratios(
        pipeline=pipeline,
        reference_table=fine_reference_table,
    )
    report(
        "pipeline vs same-integrator fine grid (resolution only)",
        _ratios_pass(fine_ratios),
        "largest difference/band %.3e" % max(fine_ratios),
    )

    # Mutation catch power.  Scaling every Simpson weight on BOTH sides leaves
    # the shared-function resolution control green.  The adaptive integral
    # retains the physical normalization and rejects the same pipeline.
    mutated_pipeline = _distance_pipeline_with_simpson(
        z_grid=z_grid,
        h_grid=h_grid,
        simpson_fn=_scaled_simpson_weights,
    )
    shared_mutated_chi = _scaled_simpson_weights(
        fine_redshift_grid,
        C_KMS / lcdm_h(fine_redshift_grid),
    )
    shared_mutated_reference_table = {}
    for redshift in PIPELINE_QUERY_REDSHIFTS:
        chi_reference = float(np.interp(
            redshift,
            fine_redshift_grid,
            shared_mutated_chi,
        ))
        chi_band = PIPELINE_RELATIVE_TOLERANCE * abs(chi_reference)
        shared_mutated_reference_table[redshift] = _flat_distance_reference(
            comoving_distance=chi_reference,
            comoving_band=chi_band,
            redshift=redshift,
        )
    shared_mutated_ratios = _pipeline_ratios(
        pipeline=mutated_pipeline,
        reference_table=shared_mutated_reference_table,
    )
    report(
        "Simpson-weight mutation: shared fine-grid reference is blind",
        _ratios_pass(shared_mutated_ratios),
        "largest difference/band %.3e" % max(shared_mutated_ratios),
    )

    independent_mutated_ratios = _pipeline_ratios(
        pipeline=mutated_pipeline,
        reference_table=independent_reference_table,
    )
    independent_mutation_is_finite = all(
        np.isfinite(ratio) for ratio in independent_mutated_ratios
    )
    independent_mutation_misses_every_band = (
        min(independent_mutated_ratios) > 1.0
    )
    report(
        "Simpson-weight mutation: independent quadrature rejects it",
        independent_mutation_is_finite
        and independent_mutation_misses_every_band,
        "smallest difference/band %.3e" % min(independent_mutated_ratios),
    )

    # Nonfinite catch power.  Replace only the comoving-distance callable,
    # then evaluate the SAME acceptance predicate as the control.  The shared
    # ratio helper maps NaN to infinity, so one invalid value makes the whole
    # distance-pipeline verdict false instead of disappearing inside max().
    nonfinite_pipeline = dict(pipeline)
    nonfinite_pipeline["chi"] = _nonfinite_distance
    nonfinite_ratios = _pipeline_ratios(
        pipeline=nonfinite_pipeline,
        reference_table=independent_reference_table,
    )
    nonfinite_pipeline_passes = _ratios_pass(nonfinite_ratios)
    nonfinite_value_reached_ratio_helper = any(
        np.isinf(ratio) for ratio in nonfinite_ratios
    )
    report(
        "nonfinite-distance mutation makes the pipeline comparison red",
        not nonfinite_pipeline_passes
        and nonfinite_value_reached_ratio_helper,
        "acceptance verdict %s" % nonfinite_pipeline_passes,
    )


def check_geometry(device):
    z = np.linspace(0.001, 3.0, 64)
    g = np.random.default_rng(3)
    Y = lcdm_h(z)[None, :] * (1.0 + 0.05 * g.standard_normal((400, 64)))
    geom = GridGeometry.from_targets(device=device, targets=Y, z=z,
                                     quantity="Hubble", units="km/s/Mpc",
                                     law="log_offset", offset=1.0)
    t = torch.randn(6, 64)
    back = geom.encode(geom.decode(t))
    rel = (back - t).abs().max().item()
    report("log_offset law: encode(decode(x)) to float32 round-off",
           rel < 1e-4, "max |d| %.1e" % rel)
    st0 = geom.state()
    geom2 = GridGeometry.from_state(device=device, state=st0)
    st1 = geom2.state()
    ok = set(st0) == set(st1)
    for k in st0:
        a, b = st0[k], st1[k]
        ok = ok and (torch.equal(a, b) if isinstance(a, torch.Tensor)
                     else a == b)
    report("grid state round-trip byte-identical", ok,
           "%d keys incl. law/offset/quantity" % len(st0))
    try:
        GridGeometry.from_targets(device=device, targets=Y, z=z,
                                  quantity="Hubble", units="km/s/Mpc",
                                  law="nope")
        report("unknown law raises", False, "no raise")
    except ValueError:
        report("unknown law raises", True, "ValueError")
    try:
        GridGeometry.from_targets(device=device, targets=-np.abs(Y), z=z,
                                  quantity="Hubble", units="km/s/Mpc",
                                  law="log_offset", offset=0.5)
        report("log-positivity guard raises", False, "no raise")
    except ValueError:
        report("log-positivity guard raises", True, "ValueError")
    try:
        GridGeometry.from_targets(
            device=device, targets=np.full((100, 64), 70.0), z=z,
            quantity="Hubble", units="km/s/Mpc", law="none")
        report("un-standardizable guard raises", False, "no raise")
    except ValueError:
        report("un-standardizable guard raises", True, "ValueError")


def grid_recipe(nz):
    return {"cls": "emulator.designs.plain.ResMLP",
            "name": "resmlp",
            "ia": None,
            "input_dim": N_IN,
            "output_dim": nz,
            "compile_mode": None,
            "needs_geom": False,
            "kwargs": {"int_dim_res": 16,
                       "n_blocks": 2,
                       "block_opts": {"act": {"type": "H", "n_gates": 3},
                                      "norm": "affine"}}}


def save_synthetic_grid(root, device, tmp, label, quantity="Hubble",
                        units="km/s/Mpc", law="log_offset", offset=1.0,
                        z=None, seed=0, support=None):
    """Build, then save, a tiny synthetic grid emulator under `root`.

    `label` fixes the identity of the scientific record the file carries:
    doubles that belong to one dataset are handed the same label, doubles that
    must be told apart are handed different ones.

    `support` is the region the double stands for, as a mapping name -> (low,
    high). A double the gate PREDICTS THROUGH declares one, because a real
    emulator was drawn from an interval and is refused outside it. A double
    that is only saved, rebuilt, compared, or offered to an adapter that turns
    it away declares None, and then refuses every prediction — the honest
    record for a double nobody asks a question of, not a gap in one.
    """
    if z is None:
        z = np.linspace(0.001, 3.0, 64)
    covmat = os.path.join(tmp, "grid_%d.covmat" % seed)
    write_covmat(covmat, IN_NAMES, seed=seed + 1)
    pgeom = ParamGeometry.from_covmat(
        device=device, center=np.array([0.31, 67.0, -1.0]),
        covmat_path=covmat)
    g = np.random.default_rng(seed + 2)
    base = lcdm_h(z) if quantity == "Hubble" else 4000.0 + 3.0 * z
    Y = base[None, :] * (1.0 + 0.05 * g.standard_normal((400, len(z))))
    geom = GridGeometry.from_targets(device=device, targets=Y, z=z,
                                     quantity=quantity, units=units,
                                     law=law, offset=offset)
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=len(z), int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    config = {"data": {"grid": {"quantity": quantity,
                                "units": units,
                                "law": law,
                                "z_file": "z.npy"},
                       "train_dv": "t.npy",
                       "val_dv": "v.npy",
                       "train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat)},
              "train_args": {"nepochs": 1}}
    if law == "log_offset":
        config["data"]["grid"]["offset"] = offset
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
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=grid_recipe(len(z)),
                  facts_yaml=fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label=label,
                      family="grid",
                      support=support),
                  attrs={"rescale": "none", "quantity": quantity})
    return pgeom, geom, model, covmat


def check_roundtrip(tmp, device, law):
    root = os.path.join(tmp, "emul_grid_" + law)
    off = 1.0 if law == "log_offset" else 0.0
    # predicted through, so it declares the region it stands for.
    pgeom, geom, model, _ = save_synthetic_grid(
        root, device, tmp, label="bsn-identity/round-trip-" + law,
        law=law, offset=off, seed=30, support=GRID_SUPPORT)
    theta = np.array([[0.32, 68.0, -0.98]])
    x = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        ref = geom.decode(model(pgeom.encode(x)))[0].cpu().numpy()
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    ok = (np.array_equal(got["Hubble"], ref)
          and np.array_equal(got["z"], geom.z.cpu().numpy())
          and getattr(pred, "_grid", False)
          and pred.quantity == "Hubble" and pred.units == "km/s/Mpc")
    report("predict round-trip bitwise (%s law)" % law, ok,
           "max|d| %.1e" % np.abs(got["Hubble"] - ref).max())
    _, _, _, info = rebuild_emulator(root, device, compile_model=False)
    report("rebuild info: grid flags (%s)" % law,
           info["grid"] and info["grid_quantity"] == "Hubble"
           and info["grid_law"] == law and not info["cmb"]
           and info["amplitude_law"] is None,
           "law %s, amplitude_law %s" % (info["grid_law"],
                                         info["amplitude_law"]))
    return root


def check_domain_law(root, tmp, device):
    """predict() refuses a point the artifact's record does not cover.

    Two refusals, because a record can fail to cover a point in two different
    ways and only one of them is about the point:

      no support    the record declares no interval for any coordinate. Its
                    bounds are not wide, they are absent, so there is no region
                    it may be asked about at all. That is the shape of a test
                    double, and a test double must never answer a likelihood.

      outside it    the record declares a box, and the point is outside it. The
                    emulator would not fail there; it would extrapolate, and
                    hand back a confident H(z) of the right shape and the wrong
                    value — which is the quietest way to move a distance ladder.

    Both arms are read by their WORDS. float("n/a") raises the same ValueError
    class a refusal raises, so an arm that only asked "did it raise?" would go
    green on a record that crashed instead of refusing, and the law it exists to
    prove would never have run.

    Arguments:
      root   = the round-trip double's path root. It declares GRID_SUPPORT, and
               is the artifact the outside-the-box arm asks off its region.
      tmp    = the tempdir this gate's fixtures live in.
      device = the torch device to rebuild on.
    """
    pred = EmulatorPredictor(root, device, compile_model=False)
    try:
        pred.predict(OUTSIDE_GRID_BOX)
        report("a point outside the declared box is refused",
               False, "did not raise")
    except ValueError as e:
        report_refusal("a point outside the declared box is refused", e,
                       needle="which is outside it",
                       law="the domain law (the point leaves the box)")

    # a double that declares no support: saved, rebuilt, and then asked a
    # question it has no region to answer in.
    root_undeclared = os.path.join(tmp, "emul_grid_undeclared")
    save_synthetic_grid(root_undeclared, device, tmp,
                        label="bsn-identity/undeclared-support",
                        quantity="Hubble", units="km/s/Mpc",
                        law="log_offset", offset=1.0, seed=100)
    pred_undeclared = EmulatorPredictor(root_undeclared, device,
                                        compile_model=False)
    try:
        pred_undeclared.predict(INSIDE_GRID_BOX)
        report("a double that declares no support refuses every predict",
               False, "did not raise")
    except ValueError as e:
        report_refusal(
            "a double that declares no support refuses every predict", e,
            needle="declares no support",
            law="the domain law (no region was ever declared)")


def check_npce(tmp, device):
    """NPCE on the grid family (the 2026-07-12 family-wide ruling): the
    residual base + refiner algebra is exact under the diagonal metric,
    decode composes base + net THROUGH the target law (log_offset), and
    save -> rebuild -> predict is bitwise (_build_diag_decoder)."""
    from emulator.designs.pce import PCEEmulator
    from emulator.losses.pce import PCEResidualDiagChi2
    covmat = os.path.join(tmp, "grid_npce.covmat")
    write_covmat(covmat, IN_NAMES, seed=61)
    pgeom = ParamGeometry.from_covmat(
        device=device, center=np.array([0.31, 67.0, -1.0]),
        covmat_path=covmat)
    z = np.linspace(0.001, 3.0, 64)
    g = np.random.default_rng(62)
    C = np.column_stack([g.normal(0.31, 0.01, 400),
                         g.normal(67.0, 2.0, 400),
                         g.normal(-1.0, 0.05, 400)]).astype("float32")
    # H(z) rows that REALLY move with the sampled H0, so the LOO gate
    # keeps a mode and the fitted base is alive (the smoke-gate rule).
    Y = (lcdm_h(z)[None, :] * (C[:, 1:2] / 67.0)
         * (1.0 + 0.01 * g.standard_normal((400, z.size))))
    geom = GridGeometry.from_targets(device=device, targets=Y, z=z,
                                     quantity="Hubble", units="km/s/Mpc",
                                     law="log_offset", offset=1.0)
    tC = torch.from_numpy(C).to(device)
    X_white = pgeom.encode(tC)
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
    y = torch.randn(8, z.size, device=device)
    report("NPCE decode composes base + net through the log law, bitwise",
           torch.equal(chi2fn.decode(y, X_white[:8]),
                       geom.decode(y + base)), "")
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=z.size, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    root = os.path.join(tmp, "emul_grid_npce")
    config = {"data": {"grid": {"quantity": "Hubble",
                                "units": "km/s/Mpc",
                                "law": "log_offset",
                                "offset": 1.0,
                                "z_file": "z.npy"},
                       "train_dv": "t.npy",
                       "val_dv": "v.npy",
                       "train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat)},
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
                  resolved_model=grid_recipe(z.size),
                  pce=pce, pce_form="residual",
                  facts_yaml=fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label="bsn-identity/npce-hubble",
                      family="grid",
                      support=GRID_SUPPORT),
                  attrs={"rescale": "none", "quantity": "Hubble"})
    theta = np.array([[0.32, 68.0, -0.98]])
    x1 = torch.as_tensor(theta, dtype=pgeom.center.dtype, device=device)
    with torch.no_grad():
        x1e = pgeom.encode(x1)
        ref = geom.decode(model(x1e) + pce(x1e))[0].cpu().numpy()
    pred = EmulatorPredictor(root, device, compile_model=False)
    got = pred.predict({nm: float(theta[0, i])
                        for i, nm in enumerate(IN_NAMES)})
    report("NPCE save -> rebuild -> predict composes base + net bitwise",
           np.array_equal(got["Hubble"], ref),
           "max|d| = %.2e" % np.abs(got["Hubble"] - ref).max())


def _load_emul_baosn_stubbed():
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
    path = root / "cobaya_theory" / "emul_baosn.py"
    spec = importlib.util.spec_from_file_location("emul_baosn_shim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.emul_baosn


def _build(cls, roots):
    t = cls()
    t.extra_args = {"device": "cpu", "emulators": list(roots)}
    t.initialize()
    return t


def check_adapter(tmp, device):
    cls = _load_emul_baosn_stubbed()
    # both halves are served, so both are predicted through (calculate runs
    # each of them on every point): both declare the region they stand for.
    root_h = os.path.join(tmp, "ad_h")
    save_synthetic_grid(root_h, device, tmp, label=ADAPTER_PAIR_LABEL,
                        quantity="Hubble",
                        units="km/s/Mpc", law="log_offset", offset=1.0,
                        z=np.linspace(0.001, 3.0, 64), seed=40,
                        support=GRID_SUPPORT)
    root_dm = os.path.join(tmp, "ad_dm")
    save_synthetic_grid(root_dm, device, tmp, label=ADAPTER_PAIR_LABEL,
                        quantity="D_M",
                        units="Mpc", law="none", offset=0.0,
                        z=np.linspace(1000.0, 1200.0, 24), seed=50,
                        support=GRID_SUPPORT)

    t = _build(cls, [root_h, root_dm])
    report("pair layout: SN window + rec window",
           t._sn_max == 3.0 and t._rec_min == 1000.0
           and t._rec_max == 1200.0,
           "sn_max %.1f rec [%.0f, %.0f]" % (t._sn_max, t._rec_min,
                                             t._rec_max))
    # must_provide desert leg + valid legs
    t.must_provide(Hubble={"z": np.array([0.1, 2.0])},
                   comoving_radial_distance={"z": np.array([1090.0])})
    try:
        t.must_provide(angular_diameter_distance={"z": np.array([500.0])})
        report("desert must_provide raises", False, "no raise")
    except ValueError as e:
        report_refusal("desert must_provide raises", e,
                       needle="never emulated",
                       law="the desert law")

    # calculate + the piecewise getters vs the pipeline / the artifact.
    # Every comparison is LIKE FOR LIKE (the run-10 lesson): the check
    # rebuilds the SAME interpolators the adapter builds — the pipeline
    # splines for the SN window and the adapter's own cubic interp1d
    # for the recombination-window D_M artifact (the old leg compared
    # the adapter's cubic against a LINEAR np.interp at rtol 1e-3,
    # which flakes on the curvature of an unseeded synthetic net).
    # Same-computation comparisons stay exact; each assertion reports
    # on its own line so a red names its sub-leg.
    point = {"omegam": 0.31,
             "H0": 67.0,
             "w": -1.0}
    state = {}
    t.calculate(state, want_derived=True, **point)
    t.current_state = state
    out_h = t.p_h.predict(point)
    itp = distance_interpolators(z_grid=out_h["z"], h_grid=out_h["Hubble"])
    zq = np.array([0.3, 1.5])
    got_chi = t.get_comoving_radial_distance(zq)
    report("piecewise chi == the pipeline (SN window, exact)",
           np.allclose(got_chi, itp["chi"](zq), rtol=0, atol=0),
           "max|d| %.1e" % np.abs(got_chi - itp["chi"](zq)).max())
    got_da = t.get_angular_diameter_distance(zq)
    report("piecewise D_A == chi/(1+z) (SN window)",
           np.allclose(got_da, itp["chi"](zq) / (1 + zq)),
           "max|d| %.1e"
           % np.abs(got_da - itp["chi"](zq) / (1 + zq)).max())
    got_dl = t.get_luminosity_distance(zq)
    report("piecewise D_L == chi*(1+z) (SN window)",
           np.allclose(got_dl, itp["chi"](zq) * (1 + zq)),
           "max|d| %.1e"
           % np.abs(got_dl - itp["chi"](zq) * (1 + zq)).max())
    out_dm = t.p_dm.predict(point)
    dm_itp = interpolate.interp1d(out_dm["z"], out_dm["D_M"],
                                  kind='cubic',
                                  assume_sorted=True,
                                  fill_value="extrapolate")
    zr = np.array([1090.0])
    want_dm = dm_itp(zr)
    got_dm = t.get_comoving_radial_distance(zr)
    report("piecewise chi == the D_M artifact (rec window, same cubic)",
           np.allclose(got_dm, want_dm),
           "max|d| %.1e" % np.abs(got_dm - want_dm).max())
    pair = t.get_angular_diameter_distance_2([[0.3, 1.5]])
    want_pair = (itp["chi"](1.5) - itp["chi"](0.3)) / 2.5
    report("D_A_2 == (chi2 - chi1)/(1 + z2)",
           np.allclose(pair, [want_pair]),
           "max|d| %.1e" % np.abs(np.asarray(pair) - want_pair).max())
    # H units + the H-outside-SN loud error. This one leg bundles three
    # sub-checks, so it cannot use report_refusal (that helper reports a leg of
    # its own); each catch needles its guard's message by hand instead, for the
    # same reason: a bare catch would accept any ValueError at all, and both
    # guards would go untested the day an earlier law starts refusing first.
    h1 = t.get_Hubble(np.array([1.0]))
    h2 = t.get_Hubble(np.array([1.0]), units="1/Mpc")
    ok = np.allclose(h1 / C_KMS, h2)
    detail = "km/s/Mpc vs 1/Mpc, both guards named their law"
    try:
        t.get_Hubble(np.array([1090.0]))
        ok = False
        detail = "the D_M-window H query did not raise"
    except ValueError as e:
        if "outside the SN-range window" not in str(e):
            ok = False
            detail = "the window guard refused the WRONG law: " + str(e)
    try:
        t.get_Hubble(np.array([1.0]), units="parsecs")
        ok = False
        detail = "the 'parsecs' units query did not raise"
    except ValueError as e:
        if "units must be 'km/s/Mpc' or '1/Mpc'" not in str(e):
            ok = False
            detail = "the units guard refused the WRONG law: " + str(e)
    report("get_Hubble units + window guards", ok, detail)
    # desert getter leg
    try:
        t.get_comoving_radial_distance(np.array([500.0]))
        report("desert getter raises", False, "no raise")
    except ValueError as e:
        report_refusal("desert getter raises", e,
                       needle="never emulated",
                       law="the desert law")

    # pair-validation legs
    try:
        _build(cls, [root_h])
        report("missing D_M raises", False, "no raise")
    except ValueError as e:
        # NOTE the needle names the pair-COUNT law, not the missing-quantity
        # one. A one-root emulators list is refused by the "exactly TWO" guard
        # (cobaya_theory/emul_baosn.py:90) before any artifact is loaded, so
        # the by-quantity scan that would say "no loaded artifact declares
        # quantity 'D_M'" (:134) is never reached from here. Needling that
        # message would demand text this call site cannot produce; needling the
        # one that does fire is what makes the leg refuse an identity error.
        # The missing-quantity guard has a leg of its own (the
        # check_missing_quantity leg below): reaching it needs a TWO-root list
        # of distinct quantities with no D_M among them, which is exactly the
        # fixture that leg builds.
        report_refusal("missing D_M raises", e,
                       needle="exactly TWO",
                       law="the pair-count law")
    root_h2 = os.path.join(tmp, "ad_h2")
    # a second Hubble emulator, built to be refused beside the first. It carries
    # the PAIR's label on purpose: two artifacts declaring one quantity, both
    # trained off ONE generator dump, is exactly the ambiguity the duplicate law
    # exists to refuse -- one dataset, one identity. Give this double an
    # identity of its own instead and the served pair stops being one dataset:
    # the identity law refuses it first, and the duplicate law -- the law this
    # leg exists to prove -- is never reached.
    save_synthetic_grid(root_h2, device, tmp,
                        label=ADAPTER_PAIR_LABEL,
                        quantity="Hubble",
                        units="km/s/Mpc", law="none", offset=0.0,
                        z=np.linspace(0.001, 2.0, 32), seed=60)
    try:
        _build(cls, [root_h, root_h2])
        report("duplicate quantity raises", False, "no raise")
    except ValueError as e:
        report_refusal("duplicate quantity raises", e,
                       needle="two artifacts both declare quantity",
                       law="the duplicate-quantity law")

    # two artifacts fitted to DIFFERENT datasets -> loud. The pair is combined
    # into ONE expansion history, so it has to come from one dataset: two runs
    # of the same YAML agree on every fixed fact and every bound and still drew
    # different points, and only the identity can tell them apart.
    #
    # The pair handed over is topologically VALID on purpose -- one 'Hubble' and
    # one 'D_M', the units each half is served in, windows that do not overlap
    # -- because the adapter runs those laws FIRST. Hand it a pair that also
    # broke one of them and the earlier law would fire, and the needle below
    # would be naming a law that never ran.
    root_h_other = os.path.join(tmp, "ad_h_other")
    save_synthetic_grid(root_h_other, device, tmp,
                        label="bsn-identity/adapter-foreign-dataset",
                        quantity="Hubble",
                        units="km/s/Mpc", law="log_offset", offset=1.0,
                        z=np.linspace(0.001, 3.0, 64), seed=110)
    try:
        _build(cls, [root_h_other, root_dm])
        report("mismatched dataset identity raises", False, "no raise")
    except ValueError as e:
        report_refusal("mismatched dataset identity raises", e,
                       needle="different datasets",
                       law="the dataset-identity law")


def check_missing_quantity(tmp, device):
    """emul_baosn refuses a pair that declares neither quantity it serves.

    The adapter serves exactly two quantities, one 'Hubble' and one 'D_M', and
    a set that carries neither of them cannot be assembled into an expansion
    history at all. That guard (cobaya_theory/emul_baosn.py:134) has never been
    reached from this gate, and the reason is worth stating, because it is the
    same trap the missing-D_M leg's comment describes: a ONE-root list is
    refused by the "exactly TWO" law before a single artifact is loaded, so no
    fixture that hands over one root can ever arrive here.

    The fixture is therefore TWO valid grid artifacts with distinct quantities,
    neither of them one the adapter serves. They pass the count law (two roots),
    the wrong-kind law (both grid), and the duplicate law (distinct quantities),
    and then die on the law this leg exists to prove. They share one label, so
    they are one dataset: a pair with one identity leaves the dataset-identity
    law nothing to refuse, and the guard this leg targets is the one that
    speaks.

    Arguments:
      tmp    = the tempdir this gate's fixtures live in.
      device = the torch device to save + rebuild on.
    """
    cls = _load_emul_baosn_stubbed()
    # never predicted through (initialize refuses the set), so neither declares
    # a support: that is the honest record for a double nobody may ask anything.
    root_dv = os.path.join(tmp, "mq_dv")
    save_synthetic_grid(root_dv, device, tmp, label=MISSING_QUANTITY_LABEL,
                        quantity="D_V", units="Mpc", law="none", offset=0.0,
                        z=np.linspace(0.001, 3.0, 64), seed=130)
    root_dh = os.path.join(tmp, "mq_dh")
    save_synthetic_grid(root_dh, device, tmp, label=MISSING_QUANTITY_LABEL,
                        quantity="D_H", units="Mpc", law="none", offset=0.0,
                        z=np.linspace(1000.0, 1200.0, 24), seed=140)
    try:
        _build(cls, [root_dv, root_dh])
        report("a pair declaring neither served quantity raises",
               False, "no raise")
    except ValueError as e:
        report_refusal("a pair declaring neither served quantity raises", e,
                       needle="no loaded artifact declares quantity",
                       law="the missing-quantity law")


def check_finetune(tmp, device):
    root = os.path.join(tmp, "ft_grid_src")
    pgeom, geom, model, covmat = save_synthetic_grid(
        root, device, tmp, label="bsn-identity/finetune-hubble-source",
        law="log_offset", offset=1.0, seed=70)
    source = warmstart.load_source(root=root, device=device)
    report("load_source accepts a grid artifact",
           type(source.geom).__name__ == "GridGeometry",
           "geom %s" % type(source.geom).__name__)
    g = np.random.default_rng(80)
    C = np.column_stack([g.normal(0.31, 0.01, 64),
                         g.normal(67.0, 2.0, 64),
                         g.normal(-1.0, 0.05, 64)]).astype("float32")
    train_set = {"C": C,
                 "idx": np.arange(64),
                 "C_mean": C.mean(axis=0)}
    new_pgeom, extra = warmstart.extend_input_geometry(
        source=source, covmat_path=covmat,
        train_mean=train_set["C_mean"], device=device)
    model_opts = warmstart.recipe_to_model_opts(source.recipe)
    try:
        init_state, verdict, _ = warmstart.build_warm_start(
            source=source, new_pgeom=new_pgeom, pinned_geom=source.geom,
            model_opts=model_opts, train_set=train_set,
            extra_names=extra, device=device)
        report("grid warm start reproduces the source at epoch 0",
               init_state is not None, verdict.strip()[:60])
    except ValueError as e:
        report("grid warm start reproduces the source at epoch 0",
               False, str(e)[:80])
    # from_config legs: wrong-kind + metadata mismatch (before staging).
    def ft_cfg(grid_block, from_root):
        return {"data": {"grid": grid_block,
                         "train_dv": "t.npy",
                         "val_dv": "v.npy",
                         "train_params": "t.1.txt",
                         "val_params": "v.1.txt",
                         "train_covmat": covmat,
                         "n_train": 10,
                         "n_val": 5,
                         "split_seed": 0},
                "train_args": {"nepochs": 1,
                               "bs": 8,
                               "finetune": {"from": from_root}}}
    good = {"quantity": "Hubble",
            "units": "km/s/Mpc",
            "law": "log_offset",
            "offset": 1.0,
            "z_file": "z.npy"}
    bad = dict(good)
    bad["offset"] = 2.0
    try:
        EmulatorExperiment.from_config(ft_cfg(bad, root),
                                       device=torch.device("cpu"))
        report("metadata mismatch raises", False, "no raise")
    except ValueError as e:
        report("metadata mismatch raises",
               "grid-metadata mismatch" in str(e), "ValueError")
    dm_root = os.path.join(tmp, "ft_dm_src")
    # a distance source offered to a Hubble run, built to be refused: a
    # different run, so a different identity.
    save_synthetic_grid(dm_root, device, tmp,
                        label="bsn-identity/finetune-distance-source",
                        quantity="D_M",
                        units="Mpc", law="none", offset=0.0,
                        z=np.linspace(1000.0, 1200.0, 24), seed=90)
    try:
        EmulatorExperiment.from_config(ft_cfg(good, dm_root),
                                       device=torch.device("cpu"))
        report("cross-quantity source raises", False, "no raise")
    except ValueError as e:
        report("cross-quantity source raises",
               "grid-metadata mismatch" in str(e), "ValueError")


def main():
    print("bsn-identity: pipeline + law + round-trip + adapter "
          "+ finetune legs")
    # seed the GLOBAL torch RNG: the synthetic ResMLPs' weights come
    # from it, and an unseeded net makes any tolerance-adjacent leg
    # unreproducible run to run (the run-10 flake).
    torch.manual_seed(0)
    device = torch.device("cpu")
    # Each declared board leg brackets its block with a FAILURES snapshot and
    # emits exactly one '##AID' line (emit_aid). The declared evidence set
    # (ai/gates/board.py Gate id="bsn-identity") is exactly these aids -- one per
    # bracket, no stray manifest line for a sub-check.
    with tempfile.TemporaryDirectory() as tmp:
        n = len(FAILURES)
        check_simpson()
        emit_aid("bsn-identity.simpson-polynomial-nodes", n)

        n = len(FAILURES)
        check_pipeline()
        emit_aid("bsn-identity.distance-pipeline-consistency", n)

        n = len(FAILURES)
        check_geometry(device)
        root = check_roundtrip(tmp, device, law="log_offset")
        check_roundtrip(tmp, device, law="none")
        check_domain_law(root, tmp, device)
        emit_aid("bsn-identity.geometry-and-artifact-round-trip", n)

        n = len(FAILURES)
        check_npce(tmp, device)
        emit_aid("bsn-identity.npce-composition", n)

        n = len(FAILURES)
        check_adapter(tmp, device)
        emit_aid("bsn-identity.adapter-piecewise-contract", n)

        n = len(FAILURES)
        check_finetune(tmp, device)
        emit_aid("bsn-identity.finetune-parity", n)

        n = len(FAILURES)
        check_missing_quantity(tmp, device)
        emit_aid("bsn-identity.missing-quantity-refused", n)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: bsn-identity all checks green")


if __name__ == "__main__":
    main()
