#!/usr/bin/env python3
"""scalar-identity gate: the scalar-emulator save/rebuild/predict
identity and every scalar-path loud error, torch only (no cosmolike).

It builds a tiny synthetic scalar emulator by hand (a ParamGeometry over a
written covmat + a ScalarGeometry over synthetic targets + a small ResMLP),
saves it with save_emulator, rebuilds it, and asserts:
  - predict returns a {name: value} dict that reproduces the pre-save model
    bitwise (save/rebuild preserves the weights + the standardization);
  - ScalarGeometry.state() round-trips byte-identical;
  - the scalar-path loud errors fire: a constant (un-standardizable)
    output column, a duplicated .paramnames name, a head architecture
    on a scalar run;
  - the cobaya adapter emul_scalars derives its provides from the artifacts
    and raises on a duplicate output, an input/provide overlap, a bad
    `provides` subset, and a wrong-kind (data-vector) artifact;
  - the NPCE check_npce leg (the 2026-07-12 family-wide ruling): the
    residual base + refiner algebra bitwise under the standardized
    metric, save -> rebuild -> predict composing base + net exactly in
    the {name: value} dict.

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
import yaml

from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.output import DataVectorGeometry
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry
from emulator.parameter_table import resolve_parameter_table
from emulator.experiment import EmulatorExperiment
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor
from emulator import fixed_facts

FAILURES = []

N_IN    = 3                          # input parameters
IN_NAMES  = ["omegabh2", "omegach2", "thetastar"]
OUT_NAMES = ["H0", "omegam"]         # emulated derived parameters
N_OUT   = len(OUT_NAMES)

# The adapter legs serve two scalar artifacts side by side and union what they
# provide into one theory block. They are sampled over the same inputs, in the
# same order, and they split one derived-parameter set between them: one
# dataset, so one identity, which is what handing them the same label does.
ADAPTER_PAIR_LABEL = "scalar-identity/adapter-derived-pair"

# The wrong-kind fixture: a data-vector emulator offered where a scalar one is
# required. Both places that build it build the same double for the same
# purpose, so it is one double with one identity, named once here.
DATA_VECTOR_DOUBLE_LABEL = "scalar-identity/data-vector-double"

# The region a double this gate PREDICTS THROUGH declares. An emulator now
# refuses any point outside the interval its record was drawn over, so a double
# the gate asks a question of has to stand in for a real emulator's region: a
# double that declares none is refused at the door, which is the correct answer
# for a double nobody may ask anything.
#
# This double's inputs are whitened around a standard-normal center and the
# points the gate asks about are standard-normal draws, so the region a real
# emulator of this shape would have been drawn from is the five-sigma box of
# that design. The box is the DESIGN's interval, never the smallest box the
# asked points happen to fall in: a support is the contract the dataset was
# generated under, not an observation of where the questions landed.
ROUND_TRIP_SUPPORT = {"omegabh2":  (-5.0, 5.0),
                      "omegach2":  (-5.0, 5.0),
                      "thetastar": (-5.0, 5.0)}

# A point that box does NOT contain, for the arm proving predict refuses
# outside it. It leaves the box on ONE coordinate: a point outside on every
# coordinate would also be refused by a box law that only ever looked at the
# first one.
OUTSIDE_ROUND_TRIP_BOX = {"omegabh2":  12.0,
                          "omegach2":  0.24,
                          "thetastar": -1.66}

# The NPCE double is sampled on physical cosmological scales (the C columns
# check_npce draws), so its region is the five-sigma box of THAT design: each
# column's mean +- 5 sigma, which is the interval a prior over these
# coordinates would have declared. It contains the point that leg predicts at,
# (0.0225, 0.121, 1.0412).
NPCE_SUPPORT = {"omegabh2":  (0.0214, 0.0234),
                "omegach2":  (0.110, 0.130),
                "thetastar": (1.036, 1.046)}


def supported_test_record(names, label, family, support):
    """Write this gate's fixed support bounds as literal decimal strings.

    CoCoA uses NumPy 1. A validation environment may contain NumPy 2, whose
    float32 representation includes ``np.float32(...)``. That text is not a
    decimal number. Keeping this conversion inside the synthetic gate avoids
    changing production formatting or the scientific-record digest.
    """
    blocks = yaml.safe_load(fixed_facts.synthetic_sidecar(
        names=names, label=label, family=family, support=None))
    domain = blocks[fixed_facts.INPUT_DOMAIN_GROUP]
    domain["constraint"] = "box"
    for key in ("requested", "resolved"):
        domain[key] = {
            name: [str(support[name][0]), str(support[name][1])]
            for name in names
        }
    fixed_facts.validate(blocks, where="the scalar-identity test record")
    return yaml.safe_dump(blocks, default_flow_style=False, sort_keys=False)


def report(label, ok, detail):
    """Print one PASS/FAIL line and remember any failure."""
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
    """Emit ONE '##AID <aid> <PASS|FAIL>' line for a whole acceptance leg.

    (queue 2) The board's run_check folds these reserved lines into the gate's
    executed set: one per drafted leg, at the leg's aggregation point, not per
    sub-check. A leg groups several report() calls (a check_* function or a
    guard cluster); the leg's verdict is FAIL if that group appended any label
    to the module-level FAILURES list since it started. `n_before` is
    len(FAILURES) captured just before the group ran. The child's exit status
    stays the single aggregate verdict; these lines add no new judgement.

    Arguments:
      aid      = the drafted board-unique leg id, "scalar-identity.<leg>".
      n_before = len(FAILURES) captured immediately before the leg's checks ran.
    """
    mark = "PASS" if len(FAILURES) == n_before else "FAIL"
    print("##AID " + aid + " " + mark)


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
    """The model_recipe a schema-3 save stores for the scalar ResMLP.

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


def save_synthetic_scalar(root, device, covmat_path, label, seed=0,
                          in_names=None, out_names=None, support=None):
    """Build, then save, a tiny synthetic scalar emulator under `root`.

    A ParamGeometry over the written covmat (inputs), a ScalarGeometry over
    synthetic targets (outputs), and a freshly initialized ResMLP. No
    training runs, the identity path only needs consistent weights.

    `label` is what this double is for. It fixes the identity of the scientific
    record the saved file carries; the comment at the save below says why the
    file carries one at all.

    `support` is the region the double stands for, as a mapping name -> (low,
    high). A double the gate PREDICTS THROUGH declares one, because a real
    emulator was drawn from an interval and is refused outside it. A double
    that is only saved, rebuilt, compared, or offered to an adapter that turns
    it away declares None, and then refuses every prediction — which is the
    honest record for a double nobody asks a question of, not a gap in one.

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
    # A saved emulator now carries the science it was born under. This one was
    # born under nothing: no generator produced it, so it declares itself a
    # test double rather than carrying no record at all. The label says what
    # the double is for, and it fixes the identity the record holds: doubles
    # that belong to one dataset are handed the same label, doubles that must
    # be told apart are handed different ones.
    save_emulator(path_root=str(root),
                  model=model,
                  param_geometry=pgeom,
                  geometry=geom,
                  config=config,
                  histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=recipe,
                  composition_mode="plain",
                  transfer_refined=False,
                  resolved_pce=None,
                  resolved_transfer=None,
                  facts_yaml=(fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label=label,
                      family="scalar",
                      support=None)
                    if support is None else supported_test_record(
                      names=pgeom.state()["names"],
                      label=label,
                      family="scalar",
                      support=support)),
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
                    return a confident number of the right shape and the wrong
                    value.

    Both arms are read by their WORDS. float("n/a") raises the same ValueError
    class a refusal raises, so an arm that only asked "did it raise?" would go
    green on a record that crashed instead of refusing, and the law it exists
    to prove would never have run.

    Arguments:
      root   = the round-trip double's path root. It declares ROUND_TRIP_SUPPORT
               and is the artifact the outside-the-box arm asks off its region.
      tmp    = the tempdir this gate's fixtures live in.
      device = the torch device to rebuild on.
    """
    pred = EmulatorPredictor(str(root), device, compile_model=False)
    try:
        pred.predict(OUTSIDE_ROUND_TRIP_BOX)
        report("a point outside the declared box is refused",
               False, "did not raise")
    except ValueError as e:
        report_refusal("a point outside the declared box is refused", e,
                       needle="which is outside it",
                       law="the domain law (the point leaves the box)")

    # a double that declares no support: saved, rebuilt, and then asked a
    # question it has no region to answer in.
    root_undeclared = os.path.join(tmp, "emul_undeclared")
    save_synthetic_scalar(root_undeclared, device,
                          os.path.join(tmp, "undeclared.covmat"),
                          label="scalar-identity/undeclared-support",
                          seed=300)
    pred_undeclared = EmulatorPredictor(str(root_undeclared), device,
                                        compile_model=False)
    inside = {}
    for name in IN_NAMES:
        inside[name] = 0.0
    try:
        pred_undeclared.predict(inside)
        report("a double that declares no support refuses every predict",
               False, "did not raise")
    except ValueError as e:
        report_refusal(
            "a double that declares no support refuses every predict", e,
            needle="declares no support",
            law="the domain law (no region was ever declared)")


def check_prediction_names(root, device):
    """predict() proves the NAMES of a point before it whitens one number.

    A bare row of numbers cannot say which parameter each number is, and a
    permutation of one has exactly the right length: it passes the only test a
    length is able to make, is whitened against the wrong parameter's columns,
    and is answered confidently and wrongly. Nothing about the numbers looks
    unusual afterwards. That is why the proof belongs at the door of predict()
    rather than in the eye of whoever later reads the chain.

    The two forms that carry their names are accepted here and shown to agree
    bitwise, and the three that cannot say what they mean are refused. Each
    refusal is read by its WORDS, not by "did it raise": the same call also
    passes through the domain law, which refuses in the same exception class,
    so a bare catch would let a name law go untested the day the point drifted
    outside the box.

    Arguments:
      root   = the round-trip double's path root (it declares
               ROUND_TRIP_SUPPORT, so the point below is servable and the name
               laws are the only ones this leg can be reading).
      device = the torch device to rebuild on.
    """
    pred = EmulatorPredictor(str(root), device, compile_model=False)
    # inside the declared box, so a refusal here can only be a name refusal.
    point = {"omegabh2": -0.8, "omegach2": 0.24, "thetastar": -1.66}
    values = []
    for name in pred.names:
        values.append(point[name])

    by_mapping = pred.predict(point)
    finite = len(by_mapping) == len(OUT_NAMES)
    for name in OUT_NAMES:
        finite = finite and name in by_mapping \
            and bool(np.isfinite(by_mapping[name]))
    report("a mapping predicts (the control)", finite,
           "outputs %s" % (sorted(by_mapping),))

    by_pair = pred.predict((list(pred.names), values))
    same = set(by_pair) == set(by_mapping)
    for name in by_mapping:
        same = same and by_pair[name] == by_mapping[name]
    report("an ordered (names, values) pair is the mapping's answer, bitwise",
           same, "%d output(s) identical" % len(by_mapping))

    # this one arm cannot use report_refusal: a row that carries no names at
    # all is not a bad value, it is the wrong KIND of input, so predict refuses
    # it with a TypeError and report_refusal's PASS line would announce a
    # ValueError that never happened. The needle is the same idea by hand — the
    # words of this law and no other law's.
    try:
        pred.predict(values)
        report("a bare row of numbers is refused", False, "did not raise")
    except TypeError as e:
        text = str(e)
        if "carries no parameter names" in text:
            report("a bare row of numbers is refused", True,
                   "TypeError names the unnamed-row law")
        else:
            report("a bare row of numbers is refused", False,
                   "refused the WRONG law: " + text)

    # this emulator's own names, in the wrong order, beside values that are in
    # the right one: the permutation the length can never catch.
    permuted = [pred.names[1], pred.names[0], pred.names[2]]
    try:
        pred.predict((permuted, values))
        report("this emulator's own names, permuted, are refused",
               False, "did not raise")
    except ValueError as e:
        report_refusal("this emulator's own names, permuted, are refused", e,
                       needle="in a different order",
                       law="the parameter-order law")

    foreign = ["sigma8", "ns", "tau"]
    try:
        pred.predict((foreign, values))
        report("names this emulator was never trained on are refused",
               False, "did not raise")
    except ValueError as e:
        report_refusal(
            "names this emulator was never trained on are refused", e,
            needle="not the parameters this emulator was trained on",
            law="the wrong-parameters law")


def check_from_targets_errors(device):
    """from_targets raises on an un-standardizable output column, and
    does NOT raise on a legitimate tiny-magnitude one."""
    try:
        ScalarGeometry.from_targets(
            device=device,
            targets=np.full((100, 1), 0.31, dtype="float32"),
            names=["const"])
        report("constant column raises", False, "did not raise")
    except ValueError:
        report("constant column raises (0.31)", True, "ValueError")
    # must-NOT-raise: a real tiny-magnitude output (center 1e-9, 10% spread ->
    # std ~ 1e-10, threshold 8*eps32*|center| ~ 1e-15) builds and standardizes
    # to unit variance; the relative guard passes it with orders to spare.
    tiny = np.random.default_rng(7).normal(
        1e-9, 1e-10, size=(4000, 1)).astype("float32")
    try:
        g = ScalarGeometry.from_targets(device=device, targets=tiny,
                                        names=["ok"])
        std = float(g.encode(torch.as_tensor(tiny, device=device)).std())
        report("tiny-magnitude column builds (std ~ 1)",
               abs(std - 1.0) < 0.05,
               "did not raise; standardized std = %.4f" % std)
    except ValueError as e:
        report("tiny-magnitude column builds", False,
               "unexpectedly raised: " + str(e))


def check_sidecar_errors(tmp):
    """The shared table resolver raises on duplicate normalized names."""
    params = os.path.join(tmp, "dup.txt")
    np.savetxt(params, np.asarray([[1.0, 0.0, 0.02, 70.0, 71.0]]))
    with open(os.path.join(tmp, "dup.paramnames"), "w") as f:
        f.write("omegabh2 x\nH0* a\nH0* b\n")
    try:
        resolve_parameter_table(params, ["omegabh2"], ["H0"])
        report("duplicate sidecar name raises", False, "no raise")
    except ValueError:
        report("duplicate sidecar name raises", True, "ValueError")


def check_head_architecture():
    """from_config raises on a head architecture (scalar is
    trunk-only). The head_block guard fires before the experiment is built,
    so dummy file names in the data block are enough."""
    cfg = {"data": {"train_params": "t.1.txt",
                    "val_params": "v.1.txt",
                    "train_covmat": "t.covmat",
                    "outputs": ["H0"],
                    "n_train": 10,
                    "n_val": 5,
                    "split_seed": 0},
           "train_args": {"nepochs": 1,
                          "bs": 8,
                          "model": {"name": "rescnn"}}}
    try:
        EmulatorExperiment.from_config(cfg, device=torch.device("cpu"))
        report("rescnn scalar run raises", False, "did not raise")
    except ValueError as e:
        report("rescnn scalar run raises", "trunk-only" in str(e),
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

    root = Path(__file__).resolve().parents[3]
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
    """emul_scalars: artifact-derived provides + every adapter loud error."""
    cls = _load_emul_scalars_stubbed()
    # two disjoint scalar emulators: A provides H0/omegam from one input set,
    # B provides rdrag from another.
    root_a = os.path.join(tmp, "emul_a")
    save_synthetic_scalar(root_a, device, os.path.join(tmp, "a.covmat"),
                          label=ADAPTER_PAIR_LABEL,
                          seed=10, in_names=["omegabh2", "omegach2"],
                          out_names=["H0", "omegam"])
    root_b = os.path.join(tmp, "emul_b")
    save_synthetic_scalar(root_b, device, os.path.join(tmp, "b.covmat"),
                          label=ADAPTER_PAIR_LABEL,
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
    # a second emulator of H0, built to be refused beside the first. It carries
    # the PAIR's label on purpose: two emulators of one derived parameter, both
    # trained off ONE generator dump, is exactly the ambiguity the duplicate law
    # exists to refuse -- one dataset, one identity. Give this double an
    # identity of its own instead and the served set stops being one dataset:
    # the identity law refuses it first, and the duplicate law -- the law this
    # leg exists to prove -- is never reached.
    save_synthetic_scalar(root_c, device, os.path.join(tmp, "c.covmat"),
                          label=ADAPTER_PAIR_LABEL,
                          seed=30, in_names=["omegabh2", "omegach2"],
                          out_names=["H0"])
    try:
        _build(cls, [root_a, root_c])
        report("duplicate output name raises", False, "no raise")
    except ValueError as e:
        report_refusal("duplicate output name raises", e,
                       needle="two emulators provide the output",
                       law="the duplicate-output law")

    # input/provide overlap -> loud: an emulator whose input is another's
    # output (H0 in -> would-be chain).
    root_d = os.path.join(tmp, "emul_d")
    # this one samples H0, which the pair above emulates. It is a different run
    # over a different input set, so it carries its own identity.
    save_synthetic_scalar(root_d, device, os.path.join(tmp, "d.covmat"),
                          label="scalar-identity/adapter-chained-input",
                          seed=40, in_names=["H0", "omegach2"],
                          out_names=["sigma8"])
    try:
        _build(cls, [root_a, root_d])
        report("input/provide overlap raises", False, "no raise")
    except ValueError as e:
        report_refusal("input/provide overlap raises", e,
                       needle="both an emulator input and an emulator output",
                       law="the no-chaining law")

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
    except ValueError as e:
        report_refusal("provides superset raises", e,
                       needle="provides is a subset check",
                       law="the provides-subset law")

    # wrong-kind: a data-vector artifact -> loud.
    root_dv = os.path.join(tmp, "emul_dv")
    _save_tiny_dv(root_dv, device)
    try:
        _build(cls, [root_dv])
        report("wrong-kind (dv artifact) raises", False, "no raise")
    except ValueError as e:
        report_refusal("wrong-kind (dv artifact) raises", e,
                       needle="not a scalar",
                       law="the wrong-kind law")

    # two artifacts fitted to DIFFERENT datasets -> loud. The served set is
    # unioned into one theory block, so it has to be one dataset; two runs that
    # agree on every fact and every bound still drew different points, and only
    # the identity can tell them apart.
    #
    # The pair handed over is topologically VALID on purpose: distinct outputs,
    # no input that is another's output, both scalar. The adapter runs those
    # configuration laws FIRST, so a pair that also broke one of them would be
    # refused by the earlier law and this arm's needle would be naming a law
    # that never ran.
    root_e = os.path.join(tmp, "emul_e")
    save_synthetic_scalar(root_e, device, os.path.join(tmp, "e.covmat"),
                          label="scalar-identity/adapter-foreign-dataset",
                          seed=50, in_names=["omegabh2", "omegach2"],
                          out_names=["rdrag"])
    try:
        _build(cls, [root_a, root_e])
        report("mismatched dataset identity raises", False, "no raise")
    except ValueError as e:
        report_refusal("mismatched dataset identity raises", e,
                       needle="different datasets",
                       law="the dataset-identity law")


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
    recipe = {"cls": "emulator.designs.plain.ResMLP",
              "name": "resmlp",
              "ia": None,
              "input_dim": n_in,
              "output_dim": out_dim,
              "compile_mode": None,
              "needs_geom": False,
              "kwargs": {"int_dim_res": 16,
                         "n_blocks": 2,
                         "block_opts": {"act": {"type": "H", "n_gates": 3},
                                        "norm": "affine"}}}
    # the record a test double declares: no generator produced this one, so it
    # says so rather than carrying no record at all.
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom,
                  config={"data": {"cosmolike_data_dir": "lsst_y1",
                                   "cosmolike_dataset": "d.dataset",
                                   "train_dv": "t.npy",
                                   "val_dv": "v.npy"},
                          "train_args": {"nepochs": 1}},
                  histories={"train_losses": [0.1],
                             "val_medians": [0.1],
                             "val_means": [0.1],
                             "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                             "thresholds": torch.tensor([0.2, 1.0, 10.0])},
                  train_args={"nepochs": 1}, resolved_train={"nepochs": 1},
                  resolved_model=recipe,
                  composition_mode="plain",
                  transfer_refined=False,
                  resolved_pce=None,
                  resolved_transfer=None,
                  facts_yaml=fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label=DATA_VECTOR_DOUBLE_LABEL,
                      family="cosmolike"),
                  attrs={"rescale": "none"})


def check_finetune(tmp, device):
    """Scalar fine-tuning: epoch-0 parity, outputs/wrong-kind legs, the
    anchor mask over padded extra columns."""
    from emulator import warmstart

    # a scalar source over two inputs; the run's covmat adds a third.
    root = os.path.join(tmp, "ft_src")
    src_cov = os.path.join(tmp, "ft_src.covmat")
    pgeom, geom, model = save_synthetic_scalar(
        root, device, src_cov, label="scalar-identity/finetune-source",
        seed=200,
        in_names=["omegabh2", "omegach2"], out_names=["H0", "omegam"])
    source = warmstart.load_source(root=root, device=device)
    report("load_source accepts a scalar artifact",
           type(source.geom).__name__ == "ScalarGeometry",
           "geom %s" % type(source.geom).__name__)

    # parity on the SAME covmat (no extras): build_warm_start raises on a
    # parity violation, so returning is the pass; assert bitwise anyway.
    g = np.random.default_rng(210)
    C2 = g.standard_normal((64, 2)).astype("float32")
    train_set = {"C": C2,
                 "idx": np.arange(64),
                 "C_mean": C2.mean(axis=0)}
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
    # exactly the appended column (the shared block-extension rule,
    # unchanged from the cosmolike fine-tune).
    ext_cov = os.path.join(tmp, "ft_ext.covmat")
    write_covmat(ext_cov, ["omegabh2", "omegach2", "thetastar"], seed=220)
    C3 = g.standard_normal((64, 3)).astype("float32")
    train_set3 = {"C": C3,
                  "idx": np.arange(64),
                  "C_mean": C3.mean(axis=0)}
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

    # Give the configuration check two real, self-contained chain tables.
    # The checks below are meant to stop on the fine-tune source, not on a
    # missing scientific record for the new training data.
    run_names = ["omegabh2", "omegach2", "thetastar"]
    train_params = os.path.join(tmp, "scalar_ft_train.1.txt")
    val_params = os.path.join(tmp, "scalar_ft_val.1.txt")
    rows = np.asarray([[1.0, 0.0, 0.022, 0.120, 1.041, 70.0, 0.30],
                       [1.0, 0.0, 0.023, 0.121, 1.042, 69.0, 0.31]])
    declarations = run_names + ["H0*", "omegam*"]
    for role, params_path in (("train", train_params),
                              ("validation", val_params)):
        np.savetxt(params_path, rows)
        stem = os.path.splitext(params_path)[0]
        with open(stem + ".paramnames", "w") as handle:
            for name in declarations:
                handle.write(name + " " + name.rstrip("*") + "\n")
        with open(stem + fixed_facts.SIDECAR_SUFFIX, "w") as handle:
            handle.write(fixed_facts.synthetic_sidecar(
                names=run_names,
                label="scalar-identity/finetune-" + role,
                family="scalar",
                support=None))

    # The combined from_config path must still report the outputs mismatch and
    # the wrong artifact family before it tries to stage these tables. The
    # fine-tune YAML carries no model block because the source recipe supplies
    # the architecture.
    def ft_cfg(outputs, from_root):
        return {"data": {"train_params": train_params,
                         "val_params": val_params,
                         "train_covmat": ext_cov,
                         "outputs": list(outputs),
                         "n_train": 10,
                         "n_val": 5,
                         "split_seed": 0},
                "train_args": {"nepochs": 1,
                               "bs": 8,
                               "finetune": {"from": from_root}}}
    try:
        EmulatorExperiment.from_config(ft_cfg(["H0"], root),
                                       device=torch.device("cpu"))
        report("outputs mismatch raises", False, "did not raise")
    except ValueError as e:
        report("outputs mismatch raises",
               "H0" in str(e) and "omegam" in str(e),
               "ValueError names both lists")
    dv_root = os.path.join(tmp, "ft_dv")
    _save_tiny_dv(dv_root, device)
    try:
        EmulatorExperiment.from_config(ft_cfg(["H0", "omegam"], dv_root),
                                       device=torch.device("cpu"))
        report("wrong-kind source raises", False, "did not raise")
    except ValueError as e:
        report("wrong-kind source raises",
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
    tC = torch.from_numpy(C).to(device)
    X_white = pgeom.encode(tC)
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
    config = {"data": {"train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat_path),
                       "outputs": list(OUT_NAMES)},
              "pce": {"form": "residual"},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1],
                 "val_medians": [0.1],
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
                  composition_mode="npce",
                  transfer_refined=False,
                  resolved_pce={"form": "residual",
                                "p_max": 2,
                                "r_max": 2,
                                "q": 0.5,
                                "k_max": 2,
                                "loo_max": 0.9,
                                "max_terms": 8},
                  resolved_transfer=None,
                  facts_yaml=supported_test_record(
                      names=pgeom.state()["names"],
                      label="scalar-identity/npce-derived",
                      family="scalar",
                      support=NPCE_SUPPORT),
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
    print("scalar-identity: save/rebuild/predict + loud errors")
    # seed the GLOBAL torch RNG so the synthetic nets are the same
    # every run (a red must reproduce — the run-10 bsn lesson).
    torch.manual_seed(0)
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "emul")
        # this double is predicted through, so it declares the region it stands
        # for; the doubles that are only saved, rebuilt, or refused by an
        # adapter declare none.
        pgeom, geom, model = save_synthetic_scalar(
            root, device, os.path.join(tmp, "src.covmat"),
            label="scalar-identity/round-trip", seed=0,
            support=ROUND_TRIP_SUPPORT)
        # Each drafted leg emits ONE ##AID line at its aggregation point (a
        # check_* function or a guard cluster). n0 = the FAILURES count just
        # before the leg, so emit_aid reads the leg's own verdict, not the
        # board-wide one; the six aids match the note's drafted anchor block.
        n0 = len(FAILURES)
        check_roundtrip(root, device, pgeom, geom, model)
        check_state(root, device, geom)
        check_domain_law(root, tmp, device)
        emit_aid("scalar-identity.artifact-round-trip", n0)

        n0 = len(FAILURES)
        check_from_targets_errors(device)
        check_sidecar_errors(tmp)
        check_head_architecture()
        emit_aid("scalar-identity.geometry-and-schema-guards", n0)

        n0 = len(FAILURES)
        check_adapter(tmp, device)
        emit_aid("scalar-identity.scalar-adapter-contract", n0)

        n0 = len(FAILURES)
        check_npce(tmp, device)
        emit_aid("scalar-identity.npce-composition", n0)

        n0 = len(FAILURES)
        check_finetune(tmp, device)
        emit_aid("scalar-identity.finetune-parity", n0)

        n0 = len(FAILURES)
        check_prediction_names(root, device)
        emit_aid("scalar-identity.prediction-names-are-proved", n0)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): " + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: scalar-identity all checks green")


if __name__ == "__main__":
    main()
