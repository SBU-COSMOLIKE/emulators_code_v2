#!/usr/bin/env python3
"""cs-adapter-identity gate: the cosmic-shear adapter's record laws, torch only.

The cosmic-shear adapter (cobaya_theory/emul_cosmic_shear.py) is the one adapter
of the five whose only board gate needs a real CosmoLike install and a GPU
(cobaya-adapter / gct_parity, which measures the data vector against the training
side). Everything that gate proves is worth proving, and none of it can be proved
on a laptop -- so the adapter's LOUD ERRORS, the refusals a misconfigured chain
must hit before it ever computes a chi2, have never been exercised anywhere at
all. Its four sibling adapters each have a torch-only identity gate that does
exactly that. This is the missing fifth.

It builds a tiny synthetic data-vector emulator by hand (a ParamGeometry over a
small covariance + a DataVectorGeometry over a masked 3x2pt layout + a small
ResMLP), saves it, and drives the shipped adapter over it with cobaya stubbed:

  - the adapter contract: it derives its required parameters from the artifact's
    stored geometry names (never from the YAML), serves the section the geometry
    declares, and refuses each of the four wrong-kind artifacts by name;

  - the three comparison laws, at the adapter's own site: two emulators fitted to
    different datasets are refused as a pair (the horizontal law, at the end of
    initialize); an emulator generated under another cosmology is refused when
    cobaya hands over the provider (the vertical law); a point outside the region
    the generator sampled is refused at predict, and a test double that declares
    no region at all is refused at every point (the domain law).

Every refusal leg needles the WORDS of the refusal it expects. A leg that only
asks "did it raise?" proves nothing here: float("n/a") raises ValueError, the
same class every refusal in this program uses, so a broken law and a working one
look identical from behind a bare `except ValueError`.

The real cobaya lifecycle, the real data vector, and the chi2 parity live in the
cobaya-adapter (GCT-C) board gate, which is workstation-owed. This gate is the
part that can be proved anywhere, so it is proved everywhere.
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
from emulator.results import save_emulator
from emulator.inference import EmulatorPredictor
from emulator import fixed_facts

FAILURES = []

# every leg this check is declared to produce, in the order it produces them.
LEG_AIDS = ("cs-adapter-identity.adapter-contract",
            "cs-adapter-identity.record-laws-refuse")

N_IN     = 3
IN_NAMES = ["omegabh2", "omegach2", "As"]
OUT_DIM  = 4                      # kept entries of the masked data vector
TOTAL    = 6                      # the full 3x2pt layout the mask scatters into

# The region the doubles stand for. A double that is PREDICTED through is
# standing in for a real emulator, and a real emulator was drawn from an
# interval: the box below is the interval this one declares. It is centred on
# the whitening geometry's centre (zero), so the points the legs ask about sit
# inside it and the point the out-of-region leg asks about sits outside it by
# construction, not by luck.
SUPPORT = {"omegabh2": (-1.0, 1.0),
           "omegach2": (-1.0, 1.0),
           "As":       (-1.0, 1.0)}
INSIDE   = {"omegabh2": 0.10, "omegach2": -0.20, "As": 0.30}
OUTSIDE  = {"omegabh2": 0.10, "omegach2": -0.20, "As": 4.00}

# Two doubles fitted to ONE dump share one label, and so one dataset identity.
# Two doubles that must be told apart carry their own.
ONE_DUMP_LABEL   = "cs-adapter-identity/one-dump"
OTHER_DUMP_LABEL = "cs-adapter-identity/a-second-dump"
NO_REGION_LABEL  = "cs-adapter-identity/declares-no-region"
SCALAR_LABEL     = "cs-adapter-identity/scalar-double"

# The cosmology a chain would be sampling. A test double pins every coordinate
# at "n/a", so it belongs to no cosmology at all and this model refuses it --
# which is the correct answer: a test double must never reach a likelihood.
SAMPLED_MODEL = {"mnu": 0.06, "w": -1.0, "omk": 0.0, "TCMB": 2.7255,
                 "nnu": 3.044}


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
    fixed_facts.validate(blocks, where="the cosmic-shear adapter test record")
    return yaml.safe_dump(blocks, default_flow_style=False, sort_keys=False)


def report(label, ok, detail):
    """Print one PASS/FAIL line and remember any failure."""
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


def report_refusal(label, error, needle, law):
    """Report one refusal leg: the adapter raised AND named its own law.

    A bare `except ValueError` accepts ANY refusal, and this adapter has several
    laws that refuse the same call. One of them fires first: a pair of artifacts
    whose records disagree is refused on IDENTITY before the law a leg meant to
    test is ever reached. A leg that only asks "did something raise?" goes green
    on that unrelated refusal, and the law it names goes untested forever.

    Arguments:
      label  = the leg name, as the board's evidence map carries it.
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
    """Print the board's terminal line for one declared leg."""
    mark = "PASS" if len(FAILURES) == n_before else "FAIL"
    print("##AID " + aid + " " + mark)
    return mark == "PASS"


def emit_unavailable(aid, blocker):
    """Print the reserved terminal for a declared leg that never ran."""
    if blocker is None:
        reason = "the child exited before this leg ran"
    else:
        reason = "upstream leg " + blocker + " did not pass"
    print("##AID " + aid + " UNAVAILABLE " + reason)


def spd(n, seed):
    """A random symmetric positive-definite matrix (n x n)."""
    g = np.random.default_rng(seed)
    a = g.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def save_synthetic_dv(root, device, label, support, seed=11):
    """Save one tiny data-vector emulator: the double this gate drives.

    Arguments:
      root    = the artifact's path root (<root>.h5 + <root>.emul).
      device  = the torch device to build on.
      label   = what this double is for. Two doubles sharing a label share one
                dataset identity, which is what a pair off one dump has.
      support = the region the double declares, name -> (low, high), or None for
                a double that declares no region and refuses every point.
      seed    = the seed of the synthetic covariances.

    Returns:
      None. The two files are written under root.
    """
    cov = spd(N_IN, seed=seed)
    lam, vecs = np.linalg.eigh(cov)
    pgeom = ParamGeometry(device=device, names=list(IN_NAMES),
                          center=np.zeros(N_IN), evecs=vecs,
                          sqrt_ev=np.sqrt(lam))

    cov_k = spd(OUT_DIM, seed=seed + 1)
    lam_k, vecs_k = np.linalg.eigh(cov_k)
    geom = DataVectorGeometry(device=device, total_size=TOTAL,
                              dest_idx=list(range(OUT_DIM)), evecs=vecs_k,
                              sqrt_ev=np.sqrt(lam_k),
                              Cinv=spd(TOTAL, seed=seed + 2),
                              center=np.zeros(OUT_DIM),
                              section_sizes=[TOTAL], probe="xi")

    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=OUT_DIM, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    recipe = {"cls": "emulator.designs.plain.ResMLP",
              "name": "resmlp",
              "ia": None,
              "input_dim": N_IN,
              "output_dim": OUT_DIM,
              "compile_mode": None,
              "needs_geom": False,
              "kwargs": {"int_dim_res": 16,
                         "n_blocks": 2,
                         "block_opts": {"act": {"type": "H", "n_gates": 3},
                                        "norm": "affine"}}}
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
                  facts_yaml=(fixed_facts.synthetic_sidecar(
                      names=pgeom.state()["names"],
                      label=label,
                      family="cosmolike",
                      support=None)
                    if support is None else supported_test_record(
                      names=pgeom.state()["names"],
                      label=label,
                      family="cosmolike",
                      support=support)),
                  attrs={"rescale": "none"})


def save_synthetic_scalar(root, device, label, seed=31):
    """Save a SCALAR emulator: the wrong-kind artifact this adapter must refuse.

    Arguments:
      root   = the artifact's path root.
      device = the torch device to build on.
      label  = the double's label, hence its dataset identity.
      seed   = the seed of the synthetic covariances.

    Returns:
      None.
    """
    cov = spd(N_IN, seed=seed)
    lam, vecs = np.linalg.eigh(cov)
    pgeom = ParamGeometry(device=device, names=list(IN_NAMES),
                          center=np.zeros(N_IN), evecs=vecs,
                          sqrt_ev=np.sqrt(lam))
    geom = ScalarGeometry(device=device, names=["H0", "omegam"],
                          center=np.zeros(2), scale=np.ones(2))
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=N_IN, output_dim=2, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    recipe = {"cls": "emulator.designs.plain.ResMLP",
              "name": "resmlp",
              "ia": None,
              "input_dim": N_IN,
              "output_dim": 2,
              "compile_mode": None,
              "needs_geom": False,
              "kwargs": {"int_dim_res": 16,
                         "n_blocks": 2,
                         "block_opts": {"act": {"type": "H", "n_gates": 3},
                                        "norm": "affine"}}}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom,
                  config={"data": {"train_dv": "t.npy", "val_dv": "v.npy"},
                          "train_args": {"nepochs": 1}},
                  histories={"train_losses": [0.1],
                             "val_medians": [0.1],
                             "val_means": [0.1],
                             "val_fracs": [torch.tensor([0.5, 0.4])],
                             "thresholds": torch.tensor([0.2, 1.0])},
                  train_args={"nepochs": 1}, resolved_train={"nepochs": 1},
                  resolved_model=recipe,
                  composition_mode="plain",
                  transfer_refined=False,
                  resolved_pce=None,
                  resolved_transfer=None,
                  facts_yaml=supported_test_record(
                      names=pgeom.state()["names"],
                      label=label,
                      family="scalar",
                      support=SUPPORT),
                  attrs={"rescale": "none", "outputs": "H0 omegam"})


class FakeParameterization:
    """The params block of a resolved model: the constants it states."""

    def __init__(self, constants):
        self._constants = constants

    def constant_params(self):
        """The parameters this run wrote as a plain number."""
        return self._constants


class FakeModel:
    """The resolved model a chain samples, as the adapter's law reads it.

    The law duck-types the model (that is how emulator/fixed_facts.py stays free
    of cobaya), so the smallest object carrying the two surfaces it reads is
    enough to drive it.
    """

    def __init__(self, constants):
        self.theory = {}
        self.parameterization = FakeParameterization(constants=constants)


class FakeProvider:
    """The cobaya Provider, which carries the resolved model.

    Verified against the installed cobaya 3.6.2: Provider.__init__(self, model,
    requirement_providers) stores the model, so an adapter reaches it through
    self.provider.model at initialize_with_provider time.
    """

    def __init__(self, model):
        self.model = model


class ProviderWithoutModel:
    """A future cobaya whose Provider no longer carries the model.

    The adapter must refuse loudly here rather than skip the law: a law that
    skips itself when it cannot run is not a law, and the chain would sample on
    with an emulator nobody proved belonged to it.
    """


def load_emul_cosmic_shear_stubbed():
    """Load the shipped emul_cosmic_shear.py with cobaya stubbed out.

    This gate is torch-only, so cobaya and cobaya.theory are stubbed before the
    adapter module is executed: the stub Theory carries the two lifecycle hooks
    the adapter overrides (initialize, initialize_with_provider), and nothing
    else. EmulatorPredictor stays real -- the artifacts, the record and the laws
    are the point of the gate, and stubbing any of them would prove nothing.

    Returns:
      the emul_cosmic_shear class, ready to instantiate.
    """
    if "cobaya" not in sys.modules:
        sys.modules["cobaya"] = types.ModuleType("cobaya")
    theory_mod = types.ModuleType("cobaya.theory")

    class _Theory:
        renames = {}
        extra_args = {}

        def initialize(self):
            pass

        def initialize_with_provider(self, provider):
            self.provider = provider

    theory_mod.Theory = _Theory
    sys.modules["cobaya.theory"] = theory_mod

    root = Path(__file__).resolve().parents[3]
    path = root / "cobaya_theory" / "emul_cosmic_shear.py"
    spec = importlib.util.spec_from_file_location("emul_cosmic_shear_shim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.emul_cosmic_shear


def build(cls, roots, dv_return=None):
    """Instantiate the stubbed adapter over these roots and initialize it.

    Arguments:
      cls       = the emul_cosmic_shear class.
      roots     = the saved-emulator path roots to serve.
      dv_return = 'section' | '3x2pt', or None to leave it at the default.

    Returns:
      the initialized theory instance.
    """
    theory = cls()
    theory.extra_args = {"device": "cpu", "emulators": list(roots)}
    if dv_return is not None:
        theory.extra_args["dv_return"] = dv_return
    theory.initialize()
    return theory


def check_adapter_contract(tmp, device):
    """The adapter reads its configuration from the artifacts, and refuses the
    kinds that are not its own.

    The requirements a chain must supply are the emulator's own stored parameter
    names, and the vector it serves is the section the stored geometry declares.
    Neither is restated in the YAML: an emulator file that had to be described in
    a config could be described wrongly, and the description would win.

    Returns:
      None.
    """
    cls = load_emul_cosmic_shear_stubbed()

    root = os.path.join(tmp, "cs_one")
    save_synthetic_dv(root=root, device=device, label=ONE_DUMP_LABEL,
                      support=SUPPORT)
    theory = build(cls, [root])

    report("the required parameters come from the artifact's geometry",
           sorted(theory.get_requirements()) == sorted(IN_NAMES),
           "requirements = " + repr(sorted(theory.get_requirements())))

    state = {}
    ok_calc = theory.calculate(state, want_derived=False, **INSIDE)
    theory.current_state = state
    served = theory.get_cosmic_shear()
    report("calculate serves the section the geometry declares",
           ok_calc is True and served.shape == (TOTAL,)
           and np.all(np.isfinite(served)),
           "section shape " + repr(tuple(served.shape)) + ", all finite")

    full = build(cls, [root], dv_return="3x2pt")
    state = {}
    full.calculate(state, want_derived=False, **INSIDE)
    full.current_state = state
    scattered = full.get_cosmic_shear()
    report("dv_return '3x2pt' scatters into the full layout",
           scattered.shape == (TOTAL,)
           and np.count_nonzero(scattered[OUT_DIM:]) == 0,
           "the entries off the mask are zero")

    # the wrong-kind guard. A scalar artifact returns a {name: value} dict, not
    # a vector, so letting it through would hand the likelihood an object of the
    # wrong shape -- silently, at the first sampled point.
    scalar_root = os.path.join(tmp, "cs_scalar")
    save_synthetic_scalar(root=scalar_root, device=device, label=SCALAR_LABEL)
    try:
        build(cls, [scalar_root])
        report("a scalar artifact is refused by the wrong-kind guard", False,
               "no raise")
    except ValueError as exc:
        report_refusal("a scalar artifact is refused by the wrong-kind guard",
                       exc,
                       needle="belongs in emul_scalars' emulators list",
                       law="the wrong-kind law")


def check_record_laws(tmp, device):
    """The three comparison laws, at the cosmic-shear adapter's own site.

    One artifact against the chain (vertical), two artifacts against each other
    (horizontal), one point against the region the generator sampled (domain).
    The adapter is where the first two are asked, because it is where the served
    set and the sampled cosmology first exist in one place; the third is asked at
    every prediction, because that is where a point first exists.

    Returns:
      None.
    """
    cls = load_emul_cosmic_shear_stubbed()

    one = os.path.join(tmp, "law_one")
    twin = os.path.join(tmp, "law_twin")
    other = os.path.join(tmp, "law_other")
    save_synthetic_dv(root=one, device=device, label=ONE_DUMP_LABEL,
                      support=SUPPORT)
    save_synthetic_dv(root=twin, device=device, label=ONE_DUMP_LABEL,
                      support=SUPPORT, seed=21)
    save_synthetic_dv(root=other, device=device, label=OTHER_DUMP_LABEL,
                      support=SUPPORT, seed=41)

    # HORIZONTAL. Two halves off one dump share a dataset identity and may be
    # served together; a half from another dump may not, and no comparison of
    # the FACTS could have caught it -- two runs of one YAML with a fresh seed
    # agree on every fact and every bound, and drew different points.
    pair = build(cls, [one, twin])
    report("two emulators off one dump are served together",
           len(pair.predictors) == 2,
           "one dataset identity, one pair")

    try:
        build(cls, [one, other])
        report("two emulators off different dumps are refused as a pair", False,
               "no raise")
    except ValueError as exc:
        report_refusal("two emulators off different dumps are refused as a pair",
                       exc,
                       needle="different datasets",
                       law="the dataset-identity law")

    # VERTICAL. The chain hands its provider over once, at setup. A test double
    # pins every coordinate at "n/a", so it belongs to no cosmology, and a real
    # chain is exactly the thing it must never be served to.
    theory = build(cls, [one])
    try:
        theory.initialize_with_provider(
            FakeProvider(model=FakeModel(constants=SAMPLED_MODEL)))
        report("an artifact born under another cosmology is refused at setup",
               False, "no raise")
    except ValueError as exc:
        report_refusal("an artifact born under another cosmology is refused at "
                       "setup",
                       exc,
                       needle="held fixed at",
                       law="the fixed-cosmology law")

    # the same law, when the provider itself cannot answer. A cobaya that no
    # longer carries the model must stop the chain, not silently void the law.
    theory = build(cls, [one])
    try:
        theory.initialize_with_provider(ProviderWithoutModel())
        report("a provider that cannot reach the model refuses loudly", False,
               "no raise")
    except ValueError as exc:
        report_refusal("a provider that cannot reach the model refuses loudly",
                       exc,
                       needle="carries no .model",
                       law="the API-drift law")

    # DOMAIN. Inside the declared region the emulator answers; outside it, it
    # would extrapolate -- a number of the right shape, the right sign, and no
    # warning of any kind. That is the number this refusal exists to withhold.
    theory = build(cls, [one])
    state = {}
    theory.calculate(state, want_derived=False, **INSIDE)
    theory.current_state = state
    report("a point inside the declared region is served",
           np.all(np.isfinite(theory.get_cosmic_shear())),
           "As = 0.3, inside [-1, 1]")

    try:
        theory.calculate({}, want_derived=False, **OUTSIDE)
        report("a point outside the declared region is refused", False,
               "no raise -- the emulator extrapolated")
    except ValueError as exc:
        report_refusal("a point outside the declared region is refused", exc,
                       needle="which is outside it",
                       law="the domain law")

    # a double that declares NO region cannot be asked about any point: its
    # bounds are not wide, they are absent. This is the arm that would go green
    # through a broken law if it only asked "did it raise?" -- reading the
    # absent bound "n/a" as a number raises ValueError too, which is why the
    # needle names the designed words instead.
    nowhere = os.path.join(tmp, "law_nowhere")
    save_synthetic_dv(root=nowhere, device=device, label=NO_REGION_LABEL,
                      support=None, seed=51)
    predictor = EmulatorPredictor(nowhere, device, compile_model=False)
    try:
        predictor.predict(INSIDE)
        report("a double that declares no region is refused at every point",
               False, "no raise -- the test double was served")
    except ValueError as exc:
        report_refusal("a double that declares no region is refused at every "
                       "point",
                       exc,
                       needle="declares no support",
                       law="the no-declared-region law")


def main():
    """Run every leg, and account for every leg the board expects.

    Returns:
      0 when every leg passed, 1 otherwise.
    """
    print("cs-adapter-identity: the cosmic-shear adapter's contract + the "
          "three record laws")
    device = torch.device("cpu")
    tmp = tempfile.mkdtemp(prefix="cs-adapter-identity-")
    emitted = set()
    blocker = None
    try:
        print("\n-- the adapter reads its configuration from the artifacts --")
        n0 = len(FAILURES)
        check_adapter_contract(tmp, device)
        aid = "cs-adapter-identity.adapter-contract"
        if not emit_aid(aid, n0):
            blocker = aid
        emitted.add(aid)

        print("\n-- the record's three laws refuse at the adapter's site --")
        n0 = len(FAILURES)
        check_record_laws(tmp, device)
        aid = "cs-adapter-identity.record-laws-refuse"
        if not emit_aid(aid, n0):
            blocker = aid
        emitted.add(aid)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        for aid in LEG_AIDS:
            if aid not in emitted:
                emit_unavailable(aid, blocker)

    print("")
    if len(FAILURES) == 0:
        print("cs-adapter-identity: ALL PASS")
        return 0
    print("cs-adapter-identity: " + str(len(FAILURES)) + " FAILURE(S): "
          + ", ".join(FAILURES))
    return 1


if __name__ == "__main__":
    sys.exit(main())
