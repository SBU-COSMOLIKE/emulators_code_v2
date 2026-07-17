#!/usr/bin/env python3
"""geo-paths gate: the geometry folder is the ONLY geometry home.

The GEO unit moved the geometry modules into emulator/geometries/. The
move originally shipped with flat legacy shims (emulator/geometries_
<name>.py) so that artifacts saved before the move — which persist
their geometry classes as FULL module paths and rebuild by importing
exactly the stored string — would keep loading. On 2026-07-11 the user
ruled that no real (science) artifact predates the move, so the shims
were retired and the contract inverted. This gate now proves:

  1  NEW-SAVE MARKERS: a fresh save writes the folder paths
     (emulator.geometries.<name>.<Class>) — the resolved-values rule
     doing the work through type().__module__ — and the artifact
     rebuilds and predicts through those stored paths.
  2  OLD PATHS ARE DEAD, LOUDLY: importing any of the six flat legacy
     modules raises ModuleNotFoundError (no silent half-alive shim),
     and none of the flat files exist on disk.
  3  CENSUS: no repo .py outside this gate references the old flat
     module names.

A pre-retirement artifact (if one ever surfaces) fails at rebuild with
the module path in the error — the loud death this gate pins down.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ai.gates.checks.artifact_fixtures import one_pass_training_recipe
import h5py
import yaml

from emulator import fixed_facts
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry
from emulator.results import save_emulator
from emulator.inference import EmulatorPredictor

# the board harness supplies the ONE shared whole-repo .py enumerator this
# gate's folder census scans -- the SAME function the board hashes as this
# gate's data-read surface (25M-16), so the scanned set and the digested set can
# never diverge (they are one function, not two lists).
_GATES = Path(__file__).resolve().parents[1]
if str(_GATES) not in sys.path:
    sys.path.insert(0, str(_GATES))
import run_board

FAILURES = []
REPO = Path(__file__).resolve().parents[3]
IN_NAMES = ["omegabh2", "omegach2"]
OUT_NAMES = ["H0", "omegam"]
LEGACY_MODULES = ["parameter", "output", "scalar", "cmb", "grid",
                  "grid2d"]


def report(label, ok, detail, aid=None):
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    # (queue 2) the per-leg assertion manifest the board folds into this gate's
    # executed set: one reserved '##AID <aid> <result>' line per acceptance leg.
    # The child's exit status stays the single aggregate verdict, not a leg.
    if aid is not None:
        print("##AID " + aid + " " + mark)
    if not ok:
        FAILURES.append(label)


def write_covmat(path, names, seed):
    g = np.random.default_rng(seed)
    a = g.standard_normal((len(names), len(names)))
    cov = a @ a.T + len(names) * np.eye(len(names))
    with open(path, "w") as f:
        f.write("# " + " ".join(names) + "\n")
        for row in cov:
            f.write(" ".join(repr(float(x)) for x in row) + "\n")


def supported_test_record(names, label, family, support):
    """Give a prediction fixture a literal, finite support box.

    CoCoA uses NumPy 1. The validation environment may contain NumPy 2, whose
    float32 ``repr`` is Python code rather than a bare decimal. This gate keeps
    production's decimal policy untouched and writes its few fixed test bounds
    as literal decimal strings instead.
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
    fixed_facts.validate(blocks, where="the geometry-path test record")
    return yaml.safe_dump(blocks, default_flow_style=False, sort_keys=False)


def save_fixture(root, device, tmp):
    """A tiny scalar artifact (the smallest full save/rebuild cycle)."""
    covmat = os.path.join(tmp, "geo.covmat")
    write_covmat(covmat, IN_NAMES, seed=1)
    pgeom = ParamGeometry.from_covmat(
        device=device, center=np.array([0.022, 0.12]),
        covmat_path=covmat)
    g = np.random.default_rng(2)
    targets = g.normal(loc=[70.0, 0.3], scale=[3.0, 0.02],
                       size=(2000, 2)).astype("float32")
    geom = ScalarGeometry.from_targets(device=device, targets=targets,
                                       names=OUT_NAMES)
    block_opts = {"act": make_activation("H", n_gates=3),
                  "norm": make_norm("affine")}
    model = ResMLP(input_dim=2, output_dim=2, int_dim_res=16,
                   n_blocks=2, block_opts=block_opts).to(device)
    recipe = {"cls": "emulator.designs.plain.ResMLP",
              "name": "resmlp",
              "ia": None,
              "input_dim": 2,
              "output_dim": 2,
              "compile_mode": None,
              "needs_geom": False,
              "kwargs": {"int_dim_res": 16,
                         "n_blocks": 2,
                         "block_opts": {"n_layers": 2,
                                        "act": {"type": "H",
                                                "n_gates": 3},
                                        "norm": "affine"}}}
    config = {"data": {"train_params": "t.1.txt",
                       "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat),
                       "outputs": list(OUT_NAMES)},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1],
                 "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train=one_pass_training_recipe(
                    thresholds=(0.2, 1.0, 10.0, 100.0)),
                  resolved_model=recipe,
                  composition_mode="plain", transfer_refined=False,
                  resolved_pce=None, resolved_transfer=None,
                  facts_yaml=supported_test_record(
                      names=pgeom.state()["names"],
                      label="geometry-paths-fixture",
                      family="scalar",
                      support={"omegabh2": (0.01, 0.04),
                               "omegach2": (0.05, 0.20)}),
                  attrs={"rescale": "none",
                         "outputs": " ".join(OUT_NAMES)})


def geometry_cls_attrs(h5_path):
    """Every attr value in the h5 that names a geometry module path."""
    found = []

    def visit(name, obj):
        for key, val in obj.attrs.items():
            if isinstance(val, bytes):
                val = val.decode()
            if isinstance(val, str) and "emulator.geometries" in val:
                found.append((name, key, val))

    with h5py.File(h5_path, "r") as f:
        f.visititems(visit)
        for key, val in f.attrs.items():
            if isinstance(val, bytes):
                val = val.decode()
            if isinstance(val, str) and "emulator.geometries" in val:
                found.append(("/", key, val))
    return found


def main():
    print("geo-paths: new-save markers + dead legacy paths + "
          "census")
    device = torch.device("cpu")

    # leg 1: the fresh save writes folder paths, and the artifact
    # rebuilds + predicts through exactly those stored strings.
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "emul_geo")
        save_fixture(root, device, tmp)
        attrs = geometry_cls_attrs(root + ".h5")
        new_style = [v for _, _, v in attrs
                     if v.startswith("emulator.geometries.")]
        old_style = [v for _, _, v in attrs
                     if v.startswith("emulator.geometries_")]
        pred = EmulatorPredictor(root, device, compile_model=False)
        out = pred.predict({"omegabh2": 0.0223, "omegach2": 0.119})
        finite = all(np.isfinite(out[k]) for k in OUT_NAMES)
        report("fresh save writes folder cls paths + rebuilds",
               len(new_style) >= 2 and not old_style and finite,
               "%d folder-path marker(s): %s; predict finite: %s" % (
                 len(new_style), sorted(set(new_style)), finite),
               aid="geo-paths.fresh-save-uses-folder-paths")

    # leg 2: the six flat legacy modules are gone — from disk AND from
    # the import system (a stored old path now dies with the module
    # name in the error, never a silent partial load).
    ok = True
    details = []
    for mod in LEGACY_MODULES:
        flat = REPO / "emulator" / ("geometries_%s.py" % mod)
        if flat.exists():
            ok = False
            details.append("%s exists on disk" % flat.name)
        # find_spec probes the import system WITHOUT importing, and is neither
        # import_module nor __import__, so the manifest census does not see a
        # dynamic import here (this site tests NON-existence -- a static import
        # of a deleted module cannot be written, and there is no module to hash;
        # find_spec is the clean existence probe). A live spec = the legacy
        # path is back.
        if importlib.util.find_spec("emulator.geometries_" + mod) is not None:
            ok = False
            details.append("emulator.geometries_%s importable" % mod)
    report("legacy flat paths are dead (disk + import)", ok,
           "; ".join(details) if details else
           "all %d absent from the import system" % len(LEGACY_MODULES),
           aid="geo-paths.legacy-flat-paths-absent")

    # leg 3: the census — no repo code references the old flat names. The file
    # set is the SHARED whole-repo enumerator (run_board.repo_py_files), the same
    # function the board hashes as this gate's data-read surface, so the scanned
    # set is exactly the digested set. The gate skips its OWN source (it names
    # the flat modules in this census logic).
    offenders = []
    for rel in run_board.repo_py_files():
        if rel == "ai/gates/checks/geo_paths.py":
            continue
        src = open(os.path.join(REPO, rel), errors="replace").read()
        for m in LEGACY_MODULES:
            if "geometries_" + m in src:
                offenders.append((rel, m))
    report("census: nothing references the old flat names",
           not offenders, repr(offenders[:5]),
           aid="geo-paths.legacy-reference-census")

    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: geo-paths all checks green")


if __name__ == "__main__":
    main()
