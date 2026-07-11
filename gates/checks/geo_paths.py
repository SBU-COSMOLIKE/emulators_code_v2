#!/usr/bin/env python3
"""geo-paths gate (GEO-A): the geometry folder move is artifact-immune.

Every schema-v2 artifact persists its geometry classes as FULL module
paths and rebuild_emulator dispatches by importing exactly the stored
string. The GEO unit moved the geometry modules into
emulator/geometries/ behind flat legacy shims; this gate proves the
three contracts:

  1  OLD-PATH REBUILD: a fresh artifact's h5 has its geometry cls
     markers rewritten to the OLD flat paths
     (emulator.geometries_<name>.<Class>) — exactly what every artifact
     saved before the move carries — and must rebuild AND predict
     bitwise-identically to the untouched artifact (the shims route the
     import to the one class object, so isinstance stays sound).
  2  NEW-SAVE MARKERS: a fresh save writes the NEW folder paths
     (emulator.geometries.<name>.<Class>) automatically — the
     resolved-values rule doing the work through type().__module__.
  3  SHIM IDENTITY + CENSUS: each legacy shim's classes ARE the folder
     modules' class objects (import alias, not a copy), and no repo
     code outside the shims references the old flat module names.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import h5py

from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry
from emulator.results import save_emulator
from emulator.inference import EmulatorPredictor

FAILURES = []
REPO = Path(__file__).resolve().parents[2]
IN_NAMES = ["omegabh2", "omegach2"]
OUT_NAMES = ["H0", "omegam"]


def report(label, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
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
    recipe = {"cls": "emulator.designs.plain.ResMLP", "name": "resmlp",
              "ia": None, "input_dim": 2, "output_dim": 2,
              "compile_mode": None, "needs_geom": False,
              "kwargs": {"int_dim_res": 16, "n_blocks": 2,
                         "block_opts": {"act": {"type": "H",
                                                "n_gates": 3},
                                        "norm": "affine"}}}
    config = {"data": {"train_params": "t.1.txt", "val_params": "v.1.txt",
                       "train_covmat": os.path.basename(covmat),
                       "outputs": list(OUT_NAMES)},
              "train_args": {"nepochs": 1}}
    histories = {"train_losses": [0.1], "val_medians": [0.1],
                 "val_means": [0.1],
                 "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
                 "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1}, resolved_model=recipe,
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
    print("geo-paths (GEO-A): old-path rebuild + new-save markers + "
          "shim identity/census")
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "emul_geo")
        save_fixture(root, device, tmp)

        # leg 2 first: the fresh save writes NEW folder paths.
        attrs = geometry_cls_attrs(root + ".h5")
        new_style = [v for _, _, v in attrs
                     if v.startswith("emulator.geometries.")]
        old_style = [v for _, _, v in attrs
                     if v.startswith("emulator.geometries_")]
        report("fresh save writes the NEW folder cls paths",
               len(new_style) >= 2 and not old_style,
               "%d new-path marker(s): %s" % (len(new_style),
                                              sorted(set(new_style))))

        # the reference prediction from the untouched artifact.
        point = {"omegabh2": 0.0223, "omegach2": 0.119}
        ref = EmulatorPredictor(root, device, compile_model=False)
        want = ref.predict(point)

        # leg 1: rewrite the markers to the OLD flat paths (what every
        # pre-GEO artifact carries) and rebuild + predict bitwise.
        old_root = os.path.join(tmp, "emul_geo_oldpaths")
        for ext in (".h5", ".emul"):
            with open(root + ext, "rb") as fsrc, \
                 open(old_root + ext, "wb") as fdst:
                fdst.write(fsrc.read())
        n_rewritten = 0
        with h5py.File(old_root + ".h5", "r+") as f:
            def rewrite(name, obj):
                nonlocal n_rewritten
                for key, val in list(obj.attrs.items()):
                    sval = val.decode() if isinstance(val, bytes) else val
                    if (isinstance(sval, str)
                            and sval.startswith("emulator.geometries.")):
                        obj.attrs[key] = sval.replace(
                            "emulator.geometries.",
                            "emulator.geometries_", 1)
                        n_rewritten += 1
            f.visititems(rewrite)
        pred_old = EmulatorPredictor(old_root, device,
                                     compile_model=False)
        got = pred_old.predict(point)
        ok = (set(got) == set(want)
              and all(got[k] == want[k] for k in want))
        report("OLD-path artifact rebuilds + predicts bitwise",
               ok and n_rewritten >= 2,
               "%d marker(s) rewritten; max|d| = %.1e" % (
                 n_rewritten,
                 max((abs(got[k] - want[k]) for k in want), default=0.0)))

    # leg 3a: shim identity (import alias, not a copy).
    import importlib
    ok = True
    for mod, cls_names in (("parameter", ["ParamGeometry"]),
                           ("output", ["DataVectorGeometry"]),
                           ("scalar", ["ScalarGeometry"]),
                           ("cmb", ["CmbDiagonalGeometry"]),
                           ("grid", ["GridGeometry"]),
                           ("grid2d", ["Grid2DGeometry"])):
        try:
            shim = importlib.import_module("emulator.geometries_" + mod)
            new = importlib.import_module("emulator.geometries." + mod)
        except Exception as e:
            # geometries_output imports cosmolike; off-workstation the
            # shim must still be import-consistent when the new module
            # is (both fail together = consistent).
            try:
                importlib.import_module("emulator.geometries." + mod)
                ok = False
                print("  shim import failed but the new module works:",
                      mod, e)
            except Exception:
                pass
            continue
        for cn in cls_names:
            if getattr(shim, cn) is not getattr(new, cn):
                ok = False
                print("  shim class is a COPY, not an alias:", mod, cn)
    report("shims alias the folder classes (isinstance sound)", ok, "")

    # leg 3b: the shim-import census — no repo code outside the shims
    # references the old flat module names.
    offenders = []
    for base, dirs, files in os.walk(REPO):
        if "__pycache__" in base or "/notes" in base \
                or "/.git" in base:
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(base, fname)
            rel = os.path.relpath(path, REPO)
            if rel.startswith("emulator/geometries_") \
                    or rel == "gates/checks/geo_paths.py":
                continue
            src = open(path, errors="replace").read()
            for m in ("parameter", "output", "scalar", "cmb", "grid",
                      "grid2d"):
                if "geometries_" + m in src:
                    offenders.append((rel, m))
    report("shim census: new code never imports the old flat paths",
           not offenders, repr(offenders[:5]))

    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: geo-paths all checks green")


if __name__ == "__main__":
    main()
