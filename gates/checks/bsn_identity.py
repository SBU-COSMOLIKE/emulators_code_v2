#!/usr/bin/env python3
"""bsn-identity gate (BSN-A): the grid-emulator save/rebuild/predict
identity, the imposed-physics distance pipeline, the target law both
ways, the emul_baosn window/piecewise legs, the D-BSN9 finetune parity,
and every grid-path loud error — torch + scipy, no CAMB.

Legs:
  - cumulative_simpson: EVEN doubled-grid points exact on cubics (the
    original z grid sits there), the odd half-chunk step bounded (the
    recorded legacy approximation), the even-point-count guard;
  - the distance pipeline (real scipy cubic) against a closed-form flat
    LCDM reference at 1e-6 across the window;
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
  - D-BSN9: warm-start epoch-0 parity from a grid source; the
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
from emulator import warmstart

FAILURES = []
IN_NAMES = ["omegam", "H0", "w"]
N_IN = len(IN_NAMES)


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


def lcdm_h(z, H0=67.36, om=0.315):
    return H0 * np.sqrt(om * (1 + z) ** 3 + (1 - om))


def check_simpson():
    z = np.linspace(0.0, 3.0, 601)
    y = 2.0 * z ** 3 - z ** 2 + 4.0 * z + 1.0
    truth = 0.5 * z ** 4 - z ** 3 / 3.0 + 2.0 * z ** 2 + z
    got = cumulative_simpson(z, y)
    e_even = np.abs(got[::2] - truth[::2]).max()
    e_odd = np.abs(got[1::2] - truth[1::2]).max()
    ok = e_even < 1e-9 and e_odd < 1e-3 and got[0] == 0.0
    try:
        cumulative_simpson(z[:-1], y[:-1])
        ok = False
    except ValueError:
        pass
    report("Simpson: even points exact on cubics, odd bounded, guard",
           ok, "even %.1e odd %.1e" % (e_even, e_odd))


def check_pipeline():
    z_grid = np.linspace(0.001, 3.0, 600)
    h_grid = lcdm_h(z_grid)
    itp = distance_interpolators(z_grid=z_grid, h_grid=h_grid)
    zr = np.linspace(0.0, 3.0, 120001)
    chi_ref = cumulative_simpson(zr, C_KMS / lcdm_h(zr))
    ok, detail = True, ""
    for zq in (0.1, 0.5, 1.0, 2.0, 2.9):
        want = np.interp(zq, zr, chi_ref)
        rel = abs(float(itp["chi"](zq)) - want) / want
        rel_da = abs(float(itp["da"](zq)) - want / (1 + zq)) \
            / (want / (1 + zq))
        rel_dl = abs(float(itp["dl"](zq)) - want * (1 + zq)) \
            / (want * (1 + zq))
        if max(rel, rel_da, rel_dl) > 1e-6:
            ok, detail = False, "z=%s rel %.1e" % (zq, rel)
    report("pipeline vs closed-form flat LCDM (cubic, 1e-6)", ok, detail)


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


def save_synthetic_grid(root, device, tmp, quantity="Hubble",
                        units="km/s/Mpc", law="log_offset", offset=1.0,
                        z=None, seed=0):
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
    save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                  geometry=geom, config=config, histories=histories,
                  train_args=config["train_args"],
                  resolved_train={"nepochs": 1},
                  resolved_model=grid_recipe(len(z)),
                  attrs={"rescale": "none", "quantity": quantity})
    return pgeom, geom, model, covmat


def check_roundtrip(tmp, device, law):
    root = os.path.join(tmp, "emul_grid_" + law)
    off = 1.0 if law == "log_offset" else 0.0
    pgeom, geom, model, _ = save_synthetic_grid(
        root, device, tmp, law=law, offset=off, seed=30)
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
    root = Path(__file__).resolve().parents[2]
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
    root_h = os.path.join(tmp, "ad_h")
    save_synthetic_grid(root_h, device, tmp, quantity="Hubble",
                        units="km/s/Mpc", law="log_offset", offset=1.0,
                        z=np.linspace(0.001, 3.0, 64), seed=40)
    root_dm = os.path.join(tmp, "ad_dm")
    save_synthetic_grid(root_dm, device, tmp, quantity="D_M",
                        units="Mpc", law="none", offset=0.0,
                        z=np.linspace(1000.0, 1200.0, 24), seed=50)

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
        report("desert must_provide raises",
               "never emulated" in str(e), "ValueError names the desert")

    # calculate + the piecewise getters vs the pipeline / the artifact
    point = {"omegam": 0.31,
             "H0": 67.0,
             "w": -1.0}
    state = {}
    t.calculate(state, want_derived=True, **point)
    t.current_state = state
    out_h = t.p_h.predict(point)
    itp = distance_interpolators(z_grid=out_h["z"], h_grid=out_h["Hubble"])
    zq = np.array([0.3, 1.5])
    ok = np.allclose(t.get_comoving_radial_distance(zq), itp["chi"](zq),
                     rtol=0, atol=0)
    ok = ok and np.allclose(t.get_angular_diameter_distance(zq),
                            itp["chi"](zq) / (1 + zq))
    ok = ok and np.allclose(t.get_luminosity_distance(zq),
                            itp["chi"](zq) * (1 + zq))
    out_dm = t.p_dm.predict(point)
    zr = np.array([1090.0])
    want_dm = np.interp(zr, out_dm["z"], out_dm["D_M"])
    got_dm = t.get_comoving_radial_distance(zr)
    ok = ok and np.allclose(got_dm, want_dm, rtol=1e-3)
    pair = t.get_angular_diameter_distance_2([[0.3, 1.5]])
    want_pair = (itp["chi"](1.5) - itp["chi"](0.3)) / 2.5
    ok = ok and np.allclose(pair, [want_pair])
    report("piecewise getters match the pipeline / artifact", ok,
           "chi/da/dl + rec D_M + D_A2")
    # H units + the H-outside-SN loud error
    h1 = t.get_Hubble(np.array([1.0]))
    h2 = t.get_Hubble(np.array([1.0]), units="1/Mpc")
    ok = np.allclose(h1 / C_KMS, h2)
    try:
        t.get_Hubble(np.array([1090.0]))
        ok = False
    except ValueError:
        pass
    try:
        t.get_Hubble(np.array([1.0]), units="parsecs")
        ok = False
    except ValueError:
        pass
    report("get_Hubble units + window guards", ok, "km/s/Mpc vs 1/Mpc")
    # desert getter leg
    try:
        t.get_comoving_radial_distance(np.array([500.0]))
        report("desert getter raises", False, "no raise")
    except ValueError:
        report("desert getter raises", True, "ValueError")

    # pair-validation legs
    try:
        _build(cls, [root_h])
        report("missing D_M raises", False, "no raise")
    except ValueError:
        report("missing D_M raises", True, "ValueError")
    root_h2 = os.path.join(tmp, "ad_h2")
    save_synthetic_grid(root_h2, device, tmp, quantity="Hubble",
                        units="km/s/Mpc", law="none", offset=0.0,
                        z=np.linspace(0.001, 2.0, 32), seed=60)
    try:
        _build(cls, [root_h, root_h2])
        report("duplicate quantity raises", False, "no raise")
    except ValueError:
        report("duplicate quantity raises", True, "ValueError")


def check_finetune(tmp, device):
    root = os.path.join(tmp, "ft_grid_src")
    pgeom, geom, model, covmat = save_synthetic_grid(
        root, device, tmp, law="log_offset", offset=1.0, seed=70)
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
        report("D-BSN9 metadata mismatch raises", False, "no raise")
    except ValueError as e:
        report("D-BSN9 metadata mismatch raises",
               "grid-metadata mismatch" in str(e), "ValueError")
    dm_root = os.path.join(tmp, "ft_dm_src")
    save_synthetic_grid(dm_root, device, tmp, quantity="D_M",
                        units="Mpc", law="none", offset=0.0,
                        z=np.linspace(1000.0, 1200.0, 24), seed=90)
    try:
        EmulatorExperiment.from_config(ft_cfg(good, dm_root),
                                       device=torch.device("cpu"))
        report("D-BSN9 cross-quantity source raises", False, "no raise")
    except ValueError as e:
        report("D-BSN9 cross-quantity source raises",
               "grid-metadata mismatch" in str(e), "ValueError")


def main():
    print("bsn-identity (BSN-A): pipeline + law + round-trip + adapter "
          "+ finetune legs")
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        check_simpson()
        check_pipeline()
        check_geometry(device)
        check_roundtrip(tmp, device, law="log_offset")
        check_roundtrip(tmp, device, law="none")
        check_npce(tmp, device)
        check_adapter(tmp, device)
        check_finetune(tmp, device)
    if FAILURES:
        print("FAIL: " + str(len(FAILURES)) + " check(s): "
              + ", ".join(FAILURES))
        sys.exit(1)
    print("PASS: bsn-identity all checks green")


if __name__ == "__main__":
    main()
