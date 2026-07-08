#!/usr/bin/env python3
"""GSV-C: save schema v2 acceptance (bitwise + drift + v1 refusal).

Trains three tiny emulators in-process (one plain, one factored ia:nla,
one NPCE pce), saves each, rebuilds from the file alone, and requires
the rebuilt model's output BITWISE-EQUAL to the live model's on a probe
batch (home note save-schema-resolved-config.md:86-93; one factored +
one NPCE save per workstation-board-2026-07.md:66-71 so the
geometry-class marker and the pce group both round-trip). Then the
drift proof: monkeypatch a code default, rebuild again, still identical
(the file, not the code, defines the emulator). Finally a v1 file is
refused. Prints every value it compares and exits nonzero on failure.

The training assembly is the driver's own path read from experiment.py:
EmulatorExperiment.from_config -> run -> save_emulator -> rebuild.
Deploy paths (the cocoa root and the dump directory) come from
gates/board_config.json; the data-directory convention (<root>/chains,
per the driver) is the one workstation-diagnostic assumption, corrected
from the first raw log if the deploy differs.

PS: bitwise = torch.equal, the model outputs match to the last bit;
drift proof = a rebuild that ignores a monkeypatched code default,
proving reconstruction reads only the h5; v1 refusal = rebuild raising
on a schema_version != 2 file.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

from emulator.activations import make_activation
from emulator.experiment import EmulatorExperiment
from emulator.results import save_emulator, rebuild_emulator
import emulator.training as training

FAILURES = []


def report(label, ok, detail):
  """Print one acceptance line and record a failure."""
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


def repo_root():
  """The repo root (two levels above gates/checks/)."""
  return Path(__file__).resolve().parents[2]


def load_deploy():
  """Read the cocoa root + dump directory from board_config.json.

  Returns:
    (device, data_dir): the torch device and the Path holding the dump
    .npy / .txt / .covmat files (<root>/chains, the driver's
    convention). Exits with a clear message if the deploy paths are
    unset (the check needs the real dumps; there is no synthetic path).
  """
  cfg = json.loads((repo_root() / "gates" / "board_config.json").read_text())
  rootdir = cfg.get("rootdir")
  driver_root = cfg.get("driver_root")
  if rootdir is None or driver_root is None:
    print("GSV-C: board_config.json rootdir / driver_root are unset; "
          "the save gate needs the real training dumps.")
    sys.exit(2)
  root = Path(driver_root)
  if not root.is_absolute():
    root = Path(rootdir) / driver_root
  data_dir = root / "chains"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return device, data_dir


def tiny_config(data_dir, *, ia=None, pce=False):
  """A tiny but real training config for one variant.

  Arguments:
    data_dir = the directory holding the dump files (paths made
               absolute so from_config needs no --root).
    ia       = None for plain, or "nla" for the factored variant.
    pce      = True to add the top-level pce block (the NPCE variant;
               exclusive with ia).

  Returns:
    the config dict (data + train_args, plus pce when asked).
  """
  data = {"train_dv": str(data_dir / "w0wa_takahashi_dvs_train_cs_16.npy"),
          "train_params":
            str(data_dir / "w0wa_takahashi_params_train_cs_16.1.txt"),
          "train_covmat":
            str(data_dir / "w0wa_takahashi_params_train_cs_16.covmat"),
          "val_dv": str(data_dir / "w0wa_takahashi_dvs_train_cs_8.npy"),
          "val_params":
            str(data_dir / "w0wa_takahashi_params_train_cs_8.1.txt"),
          "cosmolike_data_dir": "lsst_y1",
          "cosmolike_dataset": "lsst_y1_M1_GGL0.05.dataset",
          "param_cuts": {"omegabh2_hi": 0.035},
          "n_train": 200,
          "n_val": 100,
          "split_seed": 0}
  # gated_power (n_gates 3) so the drift proof's make_activation n_gates
  # patch bites; compile_mode None so the live model is never
  # torch.compiled, keeping the bitwise leg on the save contract, not
  # compile float reordering.
  model = {"name": "resmlp",
           "mlp": {"width": 64, "n_blocks": 2},
           "activation": {"type": "gated_power", "n_gates": 3},
           "compile_mode": None}
  if ia is not None:
    model["ia"] = ia
  train_args = {"nepochs": 3,
                "bs": 32,
                "loss": {"mode": "sqrt"},
                "silent": True,
                "model": model,
                "optimizer": {"weight_decay": 0.0},
                "lr": {"lr_base": 0.001, "bs_base": 64.0, "warmup_epochs": 1},
                "scheduler": {"mode": "min", "patience": 10, "factor": 0.8}}
  cfg = {"data": data, "train_args": train_args}
  if pce:
    cfg["pce"] = {"form": "residual"}
  return cfg


def train_save(cfg, device, save_root):
  """Train one tiny emulator, save it, return (exp, probe, live_out).

  Arguments:
    cfg       = the variant config dict.
    device    = the torch device.
    save_root = the path root to save under.

  Returns:
    (exp, probe, live_out): the experiment, the probe input rows, and
    the live model's raw output on the probe (the bitwise reference).
  """
  exp = EmulatorExperiment.from_config(cfg, quiet=True, device=device)
  model, train_losses, medians, means, fracs = exp.run()

  # val_set["C"] is the already-sliced (n_val,) params; index it
  # POSITIONALLY (the first 8 rows), never with val_set["idx"] (original
  # dump-row numbers from the 16k pool -> IndexError / wrong rows).
  probe = torch.as_tensor(exp.val_set["C"][:8],
                          dtype=torch.float32,
                          device=device)
  model.eval()
  with torch.no_grad():
    live_out = model(exp.pgeom.encode(probe)).detach().clone()

  save_emulator(path_root=str(save_root),
                model=model,
                param_geometry=exp.pgeom,
                geometry=exp.geom,
                config=cfg,
                histories={"train_losses": train_losses,
                           "val_medians": medians,
                           "val_means": means,
                           "val_fracs": fracs,
                           "thresholds": exp.thresholds},
                train_args=exp.train_args,
                pce=(exp.chi2fn.pce if exp.pce_opts is not None else None),
                pce_form=(exp.pce_opts["form"]
                          if exp.pce_opts is not None else None),
                resolved_train=exp.resolved_train,
                resolved_model=exp.resolved_model,
                attrs={"n_train": cfg["data"]["n_train"]})
  return exp, probe, live_out


def rebuilt_out(save_root, device, probe, *, compile_model=False):
  """Rebuild from the file alone and return the rebuilt output on probe."""
  model_r, pgeom_r, geom_r, info = rebuild_emulator(
    path_root=str(save_root),
    device=device,
    compile_model=compile_model)
  model_r.eval()
  with torch.no_grad():
    return model_r(pgeom_r.encode(probe)).detach().clone()


def run_variant(name, cfg, device, tmp):
  """Save-rebuild-bitwise one variant; return its (save_root, probe)."""
  save_root = Path(tmp) / ("emul_" + name)
  exp, probe, live_out = train_save(cfg=cfg, device=device, save_root=save_root)
  reb_out = rebuilt_out(save_root=save_root, device=device, probe=probe)
  report(name + ": rebuilt output bitwise-equal to the live model",
         torch.equal(live_out, reb_out),
         "max abs diff "
         + repr(float((live_out - reb_out).abs().max())))
  return save_root, probe


def main():
  """Run the three variants + the drift proof + the v1 refusal."""
  print("== GSV-C: save -> rebuild bitwise + drift + v1 refusal ==")
  device, data_dir = load_deploy()
  print("device " + str(device) + ", dumps " + str(data_dir))

  tmp = tempfile.mkdtemp(prefix="gsv-")
  plain_root, plain_probe = run_variant(
    "plain", tiny_config(data_dir), device, tmp)
  run_variant("factored", tiny_config(data_dir, ia="nla"), device, tmp)
  run_variant("npce", tiny_config(data_dir, pce=True), device, tmp)

  # drift proof: monkeypatch the SHARP default, make_activation's
  # n_gates (activations.py). If rebuild trusted the code default the
  # rebuilt gated_power activation would carry K=7 parameters and the
  # strict weight load / output would break; with the file-recorded
  # n_gates=3 it rebuilds unchanged. The compile-mode default is patched
  # too (a second, softer leg). The plain save uses gated_power (n_gates 3).
  base_reb = rebuilt_out(plain_root, device, plain_probe)
  saved_defaults = make_activation.__defaults__
  saved_compile = training.DEFAULT_COMPILE_MODE
  try:
    make_activation.__defaults__ = (7,)
    training.DEFAULT_COMPILE_MODE = "reduce-overhead"
    drift_reb = rebuilt_out(plain_root, device, plain_probe)
  finally:
    make_activation.__defaults__ = saved_defaults
    training.DEFAULT_COMPILE_MODE = saved_compile
  report("drift proof: rebuild ignores the monkeypatched n_gates default",
         torch.equal(base_reb, drift_reb),
         "make_activation n_gates default 3 -> 7 patched; max abs diff "
         + repr(float((base_reb - drift_reb).abs().max())))

  # v1 refusal: tamper schema_version, rebuild must raise loudly.
  import h5py
  with h5py.File(str(plain_root) + ".h5", "r+") as f:
    f.attrs["schema_version"] = 1
  refused = False
  try:
    rebuild_emulator(path_root=str(plain_root), device=device)
  except ValueError:
    refused = True
  report("v1 refusal: rebuild raises on schema_version != 2",
         refused,
         "a v1 file must be refused with a clear message")

  print("")
  if len(FAILURES) == 0:
    print("GSV-C: ALL PASS")
    return 0
  print("GSV-C: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
