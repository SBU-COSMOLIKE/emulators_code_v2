#!/usr/bin/env python3
"""save-rebuild-drift: prove a saved emulator reloads exactly.

This check defends one promise: an emulator rebuilt from its saved file
behaves exactly like the one that was just trained, even if the code's
default values change afterward. If that promise breaks, a reloaded model
quietly returns different numbers than the run that was published.

How it works, in order:
  1. Train four tiny emulators in this process: a plain one, a factored
     intrinsic-alignment one (its amplitudes are applied in the loss, not
     the network), a neural-PCE one (a polynomial base plus a small
     correcting network), and a conv-head one (rescnn: the training
     attach's bin split must PERSIST in the dv-geometry state — the
     2026-07-11 fix — for the head to rebuild without the dataset ini).
  2. Save each, rebuild each from its saved file alone, and compare the
     rebuilt model's output to the still-in-memory model's on the same
     input rows. "Exactly" means torch.equal here: every output number
     matches to the last bit.
  3. The drift test: change a code default, rebuild once more, and confirm
     the output is still identical. That can only hold if the rebuild reads
     the saved file and ignores the changed default.
  4. Confirm an old-format file (schema version 1) is refused, not silently
     mis-loaded — and a head save whose persisted bin split is deleted
     (an artifact predating the persistence) is refused naming the fix,
     never re-derived or crashed on.
Every compared value is printed; any mismatch prints a FAIL line and the
run exits non-zero.

The training path is the driver's own, all from emulator/:
EmulatorExperiment.from_config -> run -> save_emulator -> rebuild_emulator.
The dumps and the save locations come from ai/gates/board_config.json; the
dump directory follows the driver's <root>/chains convention.

Home note: ai/notes/artifacts-inference-warmstart.md, "save-rebuild-drift."
The factored and neural-PCE saves are included so the saved geometry type and
the PCE data both survive a reload.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from emulator import fixed_facts
from emulator.activations import make_activation
from emulator.experiment import EmulatorExperiment
from emulator.results import save_emulator, rebuild_emulator
import emulator.training as training

FAILURES = []


def report(label, ok, detail):
  """Print one PASS/FAIL line and remember any failure.

  A failing check appends its label to the module-level FAILURES list so
  main can count them and exit non-zero.
  """
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


def emit_aid(aid, n_before):
  """Emit ONE '##AID <aid> <PASS|FAIL>' line for a whole acceptance leg.

  (queue 2) The board's run_check folds these reserved lines into the gate's
  executed set: one per declared leg, at the leg's aggregation point. Every
  leg here is a single report() call — one variant's rebuild, the drift proof,
  or one refusal — so a leg's verdict is FAIL if that report appended a label
  to the module-level FAILURES list. The child's exit status stays the single
  aggregate verdict; the manifest only says WHICH declared leg each report is.

  Arguments:
    aid      = the board-unique leg id, "save-rebuild-drift.<leg>".
    n_before = len(FAILURES) captured immediately before the leg's check ran.
  """
  mark = "PASS" if len(FAILURES) == n_before else "FAIL"
  print("##AID " + aid + " " + mark)


def repo_root():
  """The repo root (three levels above ai/gates/checks/)."""
  return Path(__file__).resolve().parents[3]


def load_deploy():
  """Read the cocoa root + dump directory from board_config.json.

  The rootdir key follows the portable-config rule the harness uses
  (ai/gates/run_board.py): a non-null file value wins, a null file value
  resolves from the $ROOTDIR environment variable at load, and the file
  is never rewritten. This script runs as its own process and reads the
  raw file, so it applies the same resolution itself; without this, the
  shipped "rootdir": null would read as unset even on a machine where
  the harness preflight passes.

  Returns:
    (device, data_dir): the torch device and the Path holding the dump
    .npy / .txt / .covmat files (<root>/chains, the driver's
    convention). Exits with a clear message if the deploy paths are
    unset (the check needs the real dumps; there is no synthetic path).
  """
  cfg = json.loads(
    (repo_root() / "ai" / "gates" / "board_config.json").read_text())
  rootdir = cfg.get("rootdir")
  if rootdir is None:
    # empty string = unset, matching the harness's resolution.
    rootdir = os.environ.get("ROOTDIR") or None
  driver_root = cfg.get("driver_root")
  if rootdir is None or driver_root is None:
    print("save-rebuild-drift: no rootdir (board_config.json rootdir is null "
          "and $ROOTDIR is unset) or driver_root is unset; the save gate "
          "needs the real training dumps.")
    sys.exit(2)
  root = Path(driver_root)
  if not root.is_absolute():
    root = Path(rootdir) / driver_root
  data_dir = root / "chains"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return device, data_dir


def tiny_config(data_dir, *, ia=None, pce=False, head=False):
  """A tiny but real training config for one variant.

  Arguments:
    data_dir = the directory holding the dump files (paths made
               absolute so from_config needs no --root).
    ia       = None for plain, or "nla" for the factored variant.
    pce      = True to add the top-level pce block (the NPCE variant;
               exclusive with ia).
    head     = True for the conv-head variant (name rescnn + a small
               cnn block): training runs build_shear_angle_map, so the
               save must persist the bin split (bin_sizes / pm_kept in
               the dv-geometry state) for the rebuild to reconstruct
               the head without the dataset ini.

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
  if head:
    model["name"] = "rescnn"
    model["cnn"] = {"kernel_size": 5, "n_blocks": 1}
  # trim + focus are hard-required by build_run_specs (training.py:
  # dict(train_args["trim"]) / dict(train_args["focus"]), no default), so
  # a config without them raises KeyError before training. Benign off
  # values: start == end == 0 makes anneal_value return 0 every epoch, so
  # no trimming or focal weighting complicates the bitwise save contract.
  # Every key anneal_value reads (start / end / hold_epochs /
  # anneal_epochs / shape) plus focus's kappa is present.
  train_args = {"nepochs": 3,
                "bs": 32,
                "loss": {"mode": "sqrt"},
                "silent": True,
                "model": model,
                "optimizer": {"weight_decay": 0.0},
                "lr": {"lr_base": 0.001,
                       "bs_base": 64.0,
                       "warmup_epochs": 1},
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
                          "kappa": 0.15}}
  cfg = {"data": data, "train_args": train_args}
  if pce:
    cfg["pce"] = {"form": "residual"}
  return cfg


def train_save(cfg, device, save_root, label, persist_root=None):
  """Train one tiny emulator, save it, return (exp, probe, live_out).

  Arguments:
    cfg          = the variant config dict.
    device       = the torch device.
    save_root    = the path root to save under (the tmp round-trip root).
    label        = what this variant's emulator is, for the scientific record
                   the saved file carries. This check trains on the real dumps
                   but the record they were published with is not in reach
                   here, so the file declares itself a test double rather than
                   carrying no record at all. The two saves below are the one
                   emulator written to two paths, so they share the one label
                   and come out with the one identity.
    persist_root = an optional second, persistent root to save the same
                   bytes under (survives the tmp cleanup); the plain case
                   passes it so the board-owned cobaya-adapter evaluate
                   leg has an emulator to load. None saves only to
                   save_root.

  Returns:
    (exp, probe, live_out): the experiment, the probe input rows, and
    the live model's raw output on the probe (the bitwise reference).
  """
  exp = EmulatorExperiment.from_config(cfg, quiet=True, device=device)
  model, train_losses, medians, means, fracs = exp.run()

  # val_set["C"] is the already-sliced (n_val,) params; index it
  # positionally (the first 8 rows), never with val_set["idx"] (original
  # dump-row numbers from the 16k pool -> IndexError / wrong rows).
  probe = torch.as_tensor(exp.val_set["C"][:8],
                          dtype=torch.float32,
                          device=device)
  model.eval()
  with torch.no_grad():
    live_out = model(exp.pgeom.encode(probe)).detach().clone()

  composition_mode = "npce" if exp.pce_opts is not None else "plain"
  resolved_pce = (dict(exp.pce_opts)
                  if exp.pce_opts is not None else None)
  save_kwargs = dict(
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
    pce_form=(exp.pce_opts["form"] if exp.pce_opts is not None else None),
    resolved_train=exp.resolved_train,
    resolved_model=exp.resolved_model,
    composition_mode=composition_mode,
    transfer_refined=False,
    resolved_pce=resolved_pce,
    resolved_transfer=None,
    facts_yaml=fixed_facts.synthetic_sidecar(
      names=exp.pgeom.state()["names"],
      label=label,
      family="cosmolike"),
    # rescale is the resolved run value (never a literal): a fine-tune run
    # warm-starting from this artifact reads the attr and refuses a source
    # that does not record it (emulator/warmstart.py load_source).
    attrs={"n_train": cfg["data"]["n_train"],
           "rescale": exp.rescale})
  save_emulator(path_root=str(save_root), **save_kwargs)
  # a second, PERSISTENT save (same bytes, a stable root) so the board's
  # cobaya-adapter evaluate leg has an emulator to load after the tmp dir
  # is gone. The tmp save above (the bitwise round-trip) is untouched.
  if persist_root is not None:
    save_emulator(path_root=str(persist_root), **save_kwargs)
    print("persisted evaluate emulator -> " + str(persist_root)
          + ".h5 / .emul")
  return exp, probe, live_out


def rebuilt_out(save_root, device, probe, *, compile_model=False):
  """Rebuild the emulator from its saved file alone and run it on the probe.

  Returns the rebuilt model's output on the probe rows, the value the caller
  compares (bit for bit) against the still-in-memory model's output.
  """
  model_r, pgeom_r, geom_r, info = rebuild_emulator(
    path_root=str(save_root),
    device=device,
    compile_model=compile_model)
  model_r.eval()
  with torch.no_grad():
    return model_r(pgeom_r.encode(probe)).detach().clone()


def run_variant(name, cfg, device, tmp, persist_root=None, aid=None):
  """Save-rebuild-bitwise one variant; return its (save_root, probe).

  persist_root (plain case only) is forwarded to train_save for the
  second, persistent save the cobaya-adapter evaluate leg loads.
  aid is this variant's declared board leg (queue 2): the one report below
  IS the leg, so the ##AID line is emitted right here.
  """
  save_root = Path(tmp) / ("emul_" + name)
  n0 = len(FAILURES)
  exp, probe, live_out = train_save(cfg=cfg, device=device,
                                    save_root=save_root,
                                    label="save-rebuild-drift/" + name,
                                    persist_root=persist_root)
  reb_out = rebuilt_out(save_root=save_root, device=device, probe=probe)
  report(name + ": rebuilt output bitwise-equal to the live model",
         torch.equal(live_out, reb_out),
         "max abs diff "
         + repr(float((live_out - reb_out).abs().max())))
  if aid is not None:
    emit_aid(aid, n0)
  return save_root, probe


def main():
  """Run the four save-rebuild checks, the drift test, and the refusals.

  In order: read the deploy paths; then, for the plain, factored,
  neural-PCE, and conv-head variants, train a tiny emulator, save it,
  rebuild it from the file, and require the rebuilt output to match the
  live output to the last bit (the plain one is also saved to a stable
  location for the board's cobaya-adapter evaluate leg; the head one
  proves the persisted bin split reconstructs the ResCNN with no dataset
  ini). Then the drift test on the plain save: patch make_activation's
  n_gates default and the compile-mode default, rebuild, and require an
  identical output. Last the two refusals: a schema-version-1 file, and
  a head save with its persisted bin split deleted (a pre-persistence
  artifact) — each must raise loudly. Any failed step prints a FAIL
  line; main returns 1 if any failed, else 0.
  """
  print("== save-rebuild-drift ==")
  device, data_dir = load_deploy()
  print("device " + str(device) + ", dumps " + str(data_dir))

  tmp = tempfile.mkdtemp(prefix="gsv-")
  # the plain case also persists to <driver_root>/chains/gates_emul_evaluate
  # (.h5 + .emul) so the board's cobaya-adapter evaluate leg can load it.
  evaluate_root = data_dir / "gates_emul_evaluate"
  plain_root, plain_probe = run_variant(
    "plain", tiny_config(data_dir), device, tmp, persist_root=evaluate_root,
    aid="save-rebuild-drift.plain-rebuild-matches-live")
  run_variant("factored", tiny_config(data_dir, ia="nla"), device, tmp,
              aid="save-rebuild-drift.factored-rebuild-matches-live")
  run_variant("npce", tiny_config(data_dir, pce=True), device, tmp,
              aid="save-rebuild-drift.npce-rebuild-matches-live")
  # the conv-head variant: training attaches the bin split
  # (build_shear_angle_map), the save persists it (bin_sizes / pm_kept
  # in the dv-geometry state), and the rebuild reconstructs the ResCNN
  # from the files alone — before the persistence this died in the
  # constructor's bin_sizes assert.
  head_root, _ = run_variant(
    "head", tiny_config(data_dir, head=True), device, tmp,
    aid="save-rebuild-drift.head-rebuild-matches-live")

  # drift test: monkeypatch the telling default, make_activation's
  # n_gates (activations.py). If rebuild trusted the code default the
  # rebuilt gated_power activation would carry K=7 parameters and the
  # strict weight load / output would break; with the file-recorded
  # n_gates=3 it rebuilds unchanged. The compile-mode default is patched
  # too (a second, softer leg). The plain save uses gated_power (n_gates 3).
  n_drift = len(FAILURES)
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
  emit_aid("save-rebuild-drift.code-default-drift-ignored", n_drift)

  # A saved emulator now records the science it was born under: the cosmology
  # held fixed while its parameters were sampled, and the region they were
  # sampled over. The two schema versions that came before it recorded neither,
  # so neither can say which cosmology it belongs to, and an emulator that
  # cannot say that must not be served. Both are refused, and the refusal names
  # the way out, because a message that says a file is incompatible and stops is
  # not a refusal, it is a shrug. Forge each version onto a good file in turn and
  # confirm the reader refuses it and says what to do.
  import h5py

  def forge_version(version):
    # returns the refusal's message, or None when the file was accepted.
    with h5py.File(str(plain_root) + ".h5", "r+") as f:
      f.attrs["schema_version"] = version
    try:
      rebuild_emulator(path_root=str(plain_root), device=device)
    except ValueError as exc:
      return str(exc)
    return None

  n_v1 = len(FAILURES)
  said = forge_version(1)
  report("a version 1 emulator is refused, and told how to migrate",
         said is not None and "Re-generate the dataset" in said,
         (said or "accepted").splitlines()[0][:60])
  emit_aid("save-rebuild-drift.v1-schema-refusal", n_v1)

  n_v2 = len(FAILURES)
  said = forge_version(2)
  report("a version 2 emulator is refused, and told how to migrate",
         said is not None and "Re-generate the dataset" in said,
         (said or "accepted").splitlines()[0][:60])
  emit_aid("save-rebuild-drift.v2-schema-refusal", n_v2)

  # put the file back the way it was saved, so the arms below read the real
  # artifact and not the one this arm forged.
  with h5py.File(str(plain_root) + ".h5", "r+") as f:
    f.attrs["schema_version"] = fixed_facts.SCHEMA_VERSION

  n_head = len(FAILURES)
  # pre-persistence head-artifact refusal: delete the persisted bin
  # split from the head save (simulating an artifact written before
  # 2026-07-11); rebuild must raise the loud KeyError naming the
  # persistence, never re-derive the split or crash in the
  # constructor's assert.
  with h5py.File(str(head_root) + ".h5", "r+") as f:
    del f["dv_geometry"]["bin_sizes"]
    if "pm_kept" in f["dv_geometry"]:
      del f["dv_geometry"]["pm_kept"]
  refused_head = False
  named = False
  try:
    rebuild_emulator(path_root=str(head_root), device=device)
  except KeyError as e:
    refused_head = True
    named = "bin-split persistence" in str(e)
  report("old head artifact refusal: rebuild raises naming the "
         "persistence",
         refused_head and named,
         "a head file without bin_sizes must be refused, not guessed")
  emit_aid("save-rebuild-drift.old-head-artifact-refusal", n_head)

  print("")
  if len(FAILURES) == 0:
    print("save-rebuild-drift: ALL PASS")
    return 0
  print("save-rebuild-drift: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
