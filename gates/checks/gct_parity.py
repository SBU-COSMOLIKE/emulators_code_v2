#!/usr/bin/env python3
"""cobaya-adapter parity: the predictor reproduces the trained model.

EmulatorPredictor is the object the cobaya theory block calls at every MCMC
step to turn cosmological parameters into a data vector. An MCMC must sample
the very model that was trained, so if the predictor drifts even slightly
from the training-time forward pass it biases every posterior drawn from it.
This check proves the two agree ("parity" is that agreement).

How it works: train two tiny emulators (a plain one and a factored
intrinsic-alignment one, ia:nla), save each, build an EmulatorPredictor from
the saved file, and require the predictor's data vector to match the
training-side prediction on the same input rows to a relative tolerance of
1e-6. The training-side prediction is the live forward pass the predictor
rebuilds: encode the parameters, run the model, then decode (geom.decode for
the plain run, or the loss's chi2fn.decode for the factored run). The
factored case is a full round-trip (save, then rebuild from the file, then
predict): the saved geometry-type marker must rebuild the amplitude-factoring
geometry so the factored combine reproduces.

predict() returns the emulator's own probe section by default (for cosmic
shear the xi block), so the row-for-row comparison indexes that section at
the kept-entry positions (the xi section starts at offset 0). The check also
confirms the shapes: the section length equals the stored section size, the
full 3x2pt vector equals the total length, and the masked positions are
exactly 0.0. The worst relative error per variant is printed; any mismatch
exits non-zero.

The tiny train-and-save helpers are shared with gsv_bitwise_drift; the dumps
and save locations come from board_config.json.

Home note: artifacts-inference-warmstart.md:117-123 (the plain
parity probe) and :234-238 (the factored round-trip).
"""

import sys
import tempfile
from pathlib import Path

import torch

from emulator.inference import EmulatorPredictor
from gsv_bitwise_drift import load_deploy, tiny_config, train_save

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
  executed set: one per declared leg, at the leg's aggregation point. This
  child carries four legs — the predictor-vs-training parity and the
  scattered-vector shape/mask group, for each of the plain and factored
  variants. A leg's verdict is FAIL if its group appended a label to the
  module-level FAILURES list; the child's exit status stays the single
  aggregate verdict. The board gate's own evaluate legs are separate aids,
  asserted in the wrapper, because this child does not run them.

  Arguments:
    aid      = the board-unique leg id, "cobaya-adapter.<leg>".
    n_before = len(FAILURES) captured immediately before the leg's checks ran.
  """
  mark = "PASS" if len(FAILURES) == n_before else "FAIL"
  print("##AID " + aid + " " + mark)


def training_side(exp, probe, factored):
  """The training-stack physical data vectors on the probe rows.

  Arguments:
    exp      = the trained experiment (model + pgeom + geom + chi2fn).
    probe    = the probe input rows (raw params, device tensor).
    factored = True for an ia run (decode via chi2fn.decode(pred,
               x_enc)); False for plain (geom.decode(pred)).

  Returns:
    (B, n_keep) physical kept-entry data vectors as a cpu tensor.
  """
  exp.model.eval()
  with torch.no_grad():
    x_enc = exp.pgeom.encode(probe)
    pred = exp.model(x_enc)
    if factored:
      dv = exp.chi2fn.decode(pred, x_enc)
    else:
      dv = exp.geom.decode(pred)
  return dv.detach().cpu()


def run_parity(name, cfg, device, tmp, factored, parity_aid, shape_aid):
  """Train + save one variant, then compare the predictor to training.

  Arguments:
    name       = the variant label (plain / factored).
    cfg        = the variant config dict.
    device     = the torch device.
    tmp        = the temp directory to save under.
    factored   = True for the ia run (selects the decode path).
    parity_aid = this variant's declared parity leg (the rtol 1e-6
                 predictor-vs-training comparison).
    shape_aid  = this variant's declared scattered-vector leg (the section
                 length, the 3x2pt length, and the masked-zero positions).
  """
  save_root = Path(tmp) / ("emul_" + name)
  n_parity = len(FAILURES)
  exp, probe, _ = train_save(cfg=cfg, device=device, save_root=save_root)
  ts = training_side(exp=exp, probe=probe, factored=factored)

  geom       = exp.geom
  dest_idx   = geom.dest_idx.cpu()
  section0   = geom.section_sizes[0]      # the xi block length
  total_size = geom.total_size

  # two predictors from the same file: the default 'section' shape (the
  # per-probe block the likelihood glues) and the full '3x2pt' scattered
  # vector; the saved geometry carries section_sizes + probe.
  pred_sec  = EmulatorPredictor(path_root=str(save_root), device=device)
  pred_full = EmulatorPredictor(path_root=str(save_root), device=device,
                                dv_return="3x2pt")
  worst = 0.0
  n = probe.shape[0]
  i = 0
  while i < n:
    # val_set["C"] is the already-sliced (n_val,) params; index it
    # positionally (row i), never with val_set["idx"] (original dump-row
    # numbers). probe was built the same way, so row i lines up with ts[i].
    row = exp.val_set["C"][i]
    # the xi section starts at offset 0, so its dest_idx positions ARE the
    # kept entries: index the section output at dest_idx and compare to the
    # training-side kept vector (rtol 1e-6, the kept-entry comparison kept).
    sec = torch.as_tensor(pred_sec.predict(row), dtype=ts.dtype)
    got = sec[dest_idx]
    want = ts[i]
    denom = want.abs() + 1.0e-8
    rel = float(((got - want).abs() / denom).max())
    if rel > worst:
      worst = rel
    i = i + 1
  report(name + ": predictor matches the training side (rtol 1e-6)",
         worst <= 1.0e-6,
         "worst relative error " + repr(worst))
  emit_aid(parity_aid, n_parity)

  # shape + masking assertions on one representative row.
  n_shape = len(FAILURES)
  row0  = exp.val_set["C"][0]
  sec0  = torch.as_tensor(pred_sec.predict(row0), dtype=ts.dtype)
  full0 = torch.as_tensor(pred_full.predict(row0), dtype=ts.dtype)
  report(name + ": section length == stored section_sizes[0]",
         sec0.numel() == section0,
         "len " + str(sec0.numel()) + " vs section_sizes[0] " + str(section0))
  report(name + ": 3x2pt length == total_size",
         full0.numel() == total_size,
         "len " + str(full0.numel()) + " vs total_size " + str(total_size))
  # every position outside dest_idx is exactly 0.0 in the scattered vector.
  masked = torch.ones(total_size, dtype=torch.bool)
  masked[dest_idx] = False
  nonzero_masked = int((full0[masked] != 0.0).sum())
  report(name + ": masked positions exactly 0.0 in the 3x2pt vector",
         nonzero_masked == 0,
         str(nonzero_masked) + " masked positions nonzero")
  emit_aid(shape_aid, n_shape)


def main():
  """Run the plain parity probe, then the factored round-trip.

  Reads the deploy paths, then for the plain and factored variants trains a
  tiny emulator, saves it, builds two predictors from the file (the default
  'section' shape and the full '3x2pt' shape), and checks the predictor
  matches the training-side data vector to rtol 1e-6, plus the section /
  full-length / masked-zero shape assertions. Prints a note that the evaluate
  run and the MCMC smoke are driven by the board gate, not here; returns 1 if
  any check failed, else 0.
  """
  print("== cobaya-adapter parity ==")
  device, data_dir = load_deploy()
  print("device " + str(device) + ", dumps " + str(data_dir))

  tmp = tempfile.mkdtemp(prefix="gct-")
  run_parity(name="plain",
             cfg=tiny_config(data_dir),
             device=device,
             tmp=tmp,
             factored=False,
             parity_aid="cobaya-adapter.plain-predictor-parity",
             shape_aid="cobaya-adapter.plain-scattered-vector-shape-and-mask")
  run_parity(name="factored",
             cfg=tiny_config(data_dir, ia="nla"),
             device=device,
             tmp=tmp,
             factored=True,
             parity_aid="cobaya-adapter.factored-predictor-parity",
             shape_aid="cobaya-adapter.factored-scattered-vector-shape-and-mask")

  print("")
  print("note: the example evaluate run (cobaya-run vs the lsst_y1 "
        "likelihood) and the MCMC smoke are driven by the board gate "
        "(gate_gct_c), not this parity check.")
  if len(FAILURES) == 0:
    print("cobaya-adapter parity: ALL PASS")
    return 0
  print("cobaya-adapter parity: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
