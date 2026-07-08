#!/usr/bin/env python3
"""cobaya-adapter parity (spec code GCT-C): inference equals training.

WHAT: EmulatorPredictor, the object the cobaya theory block calls for
every MCMC step. WHY: an MCMC must sample the model that was trained;
a predictor that deviates even slightly biases every posterior drawn
from it. HOW: trains a tiny plain emulator and a tiny factored (ia:nla) one, saves
each, builds an EmulatorPredictor from the saved file, and requires the
predictor's data vector to match the TRAINING-side prediction on the
same probe points to rtol 1e-6 (home note cobaya-theory-adapter.md
:117-123). The factored case is the real save -> rebuild -> predict
round-trip added (:234-238): the geometry-class marker in the h5
must rebuild the AmplitudeFactorGeometry so the factored combine
reproduces. predict() returns the emulator's own probe section by
default, so the kept-entry comparison indexes that section at dest_idx
(the xi section's offset is 0); the check also asserts the shapes
(section length == stored section_sizes[0]; dv_return '3x2pt' length ==
total_size; masked positions exactly 0.0). Prints the worst relative
error per variant and exits nonzero on any mismatch.

The training-side prediction is the live path the predictor
reconstructs: pgeom.encode(theta) -> model -> decode, where decode is
geom.decode for the plain run and chi2fn.decode(pred, x_enc) for the
factored run (TemplateFactoredChi2). The tiny train + save helpers are
shared with gsv_bitwise_drift; deploy paths come from board_config.json.

PS: parity = the in-package predictor and the training stack agree on
the same probe; round-trip = save then rebuild then predict, reading
only the file; rtol = relative tolerance for the matmul-order-sensitive
float comparison.
"""

import sys
import tempfile
from pathlib import Path

import torch

from emulator.inference import EmulatorPredictor
from gsv_bitwise_drift import load_deploy, tiny_config, train_save

FAILURES = []


def report(label, ok, detail):
  """Print one acceptance line and record a failure."""
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


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


def run_parity(name, cfg, device, tmp, factored):
  """Train + save one variant, then compare the predictor to training.

  Arguments:
    name     = the variant label (plain / factored).
    cfg      = the variant config dict.
    device   = the torch device.
    tmp      = the temp directory to save under.
    factored = True for the ia run (selects the decode path).
  """
  save_root = Path(tmp) / ("emul_" + name)
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

  # shape + masking assertions on one representative row.
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


def main():
  """Run the plain parity probe + the factored round-trip."""
  print("== cobaya-adapter parity (spec code GCT-C) ==")
  device, data_dir = load_deploy()
  print("device " + str(device) + ", dumps " + str(data_dir))

  tmp = tempfile.mkdtemp(prefix="gct-")
  run_parity(name="plain",
             cfg=tiny_config(data_dir),
             device=device,
             tmp=tmp,
             factored=False)
  run_parity(name="factored",
             cfg=tiny_config(data_dir, ia="nla"),
             device=device,
             tmp=tmp,
             factored=True)

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
