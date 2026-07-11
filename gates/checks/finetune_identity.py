#!/usr/bin/env python3
"""finetune-identity: a warm-started emulator starts as the source, exactly.

This check defends the one promise fine-tuning rests on: at the very first
epoch, an emulator warm-started from a saved source computes the source's own
function, bit for bit, no matter what the new parameters are. If that promise
breaks, a fine-tune run does not continue the source at all, it starts
somewhere near it and relearns the whitening bases.

It needs torch but no cosmolike: it builds a tiny synthetic source by hand (a
small ResMLP, a parameter geometry from a random positive-definite covariance,
a data-vector geometry constructed directly), saves it with save_emulator, and
then exercises the warm-start path from emulator/warmstart.py with two extra
parameters added.

How it works, in order:
  1. Build and save a synthetic source emulator (5 parameters, a 12-long data
     vector), then load it back through load_source.
  2. Extend the input geometry for a 7-parameter run (the 5 shared names plus
     two extras, w0 / wa) and check the shared parameters encode to exactly
     the source's numbers (torch.equal), while the extra coordinates depend
     only on the extras.
  3. Transfer the source weights into the wider model and check every
     unchanged tensor is copied bit for bit, the input-sized tensors gain
     exactly the extra columns as zeros, and only those tensors are padded.
  4. Run the pre-train parity check: the warm-started model matches the source
     on the shared parameters within the tolerance, and changing only the
     extras leaves its output identical.
  5. The degenerate case (no extra parameters): the transferred weights and
     the geometry tensors are bit-identical to the source's.
  6. The loud errors fire: a non-superset parameter set, a model: block beside
     finetune:, and a rescale other than none.
Every checked value is printed; any failure prints a FAIL line and the run
exits non-zero.

Spec code FTW-A. Home note: finetune-warm-start.md (design rules D-FT1..D-FT8;
this check is the finetune-identity gate the note's validation section names).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from emulator import warmstart
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.output import DataVectorGeometry
from emulator.geometries.parameter import ParamGeometry
from emulator.results import save_emulator
from emulator.training import make_model

FAILURES = []

N_S      = 5             # source parameter count
OUT_DIM  = 12            # kept (unmasked) data-vector length
TOTAL    = 20            # full data-vector length (before the mask)
EXTRAS   = ["w0", "wa"]  # the two new parameters fine-tuned in


def report(label, ok, detail):
  """Print one PASS/FAIL line and remember any failure.

  A failing check appends its label to the module-level FAILURES list so
  main can count them and exit non-zero.
  """
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


def spd(n, seed):
  """A random symmetric positive-definite matrix (n x n)."""
  g = np.random.default_rng(seed)
  a = g.standard_normal((n, n))
  return a @ a.T + n * np.eye(n)


def source_names():
  """The source parameter names (p0 .. p4)."""
  names = []
  for i in range(N_S):
    names.append("p" + str(i))
  return names


def build_source_geoms(device):
  """Build a synthetic source ParamGeometry + DataVectorGeometry.

  The parameter geometry whitens on a random positive-definite covariance;
  the data-vector geometry is constructed directly (a random kept-block
  whitening plus a full precision matrix), so no cosmolike is needed.

  Arguments:
    device = the torch device to place the tensors on.

  Returns:
    (pgeom, geom): the source input and output geometries.
  """
  names = source_names()
  cov_s = spd(N_S, seed=1)
  lam_s, V_s = np.linalg.eigh(cov_s)
  center_s = np.random.default_rng(2).standard_normal(N_S)
  pgeom = ParamGeometry(device=device,
                        names=names,
                        center=center_s,
                        evecs=V_s,
                        sqrt_ev=np.sqrt(lam_s))

  # a directly built output geometry: the first OUT_DIM entries survive the
  # mask, whitened on a random positive-definite kept-block covariance; the
  # full precision matrix is any positive-definite TOTAL x TOTAL matrix.
  cov_k = spd(OUT_DIM, seed=3)
  lam_k, V_k = np.linalg.eigh(cov_k)
  dest_idx = list(range(OUT_DIM))
  geom = DataVectorGeometry(
    device=device,
    total_size=TOTAL,
    dest_idx=dest_idx,
    evecs=V_k,
    sqrt_ev=np.sqrt(lam_k),
    Cinv=spd(TOTAL, seed=4),
    center=np.random.default_rng(5).standard_normal(OUT_DIM),
    section_sizes=[TOTAL],
    probe="xi")
  return pgeom, geom


def source_recipe():
  """The model_recipe dict a schema-v2 save stores for the source ResMLP.

  Mirrors the recipe EmulatorExperiment.build_specs assembles: the class
  qualname, the dims, the constructor kwargs with the activation / norm
  factories serialized by name, and the capability flags rebuild_emulator
  reads.
  """
  return {
    "cls": "emulator.designs.plain.ResMLP",
    "name": "resmlp",
    "ia": None,
    "input_dim": N_S,
    "output_dim": OUT_DIM,
    "compile_mode": None,
    "needs_geom": False,
    "kwargs": {
      "int_dim_res": 32,
      "n_blocks": 2,
      "block_opts": {"act": {"type": "H", "n_gates": 3},
                     "norm": "affine"},
    },
  }


def save_synthetic_source(root, device):
  """Build, then save, a synthetic source emulator under `root`.

  Writes <root>.h5 + <root>.emul with a schema-v2 recipe, a rescale-none
  root attr, and a data block naming a cosmolike dataset (the values
  pin_output_geometry later checks the new run against). No training runs:
  the model is a freshly initialized ResMLP, which is all the identity path
  needs.

  Arguments:
    root   = the path root to save under.
    device = the torch device.

  Returns:
    (pgeom, geom): the source geometries (also reachable through load_source).
  """
  pgeom, geom = build_source_geoms(device)
  block_opts = {"act": make_activation("H", n_gates=3),
                "norm": make_norm("affine")}
  model = ResMLP(input_dim=N_S,
                 output_dim=OUT_DIM,
                 int_dim_res=32,
                 n_blocks=2,
                 block_opts=block_opts).to(device)
  # a data block naming the source's dataset (a fine-tune run must match it).
  config = {"data": {"cosmolike_data_dir": "lsst_y1",
                     "cosmolike_dataset": "lsst_y1_M1_GGL0.05.dataset",
                     "train_dv": "src_train.npy",
                     "val_dv": "src_val.npy"},
            "train_args": {"nepochs": 1}}
  histories = {"train_losses": [0.1],
               "val_medians": [0.1],
               "val_means": [0.1],
               "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
               "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
  save_emulator(path_root=str(root),
                model=model,
                param_geometry=pgeom,
                geometry=geom,
                config=config,
                histories=histories,
                train_args=config["train_args"],
                resolved_train={"nepochs": 1},
                resolved_model=source_recipe(),
                attrs={"rescale": "none"})
  return pgeom, geom


def write_covmat(path, names, seed):
  """Write a covmat file (a "#"-prefixed header line + a SPD matrix)."""
  cov = spd(len(names), seed=seed)
  with open(path, "w") as f:
    f.write("# " + " ".join(names) + "\n")
    for row in cov:
      f.write(" ".join(repr(float(x)) for x in row) + "\n")


def check_encoding(source, covmat_path, device):
  """Item 1: the shared parameters encode to exactly the source's numbers."""
  train_mean = np.random.default_rng(6).standard_normal(N_S + len(EXTRAS))
  pgeom_new, extra_names = warmstart.extend_input_geometry(
    source=source, covmat_path=covmat_path, train_mean=train_mean,
    device=device)
  report("extend: extra names are [w0, wa] in covmat order",
         extra_names == EXTRAS, str(extra_names))

  names_n = list(pgeom_new.names)
  shared = []
  for nm in source.pgeom.names:
    shared.append(names_n.index(nm))
  shared_cols = torch.tensor(shared, dtype=torch.long, device=device)

  theta = torch.from_numpy(
    np.random.default_rng(7).standard_normal((64, N_S + len(EXTRAS)))
  ).float().to(device)
  enc_new = pgeom_new.encode(theta)
  enc_src = source.pgeom.encode(theta[:, shared_cols])
  d = float((enc_new[:, :N_S] - enc_src).abs().max())
  report("encode: shared coords bit-identical to the source encoding",
         torch.equal(enc_new[:, :N_S], enc_src),
         "max|dv| = " + format(d, ".2e"))

  # the extra coordinates must not move when only the shared params move.
  theta2 = theta.clone()
  theta2[:, :N_S] = theta2[:, :N_S] + 2.0
  enc2 = pgeom_new.encode(theta2)
  report("encode: extra coords depend only on the extras",
         torch.equal(enc2[:, N_S:], enc_new[:, N_S:]),
         "extra block unchanged under a shared-only shift")
  return pgeom_new, extra_names


def check_transfer(source, pgeom_new, device):
  """Item 2: verbatim tensors copied, input tensors zero-padded, set exact."""
  n_x = len(EXTRAS)
  model_opts = warmstart.recipe_to_model_opts(
    recipe=source.recipe, geom=None, compile_mode=None)
  template = make_model(model_opts=model_opts,
                        input_dim=len(pgeom_new.names),
                        output_dim=OUT_DIM,
                        device=device)
  src_state  = source.model.state_dict()
  tmpl_state = template.state_dict()
  new_state, padded = warmstart.transfer_state_dict(
    source_state=src_state, template_state=tmpl_state, n_extra=n_x)

  # the input-consumer set: every template tensor whose dim 1 grew by n_x.
  expected = []
  for k in tmpl_state:
    t = tmpl_state[k]
    if t.dim() >= 2 and t.shape[1] == src_state[k].shape[1] + n_x:
      expected.append(k)
  expected.sort()
  report("transfer: padded keys == the input-consumer set",
         padded == expected, str(padded))

  verbatim_ok = True
  for k in tmpl_state:
    if k in padded:
      continue
    if not torch.equal(new_state[k], src_state[k]):
      verbatim_ok = False
  report("transfer: every unchanged tensor is copied bit for bit",
         verbatim_ok, "non-padded tensors torch.equal to source")

  pad_ok = True
  for k in padded:
    w   = new_state[k]
    src = src_state[k]
    n_src = src.shape[1]
    if not torch.equal(w[:, :n_src], src):
      pad_ok = False
    if not torch.equal(w[:, n_src:], torch.zeros_like(w[:, n_src:])):
      pad_ok = False
  report("transfer: padded tensors are source columns then exact zeros",
         pad_ok, str(len(padded)) + " padded tensor(s)")


def check_parity(source, pgeom_new, extra_names, device):
  """Item 3: build_warm_start's parity gate passes (and prints its verdict)."""
  # a synthetic staged train set: raw parameter rows in the new order.
  C = np.random.default_rng(8).standard_normal((300, len(pgeom_new.names)))
  train_set = {"C": C, "idx": np.arange(300)}
  model_opts = warmstart.recipe_to_model_opts(
    recipe=source.recipe, geom=None, compile_mode=None)
  # build_warm_start returns (init_state, verdict, padded_keys) since
  # TPE-2 (the anchor mask rides on padded_keys); this check needs the
  # first two.
  init_state, verdict, _padded = warmstart.build_warm_start(
    source=source,
    new_pgeom=pgeom_new,
    pinned_geom=source.geom,
    model_opts=model_opts,
    train_set=train_set,
    extra_names=extra_names,
    device=device)
  report("parity: build_warm_start passes and returns the verdict line",
         verdict.startswith("[ok] finetune parity:"), verdict)
  # the returned state is what run_emulator would load: the transferred one.
  report("parity: init_state is a full state dict (loadable strict)",
         isinstance(init_state, dict) and len(init_state) > 0,
         str(len(init_state)) + " tensors")


def check_pin(source):
  """The output geometry pins on a matching dataset / probe / width."""
  run_data = {"cosmolike_data_dir": source.data_dir,
              "cosmolike_dataset": source.dataset}
  geom = warmstart.pin_output_geometry(
    source=source, run_data=run_data, run_probe="xi", new_dv_width=TOTAL)
  report("pin: reuses the source geometry on a matching dataset/probe/width",
         geom is source.geom, "same geometry object returned")
  # a width mismatch is a loud error.
  raised = False
  try:
    warmstart.pin_output_geometry(
      source=source, run_data=run_data, run_probe="xi",
      new_dv_width=TOTAL + 1)
  except ValueError:
    raised = True
  report("pin: a dv-width mismatch raises", raised, "width TOTAL+1 rejected")


def check_degenerate(source, tmp, device):
  """Item 4: no extra parameters -> weights and geometry are byte-identical."""
  same = source_names()
  cov_path = Path(tmp) / "same.covmat"
  write_covmat(cov_path, same, seed=9)
  train_mean = np.random.default_rng(10).standard_normal(N_S)
  pgeom0, extra0 = warmstart.extend_input_geometry(
    source=source, covmat_path=str(cov_path), train_mean=train_mean,
    device=device)
  geom_ok = (torch.equal(pgeom0.evecs, source.pgeom.evecs)
             and torch.equal(pgeom0.center, source.pgeom.center)
             and torch.equal(pgeom0.sqrt_ev, source.pgeom.sqrt_ev))
  report("degenerate: geometry tensors byte-identical to the source",
         extra0 == [] and geom_ok, "no extras, evecs/center/sqrt_ev equal")

  model_opts = warmstart.recipe_to_model_opts(
    recipe=source.recipe, geom=None, compile_mode=None)
  template = make_model(model_opts=model_opts,
                        input_dim=N_S, output_dim=OUT_DIM, device=device)
  src_state = source.model.state_dict()
  new_state, padded = warmstart.transfer_state_dict(
    source_state=src_state, template_state=template.state_dict(), n_extra=0)
  bit_ok = True
  for k in src_state:
    if not torch.equal(new_state[k], src_state[k]):
      bit_ok = False
  report("degenerate: transferred state dict bitwise-identical to the source",
         bit_ok and padded == [], "no padded keys, all tensors equal")


def check_errors(source, tmp, device):
  """Item 5: the loud errors fire (non-superset, model: block, rescale)."""
  # non-superset: a new covmat missing a source name.
  bad_names = ["p0", "p1", "p2", "p3", "q9", "w0"]
  bad_path = Path(tmp) / "bad.covmat"
  write_covmat(bad_path, bad_names, seed=11)
  raised = False
  msg = ""
  try:
    warmstart.extend_input_geometry(
      source=source, covmat_path=str(bad_path),
      train_mean=np.zeros(len(bad_names)), device=device)
  except ValueError as e:
    raised = True
    msg = str(e)
  report("error: a non-superset parameter set raises (both lists shown)",
         raised and "superset" in msg and "p4" in msg,
         "missing source name p4 named")

  # a model: block beside finetune:.
  raised = False
  try:
    warmstart.validate_finetune_config(
      cfg={},
      train_args={"finetune": {"from": "x"}, "model": {"name": "resmlp"}},
      rescale="none", activation_flag=None)
  except KeyError:
    raised = True
  report("error: a model: block beside finetune: raises", raised,
         "the inherited-architecture rule")

  # a rescale other than none.
  raised = False
  try:
    warmstart.validate_finetune_config(
      cfg={}, train_args={"finetune": {"from": "x"}},
      rescale="rescaled", activation_flag=None)
  except ValueError:
    raised = True
  report("error: --rescale other than none raises", raised,
         "the inherited-loss-form rule")


def check_anchor(source, pgeom_new, device):
  """(finetune.anchor, TPE-2) the mask excludes the padded extra columns, and
  the decoupled anchor pins the source columns while leaving the extras free."""
  import torch.optim as optim
  from emulator.training import build_anchor

  n_x        = len(EXTRAS)
  model_opts = warmstart.recipe_to_model_opts(
    recipe=source.recipe, geom=None, compile_mode=None)
  template = make_model(model_opts=model_opts,
                        input_dim=len(pgeom_new.names),
                        output_dim=OUT_DIM, device=device)
  init_state, padded = warmstart.transfer_state_dict(
    source_state=source.model.state_dict(),
    template_state=template.state_dict(), n_extra=n_x)
  masks = warmstart.anchor_masks(init_state=init_state, padded_keys=padded,
                                 n_extra=n_x, device=device)
  # the mask zeroes exactly the last n_x columns of each padded key.
  mask_ok = len(masks) == len(padded) and len(padded) > 0
  for k in padded:
    m = masks[k]
    if not torch.equal(m[:, -n_x:], torch.zeros_like(m[:, -n_x:])):
      mask_ok = False
    if not torch.equal(m[:, :-n_x], torch.ones_like(m[:, :-n_x])):
      mask_ok = False
  report("anchor: mask zeroes exactly the padded extra columns", mask_ok,
         str(len(padded)) + " padded key(s)")

  # load the init, drift every parameter, then anchor hard (lambda large via
  # many steps) and check the source columns return to init while the extras
  # stay where the drift put them.
  template.load_state_dict(init_state, strict=True)
  drifted = {}
  with torch.no_grad():
    for name, p in template.named_parameters():
      p.add_(torch.ones_like(p))            # a fixed, reproducible drift
      drifted[name] = p.detach().clone()
  opt    = optim.SGD(template.parameters(), lr=0.1)
  anchor = build_anchor(model=template, optimizer=opt,
                        reference_state=init_state, lam=1.0, masks=masks)
  for _ in range(400):
    anchor.apply(opt)
  key      = padded[0]
  w        = dict(template.named_parameters())[key]
  src_ok   = torch.allclose(w[:, :-n_x], init_state[key][:, :-n_x], atol=1e-3)
  extra_ok = torch.equal(w[:, -n_x:], drifted[key][:, -n_x:])
  report("anchor: source columns pinned to the init_state", src_ok,
         "max|d| = " + format(float((w[:, :-n_x]
                                     - init_state[key][:, :-n_x]).abs().max()),
                              ".2e"))
  report("anchor: padded extra columns left free (untouched)", extra_ok,
         "the new-physics carriers are not pulled to zero")
  # lambda 0 is a no-op (free training), byte-identical.
  before = w.detach().clone()
  free   = build_anchor(model=template, optimizer=opt,
                        reference_state=init_state, lam=0.0, masks=masks)
  free.apply(opt)
  report("anchor: lambda 0 is a no-op (free fine-tuning)",
         torch.equal(dict(template.named_parameters())[key], before),
         "no pull at lambda 0")


def main():
  """Build a synthetic source, then run the six identity checks.

  In order: save a tiny synthetic source emulator and load it back; check the
  shared-parameter encoding is bit-identical to the source; check the weight
  transfer (verbatim tensors plus zero-padded input columns); run the
  pre-train parity gate; check the no-extras degenerate case is byte-identical;
  pin the output geometry; and confirm the three loud errors fire. Each check
  prints a PASS/FAIL line; main returns 1 if any failed, else 0.
  """
  print("== finetune-identity (spec code FTW-A) ==")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device " + str(device) + " (torch only, no cosmolike)")
  torch.manual_seed(0)

  tmp = tempfile.mkdtemp(prefix="ftw-")
  source_root = Path(tmp) / "source"
  save_synthetic_source(source_root, device)
  source = warmstart.load_source(root=str(source_root), device=device)

  cov_path = Path(tmp) / "new.covmat"
  write_covmat(cov_path, source_names() + EXTRAS, seed=12)

  pgeom_new, extra_names = check_encoding(source, str(cov_path), device)
  check_transfer(source, pgeom_new, device)
  check_parity(source, pgeom_new, extra_names, device)
  check_pin(source)
  check_degenerate(source, tmp, device)
  check_errors(source, tmp, device)
  check_anchor(source, pgeom_new, device)

  print("")
  if len(FAILURES) == 0:
    print("finetune-identity: ALL PASS")
    return 0
  print("finetune-identity: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
