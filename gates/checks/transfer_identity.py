#!/usr/bin/env python3
"""transfer-identity: a frozen base plus a zero correction is the base, exactly.

This check defends the promise transfer learning rests on: at epoch 0 the
composed prediction (a frozen base under a parallel correction net whose output
is zero) equals the frozen base's own decode, bit for bit, in every combination
of form (gain / sum) and space (physical / whitened), for a plain base and for
a factored (NLA-like, three-template) base.

It needs torch but no cosmolike: it builds two tiny synthetic bases by hand (a
plain ResMLP and a factored TemplateMLP, each with hand-built geometries),
saves them with save_emulator, loads them back through the transfer loader, and
exercises the transfer path with two extra parameters added.

How it works, in order:
  1. Build and save a synthetic plain base and a synthetic factored base, then
     load each through load_source(allow_factored=True).
  2. Extend the input geometry for a run with two extra parameters and check
     the base's own encoding is a bit-identical column slice of the run's
     encoding (the invariant that makes the base a slice, not a second
     evaluation).
  3. For all four form x space combinations, check the epoch-0 composed
     prediction (zero correction) is bitwise the frozen base's decode, and
     independent of the extras; check the packed target caches the base (the
     base network runs once, at encode); check the zero-init surgery zeros
     exactly the correction's final layer.
  4. The loud errors fire: a non-superset parameter set, and the config
     exclusivities (pce / rescale / finetune / model.ia / unknown form /
     the not-yet-implemented refine block).
  5. check_diagonal (the 2026-07-12 family symmetry ruling): transfer on
     the diagonal families — TransferDiagChi2 epoch-0 identity bitwise
     for both forms through a log-law GridGeometry, the packed-target
     discipline, the whitened-only rejections, the family validators'
     acceptance matrix (cmb law-conditioned), a grid transfer artifact
     rebuilding + predicting the composition bitwise, and the
     cross-family-base loud from_config error.
Every checked value is printed; any failure prints a FAIL line and the run
exits non-zero.

Home note: artifacts-inference-warmstart.md (the transfer design rules;
this check is the transfer-identity validation gate).
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
from emulator.designs.ia import TemplateMLP
from emulator.experiment import (validate_transfer, validate_cmb,
                                 validate_grid, validate_grid2d)
from emulator.geometries.grid import GridGeometry
from emulator.geometries.output import DataVectorGeometry
from emulator.geometries.parameter import ParamGeometry, AmplitudeFactorGeometry
from emulator.losses.ia import nla_coeffs
from emulator.losses.transfer import (TransferChi2, TransferDiagChi2,
                                      FORMS, SPACES)
from emulator.results import save_emulator, rebuild_emulator
from emulator.inference import EmulatorPredictor
from emulator.training import make_model

FAILURES = []

OUT_DIM = 8       # kept dv entries (one template wide)
TOTAL   = 12      # full dv length
EXTRAS  = ["w0", "wa"]
AMP     = "LSST_A1_1"


def report(label, ok, detail):
  """Print one PASS/FAIL line and remember any failure."""
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


def emit_aid(aid, n_before):
  """Emit ONE '##AID <aid> <PASS|FAIL>' line for a whole acceptance leg.

  (queue 2) The board's run_check folds these reserved lines into the gate's
  executed set: one per declared leg, at the leg's aggregation point, not one
  per sub-check. Most legs here are one check_* function; check_diagonal
  carries two (the composition legs and the cross-family refusal), so it emits
  its own pair. A leg's verdict is FAIL if its group appended any label to the
  module-level FAILURES list while it ran; the child's exit status stays the
  single aggregate verdict.

  Arguments:
    aid      = the board-unique leg id, "transfer-identity.<leg>".
    n_before = len(FAILURES) captured immediately before the leg's checks ran.
  """
  mark = "PASS" if len(FAILURES) == n_before else "FAIL"
  print("##AID " + aid + " " + mark)


def spd(n, seed):
  g = np.random.default_rng(seed)
  a = g.standard_normal((n, n))
  return a @ a.T + n * np.eye(n)


def dv_geometry(device):
  """A synthetic output geometry (the first OUT_DIM entries survive)."""
  cov_k = spd(OUT_DIM, seed=3)
  lam_k, V_k = np.linalg.eigh(cov_k)
  return DataVectorGeometry(
    device=device,
    total_size=TOTAL,
    dest_idx=list(range(OUT_DIM)),
    evecs=V_k,
    sqrt_ev=np.sqrt(lam_k),
    Cinv=spd(TOTAL, seed=4),
    center=np.random.default_rng(5).standard_normal(OUT_DIM),
    section_sizes=[TOTAL],
    probe="xi")


def param_geometry(names, device, seed):
  """A synthetic plain ParamGeometry over `names`."""
  n = len(names)
  lam, V = np.linalg.eigh(spd(n, seed))
  center = np.random.default_rng(seed + 1).standard_normal(n)
  return ParamGeometry(device=device, names=names, center=center,
                       evecs=V, sqrt_ev=np.sqrt(lam))


def base_config():
  """The data block a saved base records (a fine-tune / transfer run matches)."""
  return {"data": {"cosmolike_data_dir": "lsst_y1",
                   "cosmolike_dataset": "lsst_y1_M1_GGL0.05.dataset",
                   "train_dv": "b.npy",
                   "val_dv": "bv.npy"},
          "train_args": {"nepochs": 1}}


def histories():
  return {"train_losses": [0.1],
          "val_medians": [0.1],
          "val_means": [0.1],
          "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
          "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}


def save_plain_base(root, device):
  """Save a synthetic plain base (ResMLP + ParamGeometry)."""
  names = ["p0", "p1", "p2"]
  pg = param_geometry(names, device, seed=10)
  geom = dv_geometry(device)
  block_opts = {"act": make_activation("H", n_gates=3),
                "norm": make_norm("affine")}
  model = ResMLP(input_dim=len(names), output_dim=OUT_DIM, int_dim_res=32,
                 n_blocks=2, block_opts=block_opts).to(device)
  recipe = {"cls": "emulator.designs.plain.ResMLP",
            "name": "resmlp",
            "ia": None,
            "input_dim": len(names),
            "output_dim": OUT_DIM,
            "compile_mode": None,
            "needs_geom": False,
            "kwargs": {"int_dim_res": 32,
                       "n_blocks": 2,
                       "block_opts": {"act": {"type": "H", "n_gates": 3},
                                      "norm": "affine"}}}
  save_emulator(path_root=str(root), model=model, param_geometry=pg,
                geometry=geom, config=base_config(), histories=histories(),
                train_args=base_config()["train_args"],
                resolved_train={"nepochs": 1}, resolved_model=recipe,
                attrs={"rescale": "none"})
  return names


def save_factored_base(root, device):
  """Save a synthetic factored (nla-like, T=3) base (TemplateMLP)."""
  keep_names = ["p0", "p1", "p2"]
  names = keep_names + [AMP]                       # amps appended last
  pg_keep = param_geometry(keep_names, device, seed=20)
  pg = AmplitudeFactorGeometry(device=device, pg_keep=pg_keep,
                               amp_idx=[len(keep_names)], n_param=len(names),
                               names=names)
  geom = dv_geometry(device)
  block_opts = {"act": make_activation("H", n_gates=3),
                "norm": make_norm("affine")}
  model = TemplateMLP(input_dim=len(names), output_dim=OUT_DIM, n_amps=1,
                      n_templates=3, int_dim_res=32, n_blocks=2,
                      block_opts=block_opts).to(device)
  recipe = {"cls": "emulator.designs.ia.TemplateMLP",
            "name": "resmlp",
            "ia": "nla",
            "input_dim": len(names),
            "output_dim": OUT_DIM,
            "compile_mode": None,
            "needs_geom": False,
            "kwargs": {"n_amps": 1,
                       "n_templates": 3,
                       "int_dim_res": 32,
                       "n_blocks": 2,
                       "block_opts": {"act": {"type": "H", "n_gates": 3},
                                      "norm": "affine"}}}
  save_emulator(path_root=str(root), model=model, param_geometry=pg,
                geometry=geom, config=base_config(), histories=histories(),
                train_args=base_config()["train_args"],
                resolved_train={"nepochs": 1}, resolved_model=recipe,
                attrs={"rescale": "none"})
  return names


def write_covmat(path, names, seed):
  cov = spd(len(names), seed=seed)
  with open(path, "w") as f:
    f.write("# " + " ".join(names) + "\n")
    for row in cov:
      f.write(" ".join(repr(float(x)) for x in row) + "\n")


def make_transfer(base, geom, form, space):
  """Build a TransferChi2 for one base + form + space (as build_geometry does)."""
  if base.ia is None:
    return TransferChi2(geom=geom, base_net=base.model,
                        base_in_dim=len(base.pgeom.names), form=form,
                        space=space, n_templates=1, n_amps=0, coeff_fn=None)
  return TransferChi2(geom=geom, base_net=base.model,
                      base_in_dim=len(base.pgeom.pg_keep.names), form=form,
                      space=space, n_templates=3, n_amps=1,
                      coeff_fn=nla_coeffs)


def check_base(tag, root, device, tmp, factored):
  """Run the slice + identity + packing + surgery checks for one base."""
  base = warmstart.load_source(root=str(root), device=device,
                               allow_factored=True)
  # extend for a run with two extra parameters.
  if factored:
    new_names = ["p0", "p1", "p2"] + EXTRAS + [AMP]
  else:
    new_names = ["p0", "p1", "p2"] + EXTRAS
  cov_path = Path(tmp) / (tag + ".covmat")
  write_covmat(cov_path, new_names, seed=30)
  train_mean = np.random.default_rng(31).standard_normal(len(new_names))
  new_pgeom, extra_names = warmstart.extend_input_geometry(
    source=base, covmat_path=str(cov_path), train_mean=train_mean,
    device=device)
  report(tag + " extend: extras are [w0, wa]", extra_names == EXTRAS,
         str(extra_names))

  theta = torch.from_numpy(
    np.random.default_rng(32).standard_normal((64, len(new_names)))
  ).float().to(device)
  enc = new_pgeom.encode(theta)

  # (item 1) the base's own encoding is a bit-identical column slice.
  if factored:
    n_sp = len(base.pgeom.pg_keep.names)
    slice_enc = torch.cat([enc[:, :n_sp], enc[:, -1:]], dim=1)
    # the base's own encoding of the shared params + the raw amp.
    base_cols = []
    for nm in base.pgeom.names:
      base_cols.append(new_names.index(nm))
    base_theta = theta[:, base_cols]
    base_enc = base.pgeom.encode(base_theta)
  else:
    n_sp = len(base.pgeom.names)
    slice_enc = enc[:, :n_sp]
    base_cols = []
    for nm in base.pgeom.names:
      base_cols.append(new_names.index(nm))
    base_enc = base.pgeom.encode(theta[:, base_cols])
  report(tag + " slice: base encoding is a bitwise column slice of enc",
         torch.equal(slice_enc, base_enc),
         "max|d| = " + format(float((slice_enc - base_enc).abs().max()), ".2e"))

  geom = base.geom
  for form in FORMS:
    for space in SPACES:
      chi2fn = make_transfer(base, geom, form, space)
      # (item 3) the packed target width, and the base runs once at encode.
      nk = OUT_DIM
      want = (4 if factored else 2) * nk
      report(tag + " " + form + "/" + space + " target_dim",
             chi2fn.target_dim == want,
             str(chi2fn.target_dim) + " want " + str(want))
      calls = {"n": 0}
      def _hook(_m, _i, _o, _c=calls):
        _c["n"] += 1
      handle = base.model.register_forward_hook(_hook)
      dv_rows = np.random.default_rng(33).standard_normal((64, TOTAL))
      dv = torch.from_numpy(dv_rows).float().to(device)
      with torch.no_grad():
        target = chi2fn.encode(dv, enc)
        zero = torch.zeros(64, 3, nk, device=device) if factored \
            else torch.zeros(64, nk, device=device)
        n_after_encode = calls["n"]
        _ = chi2fn.chi2(zero, target, enc)
      handle.remove()
      report(tag + " " + form + "/" + space + " base cached (chi2 no recompute)",
             calls["n"] == n_after_encode and n_after_encode >= 1,
             "encode base calls " + str(n_after_encode) + ", chi2 added "
             + str(calls["n"] - n_after_encode))

      # (item 2) zero-init identity + extras independence, via build_transfer_start.
      model_opts = _correction_opts(base, geom)
      train_set = {"C": theta.cpu().numpy(), "idx": np.arange(64)}
      init_state, verdict = warmstart.build_transfer_start(
        chi2fn=chi2fn, model_opts=model_opts, new_pgeom=new_pgeom,
        train_set=train_set, extra_names=extra_names, device=device)
      report(tag + " " + form + "/" + space + " epoch-0 identity + extras-indep",
             verdict.startswith("[ok] transfer parity"), verdict)


def _correction_opts(base, geom):
  """The correction model_opts (mirrors build_specs' inherited-family path)."""
  block_opts = {"act": make_activation("H", n_gates=3),
                "norm": make_norm("affine")}
  if base.ia is None:
    return {"cls": ResMLP,
            "compile_mode": None,
            "int_dim_res": 16,
            "n_blocks": 1,
            "block_opts": block_opts}
  return {"cls": TemplateMLP,
          "compile_mode": None,
          "int_dim_res": 16,
          "n_blocks": 1,
          "block_opts": block_opts,
          "n_amps": 1,
          "n_templates": 3}


def check_zero_init(device):
  """(item 4) the surgery zeros exactly the correction's final Linear."""
  block_opts = {"act": make_activation("H", n_gates=3),
                "norm": make_norm("affine")}
  model = ResMLP(input_dim=5, output_dim=OUT_DIM, int_dim_res=16, n_blocks=1,
                 block_opts=block_opts).to(device)
  before = {}
  for k, v in model.state_dict().items():
    before[k] = v.detach().clone()
  last = warmstart._zero_final_linear(model)
  after = model.state_dict()
  # the last Linear's weight + bias are exactly zero.
  zeroed = torch.equal(last.weight, torch.zeros_like(last.weight)) \
      and torch.equal(last.bias, torch.zeros_like(last.bias))
  # every other tensor is untouched.
  last_ids = {id(last.weight), id(last.bias)}
  untouched = True
  for k, v in model.named_parameters():
    if id(v) in last_ids:
      continue
    if not torch.equal(v.detach(), before[k]):
      untouched = False
  report("zero-init: final Linear exactly zero", zeroed, "weight + bias == 0")
  report("zero-init: every other tensor untouched", untouched,
         "only the final Linear changed")


def check_errors(device, plain_root):
  """(item 5) the loud config errors fire."""
  base_cfg = {"transfer": {"from": "x", "form": "gain"}}
  # unknown form.
  raised = False
  try:
    validate_transfer(cfg={"transfer": {"from": "x", "form": "product"}},
                      train_args={"model": {}}, rescale="none")
  except ValueError:
    raised = True
  report("error: unknown transfer.form raises", raised, "product rejected")
  # rescale exclusivity.
  raised = False
  try:
    validate_transfer(cfg=base_cfg, train_args={"model": {}},
                      rescale="rescaled")
  except ValueError:
    raised = True
  report("error: transfer + --rescale raises", raised, "rescale must be none")
  # pce exclusivity.
  raised = False
  try:
    validate_transfer(cfg={"transfer": {"from": "x", "form": "gain"},
                           "pce": {"form": "residual"}},
                      train_args={"model": {}}, rescale="none")
  except ValueError:
    raised = True
  report("error: transfer + pce raises", raised, "exclusive losses")
  # finetune exclusivity.
  raised = False
  try:
    validate_transfer(cfg=base_cfg,
                      train_args={"model": {}, "finetune": {"from": "y"}},
                      rescale="none")
  except ValueError:
    raised = True
  report("error: transfer + finetune raises", raised, "different tools")
  # model.ia (family inherited).
  raised = False
  try:
    validate_transfer(cfg=base_cfg, train_args={"model": {"ia": "nla"}},
                      rescale="none")
  except ValueError:
    raised = True
  report("error: transfer + model.ia raises", raised, "family inherited")
  # refine knobs are explicit (no silent defaults): a refine block missing
  # base_lr_scale / anchor is a loud error, the same required-explicit rule
  # the anchor itself follows. (This leg replaced the V1 not-yet-implemented
  # rejection when the refine unit landed.)
  raised = False
  try:
    validate_transfer(cfg={"transfer": {"from": "x",
                                        "form": "gain",
                                        "refine": {"epochs": 1}}},
                      train_args={"model": {}}, rescale="none")
  except ValueError:
    raised = True
  report("error: an incomplete refine block raises (explicit knobs)", raised,
         "base_lr_scale / anchor must be stated, no silent defaults")
  # non-superset names (extend raises).
  base = warmstart.load_source(root=str(plain_root), device=device,
                               allow_factored=True)
  tmp = tempfile.mkdtemp(prefix="tpe-e-")
  bad = Path(tmp) / "bad.covmat"
  write_covmat(bad, ["p0", "p1", "q9", "w0"], seed=40)
  raised = False
  try:
    warmstart.extend_input_geometry(source=base, covmat_path=str(bad),
                                    train_mean=np.zeros(4), device=device)
  except ValueError:
    raised = True
  report("error: a non-superset parameter set raises", raised,
         "missing p2 named")


def check_lifecycle(device, tmp):
  """(the artifact lifecycle) save a transfer, rebuild, compose again.

  Saves a transfer artifact (the correction net as the main model plus the
  frozen base embedded whole), rebuilds it, and checks the composed prediction
  reproduces the in-memory composition bit for bit (the save-rebuild-drift
  pattern applied to the composed model), both through rebuild_emulator + the
  transfer decoder and through the full EmulatorPredictor path. Then confirms
  chaining is refused: the saved transfer cannot itself be loaded as a base.
  A plain base, gain/physical, names-equal (n_x = 0), with the correction's
  ordinary nonzero weights so the composition is nontrivial.
  """
  base_root = Path(tmp) / "life_base"
  save_plain_base(base_root, device)
  base = warmstart.load_source(root=str(base_root), device=device,
                               allow_factored=True)

  # a names-equal run geometry (no extras), its own fresh covmat.
  names    = list(base.pgeom.names)
  cov_path = Path(tmp) / "life.covmat"
  write_covmat(cov_path, names, seed=71)
  train_mean = np.random.default_rng(72).standard_normal(len(names))
  new_pgeom, _extra = warmstart.extend_input_geometry(
    source=base, covmat_path=str(cov_path), train_mean=train_mean,
    device=device)
  geom   = base.geom
  chi2fn = make_transfer(base, geom, "gain", "physical")

  # a correction net at its ordinary (nonzero) init, so the composition is not
  # the trivial identity; the round-trip must reproduce it bitwise.
  in_dim = getattr(new_pgeom, "encoded_dim", len(new_pgeom.names))
  corr   = make_model(model_opts=dict(_correction_opts(base, geom),
                                      compile_mode=None),
                      input_dim=in_dim, output_dim=OUT_DIM, device=device)
  corr.eval()
  theta_rows = np.random.default_rng(73).standard_normal((8, len(names)))
  theta = torch.from_numpy(theta_rows).float().to(device)
  with torch.no_grad():
    enc      = new_pgeom.encode(theta)
    composed = chi2fn.decode(corr(enc), enc)          # in-memory (8, n_keep)

  # save the transfer artifact exactly as the driver assembles it.
  corr_recipe = {"cls": "emulator.designs.plain.ResMLP",
                 "name": "resmlp",
                 "ia": None,
                 "input_dim": int(in_dim),
                 "output_dim": OUT_DIM,
                 "compile_mode": None,
                 "needs_geom": False,
                 "kwargs": {"int_dim_res": 16,
                            "n_blocks": 1,
                            "block_opts": {"act": {"type": "H", "n_gates": 3},
                                           "norm": "affine"}}}
  transfer_base = {"recipe": base.recipe,
                   "state": base.model.state_dict(),
                   "param_geometry": base.pgeom,
                   "dv_geometry": base.geom,
                   "form": "gain",
                   "space": "physical"}
  saved = Path(tmp) / "life_transfer"
  save_emulator(path_root=str(saved), model=corr, param_geometry=new_pgeom,
                geometry=geom, config=base_config(), histories=histories(),
                train_args=base_config()["train_args"],
                resolved_train={"nepochs": 1}, resolved_model=corr_recipe,
                transfer_base=transfer_base, attrs={"rescale": "none"})

  # rebuild + compose again through the transfer decoder.
  model_r, pgeom_r, geom_r, info = rebuild_emulator(
    str(saved), device, compile_model=False)
  report("lifecycle: rebuild returns the embedded base + form/space",
         info["transfer_base"] is not None and info["transfer_form"] == "gain"
         and info["transfer_space"] == "physical",
         "form " + str(info["transfer_form"]) + "/"
         + str(info["transfer_space"]))
  base_r = info["transfer_base"]
  chi2_r = TransferChi2(geom=geom_r, base_net=base_r["model"],
                        base_in_dim=len(base_r["pgeom"].names), form="gain",
                        space="physical", n_templates=1, n_amps=0,
                        coeff_fn=None)
  with torch.no_grad():
    composed_r = chi2_r.decode(model_r(pgeom_r.encode(theta)),
                               pgeom_r.encode(theta))
  report("lifecycle: save -> rebuild -> composed predict bitwise == in-memory",
         torch.equal(composed, composed_r),
         "max|d| = " + format(float((composed - composed_r).abs().max()), ".2e"))

  # the full inference path composes too (dv_return 3x2pt to skip sectioning).
  predictor = EmulatorPredictor(str(saved), device, dv_return="3x2pt")
  row = {}
  for i, nm in enumerate(names):
    row[nm] = float(theta[0, i])
  got  = predictor.predict(row)
  want = geom.unsqueeze(composed[:1]).detach().cpu().numpy()[0]
  # NOT bitwise: the predictor runs batch-1 on the dict-input inference path
  # while the reference row is sliced from the batch-64 in-memory run --
  # different matmul shapes regroup the float reductions (~1 ulp; the same
  # kernel-reassociation caveat pre-authorized in the fine-tune audit). Bitwise is
  # only demanded where the path is literally the same computation (the
  # save -> rebuild -> composed-predict leg above).
  report("lifecycle: EmulatorPredictor.predict == in-memory unsqueeze (1e-6)",
         bool(np.abs(got - want).max() <= 1.0e-6),
         "max|d| = " + format(float(np.abs(got - want).max()), ".2e"))

  # chaining refused: the saved transfer cannot be loaded as a new base.
  raised = False
  try:
    warmstart.load_source(root=str(saved), device=device, allow_factored=True)
  except ValueError:
    raised = True
  report("lifecycle: chaining refused (a transfer cannot be a base)", raised,
         "load_source rejects the embedded transfer_base group")


def check_refined_lifecycle(device, tmp):
  """(refine artifact) a refined transfer saves the drifted base to a
  drifted_state group; rebuild composes with the DRIFTED base bitwise, and the
  transfer_refined attr is two-way consistent with the group (either half alone
  is a corrupt file)."""
  base_root = Path(tmp) / "ref_base"
  save_plain_base(base_root, device)
  base = warmstart.load_source(root=str(base_root), device=device,
                               allow_factored=True)
  names    = list(base.pgeom.names)
  cov_path = Path(tmp) / "ref.covmat"
  write_covmat(cov_path, names, seed=81)
  new_pgeom, _ = warmstart.extend_input_geometry(
    source=base, covmat_path=str(cov_path),
    train_mean=np.zeros(len(names)), device=device)
  geom = base.geom

  # the pretrained clone (kept for transfer_base/state), then drift the base in
  # place to stand in for stage 2, so the loss composes with the drifted base.
  pretrained = {}
  for k, v in base.model.state_dict().items():
    pretrained[k] = v.detach().clone()
  with torch.no_grad():
    for p in base.model.parameters():
      p.add_(0.05 * torch.randn_like(p))
  drifted = base.model.state_dict()

  chi2fn = make_transfer(base, geom, "gain", "physical")
  in_dim = getattr(new_pgeom, "encoded_dim", len(new_pgeom.names))
  corr   = make_model(model_opts=dict(_correction_opts(base, geom),
                                      compile_mode=None),
                      input_dim=in_dim, output_dim=OUT_DIM, device=device)
  corr.eval()
  theta_rows = np.random.default_rng(82).standard_normal((8, len(names)))
  theta = torch.from_numpy(theta_rows).float().to(device)
  with torch.no_grad():
    enc      = new_pgeom.encode(theta)
    composed = chi2fn.decode(corr(enc), enc)          # in-memory, drifted base

  corr_recipe = {"cls": "emulator.designs.plain.ResMLP",
                 "name": "resmlp",
                 "ia": None,
                 "input_dim": int(in_dim),
                 "output_dim": OUT_DIM,
                 "compile_mode": None,
                 "needs_geom": False,
                 "kwargs": {"int_dim_res": 16,
                            "n_blocks": 1,
                            "block_opts": {"act": {"type": "H", "n_gates": 3},
                                           "norm": "affine"}}}
  transfer_base = {"recipe": base.recipe,
                   "state": pretrained,
                   "drifted_state": drifted,
                   "param_geometry": base.pgeom,
                   "dv_geometry": base.geom,
                   "form": "gain",
                   "space": "physical"}
  saved = Path(tmp) / "ref_transfer"
  save_emulator(path_root=str(saved), model=corr, param_geometry=new_pgeom,
                geometry=geom, config=base_config(), histories=histories(),
                train_args=base_config()["train_args"],
                resolved_train={"nepochs": 1}, resolved_model=corr_recipe,
                transfer_base=transfer_base, attrs={"rescale": "none"})

  model_r, pgeom_r, geom_r, info = rebuild_emulator(
    str(saved), device, compile_model=False)
  base_r = info["transfer_base"]
  chi2_r = TransferChi2(geom=geom_r, base_net=base_r["model"],
                        base_in_dim=len(base_r["pgeom"].names), form="gain",
                        space="physical", n_templates=1, n_amps=0,
                        coeff_fn=None)
  with torch.no_grad():
    composed_r = chi2_r.decode(model_r(pgeom_r.encode(theta)),
                               pgeom_r.encode(theta))
  report("refined: composed predict uses the drifted base, bitwise",
         torch.equal(composed, composed_r),
         "max|d| = " + format(float((composed - composed_r).abs().max()), ".2e"))

  # two-way consistency: strip the transfer_refined attr but keep drifted_state;
  # rebuild must refuse (either half alone is corrupt).
  import h5py
  with h5py.File(str(saved) + ".h5", "r+") as f:
    del f.attrs["transfer_refined"]
  raised = False
  try:
    rebuild_emulator(str(saved), device, compile_model=False)
  except KeyError:
    raised = True
  report("refined: two-way consistency (drifted_state without the attr raises)",
         raised, "a refined artifact must carry both halves")


def grid_base_recipe(names, nz):
  """The model_recipe a schema-v2 save stores for a grid-family ResMLP."""
  return {"cls": "emulator.designs.plain.ResMLP",
          "name": "resmlp",
          "ia": None,
          "input_dim": len(names),
          "output_dim": nz,
          "compile_mode": None,
          "needs_geom": False,
          "kwargs": {"int_dim_res": 16,
                     "n_blocks": 2,
                     "block_opts": {"act": {"type": "H", "n_gates": 3},
                                    "norm": "affine"}}}


def check_diagonal(device, tmp):
  """The 2026-07-12 symmetry ruling: transfer on the diagonal families.

  TransferDiagChi2 on a GridGeometry (log-offset law): the epoch-0
  identity is bitwise for BOTH forms (zero correction = the frozen
  base, through the law), the packed target caches the base, the
  physical space is loudly refused, validate_transfer(diagonal=True)
  resolves/rejects as ruled, the family validators accept the block
  (cmb only under amplitude_law none), a transfer artifact rebuilds
  and predicts the composition bitwise, and a cross-family base is a
  loud from_config error.

  This function carries TWO declared legs, so it emits its own aids rather
  than being wrapped by one emit_aid in main: the composition legs, and the
  cross-family refusal (a leg whose fixture is red in the register — it stays
  its own aid so the red names itself instead of hiding inside a group).
  """
  from emulator.experiment import EmulatorExperiment
  n_composition = len(FAILURES)
  names = ["p0", "p1", "p2"]
  pg = param_geometry(names, device, seed=90)
  z = np.linspace(0.001, 3.0, 32)
  g = np.random.default_rng(91)
  rows = 70.0 * (1.0 + 0.05 * g.standard_normal((300, z.size)))
  geom = GridGeometry.from_targets(device=device, targets=rows, z=z,
                                   quantity="Hubble", units="km/s/Mpc",
                                   law="log_offset", offset=1.0)
  block_opts = {"act": make_activation("H", n_gates=3),
                "norm": make_norm("affine")}
  base = ResMLP(input_dim=len(names), output_dim=z.size, int_dim_res=16,
                n_blocks=2, block_opts=block_opts).to(device)
  base.eval()
  theta_rows = np.random.default_rng(92).standard_normal((8, len(names)))
  theta = torch.from_numpy(theta_rows).float().to(device)
  dv = torch.from_numpy(rows[:8].astype("float32")).to(device)
  with torch.no_grad():
    enc = pg.encode(theta)
  for form in FORMS:
    chi2fn = TransferDiagChi2(geom=geom, base_net=base,
                              base_in_dim=len(names), form=form,
                              space="whitened")
    zero = torch.zeros(8, z.size, device=device)
    with torch.no_grad():
      composed = chi2fn.decode(zero, enc)
      base_dec = chi2fn.base_decode(enc)
      target = chi2fn.encode(dv, enc)
      c_zero = chi2fn.chi2(pred=zero, target=target,
                           params_whitened=enc)
      base_w = chi2fn._base(enc)
      want_c = ((base_w - geom.encode(dv)) ** 2).sum(dim=1)
    report("diag %s: epoch-0 identity bitwise through the law" % form,
           torch.equal(composed, base_dec),
           "max|d| %.1e" % (composed - base_dec).abs().max().item())
    report("diag %s: packed target + zero-correction chi2 exact" % form,
           target.shape == (8, 2 * z.size) and torch.equal(c_zero, want_c),
           "target %s" % (tuple(target.shape),))
  try:
    TransferDiagChi2(geom=geom, base_net=base, base_in_dim=len(names),
                     form="sum", space="physical")
    report("diag: physical space refused", False, "no raise")
  except ValueError as e:
    report("diag: physical space refused", "metric basis" in str(e),
           "ValueError names the basis")
  # validate_transfer(diagonal=True): whitened resolution for both
  # forms, the explicit-physical rejection, the gain notice, and the
  # refine rejection.
  def tr_cfg(block):
    return {"transfer": block, "data": {}}
  res, note = validate_transfer(tr_cfg({"from": "x", "form": "sum"}),
                                train_args={}, diagonal=True)
  ok = res["space"] == "whitened" and note is None
  res, note = validate_transfer(tr_cfg({"from": "x", "form": "gain"}),
                                train_args={}, diagonal=True)
  ok = ok and res["space"] == "whitened" and note is not None
  report("diag validate: whitened resolution + the gain notice", ok,
         "gain notice: %s" % ("present" if note else "MISSING"))
  try:
    validate_transfer(tr_cfg({"from": "x",
                              "form": "sum",
                              "space": "physical"}),
                      train_args={}, diagonal=True)
    report("diag validate: explicit physical raises", False, "no raise")
  except ValueError:
    report("diag validate: explicit physical raises", True, "")
  try:
    validate_transfer(tr_cfg({"from": "x",
                              "form": "sum",
                              "refine": {"epochs": 5,
                                         "base_lr_scale": 0.1,
                                         "anchor": 0.0}}),
                      train_args={}, diagonal=True)
    report("diag validate: refine rejected (frozen-base V1)", False,
           "no raise")
  except ValueError:
    report("diag validate: refine rejected (frozen-base V1)", True, "")
  # the family validators accept the block now (cmb only law-none).
  grid_data = {"grid": {"quantity": "Hubble",
                        "units": "km/s/Mpc",
                        "law": "log_offset",
                        "offset": 1.0,
                        "z_file": "z.npy"},
               "train_dv": "a",
               "val_dv": "b",
               "train_params": "c",
               "val_params": "d",
               "train_covmat": "e"}
  cfg = {"data": grid_data,
         "pce": None,
         "transfer": {"from": "x", "form": "sum"}}
  try:
    validate_grid(cfg, train_args={}, rescale="none")
    report("validate_grid accepts a transfer block", True, "")
  except ValueError as e:
    report("validate_grid accepts a transfer block", False, str(e)[:70])
  cmb_data = {"cmb": {"spectrum": "tt",
                      "covariance": "c.npz",
                      "amplitude_law": "none"},
              "train_dv": "a",
              "val_dv": "b",
              "train_params": "c",
              "val_params": "d",
              "train_covmat": "e"}
  cfg = {"data": cmb_data,
         "pce": None,
         "transfer": {"from": "x", "form": "sum"}}
  try:
    validate_cmb(cfg, train_args={}, rescale="none")
    report("validate_cmb accepts transfer under law none", True, "")
  except ValueError as e:
    report("validate_cmb accepts transfer under law none", False,
           str(e)[:70])
  cfg["data"]["cmb"] = {"spectrum": "tt",
                        "covariance": "c.npz",
                        "amplitude_law": "as_exp2tau_ref",
                        "as_name": "As",
                        "tau_name": "tau",
                        "as_ref": 2.1e-9,
                        "tau_ref": 0.0544}
  try:
    validate_cmb(cfg, train_args={}, rescale="none")
    report("validate_cmb: transfer x amplitude-law raises", False,
           "no raise")
  except ValueError as e:
    report("validate_cmb: transfer x amplitude-law raises",
           "amplitude_law: none" in str(e), "names the fix")
  # save -> rebuild -> predict: a grid transfer artifact (zero-init
  # correction + the embedded base) must predict the composition
  # bitwise, i.e. exactly the frozen base at epoch 0.
  corr = ResMLP(input_dim=len(names), output_dim=z.size, int_dim_res=8,
                n_blocks=1, block_opts=dict(block_opts)).to(device)
  warmstart._zero_final_linear(corr)
  corr.eval()
  corr_recipe = grid_base_recipe(names, int(z.size))
  corr_recipe["kwargs"]["int_dim_res"] = 8
  corr_recipe["kwargs"]["n_blocks"] = 1
  root = Path(tmp) / "diag_transfer"
  config = {"data": {"grid": {"quantity": "Hubble",
                              "units": "km/s/Mpc",
                              "law": "log_offset",
                              "offset": 1.0,
                              "z_file": "z.npy"},
                     "train_dv": "t.npy",
                     "val_dv": "v.npy",
                     "train_params": "t.1.txt",
                     "val_params": "v.1.txt",
                     "train_covmat": "c.covmat"},
            "transfer": {"from": "grid_base", "form": "sum"},
            "train_args": {"nepochs": 1}}
  save_emulator(path_root=str(root), model=corr, param_geometry=pg,
                geometry=geom, config=config, histories=histories(),
                train_args=config["train_args"],
                resolved_train={"nepochs": 1},
                resolved_model=corr_recipe,
                transfer_base={"recipe": grid_base_recipe(names,
                                                          int(z.size)),
                               "state": base.state_dict(),
                               "param_geometry": pg,
                               "dv_geometry": geom,
                               "form": "sum",
                               "space": "whitened"},
                attrs={"rescale": "none", "quantity": "Hubble"})
  chi2fn = TransferDiagChi2(geom=geom, base_net=base,
                            base_in_dim=len(names), form="sum",
                            space="whitened")
  theta1 = np.array([[0.3, -0.2, 1.1]])
  x1 = torch.as_tensor(theta1, dtype=pg.center.dtype, device=device)
  with torch.no_grad():
    ref = chi2fn.base_decode(pg.encode(x1))[0].cpu().numpy()
  pred = EmulatorPredictor(str(root), device, compile_model=False)
  got = pred.predict({nm: float(theta1[0, i])
                      for i, nm in enumerate(names)})
  report("diag transfer artifact predicts the composition bitwise",
         np.array_equal(got["Hubble"], ref),
         "max|d| %.1e" % np.abs(got["Hubble"] - ref).max())
  emit_aid("transfer-identity.diagonal-family-composition", n_composition)

  n_cross = len(FAILURES)
  # a cross-family base is a loud from_config error (before staging).
  # FIXTURE: the base must be invalid ONLY in the way under test
  # (cross-family), so save a PLAIN grid base — this leg's base net +
  # GridGeometry through the same save_emulator call, no transfer_base
  # group — at its own root. Pointing at the diag_transfer artifact
  # above would trip load_source's chaining refusal (it embeds a
  # transfer_base) before the family-kind check, and the needle test
  # would pass on the wrong message. The chaining refusal keeps its own
  # dedicated leg in check_lifecycle.
  plain_root = Path(tmp) / "plain_grid_base"
  plain_cfg = {"data": {"grid": {"quantity": "Hubble",
                                 "units": "km/s/Mpc",
                                 "law": "log_offset",
                                 "offset": 1.0,
                                 "z_file": "z.npy"},
                        "train_dv": "t.npy",
                        "val_dv": "v.npy",
                        "train_params": "t.1.txt",
                        "val_params": "v.1.txt",
                        "train_covmat": "c.covmat"},
               "train_args": {"nepochs": 1}}
  save_emulator(path_root=str(plain_root), model=base, param_geometry=pg,
                geometry=geom, config=plain_cfg, histories=histories(),
                train_args=plain_cfg["train_args"],
                resolved_train={"nepochs": 1},
                resolved_model=grid_base_recipe(names, int(z.size)),
                attrs={"rescale": "none", "quantity": "Hubble"})
  g2_cfg = {"data": {"grid2d": {"quantity": "pklin",
                                "units": "Mpc3",
                                "law": "syren_linear",
                                "z_file": "z.npy",
                                "k_file": "k.npy",
                                "train_base": "tb.npy",
                                "val_base": "vb.npy"},
                     "train_dv": "t.npy",
                     "val_dv": "v.npy",
                     "train_params": "t.1.txt",
                     "val_params": "v.1.txt",
                     "train_covmat": "c.covmat"},
            "transfer": {"from": str(plain_root), "form": "sum"},
            "train_args": {"nepochs": 1, "bs": 8}}
  try:
    EmulatorExperiment.from_config(g2_cfg, device=torch.device("cpu"))
    report("cross-family transfer base raises", False, "no raise")
  except ValueError as e:
    report("cross-family transfer base raises",
           "never" in str(e) and "families" in str(e),
           "ValueError names the rule")
  emit_aid("transfer-identity.cross-family-base-refusal", n_cross)


def main():
  """Build synthetic plain + factored bases and run the transfer-identity checks.

  In order: save and load a plain base and a factored base; for each, check the
  base encoding is a bitwise column slice of the run's encoding, and, for all
  four form x space combinations, the epoch-0 identity, the base caching, and
  the packed target width; then the zero-init surgery and the loud config
  errors. Each check prints a PASS/FAIL line; main returns 1 if any failed.
  """
  print("== transfer-identity ==")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device " + str(device) + " (torch only, no cosmolike)")
  torch.manual_seed(0)

  tmp = tempfile.mkdtemp(prefix="tpe-")
  plain_root = Path(tmp) / "plain_base"
  factored_root = Path(tmp) / "factored_base"
  save_plain_base(plain_root, device)
  save_factored_base(factored_root, device)

  # One ##AID per declared leg, at its aggregation point (see emit_aid).
  # check_diagonal emits its own two aids, so it is not wrapped here.
  n0 = len(FAILURES)
  check_base("plain", plain_root, device, tmp, factored=False)
  emit_aid("transfer-identity.plain-base-slice-and-identity", n0)

  n0 = len(FAILURES)
  check_base("factored", factored_root, device, tmp, factored=True)
  emit_aid("transfer-identity.factored-base-slice-and-identity", n0)

  n0 = len(FAILURES)
  check_zero_init(device)
  emit_aid("transfer-identity.zero-init-surgery", n0)

  n0 = len(FAILURES)
  check_errors(device, plain_root)
  emit_aid("transfer-identity.loud-config-errors", n0)

  n0 = len(FAILURES)
  check_lifecycle(device, tmp)
  emit_aid("transfer-identity.artifact-lifecycle-round-trip", n0)

  n0 = len(FAILURES)
  check_refined_lifecycle(device, tmp)
  emit_aid("transfer-identity.refined-base-lifecycle", n0)

  check_diagonal(device, tmp)

  print("")
  if len(FAILURES) == 0:
    print("transfer-identity: ALL PASS")
    return 0
  print("transfer-identity: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
