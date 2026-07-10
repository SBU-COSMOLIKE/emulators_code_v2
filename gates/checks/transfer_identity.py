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
     encoding (the D-TP3 invariant that makes the base a slice, not a second
     evaluation).
  3. For all four form x space combinations, check the epoch-0 composed
     prediction (zero correction) is bitwise the frozen base's decode, and
     independent of the extras; check the packed target caches the base (the
     base network runs once, at encode); check the zero-init surgery zeros
     exactly the correction's final layer.
  4. The loud errors fire: a non-superset parameter set, and the config
     exclusivities (pce / rescale / finetune / model.ia / unknown form /
     the not-yet-implemented refine block).
Every checked value is printed; any failure prints a FAIL line and the run
exits non-zero.

Spec code TPE-A. Home note: transfer-parallel-emulator.md (design rules
D-TP1..D-TP8; this check is the transfer-identity validation gate).
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
from emulator.experiment import validate_transfer
from emulator.geometries_output import DataVectorGeometry
from emulator.geometries_parameter import ParamGeometry, AmplitudeFactorGeometry
from emulator.losses.ia import nla_coeffs
from emulator.losses.transfer import TransferChi2, FORMS, SPACES
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
                   "train_dv": "b.npy", "val_dv": "bv.npy"},
          "train_args": {"nepochs": 1}}


def histories():
  return {"train_losses": [0.1], "val_medians": [0.1], "val_means": [0.1],
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
  recipe = {"cls": "emulator.designs.plain.ResMLP", "name": "resmlp",
            "ia": None, "input_dim": len(names), "output_dim": OUT_DIM,
            "compile_mode": None, "needs_geom": False,
            "kwargs": {"int_dim_res": 32, "n_blocks": 2,
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
  recipe = {"cls": "emulator.designs.ia.TemplateMLP", "name": "resmlp",
            "ia": "nla", "input_dim": len(names), "output_dim": OUT_DIM,
            "compile_mode": None, "needs_geom": False,
            "kwargs": {"n_amps": 1, "n_templates": 3, "int_dim_res": 32,
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
      dv = torch.from_numpy(
        np.random.default_rng(33).standard_normal((64, TOTAL))).float().to(device)
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
    return {"cls": ResMLP, "compile_mode": None, "int_dim_res": 16,
            "n_blocks": 1, "block_opts": block_opts}
  return {"cls": TemplateMLP, "compile_mode": None, "int_dim_res": 16,
          "n_blocks": 1, "block_opts": block_opts, "n_amps": 1,
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
  # rejection when TPE-2 landed refine.)
  raised = False
  try:
    validate_transfer(cfg={"transfer": {"from": "x", "form": "gain",
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
  """(the artifact lifecycle, TPE-1b) save a transfer, rebuild, compose again.

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
  theta = torch.from_numpy(
    np.random.default_rng(73).standard_normal((8, len(names)))).float().to(device)
  with torch.no_grad():
    enc      = new_pgeom.encode(theta)
    composed = chi2fn.decode(corr(enc), enc)          # in-memory (8, n_keep)

  # save the transfer artifact exactly as the driver assembles it.
  corr_recipe = {"cls": "emulator.designs.plain.ResMLP", "name": "resmlp",
                 "ia": None, "input_dim": int(in_dim), "output_dim": OUT_DIM,
                 "compile_mode": None, "needs_geom": False,
                 "kwargs": {"int_dim_res": 16, "n_blocks": 1,
                            "block_opts": {"act": {"type": "H", "n_gates": 3},
                                           "norm": "affine"}}}
  transfer_base = {"recipe": base.recipe, "state": base.model.state_dict(),
                   "param_geometry": base.pgeom, "dv_geometry": base.geom,
                   "form": "gain", "space": "physical"}
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
  # kernel-reassociation caveat pre-authorized in the FTW audit). Bitwise is
  # only demanded where the path is literally the same computation (the
  # save -> rebuild -> composed-predict leg above).
  report("lifecycle: EmulatorPredictor.predict == in-memory unsqueeze (1e-6)",
         bool(np.abs(got - want).max() <= 1.0e-6),
         "max|d| = " + format(float(np.abs(got - want).max()), ".2e"))

  # chaining refused: the saved transfer cannot be loaded as a new base (D-TP2).
  raised = False
  try:
    warmstart.load_source(root=str(saved), device=device, allow_factored=True)
  except ValueError:
    raised = True
  report("lifecycle: chaining refused (a transfer cannot be a base)", raised,
         "load_source rejects the embedded transfer_base group")


def check_refined_lifecycle(device, tmp):
  """(refine artifact, TPE-2) a refined transfer saves the drifted base to a
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
  theta = torch.from_numpy(
    np.random.default_rng(82).standard_normal((8, len(names)))).float().to(device)
  with torch.no_grad():
    enc      = new_pgeom.encode(theta)
    composed = chi2fn.decode(corr(enc), enc)          # in-memory, drifted base

  corr_recipe = {"cls": "emulator.designs.plain.ResMLP", "name": "resmlp",
                 "ia": None, "input_dim": int(in_dim), "output_dim": OUT_DIM,
                 "compile_mode": None, "needs_geom": False,
                 "kwargs": {"int_dim_res": 16, "n_blocks": 1,
                            "block_opts": {"act": {"type": "H", "n_gates": 3},
                                           "norm": "affine"}}}
  transfer_base = {"recipe": base.recipe, "state": pretrained,
                   "drifted_state": drifted, "param_geometry": base.pgeom,
                   "dv_geometry": base.geom, "form": "gain", "space": "physical"}
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


def main():
  """Build synthetic plain + factored bases and run the transfer-identity checks.

  In order: save and load a plain base and a factored base; for each, check the
  base encoding is a bitwise column slice of the run's encoding, and, for all
  four form x space combinations, the epoch-0 identity, the base caching, and
  the packed target width; then the zero-init surgery and the loud config
  errors. Each check prints a PASS/FAIL line; main returns 1 if any failed.
  """
  print("== transfer-identity (spec code TPE-A) ==")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("device " + str(device) + " (torch only, no cosmolike)")
  torch.manual_seed(0)

  tmp = tempfile.mkdtemp(prefix="tpe-")
  plain_root = Path(tmp) / "plain_base"
  factored_root = Path(tmp) / "factored_base"
  save_plain_base(plain_root, device)
  save_factored_base(factored_root, device)

  check_base("plain", plain_root, device, tmp, factored=False)
  check_base("factored", factored_root, device, tmp, factored=True)
  check_zero_init(device)
  check_errors(device, plain_root)
  check_lifecycle(device, tmp)
  check_refined_lifecycle(device, tmp)

  print("")
  if len(FAILURES) == 0:
    print("transfer-identity: ALL PASS")
    return 0
  print("transfer-identity: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
