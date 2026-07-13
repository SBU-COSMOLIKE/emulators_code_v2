#!/usr/bin/env python3
"""weight-decay census: weight decay reaches only true weight matrices.

Weight decay is the optimizer's steady pull of parameters toward zero.
Applied to the wrong parameters it quietly deforms the model: shrinking an
activation's shape parameters or a bias changes what the network can
represent. So the rule inside make_optimizer must choose parameters by
their module's role, never by tensor shape. This check proves it does.

How it works: build a toy module tree holding one of every family the rule
must sort (nn.Linear, nn.Conv1d, BinLinear, Affine, FeatureAffine, a
gated_power activation whose shape parameters are (K, dim) matrices, and
LayerNorm); run the real make_optimizer with weight decay 1e-4; then check
that the decayed group is exactly the .weight of nn.Linear, nn.Conv1d, and
BinLinear, and that everything else is left alone. That everything-else
includes the two cases a shape-based rule gets wrong: the (K, dim) shape
parameters of the gated_power activation, and the (G, out) bias of
BinLinear, both of which look like weight matrices by shape but are not.
Finally, with weight decay 0, both groups are inert, so the regrouping
cannot change any shipping run. The full census is printed; any
misclassification exits non-zero.

Terms: decayed = the parameter group the optimizer applies weight decay to;
undecayed = the group left at weight decay 0. Membership is decided by
module role (the allowlist), not by tensor shape, so a many-parameter
activation is never decayed.

Home note: training-stack.md:143-147.
"""

import sys

import torch
import torch.nn as nn
import torch.optim as optim

from emulator.activations import make_activation
from emulator.designs.blocks import (Affine,
                                     BinLinear,
                                     FeatureAffine)
from emulator.training import make_optimizer

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


class ToyTree(nn.Module):
  """One module of every family the allowlist must classify.

  Carries the three weight-matrix modules (Linear / Conv1d / BinLinear)
  and the families that must stay undecayed (Affine, FeatureAffine, a
  gated_power activation with (K, dim) parameters, LayerNorm), so the
  census exercises every branch of make_optimizer's split.
  """

  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(in_features=4, out_features=4)
    self.conv = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3)
    self.bin = BinLinear(n_tokens=2, in_features=4, out_features=4)
    self.aff = Affine()
    self.faff = FeatureAffine(size=4)
    self.act = make_activation("gated_power", n_gates=3)(4)
    self.ln = nn.LayerNorm(normalized_shape=4)


def main():
  """Build the toy tree, run make_optimizer, and check the decay census.

  In order: build ToyTree; run make_optimizer at weight decay 1e-4 and read
  back which parameters landed in the decayed group; check that group is
  exactly the three weight matrices, that the gated_power (K, dim)
  parameters and the BinLinear bias stayed undecayed, that nothing else
  leaked in, and that every parameter is in exactly one group; then rerun at
  weight decay 0 and check every group is inert. Each check prints a
  PASS/FAIL line; main returns 1 if any failed, else 0.
  """
  print("== weight-decay-census ==")
  torch.manual_seed(0)
  device = torch.device("cpu")
  model = ToyTree()

  # id -> name, so the census reports readable names.
  name_of = {}
  for name, p in model.named_parameters():
    name_of[id(p)] = name

  # the three weight matrices the allowlist must decay.
  expected_decay = set()
  expected_decay.add(id(model.lin.weight))
  expected_decay.add(id(model.conv.weight))
  expected_decay.add(id(model.bin.weight))

  opt = make_optimizer(model=model,
                       opt_opts={"cls": optim.AdamW, "weight_decay": 1.0e-4},
                       lr=1.0e-3,
                       device=device)

  # the decayed group is the one whose weight_decay is nonzero.
  decay_ids = set()
  no_decay_ids = set()
  for group in opt.param_groups:
    target = decay_ids if group["weight_decay"] > 0.0 else no_decay_ids
    for p in group["params"]:
      target.add(id(p))

  print("decayed group:")
  for pid in sorted(decay_ids, key=lambda i: name_of.get(i, "?")):
    print("    " + name_of.get(pid, "?"))
  print("undecayed group:")
  for pid in sorted(no_decay_ids, key=lambda i: name_of.get(i, "?")):
    print("    " + name_of.get(pid, "?"))

  report("decayed group == exactly the Linear/Conv1d/BinLinear weights",
         decay_ids == expected_decay,
         "got " + str(len(decay_ids)) + " expected 3")

  # the two families the old ndim >= 2 rule wrongly decayed.
  multigate_ids = []
  multigate_ids.append(id(model.act.w))
  multigate_ids.append(id(model.act.beta))
  multigate_ids.append(id(model.act.mu))
  mg_undecayed = True
  for pid in multigate_ids:
    if pid not in no_decay_ids:
      mg_undecayed = False
  report("gated_power (K, dim) w / beta / mu are UNDECAYED",
         mg_undecayed,
         "shapes " + str(tuple(model.act.w.shape)))
  report("BinLinear (G, out) bias is UNDECAYED",
         id(model.bin.bias) in no_decay_ids,
         "shape " + str(tuple(model.bin.bias.shape)))

  # every non-weight-matrix parameter is undecayed (no leaks).
  all_ids = set()
  for _, p in model.named_parameters():
    all_ids.add(id(p))
  leaked = (all_ids - expected_decay) & decay_ids
  leaked_names = []
  for pid in leaked:
    leaked_names.append(name_of.get(pid, "?"))
  report("no non-weight-matrix parameter leaked into the decayed group",
         len(leaked) == 0,
         "leaked " + str(leaked_names))
  report("every parameter is in exactly one group",
         decay_ids.isdisjoint(no_decay_ids)
         and (decay_ids | no_decay_ids) == all_ids,
         "total " + str(len(all_ids)))

  # wd 0: both groups inert, so the regrouping changes no shipping run.
  opt0 = make_optimizer(model=model,
                        opt_opts={"cls": optim.AdamW, "weight_decay": 0.0},
                        lr=1.0e-3,
                        device=device)
  wd0 = []
  all_zero = True
  for group in opt0.param_groups:
    wd0.append(group["weight_decay"])
    if group["weight_decay"] != 0.0:
      all_zero = False
  report("wd 0: every group has weight_decay 0 (the regrouping is inert)",
         all_zero,
         "group decays " + repr(wd0))

  print("")
  if len(FAILURES) == 0:
    print("weight-decay-census: ALL PASS")
    return 0
  print("weight-decay-census: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
