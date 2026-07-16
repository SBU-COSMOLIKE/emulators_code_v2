"""Focused tests for fine-tune anchoring and saved provenance.

The saved weight average must observe the post-anchor parameters. Both public
training drivers must also use one provenance assembler so scalar artifacts
cannot omit the source recorded by every other family.
"""

import ast
from pathlib import Path
import types
import unittest

import torch

from emulator import warmstart
from emulator.training import Anchor


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parsed_file(relative_path):
  path = _REPOSITORY_ROOT / relative_path
  source = path.read_text(encoding="utf-8")
  return source, ast.parse(source, filename=str(path))


def _call_lines(function, dotted_name):
  """Return source lines for calls with one exact dotted name."""
  lines = []
  for node in ast.walk(function):
    if not isinstance(node, ast.Call):
      continue

    parts = []
    current = node.func
    while isinstance(current, ast.Attribute):
      parts.append(current.attr)
      current = current.value
    if isinstance(current, ast.Name):
      parts.append(current.id)
    name = ".".join(reversed(parts))
    if name == dotted_name:
      lines.append(node.lineno)
  return sorted(lines)


class FinetunePostStepAndProvenanceTests(unittest.TestCase):

  def test_anchor_precedes_weight_average_in_training_loop(self):
    _, tree = _parsed_file("emulator/training.py")
    function = None
    for node in tree.body:
      if isinstance(node, ast.FunctionDef):
        if node.name == "training_loop_batched":
          function = node
          break
    self.assertIsNotNone(function)

    optimizer_lines = _call_lines(function, "optimizer.step")
    anchor_lines = _call_lines(function, "anchor.apply")
    average_lines = _call_lines(function, "torch._foreach_lerp_")
    self.assertEqual(len(optimizer_lines), 1)
    self.assertEqual(len(anchor_lines), 1)
    self.assertEqual(len(average_lines), 1)
    self.assertLess(optimizer_lines[0], anchor_lines[0])
    self.assertLess(anchor_lines[0], average_lines[0])

  def test_beta_zero_average_copies_anchored_weight(self):
    parameter = torch.nn.Parameter(torch.tensor([4.0]))
    reference = torch.tensor([0.0])
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    anchor = Anchor(
      entries=[(parameter, reference, None, 0)],
      lam=1.0)
    average = [torch.tensor([9.0])]

    anchor.apply(optimizer)
    with torch.no_grad():
      torch._foreach_lerp_(average, [parameter], 1.0)

    self.assertEqual(parameter.item(), 3.0)
    self.assertEqual(average[0].item(), 3.0)

  def test_shared_provenance_assembler(self):
    source = types.SimpleNamespace(root="/saved/source")
    attrs = warmstart.finetune_provenance_attrs(
      source=source,
      extra_names=["w0", "wa"])
    self.assertEqual(
      attrs,
      {
        "finetuned_from": "/saved/source",
        "finetune_extra_names": "w0 wa",
      })
    self.assertEqual(
      warmstart.finetune_provenance_attrs(
        source=None,
        extra_names=None),
      {})

  def test_both_training_drivers_call_shared_assembler(self):
    for relative_path in (
        "cosmic_shear_train_emulator.py",
        "scalar_train_emulator.py"):
      with self.subTest(driver=relative_path):
        source, tree = _parsed_file(relative_path)
        self.assertIn(
          "from emulator.warmstart import finetune_provenance_attrs",
          source)
        calls = []
        for node in ast.walk(tree):
          if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
              if node.func.id == "finetune_provenance_attrs":
                calls.append(node)
        self.assertEqual(len(calls), 1)
        keyword_names = {keyword.arg for keyword in calls[0].keywords}
        self.assertEqual(keyword_names, {"source", "extra_names"})


if __name__ == "__main__":
  unittest.main()
