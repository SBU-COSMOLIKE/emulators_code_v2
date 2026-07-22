"""CPU checks for the drift gate's child-process default isolation.

The save-rebuild-drift gate proves that rebuilding a saved emulator reads
the activation gate count from the file, not from the current source
default of ``make_activation``.  The gate observes the changed default
through files: it copies the emulator package into a temporary folder,
changes only that default line, and rebuilds the save in a child process
whose PYTHONPATH names the copy first.  No running process ever has its
behavior replaced.

The complete gate needs the configured workstation data, so these tests
exercise the same helpers on a tiny synthetic gated-power artifact: the
copy changes exactly one line, the child's rebuilt output is bit-for-bit
the parent's, and a child launched with an UNMODIFIED copy refuses with
its dedicated exit code — the control proving the harness cannot pass
while observing the ordinary default.
"""

from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from ai.gates.checks import gsv_bitwise_drift
from ai.gates.checks.artifact_fixtures import one_pass_training_recipe
from emulator import fixed_facts
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry
from emulator.model_recipe import set_runtime_compile_mode
from emulator.results import rebuild_emulator, save_emulator


def _gated_power_recipe():
  """The rebuild recipe for the tiny gated-power fixture below."""
  return {
    "cls": "emulator.designs.plain.ResMLP",
    "name": "resmlp",
    "ia": None,
    "input_dim": 2,
    "output_dim": 1,
    "compile_mode": None,
    "needs_geom": False,
    "kwargs": {
      "int_dim_res": 4,
      "n_blocks": 1,
      "block_opts": {
        "n_layers": 2,
        "act": {"type": "gated_power", "n_gates": 3},
        "norm": "affine",
      },
    },
  }


def _save_gated_power_fixture(path_root):
  """Write one tiny schema-v3 artifact whose activation is gated_power.

  The gated-power family is the one whose rebuilt parameter count
  depends on the recorded gate count, so a rebuild that trusted the
  source default instead of the file would fail its strict weight load
  in the drift child.

  Arguments:
    path_root = the artifact pair's path root (files <root>.h5 and
                <root>.emul are created).
  """
  cpu = torch.device("cpu")
  pgeom = ParamGeometry(
    device=cpu,
    names=["p0", "p1"],
    center=[0.0, 0.0],
    evecs=[[1.0, 0.0], [0.0, 1.0]],
    sqrt_ev=[1.0, 1.0])
  geom = ScalarGeometry(
    device=cpu,
    names=["derived"],
    center=[0.0],
    scale=[1.0])
  block_opts = {
    "act": make_activation("gated_power", n_gates=3),
    "norm": make_norm("affine"),
  }
  torch.manual_seed(41)
  model = ResMLP(
    input_dim=2,
    output_dim=1,
    int_dim_res=4,
    n_blocks=1,
    block_opts=block_opts).to(cpu)
  set_runtime_compile_mode(model, None)
  config = {
    "data": {},
    "train_args": {"nepochs": 1},
  }
  histories = {
    "train_losses": [0.1],
    "val_medians": [0.1],
    "val_means": [0.1],
    "val_fracs": [torch.tensor([0.5])],
    "thresholds": torch.tensor([1.0]),
  }
  save_emulator(
    path_root=str(path_root),
    model=model,
    param_geometry=pgeom,
    geometry=geom,
    config=config,
    histories=histories,
    train_args=config["train_args"],
    resolved_train=one_pass_training_recipe(
      thresholds=(1.0,), compile_mode=None),
    resolved_model=_gated_power_recipe(),
    composition_mode="plain",
    transfer_refined=False,
    resolved_pce=None,
    resolved_transfer=None,
    resolved_rescale="none",
    facts_yaml=fixed_facts.synthetic_sidecar(
      names=pgeom.state()["names"],
      label="drift-child-isolation",
      family="scalar",
      support=None),
    attrs={"rescale": "none"})


class DriftGateChildIsolationTests(unittest.TestCase):
  """Check the copy substitution, the child proof, and its control."""

  def test_modified_copy_changes_only_the_default_line(self):
    """The copied package differs from the source in exactly one line."""
    with tempfile.TemporaryDirectory(prefix="drift-copy-") as tmp:
      copy_root = gsv_bitwise_drift.prepare_drift_source_copy(
        Path(tmp) / "modified")
      changed_path = copy_root / "emulator" / "activations.py"
      changed_text = changed_path.read_text(encoding="utf-8")
      self.assertEqual(
        changed_text.count(gsv_bitwise_drift._DRIFT_MODIFIED_LINE), 1)
      self.assertNotIn(
        gsv_bitwise_drift._DRIFT_DEFAULT_LINE, changed_text)

      source_package = gsv_bitwise_drift.repo_root() / "emulator"
      for copied_file in sorted((copy_root / "emulator").rglob("*.py")):
        relative = copied_file.relative_to(copy_root / "emulator")
        source_file = source_package / relative
        with self.subTest(file=str(relative)):
          if relative == Path("activations.py"):
            continue
          self.assertEqual(copied_file.read_bytes(),
                           source_file.read_bytes())

  def test_child_rebuild_matches_parent_bitwise(self):
    """The child, seeing default 7, rebuilds the n_gates=3 save exactly."""
    with tempfile.TemporaryDirectory(prefix="drift-child-") as tmp:
      save_root = Path(tmp) / "gated_power_fixture"
      _save_gated_power_fixture(save_root)

      torch.manual_seed(7)
      probe = torch.randn(4, 2)
      cpu = torch.device("cpu")
      parent_out = gsv_bitwise_drift.rebuilt_out(
        save_root=save_root, device=cpu, probe=probe)

      modified_root = gsv_bitwise_drift.prepare_drift_source_copy(
        Path(tmp) / "modified")
      child_out, detail = gsv_bitwise_drift.run_drift_child(
        save_root=save_root,
        device=cpu,
        probe=probe,
        work_dir=tmp,
        modified_root=modified_root)
      self.assertIsNotNone(child_out, detail)
      self.assertTrue(torch.equal(parent_out.detach().cpu(), child_out),
                      "child rebuild is not bitwise-equal: " + detail)

  def test_child_refuses_an_unmodified_copy(self):
    """A child that imported the ordinary default cannot pass as a proof."""
    with tempfile.TemporaryDirectory(prefix="drift-control-") as tmp:
      save_root = Path(tmp) / "gated_power_fixture"
      _save_gated_power_fixture(save_root)

      unmodified_root = Path(tmp) / "unmodified"
      shutil.copytree(
        gsv_bitwise_drift.repo_root() / "emulator",
        unmodified_root / "emulator",
        ignore=shutil.ignore_patterns("__pycache__"))

      torch.manual_seed(7)
      probe = torch.randn(4, 2)
      child_out, detail = gsv_bitwise_drift.run_drift_child(
        save_root=save_root,
        device=torch.device("cpu"),
        probe=probe,
        work_dir=tmp,
        modified_root=unmodified_root)
      self.assertIsNone(child_out)
      self.assertIn("exit 3", detail)
      self.assertIn("not the modified copy", detail)

  def test_fixture_rebuilds_in_this_process(self):
    """The synthetic gated-power pair is a valid current-schema artifact."""
    with tempfile.TemporaryDirectory(prefix="drift-fixture-") as tmp:
      save_root = Path(tmp) / "gated_power_fixture"
      _save_gated_power_fixture(save_root)
      model, pgeom, geom, info = rebuild_emulator(
        path_root=str(save_root),
        device=torch.device("cpu"),
        compile_model=False)
      torch.manual_seed(7)
      probe = torch.randn(4, 2)
      with torch.no_grad():
        output = model(pgeom.encode(probe))
      self.assertEqual(tuple(output.shape), (4, 1))
      self.assertTrue(bool(torch.isfinite(output).all()))


if __name__ == "__main__":
  unittest.main()
