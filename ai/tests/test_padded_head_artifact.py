"""CPU checks for padded-head identity across an artifact round trip.

A structured CNN head expands four physical outputs into a two-by-three
rectangle.  Two rectangle positions are storage only.  The saved HDF5 record
therefore carries the physical coordinate map and mask, while the checkpoint
carries matching fixed model buffers.  These tests prove that a valid pair
reopens exactly and that neither half may silently replace the other.

The fixture uses the public ``save_emulator`` and ``rebuild_emulator`` APIs.
It needs no training data, CosmoLike installation, or GPU.
"""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import h5py
import torch

from ai.gates.checks.artifact_fixtures import one_pass_training_recipe

from emulator import fixed_facts, results
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.ia import TemplateResCNN
from emulator.designs.plain import ResCNN
from emulator.geometries.output import DataVectorGeometry
from emulator.geometries.parameter import ParamGeometry


CPU = torch.device("cpu")
_ARTIFACT_ID_ATTR = "artifact_id"
_CHECKPOINT_SHA256_ATTR = "checkpoint_sha256"
_COMMENT_PREFIX = b"emulators_code_v2 artifact_id v1:"


def _checkpoint_path(root):
  """Return the learned-tensor member of one saved fixture pair."""
  return Path(str(root) + ".emul")


def _record_path(root):
  """Return the geometry-and-recipe member of one saved fixture pair."""
  return Path(str(root) + ".h5")


def _sha256(path):
  """Hash the exact checkpoint bytes without asking PyTorch to load them."""
  digest = hashlib.sha256()
  with open(path, "rb") as stream:
    while True:
      block = stream.read(1024 * 1024)
      if not block:
        break
      digest.update(block)
  return digest.hexdigest()


def _output_geometry(*, positions=(0, 2, 4, 5), validity=None):
  """Describe four physical values in a six-position head rectangle."""
  if validity is None:
    validity = [[True, False, True], [False, True, True]]
  return DataVectorGeometry(
    device=CPU,
    total_size=6,
    dest_idx=torch.tensor(positions, dtype=torch.long),
    evecs=torch.eye(4),
    sqrt_ev=torch.ones(4),
    Cinv=torch.eye(6),
    center=torch.zeros(4),
    section_sizes=[6, 0, 0],
    probe="xi",
    bin_sizes=[2, 2],
    head_pad_idx=torch.tensor(positions, dtype=torch.long),
    head_valid_mask=torch.tensor(validity, dtype=torch.bool),
  )


def _model_recipe():
  """Return every constructor choice needed to rebuild the tiny CNN."""
  return {
    "cls": "emulator.designs.plain.ResCNN",
    "name": "rescnn",
    "ia": None,
    "input_dim": 2,
    "output_dim": 4,
    "compile_mode": None,
    "needs_geom": True,
    "kwargs": {
      "int_dim_res": 5,
      "kernel_size": 3,
      "rescale_kernel": False,
      "groups": 1,
      "separable": False,
      "film": False,
      "n_blocks": 1,
      "n_blocks_cnn": 2,
      "gate_init": 0.25,
      "head_act": None,
      "block_opts": {
        "n_layers": 2,
        "act": {"type": "H", "n_gates": 3},
        "norm": "affine",
      },
    },
  }


def _save_fixture(root, *, saved_geometry=None):
  """Save one deterministic structured-head model through the public API."""
  param_geometry = ParamGeometry(
    device=CPU,
    names=["p0", "p1"],
    center=[0.0, 0.0],
    evecs=[[1.0, 0.0], [0.0, 1.0]],
    sqrt_ev=[1.0, 1.0])
  model_geometry = _output_geometry()
  geometry = model_geometry if saved_geometry is None else saved_geometry
  block_opts = {
    "act": make_activation("H", n_gates=3),
    "norm": make_norm("affine"),
  }
  torch.manual_seed(184)
  model = ResCNN(
    input_dim=2,
    output_dim=4,
    int_dim_res=5,
    geom=model_geometry,
    kernel_size=3,
    rescale_kernel=False,
    groups=1,
    separable=False,
    film=False,
    n_blocks=1,
    n_blocks_cnn=2,
    gate_init=0.25,
    head_act=None,
    block_opts=block_opts).to(CPU)
  with torch.no_grad():
    final_convolution = model.convs[-1]
    final_convolution.weight.copy_(torch.linspace(
      -0.12, 0.15, steps=final_convolution.weight.numel()).reshape_as(
        final_convolution.weight))
    final_convolution.bias.copy_(torch.tensor([0.15, -0.10]))
    model.gate.fill_(0.4)
  model.eval()

  config = {"data": {}, "train_args": {"nepochs": 1}}
  histories = {
    "train_losses": [0.1],
    "val_medians": [0.1],
    "val_means": [0.1],
    "val_fracs": [torch.tensor([0.5])],
    "thresholds": torch.tensor([1.0]),
  }
  results.save_emulator(
    path_root=str(root),
    model=model,
    param_geometry=param_geometry,
    geometry=geometry,
    config=config,
    histories=histories,
    train_args=config["train_args"],
    resolved_train=one_pass_training_recipe(thresholds=(1.0,)),
    resolved_model=_model_recipe(),
    composition_mode="plain",
    transfer_refined=False,
    resolved_pce=None,
    resolved_transfer=None,
    facts_yaml=fixed_facts.synthetic_sidecar(
      names=param_geometry.state()["names"],
      label="padded-head-artifact",
      family="cosmolike",
      support=None),
    attrs={"rescale": "none"})
  return model, geometry


def _rebuild(root):
  """Reopen a fixture on the CPU without compiling its model."""
  return results.rebuild_emulator(
    path_root=str(root), device=CPU, compile_model=False)


def _rewrite_checkpoint(root, change_state):
  """Change checkpoint tensors, then honestly rebind its digest and ID.

  The rewritten pair is intentionally corrupt at the scientific-layout
  level, not at the outer file-binding level.  Rebinding makes the test reach
  the structured-head comparison instead of stopping at the earlier SHA-256
  check.
  """
  checkpoint = _checkpoint_path(root)
  with h5py.File(_record_path(root), "r") as record:
    artifact_id = record.attrs[_ARTIFACT_ID_ATTR]
  state = torch.load(checkpoint, map_location=CPU, weights_only=True)
  change_state(state)
  torch.save(state, checkpoint)
  with zipfile.ZipFile(checkpoint, "a") as archive:
    archive.comment = _COMMENT_PREFIX + artifact_id.encode("ascii")
  digest = _sha256(checkpoint)
  with h5py.File(_record_path(root), "r+") as record:
    record.attrs[_CHECKPOINT_SHA256_ATTR] = digest
  return digest


class PaddedHeadArtifactTests(unittest.TestCase):
  """Check valid reconstruction and two layout-tampering refusals."""

  def test_valid_round_trip_preserves_layout_and_forward_result(self):
    """A saved CNN reopens with the same fixed map, mask, and prediction."""
    with tempfile.TemporaryDirectory(prefix="padded-head-artifact-valid-") \
        as temp:
      root = Path(temp) / "saved"
      original, original_geometry = _save_fixture(root)
      inputs = torch.tensor([[0.2, -0.4], [1.0, 0.5]])
      with torch.no_grad():
        expected = original(inputs)
        trunk_only = original.mlp(inputs)

      self.assertGreater(
        int(torch.count_nonzero(expected - trunk_only).item()), 0,
        "fixture must exercise a live CNN correction, not its identity start")

      rebuilt, _, rebuilt_geometry, _ = _rebuild(root)
      with torch.no_grad():
        actual = rebuilt(inputs)

      self.assertIsInstance(rebuilt, ResCNN)
      torch.testing.assert_close(
        rebuilt.pad_idx, original.pad_idx, rtol=0.0, atol=0.0)
      torch.testing.assert_close(
        rebuilt.pad_valid, original.pad_valid, rtol=0.0, atol=0.0)
      torch.testing.assert_close(
        rebuilt_geometry.head_pad_idx,
        original_geometry.head_pad_idx,
        rtol=0.0,
        atol=0.0)
      torch.testing.assert_close(
        rebuilt_geometry.head_valid_mask,
        original_geometry.head_valid_mask,
        rtol=0.0,
        atol=0.0)
      torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

  def test_digest_rebound_checkpoint_missing_pad_valid_is_refused(self):
    """A checkpoint cannot discard the persisted fixed padding mask."""
    with tempfile.TemporaryDirectory(prefix="padded-head-artifact-missing-") \
        as temp:
      root = Path(temp) / "saved"
      _save_fixture(root)

      digest = _rewrite_checkpoint(
        root, lambda state: state.pop("pad_valid"))

      with h5py.File(_record_path(root), "r") as record:
        self.assertEqual(record.attrs[_CHECKPOINT_SHA256_ATTR], digest)
      with self.assertRaisesRegex(KeyError, r"no pad_valid buffer"):
        _rebuild(root)

  def test_digest_rebound_pad_index_disagreement_is_refused(self):
    """Checkpoint coordinates cannot replace the HDF5 physical map."""
    with tempfile.TemporaryDirectory(prefix="padded-head-artifact-map-") \
        as temp:
      root = Path(temp) / "saved"
      _save_fixture(root)

      def replace_map(state):
        state["pad_idx"] = torch.tensor([0, 1, 3, 5], dtype=torch.long)
        state["pad_valid"] = torch.tensor(
          [[[True, True, False], [True, False, True]]], dtype=torch.bool)

      digest = _rewrite_checkpoint(root, replace_map)

      with h5py.File(_record_path(root), "r") as record:
        self.assertEqual(record.attrs[_CHECKPOINT_SHA256_ATTR], digest)
      with mock.patch.object(
          ResCNN,
          "load_state_dict",
          side_effect=AssertionError("strict state loading must not run")) \
          as load_state:
        with self.assertRaisesRegex(
            ValueError, r"structured-head pad_idx disagrees"):
          _rebuild(root)
      load_state.assert_not_called()

  def test_save_refuses_mismatched_geometry_before_creating_any_file(self):
    """A fresh save cannot publish an incompatible structured-head layout."""
    with tempfile.TemporaryDirectory(prefix="padded-head-save-mismatch-") \
        as temp:
      root = Path(temp) / "saved"
      names_before = set(Path(temp).iterdir())
      different_geometry = _output_geometry(
        positions=(0, 1, 3, 5),
        validity=[[True, True, False], [True, False, True]])

      with mock.patch.object(
          results,
          "_new_staging_path",
          side_effect=AssertionError("staging must not begin")) as reserve:
        with self.assertRaisesRegex(
            ValueError, r"structured-head pad_idx disagrees"):
          _save_fixture(root, saved_geometry=different_geometry)

      reserve.assert_not_called()
      self.assertFalse(_checkpoint_path(root).exists())
      self.assertFalse(_record_path(root).exists())
      self.assertEqual(set(Path(temp).iterdir()), names_before)

  def test_save_refuses_missing_model_mask_before_creating_files(self):
    """A structured model without its mask cannot begin publication."""
    with tempfile.TemporaryDirectory(prefix="padded-head-save-missing-") \
        as temp:
      root = Path(temp) / "saved"
      real_state_dict = ResCNN.state_dict

      def state_without_mask(model, *args, **kwargs):
        state = real_state_dict(model, *args, **kwargs)
        state.pop("pad_valid")
        return state

      with mock.patch.object(ResCNN, "state_dict", state_without_mask), \
          mock.patch.object(
            results,
            "_new_staging_path",
            side_effect=AssertionError("staging must not begin")) as reserve:
        with self.assertRaisesRegex(KeyError, r"no pad_valid buffer"):
          _save_fixture(root)

      reserve.assert_not_called()
      self.assertEqual(list(Path(temp).iterdir()), [])

  def test_save_preflight_accepts_template_major_repeated_mask(self):
    """Each template may repeat the same physical mask in channel order."""
    geometry = _output_geometry()
    block_opts = {
      "act": make_activation("H", n_gates=3),
      "norm": make_norm("affine"),
    }
    model = TemplateResCNN(
      input_dim=3,
      output_dim=4,
      n_amps=1,
      n_templates=3,
      int_dim_res=5,
      geom=geometry,
      kernel_size=3,
      n_blocks=0,
      n_blocks_cnn=1,
      block_opts=block_opts)
    state = {
      name: value.detach().cpu()
      for name, value in model.state_dict().items()
    }
    expected_mask = geometry.head_valid_mask.unsqueeze(0).repeat(1, 3, 1)

    torch.testing.assert_close(
      state["pad_valid"], expected_mask, rtol=0.0, atol=0.0)
    results._validate_saved_head_layout(
      model_state=state,
      geometry=geometry,
      recipe={
        "cls": "emulator.designs.ia.TemplateResCNN",
        "needs_geom": True,
        "output_dim": 4,
        "kwargs": {"n_templates": 3},
      },
      where="template-major-save-preflight")


if __name__ == "__main__":
  unittest.main()
