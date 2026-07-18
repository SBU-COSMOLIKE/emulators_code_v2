"""Check simple transfer rebuilding and the public target scale."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from ai.gates.checks.artifact_fixtures import one_pass_training_recipe
from ai.tests.test_artifact_recipe_preflight import _geometries
from ai.tests.test_artifact_recipe_preflight import _histories
from ai.tests.test_artifact_recipe_preflight import _scalar_recipe
from ai.tests.test_artifact_recipe_preflight import _tiny_model
from emulator import fixed_facts, results


def _cloned_state(model):
  """Return detached tensor copies of one model's complete state."""
  return {name: value.detach().clone()
          for name, value in model.state_dict().items()}


def _transfer_base(base_model, state, *, drifted_state=None):
  """Assemble the embedded-base fields consumed by the artifact writer."""
  param_geometry, geometry = _geometries()
  payload = {
    "recipe": _scalar_recipe(),
    "model": base_model,
    "state": state,
    "param_geometry": param_geometry,
    "dv_geometry": geometry,
    "form": "gain",
    "space": "physical",
  }
  if drifted_state is not None:
    payload["drifted_state"] = drifted_state
  return payload


def _save(
    root, *, transfer_base=None, attrs=None, resolved_rescale=None,
    transfer_refined=False):
  """Save one tiny plain or transfer artifact through the public writer."""
  param_geometry, geometry = _geometries()
  transfer = transfer_base is not None
  resolved_transfer = None
  if transfer:
    resolved_transfer = {
      "form": "gain",
      "space": "physical",
      "refine": ({"fixture": "state-contract"}
                 if transfer_refined else None),
      "source_artifact_id": "1" * 32,
      "source_checkpoint_sha256": "2" * 64,
    }
  return results.save_emulator(
    path_root=str(root),
    model=_tiny_model(),
    param_geometry=param_geometry,
    geometry=geometry,
    config={"data": {}, "train_args": {"nepochs": 1}},
    histories=_histories(),
    resolved_train=one_pass_training_recipe(thresholds=(1.0,)),
    resolved_model=_scalar_recipe(),
    transfer_base=transfer_base,
    facts_yaml=fixed_facts.synthetic_sidecar(
      names=["p0"], label="artifact-transfer-state-contract",
      family="scalar", support=None),
    composition_mode="transfer" if transfer else "plain",
    transfer_refined=transfer_refined,
    resolved_pce=None,
    resolved_transfer=resolved_transfer,
    attrs=attrs,
    resolved_rescale=resolved_rescale)


class ArtifactTransferStateContractTests(unittest.TestCase):
  """Keep transfer state handling small and tensor loading strict."""

  def test_transfer_save_uses_no_duplicate_state_digests(self):
    base_model = _tiny_model()
    base = _transfer_base(base_model, _cloned_state(base_model))
    with tempfile.TemporaryDirectory(prefix="transfer-simple-save-") as temp:
      root = Path(temp) / "artifact"
      _save(root, transfer_base=base, attrs={"rescale": "none"})
      with h5py.File(str(root) + ".h5", "r") as artifact:
        transfer = artifact["transfer_base"]
        self.assertNotIn("state_sha256", transfer.attrs)
        self.assertNotIn("drifted_state_sha256", transfer.attrs)
        resolved = artifact["config_resolved_yaml"][()]
      if isinstance(resolved, bytes):
        resolved = resolved.decode("utf-8")
      self.assertNotIn("embedded_state_sha256", resolved)
      self.assertNotIn("embedded_drifted_state_sha256", resolved)
      results.rebuild_emulator(
        str(root), torch.device("cpu"), compile_model=False)

  def test_strict_loading_refuses_damaged_transfer_state(self):
    def remove_one(state):
      del state[next(iter(state))]

    def add_one(state):
      state.create_dataset("unexpected", data=np.zeros(1, dtype=np.float32))

    def change_shape(state):
      name = next(iter(state))
      dtype = state[name].dtype
      size = int(np.prod(state[name].shape)) + 1
      del state[name]
      state.create_dataset(name, data=np.zeros(size, dtype=dtype))

    cases = (
      ("missing", remove_one, "Missing key"),
      ("unexpected", add_one, "Unexpected key"),
      ("wrong-shape", change_shape, "size mismatch"),
    )
    for label, damage, message in cases:
      with self.subTest(label=label), tempfile.TemporaryDirectory(
          prefix="transfer-strict-load-") as temp:
        base_model = _tiny_model()
        base = _transfer_base(base_model, _cloned_state(base_model))
        root = Path(temp) / "artifact"
        _save(root, transfer_base=base, attrs={"rescale": "none"})
        with h5py.File(str(root) + ".h5", "r+") as artifact:
          damage(artifact["transfer_base/state"])
        with self.assertRaisesRegex(RuntimeError, message):
          results.rebuild_emulator(
            str(root), torch.device("cpu"), compile_model=False)

  def test_schema3_refuses_non_none_rescale_without_an_identity(self):
    with tempfile.TemporaryDirectory(prefix="artifact-rescale-save-") as temp:
      root = Path(temp) / "artifact"
      with mock.patch.object(
          results, "_new_staging_path",
          side_effect=AssertionError("staging must not begin")) as staging:
        with self.assertRaisesRegex(ValueError, "can publish only.*none"):
          _save(root, attrs={"rescale": "residual"})
      staging.assert_not_called()

  def test_schema3_materializes_missing_resolved_mode_from_explicit_attr(self):
    with tempfile.TemporaryDirectory(prefix="artifact-rescale-control-") as temp:
      root = Path(temp) / "artifact"
      _save(root, attrs={"rescale": "none"}, resolved_rescale=None)
      with h5py.File(str(root) + ".h5", "r") as artifact:
        self.assertEqual(artifact.attrs["rescale"], "none")
        identity = results._read_output_identity(
          artifact, where=str(root) + ".h5")[1]
      self.assertEqual(identity["loss_recipe"]["rescale"], "none")

  def test_schema3_refuses_missing_or_disagreeing_rescale_before_staging(self):
    cases = (
      (None, None, "requires attrs"),
      ({"rescale": "none"}, "residual", "disagrees"),
      ({"rescale": b"none"}, None, "explicit native string"),
    )
    with tempfile.TemporaryDirectory(prefix="artifact-rescale-bad-") as temp:
      for index, (attrs, resolved, message) in enumerate(cases):
        with self.subTest(attrs=attrs, resolved=resolved), mock.patch.object(
            results, "_new_staging_path",
            side_effect=AssertionError("staging must not begin")) as staging:
          with self.assertRaisesRegex(ValueError, message):
            _save(
              Path(temp) / ("artifact-" + str(index)), attrs=attrs,
              resolved_rescale=resolved)
          staging.assert_not_called()


if __name__ == "__main__":
  unittest.main()
