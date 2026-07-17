"""Protect transfer-state and target-transform facts at artifact boundaries.

A transfer artifact embeds a pretrained model.  Frozen transfer must embed
the exact live pretrained weights; refined transfer must embed the exact live
refined weights as its drifted state.  Both saved mappings must remain
nonempty and structurally compatible.  Schema 3 also publishes only raw
(``rescale='none'``) targets because it cannot invert the other training
transforms during prediction.
"""

import importlib
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


def _changed_state(model):
  """Return a structurally valid state that differs in one tensor value."""
  state = _cloned_state(model)
  name = next(iter(state))
  changed = state[name].clone()
  changed.reshape(-1)[0] += 1
  state[name] = changed
  return state


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
  """Refuse ambiguous transfer tensors and non-invertible save modes."""

  def test_frozen_save_refuses_empty_state_before_staging(self):
    base_model = _tiny_model()
    base = _transfer_base(base_model, {})
    with tempfile.TemporaryDirectory(prefix="transfer-empty-save-") as temp:
      with mock.patch.object(
          results, "_new_staging_path",
          side_effect=AssertionError("staging must not begin")) as staging:
        with self.assertRaisesRegex(ValueError, "nonempty tensor-state"):
          _save(
            Path(temp) / "artifact", transfer_base=base,
            attrs={"rescale": "none"})
      staging.assert_not_called()

  def test_frozen_save_binds_embedded_state_to_live_model_before_staging(self):
    base_model = _tiny_model()
    base = _transfer_base(base_model, _changed_state(base_model))
    with tempfile.TemporaryDirectory(prefix="transfer-live-save-") as temp:
      with mock.patch.object(
          results, "_new_staging_path",
          side_effect=AssertionError("staging must not begin")) as staging:
        with self.assertRaisesRegex(
            ValueError, "does not equal the live frozen transfer model"):
          _save(
            Path(temp) / "artifact", transfer_base=base,
            attrs={"rescale": "none"})
      staging.assert_not_called()

  def test_refined_live_binding_uses_drifted_not_pretrained_values(self):
    base_model = _tiny_model()
    pretrained = _changed_state(base_model)
    drifted = _cloned_state(base_model)
    base = _transfer_base(
      base_model, pretrained, drifted_state=drifted)
    results._validate_live_transfer_state(
      base,
      transfer_refined=True,
      state_sha256=results.digest_tensor_state(pretrained),
      drifted_sha256=results.digest_tensor_state(drifted),
      where="refined fixture")

    base["drifted_state"] = _changed_state(base_model)
    with self.assertRaisesRegex(
        ValueError, "does not equal the live refined transfer model"):
      results._validate_live_transfer_state(
        base,
        transfer_refined=True,
        state_sha256=results.digest_tensor_state(pretrained),
        drifted_sha256=results.digest_tensor_state(base["drifted_state"]),
        where="refined fixture")

  def test_refined_states_must_have_matching_keys_shapes_and_dtypes(self):
    base_model = _tiny_model()
    pretrained = _cloned_state(base_model)
    drifted = _cloned_state(base_model)
    removed = next(iter(drifted))
    del drifted[removed]
    base = _transfer_base(
      base_model, pretrained, drifted_state=drifted)
    with self.assertRaisesRegex(ValueError, "different tensor names"):
      results._bind_transfer_state_digests(
        {}, {}, base, transfer_refined=True, where="refined fixture")

    drifted = _cloned_state(base_model)
    name = next(iter(drifted))
    drifted[name] = drifted[name].to(torch.float64)
    base["drifted_state"] = drifted
    with self.assertRaisesRegex(ValueError, "shape/dtype"):
      results._bind_transfer_state_digests(
        {}, {}, base, transfer_refined=True, where="refined fixture")

  def test_hdf5_transfer_states_are_nonempty_and_structurally_equal(self):
    with tempfile.TemporaryDirectory(prefix="transfer-inert-structure-") as temp:
      path = Path(temp) / "states.h5"
      with h5py.File(path, "w") as artifact:
        transfer = artifact.create_group("transfer")
        state = transfer.create_group("state")
        drifted = transfer.create_group("drifted_state")
        with self.assertRaisesRegex(ValueError, "at least one tensor"):
          results._validate_embedded_transfer_state(
            transfer, {}, transfer_refined=False, where="empty fixture")

        state.create_dataset(
          "weight", data=np.asarray([1.0, 2.0], dtype=np.float32))
        drifted.create_dataset(
          "weight", data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        state_digest = results._h5_state_digest(
          state, where="structure fixture.state")
        drifted_digest = results._h5_state_digest(
          drifted, where="structure fixture.drifted_state")
        transfer.attrs["state_sha256"] = state_digest
        transfer.attrs["drifted_state_sha256"] = drifted_digest
        resolved = {
          "embedded_state_sha256": state_digest,
          "embedded_drifted_state_sha256": drifted_digest,
        }
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
          results._validate_embedded_transfer_state(
            transfer, resolved, transfer_refined=True,
            where="structure fixture")

  def test_rebuild_refuses_empty_transfer_state_before_execution(self):
    base_model = _tiny_model()
    base = _transfer_base(base_model, _cloned_state(base_model))
    with tempfile.TemporaryDirectory(prefix="transfer-empty-read-") as temp:
      root = Path(temp) / "artifact"
      _save(root, transfer_base=base, attrs={"rescale": "none"})
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        state = artifact["transfer_base/state"]
        for name in tuple(state):
          del state[name]
      with mock.patch.object(
          importlib, "import_module",
          side_effect=AssertionError("dynamic import must not run")) as imports, \
          mock.patch.object(
            results, "_load_tensor_state_dict",
            side_effect=AssertionError("checkpoint load must not run")) as load:
        with self.assertRaisesRegex(ValueError, "at least one tensor"):
          results.rebuild_emulator(
            str(root), torch.device("cpu"), compile_model=False)
      imports.assert_not_called()
      load.assert_not_called()

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
