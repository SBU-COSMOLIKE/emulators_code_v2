"""Schema-v3 witness for rebuild-time input-name validation.

The fixed-facts blocks and their copied producer text protect one another, but
an editor can rewrite both copies together.  The rebuilt parameter geometry is
the independent source of the whitening-column order.  These tests prove that
rebuild compares that geometry against the record before loading model weights,
and that an untampered matching record still reaches the model-loading stage.
"""

import hashlib
import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch
import yaml

from emulator import fixed_facts
from emulator import results


class _GeometryFixture:
  """Minimal importable geometry for exercising the artifact reader."""

  def __init__(self, names):
    self.names = list(names)

  @classmethod
  def from_state(cls, device, state):
    """Rebuild the persisted names through results._rebuild_geometry."""
    del device
    return cls(names=state["names"])


class _ModelLoadReached(BaseException):
  """Sentinel proving rebuild advanced past all artifact validation."""


class RebuildFixedFactsNamesTest(unittest.TestCase):
  """Exercise the independent geometry-to-record comparison on schema v3."""

  geometry_names = ["alpha", "beta"]

  def _write_geometry(self, parent, group_name, names):
    """Write the minimal state consumed by results._rebuild_geometry."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group = parent.create_group(group_name)
    group.attrs["cls"] = __name__ + "._GeometryFixture"
    group.create_dataset(
      "names",
      data=np.asarray(names, dtype=object),
      dtype=string_dtype)

  def _write_artifact(self, path_root, rewritten_names=None):
    """Write a schema-v3 artifact and optionally rewrite both record copies."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    artifact_id = "1" * 32
    emul_path = path_root + ".emul"
    torch.save({"weight": torch.ones(1)}, emul_path)
    results._stamp_checkpoint_artifact_id(emul_path, artifact_id)
    with open(emul_path, "rb") as checkpoint:
      checkpoint_sha256 = hashlib.sha256(checkpoint.read()).hexdigest()
    h5_path = path_root + ".h5"
    with h5py.File(h5_path, "w") as artifact:
      artifact.attrs["schema_version"] = fixed_facts.SCHEMA_VERSION
      artifact.attrs["composition_mode"] = "plain"
      artifact.attrs["transfer_refined"] = False
      artifact.attrs["artifact_id"] = artifact_id
      artifact.attrs["checkpoint_sha256"] = checkpoint_sha256
      artifact.create_dataset(
        "model_recipe",
        data="{}",
        dtype=string_dtype)
      artifact.create_dataset(
        "config_yaml",
        data=yaml.safe_dump({"data": {}, "train_args": {},
                             "pce": None, "transfer": None},
                            sort_keys=False),
        dtype=string_dtype)
      artifact.create_dataset(
        "config_resolved_yaml",
        data=yaml.safe_dump({"data": {}, "train_args": {},
                             "composition_mode": "plain",
                             "transfer_refined": False,
                             "pce": None,
                             "transfer": None},
                            sort_keys=False),
        dtype=string_dtype)
      self._write_geometry(
        parent=artifact,
        group_name="param_geometry",
        names=self.geometry_names)
      self._write_geometry(
        parent=artifact,
        group_name="dv_geometry",
        names=["observable"])

      original = fixed_facts.synthetic_sidecar(
        names=self.geometry_names,
        label="rebuild-name-witness-original")
      fixed_facts.write_h5(f=artifact, sidecar_text=original)

      if rewritten_names is not None:
        # Coordinate the rewrite: the structured blocks and the producer text
        # still agree with one another, while the persisted geometry is left
        # untouched.  read_h5 therefore accepts this final state; the rebuild
        # comparison below is the only independent witness that can refuse it.
        for key in (fixed_facts.FIXED_FACTS_GROUP,
                    fixed_facts.INPUT_DOMAIN_GROUP,
                    fixed_facts.SIDECAR_DATASET):
          del artifact[key]
        rewritten = fixed_facts.synthetic_sidecar(
          names=rewritten_names,
          label="rebuild-name-witness-rewritten")
        fixed_facts.write_h5(f=artifact, sidecar_text=rewritten)

  def test_coordinated_record_rewrite_is_refused_before_model_load(self):
    """A jointly rewritten record cannot outrank the rebuilt geometry."""
    with tempfile.TemporaryDirectory() as temp_dir:
      path_root = os.path.join(temp_dir, "tampered")
      self._write_artifact(
        path_root=path_root,
        rewritten_names=["beta", "alpha"])

      with mock.patch.object(
          results.torch,
          "load",
          side_effect=_ModelLoadReached("model weights were requested")) \
          as load_weights:
        with self.assertRaisesRegex(
            ValueError,
            "input geometry and its record disagree"):
          results.rebuild_emulator(
            path_root=path_root,
            device=torch.device("cpu"),
            compile_model=False)
        load_weights.assert_not_called()

  def test_matching_record_reaches_model_load(self):
    """The added comparison preserves the valid schema-v3 rebuild path."""
    with tempfile.TemporaryDirectory() as temp_dir:
      path_root = os.path.join(temp_dir, "matching")
      self._write_artifact(path_root=path_root)

      with mock.patch.object(
          results.torch,
          "load",
          side_effect=_ModelLoadReached("matching record passed")):
        with self.assertRaisesRegex(_ModelLoadReached, "matching record passed"):
          results.rebuild_emulator(
            path_root=path_root,
            device=torch.device("cpu"),
            compile_model=False)


if __name__ == "__main__":
  unittest.main()
