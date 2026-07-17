"""Schema-v3 witness for rebuild-time input-name validation.

The fixed-facts blocks and their copied producer text protect one another, but
an editor can rewrite both copies together.  The rebuilt parameter geometry is
the independent source of the whitening-column order.  These tests prove that
rebuild compares that geometry against the record before loading model weights,
and that an untampered matching record still reaches the model-loading stage.
"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import torch

from ai.gates.checks.compile_recipe import save_fixture
from emulator import fixed_facts
from emulator import results
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry


class _ModelLoadReached(BaseException):
  """Sentinel proving rebuild advanced past all artifact validation."""


class RebuildFixedFactsNamesTest(unittest.TestCase):
  """Exercise the independent geometry-to-record comparison on schema v3."""

  geometry_names = ["p0", "p1"]

  def _write_artifact(self, path_root, rewritten_names=None, rescale="none"):
    """Write a schema-v3 artifact with a caller-selected rescale fact."""
    save_fixture(
      path_root=Path(path_root), compile_mode="default",
      case_label="rebuild-fixed-facts-names")
    h5_path = path_root + ".h5"
    with h5py.File(h5_path, "r+") as artifact:
      if rescale is None:
        del artifact.attrs["rescale"]
      else:
        artifact.attrs["rescale"] = rescale

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

  def test_public_rebuild_refuses_invalid_rescale_before_construction(self):
    """The public reader, not merely its helper, enforces the transform."""
    cases = (("missing", None), ("rescaled", "rescaled"),
             ("residual", "residual"), ("boolean", True),
             ("byte-string", b"none"))
    with tempfile.TemporaryDirectory() as temp_dir:
      for label, value in cases:
        with self.subTest(label=label):
          path_root = os.path.join(temp_dir, label)
          self._write_artifact(path_root=path_root, rescale=value)
          with mock.patch.object(
              ParamGeometry,
              "from_state",
              side_effect=AssertionError("geometry construction reached")) \
              as construct_parameter_geometry, mock.patch.object(
                ScalarGeometry,
                "from_state",
                side_effect=AssertionError("geometry construction reached")) \
              as construct_output_geometry, mock.patch.object(
                results.torch,
                "load",
                side_effect=AssertionError("model load reached")) \
              as load_weights:
            with self.assertRaisesRegex(
                (KeyError, ValueError), "rescale.*inverse transform"):
              results.rebuild_emulator(
                path_root=path_root,
                device=torch.device("cpu"),
                compile_model=False)
            construct_parameter_geometry.assert_not_called()
            construct_output_geometry.assert_not_called()
            load_weights.assert_not_called()

  def test_bypassing_rescale_reader_still_fails_output_identity(self):
    """A forged rescale cannot bypass the independent output identity."""
    with tempfile.TemporaryDirectory() as temp_dir:
      path_root = os.path.join(temp_dir, "bypassed-rescale")
      self._write_artifact(path_root=path_root, rescale="rescaled")
      with mock.patch.object(
          results,
          "_read_public_rescale",
          return_value="rescaled"), mock.patch.object(
            results.torch,
            "load",
            side_effect=AssertionError("model load reached")) as load_weights:
        with self.assertRaisesRegex(
            ValueError, "output identity disagrees"):
          results.rebuild_emulator(
            path_root=path_root,
            device=torch.device("cpu"),
            compile_model=False)
        load_weights.assert_not_called()


if __name__ == "__main__":
  unittest.main()
