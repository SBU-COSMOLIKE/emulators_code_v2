"""CPU tests for the grid2d constant-mask state contract.

The mask is a scientific fact.  True removes the network prediction at one
coordinate and serves the stored boost identity instead.  These tests use a
two-redshift, three-wavenumber surface so the flattened z-outer order can be
read directly:

  flat index:  0       1      2       3       4      5
  coordinate: (z0,k0) (z0,k1) (z0,k2) (z1,k0) (z1,k1) (z1,k2)
"""

import unittest

import numpy as np
import torch

from emulator.geometries.grid2d import Grid2DGeometry


class Grid2DConstMaskStateTest(unittest.TestCase):
  """Exercise explicit state and the forbidden presence-inference mutation."""

  def setUp(self):
    self.device = torch.device("cpu")
    self.z_axis = np.array(
      [0.0, 1.0],
      dtype="float64")
    self.k_axis = np.array(
      [0.01, 0.1, 1.0],
      dtype="float64")

  def _unpinned_geometry(self):
    center = np.array(
      [2.0, 3.0, 4.0, 2.5, 3.5, 4.5],
      dtype="float64")
    scale = np.ones(6, dtype="float64")
    return Grid2DGeometry.from_stats(
      device=self.device,
      center=center,
      scale=scale,
      z=self.z_axis,
      k=self.k_axis,
      quantity="pklin",
      units="Mpc3",
      law="none")

  def _pinned_geometry(self):
    # The first k coordinate in each z row is the raw-boost identity B=1.
    # Zero spread asks from_stats to pin exactly those two coordinates.
    center = np.array(
      [1.0, 1.2, 1.3, 1.0, 1.4, 1.5],
      dtype="float64")
    scale = np.array(
      [0.0, 0.2, 0.3, 0.0, 0.4, 0.5],
      dtype="float64")
    return Grid2DGeometry.from_stats(
      device=self.device,
      center=center,
      scale=scale,
      z=self.z_axis,
      k=self.k_axis,
      quantity="boost",
      units="dimensionless",
      law="none")

  def _presence_inferred_geometry(
      self,
      state):
    """Demonstrate the forbidden reader that maps a missing key to None."""
    return Grid2DGeometry(
      device=self.device,
      quantity=state["quantity"],
      units=state["units"],
      law=state["law"],
      z=state["z"],
      k=state["k"],
      center=state["center"],
      scale=state["scale"],
      const_mask=None)

  def _direct_constructor_arguments(self):
    """Return every constructor argument except the required mask."""
    return {
      "device": self.device,
      "quantity": "pklin",
      "units": "Mpc3",
      "law": "none",
      "z": self.z_axis,
      "k": self.k_axis,
      "center": np.ones(6, dtype="float64"),
      "scale": np.ones(6, dtype="float64"),
    }

  def test_direct_constructor_requires_the_mask_argument(self):
    constructor_arguments = self._direct_constructor_arguments()
    with self.assertRaisesRegex(TypeError, "const_mask"):
      Grid2DGeometry(**constructor_arguments)

  def test_unpinned_state_persists_an_all_false_mask(self):
    geometry = self._unpinned_geometry()
    state = geometry.state()

    self.assertIn("const_mask", state)
    self.assertEqual(state["const_mask"].dtype, torch.uint8)
    self.assertEqual(tuple(state["const_mask"].shape), (6,))
    self.assertEqual(int(state["const_mask"].sum().item()), 0)

    rebuilt = Grid2DGeometry.from_state(
      device=self.device,
      state=state)
    model_output = torch.tensor(
      [[0.25, -0.5, 0.75, -1.0, 1.25, -1.5]],
      dtype=torch.float32)
    direct_decode = model_output * geometry.scale + geometry.center
    self.assertTrue(torch.equal(
      geometry.decode(model_output),
      direct_decode))
    self.assertTrue(torch.equal(
      geometry.decode(model_output),
      rebuilt.decode(model_output)))

  def test_existing_low_k_pin_round_trips(self):
    geometry = self._pinned_geometry()
    state = geometry.state()
    expected_mask = torch.tensor(
      [1, 0, 0, 1, 0, 0],
      dtype=torch.uint8)

    self.assertTrue(torch.equal(state["const_mask"], expected_mask))
    rebuilt = Grid2DGeometry.from_state(
      device=self.device,
      state=state)
    model_output = torch.full(
      (1, 6),
      0.25,
      dtype=torch.float32)
    decoded = rebuilt.decode(model_output)
    self.assertEqual(float(decoded[0, 0].item()), 1.0)
    self.assertEqual(float(decoded[0, 3].item()), 1.0)
    self.assertTrue(torch.equal(
      geometry.decode(model_output),
      decoded))

  def test_missing_mask_cannot_select_the_unpinned_policy(self):
    geometry = self._pinned_geometry()
    complete_state = geometry.state()
    missing_mask_state = dict(complete_state)
    missing_mask_state.pop("const_mask")

    model_output = torch.full(
      (1, 6),
      0.25,
      dtype=torch.float32)
    intact_value = float(geometry.decode(model_output)[0, 0].item())
    presence_inferred = self._presence_inferred_geometry(
      missing_mask_state)
    mutated_value = float(
      presence_inferred.decode(model_output)[0, 0].item())

    self.assertEqual(intact_value, 1.0)
    self.assertEqual(mutated_value, 1.25)
    with self.assertRaisesRegex(KeyError, "Re-save"):
      Grid2DGeometry.from_state(
        device=self.device,
        state=missing_mask_state)

  def test_mask_shape_and_storage_type_are_validated(self):
    base = self._unpinned_geometry().state()
    two_dimensional = dict(base)
    two_dimensional["const_mask"] = torch.zeros(
      (2, 3),
      dtype=torch.uint8)
    with self.assertRaisesRegex(ValueError, "one-dimensional"):
      Grid2DGeometry.from_state(
        device=self.device,
        state=two_dimensional)

    integer_array = dict(base)
    integer_array["const_mask"] = torch.zeros(
      6,
      dtype=torch.int64)
    with self.assertRaisesRegex(TypeError, "booleans or persisted uint8"):
      Grid2DGeometry.from_state(
        device=self.device,
        state=integer_array)

    nonbinary_array = dict(base)
    nonbinary_array["const_mask"] = torch.tensor(
      [0, 0, 0, 0, 0, 2],
      dtype=torch.uint8)
    with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
      Grid2DGeometry.from_state(
        device=self.device,
        state=nonbinary_array)


if __name__ == "__main__":
  unittest.main()
