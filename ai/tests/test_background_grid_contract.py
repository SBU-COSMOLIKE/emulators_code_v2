"""Focused CPU checks for the background quantity, units, and offset rules."""

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from emulator.experiment import validate_grid
from emulator.geometries.grid import (
  BACKGROUND_QUANTITY_UNITS,
  GridGeometry,
)


ROOT = Path(__file__).resolve().parents[2]


def _grid_config(quantity, units, law="none", offset=None):
  """Return one complete background configuration for validation."""
  grid = {
    "quantity": quantity,
    "units": units,
    "law": law,
    "z_file": "background_z.npy",
  }
  if offset is not None:
    grid["offset"] = offset
  return {
    "data": {
      "grid": grid,
      "train_dv": "train.npy",
      "val_dv": "val.npy",
      "train_params": "train.1.txt",
      "val_params": "val.1.txt",
      "train_covmat": "train.covmat",
    },
  }


def _training_rows():
  """Return positive, nonconstant rows for a two-point redshift grid."""
  redshifts = np.array([0.1, 0.2], dtype="float64")
  targets = np.array(
    [
      [70.0, 72.0],
      [71.0, 74.0],
      [72.0, 76.0],
    ],
    dtype="float64",
  )
  return redshifts, targets


def _load_adapter_module():
  """Load the Cobaya adapter with a small Theory stub for this CPU test."""
  cobaya_module = types.ModuleType("cobaya")
  theory_module = types.ModuleType("cobaya.theory")

  class _Theory:
    renames = {}
    extra_args = {}

    def initialize(self):
      return None

  theory_module.Theory = _Theory
  module_path = ROOT / "cobaya_theory" / "emul_baosn.py"
  spec = importlib.util.spec_from_file_location(
    "background_grid_contract_adapter",
    module_path,
  )
  module = importlib.util.module_from_spec(spec)
  replacements = {
    "cobaya": cobaya_module,
    "cobaya.theory": theory_module,
  }
  with mock.patch.dict(sys.modules, replacements):
    spec.loader.exec_module(module)
  return module


class BackgroundGridContractTests(unittest.TestCase):
  """Check every production boundary that owns background metadata."""

  def test_registry_contains_the_two_served_pairs(self):
    self.assertEqual(
      BACKGROUND_QUANTITY_UNITS,
      {
        "Hubble": "km/s/Mpc",
        "D_M": "Mpc",
      },
    )

  def test_config_accepts_each_quantity_and_either_target_law(self):
    cases = (
      ("Hubble", "km/s/Mpc", "none", None),
      ("Hubble", "km/s/Mpc", "log_offset", 1.0),
      ("D_M", "Mpc", "none", None),
      ("D_M", "Mpc", "log_offset", 1.0),
    )
    for quantity, units, law, offset in cases:
      with self.subTest(quantity=quantity, law=law):
        config = _grid_config(
          quantity=quantity,
          units=units,
          law=law,
          offset=offset,
        )
        validated = validate_grid(
          cfg=config,
          train_args={"model": {"ia": "none"}},
          rescale="none",
        )
        self.assertEqual(validated["quantity"], quantity)
        self.assertEqual(validated["units"], units)

  def test_config_rejects_unknown_mismatched_and_nonstring_pairs(self):
    bad_pairs = (
      ("Hubble", "Mpc"),
      ("D_M", "km/s/Mpc"),
      ("distance", "Mpc"),
      ("Hubble", "bananas"),
      ("Hubble", True),
      ("Hubble", None),
      ("Hubble", 3.14),
      (True, "km/s/Mpc"),
    )
    for quantity, units in bad_pairs:
      with self.subTest(quantity=quantity, units=units):
        config = _grid_config(quantity=quantity, units=units)
        with self.assertRaisesRegex(ValueError, "accepted pairs"):
          validate_grid(
            cfg=config,
            train_args={"model": {"ia": "none"}},
            rescale="none",
          )

  def test_config_rejects_nonfinite_boolean_and_quoted_offsets(self):
    bad_offsets = (
      math.inf,
      -math.inf,
      math.nan,
      True,
      "1.0",
    )
    for offset in bad_offsets:
      with self.subTest(offset=offset):
        config = _grid_config(
          quantity="Hubble",
          units="km/s/Mpc",
          law="log_offset",
          offset=offset,
        )
        with self.assertRaisesRegex(ValueError, "finite real number"):
          validate_grid(
            cfg=config,
            train_args={"model": {"ia": "none"}},
            rescale="none",
          )

  def test_geometry_rejects_bad_pairs_before_computing_statistics(self):
    redshifts, targets = _training_rows()
    bad_pairs = (
      ("Hubble", "Mpc"),
      ("distance", "Mpc"),
      ("Hubble", True),
    )
    for quantity, units in bad_pairs:
      with self.subTest(quantity=quantity, units=units):
        with self.assertRaisesRegex(ValueError, "accepted pairs"):
          GridGeometry.from_targets(
            device=torch.device("cpu"),
            targets=targets,
            z=redshifts,
            quantity=quantity,
            units=units,
            law="none",
          )

  def test_geometry_rejects_bad_offsets_before_float_conversion(self):
    redshifts, targets = _training_rows()
    bad_offsets = (
      math.inf,
      -math.inf,
      math.nan,
      True,
      "inf",
    )
    for offset in bad_offsets:
      with self.subTest(offset=offset):
        with self.assertRaisesRegex(ValueError, "offset must be a finite"):
          GridGeometry.from_targets(
            device=torch.device("cpu"),
            targets=targets,
            z=redshifts,
            quantity="Hubble",
            units="km/s/Mpc",
            law="log_offset",
            offset=offset,
          )

  def test_geometry_names_each_nonfinite_training_boundary(self):
    redshifts, targets = _training_rows()
    raw_nonfinite = targets.copy()
    raw_nonfinite[0, 0] = math.nan
    with self.assertRaisesRegex(ValueError, "raw target rows"):
      GridGeometry.from_targets(
        device=torch.device("cpu"),
        targets=raw_nonfinite,
        z=redshifts,
        quantity="Hubble",
        units="km/s/Mpc",
        law="none",
      )

    overflow_rows = np.full((3, 2), 1.0e308, dtype="float64")
    with self.assertRaisesRegex(ValueError, r"target \+ offset"):
      GridGeometry.from_targets(
        device=torch.device("cpu"),
        targets=overflow_rows,
        z=redshifts,
        quantity="Hubble",
        units="km/s/Mpc",
        law="log_offset",
        offset=1.0e308,
      )

    scale_overflow_rows = np.array(
      [
        [-1.0e308, -1.0e308],
        [1.0e308, 1.0e308],
      ],
      dtype="float64",
    )
    with self.assertRaisesRegex(ValueError, "law-space scale"):
      GridGeometry.from_targets(
        device=torch.device("cpu"),
        targets=scale_overflow_rows,
        z=redshifts,
        quantity="D_M",
        units="Mpc",
        law="none",
      )

  def test_rebuild_rejects_forged_pair_offset_center_and_scale(self):
    redshifts, targets = _training_rows()
    geometry = GridGeometry.from_targets(
      device=torch.device("cpu"),
      targets=targets,
      z=redshifts,
      quantity="Hubble",
      units="km/s/Mpc",
      law="log_offset",
      offset=1.0,
    )
    valid_state = geometry.state()

    bad_states = (
      ("pair", {"units": "Mpc"}, "accepted pairs"),
      ("offset", {"offset": torch.tensor(math.inf)}, "saved offset"),
      (
        "center",
        {"center": torch.tensor([math.nan, 0.0])},
        "GridGeometry.center",
      ),
      (
        "scale",
        {"scale": torch.tensor([1.0, math.inf])},
        "GridGeometry.scale",
      ),
    )
    for name, replacement, message in bad_states:
      with self.subTest(field=name):
        forged_state = dict(valid_state)
        forged_state.update(replacement)
        with self.assertRaisesRegex(ValueError, message):
          GridGeometry.from_state(
            device=torch.device("cpu"),
            state=forged_state,
          )

  def test_adapter_reads_the_shared_registry_and_rejects_a_bad_pair(self):
    adapter_module = _load_adapter_module()
    self.assertIs(
      adapter_module.BACKGROUND_QUANTITY_UNITS,
      BACKGROUND_QUANTITY_UNITS,
    )

    predictor_facts = {
      "bad_hubble": ("Hubble", "Mpc"),
      "distance": ("D_M", "Mpc"),
    }

    class _Predictor:
      def __init__(self, path, device, compile_model=False):
        del device, compile_model
        key = Path(path).name
        quantity, units = predictor_facts[key]
        self._grid = True
        self._scalar = False
        self._cmb = False
        self._grid2d = False
        self.quantity = quantity
        self.units = units
        self.names = []

    adapter_module.EmulatorPredictor = _Predictor
    adapter = adapter_module.emul_baosn()
    adapter.extra_args = {
      "device": "cpu",
      "emulators": [str(ROOT / "bad_hubble"), str(ROOT / "distance")],
    }
    with self.assertRaisesRegex(ValueError, r"\('Hubble', 'Mpc'\)"):
      adapter.initialize()


if __name__ == "__main__":
  unittest.main()
