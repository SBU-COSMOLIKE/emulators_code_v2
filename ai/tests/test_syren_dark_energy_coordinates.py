"""CPU checks for the shared Syren dark-energy coordinate resolver.

The analytic matter-power base consumes ``(w0, wa)`` even when a Cobaya
configuration exposes ``w``, ``w0pwa``, or redundant forms.  These tests keep
the conversion in one place and prove that missing evolution information can
no longer become ``wa = 0`` by accident.
"""

import unittest

import numpy as np

from emulator.syren_base import DARK_ENERGY_COORDINATE_ATOL
from emulator.syren_base import SYREN_AMPLITUDE_ATOL
from emulator.syren_base import SYREN_AMPLITUDE_RTOL
from emulator.syren_base import resolve_dark_energy_coordinates
from emulator.syren_base import syren_params_from


class DarkEnergyCoordinateTests(unittest.TestCase):

  def test_complete_direct_and_transformed_coordinates_agree(self):
    direct = resolve_dark_energy_coordinates({"w0": -0.8, "wa": 0.3})
    transformed = resolve_dark_energy_coordinates(
      {"w": -0.8, "w0pwa": -0.5})
    explicit_cpl = resolve_dark_energy_coordinates(
      {"w": -0.8, "w0pwa": -0.5}, dark_energy_law="w0wa-cpl")
    self.assertEqual(direct, (-0.8, 0.3))
    self.assertAlmostEqual(transformed[0], direct[0])
    self.assertAlmostEqual(transformed[1], direct[1])
    self.assertEqual(explicit_cpl, transformed)

  def test_every_redundant_form_must_agree(self):
    inside = 0.5 * DARK_ENERGY_COORDINATE_ATOL
    resolved = resolve_dark_energy_coordinates({
      "w": -0.8,
      "w0": -0.8 + inside,
      "wa": 0.3,
      "w0pwa": -0.5 + inside,
    })
    self.assertEqual(resolved, (-0.8, 0.3))

    with self.assertRaisesRegex(ValueError, "w0pwa.*w0 \\+ wa"):
      resolve_dark_energy_coordinates({
        "w": -0.8,
        "w0": -0.8,
        "wa": 0.3,
        "w0pwa": -0.4,
      })

  def test_alias_conflict_is_reported_before_missing_wa(self):
    with self.assertRaisesRegex(ValueError, "aliases 'w'.*'w0'.*disagree"):
      resolve_dark_energy_coordinates({"w": -1.0, "w0": -0.7})

  def test_tolerance_is_absolute_and_tied_to_float32_storage(self):
    self.assertEqual(
      DARK_ENERGY_COORDINATE_ATOL,
      4.0 * np.finfo(np.float32).eps)
    inside = 0.5 * DARK_ENERGY_COORDINATE_ATOL
    outside = 2.0 * DARK_ENERGY_COORDINATE_ATOL
    self.assertEqual(
      resolve_dark_energy_coordinates({
        "w": -1.0,
        "w0": -1.0 + inside,
        "wa": 0.2,
      }),
      (-1.0, 0.2))
    with self.assertRaisesRegex(ValueError, "relative tolerance is zero"):
      resolve_dark_energy_coordinates({
        "w": -1.0,
        "w0": -1.0 + outside,
        "wa": 0.2,
      })

  def test_incomplete_coordinates_never_imply_constant_w(self):
    with self.assertRaisesRegex(ValueError, "cannot resolve wa"):
      resolve_dark_energy_coordinates({"w": -0.8})
    with self.assertRaisesRegex(ValueError, "cannot resolve w0"):
      resolve_dark_energy_coordinates({"w0pwa": -0.5})
    with self.assertRaisesRegex(ValueError, "cannot resolve w0"):
      resolve_dark_energy_coordinates({"wa": 0.3})
    with self.assertRaisesRegex(ValueError, "cannot resolve w0"):
      resolve_dark_energy_coordinates({})

  def test_constant_w_law_is_the_only_single_coordinate_zero_wa_path(self):
    self.assertEqual(
      resolve_dark_energy_coordinates(
        {"w": -0.7}, dark_energy_law="constant-w"),
      (-0.7, 0.0))
    self.assertEqual(
      resolve_dark_energy_coordinates(
        {"w0pwa": -0.7}, dark_energy_law="constant-w"),
      (-0.7, 0.0))
    with self.assertRaisesRegex(ValueError, "constant-w requires wa=0"):
      resolve_dark_energy_coordinates(
        {"w": -0.7, "wa": 0.2}, dark_energy_law="constant-w")
    with self.assertRaisesRegex(ValueError, "needs 'w'.*'w0'.*'w0pwa'"):
      resolve_dark_energy_coordinates({}, dark_energy_law="constant-w")

  def test_cosmological_constant_law_supplies_and_checks_both_values(self):
    self.assertEqual(
      resolve_dark_energy_coordinates(
        {}, dark_energy_law="cosmological-constant"),
      (-1.0, 0.0))
    self.assertEqual(
      resolve_dark_energy_coordinates(
        {"w": -1.0, "w0": -1.0, "wa": 0.0, "w0pwa": -1.0},
        dark_energy_law="cosmological-constant"),
      (-1.0, 0.0))
    for params, message in (
        ({"w": -0.9}, "requires w0=-1"),
        ({"wa": 0.1}, "requires wa=0"),
        ({"w0pwa": -0.9}, "requires w0pwa=-1"),
    ):
      with self.subTest(params=params):
        with self.assertRaisesRegex(ValueError, message):
          resolve_dark_energy_coordinates(
            params, dark_energy_law="cosmological-constant")

  def test_cpl_law_does_not_supply_a_missing_coordinate(self):
    with self.assertRaisesRegex(ValueError, "cannot resolve wa"):
      resolve_dark_energy_coordinates(
        {"w0": -0.8}, dark_energy_law="w0wa-cpl")
    with self.assertRaisesRegex(ValueError, "cannot resolve w0"):
      resolve_dark_energy_coordinates(
        {"wa": 0.3}, dark_energy_law="w0wa-cpl")

  def test_unknown_or_nontext_law_is_refused(self):
    with self.assertRaisesRegex(ValueError, "dark_energy_law must be one of"):
      resolve_dark_energy_coordinates(
        {"w": -0.8, "wa": 0.3}, dark_energy_law="cpl")
    with self.assertRaisesRegex(TypeError, "native text or None"):
      resolve_dark_energy_coordinates(
        {"w": -0.8, "wa": 0.3}, dark_energy_law=True)

  def test_python_and_numpy_real_scalars_are_accepted(self):
    resolved = resolve_dark_energy_coordinates({
      "w": np.float32(-0.8),
      "w0": np.float64(-0.8),
      "wa": np.float32(0.3),
      "w0pwa": np.float64(-0.5),
    })
    self.assertAlmostEqual(resolved[0], -0.8, places=6)
    self.assertAlmostEqual(resolved[1], 0.3, places=6)

  def test_each_coordinate_must_be_a_finite_real_nonboolean_scalar(self):
    invalid = (True, np.bool_(False), "-0.8", np.asarray(-0.8), 1.0 + 2.0j,
               10 ** 1000, float("nan"), float("inf"))
    for name in ("w", "w0", "wa", "w0pwa"):
      for value in invalid:
        params = {"w0": -0.8, "wa": 0.3}
        params[name] = value
        with self.subTest(name=name, value=repr(value)):
          with self.assertRaises((TypeError, ValueError)):
            resolve_dark_energy_coordinates(params)

  def test_syren_parameter_tuple_delegates_to_the_resolver(self):
    params = {
      "As_1e9": 2.1,
      "ns": 0.966,
      "H0": 67.3,
      "omegab": 0.049,
      "omegam": 0.31,
      "w": -0.8,
      "w0pwa": -0.5,
    }
    values = syren_params_from(params)
    self.assertEqual(len(values), 7)
    self.assertEqual(values[:5], (2.1, 0.966, 67.3, 0.049, 0.31))
    self.assertAlmostEqual(values[5], -0.8)
    self.assertAlmostEqual(values[6], 0.3)

    constant = dict(params)
    del constant["w0pwa"]
    values = syren_params_from(constant, dark_energy_law="constant-w")
    self.assertEqual(values[5:], (-0.8, 0.0))

  def test_syren_tuple_reports_alias_conflict_before_other_missing_values(self):
    with self.assertRaisesRegex(ValueError, "aliases 'w'.*'w0'.*disagree"):
      syren_params_from({"w": -1.0, "w0": -0.7})


class SyrenAmplitudeAliasTests(unittest.TestCase):

  def _base_params(self):
    return {
      "ns": 0.966,
      "H0": 67.3,
      "omegab": 0.049,
      "omegam": 0.31,
      "w0": -1.0,
      "wa": 0.0,
    }

  def test_as_1e9_only_is_unchanged(self):
    params = self._base_params()
    params["As_1e9"] = 2.1
    values = syren_params_from(params)
    self.assertEqual(values[0], 2.1)

  def test_as_only_is_unchanged(self):
    params = self._base_params()
    params["As"] = 2.1e-9
    values = syren_params_from(params)
    self.assertEqual(values[0], 2.1e-9 * 1e9)

  def test_consistent_two_names_canonicalize_to_as_1e9(self):
    params = self._base_params()
    params["As_1e9"] = 2.1
    params["As"] = 2.1e-9
    values = syren_params_from(params)
    self.assertEqual(values[0], 2.1)

  def test_conflicting_two_names_refuse_naming_both(self):
    params = self._base_params()
    params["As_1e9"] = 2.1
    params["As"] = 9e-9
    message = "conflicting primordial amplitudes"
    expected_left = repr(2.1)
    expected_right = repr(9e-9 * 1e9)
    try:
      syren_params_from(params)
      self.fail("expected ValueError for conflicting amplitude names")
    except ValueError as error:
      text = str(error)
      self.assertIn(message, text)
      self.assertIn(expected_left, text)
      self.assertIn(expected_right, text)

  def test_amplitude_tolerance_is_absolute_and_relative(self):
    self.assertEqual(
      SYREN_AMPLITUDE_ATOL,
      4.0 * np.finfo(np.float32).eps)
    self.assertEqual(
      SYREN_AMPLITUDE_RTOL,
      4.0 * np.finfo(np.float32).eps)
    base = 2.1
    allowance = SYREN_AMPLITUDE_ATOL + SYREN_AMPLITUDE_RTOL * base
    inside_offset = 0.5 * allowance
    outside_offset = 2.0 * allowance
    consistent = self._base_params()
    consistent["As_1e9"] = base
    consistent["As"] = (base + inside_offset) * 1e-9
    values = syren_params_from(consistent)
    self.assertEqual(values[0], base)
    conflicting = self._base_params()
    conflicting["As_1e9"] = base
    conflicting["As"] = (base + outside_offset) * 1e-9
    with self.assertRaisesRegex(ValueError, "conflicting primordial amplitudes"):
      syren_params_from(conflicting)


if __name__ == "__main__":
  unittest.main()
