"""CPU tests for the laws deciding whether two saved emulators may be paired.

A pair is combined into one prediction, so both halves must come from one
cosmology and one sampled coordinate set. ``check_horizontal`` is the law that
enforces this, and ``check_support`` is the law that decides whether a served
emulator may be asked about a point at all.

Mutation testing found these unguarded: disabling the coordinate-name
comparison, the fact-key comparison inside ``_same_fact``, or the unknown
support-kind refusal each left the whole suite green. ``check_horizontal`` is
named by no test in the repository at all.
"""

import unittest

from emulator import fixed_facts


def facts_block(**overrides):
  """Return one complete fixed-facts mapping with the named keys replaced.

  Arguments:
    overrides = the fact keys to change from the shared baseline.

  Returns:
    A mapping carrying every key in FIXED_FACTS_KEYS.
  """
  block = {
    "block_version": 1,
    "generator": "dataset_generator_background.py",
    "family": "background",
    "cosmology_fixed": {"omega_b_h2": 0.02237, "n_s": 0.9649},
    "neutrino_convention": "one massive",
    "flat_only": True,
    "dark_energy_law": "w0wa",
    "dark_energy_inputs": ["w", "wa"],
    "cl_units": "n/a",
    "base_identity": "none",
    "param_dtype": "float64",
    "decimal_policy": "float32-shortest",
  }
  block.update(overrides)
  return block


def blocks(names, **fact_overrides):
  """Return one saved emulator's two record groups.

  Arguments:
    names          = the sampled coordinate names, in order.
    fact_overrides = fixed-fact keys to change from the baseline.

  Returns:
    A mapping with the fixed-facts and input-domain groups.
  """
  return {
    fixed_facts.FIXED_FACTS_GROUP: facts_block(**fact_overrides),
    fixed_facts.INPUT_DOMAIN_GROUP: {"names": list(names)},
  }


class HorizontalPairingTests(unittest.TestCase):
  """Require both halves of a pair to describe one universe and one domain."""

  def test_two_matching_emulators_may_be_served_together(self):
    """The baseline must pass, or the refusals below prove nothing."""
    fixed_facts.check_horizontal(
      blocks(["H0", "omega_m"]), blocks(["H0", "omega_m"]),
      "first.h5", "second.h5")

  def test_different_sampled_coordinates_are_refused(self):
    """A coordinate one half never saw cannot be asked of the pair.

    The half that never sampled it would answer as though it were held
    fixed, so the two halves would be evaluated at different points inside
    one prediction.
    """
    with self.assertRaisesRegex(
        ValueError, "not sampled over the same coordinates"):
      fixed_facts.check_horizontal(
        blocks(["H0", "omega_m"]), blocks(["H0", "omega_m", "n_s"]),
        "first.h5", "second.h5")

  def test_the_same_coordinates_in_a_different_order_are_refused(self):
    """Order is identity here: column two of one is column one of the other."""
    with self.assertRaisesRegex(
        ValueError, "not sampled over the same coordinates"):
      fixed_facts.check_horizontal(
        blocks(["H0", "omega_m"]), blocks(["omega_m", "H0"]),
        "first.h5", "second.h5")

  def test_a_differing_cosmology_names_the_coordinate_that_differs(self):
    """A refusal must name the one fact that differs, not print two dicts."""
    with self.assertRaisesRegex(ValueError, r"cosmology_fixed\['n_s'\]"):
      fixed_facts.check_horizontal(
        blocks(["H0"]),
        blocks(["H0"],
               cosmology_fixed={"omega_b_h2": 0.02237, "n_s": 0.96}),
        "first.h5", "second.h5")

  def test_a_differing_neutrino_convention_is_refused(self):
    """Two universes with different neutrino treatments are not one pair."""
    with self.assertRaisesRegex(ValueError, "different universes"):
      fixed_facts.check_horizontal(
        blocks(["H0"]),
        blocks(["H0"], neutrino_convention="three degenerate"),
        "first.h5", "second.h5")


class SameFactTests(unittest.TestCase):
  """Require persisted-fact comparison to notice a missing or extra key."""

  def test_two_mappings_with_different_keys_are_not_the_same_fact(self):
    """A fact one record carries and the other drops is a difference."""
    self.assertFalse(
      fixed_facts._same_fact({"a": 1, "b": 2}, {"a": 1}))
    self.assertFalse(
      fixed_facts._same_fact({"a": 1}, {"a": 1, "b": 2}))

  def test_two_mappings_with_the_same_keys_and_values_are_the_same_fact(self):
    """The accepting path must still work."""
    self.assertTrue(
      fixed_facts._same_fact({"a": 1, "b": 2}, {"b": 2, "a": 1}))


class SupportKindTests(unittest.TestCase):
  """Require an unrecognized support kind to stop the prediction."""

  def _compiled(self, constraint):
    """Return a compiled support declaring the given constraint kind."""
    return {
      "where": "saved.h5",
      "constraint": constraint,
      "generator": "dataset_generator_background.py",
      "names": ["H0"],
      "low": {"H0": 60.0},
      "high": {"H0": 80.0},
    }

  def test_a_box_support_accepts_a_point_inside_it(self):
    """The baseline must pass, or the refusals below prove nothing."""
    fixed_facts.check_support(self._compiled("box"), {"H0": 70.0})

  def test_an_unknown_support_kind_is_refused(self):
    """A kind this code cannot compare against must not be guessed at."""
    with self.assertRaisesRegex(ValueError, "does not know how to compare"):
      fixed_facts.check_support(self._compiled("simplex"), {"H0": 70.0})

  def test_an_undeclared_support_is_refused(self):
    """An emulator with no declared region is a test double, not a model."""
    with self.assertRaisesRegex(ValueError, "declares no support"):
      fixed_facts.check_support(self._compiled("undeclared"), {"H0": 70.0})


if __name__ == "__main__":
  unittest.main()
