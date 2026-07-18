"""Check the deliberately narrow fixed-value comparison contract.

Cobaya permits arbitrary renamed and calculated parameters, so this check does
not try to prove that two parameterizations describe the same cosmology.  It
compares only concrete constants that use the same name on both sides.  A
missing, renamed, or unavailable value is unchecked and remains the caller's
responsibility.
"""

from types import SimpleNamespace
import unittest

from emulator import fixed_facts


class _Parameterization:
  """Expose the resolved model's plain constant-parameter mapping."""

  def __init__(self, constants=()):
    self._constants = dict(constants)

  def constant_params(self):
    return dict(self._constants)


def _model(constants=(), theory_defaults=()):
  """Return a small model-shaped object for ``resolved_constants``."""
  theory = {}
  if theory_defaults:
    theory["boltzmann"] = SimpleNamespace(extra_args=dict(theory_defaults))
  return SimpleNamespace(
    theory=theory,
    parameterization=_Parameterization(constants=constants))


def _blocks(held):
  """Return the one artifact field read by ``check_fixed_values``."""
  return {
    fixed_facts.FIXED_FACTS_GROUP: {
      "cosmology_fixed": dict(held),
    },
  }


class FixedValueBestEffortTests(unittest.TestCase):
  """Require direct mismatches without claiming parameter equivalence."""

  def test_direct_same_name_match_is_accepted(self):
    """A concrete value under the artifact's exact name can be compared."""
    fixed_facts.check_fixed_values(
      _blocks({"w": -0.9}), {"w": -0.9}, "fixed-w artifact")

  def test_direct_same_name_mismatch_names_both_values(self):
    """A direct disagreement is the one case this check refuses."""
    with self.assertRaises(ValueError) as caught:
      fixed_facts.check_fixed_values(
        _blocks({"w": -0.9}), {"w": -0.8}, "fixed-w artifact")
    message = str(caught.exception)
    self.assertIn("w", message)
    self.assertIn("-0.9", message)
    self.assertIn("-0.8", message)

  def test_missing_name_is_unchecked(self):
    """Silence is not treated as either agreement or disagreement."""
    fixed_facts.check_fixed_values(
      _blocks({"w": -0.9}), {}, "fixed-w artifact")

  def test_renamed_coordinate_is_unchecked(self):
    """The basic check does not guess that ``w0`` is the artifact's ``w``."""
    fixed_facts.check_fixed_values(
      _blocks({"w": -0.9}), {"w0": -0.8}, "fixed-w artifact")

  def test_not_applicable_value_is_unchecked(self):
    """An artifact's unavailable value is not a concrete value to compare."""
    fixed_facts.check_fixed_values(
      _blocks({"w": fixed_facts.NOT_APPLICABLE}),
      {"w": -0.8},
      "LCDM artifact")

  def test_resolver_preserves_names_without_canonicalizing_aliases(self):
    """Resolved constants remain the concrete names the model exposes."""
    resolved = fixed_facts.resolved_constants(_model(
      constants={"w0": -0.9, "w0pwa": -0.7}))
    self.assertEqual(resolved, {"w0": -0.9, "w0pwa": -0.7})
    self.assertNotIn("w", resolved)
    self.assertNotIn("wa", resolved)

  def test_resolver_keeps_existing_source_precedence(self):
    """A parameter constant still overrides an exact-name theory default."""
    resolved = fixed_facts.resolved_constants(_model(
      constants={"w": -0.9}, theory_defaults={"w": -1.0, "wa": 0.0}))
    self.assertEqual(resolved["w"], -0.9)
    self.assertEqual(resolved["wa"], 0.0)


if __name__ == "__main__":
  unittest.main()
