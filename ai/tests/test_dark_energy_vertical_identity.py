"""Check fixed dark-energy identity between an artifact and a live model.

The generator writes the physical names ``w`` and ``wa`` even when Cobaya
calls the present-day value ``w0`` or calculates ``wa`` from ``w0pwa``.  The
serving check must make the same conversion.  Otherwise a valid emulator is
refused because two equivalent names look different, or a fixed emulator is
mistakenly accepted by a chain that samples the same coordinate.
"""

from types import SimpleNamespace
import unittest

from emulator import fixed_facts
from emulator.syren_base import DARK_ENERGY_COORDINATE_ATOL


class _Parameterization:
  """Expose the three public Cobaya surfaces used by the identity reader."""

  def __init__(self, *, constants=(), sampled=(), dependencies=()):
    self._constants = dict(constants)
    self._sampled = dict(sampled)
    self.input_dependencies = {
      name: set(required) for name, required in dict(dependencies).items()
    }

  def constant_params(self):
    return dict(self._constants)

  def sampled_params(self):
    return dict(self._sampled)


def _model(*, constants=(), sampled=(), dependencies=(), theory_defaults=()):
  """Return a resolved-model stand-in with optional theory defaults."""
  theory = {}
  if theory_defaults:
    theory["boltzmann"] = SimpleNamespace(
      extra_args=dict(theory_defaults))
  return SimpleNamespace(
    theory=theory,
    parameterization=_Parameterization(
      constants=constants, sampled=sampled, dependencies=dependencies))


def _blocks(law, held):
  """Return the two fields read by the production vertical identity check."""
  return {
    fixed_facts.FIXED_FACTS_GROUP: {
      "dark_energy_law": law,
      "cosmology_fixed": dict(held),
    },
  }


class DarkEnergyVerticalIdentityTests(unittest.TestCase):
  """Require canonical aliases, explicit defaults, and sampled-value refusal."""

  def test_fixed_w0_matches_artifact_canonical_w(self):
    """A live constant named w0 is compared with the artifact's saved w."""
    resolved = fixed_facts.resolved_constants(_model(
      constants={"w0": -0.9}))
    self.assertEqual(resolved["w"], -0.9)
    self.assertEqual(resolved["wa"], 0.0)
    fixed_facts.check_vertical(
      _blocks("constant-w", {"w": -0.9, "wa": "n/a"}),
      resolved,
      "fixed-w artifact",
    )

  def test_fixed_transformed_pair_matches_canonical_cpl_artifact(self):
    """Fixed w0 and w0pwa become the saved physical w and nonzero wa."""
    resolved = fixed_facts.resolved_constants(_model(
      constants={"w0": -0.9, "w0pwa": -0.7}))
    self.assertEqual(resolved["w"], -0.9)
    self.assertAlmostEqual(resolved["wa"], 0.2)
    fixed_facts.check_vertical(
      _blocks("w0wa-cpl", {"w": -0.9, "wa": 0.2}),
      resolved,
      "fixed-CPL artifact",
    )

  def test_explicit_lcdm_record_matches_model_with_no_coordinates(self):
    """The saved law gives n/a dark-energy slots their physical values."""
    resolved = fixed_facts.resolved_constants(_model())
    self.assertEqual((resolved["w"], resolved["wa"]), (-1.0, 0.0))
    fixed_facts.check_vertical(
      _blocks(
        "cosmological-constant", {"w": "n/a", "wa": "n/a"}),
      resolved,
      "LCDM artifact",
    )

  def test_lcdm_artifact_refuses_chain_that_samples_dark_energy(self):
    """An artifact fixed to LCDM cannot borrow defaults for sampled values."""
    resolved = fixed_facts.resolved_constants(_model(
      sampled={"w": None, "w0pwa": None},
      dependencies={"wa": {"w", "w0pwa"}},
      theory_defaults={"w": -1.0, "wa": 0.0},
    ))
    self.assertNotIn("w", resolved)
    self.assertNotIn("wa", resolved)
    with self.assertRaisesRegex(ValueError, "does not say what w is"):
      fixed_facts.check_vertical(
        _blocks(
          "cosmological-constant", {"w": "n/a", "wa": "n/a"}),
        resolved,
        "LCDM artifact",
      )

  def test_different_fixed_alias_value_refuses_with_both_numbers(self):
    """Canonical spelling does not weaken the scientific equality check."""
    resolved = fixed_facts.resolved_constants(_model(
      constants={"w0": -0.8}))
    with self.assertRaisesRegex(
        ValueError, r"held fixed at -0\.9.*has w = -0\.8"):
      fixed_facts.check_vertical(
        _blocks("constant-w", {"w": -0.9, "wa": "n/a"}),
        resolved,
        "fixed-w artifact",
      )

  def test_conflicting_live_aliases_refuse_before_comparison(self):
    """The consumer never chooses between inconsistent w and w0 constants."""
    with self.assertRaisesRegex(ValueError, "aliases 'w'.*'w0'.*disagree"):
      fixed_facts.resolved_constants(_model(
        constants={"w": -0.9, "w0": -0.8}))

  def test_partial_fixed_sum_does_not_invent_present_day_value(self):
    """A lone w0pwa remains insufficient; absence never supplies w=-1."""
    with self.assertRaisesRegex(ValueError, "does not fix w or w0"):
      fixed_facts.resolved_constants(_model(
        constants={"w0pwa": -0.7}))

  def test_unknown_sampled_surface_does_not_invent_lcdm(self):
    """An unreadable Cobaya API cannot be interpreted as no sampled values."""
    class BrokenParameterization(_Parameterization):
      def sampled_params(self):
        raise RuntimeError("unavailable sampled-parameter surface")

    model = SimpleNamespace(
      theory={"boltzmann": SimpleNamespace(
        extra_args={"w": -1.0, "wa": 0.0})},
      parameterization=BrokenParameterization())
    resolved = fixed_facts.resolved_constants(model)
    self.assertNotIn("w", resolved)
    self.assertNotIn("wa", resolved)

  def test_vertical_equality_uses_the_coordinate_storage_tolerance(self):
    """Float32-equivalent values pass; a value beyond the bound refuses."""
    inside = 0.5 * DARK_ENERGY_COORDINATE_ATOL
    outside = 2.0 * DARK_ENERGY_COORDINATE_ATOL
    blocks = _blocks("constant-w", {"w": -0.9, "wa": "n/a"})
    fixed_facts.check_vertical(
      blocks,
      fixed_facts.resolved_constants(_model(
        constants={"w0": -0.9 + inside})),
      "inside-tolerance artifact",
    )
    with self.assertRaisesRegex(ValueError, "never shown that universe"):
      fixed_facts.check_vertical(
        blocks,
        fixed_facts.resolved_constants(_model(
          constants={"w0": -0.9 + outside})),
        "outside-tolerance artifact",
      )


if __name__ == "__main__":
  unittest.main()
