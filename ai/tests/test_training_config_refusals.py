"""CPU tests that pin the train_args refusals no other test stood behind.

Each refusal below was reachable from a user's YAML and would have produced a
run that trains but quietly does not do what the file asked: an anneal block
that never ramps, a berhu knot equal to its cap. Mutation testing found that
deleting any of these checks left all 821 tests green, so a later edit could
have removed one and every check would still have reported success.

The tests are grouped by the function that owns the refusal, and each one
names the configuration a user would actually write.
"""

import unittest

from emulator import training


class AnnealBlockRefusalTests(unittest.TestCase):
  """Require _validate_anneal_block to refuse a ramp that cannot ramp."""

  def _block(self, **overrides):
    """Return a valid anneal block with the named fields replaced.

    Arguments:
      overrides = the fields to change from the valid baseline.

    Returns:
      A dict suitable for _validate_anneal_block.
    """
    block = {"hold_epochs": 2, "anneal_epochs": 10, "shape": "linear"}
    block.update(overrides)
    return block

  def test_a_valid_block_is_accepted_and_copied(self):
    """The baseline must pass, or every refusal below proves nothing."""
    resolved = training._validate_anneal_block(
      self._block(), "loss.berhu")
    self.assertEqual(resolved, self._block())
    self.assertIsNot(resolved, self._block())

  def test_const_shape_is_refused_because_the_feature_never_turns_on(self):
    """`const` holds the ramp at 0, so the annealed feature stays off."""
    with self.assertRaisesRegex(ValueError, "never turns on"):
      training._validate_anneal_block(
        self._block(shape="const"), "loss.berhu")

  def test_zero_anneal_epochs_is_refused(self):
    """A zero-epoch ramp is not a ramp; the block must say >= 1."""
    with self.assertRaisesRegex(ValueError, "must be >= 1"):
      training._validate_anneal_block(
        self._block(anneal_epochs=0), "loss.berhu")

  def test_negative_hold_epochs_is_refused(self):
    """A negative hold has no meaning in an epoch-indexed schedule."""
    with self.assertRaisesRegex(ValueError, "must be >= 0"):
      training._validate_anneal_block(
        self._block(hold_epochs=-1), "loss.berhu")

  def test_a_boolean_epoch_count_is_refused(self):
    """bool is an int subclass in Python, but True is never an epoch."""
    with self.assertRaisesRegex(ValueError, "must be an integer"):
      training._validate_anneal_block(
        self._block(anneal_epochs=True), "loss.berhu")

  def test_an_unknown_shape_is_refused(self):
    """A misspelled shape must stop the run, not fall through to linear."""
    with self.assertRaisesRegex(ValueError, "unknown"):
      training._validate_anneal_block(
        self._block(shape="cosinus"), "loss.berhu")

  def test_every_accepted_shape_reaches_the_schedule(self):
    """A shape the validator allows must be one anneal_value implements.

    A shape accepted here but unknown to anneal_value would silently take
    the linear branch, so the run would anneal on a schedule the YAML did
    not ask for.
    """
    from emulator.losses.core import anneal_value
    for shape in training._ANNEAL_SHAPES:
      if shape == "const":
        continue
      opts = {"start": 0.0, "end": 1.0, "hold_epochs": 0,
              "anneal_epochs": 4, "shape": shape}
      early = anneal_value(epoch=1, opts=opts)
      late = anneal_value(epoch=4, opts=opts)
      self.assertLessEqual(early, late, shape)
      self.assertAlmostEqual(late, 1.0, msg=shape)


class BerhuKnotRefusalTests(unittest.TestCase):
  """Require validate_berhu to refuse a knot that cannot separate."""

  def test_knot_equal_to_cap_is_refused(self):
    """knot == cap leaves the berhu transition zero-width."""
    with self.assertRaisesRegex(ValueError, "needs knot < cap"):
      training.validate_berhu({"knot": 2.0, "cap": 2.0}, "berhu", "loss")

  def test_knot_above_cap_is_refused(self):
    """An inverted pair is a config error, not a silent reordering."""
    with self.assertRaisesRegex(ValueError, "needs knot < cap"):
      training.validate_berhu({"knot": 5.0, "cap": 2.0}, "berhu", "loss")

  def test_a_berhu_block_on_a_non_berhu_mode_is_refused(self):
    """The knots would be built and never read: a silent no-op."""
    with self.assertRaisesRegex(ValueError, "not a berhu mode"):
      training.validate_berhu({"knot": 1.0, "cap": 2.0}, "chi2", "loss")

  def test_a_valid_pair_resolves_with_no_anneal(self):
    """The accepting path must still work, and report no anneal."""
    resolved = training.validate_berhu(
      {"knot": 1.0, "cap": 4.0}, "berhu", "loss")
    self.assertEqual(resolved["knot"], 1.0)
    self.assertEqual(resolved["cap"], 4.0)
    self.assertIsNone(resolved["anneal"])


if __name__ == "__main__":
  unittest.main()
