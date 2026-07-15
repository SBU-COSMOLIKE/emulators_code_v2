"""Focused tests for the generator's pre-mutation run-control contract."""

import unittest

from compute_data_vectors.dataset_manifest import scope_dataset_stem
from compute_data_vectors.dataset_manifest import validate_run_control


class GeneratorRunControlTests(unittest.TestCase):

  def test_three_legal_operations_are_exhaustive(self):
    expected = {
      (0, 0): "fresh",
      (1, 0): "resume",
      (1, 1): "append",
    }
    observed = {}
    for loadchk in (0, 1):
      for append in (0, 1):
        if (loadchk, append) == (0, 1):
          with self.assertRaisesRegex(ValueError, "append extends"):
            validate_run_control(loadchk, append, 0)
          continue
        control = validate_run_control(loadchk, append, 0)
        observed[(control.loadchk, control.append)] = control.operation
    self.assertEqual(observed, expected)

  def test_none_defaults_match_fresh_full_generation(self):
    control = validate_run_control(None, None, None)
    self.assertEqual(control.loadchk, 0)
    self.assertEqual(control.append, 0)
    self.assertEqual(control.chain, 0)
    self.assertEqual(control.operation, "fresh")
    self.assertEqual(control.dataset_mode, "full")

  def test_chain_mode_is_explicit_and_independent_of_operation(self):
    for loadchk, append, operation in ((0, 0, "fresh"),
                                       (1, 0, "resume"),
                                       (1, 1, "append")):
      control = validate_run_control(loadchk, append, 1)
      self.assertEqual(control.operation, operation)
      self.assertEqual(control.dataset_mode, "chain-only")

  def test_only_native_binary_integers_are_accepted(self):
    for name, values in {
        "loadchk": (False, True, -1, 2, 1.0, "1"),
        "append": (False, True, -1, 2, 1.0, "1"),
        "chain": (False, True, -1, 2, 1.0, "1"),
    }.items():
      for value in values:
        kwargs = {"loadchk": 0, "append": 0, "chain": 0}
        kwargs[name] = value
        with self.subTest(name=name, value=value):
          with self.assertRaisesRegex(ValueError, "--" + name):
            validate_run_control(**kwargs)

  def test_normalized_record_is_immutable(self):
    control = validate_run_control(0, 0, 0)
    with self.assertRaises(Exception):
      control.append = 1

  def test_dataset_stem_scope_separates_chain_only_from_full_outputs(self):
    stem = "/tmp/example/chains/params_probe"
    self.assertEqual(scope_dataset_stem(stem, "full"), stem)
    self.assertEqual(
      scope_dataset_stem(stem, "chain-only"),
      stem + "_chain_only")
    self.assertNotEqual(
      scope_dataset_stem(stem, "full"),
      scope_dataset_stem(stem, "chain-only"))

  def test_dataset_stem_scope_rejects_unnormalized_inputs(self):
    for stem in (None, "", 7):
      with self.subTest(stem=stem):
        with self.assertRaisesRegex(ValueError, "nonempty string"):
          scope_dataset_stem(stem, "full")
    for mode in (None, "chain", "FULL", 1):
      with self.subTest(mode=mode):
        with self.assertRaisesRegex(ValueError, "normalized generator"):
          scope_dataset_stem("params", mode)


if __name__ == "__main__":
  unittest.main()
