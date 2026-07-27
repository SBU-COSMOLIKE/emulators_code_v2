"""CPU tests for layout and row-identity refusals no other test stood behind.

Mutation testing found that each check below could be deleted with all 821
tests still green, and no file under ``ai/gates`` names the functions that own
them. Every one guards a case where training would proceed on data whose row
or slot identity is not what the geometry claims: a selected row outside the
dump, the same row counted twice, a padded rectangle with no physical bins, a
persisted mask holding a value that is neither zero nor one, and a failure
flag that is blank rather than 0 or 1.

Each test names the exact malformed input a caller could really produce, and
each is paired with a baseline that must pass, so a refusal that fires for the
wrong reason cannot masquerade as coverage.
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from emulator import data_staging
from emulator.designs.blocks import resolve_padded_head_layout
from emulator.experiment import EmulatorExperiment


class PaddedHeadLayoutRefusalTests(unittest.TestCase):
  """Require resolve_padded_head_layout to refuse an unusable rectangle."""

  def _geometry(self, valid_mask, bin_sizes, pad_idx):
    """Return a stand-in geometry carrying only the three read fields.

    Arguments:
      valid_mask = the head_valid_mask tensor.
      bin_sizes  = the nonempty per-bin physical counts.
      pad_idx    = the flat slot of each physical value.

    Returns:
      An object exposing bin_sizes, head_pad_idx, and head_valid_mask.
    """
    class Geometry:
      pass
    geometry = Geometry()
    geometry.bin_sizes = bin_sizes
    geometry.head_pad_idx = pad_idx
    geometry.head_valid_mask = valid_mask
    return geometry

  def _valid_geometry(self):
    """Two bins of three and two values inside a 2 x 3 rectangle."""
    mask = torch.tensor([[True, True, True],
                         [True, True, False]])
    return self._geometry(
      valid_mask=mask,
      bin_sizes=[3, 2],
      pad_idx=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long))

  def test_a_consistent_layout_is_accepted(self):
    """The baseline must pass, or the refusals below prove nothing."""
    row_sizes, pad_idx, valid = resolve_padded_head_layout(
      geom=self._valid_geometry(), output_dim=5, where="model")
    self.assertEqual(row_sizes, [3, 2])
    self.assertEqual(int(pad_idx.numel()), 5)
    self.assertEqual(tuple(valid.shape), (1, 2, 3))

  def test_a_rectangle_with_no_bins_is_refused(self):
    """A mask with zero rows describes no physical layout at all."""
    geometry = self._geometry(
      valid_mask=torch.zeros((0, 3), dtype=torch.bool),
      bin_sizes=[3],
      pad_idx=torch.tensor([0, 1, 2], dtype=torch.long))
    with self.assertRaisesRegex(ValueError, "at least one physical bin"):
      resolve_padded_head_layout(
        geom=geometry, output_dim=3, where="model")

  def test_a_persisted_mask_value_other_than_zero_or_one_is_refused(self):
    """A saved uint8 mask must not silently coerce 2 to True.

    The rectangle below is the valid one with a single 2 written where a 1
    belongs, so a check that only asks "is this nonnegative" accepts it and
    the layout looks correct afterward. Only an exact 0/1 test refuses it.
    """
    mask = torch.tensor([[1, 1, 2],
                         [1, 1, 0]], dtype=torch.uint8)
    geometry = self._geometry(
      valid_mask=mask,
      bin_sizes=[3, 2],
      pad_idx=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long))
    with self.assertRaisesRegex(ValueError, "uint8 values must be 0 or 1"):
      resolve_padded_head_layout(
        geom=geometry, output_dim=5, where="model")

  def test_a_persisted_mask_of_zeros_and_ones_is_accepted(self):
    """The same rectangle written as legal uint8 must still pass."""
    mask = torch.tensor([[1, 1, 1],
                         [1, 1, 0]], dtype=torch.uint8)
    geometry = self._geometry(
      valid_mask=mask,
      bin_sizes=[3, 2],
      pad_idx=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long))
    row_sizes, _pad_idx, _valid = resolve_padded_head_layout(
      geom=geometry, output_dim=5, where="model")
    self.assertEqual(row_sizes, [3, 2])


class Grid2DRowIdentityRefusalTests(unittest.TestCase):
  """Require the Grid2D row mapping to refuse a selection it cannot trust.

  ``_grid2d_row_mapping`` reads only its ``src`` argument, so the tests call
  it directly with no experiment instance rather than building a full run.
  """

  def _source(self, dump_rows, source_n_rows, idx=None):
    """Return one compact resident source with the named row selection.

    Arguments:
      dump_rows     = the selected disk rows.
      source_n_rows = the original dump's row count.
      idx           = the seeded order, or None for the identity.

    Returns:
      A src mapping shaped the way load_source hands one over.
    """
    dump_rows = np.asarray(dump_rows, dtype="int64")
    kept = int(dump_rows.size)
    if idx is None:
      idx = np.arange(kept, dtype="int64")
    return {
      "C": np.zeros((kept, 2), dtype="float64"),
      "dv": np.zeros((kept, 4), dtype="float32"),
      "idx": np.asarray(idx, dtype="int64"),
      "dump_rows": dump_rows,
      "source_n_rows": source_n_rows,
    }

  def test_a_well_formed_selection_is_accepted(self):
    """The baseline must pass, or the refusals below prove nothing."""
    rows, dump_rows, read_rows, output_idx = (
      EmulatorExperiment._grid2d_row_mapping(
        None, self._source(dump_rows=[0, 2, 4], source_n_rows=5)))
    self.assertEqual(rows, 5)
    self.assertEqual(list(dump_rows), [0, 2, 4])
    self.assertEqual(list(read_rows), [0, 1, 2])
    self.assertEqual(list(output_idx), [0, 1, 2])

  def test_a_row_past_the_end_of_the_source_is_refused(self):
    """Row 5 does not exist in a five-row dump; rows are 0 through 4."""
    with self.assertRaisesRegex(ValueError, "must stay inside original row"):
      EmulatorExperiment._grid2d_row_mapping(
        None, self._source(dump_rows=[0, 2, 5], source_n_rows=5))

  def test_a_repeated_row_is_refused(self):
    """One disk row counted twice would weight that cosmology twice."""
    with self.assertRaisesRegex(ValueError, "strictly increasing"):
      EmulatorExperiment._grid2d_row_mapping(
        None, self._source(dump_rows=[0, 2, 2], source_n_rows=5))


class FailureMaskTokenRefusalTests(unittest.TestCase):
  """Require the failure mask to refuse a flag that is not 0 or 1."""

  def test_a_blank_line_is_not_a_flag(self):
    """A blank line states nothing; reading it as success invents data."""
    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, "blank.txt")
      with open(path, "w", encoding="ascii") as handle:
        handle.write("0\n\n")
      with self.assertRaisesRegex(ValueError, "literal '0' or '1'"):
        data_staging._load_failure_mask(path=path, expected_rows=2)

  def test_two_real_flags_are_accepted(self):
    """The same shape written with real flags must still pass."""
    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, "flags.txt")
      with open(path, "w", encoding="ascii") as handle:
        handle.write("0\n1\n")
      mask = data_staging._load_failure_mask(path=path, expected_rows=2)
      self.assertEqual(list(mask), [False, True])


if __name__ == "__main__":
  unittest.main()
