"""CPU tests for the packed-target batch-memory calculation."""

import unittest

import numpy as np
import torch

from emulator import batching


class _TwoInputSevenOutputModel(torch.nn.Module):
  """Small deterministic shape fixture with one registered parameter."""

  def __init__(self):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.zeros(2, 7))

  def forward(
      self,
      inputs):
    return inputs @ self.weight


class _IdentityParameterGeometry:
  """Return the float32 parameter tensor supplied by the loader."""

  def encode(
      self,
      parameters):
    return parameters


class _OrdinaryTarget:
  """Seven-value target fixture matching the model output width."""

  def __init__(self):
    self.total_size = 7
    self.dest_idx = torch.arange(7)
    self.needs_params = False

  def encode(
      self,
      data_vectors):
    return data_vectors


class _PackedTarget(_OrdinaryTarget):
  """Fourteen-value target fixture storing a base beside the truth."""

  target_dim = 14

  def encode(
      self,
      data_vectors):
    return torch.cat(
      [data_vectors, data_vectors],
      dim=1)


class PackedTargetBatchSizingTest(unittest.TestCase):
  """Exercise the arithmetic without a GPU or a scientific data file."""

  batch_size = 3
  output_width = 7
  packed_target_width = 14

  def setUp(self):
    self.model = _TwoInputSevenOutputModel()
    self.resident_bytes = (
      batching.compute_model_size_bytes(self.model)
      + 1 * 1 * 8)

  def _terms(
      self,
      target_width):
    return batching.compute_batch_byte_terms(
      model=self.model,
      bs=self.batch_size,
      sample_dims=(2,),
      dv_len=1,
      target_dim=target_width,
      target_dtype=torch.float32)

  def _budget_for_available_bytes(
      self,
      available_bytes):
    """Invert the planner's exact four-fifths allowance."""
    self.assertEqual(available_bytes % 4, 0)
    return available_bytes * 5 // 4

  def test_packed_target_adds_exactly_84_bytes(self):
    ordinary_terms = self._terms(self.output_width)
    packed_terms = self._terms(self.packed_target_width)

    additional_target_bytes = (
      self.batch_size
      * (self.packed_target_width - self.output_width)
      * torch.empty((), dtype=torch.float32).element_size())

    self.assertEqual(additional_target_bytes, 84)
    self.assertEqual(
      packed_terms["target"] - ordinary_terms["target"],
      additional_target_bytes)

  def test_ordinary_target_preserves_the_original_formula(self):
    default_terms = batching.compute_batch_byte_terms(
      model=self.model,
      bs=self.batch_size,
      sample_dims=(2,),
      dv_len=1)
    explicit_terms = self._terms(self.output_width)

    self.assertEqual(default_terms, explicit_terms)
    original_total = (
      default_terms["saved_activations"]
      + default_terms["input"]
      + 2 * default_terms["model_output"]
      + default_terms["chi2_scratch"])
    self.assertEqual(sum(default_terms.values()), original_total)
    self.assertEqual(
      batching.compute_batch_size_bytes(
        model=self.model,
        bs=self.batch_size,
        sample_dims=(2,),
        dv_len=1),
      original_total)

  def test_target_dtype_controls_the_target_term(self):
    float32_terms = self._terms(self.packed_target_width)
    float64_terms = batching.compute_batch_byte_terms(
      model=self.model,
      bs=self.batch_size,
      sample_dims=(2,),
      dv_len=1,
      target_dim=self.packed_target_width,
      target_dtype=torch.float64)

    self.assertEqual(
      float64_terms["target"],
      2 * float32_terms["target"])

  def test_corrected_width_selects_one_batch_where_old_formula_selects_two(self):
    ordinary_per_batch = sum(self._terms(self.output_width).values())
    packed_per_batch = sum(self._terms(self.packed_target_width).values())

    # This allowance has room for exactly two batches under the old
    # output-shaped-target formula. The packed target needs 84 more
    # bytes per batch, so only one corrected batch fits.
    available_bytes = self.resident_bytes + 2 * ordinary_per_batch
    budget = self._budget_for_available_bytes(available_bytes)

    old_formula_count = (
      available_bytes - self.resident_bytes) // ordinary_per_batch
    corrected_count = batching.batches_per_load(
      model=self.model,
      bs=self.batch_size,
      sample_shape=(2,),
      budget=budget,
      dv_len=1,
      target_dim=self.packed_target_width,
      target_dtype=torch.float32)

    self.assertEqual(old_formula_count, 2)
    self.assertEqual(corrected_count, 1)
    self.assertEqual(
      packed_per_batch - ordinary_per_batch,
      84)

  def test_less_than_one_complete_batch_is_refused(self):
    packed_terms = self._terms(self.packed_target_width)
    packed_per_batch = sum(packed_terms.values())
    required_bytes = self.resident_bytes + packed_per_batch
    available_bytes = required_bytes - 4
    budget = self._budget_for_available_bytes(available_bytes)

    legacy_forced_count = max(
      1,
      int((available_bytes - self.resident_bytes) // packed_per_batch))
    self.assertEqual(legacy_forced_count, 1)

    with self.assertRaises(MemoryError) as caught:
      batching.batches_per_load(
        model=self.model,
        bs=self.batch_size,
        sample_shape=(2,),
        budget=budget,
        dv_len=1,
        target_dim=self.packed_target_width,
        target_dtype=torch.float32)

    message = str(caught.exception)
    self.assertIn("required=" + str(required_bytes), message)
    self.assertIn("available=" + str(available_bytes), message)
    self.assertIn("resident=" + str(self.resident_bytes), message)
    for name, value in packed_terms.items():
      self.assertIn(name + "=" + str(value), message)

  def test_streaming_loader_threads_the_staged_width_and_dtype(self):
    row_count = 100
    parameters = np.zeros((row_count, 2), dtype="float32")
    data_vectors = np.zeros((row_count, 7), dtype="float32")
    rows = np.arange(row_count)

    ordinary = batching._build_loaders_one(
      device=torch.device("cpu"),
      C=parameters,
      dv=data_vectors,
      idx=rows,
      param_geometry=_IdentityParameterGeometry(),
      chi2fn=_OrdinaryTarget(),
      model=self.model,
      bs=self.batch_size,
      budget=5000,
      dv_len=7)
    packed = batching._build_loaders_one(
      device=torch.device("cpu"),
      C=parameters,
      dv=data_vectors,
      idx=rows,
      param_geometry=_IdentityParameterGeometry(),
      chi2fn=_PackedTarget(),
      model=self.model,
      bs=self.batch_size,
      budget=5000,
      dv_len=7)

    ordinary_load_dv = ordinary[1]
    packed_load_dv = packed[1]
    probe_rows = np.array([0, 1, 2])
    ordinary_target = ordinary_load_dv(probe_rows)
    packed_target = packed_load_dv(probe_rows)

    self.assertEqual(tuple(ordinary_target.shape), (3, 7))
    self.assertEqual(tuple(packed_target.shape), (3, 14))
    self.assertEqual(ordinary_target.dtype, torch.float32)
    self.assertEqual(packed_target.dtype, torch.float32)
    self.assertLessEqual(packed[2], ordinary[2])


if __name__ == "__main__":
  unittest.main()
