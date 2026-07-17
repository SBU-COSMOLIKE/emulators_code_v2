"""CPU evidence for the transformer token-width contract."""

import types
import unittest
from unittest import mock

import torch

from emulator.designs.blocks import TRFBlock
from emulator.designs.ia import TemplateResTRF
from emulator.designs.plain import ResTRF


def _geometry(bin_sizes, *, template=False):
  """Return one complete rectangular layout for a constructor witness."""
  sizes = list(bin_sizes)
  width = max(sizes)
  positions = []
  valid = torch.zeros((len(sizes), width), dtype=torch.bool)
  for bin_index, size in enumerate(sizes):
    for coordinate_index in range(size):
      positions.append(bin_index * width + coordinate_index)
      valid[bin_index, coordinate_index] = True
  values = {
    "bin_sizes": sizes,
    "head_pad_idx": torch.tensor(positions, dtype=torch.long),
    "head_valid_mask": valid,
  }
  if template:
    output_size = sum(sizes)
    values["evecs"] = torch.eye(output_size)
    values["sqrt_ev"] = torch.ones(output_size)
  return types.SimpleNamespace(**values)


def _fill_with_deterministic_nonzero_values(module):
  """Give every parameter a reproducible nonzero value."""
  with torch.no_grad():
    for parameter in module.parameters():
      values = torch.arange(
        1,
        parameter.numel() + 1,
        dtype=parameter.dtype,
        device=parameter.device)
      values = values.reshape(parameter.shape)
      parameter.copy_(0.03 * values + 0.01)


class TransformerTokenWidthTest(unittest.TestCase):
  """A width-one token is refused before either model allocates layers."""

  def test_plain_width_one_refuses_before_layer_construction(self):
    geometry = _geometry([4])

    with mock.patch(
        "emulator.designs.plain.nn.Linear",
        side_effect=AssertionError("learnable layer construction was reached")):
      with self.assertRaisesRegex(
          ValueError,
          r"output length is 4.*token count is 4.*token width is 1.*LayerNorm"):
        ResTRF(
          input_dim=3,
          output_dim=4,
          int_dim_res=4,
          geom=geometry,
          n_heads=1,
          n_blocks=1,
          n_blocks_trf=1,
          n_mlp_blocks=1,
          n_tokens=4,
          film=False)

  def test_factored_width_one_refuses_before_layer_construction(self):
    geometry = _geometry([1, 1, 1, 1], template=True)

    with mock.patch(
        "emulator.designs.ia.nn.Linear",
        side_effect=AssertionError("learnable layer construction was reached")):
      with self.assertRaisesRegex(
          ValueError,
          r"output length is 8.*token count is 8.*token width is 1.*LayerNorm"):
        TemplateResTRF(
          input_dim=3,
          output_dim=4,
          n_amps=1,
          n_templates=2,
          int_dim_res=4,
          geom=geometry,
          n_heads=1,
          n_blocks=1,
          n_blocks_trf=1,
          n_mlp_blocks=1,
          film=False)

  def test_block_guard_alone_fires_after_factored_trunk_allocation(self):
    """Show why TemplateResTRF also needs its own early validation call."""
    geometry = _geometry([1, 1, 1, 1], template=True)
    original_linear = torch.nn.Linear
    constructed_layers = []

    def counted_linear(*args, **kwargs):
      constructed_layers.append((args, kwargs))
      return original_linear(*args, **kwargs)

    # This mutation removes only TemplateResTRF's early call. The shared
    # TRFBlock guard still raises, but only after the trunk has allocated its
    # Linear layers. The count makes the ordering defect observable.
    with mock.patch(
        "emulator.designs.ia.validate_trf_token_width",
        return_value=None):
      with mock.patch(
          "emulator.designs.ia.nn.Linear",
          side_effect=counted_linear):
        with self.assertRaisesRegex(
            ValueError,
            r"maximum token width is 1"):
          TemplateResTRF(
            input_dim=3,
            output_dim=4,
            n_amps=1,
            n_templates=2,
            int_dim_res=4,
            geom=geometry,
            n_heads=1,
            n_blocks=1,
            n_blocks_trf=1,
            n_mlp_blocks=1,
            film=False)

    self.assertGreater(len(constructed_layers), 0)

  def test_old_width_one_block_is_input_independent(self):
    """The old divisibility-only rule builds a permanently demoted head."""
    with mock.patch(
        "emulator.designs.blocks.validate_trf_token_width",
        return_value=None):
      block = TRFBlock(
        dim=1,
        n_tokens=4,
        n_heads=1,
        n_mlp_blocks=1,
        output_length=4).double()
    _fill_with_deterministic_nonzero_values(block)

    first = torch.tensor(
      [[[0.0], [1.0], [2.0], [3.0]]],
      dtype=torch.float64,
      requires_grad=True)
    second = torch.tensor(
      [[[8.0], [-3.0], [0.5], [20.0]]],
      dtype=torch.float64)
    first_correction = block(first) - first
    second_correction = block(second) - second

    torch.testing.assert_close(
      first_correction,
      second_correction,
      rtol=0.0,
      atol=1.0e-12)
    correction_gradient = torch.autograd.grad(
      first_correction.sum(),
      first)[0]
    torch.testing.assert_close(
      correction_gradient,
      torch.zeros_like(correction_gradient),
      rtol=0.0,
      atol=0.0)

  def test_adjacent_width_two_models_remain_valid(self):
    """The closest plain, factored and direct-block controls still build."""
    plain_geometry = _geometry([4])
    plain_model = ResTRF(
      input_dim=3,
      output_dim=4,
      int_dim_res=4,
      geom=plain_geometry,
      n_heads=1,
      n_blocks=1,
      n_blocks_trf=1,
      n_mlp_blocks=1,
      n_tokens=3,
      film=False)
    self.assertEqual(plain_model.max_bin, 2)

    factored_geometry = _geometry([2, 2], template=True)
    factored_model = TemplateResTRF(
      input_dim=3,
      output_dim=4,
      n_amps=1,
      n_templates=2,
      int_dim_res=4,
      geom=factored_geometry,
      n_heads=1,
      n_blocks=1,
      n_blocks_trf=1,
      n_mlp_blocks=1,
      film=False)
    self.assertEqual(factored_model.max_bin, 2)

    block = TRFBlock(
      dim=2,
      n_tokens=3,
      n_heads=1,
      n_mlp_blocks=1,
      output_length=4).double()
    _fill_with_deterministic_nonzero_values(block)
    first = torch.tensor(
      [[[0.0, 1.0], [2.0, 4.0], [5.0, 8.0]]],
      dtype=torch.float64)
    second = torch.tensor(
      [[[1.0, 0.0], [4.0, 2.0], [8.0, 5.0]]],
      dtype=torch.float64)
    correction_difference = torch.max(torch.abs(
      (block(first) - first) - (block(second) - second))).item()
    self.assertGreater(correction_difference, 1.0e-6)


if __name__ == "__main__":
  unittest.main()
