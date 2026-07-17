"""CPU checks that padded head positions never become scientific values.

CNN and transformer heads place unequal physical bins in rectangular tensors.
The extra tensor positions are storage only: they have no angular coordinate
and must stay zero after every head block.  These tests use tiny known-answer
routes where a physical output can change only if an artificial position is
allowed to carry information.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from emulator.designs.blocks import TRFBlock
from emulator.designs.ia import TemplateResCNN, TemplateResTRF
from emulator.designs.plain import ResCNN, ResTRF
from emulator.geometries.output import (
  DataVectorGeometry,
  build_shear_angle_map,
)


CPU = torch.device("cpu")


def _identity_activation(_width):
  """Return an activation that leaves a hand-written routing witness exact."""
  return nn.Identity()


class _AddOne(nn.Module):
  """Expose an artificial slot by mapping an exact zero to one."""

  def forward(self, values):
    return values + 1.0


def _plain_geometry(bin_sizes):
  """Return a diagonal geometry with the requested physical-bin lengths."""
  sizes = list(bin_sizes)
  width = max(sizes)
  positions = []
  valid = torch.zeros((len(sizes), width), dtype=torch.bool)
  for bin_index, size in enumerate(sizes):
    for coordinate_index in range(size):
      positions.append(bin_index * width + coordinate_index)
      valid[bin_index, coordinate_index] = True
  return types.SimpleNamespace(
    bin_sizes=sizes,
    head_pad_idx=torch.tensor(positions, dtype=torch.long),
    head_valid_mask=valid)


def _template_geometry(bin_sizes):
  """Return an identity-basis template geometry for tiny CPU models."""
  output_size = sum(bin_sizes)
  geometry = _plain_geometry(bin_sizes)
  geometry.evecs = torch.eye(output_size)
  geometry.sqrt_ev = torch.ones(output_size)
  return geometry


class _ConstantTrunk(nn.Module):
  """Return one fixed physical vector for every input row."""

  def __init__(self, values):
    super().__init__()
    self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

  def forward(self, inputs):
    return self.values.unsqueeze(0).expand(inputs.shape[0], -1)


class _WriteArtificialPosition(nn.Module):
  """Copy one real value into one artificial token coordinate."""

  def __init__(self, *, source_token, source_coordinate,
               target_token, artificial_coordinate):
    super().__init__()
    self.source_token = source_token
    self.source_coordinate = source_coordinate
    self.target_token = target_token
    self.artificial_coordinate = artificial_coordinate

  def forward(self, values, *args, **kwargs):
    result = values.clone()
    result[:, self.target_token, self.artificial_coordinate] = (
      values[:, self.source_token, self.source_coordinate])
    return result


class _RouteArtificialPosition(nn.Module):
  """Move an artificial token value into a real output coordinate."""

  def __init__(self, *, token, artificial_coordinate, physical_coordinate):
    super().__init__()
    self.token = token
    self.artificial_coordinate = artificial_coordinate
    self.physical_coordinate = physical_coordinate

  def forward(self, values, *args, **kwargs):
    result = values.clone()
    result[:, self.token, self.physical_coordinate] += (
      values[:, self.token, self.artificial_coordinate])
    return result


def _install_two_block_cnn_route(model):
  """Route a value through the short bin's nonexistent second position."""
  with torch.no_grad():
    for convolution in model.convs:
      convolution.weight.zero_()
      convolution.bias.zero_()
    # Block 1: long-bin position 1 -> short-bin position 1.  The target is
    # artificial because the second bin has length one.
    model.convs[0].weight[1, 0, 1] = 1.0
    # Block 2: short-bin artificial position 1 -> its real position 0.
    model.convs[1].weight[1, 1, 2] = 1.0
    model.gate.fill_(1.0)


def _dense_cosmic_geometry():
  """Return a four-kept-value cosmic-shear geometry with identity whitening."""
  return DataVectorGeometry(
    device=CPU,
    total_size=6,
    dest_idx=torch.tensor([0, 2, 4, 5], dtype=torch.long),
    evecs=torch.eye(4),
    sqrt_ev=torch.ones(4),
    Cinv=torch.eye(6),
    center=torch.zeros(4),
    section_sizes=[6, 0, 0],
    probe="xi",
  )


def _cosmic_geometry(*, total_size, kept_positions):
  """Return an identity-whitened geometry for a chosen xi mask."""
  keep = list(kept_positions)
  kept_size = len(keep)
  return DataVectorGeometry(
    device=CPU,
    total_size=total_size,
    dest_idx=torch.tensor(keep, dtype=torch.long),
    evecs=torch.eye(kept_size),
    sqrt_ev=torch.ones(kept_size),
    Cinv=torch.eye(total_size),
    center=torch.zeros(kept_size),
    section_sizes=[total_size, 0, 0],
    probe="xi",
  )


def _geometry_with_explicit_coordinate_map(*, template):
  """Describe equal-count bins whose surviving angular slots differ."""
  # Bin 0 keeps angular slots 0 and 2. Bin 1 keeps slots 1 and 2. Both bins
  # contain two values, so bin_sizes alone cannot recover this distinction.
  geometry = (_template_geometry([2, 2]) if template
              else _plain_geometry([2, 2]))
  geometry.head_pad_idx = torch.tensor([0, 2, 4, 5], dtype=torch.long)
  geometry.head_valid_mask = torch.tensor(
    [[True, False, True], [False, True, True]], dtype=torch.bool)
  return geometry


def _geometry_with_empty_middle_row():
  """Describe two full physical rows separated by one fully masked row."""
  return types.SimpleNamespace(
    bin_sizes=[2, 2],
    head_pad_idx=torch.tensor([0, 1, 4, 5], dtype=torch.long),
    head_valid_mask=torch.tensor(
      [[True, True], [False, False], [True, True]], dtype=torch.bool),
  )


def _give_trf_block_live_branches(block):
  """Give a tiny real transformer nonzero attention and MLP branches."""
  matrices = (
    torch.tensor([[0.4, -0.2], [0.1, 0.3]]),
    torch.tensor([[0.2, 0.1], [-0.3, 0.5]]),
    torch.tensor([[0.3, 0.2], [0.4, -0.1]]),
    torch.tensor([[0.5, -0.1], [0.2, 0.4]]),
  )
  with torch.no_grad():
    for layer, matrix in zip(
        (block.wq, block.wk, block.wv, block.wo), matrices):
      layer.weight.copy_(matrix)
      layer.bias.copy_(torch.tensor([0.03, -0.02]))
    block.mlp_lins[0].weight.copy_(
      torch.tensor([[0.15, 0.25], [-0.05, 0.35]]))
    block.mlp_lins[0].bias.copy_(torch.tensor([0.01, -0.04]))


class PaddedCNNIdentityTests(unittest.TestCase):
  """Expose a two-convolution path that exists only through padding."""

  def test_plain_cnn_cannot_route_through_the_short_bins_padding(self):
    """A nonexistent short-bin position must not change its real value."""
    model = ResCNN(
      input_dim=2,
      output_dim=3,
      int_dim_res=4,
      geom=_plain_geometry([2, 1]),
      kernel_size=3,
      n_blocks=0,
      n_blocks_cnn=2,
      gate_init=1.0,
      head_act=_identity_activation,
    )
    expected = torch.tensor([[0.0, 1.0, 0.0]])
    model.mlp = _ConstantTrunk(expected[0])
    _install_two_block_cnn_route(model)

    actual = model(torch.zeros(1, 2))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

  def test_template_cnn_cannot_route_through_the_short_bins_padding(self):
    """Template channels obey the same inert-padding rule as plain channels."""
    model = TemplateResCNN(
      input_dim=2,
      output_dim=3,
      n_amps=1,
      n_templates=1,
      int_dim_res=4,
      geom=_template_geometry([2, 1]),
      kernel_size=3,
      n_blocks=0,
      n_blocks_cnn=2,
      gate_init=1.0,
      head_act=_identity_activation,
    )
    expected = torch.tensor([[[0.0, 1.0, 0.0]]])
    model.model = _ConstantTrunk(expected.reshape(-1))
    _install_two_block_cnn_route(model)

    actual = model(torch.zeros(1, 2))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

  def test_nonzero_activation_cannot_repopulate_cnn_padding(self):
    """A zero-to-one activation is masked before the following block."""
    cases = (
      ("plain", ResCNN(
        input_dim=2,
        output_dim=3,
        int_dim_res=4,
        geom=_plain_geometry([2, 1]),
        kernel_size=3,
        n_blocks=0,
        n_blocks_cnn=2,
        gate_init=1.0,
        head_act=_identity_activation,
      ), "mlp", torch.zeros(3), torch.zeros(1, 3)),
      ("template", TemplateResCNN(
        input_dim=2,
        output_dim=3,
        n_amps=1,
        n_templates=1,
        int_dim_res=4,
        geom=_template_geometry([2, 1]),
        kernel_size=3,
        n_blocks=0,
        n_blocks_cnn=2,
        gate_init=1.0,
        head_act=_identity_activation,
      ), "model", torch.zeros(3), torch.zeros(1, 1, 3)),
    )
    for label, model, trunk_name, trunk_values, expected in cases:
      with self.subTest(design=label):
        setattr(model, trunk_name, _ConstantTrunk(trunk_values))
        model.acts[0] = _AddOne()
        with torch.no_grad():
          for convolution in model.convs:
            convolution.weight.zero_()
            convolution.bias.zero_()
          # The first convolution produces exact zeros. _AddOne then writes
          # one into every rectangular slot, including the nonexistent
          # second coordinate of the short row. The second convolution can
          # reach a physical output only through that artificial coordinate.
          model.convs[1].weight[1, 1, 2] = 1.0
          model.gate.fill_(1.0)

        actual = model(torch.zeros(1, 2))

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

  def test_film_beta_cannot_activate_an_invalid_short_bin_slot(self):
    """FiLM may shift a real slot, but its broadcast shift stops at padding."""
    model = ResCNN(
      input_dim=2,
      output_dim=3,
      int_dim_res=4,
      geom=_plain_geometry([2, 1]),
      kernel_size=3,
      film=True,
      n_blocks=0,
      n_blocks_cnn=2,
      gate_init=1.0,
      head_act=_identity_activation,
    )
    model.mlp = _ConstantTrunk(torch.zeros(3))
    with torch.no_grad():
      for convolution in model.convs:
        convolution.weight.zero_()
        convolution.bias.zero_()
      # The first FiLM beta broadcasts one into both positions of the short
      # bin. Position zero is physical; position one is artificial.
      model.film_gens[0].linear.weight.zero_()
      model.film_gens[0].linear.bias[3] = 1.0
      # A real beta value reaches long-bin output zero. A leaked artificial
      # beta would also reach the short bin's only real output.
      model.convs[1].weight[0, 1, 1] = 1.0
      model.convs[1].weight[1, 1, 2] = 1.0
      model.gate.fill_(1.0)

    actual = model(torch.zeros(1, 2))

    torch.testing.assert_close(
      actual, torch.tensor([[1.0, 0.0, 0.0]]), rtol=0.0, atol=0.0)

  def test_one_cnn_block_respects_each_rows_angular_slots(self):
    """A coordinate that exists in one row may be artificial in another."""
    model = ResCNN(
      input_dim=2,
      output_dim=4,
      int_dim_res=4,
      geom=_geometry_with_explicit_coordinate_map(template=False),
      kernel_size=3,
      n_blocks=0,
      n_blocks_cnn=1,
      gate_init=1.0,
      head_act=_identity_activation,
    )
    model.mlp = _ConstantTrunk(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    observed = []
    hook = model.acts[0].register_forward_hook(
      lambda _module, _inputs, output: observed.append(output.detach().clone()))
    with torch.no_grad():
      model.convs[0].weight.zero_()
      model.convs[0].bias.zero_()
      # The center tap proposes 5 at row 1, angular slot 0. That slot is
      # artificial for row 1 and must be removed. The left tap proposes 2 at
      # angular slot 1, which is physical for row 1 and must remain.
      model.convs[0].weight[1, 0, 1] = 5.0
      model.convs[0].weight[1, 0, 0] = 2.0
      model.gate.fill_(1.0)
    try:
      actual = model(torch.zeros(1, 2))
    finally:
      hook.remove()

    torch.testing.assert_close(
      observed[0][0, 1], torch.tensor([0.0, 2.0, 0.0]),
      rtol=0.0, atol=0.0)
    torch.testing.assert_close(
      actual, torch.tensor([[1.0, 0.0, 2.0, 0.0]]),
      rtol=0.0, atol=0.0)


class PaddedTransformerIdentityTests(unittest.TestCase):
  """Expose multi-block transformer paths through a partial token."""

  def test_plain_transformer_masks_the_final_partial_token_each_block(self):
    """Ragged n_tokens storage cannot feed the final physical coordinate."""
    model = ResTRF(
      input_dim=2,
      output_dim=5,
      int_dim_res=4,
      geom=_plain_geometry([5]),
      n_heads=1,
      n_tokens=3,
      n_blocks=0,
      n_blocks_trf=2,
      n_mlp_blocks=1,
      gate_init=1.0,
    )
    expected = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    model.mlp = _ConstantTrunk(expected[0])
    model.trf = nn.ModuleList((
      _WriteArtificialPosition(
        source_token=0,
        source_coordinate=1,
        target_token=2,
        artificial_coordinate=1,
      ),
      _RouteArtificialPosition(
        token=2,
        artificial_coordinate=1,
        physical_coordinate=0,
      ),
    ))
    with torch.no_grad():
      model.gate.fill_(1.0)

    actual = model(torch.zeros(1, 2))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

  def test_template_transformer_masks_each_templates_partial_token(self):
    """A template head cannot turn its short-bin padding into a correction."""
    model = TemplateResTRF(
      input_dim=2,
      output_dim=3,
      n_amps=1,
      n_templates=1,
      int_dim_res=4,
      geom=_template_geometry([2, 1]),
      n_heads=1,
      n_blocks=0,
      n_blocks_trf=2,
      n_mlp_blocks=1,
      gate_init=1.0,
    )
    expected = torch.tensor([[[0.0, 1.0, 0.0]]])
    model.model = _ConstantTrunk(expected.reshape(-1))
    model.trf = nn.ModuleList((
      _WriteArtificialPosition(
        source_token=0,
        source_coordinate=1,
        target_token=1,
        artificial_coordinate=1,
      ),
      _RouteArtificialPosition(
        token=1,
        artificial_coordinate=1,
        physical_coordinate=0,
      ),
    ))
    with torch.no_grad():
      model.gate.fill_(1.0)

    actual = model(torch.zeros(1, 2))

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

  def test_real_transformer_ignores_sentinels_in_invalid_features(self):
    """Changing only artificial feature values cannot change real outputs."""
    block = TRFBlock(
      dim=2, n_tokens=2, n_heads=1, n_mlp_blocks=1,
      act=_identity_activation, shared_mlp=True, output_length=3)
    _give_trf_block_live_branches(block)
    valid = torch.tensor(
      [[[True, True], [True, False]]], dtype=torch.bool)
    first = torch.tensor([[[0.6, -0.4], [0.8, 1.0e6]]])
    second = first.clone()
    second[0, 1, 1] = -1.0e6

    first_result = block(first, valid)
    second_result = block(second, valid)

    expanded_valid = valid.expand_as(first_result)
    torch.testing.assert_close(
      first_result[expanded_valid], second_result[expanded_valid],
      rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
      first_result[~expanded_valid], torch.zeros_like(first_result[~expanded_valid]),
      rtol=0.0, atol=0.0)

  def test_real_transformer_excludes_an_all_masked_key_without_nan(self):
    """An empty token cannot dilute attention or produce nonfinite values."""
    one_token = TRFBlock(
      dim=2, n_tokens=1, n_heads=1, n_mlp_blocks=1,
      act=_identity_activation, shared_mlp=True, output_length=2)
    two_tokens = TRFBlock(
      dim=2, n_tokens=2, n_heads=1, n_mlp_blocks=1,
      act=_identity_activation, shared_mlp=True, output_length=2)
    _give_trf_block_live_branches(one_token)
    two_tokens.load_state_dict(one_token.state_dict())
    physical = torch.tensor([[[0.7, -0.2]]])
    with_empty = torch.tensor([[[0.7, -0.2], [90.0, -70.0]]])
    one_valid = torch.tensor([[[True, True]]])
    two_valid = torch.tensor(
      [[[True, True], [False, False]]], dtype=torch.bool)

    one_result = one_token(physical, one_valid)
    two_result = two_tokens(with_empty, two_valid)

    torch.testing.assert_close(
      two_result[:, :1, :], one_result, rtol=1.0e-6, atol=1.0e-6)
    self.assertTrue(bool(torch.isfinite(two_result).all().item()))
    torch.testing.assert_close(
      two_result[:, 1, :], torch.zeros_like(two_result[:, 1, :]),
      rtol=0.0, atol=0.0)

  def test_nonzero_mlp_activation_is_masked_before_the_next_linear(self):
    """An MLP activation cannot turn an invalid feature into a route."""
    block = TRFBlock(
      dim=2,
      n_tokens=2,
      n_heads=1,
      n_mlp_blocks=2,
      act=_identity_activation,
      shared_mlp=True,
      output_length=3,
    )
    block.mlp_acts[0] = _AddOne()
    with torch.no_grad():
      for layer in block.mlp_lins:
        layer.weight.zero_()
        layer.bias.zero_()
      # The first linear produces exact zeros, then _AddOne populates every
      # feature. The second linear routes feature 1 into feature 0. Feature 1
      # is physical in token 0 but artificial in token 1, so only token 0 may
      # receive this correction.
      block.mlp_lins[1].weight[0, 1] = 1.0
    inputs = torch.zeros(1, 2, 2)
    valid = torch.tensor(
      [[[True, True], [True, False]]], dtype=torch.bool)

    actual = block(inputs, valid)

    torch.testing.assert_close(
      actual,
      torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
      rtol=0.0,
      atol=0.0,
    )


class PaddedCoordinateRecordTests(unittest.TestCase):
  """Check coordinate-aware layout creation, consumption, and persistence."""

  def test_real_heads_keep_a_fully_masked_middle_row_inert(self):
    """Empty middle storage stays zero without shifting the later row."""
    physical = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected_rectangle = torch.tensor(
      [[[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]])

    cnn = ResCNN(
      input_dim=2, output_dim=4, int_dim_res=4,
      geom=_geometry_with_empty_middle_row(), kernel_size=3,
      n_blocks=0, n_blocks_cnn=1, gate_init=1.0,
      head_act=_identity_activation)
    cnn.mlp = _ConstantTrunk(physical)
    cnn_rows = []
    cnn_hook = cnn.acts[0].register_forward_hook(
      lambda _module, _inputs, output:
      cnn_rows.append(output.detach().clone()))
    with torch.no_grad():
      cnn.convs[0].weight.zero_()
      cnn.convs[0].bias.zero_()
      for row in range(3):
        cnn.convs[0].weight[row, row, 1] = 1.0
      cnn.gate.fill_(1.0)
    try:
      cnn_result = cnn(torch.zeros(1, 2))
    finally:
      cnn_hook.remove()

    trf = ResTRF(
      input_dim=2, output_dim=4, int_dim_res=4,
      geom=_geometry_with_empty_middle_row(), n_heads=1,
      n_blocks=0, n_blocks_trf=1, n_mlp_blocks=1,
      gate_init=1.0)
    trf.mlp = _ConstantTrunk(physical)
    trf_rows = []
    trf_hook = trf.trf[0].register_forward_hook(
      lambda _module, _inputs, output:
      trf_rows.append(output.detach().clone()))
    try:
      trf_result = trf(torch.zeros(1, 2))
    finally:
      trf_hook.remove()

    for label, rows, result in (
        ("CNN", cnn_rows[0], cnn_result),
        ("transformer", trf_rows[0], trf_result)):
      with self.subTest(head=label):
        self.assertTrue(bool(torch.isfinite(rows).all().item()))
        self.assertTrue(bool(torch.isfinite(result).all().item()))
        torch.testing.assert_close(
          rows, expected_rectangle, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
          rows[:, 1, :], torch.zeros_like(rows[:, 1, :]),
          rtol=0.0, atol=0.0)
        torch.testing.assert_close(
          rows[:, 2, :], torch.tensor([[3.0, 4.0]]),
          rtol=0.0, atol=0.0)
    torch.testing.assert_close(
      cnn_result, torch.tensor([[2.0, 4.0, 6.0, 8.0]]),
      rtol=0.0, atol=0.0)
    torch.testing.assert_close(
      trf_result, physical.unsqueeze(0), rtol=0.0, atol=0.0)

  def test_shear_map_records_original_angular_slots_not_survivor_rank(self):
    """Equal kept counts with different masks retain different slot maps."""
    geometry = _dense_cosmic_geometry()
    fake_getdist = types.ModuleType("getdist")

    with tempfile.TemporaryDirectory(prefix="padded-head-map-") as directory:
      nz_path = os.path.join(directory, "source_nz.txt")
      np.savetxt(nz_path, np.asarray(((0.1, 1.0), (0.5, 0.2))))

      class _IniFile:
        def __init__(self, _path):
          pass

        def int(self, name):
          return {"n_theta": 3, "source_ntomo": 1}[name]

        def float(self, name):
          return {"theta_min_arcmin": 1.0,
                  "theta_max_arcmin": 8.0}[name]

        def relativeFileName(self, name):
          if name != "nz_source_file":
            raise AssertionError("unexpected dataset key " + name)
          return nz_path

      fake_getdist.IniFile = _IniFile
      with mock.patch.dict(sys.modules, {"getdist": fake_getdist}), \
          mock.patch.dict(os.environ, {"ROOTDIR": directory}):
        build_shear_angle_map(
          geometry, data_dir="fixture", dataset="fixture.dataset")

    self.assertEqual(geometry.bin_sizes, [2, 2])
    self.assertTrue(
      hasattr(geometry, "head_pad_idx"),
      "the shear map must retain each kept value's original angular slot",
    )
    self.assertTrue(
      hasattr(geometry, "head_valid_mask"),
      "the shear map must retain the artificial-position mask",
    )
    torch.testing.assert_close(
      torch.as_tensor(geometry.head_pad_idx),
      torch.tensor([0, 2, 4, 5], dtype=torch.long),
      rtol=0.0,
      atol=0.0,
    )
    torch.testing.assert_close(
      torch.as_tensor(geometry.head_valid_mask),
      torch.tensor(
        [[True, False, True], [False, True, True]], dtype=torch.bool),
      rtol=0.0,
      atol=0.0,
    )

  def test_shear_map_keeps_a_fully_masked_middle_physical_bin(self):
    """An empty middle bin remains a row, so later bins cannot shift left."""
    # For two source bins and three theta values, the full xi vector has six
    # physical rows of width three. Keep data from row zero and row two, while
    # masking every value in the physical row between them.
    geometry = _cosmic_geometry(
      total_size=18, kept_positions=[0, 2, 6, 8])
    fake_getdist = types.ModuleType("getdist")

    with tempfile.TemporaryDirectory(prefix="padded-head-empty-bin-") \
        as directory:
      nz_path = os.path.join(directory, "source_nz.txt")
      np.savetxt(nz_path, np.asarray(
        ((0.1, 1.0, 0.1), (0.5, 0.2, 1.0))))

      class _IniFile:
        def __init__(self, _path):
          pass

        def int(self, name):
          return {"n_theta": 3, "source_ntomo": 2}[name]

        def float(self, name):
          return {"theta_min_arcmin": 1.0,
                  "theta_max_arcmin": 8.0}[name]

        def relativeFileName(self, name):
          if name != "nz_source_file":
            raise AssertionError("unexpected dataset key " + name)
          return nz_path

      fake_getdist.IniFile = _IniFile
      with mock.patch.dict(sys.modules, {"getdist": fake_getdist}), \
          mock.patch.dict(os.environ, {"ROOTDIR": directory}):
        build_shear_angle_map(
          geometry, data_dir="fixture", dataset="fixture.dataset")

    self.assertEqual(geometry.bin_sizes, [2, 2])
    torch.testing.assert_close(
      geometry.head_pad_idx,
      torch.tensor([0, 2, 6, 8], dtype=torch.long),
      rtol=0.0, atol=0.0)
    self.assertEqual(tuple(geometry.head_valid_mask.shape), (6, 3))
    torch.testing.assert_close(
      geometry.head_valid_mask[:3],
      torch.tensor(
        [[True, False, True],
         [False, False, False],
         [True, False, True]], dtype=torch.bool),
      rtol=0.0, atol=0.0)

  def test_each_structured_design_consumes_the_explicit_coordinate_map(self):
    """All four designs use physical slots rather than rebuilding from counts."""
    cases = (
      ("plain CNN", ResCNN, False, {
        "input_dim": 2, "output_dim": 4, "int_dim_res": 4,
        "kernel_size": 3, "n_blocks": 0, "n_blocks_cnn": 1,
      }),
      ("plain transformer", ResTRF, False, {
        "input_dim": 2, "output_dim": 4, "int_dim_res": 4,
        "n_heads": 1, "n_blocks": 0, "n_blocks_trf": 1,
        "n_mlp_blocks": 1,
      }),
      ("template CNN", TemplateResCNN, True, {
        "input_dim": 2, "output_dim": 4, "n_amps": 1,
        "n_templates": 1, "int_dim_res": 4, "kernel_size": 3,
        "n_blocks": 0, "n_blocks_cnn": 1,
      }),
      ("template transformer", TemplateResTRF, True, {
        "input_dim": 2, "output_dim": 4, "n_amps": 1,
        "n_templates": 1, "int_dim_res": 4, "n_heads": 1,
        "n_blocks": 0, "n_blocks_trf": 1, "n_mlp_blocks": 1,
      }),
    )
    expected_index = torch.tensor([0, 2, 4, 5], dtype=torch.long)
    for label, model_cls, template, options in cases:
      with self.subTest(design=label):
        geometry = _geometry_with_explicit_coordinate_map(template=template)
        model = model_cls(geom=geometry, **options)
        self.assertEqual(model.max_bin, 3)
        torch.testing.assert_close(
          model.pad_idx.cpu(), expected_index, rtol=0.0, atol=0.0)
        expected_valid = geometry.head_valid_mask.unsqueeze(0)
        if template:
          expected_valid = expected_valid.repeat(
            1, options["n_templates"], 1)
        torch.testing.assert_close(
          model.pad_valid.cpu(), expected_valid, rtol=0.0, atol=0.0)
        state = model.state_dict()
        self.assertIn("pad_idx", state)
        self.assertIn("pad_valid", state)
        torch.testing.assert_close(
          state["pad_idx"].cpu(), expected_index, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
          state["pad_valid"].cpu(), expected_valid, rtol=0.0, atol=0.0)

  def test_old_geometry_without_the_coordinate_record_is_refused(self):
    """A legacy bin-count record is refused instead of guessed silently."""
    geometry = _dense_cosmic_geometry()
    geometry.bin_sizes = [2, 2]

    with self.assertRaisesRegex(
        ValueError, "persisted physical padded-head layout"):
      ResCNN(
        input_dim=2, output_dim=4, int_dim_res=4, geom=geometry,
        kernel_size=3, n_blocks=0, n_blocks_cnn=1)

  def test_geometry_state_round_trip_preserves_map_and_validity_mask(self):
    """A rebuilt head receives the exact coordinate record that was saved."""
    geometry = _dense_cosmic_geometry()
    geometry.bin_sizes = [2, 2]
    geometry.pm_kept = np.asarray([0, 0, 1, 1], dtype="int64")
    geometry.head_pad_idx = torch.tensor([0, 2, 4, 5], dtype=torch.long)
    geometry.head_valid_mask = torch.tensor(
      [[True, False, True], [False, True, True]], dtype=torch.bool)

    state = geometry.state()

    self.assertIn("head_pad_idx", state)
    self.assertIn("head_valid_mask", state)
    rebuilt = DataVectorGeometry.from_state(device=CPU, state=state)
    torch.testing.assert_close(
      torch.as_tensor(rebuilt.head_pad_idx), geometry.head_pad_idx,
      rtol=0.0, atol=0.0)
    torch.testing.assert_close(
      torch.as_tensor(rebuilt.head_valid_mask), geometry.head_valid_mask,
      rtol=0.0, atol=0.0)

  def test_rectangular_heads_keep_their_exact_initial_identity(self):
    """Adding mask support must not perturb a layout with no padding."""
    inputs = torch.tensor(((0.2, -0.5), (-0.3, 0.7)), dtype=torch.float32)
    plain_expected = torch.tensor(
      ((0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4)),
      dtype=torch.float32)
    template_expected = plain_expected[:, None, :]
    cases = (
      ("plain CNN", ResCNN(
        input_dim=2, output_dim=4, int_dim_res=4,
        geom=_plain_geometry([2, 2]), kernel_size=3, n_blocks=0,
        n_blocks_cnn=1), "mlp", plain_expected),
      ("plain transformer", ResTRF(
        input_dim=2, output_dim=4, int_dim_res=4,
        geom=_plain_geometry([2, 2]), n_heads=1, n_blocks=0,
        n_blocks_trf=1, n_mlp_blocks=1), "mlp", plain_expected),
      ("template CNN", TemplateResCNN(
        input_dim=2, output_dim=4, n_amps=1, n_templates=1,
        int_dim_res=4, geom=_template_geometry([2, 2]), kernel_size=3,
        n_blocks=0, n_blocks_cnn=1), "model", template_expected),
      ("template transformer", TemplateResTRF(
        input_dim=2, output_dim=4, n_amps=1, n_templates=1,
        int_dim_res=4, geom=_template_geometry([2, 2]), n_heads=1,
        n_blocks=0, n_blocks_trf=1, n_mlp_blocks=1),
       "model", template_expected),
    )
    for label, model, trunk_name, expected in cases:
      with self.subTest(design=label):
        self.assertFalse(model.has_padding)
        setattr(model, trunk_name, _ConstantTrunk(expected[0].reshape(-1)))
        actual = model(inputs)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

  def test_rectangular_live_cnn_matches_the_explicit_unmasked_operation(self):
    """A complete rectangle keeps the former direct convolution path."""
    model = ResCNN(
      input_dim=2, output_dim=4, int_dim_res=4,
      geom=_plain_geometry([2, 2]), kernel_size=3,
      n_blocks=0, n_blocks_cnn=1, gate_init=0.5,
      head_act=_identity_activation)
    physical = torch.tensor([0.2, -0.4, 0.7, 0.3])
    model.mlp = _ConstantTrunk(physical)
    with torch.no_grad():
      model.convs[0].weight.copy_(torch.tensor(
        [[[0.2, 0.4, -0.1], [0.1, -0.2, 0.3]],
         [[-0.3, 0.5, 0.2], [0.4, 0.1, -0.2]]]))
      model.convs[0].bias.copy_(torch.tensor([0.05, -0.03]))
      model.gate.fill_(0.5)
    inputs = torch.tensor([[0.1, -0.2], [0.8, 0.6]])

    actual = model(inputs)
    trunk = physical.unsqueeze(0).expand(inputs.shape[0], -1)
    # This is the pre-mask rectangular implementation: reshape, convolve,
    # activate, flatten, and add the gated correction directly.
    unmasked = trunk.view(-1, 2, 2)
    unmasked = model.acts[0](model.convs[0](unmasked))
    legacy_result = trunk + model.gate * unmasked.reshape(-1, 4)

    self.assertFalse(model.has_padding)
    torch.testing.assert_close(actual, legacy_result, rtol=0.0, atol=0.0)


if __name__ == "__main__":
  unittest.main()
