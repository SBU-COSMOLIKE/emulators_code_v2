"""Focused CPU tests for warm-start perturbation finite checks.

The real warm-start functions first evaluate an unchanged batch and then add
one to every extra parameter. These tests make the unchanged batch finite and
poison exactly one row after that addition. The input-guard tests require the
input quantity in the error, so removing that guard cannot be hidden by the
later output guard. The output-guard tests require the shared finite-contract
error, so removing that guard exposes the older parity-mismatch error.
"""

import inspect
import types
import unittest
from unittest import mock

import numpy as np
import torch

from emulator import warmstart


_BAD_SOURCE_ROW = 9


class _ParameterGeometry:
  """Minimal geometry whose optional poison appears after perturbation.

  ``encode`` returns a copy. The copy matters because assigning NaN to a view
  would also modify the raw input tensor used by the other parity arm.
  """

  def __init__(
      self,
      names,
      poison_perturbed_input=False):
    self.names = list(names)
    self.poison_perturbed_input = bool(poison_perturbed_input)

  def encode(
      self,
      values):
    encoded = values.clone()
    if not self.poison_perturbed_input:
      return encoded
    if encoded.shape[1] < 2:
      return encoded

    extra_was_perturbed = encoded[:, 1] > 0.5
    selected_test_row = encoded[:, 0] > 1.0
    poison_rows = extra_was_perturbed & selected_test_row
    encoded[poison_rows, 0] = float("nan")
    return encoded


class _LinearModel(torch.nn.Module):
  """One linear layer with an optional perturbation-only output poison."""

  def __init__(
      self,
      input_dim,
      poison_perturbed_output=False):
    super().__init__()
    self.linear = torch.nn.Linear(
      in_features=input_dim,
      out_features=1,
      bias=False)
    self.poison_perturbed_output = bool(poison_perturbed_output)
    with torch.no_grad():
      self.linear.weight.zero_()
      self.linear.weight[0, 0] = 2.0

  def forward(
      self,
      encoded):
    output = self.linear(encoded)
    if not self.poison_perturbed_output:
      return output
    if encoded.shape[1] < 2:
      return output

    extra_was_perturbed = encoded[:, 1] > 0.5
    selected_test_row = encoded[:, 0] > 1.0
    poison_rows = extra_was_perturbed & selected_test_row
    output = output.clone()
    output[poison_rows, 0] = float("inf")
    return output


class _TransferLoss:
  """Minimal transfer composition with optional perturbed-output poison."""

  def __init__(
      self,
      poison_perturbed_output=False):
    self.dest_idx = torch.tensor([0], dtype=torch.long)
    self.poison_perturbed_output = bool(poison_perturbed_output)

  def base_decode(
      self,
      encoded):
    return 3.0 * encoded[:, :1]

  def decode(
      self,
      correction,
      encoded):
    output = self.base_decode(encoded) + correction
    if not self.poison_perturbed_output:
      return output

    extra_was_perturbed = encoded[:, 1] > 0.5
    selected_test_row = encoded[:, 0] > 1.0
    poison_rows = extra_was_perturbed & selected_test_row
    output = output.clone()
    output[poison_rows, 0] = float("inf")
    return output


def _training_source():
  """Return raw rows whose extra coordinate starts at zero."""
  parameters = np.zeros((12, 2), dtype=np.float32)
  parameters[4, 0] = 0.25
  parameters[_BAD_SOURCE_ROW, 0] = 2.0
  selected_rows = np.asarray([4, _BAD_SOURCE_ROW], dtype=np.int64)
  return {
    "C": parameters,
    "idx": selected_rows,
  }


def _source_object():
  """Return the source fields consumed by ``build_warm_start``."""
  source_model = _LinearModel(
    input_dim=1,
    poison_perturbed_output=False)
  source_geometry = _ParameterGeometry(names=["shared"])
  output_geometry = types.SimpleNamespace(
    dest_idx=torch.tensor([0], dtype=torch.long))
  return types.SimpleNamespace(
    model=source_model,
    pgeom=source_geometry,
    geom=output_geometry)


def _model_factory(
    poison_perturbed_output=False):
  """Return a make_model replacement with the production call signature."""

  def make_test_model(
      model_opts,
      input_dim,
      output_dim,
      device):
    del model_opts
    if output_dim != 1:
      raise AssertionError("the fixture expects one output coordinate")
    model = _LinearModel(
      input_dim=input_dim,
      poison_perturbed_output=poison_perturbed_output)
    return model.to(device)

  return make_test_model


def _guard_mutation(
    quantity_to_skip):
  """Return a guard replacement that skips one named production call.

  The returned counter proves that the intended call was reached. A future
  edit that deletes the call therefore fails the mutation test even if a
  later guard happens to raise another error.
  """
  original_guard = warmstart._require_parity_finite
  mutation_state = {"skipped_calls": 0}

  def mutated_guard(
      *args,
      **kwargs):
    if "quantity" in kwargs:
      quantity = kwargs["quantity"]
    else:
      quantity = args[1]
    if quantity == quantity_to_skip:
      mutation_state["skipped_calls"] += 1
      return None
    return original_guard(*args, **kwargs)

  return mutation_state, mutated_guard


class WarmstartPerturbedFiniteTests(unittest.TestCase):
  """Exercise both perturbation boundaries in both warm-start paths."""

  def _run_finetune(
      self,
      new_geometry,
      poison_perturbed_output=False):
    source = _source_object()
    factory = _model_factory(
      poison_perturbed_output=poison_perturbed_output)
    with mock.patch.object(
        warmstart,
        "make_model",
        side_effect=factory):
      return warmstart.build_warm_start(
        source=source,
        new_pgeom=new_geometry,
        pinned_geom=source.geom,
        model_opts={},
        train_set=_training_source(),
        extra_names=["extra"],
        device=torch.device("cpu"))

  def _run_transfer(
      self,
      new_geometry,
      poison_perturbed_output=False):
    factory = _model_factory(poison_perturbed_output=False)
    transfer_loss = _TransferLoss(
      poison_perturbed_output=poison_perturbed_output)
    with mock.patch.object(
        warmstart,
        "make_model",
        side_effect=factory):
      return warmstart.build_transfer_start(
        chi2fn=transfer_loss,
        model_opts={},
        new_pgeom=new_geometry,
        train_set=_training_source(),
        extra_names=["extra"],
        device=torch.device("cpu"))

  def test_finetune_finite_control_keeps_parity(self):
    geometry = _ParameterGeometry(names=["shared", "extra"])
    _state, verdict, padded_keys = self._run_finetune(
      new_geometry=geometry)
    self.assertTrue(verdict.startswith("[ok] finetune parity"))
    self.assertEqual(padded_keys, ["linear.weight"])

  def test_transfer_finite_control_keeps_parity(self):
    geometry = _ParameterGeometry(names=["shared", "extra"])
    _state, verdict = self._run_transfer(new_geometry=geometry)
    self.assertTrue(verdict.startswith("[ok] transfer parity"))

  def test_finetune_nan_in_perturbed_encode_names_input_and_row(self):
    geometry = _ParameterGeometry(
      names=["shared", "extra"],
      poison_perturbed_input=True)
    with self.assertRaises(ValueError) as caught:
      self._run_finetune(new_geometry=geometry)
    message = str(caught.exception)
    self.assertIn("finite contract [finetune parity]", message)
    self.assertIn("perturbed encoded new-run inputs", message)
    self.assertIn("positions: [9]", message)
    self.assertNotIn("leaked", message)

  def test_finetune_inf_in_perturbed_output_names_output_and_row(self):
    geometry = _ParameterGeometry(names=["shared", "extra"])
    with self.assertRaises(ValueError) as caught:
      self._run_finetune(
        new_geometry=geometry,
        poison_perturbed_output=True)
    message = str(caught.exception)
    self.assertIn("finite contract [finetune parity]", message)
    self.assertIn("perturbed epoch-0 new-model outputs", message)
    self.assertIn("positions: [9]", message)
    self.assertNotIn("leaked", message)

  def test_finetune_input_guard_removal_changes_the_error_quantity(self):
    geometry = _ParameterGeometry(
      names=["shared", "extra"],
      poison_perturbed_input=True)
    mutation_state, mutated_guard = _guard_mutation(
      quantity_to_skip="perturbed encoded new-run inputs")
    with mock.patch.object(
        warmstart,
        "_require_parity_finite",
        side_effect=mutated_guard):
      with self.assertRaises(ValueError) as caught:
        self._run_finetune(new_geometry=geometry)
    message = str(caught.exception)
    self.assertEqual(mutation_state["skipped_calls"], 1)
    self.assertIn("perturbed epoch-0 new-model outputs", message)

  def test_finetune_output_guard_removal_restores_the_old_misdiagnosis(self):
    geometry = _ParameterGeometry(names=["shared", "extra"])
    mutation_state, mutated_guard = _guard_mutation(
      quantity_to_skip="perturbed epoch-0 new-model outputs")
    with mock.patch.object(
        warmstart,
        "_require_parity_finite",
        side_effect=mutated_guard):
      with self.assertRaises(ValueError) as caught:
        self._run_finetune(
          new_geometry=geometry,
          poison_perturbed_output=True)
    message = str(caught.exception)
    self.assertEqual(mutation_state["skipped_calls"], 1)
    self.assertIn("extra parameters leaked", message)

  def test_transfer_nan_in_perturbed_encode_names_input_and_row(self):
    geometry = _ParameterGeometry(
      names=["shared", "extra"],
      poison_perturbed_input=True)
    with self.assertRaises(ValueError) as caught:
      self._run_transfer(new_geometry=geometry)
    message = str(caught.exception)
    self.assertIn("finite contract [transfer parity]", message)
    self.assertIn("perturbed encoded run inputs", message)
    self.assertIn("positions: [9]", message)
    self.assertNotIn("moved", message)

  def test_transfer_inf_in_perturbed_output_names_output_and_row(self):
    geometry = _ParameterGeometry(names=["shared", "extra"])
    with self.assertRaises(ValueError) as caught:
      self._run_transfer(
        new_geometry=geometry,
        poison_perturbed_output=True)
    message = str(caught.exception)
    self.assertIn("finite contract [transfer parity]", message)
    self.assertIn("perturbed epoch-0 composed prediction", message)
    self.assertIn("positions: [9]", message)
    self.assertNotIn("moved", message)

  def test_transfer_input_guard_removal_changes_the_error_quantity(self):
    geometry = _ParameterGeometry(
      names=["shared", "extra"],
      poison_perturbed_input=True)
    mutation_state, mutated_guard = _guard_mutation(
      quantity_to_skip="perturbed encoded run inputs")
    with mock.patch.object(
        warmstart,
        "_require_parity_finite",
        side_effect=mutated_guard):
      with self.assertRaises(ValueError) as caught:
        self._run_transfer(new_geometry=geometry)
    message = str(caught.exception)
    self.assertEqual(mutation_state["skipped_calls"], 1)
    self.assertIn("perturbed epoch-0 composed prediction", message)

  def test_transfer_output_guard_removal_restores_the_old_misdiagnosis(self):
    geometry = _ParameterGeometry(names=["shared", "extra"])
    mutation_state, mutated_guard = _guard_mutation(
      quantity_to_skip="perturbed epoch-0 composed prediction")
    with mock.patch.object(
        warmstart,
        "_require_parity_finite",
        side_effect=mutated_guard):
      with self.assertRaises(ValueError) as caught:
        self._run_transfer(
          new_geometry=geometry,
          poison_perturbed_output=True)
    message = str(caught.exception)
    self.assertEqual(mutation_state["skipped_calls"], 1)
    self.assertIn("extra parameters moved", message)

  def test_finetune_source_doc_lists_constructor_fields_and_read_counts(self):
    doc = inspect.getdoc(warmstart.FinetuneSource)
    signature = inspect.signature(warmstart.FinetuneSource.__init__)
    for field_name in signature.parameters:
      if field_name == "self":
        continue
      self.assertIn(field_name, doc)
    self.assertIn("one authenticated read", doc)
    self.assertIn("one matching weight-file load", doc)
    self.assertIn("``nla``", doc)
    self.assertIn("``tatt``", doc)
    self.assertIn("``None``", doc)


if __name__ == "__main__":
  unittest.main()
