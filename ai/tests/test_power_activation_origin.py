"""CPU checks for the power activations' analytic origin derivative.

The power-tail families multiply a gate by the signed power transform

    psi_p(x) = x * ((1 + |x|)^p - 1) / (p |x|),

whose analytic limit at x = 0 is slope one for every exponent p. A
sign(x) * f(|x|) form of the same values has a zero derivative at
exactly x = 0 under automatic differentiation, because sign and abs both
differentiate to zero there. Zero-initialized correction layers and
padded coordinates deliberately create exact zeros, so that form starves
part of a requested correction of every gradient while ordinary forward
checks still pass. These tests pin the exact-zero derivative, the p = 1
identity, the near-zero series accuracy, float64 gradient checks over
several learned exponents, the unchanged tail values, the constructor
bound validation, and one zero-initialized layer that must receive a
usable first gradient.
"""

import unittest

import torch

from emulator.activations import (
  GatedPowerActivation,
  PowerGatedActivation,
  _POWER_SERIES_THRESHOLD,
  activation_fcn,
  require_power_bounds,
  signed_power_transform,
)


def _input_gradient(module, values, dtype=torch.float64):
  """Return d(output)/d(input) at each value through one module.

  Arguments:
    module = the activation module (already in the matching dtype).
    values = a list of input values, one feature each.
    dtype  = the tensor dtype for the probe.

  Returns:
    the gradient tensor, same shape as the probe row.
  """
  probe = torch.tensor([values], dtype=dtype, requires_grad=True)
  output = module(probe)
  output.sum().backward()
  return probe.grad[0].detach()


class SignedPowerTransformTests(unittest.TestCase):
  """The transform itself: exact origin behavior and unchanged tails."""

  def test_p_equal_one_is_the_identity(self):
    """With p = 1: the series region bit for bit, the tail to rounding."""
    near = torch.tensor(
      [-9.0e-4, -1.0e-6, 0.0, 1.0e-6, 9.0e-4], dtype=torch.float64)
    p = torch.ones(1, dtype=torch.float64)
    self.assertTrue(torch.equal(signed_power_transform(near, p), near))

    # The direct branch keeps the sign form's rounding: (1 + u) - 1
    # drops the bits of u below 1's resolution, exactly as the previous
    # formula did, so the tail is the identity only to rounding.
    tail = torch.tensor(
      [-3.0, -0.5, 1.0e-3, 0.5, 2.0], dtype=torch.float64)
    torch.testing.assert_close(
      signed_power_transform(tail, p), tail, rtol=1.0e-12, atol=0.0)

  def test_derivative_at_exact_zero_is_one_for_every_exponent(self):
    """The zero-Jacobian assertion: a sign-form mutation fails here."""
    for exponent in (0.5, 0.75, 1.0, 1.25, 1.5):
      with self.subTest(p=exponent):
        x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        p = torch.full((3,), exponent, dtype=torch.float64)
        psi = signed_power_transform(x, p)
        psi.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x.grad)))

  def test_forward_value_at_exact_zero_is_zero(self):
    """psi(0) = 0 exactly, so identity initialization is preserved."""
    x = torch.zeros(4, dtype=torch.float64)
    p = torch.tensor([0.5, 0.9, 1.1, 1.5], dtype=torch.float64)
    psi = signed_power_transform(x, p)
    self.assertTrue(torch.equal(psi, torch.zeros_like(psi)))

  def test_tail_values_match_the_sign_form_to_rounding(self):
    """Away from the origin the repair changes nothing but rounding."""
    magnitudes = torch.logspace(-3, 2, 41, dtype=torch.float64)
    x = torch.cat([-magnitudes.flip(0), magnitudes])
    for exponent in (0.5, 1.0, 1.5):
      with self.subTest(p=exponent):
        p = torch.tensor(exponent, dtype=torch.float64)
        new_values = signed_power_transform(x, p)
        old_values = torch.sign(x) * ((1.0 + x.abs()) ** p - 1.0) / p
        torch.testing.assert_close(
          new_values, old_values, rtol=1.0e-12, atol=0.0)

  def test_series_matches_the_exact_ratio_across_the_threshold(self):
    """The near-zero series and the direct quotient agree at the seam."""
    seam = _POWER_SERIES_THRESHOLD
    x = torch.tensor(
      [seam / 10.0, seam / 2.0, seam * 0.999, seam, seam * 2.0],
      dtype=torch.float64)
    for exponent in (0.5, 1.25, 1.5):
      with self.subTest(p=exponent):
        p = torch.tensor(exponent, dtype=torch.float64)
        psi = signed_power_transform(x, p)
        exact = torch.sign(x) * ((1.0 + x.abs()) ** p - 1.0) / p
        torch.testing.assert_close(psi, exact, rtol=1.0e-9, atol=0.0)

  def test_float64_gradcheck_over_several_learned_exponents(self):
    """Automatic and numerical Jacobians agree, including through zero."""
    for exponent in (0.6, 1.0, 1.4):
      with self.subTest(p=exponent):
        p = torch.full((5,), exponent, dtype=torch.float64)
        probe = torch.tensor(
          [[-1.5, -2.0e-4, 0.0, 5.0e-4, 2.0]],
          dtype=torch.float64,
          requires_grad=True)

        def transform_only(values):
          return signed_power_transform(values, p)

        self.assertTrue(
          torch.autograd.gradcheck(
            transform_only, (probe,), raise_exception=True))


class PowerActivationModuleTests(unittest.TestCase):
  """The two owning modules: origin gradients, bounds, and liveness."""

  def test_default_init_matches_h_values_and_gradients_at_the_origin(self):
    """At default init all three families agree at [-eps, 0, +eps]."""
    values = [-1.0e-4, 0.0, 1.0e-4]
    reference = activation_fcn(3).double()
    for family in (PowerGatedActivation, GatedPowerActivation):
      with self.subTest(family=family.__name__):
        module = family(3).double()
        probe = torch.tensor([values], dtype=torch.float64)
        torch.testing.assert_close(
          module(probe), reference(probe), rtol=1.0e-9, atol=1.0e-12)
        gradient = _input_gradient(module, values)
        reference_gradient = _input_gradient(reference, values)
        torch.testing.assert_close(
          gradient, reference_gradient, rtol=1.0e-9, atol=1.0e-12)
        # The middle probe is exactly zero: gate(0) = 0.5 times the
        # transform's unit slope. Zero here is the starved-head bug.
        self.assertAlmostEqual(float(gradient[1]), 0.5, places=12)

  def test_constructors_refuse_malformed_power_bounds(self):
    """Nonpositive, nonfinite, and inverted bounds stop construction."""
    bad_pairs = (
      (0.0, 1.5),
      (-0.5, 1.5),
      (float("nan"), 1.5),
      (0.5, 0.5),
      (0.5, 0.4),
      (0.5, float("inf")),
    )
    for family in (PowerGatedActivation, GatedPowerActivation):
      for p_min, p_max in bad_pairs:
        with self.subTest(family=family.__name__, p_min=p_min,
                          p_max=p_max):
          with self.assertRaisesRegex(ValueError, "power activation"):
            family(4, p_min=p_min, p_max=p_max)
    checked_min, checked_max = require_power_bounds(0.25, 2.0)
    self.assertEqual((checked_min, checked_max), (0.25, 2.0))

  def test_zero_initialized_layer_receives_a_usable_gradient(self):
    """A zeroed layer feeding the power activation moves on step one."""
    for family in (PowerGatedActivation, GatedPowerActivation):
      with self.subTest(family=family.__name__):
        layer = torch.nn.Linear(4, 4, bias=True).double()
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
        activation = family(4).double()
        torch.manual_seed(11)
        inputs = torch.randn(8, 4, dtype=torch.float64)
        residual = activation(layer(inputs))
        loss = torch.mean((inputs + residual - 1.0) ** 2)
        loss.backward()
        gradient = layer.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(int(torch.count_nonzero(gradient)), 0)


if __name__ == "__main__":
  unittest.main()
