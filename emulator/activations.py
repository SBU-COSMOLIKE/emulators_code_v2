"""Scalar learnable activation functions for the ResBlock `act` slot.

An activation is the nonlinear function applied between a network's
layers; without one, stacked linear layers would collapse to a single
linear map.  Each class here is an nn.Module — PyTorch's base class for
anything that carries learnable parameters — computing one elementwise
activation with learnable per-feature shape parameters; a ResBlock
takes an `act` factory (a callable act(dim) -> module) and builds one
per layer.

The families build on two ingredients: sigmoid, the S-shaped curve
sigmoid(t) = 1 / (1 + e^-t) rising smoothly from 0 to 1, and Swish, the
function x * sigmoid(x).  activation_fcn is the paper's
H(x) = (gamma + (1-gamma) sigmoid(beta x)) x, a learnable
identity<->Swish interpolation; GatedActivation (K gates),
PowerGatedActivation (a bounded power tail), and GatedPowerActivation
(both) generalize it. make_activation maps a short name ("H", "power",
"multigate", "gated_power", plus the parameter-free "relu" / "tanh") to
the matching factory, so the activation can be chosen by string from a
driver or YAML.
"""

import math

import torch
import torch.nn as nn

from .validation import require_exact_int


ACTIVATION_NAMES = (
  "H", "power", "multigate", "gated_power", "relu", "tanh")

# ReLU has a zero input derivative at exactly zero (Torch's convention on
# the closed negative side). Correction heads zero their last layer, so
# using it after that layer prevents the requested head from learning.
# The power-tail families are not in this set: their signed power
# transform is computed as x times an even magnitude ratio whose analytic
# origin limit is one, so they pass a usable gradient through exact zero.
ZERO_DERIVATIVE_HEAD_ACTIVATIONS = frozenset(("relu",))

# Below this |x|, the signed power transform's ratio uses its quadratic
# Taylor series instead of the direct quotient. The truncation error is
# about |(p-1)(p-2)(p-3)| / 24 * |x|^3 relative, under 1e-9 at the
# boundary for the supported exponents -- beneath float32 resolution and
# far inside float64 gradient-check tolerances.
_POWER_SERIES_THRESHOLD = 1.0e-3


def require_power_bounds(p_min, p_max):
  """Refuse malformed power-tail bounds before any forward pass.

  The signed power transform divides by the learnable exponent
  p = p_min + (p_max - p_min) * sigmoid(rho), so p must be confined to
  a finite positive interval: p_min <= 0 would let the denominator
  reach or cross zero.

  Arguments:
    p_min = the smallest tail exponent; must be finite and > 0.
    p_max = the largest tail exponent; must be finite and > p_min.

  Returns:
    (p_min, p_max) as plain floats, unchanged in value.

  Raises:
    ValueError naming the offending bound and the required condition.
  """
  p_min = float(p_min)
  p_max = float(p_max)
  if not math.isfinite(p_min) or p_min <= 0.0:
    raise ValueError(
      "power activation p_min must be a finite positive number, got "
      + repr(p_min))
  if not math.isfinite(p_max) or p_max <= p_min:
    raise ValueError(
      "power activation p_max must be finite and greater than p_min="
      + repr(p_min) + ", got " + repr(p_max))
  return p_min, p_max


def signed_power_transform(x, p):
  """The power tail psi_p(x), with its exact analytic origin derivative.

    psi_p(x) = x * ratio(|x|),   ratio(u) = ((1 + u)^p - 1) / (p u)

  ratio is an even, smooth function of the input with the analytic
  limit one at u = 0, so psi_p is linear with slope exactly one at the
  origin for every p -- including under automatic differentiation.
  That qualifier matters: torch computes gradients by chaining the
  derivative of each operation as written, not of the ideal function,
  so two algebraically equal forms can differ in their derivative at a
  single point. A sign(x) * f(|x|) form has the same forward values
  but a zero derivative at exactly x = 0, because sign and abs both
  differentiate to zero there; a zero-initialized correction layer
  then receives no gradient and cannot begin learning. That form must
  not return.

  Both branches below are finite everywhere they are evaluated, so the
  torch.where selection cannot mix a NaN from an unselected position
  into the gradient:

    - |x| >= 1e-3: the direct quotient, with the substituted safe input
      keeping the unselected positions' denominator away from zero.
      These values match the sign-form tail to rounding.
    - |x| <  1e-3: the quadratic Taylor series of ratio (coefficients
      (p-1)/2 and (p-1)(p-2)/6), whose truncation error at the boundary
      is below float32 resolution. At p = 1 the series is exactly the
      identity; the direct branch keeps the sign form's rounding.

  shape flow: x [..., dim], p [dim] -> psi [..., dim]

  legend: dim=feature width (p broadcasts over the leading axes)

  Arguments:
    x = the activation input tensor.
    p = the per-feature exponent, confined to a finite positive
        interval by the owning constructor's require_power_bounds.

  Returns:
    psi_p(x), the same shape and dtype as x.
  """
  u = x.abs()
  near_zero = u < _POWER_SERIES_THRESHOLD
  u_safe = torch.where(near_zero, torch.ones_like(u), u)
  ratio_direct = ((1.0 + u_safe) ** p - 1.0) / (p * u_safe)
  ratio_series = (1.0 + (p - 1.0) / 2.0 * u
                  + (p - 1.0) * (p - 2.0) / 6.0 * u * u)
  ratio = torch.where(near_zero, ratio_series, ratio_direct)
  return x * ratio


def activation_factory_name(factory):
  """Map an activation factory back to its registry name ("H", "relu", ...).

  The inverse of make_activation: given the callable a ResBlock holds in its
  `act` slot, recover the short name a driver or YAML would use to request
  it. Two kinds of factory exist, and each is resolved differently:

    - a class or module passed DIRECTLY (activation_fcn, GatedActivation,
      nn.ReLU, ...) is matched by identity against the known families;
    - a closure built by make_activation — a closure is a function
      defined inside another function that remembers its maker's local
      variables, here the requested name and gate count — carries the
      requested name on itself as the `_emulator_activation_name`
      attribute, so the closure answers for any registered name,
      including future ones.

  The identity checks come first because a direct class never passed
  through make_activation and so never received the attribute.

  Arguments:
    factory = the activation factory, a callable act(dim) -> nn.Module: a
              make_activation closure, one of the activation classes, or
              nn.ReLU / nn.Tanh.

  Returns:
    the registry name as a string ("H", "power", "multigate",
    "gated_power", "relu", "tanh"), or None when the factory is not a
    registered activation (the caller decides whether that is an error).
  """
  if factory is activation_fcn:
    return "H"
  if factory is PowerGatedActivation:
    return "power"
  if factory is GatedActivation:
    return "multigate"
  if factory is GatedPowerActivation:
    return "gated_power"
  if factory is nn.ReLU:
    return "relu"
  if factory is nn.Tanh:
    return "tanh"
  return getattr(factory, "_emulator_activation_name", None)


def activation_factory_recipe(factory):
  """Describe one activation factory as plain recipe data for a saved emulator.

  A saved .h5 must record how to rebuild its activation without executing
  code from the file, so the factory is reduced to two inert values: the
  registry name and the gate count K. make_activation tags every closure it
  builds with both, and this function reads them back:

    make_activation("multigate", n_gates=4) -> {"type": "multigate",
                                                "n_gates": 4}

  A class passed directly (not through make_activation) is a special case.
  A direct gated class executes its constructor default of ONE gate, so its
  recipe says 1 rather than borrowing the public factory's three-gate
  default -- recording 3 would rebuild a different network. A direct
  nn.ReLU or nn.Tanh cannot be serialized as its registered equivalent at
  all: ResBlock calls every factory as act(dim), and nn.ReLU(dim)
  interprets that number as inplace=True instead of building the
  out-of-place module the registered closure selects. Such a factory is
  therefore recorded as "unregistered:<class name>", which the artifact
  writer refuses with instructions to wrap the class in make_activation.

  Arguments:
    factory = the activation factory to describe, a callable
              act(dim) -> nn.Module (a make_activation closure or a class).

  Returns:
    a mapping {"type": <registry name or "unregistered:<label>">,
    "n_gates": <int K>} -- plain values, safe to store in the .h5 recipe.
  """
  if factory in (nn.ReLU, nn.Tanh):
    label = getattr(factory, "__qualname__", type(factory).__qualname__)
    return {"type": "unregistered:" + label, "n_gates": 3}
  name = activation_factory_name(factory)
  if name is None:
    label = getattr(factory, "__qualname__", type(factory).__qualname__)
    return {"type": "unregistered:" + label, "n_gates": 3}
  if factory in (GatedActivation, GatedPowerActivation):
    n_gates = 1
  else:
    n_gates = getattr(factory, "_emulator_activation_n_gates", 3)
  require_exact_int(n_gates, "activation factory n_gates", minimum=1)
  return {"type": name, "n_gates": int(n_gates)}


def require_live_head_activation(factory, source):
  """Refuse a head activation whose derivative is zero at the origin.

  A correction head initializes its last linear layer to zero, so at the
  first training step every head input is exactly x = 0. An activation
  with a(0) = 0 AND a'(0) = 0 (ReLU, on Torch's closed negative side)
  then passes zero forward and zero gradient backward: part of the
  requested correction can never begin learning. Every other family --
  H, multigate, tanh, and the power tails through their analytic-origin
  ratio -- has a nonzero origin derivative, so the head trains from the
  first step.

  Arguments:
    factory = the head's activation factory, a callable
              act(dim) -> nn.Module.
    source  = where the choice came from (a YAML key or driver name),
              named in the refusal so the user can fix the right line.

  Returns:
    the same factory, unchanged, when it is safe for a head.

  Raises:
    ValueError naming the activation and the safe alternatives when the
    factory belongs to ZERO_DERIVATIVE_HEAD_ACTIVATIONS.
  """
  name = activation_factory_name(factory)
  if name in ZERO_DERIVATIVE_HEAD_ACTIVATIONS:
    raise ValueError(
      source + " uses " + repr(name) + " after a zero-initialized head "
      "layer. Its input derivative at zero is zero, so the requested "
      "correction cannot fully begin learning. Use H, multigate, power, "
      "gated_power, or tanh for the head.")
  return factory


class activation_fcn(nn.Module):
    """The paper's learnable H(x): identity<->Swish interpolation per element.

      H(x) = (gamma + (1 - gamma) * sigmoid(beta * x)) * x

    sigmoid(t) = 1 / (1 + e^-t) rises smoothly from 0 to 1, so the gate
    gamma + (1 - gamma) sigmoid(beta x) runs from gamma (x -> -inf) to 1
    (x -> +inf), making H asymptotically linear at both tails (slope
    gamma left, 1 right), non-saturating, hence better than tanh here.
    Each feature has its own learnable gamma and beta (length-`dim`
    vectors). gamma = beta = 0 at init, so H starts as 0.5 * x
    (sigmoid(0) = 0.5); training then shapes each feature's curve. The
    Gated/Power/GatedPower variants generalize this same gate.

    Arguments:
      dim = feature width (one independent gamma / beta per feature).
    """
    def __init__(self, dim):
        super(activation_fcn, self).__init__()
        self.dim   = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta  = nn.Parameter(torch.zeros((dim)))
    def forward(self,x):
        # H(x) = (gamma + (1 - gamma) sigmoid(beta x)) * x, elementwise.
        exp = torch.mul(self.beta,x)            # beta * x
        inv = torch.special.expit(exp)          # sigmoid(beta x)
        fac_2 = 1-self.gamma                     # (1 - gamma) weight
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out


class GatedActivation(nn.Module):
  """Generalized H(x): x times a learnable gate of K sigmoids.

    gate(x) = a0 + sum_k w_k * sigmoid(beta_k * (x - mu_k))
    out     = gate(x) * x

  Every term is a bounded sigmoid times x, keeping the output
  asymptotically linear (slope a0 as x->-inf, a0+sum_k w_k as
  x->+inf), non-saturating like H, never blows up.

  H = (gamma + (1-gamma) sigmoid(beta x)) x is the K=1 case
  (a0=gamma, w=1-gamma, mu=0); the general form also frees the
  positive-side slope (a0+w) and the kink center mu, and K>1 adds
  gates (a learned slope-vs-x schedule). All parameters are
  per-element vectors of length `dim`, one activation shape per
  feature (as gamma/beta were).

  Arguments:
    dim     = feature width (gamma/beta were this shape too).
    n_gates = number of sigmoid components K (default 1).
  """
  def __init__(self, dim, n_gates=1):
    super().__init__()
    K = n_gates
    # a0 = negative-tail slope (gate value as x -> -inf).
    self.a0 = nn.Parameter(torch.zeros(dim))
    # per-gate weight / sharpness / center, each (K, dim). Init
    # reproduces H's start: gate 0 (w=1, beta=0, mu=0) -> 0.5;
    # extra gates inactive (w=0) but beta=1, spread mu, ready to
    # specialize once training turns them on.
    w0    = torch.zeros(K, dim)
    beta0 = torch.zeros(K, dim)
    mu0   = torch.zeros(K, dim)
    w0[0] = 1.0
    if K > 1:
      beta0[1:] = 1.0
      # linspace positional args are (start, stop, steps): K gate
      # centers evenly spaced over [-1.5, 1.5], then drop gate 0.
      mu0[1:] = torch.linspace(-1.5, 1.5, K)[1:, None]
    self.w    = nn.Parameter(w0)
    self.beta = nn.Parameter(beta0)
    self.mu   = nn.Parameter(mu0)

  def forward(self, x):
    # unsqueeze(-2) adds a size-1 axis before the last:
    # (..., dim) -> (..., 1, dim), which broadcasts against the K
    # gate parameters (shape (K, dim)) -> (..., K, dim), matching
    # each input value against all K gates at once.
    xx = x.unsqueeze(-2)                            # (...,1,dim)
    s  = torch.sigmoid(self.beta * (xx - self.mu))  # (...,K,dim)
    gate = self.a0 + (self.w * s).sum(-2)          # (..., dim)
    return gate * x


class PowerGatedActivation(nn.Module):
  """H(x) with a learnable, bounded power tail.

  Same leaky/Swish gate as the paper's H, but the multiplied x becomes a signed power
  transform psi_p: linear near 0, ~|x|^p in the tail, with p
  learnable per element and confined to [p_min, p_max] (default
  [0.5, 1.5], between sqrt(x) and x^1.5). p = 1 recovers H.

    gate(x) = gamma + (1 - gamma) * sigmoid(beta * x)
    psi_p(x) = x * ((1 + |x|)^p - 1) / (p |x|)   (limit 1 at x=0)
    H(x)     = gate(x) * psi_p(x)

  psi_p has slope exactly 1 at x=0 for any p, and the even-ratio
  form above keeps that derivative under automatic differentiation
  (signed_power_transform owns the computation and its near-zero
  series), so a zero-initialized correction layer trains from the
  first step. p reshapes only the tail, not the behavior near 0.
  The base 1+|x| >= 1 keeps any real p finite (no NaN), and the
  sigmoid box blocks a blow-up power (safe on a narrow prior,
  unlike a raw x^n). rho=0 at init -> p=1 -> starts as H.

  Arguments:
    dim   = feature width (per-element gamma/beta/rho vectors).
    p_min = smallest tail exponent (default 0.5, sqrt-like); must
            be finite and positive.
    p_max = largest tail exponent (default 1.5, mildly super-
            linear); must be finite and greater than p_min. p
            ranges in (p_min, p_max) via a sigmoid.

  Raises:
    ValueError from require_power_bounds for a nonpositive,
    nonfinite, or inverted bound pair.
  """
  def __init__(self, dim, p_min=0.5, p_max=1.5):
    super().__init__()
    self.gamma = nn.Parameter(torch.zeros(dim))
    self.beta  = nn.Parameter(torch.zeros(dim))
    # rho sets the exponent: p = p_min + (p_max-p_min)*sig(rho).
    # rho=0 -> midpoint p=1 for [0.5,1.5] -> identity tail.
    self.rho   = nn.Parameter(torch.zeros(dim))
    self.p_min, self.p_max = require_power_bounds(p_min, p_max)

  def forward(self, x):
    # bounded learnable exponent in (p_min, p_max), per element.
    p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(
      self.rho)
    # signed power tail with derivative exactly 1 at the origin.
    psi = signed_power_transform(x, p)
    # leaky/Swish gate (your H), applied to the power transform.
    g = self.gamma + (1.0 - self.gamma) * torch.sigmoid(
      self.beta * x)
    return g * psi


class GatedPowerActivation(nn.Module):
  """The full activation: a K-gate bulk schedule times a bounded power tail.

  Merges GatedActivation (K gates) and PowerGatedActivation (tail
  exponent), the two independent generalizations of H(x).

    gate(x) = a0 + sum_k w_k * sigmoid(beta_k * (x - mu_k))
    psi_p(x) = x * ((1 + |x|)^p - 1) / (p |x|)   (limit 1 at x=0)
    H(x)     = gate(x) * psi_p(x)
    p        = p_min + (p_max - p_min) * sigmoid(rho)

  The K sigmoids shape the slope vs x in the bulk; psi_p reshapes
  only the tail, with p boxed into [p_min, p_max] so it cannot blow
  up. The even-ratio psi_p has slope exactly 1 at x=0 for any p,
  kept under automatic differentiation by signed_power_transform's
  near-zero series, so a zero-initialized correction layer trains
  from the first step. Every term is a bounded sigmoid times a mild
  power, keeping the output finite.

  Recovers H at K=1 and the default init: gate 0 (w=1, beta=0,
  mu=0) -> 0.5, and rho=0 -> p=1 -> psi=x, so H = 0.5 x at init.
  Extra gates start inactive (w=0).

  Per-element parameters: a0 (1) + {w,beta,mu} x K (3K) + rho (1)
  = 3K + 2 vectors of length `dim`.

  Arguments:
    dim     = feature width (per-element parameter vectors).
    n_gates = number of bulk sigmoid gates K (default 1).
    p_min   = smallest tail exponent (default 0.5, sqrt-like);
              must be finite and positive.
    p_max   = largest  tail exponent (default 1.5, super-linear);
              must be finite and greater than p_min.

  Raises:
    ValueError from require_power_bounds for a nonpositive,
    nonfinite, or inverted bound pair.
  """
  def __init__(self, dim, n_gates=1, p_min=0.5, p_max=1.5):
    super().__init__()
    K = n_gates
    # --- multi-gate (bulk slope schedule) ---
    self.a0 = nn.Parameter(torch.zeros(dim))   # neg-tail slope
    w0    = torch.zeros(K, dim)
    beta0 = torch.zeros(K, dim)
    mu0   = torch.zeros(K, dim)
    w0[0] = 1.0                                # gate 0 -> H init
    if K > 1:
      # extra gates: active (beta=1), spread centers, but w=0
      # (inactive) until training engages them. linspace positional
      # args are (start, stop, steps).
      beta0[1:] = 1.0
      mu0[1:] = torch.linspace(-1.5, 1.5, K)[1:, None]
    self.w    = nn.Parameter(w0)
    self.beta = nn.Parameter(beta0)
    self.mu   = nn.Parameter(mu0)
    # --- bounded tail exponent ---
    self.rho   = nn.Parameter(torch.zeros(dim))  # rho=0 -> p=1
    self.p_min, self.p_max = require_power_bounds(p_min, p_max)

  def forward(self, x):
    # bulk gate: a0 + sum_k w_k sigmoid(beta_k (x - mu_k)).
    # unsqueeze(-2) adds a size-1 axis before the last
    # ((..., dim) -> (..., 1, dim)), broadcasting x against the
    # K gates (shape (K, dim)) -> (..., K, dim).
    xx   = x.unsqueeze(-2)               # (..., 1, dim)
    s    = torch.sigmoid(self.beta * (xx - self.mu))
    gate = self.a0 + (self.w * s).sum(-2)   # (..., dim)
    # bounded learnable tail exponent in (p_min, p_max).
    p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(
      self.rho)
    # signed power tail with derivative exactly 1 at the origin.
    psi = signed_power_transform(x, p)
    return gate * psi


def make_activation(name, n_gates=3):
  """Activation factory by name, for a ResBlock's `act` slot.

  Maps a short name to a factory callable act(dim) -> module, the
  contract ResBlock's `act` expects (it calls act(size) once per
  layer), letting a driver or YAML pick the activation by string
  rather than importing a class. The gated families use
  K = n_gates gates.

  Arguments:
    name    = one of:
                "H"           -> activation_fcn, the paper's H
                                 (also the ResBlock default).
                "power"       -> PowerGatedActivation (bounded
                                 learnable tail exponent).
                "multigate"   -> GatedActivation (K = n_gates).
                "gated_power" -> GatedPowerActivation (K gates plus
                                 the tail exponent).
                "relu"        -> nn.ReLU, parameter-free.
                "tanh"        -> nn.Tanh, parameter-free (pair with
                                 model.norm per_feature to guard its
                                 saturation, per the paper).
    n_gates = number of gates K for the multi-gate families
              (default 3); ignored by "H" / "power" / "relu" / "tanh".

  Returns:
    a factory act(dim) -> nn.Module.
  """
  require_exact_int(n_gates, "activation.n_gates", minimum=1)
  if name == "H":
    def h_factory(dim):
      """Build one activation_fcn at feature width dim."""
      return activation_fcn(dim)
    factory = h_factory
  if name == "power":
    def power_factory(dim):
      """Build one PowerGatedActivation at feature width dim."""
      return PowerGatedActivation(dim)
    factory = power_factory
  if name == "multigate":
    def multigate_factory(dim):
      """Build one GatedActivation at feature width dim."""
      return GatedActivation(dim, n_gates=n_gates)
    factory = multigate_factory
  if name == "gated_power":
    def gated_power_factory(dim):
      """Build one GatedPowerActivation at feature width dim."""
      return GatedPowerActivation(dim, n_gates=n_gates)
    factory = gated_power_factory
  # parameter-free families: the factory ignores dim (ReLU / Tanh take
  # no shape argument), still honoring the act(dim) -> module contract.
  if name == "relu":
    def relu_factory(dim):
      """Build one nn.ReLU; dim is accepted (the contract) and unused."""
      return nn.ReLU()
    factory = relu_factory
  if name == "tanh":
    def tanh_factory(dim):
      """Build one nn.Tanh; dim is accepted (the contract) and unused."""
      return nn.Tanh()
    factory = tanh_factory
  if name not in ACTIVATION_NAMES:
    raise ValueError(
      f"unknown activation {name!r}; one of: "
      + " / ".join(ACTIVATION_NAMES))
  factory._emulator_activation_name = name
  factory._emulator_activation_n_gates = int(n_gates)
  return factory
