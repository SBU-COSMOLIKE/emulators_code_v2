"""Shared nn building blocks (Affine, ResBlock, rescale_kernel_size,
FiLMGenerator, BinLinear, TRFBlock).

This is the shared-blocks member of the emulator/designs/ family. It
holds the small nn.Modules that the emulator models (plain.py and
ia.py) are assembled from. Each piece sits as follows:

  ResMLP = Linear -> n_blocks x ResBlock -> Linear -> Affine
  ResCNN = ResMLP trunk + conv correction head (bare nn.Conv1d
             layers, needing no block here; the rescale_kernel flag
             resolves the width through rescale_kernel_size)
  ResTRF = ResMLP trunk + TRFBlock correction head
             (per-token unique MLPs = BinLinear)

Affine is a learnable scalar scale and shift (the default ResBlock
"norm" and the models' final layer). ResBlock is width preserving:
``(B, D) -> (B, D)``. Its learned branch ends with its final Linear.
The skip is added after that Linear, and the final normalization and
activation run after the addition. Rectangular input/output projections
belong outside the residual block. rescale_kernel_size shrinks
the conv heads' kernel as their depth grows, preserving a single
block's receptive field. FiLMGenerator predicts the conv heads'
optional per-channel, cosmology-dependent modulation (the film
flag). BinLinear and TRFBlock are the ResTRF
head's pieces: per-token unique linears and a transformer block
whose tokens are the tomographic bins.

PS: whitened = rotated into the covariance eigenbasis and scaled to unit
variance (defined in the geometry modules, geometries.parameter /
geometries.output); these blocks operate on already-whitened tensors.
"""

import torch
import torch.nn as nn

# activation_fcn (activations.py): the learned gated activation H(x) =
# gate(x)*x, the default act factory for ResBlock and the conv/TRF heads.
from ..activations import (
  activation_factory_recipe, activation_fcn, require_live_head_activation)
from ..validation import (
  require_exact_bool, require_exact_int, require_positive_int_list)


class Affine(nn.Module):
    """
    A learnable scalar scale and shift: out = x * gain + bias.

    gain and bias are single scalars (shape (1,)) broadcast over
    every element of x: one global scale and shift, not a
    per-feature transform. gain inits to 1, bias to 0, so at init it
    is the identity. Used as the ResBlock default "norm" factory and
    the final layer of ResMLP / ResCNN.

    Both are nn.Parameter, registered and trained. Weight decay is
    kept off both (make_optimizer decays only ndim >= 2 weight
    matrices): decaying gain toward 0 would attenuate the signal,
    and decaying a bias has no principled meaning.

    Arguments:
      (constructor takes no arguments; gain and bias are the only
       state, both created internally.)

    forward Arguments:
      x = input tensor of any shape; every element is transformed.

    Returns:
      x * gain + bias, the same shape as x (gain and bias broadcast
      from their size-1 shape).
    """
    def __init__(self):
        """Create the scalar gain (init 1) and bias (init 0).

        Both are nn.Parameter of shape (1,), so the module starts as
        the identity and trains both scalars.
        """
        super(Affine, self).__init__()
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        """Scale and shift every element of x.

        Arguments:
          x = input tensor of any shape.

        Returns:
          x * gain + bias, the same shape as x (gain and bias
          broadcast from their size-1 shape).
        """
        return x * self.gain + self.bias


class FeatureAffine(nn.Module):
    """
    A learnable per-feature scale and shift: out = x * gain + bias,
    gain and bias length-`size` vectors (one pair per feature) — the
    per-feature sibling of Affine.

    A "feature" is one coordinate of the width-wide hidden vector
    flowing through the trunk (the ResBlock width, model.mlp.width):
    inside a ResBlock the batch is a (B, size) tensor, and gain / bias
    hold one (g_i, b_i) pair per column, shared across the B rows.
    Affine is the scalar case (one pair for the whole tensor);
    FeatureAffine gives every feature its own operating point, the
    saturation guard the paper's affine escalates to (model.norm
    per_feature). It is the same per-feature sense in which H's gamma /
    beta are learned.

    gain inits to 1, bias to 0, so at init it is the identity. Both are
    nn.Parameter of shape (size,); weight decay is kept off both
    automatically (make_optimizer decays only ndim >= 2 weight
    matrices, and these are ndim 1), exactly as for Affine.

    Arguments:
      size = feature width (the ResBlock width): one gain / bias per
             feature.

    forward Arguments:
      x = input tensor of shape (B, size); each column is scaled and
          shifted by its own gain / bias.

    Returns:
      x * gain + bias, the same shape as x (gain and bias broadcast
      over the batch rows from their (size,) shape).
    """
    def __init__(self, size):
        """Create one gain (init 1) and bias (init 0) per feature.

        Both are nn.Parameter of shape (size,), so the module starts
        as the identity and every feature trains its own pair.

        Arguments:
          size = feature width: one gain / bias pair per column.
        """
        super(FeatureAffine, self).__init__()
        self.gain = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
    def forward(self, x):
        """Scale and shift each column by its own pair.

        Arguments:
          x = input tensor of shape (B, size); B = batch rows.

        Returns:
          x * gain + bias, the same shape as x: column i scaled by
          gain[i] and shifted by bias[i], broadcast over the rows.
        """
        return x * self.gain + self.bias


def affine_norm(size):
  """
  Norm factory for "affine": one fresh Affine() per dense layer.

  Every norm factory is called the same way, norm(size) -> module, once
  per dense layer, so every layer owns an independent gain / bias pair.
  Affine holds one scalar pair for the whole tensor, so the size argument
  is accepted (keeping the call shape uniform across factories) and
  ignored. make_norm("affine") and ResBlock's
  default norm slot both point here.

  Arguments:
    size = the layer's feature width; unused by the scalar Affine, present
           because every norm factory is called as norm(size).

  Returns:
    a new Affine module (gain = 1, bias = 0 at init, the identity).
  """
  return Affine()


def identity_norm(size):
  """
  Norm factory for "none": nn.Identity(), the no-norm ablation.

  Arguments:
    size = the layer's feature width; unused (Identity has no parameters),
           present because every norm factory is called as norm(size).

  Returns:
    a new nn.Identity module (passes its input through unchanged).
  """
  return nn.Identity()


def make_norm(name):
  """
  ResBlock norm-factory by name, for the model.norm knob.

  Maps a short name to a norm factory norm(size) -> module, the contract
  ResBlock's `norm` slot expects (invoked once per dense layer). The
  parallel of make_activation: a driver or YAML picks the trunk's
  normalization by string. Only the trunk ResBlocks read it (the TRF
  block's internal LayerNorm and the CNN head have no norm slot).

  Arguments:
    name = one of:
             "affine"      -> affine_norm, the paper's per-layer
                              g x + b (one scalar pair per layer); the
                              default — the same function object as the
                              ResBlock default norm.
             "per_feature" -> FeatureAffine, a length-size gain / bias
                              (one pair per feature; the tanh
                              saturation guard).
             "none"        -> identity_norm, no norm (an ablation).

  batchnorm is deliberately not offered (see the README model.norm
  knob): its batch coupling would confound the batch-size / EMA
  experiments, its train / eval running-stats split risks baking a mode
  under the compiled eval twin, and its buffers sit outside the EMA
  weight average. The paper prescribes the affine as batchnorm's
  replacement; per_feature is the escalation.

  Returns:
    a factory norm(size) -> nn.Module.
  """
  if name == "affine":
    return affine_norm
  if name == "per_feature":
    return FeatureAffine
  if name == "none":
    return identity_norm
  raise ValueError(
    f"unknown model.norm {name!r}; one of: affine (the paper's "
    f"per-layer g x + b) / per_feature / none")


def normalization_factory_name(factory):
  """
  Map a norm factory back to its registry name ("affine", ...), for saving.

  The inverse of make_norm, matched by identity: the three registered
  factories are module-level objects, so the factory a ResBlock holds IS
  one of them when it came from make_norm. An unknown callable is reported
  as "unregistered:<label>" rather than refused here -- the artifact writer
  owns that refusal, and its message can then name the offending class.

  Arguments:
    factory = the norm factory a ResBlock holds, a callable
              norm(size) -> nn.Module.

  Returns:
    "affine", "per_feature", or "none" for a registered factory;
    "unregistered:<qualified class name>" otherwise (plain data, safe to
    store in the .h5 recipe).
  """
  if factory is affine_norm:
    return "affine"
  if factory is FeatureAffine:
    return "per_feature"
  if factory is identity_norm:
    return "none"
  label = getattr(factory, "__qualname__", type(factory).__qualname__)
  return "unregistered:" + label


def materialized_block_recipe(block_opts):
  """
  Record every ResBlock option as plain data, defaults materialized.

  A saved emulator must rebuild its blocks exactly, so the recipe stores
  what the constructor actually used -- including the defaults the caller
  never typed (n_layers = 2, norm = affine, act = H). Reading the default
  back from future code would let the two drift; writing the resolved
  value pins it (the never-trust-defaults rule, save side).

  Arguments:
    block_opts = the block-options mapping handed to ResBlock (keys among
                 "n_layers", "norm", "act"), or None for all defaults.

  Returns:
    a mapping {"n_layers": int, "act": <activation recipe mapping>,
    "norm": <norm registry name>} of plain values for the .h5 recipe.

  Raises:
    ValueError naming any unknown option key, or a non-integer /
    non-positive n_layers.
  """
  opts = {} if block_opts is None else dict(block_opts)
  unknown = sorted(set(opts) - {"n_layers", "norm", "act"})
  if unknown:
    raise ValueError("ResBlock options contain unknown key(s) " + repr(unknown))
  n_layers = opts.get("n_layers", 2)
  require_exact_int(n_layers, "ResBlock.n_layers", minimum=1)
  norm = opts.get("norm", affine_norm)
  act = opts.get("act", activation_fcn)
  return {
    "n_layers": int(n_layers),
    "act": activation_factory_recipe(act),
    "norm": normalization_factory_name(norm),
  }


class ResBlock(nn.Module):
  """
  Width-preserving residual block: n_layers dense layers between
  two skip points, the input added back to the last layer's output
  before its norm and activation. Input and output share one width
  by design, so the skip connection is the identity (no projection
  layer needed):

    x ─┬─ Linear ─ norm ─ act ─ ... ─ Linear ─(+)─ norm ─ act ─> out
       └─────────────── identity skip ──────────┘

  Arguments:
    size     = feature width, shared by input and output.
    n_layers = number of dense layers between two skip points.
    norm     = normalization factory, invoked as norm(size).
    act      = activation factory, invoked as act(size).

  norm and act are factories, not ready-made modules: each is
  invoked once per dense layer so every layer holds an independent
  module. A shared instance would couple the layers' learnable
  normalization parameters.

  Factory examples:
    norm = nn.BatchNorm1d       (accepts size)
    norm = lambda s: Affine()   (Affine accepts no size)
    act  = activation_fcn       (accepts size)
    act  = lambda s: nn.Tanh()  (Tanh accepts no size)

  forward Arguments:
    x = input tensor of shape (B, size); B = batch rows, size = the
        block's feature width.

  Returns:
    a tensor of shape (B, size): the residual output, with the input
    skip added to the last dense layer before its norm and activation.
  """
  def __init__(self,
               size,
               n_layers = 2,
               norm = affine_norm,
               act = activation_fcn):
    """Validate the sizes and build the n_layers dense sublayers.

    Arguments:
      size     = feature width, shared by input and output.
      n_layers = number of dense layers between the two skip points.
      norm     = normalization factory, invoked as norm(size) once
                 per layer.
      act      = activation factory, invoked as act(size) once per
                 layer.
    """
    require_exact_int(size, "ResBlock.size", minimum=1)
    require_exact_int(n_layers, "ResBlock.n_layers", minimum=1)
    super().__init__()
    self.emul_block_recipe = materialized_block_recipe({
      "n_layers": n_layers, "norm": norm, "act": act})
    self.skip = nn.Identity()

    # Sublayers go in nn.ModuleList, not a plain list or numbered
    # attributes: ModuleList registers each submodule with the
    # parent, so its parameters appear in .parameters(), transfer
    # under .to(device), and are saved in the state_dict. Build the
    # n_layers dense layers, norms, and activations in one loop;
    # each is its own module (fresh norm / act per layer, never
    # shared).
    layers, norms, acts = [], [], []
    for _ in range(n_layers):
      layers.append(nn.Linear(in_features=size, out_features=size))
      norms.append(norm(size))
      acts.append(act(size))
    self.layers = nn.ModuleList(layers)
    self.norms  = nn.ModuleList(norms)
    self.acts   = nn.ModuleList(acts)

  def forward(self, x):
    """Run the dense stack with the input added back near the end.

    Arguments:
      x = input tensor of shape (B, size); B = batch rows.

    Returns:
      a tensor of shape (B, size): the last dense layer's output plus
      the identity skip, then that layer's norm and activation.
    """
    xskip = self.skip(x)
    out = x
    n = len(self.layers)
    for i in range(n):
      out = self.layers[i](out)
      # Skip added to the final linear layer's output, before its
      # norm and activation (a pre-activation residual addition).
      if i == n - 1:
        out = out + xskip
      out = self.acts[i](self.norms[i](out))
    return out


def rescale_kernel_size(kernel_size, n_blocks_cnn):
  """
  Per-block kernel width that preserves a single block's view as
  the conv head deepens.

  kernel_size is read as the width one block alone would use, so it
  states the head's target total view: a single same-padded conv of
  width k sees k positions. n stacked same-padded convs of width
  k_n see

    RF = n * (k_n - 1) + 1

  Why: write r = (k_n - 1)/2 for the kernel radius (odd kernel,
  same-padded). One conv's output at position p is a weighted sum
  of the k_n inputs p-r .. p+r, so one layer sees k_n positions.
  Stack a second conv: its output at p reads the k_n layer-1
  positions p-r .. p+r, and each of those sees its own k_n-wide
  input window. Consecutive layer-1 positions hold windows shifted
  by exactly one column (stride 1), so the union of the k_n
  windows is one contiguous window, k_n - 1 columns wider than
  each. Drawn at k_n = 3 (r = 1):

    y[p]                          layer-2 output: taps
     │         │         │        h[p-1], h[p], h[p+1]
    h[p-1]    h[p]     h[p+1]     layer 1: three k_n-wide input
     │         │         │        windows, each shifted by one
     ▼         ▼         ▼        column
    x[p-2..p] x[p-1..p+1] x[p..p+2]
                                  union = x[p-2 .. p+2]:
                                  5 = 2*(k_n - 1) + 1 positions

  Every extra layer therefore adds the same r of reach per side --
  the new kernel's outermost tap already sits r columns out, and
  looks through a window extending r further. The growth is
  additive, never multiplicative:

    RF_1 = k_n;   RF_(i+1) = RF_i + (k_n - 1)
    =>  RF_n = k_n + (n - 1)(k_n - 1) = n*(k_n - 1) + 1

               y[p]               layer n
              ╱    ╲
          ..........              the cone widens by r per side,
         ╱          ╲             per layer
    x[p - n*r] .. x[p + n*r]      input window = 2*n*r + 1
                                               = n*(k_n - 1) + 1

  (legend: p = a position along the theta axis; r = (k_n - 1)/2, the
  kernel radius per side; k_n = the per-block kernel width; n =
  n_blocks_cnn, the number of stacked conv blocks; x[.] = an input
  position, h[.] = a layer-1 output position, y[.] = the final output
  position; RF = the receptive field, the count of input positions one
  output sees.)

  (Same-padding does not change the count: where the cone hangs
  past the signal's edge it reads padded zeros, not extra data.)

  Without rescaling, deepening the stack over-grows the view (3
  blocks of k = 11, r = 5, reach 3*5 = 15 per side: RF = 31, wider
  than a whole 26-point bin). This helper instead solves RF >=
  kernel_size for the smallest odd k_n (same-padding needs odd):

    k_n = ceil((kernel_size - 1) / n_blocks_cnn) + 1, then odd-up.

  Extra depth then buys nonlinearity at a fixed total view. It also
  keeps the head size nearly flat: per-block conv parameters scale
  with k_n (C_in*C_out*k_n + C_out), so the head total ~
  C^2 * n * k_n ~ C^2 * (kernel_size - 1 + 2n). At kernel_size 27
  (one bin + margin at the LSST-Y1 run's max_bin = 26):

    n_blocks_cnn : 1    2    3    4    5
    k_n          : 27   15   11   9    7
    RF           : 27   29   31   33   31

  (RF overshoots kernel_size where odd-up rounds; it never
  undershoots.)

  Arguments:
    kernel_size  = the single-block kernel width = the target
                   receptive field (odd).
    n_blocks_cnn = number of stacked conv+activation blocks.

  Returns:
    the per-block kernel width k_n (odd int; = kernel_size when
    n_blocks_cnn is 1).
  """
  require_exact_int(kernel_size, "kernel_size", minimum=1)
  require_exact_int(n_blocks_cnn, "n_blocks_cnn", minimum=1)
  if kernel_size % 2 == 0:
    raise ValueError("kernel_size must be odd; got " + repr(kernel_size))

  # ceil((kernel_size-1)/n) via negated floor division (pure ints,
  # no float rounding), + 1 = the smallest k_n with RF >=
  # kernel_size.
  k = -(-(kernel_size - 1) // n_blocks_cnn) + 1
  # odd-up: same-padding needs an odd kernel to keep the length.
  if k % 2 == 0:
    k += 1
  return k


class FiLMGenerator(nn.Module):
  """
  Predicts a per-channel affine modulation (gamma, beta) from a
  conditioning vector: the generator half of FiLM (Feature-wise
  Linear Modulation; Dumoulin et al. 2018, and the recovered
  design note ai/notes/models-and-designs.md).

    z  (B, n_cond)            conditioning vector (here: the
       │                      non-amplitude whitened parameters)
       │  Linear(n_cond, 2*C)
       ▼
    out (B, 2*C)
       │  split at C
       ▼
    gamma (B, C), beta (B, C)     one scale and one shift per
                                  channel, per sample

  The caller applies them to a feature map h of shape (B, C, L) as

    gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)

  broadcasting over the length axis: the modulation depends on the
  cosmology and the channel, never on the position, because cosmology
  sets a global property of each channel's piece of the data
  vector, not a per-theta local correction. This is what
  re-injects parameter information into a correction head that
  otherwise only ever sees the trunk's output: without FiLM the
  head is one fixed map applied identically at every point of
  parameter space; with it, the cosmology chooses which channels
  to amplify or suppress, and by how much.

  Identity at init: the weight is zeroed and the bias set to
  gamma = 1, beta = 0, so FiLM starts as a no-op for every input,
  the same identity-start convention as the zero-init conv and
  the TRFBlock branches (the model still equals its trunk exactly
  at epoch 1 and at a two-phase handoff). Gradients reach the
  zeroed weight through the inputs, so it wakes as soon as a
  cosmology-dependent modulation helps.

  (legend: B = batch rows; n_cond = conditioning width (the
  factored heads pass the non-amplitude parameter slice, keeping
  the head amplitude-blind so the closed-form amplitude exactness
  survives); C = number of channels to modulate; L = the broadcast
  length axis, max_bin here.)

  Arguments:
    n_cond     = conditioning-vector width.
    n_channels = number of channels C to modulate.

  forward Arguments:
    z = conditioning tensor of shape (B, n_cond); B = batch rows.

  Returns:
    (gamma, beta), each of shape (B, C): the per-channel scale and
    shift for the batch (C = n_channels).
  """
  def __init__(self, n_cond, n_channels):
    """Build the one linear that emits both halves, identity at init.

    Arguments:
      n_cond     = conditioning-vector width.
      n_channels = number of channels C to modulate.
    """
    super().__init__()
    self.n_channels = n_channels
    # one linear producing both halves at once: columns [:C] are
    # gamma, [C:] are beta.
    self.linear = nn.Linear(in_features=n_cond,
                            out_features=2 * n_channels)
    # identity init: zero weight kills the z-dependence, the bias
    # supplies gamma = 1 / beta = 0 (init fns run under no_grad;
    # the slices are views into the one bias parameter).
    nn.init.zeros_(self.linear.weight)
    nn.init.ones_(self.linear.bias[:n_channels])
    nn.init.zeros_(self.linear.bias[n_channels:])

  def forward(self, z):
    """Predict the per-channel modulation for one batch.

    Arguments:
      z = conditioning tensor of shape (B, n_cond); B = batch rows.

    Returns:
      (gamma, beta), each of shape (B, C): the scale and shift halves
      (columns [:C] and [C:]) of the one linear's output.
    """
    out = self.linear(z)
    gamma = out[:, :self.n_channels]    # (B, C)
    beta  = out[:, self.n_channels:]    # (B, C)
    return gamma, beta


class BinLinear(nn.Module):
  """
  G independent Linear(in_features, out_features) layers, one per
  token, run as a single batched einsum instead of a Python loop
  over G modules. The weights stack into (G, in, out), the biases
  into (G, out); token g's rows only ever meet weight[g].

  This is the "unique per token" piece of the ResTRF head: a
  standard transformer applies one shared MLP to every token,
  whereas here each token gets its own weights. The tokens are
  physically distinct (a tomographic bin in plain ResTRF, or a
  (template, bin) pair in the factored version), and the unique
  weights also make them distinguishable to the model, doing the
  job a positional encoding does in a standard transformer, so
  ResTRF needs none.

  These per-token layers live in the correction head, after
  attention has shared information across tokens; the trunk's
  parameter sharing (the expensive cosmology map, learned once) is
  untouched.

  Arguments:
    n_tokens     = number of independent tokens G.
    in_features  = input width per token.
    out_features = output width per token.

  forward Arguments:
    x = input tensor of shape (B, G, in_features); B = batch rows,
        G = n_tokens.

  Returns:
    a tensor of shape (B, G, out_features): token g's slice passed
    through its own weight[g] and bias[g].
  """
  def __init__(self,
               n_tokens,
               in_features,
               out_features):
    """Stack G independently initialized linears into batched tensors.

    Arguments:
      n_tokens     = number of independent tokens G.
      in_features  = input width per token.
      out_features = output width per token.
    """
    require_exact_int(n_tokens, "BinLinear.n_tokens", minimum=1)
    require_exact_int(in_features, "BinLinear.in_features", minimum=1)
    require_exact_int(out_features, "BinLinear.out_features", minimum=1)
    super().__init__()
    # build G ordinary nn.Linear layers just to borrow their init,
    # then stack their weights/biases and discard them. l.weight is
    # (out, in); .t() -> (in, out) for the einsum; stack adds the
    # token axis.
    lins = []
    for _ in range(n_tokens):
      lins.append(nn.Linear(in_features=in_features,
                            out_features=out_features))
    weights, biases = [], []
    for l in lins:
      weights.append(l.weight.detach().t())
      biases.append(l.bias.detach())
    self.weight = nn.Parameter(torch.stack(weights))   # (G, in, out)
    self.bias   = nn.Parameter(torch.stack(biases))    # (G, out)

  def forward(self, x):
    """Apply token g's own weights to token g's slice, all in one go.

    Arguments:
      x = input tensor of shape (B, G, in_features); B = batch rows.

    Returns:
      a tensor of shape (B, G, out_features): every token's private
      matmul plus its own bias, run as one batched einsum.
    """
    # einsum("bgi,gio->bgo", x, weight): g appears in
    # both operands and the output, so it is a batch axis (token g
    # uses weight[g] only, all G in one batched matmul); i appears in
    # both inputs but not the output, so einsum sums over it (the
    # matmul contraction); b and o are kept.
    y = torch.einsum("bgi,gio->bgo", x, self.weight)
    # bias (G, out) broadcasts over the B axis: every sample's token
    # g gets token g's bias.
    return y + self.bias


def validate_trf_token_width(
    *,
    output_length,
    n_tokens,
    token_width):
    """Refuse a transformer token whose LayerNorm loses the input.

    LayerNorm subtracts the mean across a token's feature coordinates. A
    width-one token contains one coordinate, so that coordinate is also the
    mean. The normalized value is zero for every input. Attention and the
    token MLP can then add only an input-independent correction.

    Arguments:
      output_length = number of real output values represented by the tokens.
      n_tokens      = number of token rows passed to the transformer.
      token_width   = maximum number of feature coordinates in one token.

    Returns:
      None when the width is at least 2 (the layout is usable).

    Raises:
      ValueError describing the LayerNorm degeneracy when token_width is 1.
    """
    if token_width >= 2:
        return

    raise ValueError(
        "a transformer correction head requires a maximum token width of "
        "at least 2. The resolved output length is " + str(output_length)
        + ", the token count is " + str(n_tokens)
        + ", and the maximum token width is " + str(token_width)
        + ". LayerNorm over one coordinate subtracts that coordinate "
        "itself, so the transformer correction cannot depend on its input.")


def resolve_padded_head_layout(*, geom, output_dim, where):
  """Validate and copy one geometry's physical padded-head layout.

  A structured head stores unequal physical bins in one rectangular tensor.
  ``head_pad_idx`` maps each value in the model's output order to its exact
  rectangular slot. ``head_valid_mask`` marks the slots that contain physical
  values. Bin counts alone are insufficient because two bins can keep the same
  number of values at different angular coordinates.

  Arguments:
    geom       = geometry carrying ``bin_sizes``, ``head_pad_idx``, and
                 ``head_valid_mask``.
    output_dim = number of physical values produced by the model.
    where      = model name used in refusal messages.

  Returns:
    ``(bin_sizes, pad_idx, valid_mask)``. ``pad_idx`` is one-dimensional.
    ``valid_mask`` has shape ``(1, number of bins, physical width)`` so it
    broadcasts over a model batch.
  """
  if not hasattr(geom, "bin_sizes"):
    raise ValueError(
      where + " needs geom.bin_sizes: prepare the output layout before "
      "building the model")
  sizes = require_positive_int_list(
    geom.bin_sizes, where + ".geom.bin_sizes")

  missing = []
  for name in ("head_pad_idx", "head_valid_mask"):
    if not hasattr(geom, name):
      missing.append(name)
  if missing:
    raise ValueError(
      where + " needs the geometry's persisted physical padded-head layout "
      "(" + ", ".join(missing) + "). Bin counts cannot recover angular "
      "positions. Rebuild the geometry from current data, or retrain an old "
      "structured-head artifact")

  raw_idx = torch.as_tensor(getattr(geom, "head_pad_idx"))
  if raw_idx.ndim != 1 or raw_idx.dtype == torch.bool \
      or raw_idx.dtype.is_floating_point or raw_idx.dtype.is_complex:
    raise ValueError(
      where + ".geom.head_pad_idx must be a one-dimensional integer map")
  pad_idx = raw_idx.to(dtype=torch.long).detach().clone()

  raw_valid = torch.as_tensor(getattr(geom, "head_valid_mask"))
  if raw_valid.ndim != 2:
    raise ValueError(
      where + ".geom.head_valid_mask must have shape "
      "(number of bins, physical width)")
  if raw_valid.dtype == torch.bool:
    valid = raw_valid.detach().clone()
  elif raw_valid.dtype == torch.uint8:
    binary = torch.logical_or(raw_valid == 0, raw_valid == 1)
    if not bool(torch.all(binary).item()):
      raise ValueError(
        where + ".geom.head_valid_mask uint8 values must be 0 or 1")
    valid = raw_valid.to(dtype=torch.bool).detach().clone()
  else:
    raise ValueError(
      where + ".geom.head_valid_mask must contain booleans or persisted "
      "uint8 zeros and ones")

  n_bins = int(valid.shape[0])
  if n_bins < 1:
    raise ValueError(
      where + ".geom.head_valid_mask must contain at least one physical bin")
  physical_width = int(valid.shape[1])
  if physical_width < 1:
    raise ValueError(
      where + ".geom.head_valid_mask must contain at least one coordinate")

  valid_cpu = valid.detach().cpu()
  row_sizes = []
  nonempty_sizes = []
  for bin_index in range(n_bins):
    size = int(valid_cpu[bin_index].sum().item())
    row_sizes.append(size)
    if size > 0:
      nonempty_sizes.append(size)
  if nonempty_sizes != sizes:
    raise ValueError(
      where + ".geom.bin_sizes must equal the nonempty rows of "
      "head_valid_mask in physical order")

  expected_values = sum(row_sizes)
  if output_dim != expected_values:
    raise ValueError(
      where + ".output_dim is " + str(output_dim)
      + ", but geom.bin_sizes contains " + str(expected_values)
      + " physical values")
  if int(pad_idx.numel()) != output_dim:
    raise ValueError(
      where + ".geom.head_pad_idx has " + str(int(pad_idx.numel()))
      + " entries, but output_dim is " + str(output_dim))

  idx_cpu = pad_idx.detach().cpu()
  rectangle_size = n_bins * physical_width
  if bool(torch.any(idx_cpu < 0).item()) \
      or bool(torch.any(idx_cpu >= rectangle_size).item()):
    raise ValueError(
      where + ".geom.head_pad_idx contains a slot outside the validity mask")
  if int(torch.unique(idx_cpu).numel()) != output_dim:
    raise ValueError(
      where + ".geom.head_pad_idx must map every physical value to a "
      "different slot")

  flat_valid = valid_cpu.reshape(-1)
  if not bool(torch.all(flat_valid[idx_cpu]).item()):
    raise ValueError(
      where + ".geom.head_pad_idx points to a slot marked artificial")
  if int(flat_valid.sum().item()) != output_dim:
    raise ValueError(
      where + ".geom.head_valid_mask must mark exactly output_dim physical "
      "slots")
  mask_positions = torch.nonzero(flat_valid, as_tuple=False).reshape(-1)
  if not torch.equal(torch.sort(idx_cpu).values, mask_positions):
    raise ValueError(
      where + ".geom.head_pad_idx and head_valid_mask describe different "
      "physical slots")
  return row_sizes, pad_idx, valid.unsqueeze(0)


def keep_valid_head_positions(values, valid_mask):
  """
  Zero the artificial slots of a padded token rectangle.

  A ragged physical layout stores unequal bins in one rectangular tensor,
  so some slots hold no physical value. Every operation in the TRF head
  calls this afterward, keeping those slots at exact zero so padding can
  never leak into attention scores, normalization statistics, or the
  returned correction.

  Arguments:
    values     = a token tensor of shape (B, G, dim); B = batch rows,
                 G = tokens, dim = the padded token width.
    valid_mask = Boolean mask of shape (1, G, dim), True on physical
                 slots, broadcast over the batch; or None for a fully
                 rectangular layout.

  Returns:
    values with every artificial slot set to exact zero; when valid_mask
    is None, the input unchanged (the rectangular path stays
    bit-for-bit identical).
  """
  if valid_mask is None:
    return values
  return values.masked_fill(torch.logical_not(valid_mask), 0)


def _masked_layer_norm(values, valid_mask, layer):
  """
  Apply LayerNorm over the physical feature coordinates only.

  A padded token's artificial zeros must not enter the mean and variance:
  they would dilute both and make the normalization depend on how much
  padding a bin carries. The statistics are therefore computed with the
  mask as a weight (sum over physical slots / count of physical slots),
  and the layer's learned affine is applied afterward exactly as
  nn.LayerNorm would.

  A token with ONE physical coordinate is left unnormalized: subtracting
  its own mean would zero it for every input, silently making the branch
  input-independent (the same degeneracy validate_trf_token_width refuses
  for a whole layout).

  Arguments:
    values     = a token tensor of shape (B, G, dim).
    valid_mask = Boolean mask of shape (1, G, dim), True on physical
                 slots; or None to apply the plain nn.LayerNorm.
    layer      = the nn.LayerNorm module whose eps and learned
                 weight / bias are used.

  Returns:
    the normalized tensor, shape (B, G, dim), with artificial slots
    zeroed; when valid_mask is None, exactly layer(values).
  """
  if valid_mask is None:
    return layer(values)

  mask = valid_mask.to(dtype=values.dtype)
  count = mask.sum(dim=-1, keepdim=True)
  safe_count = torch.clamp(count, min=1.0)
  mean = (values * mask).sum(dim=-1, keepdim=True) / safe_count
  centered = values - mean
  variance = ((centered.square() * mask).sum(dim=-1, keepdim=True)
              / safe_count)
  normalized_many = centered * torch.rsqrt(variance + layer.eps)
  # One physical coordinate has no meaningful variance. Keeping that
  # coordinate unnormalized retains its input dependence; subtracting its
  # own mean would silently turn the Transformer branch into a constant.
  normalized = torch.where(count == 1, values, normalized_many)
  if layer.elementwise_affine:
    normalized = normalized * layer.weight + layer.bias
  return keep_valid_head_positions(normalized, valid_mask)


class TRFBlock(nn.Module):
  """
  One transformer block over tokens at their natural width: no
  embedding in, no projection out, because the tokens are the
  (padded) physical bin segments themselves, so dim = max_bin, the
  padded bin length. (A learned embedding is what a transformer needs
  when its sequence is synthetic, i.e. a flat latent vector split
  into tokens; here the sequence structure is physical, so the
  adapter layers and their parameters are simply not needed.)
  Self-attention across the G tokens, then a per-token MLP branch,
  both pre-norm residual branches, as in a standard pre-LN
  transformer:

    x  (B, G, dim)             G tokens (bins) of width dim
       │  LayerNorm; wq / wk / wv        (shared across tokens)
       ▼
    q, k, v  (B, G, H, d_head)           H heads, d_head = dim/H
       │  scores = q.k / sqrt(d_head); softmax over the key axis
       ▼
    att  (B, H, G, G)          per head: each query bin's weights
       │                       over all key bins
       │  att @ v; merge heads; wo       (wo zero-initialized)
       ▼
    x + attention branch
       │  LayerNorm; n_mlp_blocks x [BinLinear + act]
       │                                 (last layer zero-init)
       ▼
    x + MLP branch             the block's output (= x at init)

  (legend: B = batch rows; G = n_tokens, the number of tokens; dim
  = the per-token width; H = n_heads; d_head = dim/H, the feature
  slice each head works in.)

  Two deliberate deviations from the textbook block:
  - the tokens are physical: a tomographic bin's theta segment
    (plain ResTRF) or a (template, bin) pair's (the factored
    version), so attention shares information across bins (the
    cross-bin correlations a within-bin conv cannot see);
  - the position-wise MLP is not shared (by default): each token
    has its own n_mlp_blocks-deep stack (BinLinear), where a
    standard transformer applies one shared MLP to every token.
    The unique weights specialize each token's correction and
    stand in for the positional encoding (see BinLinear).
    shared_mlp=True shares this fixed-width MLP across tokens. A
    textbook transformer often expands the hidden width before
    projecting it back; this implementation keeps ``dim -> dim`` at
    every MLP layer. The shared option is the ablation baseline for
    token-specific weights. Caveat: with the
    MLP shared (and the attention maps always shared), nothing in
    the block tells the tokens apart structurally, so the head
    becomes permutation-equivariant over tokens, with no
    positional encoding; token identity then comes only from the
    segments' content.

  The attention projections (wq / wk / wv / wo) are shared across
  tokens, as in any transformer: shared maps are what let every
  token attend to every other with one set of weights; the
  per-token specialization lives in the MLPs.

  The block is exactly the identity at init: both branch outputs
  (wo and the last MLP layer) are zero-initialized, so x passes
  through untouched. A stack of these blocks therefore satisfies
  blocks(x) == x at init, which is what lets the ResTRF head
  define its correction as blocks(h) - h == 0, the zero-init
  identity start, with no output projection to host it. Gradients
  still reach the zeroed layers (a layer's weight gradient depends
  on its inputs, not on its own weights); the layers behind them
  wake one step later.

  LayerNorm (not the package's Affine) opens both branches: the
  softmax's saturation depends on the score scale, so attention
  wants its inputs actively normalized, and pre-LN is the
  stable-training default for transformers.

  Arguments:
    dim          = token width = max_bin, the padded bin length
                   (must be divisible by n_heads; the LSST-Y1
                   cosmic-shear run keeps max_bin = 26 theta
                   points per bin, allowing n_heads = 1, 2, or
                   13).
    n_tokens     = number of tokens G.
    n_heads      = attention heads (each head attends over all G
                   tokens with dim/n_heads of the features).
    n_mlp_blocks = depth of each token's private MLP stack; every
                   layer runs at the token width (dim -> dim), the
                   interior width pinned to the bin length by design
                   (no width knob, n_mlp_blocks sets depth only).
    act          = activation factory act(dim) -> module for the
                   MLP layers (the run's activation; defaults to
                   activation_fcn, the paper's H).
    shared_mlp   = False (default): per-token unique fixed-width MLPs
                   (BinLinear). True: one fixed-width MLP shared by every
                   token through plain nn.Linear applied position-wise.
                   This changes weight sharing, not the hidden width.
    output_length = number of real output values represented by the tokens.
                    Model constructors provide this value. A direct block
                    call may omit it when its token rectangle contains no
                    padding.

  forward Arguments:
    x = input tensor of shape (B, G, dim); B = batch rows, G =
        n_tokens, dim = the per-token width.

  Returns:
    a tensor of shape (B, G, dim): the block's output, equal to x
    at init (both residual branches are zero-initialized).
  """
  def __init__(self,
               dim,
               n_tokens,
               n_heads=2,
               n_mlp_blocks=2,
               act=activation_fcn,
               shared_mlp=False,
               output_length=None):
    """Validate the token layout and build both residual branches.

    Arguments:
      dim           = token width (the padded bin length); must be
                      divisible by n_heads.
      n_tokens      = number of tokens G.
      n_heads       = attention heads.
      n_mlp_blocks  = depth of each token's MLP stack.
      act           = activation factory act(dim) -> module.
      shared_mlp    = False for per-token unique MLPs (BinLinear);
                      True for one MLP shared position-wise.
      output_length = number of real output values the tokens
                      represent; None means the token rectangle is
                      exact (n_tokens * dim, no padding).
    """
    require_exact_int(dim, "TRFBlock.dim", minimum=1)
    require_exact_int(n_tokens, "TRFBlock.n_tokens", minimum=1)
    require_exact_int(n_heads, "TRFBlock.n_heads", minimum=1)
    require_exact_int(
      n_mlp_blocks, "TRFBlock.n_mlp_blocks", minimum=1)
    require_exact_bool(shared_mlp, "TRFBlock.shared_mlp")
    if output_length is None:
      output_length = n_tokens * dim
    require_exact_int(output_length, "TRFBlock.output_length", minimum=1)
    validate_trf_token_width(
      output_length=output_length,
      n_tokens=n_tokens,
      token_width=dim)
    super().__init__()
    if dim % n_heads != 0:
      raise ValueError(
        f"the token width ({dim} = the padded bin length) must be "
        f"divisible by n_heads ({n_heads})")
    require_live_head_activation(act, "TRFBlock.act")
    self.n_heads = n_heads
    self.d_head  = dim // n_heads

    # attention branch: pre-norm, shared Q/K/V/output projections.
    self.ln_att = nn.LayerNorm(normalized_shape=dim)
    self.wq = nn.Linear(in_features=dim, out_features=dim)
    self.wk = nn.Linear(in_features=dim, out_features=dim)
    self.wv = nn.Linear(in_features=dim, out_features=dim)
    self.wo = nn.Linear(in_features=dim, out_features=dim)

    # MLP branch: pre-norm, n_mlp_blocks layers each dim -> dim (the
    # interior width is pinned to the token width, no width knob), each
    # its own activation instance. Per-token unique (BinLinear) by default;
    # with shared_mlp one nn.Linear serves every token. A Linear on
    # a (B, G, dim) tensor applies position-wise to the last axis.
    # Every layer remains dim -> dim; there is no expansion projection.
    self.ln_mlp = nn.LayerNorm(normalized_shape=dim)
    lins, acts = [], []
    for _ in range(n_mlp_blocks):
      if shared_mlp:
        lins.append(nn.Linear(in_features=dim, out_features=dim))
      else:
        lins.append(BinLinear(n_tokens=n_tokens,
                              in_features=dim,
                              out_features=dim))
      acts.append(act(dim))
    self.mlp_lins = nn.ModuleList(lins)
    self.mlp_acts = nn.ModuleList(acts)

    # identity at init: zero both branch outputs (see docstring).
    # The final MLP activation maps 0 -> 0 (H(x) = gate(x)*x), so a
    # zeroed last layer silences the whole branch.
    nn.init.zeros_(self.wo.weight)
    nn.init.zeros_(self.wo.bias)
    nn.init.zeros_(self.mlp_lins[-1].weight)
    nn.init.zeros_(self.mlp_lins[-1].bias)

  def forward(self, x, valid_mask=None):
    """Run the attention and MLP branches over one token batch.

    Arguments:
      x          = input tensor of shape (B, G, dim): B batch rows of
                   G tokens, each dim wide.
      valid_mask = the Boolean feature mask of a ragged physical
                   layout, or None for a rectangle with no padding.

    Returns:
      a tensor of shape (B, G, dim); equal to x at init, because both
      residual branches start zero-initialized.
    """
    # A ragged physical layout passes a Boolean feature mask. Invalid
    # coordinates are removed before normalization, all attention
    # projections, every MLP operation, and both residual returns. A
    # rectangular layout passes None and keeps the original operations
    # byte-for-byte.
    x = keep_valid_head_positions(x, valid_mask)
    B, G, _ = x.shape

    # --- attention branch (pre-LN residual) ---
    h = _masked_layer_norm(x, valid_mask, self.ln_att)
    # split the feature axis into heads: (B, G, dim) -> (B, G, H,
    # d_head); view is free (the Linear output is contiguous).
    q = keep_valid_head_positions(self.wq(h), valid_mask)
    k = keep_valid_head_positions(self.wk(h), valid_mask)
    v = keep_valid_head_positions(self.wv(h), valid_mask)
    q = q.view(B, G, self.n_heads, self.d_head)
    k = k.view(B, G, self.n_heads, self.d_head)
    v = v.view(B, G, self.n_heads, self.d_head)
    # attention scores, einsum("bghd,bkhd->bhgk"): d is contracted
    # (the query-key dot product), b and h are batch axes, and the
    # kept g (query bin) x k (key bin) pair is the GxG attention
    # matrix per head. Divided by sqrt(d_head) so the dot products
    # stay O(1) and the softmax does not saturate at init.
    att = torch.einsum("bghd,bkhd->bhgk", q, k) / self.d_head ** 0.5
    overlap = None
    if valid_mask is not None:
      # Padding occupies the feature axis, not the token axis. Build one
      # query/key permission matrix per attention head from the physical
      # feature overlap. An all-padding token is never a key or value, and a
      # head with no common physical coordinate receives no attention update.
      feature_valid = valid_mask.reshape(
        1, G, self.n_heads, self.d_head)
      query_valid = feature_valid.permute(0, 2, 1, 3).unsqueeze(3)
      key_valid = feature_valid.permute(0, 2, 1, 3).unsqueeze(2)
      overlap = torch.logical_and(query_valid, key_valid).any(dim=-1)
      att = att.masked_fill(torch.logical_not(overlap), -torch.inf)
    # softmax over the key axis: each query bin's weights over all
    # bins sum to 1.
    if overlap is None:
      att = torch.softmax(att, dim=-1)
    else:
      has_key = overlap.any(dim=-1, keepdim=True)
      safe_scores = torch.where(has_key, att, torch.zeros_like(att))
      att = torch.softmax(safe_scores, dim=-1)
      att = torch.where(has_key, att, torch.zeros_like(att))
    # weighted sum of the value tokens, einsum("bhgk,bkhd->bghd"):
    # k is contracted against each query's attention row; the
    # result is one mixed d_head vector per (query bin, head).
    out = torch.einsum("bhgk,bkhd->bghd", att, v)
    # merge the heads back: (B, G, H, d_head) -> (B, G, dim).
    # reshape (not view): the einsum output need not be contiguous.
    out = out.reshape(B, G, self.n_heads * self.d_head)
    out = keep_valid_head_positions(out, valid_mask)
    attention_update = keep_valid_head_positions(self.wo(out), valid_mask)
    x = keep_valid_head_positions(x + attention_update, valid_mask)

    # --- per-bin MLP branch (pre-LN residual) ---
    h = _masked_layer_norm(x, valid_mask, self.ln_mlp)
    n = len(self.mlp_lins)
    for i in range(n):
      h = keep_valid_head_positions(self.mlp_lins[i](h), valid_mask)
      h = keep_valid_head_positions(self.mlp_acts[i](h), valid_mask)
    return keep_valid_head_positions(x + h, valid_mask)
