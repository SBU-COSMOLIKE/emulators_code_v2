"""Standard emulator models (ResMLP, ResCNN, ResTRF).

The plain member of the emulator/designs/ family: full networks
mapping whitened cosmological parameters to the whitened training
targets. Where this file sits in the training pipeline:

  cosmological parameters
     │   geometries/parameter.py  center, rotate, unit-scale (whiten in)
     ▼
  whitened inputs
     │   designs/plain.py         ResMLP, ResCNN, or ResTRF (this file)
     ▼
  whitened data vector
     │   geometries/output.py     un-whiten + scatter to full length
     ▼
  physical residual vs truth
     │   losses/core.py           contract with the inverse covariance
     ▼
  chi2 = r^T Cinv r

(legend: each box is the data at that stage and the file on each
arrow does the transform; r = the physical residual, prediction
minus truth, scattered to full 3x2pt length; Cinv = the masked
inverse covariance; chi2 = r^T Cinv r, the Mahalanobis distance.)

The diagram is drawn for cosmic shear; a CMB spectrum, a scalar set,
or a background / power-spectrum grid rides the same trunk with its
own geometry and loss — the networks here never know which family
they serve.

ResMLP is the baseline: input projection, a stack of identical
ResBlocks, output projection, final Affine. ResCNN and ResTRF add a
correction appendix on a ResMLP trunk: the trunk predicts in the full
(cov-eigenbasis) whitening, fixed buffers map its output into theta
order, a structured head corrects it there, a 1D conv along the
angular axis (ResCNN), or a transformer whose tokens are the
tomographic bins (ResTRF), and a learnable gate adds the correction
back, so swapping the architecture changes only the model.

The heads ride the diagonal family geometries too (cmb / grid /
grid2d; motivated by
arXiv 2505.22574's attention-vs-MLP outlier result for CMB spectra).
Those geometries whiten per element IN physical order (ell / z /
z-slices x k), so the trunk's prediction is already in the head's
local basis: the basis-change maps degenerate to the identity (the
W_fd / W_df buffers stay None and forward skips both matmuls), and
the channel/token split comes from the geometry's
attach_head_coords() instead of build_shear_angle_map. A scalar
output has no coordinate axis between its named values, so the
scalar family stays trunk-only (its guard in experiment.py).

Each class mixes in DesignSpec: a head_block class attribute (None /
"cnn" / "trf") plus a shared describe_spec classmethod make the class the
single source of its own head-knowledge, read alike by build_specs,
build_geometry, and the startup banner; an architecture that omits
head_block fails at class-definition time.

Whitened = rotated into the covariance eigenbasis and scaled to unit
variance under the covariance that defines the transform. This gives
decorrelated coordinates on comparable numerical scales, while learning
difficulty can still differ among directions;
done by the geometry classes (geometries.parameter /
geometries.output).
"""

import torch
import torch.nn as nn

from ..activations import (
  activation_factory_recipe, activation_fcn, require_live_head_activation)
from ..model_recipe import record_model_recipe
from ..validation import (
  require_exact_bool, require_exact_int, require_nonzero_float32)
from .blocks import (
  Affine, ResBlock, TRFBlock, FiLMGenerator, keep_valid_head_positions,
  materialized_block_recipe, rescale_kernel_size, resolve_padded_head_layout,
  validate_trf_token_width)


class DesignSpec:
  """
  Shared spec machinery for the emulator design classes.

  Every design class declares a head_block class attribute (None for a
  trunk-only model, "cnn" / "trf" for the two correction-head families);
  __init_subclass__ enforces it at class-definition time, so a new
  architecture that forgets it fails loudly at import, not silently in a
  banner. head_block is the single source of head-knowledge: build_specs
  (experiment.py) reads it to skip the inactive head's YAML block, and
  describe_spec renders the banner's model-spec line. The models mix this in
  beside nn.Module (a plain base, no __init__, so nn.Module's method
  resolution is unchanged).
  """

  def __init_subclass__(cls, **kwargs):
    """Require every design class to declare head_block at definition.

    Runs automatically when a subclass is defined, so a missing
    head_block is a class-definition-time error, not a silent
    trunk-only default.

    Arguments:
      **kwargs = forwarded unchanged to the next __init_subclass__.
    """
    super().__init_subclass__(**kwargs)
    if "head_block" not in cls.__dict__:
      raise TypeError(
        f"{cls.__name__} must declare a head_block class attribute "
        f"(None | 'cnn' | 'trf'): it is the single source of "
        f"head-knowledge (build_specs / describe_spec)")

  @classmethod
  def describe_spec(cls, model_block):
    """
    Render the model-spec banner line: only the keys this class consumes.

    A shared YAML carries cnn: and trf: for every architecture (so one file
    switches models by name: alone), but the banner shows the truth about
    this run, so the inactive head block is dropped. Renders, in order,
    name / ia / mlp / activation / norm / this class's own head block
    (head_block) / compile_mode, each only when present in model_block.

    Arguments:
      model_block = the raw train_args["model"] mapping.

    Returns:
      a str: the dict of the consumed keys, in display order.
    """
    order = ["name", "ia", "mlp", "activation", "norm"]
    if cls.head_block is not None:
      order.append(cls.head_block)
    order.append("compile_mode")
    shown = {}
    for key in order:
      if key in model_block:
        shown[key] = model_block[key]
    return str(shown)


class ResMLP(DesignSpec, nn.Module):
  """
  The baseline emulator: input projection, a stack of identical
  residual blocks, output projection, final learnable affine.

    x  (B, input_dim)     whitened cosmological parameters
       │  Linear: input_dim -> int_dim_res
       ▼
       │  n_blocks x ResBlock, all at width int_dim_res
       ▼
       │  Linear: int_dim_res -> output_dim;  Affine
       ▼
    dv (B, output_dim)    whitened data vector

  (legend: B = batch rows; input_dim = number of cosmological
  parameters; int_dim_res = internal residual width; output_dim =
  whitened data-vector length.)

  Arguments:
    input_dim   = number of cosmological parameters
    output_dim  = length of the data vector
    int_dim_res = internal (residual) width
    n_blocks    = number of residual blocks
    block_opts  = dict of ResBlock options (n_layers,
                   norm, act), the same for every block

  block_opts defaults to None, not {}: a default argument is
  created once and shared across calls, so a mutable dict would
  leak between them. All blocks share one configuration, capping
  the hyperparameter count.
  """
  head_block = None                # trunk only, no correction head

  def __init__(self,
               input_dim,
               output_dim,
               int_dim_res,
               n_blocks=3,
               block_opts=None):
    """Validate the sizes and assemble the Linear/ResBlock/Affine stack.

    Arguments:
      input_dim   = whitened-parameter width.
      output_dim  = data-vector width.
      int_dim_res = internal (residual) width.
      n_blocks    = number of residual blocks.
      block_opts  = the ResBlock options mapping shared by every
                    block, or None for the defaults.
    """
    require_exact_int(input_dim, "ResMLP.input_dim", minimum=1)
    require_exact_int(output_dim, "ResMLP.output_dim", minimum=1)
    require_exact_int(int_dim_res, "ResMLP.int_dim_res", minimum=1)
    require_exact_int(n_blocks, "ResMLP.n_blocks", minimum=0)
    if block_opts is not None and not isinstance(block_opts, dict):
      raise TypeError("ResMLP.block_opts must be a mapping or None")
    super().__init__()

    # Default to {} (not in the signature: a mutable default is
    # created once and would leak between calls).
    if block_opts is None:
      block_opts = {}
    self.model_recipe = record_model_recipe(
      class_path=type(self).__module__ + "." + type(self).__qualname__,
      name="resmlp", ia=None, input_dim=input_dim, output_dim=output_dim,
      needs_geom=False,
      kwargs={
        "int_dim_res": int(int_dim_res),
        "n_blocks": int(n_blocks),
        "block_opts": materialized_block_recipe(block_opts),
      })
    layers = []

    # param dim -> internal width
    layers.append(nn.Linear(in_features=input_dim, out_features=int_dim_res))

    # n_blocks identical residual blocks at the internal width;
    # **block_opts unpacks the dict into keyword args per ResBlock.
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))

    # internal width -> data-vector dim
    layers.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))

    # final learnable scale and shift
    layers.append(Affine())

    # Sequential registers every module, so the temporary list is fine.
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    """Run the whole stack on one batch.

    Arguments:
      x = whitened parameters of shape (B, input_dim).

    Returns:
      the whitened prediction of shape (B, output_dim).
    """
    return self.model(x)


class ResCNN(DesignSpec, nn.Module):
  """
  ResMLP trunk + a bins-as-channels 1D CNN correction appendix. The
  trunk is identical to the standalone ResMLP and predicts in the
  full (cov-eigenbasis) whitened basis, so its loss stays the
  well-conditioned chi2 = ||pred - target||^2 (identity Hessian).
  The forward pass, shapes and all:

    x  (B, input_dim)           whitened parameters
       │  self.mlp              the ResMLP trunk
       ▼
    y  (B, n_keep)              full-whitened dv (also the skip)
       │  @ W_fd                f -> d: theta order, /sigma
       ▼
    h  (B, n_keep)
       │  pad_idx scatter       original physical slots + validity mask
       ▼
    c  (B, n_bins, max_bin)     tomographic bins = conv channels
       │  n_blocks_cnn x [Conv1d + act]
       ▼
       │  pad_idx gather        drop the pad slots
       ▼
    corr (B, n_keep)
       │  @ W_df                d -> f, back to full whitening
       ▼
    out = y + gate * corr       corr = 0 at init (identity start)

  (legend: B = batch rows; n_keep = kept data-vector length, the
  unmasked entries the model emulates = output_dim; n_bins = number
  of physical tomographic (xi+/-, source-pair) bins, including an
  entirely masked row; max_bin = the full physical angular width of
  the padded rectangle; f / d = the
  full-whitened / diagonal-theta bases, see the W_fd buffers below.)

  The CNN is an additive correction in the diagonal view (theta
  order, per-element /sigma; the full-whitened basis scrambles the
  angular order, so a conv there has no locality). The theta-order
  dv splits into its (xi+/-, source-pair) tomographic bins, and the
  bins become the conv's channels: one Conv1d(n_bins -> n_bins,
  kernel_size) slides a single kernel along theta over the whole
  data vector at once. At every theta position each output bin
  reads a kernel_size-wide window of all the bins, theta-local
  and cross-bin (the bins share one angular grid, so channel mixing
  couples different bins at like angular scales, up to per-bin mask
  offsets). No channel expansion: the head's tensors never grow
  beyond the (padded) dv size, so the bandwidth wall the old
  expand-to-C-filters head hit cannot occur by construction. Each
  block is one conv + one activation (the nonlinearity between
  stacked blocks, without it two convs fold into a single
  kernel). The head hyperparameters are kernel_size (+ the
  rescale_kernel flag), n_blocks_cnn, groups, separable, film, and
  gate_init.

  groups restricts that channel mixing along the one physical cut
  the channel order offers. The channels are the bins in dv order
  the xi+ pairs first, then the xi- pairs (cosmolike's layout,
  reconstructed by build_shear_angle_map), and a grouped conv
  splits the channels into `groups` consecutive blocks that never
  mix:

    channels:   xi+ pair 1 .. P │ xi- pair 1 .. P
                                │
    groups=1:   no cut, every output bin reads every bin (the
                default: full cross-bin mixing)
    groups=2:   cut at the │, xi+ never mixes with xi-, but
                bins still mix freely within their branch

  (legend: P = n_bins/2 source pairs per xi branch; per-block conv
  parameters = n_bins * (n_bins/groups) * kernel_size + n_bins, so
  the cut also halves the head's conv weights. The boundary is
  validated against geom.pm_kept and the physical coordinate map.
  A wholly masked row remains in that map, so it cannot silently
  shift a later bin across the xi+ / xi- cut.)

  separable factors the remaining sum's two jobs, smoothing
  along theta and mixing channels, into two cheaper layers per
  block:

    c  (B, C, max_bin)
       │  depthwise Conv1d(C -> C, k, groups=C): each channel its
       │  own k-tap theta filter, no mixing        C*k weights
       ▼
       │  pointwise Conv1d(C -> C, 1, groups=groups): mixes the
       │  channels at each theta position          C*(C/groups)
       ▼
    act  (one activation per block, after the pointwise)

  versus the plain block's joint C*(C/groups)*k. No activation
  sits between the two layers, so the pair composes into a single
  constrained conv, weights w[o, c, t] = pointwise[o, c] *
  depthwise[c, t], i.e. a low-rank factorization of the plain
  block's sum, not a different operation. The assumption the
  factorization adds: the theta-smoothing profile a channel needs
  does not depend on which channel it mixes into (plausible for
  covariance-driven leakage at like angular scales; the standard
  depthwise-separable trade). The zero-init identity start moves
  to the last block's pointwise layer.

  (legend: C = n_bins, the channels; k = the per-block kernel
  width after any rescale; B = batch rows.)

  Bins differ in kept length, so each occupies one row of a
  physical-width rectangle. pad_idx scatters each survivor into its
  original angular slot, while pad_valid marks the storage-only
  cells. The mask is restored after every operation that can make an
  artificial cell nonzero, and pad_idx gathers only physical
  corrections. build_shear_angle_map attaches this layout before
  model construction.

  The head starts as an exact identity: the last conv is
  zero-initialized, so corr = 0 and the model equals its trunk at
  epoch 1 (the zero-init-residual-branch start; gradients reach the
  zeroed conv through the nonzero gate at step 1).

  The two basis-change maps are precomputed and stored as fixed
  buffers, named for the bases: f = full-whitened (the eigenbasis
  the trunk predicts in), d = diagonal (theta order, each element
  scaled by its marginal sigma). Subscripts read in multiply order:
  y_full @ W_fd goes f -> d, correction @ W_df goes d -> f (W_df =
  W_fd inverse). Buffers, not live geometry calls in forward, stay
  safe under torch.compile CUDA graphs.

  Target and loss use the full-whitening DataVectorGeometry, as the
  standalone ResMLP, so swapping ResMLP -> ResCNN changes the model
  only, not the whitening (no confound).

  On a diagonal family geometry (cmb / grid / grid2d) the same head
  applies unchanged, minus the basis change:
  those geometries whiten per element in physical order, so the
  trunk already predicts in the head's local basis and W_fd / W_df
  stay None (forward skips both matmuls — no n_keep x n_keep
  identity buffers). The channel split is the geometry's
  attach_head_coords(): cmb and grid expose ONE channel (the kernel
  slides along ell / z), grid2d one channel per z slice (the kernel
  slides along k; channel mixing couples z slices at like k).
  groups=2 stays cosmic-shear-only — the xi+/xi- cut is the one
  physical channel boundary this head knows.

  Arguments:
    input_dim    = number of cosmological parameters.
    output_dim   = data-vector length to emulate (= n_keep).
    int_dim_res  = internal width of the residual trunk.
    geom         = the output geometry carrying bin_sizes, the
                   physical padded-head map, and its validity mask
                   (attached by build_shear_angle_map on the
                   cosmolike geometry, by attach_head_coords() on a
                   diagonal family geometry). With an eigenbasis (evecs /
                   sqrt_ev) it also defines the basis buffers;
                   without one the head works in the geometry's own
                   physical order (see the class docstring).
    kernel_size  = conv kernel width (odd, same-padded), tuned as
                   if the head had one block. With rescale_kernel
                   it states the target receptive field; without,
                   it is used verbatim for every block.
    rescale_kernel = False (default): every block uses kernel_size
                   as given. True: the per-block kernel shrinks
                   with depth so the n_blocks_cnn-deep stack keeps
                   a single kernel_size-wide block's view,
                   receptive field n*(k-1)+1 >= kernel_size, see
                   rescale_kernel_size. Depth then buys
                   nonlinearity at a fixed total view (and
                   near-flat head parameters) instead of
                   over-growing the receptive field. The resolved
                   width is stored as self.kernel_size.
    groups       = channel-mixing restriction: 1 (default, dense
                   mixing) or 2 (xi+ never mixes with xi-). See
                   the groups paragraph and graph in the class
                   docstring; other values error, and the xi
                   boundary is validated against geom.pm_kept at
                   build.
    separable    = False (default): one joint conv per block.
                   True: factor each block into a depthwise
                   theta filter + a pointwise channel mix. This is a
                   constrained factorization of the joint convolution.
                   Its parameter count follows the executed channel,
                   group, kernel, and bias shapes. See the separable
                   paragraph and graph in the class docstring.
    film         = False (default): the head is one fixed map,
                   blind to the cosmology. True: re-inject the
                   parameters into every block as a per-channel
                   affine, conv -> gamma(x)*c + beta(x) -> act,
                   with one identity-initialized FiLMGenerator
                   per block (Linear(input_dim, 2*n_bins),
                   ~2*n_bins*input_dim parameters each, see the
                   generator's docstring and
                   ai/notes/models-and-designs.md). The gate stays as
                   the outer valve; FiLM modulates inside the
                   blocks, letting the cosmology choose which
                   bins to amplify and by how much.
    n_blocks     = residual blocks in the trunk.
    n_blocks_cnn = stacked conv+activation correction blocks.
    gate_init    = initial value of the scalar scaling the
                   correction. Small (default 0.1) to start near the
                   pure ResMLP; not 0, a 0 gate strands the CNN
                   with no gradient, so it never learns.
    head_act     = the CNN head's own activation factory (None ->
                   share block_opts["act"], the trunk's
                   family). build_specs builds it
                   from model.cnn.activation (or the head: activation:
                   alias); set, it pins the head family only.
    block_opts   = ResBlock options (None -> {}); its "act" is the
                   trunk family and, unless head_act is set, the CNN
                   head's too. Defaults to activation_fcn (the paper's
                   H) when block_opts sets no "act".

  needs_geom / needs_bins are capability flags EmulatorExperiment
  reads: geom injected (basis buffers + bin sizes), compile_mode
  defaulted to "default", and build_shear_angle_map run on the data
  geometry before the model is built.
  """
  needs_geom = True
  needs_bins = True
  head_block = "cnn"               # the bins-as-channels conv head

  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               kernel_size=11, rescale_kernel=False, groups=1,
               separable=False, film=False, n_blocks=3,
               n_blocks_cnn=1, gate_init=0.1, head_act=None,
               block_opts=None):
    """Validate every option, then build the trunk and the conv head.

    The class docstring's Arguments block defines each option in
    depth; one line each here:

    Arguments:
      input_dim      = whitened-parameter width.
      output_dim     = data-vector length to emulate (n_keep).
      int_dim_res    = trunk residual width.
      geom           = output geometry carrying the padded per-bin
                       layout (and the basis buffers, when any).
      kernel_size    = odd conv kernel width.
      rescale_kernel = True shrinks per-block kernels with depth at a
                       fixed total receptive field.
      groups         = channel mixing: 1 dense, 2 = the xi+/xi- split.
      separable      = True factors each conv into a depthwise filter
                       plus a pointwise channel mix.
      film           = True re-injects the cosmology as per-channel
                       modulation inside each block.
      n_blocks       = trunk residual blocks.
      n_blocks_cnn   = conv correction blocks.
      gate_init      = starting value of the correction gate.
      head_act       = the head's own activation factory; None shares
                       the trunk's.
      block_opts     = ResBlock options mapping, or None for {}.
    """
    require_exact_int(input_dim, "ResCNN.input_dim", minimum=1)
    require_exact_int(output_dim, "ResCNN.output_dim", minimum=1)
    require_exact_int(int_dim_res, "ResCNN.int_dim_res", minimum=1)
    require_exact_int(n_blocks, "ResCNN.n_blocks", minimum=0)
    require_exact_int(kernel_size, "ResCNN.kernel_size", minimum=1)
    if kernel_size % 2 == 0:
      raise ValueError(
        "ResCNN.kernel_size must be odd so same-padding keeps the length")
    require_exact_int(n_blocks_cnn, "ResCNN.n_blocks_cnn", minimum=1)
    require_exact_int(groups, "ResCNN.groups", minimum=1)
    if groups not in (1, 2):
      raise ValueError("ResCNN.groups must be 1 or 2; got " + repr(groups))
    require_exact_bool(rescale_kernel, "ResCNN.rescale_kernel")
    require_exact_bool(separable, "ResCNN.separable")
    require_exact_bool(film, "ResCNN.film")
    require_nonzero_float32(gate_init, "ResCNN.gate_init")
    if block_opts is not None and not isinstance(block_opts, dict):
      raise TypeError("ResCNN.block_opts must be a mapping or None")
    sizes, pad_idx, pad_valid = resolve_padded_head_layout(
      geom=geom, output_dim=output_dim, where="ResCNN")
    super().__init__()
    if block_opts is None:
      block_opts = {}
    cnn_act = (head_act if head_act is not None
               else block_opts.get("act", activation_fcn))
    require_live_head_activation(cnn_act, "ResCNN head activation")
    self.model_recipe = record_model_recipe(
      class_path=type(self).__module__ + "." + type(self).__qualname__,
      name="rescnn", ia=None, input_dim=input_dim, output_dim=output_dim,
      needs_geom=True,
      kwargs={
        "int_dim_res": int(int_dim_res),
        "kernel_size": int(kernel_size),
        "rescale_kernel": bool(rescale_kernel),
        "groups": int(groups),
        "separable": bool(separable),
        "film": bool(film),
        "n_blocks": int(n_blocks),
        "n_blocks_cnn": int(n_blocks_cnn),
        "gate_init": float(gate_init),
        "head_act": (None if head_act is None
                     else activation_factory_recipe(head_act)),
        "block_opts": materialized_block_recipe(block_opts),
      })

    # ResMLP main path: standalone ResMLP layer stack, output in the
    # full-whitened basis (well conditioned).
    mlp = [nn.Linear(in_features=input_dim, out_features=int_dim_res)]
    for _ in range(n_blocks):
      mlp.append(ResBlock(int_dim_res, **block_opts))
    mlp.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))
    mlp.append(Affine())
    self.mlp = nn.Sequential(*mlp)

    # The geometry owns the physical rectangle. pad_idx maps each kept
    # output to its original angular slot, and pad_valid marks every slot
    # that is a measurement rather than storage padding.
    self.n_bins = len(sizes)
    self.max_bin = int(pad_valid.shape[-1])
    self.register_buffer("pad_idx", pad_idx)
    self.register_buffer("pad_valid", pad_valid)
    self.has_padding = not bool(torch.all(pad_valid).item())

    # the head: n_blocks_cnn x (one bins-as-channels conv + one
    # activation). head_act (the model.cnn.activation pin) wins when set;
    # else the run's shared family (block_opts["act"], the --activation
    # choice injected by EmulatorExperiment), falling back to
    # activation_fcn (the paper's H); act(max_bin) gives per-position
    # parameters, broadcast over the bin axis.
    # rescale_kernel: kernel_size was tuned for a single block, so
    # shrink the per-block kernel with depth to keep that block's
    # view, receptive field n*(k-1)+1 >= kernel_size, see
    # rescale_kernel_size, instead of over-growing it.
    if rescale_kernel:
      kernel_size = rescale_kernel_size(kernel_size=kernel_size,
                                        n_blocks_cnn=n_blocks_cnn)
    # the resolved per-block width, inspectable after a rescale.
    self.kernel_size = int(kernel_size)
    self.separable = separable

    # groups: only the xi-branch cut is a physical channel boundary here
    # (see the docstring). For each kept output, pad_idx identifies its
    # physical rectangle row. Check that rows before the midpoint are xi+
    # and rows after it are xi-. A fully masked row remains in the rectangle,
    # so it cannot shift the boundary while escaping this check.
    if groups == 2:
      if not hasattr(geom, "pm_kept") or self.n_bins % 2 != 0:
        raise ValueError(
          "ResCNN.groups=2 needs geom.pm_kept and an even bin count")
      half = self.n_bins // 2
      pm_values = torch.as_tensor(geom.pm_kept).reshape(-1).cpu()
      if int(pm_values.numel()) != output_dim:
        raise ValueError(
          "ResCNN.groups=2 needs one geom.pm_kept value per output")
      split_ok = True
      for element_index in range(output_dim):
        physical_bin = int(
          (self.pad_idx[element_index] // self.max_bin).item())
        expected_pm = 0 if physical_bin < half else 1
        if int(pm_values[element_index].item()) != expected_pm:
          split_ok = False
      if not split_ok:
        raise ValueError(
          "ResCNN.groups=2 needs the first half of the bins to be xi+ "
          "and the second half xi- according to the physical coordinate map")

    pad = (kernel_size - 1) // 2
    convs, acts = [], []
    for _ in range(n_blocks_cnn):
      if separable:
        # depthwise-separable factorization (see the class
        # docstring): a per-channel k-tap theta filter (groups =
        # n_bins: no mixing), then a pointwise 1x1 channel mix
        # honoring `groups`. No activation between the two, the
        # pair is a low-rank factorization of the plain block's
        # conv, and the block's one activation follows as usual.
        # Sequential keeps forward unchanged (convs[i] is callable
        # either way).
        convs.append(nn.Sequential(
          nn.Conv1d(in_channels=self.n_bins,
                    out_channels=self.n_bins,
                    kernel_size=kernel_size,
                    padding=pad,
                    groups=self.n_bins),
          nn.Conv1d(in_channels=self.n_bins,
                    out_channels=self.n_bins,
                    kernel_size=1,
                    groups=groups)))
      else:
        convs.append(nn.Conv1d(in_channels=self.n_bins,
                               out_channels=self.n_bins,
                               kernel_size=kernel_size,
                               padding=pad,
                               groups=groups))
      acts.append(cnn_act(self.max_bin))
    self.convs = nn.ModuleList(convs)
    self.acts  = nn.ModuleList(acts)

    # zero-init the last mixing layer: corr = 0 at init (the
    # activation maps 0 -> 0), so the model starts as its trunk
    # exactly; the zeroed layer gets real gradients through the
    # nonzero gate at step 1, earlier layers wake one step later.
    # In a separable block the zero lives on the pointwise (second)
    # conv; the depthwise filter keeps its init (zeroing both would
    # zero the pointwise's input and stall its wake-up).
    last = self.convs[-1][1] if separable else self.convs[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)

    # FiLM (film=True): one identity-initialized generator per
    # block predicts a per-bin (gamma, beta) from the parameters,
    # re-injecting cosmology the head otherwise never sees. At
    # init gamma = 1 / beta = 0, so the identity start above is
    # untouched. None (default) = the fixed, parameter-blind head.
    self.film_gens = None
    if film:
      gens = []
      for _ in range(n_blocks_cnn):
        gens.append(FiLMGenerator(n_cond=input_dim,
                                  n_channels=self.n_bins))
      self.film_gens = nn.ModuleList(gens)

    # learnable scalar gate on the correction (small init, not 0).
    self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # training phase, set by set_train_phase: "joint" (default,
    # everything trains), "trunk" (head frozen and bypassed, the
    # model runs as a pure ResMLP at ResMLP cost), "head" (trunk
    # frozen and run under no_grad, backward touches the head only).
    # A plain Python attribute: torch.compile guards on it and
    # recompiles once per phase switch.
    self._phase = "joint"

    # Frozen basis-change buffers (move with .to(device), not
    # trained). x @ W_fd maps f -> d, x @ W_df maps d -> f. sigma =
    # per-element scale sqrt(diag cov); evecs/sqrt_ev the full basis.
    #   full-whitened y -> physical -> theta order (/sigma):
    #     W_fd = diag(sqrt_ev) evecs.T diag(1/sigma)
    #   theta-order correction -> physical -> full-whitened:
    #     W_df = diag(sigma) evecs diag(1/sqrt_ev)  (= W_fd^{-1})
    # A diagonal family geometry (cmb / grid / grid2d) has
    # no eigenbasis: it whitens per element in physical order, so
    # the trunk already predicts in the head's local basis and the
    # basis change IS the identity — both maps stay None and forward
    # skips the matmuls (never build n_keep x n_keep identities).
    if hasattr(geom, "evecs"):
      evecs   = geom.evecs.detach()
      sqrt_ev = geom.sqrt_ev.detach()
      sigma   = torch.sqrt(((evecs * sqrt_ev) ** 2).sum(1))
      self.register_buffer(
        "W_fd", (sqrt_ev[:, None] * evecs.t()) / sigma[None, :])
      self.register_buffer(
        "W_df", (sigma[:, None] * evecs) / sqrt_ev[None, :])
    else:
      self.W_fd = None
      self.W_df = None

  def set_train_phase(self, phase):
    """Switch the two-phase training mode (run_emulator calls this).

    Identical rules to TemplateResCNN.set_train_phase — the plain
    heads carry it because two-phase training is not an IA-template
    privilege (ANY trunk+head design may
    train in two phases, on every family the heads ride):
      "joint" = everything trains, head active (the default).
      "trunk" = head frozen and bypassed: forward returns the bare
                trunk, so phase-1 epochs cost exactly a ResMLP (no
                head compute, no head gradients). With the zero-init
                head this changes nothing numerically, corr was
                already 0.
      "head"  = trunk frozen and run under no_grad: backward touches
                only the conv head + gate, so phase-2 epochs skip
                the whole trunk backward. The head starts from its
                zero-init identity, so the loss is continuous across
                the switch.

    Arguments:
      phase = "joint" | "trunk" | "head".
    """
    if phase not in ("joint", "trunk", "head"):
      raise ValueError(f"unknown train phase {phase!r}; "
                       "use 'joint', 'trunk', or 'head'")
    self._phase = phase
    trunk_on = phase in ("joint", "trunk")
    head_on  = phase in ("joint", "head")
    for p in self.mlp.parameters():
      p.requires_grad_(trunk_on)
    for p in self.convs.parameters():
      p.requires_grad_(head_on)
    for p in self.acts.parameters():
      p.requires_grad_(head_on)
    if self.film_gens is not None:
      # the FiLM generators are head parameters: frozen with the
      # head in the trunk phase, trained with it in the head phase.
      for p in self.film_gens.parameters():
        p.requires_grad_(head_on)
    self.gate.requires_grad_(head_on)

  def forward(self, x):
    """Predict with the trunk, then add the gated conv correction.

    The class docstring draws the full shape flow. In the frozen
    "head" phase the trunk runs without an autograd graph; in the
    "trunk" phase the head is skipped entirely (frozen at its
    identity init, its output is already known).

    Arguments:
      x = whitened parameters of shape (B, input_dim).

    Returns:
      the whitened prediction of shape (B, output_dim).
    """
    # trunk prediction in the full-whitened basis (the bulk map).
    if self._phase == "head":
      with torch.no_grad():
        y = self.mlp(x)               # (B, n_keep)
    else:
      y = self.mlp(x)                 # (B, n_keep)
    if self._phase == "trunk":
      return y
    # (reminder: W_fd = f -> d, full-whitened -> diagonal theta
    # order; W_df = d -> f, its inverse. The subscripts read in
    # multiply order: x @ W_fd starts in f and lands in d. On an
    # identity-basis family geometry both are None: y is already in
    # the head's local order, see __init__.)
    h = y if self.W_fd is None else y @ self.W_fd
    # scatter into the padded per-bin layout: each bin one channel.
    padded = h.new_zeros(h.shape[0], self.n_bins * self.max_bin)
    padded[:, self.pad_idx] = h
    c = padded.view(-1, self.n_bins, self.max_bin)
    valid_mask = self.pad_valid if self.has_padding else None
    n = len(self.convs)
    for i in range(n):
      if self.separable:
        c = self.convs[i][0](c)
        c = keep_valid_head_positions(c, valid_mask)
        c = self.convs[i][1](c)
      else:
        c = self.convs[i](c)               # cross-bin, theta-local
      c = keep_valid_head_positions(c, valid_mask)
      if self.film_gens is not None:
        # FiLM re-injection: a per-bin affine whose coefficients
        # depend on the parameters (identity at init). unsqueeze
        # broadcasts (B, n_bins) over the theta axis, the
        # modulation is per channel, never per position.
        gamma, beta = self.film_gens[i](x)
        c = gamma.unsqueeze(-1) * c + beta.unsqueeze(-1)
        c = keep_valid_head_positions(c, valid_mask)
      c = self.acts[i](c)
      c = keep_valid_head_positions(c, valid_mask)
    # gather the real entries back out of the padding, return to the
    # full-whitened basis when one exists (reminder: @ W_df goes
    # d -> f; None = identity), add through the gate.
    corr = c.reshape(-1, self.n_bins * self.max_bin)[:, self.pad_idx]
    if self.W_df is not None:
      corr = corr @ self.W_df
    return y + self.gate * corr


class ResTRF(DesignSpec, nn.Module):
  """
  ResMLP trunk + a bin-token transformer correction appendix. The
  trunk is the standalone ResMLP, predicting in the full
  (cov-eigenbasis) whitening; the head maps its output into theta
  order (ResCNN's fixed W_fd / W_df buffers), splits it into the
  (xi+/-, source-pair) tomographic bins, and runs a transformer
  whose tokens are those bins. The forward pass, shapes and all:

    x  (B, input_dim)           whitened parameters
       │  self.mlp              the ResMLP trunk
       ▼
    y  (B, n_keep)              full-whitened dv (also the skip)
       │  @ W_fd                f -> d: theta order, /sigma
       ▼
       │  pad_idx scatter       original physical slots + validity mask
       ▼
    t0 (B, n_bins, max_bin)     one token per bin, width max_bin
       │  n_blocks_trf x TRFBlock
       │                        cross-bin attention + per-bin MLPs
       ▼
    t  (B, n_bins, max_bin)
       │  corr = t - t0         what the blocks added (0 at init)
       │  pad_idx gather, @ W_df    d -> f, back to full whitening
       ▼
    out = y + gate * corr

  (legend: B = batch rows; n_keep = kept data-vector length, the
  unmasked entries the model emulates = output_dim; n_bins = number
  of physical tomographic (xi+/-, source-pair) bins, including an
  entirely masked row; max_bin = the full physical angular width of
  the padded token rectangle; f / d = the
  full-whitened / diagonal-theta bases, see the W_fd buffers below.)

  Attention shares information across bins, then each bin's own MLP
  stack specializes its correction (see TRFBlock for the two
  deviations from a textbook block). A per-bin conv would refine
  within bins but
  never across them; attention is the head for cross-bin structure
  in the trunk's residuals.

  build_shear_angle_map supplies the physical coordinate map and
  validity mask. Bins with different survivor counts still occupy
  the full physical angular width. pad_idx scatters each value into
  its original slot and gathers it back; pad_valid is reapplied
  after normalization, attention, MLP, activation and FiLM steps so
  storage-only positions cannot become latent data.

  The tokens live at their natural width: max_bin, the padded bin
  length. There is deliberately no embedding layer in and no output
  projection out, those adapters are what a transformer needs
  when its sequence is synthetic (a flat latent split into tokens,
  as in the published CMB design, where they were the parameter-
  heaviest layers); here the sequence structure is physical, so the
  raw bin segments are the tokens and the blocks' output is already
  in dv layout. The correction is corr = blocks(h) - h: every
  TRFBlock is exactly the identity at init (its branch outputs are
  zero-initialized, see TRFBlock), so corr = 0 and the model equals
  its trunk at epoch 1, the same zero-init identity start as the
  conv heads, with the same wake-up chain (the zeroed branch layers
  get real gradients through the nonzero gate at step 1).

  needs_geom / needs_bins are capability flags EmulatorExperiment
  reads: geom injected (basis buffers + bin sizes), compile_mode
  defaulted to "default", and build_shear_angle_map run on the data
  geometry before the model is built.

  On a diagonal family geometry (cmb / grid / grid2d) the same
  head applies, minus the basis change: those
  geometries whiten per element in physical order, so W_fd / W_df
  stay None and forward skips both matmuls. The token split is the
  geometry's attach_head_coords(): grid2d exposes one token per z
  slice (attention shares information across redshifts, each
  slice's private MLP specializes along k); cmb and grid expose ONE
  physical bin, which is where n_tokens comes in — attention over a
  single token has nothing to attend across, so n_tokens
  re-segments the spectrum into contiguous near-equal ell / z
  windows, the tokenization of the attention CMB emulators
  (arXiv 2505.22574), minus the embedding layers this head
  deliberately omits.

  Arguments:
    input_dim    = number of cosmological parameters.
    output_dim   = data-vector length to emulate (= n_keep).
    int_dim_res  = internal width of the residual trunk.
    geom         = the output geometry carrying bin_sizes, the
                   physical padded-head map, and its validity mask
                   (attached by build_shear_angle_map on the
                   cosmolike geometry, by attach_head_coords() on a
                   diagonal family geometry). With an eigenbasis (evecs /
                   sqrt_ev) it also defines the basis buffers;
                   without one the head works in the geometry's own
                   physical order (see the class docstring).
    n_heads      = attention heads per TRFBlock; must divide the
                   token width max_bin (the LSST-Y1 cosmic-shear
                   run keeps max_bin = 26 theta points per bin,
                   allowing n_heads = 1, 2, or 13; default 2. With
                   n_tokens set, max_bin = ceil(n / n_tokens) — pick
                   the pair so it divides).
    n_tokens     = None (default): the geometry's bins are the
                   tokens, unchanged. An int (>= 2) re-segments a
                   SINGLE-bin geometry (cmb / grid) into that many
                   contiguous near-equal windows; loud error on a
                   geometry with real physical bins (cosmic shear,
                   grid2d) — those bins carry physical meaning a
                   re-segmentation would slice through.
    n_blocks     = residual blocks in the trunk.
    n_blocks_trf = stacked transformer blocks.
    n_mlp_blocks = depth of each bin's private MLP stack inside
                   every TRFBlock; every layer runs at the token
                   width (dim -> dim), the interior pinned to the bin
                   length by design (no width knob, depth only).
    gate_init    = initial correction-gate scale (small, not 0,
                   a 0 gate strands the head with no gradient).
    shared_mlp   = False (default): per-bin unique MLPs. True: one
                   MLP shared by every bin, the textbook block,
                   the ablation isolating the unique-MLP deviation
                   (see TRFBlock's permutation-equivariance caveat).
    film         = False (default): the head is one fixed map,
                   blind to the cosmology. True: after every
                   TRFBlock, modulate the token stream with a
                   per-token affine gamma(x)*t + beta(x) from an
                   identity-initialized FiLMGenerator (one per
                   block; tokens play the conv head's channel
                   role, broadcast over the token width). The
                   stream carries the correction (corr = stream -
                   t0), so the cosmology chooses which bins'
                   corrections to amplify; identity init keeps
                   corr = 0 at epoch 1. See FiLMGenerator and
                   ai/notes/models-and-designs.md.
    head_act     = the TRF head's own activation factory (None ->
                   share block_opts["act"], the trunk's
                   family). build_specs builds it
                   from model.trf.activation (or the head: activation:
                   alias); set, it pins the head family only.
    block_opts   = ResBlock options (None -> {}); its "act" is the
                   trunk family and, unless head_act is set, reaches
                   the TRF MLPs too.
  """
  needs_geom = True
  needs_bins = True
  head_block = "trf"               # the bin-token transformer head

  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               n_heads=2, n_blocks=4, n_blocks_trf=1,
               n_mlp_blocks=2, n_tokens=None, gate_init=0.1,
               shared_mlp=False, film=False, head_act=None,
               block_opts=None):
    """Validate every option, then build the trunk and the TRF head.

    The class docstring's Arguments block defines each option in
    depth; one line each here:

    Arguments:
      input_dim    = whitened-parameter width.
      output_dim   = data-vector length to emulate (n_keep).
      int_dim_res  = trunk residual width.
      geom         = output geometry carrying the padded per-bin
                     layout (and the basis buffers, when any).
      n_heads      = attention heads per transformer block.
      n_blocks     = trunk residual blocks.
      n_blocks_trf = transformer blocks in the head.
      n_mlp_blocks = depth of each token's MLP stack.
      n_tokens     = token count for re-segmenting a complete
                     one-dimensional grid, or None for the physical
                     bins.
      gate_init    = starting value of the correction gate.
      shared_mlp   = True shares one position-wise MLP across tokens
                     instead of per-token unique weights.
      film         = True re-injects the cosmology as per-token
                     modulation.
      head_act     = the head's own activation factory; None shares
                     the trunk's.
      block_opts   = ResBlock options mapping, or None for {}.
    """
    require_exact_int(input_dim, "ResTRF.input_dim", minimum=1)
    require_exact_int(output_dim, "ResTRF.output_dim", minimum=1)
    require_exact_int(int_dim_res, "ResTRF.int_dim_res", minimum=1)
    require_exact_int(n_blocks, "ResTRF.n_blocks", minimum=0)
    require_exact_int(n_heads, "ResTRF.n_heads", minimum=1)
    require_exact_int(n_blocks_trf, "ResTRF.n_blocks_trf", minimum=1)
    require_exact_int(n_mlp_blocks, "ResTRF.n_mlp_blocks", minimum=1)
    if n_tokens is not None:
      require_exact_int(n_tokens, "ResTRF.n_tokens", minimum=2)
    require_exact_bool(shared_mlp, "ResTRF.shared_mlp")
    require_exact_bool(film, "ResTRF.film")
    require_nonzero_float32(gate_init, "ResTRF.gate_init")
    if block_opts is not None and not isinstance(block_opts, dict):
      raise TypeError("ResTRF.block_opts must be a mapping or None")
    sizes, pad_idx, pad_valid = resolve_padded_head_layout(
      geom=geom, output_dim=output_dim, where="ResTRF")
    super().__init__()
    if block_opts is None:
      block_opts = {}
    trf_act = (head_act if head_act is not None
               else block_opts.get("act", activation_fcn))
    require_live_head_activation(trf_act, "ResTRF head activation")
    self.model_recipe = record_model_recipe(
      class_path=type(self).__module__ + "." + type(self).__qualname__,
      name="restrf", ia=None, input_dim=input_dim, output_dim=output_dim,
      needs_geom=True,
      kwargs={
        "int_dim_res": int(int_dim_res),
        "n_heads": int(n_heads),
        "n_blocks": int(n_blocks),
        "n_blocks_trf": int(n_blocks_trf),
        "n_mlp_blocks": int(n_mlp_blocks),
        "n_tokens": (None if n_tokens is None else int(n_tokens)),
        "gate_init": float(gate_init),
        "shared_mlp": bool(shared_mlp),
        "film": bool(film),
        "head_act": (None if head_act is None
                     else activation_factory_recipe(head_act)),
        "block_opts": materialized_block_recipe(block_opts),
      })

    # Resolve the token layout before allocating a learnable layer. This
    # ordering makes an invalid width a configuration error and avoids
    # leaving a partially constructed model behind.
    # n_tokens: re-segment a SINGLE-bin geometry (a spectrum
    # on one axis: cmb's ell, grid's z) into contiguous near-equal
    # windows so attention has tokens to attend across — the first
    # n % T windows get one extra element, the ragged pad machinery
    # below absorbs the remainder. A geometry with real physical
    # bins already defines the tokens; re-cutting them would slice
    # through physical structure, so that is a loud error.
    if n_tokens is not None:
      if len(sizes) != 1:
        raise ValueError(
          "model.trf.n_tokens re-segments a single-bin geometry "
          "(cmb / grid), but this geometry defines "
          + str(len(sizes)) + " physical bins (tomographic bins / "
          "z slices) — those ARE the tokens; drop n_tokens")
      expected_identity = torch.arange(
        output_dim, dtype=torch.long, device=pad_idx.device)
      if not bool(torch.all(pad_valid).item()) \
          or not torch.equal(pad_idx, expected_identity):
        raise ValueError(
          "model.trf.n_tokens can re-segment only a complete one-dimensional "
          "grid with no masked physical coordinates")
      output_length = sizes[0]
      resolved_tokens = n_tokens
      if resolved_tokens < 2 or resolved_tokens > output_length:
        raise ValueError(
          "model.trf.n_tokens must be in 2.." + str(output_length)
          + " (the spectrum's length); got " + str(resolved_tokens))
      quotient, remainder = divmod(output_length, resolved_tokens)
      sizes = []
      for token_index in range(resolved_tokens):
        if token_index < remainder:
          sizes.append(quotient + 1)
        else:
          sizes.append(quotient)
      max_bin = max(sizes)
      positions = []
      for token_index, size in enumerate(sizes):
        for coordinate_index in range(size):
          positions.append(token_index * max_bin + coordinate_index)
      pad_idx = torch.tensor(
        positions, dtype=torch.long, device=pad_idx.device)
      pad_valid = torch.zeros(
        (1, len(sizes), max_bin), dtype=torch.bool,
        device=pad_valid.device)
      pad_valid.reshape(-1)[pad_idx] = True
    self.n_bins = len(sizes)
    self.max_bin = int(pad_valid.shape[-1])
    validate_trf_token_width(
      output_length=output_dim,
      n_tokens=self.n_bins,
      token_width=self.max_bin)
    if self.max_bin % n_heads != 0:
      raise ValueError(
        "ResTRF.n_heads (" + str(n_heads)
        + ") must divide the resolved padded token width ("
        + str(self.max_bin) + ")")

    # ResMLP main path: standalone ResMLP layer stack, output in the
    # full-whitened basis (well conditioned).
    mlp = [nn.Linear(in_features=input_dim, out_features=int_dim_res)]
    for _ in range(n_blocks):
      mlp.append(ResBlock(int_dim_res, **block_opts))
    mlp.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))
    mlp.append(Affine())
    self.mlp = nn.Sequential(*mlp)

    # The geometry supplies physical angular slots. Ragged n_tokens
    # segmentation above constructs the equivalent map for its contiguous
    # windows. Both buffers persist in the model state.
    self.register_buffer("pad_idx", pad_idx)
    self.register_buffer("pad_valid", pad_valid)
    self.has_padding = not bool(torch.all(pad_valid).item())

    # the head: n_blocks_trf transformer blocks straight on the
    # padded bin tokens (width = max_bin; no embedding, no output
    # projection). Every block is the identity at init, so
    # blocks(h) - h = 0 exactly. head_act (the model.trf.activation
    # pin) wins when set; else the trunk's shared family reaches the
    # TRF MLPs too.
    trf = []
    for _ in range(n_blocks_trf):
      trf.append(TRFBlock(
        self.max_bin,
        n_tokens=self.n_bins,
        n_heads=n_heads,
        n_mlp_blocks=n_mlp_blocks,
        act=trf_act,
        shared_mlp=shared_mlp,
        output_length=output_dim))
    self.trf = nn.ModuleList(trf)

    # FiLM (film=True): one identity-initialized generator per TRF
    # block predicts a per-token (gamma, beta) from the parameters,
    # re-injecting cosmology the head otherwise never sees. At init
    # gamma = 1 / beta = 0, so blocks(t0) == t0 and corr = 0 still
    # hold exactly. None (default) = the fixed, parameter-blind
    # head.
    self.film_gens = None
    if film:
      gens = []
      for _ in range(n_blocks_trf):
        gens.append(FiLMGenerator(n_cond=input_dim,
                                  n_channels=self.n_bins))
      self.film_gens = nn.ModuleList(gens)

    # learnable scalar gate on the correction (small init, not 0).
    self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # training phase, set by set_train_phase: "joint" (default,
    # everything trains), "trunk" (head frozen and bypassed, the
    # model runs as a pure ResMLP at ResMLP cost), "head" (trunk
    # frozen and run under no_grad, backward touches the head only).
    # A plain Python attribute: torch.compile guards on it and
    # recompiles once per phase switch.
    self._phase = "joint"

    # Frozen basis-change buffers, exactly ResCNN's (reminder:
    # W_fd = f -> d, full-whitened -> diagonal theta order /sigma;
    # W_df = d -> f, its inverse, subscripts in multiply order). A
    # diagonal family geometry (no eigenbasis) keeps both None: the
    # trunk already predicts in the head's local order.
    if hasattr(geom, "evecs"):
      evecs   = geom.evecs.detach()
      sqrt_ev = geom.sqrt_ev.detach()
      sigma   = torch.sqrt(((evecs * sqrt_ev) ** 2).sum(1))
      self.register_buffer(
        "W_fd", (sqrt_ev[:, None] * evecs.t()) / sigma[None, :])
      self.register_buffer(
        "W_df", (sigma[:, None] * evecs) / sqrt_ev[None, :])
    else:
      self.W_fd = None
      self.W_df = None

  def set_train_phase(self, phase):
    """Switch the two-phase training mode (run_emulator calls this).

    Identical rules to TemplateResTRF.set_train_phase — the plain
    heads carry it because two-phase training is not an IA-template
    privilege (ANY trunk+head design may
    train in two phases, on every family the heads ride):
      "joint" = everything trains, head active (the default).
      "trunk" = head frozen and bypassed: forward returns the bare
                trunk, so phase-1 epochs cost exactly a ResMLP (no
                head compute, no head gradients). With the
                identity-at-init blocks this changes nothing
                numerically, corr was already 0.
      "head"  = trunk frozen and run under no_grad: backward touches
                only the transformer head + gate, so phase-2 epochs
                skip the whole trunk backward. The blocks start from
                their identity init, so the loss is continuous
                across the switch.

    Arguments:
      phase = "joint" | "trunk" | "head".
    """
    if phase not in ("joint", "trunk", "head"):
      raise ValueError(f"unknown train phase {phase!r}; "
                       "use 'joint', 'trunk', or 'head'")
    self._phase = phase
    trunk_on = phase in ("joint", "trunk")
    head_on  = phase in ("joint", "head")
    for p in self.mlp.parameters():
      p.requires_grad_(trunk_on)
    for p in self.trf.parameters():
      p.requires_grad_(head_on)
    if self.film_gens is not None:
      # the FiLM generators are head parameters: frozen with the
      # head in the trunk phase, trained with it in the head phase.
      for p in self.film_gens.parameters():
        p.requires_grad_(head_on)
    self.gate.requires_grad_(head_on)

  def forward(self, x):
    """Predict with the trunk, then add the gated TRF correction.

    The class docstring draws the full shape flow. In the frozen
    "head" phase the trunk runs without an autograd graph; in the
    "trunk" phase the head is skipped entirely (frozen at its
    identity init, its output is already known).

    Arguments:
      x = whitened parameters of shape (B, input_dim).

    Returns:
      the whitened prediction of shape (B, output_dim).
    """
    # trunk prediction in the full-whitened basis (the bulk map).
    if self._phase == "head":
      with torch.no_grad():
        y = self.mlp(x)               # (B, n_keep)
    else:
      y = self.mlp(x)                 # (B, n_keep)
    if self._phase == "trunk":
      return y
    # (reminder: W_fd = f -> d, full-whitened -> diagonal theta
    # order; W_df = d -> f, its inverse. The subscripts read in
    # multiply order: x @ W_fd starts in f and lands in d. On an
    # identity-basis family geometry both are None: y is already in
    # the head's local order, see __init__.)
    h = y if self.W_fd is None else y @ self.W_fd
    # scatter into the padded per-bin layout: new_zeros makes the
    # (B, n_bins*max_bin) canvas (pad slots stay 0), the pad_idx
    # assignment places the n_keep real entries.
    padded = h.new_zeros(h.shape[0], self.n_bins * self.max_bin)
    padded[:, self.pad_idx] = h
    # (B, n_bins*max_bin) -> (B, n_bins, max_bin): each bin one token row, at
    # its natural width, the blocks run directly on these.
    t0 = padded.view(-1, self.n_bins, self.max_bin)
    valid_mask = self.pad_valid if self.has_padding else None
    t = t0
    n = len(self.trf)
    for i in range(n):
      t = self.trf[i](t, valid_mask)  # cross-bin attention + MLPs
      t = keep_valid_head_positions(t, valid_mask)
      if self.film_gens is not None:
        # FiLM re-injection: a per-token affine whose coefficients
        # depend on the parameters (identity at init). unsqueeze
        # broadcasts (B, n_bins) over the token width, per bin,
        # never per position.
        gamma, beta = self.film_gens[i](x)
        t = gamma.unsqueeze(-1) * t + beta.unsqueeze(-1)
        t = keep_valid_head_positions(t, valid_mask)
    # the correction is what the blocks added: every block is the
    # identity at init, so t - t0 = 0 exactly at epoch 1 (the
    # identity start, with no output projection needed to host it).
    # Gather the real entries back out of the padded layout
    # (dropping the pad slots), then return to the full-whitened
    # basis when one exists (reminder: @ W_df goes d -> f; None =
    # identity) and add through the gate.
    corr = (t - t0).reshape(
      -1, self.n_bins * self.max_bin)[:, self.pad_idx]
    if self.W_df is not None:
      corr = corr @ self.W_df
    return y + self.gate * corr
