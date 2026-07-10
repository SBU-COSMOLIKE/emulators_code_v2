"""Transfer-learning losses: a frozen base plus a parallel correction net.

The transfer member of the emulator/losses/ family. A trained full emulator
(the base) is frozen entirely; a new, smaller parallel network takes the full
new parameter space and outputs a correction, and the final data vector
combines the two per element (per template, for a factored base) before the
amplitude combine. The base is expensive to expand, so the correction only
learns the difference the new physics makes.

The combination is a class picked by two YAML flags, orthogonal to each other:

  form  = how the correction combines with the base
          gain: base * (1 + r)      (fractional correction)
          sum : base + r            (additive residual)
  space = where the composition acts
          physical: the squeezed (bin) representation
          whitened: the decorrelated eigenbasis

All four form x space combinations start as the exact frozen base when the
correction output is zero (the zero-init surgery in the builder makes epoch 0
the base): gain because x * (1 + 0) = x, sum because x + 0 = x. That identity
is checked bitwise by the pre-train parity gate.

Speed design (copied from the PCE pair, losses/pce.py): the frozen base is run
once per row at encode (load) time and packed beside the truth into a wider
staged target (target_dim, read by batching.py), so the hot chi2 never re-runs
the base, it only unpacks and composes; backward passes touch only the small
correction net. The base's input is a column slice of the run's own encoding
(the block-extension invariant, warmstart.py D-TP3), so no second geometry
evaluation and no raw-parameter plumbing into the loss are needed.

PS: frozen = evaluated under no_grad, never trained; base = the pretrained full
emulator's prediction; correction = the small SGD-trained parallel net; whiten
= rotate into the covariance eigenbasis and scale to unit variance; squeeze =
keep only the unmasked dv entries; template = one of the T whitened components
a factored (NLA / TATT) model emits, combined by the amplitude polynomial
before the chi2; Mahalanobis distance = r^T Cinv r, the covariance-weighted
squared residual the chi2 always scores in the physical representation.
"""

import torch

from .core import CosmolikeChi2


# the two combination forms and the two spaces they can act in. The recommended
# pairing for each form (materialized as the default when the YAML omits space)
# and the trade-off the off-recommendation pairing carries live in experiment.py
# (validate_transfer), which owns the config surface.
FORMS  = ("gain", "sum")
SPACES = ("physical", "whitened")
RECOMMENDED_SPACE = {"gain": "physical", "sum": "whitened"}


class TransferChi2(CosmolikeChi2):
  """
  A frozen base under a parallel correction, scored by the plain chi2.

  One class covers the whole form x space matrix and both base families (a
  plain base whose output is the whitened dv, and a factored base whose output
  is T whitened templates the amplitude polynomial combines). The frozen base
  is precomputed at encode time and packed with the truth; the hot chi2
  unpacks and composes.

    encode (at load, once per row)
      dv, enc                     enc = the run's whitened parameters
        │  base = base_net(slice(enc))     the frozen base (no_grad), on the
        │                                  block-extension column slice
        │  convert base + truth to `space`
        ▼
      target = [base ; truth]     (B, target_dim)  packed, base cached

    chi2 (hot path, per batch)
      target, pred                pred = the correction net output
        │  unpack base, truth
        │  compose: gain -> base*(1+pred); sum -> base+pred   (per template)
        │  factored: combine with coeff_fn(amps) before the residual
        ▼
      r  (B, n_keep)              physical residual (add center in physical
        │  einsum r^T Cinv_sq r    space; whitened space unwhitens it)
        ▼
      c  (B,)                     per-sample chi2, no base recompute

  (legend: B = batch rows; enc = the run's encoded (whitened) parameters, the
  model input; n_keep = kept dv entries one template is wide; T = the base's
  template count (1 for a plain base, 3 nla / 10 tatt for a factored one);
  target_dim = 2*n_keep plain, (T+1)*n_keep factored; Cinv_sq = the kept x kept
  sub-block of the inverse covariance; slice(enc) = enc[:, :n_s] plain,
  cat(enc[:, :n_s'], enc[:, -n_amp:]) factored, the base's own encoding.)

  needs_params = True: encode / chi2 / decode take the run's whitened
  parameters (to slice the base input and, for a factored base, to read the
  amplitudes for the combine).
  """
  needs_params = True
  _params = None

  def __init__(self,
               geom,
               base_net,
               base_in_dim,
               form,
               space,
               n_templates=1,
               n_amps=0,
               coeff_fn=None):
    """Hold the geometry, the frozen base, and the combination choice.

    Arguments:
      geom        = the output DataVectorGeometry (pinned from the base);
                    owns the whitening, squeeze / decode, and Cinv.
      base_net    = the frozen base network (eval mode; run under no_grad).
                    Plain base -> outputs (B, n_keep) whitened dv; factored
                    base -> outputs (B, T, n_keep) whitened templates.
      base_in_dim = the base network's whitened-input width: n_s for a plain
                    base, n_s' (the base's non-amplitude count) for a factored
                    one. The base input is the first base_in_dim columns of the
                    run's encoding (plus the raw amplitudes, factored).
      form        = "gain" (base * (1 + r)) or "sum" (base + r).
      space       = "physical" (squeezed bins) or "whitened" (eigenbasis);
                    where the composition acts.
      n_templates = the base's template count T (1 = a plain base).
      n_amps      = appended amplitude columns (0 = a plain base); read from
                    the end of the run's encoding for the combine.
      coeff_fn    = the amplitude polynomial (nla_coeffs / tatt_coeffs) for a
                    factored base; None for a plain base.
    """
    super().__init__(geom)
    if form not in FORMS:
      raise ValueError(
        "transfer form must be one of " + str(list(FORMS)) + ", got "
        + repr(form))
    if space not in SPACES:
      raise ValueError(
        "transfer space must be one of " + str(list(SPACES)) + ", got "
        + repr(space))
    self.base_net    = base_net
    self.base_in_dim = int(base_in_dim)
    self.form        = form
    self.space       = space
    self.n_templates = int(n_templates)
    self.n_amps      = int(n_amps)
    self.coeff_fn    = coeff_fn
    self.factored    = coeff_fn is not None
    # live-base mode (the transfer refine stage, D-TP10): when True the base is
    # re-evaluated WITH grad each step (it is unfrozen and training) instead of
    # unpacked from the staged target; the truth half of the packed target is
    # still used, so no re-staging is needed. False (default) = the frozen
    # stage-1 behavior, byte-identical.
    self.live        = False

  @property
  def target_dim(self):
    """Staged-target width: encode packs [base ; truth].

    Returns:
      (T + 1) * n_keep for a factored base (T base templates + the truth),
      2 * n_keep for a plain base (the base dv + the truth).
    """
    nk = self.geom.dest_idx.numel()
    if self.factored:
      return (self.n_templates + 1) * nk
    return 2 * nk

  def _base_input(self, enc):
    """The frozen base's own encoding, sliced from the run's encoding.

    The block-extension invariant (warmstart.py D-TP3) makes the shared
    columns of the run's encoding bit-identical to the base's own encoding, so
    the base input is a column slice, no second geometry evaluation.

    Arguments:
      enc = (B, encoded_dim) the run's whitened parameters (the model input).

    Returns:
      (B, base_in_dim) plain, or (B, base_in_dim + n_amps) factored: the base
      network's input (its whitened block, plus the raw amplitudes it drops
      internally for the combine).
    """
    if self.factored:
      shared = enc[:, :self.base_in_dim]        # (B, n_s') whitened block
      amps   = enc[:, -self.n_amps:]            # (B, n_amp) raw amplitudes
      return torch.cat([shared, amps], dim=1)
    return enc[:, :self.base_in_dim]            # (B, n_s) whitened block

  def set_live(self, flag):
    """Switch the base between frozen (packed) and live (re-evaluated) mode.

    Arguments:
      flag = True for the refine stage (the base trains, evaluated with grad);
             False for the frozen stage-1 default.
    """
    self.live = bool(flag)

  def _base(self, enc):
    """Run the base on the D-TP3 slice.

    Frozen (the default): under no_grad, so no gradient flows into the base and
    it is the packed reference. Live (the refine stage): with grad, so the
    unfrozen base trains.

    Arguments:
      enc = (B, encoded_dim) the run's whitened parameters.

    Returns:
      (B, n_keep) whitened dv (plain), or (B, T, n_keep) whitened templates
      (factored).
    """
    if self.live:
      return self.base_net(self._base_input(enc))
    with torch.no_grad():
      return self.base_net(self._base_input(enc))

  def _unwhiten_templates(self, tmpl_w):
    """Un-whiten each template to the physical representation (no center).

    The center is a single kept-length vector added once after the combine
    (the constant-coefficient GG template carries it), so the per-template
    un-whitening drops it; geom.unwhiten is the eigenbasis -> physical map.

    Arguments:
      tmpl_w = (B, T, n_keep) whitened templates.

    Returns:
      (B, T, n_keep) physical templates (un-whitened, center not yet added).
    """
    b, t, nk = tmpl_w.shape
    # collapse (B, T) to one axis so unwhiten (a (rows, n_keep) map) applies to
    # every template, then restore the (B, T) split.
    flat = tmpl_w.reshape(b * t, nk)
    return self.geom.unwhiten(flat).reshape(b, t, nk)

  def _compose(self, base_repr, correction):
    """Combine the base with the correction, per the form.

    Arguments:
      base_repr  = the base representation (whitened or physical; plain
                   (B, n_keep) or factored (B, T, n_keep)).
      correction = the correction net output, same shape.

    Returns:
      base_repr * (1 + correction) for gain, base_repr + correction for sum;
      the correction is exactly zero at epoch 0, so this is exactly base_repr
      there (1 + 0 = 1, + 0, both exact).
    """
    if self.form == "gain":
      return base_repr * (1.0 + correction)
    return base_repr + correction

  def _combine(self, templates, enc):
    """Combine factored templates with the amplitude polynomial.

    Arguments:
      templates = (B, T, n_keep) templates (whitened or physical).
      enc       = (B, encoded_dim) the run's encoding (amplitudes last).

    Returns:
      (B, n_keep) the amplitude-combined vector: sum_t c_t template_t.
    """
    amps = enc[:, -self.n_amps:]               # (B, n_amp) physical amplitudes
    c    = self.coeff_fn(amps)                 # (B, T) coefficients
    # operands in subscript order: c (b,t) = coefficients, templates (b,t,k);
    # sums over t to the combined vector (b,k).
    return torch.einsum("bt,btk->bk", c, templates)

  def _to_space(self, base_w, dv):
    """Convert the base output and the truth to the chosen space.

    Arguments:
      base_w = the frozen base's whitened output (plain (B, n_keep) or
               factored (B, T, n_keep)).
      dv     = (B, total_size) raw data vectors.

    Returns:
      (base_repr, truth): both in `space`. Plain physical -> geom.decode
      (center included) and geom.squeeze; plain whitened -> the base output and
      geom.encode; factored physical -> per-template un-whitened base and
      geom.squeeze; factored whitened -> the base templates and geom.encode.
    """
    if self.factored:
      if self.space == "physical":
        return self._unwhiten_templates(base_w), self.geom.squeeze(dv).float()
      return base_w, self.geom.encode(dv)
    if self.space == "physical":
      return self.geom.decode(base_w), self.geom.squeeze(dv).float()
    return base_w, self.geom.encode(dv)

  def encode(self, dv, params_whitened):
    """Raw dv -> the packed [base ; truth] target (once per row at load).

    Runs the frozen base on the D-TP3 slice, converts it and the truth to the
    chosen space, and packs them into the wider target so the hot chi2 never
    re-runs the base.

    Arguments:
      dv              = (B, total_size) raw data vectors.
      params_whitened = (B, encoded_dim) the run's encoding (for the base slice
                        and, factored, the amplitudes).

    Returns:
      (B, target_dim): [base ; truth]; base is (T*n_keep) flattened templates
      or n_keep dv, truth is n_keep.
    """
    base_w = self._base(params_whitened)
    base_repr, truth = self._to_space(base_w=base_w, dv=dv)
    if self.factored:
      base_flat = base_repr.reshape(base_repr.shape[0], -1)
      return torch.cat([base_flat, truth], dim=1)
    return torch.cat([base_repr, truth], dim=1)

  def _unpack(self, target):
    """Split a packed target into (base_repr, truth).

    Arguments:
      target = (B, target_dim) packed [base ; truth] from encode.

    Returns:
      (base_repr, truth): base_repr is (B, T, n_keep) factored / (B, n_keep)
      plain, truth is (B, n_keep).
    """
    nk = self.geom.dest_idx.numel()
    if self.factored:
      t = self.n_templates
      base = target[:, :t * nk].reshape(target.shape[0], t, nk)
      return base, target[:, t * nk:]
    return target[:, :nk], target[:, nk:]

  def _composed_physical(self, base_repr, correction, enc):
    """Compose + combine to a physical prediction (physical-space path).

    Arguments:
      base_repr  = the physical base representation (plain (B, n_keep) or
                   factored per-template (B, T, n_keep), center not yet added).
      correction = the correction net output, same shape.
      enc        = (B, encoded_dim) the run's encoding (amplitudes last).

    Returns:
      (B, n_keep) physical prediction. Factored: combine the composed
      templates, then add the center once (the base path at correction 0).
    """
    composed = self._compose(base_repr=base_repr, correction=correction)
    if self.factored:
      return self._combine(templates=composed, enc=enc) + self.geom.center
    return composed

  def _composed_whitened(self, base_repr, correction, enc):
    """Compose + combine to a whitened prediction (whitened-space path).

    Arguments:
      base_repr  = the whitened base representation (plain (B, n_keep) or
                   factored (B, T, n_keep)).
      correction = the correction net output, same shape.
      enc        = (B, encoded_dim) the run's encoding (amplitudes last).

    Returns:
      (B, n_keep) whitened prediction (the combined whitened xi, factored).
    """
    composed = self._compose(base_repr=base_repr, correction=correction)
    if self.factored:
      return self._combine(templates=composed, enc=enc)
    return composed

  def chi2(self, pred, target, params_whitened=None, full=False):
    """Per-sample chi2 of the composed prediction against the truth.

    Unpacks the base and truth cached in target, so the frozen base is never
    recomputed in the loop. The metric is always the physical Mahalanobis
    distance: physical space forms the residual directly, whitened space lets
    the base chi2 un-whiten it first.

    Arguments:
      pred            = (B, n_keep) plain / (B, T, n_keep) factored correction.
      target          = (B, target_dim) packed [base ; truth] from encode.
      params_whitened = (B, encoded_dim) the run's encoding, or None to use the
                        loss() stash (the amplitudes for the combine).
      full            = if True, contract the full-length precision
                        (diagnostics); else the kept sub-block.

    Returns:
      (B,) per-sample chi2.
    """
    if params_whitened is None:
      params_whitened = self._params
    if self.live:
      # refine stage: re-evaluate the (unfrozen, drifting) base with grad; the
      # truth half of the packed target is still valid (only the base changes),
      # so no re-staging. The base representation matches the frozen path, so
      # the stage-1 -> stage-2 handoff is loss-continuous.
      base_repr = self._base_in_space(self._base(params_whitened))
      truth     = self._truth_half(target)
    else:
      base_repr, truth = self._unpack(target)
    if self.space == "physical":
      pred_phys = self._composed_physical(
        base_repr=base_repr, correction=pred, enc=params_whitened)
      r = pred_phys - truth
      geo = self.geom
      if full:
        rf = geo.unsqueeze(r)
        return torch.einsum("bi,ij,bj->b", rf, geo.Cinv, rf)
      return torch.einsum("bi,ij,bj->b", r, geo.Cinv_sq, r)
    pred_w = self._composed_whitened(
      base_repr=base_repr, correction=pred, enc=params_whitened)
    # whitened prediction vs whitened truth: the base chi2 un-whitens the
    # residual and contracts Cinv_sq (the same physical metric).
    return CosmolikeChi2.chi2(self, pred=pred_w, target=truth, full=full)

  def decode(self, pred, params_whitened):
    """Composed physical prediction (diagnostics; recomputes the base).

    Arguments:
      pred            = (B, n_keep) plain / (B, T, n_keep) factored correction.
      params_whitened = (B, encoded_dim) the run's encoding.

    Returns:
      (B, n_keep) physical prediction. Not a hot path, so the base is
      recomputed here rather than unpacked.
    """
    base_w = self._base(params_whitened)
    if self.space == "physical":
      base_repr = self._space_base_physical(base_w)
      return self._composed_physical(
        base_repr=base_repr, correction=pred, enc=params_whitened)
    combined_w = self._composed_whitened(
      base_repr=base_w, correction=pred, enc=params_whitened)
    return self.geom.decode(combined_w)

  def _space_base_physical(self, base_w):
    """The base's physical representation (plain decode / per-template)."""
    if self.factored:
      return self._unwhiten_templates(base_w)
    return self.geom.decode(base_w)

  def _base_in_space(self, base_w):
    """The base representation in the chosen space (the base half of encode).

    Used by the live (refine) chi2 to convert the freshly-evaluated base to the
    same representation the frozen path packs, so the composition is identical
    given the same base weights.

    Arguments:
      base_w = the base's whitened output (plain (B, n_keep) / factored
               (B, T, n_keep)).

    Returns:
      the base in `space`: physical -> geom.decode (plain) / per-template
      un-whitened (factored); whitened -> base_w unchanged.
    """
    if self.space == "physical":
      return self._space_base_physical(base_w)
    return base_w

  def _truth_half(self, target):
    """The truth half of a packed target (the base half is ignored when live).

    Arguments:
      target = (B, target_dim) the packed [base ; truth] staged at load.

    Returns:
      (B, n_keep) the truth in `space` (unchanged across stages, so the refine
      stage reuses it without re-staging).
    """
    nk = self.geom.dest_idx.numel()
    if self.factored:
      return target[:, self.n_templates * nk:]
    return target[:, nk:]

  def base_decode(self, params_whitened):
    """The frozen base's own physical decode, in this loss's space.

    Equal to decode() with the correction identically zero: the parity gate
    compares the epoch-0 composed prediction against this. Computed in the
    loss's own space, so a physical-space transfer reproduces the base through
    the same per-template un-whitening the composition uses (the identity is
    then bitwise; the two spaces' base decodes agree up to the combine /
    un-whiten reassociation, which is a float-order, not a physics, difference).

    Arguments:
      params_whitened = (B, encoded_dim) the run's encoding.

    Returns:
      (B, n_keep) the frozen base's physical prediction.
    """
    base_w = self._base(params_whitened)
    if self.space == "physical":
      base_repr = self._space_base_physical(base_w)
      if self.factored:
        return self._combine(templates=base_repr, enc=params_whitened) \
            + self.geom.center
      return base_repr
    if self.factored:
      combined_w = self._combine(templates=base_w, enc=params_whitened)
      return self.geom.decode(combined_w)
    return self.geom.decode(base_w)

  def loss(self, pred, target, params_whitened, *args, **kwargs):
    """Scalar training loss from the composed-prediction chi2.

    Stashes params so the inherited reduction (which calls self.chi2 without
    params) can recover the amplitudes for a factored base.

    Arguments:
      pred            = (B, n_keep) plain / (B, T, n_keep) factored correction.
      target          = (B, target_dim) packed [base ; truth].
      params_whitened = (B, encoded_dim) the run's encoding.
      *args, **kwargs = the mode / trim / focus reduction controls, forwarded
                        positionally (never keyword pred / target before the
                        *args forwarder).

    Returns:
      a scalar loss tensor.
    """
    self._params = params_whitened
    return CosmolikeChi2.loss(self, pred, target, *args, **kwargs)
