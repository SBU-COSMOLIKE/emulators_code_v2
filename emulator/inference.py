"""Inference-time prediction on a saved emulator.

EmulatorPredictor is the in-package physics layer for running a trained
emulator outside training: it wraps rebuild_emulator (schema v2, the h5
alone) and turns a parameter dict into the physical cosmic-shear data
vector by exactly the forward path training used -- encode with the saved
ParamGeometry, the module forward, the factored-IA amplitude combine or the
NPCE base recombine when the run used one, then decode with the saved
DataVectorGeometry. The cobaya Theory adapter is a thin shell over this; all
the prediction physics lives here, testable against the training stack and
immune to code-default drift (the recipe + geometries come from the file).

PS: whitened = rotated into the covariance eigenbasis and scaled to unit
variance (the decorrelated space the network sees); encode = the geometry's
raw-params -> whitened-input transform (a factored emulator also appends the
raw IA amplitudes as the last columns, which the model drops and the combine
reads); decode = the output geometry's whitened -> physical (kept-entry)
data vector; kept entries = the unmasked positions of the full 3x2pt vector
the network emulates.
"""

import torch

from .results import rebuild_emulator


class EmulatorPredictor:
  """
  Physical-data-vector predictor for a saved emulator (schema v2).

  Built from rebuild_emulator(path_root, device): the module, the two saved
  geometries, and the physics-branch metadata (ia / pce_base / pce_form) all
  come from the h5, so nothing is re-declared here and a run predicts what
  its training-side eval would. predict() maps a parameter dict (or an
  ordered array in .names order) to the physical kept-entry data vector:

      params (in .names order)
         │  theta = (1, n_param) raw physical parameters
         │  pgeom.encode          center + whiten; append raw amps (factored)
         ▼
      X  (1, encoded_dim)         whitened model input (amps as last columns)
         │  model(X)              eval, no_grad; the model drops the amp
         │                        columns itself for a factored trunk
         ▼
      pred                        plain:    (1, n_keep) whitened dv
         │                        factored: (1, n_templates, n_keep) templates
         │                        NPCE:     (1, n_keep) refiner output
         │  _decode(pred, X)      plain:    geom.decode; factored: the
         │                        amplitude combine then geom.decode; NPCE:
         │                        the base recombine then geom.decode -- the
         │                        exact training chi2fn.decode, reused not
         │                        re-derived
         ▼
      dv_kept (1, n_keep)         physical kept-entry data vector
         │  geom.unsqueeze        scatter to dest_idx in a total_size zero
         ▼
      dv_full (1, total_size)     the full 3x2pt vector, 0 off the kept entries
         │  dv_return 'section'   slice the stored probe's block(s)
         │           '3x2pt'      keep the whole scattered vector
         ▼
      dv_out                      'section': (section_size,); '3x2pt':
                                  (total_size,); the returned vector (numpy)

  (legend: n_param = the full parameter count the geometry whitens;
  encoded_dim = pgeom.encoded_dim, the model's input width; n_keep =
  geom.dest_idx.numel(), the kept (unmasked) 3x2pt entries; n_templates =
  the factored design's template count; total_size = the full 3x2pt length
  unsqueeze restores to; section_size = the stored probe's block lengths
  summed (for xi, section_sizes[0]); dv_return = the returned-shape flag;
  the amplitudes among .names feed the combine, never the network.)

  Authority chain (the never-trust-defaults rule, read side): .names IS the
  saved ParamGeometry's stored names, in training order -- for a factored
  emulator the AmplitudeFactorGeometry's names already carry the IA
  amplitudes, so they join the required inputs automatically. The predictor
  asks the geometry; nobody keeps a second list. The kept-entry return shape
  matches the legacy Theory's use_emulator vector.
  """

  def __init__(self,
               path_root,
               device,
               compile_model=False,
               dv_return="section"):
    """Rebuild the emulator and assemble the branch-specific decoder.

    Arguments:
      path_root     = the saved emulator's path without extension (reads
                      <path_root>.h5 + .emul via rebuild_emulator).
      device        = torch.device to rebuild + run on.
      compile_model = torch.compile the module on CUDA (default False: batch-1
                      MCMC latency rarely pays off the compile cost).
      dv_return     = the returned shape (default 'section'): 'section'
                      returns this emulator's own probe block(s) sliced from
                      the scattered full vector (for a cosmic-shear emulator
                      the xi block, the length the likelihood demands);
                      '3x2pt' returns the full-length scattered vector (masked
                      positions zero). Which section comes from the artifact
                      (the geometry's stored probe), never re-declared here.

    Raises:
      ValueError on a non-schema-v2 file (rebuild_emulator refuses it), an
      unrecognized NPCE form, or a dv_return outside {'section', '3x2pt'};
      the exclusivity guard fires if a file somehow carries both a
      factored-IA design and an NPCE base.
    """
    if dv_return not in ("section", "3x2pt"):
      raise ValueError(
        "dv_return must be 'section' (this emulator's own probe block) or "
        "'3x2pt' (the full scattered vector), got " + repr(dv_return))
    self.dv_return = dv_return
    self.device = device
    (self.model,
     self.pgeom,
     self.geom,
     info) = rebuild_emulator(path_root, device,
                              compile_model=compile_model)

    self.names      = list(self.pgeom.names)
    # scalar (derived-parameter) emulator: predict returns a
    # {name: value} dict, not a data vector, so skip the dv-geometry
    # accounting (section_sizes / probe) and the physical-dv decoder that a
    # ScalarGeometry does not have. The emulated output names come off the
    # geometry; the input dtype still comes from the parameter whitening.
    self._scalar = info["scalar"]
    self._cmb    = info["cmb"]
    self._grid   = info["grid"]
    self._grid2d = info["grid2d"]
    if self._scalar:
      self.output_names = list(self.geom.names)
      self._dtype = self.pgeom.center.dtype
      self._decode = self._build_diag_decoder(
        pce_base=info["pce_base"],
        pce_form=info["pce_form"],
        transfer_base=info["transfer_base"],
        transfer_form=info["transfer_form"],
        transfer_space=info["transfer_space"])
      return
    # grid (background-function) emulator: predict returns
    # {"z": grid, quantity: row} — the raw physical function on the
    # stored grid (the target law already decoded by the geometry);
    # the distance pipeline (emulator/background.py) is applied by the
    # consumer (emul_baosn / a profile script), never re-derived here.
    if self._grid:
      self.quantity = self.geom.quantity
      self.units    = self.geom.units
      self.law      = self.geom.law
      self.z        = self.geom.z
      self._dtype   = self.pgeom.center.dtype
      self._decode  = self._build_diag_decoder(
        pce_base=info["pce_base"],
        pce_form=info["pce_form"],
        transfer_base=info["transfer_base"],
        transfer_form=info["transfer_form"],
        transfer_space=info["transfer_space"])
      return
    # grid2d (matter-power-spectrum) emulator: predict returns
    # the LAW-SPACE surface on the stored (z, k) axes — log(P/P_base)
    # under a syren law, the raw surface under "none" — keyed by the
    # quantity tag; the consumer multiplies the base back through
    # emulator/syren_base.py, exactly as emul_mps does.
    if self._grid2d:
      self.quantity = self.geom.quantity
      self.units    = self.geom.units
      self.law      = self.geom.law
      self.z        = self.geom.z
      self.k        = self.geom.k
      self._dtype   = self.pgeom.center.dtype
      self._decode  = self._build_diag_decoder(
        pce_base=info["pce_base"],
        pce_form=info["pce_form"],
        transfer_base=info["transfer_base"],
        transfer_form=info["transfer_form"],
        transfer_space=info["transfer_space"])
      return
    # CMB spectrum emulator: predict returns the physical C_ell
    # row on the stored multipole grid (a 1-D numpy array over .ell), so
    # skip the 3x2pt mask/section accounting a CmbDiagonalGeometry does
    # not have. The decoder is law-dispatched: the training chi2's decode
    # (losses/cmb.py) multiplies the imposed amplitude law back, reused
    # here not re-derived (the same single-sourcing as the dv branches).
    if self._cmb:
      self.spectrum       = self.geom.spectrum
      self.ell            = self.geom.ell
      self.units          = self.geom.units
      self.amplitude_law  = info["amplitude_law"]
      # an NPCE or transfer cmb artifact composes base + net (law "none"
      # enforced at training); otherwise the law-dispatched decode.
      if (info["pce_base"] is not None
          or info["transfer_base"] is not None):
        if info["amplitude_law"] != "none":
          kind = "an NPCE base" if info["pce_base"] is not None \
              else "a transfer base"
          raise ValueError(
            "the saved emulator carries both " + kind + " and "
            "amplitude_law " + repr(info["amplitude_law"]) + "; the two "
            "are mutually exclusive (validate_cmb), so the file is "
            "inconsistent")
        self._decode = self._build_diag_decoder(
          pce_base=info["pce_base"],
          pce_form=info["pce_form"],
          transfer_base=info["transfer_base"],
          transfer_form=info["transfer_form"],
          transfer_space=info["transfer_space"])
      else:
        self._decode = self._build_cmb_decoder(law=info["amplitude_law"],
                                               as_name=info["as_name"],
                                               tau_name=info["tau_name"],
                                               as_ref=info["as_ref"],
                                               tau_ref=info["tau_ref"])
      self._dtype = self.pgeom.center.dtype
      return
    self.dest_idx   = self.geom.dest_idx
    self.total_size = self.geom.total_size
    # section accounting the geometry persisted (None on a file that predates
    # the keys); section mode slices these, '3x2pt' ignores them.
    self.section_sizes = self.geom.section_sizes
    self.probe         = self.geom.probe

    ia       = info["ia"]
    pce_base = info["pce_base"]
    pce_form = info["pce_form"]
    # a transfer artifact embeds its frozen base (info["transfer_base"] holds
    # the rebuilt base model + both geometries; form / space say how to
    # compose). On such a run info["ia"] is the CORRECTION net's inherited
    # design, consumed by the transfer decoder, not a standalone factored run.
    transfer_base  = info.get("transfer_base")
    transfer_form  = info.get("transfer_form")
    transfer_space = info.get("transfer_space")
    if ia is not None and pce_base is not None:
      raise ValueError(
        "the saved emulator carries both a factored-IA design and an NPCE "
        "base; the two are mutually exclusive (pce excludes ia), so the "
        "file is inconsistent")
    if transfer_base is not None and pce_base is not None:
      raise ValueError(
        "the saved emulator carries both a transfer base and an NPCE base; "
        "the two are mutually exclusive (transfer excludes pce), so the file "
        "is inconsistent")

    # the physical-dv decoder: reuse the EXACT training chi2fn.decode so the
    # amplitude combine / NPCE recombine / transfer composition are
    # single-sourced, never re-derived here (the drift channel the standing
    # rule kills).
    self._decode = self._build_decoder(ia=ia,
                                       pce_base=pce_base,
                                       pce_form=pce_form,
                                       transfer_base=transfer_base,
                                       transfer_form=transfer_form,
                                       transfer_space=transfer_space)

    # the input dtype the geometry was whitened in (build theta to match, so
    # encode reproduces training exactly); unwrap the factored geometry's
    # kept-column ParamGeometry to reach the whitening tensors.
    base_pg     = getattr(self.pgeom, "pg_keep", self.pgeom)
    self._dtype = base_pg.center.dtype

  def _build_diag_decoder(self, pce_base, pce_form,
                          transfer_base=None, transfer_form=None,
                          transfer_space=None):
    """Pick the whitened-output -> physical map for a diagonal family.

    The scalar / cmb / grid / grid2d branches all decode through this:
    with an NPCE base (the 2026-07-12 family-wide ruling) or a frozen
    transfer base (the same day's symmetry ruling) it reconstructs the
    training loss purely for its decode, so the recombine keeps one
    definition (losses/pce.py / losses/transfer.py), exactly the
    single-sourcing rule of the dv branches; with neither, the module
    output is the whitened row itself and geom.decode alone inverts it
    (byte-identical to the pre-NPCE path).

    Arguments:
      pce_base      = the frozen PCEEmulator rebuilt off the h5, or None.
      pce_form      = the persisted combine form; a diagonal family
                      persists only "residual" (else a corrupt file).
      transfer_base = the rebuilt frozen transfer-base bundle ({model,
                      pgeom, geom}) or None; exclusive with pce_base.
      transfer_form / transfer_space = the persisted transfer combine
                      flags (gain|sum / "whitened" on these families).

    Returns:
      a callable (pred, x_enc) -> (1, n_out) physical row; the plain
      closure ignores x_enc, the NPCE / transfer decodes evaluate their
      base from it.
    """
    if transfer_base is not None and pce_base is not None:
      raise ValueError(
        "the saved emulator carries both a transfer base and an NPCE "
        "base; the two are mutually exclusive (transfer excludes pce), "
        "so the file is inconsistent")
    if transfer_base is not None:
      from .losses.transfer import TransferDiagChi2
      chi2 = TransferDiagChi2(
        geom=self.geom,
        base_net=transfer_base["model"],
        base_in_dim=len(transfer_base["pgeom"].names),
        form=transfer_form,
        space=transfer_space)
      # TransferDiagChi2.decode(pred, params_whitened) matches the
      # predictor's (pred, x_enc) decoder convention.
      return chi2.decode
    if pce_base is None:
      def _diag_plain_decode(pred, x_enc):
        # the module output is the whitened row itself; no base.
        return self.geom.decode(pred)
      return _diag_plain_decode
    if pce_form != "residual":
      raise ValueError(
        "the saved emulator is a diagonal-family artifact whose pce "
        "group records form " + repr(pce_form) + "; these families are "
        "residual-only (validate_pce), so the file is inconsistent")
    from .losses.pce import PCEResidualDiagChi2
    chi2 = PCEResidualDiagChi2(geom=self.geom, pce=pce_base)
    # PCEResidualDiagChi2.decode(y, params_whitened) already matches the
    # predictor's (pred, x_enc) decoder convention.
    return chi2.decode

  def _build_cmb_decoder(self, law, as_name, tau_name, as_ref, tau_ref):
    """Pick the whitened-output -> physical-C_ell map for a CMB emulator.

    Reconstructs the same loss object training used, purely for its
    decode (the amplitude law multiplied back for "as_exp2tau_ref", the
    plain un-whiten for "none"), so the law math keeps one definition
    (losses/cmb.py). Mirrors _build_decoder for the data-vector kinds.

    Arguments:
      law      = the imposed amplitude-law name the artifact persisted
                 ("none" / "as_exp2tau_ref"); make_cmb_chi2 rejects an
                 unknown name loudly and refuses the retired "as_exp2tau".
      as_name  = the raw linear amplitude column name ("" for "none").
      tau_name = the optical-depth column name ("" for "none").
      as_ref   = the fiducial A_s_ref the order-one law measures A_s
                 against (a persisted float; None for "none").
      tau_ref  = the fiducial tau_ref (a persisted float; None for "none").

    Returns:
      a callable (pred, x_enc) -> (1, n_ell) physical C_ell; the "none"
      closure ignores x_enc, the "as_exp2tau_ref" decode reads A_s / tau
      from it through the saved param geometry.
    """
    from .losses.cmb import make_cmb_chi2
    if law == "none":
      chi2 = make_cmb_chi2(geom=self.geom, law=law)

      def _cmb_plain_decode(pred, x_enc):
        # the module output is the whitened spectrum itself; no law.
        return chi2.decode(pred)
      return _cmb_plain_decode
    chi2 = make_cmb_chi2(geom=self.geom,
                         law=law,
                         param_geometry=self.pgeom,
                         as_name=as_name,
                         tau_name=tau_name,
                         as_ref=as_ref,
                         tau_ref=tau_ref)
    # CmbFactoredChi2.decode(pred, params_whitened) already matches the
    # predictor's (pred, x_enc) decoder convention.
    return chi2.decode

  def _build_decoder(self, ia, pce_base, pce_form,
                     transfer_base=None, transfer_form=None,
                     transfer_space=None):
    """Pick the whitened-output -> physical-dv map for this run's branch.

    Reconstructs the same loss object training used, purely for its decode
    (geom + the amplitude polynomial, geom + the frozen NPCE base, or the
    frozen transfer base composed by form/space), so the combine / recombine /
    compose math keeps one definition. The plain branch needs no loss object,
    the module output IS the whitened dv, so geom.decode alone.

    Arguments:
      ia             = the factored design name (nla / tatt) or None.
      pce_base       = the frozen PCEEmulator base or None.
      pce_form       = the NPCE form (residual / ratio) or None.
      transfer_base  = the rebuilt frozen transfer base bundle ({model, pgeom,
                       geom}) or None. When set it wins: the module output is
                       the CORRECTION, composed with the base by the transfer
                       decoder (its own family read from the base geometry +
                       ia). ia here is the correction's inherited design.
      transfer_form  = the transfer combination form (gain / sum) or None.
      transfer_space = the transfer composition space (physical / whitened).

    Returns:
      a callable (pred, x_enc) -> (1, n_keep) physical dv; the plain closure
      ignores x_enc, the factored / NPCE / transfer branches read the appended
      amplitudes / evaluate the base from it.
    """
    geom = self.geom
    if transfer_base is not None:
      # the transfer decoder composes the frozen base with the correction
      # on the base's own column slice, exactly as training did
      # (TransferChi2.decode single-sourced). The base family (plain vs
      # factored) is read off the embedded base geometry; a factored
      # base's coeff_fn / template count come from the correction's
      # inherited design (ia).
      from .losses.transfer import TransferChi2
      base_pg = transfer_base["pgeom"]
      if type(base_pg).__name__ == "AmplitudeFactorGeometry":
        from .experiment import IA_DESIGNS
        des         = IA_DESIGNS[ia]
        base_in_dim = len(base_pg.pg_keep.names)
        n_amps      = base_pg.n_amps
        n_templates = des["n_templates"]
        coeff_fn    = des["coeff_fn"]
      else:
        base_in_dim = len(base_pg.names)
        n_amps      = 0
        n_templates = 1
        coeff_fn    = None
      chi2 = TransferChi2(geom=geom,
                          base_net=transfer_base["model"],
                          base_in_dim=base_in_dim,
                          form=transfer_form,
                          space=transfer_space,
                          n_templates=n_templates,
                          n_amps=n_amps,
                          coeff_fn=coeff_fn)
      return chi2.decode

    if ia is not None:
      from .losses.ia import TemplateFactoredChi2
      from .experiment import IA_DESIGNS
      if ia not in IA_DESIGNS:
        raise ValueError(
          f"unknown factored-IA design {ia!r}; the saved recipe must name a "
          f"design in IA_DESIGNS ({sorted(IA_DESIGNS)})")
      chi2 = TemplateFactoredChi2(geom=geom,
                                  coeff_fn=IA_DESIGNS[ia]["coeff_fn"],
                                  n_amps=self.pgeom.n_amps)
      return chi2.decode

    if pce_base is not None:
      from .losses.pce import PCEResidualChi2, PCERatioChi2
      if pce_form == "residual":
        chi2 = PCEResidualChi2(geom=geom, pce=pce_base)
      elif pce_form == "ratio":
        chi2 = PCERatioChi2(geom=geom, pce=pce_base)
      else:
        raise ValueError(
          f"unknown NPCE form {pce_form!r}; the pce group must record "
          "'residual' (base + net) or 'ratio' (base * (1 + net))")
      return chi2.decode

    def _plain_decode(pred, x_enc):
      # the module output is the whitened dv itself; no combine / recombine.
      return geom.decode(pred)
    return _plain_decode

  def _as_row(self, params):
    """Order the inputs into a single (1, n_param) tensor.

    Arguments:
      params = either a mapping name -> value (read in .names order) or an
               already-ordered sequence / array in .names order.

    Returns:
      (1, n_param) tensor in the geometry's whitening dtype on self.device.

    Raises:
      KeyError naming the first required parameter a mapping is missing;
      ValueError when an ordered sequence has the wrong length.
    """
    if isinstance(params, dict):
      row = []
      for n in self.names:
        if n not in params:
          raise KeyError(
            f"predict() is missing required parameter {n!r}; the saved "
            f"emulator needs {self.names}")
        row.append(params[n])
    else:
      row = list(params)
      if len(row) != len(self.names):
        raise ValueError(
          f"predict() got {len(row)} values but the emulator needs "
          f"{len(self.names)} ({self.names})")
    return torch.as_tensor(row, dtype=self._dtype,
                           device=self.device).reshape(1, -1)

  def predict(self, params):
    """Predict the physical data vector at the configured dv_return shape.

    Arguments:
      params = a mapping name -> value, or an ordered sequence in .names
               order (the amplitudes among .names are consumed by the
               factored combine, never entered into the network).

    Returns:
      For a scalar (derived-parameter) emulator: a {name: value} dict, one
      entry per emulated output; the dv_return / section machinery
      does not apply. For a CMB spectrum emulator: a 1-D numpy array of
      physical C_ell on the stored multipole grid .ell (the
      imposed amplitude law already multiplied back); dv_return does not
      apply. For a data-vector emulator: a 1-D numpy array.
      dv_return 'section' (the default): this emulator's own probe block(s),
      shape (section_size,); for a cosmic-shear emulator the xi block, the
      length the likelihood glues per probe. dv_return '3x2pt': the full
      scattered vector (total_size,), the kept entries at their dest_idx
      positions and 0 everywhere else.
    """
    x     = self._as_row(params)
    x_enc = self.pgeom.encode(x)
    with torch.no_grad():
      pred = self.model(x_enc)
    # scalar (derived-parameter) emulator: destandardize the outputs
    # (the decoder built at init: geom.decode alone, or the NPCE base + net
    # recombine) and return a {name: value} dict, not a data vector; there
    # is no mask to unsqueeze through and no section to slice.
    if self._scalar:
      out = self._decode(pred, x_enc)[0]
      result = {}
      for i, nm in enumerate(self.output_names):
        result[nm] = float(out[i])
      return result
    # grid (background-function) emulator: decode inverts the
    # target law (e.g. exp(y) - offset) and returns the physical
    # function keyed by its quantity tag, with the stored grid beside it.
    if self._grid:
      row = self._decode(pred, x_enc)[0].detach().cpu().numpy()
      return {"z": self.z.detach().cpu().numpy(),
              self.quantity: row}
    # grid2d emulator: decode destandardizes to LAW SPACE (the
    # base multiply-back is the consumer's one step); the
    # flattened row is reshaped back to the (nz, nk) surface.
    if self._grid2d:
      nz = int(self.z.numel())
      nk = int(self.k.numel())
      surface = self._decode(pred, x_enc)[0].detach().cpu().numpy()
      return {"z": self.z.detach().cpu().numpy(),
              "k": self.k.detach().cpu().numpy(),
              self.quantity: surface.reshape(nz, nk)}
    # CMB spectrum emulator: decode multiplies the amplitude law
    # back (law-dispatched at build time), returning the physical C_ell on
    # the stored multipole grid .ell; no mask to unsqueeze through and no
    # section to slice.
    if self._cmb:
      cl = self._decode(pred, x_enc)         # (1, n_ell) physical C_ell
      return cl[0].detach().cpu().numpy()
    dv_kept = self._decode(pred, x_enc)      # (1, n_keep) kept-entry
    dv_full = self.geom.unsqueeze(dv_kept)   # (1, total_size), 0 off dest_idx
    if self.dv_return == "3x2pt":
      out = dv_full[0]
    else:
      out = self._section(dv_full)
    return out.detach().cpu().numpy()

  def _section(self, dv_full):
    """Slice this emulator's own probe block(s) from the full vector.

    Concatenates, in probe order, each block the stored probe spans (block k
    starts at sum(section_sizes[:k]) and runs section_sizes[k]); for a
    cosmic-shear emulator that is the single xi block, full[0:section_sizes
    [0]]. The training data vector is a separate story and stays full length.

    Arguments:
      dv_full = (1, total_size) the scattered full 3x2pt vector.

    Returns:
      (section_size,) the probe's block(s), concatenated in probe order.

    Raises:
      ValueError when the geometry predates section_sizes / probe (an older
      schema-v2 file), naming the two ways out.
    """
    if self.section_sizes is None or self.probe is None:
      raise ValueError(
        "dv_return='section' needs the geometry's section_sizes + probe, "
        "which this saved emulator predates (an older schema-v2 file). Two "
        "ways out: re-save with the current code (from_cosmolike records "
        "them), or build the predictor with dv_return='3x2pt' for the full "
        "scattered vector.")
    blocks = []
    for block_id in self.geom.PROBE_BLOCKS[self.probe]:
      start  = sum(self.section_sizes[:block_id])
      length = self.section_sizes[block_id]
      blocks.append(dv_full[0, start:start + length])
    if len(blocks) == 1:
      return blocks[0]
    return torch.cat(blocks)
