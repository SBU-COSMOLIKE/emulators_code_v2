"""Inference-time prediction from a saved emulator artifact pair.

``EmulatorPredictor`` rebuilds one trained model from two required files.
The ``.emul`` file contains the model's registered tensors. The ``.h5`` file
contains the constructor recipe, geometries, scientific facts, configuration,
and histories. Prediction then follows the same common prefix used during
training: order the raw parameters, encode them, call the model, and apply the
saved decoder. The final return value depends on the observable family:

=====================  ======================================================
family                 ``predict`` return
=====================  ======================================================
cosmic-shear vector    one physical NumPy vector, either the stored section
                       or the full scattered 3x2pt layout
scalar outputs         a dictionary from output name to Python float
CMB spectrum           one NumPy vector of physical ``C_ell`` values on the
                       stored multipole grid
background grid        a dictionary containing ``z`` and the named physical
                       function on that grid
matter-power grid      a dictionary containing ``z``, ``k``, and the named
                       law-space surface; the adapter applies the stored base
=====================  ======================================================

A saved emulator also carries the science it was born under: the cosmology its
dataset was generated with held fixed, and the region of parameter space the
generator sampled. ``EmulatorPredictor`` retains that record and exposes the
three questions a consumer must be able to ask of it, one method each:

=====================  ======================================================
method                 the question it answers
=====================  ======================================================
``check_belongs_to``   does this emulator belong to the cosmology now being
                       sampled? (the artifact against the chain)
``check_pairs_with``   do these two emulators belong to each other? (one
                       artifact against another)
``check_may_serve``    may this emulator be asked about this point? (the
                       training region against one point)
=====================  ======================================================

The laws those methods run, and the refusals they raise, live in
``emulator/fixed_facts.py``. This module owns only the site at which they are
asked: the artifact this predictor rebuilt, and the identity to name when one
of them refuses.

Two of the three are asked once, at the start of a chain, by whoever assembles
the emulators being served; the module-level ``check_artifacts_belong_to`` and
``check_artifacts_pair_up`` below are that site, shared by the five cobaya
adapters so that the five of them do not become five authors of one refusal.
The third — the training region against one point — is asked by ``predict``
itself, on every call, because a point outside the region is answered
confidently and wrongly by a network that cannot know it is extrapolating.

``torch.no_grad()`` disables gradient recording because inference does not
update model weights. ``detach()`` removes a tensor from any gradient graph
without changing its numerical values. ``cpu()`` places a tensor in CPU
memory and copies only when a device move is required. ``numpy()`` exposes a
CPU tensor as a NumPy array. These operations appear together at the return
boundary because Cobaya and analysis scripts consume NumPy values.

PS: whitened = rotated into the covariance eigenbasis and scaled to unit
variance (the decorrelated space the network sees); encode = the geometry's
raw-params -> whitened-input transform (a factored emulator also appends the
raw IA amplitudes as the last columns, which the model drops and the combine
reads); decode = the output geometry's whitened -> physical (kept-entry)
data vector; kept entries = the unmasked positions of the full 3x2pt vector
the network emulates; the record = the two blocks of persisted science the
artifact carries, the cosmology held fixed (fixed_facts) and the sampled
region (input_domain).
"""

import math
import numbers

import torch

# the scientific record and its three comparison laws. The module is
# torch-free, so importing it here costs the predictor nothing.
from . import fixed_facts
from .results import rebuild_emulator


def _require_prediction_tensor(value, *, stage, shape, where):
  """Validate one tensor at a named public-prediction boundary.

  A later operation can hide an earlier failure.  For example, a decoder may
  replace a constant model coordinate with its saved center, making a NaN in
  the model output disappear.  Each stage is therefore checked before the
  next stage runs, not only at the final NumPy return.

  Arguments:
    value = object produced by the prediction stage.
    stage = short human-readable stage name used in a refusal.
    shape = exact tuple of dimensions required at this stage.
    where = saved artifact root, so a refusal identifies the file.

  Returns:
    value unchanged after validation.

  Raises:
    TypeError when value is not a real floating-point Torch tensor.
    ValueError when its shape differs or any entry is NaN or infinity.
  """
  expected = tuple(int(x) for x in shape)
  if not torch.is_tensor(value) or not torch.is_floating_point(value):
    raise TypeError(
      where + ": public prediction stage " + repr(stage)
      + " must return a real floating-point Torch tensor, got "
      + type(value).__name__ + ".")
  if tuple(value.shape) != expected:
    raise ValueError(
      where + ": public prediction stage " + repr(stage)
      + " must have exact shape " + repr(expected) + ", got "
      + repr(tuple(value.shape)) + ".")
  if not bool(torch.isfinite(value).all()):
    raise ValueError(
      where + ": public prediction stage " + repr(stage)
      + " produced NaN or infinity; the value is refused before the next "
      "stage can hide or publish it.")
  return value


def _select_composition(composition_mode, pce_base, transfer_base):
  """Select decoder payloads from the validated authoritative mode.

  ``rebuild_emulator`` has already checked the HDF5 enum against its exact
  required/forbidden group set.  This second boundary keeps inference from
  drifting back to presence-based dispatch if a caller constructs or mutates
  an ``info`` record in memory.
  """
  if type(composition_mode) is not str or composition_mode not in (
      "plain", "npce", "transfer"):
    raise ValueError(
      "composition_mode must be 'plain', 'npce', or 'transfer', got "
      + repr(composition_mode))
  expect_pce = composition_mode == "npce"
  expect_transfer = composition_mode == "transfer"
  have_pce = pce_base is not None
  have_transfer = transfer_base is not None
  if have_pce != expect_pce or have_transfer != expect_transfer:
    raise ValueError(
      "validated composition_mode=" + repr(composition_mode)
      + " requires pce_base=" + repr(expect_pce)
      + " and transfer_base=" + repr(expect_transfer)
      + ", but rebuild info carries pce_base=" + repr(have_pce)
      + " and transfer_base=" + repr(have_transfer))
  return pce_base, transfer_base


def _is_named_pair(params):
  """Is this input the (names, values) pair, or a bare row of numbers?

  The two look alike from the outside: both are sequences. The predictor has to
  tell them apart before it can read one and refuse the other, and it tells
  them apart by what sits in the first slot. The pair holds a sequence of
  parameter names there. A bare row holds a number.

  The case that has to come out right is a two-parameter emulator handed a bare
  row of two numbers. It is a 2-item sequence, exactly like a pair, and the
  length cannot separate them. The first slot can: a number is not a sequence
  of names.

  Arguments:
    params = the object handed to predict, already known not to be a mapping.

  Returns:
    True when params is a 2-item sequence whose first item is a sequence of
    strings, which is the (names, values) pair; False for anything else,
    including a bare row of numbers.
  """
  if not isinstance(params, (tuple, list)):
    return False
  if len(params) != 2:
    return False
  head = params[0]
  # A string is itself a sequence of strings, one character at a time, so it
  # has to be ruled out before the slot is read: ("H0", 0.7) is a name beside a
  # number, not a pair of sequences.
  if isinstance(head, str):
    return False
  try:
    first = head[0]
  except (TypeError, IndexError, KeyError):
    # a number is not subscriptable, and an empty name list has no first entry.
    # Neither is the pair.
    return False
  # numpy's string scalar is a subclass of str, so a name list read back out of
  # a saved file answers True here without a numpy import.
  return isinstance(first, str)


def check_artifacts_belong_to(predictors, provider, adapter):
  """Vertical law at the cobaya site: every served artifact against the chain.

  A cobaya adapter is handed its provider once, when the chain is set up, and
  the provider carries the resolved model — the same object the dataset
  generator read when it wrote the record. So the question "does this emulator
  belong to the cosmology being sampled?" can be asked exactly once per chain,
  before the first point is evaluated, rather than once per point: the facts
  cannot change while a chain runs.

  This is the site, shared by all five adapters. A copy of it in each adapter
  would be five authors of one refusal, and the refusal is the product here: a
  chain that would have silently answered about the wrong universe stops instead
  and says which coordinate it stopped on.

  Arguments:
    predictors = the EmulatorPredictors this adapter is serving.
    provider   = the cobaya Provider the adapter was initialized with. It is
                 duck-typed: the only surface read is ``.model``, the resolved
                 global model cobaya built.
    adapter    = the adapter's own name, named in the API-drift refusal.

  Returns:
    None. The function is called for its refusals.

  Raises:
    ValueError when the provider cannot hand over the model (a cobaya whose
    Provider no longer carries it), or when any served artifact was generated
    under a cosmology this chain is not sampling.
  """
  model = getattr(provider, "model", None)
  if model is None:
    # The alternative to refusing here is skipping the law, and a law that
    # skips itself when it cannot run is not a law: the chain would sample on,
    # and the emulator it was never allowed to serve would answer every point.
    try:
      import cobaya
      version = getattr(cobaya, "__version__", "unknown")
    except ImportError:
      version = "not importable"
    raise ValueError(
      adapter + ": the cobaya provider handed to this theory carries no "
      ".model, so the cosmology being sampled cannot be read and the saved "
      "emulators cannot be shown to belong to it. This adapter needs the "
      "resolved global model, which cobaya's Provider has stored as .model "
      "since 3.x; the cobaya found here is version " + repr(version) + ". "
      "The check is not skipped, because an emulator generated under a "
      "different cosmology answers every point confidently and wrongly.")

  resolved = fixed_facts.resolved_constants(model=model)
  for predictor in predictors:
    predictor.check_belongs_to(resolved_model=resolved)


def check_artifacts_pair_up(predictors):
  """Horizontal law at the cobaya site: the served set is ONE dataset.

  Every artifact an adapter serves is combined into one prediction, so all of
  them must come from one generator dump and one cosmology. The law is an
  equality, so it is transitive: comparing every artifact against the first
  proves the whole set agrees, and each refusal still names the two files it
  refused between.

  It runs LAST, after the adapter's own configuration laws (wrong kind, pair
  count, duplicate output, no chaining, one shared grid). A misconfigured set is
  a misconfiguration, and it must be refused as one: told that two emulators
  were fitted to different datasets, the reader of a scalar chain whose input
  was accidentally another emulator's output would go off to regenerate both
  halves from one run, which is impossible advice — the two halves were never
  one dataset to begin with.

  Arguments:
    predictors = the EmulatorPredictors this adapter is serving, in load order.

  Returns:
    None. The function is called for its refusals.

  Raises:
    ValueError naming the two artifacts and the fact they disagree about.
  """
  for i in range(1, len(predictors)):
    predictors[0].check_pairs_with(predictors[i])


class EmulatorPredictor:
  """
  Physical-observable predictor for a saved schema-3 emulator.

  ``rebuild_emulator(path_root, device)`` reads both ``path_root.emul`` and
  ``path_root.h5``. It returns the model, the two saved geometries, and the
  branch metadata. Nothing in this constructor re-declares those saved facts.
  The diagram below shows the cosmic-shear branch, where ``predict`` maps the
  parameters to a physical data vector. Every input form names its parameters:
  a mapping from name to value, or a pair holding the names beside their values
  (see ``_as_row``). A bare row of numbers is refused, because a row of numbers
  cannot say which parameter each number is.

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
         │                        the base recombine, then geom.decode; the
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
  saved ParamGeometry's stored names in training order. For a factored
  emulator the AmplitudeFactorGeometry's names already carry the IA
  amplitudes, so they join the required inputs automatically. The predictor
  asks the geometry; nobody keeps a second list. The kept-entry return shape
  matches the legacy Theory's use_emulator vector.

  The scientific record travels the same way. ``rebuild_emulator`` hands back
  the two blocks the file carries and ``.record`` holds them. The two blocks
  are also read out singly: ``.fixed_facts`` is the cosmology the dataset was
  generated under, ``.input_domain`` the region it was sampled over. The four
  ``check_*`` / ``served_support_with`` methods below are the
  sites at which the laws are asked about this artifact. ``predict`` asks the
  domain law itself, on every point, against the support compiled once at load;
  the two equality laws are asked once per chain by whoever assembles the
  emulators being served (the cobaya adapters, through the module-level sites
  above).
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
      ValueError on a non-schema-3 file (rebuild_emulator refuses it), an
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

    # The science the artifact was born under, kept rather than dropped. The
    # rebuild has always read these two blocks; nothing downstream held on to
    # them, so a consumer had nothing to compare against and every comparison
    # law had nowhere to run. That is why a w-varying emulator could be served
    # to a cosmological-constant chain, and why a point outside the sampled
    # region got a confident answer instead of a refusal.
    #
    # This binding runs before every family branch below returns, because the
    # record is not a property of the cosmic-shear branch or the scalar branch.
    # It is a property of the file, and every family has one.
    self.record = {fixed_facts.FIXED_FACTS_GROUP:  info["fixed_facts"],
                   fixed_facts.INPUT_DOMAIN_GROUP: info["input_domain"]}
    self.fixed_facts  = self.record[fixed_facts.FIXED_FACTS_GROUP]
    self.input_domain = self.record[fixed_facts.INPUT_DOMAIN_GROUP]
    self.composition_mode = info["composition_mode"]
    self.transfer_refined = info["transfer_refined"]
    pce_base, transfer_base = _select_composition(
      composition_mode=self.composition_mode,
      pce_base=info["pce_base"],
      transfer_base=info["transfer_base"])
    pce_form = info["pce_form"]
    transfer_form = info["transfer_form"]
    transfer_space = info["transfer_space"]
    # the artifact's identity, named in every refusal the laws raise. A refusal
    # that does not say WHICH file it refused sends the reader back to a config
    # to guess, and a chain that serves several emulators has several to guess
    # between.
    self._where = str(path_root)
    # the sampled region, parsed out of the record's text ONCE. predict()
    # compares every point against it, so a chain pays this parse a single time
    # instead of once per step. fixed_facts compiles it and fixed_facts compares
    # against it; this class only holds what it was handed.
    self._support = fixed_facts.compile_support(blocks=self.record,
                                                where=self._where)

    self.names = list(self.pgeom.names)
    recipe = info["model_recipe"]
    self._input_dim = int(recipe["input_dim"])
    self._decoded_dim = int(self.geom.dest_idx.numel())
    model_output_dim = int(recipe["output_dim"])
    if model_output_dim != self._decoded_dim:
      raise ValueError(
        self._where + ": model_recipe output_dim="
        + repr(model_output_dim) + " disagrees with the saved output "
        "geometry width " + repr(self._decoded_dim) + ".")
    if recipe["ia"] is None:
      self._model_output_shape = (1, model_output_dim)
    else:
      n_templates = int(recipe["kwargs"]["n_templates"])
      self._model_output_shape = (1, n_templates, model_output_dim)
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
        composition_mode=self.composition_mode,
        pce_base=pce_base,
        pce_form=pce_form,
        transfer_base=transfer_base,
        transfer_form=transfer_form,
        transfer_space=transfer_space)
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
        composition_mode=self.composition_mode,
        pce_base=pce_base,
        pce_form=pce_form,
        transfer_base=transfer_base,
        transfer_form=transfer_form,
        transfer_space=transfer_space)
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
        composition_mode=self.composition_mode,
        pce_base=pce_base,
        pce_form=pce_form,
        transfer_base=transfer_base,
        transfer_form=transfer_form,
        transfer_space=transfer_space)
      return
    # CMB spectrum emulator: predict returns the physical C_ell
    # row on the stored multipole grid (a 1-D numpy array over .ell), so
    # skip the 3x2pt mask/section accounting a CmbDiagonalGeometry does
    # not have. The decoder is law-dispatched: the training chi2's decode
    # (losses/cmb.py) divides out the factor applied before encoding. The
    # predictor reuses that decoder rather than copying the law equation.
    if self._cmb:
      self.spectrum       = self.geom.spectrum
      self.ell            = self.geom.ell
      self.units          = self.geom.units
      self.amplitude_law  = info["amplitude_law"]
      # an NPCE or transfer cmb artifact composes base + net (law "none"
      # enforced at training); otherwise the law-dispatched decode.
      if self.composition_mode != "plain":
        if info["amplitude_law"] != "none":
          kind = ("an NPCE base" if self.composition_mode == "npce"
                  else "a transfer base")
          raise ValueError(
            "the saved emulator carries both " + kind + " and "
            "amplitude_law " + repr(info["amplitude_law"]) + "; the two "
            "are mutually exclusive (validate_cmb), so the file is "
            "inconsistent")
        self._decode = self._build_diag_decoder(
          composition_mode=self.composition_mode,
          pce_base=pce_base,
          pce_form=pce_form,
          transfer_base=transfer_base,
          transfer_form=transfer_form,
          transfer_space=transfer_space)
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
    # a transfer artifact embeds its frozen base (info["transfer_base"] holds
    # the rebuilt base model + both geometries; form / space say how to
    # compose). On such a run info["ia"] is the CORRECTION net's inherited
    # design, consumed by the transfer decoder, not a standalone factored run.
    if ia is not None and self.composition_mode == "npce":
      raise ValueError(
        "the saved emulator carries both a factored-IA design and an NPCE "
        "base; the two are mutually exclusive (pce excludes ia), so the "
        "file is inconsistent")

    # the physical-dv decoder: reuse the EXACT training chi2fn.decode so the
    # amplitude combine / NPCE recombine / transfer composition are
    # single-sourced, never re-derived here (the drift channel the standing
    # rule kills).
    self._decode = self._build_decoder(ia=ia,
                                       composition_mode=self.composition_mode,
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

  # ----- the three questions a consumer must be able to ask this artifact ----
  #
  # They are three different questions, and answering one does not answer any
  # other. An emulator can belong to the cosmology being sampled and still be
  # asked about a point it never saw. Two emulators can be a matched pair and
  # both belong to the wrong universe. A point can sit inside one emulator's
  # region and outside its partner's. Each method below asks exactly one of the
  # three, at the one site that knows which artifact is being asked about.
  #
  # The laws are in emulator/fixed_facts.py and stay there: this class owns the
  # SITE of a comparison, that module owns the LAW and the words it refuses in.
  # A refusal restated here would be a second author of the same sentence, and
  # two authors of one sentence are how the two copies drift apart.

  def check_belongs_to(self, resolved_model):
    """Vertical: does this emulator belong to the cosmology being sampled?

    The emulator's dataset was generated with some coordinates held fixed: a
    neutrino mass, an equation of state, a curvature. They are not inputs of
    the network. They are properties of the universe it learned, and they are
    not visible in anything it returns. A chain that holds any of them at a
    different value is asking a question about a different universe, and this
    emulator will answer it: confidently, in the right shape, with the right
    sign, and wrong.

    Why this is not the other two questions. It compares ONE artifact against
    the chain that wants to use it, and it reads only the coordinates the
    artifact HELD FIXED. The horizontal question (check_pairs_with) compares
    two artifacts against each other and never looks at the chain at all. The
    domain question (check_may_serve) compares one point against the region
    this artifact was trained over, and never looks at what was held fixed. An
    emulator that passes this law can still be asked about a point far outside
    the region it saw.

    Arguments:
      resolved_model = the cosmology being sampled, as a plain mapping from
                       coordinate name to value, RESOLVED rather than
                       requested: a default the YAML left unstated has been
                       materialized by the time the model object exists, and it
                       is the materialized value the chain actually samples.
                       The consumer resolves it (from cobaya, from a script)
                       and hands it in; neither this module nor fixed_facts
                       imports cobaya.

    Returns:
      None. The method is called for its refusal.

    Raises:
      ValueError naming the coordinate that disagrees, the value this artifact
      was generated with, the value being sampled, and what to do about it.
    """
    fixed_facts.check_vertical(blocks=self.record,
                               resolved_model=resolved_model,
                               where=self._where)

  def check_pairs_with(self, other):
    """Horizontal: do these two emulators belong to each other?

    Emulators are served in pairs all the time: a Hubble rate beside an angular
    diameter distance, a linear power spectrum beside its nonlinear boost, a TT
    spectrum beside an EE. Each pair is combined into ONE prediction, so each
    pair must come from one dataset and one cosmology. Two independent
    generator runs of the same YAML agree on every fixed fact and every bound
    and still drew different points, so comparing their facts cannot tell them
    apart. The dataset identity can, and the law compares it first.

    Why this is not the other two questions. It compares two artifacts against
    EACH OTHER, and the chain never enters it: a pair can match each other
    exactly and both belong to a universe nobody is sampling, which is what
    check_belongs_to is for. Nor does it say anything about where the pair may
    be evaluated: two emulators from one dump agree here and may still have
    been sampled over different regions, which is what served_support_with is
    for.

    Arguments:
      other = the other EmulatorPredictor, the one this artifact would be
              served beside.

    Returns:
      None. The method is called for its refusal.

    Raises:
      ValueError naming the fact the two disagree about, both artifacts by
      path, both of their values, and what to do about it.
    """
    fixed_facts.check_horizontal(blocks_a=self.record,
                                 blocks_b=other.record,
                                 where_a=self._where,
                                 where_b=other._where)

  def check_may_serve(self, point):
    """Domain: may this emulator be asked about this point?

    Inside the region it was trained over, the emulator interpolates. Outside
    it, the emulator extrapolates: it returns a number of the right shape, with
    the right sign, and no warning of any kind. The region is the contract the
    dataset was generated under, and this is the refusal that enforces it.

    Why this is not the other two questions. It compares ONE point against ONE
    artifact's sampled region, and it reads only the coordinates the generator
    SAMPLED. The two equality questions read the coordinates that were held
    FIXED, which are exactly the ones a point cannot vary. This is also the
    only one of the three that intersects rather than matches: a pair's served
    region is the overlap of the two, which is why served_support_with exists
    and why this method answers only for this artifact.

    predict() runs this same law on every point it is handed, so a consumer
    does not have to remember to. This method stays as the surface that asks
    the question WITHOUT asking for a prediction: a script that wants to know
    whether a point is servable, or that walks a proposed region and reports
    where the refusals would start, asks here and gets the refusal by itself.

    Arguments:
      point = the point being asked about, a mapping from name to value. Only
              the coordinates the generator sampled are read, so a mapping that
              carries more of them (a whole cobaya parameter block, say) is
              fine.

    Returns:
      None. The method is called for its refusal.

    Raises:
      ValueError naming the coordinate, the interval this artifact was trained
      over, the value it is being asked about, and what to do about it; also
      when the artifact declares no support at all, which is the shape of a
      test double and must never be served.
    """
    fixed_facts.check_domain(blocks=self.record,
                             point=point,
                             where=self._where)

  def served_support_with(self, other):
    """The region a PAIR of emulators may be served over: their overlap.

    Two emulators combined into one prediction can only be asked about a point
    both of them were trained over. The served region is the intersection of
    the two, never the union: a point inside one half and outside the other is
    a point where one half extrapolates, and the combined answer inherits that
    silently.

    This is the reporting half of the domain question, for a pair. It answers
    "where may we ask?" rather than "may we ask here?", so a consumer can print
    the region it is allowed to sample instead of discovering it one refusal at
    a time.

    Arguments:
      other = the other EmulatorPredictor, the one this artifact would be
              served beside.

    Returns:
      a mapping from parameter name to (low, high), the overlapped region, as
      Python floats.

    Raises:
      ValueError when either artifact declares no box of intervals to intersect
      (a test double), or when the two regions do not overlap on some
      coordinate, in which case the pair has no point it can be asked about at
      all.
    """
    return fixed_facts.served_support(blocks_a=self.record,
                                      blocks_b=other.record,
                                      where_a=self._where,
                                      where_b=other._where)

  def _build_diag_decoder(self, composition_mode, pce_base, pce_form,
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
      composition_mode = validated plain / npce / transfer fact.
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
    pce_base, transfer_base = _select_composition(
      composition_mode=composition_mode,
      pce_base=pce_base,
      transfer_base=transfer_base)
    if composition_mode == "transfer":
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
    if composition_mode == "plain":
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

    Reconstructs the same loss object training used, solely to reuse its
    decode method. For ``as_exp2tau_ref``, decode divides out the factor
    that encode applied. For ``none``, decode only reverses the geometry's
    centering and scaling. The equation therefore has one owner in
    ``losses/cmb.py``.

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

  def _build_decoder(self, ia, composition_mode, pce_base, pce_form,
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
      composition_mode = validated plain / npce / transfer fact.
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
    pce_base, transfer_base = _select_composition(
      composition_mode=composition_mode,
      pce_base=pce_base,
      transfer_base=transfer_base)
    if composition_mode == "transfer":
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

    if composition_mode == "npce":
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

  def _ordered_values(self, params):
    """Order the inputs into this emulator's own parameter order, by name.

    Every input form this method accepts carries the parameter names beside the
    values, and the names are proved against the order the whitening geometry
    holds:

      a mapping   {"omegabh2": 0.0223, "omegach2": 0.119}
                  read in .names order, so it has no order to get wrong.

      a pair      (["omegabh2", "omegach2"], [0.0223, 0.119])
                  the names are checked against .names, then the values are
                  read in that order.

    A bare row of numbers is refused. It carries no names, so nothing in it can
    say which parameter each number is. A permuted row has exactly the right
    length, passes the only test a length is able to make, and is then whitened
    against the wrong parameter's columns: every prediction that follows is
    confident and wrong, and nothing about the numbers looks unusual. Length is
    not a defence against a permutation, and it was the only defence here.

    A caller that has already proved its own order against .names does not come
    through this method; it builds the row through _as_row_trusted.

    The values are handed back as a plain list rather than as the tensor,
    because predict() has one more question to ask before any number reaches the
    network: whether the point they spell out is inside the region this emulator
    was trained over. The domain law reads values, not tensors.

    Arguments:
      params = a mapping from parameter name to value, or a (names, values)
               pair: a 2-item sequence whose first item is the parameter names
               and whose second is their values, in the same order.

    Returns:
      the values, in this emulator's own parameter order (.names order).

    Raises:
      KeyError naming the first required parameter a mapping is missing;
      ValueError when a pair's names are not this emulator's names in this
      emulator's order, naming both orders; TypeError when the input carries no
      names at all.
    """
    if isinstance(params, dict):
      row = []
      for n in self.names:
        if n not in params:
          raise KeyError(
            f"predict() is missing required parameter {n!r}; the saved "
            f"emulator needs {self.names}")
        row.append(params[n])
      return row

    if _is_named_pair(params):
      names, values = params
      given = list(names)
      # One list comparison, once, on the hot path. It stops at the first name
      # that differs and it builds nothing; the sorted() below pays only on the
      # way to a refusal, where the cost of a sort is not a cost at all.
      if given != self.names:
        if sorted(given) == sorted(self.names):
          fault = (
            "predict() was handed this emulator's own parameters in a "
            "different order. That is the failure this check exists to catch: "
            "the values would be whitened against the wrong parameter's "
            "columns, every prediction would be confident and wrong, and "
            "nothing about the numbers would look unusual.\n")
        else:
          fault = (
            "predict() was handed names that are not the parameters this "
            "emulator was trained on.\n")
        raise ValueError(
          fault
          + "  the names handed in:      " + repr(given) + "\n"
          + "  the emulator's own order: " + repr(self.names) + "\n"
          + "The order is not an incidental detail of the file. It is the "
            "order the whitening matrices were built in, and the saved "
            "geometry holds it. Hand in a mapping from name to value, which "
            "has no order to get wrong, or hand the values in the emulator's "
            "own order.")
      return list(values)

    raise TypeError(
      "predict() was handed a bare ordered sequence (a "
      + type(params).__name__ + "), which carries no parameter names. A row of "
      "numbers cannot say which parameter each number is, so a permutation of "
      "it has exactly the right length, passes the only test a length is able "
      "to make, and is whitened against the wrong parameter's columns: the "
      "prediction is then confident and wrong, and nothing about the numbers "
      "looks unusual. Two input forms carry their names, and either is "
      "accepted here:\n"
      "  a mapping:            {" + repr(self.names[0]) + ": value, ...}\n"
      "  a (names, values) pair: (" + repr(self.names) + ", values)\n"
      "A caller inside this class that has already proved its values are in "
      "the emulator's own order builds the row through _as_row_trusted, the "
      "internal path this refusal guards.")

  def _as_row_trusted(self, values):
    """Build the (1, n_param) tensor from values whose order is already proved.

    The internal path, and the one place a row becomes a tensor. Its caller has
    established that the values arrive in this emulator's own parameter order:
    _ordered_values establishes it by reading a mapping in .names order, or by
    checking a pair's names against .names. Nothing in this method can establish
    it, because a row of numbers carries nothing to establish it with. The
    length is checked, and a permutation has the right length.

    Do not call this from outside the class on a row whose order was not proved
    against .names first. The whole point of the public refusal above is that
    an unproved row cannot be told apart from a wrong one.

    Arguments:
      values = the parameter values, in .names order.

    Returns:
      (1, n_param) tensor in the geometry's whitening dtype on self.device.

    Raises:
      ValueError when the row's length is not this emulator's parameter count.
    """
    row = list(values)
    if len(row) != len(self.names):
      raise ValueError(
        f"predict() got {len(row)} values but the emulator needs "
        f"{len(self.names)} ({self.names})")
    return torch.as_tensor(row, dtype=self._dtype,
                           device=self.device).reshape(1, -1)

  def _validate_ordered_values(self, values):
    """Require one finite real scalar for every saved parameter name.

    Python and NumPy real scalar types are accepted.  Booleans are refused
    explicitly because they are integers in Python and would otherwise enter
    whitening as 0 or 1.  Arrays, tensors, strings, NaN, and infinity are also
    refused before the physical-support comparison or any tensor conversion.

    Arguments:
      values = candidate values already placed in ``self.names`` order.

    Returns:
      a plain list with the same scalar objects and order.
    """
    row = list(values)
    if len(row) != len(self.names):
      raise ValueError(
        "predict() got " + str(len(row)) + " values but the emulator needs "
        + str(len(self.names)) + " (" + repr(self.names) + ")")
    for name, value in zip(self.names, row):
      if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
          "predict() parameter " + repr(name)
          + " must be a finite real scalar, not a Boolean, array, tensor, "
          "string, or other object; got " + repr(value) + " (type "
          + type(value).__name__ + ").")
      try:
        finite = math.isfinite(float(value))
      except (OverflowError, TypeError, ValueError):
        finite = False
      if not finite:
        raise ValueError(
          "predict() parameter " + repr(name)
          + " must be finite; got " + repr(value) + ".")
    return row

  def predict(self, params):
    """Predict the physical data vector at the configured dv_return shape.

    Two things are proved about the point before any number reaches the network,
    and both are proved here because this is the one door every consumer walks
    through:

      the NAMES    each value is the value of the parameter it is paired with,
                   proved against the order the whitening geometry was built in
                   (_ordered_values; a bare row of numbers carries no names and
                   is refused).

      the REGION   the point lies inside the region the generator sampled, read
                   from the artifact's own record (fixed_facts.check_support).

    Neither is optional, and for one reason: an emulator asked outside its
    training region does not fail. It extrapolates — a number of the right
    shape, with the right sign, and no warning — exactly as a permuted row is
    whitened against the wrong columns and answered confidently. A silently
    wrong answer must be refused at the door, not left to each consumer to
    remember to ask for. A development script that wants to WATCH the
    extrapolation is not a consumer of predictions; it drives the internal
    surface (_as_row_trusted) that this refusal guards.

    Arguments:
      params = a mapping from parameter name to value, or a (names, values)
               pair whose names are checked against .names (see _ordered_values:
               a bare ordered row carries no names and is refused). The
               amplitudes among .names are consumed by the factored combine,
               never entered into the network.

    Returns:
      For a scalar (derived-parameter) emulator: a {name: value} dict, one
      entry per emulated output; the dv_return / section machinery
      does not apply. For a CMB spectrum emulator: a 1-D numpy array of
      physical C_ell on the stored multipole grid .ell (the
      imposed amplitude law already reversed); dv_return does not
      apply. For a data-vector emulator: a 1-D numpy array.
      dv_return 'section' (the default): this emulator's own probe block(s),
      shape (section_size,); for a cosmic-shear emulator the xi block, the
      length the likelihood glues per probe. dv_return '3x2pt': the full
      scattered vector (total_size,), the kept entries at their dest_idx
      positions and 0 everywhere else.

    Raises:
      KeyError / ValueError / TypeError from the name proof (_ordered_values);
      ValueError from the domain law, naming the coordinate, the region this
      artifact was trained over, and the value it was asked about.
    """
    values = self._validate_ordered_values(self._ordered_values(params))
    # the point, in the emulator's own order, for the domain law: the values are
    # named by construction here, whichever form the caller handed in.
    point = {}
    for i in range(len(self.names)):
      point[self.names[i]] = values[i]
    fixed_facts.check_support(compiled=self._support, point=point)

    x = _require_prediction_tensor(
      self._as_row_trusted(values=values),
      stage="raw parameter row",
      shape=(1, len(self.names)),
      where=self._where)
    x_enc = _require_prediction_tensor(
      self.pgeom.encode(x),
      stage="parameter encoding",
      shape=(1, self._input_dim),
      where=self._where)
    with torch.no_grad():
      pred = self.model(x_enc)
    pred = _require_prediction_tensor(
      pred,
      stage="model evaluation",
      shape=self._model_output_shape,
      where=self._where)
    decoded = _require_prediction_tensor(
      self._decode(pred, x_enc),
      stage="physical decoding",
      shape=(1, self._decoded_dim),
      where=self._where)
    # scalar (derived-parameter) emulator: destandardize the outputs
    # (the decoder built at init: geom.decode alone, or the NPCE base + net
    # recombine) and return a {name: value} dict, not a data vector; there
    # is no mask to unsqueeze through and no section to slice.
    if self._scalar:
      out = decoded[0]
      result = {}
      for i, nm in enumerate(self.output_names):
        result[nm] = float(out[i])
      return result
    # grid (background-function) emulator: decode inverts the
    # target law (e.g. exp(y) - offset) and returns the physical
    # function keyed by its quantity tag, with the stored grid beside it.
    if self._grid:
      row = decoded[0].detach().cpu().numpy()
      return {"z": self.z.detach().cpu().numpy(),
              self.quantity: row}
    # grid2d emulator: decode destandardizes to LAW SPACE (the
    # base multiply-back is the consumer's one step); the
    # flattened row is reshaped back to the (nz, nk) surface.
    if self._grid2d:
      nz = int(self.z.numel())
      nk = int(self.k.numel())
      surface = decoded[0].detach().cpu().numpy()
      return {"z": self.z.detach().cpu().numpy(),
              "k": self.k.detach().cpu().numpy(),
              self.quantity: surface.reshape(nz, nk)}
    # CMB spectrum emulator: decode reverses the selected target law and
    # returns physical C_ell on the stored multipole grid. This family has
    # no mask to scatter through and no data-vector section to slice.
    if self._cmb:
      return decoded[0].detach().cpu().numpy()
    dv_full = _require_prediction_tensor(
      self.geom.unsqueeze(decoded),
      stage="data-vector scattering",
      shape=(1, self.total_size),
      where=self._where)
    if self.dv_return == "3x2pt":
      out = dv_full[0]
      expected_width = self.total_size
    else:
      out = self._section(dv_full)
      expected_width = sum(
        self.section_sizes[block_id]
        for block_id in self.geom.PROBE_BLOCKS[self.probe])
    out = _require_prediction_tensor(
      out,
      stage="returned data vector",
      shape=(expected_width,),
      where=self._where)
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
      ValueError when the saved geometry does not carry section_sizes / probe,
      naming the two ways out.
    """
    if self.section_sizes is None or self.probe is None:
      raise ValueError(
        "dv_return='section' needs the geometry's section_sizes + probe, "
        "which this saved emulator does not carry. Two "
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
