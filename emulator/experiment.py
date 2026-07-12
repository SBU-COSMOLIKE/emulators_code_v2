"""The EmulatorExperiment: one configured cosmic-shear emulator run.

This class factors the driver setup boilerplate (parse the config, pick the
device, stage the sources, build the parameter + data-vector geometries and
the chi2, assemble the run_emulator spec dicts, train) into one reusable
object a driver or sweep script (over N_train, or one hyperparameter) need
not copy. What exp.run() orchestrates:

    YAML config
       │  from_yaml / from_config   validate blocks, collapse search
       │                            ranges, (name, ia) -> model class
       ▼
    stage_train / stage_val         load the dumps, apply the physics
       │                            cuts, split the rows
       ▼
    build_geometry                  ParamGeometry (whiten in) +
       │                            DataVectorGeometry (whiten out)
       │                            + the chi2 (one cosmolike read);
       │                            needs_bins models also get the
       │                            shear angle map (bin_sizes)
       ▼
    train                           build_specs -> run_emulator
       │                            (model / optimizer / scheduler /
       │                            loaders; one- or two-phase)
       ▼
    model + histories on the instance, ready for diagnostics
       (frac_above, eval_source_chi2, plotting)

(legend: each box is the run's state at that stage, and the label on
each arrow is the EmulatorExperiment method that produces it; ia =
the model.ia key (None / "nla" / "tatt"), the factored-design
choice; needs_bins = the bin-token head's capability flag, which
asks build_geometry for the per-bin split.)

Build it from a YAML file (from_yaml) or a parsed config mapping (from_config,
e.g. load the YAML once and rebuild from a tweaked copy per sweep point); both
resolve the model class from train_args.model.name through MODELS. The
expensive pieces are built by explicit methods and cached:

  - a single run (the driver): exp = from_yaml(...); exp.run().
  - an N_train sweep (geometry per training subset):
      for N in sizes:
        exp.stage_train(n_train=N); exp.stage_val()
        exp.build_geometry(); exp.train()
        f = exp.frac_above(0.2)
  - a hyperparameter sweep (data + geometry fixed, only the spec varies):
      exp.stage_train(); exp.stage_val(); exp.build_geometry()
      for v in values:
        exp.train(train_args=tweaked_copy_with(v))

PS: to whiten is to rotate into the covariance eigenbasis and scale to
unit variance, so correlated quantities become decorrelated; encoded =
a dv put through the geometry's encode (kept entries, centered,
whitened); loader = a closure load(rows) -> a ready-to-train batch on
the device, hiding where the data lives (resident on the GPU, streamed
from RAM, or read from a disk memmap); dump = the full on-disk array
from the data-generation run, one row per cosmology (the dv dump is
the .npy, the param dump the .txt); memmap = a NumPy array backed by
that on-disk file, read in slices so it is never fully loaded.
"""

import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from .data_staging import (
  read_param_names, load_source, load_scalar_source, phys_cut_idx)
from .geometries.parameter import ParamGeometry, AmplitudeFactorGeometry
from .losses.core import make_chi2
from .designs.plain import ResMLP, ResCNN, ResTRF
from .designs.ia import (TemplateMLP, TemplateResCNN,
                         TemplateResTRF)
from .losses.ia import (TemplateFactoredChi2, nla_coeffs,
                        tatt_coeffs)
from .activations import make_activation
from .designs.blocks import make_norm
from . import warmstart
from .losses.transfer import (
  TransferChi2, TransferDiagChi2, FORMS, SPACES, RECOMMENDED_SPACE)
from .training import (
  run_emulator, build_run_specs, pick_device, make_logger,
  default_train_args, eval_source_chi2, DEFAULT_COMPILE_MODE,
  validate_phase_block, _PHASE_BLOCK_KEYS,
  validate_loss, _loss_migration_message)


# (architecture, ia) -> model class. Two orthogonal YAML choices:
# train_args.model.name picks the architecture (resmlp = residual MLP;
# rescnn = + a theta-order 1D CNN correction head; restrf = + a
# bin-token transformer correction head, attention across the
# tomographic bins with per-bin unique MLPs), and the separate
# train_args.model.ia key layers a factored intrinsic-alignment design
# on it (absent/None = the plain emulator; "nla" = the model emits
# three templates from the non-amplitude inputs and the loss combines
# them in closed form as K0 + A1*K1 + A1^2*K2, so the amplitude never
# enters the network, exact generalization in A1; "tatt" = the
# same factoring with 3 amplitudes (a1, a2, b_TA) and 10 templates,
# exact in all three, see tatt_coeffs). The classes carry
# capability flags build_geometry / build_specs read: factored
# (AmplitudeFactorGeometry input + the template-combining loss),
# needs_geom (geom injected for the fixed full<->theta basis buffers;
# compile_mode defaulted to "default"), and needs_bins
# (build_shear_angle_map run on the cosmolike data geometry — or
# attach_head_coords() on a diagonal family geometry — attaching the
# per-bin split the heads need).
MODELS = {("resmlp", None):   ResMLP,
          ("rescnn", None):   ResCNN,
          ("restrf", None):   ResTRF,
          ("resmlp", "nla"):  TemplateMLP,
          ("rescnn", "nla"):  TemplateResCNN,
          ("restrf", "nla"):  TemplateResTRF,
          # tatt reuses the same factored classes: only the
          # IA_DESIGNS entry (amplitude columns, polynomial,
          # template count) differs, the classes are generic in
          # n_amps / n_templates.
          ("resmlp", "tatt"): TemplateMLP,
          ("rescnn", "tatt"): TemplateResCNN,
          ("restrf", "tatt"): TemplateResTRF}

# the amplitude column the NLA design factors out of the network input.
# LSST_A1_1 is the NLA amplitude (enters xi as a linear field
# coefficient, so it factors exactly); LSST_A1_2 is the redshift-
# evolution power eta, which sits inside the projection integral and
# stays an emulated input.
NLA_AMP_NAMES = ["LSST_A1_1"]

# the three amplitude columns the TATT design factors out, in
# tatt_coeffs order [a1, a2, b_TA]: LSST_A1_1 (the linear/TA
# amplitude), LSST_A2_1 (the quadratic/TT amplitude), LSST_BTA_1
# (the density-weighting bias). The redshift-evolution powers
# (LSST_A1_2, LSST_A2_2) sit inside the projection integrals and
# stay emulated inputs, exactly as NLA's eta does.
TATT_AMP_NAMES = ["LSST_A1_1", "LSST_A2_1", "LSST_BTA_1"]

# one entry per factored IA design (the model.ia key): the amplitude
# columns the input geometry appends (in coeff_fn order), the amplitude
# polynomial, and the template count the model emits. Everything
# downstream (AmplitudeFactorGeometry, TemplateFactoredChi2, the
# models' n_amps / n_templates, the conv head's groups values) reads
# this entry, a new design is a new entry, never new code paths.
# note (tatt): the entry is live, but the template training dumps do
# not exist yet, a tatt run needs dv dumps holding the 10 templates.
IA_DESIGNS = {"nla":  {"amp_names":   NLA_AMP_NAMES,
                       "coeff_fn":    nla_coeffs,
                       "n_templates": 3},
              "tatt": {"amp_names":   TATT_AMP_NAMES,
                       "coeff_fn":    tatt_coeffs,
                       "n_templates": 10}}

# The nested model-block schema. The YAML groups each component's
# knobs in its own sub-block (mlp / activation / cnn / trf), so the
# nesting carries the context and the keys stay short, no
# n_blocks-vs-n_blocks_cnn suffixes. The tables below map each
# sub-block's YAML keys onto the model constructors' flat argument
# names (the constructors are internal API; the YAML is the
# interface). An unknown key raises, listing what is allowed. The
# cnn / trf "activation" key maps to head_act (the per-head activation
# pin), but build_specs special-cases its value (a {type, n_gates}
# factory spec, not a scalar) before the generic copy.
MODEL_BLOCK_KEYS = {
  "mlp": {"width":        "int_dim_res",
          "n_blocks":     "n_blocks"},
  "cnn": {"kernel_size":    "kernel_size",
          "rescale_kernel": "rescale_kernel",
          "groups":         "groups",
          "separable":      "separable",
          "film":           "film",
          "n_blocks":       "n_blocks_cnn",
          "gate_init":      "gate_init",
          "activation":     "head_act"},
  "trf": {"n_heads":      "n_heads",
          "n_blocks":     "n_blocks_trf",
          "n_mlp_blocks": "n_mlp_blocks",
          "n_tokens":     "n_tokens",
          "shared_mlp":   "shared_mlp",
          "film":         "film",
          "gate_init":    "gate_init",
          "activation":   "head_act"},
}

def _head_activation_spec(value, source):
  """
  Validate a per-head activation value into a {type, n_gates} spec.

  The per-head pin (model.cnn / .trf.activation, or the head: activation:
  alias) takes the same two shapes as the top-level model.activation: a
  bare type string, or a mapping {type, n_gates}. Unlike the top-level key
  it is strict (an unknown sub-key raises instead of being silently
  ignored), so a per-head typo cannot quietly fall back to the shared
  family. A standalone pure function (no torch), so build_specs stays
  unit-testable.

  Arguments:
    value  = the raw YAML value: a str (the type), or a mapping with
             "type" (required) and an optional "n_gates".
    source = the config path, named in every error (e.g.
             "model.trf.activation").

  Returns:
    {"type": str, "n_gates": int} (n_gates default 3, the make_activation
    default), ready for make_activation(type, n_gates=...).

  Raises:
    TypeError if value is neither a str nor a mapping; ValueError on a
    mapping without "type" or carrying a key outside {type, n_gates}.
  """
  if isinstance(value, str):
    return {"type": value, "n_gates": 3}
  if not isinstance(value, dict):
    raise TypeError(
      f"{source} must be a type string or a mapping {{type, n_gates}}, "
      f"got {type(value).__name__}")
  unknown = set(value) - {"type", "n_gates"}
  if unknown:
    raise ValueError(
      f"unknown key(s) {sorted(unknown)} in {source}; a per-head "
      f"activation takes only type / n_gates")
  if "type" not in value:
    raise ValueError(
      f"{source} needs a 'type' (the activation family, e.g. H / "
      f"gated_power)")
  return {"type": str(value["type"]),
          "n_gates": int(value.get("n_gates", 3))}


def _resolve_head_activation(canonical, alias, head_block, trunk_epochs,
                             freeze_trunk):
  """
  Resolve the per-head activation pin (canonical vs alias) and license it.

  The head's activation may be pinned canonically
  (model.<head_block>.activation) or through the head: activation: alias;
  they are one setting, so giving both is a config error (no silent
  winner, even when the two agree — the loss berhu: / mode-named
  precedent). The pin is licensed only by a frozen-trunk head phase: the
  head family is the head-phase family, so it needs trunk_epochs > 0 and
  freeze_trunk true (absent = true). When the trunk and head train
  together the network keeps one family. Pure (no torch).

  Arguments:
    canonical    = the {type, n_gates} spec from model.<head>.activation,
                   or None (already validated by _head_activation_spec).
    alias        = the {type, n_gates} spec from head: activation:, or
                   None (already validated).
    head_block   = the active head's block name ("cnn" / "trf"), named in
                   every message as the canonical spelling.
    trunk_epochs = the resolved trunk_epochs (0 = no two-phase schedule).
    freeze_trunk = the resolved freeze_trunk (None / True = a frozen head
                   phase, False = joint).

  Returns:
    the pin spec {type, n_gates}, or None when neither spelling is set.

  Raises:
    ValueError when both spellings are given, or when a pin is set without
    a frozen-trunk head phase (the license rule).
  """
  if canonical is not None and alias is not None:
    raise ValueError(
      f"the head activation is set twice: model.{head_block}.activation "
      f"and the head: activation: alias name the same family; keep one "
      f"(the canonical model.{head_block}.activation is recommended, it "
      f"also reads on single-phase YAMLs)")
  pin = canonical if canonical is not None else alias
  if pin is None:
    return None
  if not (trunk_epochs > 0 and freeze_trunk is not False):
    raise ValueError(
      f"model.{head_block}.activation: a per-head activation needs a "
      f"frozen-trunk head phase (trunk_epochs > 0 and freeze_trunk "
      f"true): the head family is the head-phase family. With the trunk "
      f"and head training together the network keeps one family — set "
      f"model.activation only.")
  return pin


def _activation_flag_notice(flag_type, head_block, head_pin):
  """
  The startup warning when an explicit --activation flag meets a head pin.

  Ruling (a): --activation (and model.activation) set the trunk + default
  family; an explicit per-head pin (model.<head>.activation or the head:
  activation: alias) holds for the head. When the two name different
  families the pin silently wins for the head, so one line names both. No
  warning when the flag is absent (the drivers pass None), when no head pin
  exists, or when the flag and the pin agree (no surprise). Pure (no
  torch), so it is exercised case-by-case off the tree.

  Arguments:
    flag_type  = the explicit --activation type string, or None when the
                 flag was absent (the driver default).
    head_block = the active head's block name ("cnn" / "trf"), or None
                 (a resmlp / plain model has no head to pin).
    head_pin   = the active head's activation pin, from either spelling: a
                 type string, a mapping {type, n_gates}, or None.

  Returns:
    the warning string, or None when no surprise is possible.
  """
  if flag_type is None or head_block is None or head_pin is None:
    return None
  pin_type = (head_pin.get("type") if isinstance(head_pin, dict)
              else head_pin)
  if pin_type is None or str(pin_type) == str(flag_type):
    return None
  return (f"warning: --activation {flag_type} sets the trunk/default "
          f"only; the head keeps its model.{head_block}.activation pin "
          f"({pin_type})")


def _pinned_head_warning(train_args, head_block, what_varies):
  """
  One-line warning when the active head is pinned while a driver sweeps the
  shared activation family.

  The bake-off and a model.activation sweep write the shared / trunk family
  per curve; a per-head pin (model.<head>.activation or the head:
  activation: alias) stays fixed across every curve, so it is worth a
  heads-up (unlike the from_config warning, this fires whenever a pin
  exists, since the swept value changes per curve). None when there is no
  head (resmlp) or no pin. Pure (no torch).

  Arguments:
    train_args  = the resolved train_args mapping.
    head_block  = the active head's block name ("cnn" / "trf"), or None.
    what_varies = the tail clause naming what the driver sweeps (e.g.
                  "it stays fixed across the bake-off").

  Returns:
    the warning string, or None.
  """
  if head_block is None:
    return None
  model = train_args.get("model", {})
  pin = model.get(head_block, {}).get("activation")
  if pin is None and isinstance(train_args.get("head"), dict):
    pin = train_args["head"].get("activation")
  if pin is None:
    return None
  pin_type = pin.get("type") if isinstance(pin, dict) else pin
  return (f"warning: model.{head_block}.activation pins the head family "
          f"({pin_type}); {what_varies}")


# default reported delta-chi2 cutoffs; the first (0.2) is the emulator goal
# and the best-model-selection metric.
DEFAULT_THRESHOLDS = torch.tensor([0.2, 0.5, 1.0, 10.0, 100.0])

# the allowed "data" block keys; from_config rejects any other. The
# physical window cuts live in the nested param_cuts sub-block (its own
# whitelist below), not flat here. Keep in sync with the data-block
# docstring in __init__.
DATA_KEYS = {
  "train_dv", "train_params", "train_covmat",
  "val_dv", "val_params",
  "cosmolike_data_dir", "cosmolike_dataset",
  "param_cuts",
  "n_train", "n_val",
  "split_seed", "ram_frac",
  # scalar (derived-parameter) run: the emulated output names. Its presence
  # switches from_config to the scalar path (validate_scalar), where the
  # dv / cosmolike keys are forbidden and param_cuts is optional.
  # CMB-spectrum run: the data.cmb sub-block {spectrum, covariance,
  # amplitude_law, as_name, tau_name}. Its presence switches from_config to
  # the CMB path (validate_cmb): dv files required, cosmolike keys
  # forbidden, param_cuts optional, the covariance from the
  # compute_cmb_covariance.py script.
  "cmb",
  # grid (background-function) run: the data.grid sub-block
  # {quantity, units, law, offset, z_file}. Its presence switches
  # from_config to the grid path (validate_grid): dv files required (rows
  # over the stored z grid), cosmolike keys forbidden, param_cuts optional.
  "grid",
  # grid2d (matter-power-spectrum) run: the data.grid2d sub-block
  # {quantity, units, law, z_file, k_file, k_stride, train_base,
  # val_base}. Its presence switches from_config to the grid2d path
  # (validate_grid2d): dv files required (flattened (z, k) rows), the
  # syren-law base files beside them, cosmolike keys forbidden.
  "grid2d",
  "outputs",
}

# the keys the nested data.param_cuts sub-block accepts (the physical
# window bounds threaded to phys_cut_idx). omegabh2_hi is required
# inside the block (the former mandatory omegabh2_cut, renamed); the
# other seven are optional. The whitelist is the omegamh2-vs-omegam2h2
# one-character typo guard.
PARAM_CUTS_KEYS = {
  "omegabh2_lo", "omegabh2_hi",
  "omegam2h2_lo", "omegam2h2_hi",
  "omegamh2_lo", "omegamh2_hi",
  "omegamh2ns_lo", "omegamh2ns_hi",
}


def _param_cuts_migration_message(data, flat):
  """Build the paste-ready param_cuts block for the flat-key migration.

  Arguments:
    data = the flat "data" mapping (read for the offending keys' values).
    flat = the cut keys found flat under data (including the legacy
           omegabh2_cut).

  Returns:
    a multi-line message whose body is a valid, paste-ready
    param_cuts: block, with omegabh2_cut renamed to omegabh2_hi and
    every offending key's value carried over.
  """
  lines = [
    "the physical cut keys moved into a nested data.param_cuts sub-block,",
    "and omegabh2_cut was renamed omegabh2_hi. Replace the flat keys "
    "under data: with:",
    "",
    "  param_cuts:",
  ]
  for k in flat:
    new_k = "omegabh2_hi" if k == "omegabh2_cut" else k
    lines.append(f"    {new_k}: {data[k]}")
  return "\n".join(lines)


def validate_param_cuts(data):
  """
  Validate the data.param_cuts sub-block (the physical window cuts).

  A standalone pure function (no torch), so it is unit-testable in
  isolation. Raises loudly on the migration from the old flat layout,
  on the renamed / unknown keys, and on a missing required bound;
  otherwise returns the param_cuts mapping unchanged.

  Arguments:
    data = the parsed "data" block mapping.

  Returns:
    the validated data["param_cuts"] mapping.

  Raises:
    ValueError on: any cut key (including omegabh2_cut) still flat under
    data (a migration error printing the paste-ready block); a missing
    param_cuts block; omegabh2_cut written inside param_cuts (naming the
    rename); an unknown param_cuts key; a missing required omegabh2_hi.
    TypeError if param_cuts is not a mapping.
  """
  # the old flat layout: any cut key (incl. the renamed omegabh2_cut)
  # directly under data -> a migration error printing the block to
  # paste in its place.
  flat = []
  for k in data:
    if k in PARAM_CUTS_KEYS or k == "omegabh2_cut":
      flat.append(k)
  if flat:
    raise ValueError(_param_cuts_migration_message(data, flat))
  # param_cuts is required.
  if "param_cuts" not in data:
    raise ValueError(
      "data is missing the required 'param_cuts' block; add e.g.\n"
      "  param_cuts:\n"
      "    omegabh2_hi:  0.035   # required (the former omegabh2_cut)\n"
      "    omegabh2_lo:  0.005")
  pc = data["param_cuts"]
  if not isinstance(pc, dict):
    raise TypeError(
      f"data.param_cuts must be a mapping of window bounds, got "
      f"{type(pc).__name__}")
  # omegabh2_cut written inside param_cuts -> name the rename.
  if "omegabh2_cut" in pc:
    raise ValueError(
      "data.param_cuts has 'omegabh2_cut'; it was renamed 'omegabh2_hi' "
      "(rename the key, keep the value)")
  # unknown key inside param_cuts (the omegamh2-vs-omegam2h2 typo guard).
  unknown = set(pc) - PARAM_CUTS_KEYS
  if unknown:
    raise ValueError(
      f"unknown data.param_cuts key(s): {sorted(unknown)}; allowed: "
      f"{sorted(PARAM_CUTS_KEYS)}")
  # omegabh2_hi is the one required bound (the former omegabh2_cut).
  if "omegabh2_hi" not in pc:
    raise ValueError(
      "data.param_cuts is missing the required 'omegabh2_hi' (the upper "
      "omega_b h^2 bound, the former omegabh2_cut)")
  return pc


def validate_scalar(cfg, train_args, rescale="none"):
  """
  Validate a scalar (derived-parameter) run and return its output names.

  A scalar run is signalled by a data.outputs list: inputs and outputs are
  both named columns of one parameter .txt, with no data vector and no
  cosmolike. This enforces that exclusivity and forbids the
  data-vector-only features, then returns the validated output names. A
  standalone pure function (no torch), unit-testable in isolation.

  Arguments:
    cfg        = the parsed config mapping (reads cfg["data"], cfg["pce"],
                 cfg["transfer"]).
    train_args = the resolved train_args (reads finetune / model.ia).
    rescale    = the driver's --rescale value (a scalar run forbids any
                 rescaling; default "none").

  Returns:
    the data.outputs list (non-empty, unique names).

  Raises:
    ValueError if outputs is empty / has duplicate names; if any
    data-vector or cosmolike key is present (train_dv / val_dv /
    cosmolike_data_dir / cosmolike_dataset); if a required param file /
    covmat is missing; if a data-vector-only feature is requested
    (rescale, model.ia, transfer, finetune). The pce: block is legal
    here (the 2026-07-12 family-wide NPCE ruling); validate_pce vets it
    with diagonal=True on the from_config scalar branch.
  """
  data = cfg["data"]
  outputs = data["outputs"]
  if not isinstance(outputs, list) or len(outputs) == 0:
    raise ValueError(
      "data.outputs must be a non-empty list of derived-parameter names "
      f"to emulate (e.g. [H0, omegam]); got {outputs!r}")
  if len(set(outputs)) != len(outputs):
    seen = {}
    dups = []
    for nm in outputs:
      seen[nm] = seen.get(nm, 0) + 1
    for nm, c in seen.items():
      if c > 1:
        dups.append(nm)
    raise ValueError(
      f"data.outputs has duplicate names {sorted(dups)!r}; each emulated "
      "output must be listed once")
  # exclusivity: a scalar run carries no data vector and no cosmolike.
  forbidden = []
  for k in ("train_dv", "val_dv", "cosmolike_data_dir", "cosmolike_dataset"):
    if k in data:
      forbidden.append(k)
  if forbidden:
    raise ValueError(
      f"a scalar run (data.outputs present) must not carry data-vector or "
      f"cosmolike keys, but has {sorted(forbidden)!r}; the scalar path reads "
      "only the parameter .txt (inputs) and its named output columns")
  # scalar needs the parameter files + the input covmat.
  for k in ("train_params", "val_params", "train_covmat"):
    if k not in data:
      raise ValueError(
        f"a scalar run needs data.{k} (the parameter .txt / input covmat); "
        "it is missing")
  # the data-vector-only features do not compose with a scalar run (V1).
  if rescale != "none":
    raise ValueError(
      f"--rescale {rescale!r} is a data-vector concept; a scalar run has no "
      "analytic rescaling (drop --rescale / leave it none)")
  ia = train_args.get("model", {}).get("ia")
  if ia not in (None, "none"):
    raise ValueError(
      f"train_args.model.ia {ia!r} is an intrinsic-alignment (data-vector) "
      "design; a scalar run has no ia (remove it)")
  if cfg.get("transfer") is not None:
    raise ValueError(
      "transfer learning is out of scope for scalar emulators (a "
      "recorded ruling); remove the transfer: block")
  # fine-tuning IS supported: the finetune block is validated by
  # warmstart.validate_finetune_config on the from_config scalar
  # branch, and the source ScalarGeometry is pinned in build_geometry
  # after the outputs-equal admissibility check.
  return outputs


def validate_cmb(cfg, train_args, rescale="none"):
  """
  Validate a CMB-spectrum run and return its data.cmb block.

  A CMB run is signalled by the data.cmb sub-block: one spectrum's C_ell
  rows are the data vector (train_dv / val_dv .npy dumps), the loss
  covariance comes from the compute_cmb_covariance.py script's .npz,
  and the primary amplitude may be imposed by a law instead of learned
  (a registry name persisted in the artifact).
  This enforces the exclusivity and the forbidden features, then returns
  the validated block. A standalone pure function (no torch),
  unit-testable in isolation.

  Arguments:
    cfg        = the parsed config mapping (reads cfg["data"],
                 cfg["pce"], cfg["transfer"]).
    train_args = the resolved train_args (reads finetune / model.ia).
    rescale    = the driver's --rescale value (a CMB run forbids any
                 analytic rescaling; default "none").

  Returns:
    the data.cmb mapping (spectrum / covariance / amplitude_law
    validated; as_name / tau_name present exactly when the law needs
    them).

  Raises:
    ValueError on: a missing or unknown sub-key; a spectrum outside
    tt/te/ee/pp; an amplitude law outside the registry; law-column
    names missing (as_exp2tau) or present (none); a cosmolike or
    scalar key beside data.cmb; a missing dv/params/covmat file key;
    a data-vector-only feature (rescale, model.ia); a pce: or
    transfer: block beside an amplitude_law other than "none" (each
    replaces the target construction — one at a time). The pce: and
    transfer: blocks are otherwise legal (the 2026-07-12 family-wide
    rulings); validate_pce / validate_transfer vet them with
    diagonal=True on the from_config cmb branch.
  """
  # the amplitude-law registry lives with the CMB loss; imported here
  # (not at module top) so this validator stays importable in the same
  # torch-light contexts as the rest of the config logic.
  from .losses.cmb import AMPLITUDE_LAWS

  data = cfg["data"]
  cmb = data["cmb"]
  if not isinstance(cmb, dict):
    raise ValueError(
      "data.cmb must be a mapping {spectrum, covariance, amplitude_law"
      "[, as_name, tau_name]}; got " + repr(type(cmb).__name__))
  allowed = {"spectrum", "covariance", "amplitude_law", "as_name",
             "tau_name"}
  unknown = sorted(set(cmb) - allowed)
  if unknown:
    raise ValueError(
      "unknown data.cmb key(s) " + repr(unknown) + "; allowed: "
      + repr(sorted(allowed)))
  for key in ("spectrum", "covariance", "amplitude_law"):
    if key not in cmb:
      raise ValueError(
        "data.cmb needs the " + repr(key) + " key (spectrum = which C_ell "
        "to emulate; covariance = the compute_cmb_covariance.py script's "
        ".npz; amplitude_law = the imposed-amplitude registry name); it "
        "is missing")
  spectrum = str(cmb["spectrum"]).lower()
  if spectrum not in ("tt", "te", "ee", "pp"):
    raise ValueError(
      "data.cmb.spectrum must be one of tt / te / ee / pp; got "
      + repr(cmb["spectrum"]))
  law = str(cmb["amplitude_law"])
  if law not in AMPLITUDE_LAWS:
    raise ValueError(
      "data.cmb.amplitude_law " + repr(law) + " is not in the registry "
      + repr(sorted(AMPLITUDE_LAWS)) + " (persisted by name, never a "
      "default)")
  need = AMPLITUDE_LAWS[law]
  for key in need:
    if key not in cmb:
      raise ValueError(
        "amplitude law " + repr(law) + " reads named parameter columns "
        "and needs data.cmb." + key + "; it is missing")
  if not need:
    extra = []
    for key in ("as_name", "tau_name"):
      if key in cmb:
        extra.append(key)
    if extra:
      raise ValueError(
        "amplitude law 'none' reads no parameter columns; drop "
        + repr(extra) + " from data.cmb")
  # exclusivity: a CMB run has no cosmolike and is not a scalar run.
  forbidden = []
  for key in ("cosmolike_data_dir", "cosmolike_dataset", "outputs"):
    if key in data:
      forbidden.append(key)
  if forbidden:
    raise ValueError(
      "a CMB run (data.cmb present) must not carry " + repr(sorted(
      forbidden)) + "; the CMB path reads dv dumps + the covariance "
      ".npz, no cosmolike and no scalar outputs")
  # the CMB run DOES use dv dumps (unlike scalar): all five files.
  for key in ("train_dv", "val_dv", "train_params", "val_params",
              "train_covmat"):
    if key not in data:
      raise ValueError(
        "a CMB run needs data." + key + " (the C_ell dumps ride the "
        "same dv/params staging as the cosmolike path); it is missing")
  # the data-vector-only features do not compose with a CMB run (V1).
  if rescale != "none":
    raise ValueError(
      "--rescale " + repr(rescale) + " is a cosmolike data-vector "
      "concept; a CMB run imposes its amplitude through "
      "data.cmb.amplitude_law instead (drop --rescale / leave it none)")
  ia = train_args.get("model", {}).get("ia")
  if ia not in (None, "none"):
    raise ValueError(
      "train_args.model.ia " + repr(ia) + " is an intrinsic-alignment "
      "(cosmic-shear) design; a CMB run has no ia (remove it)")
  # NPCE rides the CMB family (the 2026-07-12 family-wide ruling), but
  # only under amplitude_law "none": the as_exp2tau law's loss owns the
  # target construction (CmbFactoredChi2), the same one-at-a-time
  # exclusivity as pce vs rescale / model.ia. validate_pce vets the
  # block itself (with diagonal=True) on the from_config cmb branch.
  if (cfg.get("pce") is not None
      and str(cmb.get("amplitude_law")) != "none"):
    raise ValueError(
      "the pce: block and data.cmb.amplitude_law "
      + repr(cmb.get("amplitude_law")) + " are exclusive: each replaces "
      "the target construction (pce fits a closed-form base; the law "
      "rescales the target per row). Use one at a time (the NPCE base "
      "needs amplitude_law: none)")
  # transfer learning IS in scope since the 2026-07-12 symmetry ruling
  # (an earlier deferral, closed: "it is weird to have a feature not
  # symmetric to all cases") — but only under amplitude_law "none":
  # the imposed law and the transfer target construction each own the
  # target, one at a time (the same exclusivity as pce). The block
  # itself is vetted by validate_transfer (diagonal=True) on the
  # from_config cmb branch.
  if (cfg.get("transfer") is not None
      and str(cmb.get("amplitude_law")) != "none"):
    raise ValueError(
      "the transfer: block and data.cmb.amplitude_law "
      + repr(cmb.get("amplitude_law")) + " are exclusive: each replaces "
      "the target construction (the transfer loss composes a frozen "
      "base; the law rescales the target per row). Use one at a time "
      "(a transfer run needs amplitude_law: none)")
  # fine-tuning IS in scope: the finetune block is validated by
  # warmstart.validate_finetune_config on the from_config cmb branch
  # and the source geometry is pinned in build_geometry (the same
  # source-geometry pin as the cosmolike warm start).
  return dict(cmb)


def validate_grid(cfg, train_args, rescale="none"):
  """
  Validate a grid (background-function) run and return its data.grid block.

  A grid run is signalled by the data.grid sub-block: the data vector is
  a function of redshift on a stored grid — H(z) on the SN range or the
  comoving distance D_M(z) on the recombination window (the BSN
  two-regime design) — standardized through a GridGeometry
  whose target law (e.g. log(H + offset)) is persisted in the artifact.
  This enforces the exclusivity and the forbidden features, then returns
  the validated block. A standalone pure function (no torch).

  Arguments:
    cfg        = the parsed config mapping (reads cfg["data"],
                 cfg["pce"], cfg["transfer"]).
    train_args = the resolved train_args (reads model.ia).
    rescale    = the driver's --rescale value (a grid run forbids any
                 analytic rescaling; default "none").

  Returns:
    the data.grid mapping (quantity / units / law validated; offset
    present exactly when the law needs it).

  Raises:
    ValueError on: a missing or unknown sub-key; a quantity outside
    Hubble / D_M; a law outside the TARGET_LAWS registry; an offset
    missing (log_offset) or present (none); a cosmolike / scalar / cmb
    key beside data.grid; a missing dv/params/covmat file key; a
    data-vector-only feature (rescale, model.ia). The pce: and
    transfer: blocks are legal (the 2026-07-12 family-wide rulings —
    the transfer forbid was overturned by the user); validate_pce /
    validate_transfer vet them with diagonal=True on the from_config
    grid branch.
  """
  # the target-law registry lives with the grid geometry; imported here
  # (not at module top) so this validator stays importable in the same
  # torch-light contexts as the rest of the config logic.
  from .geometries.grid import TARGET_LAWS

  data = cfg["data"]
  grid = data["grid"]
  if not isinstance(grid, dict):
    raise ValueError(
      "data.grid must be a mapping {quantity, units, law[, offset], "
      "z_file}; got " + repr(type(grid).__name__))
  allowed = {"quantity", "units", "law", "offset", "z_file"}
  unknown = sorted(set(grid) - allowed)
  if unknown:
    raise ValueError(
      "unknown data.grid key(s) " + repr(unknown) + "; allowed: "
      + repr(sorted(allowed)))
  for key in ("quantity", "units", "law", "z_file"):
    if key not in grid:
      raise ValueError(
        "data.grid needs the " + repr(key) + " key (quantity = which "
        "background function the rows hold; units = its units string; "
        "law = the TARGET_LAWS name; z_file = the generator's _z.npy "
        "grid sidecar); it is missing")
  quantity = str(grid["quantity"])
  if quantity not in ("Hubble", "D_M"):
    raise ValueError(
      "data.grid.quantity must be 'Hubble' (the SN-range H(z) emulator) "
      "or 'D_M' (the recombination-window comoving distance); got "
      + repr(grid["quantity"]))
  law = str(grid["law"])
  if law not in TARGET_LAWS:
    raise ValueError(
      "data.grid.law " + repr(law) + " is not in the registry "
      + repr(sorted(TARGET_LAWS)) + " (persisted by name, never a "
      "default)")
  if law == "log_offset" and "offset" not in grid:
    raise ValueError(
      "the log_offset law needs data.grid.offset (the additive constant "
      "in log(target + offset), the legacy emulbaosn convention — state "
      "it explicitly, never a default); it is missing")
  if law == "none" and "offset" in grid:
    raise ValueError(
      "law 'none' has no offset; drop data.grid.offset")
  # exclusivity: a grid run has no cosmolike, is not scalar, not CMB.
  forbidden = []
  for key in ("cosmolike_data_dir", "cosmolike_dataset", "outputs", "cmb"):
    if key in data:
      forbidden.append(key)
  if forbidden:
    raise ValueError(
      "a grid run (data.grid present) must not carry " + repr(sorted(
      forbidden)) + "; the grid path reads dv dumps + the grid sidecar, "
      "no cosmolike, no scalar outputs, no cmb block")
  # the grid run DOES use dv dumps: all five files.
  for key in ("train_dv", "val_dv", "train_params", "val_params",
              "train_covmat"):
    if key not in data:
      raise ValueError(
        "a grid run needs data." + key + " (the background dumps ride "
        "the same dv/params staging as the cosmolike path); it is "
        "missing")
  if rescale != "none":
    raise ValueError(
      "--rescale " + repr(rescale) + " is a cosmolike data-vector "
      "concept; a grid run imposes its target transform through "
      "data.grid.law instead (drop --rescale / leave it none)")
  ia = train_args.get("model", {}).get("ia")
  if ia not in (None, "none"):
    raise ValueError(
      "train_args.model.ia " + repr(ia) + " is an intrinsic-alignment "
      "(cosmic-shear) design; a grid run has no ia (remove it)")
  # transfer learning IS in scope since the 2026-07-12 symmetry ruling
  # (an earlier permanent forbid overturned by the user: "I misspoke -
  # ... it is easy to allow it to BAO/SN"); validate_transfer
  # (diagonal=True) vets the block on the from_config grid branch, and
  # build_geometry pins the base's grid/quantity/law loudly.
  # fine-tuning IS in scope: validated on the from_config grid
  # branch; the source geometry is pinned in build_geometry.
  return dict(grid)


def validate_grid2d(cfg, train_args, rescale="none"):
  """
  Validate a grid2d (matter-power-spectrum) run; return its data.grid2d.

  A grid2d run is signalled by the data.grid2d sub-block: the data
  vector is one MPS quantity's flattened (z, k) surface — the linear
  P(k, z) ("pklin") or the nonlinear boost ("boost") — standardized in
  LAW SPACE. Under a syren law the target is log(quantity / base) where
  the base is the analytic syren formula the emulator corrects; the
  base rows come from the generator's *_base dump files
  (train_base / val_base), never recomputed at training. A
  standalone pure function (no torch).

  Arguments:
    cfg        = the parsed config mapping (reads cfg["data"],
                 cfg["pce"], cfg["transfer"]).
    train_args = the resolved train_args (reads model.ia).
    rescale    = the driver's --rescale value (forbidden off "none").

  Returns:
    the data.grid2d mapping, validated.

  Raises:
    ValueError on: a missing or unknown sub-key; a quantity outside
    pklin / boost; a units string not matching the quantity (pklin =
    "Mpc3", boost = "dimensionless"); a law outside TARGET_LAWS_2D or
    paired with the wrong quantity (syren_linear corrects pklin,
    syren_halofit corrects boost); base files missing under a syren
    law or present under "none"; a bad k_stride; another family's key
    beside data.grid2d; a missing dv/params/covmat file key; rescale /
    model.ia. The pce: and transfer: blocks are legal (the 2026-07-12
    family-wide rulings — the transfer forbid was overturned by the
    user, "this for sure should be allowed for MPS"); validate_pce /
    validate_transfer vet them with diagonal=True on the from_config
    grid2d branch.
  """
  from .geometries.grid2d import TARGET_LAWS_2D

  data = cfg["data"]
  g2 = data["grid2d"]
  if not isinstance(g2, dict):
    raise ValueError(
      "data.grid2d must be a mapping {quantity, units, law, z_file, "
      "k_file[, k_stride, train_base, val_base]}; got "
      + repr(type(g2).__name__))
  allowed = {"quantity", "units", "law", "z_file", "k_file",
             "k_stride", "train_base", "val_base"}
  unknown = sorted(set(g2) - allowed)
  if unknown:
    raise ValueError(
      "unknown data.grid2d key(s) " + repr(unknown) + "; allowed: "
      + repr(sorted(allowed)))
  for key in ("quantity", "units", "law", "z_file", "k_file"):
    if key not in g2:
      raise ValueError(
        "data.grid2d needs the " + repr(key) + " key (quantity = which "
        "MPS surface the rows hold; units = its units string; law = "
        "the TARGET_LAWS_2D name; z_file / k_file = the generator's "
        "grid sidecars); it is missing")
  quantity = str(g2["quantity"])
  if quantity not in ("pklin", "boost"):
    raise ValueError(
      "data.grid2d.quantity must be 'pklin' (the linear P(k, z)) or "
      "'boost' (P_nl/P_lin); got " + repr(g2["quantity"]))
  want_units = "Mpc3" if quantity == "pklin" else "dimensionless"
  if str(g2["units"]) != want_units:
    raise ValueError(
      "data.grid2d.units must be " + repr(want_units) + " for quantity "
      + repr(quantity) + " (the generator's convention); got "
      + repr(g2["units"]))
  law = str(g2["law"])
  if law not in TARGET_LAWS_2D:
    raise ValueError(
      "data.grid2d.law " + repr(law) + " is not in the registry "
      + repr(sorted(TARGET_LAWS_2D)) + " (persisted by name, never a "
      "default)")
  law_of = {"pklin": "syren_linear", "boost": "syren_halofit"}
  if law != "none" and law != law_of[quantity]:
    raise ValueError(
      "data.grid2d.law " + repr(law) + " does not correct quantity "
      + repr(quantity) + "; the syren pairing is pklin <- syren_linear "
      "and boost <- syren_halofit (or 'none' for either)")
  if law != "none":
    for key in ("train_base", "val_base"):
      if key not in g2:
        raise ValueError(
          "a syren law needs data.grid2d." + key + " (the generator's "
          "*_base dump beside the raw one — the analytic base the "
          "emulator corrects, read from disk); it is "
          "missing")
  else:
    extra = []
    for key in ("train_base", "val_base"):
      if key in g2:
        extra.append(key)
    if extra:
      raise ValueError(
        "law 'none' reads no base files; drop " + repr(extra)
        + " from data.grid2d")
  if "k_stride" in g2:
    ks = g2["k_stride"]
    if isinstance(ks, bool) or not isinstance(ks, int) or ks < 1:
      raise ValueError(
        "data.grid2d.k_stride must be an integer >= 1 (keep every "
        "k_stride-th wavenumber, the top edge always kept), got "
        + repr(ks))
  # exclusivity: a grid2d run has no cosmolike and is no other family.
  forbidden = []
  for key in ("cosmolike_data_dir", "cosmolike_dataset", "outputs",
              "cmb", "grid"):
    if key in data:
      forbidden.append(key)
  if forbidden:
    raise ValueError(
      "a grid2d run (data.grid2d present) must not carry " + repr(sorted(
      forbidden)) + "; the grid2d path reads dv dumps + the grid "
      "sidecars, no cosmolike and no other family block")
  for key in ("train_dv", "val_dv", "train_params", "val_params",
              "train_covmat"):
    if key not in data:
      raise ValueError(
        "a grid2d run needs data." + key + " (the MPS dumps ride the "
        "same dv/params staging as the cosmolike path); it is missing")
  if rescale != "none":
    raise ValueError(
      "--rescale " + repr(rescale) + " is a cosmolike data-vector "
      "concept; a grid2d run imposes its target transform through "
      "data.grid2d.law instead (drop --rescale / leave it none)")
  ia = train_args.get("model", {}).get("ia")
  if ia not in (None, "none"):
    raise ValueError(
      "train_args.model.ia " + repr(ia) + " is an intrinsic-alignment "
      "(cosmic-shear) design; a grid2d run has no ia (remove it)")
  # transfer learning IS in scope since the 2026-07-12 symmetry ruling
  # (an earlier permanent forbid overturned by the user: "this for sure
  # should be allowed for MPS"); validate_transfer (diagonal=True)
  # vets the block on the from_config grid2d branch, and build_geometry
  # pins the base's (z, k) axes / quantity / law loudly.
  # fine-tuning IS in scope: validated on the from_config
  # grid2d branch; the source geometry is pinned in build_geometry.
  return dict(g2)


# the train_divisor / val_divisor -> n_train / n_val migration message,
# printed verbatim by validate_sizes. No automatic value conversion: a
# divisor kept a fraction of the pre-cut dump, an absolute count guarantees
# rows after the cuts, so the placeholder numbers below are the user's to
# set. Follows the paste-ready-YAML rule (a block-context snippet with
# example values, not a prose key list).
_SIZES_MIGRATION_MESSAGE = (
  "train_divisor / val_divisor are gone: run sizes are now the absolute "
  "row counts data.n_train / data.n_val, enforced after param_cuts, not a "
  "fraction of the pre-cut dump. The semantics changed (a divisor kept a "
  "fraction of the whole dump; a count guarantees that many rows survive "
  "the cuts), so there is no automatic conversion; choose the counts you "
  "want. Replace the two flat keys under data: with (example values, set "
  "your own):\n\n"
  "  n_train: 25000     # absolute training rows kept after param_cuts\n"
  "  n_val:   5000      # absolute validation rows, same rule")

# placeholder counts named in the missing-key error (the same scale the
# example YAMLs ship), so the message shows a concrete key: value to add.
_SIZES_PLACEHOLDER = {"n_train": 25000, "n_val": 5000}


def validate_sizes(data):
  """
  Validate the absolute run sizes data.n_train / data.n_val.

  A standalone pure function (no torch), so it is unit-testable in
  isolation. Raises loudly on the migration from the old fractional
  train_divisor / val_divisor, and on a missing / non-integer /
  non-positive count; otherwise returns the validated pair. The counts
  are enforced after param_cuts by load_source's physical-pool check, not
  here (this only checks the values are well-formed).

  Arguments:
    data = the parsed "data" block mapping.

  Returns:
    the validated (n_train, n_val) integer pair (absolute row counts).

  Raises:
    ValueError on: either legacy divisor key still present (a migration
    error naming both new keys, with the semantics-changed warning and
    placeholder example values); a missing n_train / n_val; a
    non-integer (bool rejected) or < 1 count.
  """
  # the old fractional layout -> a migration error. No automatic
  # conversion: a divisor kept a fraction of the pre-cut dump, a count
  # guarantees rows after the cuts, so the value is the user's choice.
  if "train_divisor" in data or "val_divisor" in data:
    raise ValueError(_SIZES_MIGRATION_MESSAGE)
  sizes = []
  for key in ("n_train", "n_val"):
    if key not in data:
      raise ValueError(
        f"data is missing the required '{key}' (absolute rows kept after "
        f"param_cuts, a positive int); add e.g. "
        f"{key}: {_SIZES_PLACEHOLDER[key]}")
    val = data[key]
    # bool is an int subclass, but True / False is never a row count.
    if isinstance(val, bool) or not isinstance(val, int):
      raise ValueError(
        f"data.{key} must be a positive int (absolute rows kept after "
        f"param_cuts), got {val!r}")
    if val < 1:
      raise ValueError(
        f"data.{key} must be >= 1 (absolute rows kept after param_cuts), "
        f"got {val}")
    sizes.append(val)
  return sizes[0], sizes[1]


# the pce: block schema (the eight keys). form has no default (required
# when the block is present); the rest default to PCEEmulator.from_training's.
PCE_KEYS = {
  "form", "p_max", "r_max", "q", "k_max", "loo_max", "max_terms", "max_fail",
}
_PCE_DEFAULTS = {"p_max": 4,
                 "r_max": 2,
                 "q": 0.5,
                 "k_max": 40,
                 "loo_max": 0.05,
                 "max_terms": 30,
                 "max_fail": 4}
_PCE_INT_KEYS = ("p_max", "r_max", "k_max", "max_terms", "max_fail")


def validate_pce(pce, rescale="none", ia=None, diagonal=False):
  """
  Validate the top-level pce: block (the NPCE closed-form base).

  A standalone pure function (no torch), so it is unit-testable in
  isolation. The pce: block is a sibling of data / train_args, never
  inside train_args: sweep_hyperparam deep-copies train_args per point
  but builds the geometry (hence the base) once, so a pce knob under
  train_args would sweep without refitting the base (a silent no-op, the
  trap validate_sweep_paths exists to kill). Top-level makes it
  structurally unsweepable, one study per pce config (the model.name rule).

  The base and the two exclusive alternatives each replace the chi2fn, so
  pce is rejected alongside --rescale or a model.ia design (one at a time).

  Arguments:
    pce      = the parsed top-level "pce" block, or None (the block absent).
    rescale  = the analytic-R rescale mode (a driver flag), for the
               exclusivity check; pce is exclusive with rescale != "none".
    ia       = the resolved model.ia design (None | "nla" | "tatt"), for
               the exclusivity check; pce is exclusive with an ia design.
    diagonal = True on the elementwise-whitened families (cmb / grid /
               grid2d / scalar), where only form "residual" exists: the
               ratio form is a dense-covariance concept (a fractional
               correction where whitening mixes elements); with a
               per-element whitening the residual form already gives the
               refiner per-element leverage, and on the log-law grids a
               whitened residual IS a multiplicative correction in
               linear space.

  Returns:
    None when pce is None (NPCE off, byte-identical everywhere), else the
    validated, defaults-filled mapping (form + the seven fit knobs).

  Raises:
    TypeError if pce is not a mapping.
    ValueError on: an unknown key; a missing / non-{residual, ratio} form;
    form "ratio" on a diagonal family; a non-positive-int p_max / r_max /
    k_max / max_terms / max_fail; q outside (0, 1]; loo_max <= 0; or pce
    set together with rescale != "none" or a model.ia design.
  """
  if pce is None:
    return None
  if not isinstance(pce, dict):
    raise TypeError(
      f"the pce: block must be a mapping of fit knobs, got "
      f"{type(pce).__name__}")
  unknown = set(pce) - PCE_KEYS
  if unknown:
    raise ValueError(
      f"unknown pce: key(s): {sorted(unknown)}; allowed: "
      f"{sorted(PCE_KEYS)}")
  # form is required and picks the loss shape.
  form = pce.get("form")
  if form not in ("residual", "ratio"):
    raise ValueError(
      "pce.form is required and must be 'residual' (base + net) or "
      f"'ratio' (base * (1 + net)), got {form!r}")
  if diagonal and form == "ratio":
    raise ValueError(
      "pce.form 'ratio' exists only on the cosmolike (dense-covariance) "
      "family: with an elementwise whitening the residual form already "
      "gives the refiner per-element leverage (and on a log-law grid a "
      "whitened residual IS a multiplicative correction in linear space); "
      "use form: residual")
  out = dict(_PCE_DEFAULTS)
  out.update({k: v for k, v in pce.items() if k != "form"})
  out["form"] = form
  # positive-int knobs (bool is an int subclass, but never a count).
  for k in _PCE_INT_KEYS:
    v = out[k]
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
      raise ValueError(f"pce.{k} must be a positive int (>= 1), got {v!r}")
  # q in (0, 1]; loo_max > 0.
  q = out["q"]
  if isinstance(q, bool) or not isinstance(q, (int, float)) \
     or not (0.0 < q <= 1.0):
    raise ValueError(
      f"pce.q must be in (0, 1] (the hyperbolic sparsity exponent), got {q!r}")
  lm = out["loo_max"]
  if isinstance(lm, bool) or not isinstance(lm, (int, float)) or lm <= 0:
    raise ValueError(
      f"pce.loo_max must be > 0 (the relative leave-one-out cutoff), "
      f"got {lm!r}")
  # exclusivity: pce and rescale / ia each replace the chi2fn.
  if rescale != "none":
    raise ValueError(
      "the pce: block and --rescale are exclusive: each replaces the chi2 "
      f"loss (pce fits a closed-form base; rescale={rescale!r} applies "
      "analytic R). Use one at a time.")
  if ia is not None:
    raise ValueError(
      "the pce: block and model.ia are exclusive: each replaces the chi2 "
      f"loss (pce fits a closed-form base; ia={ia!r} combines templates in "
      "closed form). Use one at a time.")
  return out


# the top-level transfer: block keys. "refine" is the optional joint-refinement
# stage: unfreeze the base for a second training stage.
TRANSFER_KEYS = ("from", "form", "space", "refine")
# the refine: sub-block keys. anchor is REQUIRED when refine: is present (an
# explicit 0.0 states free fine-tuning deliberately, never a silent default).
_REFINE_KEYS = ("epochs", "base_lr_scale", "anchor")


def validate_transfer(cfg, train_args, rescale="none", diagonal=False):
  """
  Validate the top-level transfer: block (the frozen-base parallel correction).

  A standalone pure function (no torch), so it is unit-testable in isolation.
  Like pce:, transfer: is a sibling of data / train_args (one base per study).
  Resolves the space to the form's recommended default when omitted, and
  materializes it (persist-resolved-values), returning (resolved, notice):
  the resolved block plus a one-line off-recommendation notice or None.

  Arguments:
    cfg        = the full parsed config mapping (its top-level pce: block and
                 the transfer: block are read here).
    train_args = the resolved train_args mapping (checked for the exclusive
                 finetune / two-phase keys and the inherited model.ia).
    rescale    = the analytic-R rescale mode (a driver flag); transfer is
                 exclusive with rescale != "none".
    diagonal   = True on the elementwise-whitened families (cmb / grid /
                 grid2d — the 2026-07-12 symmetry ruling admits them).
                 There the composition space is "whitened" ONLY (it is
                 those families' chi2 metric basis; a physical composition
                 is an elementwise scale away, or crosses a log-law domain
                 edge), space defaults to it for BOTH forms, an explicit
                 "physical" raises, form "gain" carries the zero-crossing
                 notice (sum is the recommendation), and transfer.refine
                 is rejected (frozen-base V1 on these families).

  Returns:
    (None, None) when transfer is absent; else (resolved, notice) where
    resolved is {from, form, space} with space materialized, and notice is the
    off-recommendation trade-off line (or None on the recommended pairing).

  Raises:
    TypeError / ValueError / KeyError naming any violated rule. A refine:
    key is validated and materialized on the cosmolike family and
    rejected on a diagonal family (frozen-base V1).
  """
  transfer = cfg.get("transfer")
  if transfer is None:
    return None, None
  if not isinstance(transfer, dict):
    raise TypeError(
      "the transfer: block must be a mapping (from / form / space), got "
      + type(transfer).__name__)
  unknown = set(transfer) - set(TRANSFER_KEYS)
  if unknown:
    raise KeyError(
      "unknown transfer: key(s): " + str(sorted(unknown)) + "; allowed: "
      + str(list(TRANSFER_KEYS)))
  if not transfer.get("from"):
    raise KeyError(
      "transfer.from is required: the frozen base artifact path root "
      "(<root>.h5 + <root>.emul, as written by save_emulator)")
  form = transfer.get("form")
  if form not in FORMS:
    raise ValueError(
      "transfer.form is required and must be one of " + str(list(FORMS))
      + " (gain = base * (1 + r); sum = base + r), got " + repr(form))
  space = transfer.get("space")
  if space is not None and space not in SPACES:
    raise ValueError(
      "transfer.space must be one of " + str(list(SPACES)) + " (physical = "
      "squeezed bins; whitened = the eigenbasis), got " + repr(space))

  # exclusivities (all loud): each of these replaces the loss form or the
  # architecture the transfer inherits.
  if cfg.get("pce") is not None:
    raise ValueError(
      "transfer: is exclusive with a pce: block (each owns the loss); "
      "use one at a time")
  if rescale != "none":
    raise ValueError(
      "a transfer run requires --rescale none (the base's loss form is "
      "inherited); got --rescale " + repr(rescale))
  if train_args.get("finetune") is not None:
    raise ValueError(
      "transfer: is exclusive with train_args.finetune (a warm start and a "
      "frozen-base correction are different tools); use one at a time")
  model_blk = train_args.get("model", {})
  if isinstance(model_blk, dict) and model_blk.get("ia") is not None:
    raise ValueError(
      "a transfer run inherits its intrinsic-alignment family from the base "
      "artifact; remove model.ia (the base's family forces the correction's)")
  for key in ("trunk", "head"):
    if key in train_args:
      raise KeyError(
        "a transfer run is single-phase (V1): remove the train_args." + key
        + " block")
  if int(train_args.get("trunk_epochs", 0) or 0) > 0:
    raise ValueError(
      "a transfer run is single-phase (V1): set trunk_epochs 0 or remove it")
  if "freeze_trunk" in train_args:
    raise KeyError(
      "a transfer run is single-phase (V1): remove train_args.freeze_trunk")

  # resolve the space (materialized). Diagonal families: whitened only
  # (their metric basis), for both forms; an explicit physical is loud.
  # Cosmolike: absent -> the form's recommended pairing; an
  # off-recommendation pairing is allowed (the user decides) and carries
  # one quiet-gated trade-off notice.
  notice = None
  if diagonal:
    if space == "physical":
      raise ValueError(
        "transfer.space 'physical' does not exist on the elementwise-"
        "whitened families: the whitened space IS their chi2 metric "
        "basis, so a physical composition is an elementwise scale away "
        "(no new capability) or crosses a log-law domain edge (a NaN "
        "risk). Drop the key or set space: whitened")
    space = "whitened"
    if form == "gain":
      notice = ("notice: transfer gain on a diagonal family composes in "
                "the whitened space, whose coordinates cross zero element "
                "by element (no gain leverage at the crossings); form sum "
                "is the recommendation here")
  elif space is None:
    space = RECOMMENDED_SPACE[form]
  elif space != RECOMMENDED_SPACE[form]:
    if form == "gain":
      notice = ("notice: transfer gain/whitened is off-recommendation "
                "(whitened coordinates cross zero everywhere, so the "
                "near-zero degeneracy is generic); gain/physical is advised")
    else:
      notice = ("notice: transfer sum/physical is off-recommendation (the "
                "additive output spans the decades whitening exists to tame); "
                "sum/whitened is advised")
  resolved = {"from": transfer["from"],
              "form": form,
              "space": space}

  # the optional refine: stage — a second, joint training stage with
  # the base unfrozen. Validated + materialized here (persist-resolved-values).
  refine = transfer.get("refine")
  if refine is not None and diagonal:
    raise ValueError(
      "transfer.refine is not offered on the cmb / grid / grid2d families "
      "in V1 (frozen-base transfer only; the stage-2 drift/anchor plumbing "
      "is audited per family before it unlocks). Remove the refine: block")
  if refine is not None:
    if not isinstance(refine, dict):
      raise TypeError(
        "transfer.refine must be a mapping (epochs / base_lr_scale / anchor), "
        "got " + type(refine).__name__)
    unknown_r = set(refine) - set(_REFINE_KEYS)
    if unknown_r:
      raise KeyError(
        "unknown transfer.refine key(s): " + str(sorted(unknown_r))
        + "; allowed: " + str(list(_REFINE_KEYS)))
    epochs = refine.get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
      raise ValueError(
        "transfer.refine.epochs must be a positive integer (the stage-2 "
        "epoch count), got " + repr(epochs))
    scale = refine.get("base_lr_scale")
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
        or scale <= 0.0):
      raise ValueError(
        "transfer.refine.base_lr_scale must be a positive number (the base "
        "group's lr as a multiple of the run lr), got " + repr(scale))
    # anchor is REQUIRED (0.0 = free fine-tuning, stated deliberately); no
    # silent default for a physics-consequential knob.
    if "anchor" not in refine:
      raise KeyError(
        "transfer.refine.anchor is required when refine: is present: the "
        "L2-SP strength lambda pulling the base back toward its pretrained "
        "weights. An explicit 0.0 states free fine-tuning deliberately")
    anchor = refine["anchor"]
    if isinstance(anchor, bool) or not isinstance(anchor, (int, float)) \
        or anchor < 0.0:
      raise ValueError(
        "transfer.refine.anchor must be a number >= 0 (the L2-SP strength; "
        "0.0 = free fine-tuning), got " + repr(anchor))
    resolved["refine"] = {"epochs": int(epochs),
                          "base_lr_scale": float(scale),
                          "anchor": float(anchor)}
  return resolved, notice


def _load_diag_transfer(cfg, train_args, kwargs, geom_cls_name):
  """Validate + load a diagonal-family transfer base (a from_config helper).

  Shared by the cmb / grid / grid2d from_config branches (the 2026-07-12
  symmetry ruling): runs validate_transfer with diagonal=True, resolves
  the device (load_source rebuilds the base on it, so __init__ must get
  the same one — kwargs is updated in place), loads the base artifact,
  and checks its KIND — the base's output geometry class must be the
  run's family's (a cosmolike or wrong-family base is a loud error here,
  before any staging). The deep pins (grids / quantity / law equality)
  happen in build_geometry, where the run's own grids exist.

  Arguments:
    cfg           = the full parsed config mapping.
    train_args    = the resolved train_args (validate_transfer reads the
                    exclusive finetune / two-phase keys).
    kwargs        = the from_config **kwargs mapping; its "device" entry
                    is resolved and written back.
    geom_cls_name = the family's output-geometry class name
                    ("CmbDiagonalGeometry" / "GridGeometry" /
                    "Grid2DGeometry"), for the wrong-kind check.

  Returns:
    (resolved, notice, base, base_root), or (None, None, None, None)
    when no transfer: block is present.

  Raises:
    everything validate_transfer raises; ValueError on a base whose
    geometry class is not the family's.
  """
  if cfg.get("transfer") is None:
    return None, None, None, None
  resolved, notice = validate_transfer(
    cfg=cfg, train_args=train_args,
    rescale=kwargs.get("rescale", "none"), diagonal=True)
  device = kwargs.get("device")
  if device is None:
    device = pick_device()
  kwargs["device"] = device
  base_root = warmstart.resolve_source_root(cfg["transfer"])
  base = warmstart.load_source(root=base_root, device=device)
  got_cls = type(base.geom).__name__
  if got_cls != geom_cls_name:
    raise ValueError(
      "transfer.from points at a " + got_cls + " artifact but this run's "
      "family needs a " + geom_cls_name + " base; a transfer never "
      "crosses families (pick a base trained by this family's driver)")
  return resolved, notice, base, base_root


def resolve_phase_args(train_args, two_phase):
  """
  Resolve the two-phase schedule keys against the model's real capability.

  A standalone pure function (no torch), so it is unit-testable in
  isolation, and it never mutates its input (a hyperparameter sweep reuses
  one train_args across points). One shared YAML can then carry the
  two-phase keys (trunk_epochs, freeze_trunk, the symmetric trunk: / head:
  blocks) and still drive a single-phase model: for such a model head:,
  trunk_epochs, and freeze_trunk are dropped and trunk: is merged into the
  top level (the trunk becomes
  the global objective, the user's rule), with a notice naming what
  happened. For a two-phase model it is an exact no-op.

  Arguments:
    train_args = the resolved train_args mapping (range-free).
    two_phase  = whether the model supports the trunk-then-head schedule
                 (hasattr(model_cls, "set_train_phase")); the caller
                 probes the class, mirroring run_emulator's duck-typed
                 instance guard.

  Returns:
    (resolved, notice): resolved = train_args unchanged when two_phase or
    when no phase keys are present, else a copy with the phase keys
    demoted; notice = a one-line string naming the demotion (for the
    quiet-gated banner), or None when nothing changed.

  Raises:
    ValueError / TypeError from validate_phase_block on a malformed trunk:
    or head: block (a flat lr_base migration, an unknown key, a bs_base in
    the phase lr, a cls in the phase scheduler, or a non-mapping block) —
    the same errors the two-phase run_emulator path raises. Also a
    ValueError on a flat top-level loss_mode / berhu key (loss options now
    nest under a loss: block), printing the paste-ready block.
  """
  # loud no-alias migration: the flat top-level loss_mode / berhu keys are
  # gone, replaced by a nested loss: block (validate_loss). Reject them here,
  # before any early return or demotion, so single- and two-phase models
  # fail identically. run_emulator's validate_loss covers the loss block
  # itself; this covers a top-level flat key that would otherwise be
  # silently dropped (experiment.train reads only train_args["loss"]).
  if "loss_mode" in train_args or "berhu" in train_args:
    raise ValueError(_loss_migration_message(
      train_args.get("loss_mode"), train_args.get("berhu"), "train_args"))

  # a two-phase model, or a plain single-phase YAML: nothing to resolve.
  has_phase = ("trunk_epochs" in train_args or "trunk" in train_args
               or "head" in train_args or "freeze_trunk" in train_args)
  if two_phase or not has_phase:
    return train_args, None

  # validate both blocks first (validate_phase_block, training.py), so a
  # typo / flat lr_base / bs_base / cls fails identically here and on the
  # two-phase path. head: is dropped below, but a malformed head: block is
  # still a config error worth catching on a shared YAML.
  validate_phase_block(train_args.get("trunk"), "trunk")
  validate_phase_block(train_args.get("head"), "head")

  # single-phase model carrying two-phase keys: demote on a shallow copy,
  # so the caller's dict (a sweep's shared self.train_args) is untouched.
  had_trunk = "trunk" in train_args
  had_head  = "head" in train_args
  had_epochs = "trunk_epochs" in train_args
  had_freeze = "freeze_trunk" in train_args
  resolved = dict(train_args)
  trunk = resolved.pop("trunk", None)
  resolved.pop("head", None)
  resolved.pop("trunk_epochs", None)
  resolved.pop("freeze_trunk", None)

  merged = []
  if isinstance(trunk, dict):
    # prefix-strip: every trunk key merges to its same-named top-level key
    # (the blocks mirror the top level, so demotion is a plain move). lr
    # overlays the top-level lr block (bs_base and any key the phase does
    # not set are preserved); scheduler and the other five full-replace,
    # the same semantics they have as a phase override.
    for key in _PHASE_BLOCK_KEYS:
      if key not in trunk:
        continue
      if key == "lr":
        lr_blk = dict(resolved.get("lr", {}))   # copy: never touch input
        for k2 in trunk["lr"]:
          lr_blk[k2] = trunk["lr"][k2]
        resolved["lr"] = lr_blk
      else:
        resolved[key] = trunk[key]
      merged.append(key)

  # the notice names only what actually happened: the keys really merged,
  # and each block only when it was present.
  parts = []
  if merged:
    parts.append(
      f"trunk: merged into the top level ({', '.join(merged)})")
  elif had_trunk:
    parts.append("trunk: ignored (no keys)")
  dropped = []
  if had_head:
    dropped.append("head:")
  if had_epochs:
    dropped.append("trunk_epochs")
  if had_freeze:
    dropped.append("freeze_trunk")
  if dropped:
    parts.append(f"{' and '.join(dropped)} ignored")
  return resolved, "single-phase model: " + "; ".join(parts)


def validate_sweep_paths(paths, two_phase):
  """
  Reject phase-schedule sweep axes a single-phase model would silently drop.

  A standalone pure function (no torch), the one source of truth both sweep
  drivers call. On a two-phase model every path is valid (returns
  silently). On a single-phase model (the caller probes
  hasattr(model_cls, "set_train_phase")) resolve_phase_args demotes the
  phase keys before training, so a search over one of them is silently
  degraded: a head. or trunk_epochs axis makes every point identical (the
  axis is dropped), and a trunk. axis sweeps the merged top-level key in
  disguise. This turns that into a loud startup error naming every
  offending path and the concrete key to sweep instead.

  Arguments:
    paths     = iterable of dotted train_args paths a sweep / search will
                vary (the sweep parameter, or a study's range-leaf paths).
    two_phase = whether the model supports the trunk-then-head schedule
                (hasattr(model_cls, "set_train_phase")); True skips the
                whole check (all phase axes are valid there).

  Returns:
    None (a pure guard: it either raises or returns).

  Raises:
    ValueError listing every offending path: a head. / trunk_epochs axis
    (dropped by resolve_phase_args on a single-phase model), or a trunk.
    axis (demoted to a top-level key), each with the concrete fix.
  """
  if two_phase:
    return None
  problems = []
  for path in paths:
    segs = str(path).split(".")
    if segs[0] in ("head", "trunk_epochs", "freeze_trunk"):
      problems.append(
        f"{path!r}: a single-phase model (no set_train_phase) has "
        f"resolve_phase_args drop this axis, so every sweep point would be "
        f"identical; sweep a real top-level train_args leaf instead")
    elif segs[0] == "trunk":
      # trunk: mirrors the top level and merges up on a single-phase
      # model, so trunk.X.Y is just the top-level X.Y. Strip the prefix
      # (e.g. trunk.lr.lr_base -> lr.lr_base, trunk.scheduler.patience ->
      # scheduler.patience).
      concrete = ".".join(segs[1:]) or "the matching top-level key"
      problems.append(
        f"{path!r}: on a single-phase model trunk: merges into the top "
        f"level, so this sweeps {concrete!r} in disguise; sweep "
        f"{concrete!r} directly")
  if problems:
    raise ValueError(
      "phase-schedule sweep axis on a single-phase model (no "
      "set_train_phase):\n  " + "\n  ".join(problems))
  return None


class EmulatorExperiment:
  """
  Configuration + environment for one single-network cosmic-shear (xi)
  emulator, reusable across a single run and across sweeps.

  The constructor builds only the cheap, config-derived state (device,
  parameter names, the quiet-gated logger); the staged data and geometry
  come from explicit cached methods, so a sweep rebuilds only what varies.
  The fixed single-emulator choices (probe = xi, AdamW, ReduceLROnPlateau,
  use_amp = False, the report thresholds) are constructor defaults, so a
  driver passes only what it varies (the YAML, rescale, activation, quiet);
  the model is the config's choice (train_args.model.name).
  """

  def __init__(self,
               data,
               train_args,
               model_cls,
               opt_cls=optim.AdamW,
               sched_cls=lr_scheduler.ReduceLROnPlateau,
               probe="xi",
               thresholds=None,
               use_amp=False,
               rescale="none",
               activation="H",
               device=None,
               quiet=False,
               raw_train_args=None):
    """
    Store the config + fixed choices; build the cheap derived state.

    Arguments:
      data       = the "data" block: input paths plus cut / split /
                   cosmolike-dataset settings. Keys:
                     train_dv = training data-vector .npy (memmapped);
                     train_params = training parameter .txt (columns:
                       weight, lnp, modeled params, chi2);
                     train_covmat = parameter covmat (header line =
                       parameter names);
                     val_dv = validation data-vector .npy;
                     val_params = validation parameter .txt;
                     cosmolike_data_dir = folder under
                       external_modules/data;
                     cosmolike_dataset = .dataset ini naming the cov /
                       mask / data-vector files;
                     param_cuts = nested sub-block of physical density
                       windows (required; validated + flattened by
                       validate_param_cuts). Its keys:
                         omegabh2_hi = required upper bound on
                           omega_b h^2 (the former flat omegabh2_cut,
                           renamed);
                         omegabh2_lo = optional lower bound on
                           omega_b h^2;
                         omegam2h2_lo / omegam2h2_hi = optional window on
                           omegam^2 h^2 = (Omega_m H0/100)^2;
                         omegamh2_lo / omegamh2_hi = optional window on
                           omegamh2 = Omega_m (H0/100)^2 (Planck ~ 0.143);
                         omegamh2ns_lo / omegamh2ns_hi = optional window
                           on omegamh2 * n_s (Planck ~ 0.138; needs the
                           ns column).
                       Omit an optional key for no cut on that side; a
                       cut key left flat under data raises a migration
                       error printing the paste-ready block;
                     n_train / n_val = absolute training / validation rows
                       to keep (required positive ints, enforced after
                       param_cuts: the cut pool must supply them, else
                       load_source raises);
                     split_seed = seed for the cut+shuffle picking train /
                       val rows;
                     ram_frac = optional (default 0.7): RAM fraction the
                       staged subset may fill.
      train_args = resolved "train_args" block, range-free (search ranges
                   collapsed to scalars, e.g. by default_train_args).
                   Top keys:
                     nepochs = passes over the training set;
                     bs = minibatch size;
                     loss = optional nested block {mode, berhu} (absent or
                       no mode -> mode "sqrt"): mode the per-sample
                       transform "sqrt" / "chi2" / "sqrt_dchi2" / "berhu"
                       (reversed Huber: sqrt below the berhu knot, chi2-like
                       above, C1 there) / "berhu_capped" (berhu with the
                       tail vote plateauing above the cap, monster-robust);
                       berhu the {knot, cap} knot sub-block (defaults 0.2 /
                       10.0), valid only beside a berhu mode (a berhu
                       sub-block on a non-berhu mode raises, see
                       validate_loss), with an optional anneal: sub-block
                       {hold_epochs, anneal_epochs, shape} (presence = on)
                       ramping the loss from plain sqrt into the full berhu
                       shape (the escalated window votes arrive late). The
                       knot block may be spelled berhu: (the family name,
                       sweep-safe) or the exact active mode (berhu_capped:
                       under mode berhu_capped); a wrong-mode block raises.
                       Per-phase overridable (full replacement) and
                       sweepable (loss.mode / loss.berhu.knot /
                       loss.berhu.cap / loss.berhu.anneal.hold_epochs);
                     silent = optional (default False): silence the run;
                     trunk_epochs = optional (default 0): two-phase
                       schedule, see run_emulator;
                     freeze_trunk = optional (default true): phase-2 mode.
                       True freezes the trunk and trains the head alone;
                       false trains trunk + head together (a joint
                       fine-tune). Needs trunk_epochs > 0; sweepable on a
                       two-phase model; demoted (dropped) on a single-phase
                       one, see run_emulator;
                     trunk / head = optional symmetric mappings of
                       per-phase overrides (lr / scheduler / loss / trim /
                       focus / clip / rewind / ema, the eight-key phase
                       whitelist) over the shared top-level
                       defaults; need trunk_epochs > 0,
                       see run_emulator. Every design WITH a
                       correction head is two-phase capable (plain
                       rescnn / restrf on every family they ride,
                       and the factored-IA templates — the
                       2026-07-12 ruling: any trunk+head design may
                       train in two phases); on a single-phase model
                       (resmlp, incl. its ia variants — no
                       set_train_phase method) train()
                       demotes these through resolve_phase_args: head:,
                       trunk_epochs, and freeze_trunk are dropped and
                       trunk: is merged into the top level (with a
                       quiet-gated notice), so one shared YAML serves both
                       model families;
                     clip = optional (default 0.0 = off): per-step
                       gradient-norm ceiling, see run_emulator;
                     rewind = optional (default False): reload the
                       best weights + optimizer snapshot at every
                       plateau lr cut, see run_emulator;
                     ema = optional mapping {horizon_epochs, anneal}
                       (absent = off = a byte-identical run): keep a
                       Polyak weight average over horizon_epochs, coupled
                       to the best snapshot / rewind; selection + reported
                       metrics use the average, the scheduler the raw
                       median, and the shipped model is the best average.
                       An optional anneal: sub-block {hold_epochs,
                       anneal_epochs, shape} (the berhu-anneal twins) ramps
                       the horizon from 0, deferring the average past the
                       terrible early era; per-phase overridable (a phase
                       ema: full-replaces, ema: null disables it there) and
                       sweepable (ema.horizon_epochs /
                       ema.anneal.hold_epochs), see run_emulator;
                   Plus six constructible sub-blocks (each a mapping):
                     model = the nested model block: "name" (the
                       architecture: resmlp | rescnn | restrf) and
                       "ia" (the factored IA design layered on it:
                       omit for plain, "nla" or "tatt"; the pair
                       picks the class), then one sub-block per
                       component,
                       "mlp" (width, n_blocks; the trunk, required),
                       "activation" ({type, n_gates} or a bare type
                       string; see the `activation` argument below),
                       "cnn" (kernel_size, rescale_kernel, groups,
                       separable, film, n_blocks, gate_init,
                       activation; name rescnn only), "trf" (n_heads,
                       n_blocks, n_mlp_blocks, n_tokens, shared_mlp,
                       film, gate_init, activation; name restrf only,
                       the tokens live at the natural bin width, so
                       there is no width knob, and the per-token MLP
                       layers run at that width too, n_mlp_blocks is
                       depth only; n_tokens re-segments a single-bin
                       family geometry — cmb's ell / grid's z — into
                       that many attention windows, and is
                       rejected where physical bins exist). The
                       head's activation ({type,
                       n_gates} or a bare string) pins its own family
                       (absent = shares model.activation, the trunk's;
                       the head trains only in phase 2, so it needs a
                       frozen-trunk head phase: trunk_epochs > 0 and
                       freeze_trunk true). head: activation: is the
                       head-only alias; trunk: activation: is an error.
                       Plus an optional flat "compile_mode".
                       build_specs translates the nesting onto the
                       constructors' flat kwargs (MODEL_BLOCK_KEYS)
                       and rejects unknown or misplaced keys;
                     optimizer = weight_decay (+ any extra AdamW kwargs);
                     lr = lr_base, bs_base, warmup_epochs (run sets
                       lr = lr_base * sqrt(bs / bs_base));
                     scheduler = mode, patience, factor (ReduceLROnPlateau
                       kwargs);
                     trim = trim schedule, start, end, hold_epochs,
                       anneal_epochs, shape (see anneal_value);
                     focus = focal-weight schedule, start, end,
                       hold_epochs, anneal_epochs, shape, kappa.
      model_cls  = the model class (ResMLP / ResCNN / ResTRF); from_config
                   resolves it from train_args.model.name.
      opt_cls    = optimizer class (default AdamW).
      sched_cls  = scheduler class (default ReduceLROnPlateau).
      probe      = cosmolike probe (default "xi").
      thresholds = reported delta-chi2 cutoffs (default
                   DEFAULT_THRESHOLDS); thresholds[0] selects the best model.
      use_amp    = run the forward in low-precision autocast (default False).
      rescale    = analytic-R mode forwarded to make_chi2 ("none" /
                   "rescaled" / "residual").
      activation = ResBlock activation name (make_activation): "H" /
                   "power" / "multigate" / "gated_power". from_config
                   resolves the precedence: an explicit value here (the
                   drivers' --activation) wins over the YAML's
                   train_args.model.activation, which wins over "H".
      device     = compute device (default: pick_device()).
      quiet      = if True, silence the instance logger and the
                   per-source / per-epoch prints.
      raw_train_args = un-collapsed train_args (search ranges intact), for
                   a search driver that resolves them per trial; defaults
                   to train_args (from_config supplies the raw block).
    """
    self.data       = data
    self.train_args = train_args
    self.model_cls  = model_cls
    # display name + IA design + architecture, overwritten by
    # from_config with the YAML's composed name/ia. The
    # direct-construction fallbacks: a factored class defaults to
    # "nla" (a direct-construction tatt run must set exp.ia = "tatt"
    # itself, from_config does this from the YAML); arch stays
    # None, which skips build_specs' head-block-vs-architecture check.
    self.model_name = model_cls.__name__.lower()
    self.ia = ("nla" if getattr(model_cls, "factored", False)
               else None)
    self.arch = None
    self.opt_cls    = opt_cls
    self.sched_cls  = sched_cls
    self.probe      = probe
    self.thresholds = (DEFAULT_THRESHOLDS if thresholds is None
                       else thresholds)
    self.use_amp    = use_amp
    self.rescale    = rescale
    self.activation = activation
    self.quiet      = quiet

    # make_logger / pick_device (training.py): a print(*a) gated on quiet,
    # and the compute device (cuda > mps > cpu).
    self.log        = make_logger(quiet=quiet)
    self.device     = pick_device() if device is None else device

    # un-collapsed train_args (search ranges intact) for a per-trial search
    # driver; defaults to the resolved train_args when no raw block given.
    self.raw_train_args = (train_args if raw_train_args is None
                           else raw_train_args)

    # TF32 tensor-core float32 matmuls (Ampere+); no-op on CPU / MPS. A
    # one-time global switch.
    torch.set_float32_matmul_precision("high")

    # read_param_names (data_staging.py): parameter column names from the
    # covmat's "#"-prefixed header line. Reused for the val cut (same
    # columns).
    self.names = read_param_names(data["train_covmat"])

    # artifacts the methods below build; cached across a sweep (None until
    # built).
    self.train_set = None
    self.val_set   = None
    self.pgeom     = None
    self.geom      = None
    self.chi2fn    = None
    self.model     = None
    # the ruling-(a) flag-vs-pin warning, set by from_config (which alone
    # knows whether --activation was explicit); None on direct __init__.
    self._activation_notice = None
    # the validated top-level pce: block (None = NPCE off), set by
    # from_config; every wiring point guards on it (absent = byte-identical).
    self.pce_opts = None
    # the consumed-view save recipes (save schema v2): resolved_model is
    # assembled in build_specs, resolved_train is returned by run_emulator
    # and stored by train(). None until those run.
    self.resolved_model = None
    self.resolved_train = None
    # scalar (derived-parameter) emulator: the emulated output names
    # and the run-mode flag, set by from_config. None / False on a data-vector
    # run, which every scalar branch below guards on (absent = byte-identical).
    self._scalar = False
    self.outputs = None
    # the CMB-run flag + its validated data.cmb block, set by from_config.
    # None / False on every other run, which the cmb branches guard on.
    self._cmb = False
    self.cmb  = None
    self._grid = False
    self.grid  = None
    self._grid2d = False
    self.grid2d  = None
    # the post-staging (z, k) axes a grid2d run trains on (set by the
    # staging law-transform hook; build_geometry reads them).
    self._grid2d_z = None
    self._grid2d_k = None
    # fine-tune warm start (emulator/warmstart.py): the loaded source bundle,
    # its resolved absolute path root, and the extra parameter names, all set
    # by from_config / build_geometry. None on an ordinary run, which every
    # finetune branch below guards on (absent = byte-identical).
    self._finetune = None
    self._finetune_root = None
    self._finetune_extra_names = None
    # transfer learning (emulator/losses/transfer.py): the loaded frozen base,
    # its resolved path root, the resolved form / space, and the extra
    # parameter names, set by from_config / build_geometry. None on an ordinary
    # run, which every transfer branch below guards on (absent = byte-identical).
    self._transfer_base = None
    self._transfer_refine = None
    self._transfer_pretrained_base = None
    self._transfer_root = None
    self._transfer_form = None
    self._transfer_space = None
    self._transfer_extra_names = None

  # --- alternative constructors ---
  @classmethod
  def from_config(cls, cfg, models=None, **kwargs):
    """
    Build from an already-parsed config mapping.

    Validates the required blocks, collapses train_args search ranges to
    their defaults, and resolves the (train_args.model.name,
    train_args.model.ia) pair -> a model class through `models`. Use it
    to rebuild from a tweaked copy of a config dict (one sweep point).

    Arguments:
      cfg    = mapping with a "data" block and a "train_args" block (the
               YAML schema; see __init__ for each block's keys).
      models = (name, ia) -> class registry (default MODELS: name is
               the architecture, resmlp | rescnn | restrf; ia the
               factored IA design layered on it, None | "nla" |
               "tatt").
      **kwargs = forwarded to __init__ (opt_cls, sched_cls, probe,
               thresholds, use_amp, rescale, activation, device, quiet).

    Returns:
      an EmulatorExperiment with the resolved data / train_args / model.
    """
    models = MODELS if models is None else models
    for block in ("data", "train_args"):
      if block not in cfg:
        raise KeyError(
          f"config is missing the required block: {block!r}")
    # a scalar (derived-parameter) run is signalled by data.outputs; on it
    # param_cuts is optional (a scalar chain is already the target
    # distribution, and the omega-windows reference params a scalar input set
    # may not carry), while on a data-vector run it stays required.
    is_scalar = "outputs" in cfg["data"]
    is_cmb    = "cmb" in cfg["data"]
    is_grid   = "grid" in cfg["data"]
    is_grid2d = "grid2d" in cfg["data"]
    if (int(is_scalar) + int(is_cmb) + int(is_grid) + int(is_grid2d)) > 1:
      raise ValueError(
        "data.outputs (scalar), data.cmb (CMB spectrum), data.grid "
        "(background function), and data.grid2d (matter power spectrum) "
        "are mutually exclusive; a config carries at most one of them.")
    # validate_param_cuts (below): the physical window cuts now live in
    # data.param_cuts; run this before the generic whitelist so a flat
    # cut key (the old layout) gets the migration message, not a bare
    # "unknown key". On a scalar run with no param_cuts block it is skipped.
    if (not (is_scalar or is_cmb or is_grid or is_grid2d)
        or "param_cuts" in cfg["data"]):
      validate_param_cuts(cfg["data"])
    # validate_sizes (below): n_train / n_val are absolute row counts
    # enforced after param_cuts; run this before the generic whitelist too,
    # so a legacy train_divisor / val_divisor gets the migration message,
    # not a bare "unknown key".
    validate_sizes(cfg["data"])
    # reject any other unknown "data" key.
    unknown = set(cfg["data"]) - DATA_KEYS
    if unknown:
      raise KeyError(
        f"unknown data-block key(s): {sorted(unknown)}; allowed: "
        f"{sorted(DATA_KEYS)}")
    # default_train_args (training.py): walk train_args, collapsing every
    # [default, min, max, kind] search range to its default (first) value,
    # so a tuning YAML builds a concrete run.
    ta = default_train_args(cfg["train_args"])

    # scalar (derived-parameter) run (data.outputs present): inputs and
    # outputs are named columns of one parameter .txt, no cosmolike.
    # validate_scalar enforces the exclusivity + forbidden features and
    # returns the output names; the model is a plain design (ia None).
    if is_scalar:
      outputs = validate_scalar(cfg, train_args=ta,
                                rescale=kwargs.get("rescale", "none"))
      # fine-tune warm start on the scalar path: architecture,
      # activation, and loss form inherited from a saved SCALAR source;
      # the admissibility checks run here (wrong-kind + outputs-equal,
      # both loud before any staging), the geometry pin in
      # build_geometry.
      if ta.get("finetune") is not None:
        warmstart.validate_finetune_config(
          cfg=cfg,
          train_args=ta,
          rescale=kwargs.get("rescale", "none"),
          activation_flag=kwargs.get("activation"))
        device = kwargs.get("device")
        if device is None:
          device = pick_device()
        kwargs["device"] = device
        source_root = warmstart.resolve_source_root(ta["finetune"])
        source = warmstart.load_source(root=source_root, device=device)
        from .geometries.scalar import ScalarGeometry
        if not isinstance(source.geom, ScalarGeometry):
          raise ValueError(
            "a scalar run (data.outputs present) can only fine-tune from "
            "a scalar source emulator; " + repr(source_root) + " rebuilds "
            "a " + type(source.geom).__name__ + " output geometry (a "
            "cosmolike / CMB artifact fine-tunes on its own family's "
            "path)")
        # admissibility: the emulated outputs must match EXACTLY (names and
        # order) — the pinned standardization is per output column, so a
        # different list is a different map, not a warm-startable one.
        if list(source.geom.names) != list(outputs):
          raise ValueError(
            "finetune outputs mismatch: the source emulates "
            + repr(list(source.geom.names)) + " but data.outputs is "
            + repr(list(outputs)) + "; a scalar warm start needs the "
            "same outputs in the same order")
        block_opts = source.recipe.get("kwargs", {}).get("block_opts", {})
        act_spec   = block_opts.get("act", {}) if isinstance(block_opts,
                                                             dict) else {}
        kwargs["activation"] = act_spec.get("type", "H")
        exp = cls(data=cfg["data"], train_args=ta,
                  model_cls=source.model_cls,
                  raw_train_args=cfg["train_args"], **kwargs)
        exp.pce_opts   = None
        exp.ia         = None
        exp.arch       = source.recipe.get("name")
        exp.model_name = (source.recipe.get("name")
                          or source.model_cls.__name__.lower())
        exp._activation_notice = None
        exp._scalar        = True
        exp.outputs        = list(outputs)
        exp._finetune      = source
        exp._finetune_root = source_root
        return exp
      name = str(ta["model"].get("name", "resmlp")).lower()
      if (name, None) not in models:
        raise ValueError(
          "no scalar model for architecture " + repr(name) + " (a scalar "
          "run is a plain design, ia=None; pick a name that has it)")
      # scalar is trunk-only (still standing after the correction-head
      # family lift): the conv / TRF heads correct along an output
      # COORDINATE axis (theta / ell / z / k), and a scalar output is a
      # set of NAMED values with no axis between them — no locality for
      # a conv, no windows for attention.
      # Keyed on the class's declared head_block, not the name, so a
      # future trunk-only design composes automatically.
      model_cls = models[(name, None)]
      if model_cls.head_block is not None:
        raise ValueError(
          f"model.name {name!r} has a correction head "
          f"({model_cls.head_block}): the heads correct along an output "
          "coordinate axis (theta / ell / z / k), and a scalar output "
          "is a set of named values with no axis between them. A "
          "scalar run is trunk-only; use name: resmlp")
      # activation precedence, the same rule as the normal path below: an
      # explicit --activation flag wins over model.activation, then "H".
      explicit_flag = kwargs.get("activation")
      if kwargs.get("activation") is None:
        act_blk = ta["model"].get("activation")
        if isinstance(act_blk, dict):
          kwargs["activation"] = str(act_blk.get("type", "H"))
        elif act_blk is not None:
          kwargs["activation"] = str(act_blk)
        else:
          kwargs["activation"] = "H"
      exp = cls(data=cfg["data"], train_args=ta,
                model_cls=model_cls,
                raw_train_args=cfg["train_args"], **kwargs)
      # NPCE (the 2026-07-12 family-wide ruling): the pce: block is legal
      # on the scalar family; diagonal=True = residual-only (the ratio
      # form is a dense-covariance concept). rescale / ia were already
      # rejected by validate_scalar, so the exclusivity args are their
      # known-clean values.
      exp.pce_opts   = validate_pce(cfg.get("pce"), diagonal=True)
      exp.ia         = None
      exp.arch       = name
      exp.model_name = name
      exp._scalar    = True
      exp.outputs    = list(outputs)
      # a plain design has no head block, so no per-head activation pin; the
      # notice helper returns None, matching the normal path's shape.
      exp._activation_notice = _activation_flag_notice(
        flag_type=explicit_flag,
        head_block=exp.model_cls.head_block,
        head_pin=None)
      return exp

    # CMB-spectrum run: one spectrum's C_ell rows as the data vector,
    # the covariance from the compute_cmb_covariance.py .npz, the
    # amplitude law imposed by the chi2 wrapper (losses/cmb.py).
    # validate_cmb enforced the exclusivity + forbidden features above
    # the model choice.
    if is_cmb:
      cmb = validate_cmb(cfg, train_args=ta,
                         rescale=kwargs.get("rescale", "none"))
      # transfer learning (the 2026-07-12 symmetry ruling): validate +
      # load the frozen base before the model construction below; the
      # correction net keeps its own model: block, and the
      # exp._transfer_* stash lands after the construction.
      tr_resolved, tr_notice, tr_base, tr_root = _load_diag_transfer(
        cfg=cfg, train_args=ta, kwargs=kwargs,
        geom_cls_name="CmbDiagonalGeometry")
      # fine-tune warm start on the CMB path: the architecture,
      # activation, and loss form are inherited from a saved CMB source
      # emulator, exactly the cosmolike finetune flow; the CMB-specific
      # geometry pin happens in build_geometry. The source must itself be
      # a CMB artifact (wrong-kind loud here, before any staging).
      if ta.get("finetune") is not None:
        warmstart.validate_finetune_config(
          cfg=cfg,
          train_args=ta,
          rescale=kwargs.get("rescale", "none"),
          activation_flag=kwargs.get("activation"))
        device = kwargs.get("device")
        if device is None:
          device = pick_device()
        kwargs["device"] = device
        source_root = warmstart.resolve_source_root(ta["finetune"])
        source = warmstart.load_source(root=source_root, device=device)
        from .geometries.cmb import CmbDiagonalGeometry
        if not isinstance(source.geom, CmbDiagonalGeometry):
          raise ValueError(
            "a CMB run (data.cmb present) can only fine-tune from a CMB "
            "source emulator; " + repr(source_root) + " rebuilds a "
            + type(source.geom).__name__ + " output geometry (a "
            "cosmolike / scalar artifact fine-tunes on its own family's "
            "path)")
        block_opts = source.recipe.get("kwargs", {}).get("block_opts", {})
        act_spec   = block_opts.get("act", {}) if isinstance(block_opts,
                                                             dict) else {}
        kwargs["activation"] = act_spec.get("type", "H")
        exp = cls(data=cfg["data"], train_args=ta,
                  model_cls=source.model_cls,
                  raw_train_args=cfg["train_args"], **kwargs)
        exp.pce_opts   = None
        exp.ia         = None
        exp.arch       = source.recipe.get("name")
        exp.model_name = (source.recipe.get("name")
                          or source.model_cls.__name__.lower())
        exp._activation_notice = None
        exp._cmb           = True
        exp.cmb            = dict(cmb)
        exp._finetune      = source
        exp._finetune_root = source_root
        return exp
      name = str(ta["model"].get("name", "resmlp")).lower()
      if (name, None) not in models:
        raise ValueError(
          "no plain model for architecture " + repr(name) + " (a CMB run "
          "uses the plain designs, ia=None; pick a name that has one)")
      model_cls = models[(name, None)]
      # the conv/TRF heads ride this family (user order 2026-07-11;
      # arXiv 2505.22574's attention-vs-MLP outlier result for CMB
      # spectra). The diagonal CMB whitening keeps the multipole order, so
      # the heads' basis change degenerates to the identity (the models
      # keep W_fd / W_df as None) and the channel/token split is
      # CmbDiagonalGeometry.attach_head_coords() in build_geometry — one
      # bin, coordinate = ell; model.trf.n_tokens re-segments it so
      # attention has windows to attend across.
      # activation precedence, the same rule as the scalar/normal paths:
      # an explicit --activation flag wins over model.activation, then H.
      explicit_flag = kwargs.get("activation")
      if kwargs.get("activation") is None:
        act_blk = ta["model"].get("activation")
        if isinstance(act_blk, dict):
          kwargs["activation"] = str(act_blk.get("type", "H"))
        elif act_blk is not None:
          kwargs["activation"] = str(act_blk)
        else:
          kwargs["activation"] = "H"
      exp = cls(data=cfg["data"], train_args=ta,
                model_cls=model_cls,
                raw_train_args=cfg["train_args"], **kwargs)
      # NPCE (the 2026-07-12 family-wide ruling): legal here, residual
      # only (diagonal=True); validate_cmb already rejected the block
      # beside an amplitude_law other than "none".
      exp.pce_opts   = validate_pce(cfg.get("pce"), diagonal=True)
      exp.ia         = None
      exp.arch       = name
      exp.model_name = name
      exp._cmb       = True
      exp.cmb        = dict(cmb)
      # the per-head activation pin, read for the startup notice exactly
      # as on the cosmolike path (canonical model.<head>.activation or
      # the head: alias); build_specs licenses / rejects it later.
      head_block = exp.model_cls.head_block
      head_pin   = None
      if head_block is not None:
        head_pin = ta["model"].get(head_block, {}).get("activation")
        if head_pin is None and isinstance(ta.get("head"), dict):
          head_pin = ta["head"].get("activation")
      exp._activation_notice = _activation_flag_notice(
        flag_type=explicit_flag,
        head_block=head_block,
        head_pin=head_pin)
      if tr_base is not None:
        exp._transfer_base         = tr_base
        exp._transfer_root         = tr_root
        exp._transfer_refine       = None
        exp._transfer_form         = tr_resolved["form"]
        exp._transfer_space        = tr_resolved["space"]
        exp._transfer_space_notice = tr_notice
      return exp

    # grid (background-function) run: one background quantity's
    # rows over a stored z grid as the data vector, standardized through
    # a GridGeometry whose target law is persisted in the artifact.
    # validate_grid enforced the exclusivity + forbidden features.
    if is_grid:
      grid = validate_grid(cfg, train_args=ta,
                           rescale=kwargs.get("rescale", "none"))
      # transfer learning (the 2026-07-12 symmetry ruling): validate +
      # load the frozen base before the model construction below.
      tr_resolved, tr_notice, tr_base, tr_root = _load_diag_transfer(
        cfg=cfg, train_args=ta, kwargs=kwargs,
        geom_cls_name="GridGeometry")
      # fine-tune warm start on the grid path: architecture,
      # activation, and loss form inherited from a saved GRID source of
      # the SAME quantity/units/law/offset; the z-grid check (needs the
      # z_file) and the geometry pin live in build_geometry.
      if ta.get("finetune") is not None:
        warmstart.validate_finetune_config(
          cfg=cfg,
          train_args=ta,
          rescale=kwargs.get("rescale", "none"),
          activation_flag=kwargs.get("activation"))
        device = kwargs.get("device")
        if device is None:
          device = pick_device()
        kwargs["device"] = device
        source_root = warmstart.resolve_source_root(ta["finetune"])
        source = warmstart.load_source(root=source_root, device=device)
        from .geometries.grid import GridGeometry
        if not isinstance(source.geom, GridGeometry):
          raise ValueError(
            "a grid run (data.grid present) can only fine-tune from a "
            "grid source emulator; " + repr(source_root) + " rebuilds a "
            + type(source.geom).__name__ + " output geometry (a "
            "cosmolike / scalar / CMB artifact fine-tunes on its own "
            "family's path)")
        sgeom = source.geom
        want = (str(grid["quantity"]), str(grid["units"]),
                str(grid["law"]),
                float(grid.get("offset", 0.0)))
        have = (sgeom.quantity, sgeom.units, sgeom.law, sgeom.offset)
        if want != have:
          raise ValueError(
            "finetune grid-metadata mismatch: the source persisted "
            "(quantity, units, law, offset) = " + repr(have) + " but "
            "data.grid states " + repr(want) + "; a grid warm start "
            "needs the same quantity, units, law, and offset")
        block_opts = source.recipe.get("kwargs", {}).get("block_opts", {})
        act_spec   = block_opts.get("act", {}) if isinstance(block_opts,
                                                             dict) else {}
        kwargs["activation"] = act_spec.get("type", "H")
        exp = cls(data=cfg["data"], train_args=ta,
                  model_cls=source.model_cls,
                  raw_train_args=cfg["train_args"], **kwargs)
        exp.pce_opts   = None
        exp.ia         = None
        exp.arch       = source.recipe.get("name")
        exp.model_name = (source.recipe.get("name")
                          or source.model_cls.__name__.lower())
        exp._activation_notice = None
        exp._grid          = True
        exp.grid           = dict(grid)
        exp._finetune      = source
        exp._finetune_root = source_root
        return exp
      name = str(ta["model"].get("name", "resmlp")).lower()
      if (name, None) not in models:
        raise ValueError(
          "no plain model for architecture " + repr(name) + " (a grid "
          "run uses the plain designs, ia=None; pick a name that has "
          "one)")
      model_cls = models[(name, None)]
      # the conv/TRF heads ride the grid family too (user order
      # 2026-07-11). The diagonal standardization keeps the
      # z order, so the basis change is the identity (W_fd / W_df stay
      # None) and the split is GridGeometry.attach_head_coords() in
      # build_geometry — one bin, coordinate = z; model.trf.n_tokens
      # re-segments it for attention.
      # activation precedence, the same rule as the scalar/cmb paths.
      explicit_flag = kwargs.get("activation")
      if kwargs.get("activation") is None:
        act_blk = ta["model"].get("activation")
        if isinstance(act_blk, dict):
          kwargs["activation"] = str(act_blk.get("type", "H"))
        elif act_blk is not None:
          kwargs["activation"] = str(act_blk)
        else:
          kwargs["activation"] = "H"
      exp = cls(data=cfg["data"], train_args=ta,
                model_cls=model_cls,
                raw_train_args=cfg["train_args"], **kwargs)
      # NPCE (the 2026-07-12 family-wide ruling): legal here, residual
      # only (diagonal=True); the base fits the law-space whitened rows.
      exp.pce_opts   = validate_pce(cfg.get("pce"), diagonal=True)
      exp.ia         = None
      exp.arch       = name
      exp.model_name = name
      exp._grid      = True
      exp.grid       = dict(grid)
      # the per-head activation pin, read for the startup notice exactly
      # as on the cosmolike path; build_specs licenses / rejects it.
      head_block = exp.model_cls.head_block
      head_pin   = None
      if head_block is not None:
        head_pin = ta["model"].get(head_block, {}).get("activation")
        if head_pin is None and isinstance(ta.get("head"), dict):
          head_pin = ta["head"].get("activation")
      exp._activation_notice = _activation_flag_notice(
        flag_type=explicit_flag,
        head_block=head_block,
        head_pin=head_pin)
      if tr_base is not None:
        exp._transfer_base         = tr_base
        exp._transfer_root         = tr_root
        exp._transfer_refine       = None
        exp._transfer_form         = tr_resolved["form"]
        exp._transfer_space        = tr_resolved["space"]
        exp._transfer_space_notice = tr_notice
      return exp

    # grid2d (matter-power-spectrum) run: one MPS quantity's
    # flattened (z, k) surface as the data vector, law-transformed at
    # staging and standardized through a Grid2DGeometry.
    # validate_grid2d enforced the exclusivity + forbidden features.
    if is_grid2d:
      grid2d = validate_grid2d(cfg, train_args=ta,
                               rescale=kwargs.get("rescale", "none"))
      # transfer learning (the 2026-07-12 symmetry ruling): validate +
      # load the frozen base before the model construction below.
      tr_resolved, tr_notice, tr_base, tr_root = _load_diag_transfer(
        cfg=cfg, train_args=ta, kwargs=kwargs,
        geom_cls_name="Grid2DGeometry")
      # fine-tune warm start on the grid2d path: architecture,
      # activation, and loss form inherited from a saved grid2d source
      # of the SAME quantity/units/law; the (z, k) axes check and the
      # geometry pin live in build_geometry (the axes exist only after
      # the staging transform ran).
      if ta.get("finetune") is not None:
        warmstart.validate_finetune_config(
          cfg=cfg,
          train_args=ta,
          rescale=kwargs.get("rescale", "none"),
          activation_flag=kwargs.get("activation"))
        device = kwargs.get("device")
        if device is None:
          device = pick_device()
        kwargs["device"] = device
        source_root = warmstart.resolve_source_root(ta["finetune"])
        source = warmstart.load_source(root=source_root, device=device)
        from .geometries.grid2d import Grid2DGeometry
        if not isinstance(source.geom, Grid2DGeometry):
          raise ValueError(
            "a grid2d run (data.grid2d present) can only fine-tune from "
            "a grid2d source emulator; " + repr(source_root)
            + " rebuilds a " + type(source.geom).__name__ + " output "
            "geometry (another family's artifact fine-tunes on its own "
            "path)")
        sgeom = source.geom
        want = (str(grid2d["quantity"]), str(grid2d["units"]),
                str(grid2d["law"]))
        have = (sgeom.quantity, sgeom.units, sgeom.law)
        if want != have:
          raise ValueError(
            "finetune grid2d-metadata mismatch: the source persisted "
            "(quantity, units, law) = " + repr(have) + " but "
            "data.grid2d states " + repr(want) + "; a grid2d warm "
            "start needs the same quantity, units, and law")
        block_opts = source.recipe.get("kwargs", {}).get("block_opts", {})
        act_spec   = block_opts.get("act", {}) if isinstance(block_opts,
                                                             dict) else {}
        kwargs["activation"] = act_spec.get("type", "H")
        exp = cls(data=cfg["data"], train_args=ta,
                  model_cls=source.model_cls,
                  raw_train_args=cfg["train_args"], **kwargs)
        exp.pce_opts   = None
        exp.ia         = None
        exp.arch       = source.recipe.get("name")
        exp.model_name = (source.recipe.get("name")
                          or source.model_cls.__name__.lower())
        exp._activation_notice = None
        exp._grid2d        = True
        exp.grid2d         = dict(grid2d)
        exp._finetune      = source
        exp._finetune_root = source_root
        return exp
      name = str(ta["model"].get("name", "resmlp")).lower()
      if (name, None) not in models:
        raise ValueError(
          "no plain model for architecture " + repr(name) + " (a grid2d "
          "run uses the plain designs, ia=None; pick a name that has "
          "one)")
      model_cls = models[(name, None)]
      # the conv/TRF heads ride the grid2d family too (user order
      # 2026-07-11). The flattening is z-outer and the
      # standardization keeps the grid order, so the basis change is the
      # identity (W_fd / W_df stay None) and the split is
      # Grid2DGeometry.attach_head_coords() in build_geometry — one bin
      # PER Z SLICE: conv channels = z slices sliding along k, TRF
      # tokens = z slices (n_tokens is rejected: the slices ARE the
      # tokens).
      # activation precedence, the same rule as every family path.
      explicit_flag = kwargs.get("activation")
      if kwargs.get("activation") is None:
        act_blk = ta["model"].get("activation")
        if isinstance(act_blk, dict):
          kwargs["activation"] = str(act_blk.get("type", "H"))
        elif act_blk is not None:
          kwargs["activation"] = str(act_blk)
        else:
          kwargs["activation"] = "H"
      exp = cls(data=cfg["data"], train_args=ta,
                model_cls=model_cls,
                raw_train_args=cfg["train_args"], **kwargs)
      # NPCE (the 2026-07-12 family-wide ruling): legal here, residual
      # only (diagonal=True); the base fits the staged law-space rows
      # (arXiv 2404.12344 runs exactly this — an NPCE on the boost).
      exp.pce_opts   = validate_pce(cfg.get("pce"), diagonal=True)
      exp.ia         = None
      exp.arch       = name
      exp.model_name = name
      exp._grid2d    = True
      exp.grid2d     = dict(grid2d)
      # the per-head activation pin, read for the startup notice exactly
      # as on the cosmolike path; build_specs licenses / rejects it.
      head_block = exp.model_cls.head_block
      head_pin   = None
      if head_block is not None:
        head_pin = ta["model"].get(head_block, {}).get("activation")
        if head_pin is None and isinstance(ta.get("head"), dict):
          head_pin = ta["head"].get("activation")
      exp._activation_notice = _activation_flag_notice(
        flag_type=explicit_flag,
        head_block=head_block,
        head_pin=head_pin)
      if tr_base is not None:
        exp._transfer_base         = tr_base
        exp._transfer_root         = tr_root
        exp._transfer_refine       = None
        exp._transfer_form         = tr_resolved["form"]
        exp._transfer_space        = tr_resolved["space"]
        exp._transfer_space_notice = tr_notice
      return exp

    # fine-tune warm start (train_args.finetune present): the architecture,
    # activation, and loss form are all inherited from a saved source
    # emulator, so this path bypasses the model-name resolution below. It
    # validates the finetune YAML surface, loads + validates the source once
    # (on the run's device), and builds the experiment on the source's class.
    # See emulator/warmstart.py and notes/artifacts-inference-warmstart.md.
    if ta.get("finetune") is not None:
      warmstart.validate_finetune_config(
        cfg=cfg,
        train_args=ta,
        rescale=kwargs.get("rescale", "none"),
        activation_flag=kwargs.get("activation"))
      # resolve the device now (load_source rebuilds the source on it); pass
      # it through so __init__ does not pick a different one.
      device = kwargs.get("device")
      if device is None:
        device = pick_device()
      kwargs["device"] = device
      source_root = warmstart.resolve_source_root(ta["finetune"])
      source = warmstart.load_source(root=source_root, device=device)
      # the trunk activation the source fixed, for the banner + save attrs
      # (the network already pins it; a finetune run never restates it).
      block_opts = source.recipe.get("kwargs", {}).get("block_opts", {})
      act_spec   = block_opts.get("act", {}) if isinstance(block_opts, dict) \
          else {}
      kwargs["activation"] = act_spec.get("type", "H")
      exp = cls(data=cfg["data"],
                train_args=ta,
                model_cls=source.model_cls,
                raw_train_args=cfg["train_args"],
                **kwargs)
      # a plain source (validated): no factored IA, no NPCE base.
      exp.pce_opts   = None
      exp.ia         = None
      exp.arch       = source.recipe.get("name")
      exp.model_name = (source.recipe.get("name")
                        or source.model_cls.__name__.lower())
      exp._activation_notice = None
      exp._finetune      = source
      exp._finetune_root = source_root
      return exp

    # transfer learning (a top-level transfer: block): a frozen base under a
    # parallel correction net (emulator/losses/transfer.py). The correction
    # keeps its own model: block (capacity is the user's knob here), but its
    # intrinsic-alignment family is inherited from the base; a YAML
    # model.ia was already rejected by validate_transfer.
    if cfg.get("transfer") is not None:
      resolved_tr, space_notice = validate_transfer(
        cfg=cfg, train_args=ta, rescale=kwargs.get("rescale", "none"))
      device = kwargs.get("device")
      if device is None:
        device = pick_device()
      kwargs["device"] = device
      base_root = warmstart.resolve_source_root(cfg["transfer"])
      base = warmstart.load_source(
        root=base_root, device=device, allow_factored=True)
      # the correction architecture is the YAML model.name; the ia family is
      # forced from the base.
      name      = str(ta["model"].get("name", "resmlp")).lower()
      ia_forced = base.ia
      if (name, ia_forced) not in models:
        raise ValueError(
          "no correction model for architecture " + repr(name) + " with the "
          "base's inherited family " + repr(ia_forced) + " (the base fixes the "
          "family; pick a name that has it)")
      # activation precedence (same as the normal path; the correction has its
      # own activation): an explicit --activation wins over model.activation,
      # which wins over the "H" default.
      explicit_flag = kwargs.get("activation")
      if kwargs.get("activation") is None:
        act_blk = ta["model"].get("activation")
        if isinstance(act_blk, dict):
          kwargs["activation"] = str(act_blk.get("type", "H"))
        elif act_blk is not None:
          kwargs["activation"] = str(act_blk)
        else:
          kwargs["activation"] = "H"
      exp = cls(data=cfg["data"], train_args=ta,
                model_cls=models[(name, ia_forced)],
                raw_train_args=cfg["train_args"], **kwargs)
      exp.pce_opts   = None
      exp.ia         = ia_forced
      exp.arch       = name
      exp.model_name = name if ia_forced is None else f"{name}_{ia_forced}"
      head_block = exp.model_cls.head_block
      head_pin   = None
      if head_block is not None:
        head_pin = ta["model"].get(head_block, {}).get("activation")
        if head_pin is None and isinstance(ta.get("head"), dict):
          head_pin = ta["head"].get("activation")
      exp._activation_notice = _activation_flag_notice(
        flag_type=explicit_flag, head_block=head_block, head_pin=head_pin)
      exp._transfer_base         = base
      exp._transfer_root         = base_root
      exp._transfer_refine       = resolved_tr.get("refine")
      exp._transfer_form         = resolved_tr["form"]
      exp._transfer_space        = resolved_tr["space"]
      exp._transfer_space_notice = space_notice
      return exp

    # read (not pop) name / ia, build_specs strips both from the
    # spread, so they never reach the model constructor. A YAML `ia:
    # none` parses as the string "none" (YAML's nulls are null/~), so
    # accept it as None too.
    name = str(ta["model"].get("name", "resmlp")).lower()
    ia   = ta["model"].get("ia")
    ia   = None if ia in (None, "none") else str(ia).lower()
    if (name, ia) not in models:
      archs = []
      ias   = []
      for n, i in models:
        if n not in archs:
          archs.append(n)
        if i is not None and i not in ias:
          ias.append(i)
      raise ValueError(
        f"unknown model: name={name!r}, ia={ia!r}. name picks the "
        f"architecture ({' | '.join(sorted(archs))}); the separate ia "
        f"key layers a factored intrinsic-alignment design on it "
        f"({' | '.join(sorted(ias))}; omit it for the plain emulator)")
    # pce (top-level, optional): validate + the exclusivity checks. ia is
    # resolved above; rescale is the driver flag (kwargs). Kept a sibling of
    # data / train_args on purpose (validate_pce): one base per study.
    pce_opts = validate_pce(cfg.get("pce"),
                            rescale=kwargs.get("rescale", "none"),
                            ia=ia)
    # activation precedence, resolved once here: an explicit caller choice
    # (the drivers' --activation flag; they pass None when the flag is
    # absent) wins over the YAML's model.activation block, which wins
    # over the "H" default. The block nests {type, n_gates}; a bare
    # string (activation: H) is accepted as shorthand for the type.
    # build_specs consumes n_gates and drops the block (it is not a
    # model-constructor kwarg).
    explicit_flag = kwargs.get("activation")   # the raw flag: None = absent
    if kwargs.get("activation") is None:
      act_blk = ta["model"].get("activation")
      if isinstance(act_blk, dict):
        kwargs["activation"] = str(act_blk.get("type", "H"))
      elif act_blk is not None:
        kwargs["activation"] = str(act_blk)
      else:
        kwargs["activation"] = "H"
    exp = cls(data=cfg["data"], train_args=ta,
              model_cls=models[(name, ia)],
              raw_train_args=cfg["train_args"], **kwargs)
    # the composed display name (run_tag / the banner / file names):
    # the architecture, suffixed by the IA design when one is layered
    # (resmlp_nla, rescnn_nla). exp.ia drives the factored-design
    # lookups (IA_DESIGNS); exp.arch gates head-block filtering in
    # build_specs (which reads model_cls.head_block, the class's own
    # head-knowledge; None on direct construction translates every block).
    # stash the validated pce config (None = NPCE off); build_geometry,
    # print_design, and the save wiring all guard on exp.pce_opts.
    exp.pce_opts = pce_opts
    exp.model_name = name if ia is None else f"{name}_{ia}"
    exp.ia   = ia
    exp.arch = name
    # ruling (a) amendment: an explicit --activation flag meeting a
    # differing per-head pin is a surprise (the pin silently wins for the
    # head). Build the one-line warning once, here — the only place that
    # knew the flag was explicit (the drivers pass None when absent) — and
    # let print_design emit it, quiet-gated. The pin reads from either
    # spelling: canonical model.<head>.activation or the head: alias.
    head_block = exp.model_cls.head_block
    head_pin = None
    if head_block is not None:
      head_pin = ta["model"].get(head_block, {}).get("activation")
      if head_pin is None and isinstance(ta.get("head"), dict):
        head_pin = ta["head"].get("activation")
    exp._activation_notice = _activation_flag_notice(
      flag_type=explicit_flag, head_block=head_block, head_pin=head_pin)
    return exp

  @classmethod
  def from_yaml(cls, path, models=None, **kwargs):
    """
    Build from a YAML config file.

    Thin wrapper: read the file, then from_config (see it for the
    resolution and **kwargs).

    Arguments:
      path   = path to the YAML config (data + train_args blocks).
      models = (name, ia) -> class registry (default MODELS; keyed
               by the architecture-name and IA-design tuple, as in
               from_config).
      **kwargs = forwarded to from_config -> __init__.

    Returns:
      an EmulatorExperiment.
    """
    with open(path) as f:
      cfg = yaml.safe_load(f)
    return cls.from_config(cfg, models=models, **kwargs)

  # --- the startup banner ---
  def print_design(self):
    """
    Print the resolved run design to stdout (the startup banner).

    Announces the full design before anything trains, so a stale YAML
    is caught at launch and not one 17-minute training (or one whole
    sweep / study) later. Shared by every driver; quiet-gated through
    self.log, so --quiet silences it with the rest.

    Every line is the resolved, consumed view, not the raw YAML: phases
    resolved against the model's real capability (resolve_phase_args, so a
    single-phase model that carries two-phase keys prints them demoted, with
    the notice) and the model spec filtered to the chosen architecture (the
    class's own describe_spec). The banner then cannot contradict what the
    run executes (the training-stack directive).

    Lines printed, in order:

        device / model class / activation / rescale
           |  the environment line: what runs where
           v
        [notice]                  the phase-resolution notice, only on a
           |                      single-phase model that carried phase keys
           v
        model spec                only the sub-blocks this architecture
           |                      consumes (name / ia / mlp / activation /
           |                      its own head / compile_mode), the class
           |                      describing itself via describe_spec
           v
        run: nepochs bs loss_mode (+ the two-phase split, only when the
           |                       model is two-phase and trunk_epochs > 0)
           v
        guards: clip / rewind     only when either is set
           |
           v
        one line per sub-block    optimizer / lr / scheduler / loss /
           |                      trim / focus / ema / trunk / head, each
           |                      printed only when present in train_args
           v
        cuts                      the physical omegabh2 / omegam2h2
           |                      windows from the data block
           v
        sizes                     n_train / n_val, the absolute row
                                  counts enforced after the cuts

    A sweep or a study varies pieces per point / per trial; this
    banner shows the resolved defaults those variations start from.
    """
    # display the consumed view (the training-stack directive):
    # resolve the phase schedule against the model's real capability, so a
    # single-phase model carrying two-phase keys prints them demoted (no
    # two-phase fragment, no trunk: / head: lines) exactly as train() runs
    # it. resolve_phase_args is pure (no mutation); a two-phase model
    # resolves to a no-op and prints as before.
    two_phase = hasattr(self.model_cls, "set_train_phase")
    ta, notice = resolve_phase_args(train_args=self.train_args,
                                    two_phase=two_phase)
    d  = self.data
    self.log(f"device: {self.device}  |  "
             f"model: {self.model_cls.__name__}  |  "
             f"activation: {self.activation}  |  "
             f"rescale: {self.rescale}")
    # scalar run: name the emulated outputs and the input parameters (the
    # banner has no data-vector probe / cosmolike / cuts to show).
    if self._scalar:
      self.log(f"scalar emulator: outputs {self.outputs}  |  "
               f"inputs {self.names}  |  no cosmolike")
    # cmb run: name the spectrum, the imposed law, and the covariance file
    # (the experiment facts live IN that file's provenance).
    if self._cmb:
      cov_name = str(self.cmb["covariance"]).rsplit("/", 1)[-1]
      self.log(f"cmb emulator: spectrum {self.cmb['spectrum']}  |  "
               f"amplitude_law {self.cmb['amplitude_law']}  |  "
               f"covariance {cov_name}  |  no cosmolike")
    # grid run: name the quantity, its units, the target law, and the grid
    # sidecar (the z grid itself is a file fact, loaded at build time).
    if self._grid:
      z_name = str(self.grid["z_file"]).rsplit("/", 1)[-1]
      law_str = str(self.grid["law"])
      if law_str == "log_offset":
        law_str += f" (offset {self.grid['offset']})"
      self.log(f"grid emulator: quantity {self.grid['quantity']} "
               f"[{self.grid['units']}]  |  law {law_str}  |  "
               f"z_file {z_name}  |  no cosmolike")
    # grid2d run: name the quantity, the law (the syren base it
    # corrects), and the k thinning; the axes are file facts.
    if self._grid2d:
      stride = int(self.grid2d.get("k_stride", 1))
      self.log(f"grid2d emulator: quantity {self.grid2d['quantity']} "
               f"[{self.grid2d['units']}]  |  law {self.grid2d['law']}"
               f"  |  k_stride {stride}  |  no cosmolike")
    # fine-tune warm start: name the source artifact and the extra parameters
    # being fine-tuned in (the extras = the new covmat names absent from the
    # source; computed here from the headers, before the geometry is built).
    if self._finetune is not None:
      src_names = set(self._finetune.pgeom.names)
      extras = []
      for nm in self.names:
        if nm not in src_names:
          extras.append(nm)
      extra_str = " ".join(extras) if extras else "(none)"
      self.log(f"finetune: from {self._finetune.root}  |  "
               f"extras: {extra_str}")
    # transfer: name the frozen base, the form / space, and the extras. The
    # extras are the new (non-amplitude) names absent from the base, computed
    # from the covmat header before the geometry is built.
    if self._transfer_base is not None:
      base_names = set(self._transfer_base.pgeom.names)
      extras = []
      for nm in self.names:
        if nm not in base_names:
          extras.append(nm)
      extra_str = " ".join(extras) if extras else "(none)"
      self.log(f"transfer: from {self._transfer_base.root}  |  "
               f"form {self._transfer_form}/{self._transfer_space}  |  "
               f"extras: {extra_str}")
      # refine stage: the stage-2 joint pass, when present.
      rf = self._transfer_refine
      if rf is not None:
        self.log(f"transfer refine: {rf['epochs']} epochs, base unfrozen "
                 f"(lr x{rf['base_lr_scale']:g}, anchor lambda "
                 f"{rf['anchor']:g})")
      notice_tr = getattr(self, "_transfer_space_notice", None)
      if notice_tr is not None:
        self.log(notice_tr)
    if notice is not None:
      self.log(notice)
    # NPCE base (consumed view): the fit knobs. The kept-modes / terms
    # summary is the runtime fit report (it does not exist at banner time).
    if self.pce_opts is not None:
      p = self.pce_opts
      self.log(
        f"pce: form {p['form']}  p_max {p['p_max']}  r_max {p['r_max']}  "
        f"q {p['q']}  k_max {p['k_max']}  loo_max {p['loo_max']}  "
        f"max_terms {p['max_terms']} (base fit at staging; report below)")
    # ruling (a): the flag-vs-pin surprise warning (an explicit --activation
    # meeting a differing per-head pin), built in from_config (the only
    # place that knew the flag was explicit); quiet-gated like the notice.
    if getattr(self, "_activation_notice", None) is not None:
      self.log(self._activation_notice)
    # the model class describes itself: only the sub-blocks this
    # architecture consumes (its own head, never the inactive cnn: / trf:).
    # A finetune run has no model: block (forbidden -- the architecture is
    # inherited), so it prints the source recipe's constructor kwargs
    # instead: the consumed view of the inherited spec.
    if self._finetune is not None:
      self.log("model spec: inherited from the source recipe  "
               f"{self._finetune.recipe.get('kwargs', {})}")
    else:
      self.log(f"model spec: {self.model_cls.describe_spec(ta['model'])}")
    # trunk_epochs > 0 = the two-phase schedule (trunk then frozen-trunk
    # head); print it only when active, so ordinary runs stay unchanged.
    tk = ta.get("trunk_epochs", 0)
    # phase 2 is the frozen-trunk head by default, or a joint trunk + head
    # fine-tune when freeze_trunk is false (consumed view: name what runs).
    phase2 = "head" if ta.get("freeze_trunk", True) else "joint"
    ph = (f"  (two-phase: {tk} trunk + {ta['nepochs'] - tk} {phase2})"
          if tk else "")
    self.log(f"run: nepochs {ta['nepochs']}  bs {ta['bs']}  "
             f"loss_mode {(ta.get('loss') or {}).get('mode', 'sqrt')}{ph}")
    # the stability guards (training.py run_emulator: clip = per-step
    # gradient-norm ceiling, rewind = reload the best snapshot at every
    # plateau lr cut); printed only when set, like the two-phase line.
    clip   = ta.get("clip", 0.0)
    rewind = ta.get("rewind", False)
    if clip or rewind:
      self.log(f"guards: clip {clip}  rewind {rewind}")
    # the remaining train_args sub-blocks, one dict per line (including
    # the per-phase trunk / head override blocks), so the whole resolved
    # config is on the terminal.
    for block in ("optimizer", "lr", "scheduler", "loss", "trim", "focus",
                  "ema", "trunk", "head"):
      if block in ta:
        self.log(f"{block}: {ta[block]}")
    pc = d.get("param_cuts", {})
    # skip the physical-window line when there are no cuts (a scalar run
    # with no param_cuts block); a data-vector run always has omegabh2_hi,
    # so pc is non-empty and this prints exactly as before.
    if pc:
      self.log(f"cuts: omegabh2 in "
               f"({pc.get('omegabh2_lo')}, {pc.get('omegabh2_hi')})  "
               f"omegam2h2 in "
               f"({pc.get('omegam2h2_lo')}, {pc.get('omegam2h2_hi')})  "
               f"omegamh2 in "
               f"({pc.get('omegamh2_lo')}, {pc.get('omegamh2_hi')})  "
               f"omegamh2ns in "
               f"({pc.get('omegamh2ns_lo')}, {pc.get('omegamh2ns_hi')})")
    # the absolute run sizes, next to the cuts they are enforced against
    # (load_source raises if the post-cut pool cannot supply them).
    self.log(f"sizes: n_train {d.get('n_train')}  "
             f"n_val {d.get('n_val')} (enforced after param_cuts)")

  # --- staging + geometry (the expensive, cached pieces) ---
  def _grid2d_law_rows(self, src, base_path):
    """Materialize a grid2d source's law-space rows in place.

    Reads the (z, k) sidecars named in data.grid2d, forms the law-space
    target — log(quantity / base) under a syren law, with the base rows
    read from the generator's *_base dump aligned by src["dump_rows"];
    the raw rows under "none" — then applies the optional k_stride
    column thinning (every k_stride-th wavenumber, the top edge always
    kept). REPLACES src["C"] / src["dv"] / src["idx"] with row-aligned
    in-RAM arrays (the loop and every diagnostic then see law-space
    rows and nothing else), recomputes dv_mean when present, and
    stashes the post-thinning axes on the experiment
    (self._grid2d_z / _k) for build_geometry.

    Arguments:
      src       = the load_source dict to transform in place.
      base_path = the *_base dump path (None under law "none").

    Raises:
      ValueError on a dump/sidecar width mismatch, a base dump whose
      shape disagrees with the raw dump, or non-positive values where
      the law takes a log.
    """
    g2 = self.grid2d
    z = np.asarray(np.load(g2["z_file"], allow_pickle=False),
                   dtype="float64").reshape(-1)
    k = np.asarray(np.load(g2["k_file"], allow_pickle=False),
                   dtype="float64").reshape(-1)
    nz, nk = int(z.size), int(k.size)
    rows_sorted = np.sort(np.unique(src["idx"]))
    if int(src["dv"].shape[1]) != nz * nk:
      raise ValueError(
        "the MPS dump has " + str(int(src["dv"].shape[1])) + " columns "
        "but the z_file/k_file sidecars name a " + str(nz) + " x "
        + str(nk) + " = " + str(nz * nk) + " grid; the dump and its "
        "sidecars must come from the same generator run")
    dv_rows = np.asarray(src["dv"][rows_sorted], dtype="float64")
    law = str(g2["law"])
    if law == "none":
      law_rows = dv_rows
    else:
      base = np.load(base_path, mmap_mode="r", allow_pickle=False)
      if base.shape[1] != nz * nk:
        raise ValueError(
          "the base dump " + repr(base_path) + " has "
          + str(int(base.shape[1])) + " columns, the raw dump "
          + str(nz * nk) + "; they must come from the same generator "
          "run (the *_base sibling)")
      base_rows = np.asarray(base[src["dump_rows"]], dtype="float64")
      bad = int((~(dv_rows > 0)).sum() + (~(base_rows > 0)).sum())
      if bad > 0:
        raise ValueError(
          "the " + law + " law takes log(quantity / base) and needs "
          "both strictly positive; found " + str(bad) + " non-positive "
          "entries across the staged rows (a failed generator sample "
          "left zero rows — drop it from the dump, the failfile names "
          "it)")
      law_rows = np.log(dv_rows / base_rows)
    stride = int(g2.get("k_stride", 1))
    if stride > 1:
      kept_k = np.unique(np.concatenate(
        [np.arange(0, nk, stride), np.array([nk - 1])]))
      cols = (np.arange(nz)[:, None] * nk + kept_k[None, :]).reshape(-1)
      law_rows = law_rows[:, cols]
      k = k[kept_k]
    src["C"]   = np.asarray(src["C"][rows_sorted])
    src["dv"]  = law_rows.astype("float32")
    src["idx"] = np.arange(law_rows.shape[0])
    if "dv_mean" in src:
      src["dv_mean"] = law_rows.mean(axis=0).astype("float32")
    self._grid2d_z = z
    self._grid2d_k = k

  def stage_train(self, n_train=None):
    """
    Stage the training source (cached as self.train_set).

    A generator freshly seeded from data["split_seed"] fixes the
    cut+shuffle pool, so slicing it to different sizes gives nested subsets
    the right thing for a learning-curve sweep.

    Arguments:
      n_train = absolute number of training rows to keep; None (default)
                uses the YAML data["n_train"].

    Returns:
      the training source dict.
    """
    d   = self.data
    # scalar run: inputs + outputs are named columns of one params .txt; no
    # dv .npy, no cosmolike, param_cuts optional (load_scalar_source).
    if self._scalar:
      pc  = d.get("param_cuts", {})
      gen = torch.Generator().manual_seed(int(d["split_seed"]))
      self.train_set = load_scalar_source(
        params_path=d["train_params"],
        in_names=self.names,
        out_names=self.outputs,
        n_keep=(n_train if n_train is not None else d["n_train"]),
        gen=gen,
        ram_frac=d.get("ram_frac", 0.7),
        with_means=True,
        verbose=not self.quiet,
        omegabh2_hi=pc.get("omegabh2_hi"),
        omegabh2_lo=pc.get("omegabh2_lo"),
        omegam2h2_lo=pc.get("omegam2h2_lo"),
        omegam2h2_hi=pc.get("omegam2h2_hi"),
        omegamh2_lo=pc.get("omegamh2_lo"),
        omegamh2_hi=pc.get("omegamh2_hi"),
        omegamh2ns_lo=pc.get("omegamh2ns_lo"),
        omegamh2ns_hi=pc.get("omegamh2ns_hi"))
      return self.train_set
    # cmb / grid run: the physical windows are opt-in (the scalar-path
    # pattern) — an absent block means no cuts; the cosmolike
    # path requires the block (validate_param_cuts), so .get never
    # changes it.
    if self._cmb or self._grid or self._grid2d:
      pc = d.get("param_cuts", {})
    else:
      pc = d["param_cuts"]    # the validated physical-window bounds
    gen = torch.Generator().manual_seed(int(d["split_seed"]))
    # load_source (data_staging.py): memmap the dv .npy, apply the physical
    # cuts (omega_b h^2 < omegabh2_hi; optional omegam^2 h^2 / omegamh2 /
    # omegamh2*ns windows), keep the first n_keep rows of the seeded
    # shuffle, stage in RAM if they fit (else the memmap), return
    # {C, dv, idx} (+ C_mean / dv_mean with with_means).
    self.train_set = load_source(
      dv_path=d["train_dv"],
      params_path=d["train_params"],
      names=self.names,
      omegabh2_hi=pc.get("omegabh2_hi"),
      n_keep=(n_train if n_train is not None else d["n_train"]),
      omegabh2_lo=pc.get("omegabh2_lo"),
      omegam2h2_lo=pc.get("omegam2h2_lo"),
      omegam2h2_hi=pc.get("omegam2h2_hi"),
      omegamh2_lo=pc.get("omegamh2_lo"),
      omegamh2_hi=pc.get("omegamh2_hi"),
      omegamh2ns_lo=pc.get("omegamh2ns_lo"),
      omegamh2ns_hi=pc.get("omegamh2ns_hi"),
      gen=gen,
      ram_frac=d.get("ram_frac", 0.7),
      with_means=True,
      verbose=not self.quiet)
    # grid2d run: materialize the law-space rows now — the
    # training loop consumes train_set["dv"] directly, so the syren-law
    # transform (and the optional k-stride thinning) must happen at
    # staging, once, on the CPU cold path.
    if self._grid2d:
      self._grid2d_law_rows(src=self.train_set,
                            base_path=self.grid2d.get("train_base"))
    return self.train_set

  def stage_val(self, n_val=None):
    """
    Stage the validation source (cached as self.val_set).

    Seeded from data["split_seed"] like the train source (the val file
    differs, so the same seed gives an independent selection). Carries no
    means, geometry centers come from the training source only.

    Arguments:
      n_val = absolute number of validation rows to keep; None (default)
              uses the YAML data["n_val"].

    Returns:
      the validation source dict.
    """
    d   = self.data
    # scalar run: the val targets are named columns of the val params .txt
    # (load_scalar_source, with_means=False, param_cuts optional).
    if self._scalar:
      pc  = d.get("param_cuts", {})
      gen = torch.Generator().manual_seed(int(d["split_seed"]))
      self.val_set = load_scalar_source(
        params_path=d["val_params"],
        in_names=self.names,
        out_names=self.outputs,
        n_keep=(n_val if n_val is not None else d["n_val"]),
        gen=gen,
        ram_frac=d.get("ram_frac", 0.7),
        with_means=False,
        verbose=not self.quiet,
        omegabh2_hi=pc.get("omegabh2_hi"),
        omegabh2_lo=pc.get("omegabh2_lo"),
        omegam2h2_lo=pc.get("omegam2h2_lo"),
        omegam2h2_hi=pc.get("omegam2h2_hi"),
        omegamh2_lo=pc.get("omegamh2_lo"),
        omegamh2_hi=pc.get("omegamh2_hi"),
        omegamh2ns_lo=pc.get("omegamh2ns_lo"),
        omegamh2ns_hi=pc.get("omegamh2ns_hi"))
      return self.val_set
    # cmb / grid run: the physical windows are opt-in (the scalar-path
    # pattern) — an absent block means no cuts; the cosmolike
    # path requires the block (validate_param_cuts), so .get never
    # changes it.
    if self._cmb or self._grid or self._grid2d:
      pc = d.get("param_cuts", {})
    else:
      pc = d["param_cuts"]    # the validated physical-window bounds
    gen = torch.Generator().manual_seed(int(d["split_seed"]))
    # load_source (data_staging.py): same staging as stage_train, on the
    # val files; with_means=False (val borrows the training centers).
    self.val_set = load_source(
      dv_path=d["val_dv"],
      params_path=d["val_params"],
      names=self.names,
      omegabh2_hi=pc.get("omegabh2_hi"),
      n_keep=(n_val if n_val is not None else d["n_val"]),
      omegabh2_lo=pc.get("omegabh2_lo"),
      omegam2h2_lo=pc.get("omegam2h2_lo"),
      omegam2h2_hi=pc.get("omegam2h2_hi"),
      omegamh2_lo=pc.get("omegamh2_lo"),
      omegamh2_hi=pc.get("omegamh2_hi"),
      omegamh2ns_lo=pc.get("omegamh2ns_lo"),
      omegamh2ns_hi=pc.get("omegamh2ns_hi"),
      gen=gen,
      ram_frac=d.get("ram_frac", 0.7),
      with_means=False,
      verbose=not self.quiet)
    # grid2d run: the same law transform on the val rows (the val file's
    # own base sidecar).
    if self._grid2d:
      self._grid2d_law_rows(src=self.val_set,
                            base_path=self.grid2d.get("val_base"))
    return self.val_set

  def pool_size(self):
    """
    Number of physically-cut training rows available, the natural top
    of an N_train sweep.

    Loads the training parameter file, keeps the modeled columns, applies
    the physical cuts (the omega_b h^2 bound and the optional
    omegam^2 h^2 window, same cuts as stage_train), counts the survivors.
    Order-independent, so no shuffle or staging.

    This is the ceiling for n_train: a run's n_train must not exceed it
    (load_source raises otherwise), and an N_train sweep caps its largest
    point here.

    Returns:
      the number of training rows passing the physical cuts (an int).
    """
    d  = self.data
    # the scalar / cmb / grid families keep param_cuts optional (their
    # validators never require it), so the pool count must not demand it
    # either — the family sweep drivers call this on cuts-free YAMLs.
    if self._scalar or self._cmb or self._grid or self._grid2d:
      pc = d.get("param_cuts", {})
    else:
      pc = d["param_cuts"]   # the validated physical-window bounds
    # modeled parameter columns (drop leading weight / lnp and trailing
    # chi2), as load_source does by default.
    C   = np.loadtxt(d["train_params"], dtype="float32")[:, slice(2, -1)]
    idx = np.arange(C.shape[0])
    # phys_cut_idx (data_staging.py): keep rows inside the omega_b h^2
    # bound plus the optional omegam2h2 / omegamh2 / omegamh2*ns
    # windows (same cuts as stage_train); the report is unused here,
    # only the survivor count.
    phys, _ = phys_cut_idx(C=C, idx=idx, names=self.names,
                           omegabh2_hi=pc["omegabh2_hi"],
                           omegabh2_lo=pc.get("omegabh2_lo"),
                           omegam2h2_lo=pc.get("omegam2h2_lo"),
                           omegam2h2_hi=pc.get("omegam2h2_hi"),
                           omegamh2_lo=pc.get("omegamh2_lo"),
                           omegamh2_hi=pc.get("omegamh2_hi"),
                           omegamh2ns_lo=pc.get("omegamh2ns_lo"),
                           omegamh2ns_hi=pc.get("omegamh2ns_hi"),
                           param_file=d["train_params"])
    return int(len(phys))

  def _fit_diag_pce(self, train_set):
    """Fit the NPCE base and wrap the diagonal residual refiner loss.

    The family-wide NPCE hook (the 2026-07-12 ruling): shared by the
    scalar / cmb / grid / grid2d build_geometry branches, each calling
    it after self.pgeom and self.geom exist. Mirrors the cosmolike hook
    in build_geometry: materialize the whitened fit inputs once — the
    loaders' own tensor path, so the base sees exactly the rows the
    refiner will train on (for grid2d those are the staged law-space
    rows) — fit the closed-form sparse-Legendre base, and wrap the
    residual refiner loss in place of the family's plain chi2.
    validate_pce (diagonal=True) already pinned form "residual".

    Arguments:
      train_set = the staged training source dict ("C" / "dv" / "idx").

    Returns:
      the PCEResidualDiagChi2 chi2fn (the caller assigns self.chi2fn).
    """
    from .designs.pce import PCEEmulator
    from .losses.pce import PCEResidualDiagChi2
    idx = train_set["idx"]
    # stage the training rows as device tensors, then whiten (one
    # step per line).
    C_rows  = np.asarray(train_set["C"][idx])
    dv_rows = np.asarray(train_set["dv"][idx])
    tC  = torch.from_numpy(C_rows).float().to(self.device)
    tdv = torch.from_numpy(dv_rows).float().to(self.device)
    X_white = self.pgeom.encode(tC)
    Y_white = self.geom.encode(tdv)
    fit_opts = {k: v for k, v in self.pce_opts.items() if k != "form"}
    # quiet-gated fit report (beside the loading-sources lines).
    pce = PCEEmulator.from_training(
      self.device, X_white, Y_white, silent=self.quiet, **fit_opts)
    return PCEResidualDiagChi2(geom=self.geom, pce=pce)

  def build_geometry(self, train_set=None):
    """
    Build the input + output geometries and the chi2 (cached as
    self.pgeom / self.geom / self.chi2fn).

    Whitening centers come from the training means, so this depends on the
    training subset: rebuild per subset in an N_train sweep, build once for
    a hyperparameter sweep (independent of model / train_args).

    Arguments:
      train_set = training source dict with "C_mean" / "dv_mean" / "C" /
                  "idx" (default: self.train_set, from stage_train).

    Returns:
      (pgeom, geom, chi2fn).
    """
    train_set = self.train_set if train_set is None else train_set
    d = self.data

    # fine-tune warm start (emulator/warmstart.py): the input geometry is the
    # source geometry block-extended for the new parameters, and the
    # output geometry is the source's, pinned after the dataset / probe / width
    # checks. No from_covmat, no from_cosmolike, no cosmolike import.
    # rescale is "none" (validated), so make_chi2 wraps the pinned geometry in
    # a plain CosmolikeChi2 with no cosmolike calls. A CMB or scalar
    # finetune run skips this branch: its pin lives inside its own
    # family branch below (the checks are family-specific), while the
    # input-geometry extension is shared.
    if (self._finetune is not None and not self._cmb
        and not self._scalar and not self._grid
        and not self._grid2d):
      source = self._finetune
      self.pgeom, extra_names = warmstart.extend_input_geometry(
        source=source,
        covmat_path=d["train_covmat"],
        train_mean=train_set["C_mean"],
        device=self.device)
      self._finetune_extra_names = extra_names
      # the new dv dump's full width (dv_mean is the training-mean dv), checked
      # equal to the pinned geometry's total_size inside pin_output_geometry.
      new_dv_width = int(np.asarray(train_set["dv_mean"]).reshape(-1).shape[0])
      self.geom = warmstart.pin_output_geometry(
        source=source,
        run_data=d,
        run_probe=self.probe,
        new_dv_width=new_dv_width)
      # bin-token heads (restrf; needs_bins) still attach their per-bin split
      # to the pinned geometry, reading only the dataset ini + n(z) file (no
      # cosmolike); a plain ResMLP source (V1) skips this.
      if getattr(self.model_cls, "needs_bins", False):
        from .geometries.output import build_shear_angle_map
        build_shear_angle_map(geom=self.geom,
                              data_dir=d["cosmolike_data_dir"],
                              dataset=d["cosmolike_dataset"])
      self.chi2fn = make_chi2(geom=self.geom, rescale="none")
      return self.pgeom, self.geom, self.chi2fn

    # transfer learning (a frozen base under a parallel correction): the input
    # geometry is the base geometry block-extended for the new parameters
    # (extend_input_geometry handles both a plain and a factored base),
    # and the output geometry is the base's, pinned. The chi2 is a
    # TransferChi2 wrapping the frozen base network; no cosmolike. This is the
    # COSMOLIKE branch only — the diagonal families (the 2026-07-12 symmetry
    # ruling) pin and wrap inside their own build branches below.
    if (self._transfer_base is not None
        and not (self._scalar or self._cmb or self._grid
                 or self._grid2d)):
      base = self._transfer_base
      self.pgeom, extra_names = warmstart.extend_input_geometry(
        source=base,
        covmat_path=d["train_covmat"],
        train_mean=train_set["C_mean"],
        device=self.device)
      self._transfer_extra_names = extra_names
      new_dv_width = int(np.asarray(train_set["dv_mean"]).reshape(-1).shape[0])
      self.geom = warmstart.pin_output_geometry(
        source=base,
        run_data=d,
        run_probe=self.probe,
        new_dv_width=new_dv_width)
      if getattr(base.model_cls, "needs_bins", False):
        from .geometries.output import build_shear_angle_map
        build_shear_angle_map(geom=self.geom,
                              data_dir=d["cosmolike_data_dir"],
                              dataset=d["cosmolike_dataset"])
      # the base's whitened-input width and (factored) its template family, so
      # the loss slices the base input and combines its templates.
      if base.ia is None:
        base_in_dim = len(base.pgeom.names)
        n_templates, n_amps, coeff_fn = 1, 0, None
      else:
        base_in_dim = len(base.pgeom.pg_keep.names)
        des         = IA_DESIGNS[base.ia]
        n_templates = des["n_templates"]
        n_amps      = len(des["amp_names"])
        coeff_fn    = des["coeff_fn"]
      self.chi2fn = TransferChi2(
        geom=self.geom,
        base_net=base.model,
        base_in_dim=base_in_dim,
        form=self._transfer_form,
        space=self._transfer_space,
        n_templates=n_templates,
        n_amps=n_amps,
        coeff_fn=coeff_fn)
      return self.pgeom, self.geom, self.chi2fn

    # scalar (derived-parameter) run: the input geometry is the plain
    # ParamGeometry over the covmat; the output geometry is a ScalarGeometry
    # standardizing the staged output columns; the loss is a ScalarChi2. No
    # cosmolike, no mask; returns before the cosmolike import below.
    if self._scalar:
      from .geometries.scalar import ScalarGeometry
      from .losses.scalar import make_scalar_chi2
      # fine-tune warm start: the input geometry is the source's
      # block-extended (the same block extension the cosmolike
      # fine-tune uses), and the output geometry is the SOURCE
      # ScalarGeometry pinned wholesale — its center/scale are the
      # source's training standardization, so epoch 0 reproduces the
      # source bitwise (from_config already enforced the outputs-equal
      # check).
      if self._finetune is not None:
        self.pgeom, extra_names = warmstart.extend_input_geometry(
          source=self._finetune,
          covmat_path=d["train_covmat"],
          train_mean=train_set["C_mean"],
          device=self.device)
        self._finetune_extra_names = extra_names
        self.geom = self._finetune.geom
        self.chi2fn = make_scalar_chi2(self.geom)
        return self.pgeom, self.geom, self.chi2fn
      self.pgeom = ParamGeometry.from_covmat(
        device=self.device,
        center=train_set["C_mean"],
        covmat_path=d["train_covmat"])
      # the staged output columns (the dv slot) for the kept rows;
      # ScalarGeometry computes its center / scale from them
      # (persist-resolved-values, the standardization travels with the run).
      idx     = train_set["idx"]
      targets = np.asarray(train_set["dv"])[idx]
      self.geom = ScalarGeometry.from_targets(
        device=self.device, targets=targets, names=self.outputs)
      # NPCE (the 2026-07-12 family-wide ruling): fit the closed-form
      # base on the standardized outputs and wrap the residual refiner
      # loss in place of the plain scalar chi2 (_fit_diag_pce).
      if self.pce_opts is not None:
        self.chi2fn = self._fit_diag_pce(train_set=train_set)
        return self.pgeom, self.geom, self.chi2fn
      self.chi2fn = make_scalar_chi2(self.geom)
      return self.pgeom, self.geom, self.chi2fn

    # CMB-spectrum run: the input geometry is the plain
    # ParamGeometry over the covmat; the output geometry is the diagonal
    # CmbDiagonalGeometry whose whitening sigma comes from the
    # compute_cmb_covariance.py .npz (WITH the experiment's noise —
    # eq 4); the loss is
    # the law-dispatched CMB chi2. No cosmolike; returns before the
    # cosmolike import below.
    if self._cmb:
      from .geometries.cmb import CmbDiagonalGeometry
      from .losses.cmb import make_cmb_chi2
      # input geometry: fresh ParamGeometry on a plain run; on a finetune
      # run the source input geometry block-extended for any new
      # parameters (the shared warm-start machinery).
      if self._finetune is not None:
        self.pgeom, extra_names = warmstart.extend_input_geometry(
          source=self._finetune,
          covmat_path=d["train_covmat"],
          train_mean=train_set["C_mean"],
          device=self.device)
        self._finetune_extra_names = extra_names
      else:
        self.pgeom = ParamGeometry.from_covmat(
          device=self.device,
          center=train_set["C_mean"],
          covmat_path=d["train_covmat"])
      spectrum = str(self.cmb["spectrum"]).lower()
      law      = str(self.cmb["amplitude_law"])
      as_name  = str(self.cmb.get("as_name", ""))
      tau_name = str(self.cmb.get("tau_name", ""))
      cov = np.load(self.cmb["covariance"], allow_pickle=False)
      for key in ("ell", "sigma_" + spectrum, "cl_" + spectrum):
        if key not in cov.files:
          raise ValueError(
            "covariance file " + repr(self.cmb["covariance"]) + " lacks "
            + repr(key) + "; it must come from compute_cmb_covariance.py "
            "(its interface: ell + sigma_<s> + cl_<s> at least)")
      ell   = np.asarray(cov["ell"], dtype="int64")
      sigma = np.asarray(cov["sigma_" + spectrum], dtype="float64")
      fid   = np.asarray(cov["cl_" + spectrum], dtype="float64")
      dv    = train_set["dv"]
      idx   = train_set["idx"]
      if int(dv.shape[1]) != int(ell.size):
        raise ValueError(
          "the C_ell dump has " + str(int(dv.shape[1])) + " columns but "
          "the covariance file covers " + str(int(ell.size)) + " "
          "multipoles (l = " + str(int(ell[0])) + ".." + str(int(ell[-1]))
          + "); the dump and the covariance must share one ell grid")
      # fine-tune warm start: pin the SOURCE
      # output geometry wholesale — its center is the source's training
      # mean, so the warm-started network reproduces the source bitwise at
      # epoch 0. Pinning is only honest when the new run really shares the
      # source's whitening: same spectrum, same amplitude law + named
      # columns, and the same covariance (ell grid + sigma). Each check is
      # loud with the fix named.
      if self._finetune is not None:
        sgeom = self._finetune.geom
        if sgeom.spectrum != spectrum:
          raise ValueError(
            "finetune spectrum mismatch: the source emulates "
            + repr(sgeom.spectrum) + " but data.cmb.spectrum is "
            + repr(spectrum) + "; a warm start never crosses spectra")
        if (sgeom.law, sgeom.as_name, sgeom.tau_name) != (law, as_name,
                                                          tau_name):
          raise ValueError(
            "finetune amplitude-law mismatch: the source persisted (law="
            + repr(sgeom.law) + ", as_name=" + repr(sgeom.as_name)
            + ", tau_name=" + repr(sgeom.tau_name) + ") but data.cmb has "
            "(law=" + repr(law) + ", as_name=" + repr(as_name)
            + ", tau_name=" + repr(tau_name) + "); the law is inherited, "
            "restate the source's values")
        src_ell = sgeom.ell.detach().cpu().numpy()
        if not np.array_equal(ell, src_ell):
          raise ValueError(
            "finetune ell-grid mismatch: the covariance file covers l = "
            + str(int(ell[0])) + ".." + str(int(ell[-1])) + " ("
            + str(int(ell.size)) + " multipoles) but the source geometry "
            "was whitened on l = " + str(int(src_ell[0])) + ".."
            + str(int(src_ell[-1])) + " (" + str(int(src_ell.size))
            + "); point data.cmb.covariance at the file the source "
            "trained with")
        src_sigma = sgeom.sigma.detach().cpu().numpy()
        if not np.array_equal(sigma.astype(np.float32), src_sigma):
          raise ValueError(
            "finetune covariance mismatch: sigma_" + spectrum + " in "
            + repr(self.cmb["covariance"]) + " differs from the sigma the "
            "source geometry whitens with; a warm start requires the "
            "SAME experiment covariance file the source trained with "
            "(epoch-0 parity is impossible under a different whitening)")
        self.geom = sgeom
        # conv/TRF heads (needs_bins): attach the channel/token
        # split — a pure derivation from the pinned geometry's own ell
        # grid, so a head-model source rebuilds and fine-tunes exactly.
        if getattr(self.model_cls, "needs_bins", False):
          self.geom.attach_head_coords()
        if law == "none":
          self.chi2fn = make_cmb_chi2(geom=self.geom, law=law)
        else:
          self.chi2fn = make_cmb_chi2(geom=self.geom, law=law,
                                      param_geometry=self.pgeom,
                                      as_name=as_name,
                                      tau_name=tau_name)
        return self.pgeom, self.geom, self.chi2fn
      # transfer learning (the 2026-07-12 symmetry ruling): the input
      # geometry is the base's block-extended for the new parameters
      # and the output geometry is the BASE's, pinned after the
      # same whitening checks the finetune pin makes above (identical
      # spectrum, law "none" both sides — validate_cmb forced the run's
      # side — and identical ell grid + sigma: a frozen base under a
      # different whitening could never reach epoch-0 parity). The chi2
      # is TransferDiagChi2 wrapping the frozen base network.
      if self._transfer_base is not None:
        base  = self._transfer_base
        bgeom = base.geom
        if bgeom.spectrum != spectrum:
          raise ValueError(
            "transfer spectrum mismatch: the base emulates "
            + repr(bgeom.spectrum) + " but data.cmb.spectrum is "
            + repr(spectrum) + "; a transfer never crosses spectra")
        if bgeom.law != "none":
          raise ValueError(
            "the transfer base carries amplitude_law " + repr(bgeom.law)
            + "; a CMB transfer needs a law-none base (the transfer "
            "loss owns the target construction — train or pick a base "
            "with amplitude_law: none)")
        src_ell = bgeom.ell.detach().cpu().numpy()
        if not np.array_equal(ell, src_ell):
          raise ValueError(
            "transfer ell-grid mismatch: the covariance file covers l = "
            + str(int(ell[0])) + ".." + str(int(ell[-1])) + " ("
            + str(int(ell.size)) + " multipoles) but the base geometry "
            "was whitened on l = " + str(int(src_ell[0])) + ".."
            + str(int(src_ell[-1])) + " (" + str(int(src_ell.size))
            + "); point data.cmb.covariance at the file the base "
            "trained with")
        src_sigma = bgeom.sigma.detach().cpu().numpy()
        if not np.array_equal(sigma.astype(np.float32), src_sigma):
          raise ValueError(
            "transfer covariance mismatch: sigma_" + spectrum + " in "
            + repr(self.cmb["covariance"]) + " differs from the sigma "
            "the base geometry whitens with; a transfer requires the "
            "SAME experiment covariance file the base trained with "
            "(epoch-0 parity is impossible under a different whitening)")
        self.pgeom, extra_names = warmstart.extend_input_geometry(
          source=base,
          covmat_path=d["train_covmat"],
          train_mean=train_set["C_mean"],
          device=self.device)
        self._transfer_extra_names = extra_names
        self.geom = bgeom
        if getattr(self.model_cls, "needs_bins", False):
          self.geom.attach_head_coords()
        self.chi2fn = TransferDiagChi2(
          geom=self.geom,
          base_net=base.model,
          base_in_dim=len(base.pgeom.names),
          form=self._transfer_form,
          space=self._transfer_space)
        return self.pgeom, self.geom, self.chi2fn
      # the per-row amplitude factor f (1 for the "none" law); the law
      # reads RAW parameter columns, located by the covmat-header names.
      if law == "as_exp2tau":
        names = list(self.names)
        for nm, role in ((as_name, "as_name"), (tau_name, "tau_name")):
          if nm not in names:
            raise ValueError(
              "data.cmb." + role + " " + repr(nm) + " is not among the "
              "input parameter columns " + repr(names))
        C = np.asarray(train_set["C"])
        f = (np.exp(2.0 * C[idx, names.index(tau_name)].astype("float64"))
             / C[idx, names.index(as_name)].astype("float64"))
      else:
        f = np.ones(len(idx), dtype="float64")
      # the training-mean of the amplitude-rescaled target, streamed in
      # chunks so a big memmapped dump never loads whole (C-readable
      # explicit loop; this is a cold path, run once per staging).
      total = np.zeros(int(ell.size), dtype="float64")
      chunk = 4096
      start = 0
      while start < len(idx):
        stop = min(len(idx), start + chunk)
        rows = np.asarray(dv[idx[start:stop]], dtype="float64")
        total += (rows * f[start:stop, None]).sum(axis=0)
        start = stop
      center = total / float(len(idx))
      units = "dimensionless" if spectrum == "pp" else "muK2"
      self.geom = CmbDiagonalGeometry(device=self.device,
                                      spectrum=spectrum,
                                      ell=ell,
                                      center=center,
                                      sigma=sigma,
                                      fiducial_cl=fid,
                                      units=units,
                                      law=law,
                                      as_name=as_name,
                                      tau_name=tau_name)
      # conv/TRF heads (needs_bins): attach the channel/token
      # split — one bin, coordinate = ell (attach_head_coords).
      if getattr(self.model_cls, "needs_bins", False):
        self.geom.attach_head_coords()
      # NPCE (the 2026-07-12 family-wide ruling): fit the closed-form
      # base on the whitened C_ell rows and wrap the residual refiner
      # loss (_fit_diag_pce). validate_cmb guaranteed amplitude_law
      # "none" here, so geom.encode is the loss's whole encode.
      if self.pce_opts is not None:
        self.chi2fn = self._fit_diag_pce(train_set=train_set)
        return self.pgeom, self.geom, self.chi2fn
      if law == "none":
        self.chi2fn = make_cmb_chi2(geom=self.geom, law=law)
      else:
        self.chi2fn = make_cmb_chi2(geom=self.geom, law=law,
                                    param_geometry=self.pgeom,
                                    as_name=as_name,
                                    tau_name=tau_name)
      return self.pgeom, self.geom, self.chi2fn

    # grid (background-function) run: the input geometry is the
    # plain ParamGeometry over the covmat; the output geometry is a
    # GridGeometry over the generator's persisted z grid (read from the
    # _z.npy sidecar file — resolved values); the loss is ScalarChi2
    # reused unchanged (the law lives inside the geometry's
    # encode/decode). No cosmolike; returns before the import below.
    if self._grid:
      from .geometries.grid import GridGeometry
      from .losses.scalar import make_scalar_chi2
      quantity = str(self.grid["quantity"])
      units    = str(self.grid["units"])
      law      = str(self.grid["law"])
      offset   = float(self.grid.get("offset", 0.0))
      z = np.load(self.grid["z_file"], allow_pickle=False)
      z = np.asarray(z, dtype="float64").reshape(-1)
      dv  = train_set["dv"]
      idx = train_set["idx"]
      if int(dv.shape[1]) != int(z.size):
        raise ValueError(
          "the background dump has " + str(int(dv.shape[1])) + " columns "
          "but data.grid.z_file covers " + str(int(z.size)) + " grid "
          "points; the dump and its _z.npy sidecar must come from the "
          "same generator run")
      # fine-tune warm start: pin the
      # SOURCE GridGeometry wholesale (its center/scale are the source
      # standardization -> epoch-0 parity). The metadata was checked at
      # from_config; the GRID itself is checked here, where the z_file
      # is loaded.
      if self._finetune is not None:
        sgeom = self._finetune.geom
        src_z = sgeom.z.detach().cpu().numpy()
        if not np.array_equal(z, src_z):
          raise ValueError(
            "finetune z-grid mismatch: data.grid.z_file covers z = "
            + repr([float(z[0]), float(z[-1])]) + " (" + str(int(z.size))
            + " points) but the source geometry was standardized on z = "
            + repr([float(src_z[0]), float(src_z[-1])]) + " ("
            + str(int(src_z.size)) + "); point z_file at the grid the "
            "source trained with")
        self.pgeom, extra_names = warmstart.extend_input_geometry(
          source=self._finetune,
          covmat_path=d["train_covmat"],
          train_mean=train_set["C_mean"],
          device=self.device)
        self._finetune_extra_names = extra_names
        self.geom = sgeom
        # conv/TRF heads (needs_bins): the split derives from
        # the pinned geometry's own z grid (attach_head_coords).
        if getattr(self.model_cls, "needs_bins", False):
          self.geom.attach_head_coords()
        self.chi2fn = make_scalar_chi2(self.geom)
        return self.pgeom, self.geom, self.chi2fn
      # transfer learning (the 2026-07-12 symmetry ruling): the input
      # geometry is the base's block-extended and the output
      # geometry is the BASE's, pinned after the same z-grid check the
      # finetune pin makes above plus the metadata equality (quantity /
      # units / law — the base's whitening must be the run's). The chi2
      # is TransferDiagChi2 wrapping the frozen base network.
      if self._transfer_base is not None:
        base  = self._transfer_base
        bgeom = base.geom
        src_z = bgeom.z.detach().cpu().numpy()
        if not np.array_equal(z, src_z):
          raise ValueError(
            "transfer z-grid mismatch: data.grid.z_file covers z = "
            + repr([float(z[0]), float(z[-1])]) + " (" + str(int(z.size))
            + " points) but the base geometry was standardized on z = "
            + repr([float(src_z[0]), float(src_z[-1])]) + " ("
            + str(int(src_z.size)) + "); point z_file at the grid the "
            "base trained with")
        if (bgeom.quantity, bgeom.units, bgeom.law) != (quantity, units,
                                                        law):
          raise ValueError(
            "transfer grid-metadata mismatch: the base persisted "
            "(quantity=" + repr(bgeom.quantity) + ", units="
            + repr(bgeom.units) + ", law=" + repr(bgeom.law) + ") but "
            "data.grid has (quantity=" + repr(quantity) + ", units="
            + repr(units) + ", law=" + repr(law) + "); a transfer never "
            "crosses quantities — restate the base's values")
        self.pgeom, extra_names = warmstart.extend_input_geometry(
          source=base,
          covmat_path=d["train_covmat"],
          train_mean=train_set["C_mean"],
          device=self.device)
        self._transfer_extra_names = extra_names
        self.geom = bgeom
        if getattr(self.model_cls, "needs_bins", False):
          self.geom.attach_head_coords()
        self.chi2fn = TransferDiagChi2(
          geom=self.geom,
          base_net=base.model,
          base_in_dim=len(base.pgeom.names),
          form=self._transfer_form,
          space=self._transfer_space)
        return self.pgeom, self.geom, self.chi2fn
      self.pgeom = ParamGeometry.from_covmat(
        device=self.device,
        center=train_set["C_mean"],
        covmat_path=d["train_covmat"])
      targets = np.asarray(dv[idx])
      self.geom = GridGeometry.from_targets(
        device=self.device, targets=targets, z=z,
        quantity=quantity, units=units, law=law, offset=offset)
      # conv/TRF heads (needs_bins): attach the channel/token
      # split — one bin, coordinate = z (attach_head_coords).
      if getattr(self.model_cls, "needs_bins", False):
        self.geom.attach_head_coords()
      # NPCE (the 2026-07-12 family-wide ruling): fit the closed-form
      # base on the law-space whitened rows and wrap the residual
      # refiner loss (_fit_diag_pce).
      if self.pce_opts is not None:
        self.chi2fn = self._fit_diag_pce(train_set=train_set)
        return self.pgeom, self.geom, self.chi2fn
      self.chi2fn = make_scalar_chi2(self.geom)
      return self.pgeom, self.geom, self.chi2fn

    # grid2d (matter-power-spectrum) run: the input geometry is
    # the plain ParamGeometry over the covmat; the output geometry is a
    # Grid2DGeometry over the post-staging (z, k) axes (the staging
    # hook already formed the LAW-SPACE rows and applied any k_stride,
    # so train_set["dv"] is what the network learns); the loss is
    # ScalarChi2 reused unchanged. No cosmolike; returns before the
    # import below.
    if self._grid2d:
      from .geometries.grid2d import Grid2DGeometry
      from .losses.scalar import make_scalar_chi2
      if self._grid2d_z is None or self._grid2d_k is None:
        raise RuntimeError(
          "grid2d geometry requested before staging ran (the law "
          "transform sets the (z, k) axes); call stage_train first")
      z2, k2 = self._grid2d_z, self._grid2d_k
      quantity = str(self.grid2d["quantity"])
      units    = str(self.grid2d["units"])
      law      = str(self.grid2d["law"])
      dv  = train_set["dv"]
      idx = train_set["idx"]
      if int(dv.shape[1]) != int(z2.size) * int(k2.size):
        raise ValueError(
          "the staged law-space rows have " + str(int(dv.shape[1]))
          + " columns but the post-thinning grid is "
          + str(int(z2.size)) + " x " + str(int(k2.size))
          + "; staging and geometry disagree (internal ordering bug)")
      # fine-tune warm start: pin the
      # SOURCE Grid2DGeometry wholesale after the axes check (the
      # metadata was checked at from_config; the AXES exist only now).
      if self._finetune is not None:
        sgeom = self._finetune.geom
        src_z = sgeom.z.detach().cpu().numpy()
        src_k = sgeom.k.detach().cpu().numpy()
        if not (np.array_equal(z2, src_z) and np.array_equal(k2, src_k)):
          raise ValueError(
            "finetune (z, k)-grid mismatch: the staged axes are z "
            + repr([float(z2[0]), float(z2[-1]), int(z2.size)])
            + " / k " + repr([float(k2[0]), float(k2[-1]),
                              int(k2.size)])
            + " but the source geometry was standardized on z "
            + repr([float(src_z[0]), float(src_z[-1]), int(src_z.size)])
            + " / k " + repr([float(src_k[0]), float(src_k[-1]),
                              int(src_k.size)])
            + "; use the source's generator grids and k_stride")
        self.pgeom, extra_names = warmstart.extend_input_geometry(
          source=self._finetune,
          covmat_path=d["train_covmat"],
          train_mean=train_set["C_mean"],
          device=self.device)
        self._finetune_extra_names = extra_names
        self.geom = sgeom
        # conv/TRF heads (needs_bins): the split derives from
        # the pinned geometry's own (z, k) axes (attach_head_coords).
        if getattr(self.model_cls, "needs_bins", False):
          self.geom.attach_head_coords()
        self.chi2fn = make_scalar_chi2(self.geom)
        return self.pgeom, self.geom, self.chi2fn
      # transfer learning (the 2026-07-12 symmetry ruling, "this for
      # sure should be allowed for MPS"): the input geometry is the
      # base's block-extended and the output geometry is the
      # BASE's, pinned after the same (z, k)-axes check the finetune
      # pin makes above plus the metadata equality (quantity / units /
      # law). The chi2 is TransferDiagChi2 wrapping the frozen base.
      if self._transfer_base is not None:
        base  = self._transfer_base
        bgeom = base.geom
        src_z = bgeom.z.detach().cpu().numpy()
        src_k = bgeom.k.detach().cpu().numpy()
        if not (np.array_equal(z2, src_z) and np.array_equal(k2, src_k)):
          raise ValueError(
            "transfer (z, k)-grid mismatch: the staged axes are z "
            + repr([float(z2[0]), float(z2[-1]), int(z2.size)])
            + " / k " + repr([float(k2[0]), float(k2[-1]),
                              int(k2.size)])
            + " but the base geometry was standardized on z "
            + repr([float(src_z[0]), float(src_z[-1]), int(src_z.size)])
            + " / k " + repr([float(src_k[0]), float(src_k[-1]),
                              int(src_k.size)])
            + "; use the base's generator grids and k_stride")
        if (bgeom.quantity, bgeom.units, bgeom.law) != (quantity, units,
                                                        law):
          raise ValueError(
            "transfer grid2d-metadata mismatch: the base persisted "
            "(quantity=" + repr(bgeom.quantity) + ", units="
            + repr(bgeom.units) + ", law=" + repr(bgeom.law) + ") but "
            "data.grid2d has (quantity=" + repr(quantity) + ", units="
            + repr(units) + ", law=" + repr(law) + "); a transfer never "
            "crosses quantities — restate the base's values")
        self.pgeom, extra_names = warmstart.extend_input_geometry(
          source=base,
          covmat_path=d["train_covmat"],
          train_mean=train_set["C_mean"],
          device=self.device)
        self._transfer_extra_names = extra_names
        self.geom = bgeom
        if getattr(self.model_cls, "needs_bins", False):
          self.geom.attach_head_coords()
        self.chi2fn = TransferDiagChi2(
          geom=self.geom,
          base_net=base.model,
          base_in_dim=len(base.pgeom.names),
          form=self._transfer_form,
          space=self._transfer_space)
        return self.pgeom, self.geom, self.chi2fn
      self.pgeom = ParamGeometry.from_covmat(
        device=self.device,
        center=train_set["C_mean"],
        covmat_path=d["train_covmat"])
      targets = np.asarray(dv[idx])
      self.geom = Grid2DGeometry.from_targets(
        device=self.device, targets=targets, z=z2, k=k2,
        quantity=quantity, units=units, law=law)
      # constant-column pin (law-agnostic): law-space columns constant
      # across the training cosmologies (the boost's low-k tail, under
      # any law) are pinned by the geometry, reported here once
      # (quiet-gated), never silent.
      if self.geom.const_mask is not None:
        n_pin = int(self.geom.const_mask.sum())
        self.log("grid2d: " + str(n_pin) + " constant law-space grid "
                 "point(s) pinned (decode returns the training "
                 "constant — the physics is flat there)")
      # conv/TRF heads (needs_bins): attach the channel/token
      # split — one bin per z slice, each of length nk
      # (attach_head_coords; conv channels / TRF tokens = z slices).
      if getattr(self.model_cls, "needs_bins", False):
        self.geom.attach_head_coords()
      # NPCE (the 2026-07-12 family-wide ruling): fit the closed-form
      # base on the staged law-space rows and wrap the residual refiner
      # loss (_fit_diag_pce); a constant-column pin composes in decode
      # (the geometry pins the COMBINED base + net prediction).
      if self.pce_opts is not None:
        self.chi2fn = self._fit_diag_pce(train_set=train_set)
        return self.pgeom, self.geom, self.chi2fn
      self.chi2fn = make_scalar_chi2(self.geom)
      return self.pgeom, self.geom, self.chi2fn

    # config validation first, before the cosmolike import below: a bad
    # combination should fail fast, and stays testable off-workstation.
    if getattr(self.model_cls, "factored", False) and self.rescale != "none":
      raise ValueError(
        f"model {self.model_name!r} does not compose with --rescale "
        "(the factored loss owns the target construction)")
    # lazy import: DataVectorGeometry.from_cosmolike pulls in cosmolike,
    # which lives only on the workstation, importing here keeps the module
    # importable for the config logic without cosmolike.
    from .geometries.output import DataVectorGeometry

    # The input whitening. The plain designs whiten every parameter
    # (ParamGeometry); the factored designs (the models' factored flag,
    # picked by model.ia) instead whiten only the non-amplitude
    # columns and append the raw amplitudes last
    # (AmplitudeFactorGeometry), so the model can drop them and the loss
    # can read them, the amplitudes never enter the network. Which
    # columns / polynomial / template count is the IA_DESIGNS[self.ia]
    # entry.
    if getattr(self.model_cls, "factored", False):
      des = IA_DESIGNS[self.ia]
      self.pgeom = AmplitudeFactorGeometry.from_covmat(
        device=self.device,
        center=train_set["C_mean"],
        covmat_path=d["train_covmat"],
        amp_names=des["amp_names"])
    else:
      # ParamGeometry.from_covmat (geometries.parameter.py): eigendecompose
      # the parameter covmat so encode() centers, rotates, unit-scales the
      # params the model sees.
      self.pgeom = ParamGeometry.from_covmat(
        device=self.device,
        center=train_set["C_mean"],
        covmat_path=d["train_covmat"])

    # DataVectorGeometry.from_cosmolike (geometries.output.py): the output
    # geometry, read cosmolike's cov / mask / inverse-cov, eigendecompose
    # the kept (unmasked) block, so encode()/chi2 whiten + score the dv.
    self.geom = DataVectorGeometry.from_cosmolike(
      device=self.device,
      dv_center=train_set["dv_mean"],
      data_dir=d["cosmolike_data_dir"],
      dataset=d["cosmolike_dataset"],
      probe=self.probe)

    # bin-token heads (restrf; the needs_bins flag) split the dv per
    # tomographic bin: build_shear_angle_map (geometries.output.py)
    # attaches bin_sizes to the geometry, reading only the dataset ini
    # and the n(z) file, no cosmolike.
    if getattr(self.model_cls, "needs_bins", False):
      from .geometries.output import build_shear_angle_map
      build_shear_angle_map(geom=self.geom,
                            data_dir=d["cosmolike_data_dir"],
                            dataset=d["cosmolike_dataset"])

    # NPCE (the top-level pce: block): fit the closed-form sparse-Legendre
    # base on the staged, whitened train set, then wrap the residual / ratio
    # refiner loss in place of the plain chi2. Guarded on pce_opts (absent =
    # skipped, everything below byte-identical). pce is exclusive with
    # rescale and model.ia (validate_pce), so this path never coincides with
    # the factored / make_chi2 branches below.
    if self.pce_opts is not None:
      from .designs.pce import PCEEmulator
      from .losses.pce import PCEResidualChi2, PCERatioChi2
      # materialize the whitened fit inputs once: X_white = pgeom.encode of
      # the raw params, Y_white = geom.encode of the raw dvs (from_training
      # converts to float64 numpy internally). Same tensor path the loaders
      # use, staged one step per line.
      idx = train_set["idx"]
      C_rows  = np.asarray(train_set["C"][idx])
      dv_rows = np.asarray(train_set["dv"][idx])
      tC  = torch.from_numpy(C_rows).float().to(self.device)
      tdv = torch.from_numpy(dv_rows).float().to(self.device)
      X_white = self.pgeom.encode(tC)
      Y_white = self.geom.encode(tdv)
      form     = self.pce_opts["form"]
      fit_opts = {k: v for k, v in self.pce_opts.items() if k != "form"}
      # quiet-gated fit report (beside the loading-sources lines).
      pce = PCEEmulator.from_training(
        self.device, X_white, Y_white, silent=self.quiet, **fit_opts)
      if form == "residual":
        self.chi2fn = PCEResidualChi2(geom=self.geom, pce=pce)
      else:
        self.chi2fn = PCERatioChi2(geom=self.geom, pce=pce)
      return self.pgeom, self.geom, self.chi2fn

    # TemplateFactoredChi2 (losses/ia.py): the factored-design
    # loss. It combines the model's templates in closed form (nla:
    # xi = K0 + A1*K1 + A1^2*K2 via nla_coeffs), reading each sample's
    # own amplitudes off the encoded input's last columns, then scores
    # the plain chi2 on the combined xi.
    if getattr(self.model_cls, "factored", False):
      des = IA_DESIGNS[self.ia]
      self.chi2fn = TemplateFactoredChi2(
        geom=self.geom,
        coeff_fn=des["coeff_fn"],
        n_amps=len(des["amp_names"]))
      return self.pgeom, self.geom, self.chi2fn

    # make_chi2 (losses/core.py): wrap geom in the loss, plain
    # CosmolikeChi2, or the analytic-R RescaledChi2 / ResidualBaseChi2 when
    # rescale != "none". cosmo_mid = training-cloud mean (R = 1 there for a
    # rescaled chi2; the plain chi2 ignores it).
    self.chi2fn = make_chi2(
      geom=self.geom,
      rescale=self.rescale,
      param_geometry=self.pgeom,
      cosmo_mid=train_set["C"][train_set["idx"]].mean(0),
      data_dir=d["cosmolike_data_dir"],
      dataset=d["cosmolike_dataset"])

    return self.pgeom, self.geom, self.chi2fn

  # --- per-run pieces ---
  def build_specs(self, train_args=None):
    """
    Assemble the six run_emulator spec dicts for one run.

    build_run_specs from train_args, then inject the named activation and
    for ResCNN only, the data geometry (see body comments). A
    hyperparameter sweep passes a varied train_args.

    Arguments:
      train_args = resolved train_args mapping (default:
                   self.train_args). Range-free (from default_train_args
                   / suggest_train_args). A leftover model.name is
                   stripped here, so a suggest_train_args result works too.

    Returns:
      the keyed spec dict run_emulator consumes as **specs.
    """
    train_args = self.train_args if train_args is None else train_args

    # fine-tune warm start: the model spec is inherited from the source recipe
    # (the architecture is never restated in YAML), not translated from
    # a model: block. The other five specs (optimizer / lr / scheduler / trim /
    # focus) still come from train_args as usual; the lower learning rate is
    # just a smaller lr_base in the lr: block. resolved_model records the
    # source recipe with the new input width (n_n) and the pinned output width,
    # every other value inherited (persist-resolved-values).
    if self._finetune is not None:
      source = self._finetune
      ft = train_args.get("finetune", {})
      # compile mode: the finetune block may override the source's for this
      # machine; absent -> the mode the source recipe stored.
      if "compile_mode" in ft:
        compile_mode = ft["compile_mode"]
      else:
        compile_mode = source.compile_mode
      # recipe_to_model_opts injects the geometry only when the recipe records
      # needs_geom (the conv / TRF heads); pass it unconditionally and let the
      # recipe decide, so the class flag and the recipe cannot disagree.
      model_opts = warmstart.recipe_to_model_opts(
        recipe=source.recipe,
        geom=self.geom,
        compile_mode=compile_mode)
      # build_run_specs needs a model: block; feed it an empty one for the five
      # non-model specs, then overwrite model_opts with the inherited spec.
      ta_specs = dict(train_args)
      ta_specs["model"] = {}
      specs = build_run_specs(train_args=ta_specs,
                              model_cls=source.model_cls,
                              opt_cls=self.opt_cls,
                              sched_cls=self.sched_cls)
      specs["model_opts"] = model_opts
      recipe = dict(source.recipe)
      recipe["input_dim"]    = int(len(self.pgeom.names))
      recipe["output_dim"]   = int(self.geom.dest_idx.numel())
      recipe["compile_mode"] = compile_mode
      self.resolved_model = recipe
      return specs

    # Translate the nested YAML model block into the constructors'
    # flat kwargs (MODEL_BLOCK_KEYS). name / ia picked the class
    # (from_config resolved them); the activation type was resolved by
    # from_config too (re-reading it here would let the YAML overrule
    # an explicit --activation), so only its n_gates is consumed here;
    # compile_mode passes through (make_model strips it). Everything
    # else must be one of the component sub-blocks. A head block for a
    # head this architecture does not have is ignored, not an error:
    # keeping cnn: and trf: both configured lets a run switch
    # architectures by changing name: alone. Unknown keys, top-level
    # or inside the active blocks, still raise, so a misspelled knob
    # that would affect the run fails loudly.
    ta = dict(train_args)
    model_opts = {}
    n_gates = 3
    norm_name = "affine"             # model.norm (absent = the paper's
                                     # per-layer affine, byte-identical)
    head_pin = None                  # the model.<head>.activation pin
    # the model class owns its head-knowledge (head_block: None | "cnn" |
    # "trf"); with arch known (from_config ran) skip the inactive head's
    # block, with arch None (direct construction) translate every block.
    head = self.model_cls.head_block if self.arch is not None else None
    for key, sub in ta["model"].items():
      if key in ("name", "ia"):
        continue
      if key == "activation":
        if isinstance(sub, dict) and "n_gates" in sub:
          n_gates = int(sub["n_gates"])
        continue
      if key == "norm":
        # a model-level string like activation (not a sub-block): its
        # factory is built into block_opts["norm"] after build_run_specs
        # (make_norm validates the three-value whitelist there).
        norm_name = sub
        continue
      if key == "compile_mode":
        model_opts["compile_mode"] = sub
        continue
      if key in MODEL_BLOCK_KEYS:
        # skip the inactive head's block entirely (its contents are
        # not even validated, a stale key in a block that cannot
        # affect this run should not stop it). With arch unknown
        # (direct construction) every present block is translated.
        if (self.arch is not None and key != "mlp"
            and key != head):
          continue
        table = MODEL_BLOCK_KEYS[key]
        for k2, v2 in sub.items():
          if k2 not in table:
            raise ValueError(
              f"unknown key model.{key}.{k2}; allowed: "
              f"{' / '.join(sorted(table))}")
          if k2 == "activation":
            # the per-head activation pin: its value is a factory spec
            # (type + optional n_gates), not a scalar to copy — validate
            # it now (strict), then resolve it against the head: alias +
            # license it after the loop and build the factory. Only the
            # active head's block reaches here (the inactive one is
            # skipped above), so head_pin is that head's pin.
            head_pin = _head_activation_spec(v2,
                                             f"model.{key}.activation")
            continue
          model_opts[table[k2]] = v2
        continue
      raise ValueError(
        f"unknown model key {key!r}; the model block nests its "
        "knobs: name / ia / mlp / activation / norm / cnn / trf / "
        "compile_mode")
    if "int_dim_res" not in model_opts:
      raise ValueError(
        "the model.mlp block (width, n_blocks) is required: every "
        "architecture is built on the ResMLP trunk")
    ta["model"] = model_opts

    # the per-head activation (rulings c/d, notes head-activation-per-
    # component + training-stack): the canonical pin
    # (model.<head>.activation, in head_pin) and the head: activation:
    # alias are one setting — resolve them (both given = a loud error),
    # license the pin against a frozen-trunk head phase (trunk_epochs > 0
    # and freeze_trunk not False; the head family is the head-phase
    # family), then build the factory into head_act. resolve_phase_args
    # has already dropped head: on a single-phase model, so the alias
    # reaches here only for a real head.
    head_alias = None
    head_blk = train_args.get("head")
    if isinstance(head_blk, dict) and "activation" in head_blk:
      head_alias = _head_activation_spec(head_blk["activation"],
                                         "head.activation")
    head_pin = _resolve_head_activation(
      canonical=head_pin, alias=head_alias,
      head_block=self.model_cls.head_block,
      trunk_epochs=train_args.get("trunk_epochs", 0),
      freeze_trunk=train_args.get("freeze_trunk"))
    if head_pin is not None:
      model_opts["head_act"] = make_activation(head_pin["type"],
                                               n_gates=head_pin["n_gates"])

    # build_run_specs (training.py): turn the train_args sub-blocks into the
    # six {cls, **kwargs} spec dicts run_emulator consumes (model_opts /
    # opt_opts / lr_opts / sched_opts / trim_opts / focus_opts), with this
    # experiment's fixed classes.
    specs = build_run_specs(
      train_args=ta,
      model_cls=self.model_cls,
      opt_cls=self.opt_cls,
      sched_cls=self.sched_cls)

    # make_activation (activations.py): map the activation name to a
    # factory act(dim) -> nn.Module (the paper's H, or a Power / Gated /
    # GatedPower variant). A callable, so it cannot live in the YAML; inject
    # it into the ResBlock options (setdefault keeps config-set block_opts).
    # n_gates (YAML model.activation.n_gates, default 3) sizes the
    # multi-gate families.
    specs["model_opts"].setdefault(
      "block_opts", {})["act"] = make_activation(self.activation,
                                                 n_gates=n_gates)

    # make_norm (designs/blocks.py): map model.norm to
    # the ResBlock norm factory norm(size) -> module (affine = the
    # paper's per-layer g x + b, the default and byte-identical;
    # per_feature = a dim-sized gain/bias; none = Identity), injected
    # into the same trunk block_opts, so it reaches every architecture's
    # ResBlock trunk (the TRF LayerNorm and the CNN head keep their own).
    # make_norm validates the three-value whitelist loudly.
    specs["model_opts"].setdefault(
      "block_opts", {})["norm"] = make_norm(norm_name)

    # Geometry-consuming heads (the needs_geom flag: the conv and TRF
    # models) get geom injected for their fixed full<->theta
    # basis-change buffers (+ bin_sizes for restrf); ResMLP /
    # TemplateMLP take none. compile_mode falls back to "default" for
    # them (reduce-overhead's CUDA-graph capture trips on the gated
    # skip-add); setdefault keeps a YAML-set choice.
    if getattr(self.model_cls, "needs_geom", False):
      specs["model_opts"]["geom"] = self.geom
      specs["model_opts"].setdefault("compile_mode", "default")

    # The factored models (designs/ia.py) need the
    # factored-design shape from IA_DESIGNS[self.ia]: how many
    # amplitude columns AmplitudeFactorGeometry appended (dropped from
    # the trunk input) and how many templates to emit (3 for nla:
    # GG, GI, II). setdefault keeps YAML-set overrides.
    if getattr(self.model_cls, "factored", False):
      des = IA_DESIGNS[self.ia]
      specs["model_opts"].setdefault("n_amps", len(des["amp_names"]))
      specs["model_opts"].setdefault("n_templates",
                                     des["n_templates"])

    # exp.resolved_model: a serializable rebuild recipe, assembled HERE
    # beside the specs by the same code that built them, so it cannot
    # diverge (the consumed-view doctrine for artifacts). It records every
    # constructor kwarg make_model will actually pass, callables serialized
    # by name: block_opts act -> {type, n_gates}, norm -> the make_norm name,
    # head_act -> {type, n_gates} | None. geom is not serialized (rebuild
    # passes the saved geometry; needs_geom records that it is needed); the
    # dims match run_emulator's make_model call exactly (in_dim / out_dim).
    mo = specs["model_opts"]
    in_dim = getattr(self.pgeom, "encoded_dim",
                     self.train_set["C"].shape[1])
    recipe = {
      "cls": self.model_cls.__module__ + "." + self.model_cls.__qualname__,
      "name": self.arch,
      "ia": self.ia,
      "input_dim": int(in_dim),
      "output_dim": int(self.geom.dest_idx.numel()),
      "compile_mode": mo.get("compile_mode", DEFAULT_COMPILE_MODE),
      "needs_geom": bool(getattr(self.model_cls, "needs_geom", False)),
      "kwargs": {},
    }
    for k, v in mo.items():
      # cls -> the qualname above; compile_mode -> the top level (make_model
      # consumes it, not the constructor); geom -> the saved geometry;
      # head_act -> recorded below for every head model.
      if k in ("cls", "compile_mode", "geom", "head_act"):
        continue
      if k == "block_opts":
        recipe["kwargs"]["block_opts"] = {
          "act": {"type": self.activation, "n_gates": int(n_gates)},
          "norm": norm_name,
        }
      else:
        recipe["kwargs"][k] = v
    if getattr(self.model_cls, "head_block", None) is not None:
      recipe["kwargs"]["head_act"] = (
        None if head_pin is None
        else {"type": head_pin["type"],
              "n_gates": int(head_pin["n_gates"])})
    # materialize every remaining constructor default the YAML did not set
    # (the standing rule: the recipe records values, never "it was defaulted",
    # so rebuild is immune to a default drifting later). input_dim / output_dim
    # are the top-level dims, block_opts is the factory dict above, geom is the
    # saved-geometry allowlist; every other defaulted param joins the recipe.
    import inspect
    for pn, pm in inspect.signature(self.model_cls.__init__).parameters.items():
      if pn in ("self", "input_dim", "output_dim", "block_opts", "geom"):
        continue
      if pm.default is not inspect.Parameter.empty:
        recipe["kwargs"].setdefault(pn, pm.default)
    self.resolved_model = recipe

    return specs

  def train(self, train_args=None, silent=None):
    """
    Train one model on the staged sources; return its histories.

    Uses the cached sources / geometry / chi2 (build them first via
    stage_train / stage_val / build_geometry, or call run). train_args
    overrides the resolved config for this run (a hyperparameter sweep
    passes a varied copy); the model and histories stay on the instance.

    Arguments:
      train_args = resolved train_args for this run (default:
                   self.train_args).
      silent     = override run_emulator's per-epoch printing; None
                   (default) -> train_args["silent"] or self.quiet. A
                   search driver passes silent=True so trials train quietly
                   regardless of self.quiet.

    Returns:
      (model, train_losses, medians, means, fracs), run_emulator's
      return, the model at its best frac>0.2 epoch.
    """
    train_args = self.train_args if train_args is None else train_args
    # resolve_phase_args (above): a shared YAML may carry the two-phase
    # keys (trunk_epochs / trunk: / head:), but a single-phase model
    # (resmlp, incl. its ia variants — no set_train_phase method) would
    # die in run_emulator's capability guard. Every design WITH a
    # correction head is two-phase capable: plain rescnn / restrf on
    # every family they ride, and the factored-IA templates (the
    # 2026-07-12 ruling — any trunk+head design may train in two
    # phases). For a single-phase model demote the phase keys (drop
    # head: / trunk_epochs, merge trunk: into the top level) here,
    # once, at the choke point every driver funnels through; a
    # two-phase model is an exact no-op. It never mutates train_args (a
    # sweep reuses it across points); the notice is quiet-gated like the
    # config banner.
    two_phase = hasattr(self.model_cls, "set_train_phase")
    train_args, phase_notice = resolve_phase_args(train_args=train_args,
                                                  two_phase=two_phase)
    if phase_notice is not None:
      self.log(phase_notice)
    specs = self.build_specs(train_args=train_args)
    # None -> the config/quiet default; a search driver forces silent.
    silent_run = (train_args.get("silent", False) or self.quiet
                  if silent is None else silent)

    # fine-tune warm start (emulator/warmstart.py): transfer the source
    # weights into the new (block-extended) shape and run the pre-train parity
    # gate, then hand run_emulator the transferred state as init_state. The
    # one verdict line is essential-only (printed even under a quiet run: it is
    # the checked-exactness fact, not per-epoch chatter). None on a plain run,
    # so run_emulator's init_state stays absent and the run is byte-identical.
    init_state = None
    anchor_spec = None
    if self._finetune is not None:
      init_state, verdict, padded_keys = warmstart.build_warm_start(
        source=self._finetune,
        new_pgeom=self.pgeom,
        pinned_geom=self.geom,
        model_opts=specs["model_opts"],
        train_set=self.train_set,
        extra_names=self._finetune_extra_names,
        device=self.device)
      self.log(verdict)
      # finetune.anchor (optional L2-SP): pull the trained weights back toward
      # the transferred init_state, with the padded extra columns excluded from
      # the pull. Absent = no anchor, byte-identical. weight_decay decays toward
      # 0 (away from the source), so recommend 0.0 beside a nonzero anchor.
      ft_block = train_args.get("finetune", {})
      if ft_block.get("anchor") is not None:
        lam = float(ft_block["anchor"])
        anchor_spec = {
          "reference": init_state,
          "masks": warmstart.anchor_masks(
            init_state=init_state,
            padded_keys=padded_keys,
            n_extra=len(self._finetune_extra_names),
            device=self.device),
          "lam": lam,
        }
        if (lam > 0.0
            and specs["opt_opts"].get("weight_decay", 0.0) > 0.0):
          self.log("note: finetune.anchor is set with weight_decay > 0; the "
                   "decay pulls toward 0 (away from the source), so consider "
                   "optimizer.weight_decay 0.0 beside the anchor")
    elif self._transfer_base is not None:
      # transfer: zero-init the correction's final layer and run the bitwise
      # parity gate (epoch 0 == the frozen base), then hand run_emulator the
      # zero-init state as init_state.
      init_state, verdict = warmstart.build_transfer_start(
        chi2fn=self.chi2fn,
        model_opts=specs["model_opts"],
        new_pgeom=self.pgeom,
        train_set=self.train_set,
        extra_names=self._transfer_extra_names,
        device=self.device)
      self.log(verdict)
      # transfer refine: keep a clone of the PRETRAINED base weights
      # before stage 2 unfreezes and drifts them. The saved artifact's
      # transfer_base group stays the pretrained base (provenance + the anchor
      # reference); the driver reads this clone for it. None on a frozen-only
      # transfer run.
      if self._transfer_refine is not None:
        pre = {}
        for k, v in self._transfer_base.model.state_dict().items():
          pre[k] = v.detach().clone()
        self._transfer_pretrained_base = pre

    # run_emulator (training.py): build the model / optimizer / scheduler
    # from the specs and the regime-aware loaders, train nepochs with a
    # per-epoch val pass, return the model (restored to its best frac>0.2
    # epoch) plus the histories.
    out = run_emulator(
      train_set=self.train_set,
      val_set=self.val_set,
      chi2fn=self.chi2fn,
      param_geometry=self.pgeom,
      nepochs=train_args["nepochs"],
      bs=train_args["bs"],
      # nested loss block {mode, berhu} (absent -> mode sqrt); run_emulator
      # validates it and resolves it per pass (a phase loss: full-replaces
      # it). The flat loss_mode / berhu keys are gone (resolve_phase_args
      # rejects them with the migration message).
      loss=train_args.get("loss"),
      # two-phase schedule (trunk-then-head, factored conv/TRF heads):
      # epochs of pure-trunk training before the trunk freezes and the
      # head learns the residual; 0 / absent = ordinary joint training.
      # The symmetric trunk: / head: blocks override each pass's
      # objective (the eight-key whitelist lr / scheduler / loss / trim /
      # focus / clip / rewind / ema; the head: block also accepts the
      # activation pin alias) over the shared top-level defaults; by the
      # handoff the trunk has absorbed most outliers, so the head may want
      # e.g. loss {mode: chi2} with no trim.
      trunk_epochs=train_args.get("trunk_epochs", 0),
      # freeze_trunk (None = absent = today's frozen default): false trains
      # trunk + head together in phase 2 (a joint fine-tune) instead of
      # freezing the trunk at the handoff. Needs trunk_epochs > 0.
      freeze_trunk=train_args.get("freeze_trunk"),
      trunk_opts=train_args.get("trunk"),
      head_opts=train_args.get("head"),
      # stability guards (both default off; the trunk: / head:
      # blocks can override either per phase): clip = per-step
      # gradient-norm ceiling (kills single-batch kicks from
      # monster outliers under a quadratic loss); rewind = on every
      # plateau lr cut reload the best weights + optimizer
      # snapshot, so an excursion into a bad basin costs at most
      # `patience` epochs instead of freezing the run there.
      clip=train_args.get("clip", 0.0),
      rewind=train_args.get("rewind", False),
      # optional weight ema (train_args.ema {horizon_epochs}); absent =
      # off = a byte-identical run. run_emulator validates the block and
      # couples the average to its snapshot/rewind (see training.py).
      ema=train_args.get("ema"),
      # fine-tune warm start: the transferred source weights (None on a plain
      # run). make_model loads them strict into the eager module before
      # compile; everything else in the loop stays fresh.
      init_state=init_state,
      # finetune.anchor: the decoupled L2-SP pull toward the transferred
      # weights (None on a plain / un-anchored run; the shared anchor facility,
      # training.build_anchor). The transfer refine stage's base anchor is a
      # separate spec.
      anchor=anchor_spec,
      # transfer refine: the resolved {epochs, base_lr_scale, anchor}
      # for the optional stage-2 joint pass (None = a frozen-only run). Its own
      # base-group anchor is built inside run_emulator.
      refine=self._transfer_refine,
      thresholds=self.thresholds,
      use_amp=self.use_amp,
      silent=silent_run,
      device=self.device,
      **specs)

    # run_emulator now returns resolved_train (save schema v2); store it on
    # the instance for save_emulator, and return the original 5-tuple so the
    # drivers' interface is unchanged (the added return is contained here).
    (self.model, self.train_losses, self.medians,
     self.means, self.fracs, self.resolved_train) = out

    # fine-tune warm start: record the resolved finetune block in the consumed
    # config (persist-resolved-values), so the saved run states its source with
    # the path and compile_mode materialized. run_emulator does not see the
    # finetune block, so it is added here, the one place that resolved both.
    if self._finetune is not None:
      ft = train_args.get("finetune", {})
      if "compile_mode" in ft:
        compile_mode = ft["compile_mode"]
      else:
        compile_mode = self._finetune.compile_mode
      self.resolved_train["finetune"] = {
        "from":         self._finetune.root,
        "compile_mode": compile_mode,
        "extra_names":  " ".join(self._finetune_extra_names),
      }
    # transfer: record the resolved transfer block (form + the materialized
    # space), so the saved run states what it composed (persist-resolved-values).
    if self._transfer_base is not None:
      self.resolved_train["transfer"] = {
        "from":        self._transfer_base.root,
        "form":        self._transfer_form,
        "space":       self._transfer_space,
        "extra_names": " ".join(self._transfer_extra_names),
      }
      # the resolved refine block (materialized), present only on a refined run.
      if self._transfer_refine is not None:
        self.resolved_train["transfer"]["refine"] = dict(self._transfer_refine)
    return (self.model, self.train_losses, self.medians,
            self.means, self.fracs)

  def run(self, n_train=None, train_args=None):
    """
    The full pipeline (the driver's body) in one call.

    Stage the train + val sources, build the geometry + chi2 from the
    training subset, train once. The artifacts (train_set, val_set, pgeom,
    geom, chi2fn, model) stay on the instance for diagnostics.

    Arguments:
      n_train    = absolute training-row count (default: the YAML
                   data["n_train"]), the N_train sweep knob.
      train_args = resolved train_args for this run (default:
                   self.train_args).

    Returns:
      (model, train_losses, medians, means, fracs).
    """
    self.stage_train(n_train=n_train)
    self.stage_val()
    self.build_geometry(train_set=self.train_set)
    return self.train(train_args=train_args)

  # --- a sweep metric ---
  def frac_above(self, threshold=0.2, source=None, bs=256):
    """
    Fraction of a source's points with delta-chi2 > threshold.

    Scores the trained model on a source (default the val set) with
    eval_source_chi2, the learning-curve / sweep metric (the number
    frac>thresholds[0] tracks per epoch, recomputed here).

    Arguments:
      threshold = the delta-chi2 cutoff (default 0.2, the goal).
      source    = source dict to score (default self.val_set).
      bs        = forward batch size for the scoring.

    Returns:
      the fraction over `threshold`, a float.
    """
    source = self.val_set if source is None else source
    # eval_source_chi2 (training.py): score every row of `source`, encode
    # params -> model -> per-row delta-chi2 against the encoded truth
    # (returns numpy params + dchi2, aligned row-for-row).
    _, dchi2 = eval_source_chi2(
      model=self.model,
      param_geometry=self.pgeom,
      chi2fn=self.chi2fn,
      source=source,
      device=self.device,
      bs=bs)
    return float((dchi2 > threshold).mean())
