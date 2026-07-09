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

from .data_staging import read_param_names, load_source, phys_cut_idx
from .geometries_parameter import ParamGeometry, AmplitudeFactorGeometry
from .losses.core import make_chi2
from .designs.plain import ResMLP, ResCNN, ResTRF
from .designs.ia import (TemplateMLP, TemplateResCNN,
                         TemplateResTRF)
from .losses.ia import (TemplateFactoredChi2, nla_coeffs,
                        tatt_coeffs)
from .activations import make_activation
from .designs.blocks import make_norm
from .training import (
  run_emulator, build_run_specs, pick_device, make_logger,
  default_train_args, eval_source_chi2, DEFAULT_COMPILE_MODE,
  validate_phase_block, _PHASE_BLOCK_KEYS,
  validate_loss, _loss_migration_message)


# (architecture, ia) -> model class. Two orthogonal YAML choices:
# train_args.model.name picks the architecture (resmlp = residual MLP;
# rescnn = + a theta-order 1D-CNN correction head; restrf = + a
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
# (build_shear_angle_map run on the data geometry, attaching the
# per-bin split the bin-token head needs).
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
_PCE_DEFAULTS = {
  "p_max": 4, "r_max": 2, "q": 0.5, "k_max": 40, "loo_max": 0.05,
  "max_terms": 30, "max_fail": 4,
}
_PCE_INT_KEYS = ("p_max", "r_max", "k_max", "max_terms", "max_fail")


def validate_pce(pce, rescale="none", ia=None):
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
    pce     = the parsed top-level "pce" block, or None (the block absent).
    rescale = the analytic-R rescale mode (a driver flag), for the
              exclusivity check; pce is exclusive with rescale != "none".
    ia      = the resolved model.ia design (None | "nla" | "tatt"), for
              the exclusivity check; pce is exclusive with an ia design.

  Returns:
    None when pce is None (NPCE off, byte-identical everywhere), else the
    validated, defaults-filled mapping (form + the seven fit knobs).

  Raises:
    TypeError if pce is not a mapping.
    ValueError on: an unknown key; a missing / non-{residual, ratio} form;
    a non-positive-int p_max / r_max / k_max / max_terms / max_fail; q
    outside (0, 1]; loo_max <= 0; or pce set together with rescale != "none"
    or a model.ia design.
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
                       see run_emulator. On a single-phase model (any
                       name: resmlp, including ia nla / tatt; no
                       set_train_phase, unlike rescnn / restrf) train()
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
                       n_blocks, n_mlp_blocks, shared_mlp, film,
                       gate_init, activation; name restrf only,
                       the tokens live at the natural bin width, so
                       there is no width knob, and the per-token MLP
                       layers run at that width too, n_mlp_blocks is
                       depth only). The head's activation ({type,
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
    # validate_param_cuts (below): the physical window cuts now live in
    # data.param_cuts; run this before the generic whitelist so a flat
    # cut key (the old layout) gets the migration message, not a bare
    # "unknown key".
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
    run executes (the banner-prints-consumed-view directive).

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
    # display the consumed view (the banner-prints-consumed-view directive):
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
    pc  = d["param_cuts"]     # the validated physical-window bounds
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
      omegabh2_hi=pc["omegabh2_hi"],
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
    pc  = d["param_cuts"]     # the validated physical-window bounds
    gen = torch.Generator().manual_seed(int(d["split_seed"]))
    # load_source (data_staging.py): same staging as stage_train, on the
    # val files; with_means=False (val borrows the training centers).
    self.val_set = load_source(
      dv_path=d["val_dv"],
      params_path=d["val_params"],
      names=self.names,
      omegabh2_hi=pc["omegabh2_hi"],
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
    pc = d["param_cuts"]     # the validated physical-window bounds
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
    # config validation first, before the cosmolike import below: a bad
    # combination should fail fast, and stays testable off-workstation.
    if getattr(self.model_cls, "factored", False) and self.rescale != "none":
      raise ValueError(
        f"model {self.model_name!r} does not compose with --rescale "
        "(the factored loss owns the target construction)")
    # lazy import: DataVectorGeometry.from_cosmolike pulls in cosmolike,
    # which lives only on the workstation, importing here keeps the module
    # importable for the config logic without cosmolike.
    from .geometries_output import DataVectorGeometry

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
      # ParamGeometry.from_covmat (geometries_parameter.py): eigendecompose
      # the parameter covmat so encode() centers, rotates, unit-scales the
      # params the model sees.
      self.pgeom = ParamGeometry.from_covmat(
        device=self.device,
        center=train_set["C_mean"],
        covmat_path=d["train_covmat"])

    # DataVectorGeometry.from_cosmolike (geometries_output.py): the output
    # geometry, read cosmolike's cov / mask / inverse-cov, eigendecompose
    # the kept (unmasked) block, so encode()/chi2 whiten + score the dv.
    self.geom = DataVectorGeometry.from_cosmolike(
      device=self.device,
      dv_center=train_set["dv_mean"],
      data_dir=d["cosmolike_data_dir"],
      dataset=d["cosmolike_dataset"],
      probe=self.probe)

    # bin-token heads (restrf; the needs_bins flag) split the dv per
    # tomographic bin: build_shear_angle_map (geometries_output.py)
    # attaches bin_sizes to the geometry, reading only the dataset ini
    # and the n(z) file, no cosmolike.
    if getattr(self.model_cls, "needs_bins", False):
      from .geometries_output import build_shear_angle_map
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
      # use, torch.from_numpy(...).float().to(device).
      idx = train_set["idx"]
      X_white = self.pgeom.encode(
        torch.from_numpy(np.asarray(train_set["C"][idx])).float().to(
          self.device))
      Y_white = self.geom.encode(
        torch.from_numpy(np.asarray(train_set["dv"][idx])).float().to(
          self.device))
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
    # component + freeze-trunk-joint-phase2): the canonical pin
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
    # keys (trunk_epochs / trunk: / head:), but a single-phase model (any
    # name: resmlp, including ia nla / tatt; no set_train_phase, unlike
    # rescnn / restrf) would die in run_emulator's capability guard. For
    # such a model demote the phase keys (drop head: / trunk_epochs, merge
    # trunk: into the top level) here, once, at the choke point every
    # driver funnels through; a two-phase model is an
    # exact no-op. It never mutates train_args (a sweep reuses it across
    # points); the notice is quiet-gated like the config banner.
    two_phase = hasattr(self.model_cls, "set_train_phase")
    train_args, phase_notice = resolve_phase_args(train_args=train_args,
                                                  two_phase=two_phase)
    if phase_notice is not None:
      self.log(phase_notice)
    specs = self.build_specs(train_args=train_args)
    # None -> the config/quiet default; a search driver forces silent.
    silent_run = (train_args.get("silent", False) or self.quiet
                  if silent is None else silent)

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
