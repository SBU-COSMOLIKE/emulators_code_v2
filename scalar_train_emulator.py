#!/usr/bin/env python3
"""Train one scalar (derived-parameter) emulator from a YAML.

A scalar emulator maps cosmological parameters to a small set of named
derived parameters (H0, omegam, rdrag, ...) instead of a cosmolike data
vector. Inputs and outputs are both named columns of one parameter .txt
(the covmat header names in, the YAML data.outputs list out); there is no
data vector, no mask, and no cosmolike anywhere on this path. See
ai/notes/families-scalar-cmb.md and emulator/geometries/scalar.py.

PS: standardized = each output shifted to zero mean and scaled to unit
variance (the form the network predicts); dump = the on-disk parameter
.txt from a chain run, whose columns are named by a getdist .paramnames
sidecar; run tag = the identity string appended to output file names so
runs do not overwrite each other.
"""

#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
# python .../emulators_code_v2/scalar_train_emulator.py \
#   --root projects/lsst_y1/ \
#   --fileroot emulators/training_scripts/ \
#   --yaml scalar_train_emulator.yaml
#
#- Cocoa layout: export $ROOTDIR, then --root names the project folder under
#  it ($ROOTDIR/projects/lsst_y1) and --fileroot a subfolder holding this
#  emulator's YAML. The parameter files (train_params / val_params, their
#  covmat and .paramnames sidecars) and the run products live under
#  --root/chains; the YAML under --fileroot. The driver resolves every path,
#  so it runs from $ROOTDIR regardless of cwd. No cosmolike is used.
#
#- This script sits beside the emulator/ package, so `import emulator` needs
#  no sys.path edit; run it from $ROOTDIR.
#
#- `--root` (required): project folder under $ROOTDIR (e.g. projects/lsst_y1);
#  the parameter files resolve under --root/chains.
#- `--fileroot` (required): subfolder of --root holding this emulator's YAML.
#- `--yaml` (default test.yaml): config file under --fileroot, or an absolute
#  path used as-is. Two blocks:
#  - `data`: train_params / val_params (the parameter .txt files, each with a
#    getdist .paramnames sidecar), train_covmat (the input-parameter covmat,
#    its header the input names), the `outputs` list (the derived-parameter
#    names to emulate, columns of the same .txt), absolute sizes (n_train,
#    n_val), split_seed, ram_frac. No dv / cosmolike keys (a scalar run
#    rejects them). data.param_cuts is optional here (a scalar chain is
#    already the target distribution).
#  - `train_args`: the usual knobs (nepochs, bs, loss, ...) plus the model
#    block (name must be a trunk-only design, resmlp; the conv / trf heads
#    correct along an angular axis a scalar output does not have). Small
#    resmlp widths are plenty for a scalar map.
#
#- `--save` (default `emulator`): name root for the trained-emulator files,
#  written under --root/chains with the run tag appended:
#  <save>_scalar-<outputs>-<digest>.emul (weights) + .h5 (the input
#  ParamGeometry, the output ScalarGeometry, histories, resolved config).
#  The ordered output names are readable; the digest also distinguishes the
#  completed configuration, selected rows, and source artifact.
#  rebuild_emulator
#  (emulator/results.py) reconstructs the inference-ready model from the h5
#  alone (schema 3), and its info["scalar"] flag routes EmulatorPredictor
#  to the scalar branch.
#
#- `--activation` (optional): the ResBlock activation family (H / power /
#  multigate / gated_power); overrides the YAML train_args.model.activation,
#  default H.
#
#- `--diagnostic` (optional): the name root of a multipage diagnostics PDF,
#  written under --root/chains with the run tag appended (like --save), e.g.
#  `--diagnostic diagnostic` -> diagnostic_scalar-h0-omegam-<digest>.pdf.
#  Pages:
#  the shared chi2 diagnostics (training history, coverage vs training
#  sparsity, the local-linear data floor, the hard-direction regression,
#  the parameter triangle + PCA planes) plus the scalar family pages
#  (truth-vs-predicted scatter; residual histograms in physical AND
#  standardized units; residual vs each input parameter — the bias hunt).
#
#- `--quiet` (optional): suppress all stdout (driver prints, the per-source
#  line, the per-epoch log).
#
#- Fixed single-emulator choices (AdamW, ReduceLROnPlateau, the reported
#  delta-chi2 thresholds) are EmulatorExperiment defaults; the model is the
#  YAML's choice (train_args.model.name).
#-------------------------------------------------------------------------------

import argparse
import os

# This script sits beside the emulator/ package, so launching it by path makes
# its own directory sys.path[0] and `import emulator` resolves with no path
# manipulation. Run it from $ROOTDIR; emulator.cocoa reads $ROOTDIR for paths.

from emulator.cocoa import (
  add_cocoa_path_args, resolve_cocoa_config, cocoa_output)
from emulator.experiment import EmulatorExperiment
from emulator.results import executed_composition, save_emulator
from emulator.warmstart import finetune_provenance_attrs


def run_tag(cfg, exp):
  """
  The run's identity tag for output filenames.

  <model>_ntrain<N>: the resolved model name and the N_train actually
  staged (cfg is retained for interface symmetry with the cosmic-shear
  driver's run_tag). Appended to the --diagnostic and --save name roots
  so runs do not overwrite each other and a file says what produced it.

  Arguments:
    cfg = the resolved config mapping (unused; interface symmetry).
    exp = the staged EmulatorExperiment (reads exp.arch + exp.train_set).

  Returns:
    the tag string, e.g. "resmlp_ntrain50000".
  """
  del cfg
  tags = [str(exp.arch or "resmlp").lower()]
  tags.append(f"ntrain{exp.train_set['idx'].shape[0]}")
  return "_".join(tags)


def main():
  """
  Train one scalar emulator end to end and save it.

  Resolves the cocoa paths and YAML, builds the EmulatorExperiment on the
  scalar path (data.outputs present), prints the startup banner, runs the
  full pipeline (stage the train + val sources, build the input
  ParamGeometry + output ScalarGeometry, train once), reports the best
  epoch, and saves the two artifact files (<save>_<tag>.emul weights +
  .h5 record). With --diagnostic, also writes the multipage PDF: the
  shared chi2 pages (they consume only params + per-sample chi2, so
  they apply to every family) plus the scalar truth/residual pages.
  """
  parser = argparse.ArgumentParser(prog="scalar_train_emulator")
  # --root / --fileroot / --yaml: the cocoa project layout.
  add_cocoa_path_args(parser)
  parser.add_argument("--save",
                      dest="save",
                      help="name root for the trained-emulator files, "
                           "written under --root/chains with the run tag "
                           "appended: <save>_scalar-<outputs>-<digest>.emul "
                           "(the "
                           "weights) + .h5 (geometries, histories, config)",
                      type=str,
                      default="emulator")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation: 'H' (default), 'power', "
                           "'multigate', or 'gated_power'. Overrides the "
                           "YAML train_args.model.activation",
                      type=str,
                      choices=["H", "power", "multigate", "gated_power"],
                      default=None)
  parser.add_argument("--diagnostic",
                      dest="diagnostic",
                      help="if set, save a multipage diagnostics PDF "
                           "under --root/chains, named with the run tag "
                           "(diagnostic -> diagnostic_scalar-h0-"
                           "omegam-<digest>.pdf): the shared chi2 pages "
                           "plus the scalar truth/residual pages",
                      type=str,
                      default=None)
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (driver prints, the "
                           "per-source line, the per-epoch log)",
                      action="store_true")
  # strict parse: reject a misspelled flag instead of silently ignoring it.
  args = parser.parse_args()

  # resolve_cocoa_config (cocoa.py): resolve the cocoa layout, load the YAML,
  # and rewrite its data paths to absolute, so the run does not depend on the
  # launch directory.
  cfg, _, chains = resolve_cocoa_config(args)

  # All setup lives in EmulatorExperiment; the scalar path is selected by the
  # presence of data.outputs (from_config validates the exclusivity). This
  # driver passes only what it varies (activation, quiet); no rescale (a
  # scalar run has no analytic rescaling), no probe, no cosmolike.
  exp = EmulatorExperiment.from_config(cfg,
                                       activation=args.activation,
                                       quiet=args.quiet)
  log = exp.log
  # print_design (experiment.py): the startup banner (the scalar summary
  # line, the resolved model block, and every train_args sub-block), so a
  # stale YAML is caught here and not one whole training later.
  exp.print_design()
  log("loading sources:")
  (model, train_losses, medians, means, fracs) = exp.run()

  # run_emulator already restored the best-frac>0.2 epoch; report which one.
  # fracs[i][0] is frac>0.2 at epoch i+1, median the tiebreaker (loop's rule).
  best = min(range(len(fracs)),
             key=lambda i: (fracs[i][0].item(), medians[i]))
  log(f"best epoch {best + 1}: "
      f"frac>0.2 {fracs[best][0].item():.4f}  "
      f"median {medians[best]:.4f}")

  # Persist the trained emulator. cocoa_output (cocoa.py) joins the chains/
  # folder to the name root; save_emulator (results.py) writes the two files:
  # <save>_<tag>.emul = the best-epoch weights (torch state_dict, cpu);
  # <save>_<tag>.h5   = the input ParamGeometry + output ScalarGeometry (both
  # from_state-ready), the per-epoch histories, and the full resolved config.
  identity_tag = run_tag(cfg, exp)
  save_root = cocoa_output(chains, f"{args.save}_{identity_tag}")
  # run-identity root attrs (no train_dv / val_dv: a scalar run reads only
  # parameter .txt files, so it records their basenames and its outputs).
  # rescale is recorded as the resolved "none": a scalar run has no analytic
  # rescale (that is a cosmolike data-vector concept), but the shared fine-tune
  # loader (warmstart.load_source) requires the rescale fact of every source it
  # admits. Omitting it made the scalar driver's own artifact unusable as its
  # supported fine-tune source (load_source refused "records no rescale"); the
  # value is recorded explicitly, never left to be inferred.
  attrs = {"model":        str(exp.arch or "resmlp").lower(),
           "activation":   exp.activation,
           "rescale":      "none",
           "n_train":      int(exp.train_set["idx"].shape[0]),
           "n_val":        int(exp.val_set["idx"].shape[0]),
           "best_epoch":   best + 1,
           "best_frac02":  fracs[best][0].item(),
           "best_median":  float(medians[best]),
           "device":       str(exp.device),
           "outputs":      " ".join(exp.outputs),
           "train_params": os.path.basename(cfg["data"]["train_params"]),
           "val_params":   os.path.basename(cfg["data"]["val_params"])}
  # Match every other training family: a fine-tuned scalar artifact must
  # identify the source artifact and its ordered extra parameter names.
  attrs.update(
    finetune_provenance_attrs(
      source=exp._finetune,
      extra_names=exp._finetune_extra_names))
  pce = exp.chi2fn.pce if exp.pce_opts is not None else None
  composition_mode, transfer_refined = executed_composition(
    pce=pce, transfer_base=None)
  emul_path, h5_path = save_emulator(
    path_root=save_root,
    model=model,
    param_geometry=exp.pgeom,
    geometry=exp.geom,
    config=cfg,
    histories={"train_losses": train_losses,
               "val_medians":  medians,
               "val_means":    means,
               "val_fracs":    fracs,
               "thresholds":   exp.thresholds},
    train_args=exp.train_args,
    # the NPCE base rides every family (the 2026-07-12 ruling); a scalar
    # run still has no transfer base (a recorded ruling).
    pce=pce,
    pce_form=(exp.pce_opts["form"] if exp.pce_opts is not None else None),
    # The resolved recipes (consumed view), so the saved run
    # rebuilds bit-exactly even if code defaults later drift.
    resolved_train=exp.resolved_train,
    resolved_model=exp.resolved_model,
    transfer_base=None,
    composition_mode=composition_mode,
    transfer_refined=transfer_refined,
    resolved_pce=(dict(exp.pce_opts)
                  if exp.pce_opts is not None else None),
    resolved_transfer=None,
    resolved_rescale=exp.rescale,
    # The generator's required scientific record, carried here verbatim from
    # the staged training source. Indexing is intentional: staging cannot
    # produce a train set without this record, and a missing key is a broken
    # production error rather than permission to save an older format.
    facts_yaml=exp.train_set["facts_yaml"],
    attrs=attrs)
  log(f"saved emulator -> {emul_path}")
  log(f"saved run record -> {h5_path}")

  if args.diagnostic is not None:
    # --diagnostic is a name root: the run tag is appended so runs do not
    # overwrite each other (the cosmic-shear driver's convention).
    stem, ext = os.path.splitext(args.diagnostic)
    diag_name = f"{stem}_{identity_tag}{ext or '.pdf'}"
    diag_path = cocoa_output(chains, diag_name)
    # headless output: pick a non-interactive matplotlib backend before
    # pyplot is imported (emulator.plotting imports it at load).
    os.environ.setdefault("MPLBACKEND", "Agg")
    from emulator.diagnostics import (
      coverage_diagnostic, local_linear_floor,
      hard_direction_regression, scalar_output_diagnostic)
    from emulator.plotting import plot_diagnostics
    # the shared chi2 pages are family-generic (they consume params +
    # per-sample chi2 only); the scalar loss is a plain chi2, so the
    # local-linear floor applies too.
    cov = coverage_diagnostic(model=model,
                              param_geometry=exp.pgeom,
                              chi2fn=exp.chi2fn,
                              train_set=exp.train_set,
                              val_set=exp.val_set,
                              device=exp.device)
    log(f"coverage: spearman(knn_dist, log dchi2) "
        f"{cov['spearman']:+.3f}")
    hd = hard_direction_regression(model=model,
                                   param_geometry=exp.pgeom,
                                   chi2fn=exp.chi2fn,
                                   val_set=exp.val_set,
                                   device=exp.device)
    log(f"hardness: joint log-linear R2 {hd['r2']:.3f}")
    floor = local_linear_floor(model=model,
                               param_geometry=exp.pgeom,
                               chi2fn=exp.chi2fn,
                               train_set=exp.train_set,
                               val_set=exp.val_set,
                               device=exp.device)
    log(f"floor: f_model {floor['f_model']:.3f}  "
        f"f_floor {floor['f_floor']:.3f}")
    # the scalar family pages: truth-vs-predicted, residual
    # histograms in physical + standardized units, residual vs input.
    sc = scalar_output_diagnostic(model=model,
                                  param_geometry=exp.pgeom,
                                  chi2fn=exp.chi2fn,
                                  val_set=exp.val_set,
                                  device=exp.device)
    plot_diagnostics(train_losses=train_losses,
                     medians=medians,
                     means=means,
                     fracs=fracs,
                     thresholds=exp.thresholds,
                     coverage=cov,
                     floor=floor,
                     hard_dir=hd,
                     val_set=exp.val_set,
                     names=exp.names,
                     cuts=cfg["data"].get("param_cuts", {}),
                     scalar=sc,
                     savepath=diag_path)
    log(f"saved diagnostics -> {diag_path}")


if __name__ == "__main__":
  main()
