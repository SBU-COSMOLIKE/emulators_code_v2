import numpy as np
import math, sys, traceback
from generator_core import (GeneratorCore, capture_native_output,
                            run_generator)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# The script below computes data vectors for cosmic shear (NLA, $w_0w_a$ model and Halofit.
#
#     mpirun -n 10 --report-bindings \
#       python external_modules/code/emulators/emultrfv2/compute_data_vectors/dataset_generator_lensing.py \
#         --root projects/roman_real/  \
#         --fileroot emulators/nla_cosmic_shear/ \
#         --nparams 10000 \
#         --yaml 'w0wa_takahashi_cs_cnn.yaml' \
#         --datavsfile 'w0wa_takahashi_dvs_train' \
#         --paramfile 'w0wa_takahashi_params_train' \
#         --failfile  'w0wa_takahashi_params_failed_train' \
#         --chain 0 \
#         --unif 0 \
#         --temp 64 \
#         --maxcorr 0.15 \
#         --freqchk 2000 \
#         --loadchk 0 \
#         --append 0 \
#         --boundary 1.0
#
#- The requested number of data vectors is given by the `--nparams` flag.
#
#- There are two possible samplings.
#  - The option `--unif 1` sets the sampling to follow a uniform distribution (respecting parameter boundaries set in the YAML file)
#  - The option `--unif 0` sets the sampling to follow a Gaussian distribution with the following options
#    -  The covariance matrix is set in the YAML file (keyword `params_covmat_file` inside the `train_args` block).
#       For example, our provided YAML selects the Fisher-based *w0wa_fisher_covmat.txt* covariance matrix
#    -  Temperature reduces the curvature of the likelihood (`cov = cov/T`) and is set by `--temp` flag
#    -  The correlations of the original covariance matrix are reduced to be less than `--maxcorr`.
#
#  Even on Uniform Sampling, the temperature parameter is needed to set hard boundary on parameters with Gaussian prior
#
#- For visualization purposes, setting `--chain 1` sets the script to generate the training parameters without computing the data vectors.
#
#- The output files are
#
#      # Distribution of training points ready to be plotted by GetDist
#      w0wa_takahashi_params_train_cs_64.1.txt
#      w0wa_takahashi_params_train_cs_64.covmat
#      w0wa_takahashi_params_train_cs_64.paramnames
#      w0wa_takahashi_params_train_cs_64.ranges
#
#      #Corresponding data vectors
#      w0wa_takahashi_dvs_train_cs_64.npy
#      # Training parameters in which the data vector computation failed (or not computed)
#      w0wa_takahashi_params_failed_train_cs_64.txt
#
#- The flags `--freqchk`, `--loadchk`, and `--append` are related to checkpoints.
#  - The option `--freqchk` sets the frequency at which the code saves checkpoints (chk).
#  - The options `--loadchk` and `--append` specify whether the code loads the parameters and data vectors from a chk.
#    In the two cases below, the code determines which remaining data vectors to compute based on the flags saved in the `--failfile` file.
#      - Case 1 (`--loadchk 1` and `--append 1`): the code loads params from the chk and appends `~nparams` models to it.
#      - Case 2 (`--loadchk 1` and `--append 0`): the code loads the params.
#
# The sampling, checkpointing, output-file, and MPI machinery all live in
# generator_core.py (shared with the other dataset generators, D-CM3-A);
# this file keeps only what is cosmolike-specific: the probe whitelist and
# the per-sample data-vector computation.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Class Definition
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class dataset(GeneratorCore):
  """
  Cosmolike data-vector generator: one likelihood.get_datavector call
  per sample; the data vector is a single flat vector, so the core's
  default single-2D-array store is used unchanged.
  """
  VALID_PROBES = ("cs", "ggl", "gc")
  EXTRA_TRAIN_KEYS = ()

  def _compute_dvs_from_sample(self, sample):
    # Define fortran errors we want to capture ---------------------------------
    camb_error_keywords = {"ERROR", "error", "Did not converge"}

    # Compute data vector (within using cobaya API) ----------------------------
    idx = self.reorder_idx_from_ord_to_yaml()
    param = dict(self.model.parameterization.to_input(
        sampled_params_values=dict(zip(self.names, sample[idx])))
    )
    self.model.provider.set_current_input_params(param)

    # Check prior before attempting computation --------------------------------
    if math.isinf(self.model.prior.logp(sample[idx])):
      raise RuntimeError(f"Prior is -inf (this should not happen). "
                         f"Values: {dict(zip(self.sampled_params, sample))}")

    # Compute data vector (within using cobaya API) ----------------------------
    likelihood = self.model.likelihood[list(self.model.likelihood.keys())[0]]

    captured = 0 # variable that will hold terminal output
    with capture_native_output() as tmp:
      for (x, _), z in zip(self.model._component_order.items(),
                           self.model._params_of_dependencies):
        x.check_cache_and_compute(
            params_values_dict = dict({p: param[p] for p in x.input_params}),
            want_derived = self.derived,
            dependency_params = list(param.keys()),
            cached = True
        )
      tmp.seek(0)
      captured = tmp.read() # copy terminal output -----------------------------

    # check for CAMB errors in the terminal output -----------------------------
    if any(kw in captured for kw in camb_error_keywords):
      raise RuntimeError(f"CAMB Fortran error: {captured.strip()}")

    return np.array(likelihood.get_datavector(**param),
                    copy = True,
                    dtype = self.dtype)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  run_generator(dataset, prog='dataset_generator_lensing')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
