#!/usr/bin/env python3
"""Compute the CMB power-spectrum covariance (Motloch & Hu 1709.03599).

This is the CMB covariance script: the CMB training loss needs a
covariance the way the lensing path gets one from cosmolike, and here
it must be COMPUTED. The model is eqs 1-7 of Motloch & Hu
(user-supplied paper, read 2026-07-10):

  Cov = G + N                                                    (eq 2)

  G^{XY,WZ}_{ll'} = delta_{ll'}/(2l+1) *
                    [C^XW_exp C^YZ_exp + C^XZ_exp C^YW_exp]      (eq 3)
  C^XY_exp,l      = C^XY_l + N^XY_l                              (eq 4)
  N^XY_l          = Delta_XY^2 exp(l(l+1) theta_FWHM^2 / 8 ln2)  (eq 1)

  N = N^(phi) + N^(E)                                            (eq 5)
  N^(phi)^{XY,WZ}_{ll'} = sum_L dC^XY_l/dC^phiphi_L *
                          Cov^phiphi_LL * dC^WZ_l'/dC^phiphi_L   (eq 6)
  N^(E)  (unlensed-EE sample variance into BB; eq 7)  — V1 records it
         and skips it: no BB emulator is planned, and eq 7 feeds only
         Cov^{XY,BB}.

The GAUSSIAN part is always computed (per-spectrum diagonals AND the
l-diagonal cross-spectrum blocks). The NON-GAUSSIAN lens-induced part
(eq 6) is behind a flag, OFF by default ("we first test with Gaussian
terms" — the user); when on, the power-spectrum derivatives are taken
by a 5-POINT STENCIL in the band amplitude (upgrading the paper's
two-point central difference), at SEVERAL step sizes, with the
convergence across steps reported per band and a loud failure when it
is not met — "getting the convergence of the 5-stencil rule with
respect to step size is always tricky" (the user), so the step study
is a first-class output.

CAMB runs WITHIN COBAYA on the high-accuracy settings the user fixed
(the yaml's theory block; see the example at the bottom of this
docstring). The cosmology is ALWAYS a fiducial flat LCDM: the params
block must fix every parameter to a value, and only LCDM parameter
names are admitted (loud otherwise). Derivatives re-lens the FIXED
unlensed spectra with a perturbed lensing potential through the CAMB
results object (provider.get_CAMBdata() ->
get_lensed_cls_with_spectrum), so the Boltzmann solve runs ONCE.

The OUTPUT is the interface the training stack consumes (the ruling
in notes/families-scalar-cmb.md): one .npz holding

  ell                    (n_ell,)  l = 2..lmax
  sigma_tt/te/ee/pp      (n_ell,)  sqrt of the Gaussian diagonal —
                                   what CmbDiagonalGeometry uses as its
                                   whitening scale (ALWAYS present)
  gauss_tt_te, gauss_tt_ee, gauss_te_ee
                         (n_ell,)  the l-diagonal Gaussian
                                   cross-spectrum covariances (eq 3
                                   off-pair blocks), for a future joint
                                   likelihood; not read by training V1
  cov_tt/te/ee           (n_ell, n_ell)  the DENSE per-spectrum block
                                   G + N^(phi); present ONLY when the
                                   non-Gaussian flag was on
  cov_tt_te, cov_tt_ee, cov_te_ee
                         (n_ell, n_ell)  the DENSE cross-spectrum
                                   blocks (eq 6 off-pair + the eq 3
                                   l-diagonal on their diagonals);
                                   present ONLY when the non-Gaussian
                                   flag was on. Together with the
                                   per-spectrum blocks these tile the
                                   full (3 n_ell, 3 n_ell) TT/TE/EE
                                   covariance a joint likelihood or a
                                   dense whitening (the planned
                                   dense-covariance training; see
                                   notes/families-scalar-cmb.md) would
                                   consume.
  provenance             json string: the fiducial parameters, noise,
                         beam, fsky, the NG flag, the stencil step
                         study, and the exact camb extra_args —
                         resolved values persisted, so the consumer
                         re-derives nothing.

PS: C_ell here is the raw power spectrum (muK^2 for T/E;
dimensionless C^phiphi for the lensing potential), never the
l(l+1)/2pi-scaled form — CAMB is asked for raw Cl and the noise
formula is in the same units. "Band" = one multipole L of the lensing
potential whose amplitude is perturbed to measure dC_l/dC^phiphi_L
(band width > 1 trades exactness for speed; the width is a knob and
is persisted). fsky rescales every covariance by 1/fsky (the standard
mode-counting approximation); the default is 1 (full sky) and it is
always recorded, never silent.

Example (run from $ROOTDIR; the YAML holds theory + params + cov_args):

    python external_modules/code/emulators_code_v2/compute_data_vectors/\
compute_cmb_covariance.py \
      --root projects/cmb/ --fileroot emulators/ \
      --yaml cmb_covariance_lcdm.yaml --output cmb_cov_s4

The YAML's three blocks:

```yaml
theory:
  camb:
    path: ./external_modules/code/CAMB
    extra_args:
      halofit_version: takahashi
      lmax: 7000
      kmax: 10
      k_per_logint: 130
      AccuracyBoost: 1.5
      lAccuracyBoost: 1.2
      lens_margin: 2050
      lens_k_eta_reference: 36000.0
      nonlinear: NonLinear_both
      recombination_model: CosmoRec
      Accuracy.AccurateBB: True
      min_l_logl_sampling: 6000
      DoLateRadTruncation: False

params:          # fiducial flat LCDM, every parameter FIXED
  As:       2.1e-9
  ns:       0.9660
  H0:       67.36
  omegabh2: 0.02237
  omegach2: 0.1200
  tau:      0.0544
  mnu:      0.06

cov_args:
  lmax: 5000             # covariance multipole range 2..lmax
  fsky: 1.0
  noise:
    delta_tt:  1.0       # muK-arcmin  (eq 1's Delta_TT)
    delta_ee:  1.4       # muK-arcmin  (Delta_EE)
    delta_te:  0.0       # muK-arcmin  (Delta_TE; 0 = uncorrelated)
    beam_fwhm: 1.0       # arcmin      (theta_FWHM)
  nongaussian:
    enabled: false       # eq 6 off by default (Gaussian-first)
    lens_lmax: 3000      # L range of the phi sum
    band_width: 20       # multipoles per perturbed band (1 = exact)
    step_fracs:          # fractional band amplitudes for the stencil
      - 0.01
      - 0.02
      - 0.04
    converge_rtol: 0.05  # max relative spread across steps, per band
```
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# this script sits beside dataset_generator_lensing.py; the emulator
# package is one level up, but nothing here needs it: the covariance is
# a standalone product consumed BY the training stack through the .npz.

C_ARCMIN_TO_RAD = math.pi / (180.0 * 60.0)

# the fiducial cosmology must be plain flat LCDM (the user directive:
# "the covariance is ALWAYS computed on a LCDM cosmology"). Only these
# parameter names may appear in the params block; every one must be a
# FIXED value. Extension-model names are rejected loudly.
LCDM_ALLOWED = ("As", "logA", "ns", "H0", "thetastar", "cosmomc_theta",
                "omegabh2", "omegach2", "tau", "mnu", "omk", "w", "wa")
LCDM_FIXED_ONLY = {"omk": 0.0,
                   "w": -1.0,
                   "wa": 0.0}


def noise_spectrum(ell, delta_arcmin, beam_fwhm_arcmin):
  """Instrumental noise power N_l, eq 1, in the C_ell units (muK^2).

  N^XY_l = Delta_XY^2 * exp(l(l+1) theta_FWHM^2 / (8 ln 2)), with
  Delta in muK-radian and theta_FWHM in radians; the arguments arrive
  in the arcmin conventions of the YAML and are converted here.

  Arguments:
    ell               = (n_ell,) multipole grid.
    delta_arcmin      = instrumental noise Delta_XY in muK-arcmin
                        (0 = no noise for this pair, e.g. TE).
    beam_fwhm_arcmin  = beam FWHM theta in arcmin.

  Returns:
    (n_ell,) noise power N_l in muK^2 (all zeros when delta is 0).
  """
  l = np.asarray(ell, dtype="float64")
  delta = float(delta_arcmin) * C_ARCMIN_TO_RAD
  theta = float(beam_fwhm_arcmin) * C_ARCMIN_TO_RAD
  return (delta ** 2) * np.exp(l * (l + 1.0) * theta ** 2 / (8.0 * math.log(2.0)))


def gaussian_blocks(ell, cls, noise, fsky):
  """The full Gaussian covariance set, eq 3 with eq-4 noise.

  Every diagonal and l-diagonal cross block over the spectra
  {TT, TE, EE} plus the phiphi cosmic variance. Pure numpy on already-
  computed C_ell, so this function is Mac-verifiable against the
  closed forms (the gate leg).

      cls  {tt, te, ee, pp}      raw fiducial C_ell over `ell`
        │   + noise (eq 4)       C_exp = C + N   (N_te from delta_te)
        ▼
      eq 3 per (XY, WZ) pair     var_tt  = 2 C^TT_exp^2        /(2l+1)
        │                        var_te  = (C^TT_exp C^EE_exp
        │                                   + C^TE_exp^2)      /(2l+1)
        │                        var_ee  = 2 C^EE_exp^2        /(2l+1)
        │                        var_pp  = 2 C^pp^2            /(2l+1)
        │                        tt_te   = 2 C^TT_exp C^TE_exp /(2l+1)
        │                        tt_ee   = 2 C^TE_exp^2        /(2l+1)
        │                        te_ee   = 2 C^EE_exp C^TE_exp /(2l+1)
        ▼
      / fsky                     the mode-counting sky-cut rescale

  (legend: C^XY_exp = C^XY + N^XY per eq 4; every line is eq 3 with
  the right (XY, WZ) index pattern, e.g. tt_te is X=Y=T, W=T, Z=E:
  C^TW C^YZ + C^XZ C^YW = 2 C^TT_exp C^TE_exp. The phiphi noise N0 is
  a recorded future knob — V1 phiphi is cosmic variance only.)

  Arguments:
    ell   = (n_ell,) multipole grid.
    cls   = dict with "tt", "te", "ee", "pp": raw fiducial C_ell.
    noise = dict with "tt", "te", "ee": the N_l arrays (eq 1); "pp"
            absent (V1 cosmic variance only).
    fsky  = observed sky fraction; every block divides by it.

  Returns:
    dict of (n_ell,) arrays: var_tt, var_te, var_ee, var_pp,
    gauss_tt_te, gauss_tt_ee, gauss_te_ee.
  """
  l = np.asarray(ell, dtype="float64")
  norm = 1.0 / ((2.0 * l + 1.0) * float(fsky))
  tt = np.asarray(cls["tt"], dtype="float64") + noise["tt"]
  te = np.asarray(cls["te"], dtype="float64") + noise["te"]
  ee = np.asarray(cls["ee"], dtype="float64") + noise["ee"]
  pp = np.asarray(cls["pp"], dtype="float64")
  out = {}
  out["var_tt"] = norm * 2.0 * tt * tt
  out["var_te"] = norm * (tt * ee + te * te)
  out["var_ee"] = norm * 2.0 * ee * ee
  out["var_pp"] = norm * 2.0 * pp * pp
  out["gauss_tt_te"] = norm * 2.0 * tt * te
  out["gauss_tt_ee"] = norm * 2.0 * te * te
  out["gauss_te_ee"] = norm * 2.0 * ee * te
  return out


def stencil_derivative(f_m2, f_m1, f_p1, f_p2, h):
  """First derivative by the 5-point central stencil.

    f'(0) ~ [ f(-2h) - 8 f(-h) + 8 f(+h) - f(+2h) ] / (12 h)

  (the center point drops out of the first-derivative stencil; the
  five points are -2h, -h, 0, +h, +2h). Error is O(h^4), which is why
  the convergence study across several h is meaningful.

  Arguments:
    f_m2, f_m1, f_p1, f_p2 = f evaluated at -2h, -h, +h, +2h; arrays
                             of one shape.
    h                      = the step (a scalar, same units as the
                             perturbation amplitude).

  Returns:
    the derivative estimate, same shape as the inputs.
  """
  return (f_m2 - 8.0 * f_m1 + 8.0 * f_p1 - f_p2) / (12.0 * h)


def band_windows(lmin, lmax, band_width):
  """Contiguous multipole bands [start, stop] covering lmin..lmax.

  Arguments:
    lmin, lmax  = the phi-sum multipole range (inclusive).
    band_width  = multipoles per band (1 = per-L, exact eq 6; wider
                  bands trade exactness for fewer re-lensings and are
                  persisted in the provenance).

  Returns:
    a list of (start, stop) inclusive pairs.
  """
  bands = []
  start = int(lmin)
  while start <= int(lmax):
    stop = min(int(lmax), start + int(band_width) - 1)
    bands.append((start, stop))
    start = stop + 1
  return bands


def assemble_lensing_blocks(deriv, w):
  """Eq 6 assembly for EVERY spectrum pair, from the band derivatives.

  N^(phi)^{XY,WZ}_{ll'} = sum_b  dC^XY_l/dA_b * w_b * dC^WZ_l'/dA_b

  where A_b is the FRACTIONAL amplitude of C^phiphi inside band b (the
  band is perturbed by clpp *= 1 + eps) and w_b the eq-6 contraction
  weight for that coordinate: the Gaussian variance of the fractional
  amplitude, which at band width 1 is exactly 2/((2L+1) fsky) and for a
  wider band is the smooth-response projection the caller builds (see
  nongaussian_blocks and notes/families-scalar-cmb.md). A pure matrix
  product on already-computed derivatives, so this function is
  Mac-verifiable against the closed form (the probe leg): the
  same-spectrum blocks are symmetric by construction, and the cross
  blocks obey cov_xy_wz == cov_wz_xy^T.

  Arguments:
    deriv = dict tt/te/ee of (n_bands, n_ell) band-derivative
            matrices dC^s_l/dA_b (the smallest-step stencil
            estimates).
    w     = (n_bands,) per-band eq-6 contraction weights (the
            fractional-amplitude variance; nongaussian_blocks builds
            them).

  Returns:
    dict of dense (n_ell, n_ell) arrays: cov_tt, cov_te, cov_ee (the
    same-spectrum blocks) and cov_tt_te, cov_tt_ee, cov_te_ee (the
    cross-spectrum blocks, rows = the first spectrum's l, columns =
    the second's l').
  """
  pairs = (("tt", "tt"),
           ("te", "te"),
           ("ee", "ee"),
           ("tt", "te"),
           ("tt", "ee"),
           ("te", "ee"))
  out = {}
  for a, b in pairs:
    if a == b:
      key = "cov_" + a
    else:
      key = "cov_" + a + "_" + b
    out[key] = (deriv[a] * w[:, None]).T @ deriv[b]
  return out


def validate_lcdm_params(params):
  """Loud check: the params block is plain flat LCDM, every value fixed.

  The covariance is only ever computed on a fiducial LCDM cosmology
  (user directive). Three rules: (1) every entry must be a plain
  number (a FIXED value — no priors, no derived lambdas: this script
  evaluates one cosmology); (2) only LCDM_ALLOWED names may appear;
  (3) the geometry/dark-energy names, if present, must sit at their
  LCDM values (omk 0, w -1, wa 0).

  Arguments:
    params = the YAML params mapping.

  Raises:
    ValueError naming the offending key(s) and the rule broken.
  """
  bad_type, bad_name, bad_value = [], [], []
  for name, value in params.items():
    if not isinstance(value, (int, float)):
      bad_type.append(name)
      continue
    if name not in LCDM_ALLOWED:
      bad_name.append(name)
      continue
    if name in LCDM_FIXED_ONLY:
      if abs(float(value) - LCDM_FIXED_ONLY[name]) > 1e-12:
        bad_value.append(name)
  problems = []
  if bad_type:
    problems.append(
      "non-fixed entries " + repr(sorted(bad_type)) + " (every parameter "
      "must be a plain number; this script evaluates ONE fiducial "
      "cosmology, never samples)")
  if bad_name:
    problems.append(
      "non-LCDM parameter name(s) " + repr(sorted(bad_name)) + " (allowed: "
      + repr(list(LCDM_ALLOWED)) + ")")
  if bad_value:
    problems.append(
      "parameter(s) " + repr(sorted(bad_value)) + " not at their LCDM "
      "value (omk 0, w -1, wa 0)")
  if problems:
    raise ValueError(
      "the covariance is ALWAYS computed on a fiducial flat LCDM "
      "cosmology; the params block breaks that: " + "; ".join(problems))


def fiducial_spectra(info, lmax):
  """One high-accuracy CAMB evaluation through cobaya; raw Cl + CAMBdata.

  Builds the cobaya model from the YAML info (the user's fixed
  high-accuracy theory block), attaches a Cl requirement, evaluates the
  single fiducial point, and returns the raw spectra and the live CAMB
  results object (the derivative machinery re-lenses through it).

  Arguments:
    info = the loaded YAML mapping (theory + params blocks used here).
    lmax = the covariance multipole range's top (Cl requested to lmax).

  Returns:
    (cls, cambdata):
      cls      = dict of raw C_ell arrays over l = 0..lmax: tt, te, ee,
                 pp (muK^2 for T/E; raw C^phiphi for pp).
      cambdata = the CAMBdata results object from the provider.
  """
  # imported here, not at module top: the Mac has no cobaya; every
  # pure-math piece above stays importable and probe-able without it.
  from cobaya.model import get_model

  model_info = {}
  model_info["theory"] = info["theory"]
  model_info["params"] = info["params"]
  # a likelihood must exist for cobaya to build a model; the one-liner
  # requests nothing and scores 0, exactly the dummy-likelihood trick
  # the dataset generator uses.
  model_info["likelihood"] = {"one": {"external": "lambda: 0.0"}}
  model = get_model(model_info)
  model.add_requirements({
    "Cl": {"tt": int(lmax),
           "te": int(lmax),
           "ee": int(lmax),
           "pp": int(lmax)},
    "CAMBdata": None,
  })
  # evaluate the single fiducial point (all params fixed, so the
  # sampled-point dict is empty).
  model.logposterior({})
  cl = model.provider.get_Cl(ell_factor=False, units="muK2")
  cambdata = model.provider.get_CAMBdata()
  cls = {}
  cls["tt"] = np.asarray(cl["tt"], dtype="float64")
  cls["te"] = np.asarray(cl["te"], dtype="float64")
  cls["ee"] = np.asarray(cl["ee"], dtype="float64")
  cls["pp"] = np.asarray(cl["pp"], dtype="float64")
  return cls, cambdata


def lensed_cls_with_clpp(cambdata, clpp, lmax):
  """Re-lens the fixed unlensed spectra with a modified C^phiphi.

  The eq-6 derivative is with respect to the LENSING POTENTIAL spectrum
  at fixed unlensed CMB, so the Boltzmann solve never reruns: CAMB's
  results object re-lenses with any supplied potential
  (get_lensed_cls_with_spectrum), and cobaya provided that object
  (provider.get_CAMBdata()) — CAMB stays "within cobaya on high
  settings" exactly as directed.

  Arguments:
    cambdata = the CAMBdata results object (fiducial, high accuracy).
    clpp     = the [L(L+1)]^2/2pi-convention lensing array over
               L = 0..Params.max_l — CAMB refuses anything shorter.
               The caller takes it whole from get_lens_potential_cls
               and perturbs one band.
    lmax     = top multipole of the returned lensed spectra.

  Returns:
    dict tt/te/ee of raw muK^2 lensed C_ell over l = 0..lmax.
  """
  lensed = cambdata.get_lensed_cls_with_spectrum(clpp=clpp,
                                                 lmax=int(lmax),
                                                 CMB_unit="muK",
                                                 raw_cl=True)
  out = {}
  out["tt"] = np.asarray(lensed[:, 0], dtype="float64")
  out["ee"] = np.asarray(lensed[:, 1], dtype="float64")
  out["te"] = np.asarray(lensed[:, 3], dtype="float64")
  return out


def nongaussian_blocks(cambdata, cls, ell, ng_cfg, fsky, log):
  """The lens-induced non-Gaussian covariance N^(phi), eq 6.

  For each L-band, the lensing potential is scaled by (1 + eps) inside
  the band and the lensed TT/TE/EE recomputed by re-lensing; the
  5-point stencil over eps gives dC^XY_l/d(band amplitude), and eq 6
  assembles the dense blocks with the phi Gaussian variance
  Cov^phiphi_LL = 2 (C^phiphi_L)^2 / ((2L+1) fsky):

      for each band b, each step h in step_fracs:
        eps in {-2h, -h, +h, +2h}  ->  re-lens  ->  Cl^XY(eps)
        dCl^XY/dA_b = stencil(...) / 1          (A_b = the band scale)
      convergence: the h-to-h relative spread per band must sit below
      converge_rtol (loud otherwise); the kept derivative is the
      smallest-h estimate.
      N^(phi)XY,WZ_{ll'} = sum_b dCl^XY_l/dA_b * w_b * dCl^WZ_l'/dA_b
      with the eq-6 contraction weight
        w_b = [sum_{L in b} 2 C^phiphi_L^2/((2L+1) fsky)]
              / [sum_{L in b} C^phiphi_L]^2
      (band width 1 = 2/((2L+1) fsky), eq 6 verbatim; wider bands are
      the smooth-response projection, valid when dCl/dC^phiphi_L is
      nearly constant across the band; a band with sum C^phiphi_L = 0
      contributes nothing, w_b = 0, never a division).

  (legend: A_b = the FRACTIONAL amplitude of C^phiphi inside band b
  (clpp *= 1 + eps), so the stencil returns dCl/dA_b =
  sum_{L in b} C^phiphi_L dCl/dC^phiphi_L, which already carries one
  factor of C^phiphi; eq 6 in that coordinate contracts with the
  variance of the FRACTIONAL amplitude, w_b above, not with C^phiphi's
  own variance, so the C^phiphi_L^2 cancels at band width 1 and leaves
  2/((2L+1) fsky); eps = the stencil offsets of A_b around 0.
  Derivation in notes/families-scalar-cmb.md.)

  Arguments:
    cambdata = fiducial CAMBdata (re-lensing machine).
    cls      = fiducial raw spectra dict (pp used for the band scale
               and the phi variance).
    ell      = (n_ell,) the covariance grid l = 2..lmax.
    ng_cfg   = the cov_args.nongaussian mapping (enabled / lens_lmax /
               band_width / step_fracs / converge_rtol).
    fsky     = sky fraction (rescales the phi variance).
    log      = print-like callable for the step-study report.

  Returns:
    (blocks, study): blocks = dict of dense (n_ell, n_ell) arrays
    holding N^(phi) ONLY (the caller adds the Gaussian l-diagonals):
    cov_tt/cov_te/cov_ee (same-spectrum) and cov_tt_te/cov_tt_ee/
    cov_te_ee (the cross-spectrum off-pair blocks, eq 6 for X != W);
    study = the convergence record (per-band relative spreads and the
    step list) plus the derivative coordinate, the band-weight policy,
    and the per-band eq-6 weights, all for the provenance.
  """
  lmax = int(ell[-1])
  lens_lmax = int(ng_cfg["lens_lmax"])
  band_width = int(ng_cfg.get("band_width", 20))
  step_fracs = list(ng_cfg["step_fracs"])
  rtol = float(ng_cfg.get("converge_rtol", 0.05))
  if len(step_fracs) < 2:
    raise ValueError(
      "cov_args.nongaussian.step_fracs needs >= 2 step sizes: the "
      "convergence of the 5-point stencil vs step size is the point of "
      "the study, one step proves nothing.")

  # The phi Gaussian variance below needs the RAW C^phiphi_L over the
  # banded range; the re-lensing call needs CAMB's own scaled
  # convention at FULL length.
  pp_raw = np.zeros(lens_lmax + 1, dtype="float64")
  n_have = min(len(cls["pp"]), lens_lmax + 1)
  pp_raw[:n_have] = cls["pp"][:n_have]
  # get_lens_potential_cls(raw_cl=False) column 0 is
  # [L(L+1)]^2 C^phiphi_L / 2pi over L = 0..Params.max_l — the exact
  # array get_lensed_cls_with_spectrum demands ("clpp must go to at
  # least Params.max_l"; a shorter array raises — the 2026-07-12
  # first-execution red). Building it by hand truncated at lens_lmax
  # would also silently DELENS every L above lens_lmax at the
  # fiducial; only the band [b_lo, b_hi] may differ from fiducial.
  clpp_fid = np.asarray(
    cambdata.get_lens_potential_cls(raw_cl=False)[:, 0], dtype="float64")

  bands = band_windows(lmin=2, lmax=lens_lmax, band_width=band_width)
  spectra = ("tt", "te", "ee")
  n_ell = len(ell)
  # dCl/dA_b per spectrum, kept per band (smallest step's estimate).
  deriv = {}
  for s in spectra:
    deriv[s] = np.zeros((len(bands), n_ell), dtype="float64")
  spreads = np.zeros(len(bands), dtype="float64")

  ell_lo = int(ell[0])
  for b, (b_lo, b_hi) in enumerate(bands):
    # derivative estimates at every step size, then the convergence
    # check across them (relative spread of the stacked estimates).
    est = {}
    for s in spectra:
      est[s] = []
    for h in step_fracs:
      lensed = {}
      for eps_mult in (-2.0, -1.0, 1.0, 2.0):
        eps = eps_mult * h
        clpp = clpp_fid.copy()
        clpp[b_lo:b_hi + 1] *= (1.0 + eps)
        lensed[eps_mult] = lensed_cls_with_clpp(cambdata=cambdata,
                                                clpp=clpp, lmax=lmax)
      for s in spectra:
        d = stencil_derivative(f_m2=lensed[-2.0][s][ell_lo:lmax + 1],
                               f_m1=lensed[-1.0][s][ell_lo:lmax + 1],
                               f_p1=lensed[1.0][s][ell_lo:lmax + 1],
                               f_p2=lensed[2.0][s][ell_lo:lmax + 1],
                               h=h)
        est[s].append(d)
    # convergence across the steps: pooled over the three spectra, the
    # max relative spread against the smallest-step estimate (guarded
    # by the estimate's own scale so near-zero derivatives don't fake
    # a failure).
    worst = 0.0
    for s in spectra:
      stack = np.stack(est[s])                       # (n_steps, n_ell)
      ref = stack[0]
      scale = np.abs(ref).max() + 1e-30
      spread = np.abs(stack - ref).max() / scale
      if spread > worst:
        worst = spread
      deriv[s][b] = ref
    spreads[b] = worst
    if worst > rtol:
      raise RuntimeError(
        "5-point stencil NOT converged in band L = [" + str(b_lo) + ", "
        + str(b_hi) + "]: relative spread " + f"{worst:.3g}" + " across "
        "step_fracs " + repr(step_fracs) + " exceeds converge_rtol "
        + repr(rtol) + ". Adjust step_fracs (the study output shows the "
        "per-band spreads so far) — this failure is loud by design.")
    if b % 25 == 0:
      log("  band " + str(b + 1) + "/" + str(len(bands)) + " L=["
          + str(b_lo) + "," + str(b_hi) + "]  spread "
          + f"{worst:.2e}")

  # the eq-6 contraction weight per band. The band is perturbed by a
  # FRACTIONAL amplitude A_b (clpp *= 1 + eps), so the stencil already
  # returns dCl/dA_b, which carries one factor of C^phiphi inside the
  # band; eq 6 in that coordinate contracts with the Gaussian variance
  # of the FRACTIONAL amplitude, not of C^phiphi itself. At band width 1
  # that is exactly Var(A_L) = Var(C^phiphi_L)/C^phiphi_L^2 =
  # 2/((2L+1) fsky). For a wider band it is the smooth-response
  # projection
  #   w_b = [sum_L 2 C^phiphi_L^2/((2L+1) fsky)] / [sum_L C^phiphi_L]^2,
  # exact when dCl/dC^phiphi_L is nearly constant across the band. See
  # notes/families-scalar-cmb.md.
  L = np.arange(0, lens_lmax + 1, dtype="float64")
  var_pp_L = np.zeros(lens_lmax + 1, dtype="float64")
  var_pp_L[2:] = 2.0 * pp_raw[2:] ** 2 / ((2.0 * L[2:] + 1.0) * fsky)
  w = np.zeros(len(bands), dtype="float64")
  for b, (b_lo, b_hi) in enumerate(bands):
    band_var_sum = var_pp_L[b_lo:b_hi + 1].sum()
    band_cl_sum  = pp_raw[b_lo:b_hi + 1].sum()
    # an all-zero band carries no signal: its fractional derivative is
    # identically zero, so it contributes nothing. w_b = 0, never a
    # divide by zero.
    if band_cl_sum == 0.0:
      w[b] = 0.0
    else:
      w[b] = band_var_sum / (band_cl_sum ** 2)

  # eq 6 assembly for every spectrum pair: the same-spectrum blocks
  # (the per-spectrum training covariance) AND the cross-spectrum
  # blocks (the off-pair terms of the paper's full matrix — together
  # they tile the joint TT/TE/EE covariance the planned
  # dense-covariance training (notes/families-scalar-cmb.md) and any
  # joint likelihood would consume).
  blocks = assemble_lensing_blocks(deriv=deriv, w=w)
  # the band policy is a resolved provenance fact: exact eq 6 at band
  # width 1, the smooth-response projection when the band is wider.
  if band_width == 1:
    band_weight_policy = "exact eq 6"
  else:
    band_weight_policy = "smooth-response band projection"
  study = {"bands": [[int(a), int(b)] for a, b in bands],
           "step_fracs": step_fracs,
           "band_width": band_width,
           "per_band_relative_spread": spreads.tolist(),
           "converge_rtol": rtol,
           "derivative_coordinate": "fractional_band_amplitude",
           "band_weight_policy": band_weight_policy,
           "per_band_weight": w.tolist()}
  return blocks, study


def main():
  """Compute the covariance per the YAML and write the .npz interface."""
  parser = argparse.ArgumentParser(prog="compute_cmb_covariance")
  parser.add_argument("--root", dest="root", type=str, required=True,
                      help="project folder under $ROOTDIR "
                           "(e.g. projects/cmb/)")
  parser.add_argument("--fileroot", dest="fileroot", type=str,
                      required=True,
                      help="subfolder of --root holding the YAML")
  parser.add_argument("--yaml", dest="yaml", type=str, required=True,
                      help="covariance config: theory (the fixed "
                           "high-accuracy camb block) + params (fixed "
                           "fiducial LCDM) + cov_args")
  parser.add_argument("--output", dest="output", type=str,
                      required=True,
                      help="output name root; writes "
                           "<root>/chains/<output>.npz")
  args, _ = parser.parse_known_args()

  import yaml as pyyaml

  root_env = os.environ.get("ROOTDIR")
  if not root_env:
    raise RuntimeError("ROOTDIR environment variable is not set")
  root = root_env.rstrip("/") + "/" + args.root.strip("/")
  yaml_path = root + "/" + args.fileroot.strip("/") + "/" + args.yaml
  with open(yaml_path) as fh:
    info = pyyaml.safe_load(fh)
  for key in ("theory", "params", "cov_args"):
    if key not in info:
      raise KeyError("covariance YAML missing the required block "
                     + repr(key) + ": " + yaml_path)

  validate_lcdm_params(info["params"])

  cov = info["cov_args"]
  lmax = int(cov["lmax"])
  fsky = float(cov.get("fsky", 1.0))
  noise_cfg = cov["noise"]
  ng_cfg = cov.get("nongaussian", {"enabled": False})

  print("compute_cmb_covariance: fiducial LCDM evaluation (one CAMB "
        "solve, high accuracy)...")
  cls_full, cambdata = fiducial_spectra(info=info, lmax=lmax)

  ell = np.arange(2, lmax + 1, dtype="int64")
  cls = {}
  for s in ("tt", "te", "ee", "pp"):
    cls[s] = cls_full[s][2:lmax + 1]

  beam = float(noise_cfg["beam_fwhm"])
  noise = {}
  noise["tt"] = noise_spectrum(ell=ell, delta_arcmin=noise_cfg["delta_tt"],
                               beam_fwhm_arcmin=beam)
  noise["ee"] = noise_spectrum(ell=ell, delta_arcmin=noise_cfg["delta_ee"],
                               beam_fwhm_arcmin=beam)
  noise["te"] = noise_spectrum(ell=ell,
                               delta_arcmin=noise_cfg.get("delta_te", 0.0),
                               beam_fwhm_arcmin=beam)

  g = gaussian_blocks(ell=ell, cls=cls, noise=noise, fsky=fsky)

  out = {}
  out["ell"] = ell
  out["sigma_tt"] = np.sqrt(g["var_tt"])
  out["sigma_te"] = np.sqrt(g["var_te"])
  out["sigma_ee"] = np.sqrt(g["var_ee"])
  out["sigma_pp"] = np.sqrt(g["var_pp"])
  out["gauss_tt_te"] = g["gauss_tt_te"]
  out["gauss_tt_ee"] = g["gauss_tt_ee"]
  out["gauss_te_ee"] = g["gauss_te_ee"]
  # the fiducial spectra ride along: the geometry persists its
  # fiducial_cl from here, single-sourced with the covariance.
  out["cl_tt"] = cls["tt"]
  out["cl_te"] = cls["te"]
  out["cl_ee"] = cls["ee"]
  out["cl_pp"] = cls["pp"]

  study = None
  if bool(ng_cfg.get("enabled", False)):
    print("non-Gaussian lens-induced part (eq 6): 5-point stencil over "
          + str(len(ng_cfg["step_fracs"])) + " step sizes...")
    blocks, study = nongaussian_blocks(cambdata=cambdata, cls=cls_full,
                                       ell=ell, ng_cfg=ng_cfg,
                                       fsky=fsky, log=print)
    diag_idx = np.arange(len(ell))
    for s in ("tt", "te", "ee"):
      dense = blocks["cov_" + s]
      # the full covariance = Gaussian diagonal + N^(phi); persisted
      # dense only under the flag (the Gaussian-only file stays small).
      dense[diag_idx, diag_idx] += g["var_" + s]
      out["cov_" + s] = dense
    for pair in ("tt_te", "tt_ee", "te_ee"):
      dense = blocks["cov_" + pair]
      # the cross-spectrum blocks: the eq 3 Gaussian cross-covariance
      # is l-diagonal, so it lands on the dense block's diagonal; the
      # eq 6 lens-induced part fills the off-diagonals.
      dense[diag_idx, diag_idx] += g["gauss_" + pair]
      out["cov_" + pair] = dense

  provenance = {
    "paper": "Motloch & Hu 1709.03599 eqs 1-7 (N^(E) recorded, skipped)",
    "fiducial_params": info["params"],
    "camb_extra_args": info["theory"]["camb"].get("extra_args", {}),
    "lmax": lmax,
    "fsky": fsky,
    "noise": {"delta_tt": noise_cfg["delta_tt"],
              "delta_ee": noise_cfg["delta_ee"],
              "delta_te": noise_cfg.get("delta_te", 0.0),
              "beam_fwhm_arcmin": beam},
    "nongaussian_enabled": bool(ng_cfg.get("enabled", False)),
    "stencil_study": study,
    "pp_noise_n0": "not included (V1 cosmic variance only; N0 is a "
                   "recorded future knob)",
  }
  out["provenance"] = json.dumps(provenance)

  os.makedirs(root + "/chains", exist_ok=True)
  out_path = root + "/chains/" + args.output + ".npz"
  np.savez(out_path, **out)
  print("covariance written -> " + out_path)
  print("  Gaussian sigmas: tt/te/ee/pp over l = 2.." + str(lmax)
        + ("  + dense NG blocks (3 per-spectrum + 3 cross)"
           if study is not None else "  (non-Gaussian OFF)"))


if __name__ == "__main__":
  main()
