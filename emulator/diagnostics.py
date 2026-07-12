"""This module runs model diagnostics over a validation set.

It provides the post-training analyses that say why the metric sits
where it does (each returns a dict the plotting reads).

The chi2-based analyses are family-generic BY CONSTRUCTION (every
family's loss exposes a per-sample chi2, and these consume only params +
per-sample chi2): coverage_diagnostic asks whether the failing val
points sit in sparse regions of the training set (a kNN-distance vs
delta-chi2 correlation, i.e. data coverage); local_linear_floor compares
the model to a local-linear interpolation of the training targets (the
data-only floor; plain chi2 only); hard_direction_regression fits log10
delta-chi2 against the (log) parameters to find which combination
predicts the per-point hardness.

The PHYSICAL-units analyses are per family (the D-CM9 dispatch):
cmb_residual_diagnostic (per-multipole residual statistics + the
high-pass wiggle content the D-CM8 roughness term targets) and
scalar_output_diagnostic (per-output truth/prediction/residual tables).
The plotting side (plot_diagnostics) turns each family dict into its
pages; a run passes only the dict its family produces, so the
cosmic-shear PDF is byte-identical when neither is given.

PS: whitened = rotated into the covariance eigenbasis and scaled to unit
variance, so correlated quantities become decorrelated and equally hard
to fit. For the diagonal CMB geometry "whitened" is per-multipole: the
residual divided by its cosmic-variance error bar sigma_ell.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .training import eval_source_chi2


def coverage_diagnostic(model,
                        param_geometry,
                        chi2fn,
                        train_set,
                        val_set,
                        device,
                        k_nn=8,
                        bs=256):
  """
  Do the failing val points sit in sparse regions of training?

  For each validation cosmology, measure its mean distance to the
  k nearest training cosmologies in whitened (param_geometry)
  parameter space (Euclidean distance there weights each direction
  by its prior spread, so no single wide param dominates) and
  relate that local sparsity to the per-point delta-chi2. A positive
  rank correlation (sparser neighbourhoods at the failures) means
  the floor is data coverage, not the model. Model-agnostic: any
  trained model works.

  Arguments:
    model          = the trained network (eval_source_chi2 sets
                     eval mode).
    param_geometry = ParamGeometry; .encode whitens the raw params
                     to the decorrelated, unit-variance metric the
                     kNN distance uses.
    chi2fn         = the loss/geometry wrapper (plain or rescaled);
                     scores the val dchi2.
    train_set      = training source dict; its used rows are the
                     interpolation anchors (the training cloud).
    val_set        = validation source dict (the points scored).
    device         = device the model is on.
    k_nn           = neighbours averaged for the local-density
                     estimate (default 8).
    bs             = forward batch size for the dchi2 scoring.

  Returns:
    a dict with:
      knn_dist  = (Nval,) mean distance to the k nearest train pts.
      dchi2     = (Nval,) per-val delta-chi2 (same row order).
      k_nn      = the k used (for axis labels downstream).
      spearman  = rank correlation of knn_dist with log10 dchi2.
      median_good / median_bad = median knn_dist of the
                  dchi2<=0.2 / dchi2>0.2 populations.
      frac_dense / frac_sparse = frac>0.2 in the densest / sparsest
                  knn_dist decile.
      coverage_limited = bool verdict (failures in sparse regions:
                  median_bad > median_good, spearman > 0.1).
  """
  # per-val delta-chi2 from the model (sorted-idx order).
  # eval_source_chi2 (training.py): runs the model over a source in
  # eval mode and returns (params, per-point delta-chi2) row-aligned.
  _, dchi2 = eval_source_chi2(model=model,
                              param_geometry=param_geometry,
                              chi2fn=chi2fn,
                              source=val_set,
                              device=device,
                              bs=bs)

  # whitened params: the training cloud (anchors) and the val
  # points. encode decorrelates + unit-scales, so Euclidean
  # distance weights every direction by its prior spread.
  tr_rows = np.sort(np.unique(train_set["idx"]))
  va_rows = np.sort(val_set["idx"])
  with torch.no_grad():
    Xtr = param_geometry.encode(torch.from_numpy(
      np.asarray(train_set["C"][tr_rows], dtype="float64")
    ).float().to(device)).cpu().numpy()
    Xva = param_geometry.encode(torch.from_numpy(
      np.asarray(val_set["C"][va_rows], dtype="float64")
    ).float().to(device)).cpu().numpy()

  # mean distance from each val point to its k nearest training
  # points: a local sparsity measure (large = under-covered).
  # cKDTree.query returns (distances, indices); keep distances.
  tree = cKDTree(Xtr)
  dists, _ = tree.query(Xva, k=k_nn)     # (Nval, k_nn)
  knn_dist = dists.mean(1)               # (Nval,)

  # quantify. log10 dchi2 (floored so a near-zero stays finite)
  # tames the heavy tail; spearman is the rank correlation.
  y = np.log10(np.maximum(dchi2, 1e-4))
  rho, _ = spearmanr(knn_dist, y)
  bad = dchi2 > 0.2
  q10, q90 = np.quantile(knn_dist, [0.1, 0.9])
  median_good = float(np.median(knn_dist[~bad]))
  median_bad  = float(np.median(knn_dist[bad]))
  frac_dense  = float(np.mean(dchi2[knn_dist <= q10] > 0.2))
  frac_sparse = float(np.mean(dchi2[knn_dist >= q90] > 0.2))
  cov = (median_bad > median_good) and (rho > 0.1)

  return {"knn_dist": knn_dist, "dchi2": dchi2, "k_nn": k_nn,
          "spearman": float(rho),
          "median_good": median_good, "median_bad": median_bad,
          "frac_dense": frac_dense, "frac_sparse": frac_sparse,
          "coverage_limited": bool(cov)}


def local_linear_floor(model,
                       param_geometry,
                       chi2fn,
                       train_set,
                       val_set,
                       device,
                       k_nn=40,
                       bs=256):
  """
  The data-only floor: a local linear map vs the trained model.

  For each val point, fit a local linear map params -> whitened
  target over its k nearest training points and predict the val
  target; that prediction's chi2 is the best a smooth local method
  extracts from the data. A linear fit is exact for a locally-linear
  map, so its error is the local nonlinearity (hardness) plus
  residual coverage. Comparing the fractions:
    f_model ~ f_floor  -> data / representation-limited (the net is
                          at what the data supports; lever = prior /
                          features / more N).
    f_model >> f_floor -> the net has headroom (arch / training).
  f_floor in the best-covered (densest) decile = pure hardness.

  Valid only for a plain CosmolikeChi2 (needs_params == False): the
  fit lives in the whitened target space chi2fn.encode(dv) builds,
  and a rescaled encode/chi2 would need each point's own R.

  Arguments:
    model          = the trained network (for the model dchi2).
    param_geometry = ParamGeometry; .encode whitens the params
                     (the kNN space) and chi2fn.encode the targets.
    chi2fn         = a plain CosmolikeChi2 (raises otherwise).
    train_set      = training source dict (the fit anchors).
    val_set        = validation source dict (the points scored).
    device         = device the model is on.
    k_nn           = neighbours for the local linear fit (default
                     40; must exceed n_param + 1).
    bs             = forward batch size for the model dchi2.

  Returns:
    a dict with dchi2_floor, dchi2_model (both (Nval,)) and the
    scalars f_floor, f_model, f_hard, median_floor, median_model.
  """
  if getattr(chi2fn, "needs_params", False):
    raise ValueError(
      "local_linear_floor needs a plain CosmolikeChi2 "
      "(this chi2fn has needs_params == True)")

  tr_rows = np.sort(np.unique(train_set["idx"]))
  va_rows = np.sort(val_set["idx"])
  with torch.no_grad():
    Xtr = param_geometry.encode(torch.from_numpy(np.asarray(
      train_set["C"][tr_rows], "float64")).float().to(device))
    Xva = param_geometry.encode(torch.from_numpy(np.asarray(
      val_set["C"][va_rows], "float64")).float().to(device))
    Ttr = chi2fn.encode(torch.from_numpy(
      np.asarray(train_set["dv"][tr_rows])).float().to(device))
    Tva = chi2fn.encode(torch.from_numpy(
      np.asarray(val_set["dv"][va_rows])).float().to(device))

  # k nearest training neighbours of each val point (param space).
  tree = cKDTree(Xtr.cpu().numpy())
  knn_d, nbr = tree.query(Xva.cpu().numpy(), k=k_nn)
  knn_dist = knn_d.mean(1)                        # coverage scalar
  nbr = torch.from_numpy(nbr).to(device)

  # local linear fit: target ~ b + A (x - x_val) over the
  # neighbours, with intercept b = the prediction at x_val.
  # Solve on CPU (batched lstsq is not on MPS), one-time.
  Xn = Xtr[nbr]                                   # (Nval, k, n_param)
  Yn = Ttr[nbr]                                   # (Nval, k, out_dim)
  dX = (Xn - Xva[:, None, :]).cpu()
  ones = torch.ones(dX.shape[0], dX.shape[1], 1)
  # design = [1, (x - x_val)], so column 0's coefficient is b.
  design = torch.cat([ones, dX], dim=-1)          # (Nval, k, n_p+1)
  coef = torch.linalg.lstsq(design, Yn.cpu()).solution
  Tlin = coef[:, 0, :].to(device)                 # intercept = pred

  dchi2_floor = chi2fn.chi2(pred=Tlin,
                            target=Tva).double().cpu().numpy()
  # eval_source_chi2 (training.py): runs the model over a source in
  # eval mode and returns (params, per-point delta-chi2) row-aligned.
  _, dchi2_model = eval_source_chi2(model=model,
                                    param_geometry=param_geometry,
                                    chi2fn=chi2fn, source=val_set,
                                    device=device, bs=bs)
  # pure hardness: the floor in the densest (best-covered) decile.
  dense = knn_dist <= np.quantile(knn_dist, 0.1)
  return {"dchi2_floor": dchi2_floor, "dchi2_model": dchi2_model,
          "f_floor": float(np.mean(dchi2_floor > 0.2)),
          "f_model": float(np.mean(dchi2_model > 0.2)),
          "f_hard": float(np.mean(dchi2_floor[dense] > 0.2)),
          "median_floor": float(np.median(dchi2_floor)),
          "median_model": float(np.median(dchi2_model))}


def hard_direction_regression(model,
                              param_geometry,
                              chi2fn,
                              val_set,
                              device,
                              bs=256,
                              log_set=None):
  """
  Which log-param combination predicts the per-point hardness?

  Fits log10 dchi2 ~ c0 + sum_i c_i z_i, with z_i = standardized
  ln(param / median) for the positive multiplicative cosmological
  params and standardized centered-linear for the additive
  nuisances (photo-z DZ, IA A1, which can be <= 0). Reports each
  feature's univariate correlation (a collinearity-robust ranking),
  the joint OLS coefficients (the alpha, beta, ... combination) and
  joint R^2 (how much of the difficulty is a clean log-linear
  direction), and the ln(omega_b h^2)-alone R^2 (does it collapse
  to that single physical-baryon direction?). Works for any chi2fn:
  the dchi2 comes from eval_source_chi2's param-aware path.

  Arguments:
    model          = the trained network.
    param_geometry = ParamGeometry; .names gives the column order.
    chi2fn         = the loss/geometry wrapper (plain or rescaled).
    val_set        = validation source dict (the points scored).
    device         = device the model is on.
    bs             = forward batch size for the dchi2 scoring.
    log_set        = parameter names ln-transformed before
                     standardizing (default the positive
                     cosmological params As_1e9 / ns / H0 / omegam
                     / omegab).

  Returns:
    a dict with labels (per feature), univariate (the per-feature
    correlations), joint_coef (the joint coefficients, no
    intercept), r2 (joint), and r2_omega (ln(omega_b h^2) alone).
  """
  if log_set is None:
    log_set = {"As_1e9", "ns", "H0", "omegam", "omegab"}
  # eval_source_chi2 (training.py): runs the model over a source in
  # eval mode and returns (params, per-point delta-chi2) row-aligned;
  # here the params feed the hard-direction regression.
  params, dchi2 = eval_source_chi2(model=model,
                                   param_geometry=param_geometry,
                                   chi2fn=chi2fn, source=val_set,
                                   device=device, bs=bs)
  names = list(param_geometry.names)
  y = np.log10(np.maximum(dchi2, 1e-4))

  # ln(param/median) for the positive multiplicative params;
  # centered-linear for the additive nuisances. Standardize so the
  # coefficients are comparable.
  feat, lab = [], []
  for j, nm in enumerate(names):
    x = params[:, j].astype("float64")
    f = np.log(x / np.median(x)) if nm in log_set else x - np.mean(x)
    feat.append((f - f.mean()) / (f.std() + 1e-30))
    lab.append(("ln " if nm in log_set else "") + nm)
  feat = np.column_stack(feat)

  # univariate (collinearity-robust): each feature's own
  # correlation with log10 dchi2.
  uni_vals = []
  for j in range(feat.shape[1]):
    uni_vals.append(np.corrcoef(feat[:, j], y)[0, 1])
  uni = np.array(uni_vals)
  # joint OLS (a column of 1s is the intercept) and its R^2.
  Z = np.column_stack([np.ones_like(y), feat])
  coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
  r2 = 1.0 - np.var(y - Z @ coef) / np.var(y)

  # does it collapse to a single direction, ln(omega_b h^2)? Only
  # answerable when the run samples omegab + H0 (the cosmic-shear
  # convention); a family whose dump parameterizes the baryon density
  # differently (e.g. omegabh2 directly) reports NaN and the joint R^2
  # above still stands.
  if "omegab" in names and "H0" in names:
    ob = params[:, names.index("omegab")].astype("float64")
    h  = params[:, names.index("H0")].astype("float64") / 100.0
    g  = np.log(ob * h ** 2 / np.median(ob * h ** 2))
    g  = (g - g.mean()) / g.std()
    Zo = np.column_stack([np.ones_like(y), g])
    co, *_ = np.linalg.lstsq(Zo, y, rcond=None)
    r2o = 1.0 - np.var(y - Zo @ co) / np.var(y)
  else:
    r2o = float("nan")

  return {"labels": lab, "univariate": uni, "joint_coef": coef[1:],
          "r2": float(r2), "r2_omega": float(r2o)}


def cmb_residual_diagnostic(model,
                            param_geometry,
                            chi2fn,
                            val_set,
                            device,
                            bs=256,
                            period_cut=None):
  """
  Per-multipole residual statistics for a CMB spectrum run (D-CM9).

  Decodes every validation prediction back to PHYSICAL C_ell (the
  training chi2fn.decode, so the imposed amplitude law is multiplied
  back) and summarizes the residual against the true spectra two ways
  per multipole: fractionally ((pred - truth) / truth; readable for
  tt / ee / pp, spiky where te crosses zero) and in error-bar units
  ((pred - truth) / sigma_ell; always well-defined — for te read this
  one). Also finds the worst validation cosmology (highest per-sample
  chi2) for a pred-vs-truth overlay, and measures the residual's
  HIGH-PASS content — the short-period wiggle spectrum the D-CM8
  roughness term penalizes, computed with the same double-boxcar
  remainder — so over-smoothing or ringing is visible at a glance.

  Arguments:
    model          = the trained network.
    param_geometry = ParamGeometry; .encode whitens the raw params.
    chi2fn         = the CMB loss wrapper (CmbDiagonalChi2 /
                     CmbFactoredChi2); its geom carries ell / sigma /
                     spectrum / units, its decode the law.
    val_set        = validation source dict ("C" / "dv" / "idx").
    device         = device the model is on.
    bs             = forward batch size.
    period_cut     = the high-pass band edge in multipoles; None reads
                     the run's configured roughness term (chi2fn._rough)
                     and falls back to 50 (the D-CM8 default band) when
                     the run trained without one.

  Returns:
    a dict with:
      ell / spectrum / units       = the geometry's grid + labels.
      frac_med, frac_lo68, frac_hi68, frac_lo95, frac_hi95
                                   = per-ell fractional-residual bands.
      sig_med, sig_lo68, sig_hi68, sig_lo95, sig_hi95
                                   = per-ell residual/sigma bands.
      worst = {"dchi2", "pred", "truth", "params"} at the highest-chi2
              val point (physical C_ell rows; params a {name: value}).
      highpass = {"median_abs_rem", "period_cut"}: the median absolute
              short-period remainder of the WHITENED residual vs ell.
  """
  geom = chi2fn.geom
  rows  = np.sort(val_set["idx"])
  C     = np.asarray(val_set["C"][rows], dtype="float64")
  truth = np.asarray(val_set["dv"][rows], dtype="float64")
  needs_p = getattr(chi2fn, "needs_params", False)

  # batched forward + decode to physical C_ell (the training decode,
  # never re-derived), and the per-sample chi2 for the worst point.
  preds = []
  chi2s = []
  model.eval()
  with torch.no_grad():
    start = 0
    while start < len(rows):
      stop  = min(len(rows), start + bs)
      x     = torch.from_numpy(C[start:stop]).float().to(device)
      x_enc = param_geometry.encode(x)
      p     = model(x_enc)
      t     = torch.from_numpy(truth[start:stop]).float().to(device)
      if needs_p:
        cl = chi2fn.decode(p, x_enc)
        tw = chi2fn.encode(t, x_enc)
      else:
        cl = chi2fn.decode(p)
        tw = chi2fn.encode(t)
      preds.append(cl.double().cpu().numpy())
      chi2s.append(chi2fn.chi2(pred=p, target=tw).double().cpu().numpy())
      start = stop
  pred  = np.concatenate(preds)
  dchi2 = np.concatenate(chi2s)

  sigma = geom.sigma.detach().cpu().numpy().astype("float64")
  frac = (pred - truth) / truth
  sig  = (pred - truth) / sigma[None, :]

  def bands(r):
    q = np.percentile(r, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
    return {"lo95": q[0], "lo68": q[1], "med": q[2],
            "hi68": q[3], "hi95": q[4]}

  fb = bands(frac)
  sb = bands(sig)

  worst = int(np.argmax(dchi2))
  names = list(param_geometry.names)
  worst_params = {}
  for j, nm in enumerate(names):
    worst_params[nm] = float(C[worst, j])

  # the D-CM8 companion: the whitened residual's short-period remainder
  # (the same double-boxcar high-pass the roughness term uses), so the
  # page shows exactly what the penalty would see.
  if period_cut is None:
    rough = getattr(chi2fn, "_rough", None)
    period_cut = rough.width if rough is not None else 50
  w = int(round(float(period_cut)))
  if w % 2 == 0:
    w += 1
  pad = w // 2
  kern = np.ones(w) / float(w)
  sm = sig
  for _ in range(2):
    padded = np.pad(sm, [(0, 0), (pad, pad)], mode="reflect")
    smoothed = np.empty_like(sm)
    for i in range(sm.shape[0]):
      smoothed[i] = np.convolve(padded[i], kern, mode="valid")
    sm = smoothed
  rem = sig - sm
  median_abs_rem = np.median(np.abs(rem), axis=0)

  return {"ell": geom.ell.detach().cpu().numpy(),
          "spectrum": geom.spectrum,
          "units": geom.units,
          "frac_med": fb["med"], "frac_lo68": fb["lo68"],
          "frac_hi68": fb["hi68"], "frac_lo95": fb["lo95"],
          "frac_hi95": fb["hi95"],
          "sig_med": sb["med"], "sig_lo68": sb["lo68"],
          "sig_hi68": sb["hi68"], "sig_lo95": sb["lo95"],
          "sig_hi95": sb["hi95"],
          "worst": {"dchi2": float(dchi2[worst]),
                    "pred": pred[worst], "truth": truth[worst],
                    "params": worst_params},
          "highpass": {"median_abs_rem": median_abs_rem,
                       "period_cut": int(w)}}


def scalar_output_diagnostic(model,
                             param_geometry,
                             chi2fn,
                             val_set,
                             device,
                             bs=256):
  """
  Per-output truth / prediction / residual tables for a scalar run
  (D-CM9's second family — the factoring must prove itself on two).

  Decodes every validation prediction back to PHYSICAL units
  (chi2fn.decode = the geometry's destandardization) and returns, per
  emulated output, the truth and prediction columns plus the residual
  in physical AND standardized units, alongside the raw input
  parameters — everything the scalar pages plot (truth-vs-predicted
  scatter, residual histograms both ways, residual vs each input: the
  bias hunt).

  Arguments:
    model          = the trained network.
    param_geometry = ParamGeometry; .encode whitens the raw params.
    chi2fn         = the ScalarChi2 wrapper (its geom holds names /
                     center / scale).
    val_set        = validation source dict ("C" / "dv" / "idx").
    device         = device the model is on.
    bs             = forward batch size.

  Returns:
    a dict with:
      names       = the emulated output names (geometry order).
      truth, pred = (Nval, n_out) physical values.
      resid_std   = (Nval, n_out) residual / the geometry's scale.
      params      = (Nval, n_param) raw input parameters.
      param_names = the input parameter names.
  """
  geom  = chi2fn.geom
  rows  = np.sort(val_set["idx"])
  C     = np.asarray(val_set["C"][rows], dtype="float64")
  truth = np.asarray(val_set["dv"][rows], dtype="float64")
  # an NPCE run's loss is param-aware (needs_params: decode evaluates
  # the frozen base from the whitened inputs) — the doctrine branch.
  needs_p = getattr(chi2fn, "needs_params", False)

  preds = []
  model.eval()
  with torch.no_grad():
    start = 0
    while start < len(rows):
      stop  = min(len(rows), start + bs)
      x     = torch.from_numpy(C[start:stop]).float().to(device)
      x_enc = param_geometry.encode(x)
      if needs_p:
        out = chi2fn.decode(model(x_enc), x_enc)
      else:
        out = chi2fn.decode(model(x_enc))
      preds.append(out.double().cpu().numpy())
      start = stop
  pred  = np.concatenate(preds)
  scale = geom.scale.detach().cpu().numpy().astype("float64")

  return {"names": list(geom.names),
          "truth": truth,
          "pred": pred,
          "resid_std": (pred - truth) / scale[None, :],
          "params": C,
          "param_names": list(param_geometry.names)}


def grid_residual_diagnostic(model,
                             param_geometry,
                             chi2fn,
                             val_set,
                             device,
                             bs=256,
                             n_derived=64):
  """
  Per-redshift residual statistics for a grid (background) run (D-BSN8).

  Decodes every validation prediction back to the PHYSICAL function
  (the geometry's decode inverts the target law) and summarizes the
  fractional residual per grid redshift (median + 68/95 bands), finds
  the worst validation cosmology (highest per-sample chi2) for a
  pred-vs-truth overlay, and — for a "Hubble" artifact — propagates a
  subsample through the REAL distance pipeline
  (emulator/background.py) to band the derived D_A / D_L fractional
  errors: pipeline(predicted H) against pipeline(true H), so the page
  tests what the network error does to the integration path, not just
  the raw function.

  Arguments:
    model          = the trained network.
    param_geometry = ParamGeometry; .encode whitens the raw params.
    chi2fn         = the ScalarChi2 over the GridGeometry (its geom
                     carries z / quantity / units / law).
    val_set        = validation source dict ("C" / "dv" / "idx").
    device         = device the model is on.
    bs             = forward batch size.
    n_derived      = validation rows propagated through the distance
                     pipeline for the derived page (a cold path — one
                     Simpson integration per row).

  Returns:
    a dict with:
      z / quantity / units       = the geometry's grid + labels.
      frac_med, frac_lo68, frac_hi68, frac_lo95, frac_hi95
                                 = per-z fractional-residual bands.
      worst = {"dchi2", "pred", "truth"} at the highest-chi2 point.
      derived = None, or (for "Hubble") {"z_eval", "da_med", "da_lo68",
                "da_hi68", "dl_med", "dl_lo68", "dl_hi68"}: fractional
                derived-distance error bands at interior redshifts.
  """
  geom  = chi2fn.geom
  rows  = np.sort(val_set["idx"])
  C     = np.asarray(val_set["C"][rows], dtype="float64")
  truth = np.asarray(val_set["dv"][rows], dtype="float64")
  # an NPCE run's loss is param-aware (needs_params: encode / decode
  # evaluate the frozen base from the whitened inputs) — the doctrine
  # branch.
  needs_p = getattr(chi2fn, "needs_params", False)

  preds = []
  chi2s = []
  model.eval()
  with torch.no_grad():
    start = 0
    while start < len(rows):
      stop  = min(len(rows), start + bs)
      x     = torch.from_numpy(C[start:stop]).float().to(device)
      x_enc = param_geometry.encode(x)
      p     = model(x_enc)
      t     = torch.from_numpy(truth[start:stop]).float().to(device)
      if needs_p:
        tw = chi2fn.encode(t, x_enc)
        preds.append(chi2fn.decode(p, x_enc).double().cpu().numpy())
      else:
        tw = chi2fn.encode(t)
        preds.append(chi2fn.decode(p).double().cpu().numpy())
      chi2s.append(chi2fn.chi2(pred=p, target=tw).double().cpu().numpy())
      start = stop
  pred  = np.concatenate(preds)
  dchi2 = np.concatenate(chi2s)

  frac = (pred - truth) / truth
  q = np.percentile(frac, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
  worst = int(np.argmax(dchi2))

  derived = None
  if geom.quantity == "Hubble":
    from .background import distance_interpolators
    z_grid = geom.z.detach().cpu().numpy()
    # interior evaluation points (the pipeline is an interpolation; stay
    # off the exact edges).
    z_eval = np.linspace(z_grid[0] + 0.05 * (z_grid[-1] - z_grid[0]),
                         z_grid[-1] * 0.95, 25)
    take = min(int(n_derived), pred.shape[0])
    da_fr = np.empty((take, z_eval.size))
    dl_fr = np.empty((take, z_eval.size))
    for i in range(take):
      itp_p = distance_interpolators(z_grid=z_grid, h_grid=pred[i])
      itp_t = distance_interpolators(z_grid=z_grid, h_grid=truth[i])
      da_p, da_t = itp_p["da"](z_eval), itp_t["da"](z_eval)
      dl_p, dl_t = itp_p["dl"](z_eval), itp_t["dl"](z_eval)
      da_fr[i] = (da_p - da_t) / da_t
      dl_fr[i] = (dl_p - dl_t) / dl_t
    qa = np.percentile(da_fr, [16.0, 50.0, 84.0], axis=0)
    ql = np.percentile(dl_fr, [16.0, 50.0, 84.0], axis=0)
    derived = {"z_eval": z_eval,
               "da_lo68": qa[0], "da_med": qa[1], "da_hi68": qa[2],
               "dl_lo68": ql[0], "dl_med": ql[1], "dl_hi68": ql[2]}

  return {"z": geom.z.detach().cpu().numpy(),
          "quantity": geom.quantity,
          "units": geom.units,
          "frac_med": q[2], "frac_lo68": q[1], "frac_hi68": q[3],
          "frac_lo95": q[0], "frac_hi95": q[4],
          "worst": {"dchi2": float(dchi2[worst]),
                    "pred": pred[worst], "truth": truth[worst]},
          "derived": derived}


def grid2d_residual_diagnostic(model,
                               param_geometry,
                               chi2fn,
                               val_set,
                               device,
                               bs=256):
  """
  Per-(z, k) residual statistics for a grid2d (matter-power) run
  (MPS-DIAG).

  Decodes every validation prediction back to LAW space — what the
  network learns; the geometry's decode un-standardizes but never
  multiplies a syren base back — and summarizes the residual over the
  validation set on the stored (z, k) grid. Under a syren law the
  law-space residual IS the log-ratio of the physical spectra,

      pred - truth = ln(P_pred / P_base) - ln(P_truth / P_base)
                   = ln(P_pred / P_truth)

  (the base cancels), so the numbers read directly as the fractional
  error of the SERVED spectrum for small residuals. Under law "none"
  the surfaces are physical and the residual is the usual
  (pred - truth) / truth. The returned res_kind names which one the
  arrays hold.

  Arguments:
    model          = the trained network.
    param_geometry = ParamGeometry; .encode whitens the raw params.
    chi2fn         = the ScalarChi2 over the Grid2DGeometry (its geom
                     carries z / k / quantity / units / law).
    val_set        = validation source dict ("C" / "dv" / "idx"); its
                     dv rows are the STAGED law-space surfaces.
    device         = device the model is on.
    bs             = forward batch size.

  Returns:
    a dict with:
      z, k                = the stored axes (k thinned by k_stride).
      quantity/units/law  = the geometry's labels.
      res_kind            = "ln-ratio" (syren laws) or "fractional".
      med_abs   (nz, nk)  = median |residual| over the val set.
      slices              = per-redshift cuts at the first / middle /
                            last z (deduplicated when nz is small):
                            {"iz", "z", "lo95", "lo68", "med", "hi68",
                             "hi95"}, each band (nk,).
      worst               = {"dchi2", "res" (nz, nk)} at the
                            highest-chi2 validation cosmology.
  """
  geom  = chi2fn.geom
  rows  = np.sort(val_set["idx"])
  C     = np.asarray(val_set["C"][rows], dtype="float64")
  truth = np.asarray(val_set["dv"][rows], dtype="float64")
  # an NPCE run's loss is param-aware (needs_params: encode / decode
  # evaluate the frozen base from the whitened inputs) — the doctrine
  # branch.
  needs_p = getattr(chi2fn, "needs_params", False)

  preds = []
  chi2s = []
  model.eval()
  with torch.no_grad():
    start = 0
    while start < len(rows):
      stop  = min(len(rows), start + bs)
      x     = torch.from_numpy(C[start:stop]).float().to(device)
      x_enc = param_geometry.encode(x)
      p     = model(x_enc)
      t     = torch.from_numpy(truth[start:stop]).float().to(device)
      if needs_p:
        tw = chi2fn.encode(t, x_enc)
        preds.append(chi2fn.decode(p, x_enc).double().cpu().numpy())
      else:
        tw = chi2fn.encode(t)
        preds.append(chi2fn.decode(p).double().cpu().numpy())
      chi2s.append(chi2fn.chi2(pred=p, target=tw).double().cpu().numpy())
      start = stop
  pred  = np.concatenate(preds)
  dchi2 = np.concatenate(chi2s)

  nz = int(geom.z.numel())
  nk = int(geom.k.numel())
  if geom.law == "none":
    res = (pred - truth) / truth
    res_kind = "fractional"
  else:
    # law space: the difference is ln(P_pred / P_truth), base-free.
    res = pred - truth
    res_kind = "ln-ratio"
  res = res.reshape(-1, nz, nk)

  med_abs = np.median(np.abs(res), axis=0)

  z_grid = geom.z.detach().cpu().numpy()
  iz_cuts = sorted(set((0, nz // 2, nz - 1)))
  slices = []
  for iz in iz_cuts:
    band = np.percentile(res[:, iz, :],
                         [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
    slices.append({"iz": iz, "z": float(z_grid[iz]),
                   "lo95": band[0], "lo68": band[1], "med": band[2],
                   "hi68": band[3], "hi95": band[4]})

  worst = int(np.argmax(dchi2))
  return {"z": z_grid,
          "k": geom.k.detach().cpu().numpy(),
          "quantity": geom.quantity,
          "units": geom.units,
          "law": geom.law,
          "res_kind": res_kind,
          "med_abs": med_abs,
          "slices": slices,
          "worst": {"dchi2": float(dchi2[worst]), "res": res[worst]}}
