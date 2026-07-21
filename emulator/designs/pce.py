"""Sparse-Legendre PCE machinery and the PCEEmulator (the NPCE base).

This is the polynomial-chaos member of the emulator/designs/ family
(the former emulator PCE subpackage): the polynomial-chaos side of
the NPCE (Neural PCE) experiment: PCEEmulator, a closed-form
sparse-Legendre expansion mapping the cosmological parameters to the
whitened data vector with no network, plus its three fit helpers,
pce_multi_index (the sparse candidate basis), pce_design (the
normalized-Legendre design matrix), and select_lars_loo (greedy term
selection with a leave-one-out stop). The companion losses/pce.py wraps
a frozen PCEEmulator as the base under a neural refiner.

Verdict for cosmic-shear xi (recorded in
ai/notes/models-and-designs.md): a PCE base only adds
capacity, it cannot lower a data-coverage floor, so the NPCE was
deprioritized. The machinery is kept for reuse and for the build
lessons baked into the docstrings below: keep the degree low (a
high-degree Legendre fit Runge-oscillates), gate each mode on its
leave-one-out error (a wiggly base poisons the refiner), and let the
refiner backstop everything the gate drops.

PS: PCE = polynomial chaos expansion, a sum of orthogonal
polynomials of the inputs; LOO = leave-one-out error, the fit's
generalization error estimated without refitting (the PRESS shortcut
in select_lars_loo); whitened = transformed to unit-variance
components (the chi2 metric basis); mode = one SVD direction of the
centered training targets.
"""

import itertools
import numbers
import numpy as np
import torch
import torch.nn as nn


def _finite_real_array(value, *, name, ndim):
  """
  Convert one array-like input to finite float64, refusing anything else.

  The PCE fit runs in float64, so every incoming array (a torch tensor, a
  list, or a numpy array of any real dtype) is converted once here, and
  the three ways an input can silently poison the fit are refused up
  front: a non-numeric dtype, a wrong dimensionality, and a NaN or
  infinity.

  Arguments:
    value = the array-like input (torch.Tensor, numpy array, or nested
            sequence of numbers).
    name  = what the value is, named in every refusal.
    ndim  = the exact required number of dimensions.

  Returns:
    a float64 numpy array of dimensionality ndim, all values finite.

  Raises:
    ValueError naming the input when its dtype is not real-numeric, its
    dimensionality differs from ndim, or it contains NaN / infinity.
  """
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu().numpy()
  raw = np.asarray(value)
  if not np.isrealobj(raw) or raw.dtype.kind not in "fiu":
    raise ValueError(
      f"{name} must contain real numerical values, got dtype {raw.dtype}")
  array = np.asarray(raw, dtype=np.float64)
  if array.ndim != ndim:
    raise ValueError(
      f"{name} must be {ndim}-dimensional, got shape {array.shape}")
  if not np.isfinite(array).all():
    raise ValueError(f"{name} must contain only finite values")
  return array


def _positive_int(value, *, name):
  """
  Read one positive-integer fit limit, refusing Booleans and floats.

  In Python True == 1, so a Boolean would slip through a plain integer
  check and set a nonsensical limit of one; a float such as 12.0 would
  hide a configuration typo. Both are refused by type, not by value.

  Arguments:
    value = the configured limit, expected to be an int (numpy integers
            accepted).
    name  = the configuration key, named in the refusal.

  Returns:
    the limit as a plain int, at least 1.

  Raises:
    ValueError when the value is a Boolean, not an integer, or below 1.
  """
  if (isinstance(value, (bool, np.bool_))
      or not isinstance(value, (int, np.integer)) or int(value) < 1):
    raise ValueError(f"{name} must be a positive integer, got {value!r}")
  return int(value)


def _positive_finite_real(value, *, name):
  """
  Read one strictly positive, finite real fit limit.

  A Boolean is refused by type (True == 1 in Python); a string such as
  "0.6" is refused rather than converted, so a YAML quoting mistake stops
  the run instead of silently parsing; NaN, infinity, zero, and negative
  values are refused by value.

  Arguments:
    value = the configured limit, expected to be a real number.
    name  = the configuration key, named in the refusal.

  Returns:
    the limit as a plain float, finite and > 0.

  Raises:
    ValueError when the value is a Boolean, not a real number, nonfinite,
    or not strictly positive.
  """
  if (isinstance(value, (bool, np.bool_))
      or not isinstance(value, numbers.Real)
      or not np.isfinite(float(value)) or float(value) <= 0.0):
    raise ValueError(
      f"{name} must be finite and strictly positive, got {value!r}")
  return float(value)


def _fixed_fit_loo(Psi, y, support, beta, *, saved_prediction=None):
  """
  Leave-one-out (LOO) error of one fixed active fit, in saved precision.

  The PRESS/LOO identity scores a least-squares fit without refitting N
  times: with the hat-matrix leverage h_n = a_n^T (A^T A)^-1 a_n of row n,

    loo = mean_n [ (y_n - yhat_n) / (1 - h_n) ]^2 / var(y)

  where yhat is the model's prediction of row n WITH row n included --
  dividing each residual by (1 - h_n) converts it to the residual the fit
  would have produced had row n been left out. The result is normalized by
  the target variance, so it is comparable across output modes.

  The prediction yhat is deliberately computed in float32, the precision
  the saved base actually uses at inference: the stored design matrix and
  coefficients are multiplied densely, including the zero coefficients
  outside the support, because promoting to float64 or shortening the
  matrix first changes cancellation and can hide error the saved base
  retains.

  Arguments:
    Psi     = (N, P) float64 design matrix over the full candidate basis.
    y       = (N,) float64 target values of one output mode.
    support = index array of the active columns (the chosen basis terms).
    beta    = coefficients of the active columns, in support order.
    saved_prediction = optional (N,) prediction to score instead of
              recomputing it (used when the caller already holds the saved
              base's own output); shape-checked against y.

  Returns:
    the normalized LOO error as a float (finite, >= 0).

  Raises:
    ValueError when the target variance is not positive, the normal matrix
    is singular or nonfinite, a leverage leaves [0, 1), or any saved-format
    quantity becomes nonfinite.
  """
  A = Psi[:, support]
  with np.errstate(over="ignore", invalid="ignore"):
    variance = np.var(y)
    normal = A.T @ A + 1e-10 * np.eye(len(support))
  if not np.isfinite(variance) or variance <= 0.0:
    raise ValueError("PCE mode-target variance must remain finite and positive")
  if not np.isfinite(normal).all():
    raise ValueError("PCE active normal matrix became nonfinite")
  try:
    inverse = np.linalg.inv(normal)
  except np.linalg.LinAlgError as error:
    raise ValueError("PCE active normal matrix could not be inverted") from error
  with np.errstate(over="ignore", invalid="ignore"):
    leverage = np.einsum("ni,ij,nj->n", A, inverse, A)
    Psi_saved = Psi.astype(np.float32)
    beta_saved = np.asarray(beta, dtype=np.float32)
    full_beta = np.zeros(Psi.shape[1], dtype=np.float32)
    full_beta[support] = beta_saved
  if (not np.isfinite(Psi_saved).all()
      or not np.isfinite(full_beta).all()):
    raise ValueError(
      "PCE active fit became nonfinite in saved float32 arithmetic")
  if saved_prediction is None:
    # Forward multiplies the stored design and coefficients in float32. Do the
    # same dense multiplication here, including zero coefficients outside the
    # support. Promoting values back to float64 or shortening the matrix first
    # can change cancellation and hide error that the saved base retains.
    with torch.no_grad():
      prediction = (
        torch.from_numpy(Psi_saved) @ torch.from_numpy(full_beta)
      ).to(torch.float64).numpy()
  else:
    prediction = _finite_real_array(
      saved_prediction, name="PCE saved-format mode prediction", ndim=1)
    if prediction.shape != y.shape:
      raise ValueError(
        "PCE saved-format mode prediction must match the target shape: "
        f"got {prediction.shape} and {y.shape}")
  residual = y - prediction
  if (not np.isfinite(leverage).all() or (leverage < 0.0).any()
      or (leverage >= 1.0).any()):
    raise ValueError(
      "PCE leave-one-out leverage must be finite and lie in [0, 1)")
  if not np.isfinite(residual).all():
    raise ValueError("PCE active fit produced nonfinite residuals")
  with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    loo = np.mean((residual / (1.0 - leverage)) ** 2) / variance
  if not np.isfinite(loo) or loo < 0.0:
    raise ValueError(
      "PCE leave-one-out error must be finite and nonnegative, got "
      f"{loo!r}")
  return float(loo)


def _pce_deg_tuples(m, pq, q):
  """
  Enumerate degree tuples (a_1, ..., a_m), each a_i >= 1, with
  sum_i a_i^q <= pq.

  This is the inner enumeration of the hyperbolic (q-norm) truncation:
  the candidate basis keeps a multi-index only when its q-norm
  (sum a_i^q)^(1/q) is within the degree budget, which for q < 1 favors
  few large per-variable degrees over many moderate ones (high-order
  interactions are pruned first). The recursion extends the tuple one
  variable at a time and abandons a branch as soon as its running sum
  of a_i^q exceeds the budget, so the pruned tree is never walked.

  Arguments:
    m  = tuple length (the number of interacting variables).
    pq = the budget: p_max raised to the power q, so the q-norm test
         (sum a_i^q)^(1/q) <= p_max becomes sum a_i^q <= pq.
    q  = the hyperbolic truncation exponent (0 < q <= 1).

  Returns:
    a generator of degree tuples, each of length m with every entry >= 1.
  """
  def rec(prefix, used):
    # prefix = the degrees chosen so far; used = their sum of a_i^q.
    # At full length the tuple is emitted; otherwise every degree d
    # that still fits the budget extends the branch.
    if len(prefix) == m:
      yield tuple(prefix)
      return
    d = 1
    while used + d ** q <= pq:
      yield from rec(prefix=prefix + [d], used=used + d ** q)
      d += 1
  yield from rec(prefix=[], used=0.0)


def pce_multi_index(n_dim, p_max=12, r_max=3, q=0.6):
  """
  Sparse candidate multi-index set A_cand (eq 11).

  Every multi-index alpha = (a_1, ..., a_n_dim) of per-variable
  degrees a_i >= 0 passes both rules of the hybrid truncation:
    - hyperbolic q-norm:  (sum_i a_i^q)^(1/q) <= p_max
    - max interaction:    #(a_i != 0)          <= r_max
  The all-zero index (the constant) is row 0.

  The q-norm scores a degree spread over many variables against one
  concentrated in a single variable. Worked example with p_max = 4,
  three terms of total degree 4:
    term          q = 1        q = 0.5
    x1^4          4  -> keep    4  -> keep
    x1^2 x2^2     4  -> keep    8  -> drop
    x1 x2 x3 x4   4  -> keep    16 -> drop
  At q = 1 the norm is the plain total degree, so a 4-way
  interaction counts as one degree-4 variable. At q < 1 spreading a
  degree over k variables costs k^(1/q) instead of k (the two- and
  four-variable terms score 8 and 16, past p_max = 4), dropping
  high-interaction cross-terms first, the sparsity-of-effects
  prior. Smaller q = sparser basis.

  Arguments:
    n_dim = number of input parameters (here 12).
    p_max = maximum total degree (the q-norm bound); the smoothness
            knob (low = smooth, high = Runge risk).
    r_max = maximum interaction order = most variables allowed
            together in one term (#(a_i != 0) <= r_max).
    q     = hyperbolic-norm exponent in (0, 1]; the sparsity knob
            (q=1 = plain total degree; smaller drops
            high-interaction terms -> sparser; see above).
  Returns:
    multi_index = (n_terms, n_dim) int array, row 0 constant.
  """
  pq   = p_max ** q
  rows = [np.zeros(n_dim, dtype=int)]      # constant term
  # every active subset of size 1..r_max, with all degree tuples
  # (>=1 on those dims) under the q-norm budget.
  for m in range(1, r_max + 1):
    for dims in itertools.combinations(range(n_dim), m):
      for degs in _pce_deg_tuples(m=m, pq=pq, q=q):
        a = np.zeros(n_dim, dtype=int)
        for d, deg in zip(dims, degs):
          a[d] = deg
        rows.append(a)
  return np.array(rows, dtype=int)


def pce_design(Xm, multi_index):
  """
  Normalized-Legendre PCE design matrix Psi (eq 10).

    Psi[n,t] = prod_l sqrt(2 a_{t,l}+1) * P_{a_{t,l}}(Xm[n,l])

  P = Legendre polynomial (orthogonal on [-1,1]); the sqrt(2a+1)
  factor makes each 1-D factor orthonormal. Legendre values from the
  three-term recurrence
    (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}, P_0=1, P_1=x.
  Pure torch, so one implementation runs on CPU (fit) and GPU
  (predict).

  Arguments:
    Xm          = (N, n_dim) inputs mapped to [-1, 1].
    multi_index = (n_terms, n_dim) long tensor of degrees.
  Returns:
    Psi = (N, n_terms) on Xm's device/dtype.
  """
  N, d = Xm.shape
  T    = multi_index.shape[0]
  maxd = int(multi_index.max())
  Psi  = torch.ones(N, T, dtype=Xm.dtype, device=Xm.device)
  for l in range(d):
    x = Xm[:, l]
    # Legendre table P_0..P_maxd for this dim: (N, maxd+1).
    cols = [torch.ones_like(x)]
    if maxd >= 1:
      cols.append(x)
    for n in range(1, maxd):
      cols.append(((2 * n + 1) * x * cols[n]
                   - n * cols[n - 1]) / (n + 1))
    tab  = torch.stack(cols, dim=-1)          # (N, maxd+1)
    a    = multi_index[:, l]                  # (T,)
    norm = torch.sqrt(2.0 * a.to(Xm.dtype) + 1.0)
    Psi  = Psi * (norm * tab[:, a])           # gather + scale
  return Psi


def select_lars_loo(Psi, y, max_terms=150, patience=10):
  """
  Greedy OMP-style selection with a leave-one-out (LOO) stop.

  At each iteration the implementation chooses the inactive basis column
  with the largest normalized residual correlation. It then refits all
  active coefficients by least squares and evaluates the PRESS/LOO score.
  It does not execute the least-angle-regression path algorithm.

  Arguments:
    Psi       = (n_samples, n_terms) PCE design matrix, each column
                one basis polynomial evaluated at the samples.
    y         = (n_samples,) target values to fit.
    max_terms = cap on the number of selected basis terms. The effective
                cap is also bounded by the usable candidate count and by
                n_samples - 1, leaving one row beyond the active count.
    patience  = stop after this many consecutive term additions
                bring no leave-one-out improvement (the early stop
                that keeps a wiggly high-degree term from poisoning
                the fit).

  Returns:
    support = int array of selected column indices.
    coef    = OLS coefficients aligned with `support`.
    loo     = relative leave-one-out MSE at the chosen model
              = mean((y - y_pred)^2 leave-one-out) / var(y)
              = 1 - R^2_LOO. 0 = perfect, 1 = no better than the
              mean; sqrt(loo) = typical error as a fraction of y's
              spread.
  """
  Psi = _finite_real_array(Psi, name="PCE design matrix", ndim=2)
  y = _finite_real_array(y, name="PCE mode target", ndim=1)
  max_terms = _positive_int(max_terms, name="PCE max_terms")
  patience = _positive_int(patience, name="PCE patience")
  n_samples, n_candidates = Psi.shape
  if n_samples < 2 or n_candidates < 1:
    raise ValueError(
      "PCE selection requires at least two training rows and one candidate "
      f"column, got design shape {Psi.shape}")
  if y.shape[0] != n_samples:
    raise ValueError(
      "PCE mode target row count must match the design matrix: "
      f"got {y.shape[0]} and {n_samples}")

  with np.errstate(over="ignore", invalid="ignore"):
    squared_norms = np.sum(Psi * Psi, axis=0)
    vy = np.var(y)
  if not np.isfinite(squared_norms).all():
    raise ValueError("PCE candidate column norms must remain finite")
  usable = squared_norms > 0.0
  if not usable[0]:
    raise ValueError("PCE constant candidate column must have positive norm")
  cn = np.ones_like(squared_norms)
  cn[usable] = np.sqrt(squared_norms[usable])
  if not np.isfinite(vy) or vy <= 0.0:
    raise ValueError("PCE mode-target variance must remain finite and positive")

  active    = [0]                             # constant term
  best_loo  = np.inf
  best_supp = [0]
  best_beta = None
  since     = 0

  # A model with one coefficient per row can interpolate its own data while
  # leaving no independent information for a leave-one-out check.  Keep at
  # least one row beyond the active term count.
  available_count = int(np.count_nonzero(usable))
  term_limit = min(max_terms, available_count, n_samples - 1)
  for _ in range(term_limit):
    A    = Psi[:, active]
    with np.errstate(over="ignore", invalid="ignore"):
      G = A.T @ A + 1e-10 * np.eye(len(active))
    if not np.isfinite(G).all():
      raise ValueError("PCE active normal matrix became nonfinite")
    try:
      Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError as error:
      raise ValueError("PCE active normal matrix could not be inverted") from error
    with np.errstate(over="ignore", invalid="ignore"):
      beta = Ginv @ (A.T @ y)
      resid = y - A @ beta                    # in-sample resid
    if not np.isfinite(beta).all() or not np.isfinite(resid).all():
      raise ValueError("PCE active fit produced nonfinite coefficients or residuals")

    # hat = leverage h_nn = diag of A (A^T A)^-1 A^T: how much
    # point n pulls its own fit; near 1 = high influence.
    with np.errstate(over="ignore", invalid="ignore"):
      hat = np.einsum("ni,ij,nj->n", A, Ginv, A)
    if (not np.isfinite(hat).all() or (hat < 0.0).any()
        or (hat >= 1.0).any()):
      raise ValueError(
        "PCE leave-one-out leverage must be finite and lie in [0, 1)")
    # LOO (generalization) error without refitting:
    #   resid / (1 - hat) = point n's residual when n is dropped
    #     from the fit (the PRESS shortcut; 1/(1 - h_nn) inflates the
    #     in-sample residual to its leave-one-out value).
    #   / vy normalizes by the target variance, so loo is
    #     dimensionless and comparable across modes:
    #       loo = mean(LOO residual^2) / var(y) = 1 - R^2_LOO.
    #   The absolute chi2 a mode adds is loo * var(mode), so a
    #   high-variance mode must reach a very small loo to help.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
      loo = np.mean((resid / (1.0 - hat)) ** 2) / vy
    if not np.isfinite(loo) or loo < 0.0:
      raise ValueError(
        "PCE leave-one-out error must be finite and nonnegative, got "
        f"{loo!r}")
    if loo < best_loo - 1e-6:
      best_loo  = loo
      best_supp = list(active)
      best_beta = beta.copy()
      since = 0
    else:
      since += 1
    if (since >= patience or len(active) >= term_limit
        or len(active) >= available_count):
      break

    # next term: candidate column most correlated with the residual
    # (scaled by its norm); never re-pick an active one.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
      score = np.abs(Psi.T @ resid) / cn
    inactive = usable.copy()
    inactive[active] = False
    if not np.isfinite(score[inactive]).all():
      raise ValueError("PCE inactive candidate scores became nonfinite")
    remaining = np.flatnonzero(inactive)
    next_index = int(remaining[np.argmax(score[remaining])])
    active.append(next_index)

  support = np.asarray(best_supp, dtype=int)
  if (best_beta is None or support.ndim != 1 or support.size < 1
      or np.unique(support).size != support.size
      or (support < 0).any() or (support >= n_candidates).any()
      or np.asarray(best_beta).shape != support.shape
      or not np.isfinite(best_beta).all() or not np.isfinite(best_loo)):
    raise ValueError(
      "PCE selection did not produce one finite coefficient for each unique "
      "candidate in its support")
  return support, np.asarray(best_beta, dtype=np.float64), float(best_loo)


class PCEEmulator(nn.Module):
  """
  Sparse-Legendre Polynomial Chaos Expansion (PCE) emulator, the
  analytic "base" of the NPCE (Neural PCE). It maps the cosmological
  parameters to the whitened data vector with no neural network. Each
  selected support is refit by least squares as the greedy loop adds terms.

  A polynomial chaos expansion writes a quantity as a sum of
  orthogonal polynomials of the inputs. Here (eqs 9-11) each
  compressed dv coefficient lambda_i is expanded in normalized
  Legendre polynomials of the 12 parameters mapped to [-1, 1]:
    lambda_i(theta) ~ sum_alpha eta_{i,alpha} Psi_alpha(x)
  Psi_alpha is a product of 1-D Legendre polynomials (eq 10) and
  alpha runs over a sparse multi-index set from a hyperbolic q-norm
  + max-interaction truncation, pruned by greedy residual correlation with a leave-one-out
  criterion (eq 11). Sparse = only a handful of terms survive
  (sparsity-of-effects), making the fit data-efficient and
  overfit-resistant.

  What the lambda_i are: the dv targets are covariance-whitened
  (geom.encode), ensemble-centered, and SVD-compressed to K leading
  modes; those K amplitudes are the lambda_i (eq-9
  principal-component amplitudes). Compressing in the whitened basis
  is deliberate, that basis is the chi2 metric (chi2 == ||.||^2)
  so (a) the least-squares PCE on each mode directly minimizes
  the expected chi2, and (b) dropping a mode costs its
  singular-value^2 / N in mean chi2, bounding the truncation error
  in the reported metric.

  Two design rules, learned the hard way (see the NPCE notes):
    - Keep only well-predicted modes. Mode 0 (the overall amplitude,
      ~A_s/S_8 scaling) is smooth and cleanly polynomial-
      predictable; the higher "shape" modes often are not. A mode
      kept with a poor fit injects more error than it removes, so
      only modes with relative LOO < loo_max enter the base; the
      rest go to the NPCE refiner, which corrects the full dv and
      backstops everything dropped.
    - Keep the degree low. A high-degree Legendre fit Runge-
      oscillates, and subtracting a wiggly base makes the refiner's
      residual harder. Degree (p_max) is the smoothness knob; term
      count (max_terms) only adds richness within that degree, so
      max_terms is generous and the LOO, not the cap, decides each
      mode's term count.

  Drop-in model(X) -> whitened dv: X is the pgeom-whitened parameter
  batch (the input the SGD models see). Build with the from_training
  classmethod and wrap as the base of an NPCE loss (PCEResidualChi2 =
  additive, PCERatioChi2 = multiplicative).

  Forward shape flow:

      X  (B, n_dim)              pgeom-whitened parameters
         │  box map + clamp      2 (X - lo) / (hi - lo) - 1
         ▼
      Xm (B, n_dim)              Legendre domain [-1, 1]
         │  pce_design           products of 1-D Legendre factors
         ▼
      Psi (B, n_terms)           the sparse polynomial basis
         │  @ C                  fitted coefficients, mode by mode
         ▼
      Z  (B, K)                  the K mode amplitudes (lambda_i)
         │  @ Vk^T  + Ybar       SVD reconstruction + ensemble mean
         ▼
      out (B, n_keep)            whitened data vector

      (legend: B = batch rows; n_dim = number of cosmological
       parameters (LSST-Y1 example: 12); n_terms = multi-indices in
       the sparse candidate basis; K = SVD modes kept by the LOO
       gate; n_keep = kept data-vector length the network emulates;
       lo / hi / C / Vk / Ybar = the frozen buffers listed below.)

  Buffers (frozen; move with .to(device), never trained):
    lo, hi      = per-parameter [-1, 1] box-map bounds.
    multi_index = (n_terms, n_dim) Legendre degree exponents.
    C           = (n_terms, K) sparse coefficient matrix (zero off
                  each mode's support).
    Vk          = (n_keep, K) leading SVD modes the amplitudes
                  reconstruct against.
    Ybar        = (n_keep,) training-ensemble mean of the whitened dv.
  """

  def __init__(self, lo, hi, multi_index, C, Vk, Ybar):
    """Store the fitted PCE as registered buffers (no training).

    Alternative constructor from_training fits these; __init__ just
    stores them so state_dict save/load round-trips.

    Arguments:
      lo, hi      = per-input min / max used to map X to [-1, 1].
      multi_index = (n_terms, n_dim) Legendre multi-indices of the
                    selected basis terms.
      C           = (n_terms, k) coefficients onto the k leading
                    output modes.
      Vk          = (out_dim, k) the k retained output eigenvectors.
      Ybar        = (out_dim,) output mean added back after the
                    mode expansion.
    """
    super().__init__()
    self.register_buffer("lo", lo)
    self.register_buffer("hi", hi)
    self.register_buffer("multi_index", multi_index)
    self.register_buffer("C", C)
    self.register_buffer("Vk", Vk)
    self.register_buffer("Ybar", Ybar)

  @classmethod
  def from_training(cls, device, X_white, Y_white,
                    p_max=4, r_max=2, q=0.5,
                    k_max=40, loo_max=0.05,
                    max_terms=30, max_fail=4, silent=False):
    """
    Fit from whitened training inputs/targets.

    Fit pipeline (iterative greedy fitting, no gradient descent):

        X_white (N, n_dim)      Y_white (N, n_keep)
           │                       │  center on the mean Ybar
           │  box map to [-1,1]    ▼
           ▼                    Yc (N, n_keep)
        Psi (N, n_terms)           │  SVD -> modes Vt, variances S^2
           │                       ▼
           │                    z_k = Yc @ Vt[k]  mode amplitudes
           │                       │
           └───────────┬───────────┘
                       │  per mode k: select_lars_loo(Psi, z_k)
                       ▼
        keep mode k iff loo < loo_max
        (stop after max_fail consecutive misses)
                       │
                       ▼
        frozen buffers: lo, hi, multi_index, C, Vk, Ybar

        (legend: N = training rows; n_dim / n_terms / n_keep as in
         the class docstring; z_k = the k-th SVD amplitude over the
         training set; loo = relative leave-one-out MSE from
         select_lars_loo; Vt = the SVD's right singular vectors,
         row per mode.)

    Arguments:
      device   = device the buffers live on.
      X_white  = (N, n_dim) pgeom-whitened training params.
      Y_white  = (N, n_keep) covariance-whitened targets.
      p_max    = max total degree (smoothness knob); low (3-6).
      r_max    = max interaction order (vars per term).
      q        = sparsity exponent in (0,1] (q=1 = total degree;
                 smaller = sparser; example in pce_multi_index).
      k_max    = max leading SVD modes to try.
      loo_max  = finite positive limit; keep a mode only if relative LOO in
                 the saved float32 bounds, coefficients, and multiplication
                 is strictly below this value.
      max_terms = per-mode active-set cap.
      max_fail = stop after this many consecutive gate failures
                 (leading modes are the predictable ones, so a run
                 of misses means the rest miss too, avoids fitting
                 modes that will only be dropped).
      silent   = suppress the fit report.
    Returns:
      a fitted PCEEmulator on `device`.

    Raises:
      ValueError when the training arrays or fit limits are invalid, the
      numerical selection becomes nonfinite, or no attempted output mode has
      leave-one-out error strictly below `loo_max`. A failed fit returns no
      fallback model.
    """
    p_max = _positive_int(p_max, name="PCE p_max")
    r_max = _positive_int(r_max, name="PCE r_max")
    k_max = _positive_int(k_max, name="PCE k_max")
    max_terms = _positive_int(max_terms, name="PCE max_terms")
    max_fail = _positive_int(max_fail, name="PCE max_fail")
    q = _positive_finite_real(q, name="PCE q")
    if q > 1.0:
      raise ValueError(
        f"PCE q must be no larger than 1, got {q!r}")
    loo_max = _positive_finite_real(loo_max, name="PCE loo_max")

    Xn = _finite_real_array(
      X_white, name="PCE whitened training inputs", ndim=2)
    Yn = _finite_real_array(
      Y_white, name="PCE whitened training targets", ndim=2)
    if Xn.shape[0] != Yn.shape[0]:
      raise ValueError(
        "PCE training input and target row counts must match: got "
        f"{Xn.shape[0]} and {Yn.shape[0]}")
    if Xn.shape[0] < 2 or Xn.shape[1] < 1 or Yn.shape[1] < 1:
      raise ValueError(
        "PCE training requires at least two rows, one input column, and one "
        f"target column; got input shape {Xn.shape} and target shape {Yn.shape}")
    N, n_dim = Xn.shape

    def float32_buffer(value, name):
      """
      Cast one fitted float64 array to the saved float32 precision.

      The fit runs in float64 but the artifact stores float32; a value
      that overflows to infinity in the narrower type would poison every
      later prediction, so the cast refuses instead of saving it.

      Arguments:
        value = the fitted array (numpy or tensor).
        name  = what the array is, named in the refusal.

      Returns:
        a float32 torch tensor with every value finite.
      """
      tensor = torch.as_tensor(value, dtype=torch.float32)
      if not torch.isfinite(tensor).all():
        raise ValueError(
          f"PCE {name} became nonfinite when converted to float32")
      return tensor

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
      low = Xn.min(0)
      high = Xn.max(0)
      mid = 0.5 * (low + high)
      half = 0.5 * (high - low) * 1.05 + 1e-12
      lo, hi = mid - half, mid + half
    if (not np.isfinite(lo).all() or not np.isfinite(hi).all()
        or not (hi > lo).all()):
      raise ValueError(
        "PCE input bounds must remain finite and distinct during box mapping")

    # Prediction uses float32 bounds and inputs. Build the training design with
    # that exact stored arithmetic so the acceptance score describes the
    # polynomial that will actually run after saving and rebuilding.
    lo_t = float32_buffer(lo, "lower input bounds")
    hi_t = float32_buffer(hi, "upper input bounds")
    if not torch.all(hi_t > lo_t):
      raise ValueError(
        "PCE input bounds must remain distinct after conversion to float32")
    X_t = float32_buffer(Xn, "whitened training inputs")
    Xm_t = 2.0 * (X_t - lo_t) / (hi_t - lo_t) - 1.0
    Xm_t = Xm_t.clamp(-1.0, 1.0)
    if not torch.isfinite(Xm_t).all():
      raise ValueError(
        "PCE input mapping became nonfinite in saved float32 arithmetic")

    mi   = pce_multi_index(n_dim=n_dim, p_max=p_max, r_max=r_max, q=q)
    mi_t = torch.as_tensor(mi, dtype=torch.long)
    Psi = pce_design(Xm=Xm_t, multi_index=mi_t).to(torch.float64).numpy()

    with np.errstate(over="ignore", invalid="ignore"):
      Ybar = Yn.mean(0)
      Yc = Yn - Ybar
    if not np.isfinite(Ybar).all() or not np.isfinite(Yc).all():
      raise ValueError("PCE target centering produced nonfinite values")
    try:
      U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    except np.linalg.LinAlgError as error:
      raise ValueError("PCE target SVD did not converge") from error
    if (not np.isfinite(U).all() or not np.isfinite(S).all()
        or not np.isfinite(Vt).all()):
      raise ValueError("PCE target SVD produced nonfinite values")
    with np.errstate(over="ignore", invalid="ignore"):
      var = S ** 2
    if not np.isfinite(var).all():
      raise ValueError("PCE target-mode variances became nonfinite")

    # fit leading modes; keep the well-predicted ones (loo <
    # loo_max). Stop after max_fail consecutive misses.
    kfit = min(k_max, int(np.count_nonzero(var > 0.0)))
    if kfit < 1:
      raise ValueError(
        "PCE fit refused: training targets contain no varying output mode")
    cols, kept, loos, all_loo, support_sizes = [], [], [], [], []
    kept_supports, kept_targets, kept_betas = [], [], []
    fails = 0
    for k in range(kfit):
      zk = Yc @ Vt[k]
      supp, beta, loo = select_lars_loo(Psi=Psi, y=zk, max_terms=max_terms)
      supp = np.asarray(supp)
      beta = np.asarray(beta, dtype=np.float64)
      if (supp.ndim != 1 or supp.size < 1 or supp.dtype.kind not in "iu"
          or np.unique(supp).size != supp.size
          or (supp < 0).any() or (supp >= Psi.shape[1]).any()
          or beta.shape != supp.shape or not np.isfinite(beta).all()
          or isinstance(loo, (bool, np.bool_))
          or not isinstance(loo, numbers.Real)
          or not np.isfinite(float(loo)) or float(loo) < 0.0):
        raise ValueError(
          f"PCE mode {k} selection must return unique in-range support, "
          "one finite coefficient per support index, and a finite "
          "nonnegative leave-one-out error")
      supp = supp.astype(int, copy=False)
      # The design already uses the saved float32 bounds. Judge the float32
      # coefficient and multiplication that will actually run, not a slightly
      # different float64 precursor.
      with np.errstate(over="ignore", invalid="ignore"):
        saved_beta = beta.astype(np.float32).astype(np.float64)
      if not np.isfinite(saved_beta).all():
        raise ValueError(
          f"PCE mode {k} coefficients became nonfinite in float32")
      saved_loo = _fixed_fit_loo(Psi, zk, supp, saved_beta)
      all_loo.append(saved_loo)
      if saved_loo < loo_max:
        col = np.zeros(Psi.shape[1])
        col[supp] = saved_beta
        cols.append(col)
        kept.append(k)
        loos.append(saved_loo)
        support_sizes.append(int(supp.size))
        kept_supports.append(supp.copy())
        kept_targets.append(zk.copy())
        kept_betas.append(saved_beta.copy())
        fails = 0
      else:
        fails += 1
        if fails >= max_fail:
          break
    if not cols:
      best_index = int(np.argmin(all_loo))
      tried = ", ".join(
        f"mode {index}: {value:.6g}"
        for index, value in enumerate(all_loo))
      raise ValueError(
        "PCE fit refused: no mode passed the strict leave-one-out limit "
        f"loo_max={loo_max!r}; best attempted LOO={all_loo[best_index]:.6g} "
        f"at mode {best_index}; modes tried: [{tried}]")

    # Forward evaluates every retained mode in one dense matrix product. The
    # float32 reduction can differ from evaluating each column separately.
    # Remove the worst joint failure, rebuild the narrower matrix, and repeat;
    # a rejected output mode belongs to the neural refiner, not the PCE base.
    joint_rejections = []
    while cols:
      C = np.stack(cols, axis=1)
      K = len(kept)
      with torch.no_grad():
        joint_prediction = (
          torch.from_numpy(Psi.astype(np.float32))
          @ torch.from_numpy(C.astype(np.float32))
        ).to(torch.float64).numpy()
      joint_loos = []
      for column, (mode, target, support, beta) in enumerate(zip(
          kept, kept_targets, kept_supports, kept_betas)):
        joint_loo = _fixed_fit_loo(
          Psi, target, support, beta,
          saved_prediction=joint_prediction[:, column])
        joint_loos.append(joint_loo)
        all_loo[mode] = joint_loo
      failing = [
        column for column, value in enumerate(joint_loos)
        if not value < loo_max
      ]
      if not failing:
        loos = joint_loos
        break
      rejected = max(failing, key=lambda column: joint_loos[column])
      joint_rejections.append(
        f"mode {kept[rejected]}: {joint_loos[rejected]:.6g}")
      for values in (
          cols, kept, loos, support_sizes, kept_supports, kept_targets,
          kept_betas):
        del values[rejected]
    if not cols:
      best_index = int(np.argmin(all_loo))
      tried = ", ".join(
        f"mode {index}: {value:.6g}"
        for index, value in enumerate(all_loo))
      raise ValueError(
        "PCE fit refused: no mode passed the final saved-format "
        f"leave-one-out limit loo_max={loo_max!r}; best attempted "
        f"LOO={all_loo[best_index]:.6g} at mode {best_index}; modes tried: "
        f"[{tried}]; joint failures: ["
        + ", ".join(joint_rejections) + "]")

    Vk = Vt[kept].T
    K = len(kept)
    drop_chi2 = float((var.sum() - var[kept].sum()) / N)
    if (not np.isfinite(C).all() or not np.isfinite(Vk).all()
        or not np.isfinite(Ybar).all() or not np.isfinite(drop_chi2)
        or not all(np.isfinite(value) and value < loo_max
                   for value in loos)):
      raise ValueError(
        "PCE retained modes must have finite coefficients and "
        "leave-one-out errors strictly below loo_max")

    if not silent:
      act = np.asarray(support_sizes, dtype=int)
      tried_vals = []
      for l in all_loo[:8]:
        tried_vals.append(f"{l:.2e}")
      tried = ", ".join(tried_vals)
      print(f"PCE fit: N {N}  n_dim {n_dim}  "
            f"candidates {Psi.shape[1]}  fit {len(all_loo)}")
      print(f"  kept {K} (loo<{loo_max})  "
            f"mean dropped chi2 {drop_chi2:.4f}")
      print(f"  active/mode: median {int(np.median(act))}"
            f"  max {int(act.max())}")
      print(f"  tried-mode LOO[:8]: [{tried}]")

    return cls(lo=lo_t.to(device),
               hi=hi_t.to(device),
               multi_index=mi_t.to(device),
               C=float32_buffer(C, "coefficient matrix").to(device),
               Vk=float32_buffer(Vk, "output modes").to(device),
               Ybar=float32_buffer(Ybar, "target mean").to(device))

  def state(self):
    """The frozen buffers, keyed as from_state expects (h5 persistence).

    Mirrors the geometry classes' state(): a flat dict of the six
    registered buffers, so save_emulator can write a "pce" h5 group and
    from_state rebuilds the base at inference with no refit and no
    cosmolike.

    Returns:
      dict with lo / hi / multi_index / C / Vk / Ybar (the six frozen
      buffers, as tensors).
    """
    return {"lo": self.lo,
            "hi": self.hi,
            "multi_index": self.multi_index,
            "C": self.C,
            "Vk": self.Vk,
            "Ybar": self.Ybar}

  @classmethod
  def from_state(cls, state, device):
    """Rebuild a frozen PCEEmulator from a saved state() dict.

    The persistence inverse of state(): mirrors the geometry classes'
    from_state, so a saved run reconstructs base + refiner off the h5
    with no refit and no cosmolike. dtypes match from_training (lo / hi
    / C / Vk / Ybar float32, multi_index long).

    Arguments:
      state  = mapping with lo / hi / multi_index / C / Vk / Ybar
               (numpy arrays or tensors, as read back from the h5).
      device = device the rebuilt buffers live on.

    Returns:
      a PCEEmulator on `device`.
    """
    def t(v, dtype):
      # one tensor per stored buffer, on the requested device, in the
      # dtype the forward pass expects.
      return torch.as_tensor(v, dtype=dtype, device=device)
    return cls(lo=t(state["lo"], torch.float32),
               hi=t(state["hi"], torch.float32),
               multi_index=t(state["multi_index"], torch.long),
               C=t(state["C"], torch.float32),
               Vk=t(state["Vk"], torch.float32),
               Ybar=t(state["Ybar"], torch.float32))

  def forward(self, X):
    """Evaluate the closed-form PCE base at the given inputs.

    Arguments:
      X = (B, n_dim) whitened inputs (the same whitening the
          training used); mapped to [-1, 1] for the Legendre basis.

    Returns:
      (B, out_dim) base prediction Ybar + (Psi @ C) @ Vk^T.
    """
    # Use the stored number format for every call, matching the arithmetic
    # used by the fit-acceptance check before this base was saved.
    X = X.to(device=self.lo.device, dtype=self.lo.dtype)
    # Map each input to [-1, 1] (Legendre domain), then clamp so a test point
    # just outside the training box stays in range.
    Xm  = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
    Xm  = Xm.clamp(-1.0, 1.0)
    Psi = pce_design(Xm=Xm, multi_index=self.multi_index)
    Z   = Psi @ self.C
    return self.Ybar + Z @ self.Vk.t()
