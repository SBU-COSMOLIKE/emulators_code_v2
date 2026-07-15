#!/usr/bin/env python3
"""Check that validation metrics do not depend on batch partitioning.

The acceptance part drives the real ``eval_val`` entry point over the same
twelve validation rows in three shapes: one full batch, three equal batches,
and a ragged final batch.  It compares every published metric and the real
training histories with an independent float64 calculation.

The fixture also drives ``eval_source_chi2``, the production per-row scoring
surface used by the diagnostics.  A temporary observer records the score and
source-row arrays that ``eval_val`` sends through its production score-domain
boundary.  This proves that the aggregates came from the expected distinct
rows without adding a gate-only scoring loop.  A permuted source-row list
keeps row association observable.

Two mutation arms alter only the production score stream seen by
``eval_val``.  One drops an equal-sized middle batch.  The other reverses its
scores without reversing their source-row labels.  The independent reference
and diagnostic scorer stay unchanged, and both mutations must be rejected.

The ordinary-median part separately retains the even- and odd-row controls.
CUDA timing remains informational because it has no numerical acceptance
bound.  The script emits exactly one reserved evidence terminal for each of
the four board-declared legs and exits nonzero if either logical leg fails.

Home note: ai/notes/training-stack.md#eval-batch-invariance-evidence.
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn

import emulator.training as training
from emulator.training import (derive_eval_bs,
                               eval_source_chi2,
                               eval_val,
                               ordinary_median,
                               training_loop_batched,
                               _EVAL_BS_TARGET)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTOL = 1.0e-6
ATOL = 1.0e-7
N_INPUT = 2

ROW_ORDER = np.array(
  [7, 2, 10, 0, 5, 11, 1, 9, 4, 8, 3, 6],
  dtype=np.int64)
SCORE_VALUES = np.array(
  [0.05, 0.15, 0.25, 0.35, 0.75, 1.25,
   1.75, 2.25, 4.5, 7.0, 11.0, 18.0],
  dtype=np.float32)
THRESHOLD_VALUES = np.array([0.2, 1.0, 5.0], dtype=np.float32)

PARTITIONS = (
  {
    "name": "one full batch",
    "load": 12,
    "batch_size": 12,
    "expected_rows": np.array(
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64),
  },
  {
    "name": "three equal batches",
    "load": 4,
    "batch_size": 4,
    "expected_rows": np.array(
      [0, 2, 7, 10, 1, 5, 9, 11, 3, 4, 6, 8], dtype=np.int64),
  },
  {
    "name": "ragged final batch",
    "load": 5,
    "batch_size": 5,
    "expected_rows": np.array(
      [0, 2, 5, 7, 10, 1, 4, 8, 9, 11, 3, 6], dtype=np.int64),
  },
)

TRIM = {
  "shape": "linear",
  "start": 0.0,
  "end": 0.0,
  "hold_epochs": 1,
  "anneal_epochs": 1,
}
FOCUS = {
  "shape": "linear",
  "start": 0.0,
  "end": 0.0,
  "hold_epochs": 1,
  "anneal_epochs": 1,
  "kappa": 1.0,
}


class IdentityParamGeometry:
  """Leave the synthetic source parameters unchanged.

  ``eval_source_chi2`` calls ``encode`` before model evaluation.  The first
  input column is the source-row identity, so leaving it unchanged makes the
  diagnostic return independently row-addressable.
  """

  def encode(self, values):
    """Return the input tensor unchanged.

    Arguments:
      values = a two-dimensional parameter tensor.

    Returns:
      The same tensor object.
    """
    return values


class TensorRows:
  """Load named rows from one in-memory Torch tensor.

  Arguments:
    values = the complete row-indexed tensor.
  """

  def __init__(self, values):
    self.values = values

  def __call__(self, rows):
    """Return the selected rows.

    Arguments:
      rows = a one-dimensional integer array of row coordinates.

    Returns:
      The indexed tensor rows in the requested order.
    """
    return self.values[rows]


class PublishedScoreLoss:
  """Publish one distinct score coupled to both inputs of each source row."""

  needs_params = False

  def encode(self, values):
    """Leave the synthetic target unchanged for diagnostic scoring.

    Arguments:
      values = the target tensor.

    Returns:
      The same target tensor.
    """
    return values

  def chi2(self, pred, target):
    """Average the matching model and target copies of the known score.

    Both copies equal the known score on an honest row.  If parameters and
    targets become associated with different source rows, their average moves
    away from that row's expected value.  The test therefore observes both
    loaders rather than accepting a target-only score.

    Arguments:
      pred   = the model prediction.  Column zero stores the score copy.
      target = the target tensor.  Column zero stores the other score copy.

    Returns:
      A one-dimensional float32 score tensor.
    """
    return 0.5 * (pred[:, 0] + target[:, 0])

  def loss(self,
           pred,
           target,
           mode="sqrt",
           trim=None,
           focus=None,
           focus_scale=None,
           berhu_knot=None,
           berhu_cap=None,
           berhu_s=None):
    """Return a finite zero with a real gradient for the history probe.

    Arguments:
      pred        = the model prediction tensor.
      target      = the target tensor.  Unused by this training fixture.
      mode        = the training loss mode.  Unused here.
      trim        = the trim fraction.  Unused here.
      focus       = the focal exponent.  Unused here.
      focus_scale = the focal scale.  Unused here.
      berhu_knot  = the lower berHu knot.  Unused here.
      berhu_cap   = the upper berHu cap.  Unused here.
      berhu_s     = the berHu blend.  Unused here.

    Returns:
      A finite scalar whose derivative with respect to ``pred`` is zero.
    """
    del target, mode, trim, focus, focus_scale, berhu_knot, berhu_cap, berhu_s
    return pred.sum() * 0.0


class RecordingPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
  """Record the validation median passed to the real plateau scheduler."""

  def __init__(self, optimizer):
    """Initialize the scheduler and its metric record.

    Arguments:
      optimizer = the optimizer updated by the training loop.
    """
    self.metrics_seen = []
    super().__init__(optimizer=optimizer)

  def step(self, metrics, epoch=None):
    """Record one ranking metric, then run the shipped scheduler method.

    Arguments:
      metrics = the validation median supplied by the training loop.
      epoch   = an optional explicit epoch number.

    Returns:
      The superclass result, which is ``None``.
    """
    self.metrics_seen.append(float(metrics))
    return super().step(metrics=metrics, epoch=epoch)


class ValidationScreenObserver:
  """Observe or mutate the real validation score-domain boundary.

  The observer replaces ``emulator.training.screen_chi2`` only for one
  ``eval_val`` call and delegates to the real function.  The captured score
  tensor is the post-batch, post-padding-slice production tensor that the
  published reductions consume.

  Arguments:
    original = the real ``screen_chi2`` function.
    mutation = ``None``, ``drop-middle``, or ``reassociate-middle``.
  """

  def __init__(self, original, mutation=None):
    self.original = original
    self.mutation = mutation
    self.scores = None
    self.positions = None

  def __call__(self, chi2, loss, label, positions=None):
    """Capture the validation arrays and delegate to the real boundary.

    Arguments:
      chi2      = the production per-row score tensor.
      loss      = the loss object that owns the score domain.
      label     = the score consumer name.
      positions = source rows aligned with ``chi2``.

    Returns:
      The real ``screen_chi2`` result, possibly after the requested mutation.
    """
    checked = chi2
    checked_positions = positions
    if label == "validation":
      if self.mutation == "drop-middle":
        checked = torch.cat((chi2[:4], chi2[8:]))
        checked_positions = np.concatenate((positions[:4], positions[8:]))
      elif self.mutation == "reassociate-middle":
        checked = chi2.clone()
        checked[4:8] = torch.flip(checked[4:8], dims=(0,))
    screened = self.original(chi2=checked,
                             loss=loss,
                             label=label,
                             positions=checked_positions)
    if label == "validation":
      # Capture the returned tensor, not the boundary's input.  This is the
      # normalized production surface that eval_val actually reduces.
      self.scores = screened.detach().cpu().numpy().copy()
      self.positions = np.asarray(checked_positions, dtype=np.int64).copy()
    return screened


def make_fixture(device):
  """Build the row-addressable parameters, score targets, model, and loss.

  Arguments:
    device = the Torch device for model evaluation.

  Returns:
    ``(parameters, targets, model, loss)``.  Parameters and targets are Torch
    tensors on ``device``; model and loss drive both production scorers.
  """
  row_ids = np.arange(SCORE_VALUES.size, dtype=np.float32)
  parameters_np = np.column_stack((row_ids, SCORE_VALUES))
  parameters = torch.from_numpy(parameters_np).to(device=device)
  targets = torch.from_numpy(SCORE_VALUES.reshape(-1, 1)).to(device=device)
  model = nn.Linear(N_INPUT, 1, bias=False).to(device=device)
  with torch.no_grad():
    model.weight.zero_()
    model.weight[0, 1] = 1.0
  model.eval()
  return parameters, targets, model, PublishedScoreLoss()


def make_eval_data(parameters, targets):
  """Build the loader dictionary accepted by ``eval_val``.

  Arguments:
    parameters = the complete parameter tensor.
    targets    = the complete target tensor.

  Returns:
    A dictionary containing the two row loaders and the permuted row list.
  """
  return {
    "load_C": TensorRows(values=parameters),
    "load_dv": TensorRows(values=targets),
    "idx": ROW_ORDER.copy(),
  }


def independent_reference():
  """Calculate all published validation values in independent float64 NumPy.

  Returns:
    A dictionary with ``median``, ``mean``, and ``fractions``.
  """
  scores = SCORE_VALUES.astype(np.float64)
  thresholds = THRESHOLD_VALUES.astype(np.float64)
  fractions = np.mean(scores[:, None] > thresholds[None, :], axis=0)
  return {
    "median": float(np.median(scores)),
    "mean": float(np.mean(scores, dtype=np.float64)),
    "fractions": fractions,
  }


def diagnostic_row_map(parameters, targets, model, loss):
  """Read the production diagnostics' row-aligned per-sample score surface.

  Arguments:
    parameters = the complete parameter tensor.
    targets    = the complete target tensor.
    model      = the fixture model.
    loss       = the fixture loss.

  Returns:
    A ``{source row: float64 score}`` dictionary.
  """
  source = {
    "C": parameters.detach().cpu().numpy().astype(np.float64),
    "dv": targets.detach().cpu().numpy(),
    "idx": ROW_ORDER.copy(),
  }
  raw_parameters, scores = eval_source_chi2(
    model=model,
    param_geometry=IdentityParamGeometry(),
    chi2fn=loss,
    source=source,
    device=DEVICE,
    bs=5)
  result = {}
  for row_values, score in zip(raw_parameters, scores):
    result[int(row_values[0])] = float(score)
  return result


def run_eval(parameters, targets, model, loss, partition, mutation=None):
  """Run real ``eval_val`` once while observing its per-row score boundary.

  Arguments:
    parameters = the complete parameter tensor.
    targets    = the complete target tensor.
    model      = the fixture model.
    loss       = the fixture loss.
    partition  = one ``PARTITIONS`` dictionary.
    mutation   = optional observer mutation name.

  Returns:
    A dictionary containing the published metrics and captured row arrays.
  """
  observer = ValidationScreenObserver(original=training.screen_chi2,
                                      mutation=mutation)
  training.screen_chi2 = observer
  try:
    median, mean, fractions = eval_val(
      model=model,
      lossfn=loss,
      data=make_eval_data(parameters=parameters, targets=targets),
      load=partition["load"],
      bs=partition["batch_size"],
      # eval_val moves the scores to CPU before this comparison.  Keep the
      # thresholds on CPU even when the model runs on CUDA.
      thresholds=torch.from_numpy(THRESHOLD_VALUES))
  finally:
    training.screen_chi2 = observer.original
  return {
    "median": median,
    "mean": mean,
    "fractions": fractions.detach().cpu().numpy(),
    "scores": observer.scores,
    "positions": observer.positions,
  }


def result_matches_reference(result, reference, row_map, partition):
  """Compare one real evaluation with every independent expected value.

  Arguments:
    result    = the dictionary returned by ``run_eval``.
    reference = the dictionary returned by ``independent_reference``.
    row_map   = the production diagnostic score keyed by source row.
    partition = the partition dictionary used for this evaluation.

  Returns:
    True only when metrics, row order, row identities, and scores all agree.
  """
  scalar_ok = np.isclose(result["median"],
                         reference["median"],
                         rtol=RTOL,
                         atol=ATOL)
  scalar_ok = scalar_ok and np.isclose(result["mean"],
                                       reference["mean"],
                                       rtol=RTOL,
                                       atol=ATOL)
  fraction_ok = np.allclose(result["fractions"],
                            reference["fractions"],
                            rtol=RTOL,
                            atol=ATOL)
  wanted_positions = partition["expected_rows"]
  order_ok = np.array_equal(result["positions"], wanted_positions)
  rows_ok = result["scores"] is not None
  rows_ok = rows_ok and result["scores"].size == wanted_positions.size
  if rows_ok:
    for position, score in zip(result["positions"], result["scores"]):
      expected_score = row_map.get(int(position))
      if expected_score is None:
        rows_ok = False
        break
      if not np.isclose(float(score), expected_score, rtol=0.0, atol=0.0):
        rows_ok = False
        break
  return bool(scalar_ok and fraction_ok and order_ok and rows_ok)


def run_history(parameters, targets, partition):
  """Run one real training epoch and return its ranking and history values.

  The model has a finite zero gradient, so its validation scores stay fixed.
  ``training_loop_batched`` still executes the real baseline evaluation,
  epoch evaluation, history appends, best-model ranking, and scheduler step.

  Arguments:
    parameters = the complete parameter tensor.
    targets    = the complete target tensor.
    partition  = one ``PARTITIONS`` dictionary.  Its load makes the training
                 loop derive the matching validation batch size.

  Returns:
    A dictionary containing the one-epoch median, mean, fractions, and the
    metric seen by the real plateau scheduler.
  """
  _, _, model, loss = make_fixture(device=DEVICE)
  optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
  scheduler = RecordingPlateau(optimizer=optimizer)
  source = make_eval_data(parameters=parameters, targets=targets)
  source["load"] = partition["load"]
  data = {
    "train": dict(source),
    "val": dict(source),
  }
  _, medians, means, fractions = training_loop_batched(
    nepochs=1,
    optimizer=optimizer,
    scheduler=scheduler,
    model=model,
    bs=1,
    lossfn=loss,
    mode="sqrt",
    data=data,
    trim_opts=TRIM,
    focus_opts=FOCUS,
    # The validation reductions live on CPU, including on a CUDA run.
    thresholds=torch.from_numpy(THRESHOLD_VALUES),
    silent=True)
  return {
    "median": medians[0],
    "mean": means[0],
    "fractions": fractions[0].detach().cpu().numpy(),
    "scheduler_metrics": scheduler.metrics_seen,
  }


def history_matches_reference(history, reference):
  """Compare every returned history and the scheduler input with float64 truth.

  Arguments:
    history   = the dictionary returned by ``run_history``.
    reference = the independent float64 reference dictionary.

  Returns:
    True when all history and ranking values agree.
  """
  median_ok = np.isclose(history["median"],
                         reference["median"],
                         rtol=RTOL,
                         atol=ATOL)
  mean_ok = np.isclose(history["mean"],
                       reference["mean"],
                       rtol=RTOL,
                       atol=ATOL)
  fraction_ok = np.allclose(history["fractions"],
                            reference["fractions"],
                            rtol=RTOL,
                            atol=ATOL)
  scheduler_ok = len(history["scheduler_metrics"]) == 1
  if scheduler_ok:
    scheduler_ok = np.isclose(history["scheduler_metrics"][0],
                              reference["median"],
                              rtol=RTOL,
                              atol=ATOL)
  return bool(median_ok and mean_ok and fraction_ok and scheduler_ok)


def check_partition_invariance():
  """Run the three real partitions, histories, and batch-stream mutations.

  Returns:
    True when every production and catch-power comparison passes.
  """
  parameters, targets, model, loss = make_fixture(device=DEVICE)
  reference = independent_reference()
  row_map = diagnostic_row_map(parameters=parameters,
                               targets=targets,
                               model=model,
                               loss=loss)
  print("  float64 reference: median=%.9g mean=%.9g fractions=%s"
        % (reference["median"],
           reference["mean"],
           np.array2string(reference["fractions"], precision=9)))
  expected_map = {}
  for row in range(SCORE_VALUES.size):
    expected_map[row] = float(SCORE_VALUES[row])
  diagnostic_ok = row_map == expected_map
  print("  production diagnostic per-row surface:",
        "PASS" if diagnostic_ok else "FAIL")
  print("    row-to-score map=" + str(row_map))

  all_ok = diagnostic_ok
  for partition in PARTITIONS:
    result = run_eval(parameters=parameters,
                      targets=targets,
                      model=model,
                      loss=loss,
                      partition=partition)
    eval_ok = result_matches_reference(result=result,
                                       reference=reference,
                                       row_map=row_map,
                                       partition=partition)
    history = run_history(parameters=parameters,
                          targets=targets,
                          partition=partition)
    history_ok = history_matches_reference(history=history,
                                           reference=reference)
    all_ok = all_ok and eval_ok and history_ok
    print("  " + partition["name"] + ": eval=" + str(eval_ok)
          + " histories=" + str(history_ok)
          + " metrics=(%.9g, %.9g, %s)"
          % (result["median"],
             result["mean"],
             np.array2string(result["fractions"], precision=9)))
    print("    rows=" + str(result["positions"].tolist()))
    print("    history=(%.9g, %.9g, %s) scheduler=%s"
          % (history["median"],
             history["mean"],
             np.array2string(history["fractions"], precision=9),
             str(history["scheduler_metrics"])))

  equal_partition = PARTITIONS[1]
  dropped = run_eval(parameters=parameters,
                     targets=targets,
                     model=model,
                     loss=loss,
                     partition=equal_partition,
                     mutation="drop-middle")
  drop_caught = not result_matches_reference(result=dropped,
                                              reference=reference,
                                              row_map=row_map,
                                              partition=equal_partition)
  reassociated = run_eval(parameters=parameters,
                          targets=targets,
                          model=model,
                          loss=loss,
                          partition=equal_partition,
                          mutation="reassociate-middle")
  reassociation_caught = not result_matches_reference(
    result=reassociated,
    reference=reference,
    row_map=row_map,
    partition=equal_partition)
  print("  mutation, dropped middle batch: caught=" + str(drop_caught))
  print("    observed rows=" + str(dropped["positions"].size)
        + " mean=%.9g" % dropped["mean"])
  print("  mutation, reversed scores under unchanged row labels: caught="
        + str(reassociation_caught))
  print("    middle rows=" + str(reassociated["positions"][4:8].tolist())
        + " scores=" + str(reassociated["scores"][4:8].tolist()))
  return bool(all_ok and drop_caught and reassociation_caught)


def fixed_score_data(values):
  """Build a tiny validation source whose first target column is ``values``.

  Arguments:
    values = a sequence of non-negative per-row scores.

  Returns:
    ``(data, model, loss)`` for a direct ``eval_val`` call.
  """
  value_array = np.asarray(values, dtype=np.float32)
  count = value_array.size
  parameters = torch.zeros(count, N_INPUT, device=DEVICE)
  parameters[:, 1] = torch.from_numpy(value_array).to(device=DEVICE)
  targets = torch.from_numpy(value_array.reshape(-1, 1)).to(device=DEVICE)
  model = nn.Linear(N_INPUT, 1, bias=False).to(device=DEVICE)
  with torch.no_grad():
    model.weight.zero_()
    model.weight[0, 1] = 1.0
  data = {
    "load_C": TensorRows(values=parameters),
    "load_dv": TensorRows(values=targets),
    "idx": np.arange(count),
  }
  return data, model, PublishedScoreLoss()


def check_ordinary_median():
  """Check the even midpoint, odd control, batching, and old reduction arm.

  Returns:
    True when the helper and real evaluation publish the ordinary median.
  """
  even = torch.tensor([0.0, 1.0, 9.0, 10.0])
  odd = torch.tensor([0.0, 1.0, 9.0])
  helper_ok = ordinary_median(even) == 5.0
  helper_ok = helper_ok and ordinary_median(odd) == 1.0

  data_even, model_even, loss_even = fixed_score_data([0.0, 1.0, 9.0, 10.0])
  data_odd, model_odd, loss_odd = fixed_score_data([0.0, 1.0, 9.0])
  thresholds = torch.from_numpy(THRESHOLD_VALUES)
  even_ok = True
  for batch_size in (1, 2, 3, 4):
    median, _, _ = eval_val(model=model_even,
                            lossfn=loss_even,
                            data=data_even,
                            load=4,
                            bs=batch_size,
                            thresholds=thresholds)
    even_ok = even_ok and median == 5.0
  odd_median, _, _ = eval_val(model=model_odd,
                              lossfn=loss_odd,
                              data=data_odd,
                              load=3,
                              bs=2,
                              thresholds=thresholds)
  odd_ok = odd_median == 1.0
  old_median = float(even.median())
  mutation_caught = old_median == 1.0 and old_median != ordinary_median(even)
  print("  helper even=5 and odd=1: " + str(helper_ok))
  print("  real eval_val even across batch sizes 1..4: " + str(even_ok))
  print("  real eval_val odd control: " + str(odd_ok))
  print("  lower-middle Tensor.median mutation: caught="
        + str(mutation_caught))
  return bool(helper_ok and even_ok and odd_ok and mutation_caught)


class TimingLoss:
  """Return the per-row squared-error sum for the informational timing run."""

  needs_params = False

  def chi2(self, pred, target):
    """Calculate one squared-error sum per row.

    Arguments:
      pred   = the model prediction tensor.
      target = the target tensor.

    Returns:
      A one-dimensional per-row score tensor.
    """
    return ((pred - target) ** 2).sum(dim=1)


def run_timing():
  """Print CUDA timings without turning them into an acceptance verdict."""
  if DEVICE.type != "cuda":
    print("  CPU run: CUDA timing is unavailable")
    return

  n_rows = 5000
  n_output = 780
  parameters = torch.randn(n_rows, 8, device=DEVICE)
  targets = torch.randn(n_rows, n_output, device=DEVICE)
  model = nn.Linear(8, n_output).to(device=DEVICE).eval()
  loss = TimingLoss()
  data = {
    "load_C": TensorRows(values=parameters),
    "load_dv": TensorRows(values=targets),
    "idx": np.arange(n_rows),
  }
  thresholds = torch.from_numpy(THRESHOLD_VALUES)
  derived = derive_eval_bs(n_val=n_rows,
                           target=_EVAL_BS_TARGET,
                           load=n_rows)

  def timed(batch_size, repetitions=50):
    """Measure the average duration of one CUDA validation pass.

    Arguments:
      batch_size = rows in one evaluation batch.
      repetitions = timed validation passes after warmup.

    Returns:
      Mean wall-clock seconds per validation pass.
    """
    for _ in range(3):
      eval_val(model=model,
               lossfn=loss,
               data=data,
               load=n_rows,
               bs=batch_size,
               thresholds=thresholds)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
      eval_val(model=model,
               lossfn=loss,
               data=data,
               load=n_rows,
               bs=batch_size,
               thresholds=thresholds)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repetitions

  small_seconds = timed(batch_size=32)
  derived_seconds = timed(batch_size=derived)
  print("  batch 32: %.2f ms" % (small_seconds * 1.0e3))
  print("  batch %d: %.2f ms" % (derived, derived_seconds * 1.0e3))


def main():
  """Run both logical legs, print evidence terminals, and return an exit code.

  Returns:
    Zero when the two logical legs pass, otherwise one.
  """
  torch.manual_seed(0)
  print("device " + str(DEVICE) + ", eval target " + str(_EVAL_BS_TARGET))

  print("\n=== Part 1: production partition invariance ===")
  partition_ok = check_partition_invariance()
  print("Part 1:", "PASS" if partition_ok else "FAIL")
  print("##AID eval-batch-invariance.partition-invariance",
        "PASS" if partition_ok else "FAIL")

  print("\n=== Part 1b: ordinary median ===")
  median_ok = check_ordinary_median()
  print("Part 1b:", "PASS" if median_ok else "FAIL")
  print("##AID eval-batch-invariance.ordinary-median",
        "PASS" if median_ok else "FAIL")

  print("\n=== Part 2: informational CUDA timing ===")
  run_timing()
  print("##AID eval-batch-invariance.cuda-timing UNAVAILABLE CUDA durations "
        "have no acceptance bound")
  print("##AID eval-batch-invariance.production-timing-claim UNAVAILABLE the "
        "gate makes no production-run speed claim")
  return 0 if partition_ok and median_ok else 1


if __name__ == "__main__":
  sys.exit(main())
