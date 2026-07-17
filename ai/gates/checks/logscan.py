"""Pure text acceptance helpers for the gates harness.

A workstation gate runs a driver and then asserts on what the driver
printed: a banner substring the home note quotes verbatim, a set of
required lines, or the byte identity of two runs' selected lines (the
golden determinism proof). These functions are that assertion layer.
They are pure (a text string in, a verdict out), stdlib only, and
import nothing heavy, so the board module can pull them in on the dev
Mac and the harness self-tests can drive them without a GPU.

Every function returns a plain verdict the caller logs: a bool, or a
(bool, detail) pair whose detail string names exactly what was missing
or where two runs first diverged. No function raises on a failed check
(that is the gate runner's job); they only report.

PS: banner = a driver's human-readable status line; selected lines =
the subset of a log matched by a grep-style regex (the home notes
compare only ``^(phase|epoch|best)`` lines, not timestamps); byte
identity = two line lists equal to the character.
"""

import math
import re


def contains(text, needle):
  """Whether a literal substring appears anywhere in the text.

  Arguments:
    text   = the captured run output (one big string).
    needle = the literal substring to find (not a regex); the home
             notes quote banner fragments verbatim, so a literal
             search is what faithfully implements them.

  Returns:
    True if needle is a substring of text, else False.
  """
  return needle in text


def contains_all(text, needles):
  """Whether every required literal substring appears in the text.

  Arguments:
    text    = the captured run output.
    needles = a sequence of literal substrings that must all be
              present (e.g. a trunk banner and a head banner from one
              run).

  Returns:
    (ok, missing) where ok is True only if every needle is present and
    missing is the list of needles that were absent (empty on ok).
  """
  missing = []
  for needle in needles:
    if needle not in text:
      missing.append(needle)
  return (len(missing) == 0, missing)


def search(text, pattern):
  """Whether a regex matches anywhere in the text (multiline).

  Arguments:
    text    = the captured run output.
    pattern = a regular expression; matched with re.MULTILINE so ``^``
              and ``$`` anchor to line boundaries.

  Returns:
    True if the pattern is found, else False.
  """
  return re.search(pattern, text, flags=re.MULTILINE) is not None


def matching_lines(text, pattern):
  """The lines of the text matched by a grep-style regex, in order.

  Mirrors the home notes' ``grep -E '<pattern>'`` prefilter used
  before a diff: a line is kept when the pattern is found ANYWHERE in
  it (grep semantics), not only anchored at the start.

  Arguments:
    text    = the captured run output.
    pattern = the grep-style regex (e.g. ``^(phase|epoch|best)``).

  Returns:
    a list of the whole matching lines, newline-stripped, in file
    order.
  """
  kept = []
  compiled = re.compile(pattern)
  for line in text.splitlines():
    if compiled.search(line) is not None:
      kept.append(line)
  return kept


def finite_sweep_points(text, *, expected_sizes, expected_threshold):
  """Validate the result rows printed by one training-size sweep.

  A worker may report a failed training point as ``nan`` while the parent
  process continues.  Counting result-shaped lines would then mistake two
  failures for a completed sweep.  This parser requires exactly one finite
  fraction in the physical interval [0, 1] for every requested training size.

  Arguments:
    text               = the complete captured sweep output.
    expected_sizes     = the distinct integer training sizes requested.
    expected_threshold = the finite threshold named inside ``f(>...)``.

  Returns:
    ``(ok, detail)``.  The detail names the parsed rows on success and the
    first malformed, duplicate, missing, unexpected, or nonfinite value on
    failure.
  """
  sizes = tuple(int(value) for value in expected_sizes)
  if len(sizes) == 0 or len(set(sizes)) != len(sizes):
    return (False, "expected training sizes must be nonempty and distinct")
  if not math.isfinite(expected_threshold):
    return (False, "expected threshold must be finite")
  pattern = re.compile(
    r"^\s*N_train\s+([0-9]+)\s+f\(>([^)]+)\)\s+([^\s]+)(?:\s|$)")
  observed = {}
  for line in text.splitlines():
    match = pattern.match(line)
    if match is None:
      if re.search(r"N_train\s+[0-9]+\s+failed:", line) is not None:
        return (False, "worker reported a failed sweep point: " + line.strip())
      if "N_train" in line and "f(>" in line:
        return (False, "malformed sweep result row: " + line.strip())
      continue
    size = int(match.group(1))
    try:
      threshold = float(match.group(2))
      fraction = float(match.group(3))
    except ValueError:
      return (False, "malformed numeric sweep row: " + line.strip())
    if size not in sizes:
      return (False, "unexpected training size " + str(size))
    if size in observed:
      return (False, "duplicate result for training size " + str(size))
    if not math.isfinite(threshold) or threshold != expected_threshold:
      return (False, "training size " + str(size)
              + " used threshold " + repr(threshold)
              + ", expected " + repr(expected_threshold))
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
      return (False, "training size " + str(size)
              + " produced invalid fraction " + repr(fraction))
    observed[size] = fraction
  missing = [size for size in sizes if size not in observed]
  if missing:
    return (False, "missing result for training size(s) " + repr(missing))
  detail = ", ".join(
    str(size) + " -> " + repr(observed[size]) for size in sizes)
  return (True, detail)


def byte_identity(text_a, text_b, pattern, strip=None):
  """Compare two runs' selected lines for character-exact equality.

  The golden byte-identity proof: extract the pattern-matching lines
  from each run (the home notes diff only ``^(phase|epoch|best)`` and
  friends, never the whole noisy log) and require the two lists equal
  line for line. This implements the notes' ``diff <(grep ...) <(grep
  ...)  # EMPTY`` step, but in-process so the harness owns the verdict
  and can name the first divergence in the log.

  Arguments:
    text_a  = the first run's captured output (e.g. the pinned
              pre-feature build in the temporary worktree).
    text_b  = the second run's captured output (the current tree).
    pattern = the grep-style regex selecting the lines to compare.
    strip   = an optional regex removed from every selected line on both
              sides before comparison (re.sub(strip, "", line)), used to
              drop a machine-noise field (the trailing wall-clock column)
              from an otherwise deterministic line; the divergence detail
              then shows the stripped lines actually compared. None
              compares the selected lines verbatim.

  Returns:
    (equal, detail). equal is True only when the selected line lists
    match exactly. detail is "" on equality; otherwise it names the
    line-count mismatch or the first differing line (both sides, as
    compared, i.e. after any strip), so the raw log records WHERE the
    determinism broke.
  """
  lines_a = matching_lines(text=text_a, pattern=pattern)
  lines_b = matching_lines(text=text_b, pattern=pattern)

  if strip is not None:
    stripped_a = []
    for line in lines_a:
      stripped_a.append(re.sub(strip, "", line))
    stripped_b = []
    for line in lines_b:
      stripped_b.append(re.sub(strip, "", line))
    lines_a = stripped_a
    lines_b = stripped_b

  if len(lines_a) != len(lines_b):
    detail = ("selected-line counts differ: "
              + str(len(lines_a)) + " (a) vs "
              + str(len(lines_b)) + " (b)")
    return (False, detail)

  n = len(lines_a)
  i = 0
  while i < n:
    if lines_a[i] != lines_b[i]:
      detail = ("first divergence at selected line " + str(i) + ":\n"
                + "  a: " + lines_a[i] + "\n"
                + "  b: " + lines_b[i])
      return (False, detail)
    i = i + 1

  return (True, "")


def decreasing(values, *, tol=0.0):
  """Whether a numeric series ends below where it started.

  A lenient "loss descends" check for smoke runs: the home notes ask
  that a short training's loss go down, not that it fall monotonically
  because mini-batch noise makes strict monotonicity false. The rule
  compares only the two endpoints. Both endpoints must be finite. The
  last value must sit below the first by more than a tolerance.

  Arguments:
    values = the ordered numeric series (e.g. per-epoch train loss).
    tol    = the minimum drop required, first minus last (default 0.0,
             any strict decrease passes).

  Returns:
    (ok, detail). ok is True only when values has at least two points,
    both endpoints are finite and first - last > tol. detail names the
    endpoint values and the drop. A rejected non-finite endpoint is
    named before subtraction, so an exploded run cannot look like a
    successful decrease.
  """
  if len(values) < 2:
    return (False, "need at least two points, got " + str(len(values)))

  first = values[0]
  last = values[-1]
  if not math.isfinite(first) or not math.isfinite(last):
    detail = ("need finite endpoints: first " + repr(first)
              + ", last " + repr(last))
    return (False, detail)

  drop = first - last
  detail = ("first " + repr(first) + " -> last " + repr(last)
            + " (drop " + repr(drop) + ")")
  return (drop > tol, detail)


def head_lr_cadence(text, *, phase, phase_epochs, warmup_epochs, patience,
                    factor):
  """Judge the first forced-plateau LR cut inside one named phase.

  ``training_loop_batched`` numbers epochs from one for each phase.  It starts
  the plateau scheduler after warmup.  The first scheduler observation sets
  the best metric; ``patience + 1`` later non-improving observations trigger
  the first cut.  Therefore a deliberately forced plateau must cut at phase
  epoch ``warmup_epochs + patience + 2``.

  Only epoch lines after the requested ``phase 'name':`` banner count.  This
  prevents a trunk LR change from being mistaken for the head override.

  Returns:
    ``(ok, detail)`` with the expected epoch and observed LR values.
  """
  phase_pattern = re.compile(r"^phase '([^']+)':")
  epoch_pattern = re.compile(
    r"^epoch\s+([0-9]+)\s+lr\s+([^\s]+)(?:\s|$)")
  current_phase = None
  values = {}
  saw_phase = False
  for line in text.splitlines():
    phase_match = phase_pattern.match(line)
    if phase_match is not None:
      current_phase = phase_match.group(1)
      if current_phase == phase:
        if saw_phase:
          return (False, "phase " + repr(phase) + " appears more than once")
        saw_phase = True
      continue
    if current_phase != phase:
      continue
    epoch_match = epoch_pattern.match(line)
    if epoch_match is None:
      continue
    epoch = int(epoch_match.group(1))
    try:
      lr = float(epoch_match.group(2))
    except ValueError:
      return (False, "phase " + repr(phase) + " epoch " + str(epoch)
              + " has a non-numeric lr token "
              + repr(epoch_match.group(2)))
    if not math.isfinite(lr) or lr <= 0.0:
      return (False, "phase " + repr(phase) + " epoch " + str(epoch)
              + " needs a finite positive lr, got " + repr(lr))
    if epoch in values:
      return (False, "phase " + repr(phase) + " repeats epoch " + str(epoch))
    values[epoch] = lr

  if not saw_phase:
    return (False, "missing phase " + repr(phase) + " banner")
  expected_epoch = warmup_epochs + patience + 2
  complete_epochs = list(range(1, phase_epochs + 1))
  missing_complete = [epoch for epoch in complete_epochs if epoch not in values]
  extras = sorted(epoch for epoch in values if epoch not in complete_epochs)
  if missing_complete or extras:
    return (False, "phase " + repr(phase) + " needs exactly epochs 1.."
            + str(phase_epochs) + "; missing " + repr(missing_complete)
            + ", extra " + repr(extras))
  if expected_epoch > phase_epochs:
    return (False, "expected cut epoch " + str(expected_epoch)
            + " is outside the " + str(phase_epochs) + "-epoch phase")
  base_epoch = max(1, warmup_epochs)
  needed = list(range(base_epoch, expected_epoch + 1))
  missing = [epoch for epoch in needed if epoch not in values]
  if missing:
    return (False, "phase " + repr(phase) + " is missing epoch lr values "
            + repr(missing))

  base_lr = values[base_epoch]
  for epoch in range(base_epoch, expected_epoch):
    if not math.isclose(values[epoch], base_lr,
                        rel_tol=1.0e-12, abs_tol=0.0):
      return (False, "phase " + repr(phase) + " changed lr before the "
              "expected cut: epoch " + str(epoch) + " has "
              + repr(values[epoch]) + ", base " + repr(base_lr))

  expected_lr = base_lr * factor
  cut_lr = values[expected_epoch]
  if not math.isclose(cut_lr, expected_lr,
                      rel_tol=1.0e-12, abs_tol=0.0):
    detail = ("phase " + repr(phase) + " expected first cut at epoch "
              + str(expected_epoch) + " from " + repr(base_lr) + " to "
              + repr(expected_lr) + ", observed " + repr(cut_lr))
    return (False, detail)
  for epoch in range(expected_epoch, phase_epochs + 1):
    if not math.isclose(values[epoch], cut_lr,
                        rel_tol=1.0e-12, abs_tol=0.0):
      return (False, "phase " + repr(phase) + " changed lr again at epoch "
              + str(epoch) + ": expected " + repr(cut_lr)
              + ", observed " + repr(values[epoch]))

  detail = ("phase " + repr(phase) + " forced plateau: base lr "
            + repr(base_lr) + ", expected first cut at epoch "
            + str(expected_epoch) + " to " + repr(expected_lr)
            + ", observed one cut across all " + str(phase_epochs)
            + " epochs")
  return (True, detail)


_EMA_FIRST_LIVE_PREFIX = "ema first-live:"
_EMA_FIRST_LIVE_PATTERN = re.compile(
  r"^ema first-live: epoch=([0-9]+) schedule=([^\s]+) beta=([^\s]+) "
  r"steps_per_epoch=([0-9]+) raw_median=([^\s]+) "
  r"averaged_median=([^\s]+) raw_mean=([^\s]+) "
  r"averaged_mean=([^\s]+)$")


def ema_first_live(text, *, expected_epoch, expected_schedule,
                   horizon_epochs):
  """Judge the production record from the first epoch that uses EMA.

  The record carries enough numbers to recompute beta independently from the
  documented formula ``1 - 1/(horizon * schedule * steps_per_epoch)``.  A
  printed statement that EMA is live is not sufficient: the raw and averaged
  validation metrics must also be present and finite.
  """
  lines = []
  for line in text.splitlines():
    if line.startswith(_EMA_FIRST_LIVE_PREFIX):
      lines.append(line)
  if len(lines) != 1:
    return (False, "need exactly one ema first-live record, got "
            + str(len(lines)))
  match = _EMA_FIRST_LIVE_PATTERN.fullmatch(lines[0])
  if match is None:
    return (False, "malformed ema first-live record: " + lines[0])

  epoch = int(match.group(1))
  steps = int(match.group(4))
  try:
    schedule = float(match.group(2))
    beta = float(match.group(3))
    raw_median = float(match.group(5))
    averaged_median = float(match.group(6))
    raw_mean = float(match.group(7))
    averaged_mean = float(match.group(8))
  except ValueError:
    return (False, "ema first-live record contains a non-numeric value")

  numbers = [schedule, beta, raw_median, averaged_median,
             raw_mean, averaged_mean]
  if not all(math.isfinite(value) for value in numbers):
    return (False, "ema first-live record needs finite schedule, beta, and "
            "metrics: " + repr(numbers))
  if epoch != expected_epoch:
    return (False, "ema became live at epoch " + str(epoch)
            + ", expected " + str(expected_epoch))
  if steps <= 0:
    return (False, "ema first-live steps_per_epoch must be positive, got "
            + str(steps))
  if not math.isclose(schedule, expected_schedule,
                      rel_tol=1.0e-12, abs_tol=1.0e-15):
    return (False, "ema first-live schedule is " + repr(schedule)
            + ", expected " + repr(expected_schedule))
  denom = horizon_epochs * schedule * steps
  expected_beta = 0.0 if denom < 1.0 else 1.0 - 1.0 / denom
  if not math.isclose(beta, expected_beta,
                      rel_tol=1.0e-12, abs_tol=1.0e-15):
    return (False, "ema first-live beta is " + repr(beta)
            + ", independently expected " + repr(expected_beta))
  detail = ("epoch " + str(epoch) + ", schedule " + repr(schedule)
            + ", beta " + repr(beta) + ", raw/average medians "
            + repr(raw_median) + "/" + repr(averaged_median)
            + ", raw/average means " + repr(raw_mean) + "/"
            + repr(averaged_mean))
  return (True, detail)


_TRUNK_DIGEST_PREFIX = "phase-boundary trunk digest:"
_TRUNK_DIGEST_PATTERN = re.compile(
  r"^phase-boundary trunk digest: phase=([^\s]+) "
  r"before=([0-9a-f]{64}) after=([0-9a-f]{64}) "
  r"parameters=([1-9][0-9]*) finite=1$")


def phase2_trunk_digest(text, *, expected_phase, should_change):
  """Judge one phase-2 trunk digest record from a production run.

  ``should_change`` is True for joint training and False for the frozen-head
  control.  The verdict compares the two hashes itself; no printed
  ``changed`` word or flag is trusted.
  """
  lines = []
  for line in text.splitlines():
    if line.startswith(_TRUNK_DIGEST_PREFIX):
      lines.append(line)
  if len(lines) != 1:
    return (False, "need exactly one phase-boundary trunk digest record, got "
            + str(len(lines)))
  match = _TRUNK_DIGEST_PATTERN.fullmatch(lines[0])
  if match is None:
    return (False, "malformed phase-boundary trunk digest record: "
            + lines[0])
  phase = match.group(1)
  before = match.group(2)
  after = match.group(3)
  parameter_count = int(match.group(4))
  if phase != expected_phase:
    return (False, "phase-boundary digest names phase " + repr(phase)
            + ", expected " + repr(expected_phase))
  changed = before != after
  if changed != should_change:
    wanted = "different" if should_change else "identical"
    return (False, "phase " + repr(phase) + " trunk hashes must be "
            + wanted + ", observed before=" + before + " after=" + after)
  detail = ("phase " + repr(phase) + ", " + str(parameter_count)
            + " parameters, before=" + before + ", after=" + after)
  return (True, detail)
