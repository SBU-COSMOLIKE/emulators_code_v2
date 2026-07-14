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
