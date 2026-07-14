#!/usr/bin/env python3
"""berhu-loss numerics: prove the robust loss math is right.

The berhu loss is the one training transform a run cannot cross-check
against itself: if its shape or its slope is wrong where the pieces meet,
every berhu run silently optimizes the wrong thing. This check reads the
shipped transform straight out of CosmolikeChi2._reduce and compares it to
hand-written reference formulas at many chi2 values.

Vocabulary (every value below is in chi2 units):
  berhu       the loss shape that is sqrt(chi2) for small chi2 and a
              straight line for large chi2 (a reversed Huber loss).
  knot (t1)   the chi2 value where berhu switches from the sqrt piece to
              the straight-line piece.
  cap (t2)    for berhu_capped, a second switch above which the line bends
              back into a sqrt-shaped tail, so one huge sample cannot
              dominate the gradient.
  join        a switch point (t1 or t2); the checks confirm the value and
              the slope are continuous across it.
  slope       the derivative dv/dc, read from the shipped code by autograd
              (torch's automatic differentiation) and compared to the
              closed-form derivative.

What it checks, for each (knot, cap) pair: berhu equals sqrt below the knot;
berhu_capped equals berhu below the cap; both match the reference formulas
in every region; at each join the shipped value matches the reference (to
1e-9) and the autograd slope matches the closed-form derivative (to 1e-6);
and the anneal blend is plain sqrt at s = 0 and the full berhu shape at
s = 1. It repeats everything at a non-default knot and cap. Every value is
printed; any mismatch exits non-zero.

The probes roll up into three board-declared evidence legs (queue 2):
reference-values (every value-vs-reference probe), join-derivatives (the
autograd-slope probes at the t1 and t2 joins), and anneal-endpoints (the
s = 0 / s = 1 blend probes). The script prints one reserved
'##AID <aid> <PASS|FAIL>' line per leg at the end; the exit status stays the
single aggregate verdict, not a leg.

_reduce is a method but reads no instance state (only the tensor and the
knots passed in), so it is called unbound with self = None.

Home note: training-stack.md:148-153.
"""

import sys

import torch

from emulator.losses.core import CosmolikeChi2

# _reduce reads no self state, so bind it once and call with self=None.
_reduce = CosmolikeChi2._reduce

FAILURES = []

# (queue 2) the three board-declared evidence legs this check emits. Every
# report() call names the leg it belongs to; a leg reds if ANY of its probes
# fail. The script prints exactly one '##AID <leg-aid> <PASS|FAIL>' line per
# leg at the end (emit_aids), one terminal per declared leg -- NOT one per
# probe, because many probes (both knot pairs) roll up into each leg. The
# child's exit status stays the single aggregate verdict, not a leg.
LEG_AIDS = {
    "reference-values": "berhu-loss.reference-values",
    "join-derivatives": "berhu-loss.join-derivatives",
    "anneal-endpoints": "berhu-loss.anneal-endpoints",
}
# leg name -> True once any probe on that leg fails.
LEG_FAILED = {leg: False for leg in LEG_AIDS}


def report(label, ok, detail, leg):
  """Print one acceptance line and record a failure against its evidence leg.

  Arguments:
    label  = what is being checked.
    ok     = the boolean verdict.
    detail = the values behind the verdict (always printed).
    leg    = the LEG_AIDS key this probe rolls up into ("reference-values",
             "join-derivatives", or "anneal-endpoints"); a False verdict
             reds the whole leg's single ##AID terminal.
  """
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)
    LEG_FAILED[leg] = True


def emit_aids():
  """Print the one reserved '##AID <aid> <result>' line per declared leg.

  One terminal per board-declared evidence leg (reference-values,
  join-derivatives, anneal-endpoints), aggregating every probe that rolled
  up into it: PASS only when the leg had no failing probe. run_board folds
  these into the gate's executed set and reconciles them against the
  declared evidence map.
  """
  for leg, aid in LEG_AIDS.items():
    mark = "FAIL" if LEG_FAILED[leg] else "PASS"
    print("##AID " + aid + " " + mark)


def transform(c_value, mode, knot, cap, s=None):
  """Read the shipped per-sample transform v(c) at one c value.

  With trim 0 (keep all) and focus 0 (unit weight), _reduce of a
  single-element tensor returns that element's transformed value, so
  this reads v(c) directly from CosmolikeChi2._reduce.

  Arguments:
    c_value = the chi2 value to probe (a python float).
    mode    = the loss mode ("sqrt" / "berhu" / "berhu_capped" / ...).
    knot    = t1 (a python float), passed as a 0-dim tensor.
    cap      = t2 (a python float) or None.
    s        = the anneal blend in [0, 1] or None (no blend).

  Returns:
    v(c_value) as a python float.
  """
  c = torch.tensor([float(c_value)], dtype=torch.float64)
  knot_t = torch.tensor(float(knot), dtype=torch.float64)
  cap_t = None if cap is None else torch.tensor(float(cap), dtype=torch.float64)
  s_t = None if s is None else torch.tensor(float(s), dtype=torch.float64)
  out = _reduce(None,
                c=c,
                mode=mode,
                trim=0.0,
                focus=0.0,
                focus_scale=1.0,
                berhu_knot=knot_t,
                berhu_cap=cap_t,
                berhu_s=s_t)
  return float(out)


def slope(c_value, mode, knot, cap, s=None):
  """The autograd derivative dv/dc of the transform at one c value.

  Arguments:
    c_value = the chi2 value to differentiate at.
    mode / knot / cap / s = as in transform().

  Returns:
    dv/dc at c_value as a python float (via torch.autograd.grad).
  """
  c = torch.tensor([float(c_value)], dtype=torch.float64, requires_grad=True)
  knot_t = torch.tensor(float(knot), dtype=torch.float64)
  cap_t = None if cap is None else torch.tensor(float(cap), dtype=torch.float64)
  s_t = None if s is None else torch.tensor(float(s), dtype=torch.float64)
  out = _reduce(None,
                c=c,
                mode=mode,
                trim=0.0,
                focus=0.0,
                focus_scale=1.0,
                berhu_knot=knot_t,
                berhu_cap=cap_t,
                berhu_s=s_t)
  grad = torch.autograd.grad(out, c)[0]
  return float(grad[0])


def ref_berhu(c, t1):
  """The manual berhu reference: sqrt below t1, (c + t1)/(2 sqrt t1) above."""
  if c <= t1:
    return c ** 0.5
  return (c + t1) / (2.0 * t1 ** 0.5)


def ref_berhu_capped(c, t1, t2):
  """The manual berhu_capped reference (sqrt / berhu / sqrt-shaped tail)."""
  if c <= t1:
    return c ** 0.5
  if c <= t2:
    return (c + t1) / (2.0 * t1 ** 0.5)
  return (2.0 * (t2 * c) ** 0.5 + t1 - t2) / (2.0 * t1 ** 0.5)


def ref_berhu_deriv(c, t1):
  """The manual berhu derivative: 0.5/sqrt(c) below t1, 1/(2 sqrt t1) above."""
  if c <= t1:
    return 0.5 / c ** 0.5
  return 1.0 / (2.0 * t1 ** 0.5)


def ref_berhu_capped_deriv(c, t1, t2):
  """The manual berhu_capped derivative (sqrt / linear / sqrt-tail slopes).

  d/dc of ref_berhu_capped, region by region: 0.5/sqrt(c) below t1,
  the constant 1/(2 sqrt t1) between t1 and t2, and sqrt(t2)/(2 sqrt t1
  sqrt c) above t2 (the tail term 2 sqrt(t2 c) differentiates to
  sqrt(t2)/sqrt(c)). The three regions meet with equal slope, so the
  transform's first derivative is continuous (it is C1) at both joins.
  """
  if c <= t1:
    return 0.5 / c ** 0.5
  if c <= t2:
    return 1.0 / (2.0 * t1 ** 0.5)
  return t2 ** 0.5 / (2.0 * t1 ** 0.5 * c ** 0.5)


def check_knots(t1, t2, tol, dtol):
  """Run every berhu / berhu_capped check for one (knot, cap) pair.

  Arguments:
    t1   = the knot.
    t2   = the cap.
    tol  = the value tolerance (shipped value vs the reference).
    dtol = the slope tolerance (autograd slope vs the analytic derivative).
  """
  print("knots t1 = " + repr(t1) + ", t2 = " + repr(t2) + ":")

  # berhu == sqrt strictly below the knot.
  probe_lo = t1 * 0.5
  vb = transform(probe_lo, "berhu", t1, t2)
  vs = transform(probe_lo, "sqrt", t1, t2)
  report("berhu == sqrt below the knot (c = " + repr(probe_lo) + ")",
         abs(vb - vs) < tol,
         "berhu " + repr(vb) + " vs sqrt " + repr(vs),
         leg="reference-values")

  # berhu matches the manual reference on both sides of the knot.
  points = []
  points.append(t1 * 0.5)
  points.append(t1 * 2.0)
  points.append(t2 * 2.0)
  for c in points:
    got = transform(c, "berhu", t1, t2)
    want = ref_berhu(c, t1)
    report("berhu(" + repr(c) + ") matches the reference",
           abs(got - want) < tol,
           "got " + repr(got) + " want " + repr(want),
           leg="reference-values")

  # berhu_capped == berhu strictly below the cap.
  probe_mid = (t1 + t2) * 0.5
  vc = transform(probe_mid, "berhu_capped", t1, t2)
  vbe = transform(probe_mid, "berhu", t1, t2)
  report("berhu_capped == berhu below the cap (c = " + repr(probe_mid) + ")",
         abs(vc - vbe) < tol,
         "capped " + repr(vc) + " vs berhu " + repr(vbe),
         leg="reference-values")

  # berhu_capped matches the manual reference across all three regions.
  cap_points = []
  cap_points.append(t1 * 0.5)
  cap_points.append((t1 + t2) * 0.5)
  cap_points.append(t2 * 3.0)
  for c in cap_points:
    got = transform(c, "berhu_capped", t1, t2)
    want = ref_berhu_capped(c, t1, t2)
    report("berhu_capped(" + repr(c) + ") matches the reference",
           abs(got - want) < tol,
           "got " + repr(got) + " want " + repr(want),
           leg="reference-values")

  # analytic-derivative check at the knot (berhu): at t1 +- 1e-3 the
  # shipped value matches the manual reference (tol) and the shipped
  # slope matches the closed-form derivative (dtol). Each single point is
  # probed against the closed form, replacing the old two-point C1
  # comparison (which compared points 2*eps apart across a curved join at
  # tolerances no smooth function meets: the gap was 2*eps*slope in value
  # and eps*curvature in slope, not a real discontinuity).
  for c in (t1 * (1.0 - 1.0e-3), t1 * (1.0 + 1.0e-3)):
    vg = transform(c, "berhu", t1, t2)
    vr = ref_berhu(c, t1)
    report("berhu value == reference at c = " + repr(c),
           abs(vg - vr) < tol,
           "got " + repr(vg) + " want " + repr(vr),
           leg="reference-values")
    dg = slope(c, "berhu", t1, t2)
    dr = ref_berhu_deriv(c, t1)
    report("berhu slope == analytic derivative at c = " + repr(c),
           abs(dg - dr) < dtol,
           "slope " + repr(dg) + " ref " + repr(dr),
           leg="join-derivatives")

  # analytic-derivative check at the cap (berhu_capped): the same probe at
  # t2 +- 1e-3, against the capped reference and its piecewise derivative
  # (the linear slope just below, the sqrt-tail slope just above).
  for c in (t2 * (1.0 - 1.0e-3), t2 * (1.0 + 1.0e-3)):
    vg = transform(c, "berhu_capped", t1, t2)
    vr = ref_berhu_capped(c, t1, t2)
    report("berhu_capped value == reference at c = " + repr(c),
           abs(vg - vr) < tol,
           "got " + repr(vg) + " want " + repr(vr),
           leg="reference-values")
    dg = slope(c, "berhu_capped", t1, t2)
    dr = ref_berhu_capped_deriv(c, t1, t2)
    report("berhu_capped slope == analytic derivative at c = " + repr(c),
           abs(dg - dr) < dtol,
           "slope " + repr(dg) + " ref " + repr(dr),
           leg="join-derivatives")

  # anneal endpoints: s = 0 is plain sqrt, s = 1 is the full berhu shape.
  above = t1 * 2.0
  v_s0 = transform(above, "berhu", t1, t2, s=0.0)
  v_sqrt = transform(above, "sqrt", t1, t2)
  v_s1 = transform(above, "berhu", t1, t2, s=1.0)
  v_full = transform(above, "berhu", t1, t2)
  report("berhu anneal s = 0 is plain sqrt (c = " + repr(above) + ")",
         abs(v_s0 - v_sqrt) < tol,
         "s0 " + repr(v_s0) + " sqrt " + repr(v_sqrt),
         leg="anneal-endpoints")
  report("berhu anneal s = 1 is the full berhu (c = " + repr(above) + ")",
         abs(v_s1 - v_full) < tol,
         "s1 " + repr(v_s1) + " full " + repr(v_full),
         leg="anneal-endpoints")


def main():
  """Run every berhu check at the default knot/cap, then a non-default pair.

  Calls check_knots twice: once at the shipped defaults (knot 0.2, cap 10.0)
  and once at (0.5, 5.0), each with a 1e-9 value tolerance and a 1e-6 slope
  tolerance. check_knots prints a PASS/FAIL line per probe; main returns 1 if
  any check failed, else 0.
  """
  print("== berhu-loss numerics ==")
  # default knots (train_args.loss.berhu defaults: knot 0.2, cap 10.0).
  check_knots(t1=0.2, t2=10.0, tol=1.0e-9, dtol=1.0e-6)
  # non-default knots (the same shape must hold).
  check_knots(t1=0.5, t2=5.0, tol=1.0e-9, dtol=1.0e-6)

  # (queue 2) one ##AID terminal per board-declared evidence leg, after every
  # probe on both knot pairs has run (so a leg reds if it failed at EITHER
  # pair). This is the manifest run_board reconciles against the evidence map.
  emit_aids()

  print("")
  if len(FAILURES) == 0:
    print("berhu-loss numerics: ALL PASS")
    return 0
  print("berhu-loss numerics: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
