#!/usr/bin/env python3
"""GB-C leg 1: the berhu / _reduce numerics + autograd continuity.

Drives the REAL CosmolikeChi2._reduce on fabricated per-sample chi2
tensors (home note loss-mode-berhu.md:148-153). With trim and focus
off, _reduce reduces a single-element input to its per-sample transform
v(c), so probing it at chosen c values reads v(c) straight from the
shipped code. The gate checks: berhu == sqrt below the knot;
berhu_capped == berhu below the cap; both match the manual reference
formula; the transform is C1 (value + first derivative continuous)
across BOTH knots; and it all holds for non-default knots. Prints every
value it compares and exits nonzero on any mismatch.

_reduce is an instance method but reads no instance state (only the
tensor c and the passed knots), so it is called unbound with self=None.

PS: knot t1 = the lower C1 join (sqrt below, chi2-like above); cap t2 =
the upper C1 join of berhu_capped (sqrt-shaped tail above); C1 = the
value and its first derivative are continuous at a join; autograd
continuity = the derivative from torch.autograd matches on both sides.
"""

import sys

import torch

from emulator.loss_functions import CosmolikeChi2

# _reduce reads no self state, so bind it once and call with self=None.
_reduce = CosmolikeChi2._reduce

FAILURES = []


def report(label, ok, detail):
  """Print one acceptance line and record a failure.

  Arguments:
    label  = what is being checked.
    ok      = the boolean verdict.
    detail = the values behind the verdict (always printed).
  """
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


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


def check_knots(t1, t2, tol, dtol):
  """Run every berhu / berhu_capped check for one (knot, cap) pair.

  Arguments:
    t1   = the knot.
    t2   = the cap.
    tol  = the value tolerance.
    dtol = the derivative-gap tolerance for the C1 check.
  """
  print("knots t1 = " + repr(t1) + ", t2 = " + repr(t2) + ":")

  # berhu == sqrt strictly below the knot.
  probe_lo = t1 * 0.5
  vb = transform(probe_lo, "berhu", t1, t2)
  vs = transform(probe_lo, "sqrt", t1, t2)
  report("berhu == sqrt below the knot (c = " + repr(probe_lo) + ")",
         abs(vb - vs) < tol,
         "berhu " + repr(vb) + " vs sqrt " + repr(vs))

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
           "got " + repr(got) + " want " + repr(want))

  # berhu_capped == berhu strictly below the cap.
  probe_mid = (t1 + t2) * 0.5
  vc = transform(probe_mid, "berhu_capped", t1, t2)
  vbe = transform(probe_mid, "berhu", t1, t2)
  report("berhu_capped == berhu below the cap (c = " + repr(probe_mid) + ")",
         abs(vc - vbe) < tol,
         "capped " + repr(vc) + " vs berhu " + repr(vbe))

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
           "got " + repr(got) + " want " + repr(want))

  # C1 at the knot: value + derivative continuous across t1 (berhu).
  eps = t1 * 1.0e-3
  v_lo = transform(t1 - eps, "berhu", t1, t2)
  v_hi = transform(t1 + eps, "berhu", t1, t2)
  d_lo = slope(t1 - eps, "berhu", t1, t2)
  d_hi = slope(t1 + eps, "berhu", t1, t2)
  report("berhu C1 at the knot: value continuous",
         abs(v_lo - v_hi) < tol,
         "v- " + repr(v_lo) + " v+ " + repr(v_hi))
  report("berhu C1 at the knot: derivative continuous",
         abs(d_lo - d_hi) < dtol,
         "d- " + repr(d_lo) + " d+ " + repr(d_hi))

  # C1 at the cap: value + derivative continuous across t2 (berhu_capped).
  eps2 = t2 * 1.0e-3
  vc_lo = transform(t2 - eps2, "berhu_capped", t1, t2)
  vc_hi = transform(t2 + eps2, "berhu_capped", t1, t2)
  dc_lo = slope(t2 - eps2, "berhu_capped", t1, t2)
  dc_hi = slope(t2 + eps2, "berhu_capped", t1, t2)
  report("berhu_capped C1 at the cap: value continuous",
         abs(vc_lo - vc_hi) < tol,
         "v- " + repr(vc_lo) + " v+ " + repr(vc_hi))
  report("berhu_capped C1 at the cap: derivative continuous",
         abs(dc_lo - dc_hi) < dtol,
         "d- " + repr(dc_lo) + " d+ " + repr(dc_hi))

  # anneal endpoints: s = 0 is plain sqrt, s = 1 is the full berhu shape.
  above = t1 * 2.0
  v_s0 = transform(above, "berhu", t1, t2, s=0.0)
  v_sqrt = transform(above, "sqrt", t1, t2)
  v_s1 = transform(above, "berhu", t1, t2, s=1.0)
  v_full = transform(above, "berhu", t1, t2)
  report("berhu anneal s = 0 is plain sqrt (c = " + repr(above) + ")",
         abs(v_s0 - v_sqrt) < tol,
         "s0 " + repr(v_s0) + " sqrt " + repr(v_sqrt))
  report("berhu anneal s = 1 is the full berhu (c = " + repr(above) + ")",
         abs(v_s1 - v_full) < tol,
         "s1 " + repr(v_s1) + " full " + repr(v_full))


def main():
  """Run the census over the default and a non-default (knot, cap)."""
  print("== GB-C leg 1: berhu / _reduce numerics + autograd continuity ==")
  # default knots (train_args.loss.berhu defaults: knot 0.2, cap 10.0).
  check_knots(t1=0.2, t2=10.0, tol=1.0e-9, dtol=1.0e-4)
  # non-default knots (the same shape must hold).
  check_knots(t1=0.5, t2=5.0, tol=1.0e-9, dtol=1.0e-4)

  print("")
  if len(FAILURES) == 0:
    print("GB-C leg 1: ALL PASS")
    return 0
  print("GB-C leg 1: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
