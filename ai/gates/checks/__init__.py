"""The numeric checks the board's tests run.

Two kinds of file live here. ``logscan`` holds pure text helpers:
find a banner line in a run's output, or compare the selected lines
of two runs character by character (the golden-run proof). The other
six files are stand-alone programs a test launches as a subprocess;
each one computes a number the test needs and exits nonzero on
failure, printing every value it compared into the test's log:

  ge_c_eval_bs.py       eval-batch-invariance: validation chi2 agrees
                        across eval batch sizes (rtol 1e-6).
  gb_c_berhu_reduce.py  berhu-loss: the loss transform matches the
                        hand references and is smooth at both joins.
  gwd_census.py         weight-decay-census: only weight matrices
                        land in the decayed optimizer group.
  gt_b_triangle.py      triangle-shading: the corner plot greys
                        exactly the cut panels.
  gsv_bitwise_drift.py  save-rebuild-drift: a saved emulator rebuilds
                        bit-for-bit, even against drifted defaults.
  gct_parity.py         cobaya-adapter: the MCMC-time predictor
                        matches the training side (rtol 1e-6).

The filename prefixes (gb_c, gsv, ...) are historical: they key
each test's history in ai/notes/; the board runs them by test name. No
file here imports torch, cosmolike, or cobaya at package-import time,
so board.py can use the text helpers on a machine without the heavy
stack; the six programs pull torch in only when actually run.
"""
