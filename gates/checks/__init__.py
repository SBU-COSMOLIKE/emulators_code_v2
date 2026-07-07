"""Acceptance checks for the workstation gates harness.

This package holds one module per non-trivial acceptance rule the
board's gates assert. Two kinds live here. The pure text helpers in
``logscan`` scan a run's captured stdout for the banner lines and
patterns a gate's home note names, and compare the selected lines of
two runs for byte identity. The numeric check scripts (``gwd_census``,
``ge_c_eval_bs``, ``gsv_bitwise_drift``, ``gct_parity``) are executable
programs the gates launch as subprocesses inside the cocoa env: they
import the emulator package, compute the gate's numeric acceptance
(the parameter-group census, the partition invariance, the bitwise
save/rebuild equality, the training-vs-inference parity), print every
value they check into the tee'd log, and exit nonzero on failure.

The split is deliberate: nothing in this package imports torch,
cosmolike, or cobaya at module load time, so ``board`` can import the
pure helpers on the dev Mac (no heavy stack) while the numeric scripts
pull torch in only when actually run on the workstation.

PS: tee = write a subprocess's output to the terminal and the log file
at once; banner = a driver's human-readable startup / per-epoch status
line the gate asserts on; byte identity = two runs whose selected log
lines match to the character (a determinism / no-op-change proof).
"""
