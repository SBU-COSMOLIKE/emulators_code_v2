# The gates board

A self-driving test suite for the cosmic-shear emulator. One command,
`python gates/run_board.py`, runs every deferred verification on the
workstation (the NVIDIA + cocoa machine), writes one raw log per test,
and leaves a pass/fail table. You run it; it does not need a Claude
session on the box.

## What it is

Each **test** ("gate" in the code) pins down one feature or contract:
it runs a short training or a small check script, then judges pass/fail
on the banner lines or numeric values that feature must produce. Tests
are grouped into three **tiers** (`backlog`, `new-features`,
`save-and-sample`) and run in a fixed order.

## Vocabulary

| Term | Meaning |
|------|---------|
| board | the ordered list of tests |
| gate | one test: its commands and its pass/fail rule ("test" in prose) |
| tier | the grouping `--tier` selects |
| golden run | the same config trained on the current code and on a pinned older commit; selected log lines must match exactly |
| smoke | a short training run judged on its banner lines |
| banner | a driver status line a test checks for |
| worktree | a throwaway git checkout of another commit; never touches your tree |
| preflight | the pre-GPU checks (git tip, clean tree, cocoa imports, data paths) |
| resume | a rerun skips tests already marked PASS |
| home note | the file under `notes/` that defines what a test must prove |

## How it is implemented

```
gates/
  run_board.py     the CLI + runner: preflight, order, per-test logs,
                   resume, and golden runs in a throwaway worktree
  board.py         the list of tests: a small Gate class + the tests, each
                   with its home note, a maps line, and a run function
  board_config.json  deployment paths (root, dumps, yaml dir) + per-test
                   golden bases + the smoke-config paths
  configs/         one small YAML per test-leg (the smoke run's knobs)
  checks/          the numeric check scripts a test launches (the census,
                   the eval-batch invariance, the save/rebuild bitwise,
                   the parity probe) + pure log-scan helpers
  logs/            (ships empty) one <test>.log per run + BOARD.md +
                   board_status.json
```

The runner builds a small helper per test that streams every command
into that test's log and notes each check. A test never touches
subprocess, files, or git directly; it goes through that helper, so the
log is complete and every path comes from `board_config.json`.

Some tests carry a **golden run**: they build the same config both on
the current tree and on a pinned pre-feature commit, and require the
selected log lines identical to the character. That pinned build runs
in a throwaway `git worktree` the runner always removes, so it never
disturbs your working tree. Only the EMA identity test pins a base by
default; the others run this leg only when you set their base in
`board_config.json`, and otherwise fall back to a plain smoke run.

## How to run it

```
cd $ROOTDIR                # the Cocoa root; the cocoa env already active
                           # (torch + cosmolike + cobaya importable)
G=external_modules/code/emulators_code_v2/gates

git -C $G/.. pull                    # tip must be the harness commit
edit $G/board_config.json            # fill root / driver paths once
python $G/run_board.py --check       # preflight only (no GPU time)
python $G/run_board.py --dry-run     # print the plan, run nothing
python $G/run_board.py               # the whole board, in order
git -C $G/.. add -f gates/logs
git -C $G/.. commit -m "workstation board run: logs"
git -C $G/.. push
```

The harness finds its own files from its location, so it can be run
from any directory; `$ROOTDIR` is just the natural cocoa working spot.

Preflight aborts before any GPU time on a stale git tip, a dirty watched
tree, a missing cocoa import, or a missing data path, and prints the
remedy. Known gaps (2026-07-12, fixes in progress): the dirty-tree watch
covers `emulator/`, `gates/`, and the root drivers only — keep
`compute_data_vectors/`, `cobaya_theory/`, and `syren/` clean yourself
for now; and an unknown `--gate` / `--from` name warns but proceeds
(exit 0 with no tests run), and a bad `--force-rerun` id is silently
ignored — automation must validate ids against `--list`.
Selectors: `--gate <name> [...]`, `--tier backlog|new-features|save-and-sample`,
`--from <name>`, `--dry-run`. A rerun skips tests already marked PASS
(`--force-rerun <name>` overrides for named gates; `--force-rerun-all`
reruns EVERY selected gate — the full regression pass after a batch of
library changes, composing with the selectors and never deleting the
resume map); a crash loses only the in-flight test.

## The 32 tests

| Test | What it confirms |
|------|------------------|
| ema-off-identity | EMA off leaves runs byte-identical to the pre-EMA build |
| ema-smoke | EMA on: the horizon banner prints and a plateau rewind fires |
| production-diagnostic | one --diagnostic run closes the dead-class census, the cut count, the sizes line, the shaded triangle |
| single-phase-demotion | a resmlp config with phase keys trains (previously a traceback); the two-phase model is unaffected |
| head-scheduler-override | a head scheduler with patience 10 cuts the lr on its own cadence |
| eval-batch-invariance | validation chi2 agrees across eval batch sizes to rtol 1e-6 |
| berhu-loss | a head-berhu run shows a plain-sqrt trunk banner and a berhu_capped head banner |
| loss-schema-equivalence | the same config in the nested loss schema reproduces the old epoch lines |
| berhu-anneal | the berhu shape is continuous at the hold boundary and full by epoch 15 |
| ema-anneal | the EMA average appears only after the hold, at the live point |
| param-window-cuts | a tight density window trains end to end and the pool shrinkage matches the banner |
| triangle-shading | (optional) the synthetic four-window triangle shades exactly the coverage panels |
| joint-training | freeze_trunk false trains trunk and head together; phase-2 time sits above the frozen control |
| head-activation-pin | a pinned gated_power head shows in the model-spec banner; the illegal pin errors |
| relu-tanh-norm | tanh with the per_feature / affine norm knob; the banner names the norm and the loss descends |
| weight-decay-census | weight decay touches exactly the Linear / Conv1d / BinLinear weight matrices |
| npce-training | NPCE residual and ratio train, the exclusivity errors fire, a 2-point n_train sweep refits per point |
| save-rebuild-drift | a saved emulator rebuilds bitwise-equal (plain, factored, NPCE, and the conv-head save whose persisted bin split reconstructs the ResCNN), survives a drifted code default, refuses a v1 file and a pre-persistence head file |
| cobaya-adapter | the inference predictor matches the training side to rtol 1e-6, including the factored round-trip |
| finetune-identity | warm-start mechanics: source validation, input-geometry extension, the output pin, epoch-0 parity, anchor masks, the loud config errors |
| finetune-smoke | a real fine-tune run: epoch 0 reproduces the source, then improves on the new data |
| transfer-identity | frozen-base transfer mechanics: the base loads, the correction composes (gain / sum), epoch-0 parity, the family-scope guards |
| transfer-smoke | a real frozen-base + correction run end to end with the collapse bar |
| scalar-identity | scalar save/rebuild/predict bitwise, ScalarGeometry state, auto-provides, the trunk-only head guard, every loud error leg, scalar finetune parity |
| scalar-smoke | a real scalar train on the analytic omegamh2 fixture: collapse, off-center predict, the cobaya evaluate readback, the scalar diagnostics pages |
| cmb-identity | the ruled sigma_l constants, the amplitude law both ways, save/rebuild bitwise, the roughness legs, CMB finetune parity, the ResTRF correction-head leg (identity basis, n_tokens, the head rebuild round-trip), and the five non-Gaussian covariance legs against an independent known-answer calculation (the exact contraction, the old-weight miss, raw-vs-scaled fixture integrity, the width-3 band projection, the exact zero-band weight) |
| cmb-smoke | the CMB pipeline end to end on real CAMB: generator, covariance-script execution and structure — including the non-Gaussian path run end to end with its normalization asserted in the output's provenance — training with the relative collapse bar, the cobaya Cl lifecycle, the CMB pages. The normalization's numerical truth rides cmb-identity's known-answer legs (notes/families-scalar-cmb.md) |
| bsn-identity | the background pipeline vs closed-form flat LCDM, the log_offset law, the piecewise windows + the loud desert, save/rebuild bitwise, BSN finetune parity |
| bsn-smoke | the BAOSN pipeline end to end vs CAMB's OWN background (truth available): generator + the stale-cache tripwire, two trainings, the cobaya getters within 2%, the grid pages |
| mps-identity | grid2d geometry + the staging law + the constant-column pinning + emul_mps assembly math on stub bases + MPS finetune parity + the ResCNN correction-head leg |
| mps-smoke | the MPS pipeline end to end on real CAMB (law-none path): generator, two trainings, the get_Pk round-trips, the matter-power pages |
| geo-paths | fresh saves write the geometries/ folder class paths, the legacy flat paths are DEAD (loud), and the tree census is clean |

## How to read a log

Each `logs/<test>.log` opens with a header naming the test, its home
note, and the git HEAD. Then it streams every command's full output
live. Each check writes a line

```
[harness] CHECK <what was checked>: PASS  (the value behind it)
```

and the file ends with `[harness] GATE <test>: PASS` or `FAIL <reason>`.
`logs/BOARD.md` is the one-line-per-test summary; a review reads the raw
logs, not the summary. Each test's home note (named in the header) is the
definitive spec for what that test must prove.
