# Conventions, workflow, and environment

Consolidated 2026-07-11 from py-module-style-conventions.md,
dual-fable-opus-workflow.md, ai-collaboration-preferences.md, the
readme-campaign notes (yaml-chapter, reorg-two-readmes,
precedence-appendix, run-it-definitions), the micro-convention notes
(no-global-variables, hanging-indent, construction-via-spec-dicts,
docstrings-formal-arguments, plots-no-red-green, terminal-output,
locate-notebook-edits, notebook-to-python-translation), the
environment notes (cocoa-rootdir-env, dev-machine-mac-m2-32gb,
test-workstation-gpus), the two audits, and the session-status
snapshots (retired; full texts in git history).

## Python house style (emulator/ + drivers; 90 columns)

- NAMED parameters everywhere the callee allows ("I will forget the
  meaning of position X"); irreducible positionals (matplotlib x/y,
  einsum operands, the tensor subject of cat/zeros, model(x)) stay
  positional WITH a naming comment; keep *args forwarders positional
  (keywording pred= before *args = the "multiple values" bug).
- Paren-alignment, one item per line (covers dict pairs and tuples);
  90 cols is the hard gate; over-90 fallback = 2-space hanging
  indent, one style per file. (The teaching notebook pytorch1.ipynb
  is separate: ~60 cols, hanging indent, READ-ONLY reference.)
- NO comprehensions in non-hot code — explicit C-style loops; KEEP
  vectorized numpy/torch and anything inside forward()/batch loops;
  find with an AST scan, not grep. The user is mainly a C coder: no
  "Alien Python" (walrus, nested comprehensions, lambda-where-a-def-
  reads-better, starred gymnastics, ternary PILEUPS — single
  ternaries are fine, C has ?:). Hot paths are never slowed; the fix
  there is a better comment.
- No silent module-global DATA reads in functions (symtable audit);
  the one sanctioned exception carries
  `# WARNING: reads module global X` on that line.
- Spec dicts {cls, **kwargs} + make_X helpers for every constructible
  component; computed/device args injected by the helper, never in
  the dict.
- No all-caps emphasis (acronyms + the WARNING marker exempt); no
  ` -- ` double dash — both rules extend to argparse help, log lines,
  and error messages (they are prose the user reads).

## In-file documentation

Module docstrings are prose (subject + verb). Every function gets a
formal `Arguments:` block naming EVERY parameter + `Returns:`; a
block-dict parameter enumerates every key. `PS:` jargon glossary per
file that uses a term (define-or-drop: the audience is cosmologists —
one undefined term of art in the opening sentence costs the whole
header; check scripts especially must read as plain English under
stress). Cross-module call sites get `# fn (module.py): what it does`
provenance. Math as display formulas with named symbols. Shape-flow
diagrams for tensor pipelines — every one ends with `(legend: ...)`
defining every symbol; magic numbers only as named-symbol derivations
with the LSST-Y1 concrete example. Enumeration rot is a named defect
(key counts and lists go stale — grep for them on every schema
change). Doc-only passes are PROVEN by the AST-minus-docstrings hash
census, never asserted.

## README / didactics

Two-README split (user philosophy: first learn to RUN and configure,
only later how the code works): the main README = Run it -> the YAML
chapter -> the family sections -> generation -> appendices, AI-Usage
LAST and verbatim the user's two sentences; emulator/README.md = the
code map, "every file's functions" LAST. Definitions before use
(vocabulary box; never invert the dependency); every ### YAML-knob
subsection carries its own <=10-line YAML block with template-verbatim
values; every README passage explaining a YAML concept carries a
fenced snippet of the REAL block (prose-only unacceptable); equations
verbatim from the code; dedup by pointers ("point, never restate");
ASCII flow diagrams (the user loves them). GitHub math policy (the
five-rule scanner): no backslash + ASCII punctuation inside math, no
LaTeX environments, no line-initial Markdown tokens inside $$ blocks,
no whitespace-adjacent $ spans, no code-name underscores in math
(single-letter symbols + a legend). No workstation assumptions in
user-facing docs (notes/ exempt). Acceptance for README work: the
anchor census (every #link resolves) + the path census (every
backticked repo path exists).

## Plots, terminal, YAML

- Plots: never red+green; explicit color= from the colorblind-safe
  palette ["#0072B2", "#E69F00", "#CC79A7", "#000000", "#56B4E9"];
  viridis for cmaps; vary linestyle for grayscale.
- Terminal: essential-only — headers, verdicts, one-line details,
  artifact paths; full streams to per-run LOG FILES (never thinned);
  a debug key/flag restores the mirror.
- YAML: block style, one key per line, never inline {...}; values
  column-aligned when the file does; [default, min, max, kind] range
  convention; EVERY YAML change (keys, values, comments, alignment)
  is reported as a paste-ready block in context, never prose.

## The dual-agent workflow (Fable Architect + Opus Implementer)

- Roles in .claude/FABLE_ROLE.md / OPUS_ROLE.md; the user relays
  ARCHITECT_HANDOFF / IMPLEMENTER_HANDOFF blocks; role resolved ONCE
  at session start (explicit assignment > received handoff > normal
  session); model identity is a sanity check, not the dispatcher.
- The Architect writes no function bodies but DOES write interfaces,
  schemas, and verbatim legacy numerics (paraphrased physics is how
  ports rot); blueprints state goals/contracts/edge-cases/gates,
  never steps; every handoff persists to notes/ BEFORE emission — THE
  NOTE IS THE SPEC OF RECORD; the Implementer executes the note even
  when the relayed block lags it.
- Audit is FABLE DOMAIN (hard user rule): no milestone closes without
  Architect sign-off on RAW evidence (never summaries — repeated
  over-claims proved summaries unreliable); the Architect verifies
  its own harnesses against a known-good case first; audit failures
  return as delta re-handoffs (D-XX-N IDs).
- Propose-don't-guess for design-sensitive layouts (a checkpoint
  proposal in the note for ruling); partial units are an approved
  shape (coherent gated sub-increment + honest remainder); interface
  changes are always DECLARED as deviations; forward-walk the WHOLE
  driver path when adding a config branch; every stop ends with a
  relayable handoff block.
- Git: the user runs all commit/merge/push; sessions leave
  uncommitted diffs + print the complete landing blocks with concrete
  commit sentences; only main is ever pushed (branches stay local;
  the workstation pulls main); occasional explicit TIME-BOXED
  branch-commit authorizations exist — never assumed, never extended.
- The notes ritual: every milestone gets its note updates + MEMORY.md
  index line in the SAME turn, unprompted — an unrecorded milestone
  is unfinished work.

## Environment

- Mac M2 32 GB (dev): python3 is homebrew with numpy + stdlib ONLY —
  no torch/h5py/yaml/scipy/matplotlib/cosmolike, no conda/venv. The
  Mac evidence pattern: py_compile/compileall + AST censuses
  (each-def-exactly-once; keyword-vs-signature; docstring-stripped
  hash) + numeric probes — prefer exec-ing the REAL function body
  under a tensor-like fake on KNOWN ANSWERS over a numpy mirror;
  torch legs ride the workstation board. MPS backend facts: no
  float64 on device, fp16 AMP, no CUDA graphs.
- Test workstation: 2x RTX 3060 12 GB; GPU 0 is a SHARED eGPU, GPU 1
  the internal/display GPU jobs should default to —
  CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=1 (or "1,0"),
  set BEFORE the process starts. NVWULF (8x H200) is the production
  box; nvidia-cuda-mps-control tightens co-located time-slicing.
- ROOTDIR (not in Cocoa's README): exported by
  Cocoa/set_installation_options.sh as $(pwd -P) of Cocoa/ when
  SOURCED (setup/start_cocoa.sh); drivers read os.environ["ROOTDIR"];
  everything anchors ${ROOTDIR:?}/... (projects/, external_modules/
  code/, .local/); cobaya-run must run FROM $ROOTDIR. This package's
  cocoa install path: external_modules/code/emulators/emultrfv2/.
- Multi-GPU pattern: task-parallel (never DDP), processes not threads
  (GIL + cosmolike's global C state), spawn not fork, set_device per
  worker, ram_frac=0 in parallel paths (the private-memmap-copy
  trap), LPT balancing, the Python 3.14 spawn keepalive trap (hold
  Queue/Lock refs until join), Optuna via a JournalStorage file.

## Process lessons that recur

Handoffs paste RAW scan output (claimed numbers were cleaner than
measured, three times); verify every harness on a known-good case
(the two-point C1 check, the em-dash slugger, exclusion greps eating
true positives, head -5 truncating a census); fixing layer N unmasks
layer N+1; sequential commits when units share files; the terminal is
a dashboard, the log file is the archive.
