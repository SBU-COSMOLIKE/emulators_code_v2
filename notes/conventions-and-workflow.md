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

Red-team documentation census at HEAD 32f7545 (2026-07-12): 92 Python
files; 6 lack a module docstring, 175 function/method definitions lack
a docstring, and 6 small gate-stub classes lack one. Those raw numbers
include the deliberately verbatim lensing generator, the vendored
Syren files, nested callbacks, and test doubles, so they are a census,
not permission for a 175-block context dump. The actionable first
slice is:

- add concise module contracts to the three new generator siblings
  (CMB, background, MPS); keep the verbatim lensing and vendored Syren
  exceptions explicitly recorded rather than silently normalized;
- document public/runtime boundaries first, including the five missing
  `save_emulator` arguments and every multiprocessing callback contract;
- keep trivial private callbacks/test doubles to one-line purpose
  docstrings where formal blocks would repeat the signature; a useful
  small contract beats bulk prose that hides the code;
- remove the six remaining unambiguous internal-ledger leaks from
  Python prose: `emulator/geometries/__init__.py` and
  `gates/checks/geo_paths.py` say "GEO unit";
  `emulator/losses/cmb.py` says "CME resume" and exposes "CME registry"
  in a user error; the scalar/transfer identity checks retain "FTW" in
  comments. Replace each with the plain-language fact and note-file
  pointer where a pointer is useful.

The documentation cleanup is a separate doc-only commit. Prove it with
the AST-minus-docstrings hash, rerun the exact internal-code scan, and
report the new census; do not mix it into a numerical or lifecycle fix.

The same AST pass found runtime validation expressed as `assert` in 17
places across `batching.py`, `designs/blocks.py`, `designs/plain.py`,
`designs/ia.py`, `geometries/output.py`, `geometries/parameter.py`, and
`losses/core.py`. These are not developer-only impossibilities: they guard
user data widths and positivity, model dimensions/groups/kernel parity,
required geometry metadata, and a probe/layout assumption. `python -O`
removes every one and lets invalid state reach division, reshape, indexing,
or model construction. The hardening unit replaces public/config/data
guards with explicit typed exceptions before mutation or accelerator setup;
an optimized-mode subprocess must reject the same negative fixtures with the
same messages as ordinary Python. Keep true internal invariants as explicit
exceptions too when continuing would produce scientific output rather than
an immediate harmless crash.

**Internal tracking codes stay in notes/ (user ruling 2026-07-12).**
The design-decision codes (D-CM9, D-MP2-A, TPE-2, MPS-DIAG, board
ledger keys like GB-C...) are Architect bookkeeping — the user hit
"the D-CM9 dispatch" in a docstring and could not parse it. They may
appear ONLY in notes/ (and as the registry's `spec_code` data field in
gates/board.py, documented as a notes-ledger key and never rendered).
Everywhere else — READMEs, docstrings, comments, exception/print
strings, gate report labels, YAML comments — the code is replaced by
the plain-language fact it stood for, or dropped when the sentence
already says it; pointers to note FILES ("spec:
notes/families-scalar-cmb.md") remain the crosswalk. The 2026-07-12
sweep translated ~470 hits across ~55 files; the recurring phrases
live in the glossary inside that sweep's commit message. New code must
be written this way from the start: state the fact, cite the note
file, never the code. Verification lesson (red-team catch, same day):
the sweep's grep required a hyphen/digit after the unit prefixes
(FTW-1, TPE-2...) and missed BARE codenames ("the FTW machinery",
"the CME registry" — four instances, one in an exception string). A
code-leak grep must include the bare unit names too:
`\b(FTW|TPE|GRF|GBC|POL|SPE|CME)\b` (minus Optuna's TPE sampler in
the tune driver).

Second verification-grep lesson (2026-07-12, the geometry-clip
retraction): a grep whose output feeds a NEGATIVE claim ("no X exists
anywhere") must never be truncated — the Architect piped a clip search
through `head -20` and the two real `np.clip` hits sat past the cut,
producing a false "no clip exists" correction the red team had to
reverse. Rules: (a) count first (`grep -c`) or read the whole output
before asserting absence; (b) sweep the synonym set in one pattern
(`clip|clamp|maximum|where`); (c) a negative claim carries the exact
pattern + scope it was checked with, so the reader can re-run it.

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

## The Fable/Opus workflow + the independent Codex red team

- The main Architect is Fable (`.claude/FABLE_ROLE.md`); the Implementer is
  Opus (`.claude/OPUS_ROLE.md`). Codex is a separate, independent red-team
  reviewer (`.codex/REDTEAM_ROLE.md`), not a replacement for Fable and never
  an Opus co-implementer.
- In the Fable/Opus loop, the user relays
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
  return as delta re-handoffs (D-XX-N IDs). Authorship discipline
  (red-team rule, 2026-07-12): Implementer records say "awaiting
  Architect audit" — an Implementer never pre-writes an Architect
  verdict, invents an Architect probe, or claims Architect
  co-authorship; audit text is written only by the Architect, after
  the audit.
- Codex independently red-teams the code, Python documentation, READMEs,
  notes, gates, and Implementer returns. It challenges green evidence,
  searches for skipped failure paths and counterexamples, and reports through
  `ARCHITECT_REDTEAM_HANDOFF` blocks ending exactly with
  `ARCHITECT_REDTEAM_HANDOFF ENDS`. Codex records its findings without
  impersonating or modifying Fable's role and does not merge to main.
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

Added by the 32/32 board saga (2026-07-11/12; the run-by-run table is
in gates-and-board.md):

- READ `HEAD at run:` in a gate log BEFORE reading its failure —
  three runs in one night re-tested old code and their reds were
  already fixed (the pull/merge race is the default failure mode of
  a two-machine loop).
- A gate fixture MIRRORS the shipped example YAML, never re-types
  its keys from memory (the covariance fixture re-invented two of
  the example's conventions wrong, one board run each).
- A hand-built fixture value that mirrors a REAL run value must
  DERIVE every coupled width from it, never hardcode (the 4-wide
  fracs row vs the 5-entry DEFAULT_THRESHOLDS, in all four smoke
  gates at once).
- The "$ROOTDIR for cobaya-run" rule extends to IN-PROCESS get_model
  with cocoa's relative theory paths — check scripts resolve
  `external_modules/...` paths absolutely from $ROOTDIR (subprocess
  legs with cwd=rootdir never hit it, which is why it hid).
- When a guard needs a carve-out, carve on the PHYSICS axis, not a
  config axis the physics does not respect (D-MP9: partial-constant
  = flat physics under ANY law; whole-constant = dead dump).
- When a mechanism hypothesis about third-party internals fails once
  on the real machine, stop patching around it and switch to the
  documented API path (the wants-Cl quirk vs
  logposterior(cached=False)) — and build the tripwire that can
  falsify the hypothesis BEFORE trusting the fix.

## Generator entry files must be self-teaching (red-team 45M-03, 2026-07-12, Architect-VERIFIED; queue 31)

The four production generator entry files
(compute_data_vectors/dataset_generator_{lensing,cmb,background,mps}.py)
have no module docstring (AST-verified on all four), open with imports
and C-style separator banners, and their central override
_compute_dvs_from_sample — the one family-specific physics boundary
(one sampled row in, reordered, Cobaya provider populated, physics
executed, payload returned to the shared core) — has no formal
docstring in any of the four. Three of the four tell the reader the
shared flags are documented in the lensing file. The family
_read_train_args and multi-file store hooks are uneven: prose without
Arguments:/Returns: contracts.

Contract (Implementer; in-file documentation repair, no new note file,
plus small corrections to the existing README tables if needed):

1. A real module docstring before the imports in each file: what the
   file produces and which physics engine computes it; a short flow
   diagram — one sampled row -> reorder into YAML order -> Cobaya
   provider -> physics call -> family payload -> GeneratorCore store
   hooks — with every symbol defined and the statement that the
   callback runs once per requested cosmology.
2. The subclass contract stated locally: GeneratorCore owns sampling,
   MPI, checkpoints, failure flags, and publication; the family class
   owns VALID_PROBES, extra generator-YAML keys,
   _compute_dvs_from_sample, and any multi-file store hooks.
3. _compute_dvs_from_sample gets formal Arguments: / Returns: /
   Raises: blocks with exact shapes, units, dtype, and
   dictionary/spectrum ordering.
4. The private Cobaya component loop explained step by step: what
   _component_order contains, what _params_of_dependencies
   contributes, why cached is passed, what terminal output is
   captured. SEQUENCING: unit 33 (45M-06) rewrites that loop — land 33
   first (or in the same handoff) so this documentation describes the
   surviving form, not the retired one.
5. Every store hook defines whether it allocates, writes in place,
   appends, loads, or returns a copy/view, and names every file member
   it owns.
6. One compact runnable command per family, or a direct link to the
   main README's exact family subsection, while still defining the
   local flags a reader needs.
7. All-caps emphasis and C-style separator walls replaced with
   ordinary headings and formal sentences (MPI, CMB, CAMB, MPS stay
   uppercase); internal decision codes stay in notes/.

Acceptance is mechanical: all four modules have docstrings; every
override hook has the formal blocks appropriate to its behavior; a
generated API inventory lists no undocumented family callback; the
four files remain syntax-clean.

## Public prose states the current state (red-team 45M-07, 2026-07-12, Architect-VERIFIED; queue 34)

Three verified classes of drift from current-state, reader-facing
documentation: emulator/README.md:216 narrates audit history ("board
run 12"; a fixture that "awaits one final identity-gate rerun");
emulator/warmstart.py:162 raises a user-facing error naming internal
scheduling ("it lands as unit 2" — a user cannot act on a queue
number); and ordinary words used as all-caps emphasis persist in
public prose (README examples: INPUT :24, ONE :68/:136/:183/:347).

Contract (Implementer; documentation-only):

1. Public READMEs state the current capability, the current
   limitation, and the relevant notes/ pointer. Board-run history and
   rerun bookkeeping move into gates-and-board.md (no new note file).
2. The warm-start exception becomes a plain current-state explanation
   with an actionable remedy; cross-reference the already-queued
   fine-tune-anchor unit internally; no duplicate implementation unit.
3. Prose emphasis capitals become formal, definitional wording.
   Genuine names — CMB, CAMB, MPI, GPU, LCDM, configuration constants,
   exact literals where case is part of the interface — keep their
   case.
4. Acceptance: the prose diff plus a COMPLETE repository-wide scan —
   an untruncated grep is the evidence; a clipped head result may not
   feed the zero-matches claim. No torch, workstation, or board gate
   needed.

### 45M-17 fold-in: the loss decode docstring lies about its shape (2026-07-12, Architect-VERIFIED; rides the documentation batch, units 31 + 34)

CosmolikeChi2.decode documents "(B, total_size) physical dv scattered
to full length" (losses/core.py:165) but returns
self.geom.decode(whitened_sq), and DataVectorGeometry.decode
(geometries/output.py:506-512) explicitly takes and returns the KEPT
width — unwhiten + center, no scatter; the scatter is the separate
geom.unsqueeze(...) call, as EmulatorPredictor.predict correctly
performs. A consumer following the docstring hands a shortened vector
to a full-vector likelihood, or scatters twice. Correction (docs-only
unless the audit below finds a caller relying on the false shape):
the return contract becomes (B, n_keep); state that decode inverts
the numerical transform ONLY and does not restore masked positions;
show the full-vector chain (kept = chi2fn.decode(...), then
full = geom.unsqueeze(kept)); audit every loss subclass's decode
documentation (RescaledChi2 :544, ResidualBaseChi2 :661) for the same
kept-vs-full distinction and every decode caller for a reliance on
the false shape; keep the diagonal-family wording separate — their
n_keep == total_size coincidence must not redefine the generic
contract. No new note file, no new gate: the existing inference shape
gate is the runtime evidence.

### 45M-41 fold-ins: three didactic comments teach false mechanics (2026-07-12, Architect-VERIFIED; documentation-only, each clause folded into its owning unit)

1. WEIGHT-DECAY SELECTION IS ROLE-BASED, NOT SHAPE-BASED. Affine
   (blocks.py:51) and FeatureAffine (:95) both teach "make_optimizer
   decays only ndim >= 2 weight matrices" — the abandoned rule.
   make_optimizer's own docstring states the real mechanism: the
   .weight of every nn.Linear / nn.Conv1d / BinLinear, "by module
   role, not tensor shape". Consequences worth teaching correctly: a
   future 2-D activation parameter stays UNdecayed, and a future
   weight module missing from the allowlist stays UNdecayed too (the
   safe default). FOLDED INTO UNIT 49 (optimizer execution protocol):
   both explanations replaced with the exact owner allowlist +
   safe-default statement.
2. TWO GEOMETRY ERRORS REVERSE ENCODE AND DECODE.
   ScalarGeometry.from_targets' zero-variance error says a tiny scale
   "would make decode divide by near-zero" (scalar.py:127) — but
   decode MULTIPLIES by scale; ENCODE divides ((y - center) / scale,
   :178). CmbDiagonalGeometry repeats the reversal for sigma
   (cmb.py:186 "decode would divide by it") — whiten/encode divide,
   unwhiten/decode multiply. The guards are correct; the taught
   direction is backwards. FOLDED INTO THE DOCUMENTATION BATCH
   (units 31 + 34, beside the 45M-17 decode-shape fold-in — the same
   API-docstring-truth class). NB: the red team addressed this to "a
   geometry totality unit"; no unit carries that name, and the docs
   batch is the honest owner — recorded as a deviation from their
   addressing, not from their contract.
3. THE AMP DOC CLAIMS BFLOAT16 UNIVERSALLY (training.py:1560 vs the
   float16-on-MPS selection at :1702). ALREADY IN UNIT 51's contract
   ("documentation corrected to name float16-on-MPS /
   bfloat16-on-CUDA-CPU") — cross-referenced, no double work.

Shared completion condition, all three clauses: after editing, an
UNTRUNCATED repo-wide search for the old phrases ("ndim >= 2",
"decode divide", "decode would divide", the bfloat16 use_amp claim)
returns zero stale hits — a clipped result may not feed the claim.
Replacements stay formal and definitional (owner role, tensor
operation, direction), never vague "normalization failed" prose. No
new gates: the parent units carry the executable evidence.
