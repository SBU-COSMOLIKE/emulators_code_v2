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

Domain-symbol names must not collide with a different established cosmology
quantity. In particular, reserve `h` for the dimensionless Hubble parameter
`H0/100`; the covariance finite-difference control is `step_frac` in Python
and `s_step` in prose/equations, never an unexplained local `h`. This applies
to code, comments, diagnostics, notes, and handoffs: a reader must not have to
infer which scientific quantity a one-letter name means from context.
For covariance calculations, "reasonable cosmology" means the explicit
Planck-LCDM fiducial recorded in `example_yamls/cmb_covariance_lcdm.yaml`, or
a scientifically justified neighboring cosmology. An extreme synthetic fake
can prove validator catch-power but cannot, on its own, prove the science
answer wrong.

Red Team scope ruling (user, 2026-07-13): this is a cosmological research
code run for emulator production and MCMCs, not a public security boundary.
Do not spend audit time on cybersecurity, hostile-user threat models,
permissions, secrets, network attacks, or exploit hardening. Manifest and
artifact checks are reviewed only where they affect scientific correctness,
reproducibility, stale-test truth, or the exact model/data used by an MCMC.

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

**Standing user ruling (2026-07-13): a README presents the current
library; it is not a diary of how the library was developed.** The root
README teaches what the library does, how to configure it, what it writes,
and the restrictions a user must act on. `emulator/README.md` teaches the
current ownership map and a novice reading order. `gates/README.md` teaches
how to operate and interpret the acceptance system. Dates, board-run
numbers, measured proof errors, fixture and rerun status, landing or queue
state, rejected alternatives, retired formulas, and the reasons an abandoned
design failed belong in `notes/` or the TeX manuscript, not in a README.
Scientific source attribution may remain where it defines an implemented
formula; benchmark comparisons and literature discussion belong in the
paper. A present limitation may stay only as **scope, consequence, and user
action**. It must not narrate when the limitation was found or how a future
repair is sequenced.

Parentheses carry only a short local definition, symbol, unit, or acronym.
They do not carry an essential algorithm, qualification, or second argument.
If removing a parenthetical would change what the reader must know, promote
it to a complete sentence, table row, or diagram label. The documentation
pass flags parentheticals longer than twelve words or containing more than
one clause for human review; equations, links, and code examples are exempt
from that candidate scan. This is a readability review, not a punctuation
quota.

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

### Notes-first inter-agent communication (hard user rule, 2026-07-13)

The detailed message between Fable, the Implementer and the Red Team must be
written to the appropriate `notes/` file before its chat handoff is emitted.
The note contains the complete reasoning and execution record: the bounded
scope, scientific or numerical evidence, counterexample, contract, file and
line anchors, changed files, branch or commit identity, raw-test locations,
open obligations and acceptance conditions. A chat handoff is a compact
routing summary. It cites the note and says what changed, what is ready for
review and what remains blocked. It does not duplicate the full record.

This rule applies to findings, adjudications, implementation returns, audit
holds, audit approvals, retractions and queue changes. A chat-only decision
is not durable and cannot be treated as the program's current instruction.
When a summary and its cited note disagree, the current note is the source of
record. `notes/MEMORY.md` continues to tell a cold reader which topic note or
registry to open first.

Relay transport copies (Fable addendum, 2026-07-14): the clipboard router
`tools/handoff_router.py` archives every captured chat block under
`notes/relay/` and its local gate logs beside them. Those files are
TRANSPORT COPIES for traceability only -- they are never the source of
record, never cited as evidence, and never edited. The agent-written note a
block cites remains the record; the router's gate log is corroborating
input to the Architect's audit, which still performs its own re-runs.

The mailbox (Fable addendum, 2026-07-14): `notes/mailbox/` holds pending
routing summaries as one file per message, `NNN-to-<fable|opus|sol>.md`;
`tools/mailbox_daemon.py` dispatches each to its addressee's headless CLI
and moves it to `notes/mailbox/done/`. Mailbox files are routing summaries
under the notes-first rule — the substance stays in the cited note. An
agent finishing a mailbox-dispatched turn writes its outbound handoff as
the next mailbox file, so the loop continues without a human relay; merges
and pushes to main remain the user's alone.



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
## Continued red-team documentation campaign: code must teach the current program, not its audit history (2026-07-12)

### 45M-85: internal audit identifiers remain in executable Python prose

At HEAD `05d4937`, an untruncated `rg -n '45M' emulator gates --glob '*.py'`
returns 61 lines.  They occur in module docstrings, comments, gate leg lists,
and mathematical explanations.  This is the same user-facing leak as the
earlier design-ledger codes: `45M-60` tells a new reader nothing about a
roundoff band, a safe square root, or an independent known answer.

Required documentation contract:

- Remove every red-team/audit identifier from `emulator/` and `gates/`
  prose.  Identifiers remain in `notes/` only.
- Replace each occurrence with the current-state fact.  “The contraction
  width sets the roundoff band” is useful; “45M-60” is not.
- Delete review biography (who ruled, which run reopened it, what the old
  gate said) from the runtime explanation.  Preserve that history in the
  owning note and preserve a short note-file pointer only when a reader needs
  the design record.
- Keep identifiers and function/class names that the program executes.  This
  is a prose-only removal, not a rename campaign.
- Prove completion with an untruncated zero-hit scan over Python files and an
  AST-with-docstrings-stripped hash showing no executable change.

### 45M-86: the experiment lifecycle is buried in three 700-line methods

The public orchestration surface is described locally but not teachably.
`EmulatorExperiment.from_config` is 708 lines, `build_geometry` 739,
`build_specs` 259, and `train` 206.  `training_loop_batched` is 704 lines and
`run_emulator` 770.  The comments inside them repeatedly explain a historical
ruling or say “same as the other path,” while the reader needs to know which
state exists before and after each method.

Required documentation contract:

- Put one lifecycle diagram at the class boundary: resolve paths; validate
  exactly one family; choose model class; stage train/validation; construct
  parameter and output geometries; construct the loss; build model/optimizer/
  scheduler specs; train; persist.
- For every stage, name its inputs, the instance attributes it creates, which
  work is eager or deferred, and which state a sweep deliberately reuses.
- Add a family decision table showing the scalar/CMB/grid/grid2d/cosmolike
  differences.  Do not repeat five copies of activation precedence,
  fine-tune inheritance, and transfer setup in prose.
- Define `classmethod`, `cls(...)`, `**kwargs`, capability flag, cached state,
  and alternative constructor at first use.
- Replace “same rule as” comments with the actual rule or a pointer to one
  shared, nearby definition.  A reader who enters the grid2d branch cold must
  not need to read the scalar branch first.
- Separate current mechanics from the configuration-key catalog.  A method
  docstring should teach its state transition; the README/YAML reference owns
  the exhaustive key table.
- Where a 700-line method still performs several independent transitions,
  split cold-path orchestration into named helpers.  The refactor is accepted
  only with compile, binding, leftover-pattern, and behavior gates; comments
  alone cannot make an unbounded branch cascade auditable.

### 45M-87: warm-start and transfer prose begins after the hard tensor step

`warmstart.py` and `losses/transfer.py` describe the high-level intent well,
but the executable comments jump directly to `n_s`, `n_s'`, `n_n`, “block
extension,” “parity,” “pin,” and packed `[base ; truth]` targets.  The compact
slices and concatenations are the part a first-time PyTorch reader cannot
infer.

Required documentation contract:

- Give one small named-column example: a source with three parameters, a new
  run with two extra parameters, and the exact encoded column order before
  and after extension.
- Draw the source-to-new input-weight transfer with concrete shapes.  Define
  dimension 0 as output neurons and dimension 1 as input features.  Explain
  why shared columns are copied, new columns are zero-filled, and `clone`
  creates independent storage.
- Explain every slice in `_shared_columns`, `extend_input_geometry`,
  `transfer_state_dict`, and `_base_input`.  State whether it is a view or a
  copy and which parameter names occupy it.
- Define `torch.no_grad` as “do not record operations for gradient
  calculation,” then contrast the frozen base with live refine mode.  Name
  which optimizer owns the correction and which owns the base in each stage.
- Expand the packed target with shapes: plain transfer stages
  `[base prediction ; truth]`; factored transfer stages one block per template
  plus truth.  Explain that batching caches the frozen base once, while the
  loss unpacks it on every minibatch.
- Define parity as an executed epoch-zero equality check and state what is
  compared, in which coordinate system, with which dtype/device, and why
  floating-point reduction order prevents a blanket bitwise claim.
- Until an advertised feature is reachable, documentation states that it is
  refused today.  Unreachable validation code and “lands as unit” biography
  are not a current API explanation.

### 45M-88: gate files describe audit chronology instead of teaching evidence

The identity gates open with long “Legs” inventories containing terms such as
mutation control, catch power, stale leg, law-space pin, lifecycle, monkeypatch,
and bitwise identity.  Representative 80--120-line `check_*` functions have
no docstring at all.  An AST census at HEAD finds 82 public gate functions
without a docstring.  The number is a triage measure, not a demand for 82
boilerplate blocks.

Required gate-documentation contract:

- Begin every check file with the user-visible promise it tests, the required
  dependencies, and the reason the check belongs on the board.  Define
  “gate” as a test whose failure blocks acceptance.
- For each nontrivial `check_*`, document four objects: the system under test,
  the fixture (small constructed input), the independent expected answer, and
  the deliberately broken implementation or input that proves the assertion
  can fail.
- Define test double, fake, stub, monkeypatch, fixture, known-answer test,
  control arm, mutation arm, and catch power before using them.  Prefer plain
  wording in report labels.
- Show execution order: arrange the input, call the real public boundary,
  compute an independent answer, compare, record failure, and make `main`
  return nonzero.  Explain the module-level `FAILURES` list as shared test
  state and why every helper contributes to the final exit status.
- Historical deleted legs and board-run stories move to notes.  The Python
  file documents what the current gate executes.
- A fake must state exactly which external behavior it replaces and which
  behavior it cannot prove.  “Real function” must name the boundary actually
  called.  A numerical reference must not be produced by the same helper as
  the value under test.
- Prioritize the long undocumented public checks and the nested fake APIs.
  Trivial `forward` methods may use a one-line purpose docstring; bulk text
  that repeats a signature is not an improvement.

Acceptance combines the zero-audit-code scan, the existing board behavior,
and a reviewer exercise: starting from one identity file alone, a new reader
can say what would fail if the production formula were replaced by the
mutation without consulting a note ledger.

## Structured evidence map — gate contract anchors (45M-72 foundation)

The board's structured evidence map (`Gate.evidence`) pins each migrated
gate to a stable, runner-validated anchor in its home note; the mechanism
and the audited rollout are documented in `gates-and-board.md`. The two
workflow-side gates anchor here:

<a id="cli-strict-strict-parse"></a>
**cli-strict (CLI-A) — every public executable rejects a misspelled flag.**
All eight public entry points parse with `parse_args` (never
`parse_known_args`), and two representative driver mains reject a misspelled
flag (`--activaton`) with a nonzero exit before the expensive boundary,
while a valid command line reaches it.

<a id="family-first-family-owned"></a>
**family-first (FAM-A) — every driver owns exactly one data-block family.**
A direct cosmic_shear run owns the cosmolike data-vector family and rejects
a CMB / grid / grid2d / scalar YAML naming its driver; a clean cosmic-shear
YAML trains; the per-family wrappers accept their own block. The census
confirms the four cosmic_shear drivers default `family=cosmolike`, always
check, and drop the dispatcher prose.

## Texnotes Current-gap paragraphs: triage + the currency rule (Architect, 2026-07-13)

texnotes/emulator_code_guide.tex teaches the package with labeled
"Current gap" paragraphs (20 as of 2026-07-13) so a reader never
mistakes documented intent for shipped behavior. Architect triage of
all 20 against HEAD:

- SIXTEEN document known defects that already carry adjudicated unit
  specs — no new units were needed. Map (guide line -> owner):
  :450 config-surface totality -> units 23 + 29 + 59; :565 no-cut
  sweep pool counting -> the data-selection-truth spec
  (data-generation-and-cuts.md "No-cut learning-curve pool
  counting"); :604 parallel study parent -> the parallel-truth item
  (+ unit 55); :664 bake-off liveness -> the activation-bakeoff
  liveness item; :771 dataset certification -> checkpoint-set
  integrity + units 56/57 + the ingress cluster; :2238 NPCE domain
  policy -> unit 46; :2652 fused/closure optimizer -> unit 49;
  :2720 ramp direction -> unit 18; :3032 MPS float16 scaler ->
  unit 51; :3169 finetune.anchor -> unit 24; :3734 background
  zero-anchor extrapolation -> unit 58 (wave-4 background visit
  15+58+62); :3995 diagnostics NaN totality -> the
  validation/diagnostic-memory-truth item; :4105 two-file
  transaction -> the artifact-pair-integrity item; :4261 adapter
  value-schema -> units 15/58/62 + 16/63 + 65; :4283 CMB multipole
  identity -> unit 47; :4308 finite-verdict-before-provider-read ->
  the acceptance-fact ordering spec + unit 33.
- TWO are STALE — the code already moved: :4691 (dirty-tree watch
  scope; queue 1c landed _EXECUTABLE_DIRS covering all five
  executable roots) and the raw-log half of :4723 (queue 1a made
  stale-log a first-class non-green resume state). The refresh is
  RED-TEAM work under the custody rule below.
- TWO are IN FLIGHT: the digest half of :4723 is queue 1b (building
  now); :4666 declared-vs-executed reconciliation is queue 2 (next).
- ONE was blocked on the user and is now RESOLVED: :3885 sigma8
  radius — USER RULING 2026-07-13 = R = 8 Mpc/h (recorded in
  families-background-mps.md "USER RULING (2026-07-13)").

THE CURRENCY RULE + GUIDE CUSTODY (binding; ownership corrected by
USER RULE the same day): texnotes/emulator_code_guide.tex is
RED-TEAM-OWNED — neither the Architect nor the Implementer edits it
(user instruction, 2026-07-13). A landing that changes behavior
taught by a Current-gap paragraph therefore does NOT carry the guide
edit itself; instead the landing's notes entry NAMES the affected
paragraph(s), the Architect carries the owed delta into the next
ARCHITECT_HANDOFF_FOR_THE_RED_TEAM block, and the RED TEAM updates
the guide — closing a gap rewrites the passage to the new behavior;
narrowing one rewrites it to the remaining gap. A Current-gap
paragraph is a contract surface, not decoration: a stale "gap"
teaches a defect the code no longer has, which is the same falsehood
as documenting a feature that does not exist. The queued full
line-by-line guide review (a separate Architect item) verifies the
remaining paragraphs against code and hands its findings to the red
team the same way — the Architect reads and audits the guide, only
the red team writes it.

## Landing-block resync ritual (2026-07-13)

The user's landing merges create merge commits on main that never flow
back to the working branch, so the branch and main DIVERGE after every
landing; the next `git merge` is then a true merge (editor prompt,
surprise merge commit) instead of a fast-forward — the "this caused
problems" incident of 2026-07-13. The fix is a standing ritual: after
a landing merges (and at Architect turn start when main moved), the
worktree branch is fast-forwarded up to main FROM THE WORKTREE —

    git merge --ff-only main

— which is content-identical (the merge commit adds ancestry only),
preserves any uncommitted work in the shared tree, and fails harmlessly
(`--ff-only`) if new branch commits landed meanwhile. With the branch
resynced, every subsequent landing block fast-forwards. Landing blocks
themselves stay the user's four lines, unchanged.

### Guide review of 2026-07-13 (user-authorized Architect edit)

The user ordered a direct Architect review-and-edit of the guide
("you can edit") — an explicit one-off exception to the custody rule
above; custody returns to the red team afterward. What the review
found and did:

- EDITORIAL PASS: a separate prose-quality review, run against the
  user's private review standards (which live outside this repo and
  are not restated here), required NO text changes — the guide's
  register already meets them. An apparent "{m km}" TeX typo was a
  terminal rendering artifact of {\rm ...}; the source is correct.
- CURRENCY: the red team had already refreshed everything through
  THIS MORNING's adjudications (stale-log first-class, the strip()
  head-line story, the RT-02 ownership gap incl. the derived-cached-
  tensor case, the RT-05 flat-only gap with the exact sinh
  counterexample, the RT-06 latex gap, the sigma8 8/h Mpc ruling).
  The ONE stale area was the digest story, written before 1b phases
  1-2 landed: fixed in four passages — the two-regime digest
  narrative, a pre-manifest row in the resume-state table, the
  Current-gap paragraph narrowed to population, and the Required-
  closure paragraph split into current-behavior vs the remaining
  population work. All 22 other Current-gap paragraphs re-verified
  still true against the unit map.
- PDF: rebuilt clean (pdflatex from the REPO ROOT — the figure paths
  are texnotes/-prefixed; two passes, zero errors, zero unresolved
  references) so the tracked build product matches its source. The
  red team's tracked-PDF policy ruling (untrack vs freshness check)
  stays owed.
