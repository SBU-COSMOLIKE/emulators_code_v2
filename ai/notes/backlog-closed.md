# Closed ticket archive

Closed tickets moved out of [`backlog.md`](backlog.md) so the Architect
does not load them on every turn. A closed ticket has no missing work of
its own. If later work is still required, its **What is missing** links to
an open ticket in `backlog.md`.

To reopen one, move its whole section back into `backlog.md` above
`# Parked edge cases` and add its `- OPEN` index line there.

<a id="closed-role-context-separation"></a>
## Give each AI role its own context limit

### High-level summary

The Architect and Implementer previously shared one context setting. A large
Architect allowance could therefore exceed an Ollama Implementer's smaller
model context and make even the connection check fail before work began.

Each role now has an independent command-line setting. Choosing a large
Architect context no longer changes the Implementer threshold.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED. Severity: CRITICAL.** The shared setting prevented the selected
three-role configuration from starting. Ping, runtime validation, saved ticket
identity, dispatch environments, terminal reports, tests, and guides now use
the Implementer's own value.

### What is already fixed

`--architect-context`, `--implementer-context`, and `--sol-context` control
only their named roles. The older `--claude-context` spelling remains as an
Architect-only compatibility name. Ollama still reports the model's actual
maximum separately and refuses an Implementer threshold above that maximum.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Focused tests prove that a 300000-token Architect threshold and a 64000-token
Implementer threshold remain distinct during ping, Ollama preflight, saved
runtime validation, and dispatch selection.

</details>

<a id="closed-control-plane-live-state-compatibility"></a>
## Test a proposed controller against the saved state it must inherit

### High-level summary

A replacement mailbox controller must understand work saved by the controller
that is running now. Testing only a new empty repository could approve a new
file format that fails as soon as it encounters a real active ticket,
completed landing, candidate, or recovery record.

The protected check now begins with a disposable copy of the current records.
The proposed controller must read those copies in a fresh process and preserve
their exact workflow meaning before the separate clean-start scenarios run.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** D0 copies ticket and candidate state, saved role-worktree records,
the backlog and recovery records, pending push or note-administration records,
and private candidate or landing refs into an isolated checkout. D1 must read
the copies without changing D0's live bytes or refs.

**Severity: HIGH.** An incompatible protected upgrade could halt the central
mailbox controller after landing and require recovery with the old controller.

### What is already fixed

The D0-owned test compares every active and completed cycle, every candidate
and landing identity, all copied recovery bytes, all saved worktree identities,
and every private ref. A state-schema change requires an exact migration
declaration. D0 runs the named migration only on the copy, starts a fresh D1
process, and rejects missing declarations, wrong versions, or a migration that
drops saved work. The older fresh-state, restart, landing, stale-main, and
health checks still run afterward.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Focused coverage is in
`ai/tests/test_protected_control_plane_shadow.py`. It seeds active and completed
D0 cycles, a candidate ref, push debt, backlog recovery, and three worktree
records; proves the current handoff succeeds; and supplies negative D1 examples
for an undeclared schema, incorrect migration versions, and a migration that
deletes the active ticket. The outer D0 fingerprint covers every copied live
record and the relevant Git refs before and after the disposable run.

</details>

<a id="open-syren-amplitude-aliases"></a>
## Refuse conflicting amplitude names before calculating Syren matter power

### High-level summary

The Syren matter-power formulas accept the primordial amplitude either as
`As`, the usual small number, or as `As_1e9 = 10^9 As`. A Cobaya run may make
both names available because different theory components use different
spellings.

When both names are present, the current helper chooses `As_1e9` without
checking whether it agrees with `As`. A saved network can therefore read one
amplitude while its analytic Syren starting surface uses another. Both results
remain finite, but their combination describes no single cosmology.

This affects the central matter-power prediction rather than a plot or
optional report. A concrete conflicting pair in the permanent scientific note
changes the analytic linear-power baseline by about 77 percent.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 1.**

**Red Team reopening: barred by Architect NO-GO.**

**Red Team reopen 1 — Architect NO-GO (2026-07-19).** The Red Team correctly
observed that `emul_mps.calculate` calls the linear learned predictor
(`p_lin.predict`) once before `syren_params_from` raises, so the earlier
"before any learned predictor" wording overstated the boundary (corrected
below). The durable acceptance authority (`artifacts-inference-warmstart.md`,
"Syren parameter aliases must agree", rules 2 and 9 and the amplitude
paragraph) requires refusal before either Syren analytic formula and leaves no
`Pk_grid`/interpolator/derived state on failure; the landing meets both — both
`base_pklin`/`base_boost` and the boost predictor never run and `state` stays
empty. The finding is Red-Team Low, below the ticket's user-set High severity,
and causes no wrong science, data loss, or halt; the "unmeasurable"
exact-landing budget is a squash-measurement artifact (the binding pre-landing
candidate budget was `within limit`: added 8625, deleted 20, total 8645, limit
15000). Reopening is permanently barred. Full adjudication in
`ai/notes/ticket-syren-amplitude-aliases.md`.

**CLOSED.** Accepted candidate `f30f406ee826a1a1222282370933e62b9837031b` adds
the both-present amplitude-agreement guard to
`emulator/syren_base.py::syren_params_from`, so a conflicting `As_1e9`/`As`
pair raises a `ValueError` that names both values before either Syren analytic
formula (`base_pklin`/`base_boost`) runs and before any `Pk_grid`,
interpolator, generator row, or Cobaya derived state is written. Single-name
and consistent two-name inputs keep their previous numerics. The complete
directive and audit record are in `ai/notes/ticket-syren-amplitude-aliases.md`.

**Severity: HIGH.** The mismatch can silently change the analytic starting
surface used in a central matter-power result. Medium is insufficient because
the current public EMUL2 configuration makes both amplitude names a normal
input shape, and the numerical error remains finite instead of stopping.

### What is already fixed

`syren_params_from` compares `As_1e9` with `1e9 * As` under one documented
absolute-and-relative tolerance sized to float32 storage and refuses a
disagreeing pair with a `ValueError` that names both supplied values and the
conversion. Because both production call sites invoke this one function before
any Syren formula, the generator refuses a conflicting sample before a raw or
base row is written and the adapter refuses before any `Pk_grid` or derived
state exists. `As`-only and `As_1e9`-only inputs keep their previous numerics.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Severity: HIGH BUG FIX. Accepted candidate
`f30f406ee826a1a1222282370933e62b9837031b` on cycle
`open-syren-amplitude-aliases@23d0340587bde038008ddcfa71eea2f2d658ed54`. Owner:
`emulator/syren_base.py::syren_params_from`; witnesses in
`ai/tests/test_syren_dark_energy_coordinates.py`,
`ai/tests/test_generator_dark_energy_facts.py`, and the new
`ai/tests/test_mps_amplitude_aliases.py`. The repair keeps `As`-only and
`As_1e9`-only numerics unchanged, compares `As_1e9` with `1e9 * As` under
module constants `SYREN_AMPLITUDE_ATOL`/`SYREN_AMPLITUDE_RTOL`, names both
supplied values on refusal, and proves no raw/base row or partial `Pk` state
survives a conflict. The two production call sites and the MPS generator and
adapter code are unchanged; the guard lives only in `syren_base.py`.
The audit re-ran every check against the immutable candidate: `py_compile` on
all four changed files; 18, 15, and 2 tests in the three affected modules;
`permanent_note_guard.py` PASS at the base; and the architect-audit
ticket-change guard reporting `within limit` (added 8625, deleted 20, total
8645, limit 15000). The durable rule already existed in
`ai/notes/artifacts-inference-warmstart.md` ("Syren parameter aliases must
agree"), so no permanent note changed.
</details>

<a id="closed-syren-amplitude-aliases"></a>

## Documentation and teaching

### Make backlog tickets and the gate guide readable

**High-level summary.** The backlog previously mixed open and closed work in
one long list, and internal labels such as “unit 8” did not tell a human what
was wrong. The gate guide also compressed tests, gates, board operation, and
workstation rules into language that hid their practical differences.

**Current status.** **CLOSED.** Both guides now pass the human-first contracts
and independent factual review.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The backlog has a linked Open section followed by a
grouped Closed section; every ticket separates its plain summary, status,
completed work, missing work, and technical record. The gate guide gives real
commands and visible results, covers every emulator family, and explains the
runner's actual setup, logs, states, restart behavior, and command options.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `b906147`. Evidence: 335/335 full CPU tests; 32/32
permanent-note and role-contract tests; focused CMB test; live board help,
49-gate list, and dataset-publication dry run; board self-test; exact-commit
eleven-note guard; whitespace and prohibited-word checks; backlog inventory
GO; gate fact-audit GO; and permanent-contract review GO.
</details>

### Remove the obsolete README trimming quota

**High-level summary.** A former ticket required a fixed 15 percent reduction
in README words, even when the words were useful. Detailed teaching now lives
in folder guides, so future trimming is based on clarity instead of a quota.

**Current status.** **CLOSED.** The percentage target has been retired.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** README work may make small natural cuts without
removing examples or explanations merely to reach a number.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
User directive retired the target; no separate implementation commit was
required.
</details>

### Explain every developer test

**High-level summary.** The test guide previously listed filenames with short
labels that did not tell a reader what input was used or why a refusal mattered.
It now explains how to run tests and gives a concrete example, action, accepted
result, refusal, and scientific reason for every test module.

**Current status.** **CLOSED.** The guide and its inventory check are on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Every immediate Python file in `ai/tests/` is described,
and an automated check rejects a missing, duplicate, or stale inventory row.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `a875f3f`, with the later scientific explanation pass in
`9a55c7b`. Evidence includes 333/333 final AI tests, focused inventory checks,
link/fence checks, permanent-note guard, and two final read-only reviews.
</details>

### Make the YAML workflow diagram readable on phones

**High-level summary.** The first `example_yamls` diagram was too wide for a
phone or tablet. It is now a five-step vertical path with short labels and the
same sequence stated in prose.

**Current status.** **CLOSED.** The vertical diagram is published.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Choose, copy, edit, check, and run appear from top
to bottom without overlapping labels.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `ac3b3eb`; 243/243 tests, independent phone-layout review,
diff check, and permanent-note check passed.
</details>

### Move detailed appendices out of the main README

**High-level summary.** The main README required a new user to cross many pages
of specialist material before finding the first run. It now keeps a five-step
startup path and sends YAML, data generation, Cobaya, emulator, Syren, and AI
details to their folder guides.

**Current status.** **CLOSED.** The shorter root guide is on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The root README changed from 3,717 to 1,122 lines,
and `emulator/CODE_REFERENCE.md` owns the dense code maps.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `b0aa890`; 335 links, 67 shell/YAML/Python examples, six
vertical Mermaid sources, 243/243 tests, board checks, and two reviews passed.
</details>

### Make the permanent notes durable and Python style mandatory

**High-level summary.** The permanent notes read like a dated development
diary and Python readability was treated as a preference. The eleven notes now
record neutral current rules, and readable Python is a required GO/NO-GO
condition.

**Current status.** **CLOSED.** The permanent-note and Python contracts apply.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Only the Architect may edit permanent notes, and
`python-changes-go-no-go.md` protects code intended for students and C users.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence included 30/30 focused contract tests, 6/6 backlog-bundle
reproductions, 243/243 full tests, and scientific, workflow, artifact, role,
link, and whole-diff audits.
</details>

### Correct CoCoA setup and project paths in the guides

**High-level summary.** Four guides duplicated CoCoA setup and used an
invented `projects/lsst_y1/cobaya` folder. They now point to the official CoCoA
instructions, place editable YAML directly under `projects/lsst_y1`, and
separate user-copied YAML from generated data exposed by startup links.

**Current status.** **CLOSED.** The verified paths are in all four guides.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The copy and syntax examples use the real project
layout, while nested CMB and grid paths retain their true resolution rules.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `b87b9f7f1de4e756d55d232edaeab6758bb20516`; 179/179 tests,
250 links, 215 fragments, 29 Bash fences, runtime examples, note guard, and
factual/cold-reader reviews passed.
</details>

### Add a beginner guide for the AI tools

**High-level summary.** Users had to read the large AI guide to discover which
tool to run and whether it changed files. `ai/tools/README.md` now explains all
five programs, daily commands, visible results, stopping, recovery, and bundle
transfer beside the tools themselves.

**Current status.** **CLOSED.** The tool guide is the command reference.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** `ai/README.md` keeps the first-ticket path, while
advanced runtime controls and recovery live in the tool guide.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `20862070723aa1b3d2e09d3250cfba717edb7a2d`; 179/179 tests,
181 links, 161 fragments, 45 Bash fences, focused tool reproductions, render,
guard, and two reviews passed.
</details>

### Make all AI README workflows vertical

**High-level summary.** Six AI workflow diagrams were hard to follow on narrow
screens. They now read from top to bottom and use visible actions instead of
unexplained internal labels.

**Current status.** **CLOSED.** All six rendered diagrams are taller than wide.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The prose beside each picture states the same
sequence, so no safety rule depends on interpreting the graph.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
All six final PNGs were inspected; 179/179 tests, link/anchor/fence checks,
hard-zero scans, permanent-note guard, and two reviews passed.
</details>

### Add a beginner guide for Cobaya adapters

**High-level summary.** The `cobaya_theory` folder lacked a direct path from a
saved emulator to one checked Cobaya evaluation. Its guide now covers five
adapters, setup versus evaluation, file matching, device behavior, physical
limits, and the current NumPy 1.x boundary.

**Current status.** **CLOSED.** The six-step guide and appendices are complete.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** A user can choose an adapter, manually copy a real
template to `projects/lsst_y1`, run a setup check, and inspect one result.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence includes README SHA-256 `5df8352818043dcd32c29f73c448f502447871251f0d70c09bf9fd851a305dbf`,
179/179 tests, Bash/YAML/copy/runtime/identity/link/render checks, guard, and
three reviews.
</details>

### Add a beginner guide for example YAML files

**High-level summary.** The ten shipped YAML files had no guide for choosing a
starting point or separating a syntax check from scientific validation. The
new guide shows how to choose, copy, edit, check, and run one template, then
uses appendices for special modes and path rules.

**Current status.** **CLOSED.** Every shipped YAML appears in the chooser.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The guide states that editable YAML is copied by
the user and that a successful parser check does not prove files or scientific
settings are correct.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence: 179/179 tests, ten-file census, 45 links, 42 anchors, Bash/YAML/copy
and help checks, diagram render, permanent-note guard, and three reviews.
</details>

### Simplify the first AI workflow picture

**High-level summary.** The first AI diagram introduced ten boxes before a new
reader understood the basic process. It now shows only the user request,
Architect plan, Implementer work and tests, and Architect review.

**Current status.** **CLOSED.** The four-step introduction is published.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Optional Red Team, repair loops, mailbox files, and
worktrees are explained later instead of crowding the first picture.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
The exact 1173x94 render, 31 links, 18 anchors, six Mermaid fences, 179/179
tests, permanent-note guard, and two reviews passed.
</details>

### Add a beginner guide for generating training data

**High-level summary.** The data-generator programs create the scientific
tables used for training and validation, but the folder had no single guide to
their outputs, failure flags, seeds, memory, MPI, resume, or append behavior.
The new guide gives a six-step first path and moves family detail to plain
question-led appendices.

**Current status.** **CLOSED.** The guide covers all five generator commands.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Training and validation commands are separate, and
executable examples check shapes, failure flags, and row overlap.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused data tests 64/64 and full tests 179/179 passed, along with five help
routes, a temporary chain-only example, every fence/link check, three reviews,
and the permanent-note guard.
</details>

### Explain why the three-role system saves scarce AI tokens

**High-level summary.** The AI guide used roles without first explaining why
a student would accept the extra structure. It now says that unlimited access
may make the system unnecessary and explains why expensive reasoning can be
reserved for the Architect and optional Red Team while a simpler model writes
and tests code.

**Current status.** **CLOSED.** The cost rationale opens the AI guide.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Tokens and selectable model roles are defined with
availability-qualified Opus, Sonnet, and Haiku examples.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Links, fences, prose scans, 179/179 tests, permanent-note guard, factual review,
and cold-reader review passed.
</details>

### Rewrite AI appendices in beginner language

**High-level summary.** The AI appendices introduced words such as lane,
dispatch, worktree, and schema before a reader could connect them to a file or
action. FAQs A-H now define or replace those terms where they first appear and
explain stopping, role folders, Red Team scope, recovery, and archive transfer
with concrete examples.

**Current status.** **CLOSED.** The appendix language pass is complete.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Exact watcher messages, positive and zero cycle
behavior, role selection, and bundle hash checks now have local explanations.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Exact-output 8/8, two-role runtime 18/18 with 21 mutations killed, 179/179 full
tests, Mermaid/anchor checks, and two cold-reader reviews passed.
</details>

### Protect README and Python explanations with a GO/NO-GO contract

**High-level summary.** README prose and explanatory Python text could become
factually polished but unusable for a physics student. The eleventh permanent
note now requires concrete examples, local definitions, exact evidence, and
anti-AI writing checks, while the SHA guard prevents accidental note drift.

**Current status.** **CLOSED.** The contract and guard apply to every role.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Implementer and Red Team cannot edit the eleven
notes; the Architect pins and rechecks their exact bytes.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused guard/role/handoff tests 41/41, bundle regression 6/6, full tests
179/179, whitespace check, and independent guard review passed.
</details>

### Remove the overlapping mailbox-diagram label

**High-level summary.** A self-loop label in the mailbox lifecycle picture
rendered on top of another label. The loop was removed and the unchanged
failure meaning was written beside the work-in-progress box.

**Current status.** **CLOSED.** The diagram is readable.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The layout and prose now communicate the same
archive behavior without overlapping text.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
README contract and whitespace checks passed.
</details>

### Rename `texnotes` to `documentation`

**High-level summary.** Teaching sources and the activation-function notebook
were stored under an unclear folder name. They now live under
`documentation/`, and links, build paths, and custody text use that name.

**Current status.** **CLOSED.** The rename and rebuilt PDF are complete.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** `activation_functions_teaching.nb` moved with the
documentation tree and the generated guide builds from current paths.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Independent path and build audit issued GO.
</details>

### Make the AI guide role-first and visual

**High-level summary.** The AI guide mixed role rules, command details, and
internal mechanics before explaining who decides and who changes code. It now
starts with stable role boundaries, selectable Claude models, Architect-only
GO/NO-GO, bounded Red Team scope, and visual workflows.

**Current status.** **CLOSED.** The role-first guide is on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Exact tool reference moved to the tools guide, and
shorter paragraphs plus diagrams explain runtime behavior.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Output parity 8/8, fix-only 14/14 with 20 mutations, role models 5/5 with five
mutations, safe stop 9/9 with seven mutations, help/anchor/fence/SVG and diff
checks passed.
</details>

### Render README equations correctly

**High-level summary.** Four formulas appeared as raw bracketed text instead
of rendered mathematics. Both delta-chi-square equations and the default
activation equation now use GitHub-compatible display-math fences.

**Current status.** **CLOSED.** All four formulas render.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Only the Markdown delimiters changed; equation
bodies and scientific prose stayed the same.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `66f7046`; 130 links/assets, math and Markdown fence
checks, permanent-note guard, and whitespace check passed.
</details>

### Reorganize the root and AI READMEs around short startup paths

**High-level summary.** Both large guides mixed first-use instructions with
reference material. They now separate a short main path from question-led
appendices grouped by subject.

**Current status.** **CLOSED.** The reader-path rewrite is on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The later `c91791a` hygiene change also repaired
the disposable primary-worktree test fixture that this ticket originally left
behind.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `eb17489`; 110 links, 55 anchors, 42 YAML fences, mailbox
reproductions, 84/84 tests, diff check, and three reviews passed. The old
scratch-fixture debt is superseded by `c91791a`.
</details>

<a id="closed-ollama-documentation-model"></a>
## Use GLM-5.2 Cloud in the Ollama examples

### High-level summary

The AI guides used Qwen as their example Ollama Implementer. The preferred
documented choice is now `glm-5.2:cloud`, so a user copying either the watcher
or connection-check command should see that model consistently.

### Current status

**CLOSED.** This was accepted as a **LOW DOCUMENTATION CHANGE**. It changes
the recommended examples, not the daemon's Claude default or the user's
ability to name another Ollama model.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

The short AI guide and detailed tool guide use `glm-5.2:cloud` for Ollama
watch and `--ping` commands. They also explain that this choice requires an
Ollama account and processes prompts through Ollama's cloud service.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Only documentation examples changed. Runtime model selection remains explicit
through `--implementer-provider ollama --implementer-model MODEL`; no model
name was compiled into the daemon.

</details>

<a id="closed-ollama-ping-visible-thinking"></a>
## Let reasoning-capable Ollama models pass the connection check

### High-level summary

The Ollama connection check required the model's entire visible answer to be
the requested marker. GLM-5.2 Cloud answered correctly, but it printed its
reasoning first, so CocoaFlow reported a healthy signed-in model as
unavailable.

### Current status

**CLOSED.** This was accepted as a **HIGH BUG FIX** because the documented
Ollama model could not pass the check required before an unattended watch.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

The probe now asks Ollama to hide model reasoning. It still requires an exact,
unpredictable reply, so echoed prompts and unrelated output remain failures.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

The repair adds Ollama's `--hidethinking` option only to the no-work
connection probe. The focused provider test requires that option while the
existing nonce and echoed-prompt tests retain the strict response boundary.

</details>

## AI roles, user controls, and handoffs

### Recover a completed Implementer return after validation refuses it

**High-level summary.** The Implementer completed candidate C, but harmless
Markdown list formatting made the evidence validator reject the return. The
watcher then stayed idle instead of preserving the completed work.

**Current status.** **CLOSED. HIGH BUG FIX.** A restart now revalidates the
saved return and sends it to the Architect without rerunning the Implementer.
The evidence parser accepts the same required heading inside a Markdown list.

**Red Team reopen count: 0. Red Team reopening: allowed.**

**What is already fixed.** Candidate identity, planned subagent names, fields,
scope, and clean Git state remain exact. The character guard can also measure
the existing 8,645-character candidate within the selected 15,000 limit.

**What is missing.** Nothing for this ticket.

### Restart cheap role work without rebuilding the Architect plan

**High-level summary.** Ctrl-C can leave an Implementer or Red Team request
outside the waiting queue with unfinished work. Recovery should discard that
role's partial attempt and reuse the Architect's saved plan.

**Current status.** **CLOSED.** This was accepted as a **HIGH BUG FIX** because
the old manual recovery could halt normal ticket operation and invited unsafe
file moves.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** `--restart-implementer` and `--restart-redteam`
requeue one exact handoff. They refuse ambiguity or a completed result.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Tests cover partial edits, exact requeueing, completed-result refusal, and the
Architect's sealed backlog.
</details>

### Explain every Architect candidate review in the terminal

**High-level summary.** A formal `GO` or `NO-GO` says whether work advances,
but it does not tell a human how close the Implementer came. The Architect now
ends each candidate audit with a short assessment of the exact result. The
assessment names strengths, remaining work, file scope, and the next action.

**Current status.** **CLOSED.** This was accepted as **MEDIUM NEW
FUNCTIONALITY** requested for immediate use.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Five plain result categories avoid false numerical
precision. The seven-line assessment fits inside the terminal's existing
eight-line relay tail and leaves the secure decision-only GO message intact.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
The role-contract test pins the ordered terminal rows and the rule that the
assessment judges candidate C rather than the selected model.
</details>

<a id="closed-subagent-discretion"></a>
### Let the Architect decide when helpers add real value

**High-level summary.** Requiring a helper for every ticket could spend more
model credits coordinating a five-line correction than performing it. The
Architect now decides whether another AI session can produce an independent
result, while the Implementer remains unable to skip required work.

**Current status.** **CLOSED.** Every directive contains exactly one visible
choice: `Subagents required` with bounded named jobs, or `Subagents not
required` with a concrete Architect-authored reason.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The handoff validator refuses empty, vague,
contradictory, or Implementer-authored waivers. A no-helper handoff must repeat
the saved reason exactly. Required helpers still launch before the
Implementer's own edit, and the existing SHA-bound capability exception remains
available only after a real required launch fails. The router and daemon
witnesses cover both exact acceptance and changed-reason refusal.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Landed and pushed as `0ff77fa`. The one required adversarial review found one
extra closing bracket in the Implementer template. The Architect corrected it,
added an exact regression assertion, reran 94 focused tests, and made the final
decision without a second review round. The complete configured CoCoA suite
passed 790 tests; the router and daemon scratch reproductions also passed.

</details>

<a id="closed-structured-role-contract"></a>
### Protect one machine-readable role contract

**High-level summary.** Stable role permissions, timing limits, candidate
identity, landing authority, backlog ownership, and the single-review rule had
been repeated across Python and long prose. A later edit could change one copy
without making the contradiction obvious to the Architect.

**Current status.** **CLOSED.** The protected JSON-compatible YAML file is now
the small machine source of truth. Live watcher controls must agree with it,
and an Implementer candidate cannot change the contract, its reader, the
eleven permanent notes, their guard, or protected role files.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The strict reader refuses duplicate, missing,
unknown, wrongly typed, noncanonical, oversized, linked, or non-regular input.
The permanent-note guard protects the YAML without changing the eleven-note
Markdown census. A protected-policy draft receives one adversarial GO/NO-GO
recommendation; the Architect decides, and a correction gets no second round.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Landed and pushed as `8611e1e`. The sole adversarial review verified staged
diff `df8be80b1ec26a9f146c1b835391ea33f218faac6ff4e529570de56d879231e7`
and returned GO. All 795 configured CoCoA tests, the full disposable-worktree
mutation reproduction, and all six backlog-bundle scenarios passed.

</details>

### Prohibit new monkey patches without forcing a wholesale rewrite

**High-level summary.** A monkey patch changes existing executable behavior
while Python is running, so an apparently local test can change a later test.
The permanent Python contract and all three role contracts now reject a monkey
patch that is added, copied, retargeted, or broadened.

**Current status.** **CLOSED.** Existing sites are not one repository-wide
Critical rewrite. When bounded work encounters one, the Architect records one
separate High bug ticket and keeps the current ticket narrow.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The contract defines prohibited runtime
replacement, names ordinary local fakes that remain allowed, and prevents the
Implementer or Red Team from proposing patching as a shortcut.

**What is missing.** Nothing for the future-rule ticket. Two existing sites
encountered while writing the rule are recorded in the High queue.

<details><summary>Technical record for development tools</summary>
The role-contract test requires the rule in the Architect, Implementer, Red
Team, and permanent Python surfaces. The scoped review rejected a baseline
census and wholesale migration as disproportionate.
</details>

### Use a 4,000-character warning for one bug repair

**High-level summary.** The earlier production-size warning was close to the
size of only a few clear Python lines and made ordinary bounded repairs look
disproportionate too early. The Architect now becomes strongly suspicious
only above 4,000 added-plus-deleted characters outside tests and gates.

**Current status.** **CLOSED.** The number remains a warning, not a hard size
limit. Readability, directness, and the separate `--max` value still govern
every candidate.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The permanent Python contract, Architect role,
entry command, and regression test use the same 4,000-character value.

**What is missing.** Nothing for this policy adjustment.

<details><summary>Technical record for development tools</summary>
The calculation excludes all of `ai/tests/` and `ai/gates/`, which are
evidence surfaces rather than mature scientific production code.
</details>

### Make the Architect the only user-facing role

**High-level summary.** Direct messages to Implementer or Red Team bypassed
the role that owns scope and final decisions. Public send and ping commands now
accept only Architect requests, and the other roles return their work through
the Architect.

**Current status.** **CLOSED.** The user-to-Architect boundary is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Widespread Red Team work begins only when the user
asks the Architect to request it, and a carried handoff must remain unchanged.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Six mismatch cases refuse with zero writes; 234/234 tests and the role, router,
permanent-note, compilation, diff, and README checks passed.
</details>

### Filter Red Team discoveries by severity

**High-level summary.** Discovery runs could create tickets for edge cases the
user did not want to pursue. The `--severity` setting now lets the user request
only severe failures, probable normal-use bugs, or every concrete finding.

**Current status.** **CLOSED.** The default is `medium`, and the Architect
makes the final severity decision.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Red Team records severity, likelihood, evidence,
and whether the finding meets the user's threshold; Architect may accept,
upgrade, or downgrade it with a reason.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence: 223 tests plus severity, manual-router, worktree, fix-only, two-role,
and character-limit reproductions; permanent-note and README contracts passed.
</details>

### Limit changed characters without allowing unreadable code

**High-level summary.** Maintenance tickets needed a way to reject changes
that touch too much code. `--max` now limits added plus deleted characters, but
the Architect must still reject shortened names, collapsed logic, or removed
explanations that make Python difficult to read.

**Current status.** **CLOSED.** The limit reaches every role and final review.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Zero means unlimited, positive values use the exact
ticket base, and an unmeasurable or conflicting candidate refuses.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `859dab2`; focused contract/guard/role tests 67/67, full tests
213/213, max propagation 9/9 with four mutations, topology/no-red-team/cycle
and all preserved router suites passed.
</details>

### Give Sol its own saved worktree

**High-level summary.** Sol formerly started in the repository folder reserved
for the user. It now creates and reuses an independent `mailbox-sol` worktree,
so ordinary agent work cannot change the user's main folder.

**Current status.** **CLOSED.** Sol and Claude have separate saved work folders.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Repository, path, branch, role, tool, and notes
identity are checked before and after each Sol launch.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `1e17fe2`; topology/race runtime 19/19 with mutations killed, all
related daemon suites, security review, and documentation review passed.
</details>

### Create and reuse Claude's primary coordination worktree

**High-level summary.** Claude sessions needed a persistent work folder instead
of guessing or using the user's main checkout. The first live use now creates
or deliberately adopts one worktree and later runs validate and reuse its exact
repository, path, and branch.

**Current status.** **CLOSED.** Creation, reuse, refusal, and migration rules are
implemented.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Concurrent launchers converge, invalid folders fail
without fallback, and archived transport can be copied only under bounded
byte checks. The old disposable test-fixture drift was repaired by `c91791a`.

**What is missing.** No code is missing. A user who owns the preserved
`amazing-keller-e798b6` transport must deliberately migrate or adopt it; the
tool will not mutate that user-owned state automatically.

<details><summary>Technical record for development tools</summary>
Initial evidence: 15/15 focused runtime arms with source mutations, preserved
router suites, 44/44 tests, board self-test, compilation, and diff check. The
real pre-migration `amazing-keller-e798b6` transport remains preserved by
design and requires deliberate user migration rather than automatic mutation.
</details>

### Package unfinished backlog work for another developer

**High-level summary.** A user who runs out of credits may need to send one
snapshot of unfinished work to another developer. `backlog_bundle.py` now
creates a `.tar.xz` package, checks it without writing, and imports it only
into a new ignored review folder.

**Current status.** **CLOSED.** Deterministic package, read, and import modes are
available.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The archive binds repository and base identity,
sizes, SHA-256 values, paths, and exact bytes, and refuses overwrite, links,
special files, traversal, races, malformed manifests, and extra XZ streams.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused regression 6/6 on Python 3.14 and 3.9, 44/44 tests, mailbox checks,
compilation, diff, and independent security/policy reviews passed.
</details>

### Allow a two-role run without Red Team

**High-level summary.** Some work needs only an Architect and Implementer, but
the watcher always assumed a Red Team route. `--skip-redteam` and
`--no-red-team` now disable that route while preserving its waiting messages
for a later three-role run.

**Current status.** **CLOSED.** The two-role topology is supported.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Evidence returns directly to the Architect, Sol
sends refuse during the two-role watch, and demand plus cycle-zero count only
enabled routes.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 18/18 with 21 mutations on system, CoCoA, and macOS Python;
manual-router parity, preserved daemon suites, 44/44 tests, board, compilation,
render, and diff checks passed.
</details>

### Let the user choose Claude models by role

**High-level summary.** Architect and Implementer were tied to expensive
default model names. `--architect-model` and `--implementer-model` now accept
aliases or full Claude model IDs while the mailbox routes keep their stable
role meaning.

**Current status.** **CLOSED.** Architect and Implementer models are selected
independently.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The expected Opus-Architect and Sonnet-Implementer
pair is supported, invalid names refuse before launch, and Sol remains
unchanged.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 5/5, five mutations killed, and every preserved daemon suite
passed.
</details>

### Require detailed plans that simpler Implementers can follow

**High-level summary.** Earlier handoffs assumed that an expensive Implementer
would fill in missing design choices. Architect and Red Team directives must
now name exact files, algorithms, invariants, failures, tests, expected
results, exclusions, and stop conditions so a simpler model can execute them.

**Current status.** **CLOSED.** Incomplete or choice-leaving packets refuse.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Detailed deterministic alternatives remain allowed,
but Red Team repair proposals return to the Architect and never execute by
themselves.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `866b30b`; focused role/contract/preamble tests 31/31,
full tests 169/169, eleven router reproductions, 59-case security matrix,
compilation, diff, and three reviews passed.
</details>

<a id="closed-failure-catalog-consistency"></a>
## Keep the Implementer failure catalog synchronized with its controls

### High-level summary

The short Implementer failure catalog correctly points to existing recovery
behavior, but one explanation repeated the current 90-minute setting and its
code references could silently become stale after a function rename.

The catalog now names the configurable role-contract setting instead of its
present value. A documentation check verifies that every catalog identifier is
unique and every named Python file and symbol still exists.

### Current status

**CLOSED.** This was accepted as a **LOW BUG FIX**. Stale catalog text could
mislead a future maintainer, but it never changed runtime behavior or granted
the Implementer additional authority.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

The `timed_complexity` entry points to
`role-contract.yaml::runtime.implementer_review_minutes`. The existing
role-contract tests parse the reference catalog as documentation, confirm
unique IDs, and use Python's syntax tree to find each referenced function or
class in its named source file.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

The check deliberately does not execute the catalog or derive workflow policy
from it. Code, the validated handoff contract, and `role-contract.yaml` remain
authoritative. The test only prevents broken documentation links and a copied
configuration value from misleading a later reader.

</details>

## Scientific code, data handling, and gates

<a id="open-mps-serving-domain"></a>
<a id="closed-mps-serving-domain"></a>
### Reject matter-power requests outside calibrated ranges

**High-level summary.** Matter-power serving now accepts only finite, ordered
saved axes and finite matching surfaces. A redshift outside the saved interval
always stops. A requested wavenumber tail is allowed only through the existing
logarithmic boundary continuation, controlled by a visible option that is on
by default.

The option can be turned off when the user wants every wavenumber confined to
the saved grid. Values inside the grid use the same interpolation path in
either mode.

**Current status.** **CLOSED.** Commit `a0633ad` contains the bounded repair for
the demonstrated serving-domain failures without adding a calibration
registry or a new scientific framework.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Saved and direct interpolation axes must be
one-dimensional, finite, strictly increasing, and long enough for the spline;
wavenumbers must also be positive. Surfaces must have the exact matching shape
and finite values. Empty, nonfinite, nonpositive, or even slightly
out-of-range queries stop. Explicit log-log wavenumber tails reproduce a
known power law, while disabling the option refuses those tails. The adapter
already validated the assembled linear and nonlinear spectra before
publishing them, so that responsibility did not need another implementation.

**What is missing.** Earlier configuration errors are tracked separately in
[Validate matter-power requests before a run starts](#open-mps-request-contract).
The hypothetical case of a future Syren source edit that keeps the old law
name is parked under [Certify the vendored Syren formulas independently](#parked-syren-formula-certificate).

<details><summary>Technical record for development tools</summary>

Evidence: 811/811 developer tests; 12 focused boundary tests; both adapter
gate groups, with 11/11 strict checks and 23/23 publication checks; the
18-check matter-power adapter assembly leg; Python compilation; whitespace
check; and one required adversarial review. That review found tolerant range
comparisons and inward extrapolation bounds. The final candidate removes both
and adds direct regression witnesses. Production and user-guide changes total
3,949 added-plus-deleted characters; tests and gates are excluded from the
4,000-character warning.

</details>

<a id="closed-background-protocol"></a>
### Reject invalid redshift grids, coordinate pairs, and nonflat cosmologies

**High-level summary.** Background distances now start from an ordered Hubble
grid whose first redshift is zero. The Cobaya bridge keeps each two-redshift
request as a visible pair, and the current flat-only calculation refuses a
background dataset, saved emulator, or directly named Cobaya curvature that
is nonflat. These checks prevent plausible-looking distances from being
served for a calculation this implementation does not support.

**Current status.** **CLOSED.** Commit `a3b345e` contains the bounded repair.
Curved-distance formulas and renamed or transformed curvature parameters
remain user responsibility unless separately requested.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Training and rebuilding reject nonfinite,
negative, duplicate, reversed, or unanchored Hubble grids. The distance
integrator independently checks its grid and Hubble values. Background
generation and serving reject a directly sampled or fixed nonzero `omk`, and
two-redshift requirements and getters require exact ordered `(N, 2)` rows.
The user guide now shows a zero-anchored low-redshift grid.

**What is missing.** Nothing for this ticket. Supporting arbitrary curved
cosmologies or discovering curvature hidden behind a renamed Cobaya
parameter would require a separate feature rather than more defensive
framework code here.

<details><summary>Technical record for development tools</summary>

The exact staged candidate has SHA-256
`5979f4099f1940a824f14c96a28bcd85dd96b568a9d7b7bc1a1db6a2088dc930`.
The one required adversarial review returned NO-GO because two gate fixtures
still began above zero. Those fixtures were corrected, the authenticated
dataset smoke gate was updated to its current published paths, and the
Architect self-audited the resulting candidate without requesting another
review round. The 5,045 changed characters outside tests and gates exceed the
4,000-character warning, but they are small direct checks at the generator,
saved-grid, integrator, artifact-load, and Cobaya-request boundaries rather
than a new framework. All 802 tests and the `bsn-identity`, `bsn-smoke`, and
`transfer-identity` gates pass.

</details>

<a id="closed-cmb-serving-contract"></a>
### Reject physically impossible CMB spectra before serving them

**High-level summary.** The CMB bridge now checks the complete local result
before it gives that result to Cobaya. TT, EE, and PP must be finite and
nonnegative; TE remains signed but cannot exceed the covariance bound where
TT, TE, and EE share a stored multipole. An invalid prediction leaves the
caller's state unchanged instead of publishing a partial result.

**Current status.** **CLOSED.** Commit `2016c40` contains the corrected repair.
Unit conversion and multipole-factor support remain a separate Medium feature.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Every loaded spectrum is checked for NaN and
infinity. Negative TT, EE, or PP values refuse. The covariance comparison
uses only multipoles stored by all three relevant artifacts and remains stable
for very small and very large finite values. One rounding step in the source
prediction's floating representation admits an honest positive-semidefinite
boundary without allowing an arbitrary tolerance.

**What is missing.** Nothing for this ticket. The adapter still deliberately
refuses unit and multipole-factor conversions until their separate ticket is
implemented.

<details><summary>Technical record for development tools</summary>

The corrected staged candidate has SHA-256
`c6e70e64a6cd130d678e27ef3aa65446f1ea21bfc892ffdc74d3d6d732584b42`
and 2,447 changed characters outside tests and gates. The one required
adversarial review returned NO-GO because squaring could underflow or overflow
and because the initial witnesses did not prove the rounding step or common-
multipole rule. The bounded correction compares `abs(TE)` with
`sqrt(TT) * sqrt(EE)` and adds all three witnesses. The Architect self-audited
the corrected candidate without a second review. All 807 tests, 18 focused
tests, and both `adapter-contracts` gate groups pass; changed Python compiles
and `git diff --check` passes.

</details>

<a id="closed-compatibility-manifest-removal"></a>
### Remove the duplicate compatibility manifest from saved emulators

**High-level summary.** Each saved emulator already records the complete model
recipe, geometry state, analytic law, and composition mode used to rebuild it.
A second compatibility manifest copied those facts and added labels such as
`model:...:v1`. Those fixed labels did not inspect or hash the Python
implementation, so they made saving and loading harder to follow without
providing independent protection.

**Current status.** **CLOSED.** New plain and transfer artifacts no longer
write or require the duplicate manifest. The direct rebuilding records and
strict weight loading remain.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The semantic-label registries, manifest builders,
root and embedded-transfer datasets, rebuild checks, and output-identity fields
were removed as one coherent change. The production change deletes 431 lines
and adds 10; splitting the writer, reader, and filename identity would have
left a partly removed file format. This deliberately breaks the brief alpha
format: an artifact written while the manifest existed may need regeneration.
No dual reader or replacement registry was added.

**What is missing.** Nothing for this ticket. The smaller model-recipe reader,
transfer-state digest cleanup, and training-history load cleanup remain
separate Critical tickets above.

<details><summary>Technical record for development tools</summary>

Landed as `8030857`. The one required adversarial review returned GO and
confirmed that model
recipes, geometry and composition facts, analytic laws, transfer checks,
training histories, pair binding, and strict checkpoint loading remain. The
CPU evidence passed 216 distinct unit tests; the transfer, output-identity,
artifact-composition, and fine-tune gates also passed. The plain-and-transfer
witness saves both artifact forms, confirms neither contains the removed
dataset, and rebuilds both. Changed Python compiled and `git diff --check`
passed.

</details>

<a id="closed-model-recipe-simplification"></a>
### Keep only the model recipe checks needed to rebuild an emulator

**High-level summary.** A saved model recipe prevents a later software default
from silently rebuilding a different neural network. The former implementation
also repeated numerical rules already checked by the real constructors and
factories, which made every saved-emulator load harder to understand.

**Current status.** **CLOSED.** The recipe reader now checks the complete,
closed rebuilding description before importing model code. Numerical rules
remain with the constructor or factory that actually uses each value.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Commit `346e65b` renames the module to the clearer
`model_recipe.py`, removes the duplicate constructor-signature registry and
constructor-owned numerical checks, and keeps the six supported classes,
exact saved fields, known factory names, compile choices, positive network
dimensions, class identity, and geometry choice. Saving still requires the
live model's own recipe to equal the claimed recipe. Reopening still checks
geometry widths and loads every learned tensor strictly.

**What is missing.** Nothing for this ticket. Embedded transfer-state hashes
and training-history checks were handled by the following separate Critical
simplifications.

<details><summary>Technical record for development tools</summary>

The exact staged diff received the one required adversarial review and GO.
The focused recipe and preflight suite passed 41 tests. The broader CPU suite
passed 174 tests; compile-recipe CPU controls passed 12/12; transfer identity,
output identity, composition, and fine-tune gates passed 57, 50, 45, and 19
checks respectively. CUDA compilation was unavailable on this workstation.
Changed Python compiled, `git diff --check` passed, the permanent-note guard
passed at the exact commit, and `main` matched `origin/main` after the push.

</details>

<a id="closed-transfer-state-digest-simplification"></a>
### Remove duplicate hashes for embedded transfer-model weights

**High-level summary.** A transfer emulator already stores its base-model
tensors inside the HDF5 file and loads them strictly into the registered model.
The former path also hashed those same tensors several times and copied the
hashes into HDF5 attributes and nested configuration records.

**Current status.** **CLOSED.** Embedded transfer tensors now have one direct
rebuild path. Missing, extra, or wrong-shaped tensors fail strict model loading.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Commit `64fa00a` removes the digest declarations,
structure walks, repeated live/HDF5 comparisons, and tensor-digest helper. It
keeps `state`, refined-only `drifted_state`, the explicit refinement choice,
model recipes, geometry reconstruction, strict tensor loading, direct
`.emul`/`.h5` pair authentication, source artifact and checkpoint identity,
and output identity. A same-shaped value edit inside the HDF5 state is now
explicitly user responsibility.

**What is missing.** Nothing for this ticket. Training-history checks on the
serving path were removed by the following separate Critical simplification.

<details><summary>Technical record for development tools</summary>

The 14,659-character warning override was accepted because the production
change adds 50 characters and deletes 13,072 as one writer/reader/metadata
format cleanup; splitting it would leave an inconsistent saved format. The
single adversarial review returned GO. The full AI test suite passed 795 tests,
the focused artifact set passed 94, and the transfer, composition, and output-
identity gates passed 57, 45, and 50 checks. Documentation style passed 16
tests, changed Python compiled, the permanent-note guard passed at the exact
commit, and `main` matched `origin/main` after the push.

</details>

<a id="closed-training-history-load-simplification"></a>
### Stop revalidating training history while loading an emulator

**High-level summary.** Training histories explain how a completed run
progressed, but they do not define the neural network used for prediction. The
former rebuild path nevertheless interpreted a large optimizer, schedule,
pass-order, and history grammar before it could load a saved model.

**Current status.** **CLOSED.** Reopening now uses only the records needed for
prediction. The training description and curves remain saved provenance.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Commit `a4f8fa8` removes the duplicate training-
policy parser from `results.py`. Before publication, the writer still requires
five finite history arrays with compatible shapes. Reopening does not read the
history group or interpret pass grammar. It keeps the resolved training mapping
as opaque provenance bound into output identity, while preserving model-recipe,
geometry, composition, scientific-record, artifact-pair, and strict-weight
checks.

**What is missing.** Nothing for this ticket. Training-pass construction and
its focused tests remain because they document what work ran; they do not
control reconstruction.

<details><summary>Technical record for development tools</summary>

The one required adversarial review found the code and protected-note change
sound but returned NO-GO for one stale sentence in the public test inventory.
The Architect corrected that sentence and made the final decision without a
second review round. The final 789-test AI suite passed. Seventy-eight focused
artifact tests, the output-identity, composition, transfer, and fine-tune gates,
documentation checks, Python compilation, and whitespace checks passed. CUDA
compilation was unavailable; its 12 CPU controls passed. The size override was
accepted because the change deletes one coupled framework: 126 lines were added
and 808 were removed across code, tests, and documentation.

</details>

<a id="closed-dark-energy-coordinates"></a>
### Preserve time-varying dark energy from data generation through serving

**High-level summary.** Matter-power calculations can describe dark energy
with a present-day value and a second value that says how it changes with
cosmic time. Cobaya may sample their sum, `w0pwa = w + wa`, and calculate
`wa` before a theory component runs.

The old path could overlook that calculated value and silently replace it
with `wa = 0`. This produced a smooth finite spectrum for constant dark
energy even when the user requested a time-varying cosmology.

Generation and serving now share one checked conversion. The saved dataset
states the physical law, every generated Syren starting surface reuses it,
and the Cobaya adapter reconstructs the saved coordinate names from the
physical values before prediction.

**Current status.** **CLOSED.** Commit `32328be` implements and tests the
repair. Architect-only commit `8b7f991` records the exact rule in the
permanent scientific notes.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Direct `w`/`wa` and transformed `w`/`w0pwa`
coordinates resolve to the same pair. Repeated forms must agree under one
documented tolerance, and incomplete inputs receive zero evolution only from
an explicit constant-`w` or cosmological-constant law. Generator facts use
Cobaya's sampled and calculated parameter information; sampled values cannot
borrow theory defaults and appear fixed. The matter-power adapter requests
physical coordinates rather than dropped `w0pwa`, then rebuilds all names
before either learned or analytic prediction. The shipped EMUL2 point now
uses `w = -0.9`, `w0pwa = -0.7`, and therefore `wa = 0.2`.

**What is missing.** Nothing for this ticket. Conflicting amplitude spellings
are a separate High ticket because they involve `As` and `As_1e9`, not the
dark-energy coordinates repaired here.

<details><summary>Technical record for development tools</summary>
The full `ai/tests` discovery ran 796 tests and ended in `OK` with three
skips. Forty-two focused dark-energy tests passed. A real Cobaya 3.5.7 test
proved the nonzero calculated-`wa` route, and the NumPy-1 fixed-facts schema
gate printed `PASS` for every check. Changed Python compiled, the test README
inventory passed, `git diff --check` passed, the implementation audit returned
GO, and the eleven-note guard passed before and after the separate Architect
note update.
</details>

<a id="closed-artifact-recipe-totality"></a>
### Save every model-building setting needed to rebuild a trained model

**High-level summary.** A saved emulator must rebuild the exact model that was
trained. It may not guess from current Python defaults or accept a plausible
recipe that describes different activation curves, layer counts, geometry,
training phases, analytic formulas, or transfer-base weights.

The writer now compares the saved description with the live model and records
the complete executed training plan and transfer state. The reader validates
the model recipe and direct scientific records before it imports or constructs
model components. The training plan remains provenance and is not interpreted
while rebuilding the prediction model.

**Current status.** **CLOSED.** Commit `dd44234` implements the complete
write-and-read contract.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The closed recipe covers all six supported model
classes, their activations, normalization, and constructor values. Parameter
and output geometries, composition, intrinsic-alignment coefficients, and the
analytic target law are saved directly. The writer checks that saved history
arrays are finite and have compatible shapes. Frozen and refined transfer
states use strict model loading, and schema 3 publication refuses any rescaling
mode that public inference cannot reverse.

**What is missing.** Nothing for this ticket. CUDA compilation of saved modes
still belongs to the separate workstation ticket because this host has no
CUDA device.

<details><summary>Technical record for development tools</summary>
Evidence: 753 developer tests passed with 2 skips; 169 focused artifact tests;
all transfer-identity legs; validation-board self-test; CPU compile-recipe
controls; Python compilation; whitespace check; exact-commit permanent-note
guard; and two independent GO reviews. The CUDA compile lane returned its
explicit unavailable status instead of a false pass.
</details>

<a id="closed-generator-ingress"></a>
### Validate generator inputs before creating output files

**High-level summary.** A malformed parameter order, covariance, fiducial
value, grid, prior bound, or command setting could previously be discovered
after output work had begun. MCMC rows could also appear distinct in memory
but collapse to duplicates when saved as `float32`, leaving a smaller usable
dataset than the user requested.

The generator now validates the complete request before it creates output.
After sampling, it counts distinct rows at the precision readers actually
receive and refuses before creating a draft when that count is too small.

**Current status.** **CLOSED.** Commit `9d53a51` validates fresh input before
publication and validates resume or append state read-only before any new
locator or draft is created.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Parameter names, covariance paths and matrices,
fiducials, prior-bound conversion, shared command controls, and family grids
now have strict finite and native-type checks. Covariance normalization is
overflow-safe and exactly symmetric. Missing optional LaTeX text uses the
parameter name. Resume and append authenticate and semantically read the
active checkpoint without writable mappings; append then refuses because
exact continuation state is not yet saved.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Severity was HIGH because malformed normal inputs could create an undersized
or differently defined dataset that later training treated as valid science.
Evidence: generator-ingress 33/33; publication continuation 18/18; checkpoint,
run-control, seed, and request-contract gates; board self-test; full CPU suite
683 passed with 2 skipped; and final independent audit GO after three numeric
edge-case reproductions were added and fixed.
</details>

<a id="open-artifact-output-identity"></a>
<a id="closed-artifact-output-identity"></a>
### Give scientifically different emulator files different names

**High-level summary.** Two completed emulator runs could use the same output
name even when they represented different spectra, physical quantities,
selected data rows, loss modes, or source models. Saving the later run could
therefore replace a valid earlier result, and a recorded source path could
later point to different model bytes.

The saved name now starts with the output family and product and ends with a
32-character digest of the completed run. A CMB `TT` run and an `EE` run, for
example, receive different names. The same is true for the three analytic
rescaling loss modes and for different authenticated fine-tune or transfer
sources.

**Current status.** **CLOSED.** Commit `fa1ec12` records and checks the complete
output identity and refuses every occupied result name.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The digest binds the resolved model and training
settings, executed rescaling mode, exact published training and validation
generations and row order, composition rule, and reused source pair. CMB names
also bind the exact multipole, whitening-scale, and fiducial-spectrum arrays.

The `.h5` record saves the canonical identity and digest. Rebuild checks them
and the exact `.emul` weight digest before PyTorch loads a model. A complete
pair, either lone member, a symbolic link, or an interrupted-save marker
reserves the name. During a race, the first completed writer wins.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed on `main` as `fa1ec12`. Evidence: 610/610 CPU tests passed with one
optional skip; the registered gate passed 31 scientific-name witnesses and 19
save/publication witnesses; the CMB identity gate passed; board list and dry
run accepted the new gate; the permanent-note guard passed against the exact
commit; and two independent exact-diff reviews returned GO.
</details>

<a id="open-padded-head-identity"></a>
<a id="closed-padded-head-identity"></a>
### Stop artificial padded values from mixing with physical bins

**High-level summary.** CNN and Transformer heads use artificial positions so
physical groups with different lengths can share one rectangular tensor. The
old saved representation could not prove which positions were physical, and
some model operations could turn an artificial zero into a nonzero value that
later influenced a real prediction.

The repaired models save the complete physical-position map and validity mask.
They reapply the mask after every operation that could revive an artificial
position, and they refuse saved models whose recorded layout does not match
the model being published or rebuilt.

**Current status.** **CLOSED.** Implementation commit `32f5b48` is on local
`main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Plain and template CNN and Transformer heads retain
fully masked rows, keep padding inert through convolution, FiLM, activation,
normalization, attention, projection, residual, and MLP operations, and gather
only the recorded physical positions. Save and rebuild checks bind the fixed
model buffers to the geometry record and reject older count-only structured
artifacts that cannot prove coordinate identity.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Evidence: 60 focused model and layout tests; 36 artifact and rebuild tests; 24
permanent-note and README contract tests; the registered 24-witness
`padded-head-identity` gate; the full CMB identity gate; an isolated
matter-power structured-head check; gate-board reconciliation; Python
compilation; two clean LaTeX builds with visual page inspection; and
independent production-path, artifact, gate, and evidence reviews with GO
decisions.

</details>

<a id="open-active-model-validation"></a>
<a id="closed-active-model-validation"></a>
### Reject invalid model settings before building the model

**High-level summary.** Model settings are now checked before the program
opens training files, selects an accelerator, or creates neural-network
layers. Values with the wrong type, impossible sizes, nonfinite numbers, and
unsafe output activations therefore stop with a message that names the exact
setting the user must correct.

The same rules are repeated inside the public model constructors. A caller
that builds a model directly cannot bypass the configuration checks by
skipping the normal experiment setup.

**Current status.** **CLOSED.** Implementation commit `08172db` is on local
`main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The active MLP, CNN, or Transformer configuration
must use exact Boolean and integer values, finite positive scales, compatible
widths, and supported normalization and compilation modes. The correction
head uses a safe activation after its zero-initialized layer, while ReLU in
the earlier trunk remains supported. Settings for inactive model families do
not block a valid run, and an MLP with no residual blocks remains a valid
linear trunk.

**What is missing.** Nothing for this ticket. The separate mathematical
gradient problem at the exact origin remains recorded in
[Preserve the power activation gradient at zero](#open-power-zero-gradient).

<details><summary>Technical record for development tools</summary>

Evidence: 19/19 focused active-model tests; 546/546 developer tests in fresh
model/science and subprocess/policy interpreters; full gate-board self-test;
optimized-Python constructor witnesses; permanent-note guard and style tests;
two successful LaTeX builds and visual inspection of the changed guide pages;
and independent source and evidence reviews with GO decisions.

</details>

### Calculate sigma-eight at the conventional physical radius

**High-level summary.** The matter-power bridge previously used a literal
8-Mpc radius even though its saved wavenumbers use inverse megaparsecs. For a
cosmology with `h = 0.64`, it could therefore label a result near one as
sigma-eight when the conventional calculation is near 0.64.

The bridge now uses `R = 8/h` Mpc, requires the exact stored redshift zero,
and refuses a wavenumber grid whose measured tails or numerical resolution do
not support the integral. Cobaya supplies `H0` for this derived result without
adding that dependency to unrelated matter-power requests.

**Current status.** **CLOSED.** Implementation commit `3134cd5` and the
separate permanent-note administration commit `ee43ec0` are on local `main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The helper integrates the linear spectrum in
float64, checks an analytic known answer, and independently protects the
missing-tail, interlaced-recalculation, and largest-panel limits. The public
calculation publishes nothing until the complete result passes. The CAMB
reference stays inside the requested wavenumber range and agrees within the
declared 0.2 percent limit.

**What is missing.** Nothing for sigma-eight. General matter-power request
domains, interpolation, extrapolation, and saved calibration ranges remain in
[Reject matter-power requests outside calibrated ranges](#open-mps-serving-domain).

<details><summary>Technical record for development tools</summary>

Evidence: 515/515 developer tests; 28/28 focused sigma-eight, publication, and
README-inventory tests; every focused matter-power adapter check; direct CAMB
values `0.828513779` and `0.827662255` with relative difference `0.00102883`;
61/61 permanent-note and role tests; exact eleven-note guard; and independent
scientific, Cobaya-routing, and test/documentation reviews with final GO.

The full matter-power identity gate still stops before its adapter leg because
its older authenticated generator failure-mask fixture is absent. The full
smoke gate was not run because it generates data and trains two models; its
CAMB reference calculation was run directly. Those existing workstation
obligations remain in their own open validation tickets.

</details>

<a id="open-pce-strictness"></a>
<a id="closed-pce-strictness"></a>
### Stop the polynomial emulator from saving a fit that failed its accuracy limit

**High-level summary.** The polynomial emulator could previously keep its
first output pattern after every attempted pattern missed the accuracy limit.
It could then save a finite-looking base even though its own leave-one-out
check had rejected that base.

The fit now judges the input bounds, coefficients, and complete matrix in the
same 32-bit number format used after saving. A failing output pattern is
removed and the smaller matrix is checked again; no emulator is created when
no pattern remains.

**Current status.** **CLOSED.** Implementation commit `dd07caa` and the
separate permanent-note commit `aaac2d7` are on local `main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Inputs, fit limits, target variance, support
indices, leverage, coefficients, and every retained accuracy score must be
valid and finite. The selector cannot reuse a polynomial term or fit as many
coefficients as training rows. Equality with `loo_max` fails. The final
multi-output matrix is checked in its saved form, and a rejected output pattern
is left for the neural refiner instead of being forced into the polynomial
base.

The artifact still saves the six polynomial arrays `lo`, `hi`, `multi_index`,
`C`, `Vk`, and `Ybar`. Accuracy scores are checked and reported before saving;
they are not stored in those six arrays. An older artifact therefore cannot be
certified retroactively from its polynomial arrays alone.

**What is missing.** Nothing for strict fit acceptance. The separate Medium
ticket [Refuse polynomial-emulator requests outside the fitted parameter
range](#open-pce-domain-enforcement) owns the future serving-domain option.
Configured GPU training remains in the existing workstation validation
tickets.

<details><summary>Technical record for development tools</summary>

Evidence: 12/12 strict-PCE tests; 28/28 training-behavior tests; 527/527 full
CPU tests; gate-board self-test `ALL PASS`; scalar, CMB, background, and
matter-power NPCE checks with save/rebuild identity; Python compilation and
diff checks; 61/61 permanent-note and role-contract tests; both note guards;
and independent code, test, and documentation reviews with final GO.

The strict witnesses cover no-mode refusal without an artifact, unique and
non-saturated support, nonfinite values, equality at the limit, input-bound
rounding, coefficient cancellation, multi-output matrix rounding, removal of
only the failing joint mode, the unchanged six-array state, and exact finite
training-size sweep results.

</details>

### Refuse invalid values at every public prediction boundary

**High-level summary.** A saved emulator could previously ignore the output
transformation recorded during training. Public prediction also trusted
several intermediate arrays, so a Boolean, nonfinite number, wrong width, or
broadcastable matter-power row could reach a likelihood as an apparently
valid scientific result.

The public reader now serves only artifacts that explicitly record the
supported untransformed target. Inputs, encoded parameters, model outputs,
decoded values, CMB amplitude calculations, and adapter arithmetic are checked
before any result is published.

**Current status.** **CLOSED.** Commit `6c21155` is on `main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Missing, mistyped, or transformed `rescale` facts
stop before geometry or model loading. Every emulator family requires exact
finite prediction shapes. Background distances reject nonpositive Hubble
rates, matter-power composition rejects row or column broadcasting, scalar
adapters stage all outputs, and no adapter leaves a partial sampled-point
result after a later calculation fails.

**What is missing.** Nothing for this ticket. Workstation-only CUDA checks and
the older matter-power gate fixture remain recorded by their existing open
validation tickets; they do not change the completed CPU behavior or focused
adapter evidence here.

<details><summary>Technical record for development tools</summary>

Evidence: 507/507 developer tests; scalar, CMB, background, cosmic-shear, and
transfer identity gates; focused matter-power adapter assembly; a real-file
rescale refusal and bypass mutation; and the finite wrong-vector witness with
maximum absolute error `28.236`. The broad finite-value gate passed every CPU
arm and reported its mandatory CUDA mirror unavailable. Three independent
reviews returned GO after the Syren pre-composition shape fix and the
production-coupled test expansion.

</details>

### Keep generated datasets complete through training

**High-level summary.** A generator previously wrote related files at
different moments, so resume or append work could combine parameter rows,
payloads, axes, or failure flags that did not belong to one completed result.
Training also opened familiar flat filenames without proving that every file
came from the same generation.

**Current status.** **CLOSED.** Fresh and resumed work stays private until one
complete read-only generation is selected. Each YAML parameter filename now
finds one authenticated training or validation generation, and no mutable
flat-file fallback remains.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Generator requests and member lists are bound before
work begins. Rank-zero refusals reach every MPI worker, all writers close before
publication, and a compare-and-swap prevents stale resume work from replacing a
newer result. Cocoa pins train and validation separately and saves their member
digests, cuts, split seed, and exact staged disk-row order.

**What is missing.** Exact append, recovery of the first interrupted private
draft, persisted sampler state, and old-generation cleanup remain Medium new
functionality under [Continue generated datasets exactly and manage old
generations](#open-dataset-continuation-features). These operations fail closed
and do not change an earlier active generation.

<details><summary>Technical record for development tools</summary>
Landed on `main` as `fa8f170`. Evidence: 478/478 AI tests; 76/76 focused
dataset-publication witnesses; validation-board self-test and dry-run; exact
permanent-note guard; and independent adversarial GO after resident,
disk-backed, scalar, all-row, and wrong-permutation staging checks.
</details>

### Keep failed physics rows out of training datasets

**High-level summary.** A failed physics calculation left a finite zero vector
with the expected shape. Training could therefore mistake the placeholder for
a real cosmology and learn scientifically false behavior.

**Current status.** **CLOSED.** A full data-vector generation cannot publish
while any row is marked failed. Staging also requires the authenticated mask
and removes failed rows before cuts, seeded selection, and pool-size counting.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Only literal `0` and `1` mask rows are accepted, and
the mask length must equal the parameter and payload row counts. A requested
training size cannot be filled with a failed row. The saved row fingerprint is
checked against the disk rows that the loader actually addresses.

**What is missing.** Automatic reproducible replacement of failed rows remains
Medium new functionality under [Retry failed generator rows
reproducibly](#open-generator-failure-retry).

<details><summary>Technical record for development tools</summary>
Landed on `main` as `fa8f170`. The same 478-test suite, 76-witness publication
gate, and independent staging re-audit cover this ticket.
</details>

### Stop training before it can save an unreadable emulator

**High-level summary.** Production training could finish without the
scientific record required by public prediction. It could then save an older
or incomplete file that the same library immediately refused to reopen.

**Current status.** **CLOSED.** Production now checks both dataset records
before expensive setup and writes one readable current format.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Training validates the training and validation
facts before choosing a device, opening a warm-start or transfer artifact, or
constructing the experiment. The exact approved text is retained through
staging. Saving requires that text and the reader-required training and model
instructions, writes schema 3 only, and refuses invalid input before reading
model state or changing output files.

**What is missing.** Nothing for this production ticket. The real-workstation
[gate-fixture checks](#open-schema-v3-gate-fixtures) remain open until their
datasets are regenerated with producer-authored facts. Complete arbitrary
constructor coverage is closed in the
[saved-recipe ticket](#closed-artifact-recipe-totality).

<details><summary>Technical record for development tools</summary>
Landed as `0fe2067`, followed by the Architect-only permanent rule in
`b6c7afd`. Complete arbitrary constructor coverage is now closed in the
[saved-recipe ticket](#closed-artifact-recipe-totality). Evidence: 446/446
project tests; 17/17 focused save and refusal tests; board self-test;
parameter-table, geometry, transfer, scalar, background, CMB, cosmic-shear
adapter, and policy identity checks; and final Red Team GO after the
explicit-null recipe reproduction refused before either output file was
created. CUDA-only compilation and the real cs16/cs8 CoCoA
deployment remain unavailable and are not counted as passes.
</details>

### Publish and load each saved emulator as one authenticated pair

**High-level summary.** A saved emulator has one learned-weights file and one
scientific-record file. The two files previously had no shared fingerprint,
so a crash or file swap could join plausible but unrelated files. Loading the
weights also did not explicitly restrict PyTorch to tensor data.

**Current status.** **CLOSED.** The two files now identify one another and are
checked before model construction.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Both files are staged and checked before their
final names change. The scientific record stores a shared identifier and the
exact SHA-256 fingerprint of the weights. Ordinary failures restore the
preceding pair, an interruption leaves a refusal marker, and concurrent
writers cannot erase one another's completed pair. Rebuild refuses a swap or
malformed declaration before model construction and uses an explicit
tensor-only PyTorch load. Warm-start obtains its settings from the same checked
HDF5 file opening.

**What is missing.** Nothing for this ticket. Output-name collisions and
complete saved-recipe coverage remain separate Open tickets.

<details><summary>Technical record for development tools</summary>
Landed as `9711160de57f54546b0ee675416665131869b13c`. Evidence: 429/429
developer tests; 13/13 focused artifact-pair tests; artifact-composition
acceptance PASS; compile-recipe CPU controls PASS; Python compilation and
whitespace checks PASS; CUDA-only compile evidence unavailable on this Mac;
two independent reviews GO.
</details>

### Authenticate fixed facts in the artifact and adapter chain — Unit 84

**High-level summary.** Saved fixed scientific settings needed to remain
consistent from the training artifact to the Cobaya adapter. Unit 84 added the
first half of that shared authentication with Unit 85.

**Current status.** **CLOSED.** The joint 84/85 change is on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The artifact and adapter use the audited fixed-facts
path instead of rebuilding those values independently.

**What is missing.** Nothing in the production change. The accepted fixture
follow-up remains [an open gate-fixture ticket](#open-schema-v3-gate-fixtures).

<details><summary>Technical record for development tools</summary>
Committed with Unit 85 as `d3b9289`; joint audit is recorded in
`ai/notes/gates-and-board.md`. The historical schema-v2 fixture rider was sent
as mailbox 0161 and is not treated as hidden work in this closed item.
</details>

### Authenticate fixed facts in the artifact and adapter chain — Unit 85

**High-level summary.** Unit 85 completed the adapter half of the same
fixed-facts change as Unit 84. Treating the pair as one change keeps the saved
scientific settings and inference settings aligned.

**Current status.** **CLOSED.** The joint audit accepted both halves.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Unit 85 shares the same code version and evidence
record as Unit 84.

**What is missing.** Nothing in the production change. The accepted fixture
follow-up remains [an open gate-fixture ticket](#open-schema-v3-gate-fixtures).

<details><summary>Technical record for development tools</summary>
Committed as `d3b9289`; see the Units 84+85 audit in
`ai/notes/gates-and-board.md`.
</details>

### Save and verify an artifact's composition mode — Unit 96

**High-level summary.** A reader could infer plain, neural-PCE, or transfer
behavior from which HDF5 groups happened to exist. Schema-v3 artifacts now
declare the native composition mode and refined state, and the reader checks
that declaration against the exact payload and resolved YAML before loading
weights.

**Current status.** **CLOSED.** The composition declaration is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Writer, rebuild, inference, and warm-start agree on
the valid plain, NPCE, and transfer rows and reject presence-only artifacts.

**What is missing.** The real-dump confirmation is in
[the workstation ticket](#open-workstation-debt). The schema-v3 smoke-fixture
repair is tracked separately in [the gate-fixture ticket](#open-schema-v3-gate-fixtures).

<details><summary>Technical record for development tools</summary>
Committed as `3d47318`; four valid rows, 30/30 forged rows refused, 14/14
focused tests, 58/58 full tests, identity gates, board checks, compilation,
diff, and three reviews passed.
</details>

### Authenticate the Grid2D constant mask — Unit 96 rider

**High-level summary.** A saved Grid2D model could previously carry a constant
mask without a value that proved the ordered mask was unchanged. Saves now
record its SHA-256 value for the main geometry and transfer base, and rebuild
checks it before creating the model.

**Current status.** **CLOSED.** The one-surface mask check is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Missing declarations, changed order, and mask data
on non-Grid2D artifacts refuse.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Ten of ten mask tests and all seven MPS identity results passed; moved-pin,
count-only, and validator-bypass mutations failed as intended.
</details>

### Enforce boundary and interior support policy — Unit 94

**High-level summary.** Generated parameter samples needed one explicit rule
for points near the allowed boundary and points in the interior. The accepted
change enforces that policy and supplies the prerequisite used by Unit 8.

**Current status.** **CLOSED.** The current-main version is `f046085`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The old candidate was ported and rechecked against
current main.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Current witness 12/12 passed and all four mutation controls failed at their
named checks. Do not merge the obsolete `a0a03a9` branch again.
</details>

### Validate each generated row before marking it successful — Unit 56

**High-level summary.** A generated row could be marked successful before all
serial, MPI, resumed, dtype, shape, finiteness, and byte-readback checks agreed.
Every path now uses one predicate and clears the failure flag only after the
written bytes are read back exactly.

**Current status.** **CLOSED.** All row-writing paths use the shared check.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Corrupt or wrong-dtype resumed rows refuse without
changing payload bytes, timestamps, or the failure mask.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `e885a8d`; 24/24 focused tests, 292/292 full tests,
compilation, diff, permanent-note guard, and focused review passed.
</details>

### Centralize background quantity and unit validation — Production Unit 62

**High-level summary.** Background quantity and unit pairs were checked in
several places and could disagree. One registry now controls configuration,
geometry, rebuild, and the Cobaya background adapter.

**Current status.** **CLOSED.** All four paths share the registry.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Wrong pairs and non-string, nonfinite, Boolean, or
quoted offsets refuse before save or inference.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `c6fca01`; 9/9 focused tests, 260/260 full tests,
compilation, diff, and permanent-note guard passed.
</details>

### Require CPU-normalized saved model state — Units 64 and 70

**High-level summary.** Saved `.emul` state needed direct proof that it contains
a nonempty tensor dictionary and that every tensor is stored on the CPU. A
ninth independent result now checks those bytes without a load-time device
override.

**Current status.** **CLOSED.** The local saved-state rule is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** An inert compile-default claim was removed, while
the activation-default check and CPU-state refusal cases remain active.

**What is missing.** Nothing for this ticket. The CUDA, CosmoLike, deployment
dump, and `.cpu()` removal run is tracked in
[the workstation ticket](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
Landed and pushed as `fb5302e`; 58/58 tests, board self-test/list,
compilation, diff, helper refusal cases, and two reviews passed.
</details>

### Preserve Grid2D row identity during staging — 25M-32/33

**High-level summary.** Grid2D staging could lose the generator's seeded row
order while moving through raw, base, parameter, data-vector, and index arrays.
Resident and memory-mapped paths now preserve one exact row identity and check
all row counts before allocating transformed targets.

**Current status.** **CLOSED.** Both staging modes use the accepted order.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** N-minus-one and N-plus-one inputs refuse, while the
exact N-row input passes.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `c688489`; 278/278 tests, seven MPS identity results,
row-count witnesses, order mutation, compilation, diff, note style/guard, and
independent review passed.
</details>

### Authenticate optimization-study identity — Unit 53 repair

**High-level summary.** An optimization study could reuse results created with
different scientific inputs, family choices, or implementation rules. One
manifest now fixes that identity before workers start and prevents failed or
stale trials from becoming the winner.

**Current status.** **CLOSED.** Local study identity and worker rules are
enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Workers rebuild the manifest identity, loaded old
studies refuse, and the manifest-owned default is queued once.

**What is missing.** Nothing for this ticket. The real Optuna journal run is
tracked in [the workstation ticket](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
CoCoA Torch witness 34/34, ten critical/coupling mutations, shipped-threshold
canonicalization, exact `n_theta` environment drift, compilation, diff, and
two audits passed.
</details>

### Repair the generator-ranges gate

**High-level summary.** The range gate could miss an old header format because
GetDist might accept or reject comment rows before the intended assertion ran.
It now checks the producer-owned rows-only sidecar directly.

**Current status.** **CLOSED.** The intended sidecar rule is reached.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The exact retired header and decimal-format change
are both caught.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
See `ai/notes/gates-and-board.md`, Gate-integrity pair recovery audit.
</details>

### Repair the cross-family transfer-refusal gate

**High-level summary.** The cross-family transfer check failed early because
its fixture omitted ordinary required data. The fixture now reaches the actual
rule that forbids a transfer base from the wrong family.

**Current status.** **CLOSED.** The named scientific refusal is tested.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Removing `n_train` or `n_val` now fails only the
intended early-data control instead of masking the cross-family result.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
See `ai/notes/gates-and-board.md`, Gate-integrity pair recovery audit.
</details>

### Resolve parameter names for numbered chain files

**High-level summary.** A file such as `chain.1.txt` could fail to find the
shared `chain.paramnames` declaration. One resolver now applies the numbered
root fallback in ordinary, fixed-facts, and scalar staging.

**Current status.** **CLOSED.** All affected staging paths share the resolver.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The exact numbered file, plain table, and dotted
nonnumeric stem rules are tested before data loading.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Six of six focused tests and `stage-ram` passed; deleting the numeric-root
fallback failed the focused witness.
</details>

### Retire a stale rebase ticket — Unit 90

**High-level summary.** Unit 90 appeared unfinished even though its accepted
implementation was already part of main. The backlog entry was reconciled
against Git history instead of merging the same work again.

**Current status.** **CLOSED.** No code change was required.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Commit `50f1c63` is an ancestor of main through
`ce99f87`.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence is the Git ancestry check and the Unit 90 verdict in
`ai/notes/gates-and-board.md`.
</details>

### Validate the CMB covariance package — Unit 13

**High-level summary.** CMB covariance generation needed explicit wiring and
independent failure checks for each validator. The accepted package combines
the scientific calculation with the command path that runs it.

**Current status.** **CLOSED.** The CPU package and wiring are on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Five CPU witnesses pass and four independent
validator-removal changes fail their named program checks.

**What is missing.** Nothing for this ticket. Torch CMB identity and real-CAMB
byte identity are tracked in [the workstation ticket](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
Substance commit `2fd8a9d` plus wiring commit `7583019`; durable owner is
`ai/notes/families-scalar-cmb.md`.
</details>

### Recheck sampled parameter order before loading artifact weights

**High-level summary.** A forged schema-v3 record and matching sidecar could
agree with each other while disagreeing with the rebuilt input geometry. The
reader now compares the sampled-name order with that independent geometry
before calling `torch.load`.

**Current status.** **CLOSED.** The coordinated forgery is refused early.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The valid control advances, while bypassing the new
comparison reaches weight loading and makes the focused test fail.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
See `ai/notes/gates-and-board.md`, Rebuild-time fixed-facts name audit.
</details>

### Use an Ollama model as the Implementer

**High-level summary.** The watcher previously built both the Architect and
Implementer commands around Anthropic's Claude service. A user who could run a
capable open-weight coding model locally still had to spend Claude allowance
for the token-heavy implementation work.

The Architect and Implementer providers are now independent. The Architect
remains on Claude, while `--implementer-provider ollama` sends the Implementer
role to a named Ollama model through Ollama's supported headless coding
integration.

**Current status.** **CLOSED.** This was accepted as **LOW NEW
FUNCTIONALITY**. The default remains Claude, so existing watcher commands do
not change behavior.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The provider and model are separate command-line
choices. Ollama receives the same isolated Implementer worktree, checkpoint
hook, Architect directive, and evidence requirements. `--ping` checks the
Ollama model when selected, and the terminal names each role's provider.

**What is missing.** Nothing for this ticket. The workstation running the
watch must install Ollama, start its service, and download a coding model with
enough context for the requested ticket.

<details><summary>Technical record for development tools</summary>

Primary code: `ai/tools/mailbox_daemon.py::build_agent_commands` and
`check_provider_connectivity`. Focused unit tests replace all provider
programs, while `tools_mailbox_daemon_role_models_repro.py` verifies CLI
wiring and kills mutations that ignore the provider or model choice.

</details>

## Mailbox and watcher behavior

### Start every role turn with an empty provider conversation

**High-level summary.** A user asked whether the Architect, Implementer, and
Red Team should compact their conversations after a ticket closes so an old
ticket cannot fill the context window during the next one.

The watcher already starts a separate provider conversation for every mailbox
turn, but the launch commands did not explicitly forbid saving those sessions.
The intended fresh-context boundary therefore depended on nobody later adding
a resume option.

**Current status.** **CLOSED.** Claude dispatches are now explicitly
non-persistent and Sol dispatches are explicitly ephemeral. No separate paid
compaction turn is created.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** A focused reproduction checks both Claude routes
and the Sol route, and refuses a future command that resumes an earlier
provider session. The tool guide explains that the context limits apply only
inside one unusually long role turn.

**What is missing.** Nothing for cross-ticket context isolation. A single
role turn can still compact before it finishes if that turn alone reaches its
configured context limit; increasing the limit or splitting an oversized
ticket addresses that different case.

<details><summary>Technical record for development tools</summary>

`build_agent_commands()` supplies Claude `--no-session-persistence` and Sol
`--ephemeral`. `arm_each_dispatch_starts_fresh` proves that neither Claude
route contains `--continue` or `--resume` and that Sol does not use
`codex exec resume`.

Landed as `5b3f84f5a5f511064de9bc0ce56f50ae39b5f8d2` after the focused
command reproduction, 91 mailbox tests, and an independent adversarial GO.

</details>

### Stop cleanly when an AI account runs out of tokens

**High-level summary.** Claude or Sol can exhaust an account allowance during
a live watch. The daemon formerly saved the request but kept polling without
telling the user that more credits were required.

**Current status.** **CLOSED.** A verified provider account-limit message now
stops the watch with `Error: Architect is out of tokens`, `Error: Implementer
is out of tokens`, or `Error: Sol is out of tokens`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The exact request is preserved in `failed/`, the
relay log and role worktree remain untouched, another role that already
started is allowed to finish, and no later request starts in that watch pass.
Transient rate limits and context-size failures are not mislabeled as account
exhaustion.

**What is missing.** Nothing for safe stop-and-preserve recovery. Retrying is
a user action after credits are restored, because automatically resetting or
committing a partially edited Implementer worktree could lose work.

<details><summary>Technical record for development tools</summary>

The staleness reproduction covers all three public role names, nearby false
positives, exact inode and byte preservation, no invented timeout history, a
waiting same-role request, and a parallel Sol job that finishes first.

</details>

<a id="closed-user-main-primary-sync"></a>
### Let a clean user update on main reach every AI worktree

**High-level summary.** An ordinary commit or pull in the user's clean main
folder could leave the three saved AI worktrees one commit behind. The next
watcher command then refused to start because a user commit has no internal
ticket-landing receipt.

**Current status.** **CLOSED.** A command launched from the clean user main
folder now recognizes that exact commit and advances every clean idle AI
worktree to it.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The fallback works only when no ticket or candidate
is active, the user checkout is attached to `main`, and its files and index
are clean at the exact main commit. Dirty work, divergent history, malformed
landing requests, and moving only the main reference remain refusals. The
existing baseline helper preserves active candidate work while aligning idle
Architect, Implementer, and Red Team folders.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Committed as `35f39b4`. The disposable Git regression accepts a real user-main
commit and refuses an Implementer-authored reference-only move. Fifty-nine
unit and role-contract tests, five affected recovery arms, the complete
staleness runtime and 9/9 mutation checks, compilation, and whitespace checks
passed. The live `--once` command advanced all three role branches from
`957afc4` to `35f39b4`, reported an empty mailbox, and exited zero.
</details>

<a id="closed-clean-all-ai-worktrees"></a>
### Remove every AI-created worktree and branch on explicit request

**High-level summary.** Old AI sessions can leave enough worktrees, branches,
and mailbox history that the daemon cannot safely decide which folder is the
current one. The explicit `--clean-all` command now lets the user discard all
of that local AI work, including dirty files and unmerged commits, without
deleting ordinary user branches or worktrees.

**Current status.** **CLOSED.** Cleanup runs before primary-folder selection,
so it remains usable when old mailbox histories make `--once` ambiguous.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The command refuses a live mailbox process, removes
registered and abandoned AI folders, deletes local `claude/*`, `codex/*`, and
legacy `worktree-agent-*` branches, and leaves remote branch records, tags,
stashes, and non-AI Git work intact. New Claude role branches use `claude/*`;
new Sol branches use `codex/*`. Cleanup never runs automatically.

**What is missing.** Nothing for this ticket. The user may now run
`python3 ai/tools/mailbox_daemon.py --clean-all` from the main repository
folder when the old AI work is no longer needed.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `957afc4`. Evidence: 138 focused tests, a fresh-role
bootstrap check, exact help/README parity, compilation and whitespace checks,
and one adversarial protected-policy review all passed. The destructive command
was tested only in disposable repositories and was not run in the live clone.
</details>

<a id="closed-provider-connectivity-ping"></a>
### Check whether Claude and Sol can answer before starting work

**High-level summary.** The former ping command only placed a message in the
mailbox. It could not tell the user whether Claude or Sol was logged in and
able to answer. Bare `--ping` now makes one small live request to each service,
while `--ping --skip-redteam` checks Claude without starting Sol.

**Current status.** **CLOSED.** The direct connection check is on `main` and
has been pushed to GitHub.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Each requested service must return the exact
nonce-bearing reply within two minutes. Claude runs without tools or a saved
session; Sol runs read-only in an empty temporary folder. Failure returns a
nonzero status, and the command does not create worktrees, mailbox messages,
backlog changes, or ticket cycles.

**What is missing.** Nothing for this ticket. The user can run `--ping` for
Claude and Sol or add `--skip-redteam` for Claude alone.

<details><summary>Technical record for development tools</summary>

Landed and pushed as `24e7888`. Evidence: 107 focused unit and contract tests;
18/18 two-role runtime checks plus 16/16 mutations; 10/10 dead-mailbox runtime
checks plus 7/7 mutations; dry-run zero-write checks; compilation and diff
checks; and one exact staged adversarial review returning GO. Provider tests
used fake subprocesses, so validation spent no Claude or Sol credits. The
source was simplified before review; its remaining size covers two distinct
provider CLIs and their isolated response channels.

</details>

<a id="closed-role-contract-coverage"></a>
### Make the protected role contract cover every stable security authority

**High-level summary.** The protected YAML named only part of the authority
used by the AI tools. A later Python edit could therefore weaken a guard,
change a saved role worktree, or admit a control file without making the
protected contract visibly change.

**Current status.** **CLOSED.** Contract schema 2 is committed and pushed on
`main` as `96766d6`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The YAML now records the exact eleven notes, all
three role files, both guard files, all nine trusted tools, the Architect
backlog, candidate-forbidden Git and mailbox paths, size limits, and the exact
three role worktrees. Small consumers read those values directly. The daemon
independently compares them with the exact admission sets, live backlog and
mailbox paths, cleanup prefixes, and trusted files that it actually uses.

The permanent-note guard derives its note folder and census from the contract
and always checks both bootstrap guard files. A force-added backlog, mailbox
message, relay log, Git-control file, or role-directory file is refused from
an Implementer candidate.

**What is missing.** Nothing for this ticket. The router's obsolete primary
state schema is a separate Critical ticket and remains Open.

<details><summary>Technical record for development tools</summary>

One adversarial review returned NO-GO because the first draft compared the
YAML with a third set of literals instead of the exact live enforcement
values. The Architect corrected that binding and, as required, did not request
a second review. Tests now mutate the live admission set, trusted-tool set,
cleanup action and prefixes, backlog path, mailbox path, and relay path; every
drift is refused.

Evidence: 135 focused unit and role-contract tests; 6/6 backlog-bundle runtime
checks; 9/9 staleness mutations; the complete disposable primary-worktree
runtime and mutation matrix; compilation and whitespace checks; and an exact
base/candidate change-guard check. The non-test change exceeded the 4,000
character warning because this was one atomic schema migration across the
protected YAML, its reader, its consumers, and the matching role guidance.
Splitting those pieces would temporarily leave two disagreeing sources of
authority, which is the Critical defect this ticket removes.

</details>

### Require one adversarial review of protected policy changes

**High-level summary.** The Architect and Red Team role files and the eleven
permanent notes control how later work is planned. When Red Team is enabled,
the Architect now shows it the exact proposed wording once so that an
unnecessary, oversized, or contradictory rule change receives an independent
challenge.

**Current status.** **CLOSED.** Red Team gives one read-only advisory response.
The Architect then makes the final decision. There is no revision loop, second
review, veto, or post-landing policy review.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** `MAILBOX-TICKET: policy` names the cycle-free
review, including during fix-only maintenance. The two role files share the
Architect-only protected landing path with the permanent notes. A protected
administration turn also refuses to report “no change” after leaving an
uncommitted protected edit.

**What is missing.** Nothing for this ticket. With Red Team disabled, the
Architect records that the independent review was unavailable and applies the
same narrow protected-file checks.

<details><summary>Technical record for development tools</summary>
The one required adversarial review recommended a smaller design. The accepted
implementation uses one ordinary read-only mailbox response and the existing
protected landing path instead of a review journal, reusable approval token,
private proposal reference, or second decision round. Evidence: 80 focused
unit and role-contract tests; 38/38 disposable-worktree runtime cases,
including refusal of an uncommitted protected-role edit; and 56/56 deliberate
safeguard-removal mutations, all armed and refused for the intended reason.
</details>

### Pause long Implementer work for an Architect complexity review

**High-level summary.** An Implementer could previously spend several hours
expanding one repair before the Architect saw that the approach had become too
complicated. The watcher now asks the Implementer to pause after 90 minutes,
save coherent progress, and explain the size, remaining work, elapsed time,
and complexity of the approach.

**Current status.** **CLOSED.** The pause is a progress review inside the same
ticket and cycle. It cannot be accepted as a finished candidate or landed.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The main Implementer receives one instruction at
the next completed tool action; helper agents cannot consume it. The Architect
may authorize another bounded period or replace the plan. A later 120-minute
timeout remains available for an AI process that stops responding.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
The focused hook and role suites, temporary-repository refusal witness,
staleness, ticket-cycle, landing-debt, permanent-note, inventory, compilation,
and whitespace checks passed. The Red Team reviewed the reduced design and
the protected wording and returned GO.
</details>

### Require a checkpoint decision before implementation resumes

**High-level summary.** A 90-minute pause could previously return to the
Implementer without one explicit Architect decision. A checkpoint with no new
commit also had no immutable candidate for the Architect to inspect.

**Current status.** **CLOSED.** Every timed checkpoint now needs a new clean
commit and one fresh same-cycle, same-mode GO or NO-GO handoff before work can
resume.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Malformed or contradictory checkpoint state is
refused before the Architect starts. Checkpoint prompts omit ordinary landing
instructions, a checkpoint cannot send landing GO, and conflicting fresh
outputs are parked together rather than accepting only the convenient one.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `4e21b6f`. Evidence: 148 unit and contract tests; six
live checkpoint branches; six safeguard-removal mutations; staleness,
ticket-cycle, and landing-debt suites; compilation and whitespace checks; and
two independent adversarial reviews GO. The approximately 6.4k-character
production change exceeds the 4,000-character warning with a recorded
Critical exception: prompt authority, immutable checkpoint identity, exact
decision binding, and atomic output refusal had to close together so no
intermediate commit left the pause bypassable. No schema, saved phase, history
scan, framework, or production file was added.
</details>

### Keep GitHub commit messages readable

**High-level summary.** Landing commits previously replaced the Implementer's
explanation with an internal ticket label. A human reading the GitHub history
could not learn what changed, why it changed, or which checks passed without
opening the code diff and the tracked backlog.

**Current status.** **CLOSED.** Manual AI commits and mailbox commits now
follow the same human-first writing rule.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Every AI-authored commit needs a concrete subject
and a short Markdown body that explains the observed problem, the saved
change and its boundary, and exact evidence. The watcher preserves the exact
Architect-approved UTF-8 message in the landing commit, adds only two reserved
recovery lines, and refuses ambiguous or altered recovery messages.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Implementation commit `861acd5`; permanent-rule commit `3e22a1b`. Evidence:
429/429 project tests; 37/37 watcher runtime scenarios; 34/34 role-contract
tests; focused raw-message and crash-recovery reproduction; compilation and
whitespace checks; independent code and permanent-rule reviews GO.
</details>

### Keep mailbox dry runs read-only

**High-level summary.** A dry run could move a malformed or placeholder
message even though the user asked only to preview the action. Dry-run mode now
leaves the exact pending file in place and creates no failed record.

**Current status.** **CLOSED.** Preview and real dispatch have separate file
behavior.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Only a real dispatch claims a pending file into the
work-in-progress folder.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
See `ai/notes/gates-and-board.md`, Current-daemon transport safety audit.
</details>

### Use readable sentence-case terminal output

**High-level summary.** Daemon output used dense separators and all-capital
phrasing that was difficult to scan. User-facing lines now use sentence case
and semicolons while preserving exact protocol and acronym text.

**Current status.** **CLOSED.** Runtime and README quotations agree.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Separator and capitalization changes are covered by
focused refusal controls.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused output checks 8/8 and daemon reproduction 8/8 passed; separator and
all-caps mutations failed.
</details>

### Do not require a reply to an explicit terminal message

**High-level summary.** Every inbound mailbox message formerly demanded a
reply, including one that explicitly said the conversation was finished. An
exact terminal/no-reply message is now exempt, while any ambiguity still
requires an outbound response.

**Current status.** **CLOSED.** The single exception is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Ordinary notes-first prompts retain the required
reply and both parts of the wording are mutation-tested.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Four prompt/surface regressions and the eight-arm daemon reproduction passed;
removing the ordinary rule or terminal exception failed its focused check.
</details>

### Keep claimed mailbox work current and publish outcomes safely

**High-level summary.** A claimed message could become ambiguous when newer
work arrived, a timeout occurred, or another process touched its archive path.
Each dispatch now records one current-state snapshot and publishes only the
exact claimed file to done or failed.

**Current status.** **CLOSED.** Later work stays blocked until the claimed item
has a truthful outcome.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Timeout history, `--once` propagation, exact file
identity, line endings, dry-run behavior, and hostile history refusal are
covered.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 18/18, nine source mutations, preserved daemon/output/preamble
suites, and two independent reviews passed.
</details>

### Complete the combined daemon repair program

**High-level summary.** Several related daemon repairs were once tracked by one
umbrella line in addition to their individual tickets. This roll-up confirms
that recovery, prompts, output, archive handling, watcher warnings, fix-only,
safe stopping, and landing-debt behavior all have accepted child records.

**Current status.** **CLOSED.** This is a summary, not a separate code defect.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Each child appears as its own closed ticket in this
section with focused evidence.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Source record: daemon audits through Daemon landing-debt self-correction audit
in `ai/notes/gates-and-board.md`.
</details>

### Warn when a sent message has no live watcher

**High-level summary.** A send could succeed into a mailbox that no watcher was
reading, leaving the user to assume work had started. Send and ping now name
that mailbox and any other live watched mailbox without rerouting or failing
the successful send.

**Current status.** **CLOSED.** The warning is visible and dry-run stays
read-only.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Exact owner parsing, file identity, and link refusal
keep the warning conservative.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 9/9, seven mutations, daemon/output/preamble/staleness suites,
and two reviews passed.
</details>

### Protect fix-only mode and classify Sol tickets

**High-level summary.** Fix-only mode could not reliably distinguish a known
repair from a new discovery, especially near the discovery-demand limit.
Public Sol requests now declare closure or discovery, and fix-only launches
only accepted closure work plus the exact internal transport ping.

**Current status.** **CLOSED.** Ambiguous actions fail before launch.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Saturated discoveries are preserved with a clear
instruction, demand-nine work keeps its classification, and one mode lock
binds child and external sends.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 14/14, twenty mutations, dead-mailbox, Red Team, staleness,
output, and preamble suites plus two reviews passed.
</details>

### Provide regular windows for stopping the watcher

**High-level summary.** A busy watcher gave the user only occasional idle
moments to stop it without interrupting active work. After five completed role
runs or fifteen continuously busy minutes, it stops starting work, waits for
started work, and prints a 19-to-0 Ctrl-C window.

**Current status.** **CLOSED.** The bounded stop window is implemented.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Queue bytes survive Ctrl-C, source edits prevent the
next admission, and preview/one-pass commands keep their finite behavior.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 9/9, seven boundary mutations, preserved daemon suites, and
the Architect/Red Team wording audit passed.
</details>

### Stop the watcher after a chosen number of cycles

**High-level summary.** The user needed a planned stopping point instead of
waiting to catch one short Ctrl-C window. Positive `--cycle N` now stops after
N completed safe windows, while zero waits until no enabled message and no
open backlog ticket remain.

**Current status.** **CLOSED.** Omitted, positive, and zero modes have distinct
tested behavior.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Waiting messages remain untouched at the positive
limit, concurrent sends receive truthful watcher status, and missing or
changing backlog files keep zero mode running rather than closing early.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 20/20 with 21 mutations, all related daemon/router suites,
44/44 CoCoA tests, board self-test, compilation, and diff passed.
</details>

### Request landing when uncommitted work grows too large

**High-level summary.** A long-running branch could collect too much accepted
work without asking the Architect to land it. Above 400 changed lines, the
watcher now creates one durable landing request for that continuous episode
and prints the current debt in each demand report.

**Current status.** **CLOSED.** Landing requests are deduplicated and rearm
after the branch returns to 400 lines or fewer.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Fable checks for foreign commits before a squash,
and one repository lock prevents Fable and Sol from landing concurrently.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 15/15 and ten mutations on both runtimes, preserved suites,
44/44 CoCoA tests, board self-test, compilation, diff, and two reviews passed.
</details>

<a id="closed-implementer-context-handoff"></a>
## Preserve exact Implementer context before a replacement session

### High-level summary

An Implementer can reach the context limit while a ticket is only partly
finished. Repository files survive, but a fresh Implementer also needs to know
what succeeded, what failed, and which rejected approach would waste time if
repeated.

The watcher now asks for one small record before automatic context replacement.
The replacement reads that exact record and checks the repository instead of
receiving a summary invented by the watcher.

### Current status

**CLOSED.** This was accepted as **LOW NEW FUNCTIONALITY**. Repeating an
unsuccessful approach wastes tokens, but the former behavior did not corrupt
scientific output or erase repository work.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

The automatic context hook asks the Implementer to name the ticket, base and
current commits, candidate status, completed work, failures, rejected
approaches, uncommitted files, next action, and work that must not be repeated.
The watcher verifies those facts against the current Implementer worktree.

The record follows the existing checkpoint path. It creates no candidate and
closes no cycle. After the Architect permits continuation, the replacement
receives the exact saved path and keeps the verified unfinished worktree.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

`implementer_checkpoint_hook.py` requests the record through Claude Code's
automatic `PreCompact` hook. `mailbox_daemon.py` parses the bounded record and
verifies the cycle, base, HEAD, candidate
claim, and dirty state before reusing the existing checkpoint route. Focused
tests cover clean and dirty records, malformed fields, stale identity, exact
saved-path delivery, preservation of unfinished work, and absence of landing
instructions.

No session graph, task scheduler, supervisor, or additional role was added.
If a provider stops before creating the record, the existing out-of-token
recovery continues to preserve the request and worktree without fabricating a
summary.

</details>

<a id="closed-implementer-shared-notes-grant"></a>
## Let the Implementer open the shared notes directory it is told to use

### High-level summary

The three AI roles work in separate git worktrees so their code edits cannot
collide. One directory is deliberately shared instead: the Architect
worktree's `ai/notes`, which holds the mailbox, the directive note for the
current ticket, and the guard programs. Every dispatched role turn is told to
use that exact directory through the `MAILBOX_SHARED_NOTES` setting, and the
directive checker refuses a directive note stored anywhere else.

The launch command granted file access to that directory to the Red Team
alone. The Implementer was told to use a directory its file tools were not
allowed to open. A Claude Implementer never noticed, because Claude Code
opens an absolute path outside its working directory anyway. An Ollama-served
Implementer refused instead, and returned a checkpoint reporting that the
directive note and the mailbox were unreachable.

### Current status

**CLOSED.** This was accepted as **HIGH BUG FIX**. A dispatched Implementer
could neither read its directive nor write its return message, so the role
performed no work at all and one live ticket turn was spent producing
nothing. Medium is not sufficient because the failure was certain rather than
likely: every Ollama Implementer dispatch failed the same way, and the halt
covered the whole role rather than one option.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

Both Implementer launch commands, Claude and Ollama, now end with the same
notes grant the Red Team command already carried. The written description of
the `shared_notes` value, which called the directory the Red Team's only
extra writable directory, now names the Implementer as well.

### What is missing

Nothing for this ticket. A live Ollama Implementer dispatch has not been run
again since the repair; the next ordinary ticket exercises it.

<details><summary>Technical record for development tools</summary>

`build_agent_commands` in `ai/tools/mailbox_daemon.py` builds both Implementer
routes, and each now carries `--add-dir` with the Architect notes directory. A
focused test in `ai/tests/test_ollama_implementer_runtime.py` asserts that both
provider commands carry exactly one such grant naming that directory.

The Red Team command is unchanged and keeps the grant in its final position.
Codex spells the option with a single directory value, and the dispatcher
separates the prompt from the options with `--`, so nothing there can absorb
the message.

Full suite 814 OK.

</details>

## Repository organization and release hygiene

### Move study helpers into `emulator/studies/`

**High-level summary.** Four related modules used a repeated `study_` filename
prefix in the main emulator folder. They now form one `emulator.studies`
package, which makes their relationship visible without changing scientific
behavior.

**Current status.** **CLOSED.** The old flat files and compatibility duplicates
are absent.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Production, gates, comments, and package maps use
the package namespace.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
The Unit-53 manifest witness, 44/44 tests, board import closure, compilation
and import probes, and independent named-change audit passed.
</details>

### Keep operational backlog files out of release history

**High-level summary.** Temporary backlog and mailbox records could be mixed
with the eleven durable notes and accidentally enter a release commit. The
repository now ignores operational records, tracks exactly eleven permanent
notes, and represents accepted work with one reviewed commit.

**Current status.** **CLOSED.** The v1.0beta1 hygiene prerequisite is satisfied;
no release tag was created.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The primary-worktree reproduction force-tracks only
its disposable synthetic backlog, which also repairs the older fixture drift.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `c91791a`; primary-worktree runtime 20/20 with mutations killed,
65/65 focused tests, permanent-note guard, tracked/ignored census, README
checks, and whitespace check passed.
</details>

### Move all AI-development support under `ai/`

**High-level summary.** Tests, notes, gates, and tools were scattered among
several root folders and old entry points. `ai/README.md` is now the single
starting point, with the four support folders only under `ai/`.

**Current status.** **CLOSED.** Paths, imports, tools, documentation, and
ignored transport state use the consolidated tree.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** No old root directory, wrapper, duplicate entry
point, or compatibility link remains, and the documentation PDF builds from
the new paths.

**What is missing.** Nothing for this filesystem move. The real-data full-board
run is tracked in [the workstation ticket](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
Evidence: stale-path/filesystem audits, compilation/import probes, 44/44 tests,
nine tool reproductions, board list/dry-run/self-test, focused CPU gates, tool
help/status, and an 83-page PDF render. The development Mac correctly lacked
CUDA, CosmoLike, configured `$ROOTDIR`, and training dumps.
</details>

<a id="open-router-primary-schema-three"></a>
## Let the router read the current primary-worktree record

### High-level summary

The daemon creates the saved Architect worktree with state schema 3 and the
topology name `separate-role-worktrees-v1`. The router formerly expected the
retired schema and topology, so a valid fresh setup could fail before a role
read the authoritative backlog.

The source mismatch was repaired, but the ticket remained listed as Critical
and Open. The workflow also allowed that stale bookkeeping because landing did
not require the Architect to close and seal the exact ticket first.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** Commit `864b69e2f300b44e40a270b4343f3bd495557a80`
teaches the router the current record. Commit
`9f99d1a9441771b1c5889002e3d13f4ee3d23bce` makes a Closed, sealed ticket a
prerequisite for a new landing.

### What is already fixed

The router reads the daemon's schema-3 record and retains its checks for the
repository, branch, path, topology, and file type. The daemon now refuses GO
while the ticket is still Open, preserves candidate C, and retires a rejected
same-cycle GO after a corrected GO lands, including after restart.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

The closure gate is in `ai/tools/mailbox_daemon.py`; the Architect instruction
is in `.claude/FABLE_ROLE.md` and `ai/notes/conventions-and-workflow.md`.
`test_architect_go_needs_the_exact_ticket_closed` and
`arm_architect_receipt_binds_candidate_to_squash_landing` cover refusal,
candidate preservation, corrected GO, and restart cleanup.

</details>
