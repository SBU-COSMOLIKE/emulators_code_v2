# notes/ index

Consolidated 2026-07-11: ~85 topic notes rewritten into the compact topic and
audit set below so any model (or human) can orient fast. Every retired note
survives in git history (`git log --follow notes/<old-name>.md`); the
delta IDs preserved in these files are the search keys.

Read in this order for a cold start: (1) this index,
(2) the durable correctness registry in
red-team-audit-and-didactics-2026-07-13.md, (3) project-and-history.md, then
the topic file your task touches. Use state-2026-07-11-and-next.md for
chronology and routing, not as a live queue: its opening snapshot is stale and
its later blocks preserve decisions rather than one canonical current order.
Current sequencing comes from `gates-and-board.md`, "The consolidated
DIDACTICS execution handoff" (D1--D10), and must be checked against the
relevant topic note before implementation.

Hard communication rule: the substantive exchange between Fable, the
Implementer and the Red Team is written under `notes/` before a chat relay is
sent. Chat handoffs are summaries that cite the durable note. See
`conventions-and-workflow.md`, "Notes-first inter-agent communication." The
same rule governs `notes/mailbox/` dispatches: a mailbox file is a routing
summary, and a mailbox-started turn writes its outbound block to the next
numbered mailbox file after recording the substance under `notes/`.

- [State + chronological ledger](state-2026-07-11-and-next.md) — the
  historical run trail, adjudication batches, queue changes, retractions, and
  topic-note routing. Do not promote any one historical run snapshot or queue
  paragraph to current state.
- [Project + history](project-and-history.md) — the goal, the
  development arc by phase, the family-pattern recipe (what a new
  output family adds), the program-level lessons.
- [Training stack](training-stack.md) — losses (sqrt/chi2/berhu
  ladder + roughness), the shared anneal family, phase blocks +
  demotion, EMA + the snapshot invariant, consumed-view banners,
  sizing (absolute counts, derived eval bs, weight-decay allowlist),
  the loud no-alias migration pattern; open red-team details include
  the dead validation-safe chunk, the unbounded wide-output
  diagnostics path, and activation-bakeoff process liveness.
- [DIDACTICS-59 real-evaluation return](training-stack.md#didactics-59-red-team-return-2026-07-14)
  — real `eval_val` over full/equal/ragged partitions, production per-row
  diagnostics, histories/scheduler, and drop/reassociation catch-power.
- [Models + designs](models-and-designs.md) — ResMLP/ResCNN/ResTRF,
  the correction-head philosophy, zero-init identity discipline,
  factored IA, activations/norms, FiLM, NPCE, the science doctrine
  (sample efficiency, coverage floor), the CLOSED-experiment ledger.
- [Artifacts + inference + warm starts](artifacts-inference-warmstart.md)
  — schema v2 (never-trust-defaults), rebuild, EmulatorPredictor, the
  five cobaya adapters (python_path trap incl.), fine-tuning (FTW),
  transfer (TPE, refine, anchors), the geometry folder (GEO,
  D-GEO5 shims retired), plus the open real-Cobaya dependency-routing
  defects hidden by stubbed adapter gates and the open geometry-state /
  covariance-validation contract.
- [Families: scalar + CMB](families-scalar-cmb.md) — SPE (closed,
  the lesson bank) and CME (covinv ruling, amplitude law, covariance
  script, roughness, diagnostics dispatch) + the D-CM12 SPEC AWAITING
  AUDIT and D-CM13 IMPLEMENTED (heads on every coordinate family,
  user-ordered 2026-07-11).
- [Families: background + matter power](families-background-mps.md)
  — BSN (two-regime, imposed distances, flat-only, the Simpson
  finding) and MPS (correction-to-syren, D-MP2-A base-on-disk, the
  vendored syren/, EMUL2, MPS-DIAG).
- [Data generation + cuts](data-generation-and-cuts.md) — the four
  generators on generator_core + the covariance script, tempered/
  uniform sampling, the output contract, staging/memmap, the
  param_cuts windows and the coverage-cut lesson; open checkpoint-set
  integrity covers manifest membership, append publication, and the
  destructive load-error fallback.
- [Gates + the board](gates-and-board.md) — the harness, identity/smoke
  philosophy, dead-network rule, resume/evidence design, manifest population,
  and run history. `gates/board.py` plus `python3 gates/run_board.py --list`
  are authoritative for the current gate set; prose counts are snapshots.
  1b hardening COMPLETE (9/9 + item-7: 18 coverage, 16 runtime-loaders +
  data-read leaves, 19 owner resolvers + run-time refusal, 26 lineage) plus the
  27/28 machinery follow-up (tracked-driver watch, shared stale-member surface);
  QUEUE 2 OPENS. Remaining texnotes/D1 prose deferrals only. Now also holds the
  FIXED-FACTS ADJUDICATION (2026-07-14: all eight forks RULED — two sibling
  groups, chain-digest dataset_id amendment, schema v3, resolved-model truth,
  facts.yaml sidecar, one shared reader, three landings producer-first; landing 1
  dispatched 0105) and the skipped-leg sweep AUDITED GO (182 PASS selftest,
  mutation + independent probe re-run; scalar-smoke doctrine ruled onto Sol's
  nine-aid child, delta 0106).
- [Conventions + workflow + environment](conventions-and-workflow.md)
  — the Python/docs/README/plots/terminal/YAML house rules, the
  dual-agent workflow, git discipline, the Mac evidence pattern,
  machines and ROOTDIR.
- [User didactics + Python voice](user-didactics-and-python-voice.md)
  — who the reader is (C coder; cosmologist audience), how she likes
  to be taught (show-never-describe, define-or-drop, run-first), and
  the code register with her own quotes and before/after shapes; the
  Implementer reads it before writing code or docs.
- [Whole-package style audit (2026-07-05)](audit-package-style-2026-07-05.md)
  — dated pre-consolidation Architect record: whole-tree audit of
  emulator/ + the five drivers + README + example_yamls (structural
  axes PASS; the fix list was handed off 2026-07-05 and largely
  executed by the later doc/POL sweeps — read it as history + method,
  not an open queue). Side-finding: the june2026/claude_skills copy of
  pytorch-teaching-style was stale; the live skill lives in
  ~/data/claude_skills/.
- [Red-team Implementer handoff (2026-07-13)](red-team-implementer-handoff-2026-07-13.md) — single-page 45M inventory + per-unit commits/files/gates + the "audit hardest here" guidance (86-90 vs the Architect specs now on main, and the five 45M-72 design decisions to rule on); the review brief for the Architect audit pass. Landed on origin/main via merge 8ce72a9.
- [Red-team audit + durable handoff registries (2026-07-13)](red-team-audit-and-didactics-2026-07-13.md) — independent review of the Implementer landing; binding evidence-map rulings; the canonical pre-45M unit crosswalk; complete 45M/20M/RT/BLOAT routing with explicit unused-id tombstones; the DIDACTICS-42--100 registers; and the continuing 25M correctness index. Future Red Team handoffs are appended here or to their existing topic note before their chat copy is sent.
- [Units 41+53 persisted-identity review](red-team-audit-and-didactics-2026-07-13.md#unit-4153-redteam-01-independent-persisted-policy-and-study-manifest-review-2026-07-14) — independent executable HOLD: artifacts omit AMP dtype/scaler policy, sweep products re-publish raw activation and omit head pins, and the tuner has no scientific manifest or family-owned stable study name. Home-note readback: `training-stack.md`, "Units 41 and 53 Red Team readback".
- [Mailbox daemon incident (2026-07-14)](mailbox-daemon-incident-2026-07-14.md) — the 0014 placeholder body reached a live Opus turn because the running `--watch` process (lock pid 45327, started 00:52) predates its own fix (`55eb256`, 00:53:06): a daemon hardening commit is inert for the loop already running. Plus two more transport defects (refusals are never quarantined; `next_seq()` collides — `done/` already holds a duplicate 0008 pair) with ready-to-apply patches. RESTART THE WATCHER FIRST — until then no committed guard is protecting the loop. FOURTH DEFECT (2026-07-14, red team + Implementer): the dispatch preamble (`tools/mailbox_daemon.py:176-190`) unconditionally orders every headless turn to write an outbound, which contradicts an inbound whose binding instruction is TERMINAL/no-reply — four live firings (0128/0129/0131 in Sol's lane; **0130 in the Implementer's**, so the class is not Red-Team-specific and the repair must also narrow `.claude/OPUS_ROLE.md` 7a, which is unconditional too). "Live reproduction 4" records the COMPOUNDING with the stale-dispatch class: 0130 was compulsory to answer AND named an already-executed dispatch (0124) as live work, so a literal reading would have re-run landings 2+3 onto the uncommitted tree — which is why the ledgered currency marker must be MECHANICAL and in the dispatch banner: a message body cannot describe its own currency, because it was honest when it was written. ADJUDICATED 2026-07-14 ("Architect adjudication of 0133" in the note): repair shape ACCEPTED (conditional in words, agent keeps semantic judgment, no daemon parser), narrow-exception ruling (ambiguity defaults to outbound-required), five-surface word sweep pinned (PREAMBLE :268-282 — the :176-190 citation drifted — + daemon docstring + OPUS_ROLE 7a + this index's header prose + the conventions canonical sentence), no-second-instruction scan pre-run clean, backlog OPEN line added (reproduction 4's "both classes are OPEN" overclaimed), 0134/0136 pre-ruled quick closes, folded into the tools-review repair unit.
- [TEX-PROSE-04+05+06 adjudication: HOLD, landing unreachable](red-team-audit-and-didactics-2026-07-13.md#tex-prose-040506-architect-adjudication-hold--landing-unreachable-2026-07-14) — the handed-back SHAs (`9365e9a`, tip `5546a0f`) exist in no reachable ref or object database: `codex-tex-prose-04-06` is an UNLINKED clone, not a linked worktree, and the headless auditor cannot reach it. Transport hold, substance unadjudicated; repair = publish the branch at the exact tip (user-side fetch command printed in the entry). TEX-PROSE-07+08 inherits the reachable-ref handback requirement.
- [Unit 8 HALTED: the rebase target was never built](state-2026-07-11-and-next.md#unit-8-halted-at-the-premise-check-the-rebase-target-was-never-built-2026-07-14-opusimplementer-mailbox-0103) — dispatch 0103 told the Implementer to rebase unit 8 on "unit 94's landed seam"; unit 94 was never implemented (`nextafter` appears in SIX commits, all `notes:` prose — never in code on any ref; no `codex/unit94-*` branch while every other red-team unit has one; no GO record). Unit 8's identity manifest must digest two fields (resolved per-name bounds + 94's boundary-interior policy) that no code produces. Side defect: unit 94 is absent from `notes/backlog.md` entirely — invisible to the demand count, so it will never be dispatched, and it silently blocks unit 8, which IS on the ledger. Three ways forward in the entry; ADJUDICATED same day — see the next line.
- [Unit-8 halt adjudicated: ruling A; the invisible-unit ledger class repaired](state-2026-07-11-and-next.md#unit-8-halt-adjudicated-accepted-ruling-a--unit-94-dispatched-0117-the-ledgers-invisible-unit-class-repaired-2026-07-14-fablearchitect) — the halt is ACCEPTED (every probe re-run; one overstated proof clause corrected: c03a084 put `np.nextafter` into mps_identity.py fixtures, never into generator_core.py). RULING A: unit 94 dispatched to the red team (0117, base 204748e, red-team mode; clone-check-first; sequencing clause vs in-flight fixed-facts landing 1 on verified-disjoint generator_core regions); unit 8 re-dispatches only after 94's audit GO. Ledger sweep adds five invisible OPEN units (94, 90-rebase, unit-13 covariance, TEX-PROSE-04/06, tools-review carrier); backlog 24 -> 29.
- [Tools-review adjudication: HOLD, landing unreachable; defects confirmed live](red-team-audit-and-didactics-2026-07-13.md#tools-review-architect-adjudication-hold--landing-unreachable-every-claimed-defect-confirmed-against-reachable-code-2026-07-14) — second unlinked-clone handback (`codex/tools-review` at `96e5f26` unreachable), BUT all fourteen claimed router/daemon defects are real in the reachable code: four confirmed by scratch execution (incl. the cross-recipient collision that fired live today — the 0107-to-fable/0107-to-sol pair), the rest by line-cited inspection, plus two Architect extras (every failed dispatch crashes its lane thread on `proc.stdout=None`; the router still injects the retired "backup Implementer" sentence). Repair = publish the ref at the exact tip (commands in the entry); interim daemon mitigations listed there — `--dry-run` is NOT read-only.
- [TEX-PROSE-07+08 adjudication: HOLD, transport; preservation hash PINNED](red-team-audit-and-didactics-2026-07-13.md#tex-prose-0708-architect-adjudication-hold--landing-unreachable-preservation-hash-pinned-by-recomputation-2026-07-14) — third unreachable handback, now proven environmental (Sol's push is rejected by the read-only shared object store — no strike); the claimed `97e938bb...` field hash REPRODUCED from base 204748e by the Architect's own extraction (pinned verbatim in the entry, 120 = 40/40/40), so the audit is pre-armed at publication; user fetch block printed; RULING A: evidence-only gaps travel in delta messages, tips never rewritten (04-06's `888272b7` extraction owed via that channel); RULING B: 04-06 lands first, 07-08 rebases as an expected new handoff.
- [README DELTA audit PASS + the print-register ruling](gates-and-board.md#readme-delta-audit-architect-2026-07-14-pass--every-gate-re-run-and-reproduced-both-flagged-defects-adjudicated) — b193849 audited PASS with every gate re-run and reproduced (2026-07-14; gate 2 byte-exact with the ledger pinned at its landing-time 22 — the live 24 is post-landing dec161c growth); both deviations approved. RULING: verbatim fenced quotes of shipped output STAND even where the output breaks the register — the PRINTS are the defect (` -- ` + all-caps emphasis), fixed in the tools-review daemon-repair unit with the README's quoted lines refreshed in the same series.
- [Unit-96 adjudication: HOLD, transport; contract pinned; baseline reproduced](red-team-audit-and-didactics-2026-07-13.md#unit-96-second-implementer-adjudication-hold--landing-unreachable-contract-pinned-from-ruled-sources-baseline-gate-reproduced-2026-07-14) — fourth unreachable handback (environmental, no strike; the two-phase fetch-block expectation postdates dispatch 0102); dispatch 0102's cited training-stack unit-96 section does not exist at 204748e — Sol's recovery from the ruled sources (families-background-mps.md:1217, register :2897) is APPROVED; KEPT-CORE FLAG: gates-and-board.md:5587 rules unit 96 not pre-authorized to leave the Implementer without a fresh user ruling — the user confirms the dispatch at the fetch step; five-test const-mask baseline re-run by the Architect (five ok); audit pre-armed incl. the digest-vs-no-second-trusted-axis design question; results.py seam vs fixed-facts landing 1 sequenced RULING-B style. Follow-up (register section "UNIT-96 preservation checkpoint (0119) adjudicated"): the checkpoint is ACCEPTED (shared-store negatives re-verified by the Architect; the clone is still unlinked; no strike) and the checkpoint mailbox loop is TERMINATED (0129) — the user fetch + kept-core confirmation stays the only open action.
- [Scalar-smoke doctrine delta: audit PASS + the linked-worktree transport playbook](gates-and-board.md#scalar-smoke-doctrine-delta-audit-architect-2026-07-14-pass--the-transport-block-is-solved-in-place-commit-68f0e77-created-after-the-audit) — Sol's lock-blocked two-file delta audited with every gate re-run by the Architect (live gate rc 0, nine PASS terminals, calibration digits byte-matched; census 9 == 9 == 9; Sol's three forced exit paths reproduced plus an unscripted mid-leg-crash arm; branch selftest 170/0), then committed IN PLACE as 68f0e77 by entering the linked worktree — a delta stranded in a LINKED worktree is Architect-committable directly, unlike the unlinked-clone HOLD class; merge user-owed with a named expected conflict in the durable-record note (+497-line divergence).
- [TEX-PROSE-04+05+06 publication delta adjudicated: RULING-A debt verified closed, both TeX audits pre-armed](red-team-audit-and-didactics-2026-07-13.md#tex-prose-040506-publication-delta-adjudication-hold-unchanged-ruling-a-debt-verified-closed--888272b7-reproduced-at-base-by-independent-reimplementation-2026-07-14) — mailbox 0112 formally adjudicated (2026-07-14): transport HOLD unchanged (ref + tip object still absent; still an unlinked clone), the delta itself ACCEPTED as a compliant phase-one handback, its R2/R3 question already answered by RULING A, and Sol's exact extraction REPRODUCED at base 204748e by the Architect's byte-faithful python3 reimplementation (663 lines kept, 30,653 squeezed bytes, `888272b7...` exact; single-byte mutation arm reds) — which also pre-adjudicates the queued 0118 evidence delta; both TeX substance audits now wait ONLY on the user-side fetch printed in the entry. Follow-up (2026-07-14, register section "evidence-delta routing copy (0118) closed as pre-ruled"): 0118 fired and is CLOSED — third independent base reproduction of `888272b7...` plus a new terminal-space probe proving the `bodies_ws_nl` mismatch account byte-true; Sol's frozen-tip acknowledgment is on the register; the thread's mailbox loop is TERMINATED (0128), leaving the user fetch as the only open action.
- [Tools-review + 07+08 publication delta adjudicated: HOLD bilaterally confirmed — no agent-side transport for unlinked clones](red-team-audit-and-didactics-2026-07-13.md#tools-review--tex-prose-0708-publication-delta-adjudication-0113-accepted-no-strike--the-unlinked-clone-hold-is-bilaterally-confirmed-2026-07-14) — mailbox 0113 formally adjudicated (2026-07-14): ACCEPTED, compliant phase-one handback, no strike (the 0112 twin's discipline); both tip objects and both target refs re-verified absent (no partial leak; both source paths re-confirmed UNLINKED clones). NEW: the Architect probed the transport itself — the clone read AND a non-main `git fetch` from the home worktree both stop at approval gates no headless turn can grant, so the unlinked-clone HOLD class is user-owed from BOTH lanes (Sol's push write-blocked at the shared `.git`, Fable's fetch approval-gated at the clone path); the consolidated three-fetch user block is printed in the entry; audit re-request trigger ratified: `codex/tools-review reachable at 96e5f26`.
- [Unit-94 return (0121) adjudicated: ACCEPTED; HOLD unchanged; audit pre-armed](red-team-audit-and-didactics-2026-07-13.md#unit-94-return-0121-adjudicated-accepted--transport-hold-unchanged-the-audit-is-pre-armed-by-independent-base-reproduction-2026-07-14-fablearchitect) — the stale-arriving original unit-94 return is ACCEPTED against dispatch 0117 (second stale-dispatch firing; no strike); the unlinked-clone HOLD re-confirmed from the shared side; the Architect independently reproduced the exact-base self-test (176/0 ALL PASS from a scratch extraction of 204748e) and all four policy numerics BYTE-EXACT from the ruled nextafter-interior policy alone (incl. the f32-width-ratio pin); the landing-1 seam verified disjoint from this side (margin byte-intact at HEAD:1149, displaced +402); unit 8 stays blocked; the user fetch trip is consolidated to FOUR branches in the entry; mailbox loop terminated (0131-to-sol) with 0125/0126/0127 pre-ruled to quick closes.
- [Fixed-facts LANDINGS 2 + 3 (Implementer, 2026-07-14): unit 82 CLOSED; the three laws LANDED; the adapter half BLOCKED](gates-and-board.md#fixed-facts-landing-2-implementer-2026-07-14-the-canonical-decimal-policy-in-the-one-writer--landed-cpu-green-on-this-mac)
  — LANDING 2 closes unit 82: the `.ranges` bounds now go through
  `fixed_facts.format_value`, the 25M-06 witnesses round-trip DISTINCT and
  float32-exact (`70.00001`/`70.00002`, `0.12345674`/`0.12345676` — all four
  collapsed under `%.5e`), the view's text is byte-identical to the record's, the
  dead `hd` is cleared, and a `%.5e` restoration arm reproduces the defect on
  demand (generator_ranges 2 -> 5 PASS; its 1 red is the known hollow arm,
  re-proven pre-existing on HEAD by my own scratch layout). LANDING 3 lands the
  THREE COMPARISON LAWS (vertical/horizontal/domain + the pair's intersection) in
  the torch-free module, retains the record in `EmulatorPredictor`, and executes
  the positional-trust amendment (a bare ordered row is refused; a permuted
  `(names, values)` pair names both orders) — fixed_facts_schema 34 -> **79 PASS**,
  12 aids, every identity gate at its exact baseline (24/40/78/69). RULING 2
  clauses 1+2 done (16 arms needled, 4 duplicate arms re-fixtured). **The adapter
  wiring is CHECKPOINTED on three questions**: the VERTICAL law's resolution site
  does not exist (no adapter can reach the global resolved model — proven), where
  the DOMAIN law fires (every gate double is `undeclared`, so the blast radius is
  measured and a proposal is on the record), and whether the horizontal law's
  names clause belongs in `emul_scalars`. Sharpest finding: **`float("n/a")`
  raises ValueError — the same class every refusal uses — so a leg that only asks
  "did it raise?" stays green through a broken law.**
- [Fixed-facts LANDING 1 audit: PASS + rulings 1+2 + the commit ruling](gates-and-board.md#fixed-facts-landing-1-audit-architect-2026-07-14-pass--every-cpu-gate-re-run-and-reproduced-three-unscripted-probes-fire-both-deviations-approved-rulings-12-issued-committed-on-the-branch) — the producer-sidecar/schema-v3/shared-reader landing audited PASS (2026-07-14): every CPU gate re-run and reproduced (selftest 209/0, new schema gate 34/0), both HEAD-reds proven pre-existing (one by the Architect's own scratch-layout method), three unscripted probes fire (order law, lazy digest, board-aid census); both deviations approved; RULING 1 ratifies the synthetic label contract + constraint-gated domain law, RULING 2 re-fixtures the duplicate-law arms and needles every adapter refusal; gated-but-unaudited landings stay uncommitted, the auditing turn commits on PASS; landings 2+3 dispatched 0124; three gate/staging defect lines entered the ledger. Follow-up (gates-and-board.md section "Mailbox 0120 adjudication"): the Implementer's stale-0110 return is adjudicated ACCEPTED — stale on arrival, conduct endorsed as the reference behavior, every ask pre-satisfied; the stale-dispatch/supersession transport class enters the ledger riding the tools-review repair unit; the 0121-to-fable/0121-to-sol pair is recorded as the fourth live next_seq() collision; the thread's mailbox loop is TERMINATED. Its routing copy (0130) then arrived stale in the Implementer's lane and was CLOSED there (section "The 0130 receipt closed on arrival") with no code touched — carrying ONE correction the audit turn needs: **that entry's "`git status` is clean" claim (:11800) was true at 08:06 and is deliberately false now.** Landings 2+3 are gated-but-unaudited, so :11702 leaves them uncommitted; the audit opens on a 15-file, +2,774/-85 working tree, and a "clean" reading is the one under which re-running a landed landing looks safe.
- [Fixed-facts LANDINGS 2+3 audit: PASS + rulings 3–5 + the adapter half dispatched](gates-and-board.md#fixed-facts-landings-23-audit-architect-2026-07-14-pass--every-gate-re-run-including-the-six-torch-gates-two-unscripted-probes-fire-the-question-1-factual-gap-closed-with-cobaya-source-evidence-rulings-35-issued-the-adapter-half-dispatched)
  — both landings audited PASS (2026-07-14) with EVERY gate re-run by the
  auditing turn, torch included for the first time (79/0, 209/0, 11/0, 5/1
  identical label; 24/40/78/69, readback 16/0, transfer 55/1 known); the 25M-06
  witnesses byte-verified independently; two unscripted probes fire (a
  vertical-law weakening reds its own leg; the :10954 amendment driven live on a
  stub predictor — bare row, 2-tuple hard case, permuted pair all refuse, pair
  == mapping bitwise). Unit 82 CLOSED. The question-1 factual gap is closed
  against installed cobaya 3.6.2 source: `Provider.__init__` stores the MODEL,
  so `self.provider.model` reaches it from any adapter. RULING 3: mechanism B —
  one duck-typed `resolved_constants(model)` extracted verbatim from the
  producer into fixed_facts.py, called at initialize_with_provider, once per
  artifact; mechanism A rejected (cobaya's plumbing error bypasses 74 cl.4).
  RULING 4: predict() enforces the domain law unconditionally, no public
  opt-out, O(n_param) accept path with one author of comparison + refusal;
  synthetic_sidecar grows support=None and SERVED doubles declare a box through
  format_value. RULING 5: the horizontal names clause stands (scalars included;
  subset training is not a schema-v3 capability); the adapters run topology
  laws BEFORE the appended horizontal law, so BOTH flagged arms stay honest
  with zero fixture churn (the mps relabel is REJECTED — one dump has one
  grid); the Part 3 alias-resolver clause is descoped on the record. Two audit
  findings ride the 0140 adapter-half dispatch as binding riders: the
  amendment's gate leg was never armed (proven live only by the audit probe),
  and the board prose overclaims "each law carries a mutation arm" (vertical
  has none — arm it). Two Architect errata owned: Part 5's nonexistent
  resolution site, Part 3's nonexistent alias resolver. Follow-up
  (gates-and-board.md section "The 0132 re-fire closed on arrival"): the audit
  turn was timeout-killed between its final commit and its outbound — the
  ledgered 0140-to-opus dispatch never entered the store; the re-fired 0132
  arrived stale, was closed against the committed record, and the closure turn
  wrote 0140 from the ruled sections unchanged (the lost-outbound sub-class is
  recorded as incident-note Live reproduction 5 and rides the tools-review
  repair). A SECOND 0132 re-fire (gates-and-board.md section "The SECOND 0132
  re-fire closed on arrival") proved archival fails after a CLEAN turn too and
  head-blocks the lane; 0132 was hand-archived to done/ to unwedge
  0133/0134/0136, watcher restart still user-owed.
