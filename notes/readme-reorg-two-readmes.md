---
name: readme-reorg-two-readmes
description: "SPEC 2026-07-07 (Architect, HOLD until the YAML chapter lands): reader-first README reorganization into TWO files. User philosophy: 'first the user learns how to run and how to modify the YAML file of the code. Then later he starts understanding how the code is.' Main README: Run it FIRST (simplified; keeps the sweep: block and a SIMPLIFIED Multi-GPU section), then the YAML chapter (unchanged content, renumbered), then appendices (Pipeline DEMOTED to an appendix — it stays in the main file; chi2; activations; precedence), AI-Usage LAST and cut to the user's exact two-sentence text (screenshot + notebook paragraph deleted). NEW emulator/README.md (the code map): Layout, What each file does, Change X -> edit Y, Variants, every-file's-functions LAST — the standing last-appendix rule MOVES there; the main Contents links externally to it. SEQUENCING: strictly after the readme-yaml-chapter commit (user: 'this architecture plan goes after what the implementor is doing right now'). Gates GRO-A..E incl. move-accounting (nothing lost except the two authorized deletions). Handoff embedded, marked HOLD."
metadata:
  node_type: memory
  type: project
---

# README reorganization: two readers, two files (spec, HOLD)

User directives 2026-07-07 (second message + the sequencing
correction):

- "first the user learns how to run and how to modify the YAML file
  of the code. Then later he starts understanding how the code is
  (this should all be moved to appendices)" — and the code-map
  material moves to "another readme (one inside the emulator/
  folder)".
- SEQUENCING (the correction, binding): the YAML chapter
  ([[readme-yaml-chapter]]) is in flight and lands FIRST in the
  original README; "then we do the reorganization... after what the
  implementor is doing right now". This spec is ON HOLD until that
  unit is committed.

## Target: the main README (the how-to file)

    (alpha warning + the one-line description; + one line pointing
     to emulator/README.md, "the code map")
    ## Contents
    ## 1. Run it                  <- moves up from section 6
       (the bash examples + requirements + outputs; WITHOUT the
        gigantic YAML-tour paragraph — the YAML-chapter unit already
        shrinks it to a pointer, which now points two sections down)
       ### The sweep: block       <- STAYS (teaches run + YAML)
       ### Multi-GPU execution and packing
           <- STAYS but SIMPLIFIED ("a bit too technical and
           verbose"): keep the driver table, the one-line token rule
           (<= 20% -> 4/card, <= 40% -> 2, else exclusive), when to
           use / not use --gpu-pack in ~4 sentences, the journal
           paragraph compressed to ~3 lines. The deep VRAM-token
           mechanics live in scheduling.py docstrings (pointer).
    ## 2..11. The YAML chapter    <- as landed by readme-yaml-chapter
       (content untouched; renumbered 2-11: The YAML file, data,
        training globals, loss, optimizer+lr+scheduler, trim, focus,
        ema, model, two-phase + phase blocks)
    ## 12. Appendix: the pipeline <- current section 2, DEMOTED
       (the code-understanding walkthrough; content kept, moved
        whole — the user's "go down" list names it without sending
        it to the code-map file)
    ## 13. Appendix: the chi2 metric (Mahalanobis)
    ## 14. Appendix: activation functions
    ## 15. Appendix: precedence — who wins when settings collide
    ## 16. AI-Usage               <- ALWAYS LAST in the main README
       (verbatim replacement, the user's exact text, nothing else:)

       "AI Usage: This library (under the dev folder) was developed
       with Claude Code assistance. However, Prof. Miranda heavily
       influenced the code at every level, from macro-designed
       implementation and changes to minute Python choices."

       (the notebook link, the 1000-hours paragraph, and the
        screenshot image are DELETED — authorized.)

## Target: emulator/README.md (the code map, NEW file)

    # The emulator/ package: code map
    (one line: how to run + the YAML live in ../README.md)
    ## Contents
    ## 1. Layout                  <- moved from main section 1
    ## 2. What each file does     <- moved from main section 3
    ## 3. Change X -> edit Y      <- moved from main section 4
    ## 4. Variants                <- moved from main section 5
    ## 5. Every file's functions  <- moved; ALWAYS LAST here
    (backlink at the bottom)

STANDING RULE (amended): "every file's functions" is always the
LAST section of emulator/README.md; "AI-Usage" is always the LAST
section of the main README. Every future README edit honors both.

## The parallel/ deletion (user-approved 2026-07-07)

User: "lets delete parallel/ folder given that it is dead code
(double check that it is dead code)." DOUBLE-CHECKED by the
Architect, five legs, all conclusive at the YAML-chapter base:

1. Zero import-pattern references to the module anywhere in the
   tree (no `import`, `from .parallel`, importlib, registry entry).
2. Zero references to ParallelResCNN / GroupedCNNBlock in any .py
   or .yaml outside the folder itself.
3. emulator/__init__.py is a docstring only — the subpackage is
   reachable only by an explicit import nothing performs.
4. The full MODELS registry (all nine (name, ia) entries incl.
   tatt) has no parallel entry, so no YAML config past or present
   can name it — and results.py pickles no class paths (.emul is a
   plain state_dict), so no saved artifact can dangle.
5. ParallelResCNN's own docstring says it predates the current
   basis-buffer design (API-stale); its grouped-conv idea lives on
   as the live rescnn groups / separable knobs.

Scope addition to this unit: `git rm -r emulator/parallel/` plus
the FIVE prose mentions (enumerated, complete):

- emulator_designs.py:609 — the docstring parenthetical "A per-bin
  conv (parallel/)" -> "(a removed per-bin-conv variant; see git
  history)". Docstring-only: the AST-minus-docstrings check still
  shows zero code-node changes.
- README Contents entry (apx-parallel), the Layout line, the
  Variants table row, and the appendix parallel/ section — all in
  material this reorg moves anyway. The Variants row in the code
  map becomes the removal record: "per-bin CNN (parallel/): tested;
  the grouped conv was absorbed into rescnn's groups / separable
  knobs, the per-bin split lost to a single ResMLP; removed — see
  git history."

This is the THIRD authorized deletion in the move accounting, with
the Variants removal record as the surviving knowledge.

## The math-rendering fix (user bug report 2026-07-07, rides this unit)

GitHub renders the section-14 ema equation as raw LaTeX with
"Double subscripts: use braces to clarify". Root cause (Architect
diagnosis): the source escapes underscores (\mathrm{steps\_epoch}
style), but GitHub's markdown layer strips the backslash BEFORE
MathJax parses, so `steps_per_epoch` arrives with two live
subscript operators on one token = a hard error; single-underscore
code names (lr\_base, bs\_base, sqrt\_dchi2, berhu\_capped) survive
as WRONG renders (word-with-subscript) rather than errors.

POLICY (binding, gate-able): inside $$ math, NO code-name
identifiers carrying underscores, escaped or not. Code names live
in code spans in prose; math uses single-letter symbols with a
legend line beneath (house legend style). Legitimate single-letter
subscripts (w_i, c_i, beta_k, psi_p, p_min) are exempt — the
activation appendix renders fine and is untouched.

Concrete fixes (all in the YAML chapter, exact spots):

- Section 14 (THE error): the equation becomes
  $$\bar\theta \leftarrow \beta\,\bar\theta + (1-\beta)\,\theta
  \qquad \beta = 1 - \frac{1}{H S}$$
  with the legend: "$H$ = `horizon_epochs`, $S$ = steps per epoch."
- Section 11: the lr rule becomes symbols + legend, e.g.
  $$\mathrm{lr} = \ell\,\sqrt{B/B_0}$$ with "$\ell$ = `lr_base`,
  $B$ = `bs`, $B_0$ = `bs_base`" (exact symbols = Implementer
  latitude under the policy).
- Section 10: drop the mode-name subscripts from the display
  equations — the table + prose already name the modes in code
  spans (e.g. the capped line becomes prose "`berhu_capped` adds
  $\dfrac{2\sqrt{Kc}+k-K}{2\sqrt{k}}$ for $c > K$"; the
  sqrt_dchi2 label moves out of math likewise). Content identical,
  labels relocated.

Gate GRO-G: a scanner over every $$ block — zero `\_` and zero
multi-letter identifiers containing `_` (the single-letter-subscript
whitelist passes); the three fixed spots verified; every equation
still matches the code (the GYC-A equivalences unchanged — only
notation moved).

## Move accounting (the no-content-lost rule)

MOVED (verbatim, minus renumbering): Layout, What each file does,
Change X -> edit Y, Variants, every-file's-functions -> the code
map; Pipeline -> main-README appendix. STAYS: Run it (minus the
tour paragraph), sweep: block, Multi-GPU (compressed), the YAML
chapter, chi2 / activation / precedence appendices. DELETED (the
only authorized deletions): the AI-usage long form + screenshot;
the Multi-GPU prose depth being compressed (its facts must survive
in the compressed form or in scheduling.py docstrings — state which
in the report). Anything else missing = a gate failure.

## Cross-links

- Main Contents: an external entry "Code map: what each file does
  (emulator/README.md)" linking `emulator/README.md` (relative link;
  GitHub renders it).
- Every intra-README anchor that crossed the split becomes a
  cross-file link (e.g. the pipeline appendix's references to file
  roles -> `emulator/README.md#...`); the chi2-appendix anchors
  referenced from the YAML loss section stay in-file.
- emulator/README.md backlinks to `../README.md` top and bottom.

## Gates

- GRO-A (structure): both files match the targets above, order
  exact; AI-usage last in main, every-file's-functions last in the
  code map; Contents of both resolve (the slug checker, em-dash
  slugs; PLUS the cross-file links resolve as paths + anchors).
- GRO-B (move accounting): every section named MOVED exists at its
  new home with its content (diff-based: headers + spot lines);
  nothing outside the two authorized deletions disappears (a
  moved-lines audit over git diff).
- GRO-C (AI-usage): the section is exactly the user's two sentences
  (verbatim string check), nothing else, positioned last.
- GRO-D (simplifications): the Multi-GPU section fits the budget
  (driver table + ~10 lines of prose + the one-line token rule);
  Run it carries no YAML-block description beyond the pointer.
- GRO-E: no code-node changes (git diff = the two READMEs + notes +
  the parallel/ deletion + ONE docstring line; AST-minus-docstrings
  over every remaining .py = CODE-IDENTICAL); house scans;
  py_compile regardless.
- GRO-F (deletion completeness): emulator/parallel/ gone; tree-wide
  grep ParallelResCNN / GroupedCNNBlock empty; "parallel/" survives
  only in the Variants removal record (+ notes/, which keep
  history); `import emulator` + py_compile still clean.

## Handoff (HOLD — relay only after the YAML chapter commit)

### ARCHITECT_HANDOFF
Task: the two-README reorganization (spec:
notes/readme-reorg-two-readmes.md, read in full). Base: the
readme-yaml-chapter commit; `git log -1` must show it — else STOP.
Scope: the main-README reordering (Run it first, the YAML chapter
2..11, Pipeline demoted to appendix 12, appendices 13..15,
AI-Usage last with the verbatim two-sentence replacement); the NEW
emulator/README.md code map (Layout, What each file does, Change X
-> edit Y, Variants, every-file's-functions LAST); the Multi-GPU
simplification within its budget; the cross-links both ways; the
move accounting (three authorized deletions, nothing else lost);
PLUS the user-approved `git rm -r emulator/parallel/` with its five
prose-mention cleanups (the section above — the only .py touch is
one docstring line; AST code-identity must hold on every remaining
file); PLUS the math-rendering fix (the section above: the ema
double-subscript error + the underscore-identifier policy over all
$$ blocks, symbols + legends, content identical). Gates GRO-A..G.
Report: IMPLEMENTER_HANDOFF + resume state appended to this note,
raw gate evidence (incl. the move-accounting audit), deviations
declared. Do not commit: print the suggested commit command.
### END

## Status

SPEC DELIVERED 2026-07-07, ON HOLD behind [[readme-yaml-chapter]]
(in flight with the Implementer). Relay order: YAML chapter report
-> Architect re-audit -> user commits -> THIS handoff. Suggested
commit sentence: "Reader-first README split: Run it + the YAML
chapter lead the main README (Pipeline demoted to an appendix,
Multi-GPU compressed, AI-Usage last and simplified); the code map
(Layout / file roles / Change X -> edit Y / Variants /
every-file's-functions) moves to emulator/README.md (gates GRO-A..E
Architect-verified)".

## Implementer resume state (2026-07-07, Opus, base fc4655a)

IMPLEMENTED. Both files assembled by a deterministic slice-and-relocate
script (verbatim blocks sliced from the committed fc4655a so moved
sections stay byte-identical; only Run it, the loss/opt/ema math, the
Variants removal record, the Layout line, AI-Usage, and the intro /
Contents / cross-links are rewritten). Uncommitted; the user commits.

Files:
- `README.md` — reordered: 1 Run it (tour dropped to a pointer,
  Multi-GPU compressed to 12 prose lines), 2-11 the YAML chapter, 12
  Appendix: the pipeline (demoted), 13 chi2, 14 activations, 15
  precedence, 16 AI-Usage (verbatim two sentences, last). 1114 lines.
- `emulator/README.md` — NEW code map: 1 Layout, 2 What each file does,
  3 Change X -> edit Y, 4 Variants (parallel/ row -> removal record),
  5 Every file's functions (LAST; parallel subsection dropped;
  Mahalanobis link -> `../README.md#13-...`). Backlinks top + bottom.
  364 lines.
- `git rm -r emulator/parallel/` staged (3 files); disk residue cleared.
- Docstring-only parallel/ cleanups in `emulator_designs.py` (x2:
  module docstring + the ResTRF forward-graph parenthetical) and
  `emulator_designs_building_blocks.py` (x1). AST-minus-docstrings
  proves all three files code-identical.

Math-render fix (GRO-G): section 5 (loss) drops the mode-name
subscripts (labels -> code spans, `L(c)` display); section 6 (opt)
`\mathrm{lr}=\ell\sqrt{B/B_0}` + legend; section 9 (ema)
`\beta=1-1/(HS)` + legend (was the `steps\_per\_epoch` double-subscript
error). Scanner: 13 $$ blocks in main, 0 in the map, zero `\_` and zero
code-name underscore identifiers; the activation appendix untouched
(single-letter subscripts exempt).

Gates (all green, raw output in the handoff):
- GRO-A structure + anchors: 36 main links + 23 map links resolve;
  headers 1..16 / 1..5 in order; cross-file links resolve (path +
  anchor); every-file's-functions last in the map, AI-Usage last in main.
- GRO-B move accounting: 11 verbatim-moved sections byte-identical
  (pipeline, chi2, activation, precedence, data, trim, focus, model,
  two-phase, what-each, change-x); edited sections spot-verified.
- GRO-C AI-usage: verbatim two-sentence string match; screenshot +
  1000-hours + notebook long-form deleted; positioned last.
- GRO-D simplifications: Multi-GPU 12 prose lines (driver table + the
  one-line token rule kept; 2 GiB / lock / MPS depth -> scheduling.py);
  Run it carries only the YAML pointer.
- GRO-E: AST-minus-docstrings identical on both touched .py; diff scope
  = the two READMEs + the two docstring .py + the parallel/ deletion +
  notes, nothing else.
- GRO-F: emulator/parallel/ gone; ParallelResCNN / GroupedCNNBlock only
  in notes/; "parallel/" only in notes/ + the Variants removal record;
  py_compile clean on the whole package + drivers.
- GRO-G: math scanner green (above).

Deviations declared (see the handoff):
1. The spec's "five prose mentions (enumerated, complete)" missed TWO
   .py docstring mentions of parallel/ (emulator_designs.py:33 module
   docstring, emulator_designs_building_blocks.py:24). Cleaned all THREE
   docstring lines (not one) so no dangling parallel/ survives and GRO-F
   passes; GRO-E's "ONE docstring line" is therefore three docstring
   lines across two files, all AST code-identical.
2. AI-Usage keeps `dev` in a code span (matching the original + house
   style); the sentence text is verbatim (GRO-C strips markdown).
3. The `6a-sweep-block` / `6a-multi-gpu` explicit anchors renamed to
   `sweep-block` / `multi-gpu` (the "6a" referred to the old section 6);
   all references updated, anchors resolve.
4. Multi-GPU compression: the spawn -> per-child CUDA/cosmolike-state
   detail compressed to the word "spawned"; the NVWULF `nvidia-cuda-mps
   -control` operational tip dropped (not a code fact). All other depth
   (2 GiB overhead, the deadlock-free per-GPU lock, the token rule)
   lives in scheduling.py docstrings/comments; verified.

Awaiting Architect re-audit.

### 2026-07-07 — Architect re-audit: ACCEPTED, no deltas — and one
### spec error OWNED

Own probes on the confirmed base fc4655a:

- GRO-A: both files' headers exact (main 1..16, map 1..5); my slug
  checker over BOTH files including the cross-file links: all
  resolve. GRO-B: pipeline / chi2 / what-each / change-x
  content-verified at their new homes (line-level presence);
  structure lists confirm nothing else moved wrong. GRO-C: the
  AI-Usage section is the verbatim two sentences and NOTHING else,
  last (my first probe false-flagged on line wrapping — normalized,
  it is exact; long-form + screenshot + notebook all gone). GRO-D:
  Run it read in full (pointer + compressed Multi-GPU; anchors
  renamed sweep-block / multi-gpu, all references resolve); the
  dropped depth verified present in scheduling.py (28 matches).
  GRO-E: AST-minus-docstrings CODE-IDENTICAL on both touched .py
  (and experiment/training untouched). GRO-F: parallel/ gone; class
  names only in notes/; "parallel/" survives only in the Variants
  removal record; py_compile whole package + drivers OK. GRO-G: the
  math scanner over both files — only single-letter subscripts
  remain (mu_k / beta_k / psi_p, the exempt whitelist); zero code
  name underscores.

Deviations RULED:
1. The three-vs-one docstring cleanup: ACCEPTED — and the
  underlying spec error is MINE. My "enumerated, complete" list of
  five mentions was produced by a grep whose own exclusion filter
  ("in parallel", meant for prose about parallel workers) swallowed
  "live in parallel/." at emulator_designs.py:33 and
  building_blocks:24. Both verified present at HEAD. Second
  harness-filter lesson this cycle (the slugger's em-dash was the
  first): exclusion patterns can eat true positives — prefer
  positive matching + manual review of the excluded set. The
  Implementer's catch and fix were exactly right.
2. The `dev` code span in AI-Usage: ACCEPTED (the original had it;
  the text is verbatim under normalization).
3. The 6a -> sweep-block / multi-gpu anchor rename: ACCEPTED (the
  "6a" named a retired section number; all references updated and
  resolving).
4. The Multi-GPU compression losses: ACCEPTED (spawn detail
  compressed to "spawned"; the NVWULF MPS tip was operational
  advice, not a code fact — recorded here as the place to find it:
  enabling nvidia-cuda-mps-control tightens co-located
  time-slicing; the flag works without it).

COMMIT-READY. Suggested sentence: "Reader-first README split: Run
it + the YAML chapter lead the main README (Pipeline demoted to an
appendix, Multi-GPU compressed, AI-Usage last + simplified, the
GitHub math-render fix); the code map (Layout / file roles /
Change X -> edit Y / Variants / every-file's-functions) moves to
emulator/README.md; dead emulator/parallel/ removed (gates GRO-A..G
Architect-verified)". This closes the README arc.
