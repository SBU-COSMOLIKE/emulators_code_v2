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
- GRO-E: documentation-only (git diff = the two READMEs + notes;
  zero code / yaml); house scans; py_compile regardless.

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
move accounting (only the two authorized deletions).
DOCUMENTATION-ONLY: zero .py / .yaml changes. Gates GRO-A..E.
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
