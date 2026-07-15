# How the user reads: didactics and the Python voice

Written 2026-07-12 by user order ("add a file on ai/notes/ based on
everything you learned how I like to be didactical and the way I code
in Python"). This is the voice-and-why companion to
conventions-and-workflow.md: that note is the compressed rulebook
(mechanics, environment, process); this one records who the reader
actually is, what she said when a rule was born, and the before/after
shapes that make the rules concrete. The Implementer reads this note
BEFORE writing any code or documentation; the Architect reads it
before writing specs, READMEs, and handoffs. On any mechanical
conflict, conventions-and-workflow.md and the role files win; this
note wins on tone and emphasis.

## Who is reading

- Vivian is a cosmology professor and mainly a C coder. Python beyond
  C-like constructs is genuinely hard for her to parse — her words:
  "remember, Python is hard for me as a human to parse". She reads
  code as prose, and error messages, argparse help, and log lines ARE
  prose to her. Anything printed is documentation.
- The README audience is cosmologists who are NOT AI experts. Every
  machine-learning term (MLP, dense layer, residual block,
  normalization, saturation, warm start) gets the two-sentence
  explanation a physicist needs, at the place it is used. Physics
  terms (chi2, covariance, xi, C_ell, P(k)) may be used freely.
- Nobody reads documentation linearly (user ruling, 2026-07-10).
  Readers jump into any section cold, so no sentence may lean on
  jargon defined three sections earlier. Every passage is
  self-contained or links to its definition in place.

## How she likes to be taught

1. **Run first, mechanism later.** The two-README split exists
   because of this: the main README teaches how to RUN and configure
   (run it -> the YAML chapter -> family sections -> generation);
   emulator/README.md teaches how the code works, and "every file's
   functions" comes last even there. New didactic material follows
   the same arc: get the reader to a working command before
   explaining internals.
2. **Show, never describe.** Every passage explaining a YAML concept
   carries a fenced snippet of the REAL block from a shipped file —
   prose-only descriptions were rejected repeatedly ("I dont see the
   file", "show me the block"). File formats appear as actual table
   snippets, equations verbatim from the code, and ASCII flow
   diagrams wherever data moves (she loves them). Every YAML change in
   a report is a paste-ready block in context, never a description of
   the edit. Every landing step is a complete copy-pasteable command
   block.
3. **Define or drop.** The founding failure: a gate-check header that
   followed every existing convention still bounced — "extremely hard
   to parse (what is a contract?)". One undefined term of art in the
   opening sentence costs the whole header. Either a term is defined
   where it first appears or it is replaced with plain words.
   "Whiten" is the canonical banned-unglossed verb: at every use site
   either gloss it ("rescale so every direction weighs equally") or
   use the plain phrasing outright.
4. **Plain sentences, visual structure.** No parenthetical that
   carries a full clause — it becomes its own sentence. Knob and
   option enumerations are tables (| knob | what it does |), never
   bullet runs. Long sections get subsection skeletons; one idea per
   paragraph. No all-caps emphasis, no " -- " double dash, anywhere
   prose reaches the user (docstrings, help text, errors included).
5. **No internal bookkeeping in documentation** (user ruling,
   2026-07-12: "I dont know what D-CM9, D-CM8 [are] ... this can be
   on ai/notes/ but not on documentation"). Design-decision codes live
   in ai/notes/ only; documentation states the fact the code stood for
   and may cite the note file. Full rule: conventions-and-workflow.md
   ("In-file documentation").
6. **Verdicts on the terminal, everything in the log.** A CLI prints
   headers, verdicts, one-line details, artifact paths; the full
   stream goes to a per-run log file; a debug flag restores the
   mirror. A failure message must name its own cause and, when it is
   a config mistake, name the fix ("needs amplitude_law: none" beats
   "invalid configuration").

## The way she codes Python (the voice)

The base register is C written in Python syntax: flat, explicit,
one step visible per line on any cold path. Cleverness is a cost, not
a virtue. Hot paths (forward passes, batch loops, vectorized
numpy/torch) are never slowed for readability — the fix there is a
better comment.

- **Explicit loops over comprehensions** on cold paths. Single
  ternaries are fine ("C has ?:"); ternary pileups, walrus, nested
  comprehensions, lambda-where-a-def-reads-better, and starred
  gymnastics are "Alien Python" and get rewritten.
- **Staged calls split into named temporaries.** Her words, on a
  nested tensor-staging chain: "that is hard to parse. Splitting into
  tmp variables would make it easier to read."

  Hard for her to parse:

  ```python
  Xtr = torch.from_numpy(rows[tr_idx].astype("float32")).to(device)
  ```

  The house shape — the intermediate gets a name:

  ```python
  tr_rows = rows[tr_idx].astype("float32")
  Xtr     = torch.from_numpy(tr_rows).to(device)
  ```

- **One key-value pair per line** in any dict literal of three or
  more pairs, paren-aligned. Her words: "Please one line per
  key-value pair."

  ```python
  knn = {"k":       8,
         "metric":  "euclidean",
         "weights": "distance"}
  ```

- **Named parameters everywhere** the callee allows — "I will forget
  the meaning of position X". The handful of irreducible positionals
  (matplotlib x/y, einsum operands, model(x)) carry a naming comment
  instead.
- **Formal docstrings**: prose module headers (subject + verb), an
  `Arguments:` block naming every parameter, `Returns:`, and a
  shape-flow diagram with a `(legend: ...)` defining every symbol for
  any tensor pipeline. A magic number appears only as a named-symbol
  derivation with the concrete LSST-Y1 example.
- **Never trust defaults across a save/load boundary.** Artifacts
  persist RESOLVED values (defaults materialized by the code that
  consumed them); loaders never fall back to code defaults; a caller
  never re-declares what the artifact already records. A file that is
  missing a required key fails loudly, naming the key.
- **Loud, specific errors.** Every admissibility check raises with
  the mismatch spelled out and the corrective action named. Silent
  coercion and silent fallback are defects even when convenient.
- **90 columns, paren alignment, one item per line**, spec dicts
  {cls, **kwargs} with make_X helpers, no silent module-global data
  reads. The mechanics live in conventions-and-workflow.md ("Python
  house style").

## How to use this note

Before writing code: skim "The way she codes Python" and write in that
register from the first line — the alien-Python and docs sweeps of
2026-07-10/12 exist because retrofitting voice is expensive. Before
writing user-facing text: skim "How she likes to be taught" and check
each passage against rules 2-4 (real snippet present, every term
glossed in place, no clause-bearing parentheses, tables for
enumerations). When she pushes back on a passage, the fix is almost
never more words — it is a snippet, a gloss, or a split sentence.
