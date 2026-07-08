---
name: gates-checks-docs-plain-language
description: "USER FEEDBACK 2026-07-08, work DEFERRED until the board is green: the gates/checks/*.py docstrings are 'extremely hard to parse' — way too much jargon ('what is a contract?'), and main() has a one-line docstring where it 'should have good documentation'. The gsv header even follows the PS-jargon-defs convention and STILL failed the user: it defined bitwise/drift proof/v1 refusal but not 'contract', and the density/register is the real problem, not missing definitions. Rule: every term of art is defined in place or dropped; docstrings read as plain English a human parses on first read; main() gets a substantive docstring. Sweep ALL of gates/checks/ (gsv_bitwise_drift, gct_parity, gb_c_berhu_reduce, ge_c_eval_bs, gwd_census) after the board is green."
metadata:
  node_type: memory
  type: feedback
---

User feedback (2026-07-08, verbatim complaints): the `gsv_bitwise_drift.py`
header is "extremely hard to parse (what is a contract?)"; "main function
should have good documentation"; "there is way too much jargon on files
inside gates/checks/". Explicitly deferred: "do that later after we fixed
all tests".

**Why the existing convention didn't save us.** The gsv header follows
[[py-module-style-conventions]] — prose WHAT/WHY/HOW, a PS defining
bitwise / drift proof / v1 refusal — and the user still bounced off it.
Two failures:

1. The PS defined the wrong terms: "contract" (the word the user asked
   about) appears in the first line and is never defined. A jargon
   glossary only works if it is complete; one undefined term of art in
   the opening sentence costs the whole header.
2. Density/register: the header compresses spec codes, note line-ranges,
   and design rationale into sentences a maintainer can decode but a
   human cannot parse on first read. The check scripts are the files a
   person opens when a gate FAILS — they must read as plain English
   under stress, not as spec shorthand.

**The rule (applies to all of `gates/checks/`):**

- Every term of art is either defined where it first appears or replaced
  with plain words ("contract" -> "the promise that a reloaded emulator
  behaves exactly like the one that was trained").
- Docstrings are sentences a first-time reader parses once; spec codes
  and note line-ranges live in ONE header line, not woven through prose.
- `main()` (and every function) gets a substantive docstring: what it
  runs, in what order, and what a failure looks like — not one line.

**Scope:** `gsv_bitwise_drift.py`, `gct_parity.py`, `gb_c_berhu_reduce.py`,
`ge_c_eval_bs.py`, `gwd_census.py` (the whole `gates/checks/` folder).

**Status:** DEFERRED — queued behind the board going green (cobaya-adapter
is the last pending gate as of run 7). Do not start this sweep while gate
fixes are still landing; it would churn the same files.
