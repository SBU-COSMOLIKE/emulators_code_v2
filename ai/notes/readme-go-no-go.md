# README and Python prose GO / NO-GO contract for the Architect

## Terms used by this contract

A **unit** is one bounded change assigned to one development role. Git is the
repository's version-control system; a **tracked** file is a file Git includes
in saved repository versions. A **Python symbol** is one named module,
function, class, or method.

A **protocol token** is an exact machine-read word whose spelling is part of
an interface. **Serialized data** is text or bytes written so a program can
reconstruct structured state. A **test fixture** is the fixed input setup used
by a check.

A **directive** is the Architect's complete instruction for one unit. Its
**Acceptance checklist** lists the checks and observations required for GO.
An **exemption** records why one check does not apply. **Dispatch** is the act
of sending that directive to another role. A **gate** is a named validation
job whose required result is written before it starts.

YAML is the human-readable settings-file format used by the repository. A
**parser** is a program that reads a format and rejects malformed input.
**Mermaid** is the text format GitHub renders as a diagram. An **anchor** is a
named location that a link can target. **Output parity** means that actual
program output matches the declared reference under the stated comparison.

**FAQ** means frequently asked question. An **HTML detail block** is a
collapsible `<details>` section in Markdown. **Stale** text no longer matches
the current library. A **full-source search** examines every file in its named
scope rather than a shortened sample. A Git **branch** is a named line of saved
repository versions. A **worktree** is a separate Git working folder attached
to one branch.

This contract applies whenever a unit creates or changes either:

- a tracked README; or
- a tracked long-form explanation under `documentation/`; or
- explanatory prose inside Python: comments, docstrings, command help,
  user-facing diagnostics, and strings whose purpose is to explain behavior.

For a README, it covers the main guide, appendices, tables, diagrams,
captions, command examples, and exact program output quoted in the guide. For
Python, it covers the explanatory words, not protocol tokens, serialized data,
test fixtures, or strings that have no teaching or diagnostic purpose.

For a long-form document, the plain-language, current-state, factual-source,
real-example, and rendered-visual rules apply. README-only requirements such
as a main-guide table of contents do not apply mechanically to a TeX guide.

The target reader is a physics undergraduate who knows no AI-agent language
and may know little Git. A reader may open any section directly instead of
reading the file from the beginning. Each section must therefore explain the
terms it needs at the place where it uses them.

Architect-authored permanent-note prose follows this contract's local
definition, repository-example, neutral-audience, coherent-current-system, and
anti-AI requirements. README-only structure and visual rows do not apply to a
permanent topic note. This writing rule does not give the Implementer or Red
Team permission to edit any permanent note.

Git commit messages created by the AI-development workflow are reader-facing
Markdown on GitHub. They follow every applicable prose rule in this contract,
including plain local definitions, examples before broad ideas, short
paragraphs, one stable name for each object, current-state wording, a neutral
audience, and the anti-AI checks. README tables of contents, diagrams, links,
and page-layout checks do not apply. This file owns the exact writing format.
`ai/notes/conventions-and-workflow.md`, section **Commit messages explain the
saved change**, owns how that approved message is preserved during landing.

Only the Architect issues `GO` or `NO-GO`. The Implementer supplies the
change and its evidence. The Red Team may identify a problem and propose a
repair. Neither role replaces the Architect's final review.

## Describe one coherent current system

README files, long-form documentation, permanent notes, commit explanations,
and explanatory Python prose describe how the library works now. When a rule
changes, rewrite the owning explanation in place. Do not append a dated
correction or leave the old rule beside the new one.

The reader needs the current rule, its reason, one concrete example when the
reason is broad, and any present limitation. The reader does not need to know
which request, review, model, ticket, or development session introduced it.
Git keeps earlier tracked wording. The local backlog keeps unfinished work and
temporary review history.

These forms receive `NO-GO` when they narrate policy development:

- a calendar date or timestamp attached to a rule;
- `hard user rule`, `the user requested`, or an attributed personal
  preference;
- `new rule`, `the rule now says`, `formerly`, `previously`, `after the last
  review`, or `as of`;
- ticket numbers, audit waves, review rounds, rollout phases, model names, or
  commit identifiers used to explain why prose exists; and
- a chronological addendum that corrects an earlier paragraph instead of
  replacing it.

For example, this is `NO-GO`:

```text
Hard user rule, YYYY-MM-DD: the watcher now refuses a force push.
```

Write the current rule directly:

```text
The watcher refuses every force push because replacing saved Git history can
discard accepted work.
```

Dates and chronological words are allowed only when time is part of the
subject itself. Examples include a scientific data release named by year, a
publication citation, a user input that is a date, or an algorithm whose
ordered phases are current behavior. The Architect records why each such use
is necessary. A training phase is not policy-patch history; `phase added in
review round 3` is.

`GO` requires one consistent current explanation after the edit. A reader must
not encounter an old rule, a later correction, and a third paragraph that
reconciles them. The Architect searches the complete changed file and related
owner sections for policy dates, `hard user rule`, ticket or review labels,
and chronology phrases, then reads every match in context. A blind word ban is
not sufficient because words such as `history` and `phase` also have valid
technical meanings.

## The Architect reads this file twice

### Before writing the implementation directive

1. Read the full README or long-form-document section, or the complete Python
   symbol, that will change. For a README, also read its table-of-contents
   entry and the paragraphs immediately before and after it. For a long-form
   document, read its catalog entry and the complete neighboring sections.
2. Read the code, shipped configuration, live help, or other current source
   that proves every behavior the covered prose will describe.
3. Decide whether the material belongs in the short README path, an appendix,
   one existing long-form guide, or a new focused guide. For Python prose,
   decide whether the explanation belongs beside the code, in a docstring, in
   command help, or in the README instead.
4. Copy every applicable check in this file into the directive's
   `Acceptance checklist`. Each check must name the evidence the Implementer
   will return.
5. Mark a check not applicable only with a concrete reason. An omitted check,
   an unexplained exemption, or a choice left to the Implementer is `NO-GO`
   for dispatch.

### Before reviewing the final change

1. Reopen this file. Do not rely on memory or on the earlier directive.
2. Read the final rendered README section, every rendered page of a long-form
   document, or the complete Python symbol, not only the changed lines.
3. Re-run the factual, command, link, and rendering checks that apply.
4. Treat the Implementer's checked boxes as evidence to inspect. They are not
   the verdict.
5. Issue `GO` only when every applicable row below is supported by raw
   evidence. Otherwise issue `NO-GO` and name the exact repair.

## Decide where the material belongs

The main path gets a new user to a valid result. It answers, in order:

1. What does the user need?
2. Which command or setting does the user change?
3. Which result should appear?
4. Which action follows that result?

Theory, implementation detail, recovery internals, long explanations, and
reference material belong in appendices or one linked specialist guide.
Moving material out of the main path does not permit harder language. The same
reader standard applies everywhere.

For `README.md` and `ai/README.md`, the table of contents must visibly
separate the short main guide from **Common questions raised by developers**.
Appendices are grouped by topic and use real questions. The main guide stays
short enough that a new user does not have to read the appendices before the
first successful run.

### Use `ai/README.md` as the positive structural exemplar

For every tracked README unit, the Architect reads `ai/README.md` at the
unit's pinned starting commit. Read its opening through the first complete
worked example, its table of contents, and one complete FAQ appendix. If the
unit changes `ai/README.md` itself, compare the candidate with that accepted
starting version. This file is an example of teaching structure, not a factual
source for another package; current code and shipped files still prove
behavior.

Adapt these traits to the README's subject:

- Start with the smallest useful mental model. Add detail in later passes, and
  introduce a new term only when the next action needs it.
- Show a real file, command, setting, or visible result before asking the reader
  to retain a broad name or rule. Then state the general rule the example
  demonstrates.
- Keep the route to the first useful result in short main sections with one job
  each.
- When deeper explanation or recovery detail needs an appendix, group it under
  useful FAQ questions. Apply the same plain words and local definitions
  there.
- Draw sequences from top to bottom with short labels, and inspect them at a
  phone-sized width.
- Keep one stable name for each object. A glossary may support later lookup,
  but it may not be required reading before the first action.
- When a tool exists to solve a constraint that its name does not reveal, open
  with that concrete constraint, when the tool helps, and when it is
  unnecessary. Do this before introducing internal roles, folders, or options.
  This is an explanation, not a broad claim that the tool is important.
- Carry one small representative task through its first complete result. For
  each command, state where it runs, whether it changes files, the visible
  successful or refused result, and the next action. Do not switch to an
  unrelated example before the first task finishes.
- Teach the required route before optional roles, modes, and recovery paths.
  Introduce an optional choice where the reader makes it, and state both what
  the choice changes and what remains required.
- Keep the main README focused on the shortest safe route. Link to a specialist
  README or long-form guide for long command references or recovery detail
  instead of duplicating them, but give enough context that the link text and
  destination make sense.

`GO` requires the directive and final review to name the first useful result,
the order in which new terms appear, the real examples used for new
abstractions, and the main-guide or appendix placement decision. Include
narrow-screen evidence when a sequence diagram is useful. When no appendix or
sequence diagram is needed, record that reason instead of manufacturing one.
`GO` also requires the directive and final review to name the concrete problem
the tool solves, when it helps, whether it is optional, the one task carried
through its first complete result, and the step where each optional choice
becomes relevant.
`NO-GO` applies when the opening begins with a vocabulary list or complete
internal topology, postpones definitions to a glossary, places several actions
in one section, uses harder language in an appendix, or requires horizontal
scrolling to follow a sequence.

Small package READMEs do not need artificial appendices. They still put the
first useful command or code example before internal design detail.

A specialist folder README opens by naming the folder's complete current
scope. It does not describe a multi-family tool as belonging to one emulator
family, and it does not present a hand-maintained partial list as the complete
inventory. When the program can print its live inventory, the README gives
that command and explains one visible result.

When two nearby folders or tools have similar names, the README explains the
difference before introducing internal vocabulary. It gives one real example
from each side, including the input or command, the action, and the visible
result. One object keeps one name in teaching prose. A code identifier with a
different name is explained once in the command reference; wording such as
`test (gate in the code)` receives `NO-GO` because it teaches two names before
the difference is clear.

An operational paragraph has one job. Readiness checks, choosing work,
continuing after an interruption, reading results, and forcing a rerun belong
in separate paragraphs or headings, each with its own example when the action
is not obvious. A dense paragraph that mixes these actions receives `NO-GO`
even when every sentence is factually correct.

### Use one long-form guide for one deep question

Before creating a long-form document, the Architect searches
`documentation/README.md`, the tracked files under `documentation/`, relevant
README sections, and source names or synonyms for the requested topic. The
directive records the possible owner documents that were opened. If one
already answers the reader's question, update it or improve the link to it.
Creating a second guide for the same question is `NO-GO`.

A README gives enough context for a reader to recognize the question and then
links to the one long-form guide that owns the detailed answer. Copying the
complete explanation into both places is `NO-GO` because the two copies can
drift. `documentation/candidate_to_landing.tex` is the positive example for
one focused mechanism. `documentation/emulator_code_guide.tex` is the
positive example for a complete library manual; a focused request must not
expand into another manual of that size.

The long-form directive names the exact reader question, audience, scope,
source files and symbols, existing-document census, worked example, important
failure path, README link, source and compiled output, build command, and
page-render command. Final `GO` requires page-by-page visual inspection and a
fresh comparison of every implementation claim with current code.

`conventions-and-workflow.md`, section **Feature-specific long-form
documentation**, owns ticket priority, role ownership, and the complete
search-first workflow.

Python comments explain a non-obvious reason, invariant, unit, shape, failure,
or compatibility rule. They do not narrate the next line of code. Docstrings
state the callable's real inputs, outputs, shapes, units, side effects, and
errors. Command help and diagnostics tell the user what happened and what to
do next. If an explanation teaches a general workflow rather than one nearby
symbol, put it in the README and keep only a short pointer in Python.

## Commit messages use the same reader standard

A commit message is the short explanation shown beside a saved Git change on
GitHub. A reader should not need the backlog, a role transcript, or the file
diff to learn why the change exists. The Architect reviews the complete
message as prose, not only the first line.

Every AI-authored commit uses one plain-language subject followed by these
four Markdown headings in this order:

```markdown
Concrete subject that names the saved behavior

## Why this change was needed

[The observable problem, with an unfamiliar term defined where it appears.]

## What this commit changes

[The saved behavior, introduced through a concrete repository example.]

## What remains unchanged

[The behavior this commit does not change or support.]

## Checks run

- `exact command or manual check` — [visible result]
```

The subject names the saved behavior, not the development process. Internal
ticket numbers, dates, role names, branch names, and generic verbs such as
`Update`, `Improve`, or `Fix issue` receive `NO-GO`. Each body section uses
short paragraphs or bullets. A check names the exact command or manual check
and the result a reviewer saw; `tests passed` without both receives `NO-GO`.

One commit should explain one specific change. Independent changes belong in
separate commits. When two edits must land together to preserve one behavior,
the first body section explains that relationship in plain language.

For example, a commit that prevents failed physics rows from reaching model
training can use this message:

```markdown
Keep failed generator rows out of training data

## Why this change was needed

A failed physics calculation could leave a placeholder row. The training
loader could treat that row as a valid sample.

## What this commit changes

The loader reads the saved list of failed row numbers and removes each named
row before choosing training samples.

## What remains unchanged

Tables created by the scalar-data program keep their existing behavior.

## Checks run

- `conda run -n cosmology python -m unittest ai.tests.test_failed_row_staging`
  — 8 tests ran and the final result was `OK`.
```

`GO` requires the Architect to read the exact subject and body as Markdown,
confirm all four headings and their order, and apply the same vocabulary,
examples, paragraph-length, neutral-audience, and anti-AI rows used for a
README. Recovery lines added by the mailbox program may follow the four
sections. They do not replace or interrupt them.

## Permanent notes are outside every implementation and Red Team unit

The Implementer and Red Team never edit the eleven permanent notes, whatever
the ticket type. Only the Architect may update one, and that is a separate,
explicit policy step. Every directive sent to an Implementer or Red Team puts
all eleven note paths and `ai/tools/permanent_note_guard.py` under
`Do not change`.

```text
ai/notes/MEMORY.md
ai/notes/artifacts-inference-warmstart.md
ai/notes/conventions-and-workflow.md
ai/notes/data-generation-and-cuts.md
ai/notes/families-background-mps.md
ai/notes/families-scalar-cmb.md
ai/notes/models-and-designs.md
ai/notes/project-and-history.md
ai/notes/readme-go-no-go.md
ai/notes/training-stack.md
ai/notes/python-changes-go-no-go.md
ai/tools/permanent_note_guard.py
```

SHA-256 is a fixed-length fingerprint calculated from exact file bytes. A Git
commit is one saved repository snapshot. A Git worktree is a separate folder
that checks out one branch without changing another working folder. `HEAD` is
the commit selected in that worktree. The staging area contains changes chosen
for the next commit, while working files are the files presently on disk.

The expected SHA-256 values do not live in an editable checksum file. The
Architect pins the full starting commit in the directive. Before dispatch and
again before any final `GO`, the Architect runs:

```bash
python3 ai/tools/permanent_note_guard.py \
  --repo EXACT_WORKTREE \
  --base FULL_STARTING_COMMIT
```

The command calculates the expected bytes from that Git commit and compares
them with current `HEAD`, the Git staging area, and the working files. The
Architect reruns it; an Implementer's pasted result is not the final check.
Any mismatch is `NO-GO` for the implementation unit. An intentional note
update belongs to the Architect and becomes a future unit's new starting
commit after it is reviewed and committed.

## Review 1: the instruction is complete

The Architect answers these questions before dispatch.

| Check | `GO` for the directive | `NO-GO` for the directive |
| --- | --- | --- |
| Reader outcome | Names the exact question the changed section will answer and the next action the reader can take | Says only “improve,” “clarify,” “document,” or another subjective goal |
| Placement | Names the README location or the exact Python comment, docstring, help, or diagnostic location and explains why it belongs there | Leaves placement, heading order, comment location, or docstring scope to the Implementer |
| Source of truth | Names the exact code, shipped file, command, or current policy that supports each behavior | Relies on memory, chat, an old README statement, or a temporary audit note |
| Coherent current account | Names the superseded passage that will be replaced and the one current rule that will remain; lists any necessary scientific date or real algorithmic phase | Adds a correction, date, “hard user rule,” ticket reference, review round, or policy-patch paragraph beside older prose |
| Vocabulary | Lists the non-physics technical words the passage needs and gives a plain definition or replacement for each | Assumes that a later section, glossary, or software background will rescue an undefined term |
| Abstraction examples | Identifies every new broad idea and supplies one or two real repository examples that will make the idea concrete | Says only “add an example,” leaves the Implementer to invent one, or permits a broad title or definition with no real case |
| Neighbor distinction | For similarly named folders or tools, supplies one real example from each and states the visible difference | Defines the names with circular wording or says one is the other “in code” |
| Edit plan | Names every file, heading, paragraph, table, diagram, link, and exact-output block that may change | Gives a broad file-level request and asks the Implementer to choose the structure |
| Examples | Supplies or identifies a real copy-paste example, where to run it, whether it changes files, and the visible successful result | Requests an illustrative example that need not pass the real parser or command |
| Visual choice | For a README, states which sequence, branch, comparison, or feedback loop a diagram will clarify; otherwise records `not applicable` for Python-only prose | Requests “more graphs” without naming the relationship each graph must show, or omits the applicability decision |
| Tests | Gives exact validation commands and expected results for links, examples, output parity, rendering, Python syntax, help, diagnostics, and docstring behavior that the change affects | Leaves the Implementer to invent the acceptance tests |
| Boundaries | Names facts, safety rules, commands, and unrelated sections that must not change | Allows a clarity edit to silently change behavior or widen scope |

An Implementer may be Sonnet, Haiku, or an open-source model. The directive
must be executable without retained chat, hidden design choices, or “use your
best judgment.”

## Review 2: the reader can use the result

| Check | `GO` | `NO-GO` |
| --- | --- | --- |
| Direct answer | A README opens with the heading's answer or action; Python prose starts with the callable's action, the comment's reason, or the diagnostic's problem | The passage begins with background, history, or an abstract system description |
| Cold-reader test | A physics undergraduate can restate the passage and identify the next action without opening another section | Understanding depends on software-engineering or AI knowledge that the passage never supplies |
| Local definitions | Every unfamiliar term is replaced or defined in concrete words at its first use in that section | A definition uses another undefined term or points only to a distant definition |
| Examples for abstractions | A broad idea is followed or introduced by one or two real cases that name an input or file, the action, and the visible result | The reader receives only a broad label, a circular definition, or a toy example when a current repository case exists |
| Specialist scope | The opening names every major family or job the folder currently serves, or points to a live inventory command | The opening narrows the tool to one family or relies on a stale partial list |
| Stable names | One object keeps one teaching name; a different code identifier is introduced only where the reader needs it | The prose rotates among synonyms or teaches `X (called Y in code)` before explaining the boundary |
| One job per paragraph | Each paragraph explains one action, fact, warning, or consequence | One paragraph mixes setup, mechanism, exception, and recovery |
| Manageable length | Ordinary prose is split before it becomes a wall of text; a paragraph over four sentences or about 100 words has a recorded reason to remain whole | Long prose is retained only because it is technically correct or appears in an appendix |
| Parentheses | Parentheses contain a short definition, symbol, unit, or acronym | Removing a parenthetical would remove an essential rule or a second argument |
| Neutral audience | Prose uses roles such as **user**, **Architect**, or **maintainer** only when the role matters | Prose encodes a named person's preferences, pronouns, ownership, or development diary |
| Current state | The README or Python prose says what the library or symbol does now and what the reader should do now | It narrates migration history, ticket history, rejected designs, intermediate commits, or old paths |
| Coherent system | Related sections tell one compatible current story and place each rule in one owner location | The reader must reconstruct the rule from dated patches, successive corrections, duplicated policy statements, or “hard user rule” labels |
| Complete command | The reader knows prerequisites, working folder, command, expected result, and whether the command changes files; command help names required values and defaults | A command appears without enough context to run or interpret it safely |
| Current limitation | A limitation states its present scope, consequence, and user action | It explains when the limitation was found or promises an unimplemented repair |

Use the **novice paraphrase test** for every definition: ask whether the target
reader could say the same fact in ordinary words. If not, the definition is
still too technical.

### Use real examples to explain abstractions

An **abstraction** is a broad name or rule that covers several concrete cases.
Terms such as **test**, **gate**, **publication**, **identity**, and **code
change** are abstractions when the surrounding text does not yet show what the
reader can see or do.

For a README, a definition alone is not enough. The Architect gives `GO` only
when each unfamiliar abstraction has one or two nearby examples from the
current repository. Each example names:

1. a concrete input, filename, setting, command, or program state;
2. the exact action performed; and
3. the result the user can observe.

For example, “a test checks one behavior” is still broad. A concrete example
can say that one test gives a parameter table a single row and confirms that
the training program still receives a two-dimensional table. A second example
can say that another test damages a saved progress file in a temporary folder
and confirms that loading stops without changing the original files. Those
cases let the reader understand the general word **test**.

A broad README title also follows this rule. “Checks for code changes” receives
`NO-GO` unless the opening immediately names the kinds of changes or gives real
examples. “Tests for data handling, training rules, and AI tools” tells the
reader what the folder actually checks.

After the examples, the prose returns to the general rule and states the
boundary that the examples illustrate. An example must not look like the only
supported case. An analogy may help after the real cases are present, but an
analogy never replaces repository examples.

Use one example when one case makes the rule clear. Use two when a contrast is
the lesson, such as ordinary success versus refusal, or a small test versus a
final gate. More examples receive `GO` only when each adds a different boundary
the reader needs.

Terms that repeatedly failed this test include `lane`, `drain`, `ledger`,
`identity`, `schema`, `dispatch`, `publication`, `state transition`,
`inflight`, `worktree`, `branch`, and `commit`. These words are not forbidden.
They must be replaced or immediately tied to something visible: a file, a
folder, a saved project version, a command, or an action.

Examples:

| `NO-GO` wording | `GO` wording |
| --- | --- |
| “Publication and dispatch are filesystem state transitions.” | “Sending a request saves a Markdown file. Just before starting the role, the watcher moves that file into `inflight/`, the work-in-progress folder.” |
| “Zero mode drains the ledger.” | “With `--cycle 0`, the watcher exits only after no enabled message is waiting and no backlog line begins with `- OPEN`.” |
| “The schema-1 identity is invalid.” | “The saved folder information came from an older version of the tool, so the current tool cannot safely choose a work folder.” |

## Review 3: structure and visuals help the reader

This review applies to READMEs. For a Python-only prose change, mark its rows
`not applicable: no README structure or visual changed` instead of silently
omitting them.

| Check | `GO` | `NO-GO` |
| --- | --- | --- |
| Table of contents | Main sections and appendix groups are visibly separate; every link resolves | Startup steps and reference material are mixed in one long list |
| Headings | A main heading names a reader action; an appendix heading is a useful question | A heading is an internal noun phrase such as “Demand guard” or “Worktree topology” |
| Examples | YAML and code fences use real shipped structure and are valid when copied | A fence only resembles valid input or omits a required key |
| Tables | Options and repeated field comparisons use columns with plain headings | Dense option lists are buried in prose or unexplained bullets |
| Diagrams | A diagram makes a sequence, branch, comparison, or loop easier to see than prose alone | The diagram repeats a simple fact, has ambiguous labels, or needs more explanation than it saves |
| Narrow screens | A sequence flows from top to bottom with short labels; the first diagram is a small mental model | The opening diagram presents the complete topology, or a left-to-right flow requires horizontal scrolling on a phone-sized view |
| Mermaid rendering | GitHub-compatible rendering succeeds; punctuation and command-line flags are inside quoted labels; no label overlaps or is clipped | The source looks plausible but the rendered diagram was not inspected |
| Display equations | GitHub-supported `$$ ... $$` blocks render, and nearby prose defines every symbol | Plain `[ ... ]` is used as pseudo-math, the equation was not rendered, or its symbols are unexplained |
| Text fallback | The prose states every essential rule shown in a diagram | A reader must decode the diagram to learn a safety rule |
| Appendix language | Appendices use the same concrete words, short paragraphs, and local definitions as the main guide | “It is only an appendix” is used to excuse specialist language |

More diagrams is not the same as better teaching. Use a diagram only when the
review record names the relationship it clarifies. Folder ownership that fits
in three bullets should remain three bullets. A multi-step mailbox life cycle
usually benefits from a diagram.

Sequential diagrams default to top-to-bottom flow because that shape remains
readable on phones and tablets. A left-to-right diagram needs a recorded reason
and inspection at a narrow width. The first diagram in a guide teaches the
smallest useful mental model; a later appendix may show the complete topology.

## Review 4: every statement matches the project

These failures are immediate `NO-GO` because they can make a reader run the
wrong command or believe a false result.

| Check | Required evidence |
| --- | --- |
| Commands and options | Run each new or changed `--help`, `--dry-run`, or safe example. Record the command, return code, and relevant output |
| YAML and configuration | Parse every copy-paste YAML fence with the real parser or a focused syntax check, then confirm required keys and values against a shipped file |
| Paths and symbols | Confirm every backticked repository path exists and every named option or symbol exists in the current source |
| Defaults and exact output | Compare with live code. If repository-owned output is unclear, change the output, README quotation, and parity test together |
| Internal links | Check every changed anchor and all links affected by a renamed heading; duplicate anchors are `NO-GO` |
| Local assets | Confirm every image and other local asset exists at the linked path |
| Equations | Render every changed display equation with GitHub-supported delimiters and confirm that the surrounding prose defines its symbols |
| External links and citations | Open every new source. Verify its identity and confirm that it supports the sentence that cites it |
| Numbers and thresholds | Trace each new number to code, a shipped file, a table, or a cited source |
| Safety behavior | Confirm that shorter prose did not remove a failure case, ordering rule, timeout, refusal, or recovery step |
| Python comments and docstrings | Compare the prose with the complete symbol: signature, types, shapes, units, defaults, side effects, exceptions, and non-obvious invariants |
| Help and diagnostics | Exercise the relevant command or failure path and compare the actual text, return code, and next action with the prose |

Never truncate a search that supports a negative claim. A command using
`head` cannot prove that no stale path, undefined term, or prompt fragment
exists later in the output. Count or inspect the complete result.

## Review 5: the prose is concrete, not AI-polished fog

**Anti-AI writing does not mean deliberately rough writing.** It means that
each sentence carries this repository's facts, the reader's action, or a
specific reason. Generic generated scaffolding, praise, symmetry, and
restatement do not earn space.

### Check the strongest evidence first

The strongest warning signs are factual, not stylistic:

1. Prompt residue or placeholders remain in the source.
2. A reference, command, path, number, or example is invented or mismatched.
3. A polished paragraph omits the exact details a reader needs.
4. A claim praises a result more than it specifies the result.

Use this order during review:

| Evidence | What the Architect checks | Decision |
| --- | --- | --- |
| Direct residue | Prompt fragments, placeholders, model self-reference, unfinished instructions, copied chat openings | Any unexplained instance is `NO-GO` |
| False or mismatched fact | Command, path, option, default, output, citation, link, number, or diagram disagrees with the current project | Any instance is `NO-GO` |
| Missing hard detail | The prose says a method is safe, standard, automatic, easy, or supported but omits the condition, failure, file, option, or expected result | `NO-GO` until the exact detail is present |
| Reader failure | A cold reader cannot tell what to do, what changes, or what success means | `NO-GO` even when every sentence is technically true |
| Repeated AI-shaped style | Several sentence, paragraph, or vocabulary patterns flatten the section into generic prose | Inspect and rewrite the repeated pattern; a single occurrence is not enough |

### Remove prompt residue completely

The changed prose must contain none of these residues unless it is explicitly
quoting them as a warning:

- `Certainly, here is`
- `as an AI`
- `TBD`, `TODO`, or a template placeholder
- `insert citation`
- `replace with`
- prompt instructions or model commentary

Also search for unfinished bracketed fields such as `[describe result]`,
`<insert path>`, `example goes here`, `fill this in`, and `coming soon`.
Legitimate literal placeholders inside a command reference must be named as
placeholders and followed by a real example.

Do not inspect only the rendered page. Prompt residue can hide in Markdown
comments, HTML comments, collapsed `<details>` blocks, link labels, image alt
text, and Mermaid source.

### Replace generic praise with checkable facts

Use concrete actors and verbs. “The watcher moves the file” is easier to
check than “a filesystem transition occurs.” “`--cycle 2` admits no more than
two tickets; in the normal setup each ticket needs a daemon-recorded landing L
and its matching Red Team return before exit” carries more information than “this
provides useful runtime control.”

The following claims need evidence or removal:

- `easy`, `simple`, `intuitive`, `automatic`, `safe`, `fast`, `lightweight`,
  `flexible`, `powerful`, `production-ready`, `seamless`, `complete`, and
  `supported`;
- “handles errors,” “works out of the box,” “uses best practices,” “improves
  performance,” and “provides a better experience”;
- “standard approach,” “conventional method,” or “robust workflow” without
  the exact algorithm, version, condition, test, or failure behavior.

For example, replace “safe automatic shutdown” with the actual condition:
“The watcher stops starting jobs, waits for jobs already running, and then
prints the Ctrl-C countdown.” Replace “fast” with a measured time and machine,
or remove the claim.

### Apply the README vocabulary bans

In new or changed prose covered by this contract:

- do not use `thereby`;
- do not use `commendable`, `innovative`, `meticulous`, `intricate`,
  `notable`, or `versatile` as adjectives;
- replace decorative `delve`, `crucial`, `comprehensive`, `notably`,
  `underscores`, `highlights`, `showcases`, `sheds light on`, `leveraging`,
  and `utilize` with the fact they were trying to decorate;
- keep `robust`, `robustness`, and other domain terms when they carry a
  precise technical meaning.

#### Hard-zero words

The following words are prohibited in new or changed prose covered by this
contract. An exact command, code identifier, external title, or quotation may
contain one only when changing it would make the reference false. The review
record names every such exception.

```text
commendable
comprehensive
crucial
crucially
delve
delves
delving
dwelve
innovative
intricate
meticulous
multifaceted
notable
notably
pivotal
thereby
transformative
versatile
```

These forms are prohibited for the same reason:

```text
showcase
showcases
showcasing
underscore
underscores
underscoring
unveil
unveils
unveiling
```

#### Hard-zero phrases

Do not use these generated phrases in new or changed covered prose:

```text
as an AI
certainly, here is
complex interplay
deeper understanding
evolving landscape
future work should explore
important implications
it is important to note
it is worth noting
plays a crucial role
provides valuable insights
sheds light on
stands as a testament
taken together, these results
the realm of
valuable insights
```

Also remove bot closers and chat residue such as `I hope this helps`, `Let me
know if you would like`, `Here is the revised version`, `In conclusion`, and
`Overall` when they merely end a generated answer.

#### Words that require a technical reason

The words below are not automatically forbidden because some have real
technical meanings. They are `NO-GO` when they praise, inflate, or connect
prose without adding a checkable fact.

| Word family | Acceptable technical use | `NO-GO` decorative use |
| --- | --- | --- |
| `robust`, `robustness` | Names an actual sensitivity test, failure test, or statistical property | “a robust and reliable workflow” |
| `framework` | Names a specific class, file format, or defined interface | “a comprehensive framework for innovation” |
| `paradigm` | Part of an established scientific name used by the project | “a new paradigm for seamless development” |
| `significant` | Gives a statistical meaning or measured threshold | “a significant improvement” without a number |
| `efficient` | Gives time, memory, scaling, or a comparison | “an efficient solution” without a measurement |
| `standard` | Names the exact standard, version, or repository convention | “the standard approach” without the method |
| `safe` | Names the invariant, ordering rule, or test that keeps data safe | “a safe process” without the condition |
| `automatic` | Names the trigger, exact action, and refusal case | “automatic management” without behavior |

#### Decorative vocabulary to replace

The following words and phrases usually hide a simpler verb or a missing
fact. Replace them unless the review record identifies a literal technical
meaning:

```text
actionable insights
advance our understanding
at the forefront
broader implications
cornerstone
cutting-edge
dynamic
ecosystem
elevate
empower
enhance
facilitate
foster
harness
holistic
impactful
in order to
in the context of
insightful
journey
landscape
leverage
meaningful
nuanced
offer insights
pave the way
poised to
powerful
realm
roadmap
seamless
state-of-the-art
streamline
synergy
tapestry
unlock
utilize
valuable
vibrant
within the context of
```

Prefer `use` to `utilize` or `leverage`. Prefer `is` to `serves as`. Prefer
the name of the operation to `streamline`, `enhance`, or `facilitate`.

The following generated-sounding phrases are also `NO-GO` in new prose unless
the review record shows that they are literal names or technically necessary:

- `It is important to note`, `It is worth noting`, `This is important
  because`, and `A few points should be noted`;
- `plays a crucial role`, `serves as`, `stands as`, `acts as`, and `represents
  a key step` when the plain verb `is`, `uses`, `checks`, or `runs` says the
  fact;
- `rich`, `vibrant`, `pivotal`, `transformative`, `cutting-edge`,
  `state-of-the-art`, `multifaceted`, `nuanced`, `evolving landscape`,
  `ecosystem`, `realm`, `tapestry`, and `cornerstone` as praise;
- `valuable insight`, `meaningful contribution`, `important implication`,
  `deeper understanding`, `comprehensive perspective`, and `future work
  should explore`;
- `unlock`, `foster`, `enhance`, `streamline`, `empower`, `facilitate`,
  `harness`, and `unveil` when a concrete verb names the operation;
- `Moreover`, `Furthermore`, `Additionally`, `Notably`, `Importantly`,
  `Consequently`, `Taken together`, and `Overall` when they merely make a
  paragraph sound connected.

Do not replace one banned phrase with a synonym that performs the same empty
job. Delete the decoration or write the missing fact.

### Inspect sentence-level patterns

Look for repeated patterns across the changed section:

- many medium-length sentences with the same shape;
- several paragraphs with the same announce-explain-caveat-summary arc;
- repeated “This suggests,” “While X, Y,” or “By doing X, we Y” openings;
- repeated endings such as “highlighting” or “underscoring”;
- several abstract nouns where a person or program could perform an action;
- repeated roadmap sentences that announce, perform, and recap a simple step;
- newly added question-and-answer transitions that do not match the
  established question-led appendix structure.

The Architect explicitly checks these sentence shapes:

1. `This suggests/indicates/demonstrates/highlights that ...`
2. `While X, Y ...`
3. `By doing X, we Y ...`
4. `X, thereby Y ...`
5. `X, highlighting/indicating/reinforcing Y ...`
6. `Although X, it is important to note Y ...`
7. `Not only X, but also Y ...`
8. `X does not merely do A; it also does B ...`

A single useful sentence may have one of these shapes, except for the hard
ban on `thereby`. Three or more rhetorical uses in one changed section require
an explicit review. Rewrite the repeated frame with direct statements.

Check for abstract-noun stacking. A sentence such as “The validation of the
configuration enables the identification of the availability of the route”
hides every actor. Write “The watcher checks the configuration and reports
whether the route is available.” Four or more nearby nouns ending in
`-tion`, `-ment`, `-ity`, `-ance`, `-ence`, `-ness`, or `-ization` trigger
manual review.

Check hedge stacking. `May`, `might`, `could`, `appears`, `potentially`,
`somewhat`, and `to some extent` can express real uncertainty. Two or more in
one sentence usually mean the writer has not stated the actual condition.
Prefer “This happens only when X” or “This case is not tested.”

Check causal words. `Therefore`, `thus`, `hence`, `consequently`, and `as a
result` must connect facts that actually prove the conclusion. They cannot
make a weak step look stronger.

### Inspect paragraph-level patterns

Generated paragraphs often repeat this five-part arc:

1. announce the topic;
2. explain it in broad terms;
3. add one technical detail;
4. add a balanced caveat;
5. summarize the paragraph.

README paragraphs may be shorter and less symmetrical. A two-sentence warning
is fine. A one-sentence transition is fine. A longer explanation is fine when
the mechanism needs it. Repeating the same polished arc across a section is
`NO-GO` because the reader must work through repeated introductions and
summaries to find the action.

Inspect ten consecutive paragraphs when the changed area is large. A regular
sequence such as 5, 5, 4, 5, 6, 5, 4, 5, 5, 6 sentences deserves review. A
README normally has a less regular mix of short warnings, examples, tables,
and fuller explanations. There is no required mathematical variance; the
Architect records the repeated visible pattern and decides whether it hides
information.

Do not add a roadmap sentence before every table or example. “The next
section explores...,” “We now examine...,” and “Before turning to...” should
be removed when the heading already says the same thing.

Do not announce, show, and then recap a command whose result is already clear.
Use the space to say where to run it, what it changes, and what output means.

### Inspect lists, questions, and formatting

Perfectly balanced groups of three are not automatically clearer. Keep three
items when there are exactly three real items. Do not reshape two or four
facts into three for rhythm.

Frequently asked question (FAQ) headings are required in the appendices
because they help readers jump to a real problem. Do not add decorative
question-and-answer prose inside every FAQ, such as “What does this mean? The
answer lies in...”. Answer the heading directly.

Bold lead-ins, colons, semicolons, parentheses, passive voice, and complete
sentences are not AI evidence by themselves. Review repetition and reader
cost. Do not enforce a business-writing rule that damages valid Markdown,
code, equations, or scientific notation.

Do not rotate through synonyms merely to avoid repeating the correct term.
If the code calls it a watcher, keep calling it a watcher. Switching between
watcher, daemon, monitor, controller, and orchestrator makes a beginner think
they are different programs.

### Inspect README-specific failure patterns

| README area | AI-shaped failure | Required repair |
| --- | --- | --- |
| Opening | Grand description of importance before saying what the project does | State the concrete library purpose and the first useful link or command |
| Installation | “Install the dependencies” without names, versions, environment, or a verification command | Give the exact command and the visible successful check |
| Configuration | Describes a flexible system but omits a valid copy-paste block | Show a real minimal block and explain each non-obvious key locally |
| Runtime | Describes the happy path and hides refusal, timeout, or cleanup behavior | Name the important failure, what remains safe, and the user's next action |
| Model or algorithm section | Gives tutorial language for standard ideas but skips this library's unusual choice | Shorten the generic tutorial and explain the repository-specific behavior |
| Results | Says output is insightful, accurate, or robust without naming files, units, thresholds, or comparison | Name the artifact and the exact quantity a user should inspect |
| Troubleshooting | Gives a broad cause such as “configuration issue” | Use a symptom, likely meaning, and first action table |
| Appendix | Uses internal language because the material is optional | Apply the same cold-reader test and local definitions as the main guide |
| Diagram | Adds attractive boxes without clarifying sequence, branching, comparison, or feedback | Remove it or state the relationship it makes easier to see |
| Caption or alt text | Interprets an image without saying what is plotted, the units, or the direction of arrows | State construction, axes or nodes, units when relevant, and the intended reading |
| Python comment | Restates the next line or adds a polished paragraph without a non-obvious reason | State the invariant, unit, shape, refusal reason, or compatibility constraint the code alone does not reveal |
| Python docstring | Gives a broad purpose but omits real arguments, return shape, units, side effects, or errors | Match the current signature and state the behavior a caller needs |
| Help or diagnostic string | Says only that input is invalid or an operation failed | Name the bad value or condition and the user's next safe action without exposing secrets |
| Git commit message | Begins with an internal ticket, role, branch, or generic action and leaves the reader to inspect the diff | Use a concrete plain-language subject, then a short Markdown body that explains the observed problem, the saved change and its boundary, and the exact evidence |

### Preserve the useful human signals

Good README prose has selective emphasis. It says which restriction matters,
which command is the normal one, which option is unusual, and why a failure
must stop. It may contain a short fragment such as “Do not delete the folder.”
It may use an uneven paragraph rhythm. It repeats a technical term when that
term is the clearest name.

Concrete limitations are useful. “CUDA validation was not run on this
machine” is actionable when followed by the exact owed command. “Some cases
may require further testing” is not.

The README should expose real friction. If an operation can refuse, time out,
leave a file waiting, require a worktree, or need a separate machine, say so.
A guide where every operation succeeds effortlessly is probably hiding the
steps a new user most needs.

### Use measurements as prompts, not verdicts

For a large prose change, the Architect records:

- sentence word counts for at least one 30-sentence window;
- sentences per paragraph across at least ten neighboring paragraphs;
- counts of repeated `This + verb` openings;
- counts of participial endings such as `, highlighting` or `, indicating`;
- counts of paragraph-opening transition words;
- the most repeated non-technical three-to-six-word phrase;
- the single most repeated visible rhetorical move.

Low variation or repeated phrases direct the Architect to inspect the text.
They do not determine the verdict. Technical repetition such as
`ai/notes/backlog.md` or `--cycle 0` is often necessary. Rhetorical repetition
such as “provides valuable insight” is not.

Do not use an AI detector as evidence for `GO` or `NO-GO`. The decision is
about factual truth, reader success, and visible repeated patterns in the
actual prose.

### Apply the false-positive rule

These patterns are review prompts, not proof. One list, polished sentence,
semicolon, passive verb, FAQ question, or technical word is never enough for
`NO-GO`. Name the actual reader problem or repeated pattern. Do not flatten
accurate README prose merely to satisfy an automated style score.

## Review 6: required evidence is complete

Every covered prose review records the applicable evidence below:

- `git diff --check`;
- complete internal-link, duplicate-anchor, local-asset, and backticked-path
  checks;
- balanced Markdown fences and HTML detail blocks;
- parsing or execution of every changed copy-paste example;
- a source or focused test that confirms every repository example used to
  explain an abstraction;
- a GitHub-compatible render and visual inspection of every changed Mermaid
  diagram;
- a successful source build and page-by-page rendered inspection of every
  changed long-form PDF, including clipping, overlap, figures, references,
  source-to-PDF freshness, and readable narrow columns;
- exact help or output parity when quoted program text changes;
- full-source searches for residue, stale historical wording, and the
  technical terms the directive promised to define;
- focused behavior tests for claims about roles, work folders, timeouts,
  stopping, refusals, or file movement;
- `python3 -m py_compile` and focused tests for changed Python comments,
  docstrings, help, or diagnostics, including exact-output parity where used;
- `permanent_note_guard.py` using the directive's exact worktree and full
  starting commit, rerun by the Architect;
- the repository's relevant documentation tests;
- one cold-reader review of the final section in context.

Automated green results are necessary but not sufficient. The Architect must
still read the rendered README or complete Python symbol and decide whether
the target reader can act on the prose.

## Decision record

Use this shape in the temporary ticket note:

```markdown
### Reader-facing documentation / Python prose GO / NO-GO review

- **Verdict:** GO | NO-GO
- **Files and sections or symbols:** [exact paths and headings or symbols]
- **Pinned permanent-note base:** [full commit + guard command and PASS line]
- **Reader outcome:** [question answered and next action]
- **Location:** [main guide, appendix, Python symbol, help, or diagnostic; give the reason]
- **Terms checked:** [plain definitions or replacements]
- **Examples and visuals:** [what was run or rendered]
- **Technical truth:** [code, shipped files, commands, links, and numbers checked]
- **Voice and residue:** [complete searches and any adjudicated exceptions]
- **Evidence:** [exact commands, return codes, and artifact paths]
- **Failed rows and repair:** [none for GO; exact locations and changes for NO-GO]
```

`GO` means every applicable row is satisfied and supported by raw evidence.

`NO-GO` names each failed row, the exact file and passage, the concrete repair,
and the evidence that must be rerun. “Improve clarity,” “make it didactic,” or
“use best judgment” is not a repair instruction.

## Supporting sources

This contract turns two permanent policy notes into an operational decision:

- `ai/notes/conventions-and-workflow.md`, section **README and teaching contract**;
- `ai/notes/python-changes-go-no-go.md`, sections **Scope and decision
  authority** and **Teaching text inside Python**.

Style checks remain secondary to technical truth and reader success. They
must never be used as an automatic score of who wrote the text.
