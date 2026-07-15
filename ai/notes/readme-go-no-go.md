# README and Python prose GO / NO-GO contract for the Architect

This contract applies whenever a unit creates or changes either:

- a tracked README; or
- explanatory prose inside Python: comments, docstrings, command help,
  user-facing diagnostics, and strings whose purpose is to explain behavior.

For a README, it covers the main guide, appendices, tables, diagrams,
captions, command examples, and exact program output quoted in the guide. For
Python, it covers the explanatory words, not protocol tokens, serialized data,
test fixtures, or strings that have no teaching or diagnostic purpose.

The target reader is a physics undergraduate who knows no AI-agent language
and may know little Git. A reader may open any section directly instead of
reading the file from the beginning. Each section must therefore explain the
terms it needs at the place where it uses them.

Only the Architect issues `GO` or `NO-GO`. The Implementer supplies the
change and its evidence. The Red Team may identify a problem and propose a
repair. Neither role replaces the Architect's final review.

## The Architect reads this file twice

### Before writing the implementation directive

1. Read the full README section or Python symbol that will change. For a
   README, also read its table-of-contents entry and the paragraphs
   immediately before and after it.
2. Read the code, shipped configuration, live help, or other current source
   that proves every behavior the covered prose will describe.
3. For a README, decide which information belongs in the short main path and
   which belongs in an appendix. For Python prose, decide whether the
   explanation belongs beside the code, in a docstring, in command help, or
   in the README instead.
4. Copy every applicable check in this file into the directive's
   `Acceptance checklist`. Each check must name the evidence the Implementer
   will return.
5. Mark a check not applicable only with a concrete reason. An omitted check,
   an unexplained exemption, or a choice left to the Implementer is `NO-GO`
   for dispatch.

### Before reviewing the final change

1. Reopen this file. Do not rely on memory or on the earlier directive.
2. Read the final rendered README section or the complete Python symbol, not
   only the changed lines.
3. Re-run the factual, command, link, and rendering checks that apply.
4. Treat the Implementer's checked boxes as evidence to inspect. They are not
   the verdict.
5. Issue `GO` only when every applicable row below is supported by raw
   evidence. Otherwise issue `NO-GO` and name the exact repair.

## Decide where the material belongs

The main path gets a new user to a valid result. It answers, in order:

1. What do I need?
2. What do I run or configure?
3. What should I see?
4. What should I do next?

Theory, implementation detail, recovery internals, long explanations, and
reference material belong in appendices. Moving material to an appendix does
not permit harder language. The same reader standard applies everywhere.

For `README.md` and `ai/README.md`, the table of contents must visibly
separate the short main guide from **Common questions raised by developers**.
Appendices are grouped by topic and use real questions. The main guide stays
short enough that a new user does not have to read the appendices before the
first successful run.

Small package READMEs do not need artificial appendices. They still put the
first useful command or code example before internal design detail.

Python comments explain a non-obvious reason, invariant, unit, shape, failure,
or compatibility rule. They do not narrate the next line of code. Docstrings
state the callable's real inputs, outputs, shapes, units, side effects, and
errors. Command help and diagnostics tell the user what happened and what to
do next. If an explanation teaches a general workflow rather than one nearby
symbol, put it in the README and keep only a short pointer in Python.

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
ai/notes/user-didactics-and-python-voice.md
ai/tools/permanent_note_guard.py
```

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

## Gate 1: the instruction is complete

The Architect answers these questions before dispatch.

| Check | `GO` for the directive | `NO-GO` for the directive |
| --- | --- | --- |
| Reader outcome | Names the exact question the changed section will answer and the next action the reader can take | Says only “improve,” “clarify,” “document,” or another subjective goal |
| Placement | Names the README location or the exact Python comment, docstring, help, or diagnostic location and explains why it belongs there | Leaves placement, heading order, comment location, or docstring scope to the Implementer |
| Source of truth | Names the exact code, shipped file, command, or current policy that supports each behavior | Relies on memory, chat, an old README statement, or a temporary audit note |
| Vocabulary | Lists the non-physics technical words the passage needs and gives a plain definition or replacement for each | Assumes that a later section, glossary, or software background will rescue an undefined term |
| Edit plan | Names every file, heading, paragraph, table, diagram, link, and exact-output block that may change | Gives a broad file-level request and asks the Implementer to choose the structure |
| Examples | Supplies or identifies a real copy-paste example, where to run it, whether it changes files, and the visible successful result | Requests an illustrative example that need not pass the real parser or command |
| Visual choice | For a README, states which sequence, branch, comparison, or feedback loop a diagram will clarify; otherwise records `not applicable` for Python-only prose | Requests “more graphs” without naming the relationship each graph must show, or omits the applicability decision |
| Tests | Gives exact validation commands and expected results for links, examples, output parity, rendering, Python syntax, help, diagnostics, and docstring behavior that the change affects | Leaves the Implementer to invent the acceptance tests |
| Boundaries | Names facts, safety rules, commands, and unrelated sections that must not change | Allows a clarity edit to silently change behavior or widen scope |

An Implementer may be Sonnet, Haiku, or an open-source model. The directive
must be executable without retained chat, hidden design choices, or “use your
best judgment.”

## Gate 2: the reader can use the result

| Check | `GO` | `NO-GO` |
| --- | --- | --- |
| Direct answer | A README opens with the heading's answer or action; Python prose starts with the callable's action, the comment's reason, or the diagnostic's problem | The passage begins with background, history, or an abstract system description |
| Cold-reader test | A physics undergraduate can restate the passage and identify the next action without opening another section | Understanding depends on software-engineering or AI knowledge that the passage never supplies |
| Local definitions | Every unfamiliar term is replaced or defined in concrete words at its first use in that section | A definition uses another undefined term or points only to a distant definition |
| One job per paragraph | Each paragraph explains one action, fact, warning, or consequence | One paragraph mixes setup, mechanism, exception, and recovery |
| Manageable length | Ordinary prose is split before it becomes a wall of text; a paragraph over four sentences or about 100 words has a recorded reason to remain whole | Long prose is retained only because it is technically correct or appears in an appendix |
| Parentheses | Parentheses contain a short definition, symbol, unit, or acronym | Removing a parenthetical would remove an essential rule or a second argument |
| Current state | The README or Python prose says what the library or symbol does now and what the reader should do now | It narrates migration history, ticket history, rejected designs, intermediate commits, or old paths |
| Complete command | The reader knows prerequisites, working folder, command, expected result, and whether the command changes files; command help names required values and defaults | A command appears without enough context to run or interpret it safely |
| Current limitation | A limitation states its present scope, consequence, and user action | It explains when the limitation was found or promises an unimplemented repair |

Use the **novice paraphrase test** for every definition: ask whether the target
reader could say the same fact in ordinary words. If not, the definition is
still too technical.

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

## Gate 3: structure and visuals help the reader

This gate applies to READMEs. For a Python-only prose change, mark its rows
`not applicable: no README structure or visual changed` instead of silently
omitting them.

| Check | `GO` | `NO-GO` |
| --- | --- | --- |
| Table of contents | Main sections and appendix groups are visibly separate; every link resolves | Startup steps and reference material are mixed in one long list |
| Headings | A main heading names a reader action; an appendix heading is a useful question | A heading is an internal noun phrase such as “Demand guard” or “Worktree topology” |
| Examples | YAML and code fences use real shipped structure and are valid when copied | A fence only resembles valid input or omits a required key |
| Tables | Options and repeated field comparisons use columns with plain headings | Dense option lists are buried in prose or unexplained bullets |
| Diagrams | A diagram makes a sequence, branch, comparison, or loop easier to see than prose alone | The diagram repeats a simple fact, has ambiguous labels, or needs more explanation than it saves |
| Mermaid rendering | GitHub-compatible rendering succeeds; punctuation and command-line flags are inside quoted labels; no label overlaps or is clipped | The source looks plausible but the rendered diagram was not inspected |
| Text fallback | The prose states every essential rule shown in a diagram | A reader must decode the diagram to learn a safety rule |
| Appendix language | Appendices use the same concrete words, short paragraphs, and local definitions as the main guide | “It is only an appendix” is used to excuse specialist language |

More diagrams is not the same as better teaching. Use a diagram only when the
review record names the relationship it clarifies. Folder ownership that fits
in three bullets should remain three bullets. A multi-step mailbox life cycle
usually benefits from a diagram.

## Gate 4: every statement matches the project

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
| External links and citations | Open every new source. Verify its identity and confirm that it supports the sentence that cites it |
| Numbers and thresholds | Trace each new number to code, a shipped file, a table, or a cited source |
| Safety behavior | Confirm that shorter prose did not remove a failure case, ordering rule, timeout, refusal, or recovery step |
| Python comments and docstrings | Compare the prose with the complete symbol: signature, types, shapes, units, defaults, side effects, exceptions, and non-obvious invariants |
| Help and diagnostics | Exercise the relevant command or failure path and compare the actual text, return code, and next action with the prose |

Never truncate a search that supports a negative claim. A command using
`head` cannot prove that no stale path, undefined term, or prompt fragment
exists later in the output. Count or inspect the complete result.

## Gate 5: the prose is concrete, not AI-polished fog

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
check than “a filesystem transition occurs.” “`--cycle 2` exits after two
completed cycles” carries more information than “this provides useful runtime
control.”

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
- newly added question-and-answer transitions that do not match the user's
  own question-led appendix structure.

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

FAQ headings are required in the appendices because they help readers jump to
a real problem. Do not add decorative question-and-answer prose inside every
FAQ, such as “What does this mean? The answer lies in...”. Answer the heading
directly.

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

## Gate 6: required evidence is complete

Every covered prose review records the applicable evidence below:

- `git diff --check`;
- complete internal-link, duplicate-anchor, local-asset, and backticked-path
  checks;
- balanced Markdown fences and HTML detail blocks;
- parsing or execution of every changed copy-paste example;
- a GitHub-compatible render and visual inspection of every changed Mermaid
  diagram;
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
### README / Python prose GO / NO-GO review

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

- `ai/notes/conventions-and-workflow.md`, section **README / didactics**;
- `ai/notes/user-didactics-and-python-voice.md`, sections **Who is reading**
  and **How she likes to be taught**.

Style checks remain secondary to technical truth and reader success. They
must never be used as an automatic score of who wrote the text.
