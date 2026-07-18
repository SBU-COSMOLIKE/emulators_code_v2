# GO/NO-GO contract for the style of Python changes

Python style is a release condition, not a preference. This contract applies
to every Python file saved by Git, the repository's version-control system:
production code, command programs, tests, gate commands, tools,
comments, docstrings, command help, diagnostics, and explanatory strings. Such
a file is called **tracked** below.

The Architect reads the full contract before sending an implementation
instruction and again before the final verdict. Missing evidence for an
applicable check requires NO-GO. The Implementer and Red Team never edit this
contract.

## Terms used by this contract

A **diff** is the line-by-line comparison between two saved versions. A
`path::symbol` label combines a file path with the function, class, or method
that owns the change. A **forward pass** is one model evaluation from input to
output. A **batch** is one group of rows evaluated together.

A **vectorized operation** computes many array entries in one NumPy or PyTorch
operation. NumPy is the array package used by the library, and PyTorch is its
tensor and machine-learning package. A numerical **kernel** is the repeated
calculation at the center of an operation. An **accelerator** is hardware such
as a graphics processing unit (GPU) that runs those calculations.
**De-vectorizing** replaces the array operation with slower
element-by-element Python work.

A numerical **data type (dtype)** states how values are represented, such as
float32 or float64. A **schema** is the required fields, types, and meanings of
a structured record. A **gate** is a named validation job whose required
result is written before it starts. A **gate registry** is the list that
connects each named gate to the command that runs it. **Dispatch** is the
Architect's act of sending a complete instruction to the Implementer or Red
Team. An
**artifact** is a saved model result containing weights and the facts
needed to rebuild them.

`einsum` is the NumPy or PyTorch operation that expresses a tensor contraction
with index letters. A Git **branch** is a named line of saved repository
versions. A **worktree** is a separate Git working folder attached to one
branch, so an agent can edit without changing the user's folder.

The style rules name several Python shortcuts. A **comprehension** builds a
collection inside brackets, for example `[convert(item) for item in items]`.
A **conditional expression**, sometimes called a ternary, chooses a value
inside one expression: `left if condition else right`. An **assignment
expression** uses `:=` to assign a value inside a larger expression. A
**lambda** is an unnamed, single-expression function. **Starred unpacking**
uses `*values` or `**options` to insert items from another sequence or mapping.
**Metaprogramming** creates or changes functions or classes while the program
runs. **Module-global state** is a value defined at the top level of a Python
file and shared by calls that do not receive the value explicitly.

A **monkey patch** replaces existing executable behavior while Python is
running. Examples include replacing an imported function or method, changing
`sys.modules`, `__defaults__`, `__code__`, or `__class__`, and using `patch`,
`patch.object`, `patch.dict`, or pytest's `monkeypatch` fixture.

## Scope and decision authority

The Architect alone issues the final style verdict. A correct result written
in an unnecessarily difficult form receives NO-GO. A small character budget
never permits compressed, cryptic, or misleading Python.

The intended reader is a library user or physics student who understands
C-like control flow but may not know advanced Python idioms. Code must remain
easy to trace one operation at a time. User-facing prose must define software
and machine-learning terms where the terms first matter.

### Do not add monkey patches

New monkey patches are prohibited in production code, tests, gates, and
tools. A candidate that adds, copies, retargets, or broadens one receives
NO-GO. Use an explicit fake argument, a subclass defined before use, a
temporary file, or a separate process whose changed files, arguments, or
environment are selected before import. The child process must not replace
live behavior after Python starts.

Importing a module or assigning an alias alone is not a monkey patch;
replacing behavior through that alias is. Assigning ordinary non-executable
data to a new local object during construction is allowed when that object is
passed explicitly and is not installed into an imported module, class,
registry, or process-global table. Replacing a method or function on even a
local instance is a monkey patch.

Do not turn an unrelated ticket into a repository-wide cleanup. For every
existing site encountered during bounded work, the Architect records one
separate **High bug-fix ticket**. The Red Team reports each site to the
Architect; only the Architect writes the ticket. Do not search for additional
sites unless the user explicitly requests that search.

### Keep the repair proportional to the problem

A narrow bug normally needs a narrow production-code change. Robustness is
useful, but it does not justify building a new framework around one failure.

The Architect gives NO-GO when a candidate adds a registry, policy layer,
general validation system, or other large abstraction where a short direct
check would fix the named bug. A large production diff needs a concrete
explanation of why the smaller direct design is unsafe. Without that
explanation and explicit user approval, the ticket must be split or simplified.

Tests under `ai/tests/` and gates under `ai/gates/` may be longer than
the production fix because they show valid and invalid examples. They must
still be readable, but their useful examples are not evidence that production
code should also grow.

Treat `emulator/`, `compute_data_vectors/`, and `cobaya_theory/` as the
scientific reading path. The Architect must keep changes there small, direct,
and understandable line by line. Prefer deleting duplicated machinery, using
an existing function, or adding one local check over creating a new subsystem.

For one bug, count added characters plus deleted characters outside
`ai/tests/` and `ai/gates/`. More than 1,500 characters creates a strong
presumption of NO-GO. It is not an automatic rejection: the Architect may
accept the change only when the directive explains specifically why a smaller
direct repair would be unsafe and why the ticket cannot be split into complete
independent fixes. Passing tests or `--max 0` is not that justification.

A bounded repair may remove an actionable ticket's demonstrated failure while
leaving a harmless exceptional case below Low. When complete coverage would add
disproportionate complexity, choosing the simple bounded repair is acceptable
and may be preferable. The Architect closes the actionable ticket and records
the exact remainder as a linked, parked **LOW — EDGE CASE** bug ticket without
claiming complete coverage.

This parked class is not ordinary Low work. It has no `- OPEN` line, is not a
`--severity` choice, and is never dispatched automatically. Only an explicit
user request naming that exact ticket authorizes the Architect to activate it
as Low work. Do not use this rule to park a probable failure, wrong primary
scientific result, data loss, or broken core operation.

### Keep user responsibility visible

Add a protective check when it is simple, cheap, and intuitive at the boundary
where the value enters. Do not build a new framework to anticipate every way a
user could express an equivalent scientific choice. Arbitrary renamed,
derived, or transformed parameters are the user's responsibility unless the
library already has a small, explicit rule for that exact representation.

A best-effort check states what it actually compares. Missing names never prove
that two cosmologies disagree, and matching names never prove that arbitrary
parameterizations are scientifically equivalent. Prefer a direct comparison
of values both sides name over symbolic interpretation, duplicated model
resolution, or another saved identity layer. When the library cannot establish
the fact simply, document the limitation and leave the scientific choice with
the user. Do not claim that an incomplete comparison is a proof.

The Architect applies this rule throughout the scientific reading path named
above. A plan that adds a helper family, registry, digest, schema, or validation
subsystem for compatibility the user can verify from configuration receives
NO-GO unless a direct check cannot protect a demonstrated primary result and
the user explicitly accepts the additional design.

### Refuse unsupported dependency versions

Do not add compatibility branches for a dependency version outside the
declared CoCoA environment. Detect the unsupported major version at the first
shared boundary and stop with one clear error. Supporting a new major version
requires its own migration ticket and validation; a test or gate must not
quietly emulate that future support.

Use neutral audience nouns:

- **the user** for a person running or configuring the library;
- **the reader** for a person reading code or documentation;
- **the Architect**, **the Implementer**, and **the Red Team** when a workflow
  role matters.

Do not use a person's name, personal pronouns, attributed personal quotations,
development dates, ticket chronology, or diary narration in explanatory
Python prose. State the current behavior and its reason directly.

### Explain current code, not the policy patches that produced it

Comments, docstrings, command help, diagnostics, error text, and explanatory
strings tell the reader what the current code does and why that behavior is
necessary. They do not preserve the sequence of requests or reviews that led
to the code.

When behavior changes, replace the old explanation in place. Do not add a
dated correction, `hard user rule`, ticket number, audit wave, review round,
model name, or sentence such as `this now does X`. Keep the lasting reason for
the behavior and remove the development chronology.

NO-GO:

```python
# Hard user rule from the latest review: now reject a dirty worktree.
```

GO:

```python
# Refuse a dirty worktree so uncommitted user files cannot enter the landing.
```

A date remains valid when the program reads or calculates that date, when a
scientific dataset or publication is identified by year, or when a public
compatibility interface contains the date. Words such as `history`, `phase`,
and `previous` remain valid when they name real runtime data or algorithmic
order. They receive NO-GO when they merely narrate how a policy accumulated.

The Architect reviews the complete Python symbol and its related module prose.
GO requires one compatible current explanation, not an old comment followed by
a later exception. The directive lists every chronology-like match and either
removes it or records the concrete scientific, runtime, algorithmic, or
compatibility reason it must remain.

## Architect review before dispatch

The implementation directive receives GO only after the Architect completes
all applicable work below:

1. Read every affected function or class in full, plus nearby callers and
   tests. A diff fragment is not enough.
2. Mark every changed path as **cold** or **hot**.
   - A cold path includes configuration, validation, setup, file handling,
     command parsing, reporting, and one-time object construction.
   - A hot path includes a forward pass, vectorized numerical kernel, batch
     loop, or repeated accelerator operation.
3. Name every changed `path::symbol` and the exact behavior expected from that
   symbol.
4. Resolve code structure, intermediate names, argument style, validation
   order, saved fields, error behavior, comments, docstrings, tests, and
   performance constraints before assigning the Implementer.
5. State the forbidden forms that are relevant to the change. Do not ask the
   Implementer to choose between a compact idiom and an explicit form.
6. Copy every applicable acceptance row from this contract into the directive.
   An unexplained `N/A` receives NO-GO.
7. State the character-change ceiling. A zero ceiling value means unlimited.
   A positive ceiling counts additions plus deletions as Unicode code points
   across the complete tracked ticket diff from its bound full base to clean
   `HEAD`. A replacement counts both sides. An exact-boundary result is
   accepted. The ceiling does not weaken any readability rule.

The Architect returns NO-GO before dispatch when a consequential design or
style choice remains unresolved.

## Required Python shape

### Cold paths

Cold paths use explicit, C-like control flow. Each consequential operation is
visible on a separate line.

| Condition | GO | NO-GO |
|---|---|---|
| Loops | An explicit loop shows state changes and failure points. | A comprehension hides branching, mutation, validation, or more than one transformation. |
| Conditions | A normal `if` block or one simple ternary is immediately readable. | Nested ternaries, chained expression tricks, or a condition embedded in a long call. |
| Function calls | Named parameters identify meanings whenever the callee permits names. | Several unexplained positional arguments force the reader to remember argument order. |
| Staging | Named intermediate variables separate selection, conversion, device movement, and validation. | One nested expression performs several operations with different failure modes. |
| Containers | A multi-item mapping or structured sequence places one item on each line. | A packed literal hides keys, values, shapes, or units. |
| Names | Names state the scientific quantity, representation, unit, or role. | One-letter names outside conventional local mathematics, or names that collide with domain symbols. |
| Errors | Validation happens before mutation, accelerator setup, publication, or destructive file work. | A late check permits partial state or a silent fallback. |

Use an explicit loop instead of a comprehension when the loop performs
validation, mutation, logging, exception handling, or multiple transformations.
A short comprehension is acceptable only when the mapping is direct and the
result is clearer than the equivalent loop.

### Staged numerical operations

Split operations with different meanings into named intermediate variables.

NO-GO:

```python
training = torch.from_numpy(rows[train_indices].astype("float32")).to(device)
```

GO:

```python
training_rows = rows[train_indices].astype("float32")
training = torch.from_numpy(training_rows)
training = training.to(device=device)
```

The staged form exposes row selection, dtype conversion, tensor construction,
and device movement. A future error can name the failing operation.

### Containers and alignment

A dictionary with three or more entries places one key-value pair on each
line. Parentheses align with the opening delimiter when alignment improves
scanning.

```python
knn_options = {
    "k": 8,
    "metric": "euclidean",
    "weights": "distance",
}
```

The same one-item-per-line rule applies to long argument lists, public schema
lists, gate registries, and scientific tuples whose entries need explanation.

### Forbidden dense forms

The following forms receive NO-GO when a clearer ordinary function, loop, or
temporary variable exists:

- assignment expressions using `:=`;
- nested comprehensions;
- several ternaries in one expression;
- a `lambda` that would read more clearly as a named function;
- starred unpacking that hides the source or order of values;
- dense metaprogramming for ordinary configuration or validation;
- chained calls that combine selection, conversion, mutation, and device
  movement;
- silent reads from mutable module-global state;
- shortened names or collapsed logic introduced only to satisfy `--max`.

## Hot-path exception

Readability work must not de-vectorize or slow a numerical hot path. Preserve
vectorized NumPy and Torch operations in forward passes, repeated batch work,
and accelerator kernels.

Complex hot code still requires:

- descriptive names at stable boundaries;
- a plain comment stating the mathematical reason or shape invariant;
- a shape and dtype explanation in the owning docstring;
- benchmark evidence when the execution shape changes;
- a regression test for the numerical result.

The hot-path exception permits dense numerical syntax only when the syntax is
needed for performance or expresses the mathematics more directly. The
exception does not permit unclear setup, validation, or error handling around
the kernel.

## Interfaces, persistence, and failures

### Public and internal interfaces

- Use named parameters whenever the called interface supports names.
- Document unavoidable positional conventions such as `model(x)`, plotting
  coordinates, and `einsum` operands near the call or in the docstring.
- Keep return shapes and units explicit. A changed shape is an interface
  change, not a refactor detail.
- Validate types, finite values, shapes, ranges, and cross-field consistency
  before using a value.

### Saved artifacts

- Save resolved values, including defaults materialized by the code that used
  the values.
- Require every saved key needed to rebuild behavior.
- Never substitute a current code default for a missing saved key.
- Never let a caller redeclare a fact already owned by the artifact.
- Name the missing or conflicting field in the refusal message.

### Failure messages

A failure message receives GO only when the message states:

1. what failed;
2. the observed value or conflicting fields when safe to print;
3. the required condition;
4. the corrective action when a user can correct the input.

`invalid configuration` is insufficient. A message such as
`data.cmb.amplitude_law must be 'none' for this artifact` identifies the
location, accepted value, and repair.

Silent coercion, silent fallback, warning-only corruption, and partial
publication receive NO-GO.

## Teaching text inside Python

Anything printed or returned as explanatory text is documentation. Comments,
docstrings, command help, diagnostics, and errors must satisfy
[`readme-go-no-go.md`](readme-go-no-go.md) in addition to this contract.

### Module and function docstrings

A public or non-obvious function docstring includes the applicable items:

- a first sentence with a subject and verb;
- an `Arguments:` block naming every parameter;
- a `Returns:` block with type, shape, and units;
- raised errors or refusal conditions;
- side effects such as files, device state, or mutable caches;
- a shape-flow diagram for a tensor pipeline;
- a legend defining every shape symbol;
- the scientific meaning of a constant or threshold.

A tensor shape flow may use this form:

```text
parameters [B, P]
    -> trunk [B, H]
    -> correction [B, D]
    -> prediction [B, D]

legend: B=batch rows, P=parameters, H=hidden width, D=data-vector length
```

### Comments

A comment explains a reason, invariant, scientific convention, shape, unit, or
non-obvious failure boundary. A comment does not narrate the next Python line.

NO-GO:

```python
# Add one to the counter.
counter += 1
```

GO:

```python
# Count accepted rows only; rejected rows must not shift checkpoint indices.
accepted_rows += 1
```

## Formatting and naming

- Keep Python lines within 90 columns unless a URL or other indivisible value
  makes the limit harmful.
- Prefer ordinary parentheses over backslash continuation.
- Use one logical operation per line on cold paths.
- Use scientific names with units where ambiguity is possible, such as
  `radius_mpc_over_h`, `redshift_grid`, or `covariance_cholesky`.
- Avoid a local name that changes meaning within one function.
- Keep schema dictionaries and option tables visibly aligned by structure,
  not by fragile runs of spaces.
- Use helper constructors for repeated model specifications instead of
  duplicating `{class, options}` assembly.

## Mandatory directive block

Every Architect directive for a Python change records:

```text
### Python style plan
- Scope: <tracked Python paths and exact symbols>
- Hot/cold classification: <one classification per changed path>
- Required code shape: <loops, temporaries, arguments, containers, names>
- Forbidden forms: <applicable hard NO-GO forms>
- Interfaces: <arguments, returns, shapes, dtypes, units, saved fields>
- Failure behavior: <validation order and exact refusal requirements>
- Teaching text: <docstrings, comments, help, diagnostics, errors>
- Tests and static checks: <commands and expected discriminating result>
- Performance evidence: <benchmark or N/A with reason>
- Character-change ceiling: <nonnegative value and counting rule>
```

A missing row, delegated design choice, or unexplained `N/A` gives NO-GO.

## Mandatory evidence block

The Implementer returns:

```text
### Python style evidence
- Changed symbols: <path::symbol list>
- Style rows: <evidence for every applicable contract row>
- Tests: <exact commands, return codes, and important output>
- Static checks: <format, syntax, abstract syntax tree (AST), or targeted scans>
- Performance: <before/after measurement or justified N/A>
- Prose contract: <README/Python prose GO/NO-GO evidence>
- Character count: <added plus deleted characters>
- Deviations or blockers: <none, or exact unresolved item>
```

Checkboxes without raw commands or inspected code do not count as evidence.

## Architect review before final verdict

The Architect completes all steps below:

1. Reopen this contract.
2. Read every changed function or class in full, not only the diff.
3. Compare each changed path with the directive's hot/cold classification.
4. Inspect every applicable style row and every claimed exception.
5. Run the listed tests and static checks from the accepted worktree.
6. Compare performance when a hot path or allocation pattern changed.
7. Apply the README/Python prose contract to explanatory text.
8. Count added plus deleted characters and reject obfuscation regardless of
   the configured ceiling.

Use this verdict record:

```text
### Python change style verdict
| Condition | GO or NO-GO | Raw evidence |
|---|---|---|
| Cold-path control flow | ... | ... |
| Hot-path preservation | ... | ... |
| Named staging and arguments | ... | ... |
| Containers, names, and formatting | ... | ... |
| Interfaces and persistence | ... | ... |
| Validation and failures | ... | ... |
| Docstrings, comments, and user text | ... | ... |
| Tests and performance | ... | ... |
| Character ceiling without obfuscation | ... | ... |

Verdict: GO or NO-GO
Required repair for every NO-GO: <path::symbol, exact change, rerun command>
```

The final verdict is GO only when every applicable row is GO. A NO-GO verdict
must give a concrete repair and the command that will recheck the repair.

## Hard NO-GO conditions

Any condition below requires NO-GO:

- a consequential code or style choice was left to the Implementer;
- a candidate adds, copies, retargets, or broadens a monkey patch;
- a changed path lacks a hot/cold classification;
- a dense expression hides multiple operations or failure points;
- code was minified, names were shortened, or explanations were removed to
  satisfy a character ceiling;
- a hot-path rewrite lacks numerical and performance evidence;
- a save/load change relies on a default that was not persisted;
- validation happens after mutation, publication, or expensive setup without
  a proven need;
- an error silently falls back, coerces, or hides the correction;
- explanatory text is personal, development-dated, vague, undefined, narrates
  policy or review history, or uses chronology without a scientific, runtime,
  algorithmic, or compatibility need;
- an applicable acceptance row lacks raw evidence;
- the Implementer or Red Team edited this contract or another permanent note.

No deadline, model limitation, character budget, or passing behavior test
overrides a hard NO-GO condition.
