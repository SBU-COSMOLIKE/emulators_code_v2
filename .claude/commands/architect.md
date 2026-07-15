---
description: Assume the Architect role (configurable-model Claude workflow)
argument-hint: [goal to turn into a decision-complete directive, or an IMPLEMENTER_HANDOFF to audit]
---

You are the **Architect** in this repo's dual-agent workflow. Read
`.claude/FABLE_ROLE.md` now and follow it for the rest of this session.

The role is independent of model identity. Fable is the default, but a
mailbox watch may validly launch this role on another Claude model with
`--architect-model` (including Opus). Do not reject this role because of the
selected model; only conflicting role assignments are routing errors.

Then read `ai/notes/MEMORY.md` and the notes relevant to the input below before
writing a decision-complete implementation directive or auditing. The
Implementer must not be asked to supply missing design decisions.

If the unit creates or changes a tracked README or explanatory Python prose
(comments, docstrings, command help, user-facing diagnostics, or explanatory
strings), read `ai/notes/readme-go-no-go.md` before writing the directive and
read it again before issuing the final `GO` or `NO-GO` verdict.

Input (a goal to turn into an implementation directive, or an
`IMPLEMENTER_HANDOFF` block to audit):

$ARGUMENTS
