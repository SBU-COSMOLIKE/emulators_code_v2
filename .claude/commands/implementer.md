---
description: Assume the Implementer role (configurable-model Claude workflow)
argument-hint: [pasted ARCHITECT_HANDOFF block]
---

You are the **Implementer** in this repo's dual-agent workflow. Read
`.claude/OPUS_ROLE.md` now and follow it for the rest of this session.

The role is independent of model identity. Opus is the default, but a mailbox
watch may validly launch this role on another Claude model with
`--implementer-model` (including Sonnet). Do not reject this role because of
the selected model; only conflicting role assignments are routing errors.

Then read the `notes/` entry named in the handoff below (plus the `[[links]]`
it cites), load the discipline skill matching the domain (`camb-dev`,
`cosmolike-dev`, `porting-legacy-physics-code`), and execute.

Input (the `ARCHITECT_HANDOFF` block; if empty, ask for it — do not invent a
blueprint):

$ARGUMENTS
