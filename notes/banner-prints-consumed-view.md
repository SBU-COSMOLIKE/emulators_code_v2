---
name: banner-prints-consumed-view
description: "STANDING DESIGN DIRECTIVE (user, 2026-07-06): every printed configuration surface (print_design, run banners, spec dumps) shows the RESOLVED, CONSUMED view — what the run will actually execute — never the raw YAML. Two same-day user reports fixed one bug each but exposed the class: (1) a single-phase resmlp run printed '(two-phase: 2000 trunk + -1000 head)' from raw trunk_epochs (D-P2); (2) the model-spec line dumped raw cnn:/trf: blocks that TemplateMLP ignores (D-P3). In both cases the CONSUMPTION was already right (resolve_phase_args demotes; build_specs ignores inactive head blocks so one YAML serves every architecture) — only the display lied. Rule going forward: displays reuse the SAME resolution functions the execution path uses (resolve_phase_args, ARCH_HEAD filtering, validate_loss), so display and behavior cannot drift; every new config feature's spec must state its banner rendering in resolved form; every Architect audit checks display surfaces alongside execution wiring (GP-B checked train() and missed print_design — the audit gap that let D-P2 through)."
metadata:
  node_type: memory
  type: project
---

# Standing directive: banners print the consumed view, never the raw YAML

User directive 2026-07-06, issued after the second display bug of the
day: "it should only print what is necessary for the chosen model" —
generalized here into an architecture rule at the user's request
("update your architecture directive").

## The rule

1. **Display = resolution output.** Any printed configuration surface
   (print_design, phase banners, spec dumps, notices) renders the
   configuration AS THE RUN WILL CONSUME IT: phases resolved against
   the model's capability, model blocks filtered to the chosen
   architecture (ARCH_HEAD), loss blocks in canonical form. The raw
   YAML belongs in the YAML file; the banner's job is the truth about
   this run.
2. **Displays reuse the execution path's resolution functions** —
   resolve_phase_args, the ARCH_HEAD selection, validate_loss — never
   a parallel re-implementation. Shared code is what makes drift
   structurally impossible.
3. **Tolerant consumption, truthful display.** The shared-YAML
   philosophy (irrelevant blocks ignored so one file serves resmlp /
   rescnn / restrf and single- / two-phase models) stays; the display
   of only-what-is-consumed IS the notice that the rest was ignored.
4. **Spec obligation:** every future feature that adds or moves config
   keys states, in its spec, what the banner prints in resolved form
   (this joins verbatim numerics and validation gates as a mandatory
   blueprint section).
5. **Audit obligation:** every Architect re-audit checks display
   surfaces alongside execution wiring. The lesson: GP-B verified
   train()'s resolution and never looked at print_design — that gap
   shipped D-P2.

## The two instances that defined the class (both user-found)

- D-P2: single-phase resmlp + trunk_epochs 2000 / nepochs 1000 printed
  `(two-phase: 2000 trunk + -1000 head)` — raw arithmetic on keys the
  run demotes ([[resolve-phase-args-single-phase]] D-P2).
- D-P3: `model spec: {...}` dumped the raw model dict including the
  cnn: and trf: blocks TemplateMLP ignores (build_specs' deliberate
  inactive-head tolerance) — the display contradicted the consumption
  (fix folded into the D-P2v2 delta, same note).

## The implementation idiom (user's tip, 2026-07-06): the class prints
## itself

Central filtering in print_design would duplicate the consumed-view
knowledge build_specs already holds. Instead, each design class owns
its description: a `head_block` class attribute (None | "cnn" | "trf")
plus one shared `describe_spec(model_block)` classmethod that renders
name / ia / mlp / activation / the class's OWN head block /
compile_mode — and nothing else. print_design's model-spec line
becomes a pure delegation:

    self.log(f"model spec: {self.model_cls.describe_spec(ta['model'])}")

Deeper win: `head_block` on the class becomes THE single source of
head-knowledge — ARCH_HEAD (the experiment-side name->head dict that
build_specs / build_geometry consult) is retired in favor of
model_cls.head_block, so consumption, geometry, and display all read
one attribute that lives with the code it describes. A new
architecture that forgets head_block fails loudly at class definition
(the shared describe/build machinery requires it), not silently in a
banner.
