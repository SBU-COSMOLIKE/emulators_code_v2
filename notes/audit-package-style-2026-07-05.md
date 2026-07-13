---
name: audit-package-style-2026-07-05
description: "Architect whole-tree style/doc audit (2026-07-05) of emulator/ + 5 drivers + README + example_yamls against the pytorch-teaching-style skill and the notes/ conventions. PASS on the structural axes (no data-global reads, keyword names all valid, probe generalization, shared-budget threading, weight-decay split, spec dicts, driver flag parity). Fix list handed to the Implementer: user-directed deletion of dead NLATemplateMLP + NLAInputGeometry; training.py docstring-vs-code drift (vidx/idx, flat-vs-nested data dict, missing device arg + unusable 'gpu' string default, undocumented kappa); plot_xi docstring bugs + gist_rainbow red+green default; README one-generation-behind (CNNBlock stale, FiLMGenerator/print_design/audit_devices/plot_sweep_curve/TemplateRes* missing, 2 broken chi2 anchors, model.mlp/activation/trf undocumented); systematic 07-04 regressions (~50 caps-emphasis in .py + ~39 in YAMLs, ~315 double-dash, 7 legend-less diagrams); PS: jargon gaps; Arguments retrofit (PCE zero blocks); provenance/paren/blank-line/named-param passes. ARCHITECT_HANDOFF block at the end. Side-finding: the skill copy under june2026/claude_skills/ is stale (Jun 18); the live one is ~/data/claude_skills/pytorch-teaching-style/SKILL.md."
metadata:
  node_type: memory
  type: project
---

# Whole-package style + documentation audit (2026-07-05, Architect)

Scope: every `.py` in the tree (`emulator/` incl. `IA/`, `PCE/`, `parallel/`;
the five root drivers), `README.md`, and `example_yamls/`, audited against the
pytorch-teaching-style skill plus [[py-module-style-conventions]],
[[docstrings-formal-arguments-block]], [[no-global-variables-in-functions]],
[[plots-no-red-green]], [[construction-via-spec-dicts]],
[[weight-decay-only-on-weight-matrices]], [[probe-generalization-bugs]],
[[shared-budget-across-sequential-calls]].

Method: mechanical scans run by the Architect (AST parse, symtable
data-global scan, keyword-vs-signature validation, >90-col scan, comprehension
AST scan, all-caps + double-dash token scans over comments/docstrings, diagram
legend scan, mutable-default scan, jargon-vs-PS scan), then six parallel
review subagents (one file group each, tight per-rule spec, file:line
evidence), then Architect spot-verification of every load-bearing claim
against the source before it entered this note.

Side-finding for the user (not the Implementer): the skill copy at
`~/data/COCOA/june2026/claude_skills/pytorch-teaching-style/SKILL.md` is the
stale Jun-18 6 KB version; the live 69 KB Jul-4 version lives at
`~/data/claude_skills/pytorch-teaching-style/SKILL.md`. Re-sync so future
sessions load the one with the shape-flow spec, the .py-package section, and
the no-globals section.

## What passed (verified clean, no action)

- Every module parses; module docstrings present in all 26 files.
- Zero silent data-global reads (symtable free-name scan, whole tree).
- Zero invalid keyword names: every call-site keyword matches a real in-repo
  signature (AST cross-check).
- Probe generalization: global `dest_idx` used everywhere; the xi-only paths
  (`build_shear_angle_map`, the `rescale` path of `make_chi2`) are
  assert-guarded so they fail loudly, not silently.
- Shared-budget threading across sources is correct
  (`budget - used_tr` handed to the second `_build_loaders_one` call).
- Weight-decay `ndim >= 2` split, spec-dict construction, keyed `**splat`
  specs, `nn.ModuleList` use, classmethod constructors: all conform.
- All five drivers: header-comment flag list == argparse flag list, exactly,
  both directions; top-level functions carry complete `Arguments:` blocks.
- `experiment.py` block-key docs match the code's actual key reads one-for-one
  (data block 15 keys; `MODEL_BLOCK_KEYS` incl. `kappa` under focus).
- `example_yamls/`: pure block style, no inline flow; the
  `[default, min, max, kind]` range convention used as documented.

## Findings (by priority)

### P1 — user-directed deletion (2026-07-05 directive: avoid code bloat)

1. `NLATemplateMLP` (`emulator/IA/emulator_designs.py:11`) and
   `NLAInputGeometry` (`emulator/geometries_parameter.py:252`) are dead code:
   the `MODELS` registry wires `TemplateMLP`/`TemplateResCNN`/`TemplateResTRF`
   with `AmplitudeFactorGeometry`; nothing constructs the two NLA-specific
   predecessors, and `NLAInputGeometry` has no `state()`/`from_state`, so
   `save_emulator` would crash on it anyway. Delete both classes and every
   stale mention: `IA/emulator_designs.py` module docstring + the
   `NLAInputGeometry.encode` input-layout reference (line ~19),
   `geometries_parameter.py` module docstring (line ~7), any README hits.

### P2 — docstring-vs-code drift (docs actively wrong)

2. `training.py` `eval_val`: Arguments says key `vidx`; code reads
   `data["idx"]` (line 510). Fix the docstring.
3. `training.py` `training_loop_batched`: Arguments describes a flat
   `data` dict with `tidx`; code reads nested `data["train"]["load_C"/"load_dv"/
   "idx"/"load"]` and `data["val"]` (704-711, 830, 940). Rewrite the entry.
4. `training.py` `run_emulator`: Arguments omits `device`; the signature
   default `device='gpu'` is a plain string that would crash every
   `device.type` branch. Make `device` a required parameter (all callers pass
   a real `torch.device`; verify call sites) and document it.
5. `training.py` `focus_opts`: code consumes `kappa`
   (`focus_opts.get("kappa", 1.0)`, line 740) but neither focus_opts docstring
   in this file lists it. Add it (experiment.py already documents it).
6. `training.py:309`: `_walk_train_args` docstring still says "the
   comprehensions copy" — the body is an explicit loop now. `training.py:706`:
   comment references `C0/dv0`, symbols that exist nowhere in the file.
7. `experiment.py` `from_yaml`: documents `models` as "name -> class registry";
   the registry is keyed by `(name, ia)` tuples (from_config's doc is correct).
8. `plotting.py` `plot_xi`: docstring says `pm = "p", "m", or "pm"` but the
   body tests `pm > 0` (int); Returns says "figure (0 on malformed input)" but
   the code returns `(fig, axes)` or None under `show`. Fix the docstring only
   (the body keeps its declared byte-faithful port style).

### P3 — accessibility (rule is categorical, carve-out does not cover it)

9. `plotting.py` `plot_xi` default `cmap='gist_rainbow'` colors curves
   `cm(x/len(xi))` — red and green lines in one panel, plus a same-map
   colorbar. Change the default to a sequential colorblind-safe map
   (`"viridis"`, matching lines 152 and 912). One-token default change; body
   otherwise untouched.

### P4 — README one generation behind the code

10. Deleted `CNNBlock` still listed (README lines 58 and 336); line 336 also
    omits `BinLinear`/`TRFBlock`/`FiLMGenerator`; line 337 omits `ResTRF`.
11. Two broken anchors: lines 93 and 779 link
    `#7-appendix-the-chi2-metric-mahalanobis`; the chi2 appendix is section 8.
12. Per-file appendix omissions: `FiLMGenerator` + `rescale_kernel_size`
    (building_blocks), `audit_devices` (training), `print_design`
    (experiment), `plot_sweep_curve` (plotting), `TemplateResCNN` /
    `TemplateResTRF` / `nla_coeffs` (IA).
13. Nested model schema half-documented: `model.mlp:`, `model.activation:`,
    `model.trf:` sub-blocks have zero README hits (`model.cnn` knobs are
    documented); the step-5 ia map names resmlp/rescnn only.
14. State in words (README §6 or example_yamls headers) that sweep_ntrain and
    bakeoff_activation reuse the train_single YAML unchanged.
15. Driver-header staleness: `train_single` docstring + header say "ResMLP or
    ResCNN" (its own line 46 lists restrf); bakeoff header line 42 same;
    bakeoff also hand-rolls a two-line banner instead of
    `exp.print_design()` — the 07-04x milestone says the banner is shared by
    all drivers; wire it in.

### P5 — systematic style regressions (concentrated in the 07-04 layer)

16. All-caps emphasis: ~50 instances in `.py` comments/docstrings (top:
    training.py ~15, IA/emulator_designs.py ~12, experiment.py ~8, sweep/tune
    drivers ~10; e.g. NEVER/SAME/BEFORE/ARE/ONE/FRESH/BEST) + ~39 in
    example_yamls comments (ONLY/OFF/AND/BEST/ONE/CLASS/FAMILY/LAST...).
    The rule allows caps only for acronyms/notation and the sanctioned
    `⚠️ WARNING` marker. The 07-01 de-caps pass was regressed by the 07-04
    work; re-run it over the whole tree including the YAMLs.
17. Double-dash punctuation: ~315 ` -- ` instances package-wide (training.py
    46, IA/emulator_designs.py 40, experiment.py 33, loss_functions.py 30...).
    Rule: commas/colons/parens/"i.e."; "eliminate most, do not obsess".
    Target: below ~30 package-wide, zero in module docstrings.
18. Shape-flow diagrams missing `(legend: ...)` (7 sites; 11 sibling diagrams
    have proper legends, so this is drift): emulator_designs.py module +
    ResMLP (undefined `B`), building_blocks `rescale_kernel_size`,
    experiment.py module, loss_functions `_reduce` (undefined `eps`, `B`),
    training.py `eval_val` (undefined `n_val`) + `run_emulator`. Also
    undefined symbols inside existing text/diagrams: `G` in the ResTRF class
    docstring (legend defines n_bins, not G), the `0.8` headroom factor in
    batching.py's module diagram, `ram_frac` in data_staging.py's diagram
    legend, `L` in scheduling.py's lane diagram.
19. Jargon used with no in-file definition (rule 7: PS: line or inline, per
    file): IA/emulator_designs.py + IA/loss_functions.py (whitened, encoded,
    squeezed), experiment.py (loaders, dumps, memmapped, encoded, whiten),
    diagnostics.py (whitened), plotting.py (whitened, dump), batching.py
    (memmap, dump missing from its PS), data_staging.py (loader, whitening
    missing from its PS), PCE/loss_functions.py (whitened, loader missing),
    analytics.py (squeezed), training.py PS missing encoded/dump/memmap, all
    five drivers (dump/memmap/loaders/resident/whitened as used).
    Canonical definitions live in [[py-module-style-conventions]] rule 7.
20. `Arguments:` retrofit: `PCE/loss_functions.py` has zero blocks (its IA
    twin has 13) — retrofit every method; `PCE/emulator_designs.py`
    `select_lars_loo` (the `patience` knob is documented nowhere),
    `_pce_deg_tuples`, `PCEEmulator.__init__`/`forward`; the short geometry
    transform methods (`whiten`/`unwhiten`/`encode`/`decode` families in both
    geometry files, `LogParamGeometry.__init__`, `from_state` overloads);
    `loss_functions.py` `chi2`, RescaledChi2/ResidualBaseChi2 method families;
    `plotting.py` `_finish`, `_cut_role`; driver closures (`objective`,
    `log_trial`, `job_tokens`, `on_result`).
21. Cross-module provenance comments (`# fn (module.py): what it does`)
    largely absent at call sites in: emulator_designs.py (~12 sites: ResBlock,
    Affine, activation_fcn, FiLMGenerator, TRFBlock, rescale_kernel_size),
    IA/emulator_designs.py (same families), IA/loss_functions.py +
    PCE/loss_functions.py (CosmolikeChi2.chi2/.loss, geom.encode/.decode),
    loss_functions.py `_analytic_R` (analytics.py), training.py
    `build_loaders` (batching.py) + `anneal_value`, experiment.py
    `TemplateFactoredChi2` (IA/loss_functions.py), diagnostics.py
    `eval_source_chi2` (training.py) x3, drivers (`save_emulator`,
    `save_learning_curves`, `resolve_cocoa_config`, `suggest_train_args`,
    scheduling imports in sweep_hyperparam).
22. Paren-alignment lapses cluster in: `def` signatures packing 3-6 params per
    line (ResCNN/ResTRF/TRFBlock/GroupedCNNBlock/ParallelResCNN inits,
    analytics + batching + data_staging + scheduling + diagnostics defs,
    run_emulator/eval_val), `register_buffer(` two-args-on-one-line (~10
    sites in the two designs files), experiment.py's systematic 2-space
    hanging indent on every major call (load_source, from_covmat,
    from_cosmolike, make_chi2, build_run_specs, run_emulator,
    eval_source_chi2), unpack tuples in drivers (3+2 per line), and the
    scattered sites each reviewer listed. Ordinary call sites already comply;
    normalize the listed families.
23. Blank-line grouping walls: experiment.py `from_config` (56 lines),
    `build_specs` (52), `print_design` (32); training.py chunk/batch loop
    (48), eval_val stream loop (26); IA `TemplateResCNN.forward` (44) +
    `__init__` (32), `TemplateResTRF.forward` (35); ResTRF.forward (34);
    scheduling `run_gpu_pool` launch block (27); PCE `from_training` fit loop
    (26); driver argparse runs (55-75, table-like: group, do not force);
    train_single diagnostics branch (73).
24. Named-parameter pass (positional where keyword-able, or missing the naming
    comment): `ResBlock(int_dim_res, **block_opts)` `size=` (6 sites across 3
    files), `BinLinear(n_tokens, dim, dim)`, `TRFBlock(self.max_bin, ...)`
    `dim=`, `torch.full((...), float(gate_init))` `fill_value=`,
    `ci.init_binning(ntheta, tmin, tmax)` + `ci.init_data_real(...)`,
    `chi2fn.encode(dv)` / `param_geometry.encode(...)` (batching, training,
    diagnostics), optuna `suggest_int/float` `low=/high=`,
    `np.geomspace(..., num=)`, `cocoa_output(...)`, `lpt_assign` /
    `even_assign` / `set_by_path` / `suggest_train_args`,
    `get_device_properties(0)` bare index, `plt.subplots(1, 2)`,
    plotting private panel helpers, `torch.linspace(-1.5, 1.5, K)`,
    reduction dims (`.sum(-2)`, `.sum(1)`), `torch.save(sd, emul_path)`,
    `Xm.clamp(-1.0, 1.0)`, ctx.Process args tuples (naming comments).
25. Width: 6 driver-header example-command lines at 95-102 cols (fit 90 by
    reducing the comment indent). Comprehensions: 4 construction-time
    `all(pm == ... for pm in ...)` generator expressions
    (emulator_designs.py:371-372, IA/emulator_designs.py:461-462) -> explicit
    loops per rule 13 (non-hot).
26. Module-docstring prose: ~10 files open with verbless noun-phrase
    fragments (geometries_output.py "The output side: ...", analytics,
    batching, data_staging, diagnostics, plotting, IA both + __init__, PCE
    __init__, tune/sweep_ntrain/bakeoff drivers, training.py's double
    fragment, experiment.py first line). Give each a subject + verb opener;
    one sentence is enough.

### P6 — verify-then-fix (needs the workstation, cosmolike not available here)

27. `geometries_output.py`: `ci.init_probes(possible_probes=probe)` is called
    twice (lines 209 and 217) with no comment. On the workstation, verify
    whether the second call is load-bearing (e.g. the redshift init resets
    probe state). If yes: comment why, citing the observed behavior. If no:
    delete the duplicate. Do not guess statically.
28. `batching.py:357`: streaming regimes call `batches_per_load(budget=budget)`
    whose internal resident term counts model + Cinv but not the
    already-staged `enc_params` (small, but the running-remainder rule).
    Either subtract it or add the honest both-directions accounting comment.
29. `IA/emulator_designs.py:458`: the branch-cut check samples only each
    bin's first kept element (`pm_kept[start]`); a mixed-pm bin would pass
    silently. Either check every element of each bin (construction-time, not
    hot) or document the homogeneity premise at the assert.

## Validation gates (the Implementer runs these; the Architect re-audits)

- G1 `grep -rn "NLATemplateMLP\|NLAInputGeometry" --include="*.py" . README.md`
  -> 0 hits; `python -c "import emulator, emulator.IA, emulator.PCE,
  emulator.parallel"` clean.
- G2 the eight P2 drift items fixed verbatim (re-read the named lines).
- G3 all-caps scan (token walk over comments+docstrings, words of 3+ caps,
  minus acronyms/constant-names/the WARNING marker) -> 0 emphasis hits in
  `.py` and `example_yamls/`.
- G4 ` -- ` count in comments/docstrings < 30 package-wide, 0 in module
  docstrings.
- G5 every docstring containing a `│`/`▼` diagram ends with `(legend: ...)`;
  the named undefined symbols (B, eps, n_val, G, 0.8 headroom, ram_frac, L)
  are defined.
- G6 jargon-vs-PS scan: every file listed in item 19 defines the terms it
  uses (PS: line or inline).
- G7 no line > 90 cols in any `.py`; AST comprehension scan -> 0 non-hot hits.
- G8 README: `grep -c CNNBlock README.md` -> 0; both chi2 anchors point at
  `#8-...`; the item-12 functions appear in their appendices; `model.mlp` /
  `model.activation` / `model.trf` documented; bakeoff calls
  `exp.print_design()`.
- G9 whole tree `py_compile` clean + an AST-minus-docstrings diff proving the
  doc passes (16-26) touched only comments/docstrings except the sanctioned
  code edits (deletions of item 1, `device` signature of item 4, cmap default
  of item 9, loops of item 25, alignment/blank-line whitespace, item 27/28/29
  outcomes).
- G10 (workstation) item 27 resolved with evidence (chi2 identical or the
  explaining comment cites what broke).

## Resume state (Implementer appends below)

(none yet)
