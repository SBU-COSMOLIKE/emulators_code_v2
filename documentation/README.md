# Long-form documentation

This folder holds explanations that are useful for understanding or
maintaining the library but are too detailed for the main READMEs. Start with
the document whose question matches the topic. A PDF is ready to read; its
LaTeX source is the editable version.

## Complete library guides

| Question | Read | Source |
| --- | --- | --- |
| How does a cosmological sample move through the complete emulator library? | [CoCoA SONIC emulator guide](emulator_code_guide.pdf) | [LaTeX](emulator_code_guide.tex) |
| What does each of the forty acceptance gates prove, and how is the board run safely? | [Gate-board companion](emulator_gate_board.pdf) | [LaTeX](emulator_gate_board.tex) |
| How do I read the AI ticket-system code under `ai/tools/` — the mailbox, roles, guards, and landing machinery? | [CoCoA-Flow guide](cocoa_flow_guide.pdf) | [LaTeX](cocoa_flow_guide.tex) |

The emulator guide is a full-library manual. It follows generated data through
training, saved artifacts, scientific checks, and inference. The CoCoA-Flow
guide is the matching manual for the workflow machinery: it follows one
requested change from the user's request file to the accepted commit on
`main`, one `ai/tools/` module at a time, and closes with a staged reading
route through the code. Operating instructions stay in
[`ai/README.md`](../ai/README.md) and [`ai/tools/README.md`](../ai/tools/README.md);
the guide teaches the implementation behind them.

## Focused guides

| Question | Read | Source |
| --- | --- | --- |
| How does an Architect-approved candidate become a safe commit on `main`? | [Candidate-to-landing guide](candidate_to_landing.pdf) | [LaTeX](candidate_to_landing.tex) |
| Why does the cosmic-shear emulator divide out an analytic approximation? | [Analytic-scaling note](analytic_scaling.pdf) | [LaTeX](analytic_scaling.tex) |
| How do the shipped activation functions change a signal? | [Activation-functions notebook](activation_functions_teaching.nb) | Wolfram notebook |

A focused guide owns one bounded reader question. It does not repeat the
complete library manual.

## Figures and build helpers

- [`figures/`](figures/) contains vector figures used by the emulator guide.
- [`artwork/`](artwork/) contains the guide's frontispiece.
- [`make_figures.py`](make_figures.py) rebuilds the numbered vector figures.
- [`render_readme_previews.py`](render_readme_previews.py) creates the selected
  PNG previews used outside the PDF.

## Before adding another document

Search this catalog, the existing LaTeX sources, and the relevant README
sections before creating a file. If an existing document already answers the
same reader question, update that document or improve the link to it. Two
guides that teach the same mechanism will eventually disagree.

Create another focused guide only when both conditions are true:

1. the topic is important for understanding or maintaining the library; and
2. a complete explanation would make the relevant README too long.

The README should keep a short introduction and link to the one guide that
owns the detailed explanation.

## Build and inspect the candidate-to-landing guide

From the repository's top folder, run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=documentation \
  documentation/candidate_to_landing.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=documentation \
  documentation/candidate_to_landing.tex
```

Expected result: both commands exit with code `0` and
`documentation/candidate_to_landing.pdf` contains five readable pages. Render
and inspect every page before committing a changed PDF; a successful LaTeX
exit alone cannot detect clipped or overlapping content.

## Build and inspect the CoCoA-Flow guide

From the repository's top folder, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=documentation \
  documentation/cocoa_flow_guide.tex
```

Expected result: the command exits with code `0` and
`documentation/cocoa_flow_guide.pdf` contains 23 readable pages. Run it from
the top folder: the frontispiece path inside the source is relative to the
repository root, so a build started inside `documentation/` stops with a
missing-file error. The same rendered-page inspection rule as above applies
before committing a changed PDF.
