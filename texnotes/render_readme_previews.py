#!/usr/bin/env python3
"""Render browser previews for the vector figures reused by ``README.md``.

``make_figures.py`` owns the scientific content and writes vector PDF files.
GitHub displays PNG images directly, so this companion script renders only
the three PDFs embedded in the root README.  The PDF remains the source; the
PNG is a fixed-resolution preview of that source.

Run both commands from the repository root::

    python texnotes/make_figures.py
    python texnotes/render_readme_previews.py

The renderer is Poppler's ``pdftoppm`` command.  Install Poppler or place
``pdftoppm`` on ``PATH`` before running this script.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
PREVIEW_DPI = 180

# Each stem names both ``<stem>.pdf`` and the generated ``<stem>.png``.
# This is a tuple, not a generator, so the complete preview inventory is
# visible in one place and can be checked before any command runs.
README_FIGURE_STEMS = (
    "fig01_ownership_chain",
    "fig03_training_cloud",
    "fig07_activations",
)


def render_preview(
    renderer: str,
    figure_stem: str,
) -> Path:
    """Render one PDF into one PNG without changing its aspect ratio.

    Arguments:
        renderer: Absolute path to the ``pdftoppm`` executable.
        figure_stem: File name shared by the source PDF and output PNG.

    Returns:
        Path to the PNG that ``pdftoppm`` created.
    """

    source_pdf = FIGURE_DIR / f"{figure_stem}.pdf"
    output_root = FIGURE_DIR / figure_stem
    output_png = FIGURE_DIR / f"{figure_stem}.png"

    if not source_pdf.is_file():
        raise FileNotFoundError(
            f"missing vector source: {source_pdf}"
        )

    command = [
        renderer,
        "-png",
        "-r",
        str(PREVIEW_DPI),
        "-singlefile",
        str(source_pdf),
        str(output_root),
    ]
    subprocess.run(
        command,
        check=True,
    )

    if not output_png.is_file():
        raise RuntimeError(
            f"renderer completed without creating {output_png}"
        )
    return output_png


def main() -> None:
    """Render every figure that the root README embeds."""

    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise RuntimeError(
            "README preview rendering requires Poppler's pdftoppm on PATH"
        )

    for figure_stem in README_FIGURE_STEMS:
        output_png = render_preview(
            renderer=renderer,
            figure_stem=figure_stem,
        )
        print(f"wrote {output_png}")


if __name__ == "__main__":
    main()
