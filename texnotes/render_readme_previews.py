#!/usr/bin/env python3
"""Render browser previews for the vector figures reused by ``README.md``.

``make_figures.py`` owns the scientific content and writes vector PDF files.
GitHub displays PNG images directly, so this companion script renders only
the three PDFs embedded in the root README.  The PDF remains the source; the
PNG is a fixed-resolution preview of that source.

Run both commands from the repository root::

    python texnotes/make_figures.py
    python texnotes/render_readme_previews.py

The preferred renderer is Poppler's ``pdftoppm`` command.  When it is absent,
Ghostscript's ``gs`` renders the same PDF at the same resolution.  Either way
the committed PDF is the one source: the PNG is a rasterization of it, never a
second drawing of the same figure.  Install one of the two, or place it on
``PATH``, before running this script.
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
    "fig12_three_session_loop",
)


def find_renderer() -> tuple[str, str]:
    """Locate a PDF rasterizer, preferring Poppler over Ghostscript.

    Returns:
        The executable's absolute path, and the name this script uses to
        decide which command line that executable expects.

    Raises:
        RuntimeError: Neither renderer is on ``PATH``.
    """

    poppler = shutil.which("pdftoppm")
    if poppler is not None:
        return poppler, "pdftoppm"

    ghostscript = shutil.which("gs")
    if ghostscript is not None:
        return ghostscript, "ghostscript"

    raise RuntimeError(
        "README preview rendering requires Poppler's pdftoppm or "
        "Ghostscript's gs on PATH"
    )


def preview_command(
    renderer: str,
    kind: str,
    source_pdf: Path,
    output_png: Path,
) -> list[str]:
    """Build the rasterizing command line for the located renderer.

    Both renderers are asked for the same thing: page one of the vector
    source, at ``PREVIEW_DPI``, antialiased, with the aspect ratio the PDF
    already fixes.

    Arguments:
        renderer: Absolute path to the executable.
        kind: Either ``"pdftoppm"`` or ``"ghostscript"``.
        source_pdf: The vector figure that ``make_figures.py`` wrote.
        output_png: The preview this command must create.

    Returns:
        The argument list to hand to ``subprocess.run``.
    """

    if kind == "pdftoppm":
        # pdftoppm appends its own ".png", so it takes the stem, not the file.
        return [
            renderer,
            "-png",
            "-r",
            str(PREVIEW_DPI),
            "-singlefile",
            str(source_pdf),
            str(output_png.with_suffix("")),
        ]

    return [
        renderer,
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-dFirstPage=1",
        "-dLastPage=1",
        "-sDEVICE=png16m",
        "-dTextAlphaBits=4",
        "-dGraphicsAlphaBits=4",
        f"-r{PREVIEW_DPI}",
        f"-sOutputFile={output_png}",
        str(source_pdf),
    ]


def render_preview(
    renderer: str,
    kind: str,
    figure_stem: str,
) -> Path:
    """Render one PDF into one PNG without changing its aspect ratio.

    Arguments:
        renderer: Absolute path to the rasterizing executable.
        kind: Which command line that executable expects.
        figure_stem: File name shared by the source PDF and output PNG.

    Returns:
        Path to the PNG that the renderer created.
    """

    source_pdf = FIGURE_DIR / f"{figure_stem}.pdf"
    output_png = FIGURE_DIR / f"{figure_stem}.png"

    if not source_pdf.is_file():
        raise FileNotFoundError(
            f"missing vector source: {source_pdf}"
        )

    subprocess.run(
        preview_command(
            renderer=renderer,
            kind=kind,
            source_pdf=source_pdf,
            output_png=output_png,
        ),
        check=True,
    )

    if not output_png.is_file():
        raise RuntimeError(
            f"renderer completed without creating {output_png}"
        )
    return output_png


def main() -> None:
    """Render every figure that the root README embeds."""

    renderer, kind = find_renderer()
    print(f"renderer: {renderer} ({kind})")

    for figure_stem in README_FIGURE_STEMS:
        output_png = render_preview(
            renderer=renderer,
            kind=kind,
            figure_stem=figure_stem,
        )
        print(f"wrote {output_png}")


if __name__ == "__main__":
    main()
