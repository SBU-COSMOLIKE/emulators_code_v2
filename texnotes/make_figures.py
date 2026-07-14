#!/usr/bin/env python3
"""Build every vector figure used by ``emulator_code_guide.tex``.

The manuscript uses the generated PDF files in ``texnotes/figures`` directly.
This script uses ReportLab's PDF drawing primitives, so every line, curve,
arrow, and label remains vector artwork.  No screen capture or raster image is
embedded in the paper.

Run from the repository root with the bundled document runtime::

    python texnotes/make_figures.py

Each ``make_figure_*`` function owns one numbered figure.  Keeping one owner
per figure makes the plotted values, explanatory labels, and manuscript file
name easy to audit together.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, HexColor, black, white


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "figures"

PAGE_WIDTH = 720.0
WIDE_HEIGHT = 250.0

BLUE = HexColor("#2D69AA")
GOLD = HexColor("#D29123")
GREEN = HexColor("#32915F")
PURPLE = HexColor("#78509B")
RED = HexColor("#B94641")
GRAY = HexColor("#6F747B")
LIGHT_BLUE = HexColor("#EAF1F8")
LIGHT_GOLD = HexColor("#FBF3E3")
LIGHT_GREEN = HexColor("#EAF5EF")
LIGHT_PURPLE = HexColor("#F1ECF5")
LIGHT_RED = HexColor("#F8ECEB")
LIGHT_GRAY = HexColor("#F3F4F5")

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"


def register_fonts() -> None:
    """Register a Unicode font when macOS supplies it.

    ReportLab's built-in Helvetica is the portable fallback.  Arial is used
    when available because it includes the Greek symbols used in the plots.
    """

    regular = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    italic = Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf")

    global FONT_REGULAR, FONT_BOLD, FONT_ITALIC
    if regular.exists() and bold.exists() and italic.exists():
        pdfmetrics.registerFont(TTFont("GuideRegular", str(regular)))
        pdfmetrics.registerFont(TTFont("GuideBold", str(bold)))
        pdfmetrics.registerFont(TTFont("GuideItalic", str(italic)))
        FONT_REGULAR = "GuideRegular"
        FONT_BOLD = "GuideBold"
        FONT_ITALIC = "GuideItalic"


def new_canvas(
    filename: str,
    width: float,
    height: float,
) -> canvas.Canvas:
    """Create one vector-PDF drawing surface."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    return canvas.Canvas(
        str(path),
        pagesize=(width, height),
        bottomup=1,
    )


def finish(
    drawing: canvas.Canvas,
) -> None:
    """Close one PDF page and write it to disk."""

    drawing.showPage()
    drawing.save()


def draw_lines_centered(
    drawing: canvas.Canvas,
    x_center: float,
    y_center: float,
    lines: Sequence[str],
    font_size: float = 8.5,
    leading: float = 10.5,
    first_bold: bool = True,
) -> None:
    """Draw explicit text lines centered around ``y_center``."""

    total = leading * (len(lines) - 1)
    y = y_center + total / 2.0 - font_size * 0.32
    for index, line in enumerate(lines):
        font = FONT_BOLD if first_bold and index == 0 else FONT_REGULAR
        drawing.setFont(font, font_size)
        drawing.setFillColor(black)
        drawing.drawCentredString(x_center, y, line)
        y -= leading


def draw_box(
    drawing: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: Sequence[str],
    fill: Color,
    stroke: Color = black,
    font_size: float = 8.5,
    leading: float = 10.5,
    radius: float = 4.0,
) -> None:
    """Draw a rounded state box and its manually wrapped text."""

    drawing.setFillColor(fill)
    drawing.setStrokeColor(stroke)
    drawing.setLineWidth(1.0)
    drawing.roundRect(
        x,
        y,
        width,
        height,
        radius,
        stroke=1,
        fill=1,
    )
    draw_lines_centered(
        drawing=drawing,
        x_center=x + width / 2.0,
        y_center=y + height / 2.0,
        lines=lines,
        font_size=font_size,
        leading=leading,
    )


def draw_diamond(
    drawing: canvas.Canvas,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    lines: Sequence[str],
    fill: Color,
) -> None:
    """Draw a decision node centered at the supplied coordinates."""

    path = drawing.beginPath()
    path.moveTo(x_center, y_center + height / 2.0)
    path.lineTo(x_center + width / 2.0, y_center)
    path.lineTo(x_center, y_center - height / 2.0)
    path.lineTo(x_center - width / 2.0, y_center)
    path.close()
    drawing.setFillColor(fill)
    drawing.setStrokeColor(black)
    drawing.setLineWidth(1.0)
    drawing.drawPath(path, stroke=1, fill=1)
    draw_lines_centered(
        drawing=drawing,
        x_center=x_center,
        y_center=y_center,
        lines=lines,
        font_size=8.0,
        leading=9.5,
    )


def draw_arrow(
    drawing: canvas.Canvas,
    points: Sequence[tuple[float, float]],
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 5.0),
    color: Color = black,
    dashed: bool = False,
) -> None:
    """Draw a polyline ending in an arrow head.

    ``points`` are visited in order.  The final two points determine the arrow
    head direction.  The label is placed at the midpoint of the first segment.
    """

    drawing.setStrokeColor(color)
    drawing.setFillColor(color)
    drawing.setLineWidth(1.15)
    drawing.setDash(4, 3) if dashed else drawing.setDash()

    path = drawing.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    drawing.drawPath(path, stroke=1, fill=0)
    drawing.setDash()

    x0, y0 = points[-2]
    x1, y1 = points[-1]
    angle = math.atan2(y1 - y0, x1 - x0)
    head_length = 7.0
    head_half_width = 3.0
    back_x = x1 - head_length * math.cos(angle)
    back_y = y1 - head_length * math.sin(angle)
    left_x = back_x + head_half_width * math.sin(angle)
    left_y = back_y - head_half_width * math.cos(angle)
    right_x = back_x - head_half_width * math.sin(angle)
    right_y = back_y + head_half_width * math.cos(angle)

    head = drawing.beginPath()
    head.moveTo(x1, y1)
    head.lineTo(left_x, left_y)
    head.lineTo(right_x, right_y)
    head.close()
    drawing.drawPath(head, stroke=0, fill=1)

    if label:
        xa, ya = points[0]
        xb, yb = points[1]
        drawing.setFont(FONT_REGULAR, 7.5)
        drawing.drawCentredString(
            (xa + xb) / 2.0 + label_offset[0],
            (ya + yb) / 2.0 + label_offset[1],
            label,
        )


def draw_axis_panel(
    drawing: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    title: str,
    x_label: str,
    y_label: str,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    log_x: bool = False,
) -> tuple:
    """Draw one plot panel and return coordinate-mapping callables."""

    left = x + 38.0
    bottom = y + 28.0
    plot_width = width - 50.0
    plot_height = height - 52.0

    xmin, xmax = x_limits
    ymin, ymax = y_limits

    def map_x(value: float) -> float:
        raw = math.log10(value) if log_x else value
        lo = math.log10(xmin) if log_x else xmin
        hi = math.log10(xmax) if log_x else xmax
        return left + (raw - lo) / (hi - lo) * plot_width

    def map_y(value: float) -> float:
        return bottom + (value - ymin) / (ymax - ymin) * plot_height

    drawing.setStrokeColor(GRAY)
    drawing.setLineWidth(0.7)
    drawing.line(left, bottom, left + plot_width, bottom)
    drawing.line(left, bottom, left, bottom + plot_height)

    drawing.setFont(FONT_REGULAR, 7.0)
    for value in x_ticks:
        px = map_x(value)
        drawing.line(px, bottom - 3.0, px, bottom + 3.0)
        label = f"{value:g}"
        drawing.drawCentredString(px, bottom - 13.0, label)
    for value in y_ticks:
        py = map_y(value)
        drawing.line(left - 3.0, py, left + 3.0, py)
        drawing.drawRightString(left - 6.0, py - 2.5, f"{value:g}")

    drawing.setFillColor(black)
    drawing.setFont(FONT_BOLD, 10.0)
    drawing.drawCentredString(x + width / 2.0, y + height - 12.0, title)
    drawing.setFont(FONT_REGULAR, 8.0)
    drawing.drawCentredString(x + width / 2.0, y + 3.0, x_label)

    drawing.saveState()
    drawing.translate(x + 8.0, y + height / 2.0)
    drawing.rotate(90)
    drawing.drawCentredString(0.0, 0.0, y_label)
    drawing.restoreState()

    return map_x, map_y


def draw_curve(
    drawing: canvas.Canvas,
    x_values: Iterable[float],
    y_values: Iterable[float],
    map_x,
    map_y,
    color: Color,
    dashed: bool = False,
    dotted: bool = False,
    marker: str | None = None,
) -> None:
    """Draw one vector curve and optional point markers."""

    points = [(map_x(float(x)), map_y(float(y)))
              for x, y in zip(x_values, y_values)]
    drawing.setStrokeColor(color)
    drawing.setFillColor(color)
    drawing.setLineWidth(1.8)
    if dashed:
        drawing.setDash(6, 4)
    elif dotted:
        drawing.setDash(1, 4)
    else:
        drawing.setDash()
    path = drawing.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    drawing.drawPath(path, stroke=1, fill=0)
    drawing.setDash()

    if marker:
        for px, py in points:
            if marker == "circle":
                drawing.circle(px, py, 2.4, stroke=1, fill=1)
            elif marker == "square":
                drawing.rect(px - 2.2, py - 2.2, 4.4, 4.4, stroke=1, fill=1)
            elif marker == "triangle":
                path = drawing.beginPath()
                path.moveTo(px, py + 3.0)
                path.lineTo(px - 2.8, py - 2.0)
                path.lineTo(px + 2.8, py - 2.0)
                path.close()
                drawing.drawPath(path, stroke=1, fill=1)


def draw_legend(
    drawing: canvas.Canvas,
    x: float,
    y: float,
    entries: Sequence[tuple[str, Color, str]],
    columns: int = 1,
    column_width: float = 120.0,
) -> None:
    """Draw a borderless legend in unused margin space."""

    drawing.setFont(FONT_REGULAR, 7.7)
    row_height = 12.0
    for index, (label, color, style) in enumerate(entries):
        row = index // columns
        column = index % columns
        x0 = x + column * column_width
        y0 = y - row * row_height
        drawing.setStrokeColor(color)
        drawing.setLineWidth(1.8)
        if style == "dashed":
            drawing.setDash(6, 4)
        elif style == "dotted":
            drawing.setDash(1, 3)
        else:
            drawing.setDash()
        drawing.line(x0, y0, x0 + 22.0, y0)
        drawing.setDash()
        drawing.setFillColor(black)
        drawing.drawString(x0 + 28.0, y0 - 2.5, label)


def make_figure_01_ownership_chain() -> None:
    """Figure 1: file, host, model, artifact, and adapter ownership."""

    drawing = new_canvas("fig01_ownership_chain.pdf", PAGE_WIDTH, 220.0)
    width = 145.0
    height = 55.0
    xs = [25.0, 200.0, 375.0, 550.0]
    top_y = 145.0
    bottom_y = 38.0

    boxes = [
        (xs[0], top_y, ["1. Input files", "YAML; C:(N,P); V:(N,D)"], LIGHT_BLUE),
        (xs[1], top_y, ["2. Validated run", "absolute paths; one family"], LIGHT_BLUE),
        (xs[2], top_y, ["3. Staged sources", "NumPy arrays or memmaps"], LIGHT_BLUE),
        (xs[3], top_y, ["4. Scientific coordinates", "encoders, decoder, loss"], LIGHT_GOLD),
        (xs[3], bottom_y, ["5. Training state", "model, optimizer, history"], LIGHT_GREEN),
        (xs[2], bottom_y, ["6. Artifact pair", "weights plus HDF5 recipe"], LIGHT_PURPLE),
        (xs[1], bottom_y, ["7. Rebuilt predictor", "strict load; physical result"], LIGHT_GOLD),
        (xs[0], bottom_y, ["8. Cobaya product", "named quantity on its axis"], LIGHT_PURPLE),
    ]
    for x, y, lines, fill in boxes:
        draw_box(drawing, x, y, width, height, lines, fill)

    draw_arrow(drawing, [(170, 172), (200, 172)], "resolve")
    draw_arrow(drawing, [(345, 172), (375, 172)], "stage")
    draw_arrow(drawing, [(520, 172), (550, 172)], "fit maps")
    draw_arrow(drawing, [(622, 145), (622, 93)], "train", (14, 0))
    draw_arrow(drawing, [(550, 65), (520, 65)], "save")
    draw_arrow(drawing, [(375, 65), (345, 65)], "rebuild")
    draw_arrow(drawing, [(200, 65), (170, 65)], "serve")

    drawing.setFont(FONT_REGULAR, 8.0)
    drawing.setFillColor(GRAY)
    drawing.drawCentredString(
        PAGE_WIDTH / 2.0,
        15.0,
        "Each arrow names the code boundary that validates the next state.",
    )
    finish(drawing)


def make_figure_02_campaigns() -> None:
    """Figure 2: illustrative campaign curves."""

    drawing = new_canvas("fig02_campaigns.pdf", PAGE_WIDTH, 250.0)
    left_map_x, left_map_y = draw_axis_panel(
        drawing=drawing,
        x=15.0,
        y=35.0,
        width=335.0,
        height=205.0,
        x_limits=(800.0, 130000.0),
        y_limits=(0.0, 0.55),
        title="Training-size learning curve",
        x_label="training rows N_train (log scale)",
        y_label="failure fraction",
        x_ticks=[1000.0, 10000.0, 100000.0],
        y_ticks=[0.0, 0.2, 0.4],
        log_x=True,
    )
    n_train = [1000, 3000, 10000, 30000, 100000]
    draw_curve(drawing, n_train, [.48, .34, .20, .12, .08],
               left_map_x, left_map_y, BLUE, marker="circle")
    draw_curve(drawing, n_train, [.50, .31, .16, .09, .06],
               left_map_x, left_map_y, PURPLE, marker="square")
    draw_curve(drawing, n_train, [.53, .37, .19, .10, .065],
               left_map_x, left_map_y, GREEN, marker="triangle")
    draw_legend(
        drawing=drawing,
        x=105.0,
        y=24.0,
        entries=[("H", BLUE, "solid"),
                 ("multi-gate", PURPLE, "solid"),
                 ("power", GREEN, "solid")],
        columns=3,
        column_width=75.0,
    )

    right_map_x, right_map_y = draw_axis_panel(
        drawing=drawing,
        x=370.0,
        y=35.0,
        width=335.0,
        height=205.0,
        x_limits=(0.00008, 0.02),
        y_limits=(0.08, 0.45),
        title="One-knob sweep",
        x_label="learning rate (log scale)",
        y_label="failure fraction",
        x_ticks=[0.0001, 0.001, 0.01],
        y_ticks=[0.1, 0.2, 0.3, 0.4],
        log_x=True,
    )
    learning_rates = [.0001, .0003, .001, .003, .01]
    scores = [.38, .25, .14, .10, .29]
    draw_curve(drawing, learning_rates, scores,
               right_map_x, right_map_y, GOLD, marker="circle")
    px = right_map_x(.003)
    drawing.setStrokeColor(RED)
    drawing.setDash(5, 3)
    drawing.line(px, right_map_y(.08), px, right_map_y(.45))
    drawing.setDash()
    drawing.setFont(FONT_REGULAR, 7.5)
    drawing.setFillColor(RED)
    drawing.drawString(px + 5.0, right_map_y(.39), "best tested point")
    finish(drawing)


def make_figure_03_training_cloud() -> None:
    """Figure 3: posterior-like and widened training support."""

    drawing = new_canvas("fig03_training_cloud.pdf", PAGE_WIDTH, 210.0)
    drawing.setStrokeColor(black)
    draw_arrow(drawing, [(55, 35), (350, 35)])
    draw_arrow(drawing, [(55, 35), (55, 190)])
    drawing.setFont(FONT_ITALIC, 10.0)
    drawing.drawString(355, 30, "theta 1")
    drawing.drawString(35, 193, "theta 2")

    drawing.saveState()
    drawing.translate(190, 105)
    drawing.rotate(25)
    drawing.setFillColor(LIGHT_GOLD)
    drawing.setStrokeColor(GOLD)
    drawing.setLineWidth(1.8)
    drawing.ellipse(-125, -52, 125, 52, stroke=1, fill=1)
    drawing.setFillColor(HexColor("#C7DBF0"))
    drawing.setStrokeColor(BLUE)
    drawing.ellipse(-108, -22, 108, 22, stroke=1, fill=1)
    drawing.restoreState()

    draw_box(
        drawing,
        405,
        115,
        270,
        48,
        ["Posterior-like density", "narrow across the degeneracy"],
        LIGHT_BLUE,
        stroke=BLUE,
    )
    draw_box(
        drawing,
        405,
        52,
        270,
        48,
        ["Training density", "wider along and across the ridge"],
        LIGHT_GOLD,
        stroke=GOLD,
    )
    finish(drawing)


def make_figure_04_row_coordinates() -> None:
    """Figure 4: row identity across resident and disk-backed staging."""

    drawing = new_canvas("fig04_row_coordinates.pdf", PAGE_WIDTH, 255.0)
    draw_box(drawing, 20, 105, 125, 80,
             ["Selected global rows", "idx = [9, 2, 5]", "original-file coordinates"],
             LIGHT_BLUE)
    draw_box(drawing, 175, 105, 125, 80,
             ["Sequential disk order", "rows = [2, 5, 9]", "sorted unique gather"],
             LIGHT_BLUE)
    draw_box(drawing, 330, 105, 125, 80,
             ["Storage decision", "Do compact arrays", "and reindex fit RAM?"],
             LIGHT_GOLD)
    draw_box(drawing, 500, 155, 190, 80,
             ["Yes: resident", "C[rows], V[rows] are copies", "compact 0,1,2 -> disk 2,5,9"],
             LIGHT_GREEN)
    draw_box(drawing, 500, 55, 190, 80,
             ["No: disk-backed", "V remains a read-only memmap", "idx_src = [9, 2, 5]"],
             LIGHT_RED)
    draw_box(drawing, 175, 10, 190, 60,
             ["Coordinate sidecar", "dump_rows = [2, 5, 9]", "always original-file coordinates"],
             LIGHT_PURPLE)
    draw_arrow(drawing, [(145, 145), (175, 145)])
    draw_arrow(drawing, [(300, 145), (330, 145)])
    draw_arrow(drawing, [(455, 155), (500, 190)], "yes", (0, 7))
    draw_arrow(drawing, [(455, 135), (500, 95)], "no", (0, -8))
    draw_arrow(drawing, [(238, 105), (238, 70)])
    finish(drawing)


def make_figure_05_residual_block() -> None:
    """Figure 5: exact ResBlock order implemented in designs/blocks.py."""

    drawing = new_canvas("fig05_residual_block.pdf", PAGE_WIDTH, 225.0)
    drawing.setFillColor(LIGHT_GRAY)
    drawing.setStrokeColor(GRAY)
    drawing.setDash(5, 3)
    drawing.roundRect(125, 45, 475, 155, 6, stroke=1, fill=1)
    drawing.setDash()
    drawing.setFillColor(GRAY)
    drawing.setFont(FONT_BOLD, 8.5)
    drawing.drawString(135, 187, "one ResBlock: every internal Linear is W -> W")

    draw_box(drawing, 15, 90, 90, 55,
             ["model entry", "P_enc -> W", "outside block"], LIGHT_GRAY, stroke=GRAY)
    draw_box(drawing, 620, 90, 85, 55,
             ["model exit", "W -> K", "outside block"], LIGHT_GRAY, stroke=GRAY)
    draw_box(drawing, 140, 105, 65, 45,
             ["h_in", "B x W"], LIGHT_BLUE)
    draw_box(drawing, 235, 105, 85, 45,
             ["stages 1...L-1", "Linear W->W", "norm; activation"], LIGHT_GREEN,
             font_size=7.4, leading=9.0)
    draw_box(drawing, 350, 105, 80, 45,
             ["final Linear", "W -> W"], LIGHT_GREEN)

    drawing.setFillColor(white)
    drawing.setStrokeColor(black)
    drawing.circle(470, 127.5, 14, stroke=1, fill=1)
    drawing.setFillColor(black)
    drawing.setFont(FONT_BOLD, 13)
    drawing.drawCentredString(470, 123, "+")

    draw_box(drawing, 505, 105, 75, 45,
             ["final norm", "then activation"], LIGHT_GOLD)
    draw_box(drawing, 505, 50, 75, 35,
             ["h_out", "B x W"], LIGHT_BLUE)

    draw_arrow(drawing, [(105, 117), (140, 127)])
    draw_arrow(drawing, [(205, 127), (235, 127)])
    draw_arrow(drawing, [(320, 127), (350, 127)])
    draw_arrow(drawing, [(430, 127), (456, 127)])
    draw_arrow(drawing, [(484, 127), (505, 127)])
    draw_arrow(drawing, [(542, 105), (542, 85)])
    draw_arrow(drawing, [(580, 67), (620, 110)])
    draw_arrow(drawing, [(173, 105), (173, 68), (470, 68), (470, 113)])

    drawing.setFillColor(RED)
    drawing.setFont(FONT_BOLD, 8.0)
    drawing.drawCentredString(370, 28,
        "The skip is added after the final internal Linear and before the final norm/activation.")
    finish(drawing)


def make_figure_06_cnn_head() -> None:
    """Figure 6: structured convolutional correction head."""

    drawing = new_canvas("fig06_cnn_head.pdf", PAGE_WIDTH, 210.0)
    top_y = 125.0
    lower_y = 45.0
    xs = [20.0, 190.0, 360.0, 530.0]
    width = 145.0
    height = 50.0
    draw_box(drawing, xs[0], top_y, width, height,
             ["trunk output", "flat kept vector"], LIGHT_BLUE)
    draw_box(drawing, xs[1], top_y, width, height,
             ["fixed scatter", "declared coordinate map"], LIGHT_GOLD)
    draw_box(drawing, xs[2], top_y, width, height,
             ["bin-angle rectangle", "plus validity mask"], LIGHT_GREEN)
    draw_box(drawing, xs[3], top_y, width, height,
             ["convolution blocks", "local angular mixing"], LIGHT_PURPLE)
    draw_box(drawing, xs[2], lower_y, width, height,
             ["mask invalid cells", "after every mixing step"], LIGHT_RED)
    draw_box(drawing, xs[1], lower_y, width, height,
             ["fixed gather", "return kept order"], LIGHT_GOLD)
    draw_box(drawing, xs[0], lower_y, width, height,
             ["published result", "trunk + gate x correction"], LIGHT_GREEN)
    draw_arrow(drawing, [(165, 150), (190, 150)], "B x K")
    draw_arrow(drawing, [(335, 150), (360, 150)])
    draw_arrow(drawing, [(505, 150), (530, 150)])
    draw_arrow(drawing, [(602, 125), (602, 95), (505, 70)])
    draw_arrow(drawing, [(360, 70), (335, 70)])
    draw_arrow(drawing, [(190, 70), (165, 70)])
    drawing.setFont(FONT_REGULAR, 8.0)
    drawing.setFillColor(GRAY)
    drawing.drawCentredString(360, 18,
        "Zero is not a padding marker: a physical target may also be exactly zero.")
    finish(drawing)


def make_figure_07_activations() -> None:
    """Figure 7: baseline, multi-gate, and power-tail curves."""

    drawing = new_canvas("fig07_activations.pdf", PAGE_WIDTH, 280.0)
    panels = [
        (15.0, "Baseline H"),
        (255.0, "Bulk generalization"),
        (495.0, "Tail generalization"),
    ]
    mappings = []
    for x, title in panels:
        mappings.append(draw_axis_panel(
            drawing=drawing,
            x=x,
            y=62.0,
            width=210.0,
            height=205.0,
            x_limits=(-4.0, 4.0),
            y_limits=(-4.2, 6.2),
            title=title,
            x_label="x",
            y_label="f(x)",
            x_ticks=[-4.0, -2.0, 0.0, 2.0, 4.0],
            y_ticks=[-4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
        ))

    x_values = np.linspace(-4.0, 4.0, 241)
    def sigmoid(values):
        """Evaluate the logistic curve element by element."""

        return 1.0 / (1.0 + np.exp(-values))

    left_map_x, left_map_y = mappings[0]
    middle_map_x, middle_map_y = mappings[1]
    right_map_x, right_map_y = mappings[2]
    baseline = (0.15 + 0.85 * sigmoid(2.0 * x_values)) * x_values
    initial = 0.5 * x_values
    identity = x_values
    multi = (0.10
             + 0.35 * sigmoid(2.2 * (x_values + 1.0))
             + 0.55 * sigmoid(2.2 * (x_values - 1.0))) * x_values
    signed_power = np.sign(x_values) * (
        (1.0 + np.abs(x_values)) ** 1.4 - 1.0
    ) / 1.4
    power = (0.15 + 0.85 * sigmoid(2.0 * x_values)) * signed_power

    draw_curve(
        drawing=drawing,
        x_values=x_values,
        y_values=baseline,
        map_x=left_map_x,
        map_y=left_map_y,
        color=BLUE,
    )
    draw_curve(
        drawing=drawing,
        x_values=x_values,
        y_values=initial,
        map_x=left_map_x,
        map_y=left_map_y,
        color=GOLD,
        dashed=True,
    )
    draw_curve(
        drawing=drawing,
        x_values=x_values,
        y_values=identity,
        map_x=left_map_x,
        map_y=left_map_y,
        color=GRAY,
        dotted=True,
    )
    draw_curve(
        drawing=drawing,
        x_values=x_values,
        y_values=multi,
        map_x=middle_map_x,
        map_y=middle_map_y,
        color=PURPLE,
    )
    draw_curve(
        drawing=drawing,
        x_values=x_values,
        y_values=baseline,
        map_x=middle_map_x,
        map_y=middle_map_y,
        color=BLUE,
        dashed=True,
    )
    draw_curve(
        drawing=drawing,
        x_values=x_values,
        y_values=power,
        map_x=right_map_x,
        map_y=right_map_y,
        color=GREEN,
    )
    draw_curve(
        drawing=drawing,
        x_values=x_values,
        y_values=baseline,
        map_x=right_map_x,
        map_y=right_map_y,
        color=BLUE,
        dashed=True,
    )

    draw_legend(
        drawing=drawing,
        x=50.0,
        y=45.0,
        entries=[("gamma=.15, beta=2", BLUE, "solid"),
                 ("initial x/2", GOLD, "dashed"),
                 ("identity", GRAY, "dotted")],
        columns=1,
    )
    draw_legend(
        drawing=drawing,
        x=315.0,
        y=45.0,
        entries=[("two gates", PURPLE, "solid"),
                 ("baseline H", BLUE, "dashed")],
        columns=1,
    )
    draw_legend(
        drawing=drawing,
        x=555.0,
        y=45.0,
        entries=[("p=1.4", GREEN, "solid"),
                 ("p=1 gives H", BLUE, "dashed")],
        columns=1,
    )
    finish(drawing)


def make_figure_08_training_step() -> None:
    """Figure 8: persistent-state order for one accepted optimizer step."""

    drawing = new_canvas("fig08_training_step.pdf", PAGE_WIDTH, 185.0)
    top_y = 105.0
    lower_y = 35.0
    xs = [15.0, 155.0, 295.0, 435.0]
    width = 115.0
    height = 45.0
    draw_box(drawing, xs[0], top_y, width, height, ["load minibatch"], LIGHT_BLUE)
    draw_box(drawing, xs[1], top_y, width, height, ["forward", "and loss"], LIGHT_GREEN)
    draw_box(drawing, xs[2], top_y, width, height, ["backward", "gradients"], LIGHT_GOLD)
    draw_diamond(drawing, 500, 127, 100, 60, ["finite?"], LIGHT_RED)
    draw_box(drawing, 560, lower_y, 125, height, ["unscale and clip"], LIGHT_PURPLE)
    draw_box(drawing, 405, lower_y, 125, height, ["optimizer step"], LIGHT_GREEN)
    draw_box(drawing, 250, lower_y, 125, height, ["optional anchor"], LIGHT_GOLD)
    draw_box(drawing, 95, lower_y, 125, height, ["EMA update", "accepted state only"], LIGHT_BLUE)
    draw_arrow(drawing, [(130, 127), (155, 127)])
    draw_arrow(drawing, [(270, 127), (295, 127)])
    draw_arrow(drawing, [(410, 127), (450, 127)])
    draw_arrow(drawing, [(500, 97), (622, 80)], "yes", (0, 8))
    draw_arrow(drawing, [(560, 57), (530, 57)])
    draw_arrow(drawing, [(405, 57), (375, 57)])
    draw_arrow(drawing, [(250, 57), (220, 57)])
    draw_arrow(drawing, [(550, 127), (690, 127)],
               "no: clear gradients; advance no state", (5, 8), RED)
    finish(drawing)


def make_figure_09_warmstart() -> None:
    """Figure 9: fine-tuning and transfer branch from a source artifact."""

    drawing = new_canvas("fig09_warmstart.pdf", PAGE_WIDTH, 190.0)
    draw_box(drawing, 20, 70, 120, 50, ["source artifact"], LIGHT_BLUE)
    draw_box(drawing, 190, 115, 125, 50, ["fine-tune", "expand source model"], LIGHT_GOLD)
    draw_box(drawing, 190, 25, 125, 50, ["transfer", "freeze source base"], LIGHT_PURPLE)
    draw_box(drawing, 365, 115, 130, 50, ["inherited weights", "trainable"], LIGHT_GREEN)
    draw_box(drawing, 365, 25, 130, 50, ["new correction", "trainable"], LIGHT_GREEN)
    draw_box(drawing, 545, 70, 145, 50, ["epoch-zero parity", "before any step"], LIGHT_RED)
    draw_arrow(drawing, [(140, 95), (190, 140)])
    draw_arrow(drawing, [(140, 95), (190, 50)])
    draw_arrow(drawing, [(315, 140), (365, 140)])
    draw_arrow(drawing, [(315, 50), (365, 50)])
    draw_arrow(drawing, [(495, 140), (545, 105)])
    draw_arrow(drawing, [(495, 50), (545, 85)])
    drawing.setFont(FONT_REGULAR, 8.0)
    drawing.setFillColor(GRAY)
    drawing.drawCentredString(360, 8,
        "Both branches publish a self-contained artifact only after parity succeeds.")
    finish(drawing)


def make_figure_10_board_flow() -> None:
    """Figure 10: board execution, evidence, and terminal verdict."""

    drawing = new_canvas("fig10_board_flow.pdf", PAGE_WIDTH, 205.0)
    xs = [20.0, 180.0, 340.0, 500.0]
    top_y = 125.0
    width = 135.0
    height = 48.0
    draw_box(drawing, xs[0], top_y, width, height,
             ["preflight", "tree, imports, paths"], LIGHT_BLUE)
    draw_box(drawing, xs[1], top_y, width, height,
             ["selection", "tier, gate, resume"], LIGHT_GOLD)
    draw_box(drawing, xs[2], top_y, width, height,
             ["gate body", "check or short run"], LIGHT_GREEN)
    draw_box(drawing, xs[3], top_y, width, height,
             ["raw log", "commands and values"], LIGHT_PURPLE)
    draw_diamond(drawing, 567, 70, 135, 58,
                 ["all required", "claims true?"], LIGHT_RED)
    draw_box(drawing, 330, 15, 125, 42, ["PASS", "record digest"], LIGHT_GREEN)
    draw_box(drawing, 580, 15, 125, 42, ["FAIL", "nonzero verdict"], LIGHT_RED)
    draw_arrow(drawing, [(155, 149), (180, 149)])
    draw_arrow(drawing, [(315, 149), (340, 149)])
    draw_arrow(drawing, [(475, 149), (500, 149)])
    draw_arrow(drawing, [(567, 125), (567, 99)])
    draw_arrow(drawing, [(520, 70), (455, 36)], "yes", (0, 7))
    draw_arrow(drawing, [(614, 70), (642, 57)], "no", (0, 7))
    drawing.setFont(FONT_REGULAR, 8.0)
    drawing.setFillColor(GRAY)
    drawing.drawString(22, 88,
        "The raw log is the evidence; the board row is its index and verdict.")
    finish(drawing)


def make_figure_11_gate_dependencies() -> None:
    """Figure 11: common dependency chain across identity and smoke gates."""

    drawing = new_canvas("fig11_gate_dependencies.pdf", PAGE_WIDTH, 165.0)
    y = 92.0
    width = 130.0
    height = 48.0
    xs = [20.0, 190.0, 360.0, 530.0]
    draw_box(drawing, xs[0], y, width, height,
             ["family identity", "analytic fixture"], LIGHT_BLUE)
    draw_box(drawing, xs[1], y, width, height,
             ["save/rebuild", "artifact parity"], LIGHT_GOLD)
    draw_box(drawing, xs[2], y, width, height,
             ["adapter parity", "direct evaluate"], LIGHT_GREEN)
    draw_box(drawing, xs[3], y, width, height,
             ["short sampling", "repeated lifecycle"], LIGHT_PURPLE)
    draw_box(drawing, xs[1], 18.0, width, height,
             ["family smoke", "real provider"], LIGHT_BLUE)
    draw_arrow(drawing, [(150, 116), (190, 116)])
    draw_arrow(drawing, [(320, 116), (360, 116)])
    draw_arrow(drawing, [(490, 116), (530, 116)])
    draw_arrow(drawing, [(85, 92), (85, 42), (190, 42)])
    draw_arrow(drawing, [(320, 42), (425, 42), (425, 92)])
    finish(drawing)


def make_figure_12_three_session_loop() -> None:
    """Figure 12: the three cooperating sessions and the audited loop.

    The palette is deliberately blue, gold, purple, and gray.  Red and green
    never appear together in a house figure, so the pass and fail outcomes
    carry their verdict in the bold word, not in the fill color alone.
    """

    drawing = new_canvas("fig12_three_session_loop.pdf", PAGE_WIDTH, 400.0)

    draw_box(drawing, 260.0, 336.0, 200.0, 52.0,
             ["architect session", "designs the contract; audits every",
              "change; has the final word"], LIGHT_BLUE)
    draw_box(drawing, 285.0, 240.0, 150.0, 44.0,
             ["blueprint + gates", "the contract and the checks",
              "that decide it"], LIGHT_GRAY)
    draw_box(drawing, 25.0, 232.0, 180.0, 58.0,
             ["implementer session", "writes the complete code;",
              "runs the gates on it"], LIGHT_GOLD)
    draw_box(drawing, 515.0, 232.0, 180.0, 58.0,
             ["red-team session", "attacks and probes: bugs,",
              "weak tests, stale docs"], LIGHT_PURPLE)
    draw_diamond(drawing, 360.0, 150.0, 200.0, 66.0,
                 ["audit against", "raw evidence"], LIGHT_BLUE)
    draw_box(drawing, 90.0, 58.0, 195.0, 44.0,
             ["pass", "milestone recorded in notes/"], LIGHT_BLUE)
    draw_box(drawing, 435.0, 58.0, 195.0, 44.0,
             ["fail", "delta re-handoff"], LIGHT_GOLD)

    draw_arrow(drawing, [(360, 336), (360, 284)])
    draw_arrow(drawing, [(285, 262), (205, 262)], "architect handoff")
    draw_arrow(drawing, [(435, 262), (515, 262)], "red-team handoff")
    draw_arrow(drawing, [(115, 232), (115, 150), (260, 150)],
               "implementer handoff", (16, 0))
    draw_arrow(drawing, [(605, 232), (605, 150), (460, 150)],
               "findings", (-14, 0))
    draw_arrow(drawing, [(360, 240), (360, 183)])
    draw_arrow(drawing, [(310, 133), (187, 102)], "pass", (0, 7))
    draw_arrow(drawing, [(410, 133), (532, 102)], "fail", (0, 7))
    draw_arrow(drawing, [(630, 80), (708, 80), (708, 362), (460, 362)],
               "the loop repeats", (0, 7))

    draw_legend(
        drawing=drawing,
        x=22.0,
        y=36.0,
        entries=[
            ("architect: designs the contract, audits every change",
             BLUE, "solid"),
            ("implementer: writes the code and runs the gates",
             GOLD, "solid"),
            ("red team: attacks the code and files findings",
             PURPLE, "solid"),
            ("gates: machine-run checks whose raw output decides the verdict",
             GRAY, "solid"),
        ],
        columns=2,
        column_width=350.0,
    )

    drawing.setFont(FONT_REGULAR, 8.0)
    drawing.setFillColor(GRAY)
    drawing.drawCentredString(
        PAGE_WIDTH / 2.0,
        8.0,
        "Code and findings both enter the audit as evidence; only the "
        "architect's ruling changes the code.",
    )
    finish(drawing)


def main() -> None:
    """Generate the complete numbered figure set."""

    register_fonts()
    builders = [
        make_figure_01_ownership_chain,
        make_figure_02_campaigns,
        make_figure_03_training_cloud,
        make_figure_04_row_coordinates,
        make_figure_05_residual_block,
        make_figure_06_cnn_head,
        make_figure_07_activations,
        make_figure_08_training_step,
        make_figure_09_warmstart,
        make_figure_10_board_flow,
        make_figure_11_gate_dependencies,
        make_figure_12_three_session_loop,
    ]
    for build in builders:
        build()
        print(f"wrote {build.__name__.replace('make_figure_', 'fig')}")


if __name__ == "__main__":
    main()
