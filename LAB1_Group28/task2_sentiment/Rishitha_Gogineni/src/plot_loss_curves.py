"""Create reproducible SVG loss curves without external plotting packages."""

from html import escape
from pathlib import Path
import re

FINAL_RUNS = [
    ("Baseline", "baseline_20260926_130146.log"),
    ("DPCNN", "dpcnn_20260926_152502.log"),
    ("TextCNN", "textcnn_20260926_133806.log"),
]


def load_history(path):
    pattern = re.compile(
        r"epoch=(?P<epoch>\d+).*?train_loss=(?P<train_loss>\d+\.\d+).*?"
        r"val_loss=(?P<val_loss>\d+\.\d+)"
    )
    return [
        match.groupdict()
        for match in pattern.finditer(path.read_text(encoding="utf-8"))
    ]


def point_string(values, x0, y0, width, height, minimum, maximum):
    count = len(values)
    points = []
    for index, value in enumerate(values):
        x = x0 + width * index / max(count - 1, 1)
        y = y0 + height * (maximum - value) / (maximum - minimum)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def main():
    root = Path(__file__).resolve().parents[1]
    width, height = 1440, 470
    panel_width, panel_height = 390, 285
    left, top = 55, 105
    panels = []

    for panel_index, (model_name, filename) in enumerate(FINAL_RUNS):
        history = load_history(root / "logs" / "raw" / filename)
        epochs = [int(row["epoch"]) for row in history]
        train_loss = [float(row["train_loss"]) for row in history]
        validation_loss = [float(row["val_loss"]) for row in history]
        best_epoch = epochs[validation_loss.index(min(validation_loss))]
        minimum = min(train_loss + validation_loss)
        maximum = max(train_loss + validation_loss)
        padding = max((maximum - minimum) * 0.08, 0.01)
        minimum -= padding
        maximum += padding
        x0 = left + panel_index * (panel_width + 75)
        y0 = top
        selected_x = x0 + panel_width * (best_epoch - epochs[0]) / max(len(epochs) - 1, 1)

        grid = []
        for tick in range(5):
            value = minimum + (maximum - minimum) * tick / 4
            y = y0 + panel_height * (maximum - value) / (maximum - minimum)
            grid.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" y2="{y:.1f}" class="grid"/>'
                f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{value:.3f}</text>'
            )
        x_ticks = "".join(
            f'<text x="{x0 + panel_width * index / max(len(epochs) - 1, 1):.1f}" y="{y0 + panel_height + 22}" class="tick" text-anchor="middle">{epoch}</text>'
            for index, epoch in enumerate(epochs)
        )
        panels.append(
            f'<text x="{x0 + panel_width / 2:.1f}" y="{y0 - 28}" class="title" text-anchor="middle">{escape(model_name)}</text>'
            + "".join(grid)
            + f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" class="border"/>'
            + f'<line x1="{selected_x:.1f}" y1="{y0}" x2="{selected_x:.1f}" y2="{y0 + panel_height}" class="selected"/>'
            + f'<polyline points="{point_string(train_loss, x0, y0, panel_width, panel_height, minimum, maximum)}" class="train"/>'
            + f'<polyline points="{point_string(validation_loss, x0, y0, panel_width, panel_height, minimum, maximum)}" class="validation"/>'
            + x_ticks
            + f'<text x="{x0 + panel_width / 2:.1f}" y="{y0 + panel_height + 48}" class="label" text-anchor="middle">Epoch</text>'
            + f'<text x="{x0 + panel_width / 2:.1f}" y="{y0 + panel_height + 70}" class="note" text-anchor="middle">Selected checkpoint: epoch {best_epoch}</text>'
        )

    legend = (
        '<line x1="550" y1="55" x2="580" y2="55" class="train"/>'
        '<text x="588" y="60" class="label">Training loss</text>'
        '<line x1="715" y1="55" x2="745" y2="55" class="validation"/>'
        '<text x="753" y="60" class="label">Validation loss</text>'
        '<line x1="930" y1="55" x2="960" y2="55" class="selected"/>'
        '<text x="968" y="60" class="label">Selected epoch</text>'
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
.title{{font:600 18px Arial,sans-serif;fill:#1f2937}} .label{{font:14px Arial,sans-serif;fill:#374151}} .note{{font:12px Arial,sans-serif;fill:#4b5563}} .tick{{font:12px Arial,sans-serif;fill:#4b5563}} .grid{{stroke:#e5e7eb;stroke-width:1}} .border{{fill:none;stroke:#9ca3af;stroke-width:1}} .train{{fill:none;stroke:#2563eb;stroke-width:3}} .validation{{fill:none;stroke:#dc2626;stroke-width:3}} .selected{{stroke:#6b7280;stroke-width:2;stroke-dasharray:6 5}}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="720" y="28" class="title" text-anchor="middle">Training and Validation Loss for Final Task 2 Runs</text>
{legend}
{''.join(panels)}
</svg>'''
    output = root / "outputs" / "plots" / "training_validation_loss_curves.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(f"Saved: {output.relative_to(root)}")


if __name__ == "__main__":
    main()
