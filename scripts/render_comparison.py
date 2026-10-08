"""Render all eight fixed comparisons from saved outputs, without truncation."""
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(line) for line in (ROOT / "data/eval/side_by_side.jsonl").read_text(encoding="utf-8").splitlines()][:8]
def wrap(text, width):
    return "\n".join(textwrap.fill(paragraph, width=width) for paragraph in text.splitlines() if paragraph.strip())
cells = [[wrap(f"{r['id']} · {r['category']}\n{r['prompt']}", 38), wrap(r["sft"], 87), wrap(r["dpo"], 87)] for r in rows]
line_counts = [max(text.count("\n") + 1 for text in row) + 2 for row in cells]
total = sum(line_counts) + 4
fig, ax = plt.subplots(figsize=(25, total * 0.28 + 1.8))
ax.axis("off")
ax.set_title("NB4 · 8 fixed prompts · SFT vs SFT+DPO\nFull saved responses; original tool tags retained", fontsize=16, pad=16)
table = ax.table(cellText=cells, colLabels=["Prompt", "SFT", "SFT+DPO"], colWidths=[0.18, 0.41, 0.41], cellLoc="left", bbox=[0, 0, 1, 0.98])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (row, col), cell in table.get_celld().items():
    cell.PAD = 0.035
    cell.get_text().set_va("center")
    cell.set_height((4 if row == 0 else line_counts[row - 1]) / total)
    if row == 0:
        cell.set_facecolor("#2e548a")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    else:
        cell.set_facecolor("#f0f4fa" if row % 2 else "#ffffff")
fig.savefig(ROOT / "submission/screenshots/04-side-by-side-table.png", dpi=110, bbox_inches="tight")
plt.close(fig)
