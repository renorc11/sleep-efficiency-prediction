from __future__ import annotations

# WITHDRAWN (2026-11-22). This script compared the project's ROC-AUC, computed from a
# rescaled regression output, with AUROC from a clinical sleep-disorder classification
# paper. Those numbers are not comparable. See ../WITHDRAWN.md. Its output paths also
# no longer resolve after the move, so it refuses to run.
raise SystemExit("Withdrawn: unsupported base-paper comparison. See docs/withdrawn/base_paper_comparison/WITHDRAWN.md")

import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "docs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

MODEL_COMPARISON = ROOT / "ml" / "reports" / "model_comparison.json"

with MODEL_COMPARISON.open("r", encoding="utf-8") as f:
    project = json.load(f)

# Base paper values (JMIR 2023; e46520)
base_smc = {
    "OSA": {"auroc": 0.897, "auprc": 0.877},
    "COMISA": {"auroc": 0.947, "auprc": 0.786},
    "Insomnia": {"auroc": 0.922, "auprc": 0.611},
}
base_ind = {
    "OSA": {"auroc": 0.930, "auprc": 0.940},
    "COMISA": {"auroc": 0.949, "auprc": 0.854},
    "Insomnia": {"auroc": 0.849, "auprc": 0.235},
}

# ---------- Chart 1: AUROC comparison ----------
fig1, ax1 = plt.subplots(figsize=(12.5, 6.5))
labels = ["Base OSA", "Base COMISA", "Base Insomnia", "Your RF", "Your XGB", "Your Hybrid"]
values = [
    base_smc["OSA"]["auroc"],
    base_smc["COMISA"]["auroc"],
    base_smc["Insomnia"]["auroc"],
    project["rf"]["roc_auc"],
    project["xgb"]["roc_auc"],
    project["hybrid"]["roc_auc"],
]
colors = ["#5B8FF9", "#5B8FF9", "#5B8FF9", "#61DDAA", "#61DDAA", "#61DDAA"]

bars = ax1.bar(labels, values, color=colors)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("Score")
ax1.set_title("Base Paper (SMC) vs Your Project: ROC-AUC / AUROC")
ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.tick_params(axis="x", rotation=20)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

for b, v in zip(bars, values):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

legend_txt = "Base bars: AUROC (OSA/COMISA/Insomnia). Your bars: ROC-AUC (Good/Poor sleep classification)."
ax1.text(0.01, -0.2, legend_txt, transform=ax1.transAxes, fontsize=9)
fig1.tight_layout()

png1 = REPORT_DIR / "charts" / "base_vs_project_auc_bar.png"
pdf1 = REPORT_DIR / "charts" / "base_vs_project_auc_bar.pdf"
svg1 = REPORT_DIR / "charts" / "base_vs_project_auc_bar.svg"
png1.parent.mkdir(parents=True, exist_ok=True)
fig1.savefig(png1, dpi=600, bbox_inches="tight", facecolor="white")
fig1.savefig(pdf1, bbox_inches="tight", facecolor="white")
fig1.savefig(svg1, bbox_inches="tight", facecolor="white")

# ---------- Chart 2: base AUROC/AUPRC per dataset ----------
fig2, axs = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)

disorders = ["OSA", "COMISA", "Insomnia"]
x = np.arange(len(disorders))
width = 0.35

smc_auroc = [base_smc[d]["auroc"] for d in disorders]
smc_auprc = [base_smc[d]["auprc"] for d in disorders]
ind_auroc = [base_ind[d]["auroc"] for d in disorders]
ind_auprc = [base_ind[d]["auprc"] for d in disorders]

axs[0].bar(x - width / 2, smc_auroc, width, label="AUROC", color="#5B8FF9")
axs[0].bar(x + width / 2, smc_auprc, width, label="AUPRC", color="#F6BD16")
axs[0].set_title("Base Paper: SMC Testing")
axs[0].set_xticks(x)
axs[0].set_xticklabels(disorders)
axs[0].set_ylim(0, 1.05)
axs[0].grid(axis="y", linestyle="--", alpha=0.35)
axs[0].legend()
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)

axs[1].bar(x - width / 2, ind_auroc, width, label="AUROC", color="#5B8FF9")
axs[1].bar(x + width / 2, ind_auprc, width, label="AUPRC", color="#F6BD16")
axs[1].set_title("Base Paper: Independent EUMCSH")
axs[1].set_xticks(x)
axs[1].set_xticklabels(disorders)
axs[1].grid(axis="y", linestyle="--", alpha=0.35)
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)

fig2.suptitle("Base Paper Detailed Performance by Disorder")
fig2.tight_layout()

png2 = REPORT_DIR / "charts" / "base_paper_disorder_bars.png"
pdf2 = REPORT_DIR / "charts" / "base_paper_disorder_bars.pdf"
svg2 = REPORT_DIR / "charts" / "base_paper_disorder_bars.svg"
fig2.savefig(png2, dpi=600, bbox_inches="tight", facecolor="white")
fig2.savefig(pdf2, bbox_inches="tight", facecolor="white")
fig2.savefig(svg2, bbox_inches="tight", facecolor="white")

# ---------- PDF report with text + charts ----------
out_pdf = REPORT_DIR / "BasePaper_vs_YourProject_Comparison_WithCharts.pdf"

with PdfPages(out_pdf) as pdf:
    # Page 1: Summary text
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")

    title = "Base Paper vs Your Project: Quantitative Comparison"
    body = """
Base paper: "Predicting the Risk of Sleep Disorders Using a Machine Learning–Based Simple Questionnaire" (JMIR 2023; e46520)

Base paper reported metrics:
- SMC test set: OSA AUROC 0.897, COMISA AUROC 0.947, Insomnia AUROC 0.922
- Independent set: OSA AUROC 0.930, COMISA AUROC 0.949, Insomnia AUROC 0.849
- Also reported AUPRC values per disorder (see chart pages).

Your project (current trained run):
- RF: ROC-AUC 0.987, Accuracy 0.960, F1 0.903, RMSE 3.606, R^2 0.933
- XGB: ROC-AUC 0.971, Accuracy 0.947, F1 0.875, RMSE 4.248, R^2 0.907
- Hybrid: ROC-AUC 0.982, Accuracy 0.920, F1 0.813, RMSE 6.165, R^2 0.804

Interpretation note:
This is a task-level comparison, not direct benchmark equivalence:
- Base paper predicts disorder risks (OSA/COMISA/Insomnia).
- Your system predicts sleep efficiency (%) + Good/Poor label.

Conclusion:
- Both works show strong discrimination capability.
- Your project adds full-stack deployment, temporal screen behavior integration, and personalized explainability.
"""
    ax.text(0, 0.98, title, fontsize=16, fontweight="bold", va="top")

    wrapped = "\n".join(textwrap.fill(line, width=95) if line.strip() else "" for line in body.strip("\n").split("\n"))
    ax.text(0, 0.93, wrapped, fontsize=10.5, va="top", linespacing=1.45)
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: Chart 1
    pdf.savefig(fig1)
    # Page 3: Chart 2
    pdf.savefig(fig2)

plt.close(fig1)
plt.close(fig2)

print(f"Generated: {out_pdf}")
print(f"Saved charts: {png1}, {pdf1}, {svg1}, {png2}, {pdf2}, {svg2}")
