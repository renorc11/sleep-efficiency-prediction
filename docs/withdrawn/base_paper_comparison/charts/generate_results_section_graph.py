# WITHDRAWN (2026-11-22). This script compared the project's ROC-AUC, computed from a
# rescaled regression output, with AUROC from a clinical sleep-disorder classification
# paper. Those numbers are not comparable. See ../WITHDRAWN.md. Its output paths also
# no longer resolve after the move, so it refuses to run.
raise SystemExit("Withdrawn: unsupported base-paper comparison. See docs/withdrawn/base_paper_comparison/WITHDRAWN.md")

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'docs' / 'charts'
out.mkdir(parents=True, exist_ok=True)

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

# Your project metrics
with open(ROOT / 'ml' / 'reports' / 'model_comparison.json', 'r', encoding='utf-8') as f:
    m = json.load(f)

# Base paper reported numbers (JMIR e46520, SMC test set)
base_disorders = ['OSA', 'COMISA', 'Insomnia']
base_auroc = [0.897, 0.947, 0.922]
base_auprc = [0.877, 0.786, 0.611]

# Your project (classification-style metrics)
your_models = ['RF', 'XGBoost', 'Hybrid']
your_acc = [m['rf']['accuracy'], m['xgb']['accuracy'], m['hybrid']['accuracy']]
your_f1 = [m['rf']['f1'], m['xgb']['f1'], m['hybrid']['f1']]

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
fig.patch.set_facecolor('white')

# Left subplot: Base paper
x = np.arange(len(base_disorders))
width = 0.36
b1 = axs[0].bar(x - width/2, base_auroc, width, label='AUROC', color='#4C78A8')
b2 = axs[0].bar(x + width/2, base_auprc, width, label='AUPRC', color='#F58518')
axs[0].set_title('Reference Study Results (SMC Test Set)', fontweight='bold')
axs[0].set_xticks(x)
axs[0].set_xticklabels(base_disorders)
axs[0].set_ylim(0, 1.1)
axs[0].set_ylabel('Score')
axs[0].legend(loc='upper right')
axs[0].grid(axis='y', linestyle='--', alpha=0.28)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.2f}', ha='center', va='bottom', fontsize=9)

# Right subplot: Your project
x2 = np.arange(len(your_models))
c1 = axs[1].bar(x2 - width/2, your_acc, width, label='Accuracy', color='#54A24B')
c2 = axs[1].bar(x2 + width/2, your_f1, width, label='F1-Score', color='#E45756')
axs[1].set_title('Proposed System Results', fontweight='bold')
axs[1].set_xticks(x2)
axs[1].set_xticklabels(your_models)
axs[1].set_ylim(0, 1.1)
axs[1].set_ylabel('Score')
axs[1].legend(loc='upper right')
axs[1].grid(axis='y', linestyle='--', alpha=0.28)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

for bar in list(c1) + list(c2):
    h = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.2f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('Comparative Results: Reference Study vs Proposed System', fontsize=15, fontweight='bold')
fig.text(
    0.5,
    0.01,
    'Note: Metrics differ by task definition. Reference study reports disorder-risk AUROC/AUPRC; proposed system reports sleep-quality Accuracy/F1.',
    ha='center',
    fontsize=9,
)

plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])
out_png = out / 'results_section_base_vs_project_graph.png'
out_pdf = out / 'results_section_base_vs_project_graph.pdf'
out_svg = out / 'results_section_base_vs_project_graph.svg'
fig.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
fig.savefig(out_svg, bbox_inches='tight', facecolor='white')
print(out_png)
print(out_pdf)
print(out_svg)
