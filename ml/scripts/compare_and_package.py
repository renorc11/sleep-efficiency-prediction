from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def run() -> None:
    rep = Path("ml/reports")
    art = Path("ml/artifacts")

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

    with (rep / "tabular_metrics.json").open("r", encoding="utf-8") as f:
        tab = json.load(f)
    with (rep / "hybrid_metrics.json").open("r", encoding="utf-8") as f:
        hybrid = json.load(f)["hybrid"]

    comparison = {
        "rf": tab["rf"],
        "xgb": tab["xgb"],
        "hybrid": hybrid,
    }

    with (rep / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Save metadata for backend loading.
    with (art / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": tab["feature_columns"],
                "quality_threshold": tab["quality_threshold"],
                "metrics_summary": comparison,
            },
            f,
            indent=2,
        )

    metrics = ["mae", "rmse", "r2", "f1", "roc_auc"]
    models = ["rf", "xgb", "hybrid"]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    width = 0.25
    for i, m in enumerate(models):
        values = [comparison[m][k] for k in metrics]
        ax.bar([x + i * width for x in range(len(metrics))], values, width=width, label=m.upper())

    ax.set_xticks([x + width for x in range(len(metrics))])
    ax.set_xticklabels([k.upper() for k in metrics])
    ax.set_title("Model Performance Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(rep / "model_comparison.png", dpi=600, facecolor="white")
    plt.savefig(rep / "model_comparison.pdf", facecolor="white")
    plt.savefig(rep / "model_comparison.svg", facecolor="white")

    print("Saved model comparison report and metadata.")


if __name__ == "__main__":
    run()
