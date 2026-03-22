import wandb
import pandas as pd
from pathlib import Path


ENTITY = "dani-allegue"
PROJECT = "TabModels"

RUN_ID_TO_EVAL = {
    "7osbiwqr": "vanilla (20k)",
    "giuu6flf" : "clogas-v1 (20k)"
}

DATASETS_FILE = "benchmarking/datasets.txt"

METRICS = ["accuracy", "auc", "log_loss"]

datasets = [    
    d.strip()
    for d in Path(DATASETS_FILE).read_text().splitlines()
    if d.strip()
]

api = wandb.Api()

rows = []

def find_metric(summary, dataset, metric):
    for k, v in summary.items():
        if dataset in k and metric in k:
            return v
    return None


for run_id, eval_name in RUN_ID_TO_EVAL.items():
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    print(f"\n🔍 Processing run {run_id}")

    for dataset in datasets:
        row = {
            "run_id": run_id,
            "eval": eval_name,
            "dataset": dataset,
        }

        for metric in METRICS:
            row[metric] = find_metric(run.summary, dataset, metric)

        if any(row[m] is not None for m in METRICS):
            rows.append(row)
        else:
            print(f"⚠️  No metrics found for dataset={dataset}")

df = pd.DataFrame(rows)

df.to_csv("eval_metrics.csv", index=False)
df.to_parquet("eval_metrics.parquet")

print(f"\n✅ Saved {len(df)} rows")

import numpy as np
import matplotlib.pyplot as plt

agg_raw = (
    df.groupby("eval")[["accuracy", "auc"]]
      .mean()
      .reset_index()
)

norm_df = df.copy()
for metric in ["accuracy", "auc"]:
    norm_df[metric] = (
        norm_df.groupby("dataset")[metric]
        .transform(lambda x: (x - x.mean()) / x.std())
    )

agg_norm = (
    norm_df.groupby("eval")[["accuracy", "auc"]]
           .mean()
           .reset_index()
)

baseline = df[df["eval"] == "vanilla (20k)"]

delta_df = df.merge(
    baseline,
    on="dataset",
    suffixes=("", "_vanilla")
)

delta_df["accuracy_gain"] = delta_df["accuracy"] - delta_df["accuracy_vanilla"]
delta_df["auc_gain"] = delta_df["auc"] - delta_df["auc_vanilla"]

agg_delta = (
    delta_df.groupby("eval")[["accuracy_gain", "auc_gain"]]
            .mean()
            .reset_index()
)

winners = (
    df.loc[df.groupby("dataset")["auc"].idxmax()]
      .groupby("eval")
      .size()
      .reset_index(name="wins")
)

print("\n🏆 Datasets won by each model (best AUC):\n")
winner_rows = df.loc[df.groupby("dataset")["auc"].idxmax()]
for eval_name, group in winner_rows.groupby("eval"):
    datasets_won = sorted(group["dataset"].tolist())
    print(f"🔹 {eval_name} ({len(datasets_won)} wins)")
    for d in datasets_won:
        print(f"   - {d}")
    print()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
x = np.arange(len(agg_raw))
width = 0.35

axes[0].bar(x - width/2, agg_raw["accuracy"], width, label="Accuracy")
axes[0].bar(x + width/2, agg_raw["auc"], width, label="AUC")
axes[0].set_title("Raw Average Metrics")
axes[0].set_xticks(x)
axes[0].set_xticklabels(agg_raw["eval"], rotation=30, ha="right")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.4)

axes[1].bar(x - width/2, agg_norm["accuracy"], width, label="Accuracy (z)")
axes[1].bar(x + width/2, agg_norm["auc"], width, label="AUC (z)")
axes[1].set_title("Normalized Performance (Per Dataset)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(agg_norm["eval"], rotation=30, ha="right")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.4)

axes[2].bar(x - width/2, agg_delta["accuracy_gain"], width, label="Accuracy Δ")
axes[2].bar(x + width/2, agg_delta["auc_gain"], width, label="AUC Δ")
axes[2].axhline(0, linestyle="--", linewidth=1)
axes[2].set_title("Improvement over eval-vanilla")
axes[2].set_xticks(x)
axes[2].set_xticklabels(agg_delta["eval"], rotation=30, ha="right")
axes[2].legend()
axes[2].grid(axis="y", alpha=0.4)

axes[3].bar(winners["eval"], winners["wins"])
axes[3].set_title("Win Count (Best AUC per Dataset)")
axes[3].set_ylabel("Number of Datasets Won")
axes[3].set_xticklabels(winners["eval"], rotation=30, ha="right")
axes[3].grid(axis="y", alpha=0.4)

plt.suptitle("Comprehensive Evaluation Comparison", fontsize=14)
plt.tight_layout()
#plt.savefig("eval_comparison_dashboard.pdf", dpi=200)
plt.show()
