"""
CLoGAS Gating Analysis
======================
Pulls per-block per-head CLoGAS diagnostics from wandb across multiple runs,
aggregates them, and visualises what the gating mechanism has learned.

Metrics (logged each training step):
  clogas/block_{i}/head_{j}/tau          — per-head softmax temperature
  clogas/block_{i}/head_{j}/mean_g       — mean P(correct class) at key positions
  clogas/block_{i}/head_{j}/mean_bias    — mean additive log-bias (always ≤ 0)
  clogas/block_{i}/head_{j}/gamma_entropy — entropy H(gamma) of the class dist.
  clogas/block_{i}/beta                  — scalar gating strength (shared)

Usage
-----
Edit RUN_IDS and LABELS below, then run:
    python benchmarking/clogas_analysis.py
"""

import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import wandb

# ── Configuration ────────────────────────────────────────────────────────────

ENTITY  = "dani-allegue"
PROJECT = "TabModels"

# Three run IDs to aggregate over
RUN_IDS = [
    "uoil1p5a",
    "00qwggqs",
]
LABELS = {rid: rid for rid in RUN_IDS}   # optionally give human-readable names

METRICS = ["tau", "mean_g", "mean_bias", "gamma_entropy"]

# Downsample history to at most this many steps per run (speeds up fetching)
MAX_HISTORY_SAMPLES = 2000

# ── Helpers ───────────────────────────────────────────────────────────────────

_KEY_RE = re.compile(
    r"^clogas/block_(\d+)/head_(\d+)/(\w+)$"
)
_BETA_RE = re.compile(r"^clogas/block_(\d+)/beta$")


def _parse_key(key: str):
    """Return (block, head, metric) or None for non-clogas keys."""
    m = _KEY_RE.match(key)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None


def fetch_run_history(api: wandb.Api, run_id: str) -> pd.DataFrame:
    """Download the full metric history for one run, downsampled."""
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    print(f"  fetching {run_id} ({run.name}) …", flush=True)

    # Identify clogas keys present in this run's summary
    clogas_keys = [k for k in run.summary.keys() if k.startswith("clogas/")]
    beta_keys   = [k for k in clogas_keys if _BETA_RE.match(k)]
    head_keys   = [k for k in clogas_keys if _KEY_RE.match(k)]
    all_keys    = head_keys + beta_keys + ["_step"]

    if not head_keys:
        warnings.warn(f"Run {run_id} has no clogas head metrics in summary — skipping.")
        return pd.DataFrame()

    samples = run.history(keys=all_keys, samples=MAX_HISTORY_SAMPLES, pandas=True)
    samples["run_id"] = run_id
    return samples


def build_records(df_history: pd.DataFrame) -> list[dict]:
    """
    Melt the wide history dataframe into long-format records:
      {run_id, step, block, head, metric, value}
    Also adds beta as a separate metric with head=-1.
    """
    records = []
    for _, row in df_history.iterrows():
        step   = row.get("_step", np.nan)
        run_id = row["run_id"]
        for col, val in row.items():
            if pd.isna(val) or col in ("_step", "run_id"):
                continue
            parsed = _parse_key(col)
            if parsed:
                block, head, metric = parsed
                records.append(dict(run_id=run_id, step=step,
                                    block=block, head=head,
                                    metric=metric, value=float(val)))
            else:
                bm = _BETA_RE.match(col)
                if bm:
                    records.append(dict(run_id=run_id, step=step,
                                        block=int(bm.group(1)), head=-1,
                                        metric="beta", value=float(val)))
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

api = wandb.Api()

print("Fetching run histories …")
all_history = []
for rid in RUN_IDS:
    h = fetch_run_history(api, rid)
    if not h.empty:
        all_history.append(h)

if not all_history:
    raise RuntimeError("No data fetched. Check RUN_IDS and that CLoGAS was enabled.")

raw_df = pd.concat(all_history, ignore_index=True)

print(f"Building records from {len(raw_df)} history rows …")
records = build_records(raw_df)
df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No clogas metrics found in history. Make sure diagnostics were logged.")

blocks = sorted(df["block"].unique())
heads  = sorted(df[df["head"] >= 0]["head"].unique())
n_blocks = len(blocks)
n_heads  = len(heads)

print(f"Found {n_blocks} blocks, {n_heads} heads, metrics: {sorted(df['metric'].unique())}")

# Save raw long-format data
df.to_csv("benchmarking/clogas_metrics.csv", index=False)
print("Saved benchmarking/clogas_metrics.csv")


# ── 1. Training curves (mean ± std across runs) ───────────────────────────────

def plot_training_curves(df: pd.DataFrame, metric: str, blocks_sample=None, heads_sample=None):
    """Plot aggregated training curves for a given metric."""
    blist = blocks_sample or blocks
    hlist = heads_sample or heads
    fig, axes = plt.subplots(len(blist), len(hlist),
                             figsize=(3.5 * len(hlist), 2.8 * len(blist)),
                             sharex=False, sharey=False, squeeze=False)

    sub = df[(df["metric"] == metric) & (df["head"].isin(hlist)) & (df["block"].isin(blist))]

    for bi, block in enumerate(blist):
        for hi, head in enumerate(hlist):
            ax = axes[bi][hi]
            cell = sub[(sub["block"] == block) & (sub["head"] == head)]
            if cell.empty:
                ax.set_visible(False)
                continue
            agg = cell.groupby("step")["value"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["step"], agg["mean"], linewidth=1.5)
            ax.fill_between(agg["step"],
                            agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            alpha=0.25)
            ax.set_title(f"B{block} H{head}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))
            ax.grid(alpha=0.3)
            if hi == 0:
                ax.set_ylabel(metric, fontsize=7)
            if bi == len(blist) - 1:
                ax.set_xlabel("step", fontsize=7)

    fig.suptitle(f"CLoGAS – {metric} per block/head (mean±std over {len(RUN_IDS)} runs)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ── 2. Final-value heatmaps ───────────────────────────────────────────────────

def final_value_heatmap(df: pd.DataFrame):
    """Heatmap of mean final-step value for each (block, head, metric)."""
    head_df  = df[df["head"] >= 0]
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(3, n_blocks * 0.6 + 1.5)))

    for mi, metric in enumerate(METRICS):
        ax = axes[mi]
        sub = head_df[head_df["metric"] == metric]
        # Take last 5% of steps as "final" (robust to noisy endings)
        last_step = sub["step"].max()
        cutoff    = last_step * 0.95
        final     = sub[sub["step"] >= cutoff].groupby(["block", "head"])["value"].mean().reset_index()
        pivot     = final.pivot(index="block", columns="head", values="value")
        pivot     = pivot.reindex(index=blocks, columns=heads)

        im = ax.imshow(pivot.values, aspect="auto",
                       cmap="RdYlGn" if metric == "mean_g" else "viridis")
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("Head")
        ax.set_ylabel("Block")
        ax.set_xticks(range(len(heads)));  ax.set_xticklabels(heads, fontsize=7)
        ax.set_yticks(range(len(blocks))); ax.set_yticklabels(blocks, fontsize=7)

        # Annotate cells
        for (bi, block), (hi, head) in [(a, b) for a in enumerate(blocks) for b in enumerate(heads)]:
            v = pivot.at[block, head]
            if not np.isnan(v):
                ax.text(hi, bi, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.5 else "black")

    fig.suptitle(f"CLoGAS final values (mean over last 5% steps, {len(RUN_IDS)} runs)")
    fig.tight_layout()
    return fig


# ── 3. Beta (gating strength) over time ──────────────────────────────────────

def plot_beta_curves(df: pd.DataFrame):
    beta_df = df[df["metric"] == "beta"]
    if beta_df.empty:
        print("No beta data found — skipping beta plot.")
        return None

    fig, axes = plt.subplots(1, n_blocks, figsize=(4 * n_blocks, 3), squeeze=False)
    for bi, block in enumerate(blocks):
        ax = axes[0][bi]
        sub = beta_df[beta_df["block"] == block]
        agg = sub.groupby("step")["value"].agg(["mean", "std"]).reset_index()
        ax.plot(agg["step"], agg["mean"], linewidth=1.5, color="tab:orange")
        ax.fill_between(agg["step"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        alpha=0.2, color="tab:orange")
        ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")
        ax.set_title(f"Block {block} β", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))

    fig.suptitle(f"CLoGAS β (gating strength) over training ({len(RUN_IDS)} runs mean±std)")
    fig.tight_layout()
    return fig


# ── 4. Per-head specialisation: final tau vs mean_g scatter ──────────────────

def plot_tau_vs_g(df: pd.DataFrame):
    """Scatter: final tau vs final mean_g, one point per (block, head)."""
    head_df  = df[df["head"] >= 0]
    last_step = head_df["step"].max()
    cutoff    = last_step * 0.95
    final     = head_df[head_df["step"] >= cutoff].groupby(["block", "head", "metric"])["value"].mean().reset_index()
    pivot     = final.pivot_table(index=["block", "head"], columns="metric", values="value").reset_index()

    if "tau" not in pivot.columns or "mean_g" not in pivot.columns:
        print("Missing tau or mean_g — skipping scatter.")
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(pivot["tau"], pivot["mean_g"],
                         c=pivot["block"], cmap="tab10",
                         s=80, alpha=0.8, edgecolors="k", linewidths=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Block index")

    for _, row in pivot.iterrows():
        ax.annotate(f"B{int(row['block'])}H{int(row['head'])}",
                    (row["tau"], row["mean_g"]),
                    fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("τ  (temperature — low = sharp class selection)")
    ax.set_ylabel("mean g  (P(correct class) at key positions)")
    ax.set_title("CLoGAS: temperature vs class selectivity per head\n"
                 "Top-right = sharp AND class-selective heads")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── 5. Summary table ─────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    head_df  = df[df["head"] >= 0]
    last_step = head_df["step"].max()
    cutoff    = last_step * 0.95
    final     = (head_df[head_df["step"] >= cutoff]
                 .groupby(["block", "head", "metric"])["value"]
                 .mean()
                 .reset_index()
                 .pivot_table(index=["block", "head"], columns="metric", values="value")
                 .reset_index())

    print("\n── Final CLoGAS values (mean over last 5% of training, all runs) ──")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_rows", 200):
        print(final.to_string(index=False))
    final.to_csv("benchmarking/clogas_final_values.csv", index=False)
    print("\nSaved benchmarking/clogas_final_values.csv")

    # Most class-selective heads (highest mean_g)
    if "mean_g" in final.columns:
        top = final.nlargest(5, "mean_g")[["block", "head", "mean_g", "tau", "gamma_entropy"]]
        print("\n Top-5 most class-selective heads (highest mean_g):")
        print(top.to_string(index=False))

    # Coldest heads (lowest tau → sharpest class selection)
    if "tau" in final.columns:
        cold = final.nsmallest(5, "tau")[["block", "head", "tau", "mean_g"]]
        print("\n Top-5 heads with lowest temperature (sharpest softmax):")
        print(cold.to_string(index=False))


# ── Run all analyses ──────────────────────────────────────────────────────────

print_summary(df)

# Sample at most 4 blocks and 4 heads for the curve grid (manageable)
blocks_sample = blocks[:4] if len(blocks) > 4 else blocks
heads_sample  = heads[:4]  if len(heads)  > 4 else heads

for metric in METRICS:
    fig = plot_training_curves(df, metric, blocks_sample, heads_sample)
    fig.savefig(f"benchmarking/clogas_curve_{metric}.pdf", bbox_inches="tight")
    print(f"Saved benchmarking/clogas_curve_{metric}.pdf")

fig_heat = final_value_heatmap(df)
fig_heat.savefig("benchmarking/clogas_heatmap.pdf", bbox_inches="tight")
print("Saved benchmarking/clogas_heatmap.pdf")

fig_beta = plot_beta_curves(df)
if fig_beta:
    fig_beta.savefig("benchmarking/clogas_beta.pdf", bbox_inches="tight")
    print("Saved benchmarking/clogas_beta.pdf")

fig_scatter = plot_tau_vs_g(df)
if fig_scatter:
    fig_scatter.savefig("benchmarking/clogas_tau_vs_g.pdf", bbox_inches="tight")
    print("Saved benchmarking/clogas_tau_vs_g.pdf")

plt.show()
