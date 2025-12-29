#!/usr/bin/env python
"""
Evaluate a (TabICL-style) checkpoint on TALENT-format datasets using
the same regime as the original TabICL paper.

- Uses TabICLClassifier from the `tabicl` package.
- Assumes datasets are stored in LAMDA-TALENT format:
    dataset_root/
        <dataset_name>/
            N_train.npy (optional)
            C_train.npy (optional)
            y_train.npy
            N_val.npy   (optional)
            C_val.npy   (optional)
            y_val.npy
            N_test.npy  (optional)
            C_test.npy  (optional)
            y_test.npy
            info.json   (optional, contains "task_type", etc.)

You control which datasets are evaluated via a text file listing names.
"""

import argparse
import json
import os
from pathlib import Path
import wandb

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from tabicl import TabICLClassifier


def load_split(dataset_dir: Path, split: str):
    """Load one split (train/val/test) from a TALENT-style dataset folder."""
    y_path = dataset_dir / f"y_{split}.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing labels file: {y_path}")

    y = np.load(y_path, allow_pickle=True)

    num_path = dataset_dir / f"N_{split}.npy"
    cat_path = dataset_dir / f"C_{split}.npy"

    num_df = None
    cat_df = None

    if num_path.exists():
        num = np.load(num_path, allow_pickle=True)
        if num.ndim == 1:
            num = num[:, None]
        num_cols = [f"num_{i}" for i in range(num.shape[1])]
        num_df = pd.DataFrame(num, columns=num_cols)

    if cat_path.exists():
        cat = np.load(cat_path, allow_pickle=True)
        if cat.ndim == 1:
            cat = cat[:, None]
        cat_cols = [f"cat_{i}" for i in range(cat.shape[1])]
        cat_df = pd.DataFrame(cat, columns=cat_cols)
        # Mark categorical columns properly so TabICL treats them as such
        for c in cat_df.columns:
            cat_df[c] = cat_df[c].astype("category")

    if num_df is not None and cat_df is not None:
        X = pd.concat([num_df, cat_df], axis=1)
    elif num_df is not None:
        X = num_df
    elif cat_df is not None:
        X = cat_df
    else:
        raise ValueError(f"No features found for split '{split}' in {dataset_dir}")

    return X, y


def load_task_type(dataset_dir: Path):
    """Read task_type from info.json if present; otherwise return None."""
    info_path = dataset_dir / "info.json"
    if not info_path.exists():
        return None

    with info_path.open("r") as f:
        info = json.load(f)
    return info.get("task_type", None)


def evaluate_dataset(
    dataset_name: str,
    dataset_root: Path,
    clf: TabICLClassifier,
    only_leq_10_classes: bool = False,
):
    """Fit on train, evaluate on test, return metrics dict or None if skipped."""
    ds_dir = dataset_root / dataset_name
    if not ds_dir.exists():
        print(f"[WARN] Dataset folder not found, skipping: {ds_dir}")
        return None

    task_type = load_task_type(ds_dir)
    if task_type not in (None, "binclass", "multiclass"):
        print(f"[INFO] Non-classification task ({task_type}) in {dataset_name}, skipping.")
        return None

    # Load splits
    X_train, y_train = load_split(ds_dir, "train")
    # val is not used by TabICL in the paper regime, but we load it in case
    # you want to inspect it or extend the script.
    try:
        X_val, y_val = load_split(ds_dir, "val")
    except FileNotFoundError:
        X_val, y_val = None, None

    X_test, y_test = load_split(ds_dir, "test")

    # Optionally enforce ≤10 classes (main 171 datasets in the paper)
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    if only_leq_10_classes and n_classes > 10:
        print(f"[INFO] {dataset_name}: {n_classes} classes (>10), skipping.")
        return None

    print(f"[INFO] Evaluating {dataset_name} (classes={n_classes}, n_train={len(y_train)}, n_test={len(y_test)})")

    # Fit on training data only, as in the paper
    clf.fit(X_train, y_train)

    # Evaluate on test
    y_pred = clf.predict(X_test)
    # Some datasets may be degenerate; guard against errors
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception as e:
        print(f"[WARN] Failed to get predict_proba for {dataset_name}: {e}")
        y_proba = None

    metrics = {
        "dataset": dataset_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(n_classes),
    }

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))

    # Log loss & AUC (if probabilities available)
    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_test, y_proba))
        except Exception as e:
            print(f"[WARN] log_loss failed for {dataset_name}: {e}")
            metrics["log_loss"] = float("nan")

        try:
            if n_classes == 2:
                # binary AUC
                if y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    # If probabilities are of shape (N,), treat as positive class prob
                    auc = roc_auc_score(y_test, y_proba)
            else:
                # multiclass AUC with one-vs-rest, macro-averaged
                auc = roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            metrics["auc"] = float(auc)
        except Exception as e:
            print(f"[WARN] AUC failed for {dataset_name}: {e}")
            metrics["auc"] = float("nan")
    else:
        metrics["log_loss"] = float("nan")
        metrics["auc"] = float("nan")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a TabICL checkpoint on TALENT datasets "
                    "using the original paper's inference regime."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the TabICL checkpoint."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory containing TALENT-style dataset folders."
    )
    parser.add_argument(
        "--dataset-list",
        type=str,
        required=True,
        help="Text file with dataset names (one per line)."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="tabicl_eval_results.csv",
        help="Where to save the per-dataset metrics CSV."
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--only-leq-10-classes", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--checkpoint-version",
        type=str,
        default="tabicl-classifier-v1-0208.ckpt"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="TabICL-Eval",
        help="W&B project name."
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (optional)."
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    checkpoint_dir = Path(args.checkpoint_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_dir}")

    # --------------------------------------------------------
    # W&B INITIALIZATION
    # --------------------------------------------------------
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"eval-{checkpoint_dir.stem}",
        config={
            "checkpoint": str(checkpoint_dir),
            "dataset_root": str(dataset_root),
            "dataset_list_file": args.dataset_list,
            "checkpoint_version": args.checkpoint_version,
            "device": args.device,
            "only_leq_10_classes": args.only_leq_10_classes,
        }
    )

    # --------------------------------------------------------
    # Read dataset names
    # --------------------------------------------------------
    with open(args.dataset_list, "r") as f:
        dataset_names = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    print(f"[INFO] Will evaluate {len(dataset_names)} datasets.")

    clf = TabICLClassifier(
        n_estimators=32,
        norm_methods=["none", "power"],
        feat_shuffle_method="latin",
        class_shift=True,
        outlier_threshold=4.0,
        softmax_temperature=0.9,
        average_logits=True,
        use_hierarchical=True,
        batch_size=8,
        use_amp=False,  # paper setting
        model_path=str(checkpoint_dir),
        allow_auto_download=False,
        checkpoint_version=args.checkpoint_version,
        device=args.device,
        random_state=42,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        inference_config=None,
    )

    all_metrics = []
    wandb_table = wandb.Table(
        columns=[
            "dataset", "n_train", "n_test", "n_classes",
            "accuracy", "log_loss", "auc"
        ]
    )

    for ds_name in dataset_names:
        m = evaluate_dataset(
            ds_name,
            dataset_root=dataset_root,
            clf=clf,
            only_leq_10_classes=args.only_leq_10_classes,
        )
        if m is not None:
            all_metrics.append(m)

            # Add row to W&B table
            wandb_table.add_data(
                m["dataset"],
                m["n_train"],
                m["n_test"],
                m["n_classes"],
                m["accuracy"],
                m["log_loss"],
                m["auc"]
            )

            # Log per-dataset metric directly for tracking
            wandb.log({
                f"{m['dataset']}/accuracy": m["accuracy"],
                f"{m['dataset']}/auc": m["auc"],
                f"{m['dataset']}/log_loss": m["log_loss"],
            })

    if not all_metrics:
        print("[WARN] No datasets were successfully evaluated. Nothing to write.")
        return

    df = pd.DataFrame(all_metrics).sort_values("dataset")
    output_path = Path(args.output_csv)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved results for {len(df)} datasets to {output_path}")

    # Upload CSV as artifact
    artifact = wandb.Artifact(
        name=f"eval-results-{checkpoint_dir.stem}",
        type="evaluation",
    )
    artifact.add_file(str(output_path))
    run.log_artifact(artifact)

    # Log summary metrics
    wandb.summary["mean_accuracy"] = df["accuracy"].mean()
    wandb.summary["mean_auc"] = df["auc"].mean()
    wandb.summary["mean_log_loss"] = df["log_loss"].mean()

    run.finish()

if __name__ == "__main__":
    main()
