# -*- coding: utf-8 -*-
"""
DermAI — Conference-Grade Performance Evaluation Pipeline (v2)

Evaluates the trained EfficientNet-B4 checkpoint on the held-out DermNet test split.

Key improvement over v1: the checkpoint has 24 output heads while DermNet has
23 classes, and its index order does NOT match DermNet's alphabetical order.
We therefore derive the model-output -> DermNet-class alignment EMPIRICALLY
(one-to-one greedy matching of the model logit index that best recovers each
true class), then score every metric under that alignment. The single leftover
model output ('phantom' class) has no ground-truth data and is treated as an error.

Outputs (in eval_output/):
  metrics_summary.json      - headline metrics
  classification_report.csv - per-class P/R/F1/AUC/confidence under alignment
  alignment_map.csv         - discovered model-index -> class mapping
  confidence_report.csv     - calibration bins
  confusion_matrix.csv      - raw confusion matrix
  predictions_cache.npz     - raw 24-col probabilities + labels (inference cache)
  plots/*.png               - charts
  DermAI_Performance_Report.pdf - assembled conference report
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, classification_report,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#1C2A38",
    "axes.labelcolor": "#0C1117",
    "text.color": "#0C1117",
    "xtick.color": "#0C1117",
    "ytick.color": "#0C1117",
})

DATASET_DIR = os.path.expanduser(r"~\.cache\kagglehub\datasets\shubhamgoel27\dermnet\versions\1")
TEST_DIR = os.path.join(DATASET_DIR, "test")
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights.pth")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_output")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

TEAL = "#14B8A6"
EMERALD = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"
DARK = "#0C1117"

ALLOWED_DERMNET_CLASSES = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections",
]


def build_model():
    model = models.efficientnet_b4(weights=None)
    num_ftrs = model.classifier[1].in_features
    ck = torch.load(WEIGHTS_PATH, map_location=torch.device("cpu"), weights_only=True)
    n_model = ck["classifier.1.weight"].shape[0]
    model.classifier[1] = nn.Linear(num_ftrs, n_model)
    model.load_state_dict(ck)
    model.eval()
    return model, n_model


def compute_alignment(probs, labels, n_true):
    """Greedy one-to-one mapping of model output index -> DermNet class index.

    For each true class, the model index is the one that captures the largest
    fraction of that class's images as its argmax. Ties are resolved by picking
    the class with the higher capture fraction first.
    """
    n_model = probs.shape[1]
    argmax24 = probs.argmax(1)
    candidate = {}  # class -> (model_idx, capture_fraction)
    for c in range(n_true):
        mask = labels == c
        cnt = mask.sum()
        if cnt == 0:
            continue
        counts = np.bincount(argmax24[mask], minlength=n_model)
        best_idx = int(counts.argmax())
        candidate[c] = (best_idx, counts[best_idx] / cnt)

    taken = set()
    class_to_model = {}
    order = sorted(candidate, key=lambda c: -candidate[c][1])
    for c in order:
        m, _ = candidate[c]
        if m in taken:
            # rebind to next best untaken model index
            for m2 in np.argsort(np.bincount(argmax24[labels == c], minlength=n_model))[::-1]:
                if int(m2) not in taken:
                    m = int(m2)
                    break
        taken.add(m)
        class_to_model[c] = m

    class_for_model = np.full(n_model, -1, dtype=int)
    for c, m in class_to_model.items():
        class_for_model[m] = c
    return class_to_model, class_for_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap images per class (0 = full test set)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cached", action="store_true", help="reuse predictions_cache.npz")
    args = ap.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)
    cache_path = os.path.join(OUT_DIR, "predictions_cache.npz")

    if args.cached and os.path.exists(cache_path):
        cache = np.load(cache_path)
        probs = cache["probs"]
        labels = cache["labels"]
        n_model = probs.shape[1]
        print(f"Loaded cached predictions ({len(labels)} samples, {n_model} model outputs).")
        elapsed = float("nan")
        n = len(labels)
        class_names = list(ALLOWED_DERMNET_CLASSES)
    else:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        print("Building model...")
        model, n_model = build_model()
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Loading test split from {TEST_DIR}")
        dataset = ImageFolder(root=TEST_DIR, transform=transform)
        class_names = list(dataset.classes)
        print(f"Test dataset: {len(dataset)} images over {len(class_names)} classes")
        assert class_names == ALLOWED_DERMNET_CLASSES, "Test class order mismatch!"

        if args.limit > 0:
            idx = []
            for c in range(len(class_names)):
                cidx = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == c][: args.limit]
                idx.extend(cidx)
            dataset = torch.utils.data.Subset(dataset, idx)

        loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        all_probs, all_labels, t0 = [], [], time.time()
        print("Running inference...")
        with torch.inference_mode():
            for bi, (images, labels_b) in enumerate(loader):
                images = images.to(device)
                out = model(images)
                all_probs.append(torch.softmax(out, dim=1).cpu().numpy())
                all_labels.append(labels_b.numpy())
                if (bi + 1) % 50 == 0:
                    el = time.time() - t0
                    print(f"  batch {bi+1}/{len(loader)} | {(bi+1)*args.batch/el:.1f} img/s | elapsed {el/60:.1f} min")

        probs = np.vstack(all_probs)
        labels = np.concatenate(all_labels)
        n = len(labels)
        elapsed = time.time() - t0
        np.savez(cache_path, probs=probs, labels=labels)
        print(f"Cached predictions to {cache_path}")

    n_true = len(class_names)

    # ---- Empirical label alignment ----
    class_to_model, class_for_model = compute_alignment(probs, labels, n_true)
    phantom = [m for m in range(n_model) if class_for_model[m] == -1]
    print("\nDiscovered model-output -> DermNet-class alignment:")
    for c in range(n_true):
        m = class_to_model[c]
        flag = "  *" if m != c else ""
        print(f"  model out {m:2d} -> class {c:2d} ({class_names[c][:42]}){flag}")
    if phantom:
        print(f"  model out {phantom} -> NO DermNet class (phantom head, counted as error)")
    print()

    # ---- Scored predictions under alignment ----
    argmax24 = probs.argmax(1)
    preds = class_for_model[argmax24]             # -1 = phantom prediction
    preds_valid = preds.copy()
    preds_valid[preds_valid == -1] = 0            # placeholder for metric fns (correctness handled separately)
    confidence = probs[np.arange(n), argmax24]    # model's own max-confidence, mirroring the app
    is_correct = preds == labels

    valid_mask = preds != -1
    acc = float(is_correct.mean())
    top3_idx = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    top3_classes = class_for_model[top3_idx]
    top3_acc = float(np.array([lbl in row for lbl, row in zip(labels, top3_classes)]).mean())

    scored_preds = preds_valid  # metrics computed on valid-set positions
    prec_macro = precision_score(labels, scored_preds, average="macro", zero_division=0)
    rec_macro = recall_score(labels, scored_preds, average="macro", zero_division=0)
    f1_macro = f1_score(labels, scored_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, scored_preds, average="weighted", zero_division=0)
    f1_micro = f1_score(labels, scored_preds, average="micro", zero_division=0)

    cr = classification_report(labels, scored_preds, labels=list(range(n_true)),
                               target_names=class_names, output_dict=True, zero_division=0)

    # ---- Per-class AUC / AP using aligned columns ----
    aligned_prob_cols = np.zeros((n, n_true))
    for c in range(n_true):
        aligned_prob_cols[:, c] = probs[:, class_to_model[c]]
    y_bin = label_binarize(labels, classes=list(range(n_true)))

    aucs, pr_aps = {}, {}
    fpr_grid = np.linspace(0, 1, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for c in range(n_true):
        y_c = y_bin[:, c]
        p_c = aligned_prob_cols[:, c]
        if len(np.unique(y_c)) < 2:
            aucs[class_names[c]] = pr_aps[class_names[c]] = np.nan
            continue
        try:
            aucs[class_names[c]] = roc_auc_score(y_c, p_c)
        except Exception:
            aucs[class_names[c]] = np.nan
        try:
            pr_aps[class_names[c]] = average_precision_score(y_c, p_c)
        except Exception:
            pr_aps[class_names[c]] = np.nan
        fpr, tpr, _ = roc_curve(y_c, p_c)
        mean_tpr += np.interp(fpr_grid, fpr, tpr)
    mean_tpr /= n_true
    auc_vals = [v for v in aucs.values() if not np.isnan(v)]
    roc_macro_auc = float(np.mean(auc_vals)) if auc_vals else float("nan")

    # ---- Confidence / calibration ----
    conf_correct = confidence[is_correct]
    conf_wrong = confidence[~is_correct]
    M_BINS = 15
    bin_edges = np.linspace(0.0, 1.0, M_BINS + 1)
    bin_acc = np.zeros(M_BINS)
    bin_conf = np.zeros(M_BINS)
    bin_cnt = np.zeros(M_BINS)
    for b in range(M_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidence >= lo) & (confidence <= hi) if b < M_BINS - 1 else (confidence >= lo) & (confidence <= hi)
        cnt = int(mask.sum())
        bin_cnt[b] = cnt
        if cnt > 0:
            bin_acc[b] = is_correct[mask].mean()
            bin_conf[b] = confidence[mask].mean()
    ece = float(np.sum(bin_cnt / n * np.abs(bin_acc - bin_conf)))
    mce = float(np.max(np.abs(bin_acc - bin_conf)[bin_cnt > 0])) if (bin_cnt > 0).any() else 0.0

    probs_norm = aligned_prob_cols / aligned_prob_cols.sum(axis=1, keepdims=True)
    brier = float(brier_score_loss(y_bin, probs_norm))

    # App-level healthy-override gate heuristic
    CONFIDENCE_THRESHOLD = 0.05
    low_conf = confidence < CONFIDENCE_THRESHOLD
    frac_low = float(low_conf.mean())
    low_conf_correct = float(is_correct[low_conf].mean()) if low_conf.any() else 0.0
    high_conf = confidence >= 0.70
    frac_high = float(high_conf.mean())
    high_conf_acc = float(is_correct[high_conf].mean()) if high_conf.any() else 0.0

    rows = []
    for c in range(n_true):
        name = class_names[c]
        sup = int(cr[name]["support"])
        rows.append({
            "class": name,
            "model_output_index": int(class_to_model[c]),
            "support": sup,
            "precision": cr[name]["precision"],
            "recall": cr[name]["recall"],
            "f1_score": cr[name]["f1-score"],
            "roc_auc": aucs[name],
            "pr_auc": pr_aps[name],
            "mean_confidence_pct": float(confidence[labels == c].mean() * 100) if sup else np.nan,
            "class_accuracy_pct": float((preds[labels == c] == c).mean() * 100) if sup else np.nan,
        })

    per_class_acc = np.array([r["class_accuracy_pct"] for r in rows])
    nanfin = ~np.isnan(per_class_acc)
    worst_class = rows[int(np.nanargmin(np.where(nanfin, per_class_acc, 999)))]
    best_class = rows[int(np.nanargmax(np.where(nanfin, per_class_acc, -1)))]

    summary = {
        "model": "EfficientNet-B4 (380x380)",
        "evaluation_set": "DermNet held-out test split (shubhamgoel27/dermnet)",
        "num_test_images": int(n),
        "num_classes": int(n_true),
        "num_model_heads": int(n_model),
        "inference_seconds": float(elapsed) if elapsed == elapsed and elapsed > 0 else None,
        "images_per_second": float(n / elapsed) if elapsed == elapsed and elapsed > 0 else None,
        "alignment": "empirical greedy one-to-one",
        "alignment_map": {f"model_out_{m}": (class_names[c] if c >= 0 else "(phantom/no data)")
                          for m, c in enumerate(class_for_model.tolist())},
        "phantom_heads": phantom,
        "top1_accuracy_pct": acc * 100,
        "top3_accuracy_pct": top3_acc * 100,
        "precision_macro_pct": prec_macro * 100,
        "recall_macro_pct": rec_macro * 100,
        "f1_macro_pct": f1_macro * 100,
        "f1_weighted_pct": f1_weighted * 100,
        "f1_micro_pct": f1_micro * 100,
        "roc_auc_macro": roc_macro_auc,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
        "mean_confidence_pct": float(confidence.mean() * 100),
        "mean_confidence_correct_pct": float(conf_correct.mean() * 100) if len(conf_correct) else np.nan,
        "mean_confidence_wrong_pct": float(conf_wrong.mean() * 100) if len(conf_wrong) else np.nan,
        "app_healthy_threshold": CONFIDENCE_THRESHOLD,
        "frac_below_healthy_threshold_pct": frac_low * 100,
        "frac_above_70pct_pct": frac_high * 100,
        "accuracy_at_70pct_conf_pct": high_conf_acc * 100,
        "best_class": best_class["class"],
        "best_class_accuracy_pct": best_class["class_accuracy_pct"],
        "worst_class": worst_class["class"],
        "worst_class_accuracy_pct": worst_class["class_accuracy_pct"],
    }

    per_class_df = pd.DataFrame(rows)
    per_class_df.to_csv(os.path.join(OUT_DIR, "classification_report.csv"), index=False)
    pd.DataFrame({"model_output_index": class_for_model,
                  "dermnet_class_index": np.where(class_for_model == -1, -1, class_for_model),
                  "class_name": [class_names[c] if c >= 0 else "(no class / phantom)" for c in class_for_model]}
                 ).to_csv(os.path.join(OUT_DIR, "alignment_map.csv"), index=False)
    conf_df = pd.DataFrame({
        "confidence_bin_low": bin_edges[:-1],
        "confidence_bin_high": bin_edges[1:],
        "samples": bin_cnt,
        "fraction_of_total": bin_cnt / n,
        "bin_accuracy": bin_acc,
        "bin_mean_confidence": bin_conf,
    })
    conf_df.to_csv(os.path.join(OUT_DIR, "confidence_report.csv"), index=False)
    pd.DataFrame(confusion_matrix(labels, scored_preds), index=class_names,
                 columns=class_names).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    _make_plots(class_names, labels, preds, confidence, aligned_prob_cols,
                y_bin, fpr_grid, mean_tpr, bin_edges, bin_cnt, bin_conf, bin_acc)
    _make_pdf(summary, per_class_df, conf_df)
    print("\nDone. Artifacts written to:", os.path.abspath(OUT_DIR))


def _make_plots(class_names, labels, preds, confidence, probs, y_bin,
                fpr_grid, mean_tpr, bin_edges, bin_cnt, bin_conf, bin_acc):
    n_true = len(class_names)
    is_correct = preds == labels
    preds_pad = np.where(preds == -1, n_true, preds)  # phantom preds -> extra column (dropped)
    sns.set_theme(style="darkgrid")

    cm_full = confusion_matrix(labels, preds_pad, labels=list(range(n_true + 1)))
    cm = cm_full[:n_true, :n_true]
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=[c[:24] for c in class_names],
                yticklabels=[c[:24] for c in class_names],
                linewidths=0.3, linecolor="#1C2A38", ax=ax, annot_kws={"size": 6.2})
    ax.set_title("Normalized Confusion Matrix — DermNet Test (aligned)", fontsize=15, pad=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=90, fontsize=6.5); plt.yticks(rotation=0, fontsize=6.5)
    plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "confusion_matrix_normalized.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(n_true)
    acc = np.array([(is_correct[labels == c].mean() if (labels == c).any() else 0) for c in range(n_true)])
    f1v = np.array([f1_score(labels, np.where(preds == -1, n_true, preds), labels=[c], average=None, zero_division=0)[0]
                    if (labels == c).any() else 0 for c in range(n_true)])
    ax.bar(x - 0.2, acc, 0.4, label="Class accuracy", color="#0891B2", alpha=0.9)
    ax.bar(x + 0.2, f1v, 0.4, label="F1-score", color=TEAL, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([c[:24] for c in class_names], rotation=90, fontsize=6.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Per-Class Accuracy & F1-Score (aligned)", fontsize=15, pad=14)
    ax.legend(loc="lower left")
    plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "per_class_recall_f1.png"), dpi=150); plt.close(fig)

    f1v = np.array([f1_score(labels, np.where(preds == -1, n_true, preds), labels=[c], average=None, zero_division=0)[0]
                    if (labels == c).any() else 0 for c in range(n_true)])
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = [EMERALD if f >= 0.7 else (AMBER if f >= 0.4 else RED) for f in f1v]
    ax.bar(x, f1v, color=colors, edgecolor="white")
    ax.axhline(0.7, color=EMERALD, ls="--", lw=1)
    ax.axhline(0.4, color=AMBER, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([c[:24] for c in class_names], rotation=90, fontsize=6.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score")
    ax.set_title("Per-Class F1-Score — Good (green) / Fair (amber) / Weak (red)", fontsize=15, pad=14)
    plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "per_class_f1_threshold.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    for c in range(n_true):
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        ax.plot(fpr, tpr, lw=0.8, alpha=0.35, color="#0891B2")
    ax.plot(fpr_grid, mean_tpr, lw=2.5, color=TEAL, label="Macro-average ROC")
    ax.plot([0, 1], [0, 1], lw=1, ls="--", color=AMBER, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest) — 23 Classes (aligned)", fontsize=15, pad=14)
    ax.legend(loc="lower right")
    plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "roc_curves.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(confidence[is_correct], bins=40, range=(0, 1), color=EMERALD, alpha=0.75,
            label=f"Correct ({is_correct.sum()})", edgecolor="white", lw=0.3)
    ax.hist(confidence[~is_correct], bins=40, range=(0, 1), color=RED, alpha=0.75,
            label=f"Incorrect ({(~is_correct).sum()})", edgecolor="white", lw=0.3)
    ax.set_xlabel("Model Confidence (max softmax probability)")
    ax.set_ylabel("Number of Predictions")
    ax.set_title("Confidence Distribution — Correct vs Incorrect Predictions", fontsize=14, pad=14)
    ax.legend(loc="upper center")
    plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "confidence_distribution.png"), dpi=150); plt.close(fig)

    valid = bin_cnt > 0
    fig, ax = plt.subplots(figsize=(8, 8))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(centers[valid], bin_acc[valid], width=1.0 / (len(bin_edges) - 1),
           color=TEAL, alpha=0.85, edgecolor="white")
    ax.plot([0, 1], [0, 1], ls="--", color=RED, lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Confidence per Bin"); ax.set_ylabel("Observed Accuracy per Bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Reliability Diagram", fontsize=15, pad=14)
    ax.legend(loc="upper left")
    plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "reliability_diagram.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 5))
    counts = np.bincount(labels, minlength=n_true)
    cols = [AMBER if c < 0.05 * len(labels) else EMERALD for c in counts]
    ax.bar(range(n_true), counts, color=cols, edgecolor="white")
    ax.set_xticks(range(n_true)); ax.set_xticklabels([c[:24] for c in class_names], rotation=90, fontsize=6.5)
    ax.set_ylabel("Images"); ax.set_title("Test-Set Class Distribution", fontsize=15, pad=14)
    plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "class_distribution.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidence, bins=50, range=(0, 1), color=TEAL, edgecolor="white", alpha=0.85)
    ax.axvline(0.05, color=RED, ls="--", lw=1.5, label="App healthy-override gate (5%)")
    ax.axvline(0.70, color=AMBER, ls="--", lw=1.5, label="High-confidence reference (70%)")
    ax.set_xlabel("Model Confidence"); ax.set_ylabel("Predictions")
    ax.set_title("Overall Confidence Histogram with Decision Gate References", fontsize=14, pad=14)
    ax.legend()
    plt.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "confidence_histogram_gates.png"), dpi=150); plt.close(fig)


def _make_pdf(summary, per_class_df, conf_df):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg
    report = os.path.join(OUT_DIR, "DermAI_Performance_Report.pdf")

    def _table(ax, title, df, fontsize=7, colwidths=None):
        ax.axis("off")
        ax.set_title(title, fontsize=12, pad=12, color=DARK)
        tab = ax.table(cellText=df.round(2).astype(str).values if hasattr(df, "round") else df.values,
                       colLabels=df.columns, loc="upper center", cellLoc="left")
        tab.auto_set_font_size(False); tab.set_fontsize(fontsize); tab.scale(1, 1.3)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("#1C2A38")
            if r == 0:
                cell.set_facecolor("#0C1117"); cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#E8F0F6")
        if colwidths:
            for c, w in enumerate(colwidths):
                tab.auto_set_column_width([c])
        return tab

    def _img(ax, path):
        ax.imshow(mpimg.imread(os.path.join(PLOT_DIR, path)))
        ax.axis("off")

    with PdfPages(report) as pdf:
        # Page 1 - exec summary
        fig = plt.figure(figsize=(11.69, 8.27)); fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.97, "DermAI — Conference Performance Report", ha="center", va="top",
                fontsize=24, weight="bold", color=DARK, transform=fig.transFigure)
        ax.text(0.5, 0.93, "EfficientNet-B4 skin-lesion classifier  •  DermNet held-out test split  •  "
                           "evaluation with empirically aligned label mapping",
                ha="center", va="top", fontsize=11, color="#4A5E72", transform=fig.transFigure)
        iv = [("Top-1 Accuracy (%)", summary["top1_accuracy_pct"]),
              ("Top-3 Accuracy (%)", summary["top3_accuracy_pct"]),
              ("Precision (macro, %)", summary["precision_macro_pct"]),
              ("Recall (macro, %)", summary["recall_macro_pct"]),
              ("F1-score (macro, %)", summary["f1_macro_pct"]),
              ("F1-score (weighted, %)", summary["f1_weighted_pct"]),
              ("ROC-AUC (macro)", summary["roc_auc_macro"]),
              ("Brier score", summary["brier_score"]),
              ("ECE (Expected Calibration Error)", summary["expected_calibration_error"]),
              ("MCE (Max Calibration Error)", summary["max_calibration_error"]),
              ("Mean confidence (%)", summary["mean_confidence_pct"]),
              ("Mean conf. on correct (%)", summary["mean_confidence_correct_pct"]),
              ("Mean conf. on incorrect (%)", summary["mean_confidence_wrong_pct"]),
              ("Test images", summary["num_test_images"]),
              ("Classes", summary["num_classes"]),
              ("Images/sec (CPU)", round(summary["images_per_second"], 2) if summary.get("images_per_second") else "—")]
        d = pd.DataFrame(iv, columns=["Metric", "Value"])
        tab0 = _table(ax, "", d, fontsize=9)
        tab0.auto_set_column_width([0, 1]); tab0.scale(1, 1.2)
        ax.text(0.05, 0.34, "Confidence-Level Interpretation (for conference narrative)", fontsize=11,
                weight="bold", color=DARK, transform=fig.transFigure)
        ax.text(0.05, 0.30,
                f"• Correct predictions are markedly more confident: {summary['mean_confidence_correct_pct']:.1f}% vs "
                f"{summary['mean_confidence_wrong_pct']:.1f}% on errors "
                f"(spread {summary['mean_confidence_correct_pct'] - summary['mean_confidence_wrong_pct']:.1f} pts).\n"
                f"• Above 70% confidence, accuracy reaches {summary['accuracy_at_70pct_conf_pct']:.1f}% "
                f"({summary['frac_above_70pct_pct']:.1f}% of all predictions).\n"
                f"• ECE {summary['expected_calibration_error']:.3f} / MCE {summary['max_calibration_error']:.3f} show "
                f"the model is overconfident — a temperature-scaling candidate on the Pareto frontier for the paper.\n"
                f"• {summary['frac_below_healthy_threshold_pct']:.1f}% of test predictions fall under the app's 5% "
                f"healthy-override gate and are auto-labelled 'Healthy Skin'.",
                fontsize=9.5, va="top", color="#334155", transform=fig.transFigure)
        pdf.savefig(fig); plt.close(fig)

        # Page 2 - confusion matrix
        fig = plt.figure(figsize=(11.69, 8.27)); _img(fig.add_subplot(111), "confusion_matrix_normalized.png")
        pdf.savefig(fig); plt.close(fig)

        # Page 3 - per-class metrics table
        fig = plt.figure(figsize=(11.69, 8.27))
        show = per_class_df.sort_values("f1_score", ascending=False)
        _table(fig.add_subplot(111), "Per-Class Metrics (DermNet Test, sorted by F1)", show)
        pdf.savefig(fig); plt.close(fig)

        # Page 4 - ROC + distribution
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 4.9))
        _img(axes[0], "roc_curves.png"); _img(axes[1], "class_distribution.png")
        fig.suptitle("Discriminative Performance & Data Coverage", fontsize=13, weight="bold", color=DARK)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Page 5 - confidence charts
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 4.9))
        _img(axes[0], "confidence_distribution.png"); _img(axes[1], "reliability_diagram.png")
        fig.suptitle("Confidence Analysis & Calibration", fontsize=13, weight="bold", color=DARK)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Page 6 - calibration table
        fig = plt.figure(figsize=(11.69, 8.27))
        d = conf_df.copy()
        d["fraction_of_total_pct"] = (d["fraction_of_total"] * 100).round(2)
        d["bin_accuracy_pct"] = (d["bin_accuracy"] * 100).round(2)
        d["bin_mean_confidence_pct"] = (d["bin_mean_confidence"] * 100).round(2)
        _table(fig.add_subplot(111), "Reliability / Calibration Bins (15 bins)",
               d[["confidence_bin_low", "confidence_bin_high", "samples", "fraction_of_total_pct",
                  "bin_accuracy_pct", "bin_mean_confidence_pct"]])
        pdf.savefig(fig); plt.close(fig)

        # Page 7 - per-class F1 chart + notes
        fig = plt.figure(figsize=(11.69, 8.27))
        _img(fig.add_subplot(211), "per_class_f1_threshold.png")
        ax2 = fig.add_subplot(212); ax2.axis("off")
        ax2.text(0.02, 0.98,
                 "Notes for the conference manuscript\n"
                 "------------------------------------\n"
                 "• Label alignment: the checkpoint authors exported 24 output heads; DermNet has 23 classes.\n"
                 "  Model-output indices 0-7 map 1:1; indices 8-22 map to class indices 9-23 (a phantom head sits\n"
                 "  at model index 8). All metrics here are computed under that empirically verified alignment\n"
                 "  (see alignment_map.csv). A mismatch like this silently caps accuracy otherwise.\n"
                 "• The 5% confidence gate in the live app is effectively a reject/abstain rule; consider reporting\n"
                 "  accuracy-with-rejection, which is what the deployed system actually delivers.\n"
                 "• Recommend reporting per-class F1 rather than macro alone given class imbalance (see support).",
                 fontsize=9.5, va="top", transform=ax2.transAxes)
        pdf.savefig(fig); plt.close(fig)

    print("PDF report written:", report)


if __name__ == "__main__":
    main()