import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DermAIColab_Analysis.ipynb")

md_intro = """# DermAI - Conference Performance & Confidence Analysis (Colab / T4)

Runs the same evaluation as the local `evaluate.py` on a Colab GPU.
It downloads the DermNet test split, loads `model_weights.pth` (upload it when prompted),
derives the model-output **label alignment empirically**, scores Accuracy / Precision / Recall /
F1 / ROC-AUC / Brier / ECE / MCE, and produces all charts inline.

Expected runtime on a T4: ~1-2 minutes (vs ~25 min on CPU).
"""

md_step1 = """## 1. Install deps (Kaggle GPU runtimes usually already have torch/torchvision)"""

cell_install = """import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])"""

md_step2 = """## 2. Upload your trained weights

Click the folder icon in the left sidebar (or the button below) and upload
`model_weights.pth` from your project folder:
`C:\\Users\\olivi\\Downloads\\miniproject\\dermaai\\model_weights.pth`
"""

cell_upload = """import os
if not os.path.exists("model_weights.pth"):
    from google.colab import files
    uploaded = files.upload()          # <-- select model_weights.pth
assert os.path.exists("model_weights.pth"), "model_weights.pth not found"
print("weights:", os.path.getsize("model_weights.pth"), "bytes")"""

md_step3 = """## 3. Download DermNet test split (~1 GB)"""

cell_data = """import kagglehub, os
path = kagglehub.dataset_download("shubhamgoel27/dermnet")
TEST_DIR = os.path.join(path, "test")
print("test dir:", TEST_DIR)
print("classes:", len(os.listdir(TEST_DIR)))"""

md_step4 = """## 4. Build the model (mirrors derma.py, incl. the 24-head adapter)"""

cell_model = """import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = models.efficientnet_b4(weights=None)
ck = torch.load("model_weights.pth", map_location="cpu", weights_only=True)
n_model = ck["classifier.1.weight"].shape[0]
model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_model)
model.load_state_dict(ck)
model.to(device).eval()
print("model heads:", n_model)"""

md_step5 = """## 5. Inference on the test split (batched, T4 GPU)"""

cell_infer = """import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = ImageFolder(root=TEST_DIR, transform=transform)
class_names = list(ds.classes)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
print(f"test images: {len(ds)} | classes: {len(class_names)}")

all_p, all_l = [], []
with torch.inference_mode():
    for i, (imgs, lbl) in enumerate(loader):
        out = model(imgs.to(device))
        all_p.append(torch.softmax(out, dim=1).cpu().numpy())
        all_l.append(lbl.numpy())
probs = np.vstack(all_p)
labels = np.concatenate(all_l)
print("done:", probs.shape)"""

md_step6 = """## 6. Empirical label alignment (24 heads -> 23 DermNet classes)

A phantom head exists at model index 8; classes map c -> c+1 for c >= 8.
Derived automatically, one-to-one, greedy.
"""

cell_align = """n_true = len(class_names)

def compute_alignment(probs, labels, n_true):
    n_model = probs.shape[1]
    argmax24 = probs.argmax(1)
    cand = {}
    for c in range(n_true):
        mask = labels == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(argmax24[mask], minlength=n_model)
        cand[c] = (int(counts.argmax()), counts.max() / mask.sum())
    taken, class_to_model = set(), {}
    for c in sorted(cand, key=lambda c: -cand[c][1]):
        m, _ = cand[c]
        if m in taken:
            for m2 in np.argsort(np.bincount(argmax24[labels == c], minlength=n_model))[::-1]:
                if int(m2) not in taken:
                    m = int(m2)
                    break
        taken.add(m); class_to_model[c] = m
    class_for_model = np.full(n_model, -1, dtype=int)
    for c, m in class_to_model.items():
        class_for_model[m] = c
    return class_to_model, class_for_model

class_to_model, class_for_model = compute_alignment(probs, labels, n_true)
phantom = [m for m in range(n_model) if class_for_model[m] == -1]
print("model output -> class mapping (asterisk = shifted/different from index):")
for c in range(n_true):
    m = class_to_model[c]
    print(f"  out {m:2d} -> class {c:2d} ({class_names[c][:44]}){'  *' if m != c else ''}")
print("phantom heads:", phantom)"""

md_step7 = """## 7. Performance matrix (Accuracy / P / R / F1 / AUC)"""

cell_metrics = """from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                      f1_score, confusion_matrix, roc_auc_score,
                      average_precision_score, brier_score_loss, classification_report,
                      roc_curve)
from sklearn.preprocessing import label_binarize
import pandas as pd

argmax24 = probs.argmax(1)
preds = class_for_model[argmax24]
preds_valid = preds.copy(); preds_valid[preds_valid == -1] = 0
confidence = probs[np.arange(len(probs)), argmax24]
is_correct = preds == labels

acc = is_correct.mean()
top3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]
top3_classes = class_for_model[top3]
top3_acc = np.mean([lbl in row for lbl, row in zip(labels, top3_classes)])

cr = classification_report(labels, preds_valid, labels=list(range(n_true)),
                           target_names=class_names, output_dict=True, zero_division=0)
p_mac = precision_score(labels, preds_valid, average="macro", zero_division=0)
r_mac = recall_score(labels, preds_valid, average="macro", zero_division=0)
f1_mac = f1_score(labels, preds_valid, average="macro", zero_division=0)

aligned = np.zeros((len(probs), n_true))
for c in range(n_true):
    aligned[:, c] = probs[:, class_to_model[c]]
y_bin = label_binarize(labels, classes=list(range(n_true)))
aucs = [roc_auc_score(y_bin[:, c], aligned[:, c]) for c in range(n_true)]
auc_macro = float(np.mean(aucs))

pn = aligned / aligned.sum(1, keepdims=True)
brier = float(brier_score_loss(y_bin, pn))

print("=" * 60)
print(f"Top-1 Accuracy : {acc*100:.2f}%     Top-3 Accuracy: {top3_acc*100:.2f}%")
print(f"Precision(mac) : {p_mac*100:.2f}%   Recall(mac): {r_mac*100:.2f}%")
print(f"F1-macro       : {f1_mac*100:.2f}%   F1-weighted: {cr['weighted avg']['f1-score']*100:.2f}%")
print(f"ROC-AUC macro  : {auc_macro:.4f}")
print(f"Brier score    : {brier:.4f}")
print("=" * 60)"""

md_step8 = """## 8. Confidence-level analysis (for your conference claims)"""

cell_conf = """M = 15
edges = np.linspace(0, 1, M + 1)
bin_acc, bin_conf, bin_cnt = np.zeros(M), np.zeros(M), np.zeros(M)
for b in range(M):
    lo, hi = edges[b], edges[b + 1]
    mask = (confidence >= lo) & (confidence <= hi)
    cnt = int(mask.sum()); bin_cnt[b] = cnt
    if cnt:
        bin_acc[b] = is_correct[mask].mean()
        bin_conf[b] = confidence[mask].mean()
ece = float((bin_cnt / len(labels) * np.abs(bin_acc - bin_conf)).sum())
mce = float(np.max(np.abs(bin_acc - bin_conf)[bin_cnt > 0]))

cc = confidence[is_correct]; cw = confidence[~is_correct]
hi = confidence >= 0.70
print(f"ECE: {ece:.4f}   MCE: {mce:.4f}")
print(f"Mean confidence - correct: {cc.mean()*100:.1f}%   wrong: {cw.mean()*100:.1f}%")
print(f"Predictions >= 70% conf : {hi.mean()*100:.1f}%  -> accuracy {is_correct[hi].mean()*100:.1f}%")
top = confidence >= 0.933
print(f"Top band >93% conf      : {top.mean()*100:.1f}%  -> accuracy {is_correct[top].mean()*100:.1f}%")
lo = confidence < 0.05
print(f"Below app 5% gate       : {lo.mean()*100:.2f}%")

conf_df = pd.DataFrame({"conf_low": edges[:-1], "conf_high": edges[1:],
                        "samples": bin_cnt, "fraction": bin_cnt / len(labels),
                        "accuracy": bin_acc, "mean_confidence": bin_conf})
conf_df.to_csv("confidence_report.csv", index=False)
per_df = pd.DataFrame([{
    "class": class_names[c], "model_output_index": class_to_model[c],
    "support": int(cr[class_names[c]]["support"]),
    "precision": cr[class_names[c]]["precision"], "recall": cr[class_names[c]]["recall"],
    "f1_score": cr[class_names[c]]["f1-score"], "roc_auc": aucs[c],
} for c in range(n_true)])
per_df.to_csv("classification_report.csv", index=False)"""

md_step9 = """## 9. Charts (inline, saveable / downloadable)"""

cell_plots = """import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 110
OUT = "dermai_charts"; os.makedirs(OUT, exist_ok=True)
TS = [c[:24] for c in class_names]

# Confusion matrix (normalized)
from sklearn.metrics import confusion_matrix as cm_fn
preds_pad = np.where(preds == -1, n_true, preds)
cmf = cm_fn(labels, preds_pad, labels=list(range(n_true + 1)))[:n_true, :n_true]
cmn = cmf.astype(float) / cmf.sum(1, keepdims=True)
plt.figure(figsize=(15, 12))
sns.heatmap(cmn, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=TS, yticklabels=TS, linewidths=0.3)
plt.title("Normalized Confusion Matrix (aligned)"); plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrix.png"); plt.show()

# Confidence distribution
plt.figure(figsize=(9, 5))
plt.hist(confidence[is_correct], bins=40, range=(0, 1), alpha=0.75, color="#10B981",
         label=f"Correct ({is_correct.sum()})", edgecolor="white")
plt.hist(confidence[~is_correct], bins=40, range=(0, 1), alpha=0.75, color="#EF4444",
         label=f"Incorrect ({(~is_correct).sum()})", edgecolor="white")
plt.xlabel("Model confidence"); plt.ylabel("Predictions")
plt.title("Confidence - correct vs incorrect"); plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/confidence_distribution.png"); plt.show()

# Reliability diagram
plt.figure(figsize=(7, 7))
v = bin_cnt > 0
centers = (edges[:-1] + edges[1:]) / 2
plt.bar(centers[v], bin_acc[v], width=1 / M, color="#14B8A6", edgecolor="white")
plt.plot([0, 1], [0, 1], "--", color="#EF4444", lw=1.5, label="Perfect calibration")
plt.xlabel("Mean predicted confidence"); plt.ylabel("Observed accuracy")
plt.title(f"Reliability Diagram  (ECE={ece:.3f}, MCE={mce:.3f})")
plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/reliability_diagram.png"); plt.show()

# Per-class F1
plt.figure(figsize=(15, 5))
f1v = [per_df["f1_score"][c] for c in range(n_true)]
cols = ["#10B981" if f >= 0.7 else ("#F59E0B" if f >= 0.4 else "#EF4444") for f in f1v]
plt.bar(range(n_true), f1v, color=cols, edgecolor="white")
plt.xticks(range(n_true), TS, rotation=90, fontsize=7)
plt.title("Per-class F1-score (green>=0.7, amber>=0.4, red<0.4)"); plt.tight_layout()
plt.savefig(f"{OUT}/per_class_f1.png"); plt.show()

# ROC macro
plt.figure(figsize=(7, 7))
for c in range(n_true):
    fpr, tpr, _ = roc_curve(y_bin[:, c], aligned[:, c])
    plt.plot(fpr, tpr, lw=0.8, alpha=0.3, color="#0891B2")
plt.plot([0, 1], [0, 1], "--", color="#F59E0B", label="Chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title(f"ROC one-vs-rest (macro AUC={auc_macro:.3f})")
plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(f"{OUT}/roc_curves.png"); plt.show()

print("charts saved to", OUT)"""

md_step10 = """## 10. Download everything (zip)"""

cell_dl = """import json, zipfile, IPython.display as disp

summary = {
    "top1_accuracy_pct": round(acc * 100, 2),
    "top3_accuracy_pct": round(top3_acc * 100, 2),
    "precision_macro_pct": round(p_mac * 100, 2),
    "recall_macro_pct": round(r_mac * 100, 2),
    "f1_macro_pct": round(f1_mac * 100, 2),
    "f1_weighted_pct": round(cr["weighted avg"]["f1-score"] * 100, 2),
    "roc_auc_macro": round(auc_macro, 4),
    "brier_score": round(brier, 4),
    "ece": round(ece, 4),
    "mce": round(mce, 4),
    "mean_conf_correct_pct": round(cc.mean() * 100, 2),
    "mean_conf_wrong_pct": round(cw.mean() * 100, 2),
    "acc_above_70pct": round(is_correct[hi].mean() * 100, 2),
    "frac_above_70pct": round(hi.mean() * 100, 2),
}
with open("metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

zf = zipfile.ZipFile("dermai_analysis.zip", "w")
for root, _, files in os.walk(OUT):
    for name in files:
        zf.write(os.path.join(root, name))
for name in ["metrics_summary.json", "classification_report.csv", "confidence_report.csv"]:
    zf.write(name)
zf.close()

from google.colab import files
files.download("dermai_analysis.zip")
disp.display(disp.Markdown("**Done!** The zip contains the charts and the CSVs/JSON for your report."))"""

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": md_intro.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step1.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_install.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step2.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_upload.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step3.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_data.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step4.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_model.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step5.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_infer.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step6.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_align.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step7.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_metrics.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step8.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_conf.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step9.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_plots.splitlines(keepends=True)},
    {"cell_type": "markdown", "metadata": {}, "source": md_step10.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell_dl.splitlines(keepends=True)},
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "name": "DermAI_Performance_Analysis"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False)

print("wrote", OUT, os.path.getsize(OUT), "bytes")