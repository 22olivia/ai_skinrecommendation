# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
import torch

BASE = r'C:\Users\olivi\Downloads\miniproject\dermaai'
OUT = os.path.join(BASE, 'paper_figs')
os.makedirs(OUT, exist_ok=True)
TEST = r'C:\Users\olivi\.cache\kagglehub\datasets\shubhamgoel27\dermnet\versions\1\test'

SHORT = {
 0:'Acne/Rosacea',1:'Actinic Keratosis',2:'Atopic Derm.',3:'Bullous',4:'Bacterial Inf.',
 5:'Eczema',6:'Exanthems',7:'Alopecia',8:'Herpes/HPV',9:'Pigmentation',10:'Lupus',
 11:'Melanoma',12:'Nail Fungus',13:'Contact Derm.',14:'Psoriasis',15:'Scabies',
 16:'Seb. Keratoses',17:'Systemic',18:'Fungal Inf.',19:'Urticaria',20:'Vascular Tumors',
 21:'Vasculitis',22:'Warts'}

d = np.load(os.path.join(BASE, 'eval_output', 'predictions_cache.npz'))
probs, labels = d['probs'], d['labels']           # (4002,24), (4002,)
labels_t = labels                                  # test-class index 0..22
idx_map = np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
model_idx = idx_map[labels_t]                      # aligned head
N = len(labels_t)
top1 = probs[np.arange(N), model_idx]
pred_model = probs.argmax(1)
correct = top1 > 0.80  # placeholder, recompute below
correct_t = (probs[np.arange(N), model_idx].copy());
# class-aware correctness (argmax over 24 heads == aligned model head)
pred_correct = pred_model == model_idx

def short(i):
    return SHORT[i]

def cmap_coolwarm_norm(mat):
    from matplotlib.colors import Normalize
    return mat

# ---------------- Fig: confusion matrix 23x23 ----------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_t, pred_model)
cm_n = cm.astype(float)
for i in range(23):
    s = cm_n[i].sum()
    if s > 0:
        cm_n[i] = cm_n[i] / s
fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=220)
im = ax.imshow(cm_n, cmap='cividis', vmin=0, vmax=1)
ax.set_xticks(range(23)); ax.set_xticklabels([short(i) for i in range(23)], rotation=90, fontsize=4.2)
ax.set_yticks(range(23)); ax.set_yticklabels([short(i) for i in range(23)], fontsize=4.6)
ax.set_xlabel('Predicted head (aligned)', fontsize=7)
ax.set_ylabel('True class', fontsize=7)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.ax.tick_params(labelsize=6)
for i in range(23):
    for j in range(23):
        v = cm_n[i, j]
        if v > 0.08:
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=3.6,
                    color='white' if v < 0.6 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_confmat.png'), bbox_inches='tight')
plt.close()

# ---------------- Fig: ROC curves ----------------
from sklearn.metrics import roc_auc_score, roc_curve
fpr_all, tpr_all = [], []
for c in range(23):
    y = (labels_t == c).astype(int)
    score = probs[:, idx_map[c]]
    fpr, tpr, _ = roc_curve(y, score)
    fpr_all.append(fpr); tpr_all.append(tpr)
    a = score[y == 1]
    b = score[y == 0]
# macro via averaging per-class auc (same as reported)
aucs = [roc_auc_score((labels_t == c).astype(int), probs[:, idx_map[c]]) for c in range(23)]
fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=220)
for c in range(23):
    ax.plot(fpr_all[c], tpr_all[c], color='#7aa6c2', lw=0.8, alpha=0.55)
mean_fpr = np.linspace(0, 1, 200)
mean_tpr = np.mean([np.interp(mean_fpr, fpr_all[c], tpr_all[c]) for c in range(23)], axis=0)
mean_tpr[0] = 0.0
ax.plot(mean_fpr, mean_tpr, color='#1f3b73', lw=2.2, ls='--',
        label=f'Macro ROC (AUC = {np.mean(aucs):.3f})')
ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
ax.set_xlabel('False positive rate', fontsize=8)
ax.set_ylabel('True positive rate', fontsize=8)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=7.5, loc='lower right')
ax.grid(alpha=0.25, lw=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_roc.png'), bbox_inches='tight')
plt.close()

# ---------------- Fig: confidence distribution ----------------
fig, ax = plt.subplots(figsize=(4.2, 2.6), dpi=220)
conf_c = probs[np.arange(N), model_idx][pred_correct]
conf_w = probs[np.arange(N), model_idx][~pred_correct]
bins = np.linspace(0, 1, 41)
ax.hist(conf_c, bins=bins, color='#2e6f40', alpha=0.75, label=f'Correct (n={conf_c.size})')
ax.hist(conf_w, bins=bins, color='#c0392b', alpha=0.65, label=f'Incorrect (n={conf_w.size})')
ax.axvline(conf_c.mean(), color='#2e6f40', ls='--', lw=1.2)
ax.axvline(conf_w.mean(), color='#c0392b', ls='--', lw=1.2)
ax.set_xlabel('Top-1 softmax confidence', fontsize=8)
ax.set_ylabel('Samples', fontsize=8)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25, lw=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_confidence.png'), bbox_inches='tight')
plt.close()

# ---------------- Fig: reliability diagram ----------------
nb = 10
conf_all = probs[np.arange(N), model_idx]
acc_all = pred_correct.astype(float)
edges = np.linspace(0, 1, nb + 1)
bin_conf, bin_acc, bin_cnt = [], [], []
for i in range(nb):
    m = (conf_all >= edges[i]) & (conf_all < edges[i + 1])
    if m.sum() > 0:
        bin_conf.append(conf_all[m].mean()); bin_acc.append(acc_all[m].mean()); bin_cnt.append(m.sum())
bin_conf = np.array(bin_conf); bin_acc = np.array(bin_acc)
# ECE (same definition as evaluate.py)
ece = (np.abs(bin_conf - bin_acc) * (np.array(bin_cnt) / N)).sum()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7), dpi=220)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.plot(bin_conf, bin_acc, 'o-', color='#1f3b73', lw=1.6, ms=4)
ax.fill_between(bin_conf, bin_conf, bin_acc, alpha=0.15, color='#1f3b73')
ax.set_xlabel('Mean confidence', fontsize=8); ax.set_ylabel('Accuracy', fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title(f'Reliability (ECE = {ece:.3f})', fontsize=8)
ax.grid(alpha=0.25, lw=0.4)
ax2.bar(np.arange(nb), np.array(bin_cnt), color='#7aa6c2')
ax2.set_xlabel('Confidence bin (#10)', fontsize=8); ax2.set_ylabel('Samples', fontsize=8)
ax2.set_title('Binned sample count', fontsize=8)
ax2.grid(alpha=0.25, lw=0.4, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_reliability.png'), bbox_inches='tight')
plt.close()

# ---------------- Fig: architecture diagram ----------------
fig, ax = plt.subplots(figsize=(11.5, 8.65), dpi=150)
ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
def box(x, y, w, h, text, fc='#eaf2fa', ec='#1f3b73', fs=9.5, lw=1.2):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.6',
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color='#111')
def arrow(x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.4))
box(3, 82, 14, 11, 'Client\n(Upload photo)', fc='#fff3e0', ec='#b3600a')
arrow(10.5, 18, 82)
box(22, 82, 13, 13, 'Flask Server\nPOST /api/analyze', fc='#e8f5e9', ec='#2e6f40')
arrow(28.5, 38, 82)
box(42, 88, 13, 9, 'Preprocess\n380x380, norm.', fs=8.5)
arrow(35, 42, 84)
box(42, 74, 20, 9, 'EfficientNet-B4\n(ImageNet weights)', fc='#e8eaf6', ec='#283593', fs=9)
arrow(52, 52, 78)
box(34, 64, 12, 8, 'Classifier\nDropout+Linear->24', fs=8)
# right stream: softmax -> predictions -> report
arrow(40, 56, 64)
box(60, 62, 14, 10, 'Softmax\nTop-1 / Top-3', fc='#eaf2fa')
arrow(67, 74, 66)
box(78, 62, 15, 12, 'Confidence gate\n(<5% -> healthy)', fs=8.5)
# gradcam stream
arrow(52, 62, 74)
box(62, 78, 14, 8, 'Grad-CAM\n(features[-1])', fc='#f3e5f5', ec='#6a1b9a', fs=8.5)
box(78, 78, 13, 8, 'Heatmap + blend\n(JET, 40%)', fs=8)
arrow(69, 78, 74)
arrow(76, 78, 62)
arrow(85, 91, 62)
box(94, 52, 12, 16, 'JSON\nreport,\nheatmap\nbase64', fc='#fff8e1', ec='#795548', fs=7.5)
box(50, 44, 34, 12, 'Report builder\nICD-10 code, severity, remedies,\ndiet, products, referral flag, Top-3', fs=8)
arrow(67, 56, 52)
arrow(56, 67, 52)
box(50, 26, 34, 12, 'Evaluation harness (scikit-learn)\nTop-1/Top-3 acc, macro F1, ROC-AUC,\nECE, MCE, Brier, reliability', fs=8)
arrow(67, 44, 32)
ax.text(50, 5, 'Fig. 1 placeholder caption removed at insert time', ha='center', fontsize=8, alpha=0)
plt.savefig(os.path.join(OUT, 'fig_arch.png'), bbox_inches='tight', facecolor='white')
plt.close()

print('data-driven figures done')

# ---------------- Grad-CAM example figure + latency ----------------
sys.path.insert(0, BASE)
import importlib.util
spec = importlib.util.spec_from_file_location('derma', os.path.join(BASE, 'derma.py'))
derma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(derma)
model, transform = derma.model, derma.transform
model.eval()
import cv2
from PIL import Image
import time

def run_case(img_path, top_gt):
    import torch
    pil_img = Image.open(img_path).convert('RGB')
    t0 = time.perf_counter()
    inp = transform(pil_img).unsqueeze(0)
    inp.requires_grad = True
    t_pre = time.perf_counter()
    model.zero_grad()
    derma.gradients = None; derma.activations = None
    outs = model(inp)
    t_fwd = time.perf_counter()
    probs_t = torch.softmax(outs, dim=1).squeeze()
    top_idx = int(outs.argmax(1).item())
    conf = float(probs_t[top_idx].item())
    outs[0, top_idx].backward(retain_graph=True)
    pooled = torch.mean(derma.gradients, dim=[0, 2, 3])
    act = derma.activations.clone()
    for i in range(act.shape[1]):
        act[:, i, :, :] *= pooled[i]
    heat = torch.mean(act, dim=1).squeeze().detach().numpy()
    heat = np.maximum(heat, 0); heat /= (heat.max() + 1e-8)
    t_gc = time.perf_counter()
    img_np = cv2.imdecode(np.frombuffer(open(img_path, 'rb').read(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img_np.shape[:2]
    heat_r = cv2.resize(heat, (w, h))
    heat_c = cv2.applyColorMap(np.uint8(255 * heat_r), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(img_np, 0.6, heat_c, 0.4, 0)
    t_bl = time.perf_counter()
    dur = {'fwd': t_fwd - t_pre, 'gradcam': t_gc - t_fwd, 'blend': t_bl - t_gc}
    return blend, conf, top_idx, dur

# pick exemplars
def first(cond, n=1):
    idxs = np.where(cond)[0]
    return [int(i) for i in idxs[:n]]

test_names = sorted(os.listdir(TEST))
fnames_by_idx = []
for i, nm in enumerate(test_names):
    fnames_by_idx.append(sorted(os.listdir(os.path.join(TEST, nm))))

def path_for(i, k):
    return os.path.join(TEST, test_names[i], fnames_by_idx[i][k])

ex = []
# correct Acne (label 0)
for gi in first((labels_t == 0) & pred_correct & (top1 > 0.98), 1):
    ex.append((gi, 'Acne (correct)'))
# correct Nail Fungus (label 12)
for gi in first((labels_t == 12) & pred_correct & (top1 > 0.95), 1):
    ex.append((gi, 'Nail Fungus (correct)'))
# wrong Herpes (label 8) -> predicted something else
for gi in first((labels_t == 8) & ~pred_correct, 1):
    ex.append((gi, 'Herpes/HPV (incorrect)'))
# wrong Psoriasis (label 14) pred Eczema (head align idx for Eczema =5)
ecz_aligned = idx_map[5]
for gi in first((labels_t == 14) & ~pred_correct & (pred_model == ecz_aligned), 1):
    ex.append((gi, 'Psoriasis (incorrect)'))

fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0), dpi=200)
lat_list = []
for ax_, (gi, title) in zip(axes.ravel(), ex):
    folder = test_names[int(labels_t[gi])]
    flist = sorted(os.listdir(os.path.join(TEST, folder)))
    img_path = os.path.join(TEST, folder, flist[gi % len(flist)])
    blend, conf, top_idx, dur = run_case(img_path, int(labels_t[gi]))
    lat_list.append(dur)
    blend_bgr = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
    ax_.imshow(blend_bgr)
    ax_.set_title(f'{title}\nconf = {conf*100:.1f}%, pred head {top_idx}', fontsize=8.5)
    ax_.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_gradcam.png'), bbox_inches='tight', facecolor='white')
plt.close()

import statistics
fwd = [l['fwd'] for l in lat_list]; gc = [l['gradcam'] for l in lat_list]; bl = [l['blend'] for l in lat_list]
print('latency ms (n=%d): forward mean=%.0f  gradcam mean=%.0f  blend mean=%.0f  total mean=%.0f'
      % (len(lat_list), statistics.mean(fwd)*1e3, statistics.mean(gc)*1e3, statistics.mean(bl)*1e3,
         (statistics.mean(fwd)+statistics.mean(gc)+statistics.mean(bl))*1e3))
np.save(os.path.join(BASE, 'eval_output', 'latency_measured.npy'),
        np.array([statistics.mean(fwd), statistics.mean(gc), statistics.mean(bl)]))
print('figures written to', OUT)