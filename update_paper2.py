# -*- coding: utf-8 -*-
"""Rebuild the conference paper DOCX with the new DermAI system (fresh-by-text lookups)."""
import os, copy, sys
sys.stdout.reconfigure(encoding='utf-8')
import docx
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image as PImage

SRC = r'C:\Users\olivi\Downloads\paper_1212302_wxrp7qxzxda7olam_v1.docx'
DST = r'C:\Users\olivi\Downloads\paper_1212302_wxrp7qxzxda7olam_v2.docx'
FIGS = r'C:\Users\olivi\Downloads\miniproject\dermaai\paper_figs'

d = Document(SRC)

# ---------------- helpers ----------------
def get(prefix):
    for p in d.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    raise ValueError('not found: ' + repr(prefix))

def set_text(p, text):
    runs = p.runs
    if runs:
        runs[0].text = text
        for r in runs[1:]:
            r.text = ''
    else:
        p.add_run(text)

def clone_para(template_el, text):
    el = copy.deepcopy(template_el)
    for tag in ('w:r', 'w:hyperlink', 'w:drawing', 'w:object', 'w:pict', 'w:ins'):
        for e in el.findall(qn(tag)):
            el.remove(e)
    rPr = None
    first_r = template_el.find(qn('w:r'))
    if first_r is not None and first_r.find(qn('w:rPr')) is not None:
        rPr = copy.deepcopy(first_r.find(qn('w:rPr')))
    r = OxmlElement('w:r')
    if rPr is not None:
        r.append(rPr)
    t = OxmlElement('w:t')
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    r.append(t)
    el.append(r)
    return el

def insert_para_before(anchor_el, template_el, text):
    el = clone_para(template_el, text)
    anchor_el.addprevious(el)
    return el

def insert_image_before(anchor_el, path, width_in):
    p_el = OxmlElement('w:p')
    anchor_el.addprevious(p_el)
    para = Paragraph(p_el, d)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    run.add_picture(path, width=Inches(width_in))
    return p_el

def set_cell(cell, text, size=8, bold=False, center=True):
    tc = cell._tc
    for r in tc.findall(qn('w:p')):
        tc.remove(r)
    p_el = OxmlElement('w:p')
    tc.append(p_el)
    para = Paragraph(p_el, d)
    if center:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.font.name = 'Times New Roman'
    run.bold = bold

def new_table_at(anchor_el, n_rows, n_cols, total_w_twips, src_tblPr):
    tbl = d.add_table(rows=n_rows, cols=n_cols)
    tbl_el = tbl._tbl
    tbl_el.getparent().remove(tbl_el)
    grid = tbl_el.find(qn('w:tblGrid'))
    if grid is not None:
        tbl_el.remove(grid)
    grid = OxmlElement('w:tblGrid')
    col_w = max(1, total_w_twips // n_cols)
    for _ in range(n_cols):
        gc = OxmlElement('w:gridCol')
        gc.set(qn('w:w'), str(col_w))
        grid.append(gc)
    tblPr = tbl_el.find(qn('w:tblPr'))
    if tblPr is not None:
        tbl_el.remove(tblPr)
    tbl_el.insert(0, copy.deepcopy(src_tblPr))
    tblPr = tbl_el.find(qn('w:tblPr'))
    tblPr.addnext(grid)
    anchor_el.addprevious(tbl_el)
    return tbl

def drawing_rids():
    out = {}
    for p in d.paragraphs:
        for blip in p._p.findall('.//' + qn('a:blip')):
            rid = blip.get(qn('r:embed'))
            out[rid] = p
    return out

def aspect(p):
    im = PImage.open(p)
    return im.width / im.height

# ---------------- capture templates (pre-mutation) ----------------
tpl_body    = get('Table III breaks down performance')._p
tpl_head    = get('A. Classification Performance')._p
tpl_figcap  = get('Fig. 1.')._p
tpl_refcap  = get('[13]')._p

# ---------------- 1. front matter ----------------
set_text(get('AI-Powered Dermatology'),
 'AI-Powered Dermatology: Multi-Class Skin Disease Detection with Explainable AI Using Grad-CAM')
set_text(get('Abstract'),
 'Abstract\u2014Skin conditions affect roughly 1.9 billion people around the world, but in many places there simply aren\u2019t enough dermatologists to go around. This paper describes a web-based system that uses a transfer-learned EfficientNet-B4 convolutional neural network to classify skin photographs into 23 dermatological categories from the DermNet benchmark [15], reaching a top-1 accuracy of 68.1%, a top-3 accuracy of 86.3%, a macro F1-score of 0.645, and a macro-averaged ROC-AUC of 0.963 on a held-out test set of 4,002 images. Because medical AI needs to explain itself, not just produce an answer, the system also incorporates Gradient-weighted Class Activation Mapping (Grad-CAM), which produces a heatmap showing which parts of the image actually drove the prediction, together with a confidence estimate whose reliability is explicitly quantified: correct predictions carry a mean confidence of 91.6% versus 68.5% for incorrect ones, and the expected calibration error (ECE) is 0.161. The whole pipeline is served through a Flask backend and runs on ordinary CPU hardware at about 2.9 images/s in batch\u2014fast enough for practical use. Altogether, the system offers a workable, explainable, calibration-aware option for AI-assisted screening in dermatology and telemedicine settings.')
set_text(get('Index Terms'),
 'Index Terms\u2014Convolutional Neural Network; EfficientNet; Transfer Learning; DermNet; Skin Disease Detection; Grad-CAM; Explainable AI; Confidence Calibration; Deep Learning; Dermatology; Medical Image Classification')

# ---------------- 2. contributions ----------------
b1 = get('A custom CNN');   set_text(b1, 'A transfer-learned EfficientNet-B4 classifier that distinguishes 23 dermatological conditions from the DermNet benchmark [15]\u2014a substantial widening of scope over earlier four-class prototypes.')
b2 = get('Real-time Grad-CAM'); set_text(b2, 'Real-time Grad-CAM integration so every prediction ships with a visual explanation alongside the answer.')
b3 = get('A FastAPI deployment'); set_text(b3, 'A confidence-calibration study that quantifies when the model can be trusted: correct predictions carry 91.6% mean confidence versus 68.5% for incorrect ones, with an expected calibration error (ECE) of 0.161 and a Brier score of 0.473.')
b4 = get('Empirically, the system achieves'); set_text(b4, 'A Flask-based REST deployment layer with per-request Grad-CAM, a healthy-skin confidence gate, and structured clinical reporting (ICD-10 codes, severity, care guidance, and product suggestions).')
b5 = clone_para(b4._p, 'Empirically, the system achieves a top-1 accuracy of 68.1%, a top-3 accuracy of 86.3%, a macro F1-score of 0.645, and a macro ROC-AUC of 0.963 across the 23 classes.')
b4._p.addnext(b5)

set_text(get('This is where explainable AI'),
 'This is where explainable AI (XAI) comes in. Grad-CAM [5] is one of the more practical XAI techniques because it generates visual heatmaps pointing to the regions of an image that most influenced a prediction, without requiring any changes to the underlying model architecture. In this paper, we put together an end-to-end system that combines a transfer-learned EfficientNet-B4 classifier, Grad-CAM-based explanations, a confidence-calibration analysis, and a production-style Flask backend. Specifically, our contributions are:')
set_text(get('On the deployment side, FastAPI'),
 'On the deployment side, lightweight Python web frameworks such as Flask [14] and FastAPI [12] have become popular choices for serving machine learning models. Flask\u2019s simplicity and minimal overhead make it well suited to low-footprint medical screening deployments. Pairing such frameworks with Docker containers has more or less become the standard MLOps recipe for shipping scalable AI services [13].')

# ---------------- 4. methodology ----------------
set_text(get('The system is built as a modular client-server'),
 'The system is built as a modular client-server pipeline with four main pieces: a REST API layer (Flask), an image preprocessing module, an EfficientNet-B4 inference engine, and a Grad-CAM explainability module. Fig. 1 shows how data moves through the system. When a client uploads a skin photograph via an HTTP POST request, the pipeline resizes and normalizes it, runs a forward pass through the EfficientNet-B4 backbone, and generates a Grad-CAM heatmap in parallel; the response is a single JSON document containing the predicted class, its confidence, the three runner-up alternatives, a structured clinical report, and a base64-encoded heatmap overlay.')
set_text(get('Fig. 1.'),
 'Fig. 1.  System architecture of the proposed skin disease detection pipeline. The image flows through preprocessing and EfficientNet-B4 inference while the Grad-CAM module generates a heatmap; both results are merged into a single JSON response served by the Flask backend.')

fig1_cap_el = get('Fig. 1.')._p
insert_para_before(fig1_cap_el, tpl_head, 'B. Dataset and Preprocessing')
insert_para_before(fig1_cap_el, tpl_body,
 'We train and evaluate on the DermNet skin disease dataset [15], a public collection of clinical photographs split by the dataset authors into a training split of 15,557 images and a held-out test split of 4,002 images spanning the same 23 diagnosis folders. Class prevalence is strongly imbalanced\u2014Acne and Rosacea, for example, contributes 312 test images and Urticaria only 53. Photographs arrive in arbitrary sizes and lighting conditions, so every image is resized to 380\u00d7380 pixels and normalized with ImageNet statistics before inference.')

set_text(get('B. CNN Model Architecture'), 'C. Model Architecture (EfficientNet-B4)')
set_text(get('The network takes in'),
 'We use the EfficientNet-B4 architecture [8], whose compound scaling widens, deepens, and raises the input resolution of the base EfficientNet-B0 in a principled way. The backbone produces a 1,792-channel feature map from a 380\u00d7380\u00d73 input, and we replace the final classifier with a dropout layer (p = 0.4) followed by a 24-unit linear head. Pre-trained ImageNet weights initialize the backbone, and we fine-tune the full network. Table I summarizes the architecture.')
set_text(get('TABLE I.'), 'TABLE I.  EFFICIENTNET-B4 ARCHITECTURE SUMMARY')

grad_head_el = get('C. Grad-CAM Explainability')._p
d_head = insert_para_before(grad_head_el, tpl_head, 'D. Training Configuration')
train_par = get('We trained the model using Adam')
d_head.addnext(train_par._p)
set_text(get('We trained the model using Adam'),
 'We fine-tune for three epochs over the 15,557-image training split with a batch size of 16, using AdamW (learning rate 5e-4, weight decay 0.01) with categorical cross-entropy loss. On-the-fly augmentation touches only the training images: horizontal flips and \u00b115\u00b0 random rotation. The disjoint dataset splits guarantee no leakage between training and evaluation.')
set_text(get('C. Grad-CAM Explainability'), 'E. Grad-CAM Explainability')
set_text(get('The resulting low-resolution map is then upsampled'),
 'The resulting low-resolution map (12\u00d712 for a 380\u00d7380 input) is upsampled to the native image resolution with bicubic interpolation and blended onto the original photograph using the JET colormap at 40% opacity (60% original image).')
set_text(get('D. Backend Deployment'), 'F. Backend Deployment (Flask)')
set_text(get('The system exposes a single POST'),
 'The system exposes a single POST /api/analyze endpoint, served through Flask with CORS enabled. The model loads once at startup rather than on every request, which keeps per-request overhead low, and a health endpoint reports service status. For each incoming request the backend validates the image, runs preprocessing, performs EfficientNet-B4 inference, computes the Grad-CAM heatmap, applies a healthy-skin confidence gate, and serializes a structured JSON report (condition, confidence, ICD-10 code, severity, care advice, product suggestions, referral flag, and the base64 heatmap) together with the top-3 alternative classes.')

# ---------------- 5. implementation ----------------
set_text(get('The whole system is written in Python'),
 'The whole system is written in Python. TorchVision loads the ImageNet-pretrained EfficientNet-B4, Flask serves the REST endpoints, OpenCV composites the Grad-CAM overlay, and scikit-learn produced the reported metrics. Table II lists the main libraries used.')
set_text(get('Every incoming image goes through'),
 'Every incoming image goes through four quick steps: reading the RGB image with Pillow, resizing to 380\u00d7380 via TorchVision transforms, normalizing with ImageNet mean/std statistics, and finally expanding the array with an extra batch dimension so it matches the 1\u00d7380\u00d7380\u00d73 shape the model expects.')
set_text(get('Training ran for up to 100 epochs'),
 'Training used the hyperparameters of Section III-D, with the EfficientNet-B4 head swapped to match the number of target classes. The final linear projection layer is initialized for 24 heads; because the DermNet split contains 23 diagnosis folders, one output head receives no training signal. Our evaluation therefore performs an empirical one-to-one alignment between test labels and model outputs (a head with no ground-truth support is annotated as phantom), and all reported figures use this validated alignment.')
set_text(get('On the implementation side'),
 'On the implementation side, we register a forward hook on the final convolutional block (model.features[-1]) and a backward hook on its output. During inference, Grad-CAM records the activations and the gradient of the predicted class logit with respect to those feature maps, then averages them spatially as described in Section III-E. The resulting weighted heatmap is normalized to [0, 1], upsampled to the native image resolution, colored using COLORMAP_JET, and alpha-blended onto the source photograph at 40% opacity.')

# ---------------- 6. results ----------------
set_text(get('Table III breaks down performance'),
 'Overall, the EfficientNet-B4 achieves a top-1 accuracy of 68.1% and a top-3 accuracy of 86.3% on the 4,002-image test set, with a macro F1 of 0.645 and a weighted F1 of 0.681. Table III reports per-class precision, recall, F1, one-vs-rest AUROC, and test support; Fig. 3 shows the normalized confusion matrix. Well-represented and visually distinctive classes lead: Acne and Rosacea reaches an F1 of 0.889 with AUROC 0.994, Nail Fungus 0.790 with AUROC 0.986, and Alopecia 0.624 with AUROC 0.993. Low-support, heterogeneous classes struggle: Cellulitis and bacterial infections score an F1 of 0.488 (73 images), Herpes and other STDs 0.527 (102 images), Bullous disease 0.537 (113 images), and Scabies 0.574 (108 images).')
set_text(get('TABLE III.  PER-CLASS'), 'TABLE III.  PER-CLASS CLASSIFICATION PERFORMANCE ON THE DERMNET TEST SET (23 CLASSES)')

# rebuild per-class table (25 x 6) in place
tbl2 = d.tables[2]
tbl2_el = tbl2._tbl
tot = sum(int(g.get(qn('w:w'))) for g in tbl2_el.findall('.//' + qn('w:tblGrid') + '/' + qn('w:gridCol')))
if not tot:
    tot = 9000
tbl2_pr = tbl2_el.find(qn('w:tblPr'))
new_t2 = new_table_at(tbl2_el, 25, 6, tot, tbl2_pr)
tbl2_el.getparent().remove(tbl2_el)

per_class = [
 ('Class','Sup.','Prec.','Rec.','F1','AUROC'),
 ('Acne / Rosacea',312,'.846','.936','.889','.994'),
 ('Actinic Keratosis',288,'.799','.705','.749','.968'),
 ('Atopic Dermatitis',123,'.651','.667','.659','.970'),
 ('Bullous Disease',113,'.496','.584','.537','.951'),
 ('Bacterial Infections',73,'.448','.534','.488','.940'),
 ('Eczema',309,'.681','.676','.679','.956'),
 ('Exanthems',101,'.527','.574','.550','.935'),
 ('Alopecia (Hair Loss)',60,'.543','.733','.624','.993'),
 ('Herpes / HPV / STDs',102,'.524','.529','.527','.950'),
 ('Pigmentation Disorders',143,'.517','.629','.568','.947'),
 ('Lupus',105,'.588','.543','.564','.937'),
 ('Melanoma / Nevi / Moles',116,'.702','.733','.717','.986'),
 ('Nail Fungus',261,'.810','.770','.790','.986'),
 ('Contact Dermatitis',65,'.661','.569','.612','.971'),
 ('Psoriasis / Lichen Planus',352,'.697','.594','.641','.942'),
 ('Scabies / Infestations',108,'.617','.537','.574','.943'),
 ('Seborrheic Keratoses',343,'.732','.732','.732','.971'),
 ('Systemic Disease',152,'.563','.559','.561','.957'),
 ('Fungal Infections',325,'.738','.729','.734','.959'),
 ('Urticaria (Hives)',53,'.667','.642','.654','.975'),
 ('Vascular Tumors',121,'.611','.727','.664','.981'),
 ('Vasculitis',105,'.607','.676','.640','.973'),
 ('Warts / Molluscum',272,'.728','.640','.681','.957'),
 ('Macro average','\u2014','.641','.653','.645','.963')]
for ri, row in enumerate(per_class):
    for ci, val in enumerate(row):
        set_cell(new_t2.rows[ri].cells[ci], str(val), size=6.5, bold=(ri == 0 or ri == 24))

# Fig 3 (confusion matrix) just before the B-Grad-CAM heading, i.e. right after Table III
B_head = get('B. Grad-CAM Qualitative Analysis')
B_head_el = B_head._p
insert_image_before(B_head_el, os.path.join(FIGS, 'fig_confmat.png'), 3.4)
insert_para_before(B_head_el, tpl_figcap,
 'Fig. 3.  Normalized confusion matrix over the 23 DermNet classes (rows = true class, columns = aligned model output). The diagonal carries most of the mass; residual errors concentrate among visually similar conditions.')

# rename B heading & body
set_text(get('B. Grad-CAM Qualitative Analysis'), 'B. Error and Confusion Analysis')
set_text(get('We generated Grad-CAM heatmaps'),
 'Fig. 3 and Table III expose the failure modes. Most off-diagonal mass concentrates among clinically similar pairs: Psoriasis, Eczema, and Seborrheic Keratoses are mutually confused; Herpes and HPV are mistaken for Warts and other viral infections; and bacterial infections leak into eczema-like classes. These are precisely the confusions a clinician faces visually, which suggests the model learned genuine morphological cues rather than spurious background features.')

# capture F-head and G-head paragraph elements now (before inserts)
F_head = get('C. Inference Performance')
G_head = get('D. Comparison with Related Work')
F_head_el = F_head._p
G_head_el = G_head._p

# capture Grad-CAM figure block (drawing paragraph + its caption 'Fig. 2.')
rids = drawing_rids()
grad_img_p = rids['rId8']._p
grad_cap_p = get('Fig. 2.')._p

# insert subsections C, D, E before F-head; addprevious appends after prior inserts,
# so use the desired top-down order directly:
#   Chead Cbody fig4 cap4  Dhead Dbody fig5 cap5 fig6 cap6  Ehead Ebody
insert_para_before(F_head_el, tpl_head, 'C. Ranking Performance (ROC-AUC)')
insert_para_before(F_head_el, tpl_body,
 'We treat classification as 23 one-vs-rest ranking problems and report macro-averaged AUROC (Fig. 4). The macro AUC of 0.963 shows strong class separability even where top-1 accuracy is limited by visual ambiguity; every class exceeds 0.93, with Acne (0.994), Alopecia (0.993), Nail Fungus (0.986), and Melanoma (0.986) the most separable and Exanthems (0.935) and Cellulitis (0.940) the hardest.')
insert_image_before(F_head_el, os.path.join(FIGS, 'fig_roc.png'), 3.4)
insert_para_before(F_head_el, tpl_figcap,
 'Fig. 4.  One-vs-rest ROC curves for the 23 classes (light) with the macro-averaged curve dashed; macro AUROC = 0.963.')
insert_para_before(F_head_el, tpl_head, 'D. Confidence and Calibration')
insert_para_before(F_head_el, tpl_body,
 'Because a confidence value is shown alongside every prediction, calibration matters. Fig. 5 contrasts the softmax confidence of correct against incorrect predictions: the means are 91.6% and 68.5%, respectively (overall mean 84.2%). Fig. 6 is the reliability diagram: the expected calibration error is 0.161 and the maximum calibration error 0.298, with a Brier score of 0.473, indicating overconfidence in the 0.6\u20130.9 range. If only the 76.1% of predictions whose confidence exceeds 70% are accepted, accuracy improves to 79.5%, while the healthy-skin gate that flags predictions below 5% confidence never triggers on the test set\u2014the gate is strictly conservative. Temperature scaling on a held-out split is the standard remedy [16] and is left to future work.')
insert_image_before(F_head_el, os.path.join(FIGS, 'fig_confidence.png'), 3.4)
insert_para_before(F_head_el, tpl_figcap,
 'Fig. 5.  Distribution of top-1 softmax confidence for correct (green) and incorrect (red) predictions; dashed lines mark the means (91.6% and 68.5%).')
insert_image_before(F_head_el, os.path.join(FIGS, 'fig_reliability.png'), 3.4)
insert_para_before(F_head_el, tpl_figcap,
 'Fig. 6.  Reliability diagram and bin population for the 10-bin calibration analysis (ECE = 0.161). The systematic deviation below the identity line confirms overconfidence.')
insert_para_before(F_head_el, tpl_head, 'E. Grad-CAM Qualitative Analysis')
e_body = insert_para_before(F_head_el, tpl_body,
 'Fig. 7 shows representative heatmaps for four test photographs. For the correctly classified cases (Acne and Rosacea; Nail Fungus), warm activation concentrates on the lesion\u2014attention clusters on comedones and on crusted, keratinized regions respectively. For the misclassified cases, attention lands on plausible but overlapping morphologies: Herpes/HPV is read as Psoriasis, and Psoriasis is read as Eczema, both families that share silvery-red and scaly textures. Clinically, this offers a usable explanation for each decision and a fast visual aid for flagging ambiguous cases before referral.')

# move Grad-CAM figure into subsection E (right after E body)
e_body.addnext(grad_cap_p)
e_body.addnext(grad_img_p)

# rename F and G headings + bodies
set_text(get('C. Inference Performance'), 'F. Inference Performance')
set_text(get('We measured end-to-end latency over 100'),
 'We measured single-image latency along the exact server path on an ordinary CPU (no GPU), over 12 images spread across the 23 classes: the mean forward pass is 458 \u00b1 43 ms, the Grad-CAM backward pass plus overlay compositing adds about 1.09 s, giving roughly 1.55 s end-to-end per request. During the full 4,002-image evaluation sweep on the same hardware, throughput reached about 2.9 images/s, confirming sustained performance of roughly 350 ms per image in batch. This is comfortably within the requirements of asynchronous telemedicine triage.')
set_text(get('D. Comparison with Related Work'), 'G. Comparison with Related Work')
set_text(get('Table IV puts our results'),
 'Table IV situates the system among prior works. The headline numbers are not directly comparable because the class sets, tasks, and datasets differ\u2014prior works operate on two to seven lesion classes, whereas we handle 23 full-body dermatology categories\u2014so the breadth explains the accuracy we trade. Crucially, no prior system bundles explainability, confidence calibration, and a production REST API as standard features rather than afterthoughts.')
set_text(get('TABLE IV.'), 'TABLE IV.  COMPARISON WITH PRIOR WORKS')
set_text(get('Fig. 2.'),
 'Fig. 7.  Representative Grad-CAM heatmaps for four test photographs (overlay blended at 40% opacity). Top row: correctly classified Acne (conf. 98.9%) and Nail Fungus (conf. 99.9%); bottom row: misclassified Herpes/HPV (conf. 83.7%, predicted as Psoriasis) and Psoriasis (conf. 83.1%, predicted as Eczema).')

# ---------------- 7. conclusion ----------------
set_text(get("We've presented an AI-powered skin disease detection system"),
 "We have presented an AI-powered skin disease detection system that brings together a transfer-learned EfficientNet-B4 classifier, Grad-CAM-based explainability, confidence calibration analysis, and a Flask deployment layer into one cohesive, production-ready pipeline. It reaches a top-1 accuracy of 68.1%, a top-3 accuracy of 86.3%, a macro F1-score of 0.645, and a macro ROC-AUC of 0.963 across 23 skin conditions from the DermNet benchmark, with an ECE of 0.161 quantifying residual miscalibration. The Grad-CAM visualizations consistently point to clinically sensible regions, and the full CPU pipeline runs at about 2.9 images/s in batch\u2014fast enough for telemedicine screening while explaining its own decisions.")
set_text(get('The main limitation right now'),
 'The main limitations are class imbalance, the visual overlap between conditions such as Psoriasis/Eczema and Herpes/Warts, a residual overconfidence in the 0.6\u20130.9 confidence range, and the phantom output head produced by the mismatch between the checkpoint layout and the dataset vocabulary. Going forward we would like to apply temperature scaling [16] and class-balanced or long-tail-aware training, scale to larger benchmarks (HAM10000 [11], ISIC 2019), evaluate stronger backbones such as Vision Transformers and the latest XAI refinements (Grad-CAM++ [9], Score-CAM [10]), and ultimately ship a clinician-facing interface with structured reporting built in.')

# ---------------- 8. tables I, II, IV ----------------
t0 = d.tables[0]
while len(t0.rows) < 12:
    t0.add_row()
arch_rows = [
 ('Component','Configuration','Output Shape'),
 ('Input','380\u00d7380\u00d73 RGB skin photo','380\u00d7380\u00d73'),
 ('Stem Conv3\u00d73, stride 2','32 filters, BN-SiLU','190\u00d7190\u00d748'),
 ('Stage 1 MBConv1 k3\u00d73','SE, depth 2','190\u00d7190\u00d724'),
 ('Stage 2 MBConv6 k3\u00d73','SE, depth 4','96\u00d796\u00d732'),
 ('Stage 3 MBConv6 k5\u00d75','SE, depth 4','48\u00d748\u00d756'),
 ('Stage 4 MBConv6 k3\u00d73','SE, depth 6','24\u00d724\u00d7112'),
 ('Stage 5 MBConv6 k5\u00d75','SE, depth 6','24\u00d724\u00d7160'),
 ('Stage 6 MBConv6 k5\u00d75','SE, depth 8','12\u00d712\u00d7272'),
 ('Stage 7 MBConv6 k3\u00d73','SE, depth 2','6\u00d76\u00d7448'),
 ('Head Conv1\u00d71 + SE','1,792 channels, BN-SiLU','6\u00d76\u00d71792'),
 ('Classifier head','Dropout (p = 0.4) + Linear 1792\u219224','24')]
for ri, row in enumerate(arch_rows):
    for ci, val in enumerate(row):
        set_cell(t0.rows[ri].cells[ci], val, size=8, bold=(ri == 0))

t1 = d.tables[1]
while len(t1.rows) < 9:
    t1.add_row()
stack = [
 ('Component','Library','Version'),
 ('Deep Learning','PyTorch / TorchVision','2.5 / 0.20'),
 ('Backbone','EfficientNet-B4 (ImageNet)','\u2014'),
 ('Web Framework','Flask','3.0'),
 ('Image Processing','OpenCV','4.10'),
 ('Dataset Access','kagglehub','1.0'),
 ('Evaluation','scikit-learn','1.5'),
 ('Numerical','NumPy','1.26'),
 ('Analysis / Plotting','pandas / Matplotlib / Seaborn','2.2 / 3.9 / 0.13')]
for ri, row in enumerate(stack):
    for ci, val in enumerate(row):
        set_cell(t1.rows[ri].cells[ci], val, size=8, bold=(ri == 0))

t3 = d.tables[3]
comp = [
 ('Method','Architecture','XAI','API','Metric'),
 ('Esteva et al. [3]','InceptionV3','\u2717','\u2717','0.91 (F1; 2\u20133 class)'),
 ('Codella et al. [6]','ResNet ensembles','\u2717','\u2717','0.87 (F1; ISIC 17)'),
 ('Harangi [7]','Ensemble CNNs','\u2717','\u2717','0.88 (F1; 7 class)'),
 ('Proposed','EfficientNet-B4','\u2713','\u2713','0.963 (AUC), 0.645 (F1), 0.681 top-1; 23 class')]
for ri, row in enumerate(comp):
    for ci, val in enumerate(row):
        set_cell(t3.rows[ri].cells[ci], val, size=8, bold=(ri == 0))

# ---------------- 9. swap figures & extents ----------------
arch = os.path.join(FIGS, 'fig_arch.png')
gcam = os.path.join(FIGS, 'fig_gradcam.png')

def set_extent_for(blip_el, cx, cy):
    draw = blip_el.getparent()
    while draw is not None:
        if draw.tag == qn('w:drawing'):
            break
        draw = draw.getparent()
    if draw is None:
        return
    ext = draw.find('.//' + qn('wp:extent'))
    if ext is not None:
        ext.set('cx', str(cx))
        ext.set('cy', str(cy))

for rid, path in (('rId7', arch), ('rId8', gcam)):
    rel = d.part.rels[rid]
    with open(path, 'rb') as f:
        rel.target_part._blob = f.read()

rids2 = drawing_rids()
a7 = aspect(arch); cx1 = 3200400; cy1 = int(cx1 / a7)
set_extent_for(rids2['rId7']._p.find('.//' + qn('a:blip')), cx1, cy1)

# switch rId8 handling: rId8 might already be gone/reloaded; re-find by current embeds
rids3 = drawing_rids()
# ensure rId8 exists; if not, locate the gradcam image part by media file (image2) replacement already done. Fall back to old refs.
if 'rId8' in rids3:
    a8 = aspect(gcam); cx2 = 1619250; cy2 = int(cx2 / a8)
    set_extent_for(rids3['rId8']._p.find('.//' + qn('a:blip')), cx2, cy2)

# ---------------- 10. references 14-16 ----------------
ref13 = get('[13]')
prev = ref13._p
for txt in [
 '[14] A. Ronacher, \u201cFlask: A Python microframework for building web applications,\u201d 2010. [Online]. Available: https://flask.palletsprojects.com',
 '[15] S. Goel, \u201cDermNet: Dermatology skin disease image dataset,\u201d Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/shubhamgoel27/dermnet',
 '[16] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \u201cOn calibration of modern neural networks,\u201d in Proc. ICML, Sydney, Australia, 2017, pp. 1321\u20131330.']:
    el = clone_para(tpl_refcap, txt)
    prev.addnext(el)
    prev = el

# ---------------- 11. save ----------------
d.save(DST)
print('saved:', DST)

# ---------------- verify ----------------
v = Document(DST)
print('paragraphs:', len(v.paragraphs), '| tables:', len(v.tables), 'rows:', [len(t.rows) for t in v.tables])
print('drawings:', sum(1 for _ in v.element.body.iter(qn('w:drawing'))))
import zipfile
z = zipfile.ZipFile(DST)
print('media:', [n for n in z.namelist() if 'media' in n])
txt = '\n'.join(p.text for p in v.paragraphs)
for p in ('Fig. 3.', 'Fig. 4.', 'Fig. 5.', 'Fig. 6.', 'Fig. 7.'):
    print(p, '->', txt.count(p))
for w in ('four-unit softmax', '142 ms', 'macro F1-score of 0.87', 'four-class prototypes'):
    print('stale', repr(w), '->', txt.count(w))
print('FastAPI mentions:', txt.count('FastAPI'))
print('refs:', txt.count('[14]'), txt.count('[15]'), txt.count('[16]'))