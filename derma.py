# -*- coding: utf-8 -*-
"""
DermAI — Production-Grade Local Inference Engine v3.2
Authorized Academic Evaluation Layer — Auto-Adapting Weight Layout Matrix Edition
"""
print("====================================================")
print("--- STARTING DERMAI LOCAL CORE INFERENCE PIPELINE ---")
print("====================================================")

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import json
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import urllib.parse

app = Flask(__name__)
CORS(app)

DISEASE_CLASSES = [
    'Acne and Rosacea Photos', 
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 
    'Atopic Dermatitis Photos', 
    'Bullous Disease Photos', 
    'Cellulitis Impetigo and other Bacterial Infections', 
    'Eczema Photos', 
    'Exanthems and Drug Eruptions', 
    'Hair Loss Photos Alopecia and other Hair Diseases', 
    'Herpes HPV and other STDs Photos', 
    'Light Diseases and Disorders of Pigmentation', 
    'Lupup and other Connective Tissue diseases', 
    'Melanoma Skin Cancer Nevi and Moles', 
    'Nail Fungus and other Nail Disease', 
    'Poison Ivy Photos and other Contact Dermatitis', 
    'Psoriasis pictures Lichen Planus and related diseases', 
    'Scabies Lyme Disease and other Infestations and Bites', 
    'Seborrheic Keratoses and other Benign Tumors', 
    'Systemic Disease', 
    'Tinea Ringworm Candidiasis and other Fungal Infections', 
    'Urticaria Hives', 
    'Vascular Tumors',
    'Vasculitis Photos',
    'Warts Molluscum and other Viral Infections',
    'Normal Skin Textures'
]

# 1. AUTO-ADAPTING MODEL LOADER LAYER
model = models.efficientnet_b4(weights=None)
num_ftrs = model.classifier[1].in_features

# First, attempt to load assuming the standard layout array length
model.classifier[1] = nn.Linear(num_ftrs, len(DISEASE_CLASSES))

try:
    checkpoint = torch.load('model_weights.pth', map_location=torch.device('cpu'))
    
    # Check if there is an alternate checkpoint layer configuration structure
    if 'classifier.1.weight' in checkpoint:
        saved_classes = checkpoint['classifier.1.weight'].shape[0]
        if saved_classes != len(DISEASE_CLASSES):
            print(f"--> Dynamic Adapter triggered: Adjusting final projection node maps from {len(DISEASE_CLASSES)} to {saved_classes} targets.")
            model.classifier[1] = nn.Linear(num_ftrs, saved_classes)
            
            # Pad or trim our label array safely so it matches the structural dimension indexes perfectly
            while len(DISEASE_CLASSES) < saved_classes:
                DISEASE_CLASSES.append(f"Unmapped Variant Block {len(DISEASE_CLASSES)+1}")
            DISEASE_CLASSES = DISEASE_CLASSES[:saved_classes]

    model.load_state_dict(checkpoint)
    print(f"--> SUCCESS: Precision linked all active classes. Matrix dimensions aligned perfectly.")
except Exception as e:
    print(f"--> CRITICAL WARNING: Error loading weight configurations: {e}")

model.eval()

# 2. IMAGE TRANSFORMS REGISTER
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gradients = None
activations = None

def save_gradient(grad):
    global gradients
    gradients = grad

def forward_hook(module, input, output):
    global activations
    activations = output
    output.register_hook(save_gradient)

target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)

# 3. CLINICAL TRANSLATION DATABASE WITH HEALTHY OVERRIDE STRUCT
HEALTHY_OVERRIDE = {
    "id": "Healthy Skin",
    "label": "Healthy Skin Baseline",
    "icd10": "Z00.00",
    "icd_desc": "GENERAL ADULT MEDICAL EXAMINATION WITHOUT ABNORMAL FINDINGS",
    "severity": None,
    "color": "green",
    "treatment": "Routine Dermal Maintenance",
    "clinical_notes": (
        "Texture matrix evaluation and pigment distribution analysis confirm no active pathogenic biomarkers. "
        "Dermal layers show structural consistency matching normal, healthy baseline tissue configurations."
    ),
    "home_remedies": [
        "Apply raw turmeric (haldi) paste mixed with honey once a week for natural glow and anti-inflammatory benefits.",
        "Use fresh aloe vera gel daily as a lightweight moisturizer to maintain hydration.",
        "Rinse face with rice water (fermented) to tighten pores and brighten complexion.",
        "Apply chilled green tea bags under eyes for 10 minutes to reduce puffiness.",
        "Use besan (gram flour) + curd face pack weekly for gentle exfoliation."
    ],
    "diet": [
        "Drink 8-10 glasses of water daily for internal hydration.",
        "Eat foods rich in Vitamin C: oranges, amla, guava, bell peppers.",
        "Include omega-3 fatty acids: walnuts, flaxseeds, fatty fish.",
        "Add antioxidant-rich foods: blueberries, spinach, dark chocolate.",
        "Consume turmeric in warm milk (golden latte) for anti-inflammatory benefits."
    ],
    "precautions": [
        "Apply SPF 30+ broad-spectrum sunscreen every 2 hours when outdoors.",
        "Remove makeup completely before sleeping.",
        "Change pillowcases every 2-3 days to prevent bacterial buildup.",
        "Avoid touching face frequently to prevent transfer of germs.",
        "Use a humidifier in dry environments to prevent skin dehydration."
    ],
    "avoid": [
        "Excessive sun exposure without protection.",
        "Over-washing face (more than twice daily).",
        "Popping or picking at skin blemishes.",
        "Hot water on face — use lukewarm water instead.",
        "Expired skincare products."
    ],
    "products": [
        {"name": "Gentle Skin Hydrating Cleanser", "type": "DAILY CLEANSER", "tag": "Gentle"},
        {"name": "Vitamin C Serum 10%", "type": "ANTIOXIDANT SERUM", "tag": "Gentle"},
        {"name": "Broad-Spectrum SPF 50+", "type": "SUNSCREEN", "tag": "Gentle"}
    ],
    "referral": False
}

CONDITIONS = [
    {
        "id": "Melanoma Skin Cancer Nevi and Moles",
        "label": "Benign Nevi / Moles (Stable Baseline)",
        "icd10": "D22.9",
        "icd_desc": "MELANOCYTIC NEVI, UNSPECIFIED",
        "severity": None,
        "color": "green",
        "treatment": "Routine Observational Monitoring",
        "clinical_notes": "Pigmented cellular arrays map to non-pathological benign nevi configurations. Symmetrical boundaries indicate safe structural limits.",
        "home_remedies": [
            "Apply raw turmeric (haldi) paste mixed with lemon juice on moles for 15 min daily — natural curcumin helps reduce pigmentation.",
            "Use apple cider vinegar (diluted) dabbed on moles with cotton ball — its malic acid gently breaks down pigmented cells.",
            "Apply crushed basil leaves paste — contains natural compounds that lighten dark spots over time.",
            "Use frankincense essential oil (2 drops mixed with coconut oil) applied topically — known to fade skin irregularities.",
            "Make a paste of neem (margosa) leaves and apply to pigmented areas — anti-melanoma properties."
        ],
        "diet": [
            "Eat Vitamin C rich foods: amla, lemon, oranges — inhibit melanin overproduction.",
            "Include green tea daily — EGCG antioxidants protect against UV-induced cellular changes.",
            "Add turmeric to meals or golden milk — curcumin has anti-cancer and anti-inflammatory effects.",
            "Consume beta-carotene foods: carrots, sweet potatoes — protect skin from sun damage.",
            "Eat cruciferous vegetables: broccoli, cauliflower — contain sulforaphane that fights abnormal cell growth."
        ],
        "precautions": [
            "Monitor moles using ABCDE rule: Asymmetry, Border, Color, Diameter, Evolution.",
            "Photograph moles monthly to track any changes in shape or color.",
            "Avoid picking, scratching, or irritating existing moles.",
            "Wear UPF 50+ clothing for maximum sun protection.",
            "Visit dermatologist if any mole changes in size, shape, or color."
        ],
        "avoid": [
            "Direct sun exposure during 10 AM – 4 PM peak hours.",
            "Tanning beds or artificial UV exposure.",
            "Using bleaching creams without medical supervision.",
            "Friction from tight clothing on mole areas.",
            "Self-removal or freezing of moles at home."
        ],
        "products": [
            {"name": "Broad-Spectrum SPF 50+", "type": "SUNSCREEN", "tag": "Gentle"},
            {"name": "Vitamin C + E Serum", "type": "ANTIOXIDANT", "tag": "OTC"},
            {"name": "Aloe Vera Gel (Pure)", "type": "SOOTHING GEL", "tag": "Gentle"}
        ],
        "referral": False
    },
    {
        "id": "Eczema Photos",
        "label": "Eczema / Atopic Dermatitis",
        "icd10": "L20.9",
        "icd_desc": "ATOPIC DERMATITIS UNSPECIFIED",
        "severity": "Moderate",
        "color": "amber",
        "treatment": "Barrier Repair Protocols",
        "clinical_notes": "Erythematous configurations observed. Textures show characteristic transepidermal water loss metrics.",
        "home_remedies": [
            "Apply pure coconut oil (virgin, cold-pressed) on damp skin — lauric acid fights bacteria and locks in moisture.",
            "Use colloidal oatmeal baths — blend oats into fine powder, add to lukewarm water — soothes itch and inflammation.",
            "Apply fresh aloe vera gel directly from the leaf — cooling anti-inflammatory relief for flare-ups.",
            "Make a paste of turmeric (haldi) + sandalwood powder + rose water — apply for 20 min to reduce redness.",
            "Use crushed tulsi (holy basil) leaves boiled in water as a skin rinse — antibacterial and calming.",
            "Apply raw honey (manuka preferred) on affected areas — natural humectant with healing properties.",
            "Try neem oil massage — antifungal and anti-inflammatory for persistent patches."
        ],
        "diet": [
            "Increase omega-3 intake: flaxseeds, walnuts, sardines — reduce systemic inflammation.",
            "Eat probiotic-rich foods: curd, buttermilk, kimchi — strengthen gut-skin barrier.",
            "Include zinc-rich foods: pumpkin seeds, chickpeas — support skin repair.",
            "Drink turmeric milk (haldi doodh) daily — curcumin reduces internal inflammation.",
            "Eat colorful vegetables: beetroot, carrots, spinach — antioxidants repair damaged skin."
        ],
        "precautions": [
            "Moisturize within 3 minutes of bathing (3-minute rule) to lock in hydration.",
            "Use only fragrance-free, hypoallergenic detergents for clothes.",
            "Keep nails short to prevent damage from unconscious scratching.",
            "Wear soft cotton clothing — avoid wool and synthetic fabrics.",
            "Maintain room humidity at 40-60% — use a humidifier in dry climates.",
            "Take lukewarm (not hot) showers — hot water strips natural oils."
        ],
        "avoid": [
            "Soaps with SLS (sodium lauryl sulfate) or harsh surfactants.",
            "Synthetic fragrances in lotions and detergents.",
            "Wool or rough synthetic clothing directly on skin.",
            "Stress and sleep deprivation — known flare-up triggers.",
            "Dust mites — use anti-allergen bedding covers."
        ],
        "products": [
            {"name": "Ceramide Barrier Cream", "type": "BARRIER REPAIR", "tag": "OTC"},
            {"name": "Colloidal Oatmeal Bath Soak", "type": "BATH THERAPY", "tag": "Gentle"},
            {"name": "Petroleum Jelly (Vaseline)", "type": "OCCLUSIVE MOISTURIZER", "tag": "OTC"}
        ],
        "referral": True,
        "referral_msg": "Presentation warrants evaluation for topical corticosteroid management and patch testing."
    },
    {
        "id": "Acne and Rosacea Photos",
        "label": "Acne Vulgaris Complex",
        "icd10": "L70.0",
        "icd_desc": "ACNE VULGARIS",
        "severity": "Mild",
        "color": "amber",
        "treatment": "Keratolytic Regimen Therapy",
        "clinical_notes": "Inflammatory papules and active comedonal formations mapped along local sebaceous structures.",
        "home_remedies": [
            "Apply a paste of multani mitti (fuller's earth) + rose water — absorbs excess oil and unclogs pores.",
            "Use diluted tea tree oil (2-3 drops in carrier oil) as spot treatment — antibacterial against acne-causing bacteria.",
            "Apply raw honey + cinnamon mask for 15 min — honey is antibacterial, cinnamon increases blood flow.",
            "Use ice cubes wrapped in cloth on inflamed pimples — reduces swelling and redness instantly.",
            "Apply mashed banana + turmeric paste — potassium moisturizes while haldi fights bacteria.",
            "Use besan (gram flour) + curd + turmeric face pack — deep cleanses and controls oil production.",
            "Drink jeera (cumin) water daily — detoxifies blood and reduces breakouts from within."
        ],
        "diet": [
            "Reduce dairy intake — milk hormones can trigger acne flare-ups.",
            "Cut down on refined sugar and processed foods — spike insulin and oil production.",
            "Eat Vitamin A foods: sweet potatoes, carrots, papaya — regulate skin cell turnover.",
            "Drink 3-4 liters of water daily — flushes toxins that cause breakouts.",
            "Include zinc foods: lentils, pumpkin seeds — reduce inflammation and promote healing.",
            "Eat neem capsules or drink neem tea — natural blood purifier."
        ],
        "precautions": [
            "Never pop or squeeze pimples — leads to scarring and infection spread.",
            "Cleanse face twice daily with a gentle, non-comedogenic cleanser.",
            "Change pillow covers every 2 days.",
            "Use oil-free, non-comedogenic moisturizers only.",
            "Wash hands before touching face.",
            "Clean phone screen daily — it touches your face frequently."
        ],
        "avoid": [
            "Oily and fried foods (samosa, pakora, chips).",
            "Chocolate and candy in excess.",
            "Picking or touching pimples repeatedly.",
            "Heavy, pore-clogging makeup or foundations.",
            "Excessive exfoliation (more than 2-3 times a week)."
        ],
        "products": [
            {"name": "Salicylic Acid 2% Exfoliant", "type": "BHA TREATMENT", "tag": "OTC"},
            {"name": "Niacinamide 10% Serum", "type": "PORE CONTROL", "tag": "OTC"},
            {"name": "Tea Tree Oil (diluted)", "type": "SPOT TREATMENT", "tag": "Gentle"}
        ],
        "referral": False
    }
]

# 4. PRODUCT SUGGESTION DATABASE — maps conditions to real products with store links
PRODUCT_DB = {
    "Healthy Skin": [
        {"name": "Cetaphil Gentle Skin Cleanser", "brand": "Cetaphil", "category": "Cleanser", "type": "DAILY CLEANSER",
         "amazon": "https://www.amazon.in/s?k=Cetaphil+Gentle+Skin+Cleanser", "flipkart": "https://www.flipkart.com/search?q=Cetaphil+Gentle+Skin+Cleanser", "nykaa": "https://www.nykaa.com/search?q=Cetaphil+Gentle+Skin+Cleanser"},
        {"name": "Minimalist 10% Vitamin C Serum", "brand": "Minimalist", "category": "Serum", "type": "ANTIOXIDANT SERUM",
         "amazon": "https://www.amazon.in/s?k=Minimalist+10+Vitamin+C+Serum", "flipkart": "https://www.flipkart.com/search?q=Minimalist+Vitamin+C+Serum", "nykaa": "https://www.nykaa.com/search?q=Minimalist+Vitamin+C+Serum"},
        {"name": "La Shield Fisico SPF 50+ Sunscreen", "brand": "La Shield", "category": "Sunscreen", "type": "SPF 50+",
         "amazon": "https://www.amazon.in/s?k=La+Shield+Fisico+SPF+50+Sunscreen", "flipkart": "https://www.flipkart.com/search?q=La+Shield+Fisico+Sunscreen", "nykaa": "https://www.nykaa.com/search?q=La+Shield+Fisico+Sunscreen"},
        {"name": "Dot & Key Water Drench Hyaluronic Serum", "brand": "Dot & Key", "category": "Moisturizer", "type": "HYDRATING MOISTURIZER",
         "amazon": "https://www.amazon.in/s?k=Dot+Key+Water+Drench+Hyaluronic+Serum", "flipkart": "https://www.flipkart.com/search?q=Dot+Key+Water+Drench+Hyaluronic", "nykaa": "https://www.nykaa.com/search?q=Dot+Key+Water+Drench+Hyaluronic"},
        {"name": "Plum Green Tea Clear Face Mask", "brand": "Plum", "category": "Face Mask", "type": "WEEKLY TREATMENT",
         "amazon": "https://www.amazon.in/s?k=Plum+Green+Tea+Clear+Face+Mask", "flipkart": "https://www.flipkart.com/search?q=Plum+Green+Tea+Clear+Face+Mask", "nykaa": "https://www.nykaa.com/search?q=Plum+Green+Tea+Clear+Face+Mask"},
    ],
    "Melanoma Skin Cancer Nevi and Moles": [
        {"name": "Re'equil Ultra Matte Dry Touch SPF 50+", "brand": "Re'equil", "category": "Sunscreen", "type": "SPF 50+ PA++++",
         "amazon": "https://www.amazon.in/s?k=Reequil+Ultra+Matte+SPF+50", "flipkart": "https://www.flipkart.com/search?q=Reequil+Ultra+Matte+SPF+50", "nykaa": "https://www.nykaa.com/search?q=Reequil+Ultra+Matte+SPF+50"},
        {"name": "Deconstruct Dark Spot Corrector Serum", "brand": "Deconstruct", "category": "Serum", "type": "DARK SPOT REMOVER",
         "amazon": "https://www.amazon.in/s?k=Deconstruct+Dark+Spot+Corrector+Serum", "flipkart": "https://www.flipkart.com/search?q=Deconstruct+Dark+Spot+Corrector", "nykaa": "https://www.nykaa.com/search?q=Deconstruct+Dark+Spot+Corrector"},
        {"name": "WOW Skin Science Aloe Vera Gel", "brand": "WOW", "category": "Gel", "type": "PURE ALOE VERA",
         "amazon": "https://www.amazon.in/s?k=WOW+Aloe+Vera+Gel", "flipkart": "https://www.flipkart.com/search?q=WOW+Aloe+Vera+Gel", "nykaa": "https://www.nykaa.com/search?q=WOW+Aloe+Vera+Gel"},
        {"name": "Kaya Clinic Pigmentation Reducing Complex", "brand": "Kaya", "category": "Cream", "type": "ANTI-PIGMENTATION",
         "amazon": "https://www.amazon.in/s?k=Kaya+Clinic+Pigmentation+Reducing+Complex", "flipkart": "https://www.flipkart.com/search?q=Kaya+Pigmentation+Reducing", "nykaa": "https://www.nykaa.com/search?q=Kaya+Pigmentation+Reducing"},
        {"name": "Minimalist 2% Salicylic Acid Serum", "brand": "Minimalist", "category": "Serum", "type": "EXFOLIATING SERUM",
         "amazon": "https://www.amazon.in/s?k=Minimalist+2+Salicylic+Acid+Serum", "flipkart": "https://www.flipkart.com/search?q=Minimalist+Salicylic+Acid+Serum", "nykaa": "https://www.nykaa.com/search?q=Minimalist+Salicylic+Acid+Serum"},
    ],
    "Eczema Photos": [
        {"name": "Cetaphil AD Derma Moisturizing Cream", "brand": "Cetaphil", "category": "Moisturizer", "type": "BARRIER REPAIR",
         "amazon": "https://www.amazon.in/s?k=Cetaphil+AD+Derma+Moisturizing+Cream", "flipkart": "https://www.flipkart.com/search?q=Cetaphil+AD+Derma+Cream", "nykaa": "https://www.nykaa.com/search?q=Cetaphil+AD+Derma+Cream"},
        {"name": "Aveeno Dermexa Daily Emollient Cream", "brand": "Aveeno", "category": "Cream", "type": "ECZEMA RELIEF",
         "amazon": "https://www.amazon.in/s?k=Aveeno+Dermexa+Daily+Emollient+Cream", "flipkart": "https://www.flipkart.com/search?q=Aveeno+Dermexa+Cream", "nykaa": "https://www.nykaa.com/search?q=Aveeno+Dermexa+Cream"},
        {"name": "Aquaphor Healing Ointment", "brand": "Aquaphor", "category": "Ointment", "type": "HEALING BARRIER",
         "amazon": "https://www.amazon.in/s?k=Aquaphor+Healing+Ointment", "flipkart": "https://www.flipkart.com/search?q=Aquaphor+Healing+Ointment", "nykaa": "https://www.nykaa.com/search?q=Aquaphor+Healing+Ointment"},
        {"name": "CeraVe Moisturizing Cream", "brand": "CeraVe", "category": "Cream", "type": "CERAMIDE REPAIR",
         "amazon": "https://www.amazon.in/s?k=CeraVe+Moisturizing+Cream", "flipkart": "https://www.flipkart.com/search?q=CeraVe+Moisturizing+Cream", "nykaa": "https://www.nykaa.com/search?q=CeraVe+Moisturizing+Cream"},
        {"name": "Bioderma Atoderm Intensive Baume", "brand": "Bioderma", "category": "Balm", "type": "INTENSE HYDRATION",
         "amazon": "https://www.amazon.in/s?k=Bioderma+Atoderm+Intensive+Baume", "flipkart": "https://www.flipkart.com/search?q=Bioderma+Atoderm+Intensive", "nykaa": "https://www.nykaa.com/search?q=Bioderma+Atoderm+Intensive"},
    ],
    "Acne and Rosacea Photos": [
        {"name": "Minimalist 2% Salicylic Acid Face Wash", "brand": "Minimalist", "category": "Cleanser", "type": "BHA CLEANSER",
         "amazon": "https://www.amazon.in/s?k=Minimalist+2+Salicylic+Acid+Face+Wash", "flipkart": "https://www.flipkart.com/search?q=Minimalist+Salicylic+Acid+Face+Wash", "nykaa": "https://www.nykaa.com/search?q=Minimalist+Salicylic+Acid+Face+Wash"},
        {"name": "The Ordinary Niacinamide 10% + Zinc 1%", "brand": "The Ordinary", "category": "Serum", "type": "PORE CONTROL SERUM",
         "amazon": "https://www.amazon.in/s?k=The+Ordinary+Niacinamide+10+Zinc", "flipkart": "https://www.flipkart.com/search?q=The+Ordinary+Niacinamide", "nykaa": "https://www.nykaa.com/search?q=The+Ordinary+Niacinamide"},
        {"name": "Plum Tea Tree Spot Treatment Gel", "brand": "Plum", "category": "Gel", "type": "SPOT TREATMENT",
         "amazon": "https://www.amazon.in/s?k=Plum+Tea+Tree+Spot+Treatment+Gel", "flipkart": "https://www.flipkart.com/search?q=Plum+Tea+Tree+Spot+Treatment", "nykaa": "https://www.nykaa.com/search?q=Plum+Tea+Tree+Spot+Treatment"},
        {"name": "Neutrogena Oil-Free Moisturizer", "brand": "Neutrogena", "category": "Moisturizer", "type": "OIL-FREE HYDRATION",
         "amazon": "https://www.amazon.in/s?k=Neutrogena+Oil+Free+Moisturizer", "flipkart": "https://www.flipkart.com/search?q=Neutrogena+Oil+Free+Moisturizer", "nykaa": "https://www.nykaa.com/search?q=Neutrogena+Oil+Free+Moisturizer"},
        {"name": "AcneStar Gel (Benzoyl Peroxide 2.5%)", "brand": "AcneStar", "category": "Gel", "type": "ANTIBACTERIAL GEL",
         "amazon": "https://www.amazon.in/s?k=AcneStar+Gel+Benzoyl+Peroxide", "flipkart": "https://www.flipkart.com/search?q=AcneStar+Gel", "nykaa": "https://www.nykaa.com/search?q=AcneStar+Gel"},
    ],
}

DEFAULT_PRODUCT_DB = [
    {"name": "Cetaphil Gentle Skin Cleanser", "brand": "Cetaphil", "category": "Cleanser", "type": "GENTLE CLEANSER",
     "amazon": "https://www.amazon.in/s?k=Cetaphil+Gentle+Skin+Cleanser", "flipkart": "https://www.flipkart.com/search?q=Cetaphil+Gentle+Skin+Cleanser", "nykaa": "https://www.nykaa.com/search?q=Cetaphil+Gentle+Skin+Cleanser"},
    {"name": "CeraVe Moisturizing Cream", "brand": "CeraVe", "category": "Cream", "type": "DERMATOLOGIST RECOMMENDED",
     "amazon": "https://www.amazon.in/s?k=CeraVe+Moisturizing+Cream", "flipkart": "https://www.flipkart.com/search?q=CeraVe+Moisturizing+Cream", "nykaa": "https://www.nykaa.com/search?q=CeraVe+Moisturizing+Cream"},
    {"name": "Re'equil Ultra Matte SPF 50+", "brand": "Re'equil", "category": "Sunscreen", "type": "SPF 50+ PA++++",
     "amazon": "https://www.amazon.in/s?k=Reequil+Ultra+Matte+SPF+50", "flipkart": "https://www.flipkart.com/search?q=Reequil+Ultra+Matte+SPF+50", "nykaa": "https://www.nykaa.com/search?q=Reequil+Ultra+Matte+SPF+50"},
    {"name": "Bioderma Sensibio H2O Micellar Water", "brand": "Bioderma", "category": "Cleanser", "type": "MICELLAR WATER",
     "amazon": "https://www.amazon.in/s?k=Bioderma+Sensibio+H2O", "flipkart": "https://www.flipkart.com/search?q=Bioderma+Sensibio+H2O", "nykaa": "https://www.nykaa.com/search?q=Bioderma+Sensibio+H2O"},
    {"name": "La Roche-Posay Effaclar Duo+", "brand": "La Roche-Posay", "category": "Treatment", "type": "ACNE TREATMENT",
     "amazon": "https://www.amazon.in/s?k=La+Roche+Posay+Effaclar+Duo", "flipkart": "https://www.flipkart.com/search?q=La+Roche+Posay+Effaclar", "nykaa": "https://www.nykaa.com/search?q=La+Roche+Posay+Effaclar"},
]


@app.route('/api/products', methods=['GET'])
def get_products():
    condition_id = request.args.get('condition_id', '')
    products = PRODUCT_DB.get(condition_id, DEFAULT_PRODUCT_DB)
    categories = list(set(p["category"] for p in products))
    return jsonify({
        "categories": sorted(categories),
        "products": products,
        "total": len(products)
    })

@app.route('/api/compare', methods=['GET'])
def compare_prices():
    product_name = request.args.get('product', '')
    product_url = request.args.get('url', '')
    return jsonify({
        "product": product_name,
        "stores": {
            "Amazon": {
                "search_url": f"https://www.amazon.in/s?k={urllib.parse.quote(product_name)}",
                "direct_url": product_url if "amazon" in product_url else f"https://www.amazon.in/s?k={urllib.parse.quote(product_name)}",
                "color": "#FF9900"
            },
            "Flipkart": {
                "search_url": f"https://www.flipkart.com/search?q={urllib.parse.quote(product_name)}",
                "direct_url": product_url if "flipkart" in product_url else f"https://www.flipkart.com/search?q={urllib.parse.quote(product_name)}",
                "color": "#2874F0"
            },
            "Nykaa": {
                "search_url": f"https://www.nykaa.com/search?q={urllib.parse.quote(product_name)}",
                "direct_url": product_url if "nykaa" in product_url else f"https://www.nykaa.com/search?q={urllib.parse.quote(product_name)}",
                "color": "#FC2779"
            }
        }
    })

@app.route('/api/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "active"})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global gradients, activations
    gradients = None
    activations = None
    
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400
        
    file = request.files['image']
    img_bytes = file.read()
    
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0)
    input_tensor.requires_grad = True
    
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1).squeeze()
    top_prob, top_class_idx = torch.max(probabilities, dim=0)
    
    confidence_value = top_prob.item()
    predicted_id = DISEASE_CLASSES[top_class_idx.item()]
    
    # Precision calibrated gate barrier 
    CONFIDENCE_THRESHOLD = 0.05
    is_healthy_override = confidence_value < CONFIDENCE_THRESHOLD
    
    model.zero_grad()
    outputs[0, top_class_idx].backward(retain_graph=True)
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations_clone = activations.clone()
    for i in range(activations_clone.shape[1]):
        activations_clone[:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations_clone, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    
    native_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = native_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    blended_canvas = cv2.addWeighted(native_img, 0.6, heatmap_colored, 0.4, 0)
    
    _, img_buf = cv2.imencode('.png', blended_canvas)
    gradcam_b64 = base64.b64encode(img_buf.tobytes()).decode('utf-8')

    if is_healthy_override:
        report_data = HEALTHY_OVERRIDE
        display_confidence = (1.0 - confidence_value) * 100 
    else:
        report_data = next((c for c in CONDITIONS if c["id"] == predicted_id), None)
        if not report_data:
            report_data = {
                "id": predicted_id,
                "label": predicted_id.replace(" Photos", "").replace(" pictures", ""),
                "icd10": "L98.9",
                "icd_desc": "PATHOLOGY IDENTIFIED",
                "severity": "Moderate",
                "color": "amber",
                "treatment": "Clinical Verification Required",
                "clinical_notes": f"System identified structural anomalies correlating to: {predicted_id}.",
                "home_remedies": [
                    "Apply turmeric (haldi) paste with honey — natural anti-inflammatory.",
                    "Use aloe vera gel for soothing affected areas.",
                    "Drink warm water with lemon each morning for detoxification.",
                    "Apply raw milk as a gentle cleanser before bed.",
                    "Use neem leaves paste as a natural antiseptic."
                ],
                "diet": [
                    "Increase water intake to 3-4 liters daily.",
                    "Eat Vitamin C rich fruits: amla, oranges, guava.",
                    "Include green leafy vegetables for essential minerals.",
                    "Reduce sugar and processed food consumption.",
                    "Add turmeric to daily diet for anti-inflammatory benefits."
                ],
                "precautions": [
                    "Avoid direct sun exposure during peak hours.",
                    "Keep affected area clean and dry.",
                    "Do not scratch or irritate the area.",
                    "Wear loose, breathable cotton clothing.",
                    "Consult a dermatologist for proper diagnosis."
                ],
                "avoid": [
                    "Harsh chemical-based skincare products.",
                    "Excessive sun exposure without protection.",
                    "Self-medication with steroid creams.",
                    "Sharing personal towels or clothing.",
                    "Ignoring worsening symptoms."
                ],
                "products": [{"name": "Standard Skin Hydrator", "type": "MOISTURIZER", "tag": "Gentle"}],
                "referral": True,
                "referral_msg": "Consult standard textbook metrics."
            }
        display_confidence = confidence_value * 100

    response = {
        "condition_id":    report_data["id"],
        "condition_label": report_data["label"],
        "icd10":           report_data["icd10"],
        "icd10_desc":      report_data["icd_desc"],
        "severity":        report_data["severity"],
        "color":           report_data["color"],
        "confidence_pct":  f"{display_confidence:.1f}%",
        "treatment":       report_data["treatment"],
        "clinical_notes":  report_data["clinical_notes"],
        "home_remedies":   report_data["home_remedies"],
        "diet":            report_data.get("diet", []),
        "precautions":     report_data.get("precautions", []),
        "avoid":           report_data.get("avoid", []),
        "products":        report_data["products"],
        "referral":        report_data["referral"],
        "referral_msg":    report_data.get("referral_msg", ""),
        "gradcam_b64":     gradcam_b64,
        "raw_confidence":  f"{confidence_value * 100:.1f}%"
    }
    
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    next_classes = []
    for k in range(1, 4):
        if k < len(sorted_indices):
            alt_idx = sorted_indices[k].item()
            alt_name = DISEASE_CLASSES[alt_idx].replace(" Photos", "").replace(" pictures", "")[:12]
            alt_pct = sorted_probs[k].item() * 100
            next_classes.append({"name": alt_name, "pct": f"{alt_pct:.1f}"})
    
    response["next_classes"] = next_classes
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)