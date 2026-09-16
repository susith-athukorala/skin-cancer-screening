import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models

st.set_page_config(page_title="3-Tier Skin Cancer Screener", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- AUTO-DOWNLOAD WEIGHTS FROM GITHUB RELEASE IF NOT LOCAL ---
os.makedirs("models", exist_ok=True)
GK_URL = "https://github.com/susith-athukorala/skin-cancer-screening/releases/download/v1.0/tier1_gatekeeper_v2.pth"
MODEL_URL = "https://github.com/susith-athukorala/skin-cancer-screening/releases/download/v1.0/skin_cancer_hierarchical_model.pth"

gk_path = os.path.join("models", "tier1_gatekeeper_v2.pth")
model_path = os.path.join("models", "skin_cancer_hierarchical_model.pth")

@st.cache_resource
def load_models():
    if not os.path.exists(gk_path):
        with st.spinner("Downloading Tier 1 weights..."):
            urllib.request.urlretrieve(GK_URL, gk_path)
            
    if not os.path.exists(model_path):
        with st.spinner("Downloading Tier 2 weights..."):
            urllib.request.urlretrieve(MODEL_URL, model_path)

    # Gatekeeper
    gk = models.mobilenet_v3_small(weights=None)
    in_feat = gk.classifier[3].in_features
    gk.classifier[3] = nn.Linear(in_feat, 2)
    gk.load_state_dict(torch.load(gk_path, map_location=device))
    gk.to(device).eval()

    # Screener
    class HierarchicalEfficientNet(nn.Module):
        def __init__(self, num_lineages=3, num_terminals=6):
            super().__init__()
            self.backbone = models.efficientnet_b0(weights=None)
            in_f = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.lineage_head = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(in_f, 128), nn.ReLU(), nn.Linear(128, num_lineages)
            )
            self.terminal_head = nn.Sequential(
                nn.Dropout(p=0.3), nn.Linear(in_f + num_lineages, 128), nn.ReLU(), nn.Linear(128, num_terminals)
            )
        def forward(self, x):
            feats = self.backbone(x)
            l_logits = self.lineage_head(feats)
            l_probs = F.softmax(l_logits, dim=1)
            t_logits = self.terminal_head(torch.cat([feats, l_probs], dim=1))
            return l_logits, t_logits

    screener = HierarchicalEfficientNet(3, 6)
    screener.load_state_dict(torch.load(model_path, map_location=device))
    screener.to(device).eval()
    return gk, screener

gatekeeper, screener = load_models()

# --- PREPROCESSING ---
class DullRazor(object):
    def __init__(self, filter_size=9, inpaint_radius=3):
        self.filter_size, self.inpaint_radius = filter_size, inpaint_radius
    def __call__(self, pil_img):
        img = np.array(pil_img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.filter_size, self.filter_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        return Image.fromarray(cv2.inpaint(img, mask, self.inpaint_radius, cv2.INPAINT_TELEA))

class ShadesOfGray(object):
    def __init__(self, power=6):
        self.power = power
    def __call__(self, pil_img):
        img = np.array(pil_img).astype(np.float32) + 1e-5
        illuminant = np.power(np.mean(np.power(img, self.power), axis=(0, 1)), 1.0 / self.power)
        illuminant = illuminant / np.linalg.norm(illuminant)
        return Image.fromarray(np.clip(img / (illuminant * np.sqrt(3)), 0, 255).astype(np.uint8))

eval_transform = transforms.Compose([
    DullRazor(), ShadesOfGray(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

gk_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

lineages = ['Melanocytic Lesion', 'Keratinocytic / Epidermal Lesion', 'Vascular Lesion']
terminal_classes = ['mel', 'bcc', 'akiec', 'nev', 'kerat', 'vasc']
terminal_labels = {
    'mel': 'Melanoma (Malignant)', 'bcc': 'Basal Cell Carcinoma (Malignant)',
    'akiec': 'Actinic Keratosis / Bowen’s', 'nev': 'Melanocytic Nevus',
    'kerat': 'Keratotic Mimic (SK / Lentigo)', 'vasc': 'Vascular Lesion'
}

def verify_specimen_salience(pil_img):
    cv_img = np.array(pil_img.convert('L'))
    h, w = cv_img.shape
    c_y1, c_y2 = int(h * 0.35), int(h * 0.65)
    c_x1, c_x2 = int(w * 0.35), int(w * 0.65)
    center_core = cv_img[c_y1:c_y2, c_x1:c_x2]
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = False
    delta_lum = abs(float(np.mean(center_core)) - float(np.mean(cv_img[border_mask])))

    ring_mask = np.zeros((h, w), dtype=bool)
    ring_mask[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)] = True
    ring_mask[c_y1:c_y2, c_x1:c_x2] = False
    local_contrast = abs(float(np.mean(center_core)) - float(np.mean(cv_img[ring_mask])))

    gx = np.mean(np.abs(cv2.Sobel(center_core, cv2.CV_64F, 1, 0, ksize=3)))
    gy = np.mean(np.abs(cv2.Sobel(center_core, cv2.CV_64F, 0, 1, ksize=3)))
    anisotropy = abs(gx - gy) / (gx + gy + 1e-5)

    if anisotropy > 0.28:
        return False, delta_lum, local_contrast, anisotropy
    if local_contrast < 4.65 or (local_contrast < 6.5 and delta_lum < 18.0):
        return False, delta_lum, local_contrast, anisotropy
    return True, delta_lum, local_contrast, anisotropy

# --- UI INTERFACE ---
st.title("🔬 3-Tier Hierarchical Skin Cancer Screening Tool by Susith")
st.markdown("Dual Tier 1 Intake Gate + Lineage Decoupling + Asymmetric Clinical Triage")

uploaded = st.file_uploader("Upload Lesion Image (Close-Up)", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(img, caption="Uploaded Specimen", use_container_width=True)
    
    is_salient, d_lum, l_contrast, aniso = verify_specimen_salience(img)
    with torch.no_grad():
        gk_probs = F.softmax(gatekeeper(gk_transform(img).unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    
    prob_lesion = float(gk_probs[1])
    
    with col2:
        if not is_salient or prob_lesion < 0.60:
            st.error("⚠️ SPECIMEN REJECTED: NO FOCAL LESION DETECTED")
            st.markdown(f"""
            - **Intake Analysis:** Specimen rejected at Intake Gate.
            - **Local Contrast:** `{l_contrast:.2f}` (Threshold >= 4.65)
            - **Directional Anisotropy:** `{aniso:.3f}` (Ceiling <= 0.280)
            - **Tier 1 MobileNet Confidence:** `{prob_lesion*100:.1f}%`
            
            *Specimen identified as featureless cutis, textured surface, or OOD artifact.*
            """)
        else:
            with torch.no_grad():
                l_logits, t_logits = screener(eval_transform(img).unsqueeze(0).to(device))
                l_probs = F.softmax(l_logits, dim=1)[0].cpu().numpy()
                t_probs = F.softmax(t_logits, dim=1)[0].cpu().numpy()
            
            prob_map = {terminal_classes[i]: float(t_probs[i]) for i in range(len(terminal_classes))}
            entropy = -np.sum(t_probs * np.log(t_probs + 1e-8))
            ambiguity = (entropy / np.log(len(terminal_classes))) * 100
            
            mel_r, bcc_r, akiec_r = prob_map['mel'], prob_map['bcc'], prob_map['akiec']
            kerat_r, nev_r = prob_map['kerat'], prob_map['nev']
            
            if ambiguity > 65.0:
                triage, color = "⚠️ HIGH UNCERTAINTY / INDETERMINATE MORPHOLOGY", "orange"
            elif mel_r >= 0.42 or (mel_r >= 0.28 and mel_r > nev_r) or (mel_r + bcc_r) >= 0.48:
                triage, color = "🔴 HIGH RISK: SUSPECTED MALIGNANCY", "red"
            elif bcc_r >= 0.38:
                triage, color = "🔴 HIGH RISK: BASAL CELL CARCINOMA", "red"
            elif akiec_r >= 0.05 or (akiec_r >= 0.015 and (l_probs[1] >= 0.10 or kerat_r >= 0.05)):
                triage, color = "🟡 PRE-MALIGNANT (Actinic Keratosis / Bowen's)", "gold"
            elif prob_map['vasc'] >= 0.50:
                triage, color = "🟢 BENIGN VASCULAR", "green"
            elif kerat_r >= 0.40:
                triage, color = "🟢 BENIGN KERATOSIS", "green"
            else:
                triage, color = "🟢 BENIGN MELANOCYTIC NEVUS", "green"
                
            st.markdown(f"### Triage: :{color}[{triage}]")
            st.markdown(f"**Dominant Lineage:** {lineages[np.argmax(l_probs)]} ({np.max(l_probs)*100:.1f}%)")
            st.markdown(f"**Ambiguity Index:** `{ambiguity:.1f}%`")
            
            st.write("#### Probabilities:")
            st.bar_chart({terminal_labels[terminal_classes[i]]: t_probs[i] for i in range(len(terminal_classes))})
