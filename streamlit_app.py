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

st.set_page_config(
    page_title="3-Tier Hierarchical Skin Cancer Screening Model by Susith",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------
# 1. Weights Mount / Auto-Download
# -------------------------------------------------------------
os.makedirs("models", exist_ok=True)

# NOTE: Adjust the repo name below if your repo is named 'skin-cancer-screening-pipeline'
REPO_NAME = "skin-cancer-screening"
GK_URL = f"https://github.com/susith-athukorala/skin-cancer-screening/releases/download/v1.0/tier1_gatekeeper_v2.pth"
MODEL_URL = f"https://github.com/susith-athukorala/skin-cancer-screening/releases/download/v1.0/skin_cancer_hierarchical_model.pth"

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

    # Tier 1 Gatekeeper
    gk = models.mobilenet_v3_small(weights=None)
    in_feat = gk.classifier[3].in_features
    gk.classifier[3] = nn.Linear(in_feat, 2)
    gk.load_state_dict(torch.load(gk_path, map_location=device))
    gk.to(device).eval()

    # Tier 2 Hierarchical Screener
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

gatekeeper, model = load_models()

# -------------------------------------------------------------
# 2. Grad-CAM Engine
# -------------------------------------------------------------
class HierarchicalGradCAM:
    def __init__(self, target_model, conv_layer):
        self.model, self.target_layer = target_model, conv_layer
        self.gradients, self.activations = None, None
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def explain(self, tensor):
        self.model.eval()
        self.model.zero_grad()
        l_logits, t_logits = self.model(tensor)
        l_probs = F.softmax(l_logits, dim=1)[0].detach().cpu().numpy()
        t_probs = F.softmax(t_logits, dim=1)[0].detach().cpu().numpy()

        top_idx = torch.argmax(t_logits, dim=1).item()
        t_logits[0, top_idx].backward()

        weights = torch.mean(self.gradients, dim=[0, 2, 3])
        act = self.activations[0]
        for i in range(act.shape[0]):
            act[i, :, :] *= weights[i]
        cam = torch.mean(act, dim=0).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam /= np.max(cam)
        return cam, l_probs, t_probs, top_idx

cam_engine = HierarchicalGradCAM(model, model.backbone.features[-1])

# -------------------------------------------------------------
# 3. Pre-processing Transforms
# -------------------------------------------------------------
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

hair_remover = DullRazor(filter_size=9, inpaint_radius=3)

eval_transform = transforms.Compose([
    hair_remover,
    ShadesOfGray(power=6),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

gatekeeper_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

lineages = ['Melanocytic Lesion', 'Keratinocytic / Epidermal Lesion', 'Vascular Lesion']
terminal_classes = ['mel', 'bcc', 'akiec', 'nev', 'kerat', 'vasc']
terminal_labels = {
    'mel': 'Melanoma (Malignant)',
    'bcc': 'Basal Cell Carcinoma (Malignant)',
    'akiec': 'Actinic Keratosis / Bowen’s (Pre-malignant)',
    'nev': 'Melanocytic Nevus (Benign Mole)',
    'kerat': 'Keratotic Mimic (Seborrheic Keratosis / Lentigo)',
    'vasc': 'Vascular Lesion (Benign)'
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

    sobel_x = np.abs(cv2.Sobel(center_core, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = np.abs(cv2.Sobel(center_core, cv2.CV_64F, 0, 1, ksize=3))
    mean_gx = np.mean(sobel_x)
    mean_gy = np.mean(sobel_y)
    anisotropy_ratio = abs(mean_gx - mean_gy) / (mean_gx + mean_gy + 1e-5)

    if anisotropy_ratio > 0.28:
        return False, delta_lum, local_contrast, anisotropy_ratio

    if local_contrast < 4.65 or (local_contrast < 6.5 and delta_lum < 18.0):
        return False, delta_lum, local_contrast, anisotropy_ratio

    return True, delta_lum, local_contrast, anisotropy_ratio

def render_prob_bar(label, prob):
    pct = prob * 100
    st.markdown(
        f"""
        <div style="margin-bottom: 6px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 2px;">
                <span>{label}</span>
                <span style="font-weight: 600;">{pct:.1f}%</span>
            </div>
            <div style="background-color: #eee; border-radius: 4px; height: 10px; width: 100%; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #ff8c00, #ff5722); height: 100%; width: {pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# 4. Streamlit Layout
# -------------------------------------------------------------
st.title("3-Tier Hierarchical Skin Cancer Screening Model (v2) by Susith")
st.caption("Dual Tier 1 specimen validation filter + histogenetic lineage decoupling + asymmetric clinical diagnostic matrix.")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Upload Lesion (Centered Close-Up)")
    uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file is not None:
        pil_raw = Image.open(uploaded_file).convert('RGB')
        st.image(pil_raw, use_container_width=True)

with right_col:
    if uploaded_file is None:
        st.info("Upload a lesion image on the left to start screening.")
    else:
        # Tier 1 Saliency Gate
        is_salient, d_lum, l_contrast, aniso = verify_specimen_salience(pil_raw)

        gk_tensor = gatekeeper_transform(pil_raw).unsqueeze(0).to(device)
        with torch.no_grad():
            gk_logits = gatekeeper(gk_tensor)
            gk_probs = F.softmax(gk_logits, dim=1)[0].cpu().numpy()

        prob_invalid = float(gk_probs[0])
        prob_lesion = float(gk_probs[1])

        if not is_salient or prob_lesion < 0.60:
            gray_display = np.array(pil_raw.resize((224, 224)))
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.caption("Cleaned Substrate")
                st.image(gray_display, use_container_width=True)
            with sub_col2:
                st.caption("Grad-CAM Localization")
                st.image(gray_display, use_container_width=True)

            st.error("⚠️ SPECIMEN REJECTED: NO FOCAL LESION DETECTED")
            st.markdown(f"""
* **Intake Analysis:** Specimen rejected at Intake Gate.
* **Core Radial Delta:** `{d_lum:.2f}`
* **Local Contrast Gradient:** `{l_contrast:.2f}` (Threshold: >= 4.65)
* **Directional Anisotropy:** `{aniso:.3f}` (Streak ceiling: <= 0.280)
* **Tier 1 MobileNet Confidence:** `{prob_lesion*100:.1f}%`

**Clinical Pathway:** Specimen rejected. The frame lacks a discrete circumscribed lesion (identified as featureless cutis, diffuse background, or out-of-distribution artifact). Pipeline halted to avoid false positive classification. Re-center dermatoscope over a discrete lesion.
            """)

            st.markdown("### Lineage Probabilities")
            render_prob_bar("Out-of-Distribution / Non-Lesion", max(prob_invalid, 0.95))

            st.markdown("### Specific Histology Breakdown")
            render_prob_bar("Specimen Rejected", 1.0)

        else:
            # Tier 2 Dual-Head Screener + Grad-CAM
            cleaned_img = hair_remover(pil_raw)
            tensor = eval_transform(pil_raw).unsqueeze(0).to(device)
            cam_map, l_probs, t_probs, _ = cam_engine.explain(tensor)

            base_display = np.array(cleaned_img.resize((224, 224)))
            cam_resized = cv2.resize(cam_map, (224, 224))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            blended = np.uint8(0.6 * base_display + 0.4 * cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.caption("Cleaned Substrate")
                st.image(cleaned_img, use_container_width=True)
            with sub_col2:
                st.caption("Grad-CAM Localization")
                st.image(blended, use_container_width=True)

            prob_map = {terminal_classes[i]: float(t_probs[i]) for i in range(len(terminal_classes))}
            entropy = -np.sum(t_probs * np.log(t_probs + 1e-8))
            ambiguity_pct = (entropy / np.log(len(terminal_classes))) * 100

            # Tier 3 Decision Matrix
            mel_risk = prob_map['mel']
            bcc_risk = prob_map['bcc']
            akiec_risk = prob_map['akiec']
            nev_risk = prob_map['nev']
            kerat_risk = prob_map['kerat']
            combined_malignancy = mel_risk + bcc_risk
            kerat_lineage = l_probs[1]

            if ambiguity_pct > 65.0:
                triage = "⚠️ HIGH UNCERTAINTY / INDETERMINATE MORPHOLOGY"
                top_label = "Uncertain / Overlapping Signatures"
                plan = f"Model Ambiguity is elevated ({ambiguity_pct:.1f}%). Conflicting multi-class signals. Clinical dermoscopy or punch biopsy advised."
            elif mel_risk >= 0.42 or (mel_risk >= 0.28 and mel_risk > nev_risk) or combined_malignancy >= 0.48:
                triage = "🔴 HIGH RISK: SUSPECTED MALIGNANCY"
                top_label = terminal_labels['mel'] if mel_risk > bcc_risk else terminal_labels['bcc']
                plan = f"Elevated invasive neoplastic signature (Melanoma: {mel_risk*100:.1f}%, BCC: {bcc_risk*100:.1f}%). Urgent specialist excision indicated."
            elif bcc_risk >= 0.38:
                triage = "🔴 HIGH RISK: BASAL CELL CARCINOMA"
                top_label = terminal_labels['bcc']
                plan = f"Basaloid/keratinocytic pattern ({bcc_risk*100:.1f}%). Referral for confirmatory biopsy."
            elif akiec_risk >= 0.05 or (akiec_risk >= 0.015 and (kerat_lineage >= 0.10 or kerat_risk >= 0.05)):
                triage = "🟡 PRE-MALIGNANT (Actinic Keratosis / Bowen's Disease)"
                top_label = terminal_labels['akiec']
                plan = f"Epidermal keratinocytic atypia / Bowenoid change suspected (AKIEC logit: {akiec_risk*100:.1f}%, Keratotic tail: {kerat_risk*100:.1f}%). Specialist dermoscopy or biopsy indicated."
            elif prob_map['vasc'] >= 0.50:
                triage = "🟢 BENIGN VASCULAR"
                top_label = terminal_labels['vasc']
                plan = "Vascular ectasia/proliferation pattern. Reassure patient."
            elif prob_map['kerat'] >= 0.40:
                triage = "🟢 BENIGN KERATOSIS"
                top_label = terminal_labels['kerat']
                plan = "Morphology aligns with Seborrheic Keratosis / Solar Lentigo."
            else:
                triage = "🟢 BENIGN MELANOCYTIC NEVUS"
                top_label = terminal_labels['nev']
                plan = "Uniform melanocytic architecture. Routine surveillance."

            st.markdown(f"### Clinical Triage: {triage}")
            st.markdown(f"""
* **Tier 1 Intake Gate:** `Passed` ({prob_lesion*100:.1f}% lesion confidence | Delta: `{d_lum:.1f}`)
* **Stage 1 Lineage:** `{lineages[np.argmax(l_probs)]}` ({np.max(l_probs)*100:.1f}% confidence)
* **Diagnostic Call:** **{top_label}**
* **Model Ambiguity Index:** `{ambiguity_pct:.1f}%`

**Clinical Pathway:** {plan}
            """)

            st.markdown("### Lineage Probabilities")
            lineage_pairs = sorted(zip(lineages, l_probs), key=lambda x: x[1], reverse=True)
            for lin_name, lin_prob in lineage_pairs:
                render_prob_bar(lin_name, lin_prob)

            st.markdown("### Specific Histology Breakdown")
            terminal_pairs = sorted(
                [(terminal_labels[k], prob_map[k]) for k in terminal_classes],
                key=lambda x: x[1],
                reverse=True
            )
            for term_name, term_prob in terminal_pairs[:4]:
                render_prob_bar(term_name, term_prob)
