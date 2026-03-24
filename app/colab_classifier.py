"""
Applied Mashroomatics - Colab Version
Run this in Google Colab with: !streamlit run app/colab_classifier.py &
Then use localtunnel: !npx localtunnel --port 8501
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import sys
import os

# Colab paths
sys.path.insert(0, '/content/Applied_Mashroomatics')

st.set_page_config(
    page_title="Applied Mashroomatics",
    page_icon="🍄",
    layout="wide"
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model paths for Colab
YOLO_PATH = '/content/drive/MyDrive/yolo_mushroom/best.pt'
CNN_PATH = '/content/drive/MyDrive/mushroom_checkpoints/cnn_best.pth'
VIT_PATH = '/content/drive/MyDrive/mushroom_checkpoints/best_model.pth'

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@st.cache_resource
def load_vit_model():
    """Load ViT model with metadata fusion."""
    try:
        from src.models.mushroom_vit import MushroomViTWithMetadata

        ckpt = torch.load(VIT_PATH, map_location='cpu', weights_only=False)

        # Try different keys for species mapping
        species_to_id = ckpt.get('species_to_id', {})
        if not species_to_id:
            id_to_species = ckpt.get('id_to_species', {})
            if id_to_species:
                species_to_id = {v: k for k, v in id_to_species.items()}

        habitat_vocab = ckpt.get('habitat_vocab', {})
        substrate_vocab = ckpt.get('substrate_vocab', {})
        id_to_species = {v: k for k, v in species_to_id.items()} if species_to_id else {}
        num_classes = ckpt.get('num_classes', len(species_to_id) if species_to_id else 179)

        model = MushroomViTWithMetadata(
            num_classes=num_classes,
            habitat_vocab=habitat_vocab,
            substrate_vocab=substrate_vocab,
            pretrained=False
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(DEVICE).eval()

        return {
            'model': model,
            'species_to_id': species_to_id,
            'id_to_species': id_to_species,
            'habitat_vocab': habitat_vocab,
            'substrate_vocab': substrate_vocab,
            'num_classes': num_classes,
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


@st.cache_resource
def load_cnn_model():
    """Load CNN model."""
    try:
        if not os.path.exists(CNN_PATH):
            return {'loaded': False, 'error': 'Checkpoint not found'}

        import timm
        ckpt = torch.load(CNN_PATH, map_location='cpu', weights_only=False)

        num_classes = ckpt.get('num_classes', 179)
        species_to_id = ckpt.get('species_to_id', {})
        id_to_species = {v: k for k, v in species_to_id.items()}

        model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(DEVICE).eval()

        return {
            'model': model,
            'id_to_species': id_to_species,
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


@st.cache_resource
def load_yolo_model():
    """Load YOLO model."""
    try:
        if not os.path.exists(YOLO_PATH):
            return {'loaded': False, 'error': 'Checkpoint not found'}

        from ultralytics import YOLO
        model = YOLO(YOLO_PATH)

        return {'model': model, 'loaded': True}
    except Exception as e:
        return {'loaded': False, 'error': str(e)}


def get_attention_rollout(model, img_tensor, h_id, s_id, month, original_image):
    """Generate attention rollout visualization."""
    model.eval()
    with torch.no_grad():
        B = img_tensor.shape[0]

        x = model.vit.patch_embed(img_tensor)
        cls_token = model.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + model.vit.pos_embed

        meta_feat = model.meta_encoder(h_id, s_id, month).unsqueeze(1)
        meta_feat = meta_feat + model.meta_token
        x = torch.cat([x[:, :1], meta_feat, x[:, 1:]], dim=1)

        x = model.vit.pos_drop(x)

        all_attns = []
        for block in model.vit.blocks:
            y = block.norm1(x)
            B, N, C = y.shape
            qkv = block.attn.qkv(y).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            all_attns.append(attn.mean(dim=1))
            x = block(x)

        rollout = torch.eye(all_attns[0].shape[-1]).to(img_tensor.device)
        for attn in all_attns:
            attn = attn + torch.eye(attn.shape[-1]).to(img_tensor.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            rollout = attn @ rollout

        cls_attn = rollout[0, 0, 2:]
        cls_attn = cls_attn.reshape(14, 14)
        cls_attn = F.interpolate(cls_attn.unsqueeze(0).unsqueeze(0),
                                  size=(224, 224), mode='bilinear', align_corners=False)
        cls_attn = cls_attn.squeeze().cpu().numpy()
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    img_resized = original_image.resize((224, 224))
    img_array = np.array(img_resized)

    heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
    heatmap[:, :, 0] = np.clip(cls_attn * 3, 0, 1) * 255
    heatmap[:, :, 1] = np.clip(cls_attn * 3 - 1, 0, 1) * 255
    heatmap[:, :, 2] = np.clip(cls_attn * 3 - 2, 0, 1) * 255

    alpha = 0.5
    overlay = (img_array * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    return Image.fromarray(overlay)


def predict_vit(image, habitat, substrate, month, vit_data):
    """Run ViT inference with metadata."""
    if not vit_data['loaded']:
        return None, None, None, vit_data.get('error', 'Model not loaded')

    model = vit_data['model']
    habitat_vocab = vit_data['habitat_vocab']
    substrate_vocab = vit_data['substrate_vocab']
    id_to_species = vit_data['id_to_species']

    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    h_id = habitat_vocab.get(habitat, len(habitat_vocab))
    s_id = substrate_vocab.get(substrate, len(substrate_vocab))
    m = int(month) if month != "unknown" else 6

    h_tensor = torch.tensor([h_id]).to(DEVICE)
    s_tensor = torch.tensor([s_id]).to(DEVICE)
    m_tensor = torch.tensor([m]).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor, h_tensor, s_tensor, m_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top5_probs, top5_indices = torch.topk(probs, k=5)
    top5 = {id_to_species.get(idx.item(), f"Class_{idx.item()}"): prob.item()
            for prob, idx in zip(top5_probs, top5_indices)}

    top_species = id_to_species.get(top5_indices[0].item(), f"Class_{top5_indices[0].item()}")
    top_conf = top5_probs[0].item() * 100
    top_text = f"{top_species} ({top_conf:.1f}%)"

    attn_img = get_attention_rollout(model, img_tensor, h_tensor, s_tensor, m_tensor, image)

    return top5, top_text, attn_img, None


def predict_cnn(image, cnn_data):
    """Run CNN inference."""
    if not cnn_data['loaded']:
        return None, None, cnn_data.get('error', 'Model not loaded')

    model = cnn_data['model']
    id_to_species = cnn_data['id_to_species']

    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top5_probs, top5_indices = torch.topk(probs, k=5)
    top5 = {id_to_species.get(idx.item(), f"Class {idx.item()}"): prob.item()
            for prob, idx in zip(top5_probs, top5_indices)}

    top_species = id_to_species.get(top5_indices[0].item(), "Unknown")
    top_conf = top5_probs[0].item() * 100
    top_text = f"{top_species} ({top_conf:.1f}%)"

    return top5, top_text, None


def predict_yolo(image, yolo_data):
    """Run YOLO inference for safety detection."""
    if not yolo_data['loaded']:
        return None, None, yolo_data.get('error', 'Model not loaded')

    model = yolo_data['model']

    results = model(image, verbose=False)[0]

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    probs = {"Edible": 0.0, "Poisonous": 0.0}

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Assuming class 0 = edible, class 1 = poisonous (adjust if needed)
            is_poisonous = cls == 1
            color = "red" if is_poisonous else "green"
            label = f"{'Poisonous' if is_poisonous else 'Edible'} {conf*100:.0f}%"

            if is_poisonous:
                probs["Poisonous"] = max(probs["Poisonous"], conf)
            else:
                probs["Edible"] = max(probs["Edible"], conf)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 20), label, fill=color)
    else:
        probs = {"No detection": 1.0}

    return img_draw, probs, None


# ============== UI ==============
st.title("🍄 Applied Mashroomatics")
st.markdown("**Порівняй YOLO · CNN · ViT** на фото грибів")

# Load models
vit_data = load_vit_model()
cnn_data = load_cnn_model()
yolo_data = load_yolo_model()

# Model status
with st.expander("📡 Статус моделей"):
    col1, col2, col3 = st.columns(3)
    with col1:
        if yolo_data['loaded']:
            st.success("✅ YOLO")
        else:
            st.warning(f"⏳ YOLO: {yolo_data.get('error', 'Not loaded')[:30]}...")
    with col2:
        if cnn_data['loaded']:
            st.success("✅ CNN")
        else:
            st.warning(f"⏳ CNN: {cnn_data.get('error', 'Not loaded')[:30]}...")
    with col3:
        if vit_data['loaded']:
            st.success("✅ ViT")
        else:
            st.error(f"❌ ViT: {vit_data.get('error', 'Not loaded')[:30]}...")

# Layout
col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("📸 Вхідні дані")

    uploaded_file = st.file_uploader("Завантажте фото гриба", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Завантажене фото", use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Метадані")

    habitat_options = ["unknown", "coniferous", "deciduous", "mixed", "park", "garden", "meadow", "bog"]
    habitat_labels = {
        "unknown": "-- Невідомо --", "coniferous": "🌲 Хвойний ліс",
        "deciduous": "🌳 Листяний ліс", "mixed": "🌲🌳 Мішаний ліс",
        "park": "🏛️ Парк", "garden": "🏡 Сад", "meadow": "🌾 Луг", "bog": "💧 Болото"
    }
    habitat = st.selectbox("Середовище", options=habitat_options,
                           format_func=lambda x: habitat_labels.get(x, x))

    substrate_options = ["unknown", "soil", "dead wood", "litter", "moss", "bark", "grass"]
    substrate_labels = {
        "unknown": "-- Невідомо --", "soil": "🟤 Ґрунт",
        "dead wood": "🪵 Мертва деревина", "litter": "🍂 Опале листя",
        "moss": "🌿 Мох", "bark": "🌳 Кора", "grass": "🌱 Трава"
    }
    substrate = st.selectbox("Субстрат", options=substrate_options,
                             format_func=lambda x: substrate_labels.get(x, x))

    month_options = ["unknown"] + [str(i) for i in range(1, 13)]
    month_labels = {
        "unknown": "-- Невідомо --",
        "1": "Січень", "2": "Лютий", "3": "Березень", "4": "Квітень",
        "5": "Травень", "6": "Червень", "7": "Липень", "8": "Серпень",
        "9": "Вересень", "10": "Жовтень", "11": "Листопад", "12": "Грудень"
    }
    month = st.selectbox("Місяць", options=month_options,
                         format_func=lambda x: month_labels.get(x, x))

    classify_btn = st.button("🔍 Класифікувати", type="primary", use_container_width=True)

with col_output:
    st.subheader("📊 Результати")

    tab_yolo, tab_cnn, tab_vit = st.tabs(["🎯 YOLO — безпека", "🔬 CNN — види", "🧠 ViT + metadata"])

    with tab_yolo:
        st.caption("Визначає: їстівний / отруйний + bounding box")
        yolo_col1, yolo_col2 = st.columns([2, 1])
        yolo_img_placeholder = yolo_col1.empty()
        yolo_label_placeholder = yolo_col2.empty()

    with tab_cnn:
        st.caption("EfficientNet-B3 · 180 видів")
        cnn_text_placeholder = st.empty()
        cnn_label_placeholder = st.empty()

    with tab_vit:
        st.caption("ViT + метадані (habitat, substrate, month)")
        vit_text_placeholder = st.empty()
        vit_col1, vit_col2 = st.columns([1, 1])
        vit_label_placeholder = vit_col1.empty()
        vit_attn_placeholder = vit_col2.empty()

# Run classification
if classify_btn and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # YOLO
    with st.spinner("🎯 YOLO..."):
        yolo_img, yolo_probs, yolo_error = predict_yolo(image, yolo_data)
        if yolo_error:
            with tab_yolo:
                st.warning(yolo_error)
        else:
            yolo_img_placeholder.image(yolo_img, caption="Детекція", use_container_width=True)
            with yolo_label_placeholder:
                for label, prob in yolo_probs.items():
                    color = "🔴" if "Poisonous" in label else "🟢" if "Edible" in label else "⚪"
                    st.metric(f"{color} {label}", f"{prob*100:.1f}%")

    # CNN
    with st.spinner("🔬 CNN..."):
        cnn_probs, cnn_text, cnn_error = predict_cnn(image, cnn_data)
        if cnn_error:
            with tab_cnn:
                st.warning(cnn_error)
        else:
            cnn_text_placeholder.success(f"🎯 **{cnn_text}**")
            with cnn_label_placeholder:
                for species, prob in cnn_probs.items():
                    st.progress(prob, text=f"{species[:35]} ({prob*100:.1f}%)")

    # ViT
    with st.spinner("🧠 ViT + metadata..."):
        vit_probs, vit_text, attn_img, vit_error = predict_vit(image, habitat, substrate, month, vit_data)
        if vit_error:
            with tab_vit:
                st.error(vit_error)
        else:
            vit_text_placeholder.success(f"🎯 **{vit_text}**")
            with vit_label_placeholder:
                st.markdown("**Top-5 види:**")
                for species, prob in vit_probs.items():
                    st.progress(prob, text=f"{species[:30]} ({prob*100:.1f}%)")
            if attn_img:
                vit_attn_placeholder.image(attn_img, caption="Attention Map", use_container_width=True)

st.markdown("---")
st.caption("Applied Mashroomatics · ViT + Metadata Fusion · DF20-Mini Dataset")