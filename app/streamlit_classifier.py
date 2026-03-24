"""
Applied Mashroomatics - Mushroom Classification App
ViT + Metadata Fusion
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Applied Mashroomatics",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a2e;
        color: #eaeaea;
    }
    .main-header {
        background: linear-gradient(90deg, #16213e 0%, #1a1a2e 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .input-section {
        background-color: #16213e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .output-section {
        background-color: #16213e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .species-item {
        background-color: #0f3460;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.3rem 0;
    }
    .top-prediction {
        background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #e94560;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #ff6b6b;
    }
    .stSelectbox > div > div {
        background-color: #0f3460;
        border: 1px solid #16213e;
    }
    .uploadedFile {
        background-color: #0f3460;
    }
    h1, h2, h3 {
        color: #eaeaea;
    }
    .stProgress > div > div > div > div {
        background-color: #e94560;
    }
</style>
""", unsafe_allow_html=True)

# ============== Configuration ==============
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model paths
VIT_PATH = 'src/models/best_model.pth'
if os.path.exists('/content'):
    VIT_PATH = '/content/drive/MyDrive/mushroom_checkpoints/best_model.pth'

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_species_mapping():
    """Load species mapping from CSV (same as training)."""
    # Try different CSV paths
    csv_paths = [
        'data/raw/DF20M-metadata/DF20M-train_metadata_PROD.csv',
        '/content/drive/MyDrive/raw/DF20M-metadata/DF20M-train_metadata_PROD.csv',
        '../data/raw/DF20M-metadata/DF20M-train_metadata_PROD.csv'
    ]

    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['species'])
            species_list = sorted(df['species'].unique())
            species_to_id = {sp: i for i, sp in enumerate(species_list)}
            id_to_species = {i: sp for sp, i in species_to_id.items()}
            return species_to_id, id_to_species

    return {}, {}


@st.cache_resource
def load_vit_model():
    """Load ViT model with metadata fusion."""
    try:
        from src.models.mushroom_vit import MushroomViTWithMetadata

        ckpt = torch.load(VIT_PATH, map_location='cpu', weights_only=False)

        # Try checkpoint first, then CSV
        species_to_id = ckpt.get('species_to_id', {})
        if not species_to_id:
            id_to_species = ckpt.get('id_to_species', {})
            if id_to_species:
                species_to_id = {v: k for k, v in id_to_species.items()}

        # If still empty, load from CSV
        if not species_to_id:
            species_to_id, _ = load_species_mapping()

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

    # Hot colormap
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
    top5 = [(id_to_species.get(idx.item(), f"Species {idx.item()}"), prob.item())
            for prob, idx in zip(top5_probs, top5_indices)]

    top_species, top_conf = top5[0]

    attn_img = get_attention_rollout(model, img_tensor, h_tensor, s_tensor, m_tensor, image)

    return top5, top_species, top_conf, attn_img, None


# ============== UI ==============

# Load model
vit_data = load_vit_model()

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("🍄 Applied Mashroomatics")
    st.caption("ViT + Metadata | DF20-Mini Dataset | 180 Species")
with col_status:
    if vit_data['loaded']:
        st.success("✅ Model ready")
    else:
        st.error(f"❌ {vit_data.get('error', 'Error')[:50]}")
st.markdown('</div>', unsafe_allow_html=True)

# Main layout
col_input, col_output = st.columns([1, 2])

with col_input:
    st.markdown("### 📥 Input")

    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["jpg", "jpeg", "png"],
        help="Limit 200MB per file • JPG, JPEG, PNG"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.caption(f"📷 {uploaded_file.name}")

    st.markdown("---")
    st.markdown("### 📋 Metadata (optional)")

    habitat_options = {
        "unknown": "-- Select --",
        "coniferous": "coniferous",
        "deciduous": "deciduous",
        "mixed": "mixed",
        "park": "park",
        "garden": "garden",
        "meadow": "meadow",
        "bog": "bog"
    }
    habitat = st.selectbox("Habitat", options=list(habitat_options.keys()),
                           format_func=lambda x: habitat_options[x])

    substrate_options = {
        "unknown": "-- Select --",
        "soil": "soil",
        "dead wood": "dead wood",
        "litter": "litter",
        "moss": "moss",
        "bark": "bark",
        "grass": "grass"
    }
    substrate = st.selectbox("Substrate", options=list(substrate_options.keys()),
                             format_func=lambda x: substrate_options[x])

    month_options = {"unknown": "-- Select --"}
    month_options.update({str(i): str(i) for i in range(1, 13)})
    month = st.selectbox("Month", options=list(month_options.keys()),
                         format_func=lambda x: month_options[x])

    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

with col_output:
    if uploaded_file and classify_btn:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Classifying..."):
            result = predict_vit(image, habitat, substrate, month, vit_data)

            if result[4]:  # error
                st.error(result[4])
            else:
                top5, top_species, top_conf, attn_img, _ = result

                # Top prediction banner
                st.markdown(f"""
                <div class="top-prediction">
                    🎯 {top_species} ({top_conf*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)

                # Two columns: Species list + Attention map
                col_species, col_attn = st.columns([1, 1])

                with col_species:
                    st.markdown("### Top-5 Species (with metadata)")
                    for species, prob in top5:
                        st.markdown(f"""
                        <div class="species-item">
                            {species}: {prob*100:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(prob)

                with col_attn:
                    st.markdown("### Attention Map")
                    if attn_img:
                        st.image(attn_img, use_container_width=True)
                        st.caption("Yellow = high attention")

    elif not uploaded_file:
        st.info("👈 Upload a mushroom photo to get started")

# Footer
st.markdown("---")
st.caption("🍄 Applied Mashroomatics | DF20-Mini Dataset | 180 Species")