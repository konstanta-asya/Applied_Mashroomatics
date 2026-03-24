"""
Applied Mashroomatics - Mushroom Classification App
Compare YOLO · CNN · ViT on mushroom photos
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torchvision import transforms
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Applied Mashroomatics",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
  /* Hide Streamlit auto-generated pages navigation */
  [data-testid="stSidebarNav"] { display: none; }
  .main { background-color: #ffffff; }
  .block-container { padding: 2rem 3rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
      padding: 8px 20px;
      border-radius: 8px 8px 0 0;
      font-weight: 500;
  }
  .verdict-edible {
      background: #E1F5EE; color: #0F6E56;
      padding: 16px 24px; border-radius: 8px;
      font-size: 20px; font-weight: 600;
      text-align: center;
  }
  .verdict-poisonous {
      background: #FCEBEB; color: #A32D2D;
      padding: 16px 24px; border-radius: 8px;
      font-size: 20px; font-weight: 600;
      text-align: center;
  }
  div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ============== Configuration ==============
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model paths
VIT_PATH = 'src/models/best_model.pth'
CNN_PATH = 'src/models/cnn_best.pth'
YOLO_PATH = 'src/models/yolo_best.pt'

# Colab paths
if os.path.exists('/content'):
    VIT_PATH = '/content/drive/MyDrive/mushroom_checkpoints/best_model.pth'
    CNN_PATH = '/content/drive/MyDrive/mushroom_checkpoints/cnn_best.pth'
    YOLO_PATH = '/content/drive/MyDrive/yolo_mushroom/best.pt'

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============== Model Loading ==============
def load_species_mapping():
    """Load species mapping from CSV."""
    csv_paths = [
        'data/raw/DF20M-metadata/DF20M-train_metadata_PROD.csv',
        '/content/drive/MyDrive/raw/DF20M-metadata/DF20M-train_metadata_PROD.csv',
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
def load_models():
    """Load all models (cached)."""
    models = {
        'yolo': None,
        'cnn': None,
        'vit': None,
        'species_to_id': {},
        'id_to_species': {},
        'habitat_vocab': {},
        'substrate_vocab': {},
    }

    # Load species mapping from CSV
    species_to_id, id_to_species = load_species_mapping()
    models['species_to_id'] = species_to_id
    models['id_to_species'] = id_to_species

    # Load ViT
    try:
        from src.models.mushroom_vit import MushroomViTWithMetadata

        if os.path.exists(VIT_PATH):
            ckpt = torch.load(VIT_PATH, map_location='cpu', weights_only=False)

            habitat_vocab = ckpt.get('habitat_vocab', {})
            substrate_vocab = ckpt.get('substrate_vocab', {})
            num_classes = ckpt.get('num_classes', len(species_to_id) if species_to_id else 179)

            if ckpt.get('species_to_id'):
                models['species_to_id'] = ckpt['species_to_id']
                models['id_to_species'] = {v: k for k, v in ckpt['species_to_id'].items()}

            model = MushroomViTWithMetadata(
                num_classes=num_classes,
                habitat_vocab=habitat_vocab,
                substrate_vocab=substrate_vocab,
                pretrained=False
            )
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(DEVICE).eval()

            models['vit'] = model
            models['habitat_vocab'] = habitat_vocab
            models['substrate_vocab'] = substrate_vocab
    except Exception as e:
        pass

    # Load CNN
    try:
        if os.path.exists(CNN_PATH):
            import timm
            ckpt = torch.load(CNN_PATH, map_location='cpu', weights_only=False)
            num_classes = ckpt.get('num_classes', 179)
            model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(DEVICE).eval()
            models['cnn'] = model
    except Exception as e:
        pass

    # Load YOLO
    try:
        if os.path.exists(YOLO_PATH):
            from ultralytics import YOLO
            models['yolo'] = YOLO(YOLO_PATH)
    except Exception as e:
        pass

    return models


# ============== Inference Functions ==============
def draw_yolo_box(image, boxes, label, edible):
    """Draw bounding boxes on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    color = "#1D9E75" if edible else "#E24B4A"

    if boxes and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1, y1-20, x1+len(label)*8, y1], fill=color)
            draw.text((x1+4, y1-18), label, fill="white")
    else:
        w, h = img.size
        draw.rectangle([0, 0, w-1, h-1], outline=color, width=4)
        draw.text((10, 10), label, fill=color)

    return img


def safe_predict_yolo(image, model):
    """YOLO prediction with error handling."""
    if model is None:
        return {'img': image, 'edible': None, 'conf': 0.0, 'error': 'YOLO not loaded'}

    try:
        results = model(image, verbose=False)[0]
        boxes = []
        edible = True
        conf = 0.0

        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([x1, y1, x2, y2])
                cls = int(box.cls[0].item())
                conf = max(conf, box.conf[0].item())
                if cls == 1:
                    edible = False

        label = f"{'Edible' if edible else 'Poisonous'} {conf*100:.0f}%"
        img = draw_yolo_box(image, boxes, label, edible)

        return {'img': img, 'edible': edible, 'conf': conf, 'error': None}
    except Exception as e:
        return {'img': image, 'edible': None, 'conf': 0.0, 'error': str(e)}


def safe_predict_cnn(image, model, id_to_species):
    """CNN prediction with error handling."""
    if model is None:
        return {'top5': [], 'top1': 'Not loaded', 'conf': 0.0, 'error': 'CNN not loaded'}

    try:
        img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top5_probs, top5_indices = torch.topk(probs, k=5)
        top5 = [(id_to_species.get(idx.item(), f"Species {idx.item()}"), prob.item())
                for prob, idx in zip(top5_probs, top5_indices)]

        top1, conf = top5[0]
        return {'top5': top5, 'top1': top1, 'conf': conf, 'error': None}
    except Exception as e:
        return {'top5': [], 'top1': 'Error', 'conf': 0.0, 'error': str(e)}


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


def safe_predict_vit(image, model, vocabs, metadata):
    """ViT prediction with error handling."""
    if model is None:
        return {'top5': [], 'top1': 'Not loaded', 'conf': 0.0,
                'attn_img': Image.new('RGB', (224, 224), 'white'), 'error': 'ViT not loaded'}

    try:
        habitat_vocab = vocabs['habitat_vocab']
        substrate_vocab = vocabs['substrate_vocab']
        id_to_species = vocabs['id_to_species']

        img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        h_id = habitat_vocab.get(metadata['habitat'], len(habitat_vocab))
        s_id = substrate_vocab.get(metadata['substrate'], len(substrate_vocab))
        m = int(metadata['month']) if metadata['month'] != "unknown" else 6

        h_tensor = torch.tensor([h_id]).to(DEVICE)
        s_tensor = torch.tensor([s_id]).to(DEVICE)
        m_tensor = torch.tensor([m]).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor, h_tensor, s_tensor, m_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top5_probs, top5_indices = torch.topk(probs, k=5)
        top5 = [(id_to_species.get(idx.item(), f"Species {idx.item()}"), prob.item())
                for prob, idx in zip(top5_probs, top5_indices)]

        top1, conf = top5[0]
        attn_img = get_attention_rollout(model, img_tensor, h_tensor, s_tensor, m_tensor, image)

        return {'top5': top5, 'top1': top1, 'conf': conf, 'attn_img': attn_img, 'error': None}
    except Exception as e:
        return {'top5': [], 'top1': 'Error', 'conf': 0.0,
                'attn_img': Image.new('RGB', (224, 224), 'white'), 'error': str(e)}


def run_all_models(image, habitat, substrate, month, models):
    """Run all models and return results."""
    yolo_result = safe_predict_yolo(image, models['yolo'])
    cnn_result = safe_predict_cnn(image, models['cnn'], models['id_to_species'])
    metadata = {'habitat': habitat, 'substrate': substrate, 'month': month}
    vit_result = safe_predict_vit(image, models['vit'], models, metadata)

    return {
        'yolo': yolo_result,
        'cnn': cnn_result,
        'vit': vit_result,
        'metadata': metadata
    }


# ============== Load Models ==============
models = load_models()

# ============== Sidebar (navigation only) ==============
with st.sidebar:
    st.page_link("streamlit_app.py", label="Класифікатор", icon="🔬")
    st.page_link("pages/2_📖_Про_проєкт.py", label="Про проєкт", icon="📖")
    st.page_link("pages/3_🍄_Види_грибів.py", label="Види грибів", icon="🍄")
    st.page_link("pages/4_👥_Команда.py", label="Команда", icon="👥")

# ============== Main Page ==============
st.title("🍄 Applied Mashroomatics")
st.caption("Classify mushrooms using YOLO, CNN & Vision Transformer")

# Two-column layout: Input left, Results right
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 📸 Input")
    st.caption("Upload mushroom photo")
    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # Store image on upload
    if uploaded is not None:
        st.session_state['image'] = Image.open(uploaded).convert('RGB')

    # Show preview
    if 'image' in st.session_state:
        st.image(st.session_state['image'], use_container_width=True)

    st.markdown("**Metadata** <span style='color:gray;font-size:12px'>(optional)</span>",
                unsafe_allow_html=True)

    habitat = st.selectbox("Habitat",
        ["unknown", "coniferous", "deciduous", "mixed", "park", "garden", "meadow", "bog"],
        index=0)

    substrate = st.selectbox("Substrate",
        ["unknown", "soil", "dead wood", "litter", "moss", "bark", "grass"],
        index=0)

    month = st.selectbox("Month",
        [str(i) for i in range(1, 13)],
        index=5)

    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

    # Run inference on button click
    if classify_btn:
        if 'image' not in st.session_state:
            st.warning("Please upload a photo first")
            st.stop()

        image = st.session_state['image']
        with st.spinner("Running models..."):
            st.session_state['results'] = run_all_models(image, habitat, substrate, month, models)

with col_right:
    st.markdown("### 📊 Results")
    st.caption("Compare model predictions")

    # Results section: three tabs
    tab_yolo, tab_cnn, tab_vit = st.tabs([
        "🛡️ YOLO — Safety",
        "🧠 CNN — Species",
        "🔬 ViT + Metadata"
    ])

    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        yolo_result = results['yolo']
        cnn_result = results['cnn']
        vit_result = results['vit']

        # YOLO tab
        with tab_yolo:
            col_det, col_safety = st.columns([1, 1])

            with col_det:
                st.markdown("**Detection**")
                st.image(yolo_result['img'], use_container_width=True)

            with col_safety:
                st.markdown("**Safety prediction**")
                if yolo_result['error']:
                    st.warning(yolo_result['error'])
                elif yolo_result['edible'] is not None:
                    if yolo_result['edible']:
                        st.markdown(
                            '<div class="verdict-edible">✓ Edible</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="verdict-poisonous">✗ Poisonous</div>',
                            unsafe_allow_html=True
                        )
                    st.write(f"Confidence: {yolo_result['conf']*100:.1f}%")
                    st.progress(int(yolo_result['conf'] * 100))
                st.caption("⚠️ Always verify with an expert before consuming.")

        # CNN tab
        with tab_cnn:
            col_pred, col_bars = st.columns([1, 1])

            with col_pred:
                st.markdown("**Top prediction**")
                if cnn_result['error']:
                    st.warning(cnn_result['error'])
                else:
                    st.markdown(f"*{cnn_result['top1']}*")
                    st.markdown(f"**{cnn_result['conf']*100:.1f}%**")
                    st.divider()
                    st.caption("Genus")
                    genus = cnn_result['top1'].split()[0] if cnn_result['top1'] else "Unknown"
                    st.write(genus)

            with col_bars:
                st.markdown("**Top-5 species**")
                if not cnn_result['error']:
                    for species, conf in cnn_result['top5']:
                        st.markdown(f"**{species}**: {conf*100:.1f}%")
                        st.progress(float(conf))

        # ViT tab
        with tab_vit:
            col_pred, col_attn = st.columns([1, 1])

            with col_pred:
                if vit_result['error']:
                    st.warning(vit_result['error'])
                else:
                    # Green badge
                    st.markdown(f"""
                    <div style="
                        background:#1a5c2e; color:#4ade80;
                        padding:12px 20px; border-radius:8px;
                        font-size:16px; font-weight:600;
                        margin-bottom:16px;
                    ">{vit_result['top1']} ({vit_result['conf']*100:.0f}%)</div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Top-5 Species (with metadata)**")
                    for species, conf in vit_result['top5']:
                        st.markdown(f"**{species}**: {conf*100:.1f}%")
                        st.progress(float(conf))

            with col_attn:
                st.markdown("**Attention Map**")
                if not vit_result['error']:
                    st.image(vit_result['attn_img'], use_container_width=True)

    else:
        with tab_yolo:
            st.info("Upload a photo and click Classify to see results")
        with tab_cnn:
            st.info("Upload a photo and click Classify to see results")
        with tab_vit:
            st.info("Upload a photo and click Classify to see results")

# Footer
st.divider()
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:12px'>"
    "Applied Mashroomatics · DF20-Mini · 180 species · "
    "YOLOv8 · EfficientNet-B3 · ViT-Base/16 · PyTorch + Streamlit"
    "</p>",
    unsafe_allow_html=True
)