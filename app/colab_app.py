"""
Applied Mashroomatics - Streamlit App
Mushroom classification with YOLO, CNN, and ViT models.

=== LOCAL (Terminal) ===
    cd Applied_Mashroomatics
    streamlit run app/colab_app.py

=== COLAB ===
    !pip install streamlit
    !streamlit run app/colab_app.py --server.port 8501 --server.headless true &
    from google.colab import output
    output.serve_kernel_port_as_iframe(8501)
"""

import sys
import os
from pathlib import Path

# Detect environment and set paths
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

if IS_COLAB:
    PROJECT_ROOT = Path('/content/Applied_Mashroomatics')
    CHECKPOINT_BASE = Path('/content/drive/MyDrive')
else:
    # Local: project root is parent of app/
    PROJECT_ROOT = Path(__file__).parent.parent
    CHECKPOINT_BASE = PROJECT_ROOT / 'checkpoints'

# Add project root to path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import timm

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Applied Mashroomatics",
    page_icon="🍄",
    layout="wide"
)

# =============================================================================
# Constants
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Checkpoint paths (auto-detect environment)
if IS_COLAB:
    YOLO_CHECKPOINT = '/content/drive/MyDrive/mushroom_checkpoints/yolo_best.pt'
    CNN_CHECKPOINT = '/content/drive/MyDrive/mushroom_checkpoints/cnn_best.pth'
    VIT_CHECKPOINT = '/content/drive/MyDrive/mushroom_checkpoints/vit_best.pth'
else:
    # Local: all checkpoints in src/models/
    MODELS_DIR = PROJECT_ROOT / 'src' / 'models'
    YOLO_CHECKPOINT = str(MODELS_DIR / 'yolo_best.pt')
    CNN_CHECKPOINT = str(MODELS_DIR / 'cnn_best.pth')
    VIT_CHECKPOINT = str(MODELS_DIR / 'best_model.pth')

# Image preprocessing
PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dropdown options
HABITAT_CHOICES = ["unknown", "forest", "grassland", "urban", "wetland"]
SUBSTRATE_CHOICES = ["unknown", "wood", "soil", "leaves", "grass"]
MONTH_CHOICES = [str(i) for i in range(1, 13)]


# =============================================================================
# Model Loading (cached with st.cache_resource)
# =============================================================================

@st.cache_resource
def load_yolo():
    """Load YOLO model."""
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_CHECKPOINT)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_cnn():
    """Load CNN (EfficientNet-B3) model."""
    try:
        ckpt = torch.load(CNN_CHECKPOINT, map_location='cpu')

        # Extract species mapping
        if 'species_to_id' in ckpt:
            species_to_id = ckpt['species_to_id']
        elif 'config' in ckpt and 'species_to_id' in ckpt['config']:
            species_to_id = ckpt['config']['species_to_id']
        else:
            species_to_id = None

        num_classes = len(species_to_id) if species_to_id else 180

        # Create model
        model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)

        # Load weights
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)

        model = model.to(DEVICE).eval()
        id_to_species = {v: k for k, v in species_to_id.items()} if species_to_id else {}

        return model, id_to_species, None
    except Exception as e:
        return None, {}, str(e)


def load_species_from_metadata():
    """Load species_to_id from metadata CSV."""
    import pandas as pd

    # Try different possible paths
    csv_paths = [
        PROJECT_ROOT / 'data' / 'raw' / 'DF20M-metadata' / 'DF20M-train_metadata_PROD.csv',
        Path('/content/DF20M-train_metadata_PROD.csv'),  # Colab
        PROJECT_ROOT / 'DF20M-train_metadata_PROD.csv',
    ]

    for csv_path in csv_paths:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            species = sorted(df['species'].dropna().unique())
            return {sp: i for i, sp in enumerate(species)}

    return None


@st.cache_resource
def load_vit():
    """Load ViT with metadata model."""
    try:
        from src.models.mushroom_vit import MushroomViTWithMetadata

        ckpt = torch.load(VIT_CHECKPOINT, map_location='cpu')

        # Get species_to_id from checkpoint or metadata CSV
        if 'species_to_id' in ckpt:
            species_to_id = ckpt['species_to_id']
        else:
            species_to_id = load_species_from_metadata()
            if species_to_id is None:
                # Fallback: use numeric IDs
                num_classes = ckpt.get('num_classes', 179)
                species_to_id = {f"Species {i}": i for i in range(num_classes)}

        habitat_vocab = ckpt['habitat_vocab']
        substrate_vocab = ckpt['substrate_vocab']
        id_to_species = {v: k for k, v in species_to_id.items()}

        model = MushroomViTWithMetadata(
            num_classes=len(species_to_id),
            habitat_vocab=habitat_vocab,
            substrate_vocab=substrate_vocab,
            pretrained=False
        )

        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)

        model = model.to(DEVICE).eval()

        return model, id_to_species, habitat_vocab, substrate_vocab, None
    except Exception as e:
        return None, {}, {}, {}, str(e)


# =============================================================================
# Helper Functions
# =============================================================================

def get_metadata_id(value, vocab):
    """Convert metadata string to index. 'unknown' -> len(vocab) (UNK)."""
    if value == "unknown" or value not in vocab:
        return len(vocab)
    return vocab[value]


def create_attention_overlay(attn_map, original_image):
    """Create attention heatmap overlay on image."""
    import matplotlib.cm as cm

    # Resize original to 224x224
    img_resized = original_image.resize((224, 224))

    # Normalize attention
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample to 224x224
    attn_tensor = torch.tensor(attn_map).float()
    attn_up = F.interpolate(
        attn_tensor[None, None], size=(224, 224), mode='bilinear', align_corners=False
    )[0, 0].numpy()

    # Apply hot colormap
    cmap = cm.get_cmap('hot')
    attn_colored = (cmap(attn_up)[:, :, :3] * 255).astype(np.uint8)
    attn_pil = Image.fromarray(attn_colored)

    # Blend at alpha=0.5
    return Image.blend(img_resized.convert('RGB'), attn_pil, alpha=0.5)


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_yolo(image, model):
    """YOLO inference: edible/poisonous with bounding box."""
    if model is None:
        return image, None, "Model not loaded"

    try:
        results = model(image, verbose=False)
        result = results[0]

        # Draw on copy
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        if len(result.boxes) > 0:
            boxes = result.boxes
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            best_conf = float(boxes.conf[best_idx].cpu())
            best_class = int(boxes.cls[best_idx].cpu())
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()

            class_name = result.names.get(best_class, "Unknown")
            is_edible = "edible" in class_name.lower() or best_class == 0

            color = "green" if is_edible else "red"
            label = "Edible" if is_edible else "Poisonous"
            label_text = f"{label} {best_conf*100:.0f}%"

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label
            bbox = draw.textbbox((x1, y1 - 25), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 25), label_text, fill="white", font=font)

            probs = {label: best_conf, ("Poisonous" if is_edible else "Edible"): 1 - best_conf}
            return annotated, probs, None
        else:
            return annotated, {"Unknown": 1.0}, "No mushroom detected"

    except Exception as e:
        return image, None, str(e)


def predict_cnn(image, model, id_to_species):
    """CNN inference: top-5 species."""
    if model is None:
        return None, "Model not loaded"

    try:
        img_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]

        top5_probs, top5_idx = torch.topk(probs, 5)
        top5_probs = top5_probs.cpu().numpy()
        top5_idx = top5_idx.cpu().numpy()

        results = [(id_to_species.get(i, f"Species {i}"), float(p)) for i, p in zip(top5_idx, top5_probs)]
        return results, None

    except Exception as e:
        return None, str(e)


def predict_vit(image, habitat, substrate, month, model, id_to_species, habitat_vocab, substrate_vocab):
    """ViT + metadata inference: top-5 species + attention map."""
    if model is None:
        return None, None, "Model not loaded"

    try:
        img_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

        habitat_id = torch.tensor([get_metadata_id(habitat, habitat_vocab)], device=DEVICE)
        substrate_id = torch.tensor([get_metadata_id(substrate, substrate_vocab)], device=DEVICE)
        month_val = torch.tensor([int(month)], device=DEVICE)

        # Hook to capture attention
        saved_attn = [None]

        def attn_hook(module, inp, out):
            B, N, C = inp[0].shape
            qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            saved_attn[0] = attn.detach()

        hook = model.vit.blocks[-1].attn.register_forward_hook(attn_hook)

        with torch.no_grad():
            logits = model(img_tensor, habitat_id, substrate_id, month_val)
            probs = F.softmax(logits, dim=1)[0]

        hook.remove()

        # Top-5
        top5_probs, top5_idx = torch.topk(probs, 5)
        top5_probs = top5_probs.cpu().numpy()
        top5_idx = top5_idx.cpu().numpy()
        results = [(id_to_species.get(i, f"Species {i}"), float(p)) for i, p in zip(top5_idx, top5_probs)]

        # Attention map
        attn_image = None
        if saved_attn[0] is not None:
            # [batch, heads, tokens, tokens] -> CLS attention to patches
            # Skip CLS (0) and META (1), patches start at index 2
            attn = saved_attn[0]
            cls_attn = attn[0, :, 0, 2:].mean(0).cpu().numpy()  # (196,)
            attn_map = cls_attn.reshape(14, 14)
            attn_image = create_attention_overlay(attn_map, image)

        return results, attn_image, None

    except Exception as e:
        return None, None, str(e)


# =============================================================================
# Streamlit UI
# =============================================================================

st.title("🍄 Applied Mashroomatics")
st.markdown("Compare **YOLO**, **CNN**, and **ViT** on mushroom photos")

# Environment indicator
env_badge = "🌐 Colab" if IS_COLAB else "💻 Local"
st.caption(f"Running: {env_badge} | Device: {DEVICE}")

# Load models
with st.spinner("Loading models..."):
    yolo_model, yolo_err = load_yolo()
    cnn_model, cnn_id_to_species, cnn_err = load_cnn()
    vit_model, vit_id_to_species, habitat_vocab, substrate_vocab, vit_err = load_vit()

# Show model status
status_items = []
if yolo_model: status_items.append("YOLO")
if cnn_model: status_items.append("CNN")
if vit_model: status_items.append("ViT")

if status_items:
    st.success(f"Loaded: {', '.join(status_items)}")
else:
    st.error("No models loaded!")

with st.expander("Model Details", expanded=False):
    st.markdown(f"**YOLO**: {'Loaded' if yolo_model else yolo_err}")
    st.markdown(f"**CNN**: {'Loaded' if cnn_model else cnn_err}")
    st.markdown(f"**ViT**: {'Loaded' if vit_model else vit_err}")

st.markdown("---")

# Layout: Input | Output
col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("📸 Input")

    uploaded = st.file_uploader("Upload mushroom photo", type=["jpg", "jpeg", "png"])

    st.markdown("**Metadata** (optional)")
    habitat = st.selectbox("Habitat", HABITAT_CHOICES, index=0)
    substrate = st.selectbox("Substrate", SUBSTRATE_CHOICES, index=0)
    month = st.selectbox("Month", MONTH_CHOICES, index=5)

    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)

with col_output:
    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')

        if classify_btn:
            # Build tabs dynamically based on loaded models
            available_tabs = []
            tab_names = []

            if yolo_model is not None:
                tab_names.append("🎯 YOLO - Safety")
                available_tabs.append("yolo")
            if cnn_model is not None:
                tab_names.append("🔬 CNN - Species")
                available_tabs.append("cnn")
            if vit_model is not None:
                tab_names.append("🧠 ViT + Metadata")
                available_tabs.append("vit")

            if not tab_names:
                st.error("No models loaded!")
            else:
                tabs = st.tabs(tab_names)
                tab_idx = 0

                # === YOLO ===
                if "yolo" in available_tabs:
                    with tabs[tab_idx]:
                        with st.spinner("Running YOLO..."):
                            yolo_img, yolo_probs, yolo_run_err = predict_yolo(image, yolo_model)

                        st.image(yolo_img, caption="Detection result", use_container_width=True)

                        if yolo_run_err:
                            st.error(yolo_run_err)
                        elif yolo_probs:
                            st.subheader("Safety Prediction")
                            for label, prob in sorted(yolo_probs.items(), key=lambda x: -x[1]):
                                color = "🟢" if label == "Edible" else "🔴"
                                st.markdown(f"{color} **{label}**: {prob*100:.1f}%")
                                st.progress(prob)
                    tab_idx += 1

                # === CNN ===
                if "cnn" in available_tabs:
                    with tabs[tab_idx]:
                        with st.spinner("Running CNN..."):
                            cnn_results, cnn_run_err = predict_cnn(image, cnn_model, cnn_id_to_species)

                        if cnn_run_err:
                            st.error(cnn_run_err)
                        elif cnn_results:
                            top_species, top_prob = cnn_results[0]
                            st.success(f"**{top_species}** ({top_prob*100:.0f}%)")

                            st.subheader("Top-5 Species")
                            for species, prob in cnn_results:
                                st.markdown(f"**{species}**: {prob*100:.1f}%")
                                st.progress(prob)
                    tab_idx += 1

                # === ViT ===
                if "vit" in available_tabs:
                    with tabs[tab_idx]:
                        with st.spinner("Running ViT..."):
                            vit_results, attn_img, vit_run_err = predict_vit(
                                image, habitat, substrate, month,
                                vit_model, vit_id_to_species, habitat_vocab, substrate_vocab
                            )

                        if vit_run_err:
                            st.error(vit_run_err)
                        elif vit_results:
                            col_pred, col_attn = st.columns(2)

                            with col_pred:
                                top_species, top_prob = vit_results[0]
                                st.success(f"**{top_species}** ({top_prob*100:.0f}%)")

                                st.subheader("Top-5 Species (with metadata)")
                                for species, prob in vit_results:
                                    st.markdown(f"**{species}**: {prob*100:.1f}%")
                                    st.progress(prob)

                            with col_attn:
                                st.subheader("Attention Map")
                                if attn_img:
                                    st.image(attn_img, use_container_width=True)
                                else:
                                    st.info("Attention map not available")
        else:
            st.image(image, caption="Uploaded image", use_container_width=True)
            st.info("Click **Classify** to run predictions")
    else:
        st.info("👈 Upload an image to get started")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "🍄 Applied Mashroomatics | DF20-Mini Dataset | 180 Species"
    "</p>",
    unsafe_allow_html=True
)