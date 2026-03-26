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

# Model directory
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive file IDs
GDRIVE_IDS = {
    'vit': '1wYzDSxclkMJh_t1nd565r9ULMSsfGTur',   # best_model.pth
    'cnn': '1SwMehtoI6n5UGF707hqKbiq4Mio8tXye',   # cnn_model.pth
    'yolo': '1zscY8T7BGMRXPRMMmCQZpCDeGbwusige',  # yolo_model.pt
}

# Model paths
VIT_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
CNN_PATH = os.path.join(MODEL_DIR, 'cnn_model.pth')
YOLO_PATH = os.path.join(MODEL_DIR, 'yolo_model.pt')

# Local development paths (if models exist locally)
if os.path.exists('src/models/best_model.pth'):
    VIT_PATH = 'src/models/best_model.pth'
    CNN_PATH = 'src/models/cnn_model.pth'
    YOLO_PATH = 'src/models/yolo_model.pt'

# Colab paths
if os.path.exists('/content'):
    VIT_PATH = '/content/drive/MyDrive/mushroom_checkpoints/best_model.pth'
    CNN_PATH = '/content/drive/MyDrive/mushroom_checkpoints/cnn_best.pth'
    YOLO_PATH = '/content/drive/MyDrive/yolo_mushroom/yolo_model.pt'


def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive if not exists."""
    if os.path.exists(output_path):
        return True
    if file_id.startswith('YOUR_'):
        return False  # Placeholder ID
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Download error: {e}")
        return False

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

    # Try to download models from Google Drive if not exists
    if not os.path.exists(VIT_PATH):
        download_from_gdrive(GDRIVE_IDS['vit'], VIT_PATH)
    if not os.path.exists(CNN_PATH):
        download_from_gdrive(GDRIVE_IDS['cnn'], CNN_PATH)
    if not os.path.exists(YOLO_PATH):
        download_from_gdrive(GDRIVE_IDS['yolo'], YOLO_PATH)

    # Load species mapping from CSV
    species_to_id, id_to_species = load_species_mapping()
    models['species_to_id'] = species_to_id
    models['id_to_species'] = id_to_species

    # Load ViT
    try:
        from src.models.mushroom_vit import MushroomViTWithMetadata

        if os.path.exists(VIT_PATH):
            print(f"Loading ViT from {VIT_PATH}...")
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
            print("ViT loaded successfully!")
        else:
            print(f"ViT path not found: {VIT_PATH}")
    except Exception as e:
        print(f"ViT load error: {e}")

    # Load CNN
    try:
        if os.path.exists(CNN_PATH):
            import timm
            ckpt = torch.load(CNN_PATH, map_location='cpu', weights_only=False)

            # Check if it's a wrapped checkpoint or raw state_dict
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
                num_classes = ckpt.get('num_classes', 179)
            else:
                # Raw state_dict - get num_classes from classifier layer
                state_dict = ckpt
                num_classes = state_dict['classifier.weight'].shape[0]

            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
            model.load_state_dict(state_dict)
            model = model.to(DEVICE).eval()
            models['cnn'] = model
            print(f"CNN loaded successfully! Classes: {num_classes}")
        else:
            print(f"CNN path not found: {CNN_PATH}")
    except Exception as e:
        print(f"CNN load error: {e}")

    # Load YOLO (YOLOv5 model) - skip on cloud to reduce startup time
    # YOLO will be loaded lazily on first use
    if os.path.exists(YOLO_PATH):
        models['yolo_path'] = YOLO_PATH
        print(f"YOLO model found at {YOLO_PATH} (will load on first use)")
    else:
        print(f"YOLO path not found: {YOLO_PATH}")

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


@st.cache_resource
def load_yolo_model(path):
    """Load YOLO model lazily."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
        model.conf = 0.25
        return model
    except Exception as e:
        print(f"YOLO load error: {e}")
        return None


def safe_predict_yolo(image, model_or_path):
    """YOLO prediction with error handling (YOLOv5)."""
    # Lazy load YOLO model
    if isinstance(model_or_path, str):
        model = load_yolo_model(model_or_path)
    else:
        model = model_or_path

    if model is None:
        return {'img': image, 'edible': None, 'conf': 0.0, 'error': 'YOLO not loaded'}

    try:
        # YOLOv5 inference
        results = model(image)
        boxes = []
        edible = True
        conf = 0.0

        # YOLOv5 results format
        detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]
        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, det_conf, cls = det.tolist()
                boxes.append([x1, y1, x2, y2])
                conf = max(conf, det_conf)
                if int(cls) == 1:  # poisonous class
                    edible = False

        label = f"{'Edible' if edible else 'Poisonous'} {conf*100:.0f}%"
        img = draw_yolo_box(image, boxes, label, edible)

        return {'img': img, 'edible': edible, 'conf': conf, 'error': None}
    except Exception as e:
        return {'img': image, 'edible': None, 'conf': 0.0, 'error': str(e)}


def get_gradcam(model, img_tensor, original_image):
    """Generate Grad-CAM visualization for CNN."""
    model.eval()

    # Hook to capture gradients and activations
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on the last conv layer (conv_head for EfficientNet)
    target_layer = model.conv_head
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        output = model(img_tensor)
        pred_class = output.argmax(dim=1)

        # Backward pass
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1
        output.backward(gradient=one_hot)

        # Get gradients and activations
        grads = gradients[0]  # [B, C, H, W]
        acts = activations[0]  # [B, C, H, W]

        # Global average pooling of gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to image size
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

    finally:
        forward_handle.remove()
        backward_handle.remove()

    # Create heatmap overlay
    img_resized = original_image.resize((224, 224))
    img_array = np.array(img_resized)

    # Hot colormap (black -> red -> yellow -> white)
    heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
    heatmap[:, :, 0] = np.clip(cam * 3, 0, 1) * 255
    heatmap[:, :, 1] = np.clip(cam * 3 - 1, 0, 1) * 255
    heatmap[:, :, 2] = np.clip(cam * 3 - 2, 0, 1) * 255

    alpha = 0.5
    overlay = (img_array * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    return Image.fromarray(overlay)


def safe_predict_cnn(image, model, id_to_species):
    """CNN prediction with error handling."""
    if model is None:
        return {'top5': [], 'top1': 'Not loaded', 'conf': 0.0,
                'gradcam_img': None, 'error': 'CNN not loaded'}

    try:
        img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top5_probs, top5_indices = torch.topk(probs, k=5)
        top5 = [(id_to_species.get(idx.item(), f"Species {idx.item()}"), prob.item())
                for prob, idx in zip(top5_probs, top5_indices)]

        top1, conf = top5[0]

        # Generate Grad-CAM (need gradients enabled)
        img_tensor_grad = preprocess(image).unsqueeze(0).to(DEVICE)
        img_tensor_grad.requires_grad = True
        gradcam_img = get_gradcam(model, img_tensor_grad, image)

        return {'top5': top5, 'top1': top1, 'conf': conf,
                'gradcam_img': gradcam_img, 'error': None}
    except Exception as e:
        return {'top5': [], 'top1': 'Error', 'conf': 0.0,
                'gradcam_img': None, 'error': str(e)}


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


def combine_predictions(results_list):
    """Combine predictions from multiple views by averaging confidences."""
    if len(results_list) == 1:
        return results_list[0]

    # Aggregate species scores
    species_scores = {}
    for result in results_list:
        for species, conf in result.get('top5', []):
            if species not in species_scores:
                species_scores[species] = []
            species_scores[species].append(conf)

    # Average scores
    combined = []
    for species, scores in species_scores.items():
        avg_conf = sum(scores) / len(results_list)
        combined.append((species, avg_conf))

    # Sort by confidence
    combined.sort(key=lambda x: x[1], reverse=True)
    top5 = combined[:5]

    if top5:
        return {
            'top5': top5,
            'top1': top5[0][0],
            'conf': top5[0][1],
            'error': None,
            'num_views': len(results_list)
        }
    return results_list[0] if results_list else {'top5': [], 'top1': 'Error', 'conf': 0.0, 'error': 'No results'}


def run_all_models_multiview(images, habitat, substrate, month, models):
    """Run all models on multiple views and combine results."""
    metadata = {'habitat': habitat, 'substrate': substrate, 'month': month}

    # Collect results from all views
    all_yolo = []
    all_cnn = []
    all_vit = []
    yolo_imgs = []
    gradcam_imgs = []
    attn_imgs = []

    for img in images:
        # YOLO (use path for lazy loading)
        yolo_model = models.get('yolo') or models.get('yolo_path')
        yolo_result = safe_predict_yolo(img, yolo_model)
        all_yolo.append(yolo_result)
        yolo_imgs.append(yolo_result['img'])

        # CNN
        cnn_result = safe_predict_cnn(img, models['cnn'], models['id_to_species'])
        all_cnn.append(cnn_result)
        if cnn_result.get('gradcam_img'):
            gradcam_imgs.append(cnn_result['gradcam_img'])

        # ViT
        vit_result = safe_predict_vit(img, models['vit'], models, metadata)
        all_vit.append(vit_result)
        if vit_result.get('attn_img'):
            attn_imgs.append(vit_result['attn_img'])

    # Combine CNN predictions
    cnn_combined = combine_predictions(all_cnn)
    cnn_combined['gradcam_imgs'] = gradcam_imgs

    # Combine ViT predictions
    vit_combined = combine_predictions(all_vit)
    vit_combined['attn_imgs'] = attn_imgs

    # YOLO: use most confident or most "dangerous" prediction
    best_yolo = all_yolo[0]
    for yolo_result in all_yolo:
        if yolo_result.get('edible') == False:  # Poisonous takes priority
            best_yolo = yolo_result
            break
        if yolo_result.get('conf', 0) > best_yolo.get('conf', 0):
            best_yolo = yolo_result
    best_yolo['all_imgs'] = yolo_imgs

    return {
        'yolo': best_yolo,
        'cnn': cnn_combined,
        'vit': vit_combined,
        'metadata': metadata,
        'num_views': len(images)
    }


# ============== Load Models ==============
models = load_models()

# ============== Sidebar (navigation only) ==============
with st.sidebar:
    st.page_link("streamlit_app.py", label="Classifier", icon="🔬")
    st.page_link("pages/about.py", label="About", icon="📖")
    st.page_link("pages/species.py", label="Species", icon="🍄")
    st.page_link("pages/team.py", label="Team", icon="👥")

# ============== Main Page ==============
st.title("🍄 Applied Mashroomatics")
st.caption("Classify mushrooms using YOLO, CNN & Vision Transformer")

# Two-column layout: Input left, Results right
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 📸 Multi-View Input")
    st.caption("Upload 2-3 photos from different angles for better accuracy")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Store images on upload
    if uploaded_files:
        st.session_state['images'] = [Image.open(f).convert('RGB') for f in uploaded_files]

    # Show preview grid
    if 'images' in st.session_state and st.session_state['images']:
        images = st.session_state['images']
        st.markdown(f"**{len(images)} photo(s) uploaded**")

        # Display in grid
        cols = st.columns(min(len(images), 3))
        for i, img in enumerate(images[:3]):
            with cols[i % 3]:
                st.image(img, caption=f"View {i+1}", use_container_width=True)

        if len(images) == 1:
            st.info("Tip: Add more photos for better accuracy")

    st.markdown("**Metadata** <span style='color:gray;font-size:12px'>(optional, improves ViT)</span>",
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
        if 'images' not in st.session_state or not st.session_state['images']:
            st.warning("Please upload at least one photo")
            st.stop()

        images = st.session_state['images']
        with st.spinner(f"Analyzing {len(images)} view(s)..."):
            st.session_state['results'] = run_all_models_multiview(images, habitat, substrate, month, models)

with col_right:
    st.markdown("### 📊 Results")
    st.caption("Compare model predictions")

    # Results section: three tabs
    tab_yolo, tab_cnn, tab_vit = st.tabs([
        "🛡️ YOLO",
        "🧠 CNN",
        "🔬 ViT"
    ])

    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        yolo_result = results['yolo']
        cnn_result = results['cnn']
        vit_result = results['vit']
        num_views = results.get('num_views', 1)

        if num_views > 1:
            st.success(f"Combined predictions from {num_views} views")

        # YOLO tab
        with tab_yolo:
            # Show all detection images
            yolo_imgs = yolo_result.get('all_imgs', [yolo_result['img']])
            if len(yolo_imgs) > 1:
                st.markdown("**Detections (all views)**")
                img_cols = st.columns(len(yolo_imgs))
                for i, img in enumerate(yolo_imgs):
                    with img_cols[i]:
                        st.image(img, caption=f"View {i+1}", use_container_width=True)
            else:
                st.markdown("**Detection**")
                st.image(yolo_result['img'], use_container_width=True)

            st.divider()

            st.markdown("**Safety Prediction**")
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
            col_pred, col_cam = st.columns([1, 1])

            with col_pred:
                if cnn_result.get('error'):
                    st.warning(cnn_result['error'])
                else:
                    st.markdown(f"**{cnn_result['top1']}** ({cnn_result['conf']*100:.0f}%)")
                    if num_views > 1:
                        st.caption(f"Averaged from {num_views} views")

                    st.markdown("**Top-5 Species**")
                    for species, conf in cnn_result['top5']:
                        st.markdown(f"**{species}**: {conf*100:.1f}%")
                        st.progress(float(conf))

            with col_cam:
                st.markdown("**Grad-CAM**")
                gradcam_imgs = cnn_result.get('gradcam_imgs', [])
                if gradcam_imgs:
                    if len(gradcam_imgs) > 1:
                        cam_cols = st.columns(len(gradcam_imgs))
                        for i, cam_img in enumerate(gradcam_imgs):
                            with cam_cols[i]:
                                st.image(cam_img, caption=f"View {i+1}", use_container_width=True)
                    else:
                        st.image(gradcam_imgs[0], use_container_width=True)
                    st.caption("Warm colors = high importance")
                elif cnn_result.get('gradcam_img'):
                    st.image(cnn_result['gradcam_img'], use_container_width=True)
                    st.caption("Warm colors = high importance")

        # ViT tab
        with tab_vit:
            col_pred, col_attn = st.columns([1, 1])

            with col_pred:
                if vit_result.get('error'):
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

                    if num_views > 1:
                        st.caption(f"Averaged from {num_views} views")

                    st.markdown("**Top-5 Species (with metadata)**")
                    for species, conf in vit_result['top5']:
                        st.markdown(f"**{species}**: {conf*100:.1f}%")
                        st.progress(float(conf))

            with col_attn:
                st.markdown("**Attention Maps**")
                attn_imgs = vit_result.get('attn_imgs', [])
                if attn_imgs:
                    if len(attn_imgs) > 1:
                        attn_cols = st.columns(len(attn_imgs))
                        for i, attn_img in enumerate(attn_imgs):
                            with attn_cols[i]:
                                st.image(attn_img, caption=f"View {i+1}", use_container_width=True)
                    else:
                        st.image(attn_imgs[0], use_container_width=True)
                elif vit_result.get('attn_img'):
                    st.image(vit_result['attn_img'], use_container_width=True)

    else:
        with tab_yolo:
            st.info("Upload photos and click Classify to see results")
        with tab_cnn:
            st.info("Upload photos and click Classify to see results")
        with tab_vit:
            st.info("Upload photos and click Classify to see results")

# Footer
st.divider()
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:12px'>"
    "Applied Mashroomatics · DF20-Mini · 180 species · "
    "YOLOv8 · EfficientNet-B0 · ViT-Base/16 · PyTorch + Streamlit"
    "</p>",
    unsafe_allow_html=True
)