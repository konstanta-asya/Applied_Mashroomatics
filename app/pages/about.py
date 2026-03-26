import streamlit as st

st.set_page_config(page_title="About", page_icon="📖", layout="centered")
st.markdown('<style>[data-testid="stSidebarNav"]{display:none}</style>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.page_link("streamlit_app.py", label="Classifier", icon="🔬")
    st.page_link("pages/about.py", label="About", icon="📖")
    st.page_link("pages/species.py", label="Species", icon="🍄")
    st.page_link("pages/team.py", label="Team", icon="👥")

st.title("About the Project")

st.markdown("""
## Applied Mashroomatics

A deep learning project for mushroom species classification using the
**Danish Fungi 2020 Mini (DF20-Mini)** dataset.

---

### Dataset

- ~32,000 mushroom images
- 180 species classes
- Rich metadata: habitat, substrate, observation date

---

### Models

| Model | Task | Architecture |
|-------|------|--------------|
| **YOLO** | Binary safety classification | YOLOv8 |
| **CNN** | 180-species classification | EfficientNet-B0 |
| **ViT** | 180-species + metadata fusion | ViT-Base/16 |

---

### How It Works

1. **Upload** a mushroom photo
2. **Select metadata** (optional) — habitat, substrate, month
3. **Compare** predictions from all three models
4. **View attention maps** to see what the model focuses on

---

### Architecture Highlights

**Vision Transformer with Metadata Early Fusion:**
- Metadata (habitat, substrate, month) encoded as learnable embeddings
- Fused with image patches at the input layer
- Token sequence: `[CLS, META, patch_1, ..., patch_196]`

**Grad-CAM for CNN:**
- Gradient-weighted Class Activation Mapping
- Visualizes which regions influence the prediction

**Attention Rollout for ViT:**
- Aggregates attention across all transformer layers
- Shows global attention patterns

---

### Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Framework** | PyTorch |
| **Model Library** | timm |
| **Object Detection** | Ultralytics YOLOv8 |
| **Web Framework** | Streamlit |
| **Training** | Google Colab (A100 GPU) |

---

### Disclaimer

> **This tool is for educational purposes only.**
>
> Never rely solely on AI predictions for mushroom identification.
> Some poisonous species closely resemble edible ones.
> Always consult an expert mycologist before consuming wild mushrooms.
""")