import streamlit as st

st.set_page_config(page_title="Team", page_icon="👥", layout="centered")
st.markdown('<style>[data-testid="stSidebarNav"]{display:none}</style>',
            unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.page_link("streamlit_app.py", label="Classifier", icon="🔬")
    st.page_link("pages/about.py", label="About", icon="📖")
    st.page_link("pages/species.py", label="Species", icon="🍄")
    st.page_link("pages/team.py", label="Team", icon="👥")

st.title("Team")
st.caption("National University of Kyiv-Mohyla Academy · Applied Mathematics")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background:var(--background-color);
        border:0.5px solid #e0e0e0;
        border-radius:12px;
        padding:20px;
        text-align:center;
        height:100%;
    ">
        <div style="
            width:64px;height:64px;border-radius:50%;
            background:#EEEDFE;color:#534AB7;
            font-size:22px;font-weight:600;
            display:flex;align-items:center;justify-content:center;
            margin:0 auto 12px;
        ">AK</div>
        <p style="font-size:15px;font-weight:600;margin:0 0 4px;">Anastasiia Konstantynovska</p>
        <p style="font-size:12px;color:#888;margin:0 0 12px;">Applied Mathematics, Year 3</p>
        <div style="
            background:#EEEDFE;color:#534AB7;
            font-size:11px;font-weight:500;
            padding:3px 10px;border-radius:4px;
            display:inline-block;margin-bottom:8px;
        ">Team Lead</div>
        <p style="font-size:13px;color:#555;margin:8px 0 0;line-height:1.5;">
            ViT architecture &amp; training<br>
            Metadata early fusion<br>
            Project architecture<br>
            Web application
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background:var(--background-color);
        border:0.5px solid #e0e0e0;
        border-radius:12px;
        padding:20px;
        text-align:center;
        height:100%;
    ">
        <div style="
            width:64px;height:64px;border-radius:50%;
            background:#E1F5EE;color:#0F6E56;
            font-size:22px;font-weight:600;
            display:flex;align-items:center;justify-content:center;
            margin:0 auto 12px;
        ">KS</div>
        <p style="font-size:15px;font-weight:600;margin:0 0 4px;">Khrystyna Skulysh</p>
        <p style="font-size:12px;color:#888;margin:0 0 12px;">Applied Mathematics, Year 3</p>
        <div style="
            background:#E1F5EE;color:#0F6E56;
            font-size:11px;font-weight:500;
            padding:3px 10px;border-radius:4px;
            display:inline-block;margin-bottom:8px;
        ">CNN &amp; EDA</div>
        <p style="font-size:13px;color:#555;margin:8px 0 0;line-height:1.5;">
            EfficientNet-B0 training<br>
            Exploratory data analysis<br>
            Data preprocessing<br>
            Species classification
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background:var(--background-color);
        border:0.5px solid #e0e0e0;
        border-radius:12px;
        padding:20px;
        text-align:center;
        height:100%;
    ">
        <div style="
            width:64px;height:64px;border-radius:50%;
            background:#FAECE7;color:#993C1D;
            font-size:22px;font-weight:600;
            display:flex;align-items:center;justify-content:center;
            margin:0 auto 12px;
        ">MS</div>
        <p style="font-size:15px;font-weight:600;margin:0 0 4px;">Mykhailo Siryi</p>
        <p style="font-size:12px;color:#888;margin:0 0 12px;">Applied Mathematics, Year 4</p>
        <div style="
            background:#FAECE7;color:#993C1D;
            font-size:11px;font-weight:500;
            padding:3px 10px;border-radius:4px;
            display:inline-block;margin-bottom:8px;
        ">YOLO</div>
        <p style="font-size:13px;color:#555;margin:8px 0 0;line-height:1.5;">
            YOLOv8 training<br>
            Safety classification<br>
            Object detection<br>
            Edible / poisonous pipeline
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<p style="text-align:center;color:#aaa;font-size:13px;">
    National University of Kyiv-Mohyla Academy · Faculty of Informatics<br>
    Applied Mathematics · 2025–2026
</p>
""", unsafe_allow_html=True)