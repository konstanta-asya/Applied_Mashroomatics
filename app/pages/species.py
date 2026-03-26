"""
Species Browser Page
"""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Species", page_icon="🍄", layout="wide")
st.markdown('<style>[data-testid="stSidebarNav"]{display:none}</style>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.page_link("streamlit_app.py", label="Classifier", icon="🔬")
    st.page_link("pages/about.py", label="About", icon="📖")
    st.page_link("pages/species.py", label="Species", icon="🍄")
    st.page_link("pages/team.py", label="Team", icon="👥")

# Load species data
@st.cache_data
def load_species_data():
    data_path = Path(__file__).parent.parent.parent / "data" / "species_info.json"
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_species_metadata():
    data_path = Path(__file__).parent.parent.parent / "data" / "species_metadata.json"
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


species_info = load_species_data()
species_metadata = load_species_metadata()

st.title("🍄 Species Browser")
st.markdown(f"Database contains **{len(species_info)}** mushroom species")

# Filters
st.sidebar.header("🔍 Filters")

# Edibility filter
edibility_filter = st.sidebar.multiselect(
    "Edibility",
    options=["edible", "inedible", "poisonous", "deadly", "unknown"],
    default=None,
    format_func=lambda x: {
        "edible": "🟢 Edible",
        "inedible": "🟡 Inedible",
        "poisonous": "🔴 Poisonous",
        "deadly": "💀 Deadly",
        "unknown": "❓ Unknown"
    }.get(x, x)
)

# Genus filter
all_genera = sorted(set(sp.split()[0] for sp in species_info.keys()))
genus_filter = st.sidebar.multiselect("Genus", options=all_genera)

# Search
search = st.sidebar.text_input("🔎 Search", placeholder="Species name...")

# Filter species
filtered_species = {}
for species, info in species_info.items():
    # Edibility filter
    if edibility_filter and info.get("edibility") not in edibility_filter:
        continue

    # Genus filter
    genus = species.split()[0]
    if genus_filter and genus not in genus_filter:
        continue

    # Search filter
    if search:
        search_lower = search.lower()
        if (search_lower not in species.lower() and
            search_lower not in (info.get("common_name") or "").lower()):
            continue

    filtered_species[species] = info

st.markdown(f"Showing: **{len(filtered_species)}** species")

# Display species
edibility_colors = {
    "edible": "#28a745",
    "inedible": "#ffc107",
    "poisonous": "#dc3545",
    "deadly": "#721c24",
    "unknown": "#6c757d"
}

edibility_icons = {
    "edible": "🟢",
    "inedible": "🟡",
    "poisonous": "🔴",
    "deadly": "💀",
    "unknown": "❓"
}

# Group by genus
genera = {}
for species, info in sorted(filtered_species.items()):
    genus = species.split()[0]
    if genus not in genera:
        genera[genus] = []
    genera[genus].append((species, info))

# Display
for genus, species_list in sorted(genera.items()):
    with st.expander(f"**{genus}** ({len(species_list)} species)", expanded=len(genera) <= 5):
        for species, info in species_list:
            edibility = info.get("edibility", "unknown")
            icon = edibility_icons.get(edibility, "❓")
            color = edibility_colors.get(edibility, "#6c757d")
            common_name = info.get("common_name", "")

            col1, col2, col3 = st.columns([3, 2, 2])

            with col1:
                st.markdown(f"**{icon} {species}**")
                if common_name:
                    st.caption(f"{common_name}")

            with col2:
                desc = info.get("description", "")
                if desc:
                    st.caption(desc[:50] + "..." if len(desc) > 50 else desc)

            with col3:
                # Show season from metadata
                meta = species_metadata.get(species, {})
                months = meta.get("months", {})
                if months:
                    top_months = sorted(months.items(), key=lambda x: -x[1])[:3]
                    month_names = {
                        "1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr",
                        "5": "May", "6": "Jun", "7": "Jul", "8": "Aug",
                        "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
                    }
                    months_str = ", ".join(month_names.get(m, m) for m, _ in top_months)
                    st.caption(f"📅 {months_str}")

            st.divider()

# Statistics
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistics")

edibility_counts = {}
for info in species_info.values():
    ed = info.get("edibility", "unknown")
    edibility_counts[ed] = edibility_counts.get(ed, 0) + 1

for ed, count in sorted(edibility_counts.items(), key=lambda x: -x[1]):
    icon = edibility_icons.get(ed, "❓")
    st.sidebar.write(f"{icon} {ed}: **{count}**")