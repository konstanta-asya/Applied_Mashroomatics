"""
Species Browser Page
"""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Види грибів", page_icon="🍄", layout="wide")
st.markdown('<style>[data-testid="stSidebarNav"]{display:none}</style>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.page_link("streamlit_app.py", label="Класифікатор", icon="🔬")
    st.page_link("pages/2_📖_Про_проєкт.py", label="Про проєкт", icon="📖")
    st.page_link("pages/3_🍄_Види_грибів.py", label="Види грибів", icon="🍄")
    st.page_link("pages/4_👥_Команда.py", label="Команда", icon="👥")

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

st.title("🍄 Види грибів")
st.markdown(f"База даних містить **{len(species_info)}** видів грибів")

# Filters
st.sidebar.header("🔍 Фільтри")

# Edibility filter
edibility_filter = st.sidebar.multiselect(
    "Їстівність",
    options=["edible", "inedible", "poisonous", "deadly", "unknown"],
    default=None,
    format_func=lambda x: {
        "edible": "🟢 Їстівні",
        "inedible": "🟡 Неїстівні",
        "poisonous": "🔴 Отруйні",
        "deadly": "💀 Смертельні",
        "unknown": "❓ Невідомо"
    }.get(x, x)
)

# Genus filter
all_genera = sorted(set(sp.split()[0] for sp in species_info.keys()))
genus_filter = st.sidebar.multiselect("Рід", options=all_genera)

# Search
search = st.sidebar.text_input("🔎 Пошук", placeholder="Назва виду...")

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
            search_lower not in (info.get("common_name_ua") or "").lower()):
            continue

    filtered_species[species] = info

st.markdown(f"Показано: **{len(filtered_species)}** видів")

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
    with st.expander(f"**{genus}** ({len(species_list)} видів)", expanded=len(genera) <= 5):
        for species, info in species_list:
            edibility = info.get("edibility", "unknown")
            icon = edibility_icons.get(edibility, "❓")
            color = edibility_colors.get(edibility, "#6c757d")
            common_name = info.get("common_name_ua", "")

            col1, col2, col3 = st.columns([3, 2, 2])

            with col1:
                st.markdown(f"**{icon} {species}**")
                if common_name:
                    st.caption(f"🇺🇦 {common_name}")

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
                        "1": "Січ", "2": "Лют", "3": "Бер", "4": "Кві",
                        "5": "Тра", "6": "Чер", "7": "Лип", "8": "Сер",
                        "9": "Вер", "10": "Жов", "11": "Лис", "12": "Гру"
                    }
                    months_str = ", ".join(month_names.get(m, m) for m, _ in top_months)
                    st.caption(f"📅 {months_str}")

            st.divider()

# Statistics
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Статистика")

edibility_counts = {}
for info in species_info.values():
    ed = info.get("edibility", "unknown")
    edibility_counts[ed] = edibility_counts.get(ed, 0) + 1

for ed, count in sorted(edibility_counts.items(), key=lambda x: -x[1]):
    icon = edibility_icons.get(ed, "❓")
    st.sidebar.write(f"{icon} {ed}: **{count}**")