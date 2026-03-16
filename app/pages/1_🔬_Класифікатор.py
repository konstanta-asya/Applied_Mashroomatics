"""
Mushroom Classifier Page
"""

import streamlit as st
import requests
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Класифікатор", page_icon="🔬", layout="centered")


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def classify_image(image_bytes, top_k=5, use_tta=False, month=None, habitat=None, substrate=None):
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        params = {"top_k": top_k, "use_tta": use_tta}

        if month and month > 0:
            params["month"] = month
        if habitat and habitat != "-- Не вказано --":
            params["habitat"] = habitat
        if substrate and substrate != "-- Не вказано --":
            params["substrate"] = substrate

        response = requests.post(f"{API_URL}/predict", files=files, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the server is running."}
    except Exception as e:
        return {"error": str(e)}


def get_edibility_badge(edibility):
    badges = {
        "edible": ("🟢 Їстівний", "#28a745"),
        "inedible": ("🟡 Неїстівний", "#ffc107"),
        "poisonous": ("🔴 Отруйний", "#dc3545"),
        "deadly": ("💀 Смертельно отруйний", "#721c24"),
        "unknown": ("❓ Невідомо", "#6c757d"),
    }
    return badges.get(edibility, badges["unknown"])


def combine_predictions(results):
    if len(results) == 1:
        return results[0]

    species_scores = {}
    species_info = {}

    for result in results:
        for pred in result.get("predictions", []):
            species = pred["species_name"]
            conf = pred["confidence"]
            if species not in species_scores:
                species_scores[species] = []
                species_info[species] = pred
            species_scores[species].append(conf)

    combined_predictions = []
    for species, scores in species_scores.items():
        avg_conf = sum(scores) / len(results)
        pred = species_info[species].copy()
        pred["confidence"] = avg_conf
        combined_predictions.append(pred)

    combined_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_predictions = combined_predictions[:5]
    top_pred = top_predictions[0] if top_predictions else None
    is_confident = top_pred and top_pred["confidence"] >= 0.5

    warning = None
    if top_pred:
        edibility = top_pred.get("edibility", "unknown")
        if edibility in ["poisonous", "deadly"]:
            warning = f"WARNING: This mushroom ({top_pred['species_name']}) is {edibility.upper()}!"
        elif not is_confident:
            warning = "Low confidence prediction. Please provide clearer photos."

    return {
        "success": True,
        "predictions": top_predictions,
        "top_prediction": top_pred,
        "is_confident": is_confident,
        "warning": warning,
        "num_images_analyzed": len(results)
    }


# Page content
st.title("🔬 Класифікатор грибів")
st.markdown("Завантажте фото гриба для ідентифікації")

# Sidebar
with st.sidebar:
    st.header("⚙️ Налаштування")

    top_k = st.slider("Кількість результатів", 1, 10, 5)
    use_tta = st.checkbox("Test-Time Augmentation", help="Покращує точність, але працює повільніше")

    st.divider()
    st.subheader("📋 Метадані")
    st.caption("Покращують точність на 5-10%")

    month_names = {
        0: "-- Не вказано --", 1: "Січень", 2: "Лютий", 3: "Березень",
        4: "Квітень", 5: "Травень", 6: "Червень", 7: "Липень",
        8: "Серпень", 9: "Вересень", 10: "Жовтень", 11: "Листопад", 12: "Грудень"
    }
    selected_month = st.selectbox("📅 Місяць", options=list(month_names.keys()),
                                   format_func=lambda x: month_names[x], index=0)

    habitat_options = ["-- Не вказано --", "coniferous", "deciduous", "mixed", "park", "garden", "meadow", "bog"]
    habitat_names = {
        "-- Не вказано --": "-- Не вказано --", "coniferous": "🌲 Хвойний ліс",
        "deciduous": "🌳 Листяний ліс", "mixed": "🌲🌳 Мішаний ліс",
        "park": "🏛️ Парк", "garden": "🏡 Сад", "meadow": "🌾 Луг", "bog": "💧 Болото"
    }
    selected_habitat = st.selectbox("🌲 Середовище", options=habitat_options,
                                     format_func=lambda x: habitat_names.get(x, x))

    substrate_options = ["-- Не вказано --", "soil", "dead wood", "litter", "moss", "bark", "grass"]
    substrate_names = {
        "-- Не вказано --": "-- Не вказано --", "soil": "🟤 Ґрунт",
        "dead wood": "🪵 Мертва деревина", "litter": "🍂 Опале листя",
        "moss": "🌿 Мох", "bark": "🌳 Кора", "grass": "🌱 Трава"
    }
    selected_substrate = st.selectbox("🪵 Субстрат", options=substrate_options,
                                       format_func=lambda x: substrate_names.get(x, x))

    st.divider()
    if check_api_health():
        st.success("📡 API працює")
    else:
        st.error("📡 API недоступне")

# Image upload
st.subheader("📸 Завантажте зображення")
st.caption("Рекомендуємо 2 фото з різних ракурсів")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Фото 1** (верх)")
    uploaded1 = st.file_uploader("Фото 1", type=["jpg", "jpeg", "png"], key="p1", label_visibility="collapsed")

with col2:
    st.markdown("**Фото 2** (низ)")
    uploaded2 = st.file_uploader("Фото 2", type=["jpg", "jpeg", "png"], key="p2", label_visibility="collapsed")

image_sources = [src for src in [uploaded1, uploaded2] if src]

if image_sources:
    if len(image_sources) == 1:
        st.warning("⚠️ Рекомендуємо 2 фото для кращої точності")

    cols = st.columns(len(image_sources))
    images = []
    for i, (col, src) in enumerate(zip(cols, image_sources)):
        img = Image.open(src)
        images.append(img)
        with col:
            st.image(img, caption=f"Фото {i+1}", use_container_width=True)

    if st.button("🔍 Визначити гриб", type="primary", use_container_width=True):
        all_results = []

        with st.spinner(f"Аналізую {len(images)} зображень..."):
            for image in images:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                result = classify_image(
                    img_byte_arr.getvalue(), top_k=top_k, use_tta=use_tta,
                    month=selected_month, habitat=selected_habitat, substrate=selected_substrate
                )
                if "error" not in result:
                    all_results.append(result)

        if all_results:
            result = combine_predictions(all_results)
            st.session_state.result = result

# Results
if "result" in st.session_state:
    result = st.session_state.result

    st.divider()
    st.subheader("📊 Результати")

    if result.get("num_images_analyzed", 1) > 1:
        st.info(f"✅ Проаналізовано {result['num_images_analyzed']} зображення")

    top = result.get("top_prediction", {})

    if result.get("warning"):
        if "POISONOUS" in str(result["warning"]).upper() or "DEADLY" in str(result["warning"]).upper():
            st.error(f"⚠️ {result['warning']}")
        else:
            st.warning(result["warning"])

    st.markdown("### 🎯 Найбільш ймовірний вид")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        species = top.get("species_name", "Невідомо")
        common = top.get("common_name_ua", "")
        if common:
            st.markdown(f"**{common}**")
            st.caption(f"_{species}_")
        else:
            st.markdown(f"**{species}**")

    with c2:
        conf = top.get("confidence", 0)
        st.metric("Впевненість", f"{conf*100:.1f}%")

    with c3:
        edibility = top.get("edibility", "unknown")
        badge, color = get_edibility_badge(edibility)
        st.markdown(f"**{badge}**")

    if result.get("is_confident"):
        st.success("✅ Висока впевненість")
    else:
        st.warning("⚠️ Низька впевненість - зробіть краще фото")

    st.markdown("### 📋 Інші варіанти")
    for i, pred in enumerate(result.get("predictions", [])[1:], 2):
        badge, _ = get_edibility_badge(pred.get("edibility", "unknown"))
        name = pred.get("common_name_ua") or pred.get("species_name")
        st.write(f"{i}. **{name}** - {pred['confidence']*100:.1f}% {badge}")

    if st.button("🗑️ Очистити"):
        del st.session_state.result
        st.rerun()