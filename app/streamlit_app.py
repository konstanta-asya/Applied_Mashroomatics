"""
Mushroom Classifier - Home Page
"""

import streamlit as st

st.set_page_config(
    page_title="Mushroom Classifier",
    page_icon="🍄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: gray;
        margin-top: 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🍄 Mushroom Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Інтелектуальна система ідентифікації грибів</p>', unsafe_allow_html=True)

st.markdown("---")

# Hero section
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### 🎯 Визначайте гриби за фото

    Завантажте фотографію гриба і отримайте:

    - ✅ **Назву виду** (латинська + українська)
    - 📊 **Рівень впевненості** моделі
    - ⚠️ **Попередження** про отруйність
    - 📋 **Топ-5 схожих видів**

    """)

    if st.button("🔬 Почати класифікацію", type="primary", use_container_width=True):
        st.switch_page("pages/1_🔬_Класифікатор.py")

with col2:
    st.markdown("""
    ### 📸 Як отримати найкращий результат

    1. **Сфотографуйте зверху** — шапинку
    2. **Сфотографуйте знизу** — пластинки/трубочки
    3. **Вкажіть метадані** — сезон, місце
    4. **Отримайте результат** за секунди!

    > 💡 Два фото з різних ракурсів
    > підвищують точність на 10-15%
    """)

st.markdown("---")

# Stats
st.markdown("### 📊 Можливості системи")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("🍄 Видів", "179")

with c2:
    st.metric("🖼️ Датасет", "32K+")

with c3:
    st.metric("🎯 Точність", "~70%")

with c4:
    st.metric("⚡ Час", "<3 сек")

st.markdown("---")

# Features
st.markdown("### ✨ Особливості")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    #### 🧠 Vision Transformer

    Найсучасніша архітектура
    для розпізнавання зображень.
    Pretrained на ImageNet,
    fine-tuned на грибах.
    """)

with f2:
    st.markdown("""
    #### 📋 Метадані

    Використовуємо сезон,
    середовище та субстрат
    для підвищення точності
    на 5-10%.
    """)

with f3:
    st.markdown("""
    #### 🔒 Безпека

    Чіткі попередження про
    отруйні та смертельно
    небезпечні види грибів.
    """)

st.markdown("---")

# Quick links
st.markdown("### 🔗 Швидкі посилання")

l1, l2, l3, l4 = st.columns(4)

with l1:
    if st.button("🔬 Класифікатор", use_container_width=True):
        st.switch_page("pages/1_🔬_Класифікатор.py")

with l2:
    if st.button("📖 Про проєкт", use_container_width=True):
        st.switch_page("pages/2_📖_Про_проєкт.py")

with l3:
    if st.button("🍄 Види грибів", use_container_width=True):
        st.switch_page("pages/3_🍄_Види_грибів.py")

with l4:
    if st.button("👥 Команда", use_container_width=True):
        st.switch_page("pages/4_👥_Команда.py")

st.markdown("---")

# Warning
st.warning("""
⚠️ **Важливо!** Цей додаток створено в навчальних цілях.
НЕ використовуйте його як єдине джерело для визначення їстівності грибів.
Завжди консультуйтеся з експертом-мікологом!
""")

# Footer
st.markdown("""
<p style='text-align: center; color: gray; font-size: 0.8rem; margin-top: 2rem;'>
    🍄 Applied Mashroomatics | 2024 | Powered by Vision Transformer
</p>
""", unsafe_allow_html=True)