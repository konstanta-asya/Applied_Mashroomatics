"""
Team Page
"""

import streamlit as st

st.set_page_config(page_title="Команда", page_icon="👥", layout="centered")
st.markdown('<style>[data-testid="stSidebarNav"]{display:none}</style>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.page_link("streamlit_app.py", label="Класифікатор", icon="🔬")
    st.page_link("pages/2_📖_Про_проєкт.py", label="Про проєкт", icon="📖")
    st.page_link("pages/3_🍄_Види_грибів.py", label="Види грибів", icon="🍄")
    st.page_link("pages/4_👥_Команда.py", label="Команда", icon="👥")

st.title("👥 Наша команда")

st.markdown("""
Проєкт **Applied Mashroomatics** розроблено командою студентів.

---
""")

# Team members
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 👩‍💻 Учасник А
    **Data Scientist / Analyst**

    - 📊 EDA та аналіз даних
    - 🧠 CNN Baseline
    - 📈 Метрики та валідація
    - 📝 Звітність
    """)

with col2:
    st.markdown("""
    ### 👨‍💻 Учасник Б
    **ML Engineer / Backend**

    - 🤖 Vision Transformer
    - ⚙️ Backend API
    - 🚀 Оптимізація
    - 🔧 Інфраструктура
    """)

with col3:
    st.markdown("""
    ### 👩‍💻 Учасник В
    **CV Engineer / Frontend**

    - 🎯 YOLO Detection
    - 🖥️ Web UI
    - 🎨 User Experience
    - 📱 Демонстрація
    """)

st.markdown("---")

st.markdown("""
### 📅 План реалізації (6 тижнів)

| Тиждень | Фокус | Статус |
|---------|-------|--------|
| 1 | Дані та EDA | ✅ Завершено |
| 2 | Базове моделювання (MVP) | ✅ Завершено |
| 3 | Прототип застосунку | ✅ Завершено |
| 4 | Інтеграція пайплайну | 🔄 В процесі |
| 5 | Валідація та UX | ⏳ Заплановано |
| 6 | Фіналізація | ⏳ Заплановано |

---

### 🛠️ Технологічний стек

| Категорія | Інструменти |
|-----------|-------------|
| **ML/DL** | PyTorch, timm, torchvision |
| **CV** | YOLOv5, PIL, OpenCV |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Data** | Pandas, NumPy, scikit-learn |
| **DevOps** | Google Colab, Git |

---

### 📧 Контакти

Якщо у вас є питання або пропозиції щодо проєкту:

- 📩 Email: [team@example.com](mailto:team@example.com)
- 🐙 GitHub: [Applied-Mashroomatics](https://github.com/)

---

### 🙏 Подяки

- **Danish Fungi Project** — за датасет
- **Hugging Face** — за бібліотеку timm
- **Streamlit** — за чудовий фреймворк

""")