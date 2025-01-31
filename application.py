import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Загрузка модели и векторизатора
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()

# Настройка интерфейса
st.set_page_config(
    page_title="URL Classifier",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🌐 Классификатор URL")
st.markdown("""
    Это приложение предсказывает тип URL-адреса: **phishing**, **benign**, **defacement** или **malware**.
    Введите URL в поле ниже и нажмите кнопку **Предсказать**.
""")

# Поле для ввода URL
url_input = st.text_input("Введите URL:", placeholder="https://example.com")

# Кнопка для предсказания
if st.button("Предсказать"):
    if url_input:
        # Преобразование URL в числовые признаки
        url_vectorized = vectorizer.transform([url_input])

        # Предсказание
        prediction = model.predict(url_vectorized)
        prediction_proba = model.predict_proba(url_vectorized)

        # Декодирование предсказания
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        probabilities = {label_encoder.classes_[i]: prediction_proba[0][i] for i in range(len(label_encoder.classes_))}

        # Отображение результата
        st.success(f"Предсказанный тип: **{predicted_class}**")

        # Визуализация вероятностей
        st.subheader("Вероятности для каждого класса:")
        proba_df = pd.DataFrame(list(probabilities.items()), columns=["Класс", "Вероятность"])
        st.bar_chart(proba_df.set_index("Класс"))

    else:
        st.error("Пожалуйста, введите URL.")