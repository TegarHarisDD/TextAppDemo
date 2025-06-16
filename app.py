import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import os
import string

# Dapatkan path direktori saat ini
current_dir = os.path.dirname(os.path.abspath(__file__))

# Fungsi preprocessing tanpa NLTK
def preprocess_text(text):
    # Load preprocessing params
    stop_words = st.session_state.preprocess_params['stopwords']
    # Lemmatizer dihapus karena tidak digunakan
    
    # Lowercasing
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization sederhana dengan split
    tokens = text.split()
    
    # Stopword removal
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Load models
def load_models():
    if 'models_loaded' not in st.session_state:
        with st.spinner('Memuat model...'):
            try:
                print("Loading models...")
                
                # Load preprocessing params
                preprocess_path = os.path.join(current_dir, 'preprocess_params.pkl')
                if not os.path.exists(preprocess_path):
                    raise FileNotFoundError(f"File preprocess_params.pkl not found at {preprocess_path}")
                st.session_state.preprocess_params = joblib.load(preprocess_path)
                
                # Load Logistic Regression model
                model_path = os.path.join(current_dir, 'logreg_model.pkl')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"File logreg_model.pkl not found at {model_path}")
                st.session_state.logreg_model = joblib.load(model_path)
                
                # Load vectorizer
                vectorizer_path = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
                if not os.path.exists(vectorizer_path):
                    raise FileNotFoundError(f"File tfidf_vectorizer.pkl not found at {vectorizer_path}")
                st.session_state.vectorizer = joblib.load(vectorizer_path)
                
                # Load label encoder
                label_path = os.path.join(current_dir, 'label_encoder.pkl')
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"File label_encoder.pkl not found at {label_path}")
                st.session_state.label_encoder = joblib.load(label_path)
                
                st.session_state.models_loaded = True
                print("All models loaded successfully!")
                
            except Exception as e:
                st.error(f"Gagal memuat model: {str(e)}")
                st.stop()

# Konfigurasi halaman
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered"
)

# UI Header
st.title("📰 News Category Classifier")
st.markdown("""
Klasifikasi kategori berita menggunakan model Logistic Regression. 
Model dapat memprediksi 5 kategori: POLITICS, ENTERTAINMENT, SPORTS, STYLE & BEAUTY, BUSINESS.
""")

# Input teks
news_text = st.text_area(
    label="Masukkan teks berita:",
    placeholder="Contoh: The president announced new economic policies today...",
    height=150,
    key="news_input"
)

# Tombol prediksi
predict_btn = st.button("Prediksi Kategori")

# Load models
load_models()

# Fungsi prediksi
def predict_category(text):
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Transform text to TF-IDF
    text_vec = st.session_state.vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = st.session_state.logreg_model.predict(text_vec)[0]
    probabilities = st.session_state.logreg_model.predict_proba(text_vec)[0]
    
    return prediction, probabilities

# Handle prediksi
if predict_btn and news_text.strip():
    with st.spinner('Memprediksi kategori...'):
        try:
            # Get prediction
            prediction, probabilities = predict_category(news_text)
            
            # Tampilkan hasil utama
            st.subheader("Hasil Klasifikasi")
            st.success(f"Kategori prediksi: **{prediction}**")
            
            # Tampilkan probabilitas
            prob_df = pd.DataFrame({
                'Kategori': [st.session_state.label_encoder['id2label'][i] for i in range(len(probabilities))],
                'Probabilitas': probabilities
            })
            prob_df = prob_df.sort_values('Probabilitas', ascending=False)
            
            # Tampilkan grafik
            st.subheader("Probabilitas Kategori")
            st.bar_chart(
                prob_df.set_index('Kategori'),
                height=400,
                use_container_width=True
            )
            
            # Tampilkan detail probabilitas
            st.subheader("Detail Probabilitas")
            for i, prob in enumerate(probabilities):
                category = st.session_state.label_encoder['id2label'][i]
                progress = int(prob * 100)
                st.markdown(f"**{category}**")
                st.progress(progress, f"{progress}%")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
    
elif predict_btn:
    st.warning("Silakan masukkan teks berita terlebih dahulu")

# Tambahkan penjelasan model
st.markdown("---")
st.subheader("Tentang Model")
st.markdown("""
**Logistic Regression:**
- Akurasi: 86%
- Waktu prediksi: Sangat cepat (ms)
- Cocok untuk penggunaan real-time

Dataset: [News Category Dataset (Kaggle)](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
""")

# Informasi tambahan
st.markdown("---")
st.info("""
**Catatan:**
- Aplikasi ini menggunakan model Logistic Regression yang dioptimalkan
- Waktu pemuatan model hanya beberapa detik
- Hasil prediksi muncul secara instan setelah teks dimasukkan
- Preprocessing teks dilakukan tanpa dependensi NLTK
""")
