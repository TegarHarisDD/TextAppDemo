import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pandas as pd
import os
import shutil
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl

# Setup untuk mengatasi SSL certificate error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Setup path khusus untuk NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download resources yang diperlukan dengan cara manual
def setup_nltk():
    # Download semua resource yang diperlukan
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
    }
    
    for resource_name, resource_path in resources.items():
        target_path = os.path.join(nltk_data_path, resource_path)
        if not os.path.exists(target_path):
            try:
                print(f"Downloading {resource_name} resource...")
                nltk.download(resource_name, download_dir=nltk_data_path)
                
                # Jika download berhasil tapi folder tidak terbuat, buat manual
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                    print(f"Created missing directory: {target_path}")
                
                # Perbaiki path punkt_tab
                if resource_name == 'punkt':
                    punkt_tab_path = os.path.join(nltk_data_path, 'tokenizers/punkt_tab')
                    if not os.path.exists(punkt_tab_path):
                        os.makedirs(punkt_tab_path)
                        print(f"Created punkt_tab directory: {punkt_tab_path}")
                    
                    # Salin file dari punkt ke punkt_tab
                    for file_name in os.listdir(target_path):
                        if file_name.endswith('.pickle'):
                            src = os.path.join(target_path, file_name)
                            dst = os.path.join(punkt_tab_path, file_name)
                            shutil.copy2(src, dst)
                            print(f"Copied {file_name} to punkt_tab")
            except Exception as e:
                print(f"Error downloading {resource_name}: {str(e)}")
                st.error(f"Error setting up NLTK resources: {str(e)}")
                st.stop()

# Setup NLTK
setup_nltk()

# Dapatkan path direktori saat ini
current_dir = os.path.dirname(os.path.abspath(__file__))

# Fungsi preprocessing
def preprocess_text(text):
    # Load preprocessing params
    stop_words = st.session_state.preprocess_params['stopwords']
    lemmatizer = st.session_state.preprocess_params['lemmatizer']
    
    # Lowercasing
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Stopword removal and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
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
- Solusi khusus diterapkan untuk mengatasi masalah resource NLTK
""")
