import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Tentukan direktori kerja
current_dir = os.getcwd()

# 2. Cache unduhan NLTK
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    return True

download_nltk_resources()

# 3. Inisialisasi session_state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# 4. Fungsi preprocessing
def preprocess_text(text):
    stop_words = st.session_state.preprocess_params['stopwords']
    lemmatizer = st.session_state.preprocess_params['lemmatizer']
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# 5. Load semua model & params
def load_models():
    if not st.session_state.models_loaded:
        with st.spinner("Memuat model..."):
            # Paths
            p1 = os.path.join(current_dir, 'preprocess_params.pkl')
            p2 = os.path.join(current_dir, 'logreg_model.pkl')
            p3 = os.path.join(current_dir, 'tfidf_vectorizer.pkl')
            p4 = os.path.join(current_dir, 'label_encoder.pkl')
            # Cek keberadaan
            for p in (p1, p2, p3, p4):
                if not os.path.exists(p):
                    st.error(f"File tidak ditemukan: {os.path.basename(p)}")
                    st.stop()
            # Load
            st.session_state.preprocess_params = joblib.load(p1)
            st.session_state.logreg_model      = joblib.load(p2)
            st.session_state.vectorizer        = joblib.load(p3)
            st.session_state.label_encoder     = joblib.load(p4)
            st.session_state.models_loaded     = True

load_models()

# === UI ===
st.set_page_config(page_title="News Category Classifier", page_icon="📰", layout="centered")
st.title("📰 News Category Classifier")
st.markdown("Klasifikasi kategori berita: POLITICS, ENTERTAINMENT, SPORTS, STYLE & BEAUTY, BUSINESS.")

news_text = st.text_area("Masukkan teks berita:", height=150)
if st.button("Prediksi Kategori"):
    if not news_text.strip():
        st.warning("Silakan masukkan teks berita terlebih dahulu")
    else:
        with st.spinner("Memprediksi..."):
            try:
                clean = preprocess_text(news_text)
                vec   = st.session_state.vectorizer.transform([clean])
                pred  = st.session_state.logreg_model.predict(vec)[0]
                probs = st.session_state.logreg_model.predict_proba(vec)[0]

                # Tampilkan hasil
                st.subheader("Hasil Klasifikasi")
                st.success(f"Kategori: **{pred}**")

                # DataFrame probabilitas
                labels = st.session_state.label_encoder.classes_
                prob_df = pd.DataFrame({
                    'Kategori': labels,
                    'Probabilitas': probs
                }).sort_values('Probabilitas', ascending=False)

                st.subheader("Probabilitas per Kategori")
                st.bar_chart(prob_df.set_index('Kategori'), height=300)

                st.subheader("Detail Probabilitas")
                for cat, p in zip(labels, probs):
                    pct = int(p * 100)
                    c1, c2 = st.columns([2, 5])
                    with c1: st.markdown(f"**{cat}**")
                    with c2: st.progress(pct, f"{pct}%")
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
