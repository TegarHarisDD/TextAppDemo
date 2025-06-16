import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Fungsi preprocessing
def preprocess_text(text):
    # Load preprocessing params
    stop_words = preprocess_params['stopwords']
    lemmatizer = preprocess_params['lemmatizer']
    
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
@st.cache_resource
def load_models():
    print("Loading models...")
    # Load preprocessing params
    preprocess_params = joblib.load('preprocess_params.pkl')
    
    # Load Logistic Regression assets
    logreg_model = joblib.load('logreg_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    # Load DistilBERT assets
    tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_tokenizer')
    bert_model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model')
    
    # Load label encoder
    label_encoder = joblib.load('label_encoder.pkl')
    
    return {
        'preprocess_params': preprocess_params,
        'logreg_model': logreg_model,
        'vectorizer': vectorizer,
        'tokenizer': tokenizer,
        'bert_model': bert_model,
        'label_encoder': label_encoder
    }

# Konfigurasi halaman
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="wide"
)

# UI Header
st.title("📰 News Category Classifier")
st.markdown("""
Klasifikasi kategori berita menggunakan model machine learning. 
Model dapat memprediksi 5 kategori: POLITICS, ENTERTAINMENT, SPORTS, STYLE & BEAUTY, BUSINESS.
""")

# Input teks
news_text = st.text_area(
    label="Masukkan teks berita:",
    placeholder="Contoh: The president announced new economic policies today...",
    height=150
)

# Pilihan model
model_option = st.selectbox(
    "Pilih model klasifikasi:",
    ("Logistic Regression (Cepat & Akurat)", "DistilBERT (Akurasi Tinggi)")
)

# Tombol prediksi
predict_btn = st.button("Prediksi Kategori")

# Load models
if 'models' not in st.session_state:
    st.session_state.models = load_models()

# Fungsi prediksi
def predict_category(text, model_type):
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    if "Logistic Regression" in model_type:
        # Transform text to TF-IDF
        text_vec = st.session_state.models['vectorizer'].transform([cleaned_text])
        
        # Predict
        prediction = st.session_state.models['logreg_model'].predict(text_vec)[0]
        probabilities = st.session_state.models['logreg_model'].predict_proba(text_vec)[0]
        
    else:  # DistilBERT
        # Tokenize text
        inputs = st.session_state.models['tokenizer'](
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = st.session_state.models['bert_model'](**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1).numpy()[0]
        prediction_idx = np.argmax(probabilities)
        prediction = st.session_state.models['label_encoder']['id2label'][prediction_idx]
    
    return prediction, probabilities

# Handle prediksi
if predict_btn and news_text.strip():
    with st.spinner('Memprediksi kategori...'):
        try:
            # Get prediction
            prediction, probabilities = predict_category(news_text, model_option)
            
            # Tampilkan hasil
            st.subheader("Hasil Klasifikasi")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric(
                    label="Kategori Prediksi", 
                    value=prediction,
                    delta=f"{np.max(probabilities)*100:.1f}%"
                )
            
            with col2:
                # Tampilkan probabilitas
                prob_df = pd.DataFrame({
                    'Kategori': [st.session_state.models['label_encoder']['id2label'][i] for i in range(len(probabilities))],
                    'Probabilitas': probabilities
                })
                prob_df = prob_df.sort_values('Probabilitas', ascending=False)
                
                st.bar_chart(
                    prob_df.set_index('Kategori'),
                    height=300,
                    use_container_width=True
                )
            
            # Tampilkan detail probabilitas
            st.subheader("Detail Probabilitas")
            for i, prob in enumerate(probabilities):
                category = st.session_state.models['label_encoder']['id2label'][i]
                progress = int(prob * 100)
                st.markdown(f"**{category}**")
                st.progress(progress, f"{progress}%")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
    
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

**DistilBERT:**
- Akurasi: 89%
- Waktu prediksi: Sedang (beberapa detik)
- Lebih akurat untuk teks kompleks

Dataset: [News Category Dataset (Kaggle)](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
""")
