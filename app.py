# app.py (requests kütüphanesi kullanan GÜNCEL versiyon)

import streamlit as st
import requests # google-generativeai yerine bunu kullanıyoruz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import wikipedia
import json

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Dinamik RAG Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Dinamik Bilgi Tabanlı RAG Chatbot")
st.markdown("Bu chatbot, sorduğunuz soruların konusuna göre Wikipedia'dan anlık olarak bilgi çeker ve Google Gemini modelini kullanarak bu bilgilere dayanarak cevaplar üretir.")

# --- API ANAHTARI VE MODELLER ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    st.sidebar.success("API anahtarı başarıyla yüklendi.")
except Exception as e:
    st.sidebar.error("API anahtarı bulunamadı. Lütfen Streamlit Secrets'ı kontrol edin.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = load_embedding_model()
st.sidebar.info("Embedding modeli hazır.")

# --- YENİ GEMINI İSTEK FONKSİYONU ---
def generate_gemini_response(prompt):
    """requests kütüphanesi ile Gemini API'sine doğrudan istek gönderir."""
    model_name = "gemini-1.5-flash-latest" # Kullandığımız model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"

    headers = {'Content-Type': 'application/json'}

    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status() # HTTP hatalarını kontrol et (4xx veya 5xx)

        # Cevabı ayıkla
        result = response.json()
        # Güvenlik nedeniyle engellenen cevapları kontrol et
        if 'candidates' not in result or not result['candidates']:
             return "Üzgünüm, bu soruya verdiğim cevap güvenlik politikaları nedeniyle engellendi."

        return result['candidates'][0]['content']['parts'][0]['text']

    except requests.exceptions.HTTPError as http_err:
        # Özellikle 404 hatasını yakalamak için
        if response.status_code == 404:
            return f"API Hatası: Model ({model_name}) bulunamadı. Lütfen model adını kontrol edin."
        return f"HTTP Hatası: {http_err}"
    except Exception as e:
        return f"Cevap üretilirken beklenmedik bir hata oluştu: {e}"

# --- RAG MİMARİSİ FONKSİYONLARI ---
def bilgi_kaynagi_olustur(konu):
    try:
        wikipedia.set_lang("tr")
        arama_sonuclari = wikipedia.search(konu)
        if not arama_sonuclari:
            st.warning(f"'{konu}' ile ilgili bir Wikipedia sayfası bulunamadı.")
            return None, None
        sayfa_basligi = arama_sonuclari[0]
        page = wikipedia.page(sayfa_basligi, auto_suggest=False)
        chunks = []
        paragraflar = page.content.split('\n\n')
        for p in paragraflar:
            if len(p.strip()) > 100: chunks.append({'source': page.title, 'text': p.strip()})
        if not chunks:
            st.warning("Sayfa içeriği işlenecek kadar uzun paragraflara ayrılamadı.")
            return None, None
        df_chunks = pd.DataFrame(chunks)
        with st.spinner(f"'{page.title}' sayfası işleniyor..."):
            embeddings = embedding_model.encode(df_chunks['text'].tolist(), show_progress_bar=False)
        df_chunks['embeddings'] = list(embeddings)
        embeddings_array = np.array(df_chunks['embeddings'].tolist()).astype('float32')
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)
        st.success(f"'{page.title}' sayfası başarıyla işlendi.")
        return df_chunks, index
    except Exception as e:
        st.error(f"Bilgi kaynağı oluşturulurken bir hata oluştu: {e}")
        return None, None

# --- SESSION STATE VE ARAYÜZ ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df_chunks' not in st.session_state:
    st.session_state.df_chunks = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = ""

with st.sidebar:
    st.header("Konu Seçimi")
    yeni_konu = st.text_input("Yeni bir konu belirleyin:", key="new_topic_input")
    if st.button("Konuyu Yükle"):
        if yeni_konu:
            st.session_state.current_topic = yeni_konu
            st.session_state.df_chunks, st.session_state.index = bilgi_kaynagi_olustur(yeni_konu)
            st.session_state.messages = [] # Yeni konuyla sohbeti sıfırla
        else:
            st.warning("Lütfen bir konu girin.")
    if st.session_state.current_topic:
        st.success(f"Mevcut Konu: {st.session_state.current_topic}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    if st.session_state.index is None:
        st.warning("Lütfen önce kenar çubuğundan bir konu yükleyin.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Cevap hazırlanıyor..."):
                # 1. Retrieval
                soru_vector = embedding_model.encode([prompt]).astype('float32')
                k = 5
                distances, indices = st.session_state.index.search(soru_vector, k)
                relevant_chunks = [st.session_state.df_chunks.iloc[i]['text'] for i in indices[0]]
                context = "\n\n".join(relevant_chunks)

                # 2. Augmented Generation
                prompt_template = f"Aşağıdaki BİLGİLER bölümünü kullanarak SORU'yu cevapla. Sadece bu bilgileri kullan. Yetersizse 'Bu konuda bilgim yok.' de.\n\nBİLGİLER:\n{context}\n\nSORU: {prompt}\n\nCEVAP:"

                response_text = generate_gemini_response(prompt_template) # YENİ FONKSİYONU ÇAĞIR

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})