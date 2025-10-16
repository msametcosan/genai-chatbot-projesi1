# app.py

import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import wikipedia
import time

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Dinamik RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Dinamik Bilgi Tabanlı RAG Chatbot")
st.markdown("""
Bu chatbot, sorduğunuz soruların konusuna göre Wikipedia'dan anlık olarak bilgi çeker ve 
Google Gemini modelini kullanarak bu bilgilere dayanarak cevaplar üretir.
""")

# --- API ANAHTARI VE MODELLER ---

# API anahtarını Streamlit'in Secrets yönetiminden alıyoruz.
# Lokal'de çalışırken .streamlit/secrets.toml dosyası oluşturmalısınız.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    st.sidebar.success("API anahtarı başarıyla yüklendi.")
except Exception as e:
    st.sidebar.error("API anahtarı bulunamadı. Lütfen Streamlit Secrets'ı kontrol edin.")
    st.stop()


# Modelleri sadece bir kere yüklemek için cache kullanıyoruz. Bu, performansı artırır.
@st.cache_resource
def load_models():
    """Embedding ve generation modellerini yükler."""
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    generation_model = genai.GenerativeModel('gemini-1.0-pro')
    return embedding_model, generation_model


embedding_model, gen_model = load_models()
st.sidebar.info("Embedding ve Gemini modelleri hazır.")


# --- RAG MİMARİSİ FONKSİYONLARI ---

def bilgi_kaynagi_olustur(konu):
    """Belirtilen konu hakkında Wikipedia'dan bilgi alır ve bir FAISS indeksi oluşturur."""
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
            if len(p.strip()) > 100:
                chunks.append({'source': page.title, 'text': p.strip()})

        if not chunks:
            st.warning("Sayfa içeriği işlenecek kadar uzun paragraflara ayrılamadı.")
            return None, None

        df_chunks = pd.DataFrame(chunks)

        with st.spinner(f"'{page.title}' sayfasındaki metinler vektöre dönüştürülüyor..."):
            embeddings = embedding_model.encode(df_chunks['text'].tolist(), show_progress_bar=False)
            df_chunks['embeddings'] = list(embeddings)

        embeddings_array = np.array(df_chunks['embeddings'].tolist()).astype('float32')
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)

        st.success(f"'{page.title}' sayfası başarıyla işlendi ve bilgi tabanına eklendi.")
        return df_chunks, index

    except Exception as e:
        st.error(f"Bilgi kaynağı oluşturulurken bir hata oluştu: {e}")
        return None, None


# --- SESSION STATE (SOHBET HAFIZASI) YÖNETİMİ ---
# Streamlit her etkileşimde script'i baştan çalıştırır.
# Değişkenleri kaybetmemek için session_state kullanırız.
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = ""
    st.session_state.df_chunks = None
    st.session_state.index = None
    st.session_state.messages = []

# --- ARAYÜZ BİLEŞENLERİ ---

# Kenar çubuğu (Sidebar)
with st.sidebar:
    st.header("Konu Seçimi")
    yeni_konu = st.text_input("Yeni bir konu belirleyin (Örn: Albert Einstein):", key="new_topic_input")
    if st.button("Konuyu Yükle"):
        if yeni_konu:
            with st.spinner(f"'{yeni_konu}' konusu yükleniyor..."):
                st.session_state.current_topic = yeni_konu
                df, idx = bilgi_kaynagi_olustur(yeni_konu)
                st.session_state.df_chunks = df
                st.session_state.index = idx
                # Konu değiştiğinde sohbet geçmişini temizle
                st.session_state.messages = []
        else:
            st.warning("Lütfen bir konu girin.")

    if st.session_state.current_topic:
        st.success(f"Mevcut Konu: {st.session_state.current_topic}")

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni soru al
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # Önce bir konunun yüklenmiş olduğundan emin ol
    if st.session_state.index is None:
        st.warning("Lütfen önce kenar çubuğundan bir konu yükleyin.")
    else:
        # Kullanıcının mesajını sohbet geçmişine ekle ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG işlemini başlat ve cevabı üret
        with st.chat_message("assistant"):
            with st.spinner("Cevap hazırlanıyor..."):
                # 1. Retrieval
                soru_vector = embedding_model.encode([prompt]).astype('float32')
                k = 5
                distances, indices = st.session_state.index.search(soru_vector, k)
                relevant_chunks = [st.session_state.df_chunks.iloc[i]['text'] for i in indices[0]]
                context = "\n\n".join(relevant_chunks)

                # 2. Augmented Generation
                prompt_template = f"""Aşağıdaki BİLGİLER bölümünde verilen metinleri kullanarak SORU'yu cevapla.
                Cevabını yalnızca ve yalnızca sana verilen bu BİLGİLER'e dayandır. Eğer bilgiler soruyu cevaplamak için yetersizse, 'Bu konuda bilgim yok.' de.

                BİLGİLER:
                {context}

                SORU: {prompt}

                CEVAP:"""

                try:
                    response = gen_model.generate_content(prompt_template)
                    response_text = response.text
                except Exception as e:
                    response_text = f"Cevap üretilirken bir hata oluştu: {e}"

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})