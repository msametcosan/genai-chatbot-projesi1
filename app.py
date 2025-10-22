# --- 1. Adım: Gerekli Modülleri Import Etme ---

# Önbellek (Cache) hatasını çözmek için bu bölüm en üste eklendi
import os
cache_dir = "/tmp/sentence_transformers_cache/"
os.makedirs(cache_dir, exist_ok=True)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import wikipedia
import time

# --- 2. Adım: Sayfa Yapılandırması ve Başlık ---
st.set_page_config(
    page_title="Akbank GenAI Bootcamp RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Dinamik Bilgi Kaynaklı RAG Chatbot")
# DÜZELTME: Modeli 'Gemini Pro' olarak güncelledik
st.caption(
    f"Akbank GenAI Bootcamp Projesi - Wikipedia'dan alınan verilerle Gemini Pro ve FAISS kullanılarak oluşturulmuştur."
)

# --- 3. Adım: API Anahtarı Yapılandırması (Secrets'tan Okuma) ---
try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error(
            "❌ GOOGLE_API_KEY bulunamadı. Lütfen Hugging Face Spaces 'Secrets' bölümüne ekleyin.")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"❌ API anahtarı yapılandırılırken bir hata oluştu: {e}")
    st.stop()

# --- 4. Adım: Modelleri Yükleme (Cache ve Hız Optimizasyonu ile) ---
@st.cache_resource
def load_models():
    """Embedding ve Generative modelleri bir kez yükler."""
    
    # Cache dizinini, fonksiyonun içinde de açıkça tanımla
    cache_dir = "/tmp/sentence_transformers_cache/"
        
    with st.spinner("🧠 Yapay zeka modelleri yükleniyor... (Bu işlem yalnızca ilk açılışta biraz zaman alabilir)"):
        try:
            # DÜZELTME: 'cache_folder' parametresi fonksiyonun İÇİNE eklendi
            embedding_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                cache_folder=cache_dir 
            )
            
            # DÜZELTME: Yetkimiz olan 'gemini-pro-latest' modeline geri dönüldü
            gen_model = genai.GenerativeModel('gemini-pro-latest')
            
            return embedding_model, gen_model
        
        except Exception as e:
            st.error(
                f"❌ Modeller yüklenirken bir hata oluştu: {e}. Lütfen internet bağlantınızı kontrol edin veya sayfayı yenileyin.")
            st.stop()

# Modelleri yükle
embedding_model, gen_model = load_models()


# --- 5. Adım: Bilgi Kaynağı Oluşturma Fonksiyonu ---
@st.cache_data(show_spinner=False)
def bilgi_kaynagi_olustur(konu):
    """
    Belirtilen konu hakkında Wikipedia'dan bilgi alır, işler ve
    arama yapılabilecek bir FAISS indeksi ile veri setini (DataFrame) döndürür.
    """
    try:
        with st.status(f"📚 '{konu}' hakkında bilgi kaynağı oluşturuluyor...", expanded=True) as status:
            status.write(f"Wikipedia'dan '{konu}' konusu aranıyor...")
            wikipedia.set_lang("tr")
            arama_sonuclari = wikipedia.search(konu)
            if not arama_sonuclari:
                st.error(f"❌ '{konu}' ile ilgili bir Wikipedia sayfası bulunamadı.")
                status.update(label="❌ Konu bulunamadı.", state="error")
                return None, None, None

            sayfa_basligi = arama_sonuclari[0]
            page = wikipedia.page(sayfa_basligi, auto_suggest=False)
            status.write(f"📄 '{page.title}' sayfası başarıyla bulundu ve işleniyor.")
            
            chunks = []
            paragraflar = page.content.split('\n\n')
            for p in paragraflar:
                if len(p.strip()) > 100:
                    chunks.append({'source': page.title, 'text': p.strip()})
            
            if not chunks:
                st.error("❌ Sayfa içeriği işlenecek kadar uzun paragraflara ayrılamadı.")
                status.update(label="❌ Sayfa içeriği yetersiz.", state="error")
                return None, None, None

            df_chunks = pd.DataFrame(chunks)
            status.write(f"🧩 {len(df_chunks)} adet metin parçası (chunk) vektöre dönüştürülüyor...")
            
            embeddings = embedding_model.encode(df_chunks['text'].tolist(), show_progress_bar=False)
            df_chunks['embeddings'] = list(embeddings)
            
            embeddings_array = np.array(df_chunks['embeddings'].tolist()).astype('float32')
            d = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embeddings_array)
            
            status.write("✅ FAISS arama indeksi başarıyla oluşturuldu.")
            time.sleep(1)
            status.update(label=f"✅ '{page.title}' konusu sohbete hazır!", state="complete")
            
            return df_chunks, index, page.title

    except wikipedia.exceptions.PageError:
        st.error(f"❌ '{konu}' adında bir Wikipedia sayfası bulunamadı.")
        return None, None, None
    except wikipedia.exceptions.DisambiguationError as e:
        st.error(f"❌ '{konu}' birden çok anlama geliyor. Lütfen daha spesifik olun. (Örn: {e.options[:3]})")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Bilgi kaynağı oluşturulurken beklenmedik bir hata oluştu: {e}")
        return None, None, None


# --- 6. Adım: Soru Cevaplama Fonksiyonu (RAG Pipeline) ---
def soru_cevapla(soru, df_chunks, index):
    """
    Kullanıcı sorusunu ve oluşturulan bilgi kaynağını (index) kullanarak cevap üretir.
    """
    try:
        soru_vector = embedding_model.encode([soru]).astype('float32')
        k = 5
        distances, indices = index.search(soru_vector, k)
        relevant_chunks = [df_chunks.iloc[i]['text'] for i in indices[0]]
        context = "\n\n".join(relevant_chunks)
        
        prompt = f"""Aşağıdaki BİLGİLER bölümünde verilen metinleri kullanarak SORU'yu cevapla.
Cevabını yalnızca ve yalnızca sana verilen bu BİLGİLER'e dayandır. Eğer bilgiler soruyu cevaplamak için yetersizse, 'Bu konuda bilgim yok.' de.
BİLGİLER:
{context}
SORU: {soru}
CEVAP:"""

        response = gen_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Cevap üretilirken bir hata oluştu: {e}")
        return "Üzgünüm, cevap üretirken teknik bir sorunla karşılaştım."


# --- 7. Adım: Ana Uygulama Arayüzü ve Sohbet Mantığı ---

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
if "current_index" not in st.session_state:
    st.session_state.current_index = None
if "current_df" not in st.session_state:
    st.session_state.current_df = None

# --- Yan Menü (Sidebar) ---
with st.sidebar:
    st.header("Konu Seçimi")
    st.markdown("""
    Chatbot'un hangi Wikipedia konusu hakkında konuşacağını buradan belirleyebilirsiniz.
    """)
    
    yeni_konu = st.text_input("Wikipedia Konusu Girin:", placeholder="Örn: Yapay zeka")
    
    if st.button("Yeni Konuyu Ayarla", type="primary"):
        if yeni_konu:
            st.session_state.current_topic = yeni_konu
            st.session_state.messages = []
            
            df, index, title = bilgi_kaynagi_olustur(yeni_konu)
            
            if df is not None and index is not None:
                st.session_state.current_df = df
                st.session_state.current_index = index
                st.session_state.current_topic = title
            else:
                st.session_state.current_topic = ""
                st.session_state.current_df = None
                st.session_state.current_index = None
        else:
            st.sidebar.warning("Lütfen bir konu adı girin.")
            
    st.divider()
    st.markdown("Proje dökümanına [buradan](https://github.com/muratdrd/akbanka1) ulaşabilirsiniz.") # Buradaki linki kendi GitHub reponuzla değiştirmeyi unutmayın

# --- Ana Sohbet Alanı ---
if not st.session_state.current_topic:
    st.info("Lütfen sohbete başlamak için kenar çubuğundan bir Wikipedia konusu belirleyin.")
    st.stop()

st.info(f"Şu anki sohbet konusu: **{st.session_state.current_topic}** (Wikipedia'dan alındı)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"'{st.session_state.current_topic}' hakkında bir soru sorun..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Cevap oluşturuluyor..."):
            cevap = soru_cevapla(prompt, st.session_state.current_df, st.session_state.current_index)
            st.markdown(cevap)
    
    st.session_state.messages.append({"role": "assistant", "content": cevap})

