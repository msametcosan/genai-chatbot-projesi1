
# Akbank GenAI Bootcamp: Dinamik RAG Chatbot Projesi

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, RAG (Retrieval Augmented Generation) mimarisini kullanan dinamik bir sohbet botudur.

## 🚀 Projenin Amacı 

Projenin temel amacı, kullanıcının belirlediği herhangi bir Wikipedia konusunu anlık olarak "bilgi kaynağı" olarak kullanan bir RAG chatbot'u oluşturmaktır. Sabit bir veri setine bağlı kalmak yerine, kullanıcıya sohbet sırasında "hafızasını" dinamik olarak belirleme esnekliği sunulmuştur. Uygulama, Streamlit aracılığıyla interaktif bir web arayüzü üzerinden sunulmaktadır.

## 📊 Veri Seti Hakkında Bilgi 

Bu projede statik (sabit) bir veri seti kullanılmamıştır.

**Veri Kaynağı:** Wikipedia (Dinamik)

**Metodoloji:**
Kullanıcı, web arayüzündeki kenar çubuğuna (sidebar) bir konu başlığı (örn: "Yapay zeka", "Mustafa Kemal Atatürk") girdiğinde, uygulama anlık olarak:
1.  Python `wikipedia` kütüphanesini kullanarak o konuyla ilgili en alakalı Türkçe Wikipedia makalesini çeker.
2.  Makalenin tam metnini (`.content`) alır.
3.  Bu metni, anlamlı paragraflara (100 karakterden uzun "chunk"lar) böler.
4.  Bu parçaları, chatbot'un RAG mimarisi için "bilgi kaynağı" (hafıza) olarak kullanır.

## 🛠️ Kullanılan Yöntemler ve Mimari 

Proje, modern bir RAG (Retrieval Augmented Generation) mimarisi üzerine kurulmuştur.

**Çözüm Mimarisi:**
1.  **Veri Toplama (Retrieval):** Kullanıcının girdiği konu `wikipedia` kütüphanesi ile bulunur.
2.  **Embedding:** Toplanan metin parçaları (chunks), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modeli kullanılarak 384 boyutlu anlamsal vektörlere dönüştürülür.
3.  **İndeksleme (Indexing):** Bu vektörler, hızlı anlamsal arama yapılabilmesi için `faiss-cpu` (bir vektör veritabanı) içinde indekslenir.
4.  **Sorgu (Query):** Kullanıcı bir soru sorduğunda (örn: "Atatürk kimdir?"), bu soru da aynı embedding modeli ile vektöre dönüştürülür.
5.  **Arama (Search):** FAISS veritabanı kullanılarak, kullanıcının soru vektörüne en yakın (anlamsal olarak en alakalı) 5 metin parçası (context) bulunur.
6.  **Zenginleştirme (Augmentation):** Bu 5 alakalı metin parçası, bir "bilgi şablonu" (prompt) içine yerleştirilir.
7.  **Üretim (Generation):** Bu zenginleştirilmiş prompt, cevap üretmesi için `gemini-pro-latest` (Google Gemini Pro) modeline gönderilir. Modele, "Sadece sana verdiğim bu bilgileri kullanarak cevap ver" talimatı verilir.

**Kullanılan Teknolojiler:**
* **Generation Modeli:** Google `gemini-pro-latest`
* **Embedding Modeli:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Vektör Veritabanı:** `faiss-cpu`
* **Web Arayüzü:** `Streamlit`
* **Deployment (Yayınlama):** `Hugging Face Spaces` (Docker SDK ile)

## 📋 Elde Edilen Sonuçlar 

Proje başarıyla tamamlanmış ve tüm teknik gereksinimler karşılanmıştır.
* Kullanıcının girdiği herhangi bir Wikipedia konusunu temel alan, dinamik RAG mimarisi başarıyla oluşturulmuştur.
* Uygulama, `Dockerfile` ve `Streamlit` kullanılarak Hugging Face Spaces üzerinde başarıyla canlıya alınmıştır.


* ## ⚙️ Projeyi Lokal Olarak Çalıştırma Kılavuzu

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/muratdrd/akbanka1.git](https://github.com/muratdrd/akbanka1.git)
    cd akbanka1
    ```

2.  **Sanal Ortam Oluşturun (Önerilir):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux için
    .\venv\Scripts\activate  # Windows için
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Anahtarını Ayarlayın:**
    * Proje klasörünüzün içinde `.streamlit` adında bir klasör oluşturun.
    * Bu klasörün içine `secrets.toml` adında bir dosya oluşturun.
    * Dosyanın içine Google API anahtarınızı aşağıdaki formatta ekleyin:
      ```toml
      GOOGLE_API_KEY = "AIzaSy..."
      ```

5.  **Uygulamayı Başlatın:**
    ```bash
    streamlit run app.py
    ```
* 


## 🌐 Web Uygulama Linki 

Projenin canlı web arayüzüne aşağıdaki linkten erişebilirsiniz:
https://huggingface.co/spaces/murdred/akbank-genai-chatbot
** https://huggingface.co/spaces/murdred/akbank-genai-chatbot **



### 💡 Arayüz Kullanım Kılavuzu

Uygulamayı kullanmak oldukça basittir:

1.  **Konu Belirleyin:** Sol taraftaki menüde bulunan "Wikipedia Konusu Girin" metin kutusuna, hakkında sohbet etmek istediğiniz konuyu yazın (örn: `Fenerbahçe Spor Kulübü`).
2.  **Konuyu Ayarlayın:** "Yeni Konuyu Ayarla" butonuna basın. Uygulama, bu konuyla ilgili Wikipedia makalesini bulup hafızasını hazırlayacaktır.
3.  **Soru Sorun:** Ana sohbet ekranının en altındaki metin kutusuna, belirlediğiniz konuyla ilgili sorunuzu yazın ve Enter'a basın.
4.  **Cevabı Alın:** Chatbot, Wikipedia'dan aldığı bilgilere dayanarak sorunuza cevap üretecektir.
5.  ÖRNEK SORULAR : Atatürk kimdir? , Everest Dağı nerededir?, Albert Einstein ne zaman doğdu?



