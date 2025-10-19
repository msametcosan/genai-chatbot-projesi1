
# Akbank GenAI Bootcamp: Dinamik RAG Chatbot Projesi

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, RAG (Retrieval Augmented Generation) mimarisini kullanan dinamik bir sohbet botudur[cite: 2].

## 🚀 Projenin Amacı 

Projenin temel amacı, kullanıcının belirlediği herhangi bir Wikipedia konusunu anlık olarak "bilgi kaynağı" olarak kullanan bir RAG chatbot'u oluşturmaktır[cite: 2]. Sabit bir veri setine bağlı kalmak yerine, kullanıcıya sohbet sırasında "hafızasını" dinamik olarak belirleme esnekliği sunulmuştur. Uygulama, Streamlit aracılığıyla interaktif bir web arayüzü üzerinden sunulmaktadır.

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
* `PermissionError` ve `Dockerfile` yapılandırma hataları gibi sunucu taraflı sorunlar, cache (önbellek) dizininin `/tmp` olarak ayarlanması ve `Dockerfile`'ın "build" aşamasında modeli indirmeye zorlanması gibi yöntemlerle çözülmüştür.
* Başlangıçta denenen `gemini-1.5-flash` modelinin API anahtarı yetkilendirme sorunları (`404 Not Found` hatası), daha stabil olan `gemini-pro-latest` modeline dönülerek aşılmıştır.

## 🌐 Web Uygulama Linki 

Projenin canlı web arayüzüne aşağıdaki linkten erişebilirsiniz:

**[https://huggingface.co/spaces/muratdrd/akbank-genai-chatbot](https://huggingface.co/spaces/muratdrd/akbank-genai-chatbot)**



