# Akbank GenAI Bootcamp - Genel Kültür Chatbot Projesi

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, RAG (Retrieval Augmented Generation) mimarisi kullanan bir chatbot içerir.

## Projenin Amacı

Projenin ana hedefi, statik bir veri setine bağlı kalmadan, kullanıcının belirlediği herhangi bir konu hakkında gerçek zamanlı olarak bilgi toplayan ve bu bilgileri kullanarak soruları yanıtlayan akıllı bir sohbet botu geliştirmektir. Bu sayede, chatbot'un bilgi tabanı neredeyse sınırsız ve sürekli güncel kalmaktadır. Proje, son kullanıcıya bir web arayüzü aracılığıyla sunulacaktır.

## Veri Seti Hakkında Bilgi

Bu projede geleneksel, önceden hazırlanmış bir veri seti kullanılmamıştır. Bunun yerine, dinamik bir veri toplama yaklaşımı benimsenmiştir.

Veri Kaynağı: Türkçe Wikipedia.

Metodoloji: Kullanıcı sohbet sırasında yeni bir konu belirlediğinde veya hakkında bilgi olmayan bir soru sorduğunda, sistem otomatik olarak bu konuyla ilgili en alakalı Wikipedia makalesini bulur. Makalenin içeriği o anki sohbetin bilgi kaynağı (context) olarak kullanılır. Bu yöntem sayesinde chatbot, sabit bir bilgiyle sınırlı kalmaz ve her konuda konuşabilir hale gelir.

## Kullanılan Yöntemler

Proje, RAG (Retrieval-Augmented Generation)  mimarisi temel alınarak geliştirilmiştir. Çözümün iş akışı aşağıdaki adımlardan oluşmaktadır:

Konu Belirleme: Kullanıcı, konu: [konu adı] komutuyla veya doğrudan bir soru sorarak sohbetin ana konusunu belirler.

Veri Çekme (Retrieval): Belirlenen konu başlığı ile Wikipedia'dan ilgili makale çekilir.

Parçalama (Chunking): Makale metni, anlamsal bütünlüğü olan daha küçük paragraflara (chunk) ayrılır.


Vektörleştirme (Embedding): Her metin parçası, sentence-transformers kütüphanesi kullanılarak anlamsal olarak karşılığı olan sayısal vektörlere dönüştürülür.


İndeksleme (Indexing): Oluşturulan vektörler, hızlı anlamsal arama yapabilmek için FAISS vektör veritabanına yüklenir.

Anlamsal Arama: Kullanıcının sorusu da aynı modelle vektöre dönüştürülür ve FAISS üzerinde yapılan arama ile soruya en alakalı metin parçaları bulunur.


Cevap Üretimi (Generation): Bulunan alakalı metin parçaları (context) ve kullanıcının orijinal sorusu, bir prompt şablonu ile birleştirilerek Google Gemini API'sine  gönderilir. Gemini, kendisine verilen bu bağlama sadık kalarak nihai cevabı üretir.


Kullanılan Teknolojiler:


Generation Model: Google Gemini 1.0 Pro 


Embedding Model: paraphrase-multilingual-MiniLM-L12-v2 


Vector Database: FAISS (in-memory) 

Data Source: Wikipedia API


## Elde Edilen Sonuçlar

Geliştirilen chatbot, aşağıdaki yeteneklere sahiptir:

Geniş bir konu yelpazesinde, Wikipedia'da var olan bilgiler dahilinde tutarlı ve doğru cevaplar üretebilmektedir.

Cevaplarını yalnızca sağlanan bağlama dayandırdığı için, yapay zeka modellerinde sıkça görülen "halüsinasyon" (bilgi uydurma) sorunu minimize edilmiştir. Bilgi yetersiz olduğunda "Bu konuda bilgim yok." diyerek dürüst bir yanıt vermektedir.

Dinamik yapısı sayesinde bilgi tabanı esnektir ve kolayca yeni konulara adapte olabilmektedir.




Çalışma Kılavuzu:
Projeyi lokal makinenizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

Gereksinimler:

Python 3.8+

Kurulum:

Bu depoyu klonlayın:

Bash

git clone https://github.com/[msametcosan]/[genai-chatbot-projesi1].git
Proje dizinine gidin:

Bash

cd [genai-chatbot-projesi1]
Gerekli kütüphaneleri requirements.txt dosyasını kullanarak yükleyin:

Bash

pip install -r requirements.txt
(Not: Henüz oluşturmadıysanız, pip freeze > requirements.txt komutuyla projenizin bağımlılıklarını içeren bu dosyayı oluşturun.)

Google AI Studio üzerinden bir API anahtarı oluşturun ve bu anahtarı bir .env dosyasında saklayın:

GOOGLE_API_KEY="YAPAY-ZEKA-API-ANAHTARINIZ"
Ana Python betiğini çalıştırın:

Bash

python main.py


## Web Arayüzü

(Bu bölüm proje deploy edildikten sonra doldurulacaktır. Buraya link eklenecektir.)
