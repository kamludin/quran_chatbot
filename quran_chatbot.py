import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Custom CSS for enhanced UI
st.markdown("""
<style>
/* Hide Streamlit menu, footer, and header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

body {
    background-color: #f4f4f4;
    font-family: 'Arial', sans-serif;
}

.arabic {
    font-family: 'Amiri', serif;
    font-size: 24px;
    text-align: right;
    direction: rtl;
    line-height: 2;
    color: #2c3e50;
}

.english {
    font-size: 18px;
    color: #555;
    margin-top: 10px;
}

.result-block {
    padding: 20px;
    margin: 15px 0;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.bilingual-title {
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
    color: #1E88E5;
}

.bilingual-subtitle {
    font-size: 22px;
    text-align: center;
    margin-bottom: 30px;
    color: #555;
}

.search-box {
    width: 100%;
    font-size: 18px;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #ccc;
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
}
</style>
""", unsafe_allow_html=True)

# Load Quran dataset
@st.cache_resource
def load_data():
    try:
        quran_df = pd.read_csv('The Quran Dataset.csv')
        return quran_df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'The Quran Dataset.csv' exists.")
        return None

# Load embeddings and FAISS index
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    quran_df = load_data()
    
    if quran_df is None:
        return None, None, None, None
    
    bilingual_texts = [
        f"{row['surah_name_ar']} ({row['surah_no']}:{row['ayah_no_surah']}) {row['ayah_ar']} | {row['ayah_en']}"
        for _, row in quran_df.iterrows()
    ]
    
    embeddings = model.encode(bilingual_texts, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return model, index, quran_df, bilingual_texts

def main():
    st.markdown("""
    <div class="bilingual-title">
        القرآن الكريم - Quranic AI Assistant
    </div>
    <div class="bilingual-subtitle">
        اكتب سؤالك بالعربية أو الإنجليزية | Ask your question in Arabic or English
    </div>
    """, unsafe_allow_html=True)
    
    model, index, quran_df, bilingual_texts = load_embeddings()
    
    if model is None:
        return

    reciter_url = "https://everyayah.com/data/Alafasy_64kbps/{surah_no:03d}{ayah_no:03d}.mp3"
    
    # Input box with styled search bar
    user_query = st.text_input("اكتب سؤالك هنا | Enter your question here:", 
                               placeholder="مثال: آيات عن الرحمة أو which verse shows mercy", 
                               key="search_box")
    
    # Search button with better spacing
    search_btn = st.button("🔍 بحث | Search", key="search_button")

    if user_query and search_btn:
        query_embedding = model.encode([user_query])
        D, I = index.search(query_embedding.astype('float32'), k=5)
        
        st.subheader("📖 النتائج | Results:")
        for idx in I[0]:
            verse_data = quran_df.iloc[idx]
            
            audio_url = reciter_url.format(
                surah_no=int(verse_data['surah_no']),
                ayah_no=int(verse_data['ayah_no_surah'])
            )
            
            st.markdown(f"""
            <div class="result-block">
                <div class="arabic">{verse_data['ayah_ar']}</div>
                <div class="english">{verse_data['ayah_en']}</div>
                <div style="text-align: right; margin-top: 15px;">
                    <b>{verse_data['surah_name_ar']} ({verse_data['surah_no']}:{verse_data['ayah_no_surah']})</b><br>
                    <i>الجزء {verse_data['juz_no']} - السورة رقم {verse_data['surah_no']}</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.audio(audio_url, format="audio/mp3")

if __name__ == '__main__':
    main()
