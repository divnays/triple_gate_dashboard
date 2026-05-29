import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import seaborn as sns
import re
import warnings
import os
warnings.filterwarnings('ignore')

from collections import Counter
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from gensim import corpora
from gensim.models import LdaModel

# ─────────────────────────────────────────────
# PALETTE & STYLE
# ─────────────────────────────────────────────
C_BLUE   = "#00509D"
C_LBLUE  = "#89C2FF"
C_YELLOW = "#FFD60A"
C_PINK   = "#7FB3FF"
C_DARK   = "#1B263B"
C_WHITE  = "#FFFFFF"
C_GRAY   = "#F5F7FA"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Triple Gate UNY - Sentiment Dashboard",
    page_icon="assets/logo_uny.png" if os.path.exists("assets/logo_uny.png") else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stDataFrame,
div, span, p, h1, h2, h3, h4, label, button, input, select, textarea {
    font-family: 'Poppins', sans-serif !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

.stApp {
    background: linear-gradient(180deg, #DCEEFF 0%, #EAF4FF 100%);
}
header[data-testid="stHeader"] { background: transparent; }

[data-testid="metric-container"] {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 22px 26px 18px 26px;
    box-shadow: 0 8px 28px rgba(0,80,157,0.13), 0 2px 6px rgba(68,172,255,0.08);
    border: 1.5px solid #D5E8FF;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 36px rgba(0,80,157,0.18), 0 4px 12px rgba(68,172,255,0.12);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    color: #0D3B6E !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: #44ACFF !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
}

[data-testid="stTabs"] [role="tablist"] {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 6px;
    gap: 4px;
    box-shadow: 0 4px 16px rgba(68,172,255,0.13);
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 12px !important;
    padding: 10px 22px !important;
    font-weight: 700 !important;
    font-size: 0.83rem !important;
    color: #1A2235 !important;
    border: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #2073b5 !important;
    height: 3px !important;
    border-radius: 999px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #0D3B6E, #44ACFF) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(68,172,255,0.35);
}

[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; box-shadow: 0 4px 18px rgba(0,80,157,0.12); }
[data-testid="stDataFrame"] table { font-size: 0.84rem !important; border-collapse: separate; border-spacing: 0; }
[data-testid="stDataFrame"] thead tr th {
    background: linear-gradient(135deg, #0D3B6E, #1a6ec7) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 12px 16px !important;
    letter-spacing: 0.03em;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) td { background: #F0F7FF !important; }
[data-testid="stDataFrame"] tbody tr:hover td { background: #DDF0FF !important; }
[data-testid="stDataFrame"] tbody tr td {
    padding: 10px 14px !important;
    font-weight: 500 !important;
    color: #1A2235 !important;
    border-bottom: 1px solid #E5EFF9 !important;
}

.stAlert { border-radius: 12px; border-left-width: 5px; }
hr { border-color: #D0E5FF; border-width: 1.5px; }

.section-title {
    font-size: 1.08rem;
    font-weight: 800;
    color: #0D3B6E;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 3px solid #44ACFF;
    display: inline-block;
    letter-spacing: 0.01em;
}

.chart-shadow {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0,80,157,0.16), 0 2px 8px rgba(68,172,255,0.10);
    background: #FAFBFF;
    padding: 6px;
    margin-bottom: 8px;
}

[data-testid="stFileUploader"] section {
    border-radius: 14px !important;
    border: 2.5px dashed #44ACFF88 !important;
    background: #F4F9FF !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] label {
    font-weight: 700 !important;
    color: #0D3B6E !important;
    font-size: 0.9rem !important;
}

.stSpinner > div { border-top-color: #44ACFF !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER: chart shadow wrapper
# ─────────────────────────────────────────────
def chart_shadow(fig):
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    st.markdown(
        f'<div class="chart-shadow"><img src="data:image/png;base64,{img_b64}" '
        f'style="width:100%;border-radius:12px;"/></div>',
        unsafe_allow_html=True
    )
    plt.close(fig)


# ─────────────────────────────────────────────
# KAMUS & LEKSIKON
# ─────────────────────────────────────────────
slang_dict = {
    'gw':'saya','gue':'saya','gua':'saya',
    'yg':'yang','dgn':'dengan','utk':'untuk','dr':'dari',
    'jd':'jadi','jg':'juga','tp':'tapi','krn':'karena','karna':'karena',
    'tdk':'tidak','gak':'tidak','ga':'tidak','gk':'tidak',
    'nggak':'tidak','ngga':'tidak','enggak':'tidak','ora':'tidak','ra':'tidak',
    'gamau':'tidak mau','gabisa':'tidak bisa','emoh':'tidak mau',
    'bgt':'banget','bngt':'banget','sgt':'sangat',
    'udah':'sudah','udh':'sudah','sdh':'sudah','dah':'sudah','dh':'sudah',
    'wes':'sudah','skrg':'sekarang','skrang':'sekarang','abis':'habis',
    'trs':'terus','trus':'terus','blm':'belum','blum':'belum','kpn':'kapan',
    'gimana':'bagaimana','gmn':'bagaimana','gmana':'bagaimana','knp':'kenapa',
    'aja':'saja','aj':'saja','doang':'saja','jgn':'jangan',
    'nih':'ini','ni':'ini','iki':'ini','tu':'itu','tuh':'itu',
    'sih':'','loh':'','lah':'','deh':'','dong':'','mah':'',
    'wkwk':'','wkwkwk':'','haha':'','hahaha':'','hehe':'','xixi':'',
    'please':'tolong','pls':'tolong','plis':'tolong','pliss':'tolong',
    'tulung':'tolong','moga':'semoga',
    'msk':'masuk','kluar':'keluar','campus':'kampus','jl':'jalan',
    'diputer':'diputar','muterin':'memutari','muter':'putar',
    'muternya':'putar','mutar':'putar','puter':'putar','mbolak':'putar',
    'mhs':'mahasiswa',
    'ribet':'rumit','ruwett':'rumit','ngruweti':'rumit',
    'repott':'rumit','susahin':'rumit','bundet':'rumit','riweh':'rumit',
    'males':'malas','cape':'lelah','capek':'lelah','kesel':'kesal',
    'kzl':'kesal','mumet':'pusing','wagu':'aneh','ngaco':'kacau',
    'ngerasa':'merasa','ngerasain':'merasakan','nyebelin':'menjengkelkan',
    'nyusahin':'menyusahkan','nyari':'mencari','nunggu':'menunggu',
    'antri':'antre','ngantri':'mengantre',
    'opo':'apa','pie':'bagaimana','piye':'bagaimana','nek':'kalau',
    'wae':'saja','kudu':'harus','penak':'enak','gawe':'buat','angel':'sulit',
    'satpame':'satpam','tau':'tahu','kasi':'kasih','pake':'pakai',
    'rame':'ramai','gampang':'mudah','susah':'sulit',
}

kata_positif = {
    'bagus','baik','mantap','keren','oke','setuju','dukung','mendukung',
    'benar','tepat','cocok','sesuai','efektif','efisien','lancar',
    'membantu','bermanfaat','berguna','penting','perlu','aman',
    'tertib','teratur','rapi','bersih','nyaman','senang','puas',
    'bangga','suka','apresiasi','alhamdulillah','syukur','berhasil',
    'sukses','meningkat','maju','inovatif','kreatif','solusi',
    'positif','optimis','harapan','semangat','salut','selamat',
    'hebat','kece','top','jos','joss','keamanan','terjaga',
    'terlindungi','disiplin','terorganisir','terkontrol','terpantau',
}

kata_negatif = {
    'buruk','jelek','parah','payah','rusak','gagal','salah',
    'tolak','menolak','protes','marah','kesal','kecewa','sedih',
    'susah','sulit','repot','rumit','bingung','pusing','capek',
    'lelah','jengkel','sebal','benci','bosan','jenuh','muak',
    'macet','antre','mengantre','lama','lambat','telat','terlambat',
    'nyebelin','nyusahin','menyulitkan','mempersulit','menghambat',
    'larang','larangan','paksa','dipaksa','sia-sia','percuma',
    'aneh','konyol','gila','ngawur','merugikan','rugi','menderita',
    'jauh','muter','putar','memutar','mahal','bahaya','berbahaya',
    'risiko','masalah','problem','komplain','aduan',
    'kritik','mengkritik','nyerah','menyerah','malas',
    'terganggu','mengganggu',
}

topic_labels = [
    'Topik 1: Kemacetan dan Antrian',
    'Topik 2: Akses Jalan',
    'Topik 3: Kebijakan dan Peraturan',
    'Topik 4: Keamanan dan Fasilitas',
    'Topik 5: Aspirasi dan Tuntutan',
]

topic_keywords = {
    'Topik 1: Kemacetan dan Antrian':  'rumit, ruwet, konvoi, touring, sulit',
    'Topik 2: Akses Jalan':            'tidak, pusing, satpam, bijak, kampus',
    'Topik 3: Kebijakan dan Peraturan':'putar, repot, rute, gerbang, jalan',
    'Topik 4: Keamanan dan Fasilitas': 'kampus, jalan, rumit, aneh, gedung',
    'Topik 5: Aspirasi dan Tuntutan':  'telat, parkir, macet, bensin, boros',
}

TOPIC_COLORS = {
    'Topik 1: Kemacetan dan Antrian':  '#FF8FAB',
    'Topik 2: Akses Jalan':            '#00509D',
    'Topik 3: Kebijakan dan Peraturan':'#EFC52B',
    'Topik 4: Keamanan dan Fasilitas': '#89C2FF',
    'Topik 5: Aspirasi dan Tuntutan':  "#FF8337",
    'Tidak Terklasifikasi':            '#D3D3D3',
}

# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
def apply_chart_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor('#F7FAFF')
    for spine in ax.spines.values():
        spine.set_edgecolor('#D5E4F0')
        spine.set_linewidth(0.8)
    ax.tick_params(colors='#3A4A62', labelsize=9.5)
    ax.xaxis.label.set_color('#3A4A62')
    ax.yaxis.label.set_color('#3A4A62')
    if title:
        ax.set_title(title, fontsize=11.5, fontweight='800',
                     color='#0D3B6E', pad=14, fontfamily='sans-serif')
    if xlabel: ax.set_xlabel(xlabel, fontsize=9.5, fontweight='600')
    if ylabel: ax.set_ylabel(ylabel, fontsize=9.5, fontweight='600')
    ax.grid(axis='y', color='#DCE8F5', linewidth=0.7, linestyle='--', alpha=0.9)
    ax.grid(axis='x', visible=False)


# ─────────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────
noise_comments = {'sticker','stiker','gif','meme'}

def is_noise_comment(text):
    if not isinstance(text, str): return True
    text = text.lower().strip()
    if text == '': return True
    if text in noise_comments: return True
    if len(text) < 3 and not re.search(r'[a-zA-Z]', text): return True
    return False

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = remove_emoji(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = remove_repeated_chars(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text, slang_dict):
    words = text.split()
    return ' '.join([slang_dict.get(w, w) for w in words])

@st.cache_resource
def get_stemmer_and_stopwords():
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    sw_factory = StopWordRemoverFactory()
    stop_words_id = set(sw_factory.get_stop_words())
    negation_words = {'tidak','bukan','belum','tak','ga','gak','gk','nggak','ngga'}
    stop_words_id -= negation_words
    extra_stop = {
        'uny','universitas','negeri','yogyakarta','triple','gate','pintu','tiga',
        'fbsb','fmipa','fip','fik','fe','rektor','rektorat',
        'www','http','https','com','co','id','tiktok','fyp','viral',
        'share','like','subscribe','follow','video','konten','komen','komentar',
        'kak','aja','aj','banget','iya','iyaa','yah','ya',
        'nih','sih','dong','lah','kok','woi','bro','wkwk','wk','haha',
        'yg','dg','dr','krn','dll','etc','kalo','klo','kaya','kayak','kek',
        'biar','untung','terima','tidak','sticker','bikin','yek',
    }
    stop_words_id |= extra_stop
    return stemmer, stop_words_id

def remove_stopwords_fn(text, stop_words):
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words and len(w) > 1])

def stem_text_fn(text, stemmer):
    return stemmer.stem(text)

def tokenize(text):
    return text.split()

def label_sentiment(text, tokens):
    token_set  = set(tokens)
    text_words = set(text.split())
    pos_score  = len(token_set & kata_positif) + len(text_words & kata_positif)
    neg_score  = len(token_set & kata_negatif) + len(text_words & kata_negatif)
    return 'positif' if pos_score > neg_score else 'negatif'


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file_bytes):
    import io
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            df = pd.read_csv(io.BytesIO(uploaded_file_bytes), encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    cols_to_drop = [c for c in df.columns if 'sticker' in c.lower() or 'image' in c.lower()]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
    df = df[~df['text'].apply(is_noise_comment)].reset_index(drop=True)
    df.dropna(subset=['text'], inplace=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['diggCount'] = pd.to_numeric(df['diggCount'], errors='coerce').fillna(0).astype(int)
    stemmer, stop_words_id = get_stemmer_and_stopwords()
    df['text_clean']      = df['text'].apply(clean_text)
    df['text_normalized'] = df['text_clean'].apply(lambda x: normalize_slang(x, slang_dict))
    df['text_nostop']     = df['text_normalized'].apply(lambda x: remove_stopwords_fn(x, stop_words_id))
    df['text_stemmed']    = df['text_nostop'].apply(lambda x: stem_text_fn(x, stemmer))
    df['tokens']          = df['text_stemmed'].apply(tokenize)
    df = df[df['tokens'].apply(len) > 0].reset_index(drop=True)
    df['sentimen'] = df.apply(
        lambda row: label_sentiment(row['text_normalized'], row['tokens']), axis=1
    )
    return df

@st.cache_resource(show_spinner=False)
def train_svm_model(text_stemmed_list, sentimen_list):
    tfidf = TfidfVectorizer(
        max_features=5000, min_df=2, max_df=0.95,
        ngram_range=(1, 2), sublinear_tf=True
    )
    X_tfidf = tfidf.fit_transform(text_stemmed_list)
    y = pd.Series(sentimen_list)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    svm = SVC(kernel='linear', C=1.0, class_weight='balanced',
              random_state=42, probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=['negatif','positif'])
    sentimen_pred = svm.predict(X_tfidf)
    return tfidf, svm, acc, prec, rec, f1, cm, sentimen_pred

@st.cache_resource(show_spinner=False)
def train_lda_model(tokens_neg_list):
    _, stop_words_id = get_stemmer_and_stopwords()
    tokens_clean = [
        [t for t in tok if len(t) > 3 and t not in stop_words_id]
        for tok in tokens_neg_list
    ]
    tokens_clean = [t for t in tokens_clean if len(t) > 0]
    if not tokens_clean:
        return None, None, None
    dictionary = corpora.Dictionary(tokens_clean)
    dictionary.filter_extremes(no_below=2, no_above=0.85)
    corpus = [dictionary.doc2bow(doc) for doc in tokens_clean]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5,
                   random_state=42, passes=15, iterations=100,
                   alpha='auto', eta='auto', per_word_topics=True)
    return lda, dictionary, corpus

def get_dominant_topic(bow, lda_model):
    if not bow: return -1, 0.0
    dist = lda_model.get_document_topics(bow)
    if not dist: return -1, 0.0
    dom = max(dist, key=lambda x: x[1])
    return dom[0], dom[1]


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
import base64
uny_img_path = "assets/uny_campus.png"
if os.path.exists(uny_img_path):
    with open(uny_img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    bg_style = (
        "background-image: linear-gradient("
        "to right, rgba(26,34,53,0.92) 0%, rgba(13,59,110,0.75) 50%, rgba(68,172,255,0.30) 100%"
        f"), url('data:image/jpeg;base64,{img_b64}');"
        "background-size: cover; background-position: center;"
    )
else:
    bg_style = "background: linear-gradient(135deg, #1A2235 0%, #0D3B6E 60%, #44ACFF 100%);"

st.markdown(f"""
<div style="
    {bg_style}
    border-radius: 22px;
    padding: 52px 48px 40px 48px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 48px rgba(0,30,80,0.30);
">
    <div style="position:absolute;top:-80px;right:-80px;width:260px;height:260px;
        background:rgba(68,172,255,0.12);border-radius:50%;"></div>
    <div style="position:absolute;bottom:-60px;right:160px;width:180px;height:180px;
        background:rgba(255,214,10,0.07);border-radius:50%;"></div>
    <div style="display:inline-block;background:rgba(254,158,199,0.22);
        border:1px solid rgba(254,158,199,0.5);color:#FE9EC7;
        font-size:0.72rem;font-weight:700;padding:5px 14px;
        border-radius:20px;margin-bottom:16px;letter-spacing:0.1em;
        text-transform:uppercase;">Analisis Data Tak Terstruktur</div>
    <div style="font-size:2.2rem;font-weight:800;line-height:1.2;
        margin-bottom:12px;text-shadow:0 2px 12px rgba(0,0,0,0.3);">
        Analisis Sentimen<br>Kebijakan Triple Gate UNY</div>
    <p style="font-size:0.9rem;opacity:0.88;margin-bottom:20px;font-weight:400;max-width:600px;">
        Menganalisis opini publik terhadap Sistem Tiga Pintu Universitas Negeri
        Yogyakarta melalui komentar TikTok menggunakan Text Mining, SVM, dan LDA.
    </p>
    <div style="display:flex;flex-wrap:wrap;gap:8px;">
        <span style="background:rgba(255,255,255,0.15);border-radius:30px;
            padding:6px 16px;font-size:0.77rem;font-weight:700;color:#F9F6C4;
            border:1px solid rgba(255,255,255,0.2);">NLP Bahasa Indonesia</span>
        <span style="background:rgba(255,255,255,0.15);border-radius:30px;
            padding:6px 16px;font-size:0.77rem;font-weight:700;color:#F9F6C4;
            border:1px solid rgba(255,255,255,0.2);">Support Vector Machine</span>
        <span style="background:rgba(255,255,255,0.15);border-radius:30px;
            padding:6px 16px;font-size:0.77rem;font-weight:700;color:#F9F6C4;
            border:1px solid rgba(255,255,255,0.2);">Topic Modelling LDA</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UPLOAD SECTION — single unified card
# ─────────────────────────────────────────────
if 'uploaded_file_bytes' not in st.session_state:
    st.session_state['uploaded_file_bytes'] = None

st.markdown(
    '<div style="background:#F8FBFF;border-radius:18px;padding:24px 26px;border:1px solid #D8E9FF;margin-bottom:18px;">'
    '<p style="font-size:1.4rem;font-weight:800;color:#00509D;margin-bottom:6px;">Upload Dataset</p>'
    '<p style="font-size:0.78rem;color:#1e2541;margin-bottom:20px;">'
    'Unggah file CSV komentar TikTok untuk memulai analisis sentimen kebijakan Triple Gate UNY'
    '</p>'

    '<p style="font-size:0.8rem;font-weight:700;color:#1e2541;letter-spacing:0.05em;'
    'text-transform:uppercase;margin-bottom:12px;">'
    'Deskripsi Variabel Dataset'
    '</p>'

    '<div style="display:flex;gap:14px;flex-wrap:wrap;">'

        '<div style="flex:1;min-width:220px;background:white;border:1px solid #588ccf;'
        'border-radius:10px;padding:12px 14px;">'
            '<span style="font-size:0.78rem;color:#1e2541;">'
            '🔹 <b>Text</b>  (Komentar TikTok)'
            '</span>'
        '</div>'

        '<div style="flex:1;min-width:220px;background:white;border:1px solid #588ccf;'
        'border-radius:10px;padding:12px 14px;">'
            '<span style="font-size:0.78rem;color:#1e2541;">'
            '🔹 <b>DiggCount</b>  (Jumlah like komentar)' 
            '</span>'
        '</div>'

        '<div style="flex:1;min-width:220px;background:white;border:1px solid #588ccf;'
        'border-radius:10px;padding:12px 14px;">'
            '<span style="font-size:0.78rem;color:#1e2541;">'
            '▫️ Kolom lain (diabaikan otomatis)'
            '</span>'
        '</div>'

    '</div>'

    '<p style="margin-top:16px;font-size:0.72rem;color:#6B8DB3;font-weight:500;">'
    'Pastikan file berformat CSV dan memiliki kolom yang diperlukan'
    '</p>'

    '</div>',
    unsafe_allow_html=True
)
st.markdown("""
<div style="
    display:inline-block;
    padding:8px 18px;
    border-radius:999px;
    background:#1f76c9;
    color:white;
    border:1.5px solid #B7D8FF;
    font-size:0.95rem;
    font-weight:600;
    margin-bottom:10px;
    box-shadow:0 4px 12px rgba(68,172,255,0.25);
">
    🗎 Upload File CSV Komentar TikTok
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["csv"]
)


# ─────────────────────────────────────────────
# NO FILE UPLOADED
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style="background:#FFFFFF;border-radius:18px;padding:48px 40px;
        text-align:center;box-shadow:0 6px 24px rgba(68,172,255,0.10);
        margin-top:12px;border:1.5px dashed #C5DCFF;">
        <p style="font-size:1.05rem;font-weight:700;color:#0D3B6E;margin-bottom:8px;">
            Belum ada dataset yang diunggah
        </p>
        <p style="font-size:0.86rem;color:#5A7BA0;font-weight:500;">
            Upload file CSV di atas untuk memulai analisis sentimen Triple Gate UNY
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;margin-top:40px;padding:20px;
        background:linear-gradient(135deg,#ffddf3,#E8F4FF);
        border-radius:16px;border:1.5px solid #ffb8e6;">
        <p style="color:#5A7BA0;font-size:0.84rem;font-weight:600;margin:0;">
            Developed with ♡ by
            <strong style="color:#0D3B6E;">Fiki Vania Arun Fadila</strong> &amp;
            <strong style="color:#0D3B6E;">Divna Widyastuti</strong><br>
            <span style="font-size:0.78rem;color:#7A9AC0;">Universitas Negeri Yogyakarta</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# LOAD & PROCESS
# ─────────────────────────────────────────────
with st.spinner("Memuat dan memproses data..."):
    file_bytes = uploaded_file.read()
    df = load_and_preprocess(file_bytes)

with st.spinner("Melatih model SVM..."):
    tfidf, svm_model, acc, prec, rec, f1, cm, sentimen_pred = train_svm_model(
        df['text_stemmed'].tolist(), df['sentimen'].tolist()
    )
    df['sentimen_pred'] = sentimen_pred

df_neg = df[df['sentimen_pred'] == 'negatif'].copy().reset_index(drop=True)

with st.spinner("Melatih model LDA..."):
    lda_model, dictionary_neg, corpus_neg = train_lda_model(df_neg['tokens'].tolist())

if lda_model is not None:
    _, stop_words_id = get_stemmer_and_stopwords()
    tokens_all_neg = [[t for t in tok if len(t) > 2] for tok in df_neg['tokens'].tolist()]
    corpus_all_neg = [dictionary_neg.doc2bow(doc) for doc in tokens_all_neg]
    topic_assignments = [get_dominant_topic(bow, lda_model) for bow in corpus_all_neg]
    df_neg['topic_id']   = [t[0] for t in topic_assignments]
    df_neg['topic_prob'] = [t[1] for t in topic_assignments]
    df_neg['topic_label'] = df_neg['topic_id'].apply(
        lambda x: topic_labels[x] if 0 <= x < len(topic_labels) else 'Tidak Terklasifikasi'
    )
    df_neg['is_aspirasi_lda'] = df_neg['topic_label'].apply(
        lambda x: 1 if x == 'Topik 5: Aspirasi dan Tuntutan' else 0
    )
    df_aspirasi = df_neg[df_neg['is_aspirasi_lda'] == 1].copy()
    topic_dist    = df_neg['topic_label'].value_counts().reset_index()
    topic_dist.columns = ['Topik','Jumlah']
    digg_by_topic = df_neg.groupby('topic_label')['diggCount'].mean().reset_index()
    digg_by_topic.columns = ['Topik','Rata-rata Like']
    digg_by_topic = digg_by_topic.sort_values('Rata-rata Like')
    n_aspirasi = int(df_neg['is_aspirasi_lda'].sum())
else:
    n_aspirasi = 0
    df_aspirasi = pd.DataFrame()
    topic_dist = pd.DataFrame()
    digg_by_topic = pd.DataFrame()

sent_counts = df['sentimen'].value_counts()
neg_pct = sent_counts.get('negatif', 0) / len(df) * 100
pos_pct = sent_counts.get('positif', 0) / len(df) * 100


# ─────────────────────────────────────────────
# HERO STATS BANNER (after data loaded)
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1A2235 0%,#0D3B6E 60%,#44ACFF 100%);
    border-radius:18px;padding:26px 38px;margin-bottom:28px;color:white;
    box-shadow:0 10px 36px rgba(0,30,80,0.25);">
    <p style="font-size:1.5rem;font-weight:800;color:#FE9EC7;
        text-transform:letter-spacing:0.1em;margin:0 0 10px 0;">
        Hasil Analisis
    </p>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,0.14);border-radius:30px;
            padding:7px 18px;font-size:0.8rem;font-weight:700;color:#F9F6C4;
            border:1px solid rgba(255,255,255,0.2);">
            {len(df):,} Komentar Dianalisis
        </span>
        <span style="background:rgba(255,255,255,0.14);border-radius:30px;
            padding:7px 18px;font-size:0.8rem;font-weight:700;color:#F9F6C4;
            border:1px solid rgba(255,255,255,0.2);">
            {neg_pct:.1f}% Sentimen Negatif
        </span>
        <span style="background:rgba(255,255,255,0.14);border-radius:30px;
            padding:7px 18px;font-size:0.8rem;font-weight:700;color:#F9F6C4;
            border:1px solid rgba(255,255,255,0.2);">
            Akurasi SVM {acc*100:.1f}%
        </span>
        <span style="background:rgba(255,255,255,0.14);border-radius:30px;
            padding:7px 18px;font-size:0.8rem;font-weight:700;color:#F9F6C4;
            border:1px solid rgba(255,255,255,0.2);">
            {n_aspirasi} Aspirasi Teridentifikasi
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: info card
# ─────────────────────────────────────────────
def info_card(label, value, desc="", color="#44ACFF"):
    return f"""
    <div style="background:#FFFFFF;border-radius:18px;padding:22px 24px 18px 24px;
        box-shadow:0 8px 28px rgba(0,80,157,0.13),0 2px 6px rgba(68,172,255,0.08);
        border-left:5px solid {color};margin-bottom:14px;">
        <p style="font-size:0.7rem;font-weight:700;color:{color};
            text-transform:uppercase;letter-spacing:0.07em;margin:0 0 4px 0;">{label}</p>
        <p style="font-size:1.9rem;font-weight:800;color:#0D3B6E;margin:0 0 3px 0;line-height:1.1;">{value}</p>
        <p style="font-size:0.76rem;color:#5A7BA0;font-weight:500;margin:0;">{desc}</p>
    </div>
    """


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Data & EDA  ",
    "  Preprocessing  ",
    "  Klasifikasi SVM  ",
    "  Topic Modelling LDA  ",
    "  Insight & Kesimpulan  ",
])

# ══════════════════════════════════════════════
# TAB 1 — DATA & EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Ringkasan Dataset</div>', unsafe_allow_html=True)
    st.markdown("")

    cols_stat = st.columns(4)
    stat_items = [
        ("Total Komentar",  f"{len(df):,}",               "komentar valid setelah cleaning",         "#44ACFF"),
        ("Total Like",      f"{df['diggCount'].sum():,}",  "akumulasi diggCount seluruh komentar",    "#00509D"),
        ("Rata-rata Like",  f"{df['diggCount'].mean():.1f}","per komentar",                           "#FF8FAB"),
        ("Like Tertinggi",  f"{df['diggCount'].max():,}",  "komentar paling viral",                   "#E55579"),
    ]
    for col, (label, val, desc, clr) in zip(cols_stat, stat_items):
        with col:
            st.markdown(info_card(label, val, desc, clr), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Contoh Data</div>', unsafe_allow_html=True)
    st.dataframe(df[['text','diggCount']].head(10), use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Distribusi Panjang Komentar</div>',
                    unsafe_allow_html=True)
        df['panjang_teks'] = df['text'].apply(len)
        df['n_tokens']     = df['tokens'].apply(len)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='#F7FAFF')
        fig.patch.set_facecolor('#F7FAFF')

        med_ori = df['panjang_teks'].median()
        axes[0].hist(df['panjang_teks'], bins=40, color=C_BLUE,
                     edgecolor='white', alpha=0.88, linewidth=0.5)
        axes[0].axvline(med_ori, color='#FF8FAB', linestyle='--', linewidth=2,
                        label=f'Median: {med_ori:.0f}')
        axes[0].legend(fontsize=9, framealpha=0, labelcolor='#3A4A62')
        apply_chart_style(axes[0], 'Teks Asli (karakter)', 'Panjang', 'Frekuensi')

        med_tok = df['n_tokens'].median()
        axes[1].hist(df['n_tokens'], bins=30, color='#FF8FAB',
                     edgecolor='white', alpha=0.88, linewidth=0.5)
        axes[1].axvline(med_tok, color=C_BLUE, linestyle='--', linewidth=2,
                        label=f'Median: {med_tok:.0f}')
        axes[1].legend(fontsize=9, framealpha=0, labelcolor='#3A4A62')
        apply_chart_style(axes[1], 'Jumlah Token per Komentar', 'Token', 'Frekuensi')

        plt.tight_layout(pad=2.5)
        chart_shadow(fig)

    with col_b:
        st.markdown('<div class="section-title">Distribusi diggCount</div>',
                    unsafe_allow_html=True)
        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), facecolor='#F7FAFF')
        fig2.patch.set_facecolor('#F7FAFF')

        # Clip at 95th percentile so the bulk of the distribution is visible
        p95 = df['diggCount'].quantile(0.95)
        digg_clipped = df['diggCount'].clip(upper=p95)
        med_digg  = df['diggCount'].median()
        mean_digg = df['diggCount'].mean()

        axes2[0].hist(digg_clipped, bins=40, color=C_YELLOW,
                      edgecolor='white', alpha=0.92, linewidth=0.5)
        axes2[0].axvline(min(med_digg, p95), color='#FF8FAB', linestyle='--',
                         linewidth=2, label=f'Median: {med_digg:.0f}')
        axes2[0].axvline(min(mean_digg, p95), color=C_BLUE, linestyle=':',
                         linewidth=2, label=f'Mean: {mean_digg:.1f}')
        axes2[0].set_xlim(0, p95 * 1.08)
        axes2[0].xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x):,}')
        )
        axes2[0].legend(fontsize=9, framealpha=0, labelcolor='#3A4A62')
        apply_chart_style(axes2[0], 'Histogram diggCount (95% data)', 'diggCount', 'Frekuensi')

        axes2[1].boxplot(df['diggCount'], patch_artist=True,
                          widths=0.55,
                          boxprops=dict(facecolor=C_YELLOW, alpha=0.85, linewidth=0),
                          medianprops=dict(color='#FF8FAB', linewidth=2.5),
                          whiskerprops=dict(color='#AAB4C4', linewidth=1.5),
                          capprops=dict(color='#AAB4C4', linewidth=1.5),
                          flierprops=dict(marker='o', markersize=3.5,
                                          markerfacecolor=C_BLUE, alpha=0.5))
        apply_chart_style(axes2[1], 'Boxplot diggCount', '', 'diggCount')
        plt.tight_layout(pad=2.5)
        chart_shadow(fig2)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Word Cloud</div>', unsafe_allow_html=True)
        all_tokens = [t for tokens in df['tokens'] for t in tokens if len(t) > 2]
        text_wc = ' '.join(all_tokens)
        wc = WordCloud(
            width=900, height=420, background_color='#F7FAFF',
            colormap=None, max_words=150,
            color_func=lambda *args, **kwargs: np.random.choice(
                [C_BLUE, '#FF8FAB', C_DARK, '#0D3B6E', C_LBLUE]
            )
        ).generate(text_wc)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4.5), facecolor='#F7FAFF')
        fig_wc.patch.set_facecolor('#F7FAFF')
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title('Word Cloud Komentar Triple Gate UNY (Pasca Preprocessing)',
                         fontsize=11.5, fontweight='800', color='#0D3B6E', pad=12)
        chart_shadow(fig_wc)

    with col_d:
        st.markdown('<div class="section-title">Top 20 Kata Paling Sering</div>',
                    unsafe_allow_html=True)
        freq = Counter(all_tokens)
        top20 = freq.most_common(20)
        words_top, counts_top = zip(*top20)
        bar_colors = ['#FF8FAB' if w in kata_negatif else
                      "#98C9E4" if w in kata_positif else C_BLUE
                      for w in words_top]
        fig_freq, ax_freq = plt.subplots(figsize=(8, 7.5), facecolor='#F7FAFF')
        fig_freq.patch.set_facecolor('#F7FAFF')
        bars = ax_freq.barh(list(words_top)[::-1], list(counts_top)[::-1],
                            color=bar_colors[::-1], alpha=0.92, edgecolor='white',
                            linewidth=0.5, height=0.72)
        for bar, val in zip(bars, list(counts_top)[::-1]):
            ax_freq.text(val + 0.4, bar.get_y() + bar.get_height()/2,
                         str(val), va='center', fontsize=9, color='#3A4A62', fontweight='600')
        apply_chart_style(ax_freq, 'Top 20 Kata Pasca-Preprocessing', 'Frekuensi', '')
        ax_freq.grid(axis='x', color='#DCE8F5', linewidth=0.7)
        ax_freq.grid(axis='y', visible=False)
        legend_elems = [
            mpatches.Patch(color='#FF8FAB', label='Kata negatif'),
            mpatches.Patch(color='#98C9E4', label='Kata positif'),
            mpatches.Patch(color=C_BLUE,    label='Netral'),
        ]
        ax_freq.legend(handles=legend_elems, fontsize=8.5, framealpha=0.9,
                       facecolor='#F7FAFF', edgecolor='#D5E4F0', loc='lower right',
                       prop={'weight':'600'})
        plt.tight_layout(pad=2.5)
        chart_shadow(fig_freq)

    st.markdown("---")
    st.markdown('<div class="section-title">Statistik Deskriptif</div>', unsafe_allow_html=True)
    st.dataframe(df[['diggCount']].describe().T, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Ringkasan Tahapan Preprocessing</div>',
                unsafe_allow_html=True)
    st.markdown("")

    prep_stats = [
        ("Data Awal",              "1.005", "total komentar mentah dari TikTok",    "#44ACFF"),
        ("Setelah Cleaning",       "999",   "setelah hapus noise dan duplikat awal","#00509D"),
        ("Setelah Hapus Duplikat", "965",   "komentar unik tersisa",                "#EFC52B"),
        ("Data Final",             f"{len(df):,}", "siap untuk analisis lanjutan", "#FF8FAB"),
    ]
    cols_p = st.columns(4)
    for col, (label, val, desc, clr) in zip(cols_p, prep_stats):
        with col:
            st.markdown(info_card(label, val, desc, clr), unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-title">Contoh Hasil Preprocessing</div>',
                    unsafe_allow_html=True)
        sample = df[['text','text_clean','text_normalized',
                     'text_stemmed','tokens']].head(8)
        st.dataframe(sample, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Detail Setiap Tahap</div>',
                    unsafe_allow_html=True)
        tahap_data = [
            ("1", "Cleaning Text",       "Hapus emoji, URL, mention, simbol, angka - huruf kecil",   "#44ACFF"),
            ("2", "Normalisasi Slang",   f"Kamus {len(slang_dict)} kata slang ke bentuk baku",       "#EFC52B"),
            ("3", "Stopword Removal",    "896 kata henti, 9 negasi dipertahankan",                   "#FF8FAB"),
            ("4", "Stemming",            "Sastrawi Stemmer - imbuhan ke kata dasar",                  "#44ACFF"),
            ("5", "Tokenization",        "Kalimat dipecah menjadi list token per kata",              "#EFC52B"),
            ("6", "Filtering",           "Hapus komentar kosong atau tanpa token",                   "#FF8FAB"),
        ]
        for step, nama, ket, clr in tahap_data:
            st.markdown(
                f"<div style='display:flex;align-items:flex-start;background:{clr}18;"
                f"border-left:4px solid {clr};border-radius:12px;"
                f"padding:12px 16px;margin-bottom:8px;'>"
                f"<div style='min-width:30px;height:30px;border-radius:50%;"
                f"background:{clr};display:flex;align-items:center;justify-content:center;"
                f"font-weight:800;font-size:0.76rem;color:#1A2235;margin-right:12px;"
                f"flex-shrink:0;margin-top:1px;'>{step}</div>"
                f"<div><p style='margin:0;font-weight:700;font-size:0.84rem;"
                f"color:#0D3B6E;'>{nama}</p>"
                f"<p style='margin:0;font-size:0.77rem;color:#5A7BA0;font-weight:500;'>{ket}</p>"
                f"</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<div class="section-title">Token Dominan Pasca-Stemming</div>',
                unsafe_allow_html=True)
    all_stemmed = [t for tokens in df['tokens'] for t in tokens if len(t) > 2]
    freq_stem = Counter(all_stemmed).most_common(15)
    words_s, counts_s = zip(*freq_stem)
    clrs_s = ['#FF8FAB' if w in kata_negatif else
              '#98C9E4' if w in kata_positif else C_LBLUE
              for w in words_s]
    fig_s, ax_s = plt.subplots(figsize=(13, 4.5), facecolor='#F7FAFF')
    fig_s.patch.set_facecolor('#F7FAFF')
    bars_s = ax_s.bar(words_s, counts_s, color=clrs_s, alpha=0.92,
                      edgecolor='white', linewidth=0.5, width=0.66)
    for bar, val in zip(bars_s, counts_s):
        ax_s.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                  str(val), ha='center', fontsize=9.5, color='#3A4A62', fontweight='600')
    apply_chart_style(ax_s, 'Frekuensi Token Pasca-Stemming', 'Token', 'Frekuensi')
    ax_s.grid(axis='x', visible=False)
    plt.tight_layout(pad=2.5)
    chart_shadow(fig_s)


# ══════════════════════════════════════════════
# TAB 3 — SVM
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)
    st.markdown("")

    svm_stats = [
        ("Total Komentar",   f"{len(df):,}",
         "seluruh komentar teranalisis", "#44ACFF"),
        ("Sentimen Negatif", f"{sent_counts.get('negatif',0):,}",
         f"{sent_counts.get('negatif',0)/len(df)*100:.1f}% dari total", "#FF8FAB"),
        ("Sentimen Positif", f"{sent_counts.get('positif',0):,}",
         f"{sent_counts.get('positif',0)/len(df)*100:.1f}% dari total", "#98C9E4"),
    ]
    cols_sv = st.columns(3)
    for col, (label, val, desc, clr) in zip(cols_sv, svm_stats):
        with col:
            st.markdown(info_card(label, val, desc, clr), unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        colors_sent = ['#FF8FAB' if s == 'negatif' else '#98C9E4'
                       for s in sent_counts.index]
        fig_s1, axes_s1 = plt.subplots(1, 2, figsize=(11, 4.5), facecolor='#F7FAFF')
        fig_s1.patch.set_facecolor('#F7FAFF')
        fig_s1.suptitle('Distribusi Sentimen Komentar Triple Gate UNY',
                         fontsize=11.5, fontweight='800', color='#0D3B6E', y=1.02)

        bars_sv = axes_s1[0].bar(sent_counts.index, sent_counts.values,
                                  color=colors_sent, edgecolor='white',
                                  linewidth=0.5, width=0.52)
        for bar, (label, val) in zip(bars_sv, zip(sent_counts.index, sent_counts.values)):
            axes_s1[0].text(bar.get_x() + bar.get_width()/2,
                             val + 5, f'{val}\n({val/len(df)*100:.1f}%)',
                             ha='center', fontsize=9.5, fontweight='700', color='#0D3B6E')
        apply_chart_style(axes_s1[0], 'Jumlah Komentar', '', 'Jumlah')

        wedge_colors = ['#FF8FAB' if s == 'negatif' else '#98C9E4'
                        for s in sent_counts.index]
        wedges, texts, autotexts = axes_s1[1].pie(
            sent_counts.values, labels=sent_counts.index,
            colors=wedge_colors, autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':3.5},
            textprops={'fontsize':10.5, 'fontweight':'700', 'color':'#0D3B6E'},
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color('#0D3B6E')
            at.set_fontsize(11)
            at.set_fontweight('800')
        axes_s1[1].set_title('Proporsi Sentimen', fontsize=11.5, fontweight='800',
                              color='#0D3B6E', pad=10)
        plt.tight_layout(pad=2.5)
        chart_shadow(fig_s1)

    with col_b:
        st.markdown('<div class="section-title">Metrik Evaluasi Model SVM</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        metrics = [
            ("Accuracy",  f"{acc*100:.2f}%",  "#00509D", "Proporsi prediksi benar keseluruhan"),
            ("Precision", f"{prec:.4f}",       "#FF8FAB", "Ketepatan prediksi positif"),
            ("Recall",    f"{rec:.4f}",        "#89C2FF", "Kelengkapan prediksi aktual"),
            ("F1-Score",  f"{f1:.4f}",         "#E4C00A", "Rata-rata harmonik precision & recall"),
        ]
        m_cols = st.columns(2)
        for i, (name, val, color, hint) in enumerate(metrics):
            with m_cols[i % 2]:
                st.markdown(
                    f"<div style='background:{color}18;border:2.5px solid {color};"
                    f"border-radius:16px;padding:20px 18px;text-align:center;"
                    f"margin-bottom:12px;box-shadow:0 4px 14px {color}30;'>"
                    f"<p style='font-size:0.7rem;font-weight:700;color:{color};"
                    f"text-transform:uppercase;letter-spacing:0.08em;margin:0 0 6px 0;'>{name}</p>"
                    f"<p style='font-size:1.95rem;font-weight:800;color:#0D3B6E;margin:0 0 4px 0;'>{val}</p>"
                    f"<p style='font-size:0.72rem;color:#5A7BA0;margin:0;font-weight:500;'>{hint}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        st.markdown(
            "<div style='background:#EEF5FF;border-radius:12px;padding:13px 18px;"
            "font-size:0.79rem;color:#3A5A80;font-weight:600;border:1px solid #C5DCFF;'>"
            "Kernel: Linear &nbsp;|&nbsp; C: 1.0 &nbsp;|&nbsp; class_weight: balanced &nbsp;|&nbsp; random_state: 42"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cmap_custom = matplotlib.colors.LinearSegmentedColormap.from_list(
            'custom', ["#FF8FAB", '#0D3B6E'], N=256
        )
        fig_cm_plot, ax_cm = plt.subplots(figsize=(6, 4.5), facecolor='#F7FAFF')
        fig_cm_plot.patch.set_facecolor('#F7FAFF')
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_custom,
                    xticklabels=['negatif','positif'],
                    yticklabels=['negatif','positif'],
                    linewidths=3, linecolor='white',
                    ax=ax_cm, annot_kws={'size':17, 'weight':'800', 'color':'white'})
        ax_cm.set_xlabel('Prediksi', fontsize=10.5, color='#3A4A62', fontweight='700')
        ax_cm.set_ylabel('Aktual',   fontsize=10.5, color='#3A4A62', fontweight='700')
        ax_cm.set_title('Confusion Matrix - SVM Triple Gate UNY',
                         fontsize=11.5, fontweight='800', color='#0D3B6E', pad=14)
        ax_cm.tick_params(colors='#3A4A62', labelsize=10)
        plt.tight_layout(pad=2.5)
        chart_shadow(fig_cm_plot)

    with col_d:
        st.markdown('<div class="section-title">Total Like per Sentimen</div>',
                    unsafe_allow_html=True)
        digg_by_sent = df.groupby('sentimen_pred')['diggCount'].sum().reset_index()
        digg_by_sent.columns = ['Sentimen','Total Like']
        colors_dl = ['#FF8FAB' if x == 'negatif' else '#98C9E4'
                     for x in digg_by_sent['Sentimen']]
        fig_like, ax_like = plt.subplots(figsize=(6, 4.5), facecolor='#F7FAFF')
        fig_like.patch.set_facecolor('#F7FAFF')
        bars_l = ax_like.bar(digg_by_sent['Sentimen'], digg_by_sent['Total Like'],
                              color=colors_dl, edgecolor='white',
                              linewidth=0.5, width=0.52)
        for bar, val in zip(bars_l, digg_by_sent['Total Like']):
            ax_like.text(bar.get_x() + bar.get_width()/2,
                          val + 50, f'{val:,.0f}',
                          ha='center', fontsize=12, fontweight='800', color='#0D3B6E')
        apply_chart_style(ax_like, 'Total Like per Sentimen', '', 'Total diggCount')
        plt.tight_layout(pad=2.5)
        chart_shadow(fig_like)


# ══════════════════════════════════════════════
# TAB 4 — LDA
# ══════════════════════════════════════════════
with tab4:
    if lda_model is None:
        st.warning("LDA tidak dapat dijalankan karena data tidak mencukupi.")
    else:
        st.markdown('<div class="section-title">Topic Modelling LDA - Komentar Negatif</div>',
                    unsafe_allow_html=True)
        st.markdown("")

        lda_stats = [
            ("Total Komentar Negatif", f"{len(df_neg):,}",  "input untuk LDA",            "#FF8FAB"),
            ("Jumlah Topik",           "5",                  "topik LDA teridentifikasi",  "#44ACFF"),
            ("Komentar Aspirasi",      f"{n_aspirasi:,}",   "Topik 5 - tuntutan konkret", "#EFC52B"),
        ]
        cols_lda = st.columns(3)
        for col, (label, val, desc, clr) in zip(cols_lda, lda_stats):
            with col:
                st.markdown(info_card(label, val, desc, clr), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Topik dan Kata Kunci Dominan</div>',
                    unsafe_allow_html=True)
        for i, (topik, kata) in enumerate(topic_keywords.items()):
            clr = list(TOPIC_COLORS.values())[i]
            nama = topik.split(': ')[1]
            st.markdown(
                f"<div style='display:flex;align-items:center;background:{clr}20;"
                f"border-left:5px solid {clr};border-radius:12px;"
                f"padding:14px 20px;margin-bottom:8px;"
                f"box-shadow:0 3px 12px {clr}30;'>"
                f"<div style='min-width:34px;height:34px;border-radius:50%;"
                f"background:{clr};display:flex;align-items:center;justify-content:center;"
                f"font-weight:800;font-size:0.82rem;color:#1A2235;margin-right:16px;"
                f"flex-shrink:0;box-shadow:0 3px 8px {clr}50;'>{i+1}</div>"
                f"<div><p style='margin:0;font-weight:800;font-size:0.9rem;"
                f"color:#0D3B6E;'>{nama}</p>"
                f"<p style='margin:0;font-size:0.79rem;color:#5A7BA0;font-weight:500;'>"
                f"Kata kunci: {kata}</p>"
                f"</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">Jumlah Komentar per Topik</div>',
                        unsafe_allow_html=True)
            td_sorted = topic_dist.sort_values('Jumlah')
            clrs_t = [TOPIC_COLORS.get(t, '#D3D3D3') for t in td_sorted['Topik']]
            fig_t1, ax_t1 = plt.subplots(figsize=(8, 5), facecolor='#F7FAFF')
            fig_t1.patch.set_facecolor('#F7FAFF')
            bars_t1 = ax_t1.barh(td_sorted['Topik'], td_sorted['Jumlah'],
                                  color=clrs_t, alpha=0.92, edgecolor='white',
                                  linewidth=0.5, height=0.66)
            for bar, val in zip(bars_t1, td_sorted['Jumlah']):
                ax_t1.text(val + 1, bar.get_y() + bar.get_height()/2,
                           str(val), va='center', fontsize=9.5, color='#3A4A62', fontweight='600')
            apply_chart_style(ax_t1, 'Distribusi Komentar per Topik', 'Jumlah', '')
            ax_t1.grid(axis='y', visible=False)
            ax_t1.grid(axis='x', color='#DCE8F5', linewidth=0.7)
            plt.tight_layout(pad=2.5)
            chart_shadow(fig_t1)

        with col_b:
            st.markdown('<div class="section-title">Rata-rata Like per Topik</div>',
                        unsafe_allow_html=True)
            clrs_d = [TOPIC_COLORS.get(t, '#D3D3D3') for t in digg_by_topic['Topik']]
            fig_t2, ax_t2 = plt.subplots(figsize=(8, 5), facecolor='#F7FAFF')
            fig_t2.patch.set_facecolor('#F7FAFF')
            bars_t2 = ax_t2.barh(digg_by_topic['Topik'], digg_by_topic['Rata-rata Like'],
                                  color=clrs_d, alpha=0.92, edgecolor='white',
                                  linewidth=0.5, height=0.66)
            for bar, val in zip(bars_t2, digg_by_topic['Rata-rata Like']):
                ax_t2.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                           f'{val:.1f}', va='center', fontsize=9.5, color='#3A4A62', fontweight='600')
            apply_chart_style(ax_t2, 'Rata-rata Like per Topik', 'Rata-rata diggCount', '')
            ax_t2.grid(axis='y', visible=False)
            ax_t2.grid(axis='x', color='#DCE8F5', linewidth=0.7)
            plt.tight_layout(pad=2.5)
            chart_shadow(fig_t2)

        st.markdown("---")
        st.markdown('<div class="section-title">Komentar Aspirasi & Tuntutan (Topik 5)</div>',
                    unsafe_allow_html=True)
        asp_cols = st.columns(2)
        with asp_cols[0]:
            st.markdown(info_card("Komentar Aspirasi", f"{n_aspirasi:,}",
                                  "mengandung tuntutan konkret", "#EFC52B"), unsafe_allow_html=True)
        with asp_cols[1]:
            st.markdown(info_card("Bukan Aspirasi", f"{len(df_neg) - n_aspirasi:,}",
                                  "komentar negatif lainnya", "#FF8FAB"), unsafe_allow_html=True)
        if len(df_aspirasi) > 0:
            st.dataframe(
                df_aspirasi[['text','diggCount','topic_prob']]
                .sort_values('diggCount', ascending=False)
                .head(20)
                .rename(columns={'text':'Komentar',
                                  'diggCount':'Like',
                                  'topic_prob':'Probabilitas Topik'}),
                use_container_width=True
            )


# ══════════════════════════════════════════════
# TAB 5 — INSIGHT
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Temuan Utama</div>', unsafe_allow_html=True)
    st.markdown("")

    total_like_neg = df[df['sentimen_pred']=='negatif']['diggCount'].sum()
    total_like_pos = df[df['sentimen_pred']=='positif']['diggCount'].sum()

    findings = [
        ("#FF8FAB", "Dominasi Sentimen Negatif",
         f"{neg_pct:.1f}% dari {len(df):,} komentar bersifat negatif - "
         "mengindikasikan penolakan publik yang luas terhadap kebijakan Triple Gate UNY."),
        ("#00509D", "Performa Model SVM",
         f"Akurasi {acc*100:.2f}% membuktikan model mampu mengklasifikasikan "
         "sentimen komentar TikTok berbahasa Indonesia informal dengan sangat baik."),
        ("#FFE666", "Dukungan Komunitas terhadap Kritik",
         f"Komentar negatif meraih {total_like_neg:,} like vs {total_like_pos:,} like "
         "pada komentar positif - keresahan dirasakan secara kolektif dan masif."),
    ]
    if lda_model is not None and len(topic_dist) > 0:
        top_t  = topic_dist.sort_values('Jumlah', ascending=False).iloc[0]
        top_dl = digg_by_topic.sort_values('Rata-rata Like', ascending=False).iloc[0]
        findings += [
            ("#89C2FF", "Topik Paling Banyak Dibahas",
             f"{top_t['Topik']} mendominasi dengan {top_t['Jumlah']} komentar, "
             "menunjukkan dampak terbesar dirasakan pada mobilitas kampus."),
            ("#98C9E4", "Topik Paling Resonan di Publik",
             f"{top_dl['Topik']} memperoleh rata-rata {top_dl['Rata-rata Like']:.1f} like - "
             "isu keamanan dan kenyamanan kampus lebih dalam dirasakan audiens."),
        ]
        if n_aspirasi > 0:
            findings.append(
                ("#C9B8FF", "Aspirasi Publik Teridentifikasi",
                 f"{n_aspirasi} komentar mengandung tuntutan konkret terkait "
                 "keterlambatan, pemborosan BBM, dan kesulitan parkir akibat kebijakan.")
            )

    for color, title, desc in findings:
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;background:{color}50;"
            f"border:2px solid {color}100;border-radius:16px;"
            f"padding:20px 24px;margin-bottom:12px;"
            f"box-shadow:0 4px 16px {color}28;'>"
            f"<div style='min-width:12px;height:12px;border-radius:50%;"
            f"background:{color};margin-top:5px;margin-right:16px;flex-shrink:0;'></div>"
            f"<div><p style='margin:0 0 5px 0;font-weight:800;font-size:0.9rem;"
            f"color:#0D3B6E;'>{title}</p>"
            f"<p style='margin:0;font-size:0.82rem;color:#3A5A80;line-height:1.6;"
            f"font-weight:500;'>{desc}</p></div></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="section-title">Rekomendasi untuk Pihak Kampus</div>',
                unsafe_allow_html=True)
    st.markdown("")

    recs = [
        ("#00509D", "Akses Jalan & Mobilitas",
         "Tinjau ulang rute dan sistem buka-tutup pintu agar lebih efisien dan tidak membebani warga kampus."),
        ("#FF8FAB", "Kemacetan & Antrian",
         "Tambah petugas pada jam sibuk dan optimalkan manajemen arus kendaraan di setiap pintu masuk."),
        ("#FFE35A", "Keamanan & Fasilitas",
         "Perbaiki fasilitas parkir dan pastikan lingkungan kampus tetap aman dan nyaman untuk semua."),
        ("#98C9E4", "Aspirasi Mahasiswa",
         "Buka ruang dialog dengan mahasiswa untuk menyerap aspirasi secara langsung dan transparan."),
    ]
    rec_cols = st.columns(2)
    for i, (clr, title, desc) in enumerate(recs):
        with rec_cols[i % 2]:
            st.markdown(
                f"<div style='background:{clr}50;border-top:5px solid {clr};"
                f"border:1.5px solid {clr}100;border-top:5px solid {clr};"
                f"border-radius:16px;padding:20px 20px 16px 20px;margin-bottom:14px;"
                f"box-shadow:0 4px 18px {clr}28;'>"
                f"<p style='font-weight:800;font-size:0.87rem;color:#0D3B6E;margin:0 0 7px 0;'>"
                f"{title}</p>"
                f"<p style='font-size:0.79rem;color:#3A5A80;margin:0;line-height:1.6;font-weight:500;'>"
                f"{desc}</p></div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<div class="section-title">Keterbatasan Analisis</div>',
                unsafe_allow_html=True)
    limits = [
        "Data hanya berasal dari 3 video TikTok - belum merepresentasikan keseluruhan opini publik.",
        "Pelabelan sentimen menggunakan lexicon-based tanpa validasi manual menyeluruh.",
        "Distribusi data sangat tidak seimbang (95.3% negatif) sehingga performa klasifikasi positif terbatas.",
        "Sarkasme dan ironi dalam komentar belum terdeteksi secara optimal.",
    ]
    for limit in limits:
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;background:#F4F8FF;"
            f"border-left:4px solid #AAB4C4;border-radius:10px;"
            f"padding:12px 18px;margin-bottom:8px;'>"
            f"<div style='min-width:8px;height:8px;border-radius:50%;background:#AAB4C4;"
            f"margin-top:7px;margin-right:14px;flex-shrink:0;'></div>"
            f"<p style='font-size:0.83rem;color:#3A5A80;margin:0;line-height:1.6;font-weight:500;'>"
            f"{limit}</p></div>",
            unsafe_allow_html=True
        )


    st.markdown("""
    <div style="text-align:center;margin-top:40px;padding:24px;
        background:linear-gradient(135deg,#ffddf3,#E8F4FF);
        border-radius:18px;border:1.5px solid #ffb8e6;">
        <p style="font-size:1rem;font-weight:800;
            color:#0D3B6E;margin-bottom:6px;">
            ✦ End of Analysis ✦
        </p>
    </div>
    """, unsafe_allow_html=True)