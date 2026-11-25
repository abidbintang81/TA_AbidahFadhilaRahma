"""Pemuatan data, seleksi data, dan preprocessing teks (Bab III.2.2–III.2.3, III.3)."""
import html
import re

import nltk
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import config


def ensure_nltk():
    """Unduh resource NLTK bila belum ada."""
    for res, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4"),
                      ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


# ---------------------------------------------------------------------------
# 1. Pemuatan & penyeragaman kolom
# ---------------------------------------------------------------------------
def load_platform(path, platform, cols):
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "review_id": df[cols["id"]].astype(str),
        "platform": platform,
        "text_raw": df[cols["text"]],
        "rating_raw": pd.to_numeric(df[cols["rating"]], errors="coerce"),
        "date": df[cols["date"]],
        "spoiler": df[cols["spoiler"]],
        "votes": df[cols["votes"]],
    })
    factor = config.LETTERBOXD_RATING_FACTOR if platform == "Letterboxd" else 1
    out["rating_10"] = out["rating_raw"] * factor   # normalisasi rating ke skala 1–10
    return out


def load_data():
    imdb = load_platform(config.IMDB_CSV, "IMDb", config.IMDB_COLS)
    lbx = load_platform(config.LETTERBOXD_CSV, "Letterboxd", config.LETTERBOXD_COLS)
    return pd.concat([imdb, lbx], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Pembersihan ringan (dipakai BERTopic & sebagai langkah awal LDA)
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
WS_RE = re.compile(r"\s+")


def light_clean(text):
    """Hapus tag HTML (<br/>), entitas HTML (&#39;), dan URL. Struktur kalimat dipertahankan."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = TAG_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


def detect_lang(text):
    DetectorFactory.seed = config.LANGDETECT_SEED
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# 3. Seleksi data dengan pencatatan jumlah per tahap
# ---------------------------------------------------------------------------
def select_data(df):
    """Hapus data kosong, duplikat, dan non-Inggris. Mengembalikan (df_bersih, tabel_tahapan)."""
    stages = []

    def log(stage, d):
        row = {"tahap": stage}
        for p in ["IMDb", "Letterboxd"]:
            row[p] = int((d["platform"] == p).sum())
        row["Total"] = int(len(d))
        stages.append(row)

    log("Data awal", df)
    df = df.copy()
    df["text_clean"] = df["text_raw"].apply(light_clean)
    df = df[df["text_clean"].str.len() > 0]
    log("Setelah hapus ulasan kosong", df)
    df = df.drop_duplicates(subset=["platform", "text_clean"])
    log("Setelah hapus duplikat", df)
    df["lang"] = df["text_clean"].apply(detect_lang)
    df = df[df["lang"] == config.LANGUAGE]
    log("Setelah hapus non-Inggris", df)
    return df.reset_index(drop=True), stages


# ---------------------------------------------------------------------------
# 4. Preprocessing penuh untuk LDA
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()
_stop = None


def get_stopwords():
    global _stop
    if _stop is None:
        _stop = set(stopwords.words("english")) | set(config.DOMAIN_STOPWORDS)
    return _stop


def _wn_pos(tag):
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def tokenize_lemmatize(text):
    """case folding -> hapus angka & tanda baca -> tokenisasi -> stopword removal -> lemmatization."""
    t = text.lower()                                 # case folding
    t = re.sub(r"[^a-z\s]", " ", t)                  # hapus angka & tanda baca
    tokens = [w for w in t.split() if len(w) > 2]    # tokenisasi
    stop = get_stopwords()
    tokens = [w for w in tokens if w not in stop]    # stopword removal
    tagged = nltk.pos_tag(tokens) if tokens else []
    lemmas = [_lemmatizer.lemmatize(w, _wn_pos(p)) for w, p in tagged]   # lemmatization
    return [w for w in lemmas if w not in stop and len(w) > 2]


def preprocess(df):
    """Menambah kolom tokens (untuk LDA) dan menyaring ulasan terlalu pendek."""
    df = df.copy()
    df["tokens"] = df["text_clean"].apply(tokenize_lemmatize)
    df["n_tokens"] = df["tokens"].apply(len)
    df["n_words_raw"] = df["text_clean"].str.split().apply(len)
    before = df
    df = df[df["n_tokens"] >= config.MIN_TOKENS_LDA].reset_index(drop=True)
    return df, before


def preprocessing_examples(df, n=3):
    """Contoh per tahap preprocessing untuk tabel di buku TA."""
    rows = []
    for _, r in df.head(n).iterrows():
        t0 = r["text_raw"]
        t1 = light_clean(t0)
        t2 = t1.lower()
        t3 = re.sub(r"[^a-z\s]", " ", t2)
        tok = [w for w in t3.split() if len(w) > 2]
        stop = [w for w in tok if w not in get_stopwords()]
        rows.append({"platform": r["platform"], "1_teks_asli": t0, "2_cleaning": t1, "3_case_folding": t2,
                     "4_tokenisasi": tok, "5_stopword_removal": stop, "6_lemmatisasi": r["tokens"]})
    return rows
