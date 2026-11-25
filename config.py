"""
config.py — SEMUA pengaturan penelitian ada di sini.

>>> CARA MENGGANTI DATA DUMMY DENGAN DATA ASLI <<<
1. Simpan CSV hasil Apify di folder data/raw/ dengan nama:
       data/raw/imdb_obsession_reviews.csv
       data/raw/letterboxd_obsession_reviews.csv
   (atau ubah REAL_IMDB_CSV / REAL_LETTERBOXD_CSV di bawah).
2. Ubah SATU baris berikut menjadi False, lalu jalankan: python run_pipeline.py
"""
from pathlib import Path

USE_DUMMY = True   # <<< UBAH MENJADI False UNTUK MEMAKAI DATA ASLI

# ---------------------------------------------------------------------------
# Lokasi file
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DUMMY_IMDB_CSV = BASE_DIR / "data" / "dummy" / "DUMMY_imdb_obsession_reviews.csv"
DUMMY_LETTERBOXD_CSV = BASE_DIR / "data" / "dummy" / "DUMMY_letterboxd_obsession_reviews.csv"
REAL_IMDB_CSV = BASE_DIR / "data" / "raw" / "imdb_obsession_reviews.csv"
REAL_LETTERBOXD_CSV = BASE_DIR / "data" / "raw" / "letterboxd_obsession_reviews.csv"

IMDB_CSV = DUMMY_IMDB_CSV if USE_DUMMY else REAL_IMDB_CSV
LETTERBOXD_CSV = DUMMY_LETTERBOXD_CSV if USE_DUMMY else REAL_LETTERBOXD_CSV

OUTPUT_DIR = BASE_DIR / "output"
LABELS_DIR = BASE_DIR / "labels"

# Nama kolom pada CSV Apify (sesuai Tabel III-2 dan III-3 proposal)
IMDB_COLS = {"id": "id", "text": "text", "rating": "authorRating", "date": "submissionDate",
             "spoiler": "spoiler", "votes": "helpfulnessUpVotes"}
LETTERBOXD_COLS = {"id": "reviewurl", "text": "reviewtext", "rating": "rating", "date": "reviewdate",
                   "spoiler": "isspoiler", "votes": "likes"}

# ---------------------------------------------------------------------------
# Pembersihan & preprocessing (III.2.2–III.2.3)
# ---------------------------------------------------------------------------
LANGUAGE = "en"                 # hanya ulasan berbahasa Inggris (langdetect)
LANGDETECT_SEED = 0             # agar hasil deteksi bahasa stabil
MIN_TOKENS_LDA = 3              # ulasan < 3 token setelah preprocessing dibuang
LETTERBOXD_RATING_FACTOR = 2    # rating Letterboxd (0,5–5) x 2 => skala 1–10
DOMAIN_STOPWORDS = [            # stopword domain (III.2.3)
    "film", "films", "movie", "movies", "obsession", "watch", "watched", "watching",
    "one", "get", "got", "make", "made", "really", "even", "much", "would", "also",
    "thing", "lol", "see", "seen", "go", "going", "like", "still", "way",
    "horror", "saw", "well", "lot", "something", "know", "think", "give", "come", "feel",
    "felt", "time", "ten", "say", "want", "look",
]

# ---------------------------------------------------------------------------
# LDA (Gensim) — III.2.4
# ---------------------------------------------------------------------------
LDA_K_RANGE = list(range(2, 16))   # 2 s.d. 15 topik
LDA_PASSES = 20
LDA_ITERATIONS = 400
LDA_ALPHA = "auto"
LDA_ETA = "auto"
DICT_NO_BELOW = 3                  # kata muncul minimal di 3 dokumen
DICT_NO_ABOVE = 0.5                # buang kata yang muncul di >50% dokumen
K_SELECTION = "first_peak"         # "first_peak" = puncak pertama kurva C_v (K terkecil yang C_v-nya
                                   #     lebih tinggi dari tetangganya; topik tetap dapat diinterpretasikan)
                                   # "max_cv" = C_v tertinggi pada seluruh rentang 2–15
                                   # "cv_x_diversity" = C_v x topic diversity tertinggi (Dieng dkk., 2020)
LDA_SEEDS = 3                      # tiap K dilatih 3x (seed berbeda), C_v dirata-rata
LDA_K_OVERRIDE = None              # isi angka (mis. 6) bila ingin memaksa jumlah topik

# ---------------------------------------------------------------------------
# BERTopic — III.2.4
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
HDBSCAN_PARAMS = dict(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom")
MIN_TOPIC_SIZE = 15
REDUCE_OUTLIERS = True
OUTLIER_STRATEGY = "embeddings"    # strategi reduce_outliers BERTopic
NR_TOPICS = None                   # None = biarkan HDBSCAN menentukan

# ---------------------------------------------------------------------------
# Evaluasi & analisis — III.2.5, III.2.6, III.4
# ---------------------------------------------------------------------------
TOP_N_WORDS = 10                   # top-n kata untuk C_v, diversity, Jaccard
ALIGN_THRESHOLD = 0.15             # Jaccard minimal (>= 3 dari 10 kata sama) agar topik dianggap "konsisten"
ALPHA = 0.05                       # taraf signifikansi chi-square
RANDOM_STATE = 42                  # seed untuk reproduksibilitas (UMAP, LDA)
FIG_LANGS = ["id", "en"]           # gambar dibuat dalam bahasa Indonesia & Inggris
