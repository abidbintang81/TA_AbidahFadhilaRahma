"""Pemodelan topik BERTopic (Bab III.2.4): embedding -> UMAP -> HDBSCAN -> c-TF-IDF."""
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

import config
from src.preprocessing import tokenize_lemmatize


def compute_embeddings(docs, log=print):
    """Sentence embedding dengan all-MiniLM-L6-v2.

    Jika model tidak dapat diunduh (mis. tanpa internet), program memakai cadangan
    TF-IDF + SVD (LSA, 384 dimensi) agar pipeline tetap berjalan. Backend yang
    dipakai dicatat di summary.json (kunci 'embedding_backend'). Untuk hasil
    akhir TA, PASTIKAN backend = all-MiniLM-L6-v2.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        emb = model.encode(docs, show_progress_bar=False, batch_size=32)
        return np.asarray(emb), config.EMBEDDING_MODEL
    except Exception as e:  # noqa: BLE001
        log(f"   [PERINGATAN] Model embedding tidak tersedia ({type(e).__name__}). "
            "Memakai cadangan TF-IDF+SVD. JANGAN dipakai untuk hasil akhir TA.")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                            stop_words="english").fit_transform(docs)
        n_comp = min(384, X.shape[1] - 1, X.shape[0] - 1)
        emb = TruncatedSVD(n_components=n_comp, random_state=config.RANDOM_STATE).fit_transform(X)
        return normalize(emb), "FALLBACK_tfidf_svd (bukan all-MiniLM-L6-v2)"


def build_bertopic():
    umap_model = UMAP(random_state=config.RANDOM_STATE, **config.UMAP_PARAMS)
    hdbscan_model = HDBSCAN(prediction_data=True, **config.HDBSCAN_PARAMS)
    # Stopword dihapus pada tahap c-TF-IDF. Tokenizer yang sama dengan LDA
    # (termasuk lemmatisasi) dipakai agar kata teratas kedua model dapat dibandingkan
    # (C_v, diversity, Jaccard) secara setara.
    vectorizer = CountVectorizer(tokenizer=tokenize_lemmatize, lowercase=False,
                                 token_pattern=None, min_df=2)
    return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer, min_topic_size=config.MIN_TOPIC_SIZE,
                    nr_topics=config.NR_TOPICS, calculate_probabilities=False,
                    language="english", verbose=False)


def train_bertopic(docs, embeddings, log=print):
    model = build_bertopic()
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    topics = np.asarray(topics)
    n_out_before = int((topics == -1).sum())
    n_out_after = n_out_before
    if config.REDUCE_OUTLIERS and n_out_before > 0 and (topics != -1).any():
        new_topics = model.reduce_outliers(docs, list(topics), strategy=config.OUTLIER_STRATEGY,
                                           embeddings=embeddings)
        model.update_topics(docs, topics=new_topics, vectorizer_model=model.vectorizer_model)
        topics = np.asarray(new_topics)
        n_out_after = int((topics == -1).sum())
    log(f"   BERTopic: {len(set(topics) - {-1})} topik, outlier {n_out_before} -> {n_out_after}")
    return model, topics, n_out_before, n_out_after


def bertopic_top_words(model, topn=None):
    topn = topn or config.TOP_N_WORDS
    ids = sorted(t for t in model.get_topics() if t != -1)
    return ids, [[w for w, _ in model.get_topic(t)[:topn]] for t in ids]
