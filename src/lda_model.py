"""Pemodelan topik LDA dengan Gensim (Bab III.2.4)."""
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

import config


def build_corpus(token_lists):
    """Membentuk dictionary dan Document-Term Matrix (Bag of Words)."""
    dictionary = Dictionary(token_lists)
    dictionary.filter_extremes(no_below=config.DICT_NO_BELOW, no_above=config.DICT_NO_ABOVE)
    corpus = [dictionary.doc2bow(t) for t in token_lists]
    return dictionary, corpus


def train_lda(corpus, dictionary, k, seed=None):
    return LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                    passes=config.LDA_PASSES, iterations=config.LDA_ITERATIONS,
                    alpha=config.LDA_ALPHA, eta=config.LDA_ETA,
                    random_state=config.RANDOM_STATE if seed is None else seed, eval_every=None)


def lda_top_words(model, topn=None):
    topn = topn or config.TOP_N_WORDS
    return [[w for w, _ in model.show_topic(t, topn=topn)] for t in range(model.num_topics)]


def coherence_cv(topics_words, texts, dictionary):
    cm = CoherenceModel(topics=topics_words, texts=texts, dictionary=dictionary,
                        coherence="c_v", topn=config.TOP_N_WORDS, processes=1)
    return float(cm.get_coherence()), [float(x) for x in cm.get_coherence_per_topic()]


def select_k(corpus, dictionary, texts, log=print):
    """Melatih LDA untuk setiap K pada LDA_K_RANGE dan memilih K dengan rata-rata C_v tertinggi.

    Setiap K dilatih LDA_SEEDS kali (random_state berbeda) agar kurva C_v stabil;
    model akhir = model dengan seed RANDOM_STATE pada K terpilih.
    """
    rows, models = [], {}
    seeds = [config.RANDOM_STATE + i for i in range(config.LDA_SEEDS)]
    for k in config.LDA_K_RANGE:
        cvs, tds = [], []
        for s in seeds:
            m = train_lda(corpus, dictionary, k, seed=s)
            tw = lda_top_words(m)
            cv, _ = coherence_cv(tw, texts, dictionary)
            cvs.append(cv)
            words = [w for t in tw for w in t]
            tds.append(len(set(words)) / len(words))
            if s == config.RANDOM_STATE:
                models[k] = m
        rows.append({"k": k, "coherence_cv": round(float(np.mean(cvs)), 4),
                     "coherence_cv_std": round(float(np.std(cvs)), 4),
                     "coherence_cv_runs": [round(c, 4) for c in cvs],
                     "topic_diversity": round(float(np.mean(tds)), 4),
                     "topic_quality": round(float(np.mean(cvs) * np.mean(tds)), 4)})
        log(f"   LDA k={k:2d}  C_v rata-rata={np.mean(cvs):.4f} (sd {np.std(cvs):.4f}, {len(cvs)} run)")
    curve = pd.DataFrame(rows)
    best_k = choose_k(curve)
    if config.LDA_K_OVERRIDE:
        best_k = int(config.LDA_K_OVERRIDE)
    return best_k, models[best_k], curve


def choose_k(curve):
    """Aturan pemilihan K (lihat config.K_SELECTION).

    - "max_cv"        : K dengan rata-rata C_v tertinggi pada seluruh rentang.
    - "first_peak"    : K pada puncak lokal pertama kurva C_v (C_v[k] >= tetangga kiri & kanan),
                        yaitu titik sebelum kurva mendatar. K yang lebih besar umumnya hanya
                        memecah topik menjadi subtopik yang sulit diinterpretasikan (III.2.4).
                        Jika tidak ada puncak lokal, dipakai max_cv.
    - "cv_x_diversity": K dengan C_v x topic diversity tertinggi (topic quality, Dieng dkk., 2020).
    """
    cv = curve["coherence_cv"].tolist(); ks = curve["k"].tolist()
    if config.K_SELECTION == "cv_x_diversity":
        return int(curve.loc[curve["topic_quality"].idxmax(), "k"])
    if config.K_SELECTION == "first_peak":
        for i in range(1, len(cv) - 1):
            if cv[i] >= cv[i - 1] and cv[i] >= cv[i + 1] and cv[i] >= cv[0]:
                return int(ks[i])
    return int(curve.loc[curve["coherence_cv"].idxmax(), "k"])


def doc_topic_matrix(model, corpus):
    k = model.num_topics
    mat = np.zeros((len(corpus), k))
    for i, bow in enumerate(corpus):
        for t, p in model.get_document_topics(bow, minimum_probability=0.0):
            mat[i, t] = p
    return mat
