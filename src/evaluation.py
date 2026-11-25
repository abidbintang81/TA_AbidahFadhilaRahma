"""Evaluasi (C_v, topic diversity), pelabelan, pencocokan topik, chi-square, dan analisis rating."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

import config

# ---------------------------------------------------------------------------
# Topic diversity (Dieng dkk., 2020): proporsi kata unik pada top-n semua topik
# ---------------------------------------------------------------------------
def topic_diversity(topics_words, topn=None):
    topn = topn or config.TOP_N_WORDS
    words = [w for t in topics_words for w in t[:topn]]
    return float(len(set(words)) / len(words)) if words else 0.0


# ---------------------------------------------------------------------------
# Usulan label otomatis (WAJIB diperiksa peneliti & pembimbing, III.2.5)
# ---------------------------------------------------------------------------
ASPECT_LEXICON = {
    ("Akting dan performa pemeran", "Acting and performance"):
        {"performance", "actress", "acting", "navarrette", "inde", "nikki", "johnston", "cast", "role",
         "actor", "lead", "award", "chemistry", "expression", "breakout", "play"},
    ("Gore dan body horror", "Gore and body horror"):
        {"gore", "blood", "bloody", "gory", "violence", "gruesome", "effect", "practical", "body",
         "graphic", "brutal", "kitchen", "disgusting", "shock"},
    ("Jump scare, suara, dan atmosfer", "Jump scares, sound, and atmosphere"):
        {"scare", "jump", "sound", "tension", "atmosphere", "creepy", "loud", "score", "dread",
         "scary", "noise", "seat", "unsettle", "silence", "anxious", "scar", "suspense", "design", "eerie"},
    ("Premis dan alur cerita (wish)", "Premise and plot (the wish)"):
        {"premise", "wish", "willow", "concept", "story", "original", "plot", "idea", "magical",
         "careful", "rule", "unpredictable", "nightmare", "creative", "fresh"},
    ("Logika dan penulisan karakter", "Character logic and writing"):
        {"character", "bear", "decision", "dumb", "stupid", "writing", "plausibility", "flag",
         "protagonist", "sane", "relationship", "side", "behavior", "responsibility", "thin"},
    ("Humor dan nada film", "Humor and tone"):
        {"funny", "humor", "laugh", "comedy", "hilarious", "absurd", "joke", "tone", "comedic",
         "customer", "service", "iconic", "fun", "awkward"},
    ("Ending dan pacing", "Ending and pacing"):
        {"ending", "end", "final", "act", "third", "pacing", "finale", "slow", "payoff", "rush",
         "long", "minute", "drag", "circle", "devious", "letdown"},
    ("Sutradara, produksi, dan debut", "Director, production, and debut"):
        {"barker", "curry", "director", "direction", "debut", "youtube", "budget", "cinematography",
         "indie", "camera", "lighting", "production", "feature", "short", "visual", "studio"},
    ("Tema: obsesi, consent, relasi toksik", "Themes: obsession, consent, toxic love"):
        {"consent", "toxic", "entitlement", "metaphor", "nice", "guy", "trope", "theme", "love",
         "agency", "allegory", "commentary", "possessive", "loneliness", "esteem", "social"},
    ("Pengalaman menonton di bioskop", "Theater viewing experience"):
        {"theater", "audience", "crowd", "cinema", "experience", "friend", "packed", "hype",
         "screaming", "reaction", "empty", "afterwards"},
}


def suggest_labels(topics_words):
    """Label usulan berbasis leksikon aspek (dengan bobot peringkat kata)."""
    out = []
    for words in topics_words:
        best, best_score = ("Topik campuran/umum", "Mixed/general topic"), 0.0
        for lab, lex in ASPECT_LEXICON.items():
            s = sum(1 + (len(words) - i) / len(words) for i, w in enumerate(words) if w in lex)
            if s > best_score:
                best, best_score = lab, s
        out.append(best)
    # beri penanda bila label ganda
    seen = {}
    final = []
    for lid, len_ in out:
        seen[lid] = seen.get(lid, 0) + 1
        suffix = f" ({seen[lid]})" if seen[lid] > 1 else ""
        final.append((lid + suffix, len_ + suffix))
    return final


def apply_manual_labels(model_name, topic_ids, auto_labels):
    """Jika labels/<model>_labels.csv ada, label manual dari file tersebut dipakai."""
    path = config.LABELS_DIR / f"{model_name}_labels.csv"
    labels = {t: {"label_id": a[0], "label_en": a[1], "label_source": "otomatis (usulan leksikon)"}
              for t, a in zip(topic_ids, auto_labels)}
    if path.exists():
        man = pd.read_csv(path)
        for _, r in man.iterrows():
            t = int(r["topic"])
            if t in labels and isinstance(r.get("label_id"), str) and r["label_id"].strip():
                labels[t] = {"label_id": r["label_id"], "label_en": r.get("label_en", r["label_id"]),
                             "label_source": "manual (file label)"}
    return labels


# ---------------------------------------------------------------------------
# Pencocokan topik LDA–BERTopic (III.2.5)
# ---------------------------------------------------------------------------
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def jaccard_matrix(lda_words, bt_words):
    return np.array([[jaccard(a, b) for b in bt_words] for a in lda_words])


def align_topics(J, lda_ids, bt_ids):
    """Pasangan satu-satu (algoritma Hungaria) yang memaksimalkan total Jaccard."""
    r, c = linear_sum_assignment(-J)
    pairs = []
    matched_lda, matched_bt = set(), set()
    for i, j in zip(r, c):
        status = "konsisten" if J[i, j] >= config.ALIGN_THRESHOLD else "tidak cocok"
        pairs.append({"lda_topic": int(lda_ids[i]), "bertopic_topic": int(bt_ids[j]),
                      "jaccard": round(float(J[i, j]), 4), "status": status})
        if status == "konsisten":
            matched_lda.add(lda_ids[i]); matched_bt.add(bt_ids[j])
    only_lda = [int(t) for t in lda_ids if t not in matched_lda]
    only_bt = [int(t) for t in bt_ids if t not in matched_bt]
    return pairs, only_lda, only_bt


# ---------------------------------------------------------------------------
# Distribusi topik per platform & uji chi-square (III.2.6)
# ---------------------------------------------------------------------------
def distribution_table(df, topic_col):
    ct = pd.crosstab(df[topic_col], df["platform"])
    pct = ct.div(ct.sum(axis=0), axis=1) * 100
    out = pd.DataFrame({"topic": ct.index})
    for p in ct.columns:
        out[f"n_{p}"] = ct[p].values
        out[f"pct_{p}"] = pct[p].round(2).values
    out["n_total"] = ct.sum(axis=1).values
    return out, ct


def chi_square(ct):
    chi2, p, dof, expected = stats.chi2_contingency(ct.values)
    n = ct.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    resid = (ct.values - expected) / np.sqrt(expected)   # residual terstandar Pearson
    low = int((expected < 5).sum())
    return {"chi2": round(float(chi2), 4), "df": int(dof), "p_value": float(p),
            "cramers_v": round(v, 4), "n": int(n), "alpha": config.ALPHA,
            "significant": bool(p < config.ALPHA),
            "cells_expected_lt5": low, "cells_total": int(expected.size),
            "std_residuals": pd.DataFrame(resid, index=ct.index, columns=ct.columns).round(3)}


# ---------------------------------------------------------------------------
# Topik vs rating (preferensi pengguna, II.1.3 & III.2.6)
# ---------------------------------------------------------------------------
def rating_by_topic(df, topic_col):
    g = df.groupby([topic_col, "platform"])["rating_10"]
    t = g.agg(n="count", mean="mean", median="median", std="std").reset_index()
    t = t.rename(columns={topic_col: "topic"})
    allp = df.groupby(topic_col)["rating_10"].agg(n="count", mean="mean", median="median",
                                                  std="std").reset_index()
    allp = allp.rename(columns={topic_col: "topic"}); allp["platform"] = "Gabungan"
    t = pd.concat([t, allp], ignore_index=True)
    for c in ["mean", "median", "std"]:
        t[c] = t[c].round(3)
    return t


def kruskal_by_topic(df, topic_col):
    groups = [g["rating_10"].dropna().values for _, g in df.groupby(topic_col) if len(g) > 1]
    if len(groups) < 2:
        return None
    h, p = stats.kruskal(*groups)
    return {"H": round(float(h), 4), "df": len(groups) - 1, "p_value": float(p),
            "significant": bool(p < config.ALPHA)}


def mannwhitney_per_topic(df, topic_col):
    rows = []
    for t, g in df.groupby(topic_col):
        a = g.loc[g.platform == "IMDb", "rating_10"].dropna()
        b = g.loc[g.platform == "Letterboxd", "rating_10"].dropna()
        if len(a) >= 3 and len(b) >= 3:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"topic": int(t), "U": float(u), "p_value": float(p),
                         "significant": bool(p < config.ALPHA)})
    return pd.DataFrame(rows)


def preference_table(dist, rating, labels):
    """Topik dominan per platform beserta rating rata-ratanya (indikator preferensi)."""
    rows = []
    for p in ["IMDb", "Letterboxd"]:
        d = dist.sort_values(f"pct_{p}", ascending=False)
        for rank, (_, r) in enumerate(d.iterrows(), start=1):
            rr = rating[(rating.topic == r.topic) & (rating.platform == p)]
            rows.append({"platform": p, "rank": rank, "topic": int(r.topic),
                         "label_id": labels[int(r.topic)]["label_id"],
                         "label_en": labels[int(r.topic)]["label_en"],
                         "n": int(r[f"n_{p}"]), "pct": float(r[f"pct_{p}"]),
                         "mean_rating_10": float(rr["mean"].iloc[0]) if len(rr) else None,
                         "median_rating_10": float(rr["median"].iloc[0]) if len(rr) else None})
    return pd.DataFrame(rows)
