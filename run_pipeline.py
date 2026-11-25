"""
run_pipeline.py — menjalankan seluruh alur penelitian (Bab III) dari data mentah
sampai tabel, gambar, dan summary.json di folder output/.

    python run_pipeline.py

Pengaturan ada di config.py (USE_DUMMY = True/False untuk data dummy/asli).
"""
import json
import os
import time
from datetime import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import lda_model as lm  # noqa: E402
from src import visualization as viz  # noqa: E402
from src.preprocessing import ensure_nltk, load_data, preprocess, preprocessing_examples, select_data  # noqa: E402

OUT = config.OUTPUT_DIR
TAB = OUT / "tables"
FIG = OUT / "figures"
MOD = OUT / "models"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    import sys
    if "--real" in sys.argv:          # dipakai oleh aplikasi (halaman Jalankan Ulang)
        config.USE_DUMMY = False
        config.IMDB_CSV, config.LETTERBOXD_CSV = config.REAL_IMDB_CSV, config.REAL_LETTERBOXD_CSV
    elif "--dummy" in sys.argv:
        config.USE_DUMMY = True
        config.IMDB_CSV, config.LETTERBOXD_CSV = config.DUMMY_IMDB_CSV, config.DUMMY_LETTERBOXD_CSV
    t0 = time.time()
    for d in (TAB, FIG, MOD):
        d.mkdir(parents=True, exist_ok=True)
    ensure_nltk()
    np.random.seed(config.RANDOM_STATE)

    # ------------------------------------------------------------------ 1. Data
    log(f"1) Memuat data ({'DUMMY' if config.USE_DUMMY else 'ASLI'})")
    log(f"   IMDb      : {config.IMDB_CSV}")
    log(f"   Letterboxd: {config.LETTERBOXD_CSV}")
    raw = load_data()
    df, stages = select_data(raw)

    # ------------------------------------------------------------ 2. Preprocessing
    log("2) Preprocessing (cleaning, case folding, tokenisasi, stopword, lemmatisasi)")
    df, df_before_short = preprocess(df)
    stages.append({"tahap": "Setelah hapus ulasan terlalu pendek",
                   **{p: int((df.platform == p).sum()) for p in ["IMDb", "Letterboxd"]}, "Total": int(len(df))})
    stages_df = pd.DataFrame(stages)
    stages_df.to_csv(TAB / "01_tahapan_seleksi_data.csv", index=False)
    log("   " + " | ".join(f"{s['tahap']}: {s['Total']}" for s in stages))
    pd.DataFrame(preprocessing_examples(df, n=4)).to_csv(TAB / "02_contoh_preprocessing.csv", index=False)

    length_stats = df.groupby("platform")["n_words_raw"].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
    length_stats.to_csv(TAB / "03_statistik_panjang_ulasan.csv")
    rating_stats = df.groupby("platform")["rating_10"].agg(["count", "mean", "median", "std", "min", "max"]).round(3)
    rating_stats.to_csv(TAB / "04_statistik_rating.csv")

    texts = df["tokens"].tolist()
    docs = df["text_clean"].tolist()

    # ---------------------------------------------------------------------- 3. LDA
    log(f"3) LDA: memilih K pada rentang {config.LDA_K_RANGE[0]}–{config.LDA_K_RANGE[-1]} berdasarkan C_v")
    dictionary, corpus = lm.build_corpus(texts)
    log(f"   Dictionary: {len(dictionary)} kata unik, DTM {len(corpus)} dokumen")
    best_k, lda, curve = lm.select_k(corpus, dictionary, texts, log=log)
    curve.drop(columns=["coherence_cv_runs"]).to_csv(TAB / "05_lda_coherence_per_k.csv", index=False)
    log(f"   K terpilih = {best_k}")
    lda.save(str(MOD / "lda_model.gensim"))
    lda_words = lm.lda_top_words(lda)
    lda_cv, lda_cv_topic = lm.coherence_cv(lda_words, texts, dictionary)
    theta = lm.doc_topic_matrix(lda, corpus)
    df["lda_topic"] = theta.argmax(axis=1) + 1            # topik LDA bernomor 1..K
    df["lda_topic_prob"] = theta.max(axis=1).round(4)
    lda_ids = list(range(1, best_k + 1))

    # ----------------------------------------------------------------- 4. BERTopic
    log("4) BERTopic: embedding -> UMAP -> HDBSCAN -> c-TF-IDF")
    from src import bertopic_model as bm
    emb, emb_backend = bm.compute_embeddings(docs, log=log)
    log(f"   Embedding: {emb_backend}, dimensi {emb.shape}")
    bt, bt_topics, out_before, out_after = bm.train_bertopic(docs, emb, log=log)
    df["bt_topic"] = np.where(bt_topics >= 0, bt_topics + 1, -1)   # topik BERTopic bernomor 1..n, outlier -1
    bt_raw_ids, bt_words = bm.bertopic_top_words(bt)
    bt_ids = [t + 1 for t in bt_raw_ids]
    try:
        bt.save(str(MOD / "bertopic_model"), serialization="pickle")
    except Exception as e:  # noqa: BLE001
        log(f"   (model BERTopic tidak disimpan: {e})")
    # C_v BERTopic dihitung dengan korpus & dictionary yang sama dengan LDA
    bt_words_cv = [[w for w in ws if w in dictionary.token2id] for ws in bt_words]
    valid = [i for i, ws in enumerate(bt_words_cv) if len(ws) >= 2]
    bt_cv, cv_valid = lm.coherence_cv([bt_words_cv[i] for i in valid], texts, dictionary)
    bt_cv_topic = [float("nan")] * len(bt_words)
    for i, v in zip(valid, cv_valid):
        bt_cv_topic[i] = v

    # -------------------------------------------------------- 5. Evaluasi & label
    log("5) Evaluasi: C_v, topic diversity, label, pencocokan topik")
    lda_div = ev.topic_diversity(lda_words)
    bt_div = ev.topic_diversity(bt_words)
    lda_labels = ev.apply_manual_labels("lda", lda_ids, ev.suggest_labels(lda_words))
    bt_labels = ev.apply_manual_labels("bertopic", bt_ids, ev.suggest_labels(bt_words))
    metrics = {
        "LDA": {"coherence_cv": round(lda_cv, 4), "topic_diversity": round(lda_div, 4), "n_topics": best_k},
        "BERTopic": {"coherence_cv": round(bt_cv, 4), "topic_diversity": round(bt_div, 4), "n_topics": len(bt_ids)},
    }
    pd.DataFrame([{"model": k, **v} for k, v in metrics.items()]).to_csv(TAB / "06_evaluasi_model.csv", index=False)

    lda_topic_rows, bt_topic_rows = [], []
    for i, t in enumerate(lda_ids):
        ww = lda.show_topic(t - 1, topn=config.TOP_N_WORDS)
        lda_topic_rows.append({"topic": t, **{k: lda_labels[t][k] for k in ("label_id", "label_en", "label_source")},
                               "top_words": ", ".join(w for w, _ in ww),
                               "weights": ", ".join(f"{p:.4f}" for _, p in ww),
                               "coherence_cv": round(lda_cv_topic[i], 4)})
    for i, (t, rid) in enumerate(zip(bt_ids, bt_raw_ids)):
        ww = bt.get_topic(rid)[:config.TOP_N_WORDS]
        bt_topic_rows.append({"topic": t, **{k: bt_labels[t][k] for k in ("label_id", "label_en", "label_source")},
                              "top_words": ", ".join(w for w, _ in ww),
                              "weights": ", ".join(f"{p:.4f}" for _, p in ww),
                              "coherence_cv": round(bt_cv_topic[i], 4)})
    pd.DataFrame(lda_topic_rows).to_csv(TAB / "07_lda_topik_kata_teratas.csv", index=False)
    pd.DataFrame(bt_topic_rows).to_csv(TAB / "08_bertopic_topik_kata_teratas.csv", index=False)
    # template label manual (hanya dibuat bila belum ada)
    for name, rows in (("lda", lda_topic_rows), ("bertopic", bt_topic_rows)):
        p = config.LABELS_DIR / f"{name}_labels_TEMPLATE.csv"
        pd.DataFrame([{"topic": r["topic"], "top_words": r["top_words"], "label_id": r["label_id"],
                       "label_en": r["label_en"]} for r in rows]).to_csv(p, index=False)

    # contoh ulasan representatif
    reps = []
    for t in lda_ids:
        sub = df[df.lda_topic == t].sort_values("lda_topic_prob", ascending=False).head(3)
        for _, r in sub.iterrows():
            reps.append({"model": "LDA", "topic": t, "platform": r.platform, "rating_10": r.rating_10,
                         "prob": r.lda_topic_prob, "text": r.text_clean[:600]})
    for t, rid in zip(bt_ids, bt_raw_ids):
        for d in (bt.get_representative_docs(rid) or [])[:3]:
            r = df[df.text_clean == d].iloc[0]
            reps.append({"model": "BERTopic", "topic": t, "platform": r.platform, "rating_10": r.rating_10,
                         "prob": None, "text": d[:600]})
    pd.DataFrame(reps).to_csv(TAB / "09_contoh_ulasan_per_topik.csv", index=False)

    J = ev.jaccard_matrix(lda_words, bt_words)
    pd.DataFrame(J.round(4), index=[f"LDA-{t}" for t in lda_ids],
                 columns=[f"BT-{t}" for t in bt_ids]).to_csv(TAB / "10_jaccard_lda_vs_bertopic.csv")
    pairs, only_lda, only_bt = ev.align_topics(J, lda_ids, bt_ids)
    for p in pairs:
        p["lda_label"] = lda_labels[p["lda_topic"]]["label_id"]
        p["bertopic_label"] = bt_labels[p["bertopic_topic"]]["label_id"]
    pd.DataFrame(pairs).to_csv(TAB / "11_pencocokan_topik.csv", index=False)

    # ----------------------------------------------- 6. Analisis antarplatform
    log("6) Distribusi topik per platform, chi-square, dan rating per topik")
    lda_dist, lda_ct = ev.distribution_table(df, "lda_topic")
    df_bt = df[df.bt_topic != -1]
    bt_dist, bt_ct = ev.distribution_table(df_bt, "bt_topic")
    lda_dist.to_csv(TAB / "12_lda_distribusi_platform.csv", index=False)
    bt_dist.to_csv(TAB / "13_bertopic_distribusi_platform.csv", index=False)
    prop = pd.DataFrame({"topic": lda_ids})
    for p in ["IMDb", "Letterboxd"]:
        prop[f"mean_prop_{p}"] = theta[(df.platform == p).values].mean(axis=0).round(4)
    prop.to_csv(TAB / "14_lda_rata_proporsi_platform.csv", index=False)

    chi_lda = ev.chi_square(lda_ct)
    chi_bt = ev.chi_square(bt_ct)
    chi_lda["std_residuals"].to_csv(TAB / "15_lda_residual_chisquare.csv")
    chi_bt["std_residuals"].to_csv(TAB / "16_bertopic_residual_chisquare.csv")
    chi_rows = [{"model": m, **{k: v for k, v in c.items() if k != "std_residuals"}}
                for m, c in (("LDA", chi_lda), ("BERTopic", chi_bt))]
    pd.DataFrame(chi_rows).to_csv(TAB / "17_uji_chi_square.csv", index=False)
    log(f"   Chi-square LDA: chi2={chi_lda['chi2']}, df={chi_lda['df']}, p={chi_lda['p_value']:.4g}, V={chi_lda['cramers_v']}")
    log(f"   Chi-square BERTopic: chi2={chi_bt['chi2']}, df={chi_bt['df']}, p={chi_bt['p_value']:.4g}, V={chi_bt['cramers_v']}")

    rate_lda = ev.rating_by_topic(df, "lda_topic")
    rate_bt = ev.rating_by_topic(df_bt, "bt_topic")
    rate_lda.to_csv(TAB / "18_lda_rating_per_topik.csv", index=False)
    rate_bt.to_csv(TAB / "19_bertopic_rating_per_topik.csv", index=False)
    kw_lda, kw_bt = ev.kruskal_by_topic(df, "lda_topic"), ev.kruskal_by_topic(df_bt, "bt_topic")
    mw_lda, mw_bt = ev.mannwhitney_per_topic(df, "lda_topic"), ev.mannwhitney_per_topic(df_bt, "bt_topic")
    mw_lda.to_csv(TAB / "20_lda_mannwhitney_rating_platform.csv", index=False)
    mw_bt.to_csv(TAB / "21_bertopic_mannwhitney_rating_platform.csv", index=False)
    pref_lda = ev.preference_table(lda_dist, rate_lda, lda_labels)
    pref_bt = ev.preference_table(bt_dist, rate_bt, bt_labels)
    pref_lda.to_csv(TAB / "22_lda_preferensi_pengguna.csv", index=False)
    pref_bt.to_csv(TAB / "23_bertopic_preferensi_pengguna.csv", index=False)

    # Data per dokumen (untuk aplikasi)
    keep = ["review_id", "platform", "date", "rating_raw", "rating_10", "n_words_raw", "n_tokens",
            "lda_topic", "lda_topic_prob", "bt_topic", "text_clean"]
    out_docs = df[keep].copy()
    out_docs["tokens"] = df["tokens"].apply(" ".join)
    out_docs.to_csv(OUT / "documents_with_topics.csv", index=False)

    # ---------------------------------------------------------------- 7. Gambar
    log("7) Membuat gambar (id & en)")
    from umap import UMAP
    xy = UMAP(n_components=2, random_state=config.RANDOM_STATE, n_neighbors=15, min_dist=0.1,
              metric="cosine").fit_transform(emb)
    np.save(OUT / "doc_map_2d.npy", xy)
    lda_ww = [lda.show_topic(t - 1, topn=config.TOP_N_WORDS) for t in lda_ids]
    bt_ww = [bt.get_topic(r)[:config.TOP_N_WORDS] for r in bt_raw_ids]
    for lang in config.FIG_LANGS:
        F = FIG / lang
        viz.fig_cleaning(stages, F / "fig01_seleksi_data.png", lang)
        viz.fig_rating(df, F / "fig02_distribusi_rating.png", lang)
        viz.fig_length(df, F / "fig03_panjang_ulasan.png", lang)
        viz.fig_coherence(curve, best_k, F / "fig04_lda_coherence_k.png", lang)
        viz.fig_topwords(lda_ids, lda_ww, lda_labels, F / "fig05_lda_kata_teratas.png", lang, "topwords_lda", viz.C_LDA, "LDA-")
        viz.fig_topwords(bt_ids, bt_ww, bt_labels, F / "fig06_bertopic_kata_teratas.png", lang, "topwords_bt", viz.C_BT, "BT-")
        viz.fig_distribution(lda_dist, lda_labels, F / "fig07_lda_distribusi_platform.png", lang, "dist_lda", "LDA-")
        viz.fig_distribution(bt_dist, bt_labels, F / "fig08_bertopic_distribusi_platform.png", lang, "dist_bt", "BT-")
        viz.fig_lda_proportion(prop, lda_labels, F / "fig09_lda_rata_proporsi.png", lang)
        viz.fig_model_compare(metrics, F / "fig10_perbandingan_model.png", lang)
        viz.fig_jaccard(J, lda_ids, bt_ids, F / "fig11_jaccard_heatmap.png", lang)
        viz.fig_rating_topic(rate_lda, lda_labels, F / "fig12_lda_rating_per_topik.png", lang, "rate_lda", "LDA-")
        viz.fig_rating_topic(rate_bt, bt_labels, F / "fig13_bertopic_rating_per_topik.png", lang, "rate_bt", "BT-")
        viz.fig_doc_map(xy, df.platform.values, df.bt_topic.values, bt_labels, F / "fig14_bertopic_peta_dokumen.png", lang)

    # --------------------------------------------------------- 8. summary.json
    def topic_block(ids, rows, dist, rating):
        out = []
        for r in rows:
            t = r["topic"]
            d = dist[dist.topic == t]
            item = {**r, "top_words": r["top_words"].split(", "),
                    "weights": [float(x) for x in r["weights"].split(", ")]}
            for p in ["IMDb", "Letterboxd"]:
                item[f"n_{p}"] = int(d[f"n_{p}"].iloc[0]) if len(d) else 0
                item[f"pct_{p}"] = float(d[f"pct_{p}"].iloc[0]) if len(d) else 0.0
                rr = rating[(rating.topic == t) & (rating.platform == p)]
                item[f"mean_rating_{p}"] = float(rr["mean"].iloc[0]) if len(rr) else None
                item[f"median_rating_{p}"] = float(rr["median"].iloc[0]) if len(rr) else None
            rr = rating[(rating.topic == t) & (rating.platform == "Gabungan")]
            item["mean_rating_all"] = float(rr["mean"].iloc[0]) if len(rr) else None
            out.append(item)
        return out

    def strip(c):
        return {k: v for k, v in c.items() if k != "std_residuals"}

    summary = {
        "meta": {
            "is_dummy": bool(config.USE_DUMMY),
            "PERINGATAN": "HASIL DARI DATA DUMMY - BUKAN HASIL PENELITIAN ASLI" if config.USE_DUMMY else "",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "imdb_csv": str(config.IMDB_CSV.name), "letterboxd_csv": str(config.LETTERBOXD_CSV.name),
            "embedding_backend": emb_backend, "runtime_seconds": round(time.time() - t0, 1),
            "params": {"lda_k_range": [config.LDA_K_RANGE[0], config.LDA_K_RANGE[-1]], "lda_passes": config.LDA_PASSES, "k_selection": config.K_SELECTION, "lda_seeds": config.LDA_SEEDS,
                       "lda_alpha": config.LDA_ALPHA, "dict_no_below": config.DICT_NO_BELOW,
                       "dict_no_above": config.DICT_NO_ABOVE, "umap": config.UMAP_PARAMS,
                       "hdbscan": config.HDBSCAN_PARAMS, "min_topic_size": config.MIN_TOPIC_SIZE,
                       "outlier_strategy": config.OUTLIER_STRATEGY, "top_n_words": config.TOP_N_WORDS,
                       "align_threshold": config.ALIGN_THRESHOLD, "alpha": config.ALPHA,
                       "random_state": config.RANDOM_STATE, "letterboxd_rating_factor": config.LETTERBOXD_RATING_FACTOR},
        },
        "data_selection_stages": stages,
        "review_length_words": length_stats.reset_index().to_dict(orient="records"),
        "rating_10_stats": rating_stats.reset_index().to_dict(orient="records"),
        "lda": {"coherence_by_k": curve.to_dict(orient="records"), "selected_k": best_k, "k_selection_rule": config.K_SELECTION,
                "k_max_cv": int(curve.loc[curve.coherence_cv.idxmax(), "k"]), "seeds_per_k": config.LDA_SEEDS,
                "dictionary_size": len(dictionary), "coherence_cv": metrics["LDA"]["coherence_cv"],
                "topic_diversity": metrics["LDA"]["topic_diversity"],
                "topics": topic_block(lda_ids, lda_topic_rows, lda_dist, rate_lda),
                "mean_topic_proportion": prop.to_dict(orient="records"),
                "chi_square": strip(chi_lda), "kruskal_rating_by_topic": kw_lda},
        "bertopic": {"n_topics": len(bt_ids), "outliers_before_reduce": out_before,
                     "outliers_after_reduce": out_after, "coherence_cv": metrics["BERTopic"]["coherence_cv"],
                     "topic_diversity": metrics["BERTopic"]["topic_diversity"],
                     "topics": topic_block(bt_ids, bt_topic_rows, bt_dist, rate_bt),
                     "chi_square": strip(chi_bt), "kruskal_rating_by_topic": kw_bt},
        "alignment": {"jaccard_matrix": {"rows_lda": lda_ids, "cols_bertopic": bt_ids, "values": J.round(4).tolist()},
                      "pairs": pairs, "only_lda": only_lda, "only_bertopic": only_bt,
                      "threshold": config.ALIGN_THRESHOLD},
        "preference": {"lda": pref_lda.to_dict(orient="records"), "bertopic": pref_bt.to_dict(orient="records")},
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # Semua tabel dalam satu file Excel
    with pd.ExcelWriter(OUT / "hasil_semua_tabel.xlsx") as xw:
        for p in sorted(TAB.glob("*.csv")):
            pd.read_csv(p).to_excel(xw, sheet_name=p.stem[:31], index=False)

    log(f"SELESAI dalam {time.time() - t0:.0f} detik. Hasil di: {OUT}")


if __name__ == "__main__":
    main()
