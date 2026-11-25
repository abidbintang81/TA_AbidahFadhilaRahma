"""Pembuatan gambar (PNG, 200 dpi) dalam bahasa Indonesia ('id') dan Inggris ('en')."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Palet (warna tetap per entitas, bukan per urutan)
C_IMDB = "#2a78d6"      # biru  = IMDb
C_LBX = "#eb6834"       # oranye = Letterboxd
C_LDA = "#1baf7a"       # aqua  = LDA
C_BT = "#6250d6"        # violet = BERTopic
C_MUTED = "#8a8984"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
PLAT_COLOR = {"IMDb": C_IMDB, "Letterboxd": C_LBX}

T = {
    "id": {
        "stage": "Tahap", "count": "Jumlah ulasan", "cleaning_title": "Jumlah ulasan per tahap seleksi data",
        "rating_title": "Distribusi rating ternormalisasi (skala 1–10)", "rating": "Rating (1–10)",
        "share": "Persentase ulasan (%)", "length_title": "Panjang ulasan (jumlah kata)",
        "words": "Jumlah kata per ulasan", "k": "Jumlah topik (K)", "cv": "Coherence C_v",
        "coh_title": "Coherence C_v LDA untuk setiap jumlah topik", "selected": "K terpilih",
        "weight": "Bobot kata", "topwords_lda": "10 kata teratas tiap topik LDA",
        "topwords_bt": "10 kata teratas tiap topik BERTopic (c-TF-IDF)",
        "dist_lda": "Distribusi topik LDA (topik dominan) per platform",
        "dist_bt": "Distribusi topik BERTopic per platform",
        "cmp_title": "Perbandingan kualitas topik", "div": "Topic diversity",
        "jac_title": "Kesamaan Jaccard 10 kata teratas: LDA vs BERTopic",
        "rate_lda": "Rata-rata rating per topik LDA dan platform",
        "rate_bt": "Rata-rata rating per topik BERTopic dan platform",
        "mean_rating": "Rata-rata rating (1–10)", "map_title": "Peta dokumen BERTopic (UMAP 2D)",
        "prop_title": "Rata-rata proporsi topik LDA per platform", "prop": "Rata-rata proporsi (%)",
        "n_topics": "Jumlah topik",
    },
    "en": {
        "stage": "Stage", "count": "Number of reviews", "cleaning_title": "Reviews retained at each data-selection stage",
        "rating_title": "Normalized rating distribution (1–10 scale)", "rating": "Rating (1–10)",
        "share": "Share of reviews (%)", "length_title": "Review length (words)",
        "words": "Words per review", "k": "Number of topics (K)", "cv": "C_v coherence",
        "coh_title": "LDA C_v coherence by number of topics", "selected": "Selected K",
        "weight": "Word weight", "topwords_lda": "Top-10 words per LDA topic",
        "topwords_bt": "Top-10 words per BERTopic topic (c-TF-IDF)",
        "dist_lda": "LDA topic distribution (dominant topic) by platform",
        "dist_bt": "BERTopic topic distribution by platform",
        "cmp_title": "Topic quality comparison", "div": "Topic diversity",
        "jac_title": "Jaccard similarity of top-10 words: LDA vs BERTopic",
        "rate_lda": "Mean rating by LDA topic and platform",
        "rate_bt": "Mean rating by BERTopic topic and platform",
        "mean_rating": "Mean rating (1–10)", "map_title": "BERTopic document map (2D UMAP)",
        "prop_title": "Mean LDA topic proportion by platform", "prop": "Mean proportion (%)",
        "n_topics": "Number of topics",
    },
}
STAGE_EN = {"Data awal": "Initial data", "Setelah hapus ulasan kosong": "Empty removed",
            "Setelah hapus duplikat": "Duplicates removed", "Setelah hapus non-Inggris": "Non-English removed",
            "Setelah hapus ulasan terlalu pendek": "Too-short removed", "Data akhir": "Final data"}
STAGE_ID_SHORT = {"Data awal": "Data awal", "Setelah hapus ulasan kosong": "Hapus kosong",
                  "Setelah hapus duplikat": "Hapus duplikat", "Setelah hapus non-Inggris": "Hapus non-Inggris",
                  "Setelah hapus ulasan terlalu pendek": "Hapus terlalu pendek", "Data akhir": "Data akhir"}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
        "savefig.bbox": "tight", "figure.facecolor": "white",
    })


_style()


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _lab(labels, t, lang, prefix):
    key = "label_id" if lang == "id" else "label_en"
    return f"{prefix}{t} · {labels[t][key]}"


# ---------------------------------------------------------------------------
def fig_cleaning(stages, path, lang):
    tr = T[lang]
    names = [STAGE_ID_SHORT.get(s["tahap"], s["tahap"]) if lang == "id" else STAGE_EN.get(s["tahap"], s["tahap"])
             for s in stages]
    x = np.arange(len(stages)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (p, c) in enumerate(PLAT_COLOR.items()):
        vals = [s[p] for s in stages]
        bars = ax.bar(x + (i - 0.5) * w, vals, w - 0.03, color=c, label=p)
        ax.bar_label(bars, fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel(tr["count"]); ax.set_title(tr["cleaning_title"], loc="left")
    ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
    _save(fig, path)


def fig_rating(df, path, lang):
    tr = T[lang]
    bins = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7, 3.6)); w = 0.4
    for i, (p, c) in enumerate(PLAT_COLOR.items()):
        r = df.loc[df.platform == p, "rating_10"].round().clip(1, 10)
        share = r.value_counts(normalize=True).reindex(bins, fill_value=0) * 100
        ax.bar(bins + (i - 0.5) * w, share.values, w - 0.03, color=c, label=p)
    ax.set_xticks(bins); ax.set_xlabel(tr["rating"]); ax.set_ylabel(tr["share"])
    ax.set_title(tr["rating_title"], loc="left"); ax.legend(); ax.grid(axis="x", visible=False)
    _save(fig, path)


def fig_length(df, path, lang):
    tr = T[lang]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    data = [df.loc[df.platform == p, "n_words_raw"].values for p in PLAT_COLOR]
    bp = ax.boxplot(data, vert=False, tick_labels=list(PLAT_COLOR), patch_artist=True, widths=0.5,
                    medianprops=dict(color=INK, linewidth=1.5), flierprops=dict(markersize=3, markeredgecolor=C_MUTED))
    for patch, c in zip(bp["boxes"], PLAT_COLOR.values()):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("white")
    ax.set_xlabel(tr["words"]); ax.set_title(tr["length_title"], loc="left"); ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_coherence(curve, best_k, path, lang):
    tr = T[lang]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if "coherence_cv_std" in curve:
        ax.fill_between(curve.k, curve.coherence_cv - curve.coherence_cv_std,
                        curve.coherence_cv + curve.coherence_cv_std, color=C_LDA, alpha=0.15, linewidth=0)
    ax.plot(curve.k, curve.coherence_cv, color=C_LDA, lw=2, marker="o", ms=5)
    b = curve[curve.k == best_k].iloc[0]
    ax.scatter([best_k], [b.coherence_cv], s=110, color=C_LDA, edgecolor="white", linewidth=2, zorder=5)
    ax.annotate(f"{tr['selected']} = {best_k}\nC_v = {b.coherence_cv:.4f}", (best_k, b.coherence_cv),
                xytext=(10, -28), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xticks(curve.k); ax.set_xlabel(tr["k"]); ax.set_ylabel(tr["cv"])
    ax.set_title(tr["coh_title"], loc="left")
    _save(fig, path)


def fig_topwords(topic_ids, words_weights, labels, path, lang, title_key, color, prefix):
    tr = T[lang]
    n = len(topic_ids); cols = 3 if n > 4 else 2; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.9), squeeze=False)
    for ax, t, ww in zip(axes.flat, topic_ids, words_weights):
        w = [a for a, _ in ww][::-1]; v = [b for _, b in ww][::-1]
        ax.barh(w, v, color=color, height=0.7)
        ax.set_title(_lab(labels, t, lang, prefix), fontsize=8.8, loc="left")
        ax.tick_params(axis="both", labelsize=7.5); ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(tr[title_key], x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    _save(fig, path)


def fig_distribution(dist, labels, path, lang, title_key, prefix):
    tr = T[lang]
    d = dist.sort_values("topic", ascending=False)
    y = np.arange(len(d)); h = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.4))
    for i, (p, c) in enumerate(PLAT_COLOR.items()):
        bars = ax.barh(y + (0.5 - i) * h, d[f"pct_{p}"], h - 0.04, color=c, label=p)
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, color=INK2, padding=2)
    ax.set_yticks(y, [_lab(labels, int(t), lang, prefix) for t in d.topic], fontsize=8.5)
    ax.set_xlabel(tr["share"]); ax.set_title(tr[title_key], loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08 - 0.6 / (len(d) + 2)), ncol=2)
    ax.grid(axis="y", visible=False); ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
    _save(fig, path)


def fig_lda_proportion(prop, labels, path, lang):
    tr = T[lang]
    d = prop.sort_values("topic", ascending=False)
    y = np.arange(len(d)); h = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.4))
    for i, (p, c) in enumerate(PLAT_COLOR.items()):
        bars = ax.barh(y + (0.5 - i) * h, d[f"mean_prop_{p}"] * 100, h - 0.04, color=c, label=p)
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, color=INK2, padding=2)
    ax.set_yticks(y, [_lab(labels, int(t), lang, "LDA-") for t in d.topic], fontsize=8.5)
    ax.set_xlabel(tr["prop"]); ax.set_title(tr["prop_title"], loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08 - 0.6 / (len(d) + 2)), ncol=2)
    ax.grid(axis="y", visible=False); ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
    _save(fig, path)


def fig_model_compare(metrics, path, lang):
    tr = T[lang]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    keys = [("coherence_cv", tr["cv"]), ("topic_diversity", tr["div"]), ("n_topics", tr["n_topics"])]
    for ax, (k, name) in zip(axes, keys):
        vals = [metrics["LDA"][k], metrics["BERTopic"][k]]
        bars = ax.bar(["LDA", "BERTopic"], vals, color=[C_LDA, C_BT], width=0.55)
        ax.bar_label(bars, fmt="%.3f" if k != "n_topics" else "%d", fontsize=8.5, color=INK, padding=2)
        ax.set_title(name, fontsize=9.5, loc="left"); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle(tr["cmp_title"], x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    _save(fig, path)


def fig_jaccard(J, lda_ids, bt_ids, path, lang):
    tr = T[lang]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#123f78"])
    fig, ax = plt.subplots(figsize=(0.6 * len(bt_ids) + 2.6, 0.5 * len(lda_ids) + 1.6))
    im = ax.imshow(J, cmap=cmap, vmin=0, vmax=max(0.5, J.max()), aspect="auto")
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if J[i, j] > 0.3 else INK)
    ax.set_xticks(range(len(bt_ids)), [f"BT-{t}" for t in bt_ids], fontsize=8)
    ax.set_yticks(range(len(lda_ids)), [f"LDA-{t}" for t in lda_ids], fontsize=8)
    ax.grid(False); ax.set_title(tr["jac_title"], loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Jaccard")
    _save(fig, path)


def fig_rating_topic(rating, labels, path, lang, title_key, prefix):
    tr = T[lang]
    topics = sorted(rating.loc[rating.platform != "Gabungan", "topic"].unique(), reverse=True)
    y = {t: i for i, t in enumerate(topics)}
    fig, ax = plt.subplots(figsize=(8, 0.48 * len(topics) + 1.4))
    for t in topics:
        sub = rating[(rating.topic == t) & (rating.platform != "Gabungan")]
        if len(sub) == 2:
            ax.plot(sub["mean"], [y[t]] * 2, color=GRID, lw=2, zorder=1)
    for p, c in PLAT_COLOR.items():
        sub = rating[rating.platform == p]
        ax.scatter(sub["mean"], [y[t] for t in sub.topic], s=60, color=c, edgecolor="white",
                   linewidth=1.5, label=p, zorder=3)
    ax.set_yticks(list(y.values()), [_lab(labels, int(t), lang, prefix) for t in topics], fontsize=8.5)
    ax.set_xlabel(tr["mean_rating"]); ax.set_xlim(1, 10)
    ax.set_title(tr[title_key], loc="left"); ax.legend(loc="lower left")
    _save(fig, path)


def fig_doc_map(xy, platforms, topics, labels, path, lang):
    tr = T[lang]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for p, c in PLAT_COLOR.items():
        m = platforms == p
        ax.scatter(xy[m, 0], xy[m, 1], s=9, color=c, alpha=0.55, label=p, linewidths=0)
    for t in sorted(set(topics) - {-1}):
        m = topics == t
        cx, cy = np.median(xy[m, 0]), np.median(xy[m, 1])
        ax.text(cx, cy, f"BT-{t}", fontsize=8.5, fontweight="bold", color=INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID, alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(tr["map_title"], loc="left"); ax.legend(loc="best", markerscale=2)
    _save(fig, path)
