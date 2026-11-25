"""
app.py — "Catatan Ulasan": aplikasi Streamlit untuk menjelajahi hasil topic modeling ulasan
film Obsession (2025) di IMDb dan Letterboxd (LDA & BERTopic).

Jalankan:   streamlit run app.py
Tema warna ada di .streamlit/config.toml, gaya tampilan ada di blok CSS di bawah.
"""
import html
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import config

OUT = config.OUTPUT_DIR
TAB = OUT / "tables"

# ---------------------------------------------------------------------------
# Identitas visual
# ---------------------------------------------------------------------------
PAPER, PAPER_2, INK, INK_2, INK_3, RULE = "#f5f1ea", "#ece6dc", "#1d1b18", "#57524a", "#8c857a", "#d9d1c4"
ACCENT = "#9e2a2b"                       # merah layar bioskop: hanya untuk identitas, bukan data
C_IMDB, C_LBX = "#2a78d6", "#eb6834"     # sama dengan gambar di buku TA
C_LDA, C_BT = "#1baf7a", "#6250d6"
PLAT_COLOR = {"IMDb": C_IMDB, "Letterboxd": C_LBX}
MODEL_COLOR = {"LDA": C_LDA, "BERTopic": C_BT}

st.set_page_config(page_title="Catatan Ulasan · Obsession (2025)", page_icon="🎞️", layout="wide",
                   initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, textarea, button {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {{ display: none !important; }}
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ font-family: 'Fraunces', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}

/* ---------- Sidebar: tirai gelap dengan perforasi pita film ---------- */
[data-testid="stSidebar"] {{
  background: #17151a;
  border-right: none;
}}
[data-testid="stSidebar"]::before {{
  content: ""; position: absolute; top: 0; bottom: 0; left: 8px; width: 10px; z-index: 1;
  background: repeating-linear-gradient(to bottom, transparent 0 10px, #2c2830 10px 20px);
  border-radius: 2px;
}}
[data-testid="stSidebar"] * {{ color: #d8d2c8; }}
[data-testid="stSidebarContent"] {{ padding-left: 14px; }}
.masthead {{ padding: 6px 4px 18px 4px; border-bottom: 1px solid #2c2830; margin-bottom: 14px; }}
.masthead .kick {{ font: 500 10.5px/1 'IBM Plex Mono', monospace; letter-spacing: .18em; text-transform: uppercase; color: #8f877c !important; }}
.masthead .title {{ font: 700 30px/1.02 'Fraunces', serif; color: #f4efe7 !important; margin: 8px 0 6px; letter-spacing: -0.02em; }}
.masthead .title span {{ color: #e0605a !important; }}
.masthead .sub {{ font-size: 12.5px; line-height: 1.45; color: #a49c90 !important; }}
.side-label {{ font: 500 10.5px/1 'IBM Plex Mono', monospace; letter-spacing: .16em; text-transform: uppercase;
               color: #7d766c !important; margin: 18px 0 8px 4px; }}

/* navigasi: radio tanpa lingkaran, bergaya daftar isi */
.st-key-nav [role="radiogroup"] {{ gap: 2px; width: 100%; }}
.st-key-nav label[data-testid="stRadioOption"] {{
  padding: 8px 10px; border-radius: 6px; width: 100%; margin: 0; transition: background .15s;
}}
.st-key-nav label[data-testid="stRadioOption"]:hover {{ background: #221f26; }}
.st-key-nav label[data-testid="stRadioOption"] > div > div:first-child,
.st-key-lang label[data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
.st-key-nav label p {{ font-size: 14px; color: #bdb6ab !important; white-space: pre; }}
.st-key-nav label[data-selected="true"] {{ background: #2a2530; box-shadow: inset 3px 0 0 #e0605a; }}
.st-key-nav label[data-selected="true"] p {{ color: #fbf7f0 !important; font-weight: 600; }}
.st-key-lang [role="radiogroup"] {{ background: #221f26; border-radius: 8px; padding: 3px; gap: 2px; display: flex; }}
.st-key-lang label[data-testid="stRadioOption"] {{ flex: 1; justify-content: center; padding: 6px 8px; border-radius: 6px; margin: 0; }}
.st-key-lang label p {{ font: 500 12px/1 'IBM Plex Mono', monospace !important; letter-spacing: .08em; color: #a49c90 !important; }}
.st-key-lang label[data-selected="true"] {{ background: #f4efe7; }}
.st-key-lang label[data-selected="true"] p {{ color: #17151a !important; }}
/* radio di halaman utama: kontrol segmen */
[class*="st-key-pill"] [role="radiogroup"] {{ display: inline-flex; flex-wrap: wrap; gap: 4px; background: {PAPER_2};
  border-radius: 8px; padding: 4px; width: auto; }}
[class*="st-key-pill"] label[data-testid="stRadioOption"] {{ padding: 6px 13px; border-radius: 6px; margin: 0; }}
[class*="st-key-pill"] label[data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
[class*="st-key-pill"] label p {{ font: 500 13px/1 'IBM Plex Mono', monospace !important; color: {INK_2}; }}
[class*="st-key-pill"] label[data-selected="true"] {{ background: #fffdf9; box-shadow: 0 1px 2px rgba(0,0,0,.08); }}
[class*="st-key-pill"] label[data-selected="true"] p {{ color: {INK}; font-weight: 600; }}
.side-note {{ font-size: 12px; line-height: 1.45; padding: 10px 12px; border-radius: 6px; margin-top: 10px; }}
.side-note.dummy {{ background: #3a1f1f; border: 1px solid #5c2b2b; color: #f1c9c5 !important; }}
.side-note.warn {{ background: #352c19; border: 1px solid #5a4920; color: #ecd9a8 !important; }}
.side-note b {{ color: inherit !important; }}
.side-meta {{ font: 400 11px/1.5 'IBM Plex Mono', monospace; color: #7d766c !important; margin-top: 16px; }}

/* ---------- Kepala halaman ---------- */
.eyebrow {{ font: 500 11px/1 'IBM Plex Mono', monospace; letter-spacing: .18em; text-transform: uppercase; color: {ACCENT}; }}
.page-title {{ font: 600 44px/1.05 'Fraunces', serif; color: {INK}; margin: 10px 0 10px; letter-spacing: -0.025em; }}
.lede {{ font-size: 16.5px; line-height: 1.6; color: {INK_2}; max-width: 760px; margin-bottom: 8px; }}
.rule {{ height: 1px; background: {RULE}; margin: 26px 0 22px; border: 0; }}
.section {{ display: flex; align-items: baseline; gap: 12px; margin: 34px 0 12px; }}
.section .no {{ font: 500 12px/1 'IBM Plex Mono', monospace; color: {INK_3}; }}
.section .t {{ font: 600 23px/1.2 'Fraunces', serif; color: {INK}; }}
.section .d {{ font-size: 13.5px; color: {INK_3}; margin-left: auto; text-align: right; max-width: 46%; }}

/* pita data dummy */
.tape {{
  display: flex; gap: 14px; align-items: center; margin: 4px 0 22px;
  padding: 10px 16px; border-radius: 8px; color: #5b1d1d; font-size: 13.5px;
  background: repeating-linear-gradient(135deg, #f6dcd6 0 14px, #f2d2cb 14px 28px);
  border: 1px solid #e3b6ad;
}}
.tape b {{ font: 600 11px/1 'IBM Plex Mono', monospace; letter-spacing: .14em; background: #9e2a2b; color: #fff;
          padding: 5px 8px; border-radius: 4px; white-space: nowrap; }}
.tape code {{ background: rgba(255,255,255,.6); color: #5b1d1d; }}

/* ---------- Kartu angka ---------- */
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0;
         border-top: 2px solid {INK}; border-bottom: 1px solid {RULE}; margin: 8px 0 6px; }}
.kpi {{ padding: 16px 16px 14px 0; border-right: 1px solid {RULE}; margin-right: 16px; }}
.kpi:last-child {{ border-right: none; }}
.kpi .l {{ font: 500 10.5px/1.3 'IBM Plex Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: {INK_3}; }}
.kpi .v {{ font: 600 34px/1.1 'Fraunces', serif; color: {INK}; margin-top: 8px; font-variant-numeric: tabular-nums; }}
.kpi .s {{ font-size: 12.5px; color: {INK_2}; margin-top: 4px; }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }}

/* ---------- Temuan utama ---------- */
.findings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }}
.finding {{ background: #fbf8f3; border: 1px solid {RULE}; border-radius: 10px; padding: 18px 20px; }}
.finding .n {{ font: 500 11px/1 'IBM Plex Mono', monospace; color: {ACCENT}; letter-spacing: .12em; }}
.finding .h {{ font: 600 18px/1.3 'Fraunces', serif; margin: 8px 0 6px; color: {INK}; }}
.finding .p {{ font-size: 14px; line-height: 1.55; color: {INK_2}; }}

/* ---------- Kartu topik ---------- */
.tgrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }}
.tcard {{ background: #fbf8f3; border: 1px solid {RULE}; border-radius: 10px; padding: 16px 18px 14px; }}
.tcard .top {{ display: flex; align-items: center; gap: 10px; }}
.code {{ font: 500 11.5px/1 'IBM Plex Mono', monospace; padding: 5px 7px; border-radius: 4px; color: #fff; }}
.tcard .lab {{ font: 600 17.5px/1.25 'Fraunces', serif; color: {INK}; }}
.tcard .src {{ font-size: 11.5px; color: {INK_3}; margin-left: auto; white-space: nowrap; }}
.chips {{ margin: 12px 0 12px; line-height: 2.05; }}
.chip {{ display: inline-block; padding: 1px 9px; margin: 0 4px 0 0; border-radius: 999px; border: 1px solid {RULE};
         background: {PAPER}; color: {INK}; font-size: 13px; }}
.split {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: {PAPER_2}; gap: 2px; }}
.split div {{ height: 100%; }}
.tmeta {{ display: flex; justify-content: space-between; font-size: 12.5px; color: {INK_2}; margin-top: 7px; }}
.tmeta .m {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; }}

/* ---------- Putusan statistik ---------- */
.verdict {{ border-left: 3px solid {INK}; padding: 6px 0 6px 18px; margin: 10px 0 6px; }}
.verdict .big {{ font: 600 22px/1.35 'Fraunces', serif; color: {INK}; }}
.verdict .math {{ font: 400 13.5px/1.6 'IBM Plex Mono', monospace; color: {INK_2}; margin-top: 6px; }}

/* ---------- Kutipan ulasan ---------- */
.review {{ background: #fbf8f3; border: 1px solid {RULE}; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; }}
.review .hd {{ display: flex; gap: 10px; align-items: center; font-size: 12.5px; color: {INK_2}; margin-bottom: 8px; }}
.review .pl {{ font: 500 11px/1 'IBM Plex Mono', monospace; padding: 4px 7px; border-radius: 4px; color: #fff; }}
.review .rt {{ font: 600 13px/1 'IBM Plex Mono', monospace; color: {INK}; }}
.review .tx {{ font: 400 15.5px/1.6 'Fraunces', serif; color: {INK}; }}
.review .tg {{ margin-left: auto; font: 400 11.5px/1 'IBM Plex Mono', monospace; color: {INK_3}; }}

/* tabel & widget */
[data-testid="stDataFrame"] {{ border: 1px solid {RULE}; border-radius: 8px; }}
[data-testid="stCaptionContainer"] {{ color: {INK_3}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 22px; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 0; font-weight: 500; }}
[data-testid="stMetricValue"] {{ font-family: 'Fraunces', serif; }}
.stButton button[kind="primary"] {{ background: {INK}; border-color: {INK}; }}
.small {{ font-size: 13px; color: {INK_3}; }}
@media (max-width: 760px) {{
  .page-title {{ font-size: 32px; }} .section .d {{ display: none; }}
  .kpi {{ border-right: none; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Template Plotly agar semua grafik serasi
pio.templates["catatan"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color=INK_2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#e4ddd1", zerolinecolor=RULE, linecolor=RULE, ticks="", title_font=dict(size=12.5)),
    yaxis=dict(gridcolor="#e4ddd1", zerolinecolor=RULE, linecolor=RULE, ticks="", title_font=dict(size=12.5)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="", font=dict(size=12.5)),
    hoverlabel=dict(bgcolor="#1d1b18", font=dict(color="#f5f1ea", family="IBM Plex Sans"), bordercolor="#1d1b18"),
    margin=dict(l=10, r=10, t=36, b=10), bargap=0.28, bargroupgap=0.08,
    colorway=[C_IMDB, C_LBX, C_LDA, C_BT],
))
pio.templates.default = "catatan"
def show(f, where=None, **kw):
    f.update_layout(legend_title_text="")
    (where or st).plotly_chart(f, use_container_width=True, config=PCFG, **kw)


PCFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_outputs(mtime):
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    docs = pd.read_csv(OUT / "documents_with_topics.csv")
    tabs = {p.stem: pd.read_csv(p) for p in sorted(TAB.glob("*.csv"))}
    xy = np.load(OUT / "doc_map_2d.npy") if (OUT / "doc_map_2d.npy").exists() else None
    return summary, docs, tabs, xy


def esc(x):
    return html.escape(str(x))


def num(x, d=3, lang="id"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".") if lang == "id" else s


def pval(p, lang="id"):
    return ("< 0,001" if lang == "id" else "< 0.001") if p < 0.001 else num(p, 3, lang)


def header(eyebrow, title, lede=None):
    st.markdown(f'<div class="eyebrow">{esc(eyebrow)}</div><div class="page-title">{esc(title)}</div>'
                + (f'<div class="lede">{lede}</div>' if lede else ""), unsafe_allow_html=True)


def section(no, title, desc=""):
    st.markdown(f'<div class="section">{f"<span class=no>{esc(no)}</span>" if no else ""}<span class="t">{esc(title)}</span>'
                f'<span class="d">{desc}</span></div>', unsafe_allow_html=True)


def kpis(items):
    cells = "".join(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div>'
                    f'<div class="s">{s}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def tape(summary):
    if summary["meta"]["is_dummy"]:
        st.markdown('<div class="tape"><b>DATA DUMMY</b><span>Semua angka di halaman ini berasal dari data '
                    'sintetis untuk uji program, bukan temuan penelitian. Ganti data dengan mengubah '
                    '<code>USE_DUMMY = False</code> di <code>config.py</code>.</span></div>', unsafe_allow_html=True)


def tlabel(key, t, s, lang):
    for tp in s[key]["topics"]:
        if tp["topic"] == t:
            return tp["label_" + lang]
    return "Outlier" if t == -1 else str(t)


def tcode(key, t):
    return f"{'LDA' if key == 'lda' else 'BT'}-{t}"


def plat_pill(p):
    return f'<span class="pl" style="background:{PLAT_COLOR[p]}">{esc(p)}</span>'


def review_card(r, lang, tags=""):
    rating = num(r["rating_10"], 1, lang).rstrip("0").rstrip(",.") if r["rating_10"] == r["rating_10"] else "–"
    st.markdown(f'<div class="review"><div class="hd">{plat_pill(r["platform"])}'
                f'<span class="rt">★ {rating}/10</span><span class="tg">{tags}</span></div>'
                f'<div class="tx">{esc(r["text"])}</div></div>', unsafe_allow_html=True)


def topic_cards(key, s, lang):
    color = C_LDA if key == "lda" else C_BT
    cards = []
    for tp in s[key]["topics"]:
        w = np.array(tp["weights"]); w = (w - w.min()) / (w.max() - w.min() + 1e-9)
        chips = "".join(f'<span class="chip" style="font-size:{12 + 3.5 * x:.1f}px;'
                        f'{"font-weight:600;" if x > .6 else ""}">{esc(word)}</span>'
                        for word, x in zip(tp["top_words"], w))
        a, b = tp["pct_IMDb"], tp["pct_Letterboxd"]
        src = "label manual" if tp["label_source"].startswith("manual") else "label usulan"
        cards.append(
            f'<div class="tcard"><div class="top"><span class="code" style="background:{color}">{tcode(key, tp["topic"])}</span>'
            f'<span class="lab">{esc(tp["label_" + lang])}</span><span class="src">{src}</span></div>'
            f'<div class="chips">{chips}</div>'
            f'<div class="split"><div style="width:{a / (a + b + 1e-9) * 100:.1f}%;background:{C_IMDB}"></div>'
            f'<div style="flex:1;background:{C_LBX}"></div></div>'
            f'<div class="tmeta"><span><span class="dot" style="background:{C_IMDB}"></span>IMDb {num(a, 1, lang)}%</span>'
            f'<span class="m">C_v {num(tp["coherence_cv"], 3, lang)}</span>'
            f'<span>Letterboxd {num(b, 1, lang)}%<span class="dot" style="background:{C_LBX};margin:0 0 0 6px"></span></span></div></div>')
    st.markdown(f'<div class="tgrid">{"".join(cards)}</div>', unsafe_allow_html=True)


def dist_chart(key, s, lang):
    tp = pd.DataFrame(s[key]["topics"])
    tp["nama"] = tp.topic.apply(lambda t: f"{tcode(key, t)}  {tlabel(key, t, s, lang)}")
    long = tp.melt(id_vars=["nama", "topic"], value_vars=["pct_IMDb", "pct_Letterboxd"], var_name="platform", value_name="pct")
    long["platform"] = long.platform.str.replace("pct_", "")
    f = px.bar(long, y="nama", x="pct", color="platform", barmode="group", orientation="h",
               color_discrete_map=PLAT_COLOR, text="pct", custom_data=["platform"])
    f.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_size=11.5, cliponaxis=False,
                    marker_line_width=0, hovertemplate="%{customdata[0]} · %{y}<br><b>%{x:.1f}%</b> ulasan<extra></extra>")
    f.update_layout(height=110 + 58 * len(tp), yaxis_title=None, xaxis_title="% ulasan pada platform",
                    yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                    xaxis_range=[0, long.pct.max() * 1.18])
    return f


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = ["Ringkasan", "Data & Preprocessing", "Topik LDA", "Topik BERTopic", "Evaluasi & Pencocokan",
         "Perbandingan Platform", "Rating & Preferensi", "Penjelajah Ulasan", "Jalankan Ulang"]

with st.sidebar:
    st.markdown('<div class="masthead"><div class="kick">Catatan Ulasan · TA</div>'
                '<div class="title">Obsession<span>.</span></div>'
                '<div class="sub">Membaca 1.060 ulasan IMDb dan Letterboxd dengan LDA dan BERTopic untuk melihat '
                'apa yang dibicarakan penonton dan apa yang mereka sukai.</div></div>', unsafe_allow_html=True)
    has_out = (OUT / "summary.json").exists()
    st.markdown('<div class="side-label">Halaman</div>', unsafe_allow_html=True)
    with st.container(key="nav"):
        page = st.radio("Halaman", PAGES if has_out else ["Jalankan Ulang"], label_visibility="collapsed",
                        format_func=lambda p: f"{PAGES.index(p) + 1:02d}   {p}")
    st.markdown('<div class="side-label">Bahasa label topik</div>', unsafe_allow_html=True)
    with st.container(key="lang"):
        lang = st.radio("Bahasa label", ["id", "en"], horizontal=True, label_visibility="collapsed",
                        format_func=lambda x: "INDONESIA" if x == "id" else "ENGLISH")

if has_out:
    summary, docs, tabs, XY = load_outputs((OUT / "summary.json").stat().st_mtime)
    meta = summary["meta"]
    with st.sidebar:
        if meta["is_dummy"]:
            st.markdown('<div class="side-note dummy"><b>Data dummy.</b> Hasil ini untuk menguji program, '
                        'bukan temuan penelitian.</div>', unsafe_allow_html=True)
        if "FALLBACK" in meta["embedding_backend"]:
            st.markdown('<div class="side-note warn"><b>Embedding cadangan.</b> BERTopic memakai TF-IDF+SVD, '
                        'bukan all-MiniLM-L6-v2. Jalankan ulang di komputer yang terhubung internet.</div>',
                        unsafe_allow_html=True)
        st.markdown(f'<div class="side-meta">run {esc(meta["generated_at"].replace("T", " "))}<br>'
                    f'{esc(meta["imdb_csv"])}<br>{esc(meta["letterboxd_csv"])}</div>', unsafe_allow_html=True)
else:
    summary = docs = tabs = XY = None

L, B = (summary["lda"], summary["bertopic"]) if summary else (None, None)

# ---------------------------------------------------------------------------
# 01 Ringkasan
# ---------------------------------------------------------------------------
if page == "Ringkasan":
    fin = summary["data_selection_stages"][-1]
    header("01 · Ringkasan", "Apa yang dibicarakan penonton Obsession?",
           f"Ulasan dari dua komunitas film dimodelkan bersama dalam satu ruang topik, lalu dibandingkan. "
           f"Dari {num(summary['data_selection_stages'][0]['Total'], 0, 'id')} ulasan awal, {num(fin['Total'], 0, 'id')} lolos seleksi "
           f"dan dianalisis dengan LDA ({L['selected_k']} topik) dan BERTopic ({B['n_topics']} topik).")
    tape(summary)
    kpis([
        ("Ulasan dianalisis", f"{fin['Total']:,}".replace(",", "."), "setelah seleksi data"),
        ("IMDb", f"{fin['IMDb']}", f'<span class="dot" style="background:{C_IMDB}"></span>{num(fin["IMDb"] / fin["Total"] * 100, 1, lang)}% korpus'),
        ("Letterboxd", f"{fin['Letterboxd']}", f'<span class="dot" style="background:{C_LBX}"></span>{num(fin["Letterboxd"] / fin["Total"] * 100, 1, lang)}% korpus'),
        ("C_v LDA", num(L["coherence_cv"], 3, lang), f"diversity {num(L['topic_diversity'], 2, lang)}"),
        ("C_v BERTopic", num(B["coherence_cv"], 3, lang), f"diversity {num(B['topic_diversity'], 2, lang)}"),
    ])

    section("", "Temuan utama", "dihitung otomatis dari hasil model")
    f_items = []
    for key, name in (("lda", "LDA"), ("bertopic", "BERTopic")):
        pref = pd.DataFrame(summary["preference"][key])
        ti = pref[(pref.platform == "IMDb") & (pref["rank"] == 1)].iloc[0]
        tl = pref[(pref.platform == "Letterboxd") & (pref["rank"] == 1)].iloc[0]
        f_items.append((f"{name} · TOPIK DOMINAN",
                        f"IMDb: {ti['label_' + lang]}. Letterboxd: {tl['label_' + lang]}.",
                        f"Di IMDb, {num(ti.pct, 1, lang)}% ulasan paling banyak membahas <b>{esc(ti['label_' + lang])}</b> "
                        f"(rata-rata rating {num(ti.mean_rating_10, 2, lang)}). Di Letterboxd, porsi terbesar "
                        f"({num(tl.pct, 1, lang)}%) ada pada <b>{esc(tl['label_' + lang])}</b> "
                        f"(rating {num(tl.mean_rating_10, 2, lang)})."))
    cs_l, cs_b = L["chi_square"], B["chi_square"]
    sig = cs_l["significant"] and cs_b["significant"]
    f_items.append(("UJI CHI-SQUARE",
                    "Komposisi topik kedua platform " + ("berbeda nyata." if sig else "tidak selalu berbeda nyata."),
                    f"LDA: χ²({cs_l['df']}) = {num(cs_l['chi2'], 2, lang)}, p {pval(cs_l['p_value'], lang)}, V = {num(cs_l['cramers_v'], 3, lang)}. "
                    f"BERTopic: χ²({cs_b['df']}) = {num(cs_b['chi2'], 2, lang)}, p {pval(cs_b['p_value'], lang)}, "
                    f"V = {num(cs_b['cramers_v'], 3, lang)}. Nilai V di kisaran ini menunjukkan perbedaan berukuran lemah sampai sedang."))
    al = summary["alignment"]
    n_cons = sum(p["status"] == "konsisten" for p in al["pairs"])
    f_items.append(("PENCOCOKAN LDA–BERTOPIC",
                    f"{n_cons} topik ditemukan oleh kedua metode.",
                    f"Pasangan dengan Jaccard ≥ {num(al['threshold'], 2, lang)} (minimal 3 dari 10 kata teratas sama) dianggap konsisten. "
                    f"Topik pelengkap: LDA {', '.join(map(str, al['only_lda'])) or '–'}; BERTopic {', '.join(map(str, al['only_bertopic'])) or '–'}."))
    st.markdown('<div class="findings">' + "".join(
        f'<div class="finding"><div class="n">{n}</div><div class="h">{esc(h)}</div><div class="p">{p}</div></div>'
        for n, h, p in f_items) + "</div>", unsafe_allow_html=True)

    section("", "Distribusi topik BERTopic per platform", "persentase dihitung per platform agar setara")
    show(dist_chart("bertopic", summary, lang))

# ---------------------------------------------------------------------------
# 02 Data & Preprocessing
# ---------------------------------------------------------------------------
elif page == "Data & Preprocessing":
    header("02 · Data & Preprocessing", "Dari data mentah ke korpus bersih",
           "Ulasan disaring dari teks kosong, duplikat, bahasa selain Inggris, dan ulasan yang terlalu pendek. "
           "Rating Letterboxd (0,5–5) dikalikan dua agar setara dengan skala 1–10 IMDb.")
    tape(summary)
    stg = pd.DataFrame(summary["data_selection_stages"])
    section("", "Seleksi data per tahap")
    short = {"Data awal": "Data awal", "Setelah hapus ulasan kosong": "Hapus kosong", "Setelah hapus duplikat": "Hapus duplikat",
             "Setelah hapus non-Inggris": "Hapus non-Inggris", "Setelah hapus ulasan terlalu pendek": "Hapus terlalu pendek"}
    stg["Tahap"] = stg.tahap.map(short).fillna(stg.tahap)
    f = go.Figure()
    for p, c in PLAT_COLOR.items():
        f.add_bar(x=stg["Tahap"], y=stg[p], name=p, marker_color=c, text=stg[p], textposition="outside",
                  cliponaxis=False, hovertemplate=f"{p} · %{{x}}<br><b>%{{y}}</b> ulasan<extra></extra>")
    f.update_layout(barmode="group", height=360, yaxis_title="Jumlah ulasan", xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    c1, c2 = st.columns([1.55, 1])
    show(f, c1)
    with c2:
        st.dataframe(stg[["Tahap", "IMDb", "Letterboxd", "Total"]], hide_index=True, use_container_width=True, height=230)
        lost = stg.Total.iloc[0] - stg.Total.iloc[-1]
        st.markdown(f'<p class="small">{lost} ulasan ({num(lost / stg.Total.iloc[0] * 100, 1, lang)}%) dikeluarkan '
                    f'selama seleksi.</p>', unsafe_allow_html=True)

    section("", "Profil ulasan")
    c1, c2 = st.columns(2)
    with c1:
        f = px.histogram(docs, x="rating_10", color="platform", barmode="group", histnorm="percent",
                         nbins=10, color_discrete_map=PLAT_COLOR, range_x=[0.5, 10.5])
        f.update_traces(marker_line_width=0, hovertemplate="rating %{x}<br><b>%{y:.1f}%</b><extra></extra>")
        f.update_layout(height=330, yaxis_title="% ulasan", xaxis_title="Rating (1–10)", title="Distribusi rating",
                        title_font=dict(family="Fraunces, serif", size=17, color=INK))
        show(f)
    with c2:
        f = px.box(docs, x="n_words_raw", y="platform", color="platform", color_discrete_map=PLAT_COLOR, points=False)
        f.update_layout(height=330, showlegend=False, xaxis_title="Jumlah kata per ulasan", yaxis_title=None,
                        title="Panjang ulasan", title_font=dict(family="Fraunces, serif", size=17, color=INK))
        show(f)

    section("", "Contoh tahapan preprocessing", "untuk LDA; BERTopic hanya memakai tahap cleaning")
    ex = tabs["02_contoh_preprocessing"]
    i = st.select_slider("Contoh ulasan", options=list(range(len(ex))), format_func=lambda x: f"#{x + 1}")
    r = ex.iloc[i]
    steps = [("1 · Teks asli", r["1_teks_asli"]), ("2 · Cleaning", r["2_cleaning"]), ("3 · Case folding", r["3_case_folding"]),
             ("4 · Tokenisasi", r["4_tokenisasi"]), ("5 · Stopword removal", r["5_stopword_removal"]), ("6 · Lemmatisasi", r["6_lemmatisasi"])]
    for name, val in steps:
        st.markdown(f'<div class="review" style="padding:12px 18px"><div class="hd"><span class="rt">{esc(name)}</span></div>'
                    f'<div style="font:400 13px/1.6 IBM Plex Mono,monospace;color:{INK}">{esc(str(val)[:700])}</div></div>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 03/04 Topik LDA dan BERTopic
# ---------------------------------------------------------------------------
elif page in ("Topik LDA", "Topik BERTopic"):
    key = "lda" if page == "Topik LDA" else "bertopic"
    name = "LDA" if key == "lda" else "BERTopic"
    if key == "lda":
        header("03 · Topik LDA", f"{L['selected_k']} topik dari model probabilistik",
               f"LDA dilatih untuk K = {meta['params']['lda_k_range'][0]}–{meta['params']['lda_k_range'][1]}, "
               f"masing-masing {L.get('seeds_per_k', 1)} kali dengan seed berbeda. K dipilih pada puncak pertama "
               f"kurva coherence C_v (aturan <code>{esc(L.get('k_selection_rule', 'max_cv'))}</code>).")
    else:
        header("04 · Topik BERTopic", f"{B['n_topics']} topik dari klaster makna kalimat",
               "Setiap ulasan diubah menjadi sentence embedding, direduksi dengan UMAP, dikelompokkan dengan HDBSCAN, "
               "lalu kata pewakil tiap klaster dihitung dengan c-TF-IDF.")
    tape(summary)
    if key == "lda":
        cur = pd.DataFrame(L["coherence_by_k"])
        f = go.Figure()
        if "coherence_cv_std" in cur:
            f.add_scatter(x=list(cur.k) + list(cur.k[::-1]),
                          y=list(cur.coherence_cv + cur.coherence_cv_std) + list((cur.coherence_cv - cur.coherence_cv_std)[::-1]),
                          fill="toself", fillcolor="rgba(27,175,122,.13)", line=dict(width=0), hoverinfo="skip", showlegend=False)
        f.add_scatter(x=cur.k, y=cur.coherence_cv, mode="lines+markers", line=dict(color=C_LDA, width=2.5),
                      marker=dict(size=8, color=C_LDA, line=dict(color=PAPER, width=2)), showlegend=False,
                      hovertemplate="K = %{x}<br><b>C_v %{y:.4f}</b><extra></extra>")
        k = L["selected_k"]; v = cur[cur.k == k].coherence_cv.iloc[0]
        f.add_scatter(x=[k], y=[v], mode="markers", marker=dict(size=16, color=C_LDA, line=dict(color=INK, width=2)),
                      showlegend=False, hoverinfo="skip")
        f.add_annotation(x=k, y=v, text=f"<b>K terpilih = {k}</b><br>C_v {num(v, 4, lang)}", showarrow=True, arrowhead=0,
                         ax=40, ay=50, font=dict(size=12.5, color=INK), arrowcolor=INK_3)
        f.update_layout(height=330, xaxis=dict(title="Jumlah topik (K)", dtick=1), yaxis_title="C_v (rata-rata)")
        kpis([("K terpilih", k, f"K dengan C_v maksimum: {L.get('k_max_cv', '–')}"),
              ("C_v model akhir", num(L["coherence_cv"], 3, lang), "seed 42 · kurva = rata-rata 3 seed"),
              ("Topic diversity", num(L["topic_diversity"], 3, lang), "proporsi kata unik"),
              ("Kosakata", f"{L['dictionary_size']}", "setelah filter ekstrem")])
        section("", "Kurva coherence C_v", "pita = ±1 simpangan baku antar-seed")
        show(f)
    else:
        n_all = int(docs.shape[0])
        kpis([("Topik", B["n_topics"], "tanpa topik outlier"),
              ("Outlier awal", B["outliers_before_reduce"], f"{num(B['outliers_before_reduce'] / n_all * 100, 1, lang)}% ulasan"),
              ("Setelah reduce_outliers", B["outliers_after_reduce"], f"strategi {esc(meta['params']['outlier_strategy'])}"),
              ("C_v model", num(B["coherence_cv"], 3, lang), f"diversity {num(B['topic_diversity'], 3, lang)}")])
        st.markdown(f'<p class="small">Embedding: <code>{esc(meta["embedding_backend"])}</code></p>', unsafe_allow_html=True)

    section("", "Topik dan kata pewakilnya", "ukuran kata = bobot · pita = porsi IMDb vs Letterboxd")
    topic_cards(key, summary, lang)

    section("", "Ulasan representatif", "contoh untuk membantu pelabelan topik")
    reps = tabs["09_contoh_ulasan_per_topik"]; reps = reps[reps.model == name]
    ids = [tp["topic"] for tp in summary[key]["topics"]]
    with st.container(key="pill_topic"):
        t_sel = st.radio("Topik", ids, horizontal=True, format_func=lambda t: tcode(key, t), label_visibility="collapsed")
    st.markdown(f'<p class="small">{tcode(key, t_sel)} · {esc(tlabel(key, t_sel, summary, lang))}</p>', unsafe_allow_html=True)
    for _, r in reps[reps.topic == t_sel].iterrows():
        tag = f"probabilitas topik {num(r.prob, 2, lang)}" if r.prob == r.prob else "dokumen pewakil c-TF-IDF"
        review_card(r, lang, tag)

    if key == "bertopic" and XY is not None:
        section("", "Peta dokumen", "UMAP 2 dimensi · arahkan kursor untuk membaca cuplikan")
        with st.container(key="pill_map"):
            warna = st.radio("Warna", ["Topik", "Platform"], horizontal=True, key="mapc", label_visibility="collapsed")
        dm = docs.assign(x=XY[:, 0], y=XY[:, 1], cuplikan=docs.text_clean.str[:110] + "…",
                         Topik=docs.bt_topic.apply(lambda t: f"BT-{t} {tlabel('bertopic', t, summary, lang)}"))
        palette = [C_BT, "#1baf7a", "#eda100", "#e87ba4", "#2a78d6", "#008300", "#e34948", "#8a8984", "#eb6834", "#5b4b8a"]
        f = px.scatter(dm, x="x", y="y", color="Topik" if warna == "Topik" else "platform",
                       color_discrete_map=PLAT_COLOR if warna == "Platform" else None,
                       color_discrete_sequence=palette, custom_data=["platform", "rating_10", "cuplikan"], opacity=.75,
                       category_orders={"Topik": sorted(dm.Topik.unique(), key=lambda x: int(x.split()[0][3:]))})
        f.update_traces(marker=dict(size=6, line=dict(width=0)),
                        hovertemplate="<b>%{customdata[0]}</b> · ★ %{customdata[1]}<br>%{customdata[2]}<extra></extra>")
        f.update_layout(height=560, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        legend=dict(orientation="v", y=1, x=1.01, xanchor="left", yanchor="top"))
        show(f)

# ---------------------------------------------------------------------------
# 05 Evaluasi & Pencocokan
# ---------------------------------------------------------------------------
elif page == "Evaluasi & Pencocokan":
    header("05 · Evaluasi & Pencocokan", "Dua metode, satu peta topik",
           "Kualitas topik diukur dengan coherence C_v dan topic diversity pada 10 kata teratas. Topik LDA dan BERTopic "
           "lalu dipasangkan berdasarkan kemiripan kata teratas (Jaccard) agar keduanya saling melengkapi, bukan dipertandingkan.")
    tape(summary)
    c1, c2, c3 = st.columns(3)
    for col, (k, nm) in zip((c1, c2, c3), [("coherence_cv", "Coherence C_v"), ("topic_diversity", "Topic diversity"), ("n", "Jumlah topik")]):
        vals = [L["coherence_cv"], B["coherence_cv"]] if k == "coherence_cv" else \
               [L["topic_diversity"], B["topic_diversity"]] if k == "topic_diversity" else [L["selected_k"], B["n_topics"]]
        f = go.Figure(go.Bar(x=["LDA", "BERTopic"], y=vals, marker_color=[C_LDA, C_BT], width=.55,
                             text=[num(v, 3, lang) if k != "n" else str(v) for v in vals], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br><b>%{y}</b><extra></extra>"))
        f.update_layout(height=260, title=nm, title_font=dict(family="Fraunces, serif", size=17, color=INK),
                        yaxis=dict(range=[0, max(vals) * 1.25], showticklabels=False), xaxis=dict(gridcolor="rgba(0,0,0,0)"))
        show(f, col)

    al = summary["alignment"]
    section("", "Matriks kemiripan Jaccard", "baris = topik LDA · kolom = topik BERTopic")
    jm = pd.DataFrame(al["jaccard_matrix"]["values"],
                      index=[f"LDA-{t} {tlabel('lda', t, summary, lang)[:22]}" for t in al["jaccard_matrix"]["rows_lda"]],
                      columns=[f"BT-{t}" for t in al["jaccard_matrix"]["cols_bertopic"]])
    f = px.imshow(jm, text_auto=".2f", color_continuous_scale=["#fbf8f3", "#c9dcf3", "#6ea3e3", "#2a78d6", "#123f78"],
                  aspect="auto", zmin=0, zmax=max(.5, float(jm.values.max())))
    f.update_traces(hovertemplate="%{y} × %{x}<br><b>Jaccard %{z:.3f}</b><extra></extra>", xgap=2, ygap=2)
    f.update_layout(height=120 + 52 * len(jm), coloraxis_colorbar=dict(thickness=10, title=""),
                    xaxis=dict(side="top", gridcolor="rgba(0,0,0,0)"), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    show(f)

    section("", "Pasangan topik", f"algoritma Hungaria · ambang Jaccard ≥ {num(al['threshold'], 2, lang)}")
    rows = []
    for p in al["pairs"]:
        ok = p["status"] == "konsisten"
        rows.append(f'<div class="review" style="display:flex;gap:16px;align-items:center;padding:12px 18px;'
                    f'{"" if ok else "opacity:.62"}">'
                    f'<span class="code" style="background:{C_LDA}">LDA-{p["lda_topic"]}</span>'
                    f'<span style="flex:1;font:600 15px/1.3 Fraunces,serif">{esc(tlabel("lda", p["lda_topic"], summary, lang))}</span>'
                    f'<span style="font:500 13px IBM Plex Mono,monospace;color:{INK if ok else INK_3}">'
                    f'{"⟷" if ok else "··"} {num(p["jaccard"], 2, lang)}</span>'
                    f'<span style="flex:1;text-align:right;font:600 15px/1.3 Fraunces,serif">{esc(tlabel("bertopic", p["bertopic_topic"], summary, lang))}</span>'
                    f'<span class="code" style="background:{C_BT}">BT-{p["bertopic_topic"]}</span></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.markdown(f'<p class="small">Pasangan yang pudar berada di bawah ambang dan diperlakukan sebagai topik pelengkap. '
                f'Hanya di LDA: {", ".join(map(str, al["only_lda"])) or "–"} · hanya di BERTopic: '
                f'{", ".join(map(str, al["only_bertopic"])) or "–"}.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 06 Perbandingan Platform
# ---------------------------------------------------------------------------
elif page == "Perbandingan Platform":
    header("06 · Perbandingan Platform", "IMDb dan Letterboxd membicarakan hal yang berbeda?",
           "Persentase dihitung terhadap jumlah ulasan di masing-masing platform, sehingga perbedaan ukuran data "
           "(IMDb lebih banyak) tidak memengaruhi perbandingan. Perbedaan diuji dengan chi-square independensi.")
    tape(summary)
    tab1, tab2 = st.tabs(["BERTopic", "LDA"])
    for tab, key in ((tab1, "bertopic"), (tab2, "lda")):
        with tab:
            cs = summary[key]["chi_square"]
            verdict = ("Distribusi topik berbeda secara signifikan antara IMDb dan Letterboxd."
                       if cs["significant"] else "Distribusi topik tidak berbeda signifikan antara kedua platform.")
            st.markdown(f'<div class="verdict"><div class="big">{verdict}</div>'
                        f'<div class="math">χ²({cs["df"]}, N = {cs["n"]}) = {num(cs["chi2"], 3, lang)} · p {pval(cs["p_value"], lang)} · '
                        f"Cramér's V = {num(cs['cramers_v'], 3, lang)} · α = {num(cs['alpha'], 2, lang)} · "
                        f'sel E &lt; 5: {cs["cells_expected_lt5"]}/{cs["cells_total"]}</div></div>', unsafe_allow_html=True)
            show(dist_chart(key, summary, lang), key=f"d{key}")
            res = tabs["15_lda_residual_chisquare" if key == "lda" else "16_bertopic_residual_chisquare"].copy()
            tcol = res.columns[0]
            res["Topik"] = res[tcol].apply(lambda t: f"{tcode(key, t)} {tlabel(key, t, summary, lang)}")
            section("", "Topik penyumbang perbedaan", "residual terstandar · |r| > 1,96 signifikan")
            f = go.Figure()
            for p, c in PLAT_COLOR.items():
                f.add_bar(y=res["Topik"], x=res[p], name=p, orientation="h", marker_color=c,
                          hovertemplate=f"{p} · %{{y}}<br><b>r = %{{x:.2f}}</b><extra></extra>")
            for xv in (-1.96, 1.96):
                f.add_vline(x=xv, line_dash="dot", line_color=INK_3, line_width=1)
            f.update_layout(barmode="group", height=110 + 52 * len(res), xaxis_title="Residual terstandar",
                            yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"))
            show(f, key=f"r{key}")
            if key == "lda":
                section("", "Rata-rata proporsi topik", "alternatif dari topik dominan, memakai seluruh distribusi θ")
                pr = tabs["14_lda_rata_proporsi_platform"].copy()
                pr["Topik"] = pr.topic.apply(lambda t: f"LDA-{t} {tlabel('lda', t, summary, lang)}")
                for p in PLAT_COLOR:
                    pr[p] = (pr[f"mean_prop_{p}"] * 100).round(1)
                st.dataframe(pr[["Topik", "IMDb", "Letterboxd"]], hide_index=True, use_container_width=True,
                             column_config={p: st.column_config.ProgressColumn(f"{p} (%)", format="%.1f", min_value=0,
                                                                               max_value=float(pr[["IMDb", "Letterboxd"]].values.max()))
                                            for p in PLAT_COLOR})

# ---------------------------------------------------------------------------
# 07 Rating & Preferensi
# ---------------------------------------------------------------------------
elif page == "Rating & Preferensi":
    header("07 · Rating & Preferensi", "Topik mana yang disukai penonton?",
           "Preferensi dibaca dari dua sisi: seberapa sering sebuah topik dibicarakan di suatu platform, dan "
           "berapa rata-rata rating ulasan yang membahasnya (skala 1–10).")
    tape(summary)
    with st.container(key="pill_model"):
        m = st.radio("Model", ["BERTopic", "LDA"], horizontal=True, label_visibility="collapsed")
    key = m.lower(); col = "lda_topic" if key == "lda" else "bt_topic"
    tp = pd.DataFrame(summary[key]["topics"])
    rows = []
    for p in PLAT_COLOR:
        rows.append(tp.assign(platform=p, pct=tp[f"pct_{p}"], rating=tp[f"mean_rating_{p}"], n=tp[f"n_{p}"]))
    bub = pd.concat(rows)
    bub["nama"] = bub.topic.apply(lambda t: f"{tcode(key, t)} {tlabel(key, t, summary, lang)}")
    section("", "Peta preferensi", "kanan = sering dibahas · atas = rating tinggi · ukuran = jumlah ulasan")
    f = px.scatter(bub, x="pct", y="rating", size="n", color="platform", color_discrete_map=PLAT_COLOR,
                   text=bub.topic.apply(lambda t: tcode(key, t)), custom_data=["nama", "n"], size_max=42)
    f.update_traces(textposition="middle center", textfont=dict(size=10, color="white"), marker=dict(line=dict(width=1.5, color=PAPER), opacity=.88),
                    hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1f}% ulasan · rating %{y:.2f} · n = %{customdata[1]}<extra></extra>")
    f.add_hline(y=float(docs.rating_10.mean()), line_dash="dot", line_color=INK_3, line_width=1,
                annotation_text="rata-rata semua ulasan", annotation_font_size=11, annotation_font_color=INK_3)
    f.update_layout(height=470, xaxis_title="% ulasan pada platform", yaxis_title="Rata-rata rating (1–10)")
    show(f)

    section("", "Sebaran rating per topik")
    d = docs[docs[col] != -1].copy()
    d["Topik"] = d[col].apply(lambda t: f"{tcode(key, t)} {tlabel(key, t, summary, lang)}")
    f = px.box(d, x="rating_10", y="Topik", color="platform", color_discrete_map=PLAT_COLOR, points=False)
    f.update_layout(height=140 + 60 * d[col].nunique(), yaxis_title=None, xaxis_title="Rating (1–10)",
                    yaxis=dict(categoryorder="category descending", gridcolor="rgba(0,0,0,0)"))
    show(f)
    kw = summary[key]["kruskal_rating_by_topic"]
    if kw:
        st.markdown(f'<div class="verdict"><div class="big">Rating {"berbeda" if kw["significant"] else "tidak berbeda"} '
                    f'secara signifikan antartopik.</div><div class="math">Kruskal–Wallis H({kw["df"]}) = {num(kw["H"], 3, lang)} · '
                    f'p {pval(kw["p_value"], lang)}</div></div>', unsafe_allow_html=True)
    section("", "Peringkat topik per platform")
    pref = pd.DataFrame(summary["preference"][key])
    c1, c2 = st.columns(2)
    for c, p in ((c1, "IMDb"), (c2, "Letterboxd")):
        sub = pref[pref.platform == p].copy()
        sub["Topik"] = sub.topic.apply(lambda t: f"{tcode(key, t)} {tlabel(key, t, summary, lang)}")
        c.markdown(f'<p style="font:600 16px Fraunces,serif;margin:0 0 6px"><span class="dot" style="background:{PLAT_COLOR[p]}"></span>{p}</p>',
                   unsafe_allow_html=True)
        c.dataframe(sub[["Topik", "pct", "mean_rating_10"]].rename(columns={"pct": "% ulasan", "mean_rating_10": "Rating"}),
                    hide_index=True, use_container_width=True,
                    column_config={"Topik": st.column_config.TextColumn(width="medium"),
                                   "% ulasan": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=float(pref.pct.max()), width="small"),
                                   "Rating": st.column_config.NumberColumn(format="%.2f")})

# ---------------------------------------------------------------------------
# 08 Penjelajah Ulasan
# ---------------------------------------------------------------------------
elif page == "Penjelajah Ulasan":
    header("08 · Penjelajah Ulasan", "Baca ulasannya langsung",
           "Saring ulasan berdasarkan platform, topik, rating, atau kata kunci untuk memeriksa apakah label topik "
           "sesuai dengan isi ulasan.")
    tape(summary)
    c1, c2, c3 = st.columns([1.1, 1.4, 1.4])
    plat = c1.multiselect("Platform", ["IMDb", "Letterboxd"], default=["IMDb", "Letterboxd"])
    lda_opts = ["Semua"] + sorted(docs.lda_topic.unique().tolist())
    bt_opts = ["Semua"] + sorted(docs.bt_topic.unique().tolist())
    lda_t = c2.selectbox("Topik LDA", lda_opts, format_func=lambda t: t if t == "Semua" else f"LDA-{t} {tlabel('lda', t, summary, lang)}")
    bt_t = c3.selectbox("Topik BERTopic", bt_opts, format_func=lambda t: t if t == "Semua" else f"BT-{t} {tlabel('bertopic', t, summary, lang)}")
    c4, c5 = st.columns([1.2, 2])
    rr = c4.slider("Rating (1–10)", 1.0, 10.0, (1.0, 10.0), 0.5)
    q = c5.text_input("Kata kunci", placeholder="mis. ending, Nikki, jump scare")
    d = docs[docs.platform.isin(plat) & docs.rating_10.between(*rr)]
    if lda_t != "Semua":
        d = d[d.lda_topic == lda_t]
    if bt_t != "Semua":
        d = d[d.bt_topic == bt_t]
    if q:
        d = d[d.text_clean.str.contains(q, case=False, na=False, regex=False)]
    n_i, n_l = (d.platform == "IMDb").sum(), (d.platform == "Letterboxd").sum()
    st.markdown(f'<p class="small" style="margin-top:6px"><b style="color:{INK}">{len(d)}</b> ulasan · '
                f'<span class="dot" style="background:{C_IMDB}"></span>{n_i} IMDb · '
                f'<span class="dot" style="background:{C_LBX}"></span>{n_l} Letterboxd'
                + (f' · rata-rata rating {num(d.rating_10.mean(), 2, lang)}' if len(d) else "") + '</p>', unsafe_allow_html=True)
    with st.container(key="pill_view"):
        view = st.radio("Tampilan", ["Kartu", "Tabel"], horizontal=True, label_visibility="collapsed")
    if view == "Kartu":
        per = 10
        pages = max(1, int(np.ceil(len(d) / per)))
        pg = st.columns([1, 4])[0].number_input(f"Halaman (dari {pages})", 1, pages, 1) if pages > 1 else 1
        for _, r in d.iloc[(pg - 1) * per: pg * per].iterrows():
            txt = r.text_clean
            if q:
                txt_html = esc(txt)
                import re
                txt_html = re.sub(f"({re.escape(esc(q))})", r'<mark style="background:#f3dfa2;padding:0 2px">\1</mark>',
                                  txt_html, flags=re.I)
            else:
                txt_html = esc(txt)
            rating = num(r.rating_10, 1, lang)
            st.markdown(f'<div class="review"><div class="hd">{plat_pill(r.platform)}<span class="rt">★ {rating}/10</span>'
                        f'<span class="tg">LDA-{r.lda_topic} · BT-{r.bt_topic}</span></div>'
                        f'<div class="tx">{txt_html}</div></div>', unsafe_allow_html=True)
    else:
        st.dataframe(d[["platform", "rating_10", "lda_topic", "bt_topic", "text_clean"]], hide_index=True,
                     use_container_width=True, height=540,
                     column_config={"text_clean": st.column_config.TextColumn("ulasan", width="large")})
    st.download_button("Unduh hasil saringan (CSV)", d.to_csv(index=False).encode("utf-8"), "ulasan_terfilter.csv", "text/csv")

# ---------------------------------------------------------------------------
# 09 Jalankan Ulang
# ---------------------------------------------------------------------------
elif page == "Jalankan Ulang":
    header("09 · Jalankan Ulang", "Ganti data, jalankan lagi",
           "Unggah dua CSV hasil Apify untuk memakai data asli. File disimpan ke <code>data/raw/</code> lalu "
           "seluruh pipeline dijalankan ulang (sekitar 3–5 menit).")
    if summary:
        tape(summary)
    c1, c2 = st.columns(2)
    up_i = c1.file_uploader("CSV IMDb", type="csv", help="Kolom wajib: id, text, authorRating, submissionDate, ...")
    up_l = c2.file_uploader("CSV Letterboxd", type="csv", help="Kolom wajib: reviewtext, rating, reviewdate, reviewurl, ...")
    st.markdown(f'<p class="small">Mode di config.py saat ini: <code>USE_DUMMY = {config.USE_DUMMY}</code>. '
                f'Tanpa unggahan, tombol di bawah menjalankan ulang dengan data sesuai config.py.</p>', unsafe_allow_html=True)
    if st.button("Jalankan pipeline", type="primary"):
        args = []
        if up_i and up_l:
            config.REAL_IMDB_CSV.parent.mkdir(parents=True, exist_ok=True)
            config.REAL_IMDB_CSV.write_bytes(up_i.getvalue())
            config.REAL_LETTERBOXD_CSV.write_bytes(up_l.getvalue())
            args = ["--real"]
            st.success("CSV disimpan ke data/raw/. Run ini memakai data asli.")
        elif up_i or up_l:
            st.warning("Unggah kedua file (IMDb dan Letterboxd) untuk memakai data asli.")
            st.stop()
        with st.status("Menjalankan pipeline…", expanded=True) as stt:
            proc = subprocess.run([sys.executable, "run_pipeline.py", *args], cwd=config.BASE_DIR,
                                  capture_output=True, text=True)
            st.code(proc.stdout[-6000:] + proc.stderr[-3000:], language="text")
            stt.update(label="Selesai" if proc.returncode == 0 else "Gagal, lihat log di atas",
                       state="complete" if proc.returncode == 0 else "error")
        if proc.returncode == 0:
            st.cache_data.clear()
            st.success("Hasil baru tersimpan. Buka halaman lain untuk melihatnya.")
