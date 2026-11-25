# Program Topic Modeling Ulasan Film *Obsession* (2025): IMDb vs Letterboxd

Program ini menjalankan seluruh alur penelitian pada Bab III proposal TA
"Analisis Komparatif Ulasan Film Obsession pada IMDb dan Letterboxd menggunakan LDA dan BERTopic
untuk Mengidentifikasi Preferensi Pengguna", lalu menyediakan aplikasi web (Streamlit) untuk
menjelajahi hasilnya.

> **⚠️ PENTING: DATA SAAT INI ADALAH DATA DUMMY.**
> File `data/dummy/DUMMY_*.csv` berisi ulasan sintetis yang dibuat oleh `generate_dummy_data.py`
> dengan format kolom yang sama persis dengan hasil Apify. Semua angka di `output/` saat ini
> **bukan hasil penelitian**. Ganti dengan data asli sebelum dipakai di buku TA atau manuskrip.

---

## 1. Mengganti data dummy dengan data asli (satu baris)

1. Salin CSV hasil Apify ke folder `data/raw/` dengan nama:
   - `data/raw/imdb_obsession_reviews.csv` (kolom antara lain `id`, `text`, `authorRating`, `submissionDate`, ...)
   - `data/raw/letterboxd_obsession_reviews.csv` (kolom antara lain `reviewtext`, `rating`, `reviewdate`, ...)
2. Buka `config.py`, ubah baris:
   ```python
   USE_DUMMY = True    # ->  USE_DUMMY = False
   ```
3. Jalankan ulang: `python run_pipeline.py`

Jika nama kolom hasil Apify berbeda, sesuaikan `IMDB_COLS` / `LETTERBOXD_COLS` di `config.py`.

### Penanda data dummy
| Platform | Penanda |
|---|---|
| Semua | nama file diawali `DUMMY_`, URL mengarah ke `https://example.com/DUMMY/...` |
| IMDb | `id` diawali `DUMMY-rw`, `authorUserId` diawali `DUMMY-ur`, `summary` diawali `[DUMMY]` |
| Letterboxd | `username` diawali `dummy_user_`, `reviewer` diawali `DUMMY Reviewer` |
| Hasil | `output/summary.json` → `meta.is_dummy = true`; aplikasi menampilkan banner merah "DATA DUMMY" |

---

## 2. Instalasi

Butuh Python 3.10–3.13.

```bash
cd program
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Saat pertama dijalankan, program mengunduh resource NLTK (stopwords, wordnet, POS tagger) dan model
embedding `sentence-transformers/all-MiniLM-L6-v2` (sekitar 90 MB) dari Hugging Face. Pastikan ada
koneksi internet.

> **Catatan embedding.** Jika model all-MiniLM-L6-v2 gagal diunduh, program tetap berjalan
> dengan embedding cadangan TF-IDF+SVD dan menulis `meta.embedding_backend = "FALLBACK_..."` di
> `summary.json` serta peringatan di aplikasi. **Hasil dummy yang ada sekarang dibuat dengan embedding
> cadangan** karena server pembuatnya tidak bisa mengakses Hugging Face. Untuk hasil akhir TA,
> jalankan ulang di laptop sendiri dan pastikan `embedding_backend` = `sentence-transformers/all-MiniLM-L6-v2`.

---

## 3. Menjalankan

```bash
python generate_dummy_data.py   # (opsional) membuat ulang data dummy
python run_pipeline.py          # menjalankan seluruh analisis (sekitar 3–5 menit)
streamlit run app.py            # membuka aplikasi di http://localhost:8501
```

---

## 4. Alur program dan pemetaan ke Bab III

| Langkah | Bab III | File | Keterangan |
|---|---|---|---|
| Pemuatan data | III.2.2, III.3 | `src/preprocessing.py` | Menyeragamkan kolom IMDb & Letterboxd, label platform disimpan |
| Normalisasi rating | III.3 | `src/preprocessing.py` | Rating Letterboxd (0,5–5) × 2 → skala 1–10 |
| Seleksi data | III.2.2–III.2.3 | `src/preprocessing.py` | Hapus ulasan kosong, duplikat, non-Inggris (`langdetect`), terlalu pendek (< 3 token) |
| Preprocessing LDA | III.2.3 | `src/preprocessing.py` | Cleaning HTML/entitas/URL/angka/tanda baca → case folding → tokenisasi → stopword (umum + domain) → lemmatisasi WordNet dengan POS |
| Preprocessing BERTopic | III.2.3 | `src/preprocessing.py` | Hanya cleaning ringan (HTML, entitas, URL), kalimat utuh dipertahankan |
| LDA | III.2.4 | `src/lda_model.py` | Gensim; dictionary + DTM; K = 2–15, tiap K dilatih 3 seed, C_v dirata-rata |
| BERTopic | III.2.4 | `src/bertopic_model.py` | all-MiniLM-L6-v2 → UMAP (`random_state=42`) → HDBSCAN → c-TF-IDF; `reduce_outliers` untuk topik −1 |
| Korpus gabungan | III.2.4 | `run_pipeline.py` | Kedua model dilatih pada gabungan IMDb + Letterboxd |
| Evaluasi | III.2.5, III.4 | `src/evaluation.py` | C_v (Gensim) & topic diversity dengan top-10 kata yang sama untuk kedua model |
| Pelabelan topik | III.2.5 | `src/evaluation.py`, `labels/` | Label usulan otomatis (leksikon aspek) → **wajib** ditinjau dan diganti label manual |
| Pencocokan topik | III.2.5 | `src/evaluation.py` | Jaccard top-10 kata, pasangan satu-satu (algoritma Hungaria), ambang 0,15 (minimal 3 dari 10 kata teratas sama) |
| Distribusi per platform | III.2.6 | `src/evaluation.py` | LDA: topik dominan + rata-rata proporsi; BERTopic: label topik langsung |
| Uji chi-square | III.2.6, III.4 | `src/evaluation.py` | Tabel kontingensi platform × topik, α = 0,05, Cramér's V, residual terstandar |
| Preferensi pengguna | II.1.3, III.2.6 | `src/evaluation.py` | Rata-rata/median rating per topik per platform, Kruskal–Wallis, Mann–Whitney per topik |

### Pemilihan jumlah topik LDA
`config.K_SELECTION` menentukan aturan:
- `"first_peak"` (bawaan): K pada puncak pertama kurva C_v, yaitu titik sebelum kurva mendatar. Ini
  menerjemahkan kalimat proposal "nilai dengan skor coherence C_v tertinggi yang topiknya tetap dapat
  diinterpretasikan": K yang lebih besar biasanya hanya memecah topik menjadi subtopik kecil.
- `"max_cv"`: K dengan C_v tertinggi di seluruh rentang.
- `"cv_x_diversity"`: K dengan C_v × topic diversity tertinggi (topic quality, Dieng dkk., 2020).
- `LDA_K_OVERRIDE = 6` (misalnya) memaksa K tertentu setelah diskusi dengan pembimbing.

`summary.json` mencatat K terpilih (`lda.selected_k`) dan K dengan C_v maksimum (`lda.k_max_cv`).

### Label topik manual
Setelah menjalankan pipeline, folder `labels/` berisi `lda_labels_TEMPLATE.csv` dan
`bertopic_labels_TEMPLATE.csv` berisi kata teratas dan label usulan. Untuk memakai label sendiri:
1. Salin menjadi `labels/lda_labels.csv` dan `labels/bertopic_labels.csv`.
2. Isi kolom `label_id` (Indonesia) dan `label_en` (Inggris) sesuai hasil diskusi dengan pembimbing.
3. Jalankan ulang `python run_pipeline.py`. Kolom `label_source` akan berubah menjadi "manual".

Catatan: nomor topik bisa berubah jika data atau parameter berubah, jadi buat label setelah run final.

---

## 5. Isi folder `output/`

| File | Isi |
|---|---|
| `summary.json` | Ringkasan mesin-baca seluruh hasil (dipakai buku TA & manuskrip) |
| `hasil_semua_tabel.xlsx` | Semua tabel CSV dalam satu workbook (satu sheet per tabel) |
| `documents_with_topics.csv` | Setiap ulasan beserta platform, rating, topik LDA, topik BERTopic |
| `tables/01_...csv` – `23_...csv` | Tabel per tahap (lihat daftar di bawah) |
| `figures/id/*.png`, `figures/en/*.png` | 14 gambar (200 dpi) dalam bahasa Indonesia dan Inggris |
| `models/` | Model LDA (Gensim) dan BERTopic tersimpan |

Tabel: 01 tahapan seleksi data · 02 contoh preprocessing · 03 statistik panjang ulasan · 04 statistik
rating · 05 C_v per K · 06 evaluasi model · 07/08 kata teratas & label topik LDA/BERTopic · 09 contoh
ulasan per topik · 10 matriks Jaccard · 11 pencocokan topik · 12/13 distribusi topik per platform ·
14 rata-rata proporsi LDA · 15/16 residual chi-square · 17 uji chi-square · 18/19 rating per topik ·
20/21 Mann–Whitney rating antarplatform per topik · 22/23 tabel preferensi pengguna.

Gambar: fig01 seleksi data · fig02 distribusi rating · fig03 panjang ulasan · fig04 kurva C_v LDA ·
fig05/06 kata teratas LDA/BERTopic · fig07/08 distribusi topik per platform · fig09 rata-rata proporsi
LDA · fig10 perbandingan model · fig11 heatmap Jaccard · fig12/13 rating per topik · fig14 peta dokumen
BERTopic.

### Struktur `summary.json`
```
meta                    is_dummy, embedding_backend, parameter, waktu run
data_selection_stages   jumlah per tahap (IMDb, Letterboxd, Total)
review_length_words     statistik panjang ulasan per platform
rating_10_stats         statistik rating 1–10 per platform
lda                     coherence_by_k, selected_k, k_max_cv, coherence_cv, topic_diversity,
                        topics[ topic, label_id, label_en, top_words, weights, coherence_cv,
                                n_/pct_/mean_rating_/median_rating_ per platform ],
                        mean_topic_proportion, chi_square, kruskal_rating_by_topic
bertopic                n_topics, outliers_before_reduce, outliers_after_reduce, coherence_cv,
                        topic_diversity, topics[...], chi_square, kruskal_rating_by_topic
alignment               jaccard_matrix, pairs, only_lda, only_bertopic, threshold
preference              lda[], bertopic[]  (peringkat topik per platform + rating)
```

---

## 6. Aplikasi Streamlit "Catatan Ulasan"

`streamlit run app.py` membuka aplikasi di http://localhost:8501. Tampilannya bergaya jurnal film:
sidebar gelap dengan perforasi pita film, judul berhuruf serif (Fraunces), angka ber-font mono
(IBM Plex Mono), dan warna data yang sama dengan gambar di buku TA (IMDb biru, Letterboxd oranye,
LDA hijau, BERTopic ungu). Tema ada di `.streamlit/config.toml`, gaya ada di blok CSS di awal `app.py`.

Halaman:
01 Ringkasan (angka kunci + temuan utama yang dihitung otomatis) · 02 Data & Preprocessing ·
03 Topik LDA (kurva C_v, kartu topik) · 04 Topik BERTopic (kartu topik, peta dokumen interaktif) ·
05 Evaluasi & Pencocokan (matriks Jaccard, pasangan topik) · 06 Perbandingan Platform (chi-square,
residual) · 07 Rating & Preferensi (peta preferensi, Kruskal–Wallis) · 08 Penjelajah Ulasan (filter
platform/topik/rating/kata kunci, sorot kata, unduh CSV) · 09 Jalankan Ulang (unggah CSV Apify asli
lalu jalankan pipeline dari aplikasi).

Label topik bisa ditampilkan dalam Bahasa Indonesia atau English (pengalih di sidebar). Banner
"DATA DUMMY" muncul di setiap halaman selama `meta.is_dummy = true`. Tangkapan layar ada di
`output/app_screenshots/`. Font dimuat dari Google Fonts; tanpa internet aplikasi tetap jalan dengan
font cadangan.

### Pengujian black-box
Folder `pengujian/` berisi skrip Playwright `blackbox_test.js` (21 skenario, BB-01 s.d. BB-21) dan hasil terakhirnya
`hasil_blackbox.json` (25 Nov 2026, data dummy: 21/21 sesuai). Hasil ini dipakai di Bab V.6 buku TA dan di manuskrip.
Lihat `pengujian/README.md` untuk cara menjalankan ulang.

## 7. Reproduksibilitas
- Seed tetap: `RANDOM_STATE = 42` (LDA, UMAP), `LANGDETECT_SEED = 0`.
- HDBSCAN dan UMAP dapat sedikit berbeda antar-sistem operasi/versi pustaka. Catat versi pustaka
  (`pip freeze > versi_pustaka.txt`) saat run final untuk dilampirkan di buku TA.
