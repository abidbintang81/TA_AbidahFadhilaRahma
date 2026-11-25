# Pengujian black-box dashboard

`blackbox_test.js` menjalankan 21 skenario uji (BB-01 s.d. BB-21, sama dengan Tabel III-5 proposal/buku)
secara otomatis di peramban Chromium lewat Playwright. Hasil terakhir: `hasil_blackbox.json`
(25 Nov 2026, data DUMMY: 21/21 sesuai).

Cara menjalankan ulang (setelah data asli dipakai):

1. `npm install playwright && npx playwright install chromium`
2. Terminal 1: `streamlit run app.py --server.port 8501 --server.headless true`
3. Salin folder `program` ke tempat lain, hapus `output/summary.json` di salinan itu, lalu jalankan
   `streamlit run app.py --server.port 8502 --server.headless true` dari salinan (untuk skenario BB-21).
4. `node pengujian/blackbox_test.js pengujian`

Angka yang diharapkan pada BB-14 s.d. BB-16 (1006/674/332 ulasan, 52 ulasan "ending") berasal dari data
dummy. Sesuaikan di skrip dengan jumlah ulasan pada data asli, lalu jalankan ulang; `hasil_blackbox.json`
tetap di folder ini (dibaca otomatis oleh build buku TA dan manuskrip), lalu bangun ulang buku dan manuskrip.
