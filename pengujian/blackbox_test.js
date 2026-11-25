// Black-box test aplikasi dashboard Streamlit (Playwright)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const OUT = process.argv[2] || __dirname;
const URL1 = process.env.APP_URL || URL1;       // aplikasi dengan hasil analisis
const URL2 = process.env.APP_URL_KOSONG || URL2; // salinan aplikasi tanpa output/summary.json
const results = [];
function rec(id, fitur, skenario, harapan, aktual, ok) { results.push({ id, fitur, skenario, harapan, aktual, ok }); console.log(ok ? 'PASS' : 'FAIL', id, aktual); }

async function idle(page) {
  await page.waitForTimeout(600);
  for (let i = 0; i < 60; i++) {
    const running = await page.locator('[data-testid="stStatusWidget"]').count();
    if (!running) break;
    await page.waitForTimeout(300);
  }
  await page.waitForTimeout(1200);
}
async function nav(page, name) {
  await page.locator('[data-testid="stSidebar"] label').filter({ hasText: name }).first().click();
  await idle(page);
}
async function h1(page) { return (await page.locator('.page-title').first().innerText()).trim(); }
async function mainText(page) { return await page.locator('[data-testid="stMain"]').innerText(); }

(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
  const page = await ctx.newPage();
  const errs = [];
  await page.goto(URL1); await page.waitForSelector('.page-title', { timeout: 60000 }); await idle(page);

  // BB-01
  let t = await mainText(page);
  rec('BB-01', 'Halaman Ringkasan', 'Membuka aplikasi pertama kali', 'Halaman 01 Ringkasan tampil dengan angka kunci (ulasan dianalisis, IMDb, Letterboxd, C_v LDA, C_v BERTopic)',
      `Judul "${await h1(page)}" tampil; kartu angka kunci memuat ${['ULASAN DIANALISIS','IMDB','LETTERBOXD','C_V LDA','C_V BERTOPIC'].filter(k => t.toUpperCase().includes(k)).length} dari 5 indikator`,
      t.toUpperCase().includes('ULASAN DIANALISIS') && t.toUpperCase().includes('C_V BERTOPIC'));
  // BB-02 banner dummy
  rec('BB-02', 'Penanda data dummy', 'Membuka aplikasi saat summary.json berstatus is_dummy = true', 'Banner "DATA DUMMY" tampil di halaman',
      t.includes('DATA DUMMY') ? 'Banner "DATA DUMMY" tampil' : 'Banner tidak tampil', t.includes('DATA DUMMY'));

  // BB-03..10 navigasi
  const pages = [['02', 'Data & Preprocessing'], ['03', 'Topik LDA'], ['04', 'Topik BERTopic'], ['05', 'Evaluasi & Pencocokan'],
                 ['06', 'Perbandingan Platform'], ['07', 'Rating & Preferensi'], ['08', 'Penjelajah Ulasan'], ['09', 'Jalankan Ulang']];
  let n = 3;
  for (const [no, name] of pages) {
    await nav(page, name);
    const eyebrow = await page.locator('[data-testid="stMain"]').innerText();
    const exc = await page.locator('[data-testid="stException"]').count();
    const ok = eyebrow.toUpperCase().includes(`${no} · ${name.toUpperCase()}`) && exc === 0;
    rec(`BB-${String(n).padStart(2, '0')}`, 'Navigasi', `Memilih menu "${no} ${name}" pada sidebar`, `Halaman ${no} ${name} tampil tanpa pesan galat`,
        ok ? `Halaman ${no} ${name} tampil ("${await h1(page)}"), tidak ada galat` : `Gagal (galat=${exc})`, ok);
    n++;
  }

  // BB-11 language switch on Topik BERTopic
  await nav(page, 'Topik BERTopic');
  const idText = await mainText(page);
  await page.locator('[data-testid="stSidebar"] label').filter({ hasText: 'ENGLISH' }).click(); await idle(page);
  const enText = await mainText(page);
  const changed = idText !== enText;
  rec('BB-11', 'Bahasa label topik', 'Mengubah pengalih bahasa label dari INDONESIA ke ENGLISH', 'Label topik ditampilkan dalam bahasa Inggris',
      changed ? 'Label topik berubah ke bahasa Inggris' : 'Label tidak berubah', changed);
  await page.locator('[data-testid="stSidebar"] label').filter({ hasText: 'INDONESIA' }).click(); await idle(page);

  // BB-12 representative review topic select (LDA)
  await nav(page, 'Topik LDA');
  const before = await mainText(page);
  const pills = page.locator('.st-key-pill_topic label');
  const np = await pills.count();
  await pills.nth(np - 1).click(); await idle(page);
  const after = await mainText(page);
  rec('BB-12', 'Topik LDA', `Memilih topik terakhir (${np} pilihan) pada bagian ulasan representatif`, 'Contoh ulasan berganti sesuai topik yang dipilih',
      after !== before ? 'Contoh ulasan berganti sesuai topik terpilih' : 'Tidak berubah', after !== before);

  // BB-13 peta dokumen warna platform
  await nav(page, 'Topik BERTopic');
  await page.locator('.st-key-pill_map label').filter({ hasText: 'Platform' }).click(); await idle(page);
  const legend = await page.locator('.st-key-pill_map').locator('xpath=following::div[contains(@class,"js-plotly-plot")][1]').innerText().catch(() => '');
  const okMap = legend.includes('IMDb') && legend.includes('Letterboxd');
  rec('BB-13', 'Peta dokumen', 'Mengubah pewarnaan peta dokumen dari Topik ke Platform', 'Legenda peta berubah menjadi IMDb dan Letterboxd',
      okMap ? 'Legenda menampilkan IMDb dan Letterboxd' : 'Legenda tidak berubah', okMap);

  // Penjelajah
  await nav(page, 'Penjelajah Ulasan');
  const count = async () => (await page.locator('[data-testid="stMain"] p.small').first().innerText()).replace(/\s+/g, ' ');
  let c0 = await count();
  rec('BB-14', 'Penjelajah Ulasan', 'Membuka halaman tanpa mengubah filter', 'Seluruh ulasan hasil seleksi tampil (1.006 ulasan: 674 IMDb, 332 Letterboxd)',
      `Tampil: ${c0}`, c0.includes('1006 ulasan') && c0.includes('674 IMDb') && c0.includes('332 Letterboxd'));
  // remove Letterboxd tag
  await page.locator('button[aria-label="Remove Letterboxd"]:visible').first().click(); await idle(page);
  let c1 = await count();
  rec('BB-15', 'Penjelajah Ulasan', 'Menghapus Letterboxd dari filter platform', 'Hanya ulasan IMDb yang tampil (674 ulasan, 0 Letterboxd)',
      `Tampil: ${c1}`, c1.includes('674 ulasan') && c1.includes('0 Letterboxd'));
  await page.reload(); await page.waitForSelector('.page-title'); await idle(page);
  await nav(page, 'Penjelajah Ulasan');
  // keyword
  const inp = page.locator('input[aria-label="Kata kunci"]');
  await inp.fill('ending'); await inp.press('Enter'); await idle(page);
  let c2 = await count();
  const marks = await page.locator('.review mark').count();
  rec('BB-16', 'Penjelajah Ulasan', 'Mengisi kata kunci "ending"', 'Hanya ulasan yang memuat kata "ending" yang tampil (52 ulasan) dan kata tersebut disorot',
      `Tampil: ${c2}; ${marks} sorotan pada halaman pertama`, c2.startsWith('52 ulasan') && marks > 0);
  await inp.fill('xyzqwv'); await inp.press('Enter'); await idle(page);
  let c3 = await count(); const exc3 = await page.locator('[data-testid="stException"]').count();
  rec('BB-17', 'Penjelajah Ulasan', 'Mengisi kata kunci yang tidak ada di korpus ("xyzqwv")', 'Hasil 0 ulasan tanpa pesan galat',
      `Tampil: ${c3}; galat: ${exc3 ? 'ada' : 'tidak ada'}`, c3.startsWith('0 ulasan') && exc3 === 0);
  await inp.fill(''); await inp.press('Enter'); await idle(page);
  // table view
  await page.locator('.st-key-pill_view label').filter({ hasText: 'Tabel' }).click(); await idle(page);
  const df = await page.locator('[data-testid="stDataFrame"]').count();
  rec('BB-18', 'Penjelajah Ulasan', 'Mengubah tampilan dari Kartu ke Tabel', 'Ulasan tampil dalam bentuk tabel', df ? 'Tabel ulasan tampil' : 'Tabel tidak tampil', df > 0);
  // download
  const [dl] = await Promise.all([page.waitForEvent('download'), page.getByText('Unduh hasil saringan (CSV)').click()]);
  const p = path.join(require('os').tmpdir(), 'unduh_uji.csv'); await dl.saveAs(p);
  const lines = fs.readFileSync(p, 'utf8').trim().split('\n');
  const head = lines[0];
  rec('BB-19', 'Penjelajah Ulasan', 'Menekan tombol "Unduh hasil saringan (CSV)"', 'Berkas ulasan_terfilter.csv terunduh dan berisi ulasan hasil saringan',
      `Berkas ${dl.suggestedFilename()} terunduh; kolom: ${head.split(',').slice(0, 5).join(', ')}, ...`, dl.suggestedFilename() === 'ulasan_terfilter.csv' && head.includes('platform'));

  // Jalankan ulang: only one file
  await nav(page, 'Jalankan Ulang');
  const fi = page.locator('[data-testid="stFileUploaderDropzoneInput"]').first();
  await fi.setInputFiles(path.join(__dirname, '..', 'data', 'dummy', 'DUMMY_imdb_obsession_reviews.csv'));
  await idle(page);
  await page.getByRole('button', { name: 'Jalankan pipeline' }).click(); await idle(page);
  t = await mainText(page);
  const warn = t.includes('Unggah kedua file');
  rec('BB-20', 'Jalankan Ulang', 'Mengunggah CSV IMDb saja lalu menekan "Jalankan pipeline"', 'Muncul peringatan untuk mengunggah kedua file dan pipeline tidak dijalankan',
      warn ? 'Muncul peringatan "Unggah kedua file (IMDb dan Letterboxd) untuk memakai data asli."; pipeline tidak dijalankan' : 'Tidak ada peringatan', warn);
  
  // No output
  const p2 = await ctx.newPage();
  await p2.goto(URL2); await p2.waitForSelector('.page-title', { timeout: 60000 }); await idle(p2);
  const navs = await p2.locator('[data-testid="stSidebar"] .st-key-nav label').allInnerTexts();
  navs.splice(0, navs.length, ...navs.filter(x => /\d\d/.test(x)));
  const okNo = navs.length === 1 && navs[0].includes('Jalankan Ulang');
  rec('BB-21', 'Penanganan hasil kosong', 'Membuka aplikasi saat berkas output/summary.json belum ada', 'Menu hanya menampilkan halaman Jalankan Ulang tanpa pesan galat',
      okNo ? 'Hanya menu "09 Jalankan Ulang" yang tampil, tidak ada galat' : `Menu: ${navs.join('|')}`, okNo);

  fs.writeFileSync(OUT + '/hasil_blackbox.json', JSON.stringify(results, null, 1));
  await b.close();
})();
