# Dashboard Mini Kompetisi (MIKOM)

Dashboard Streamlit untuk analisis dan prediksi data Mini Kompetisi dengan machine learning.

## Fitur
- 📈 **Analisis Kategori** — Visualisasi distribusi kategori dan status per kategori
- 🎯 **Status SAGA** — Pie charts status MENANG/TERLEWAT/KALAH dan detil penyebab kalah
- 👤 **Per PIC** — Analisis status per perusahaan (PIC) dan top 5 penyebab kalah
- 🤖 **Prediksi Model** — Random Forest model untuk prediksi penawaran SAGA
- 📊 **Data Explorer** — Browse dan download data dengan statistik

## Setup & Jalankan

### 1. Pastikan file data ada
Letakkan file `Data Mini Kompetisi FIX.xlsx` di folder yang sama dengan `app.py`.
Jika ingin menyimpan hasil prediksi ke Google Sheets, letakkan juga `credentials.json` (service account key) di folder yang sama.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## Deploy ke Streamlit Cloud

1. Push folder ini ke GitHub
2. Buka https://share.streamlit.io
3. Pilih repo, branch, dan file (`app.py`)
4. Deploy!

Pastikan `Data Mini Kompetisi FIX.xlsx` juga di-push ke GitHub.

## Google Sheets (opsional)

Untuk menyimpan hasil prediksi ke Google Sheets aplikasi membutuhkan `credentials.json` (service account key) dan URL spreadsheet yang diatur di `app.py`.

Langkah singkat:

- Buat Google Cloud Service Account dan beri akses edit ke Google Sheets.
- Download key JSON dan simpan sebagai `credentials.json` di folder project.
- Pastikan spreadsheet yang dituju menerima akses service account email atau ubah sharing.

Jika tidak ingin menggunakan Google Sheets, hapus atau kosongkan `credentials.json` dan aplikasi tetap berjalan (akan menampilkan error write jika tidak dapat terhubung).

## Struktur File

```
📁 analisis data/
├── 📄 streamlit_app.py          # Main Streamlit app
├── 📄 requirements.txt           # Dependencies
├── 📄 README.md                  # Dokumentasi ini
└── 📄 Data Mini Kompetisi FIX.xlsx  # Data MIKOM
```

## Model Machine Learning

Model menggunakan **Random Forest Regressor** untuk prediksi:
- **Target:** Persentase penawaran SAGA terhadap Pagu
- **Features:** Pagu, Jumlah Penawar, Kategori, dan 5 fitur engineered
- **Metrics:** MAE, R², MAPE