# ✈️ Analisis & Peramalan Penumpang Bandara Juanda (2019–2024)

Proyek ini menggunakan data **Angkasa Pura** (Bandara Juanda, Surabaya) untuk menganalisis dan meramalkan jumlah **pesawat, penumpang, bagasi, dan kargo** pada segmen **Domestik** dan **Internasional**. Pendekatan yang digunakan memadukan **analisis statistik** (korelasi, OLS, VAR, Granger) dan **machine learning** (LSTM) sehingga cakupan tidak hanya deskriptif, tetapi juga **prediktif**.

  
> **Analisis dan Peramalan Jumlah Penumpang Domestik–Internasional di Bandara Juanda (2019–2024) dengan Model Statistik dan Machine Learning**

---

## 📌 Ikhtisar
- **Periode**: 2019–2024  
- **Fokus**: tren historis, seasonality, dampak pandemi, hubungan antar-metrik, serta **forecasting** penumpang  
- **Variabel utama**: pesawat, penumpang, bagasi (kg), kargo (kg) — **Domestik vs Internasional**  
- **Variabel konteks**: hari libur/cuti nasional, **IHK**, **Ekspor**, **Impor**

---

## 🧾 Data
- `bandara.csv` — data harian: `actualdate`, `domestik / inter`, `pesawat`, `total pax`, `transit`, `bagasi (kg)`, `kargo (kg)`
- `libur.csv` — tanggal dan `event` (libur/cuti nasional)
- `ihk.csv` — `bulan`, `IHK` (Indeks Harga Konsumen)
- `impor_ekspor.csv` — `Tanggal`, `Ekspor`, `Impor`

> Notebook akan melakukan *cleaning*, pivot Domestik/Internasional → `df_rekap`, agregasi **harian → mingguan → bulanan → tahunan**, serta pembuatan fitur (hari, bulan, rasio).

---

## 🔎 Metodologi Singkat
1. **Data Understanding & Cleaning**: standarisasi kolom, pivot, *missing handling*.  
2. **EDA**: tren (line/bar), distribusi (histogram/boxplot), seasonality (awal/akhir tahun, Lebaran, weekend effect).  
3. **Statistik**: Pearson/Spearman, Shapiro, OLS, **VAR**, **Granger** (Penumpang↔IHK, Kargo↔Ekspor/Impor).  
4. **Forecasting**: baseline **ARIMA/VAR** dan **LSTM** (deret waktu) + evaluasi **MAE/RMSE**.

---

## ✅ Hasil Utama (ringkas)
- **Domestik pulih lebih cepat**; internasional **lebih fluktuatif** dan sensitif kebijakan perbatasan.  
- **Kargo internasional melonjak** saat pandemi (cargo-only flights).  
- **Bagasi internasional ≈ 2×** domestik per penumpang.  
- **Weekend effect** jelas (Sabtu–Minggu lebih ramai untuk pesawat & bagasi).  
- **Tidak ditemukan** bukti **Granger causality** jangka pendek IHK ↔ jumlah penumpang.  
- Kerangka **LSTM** siap untuk peramalan jangka menengah.

---

**Jika Anda Ingin file nya, anda bisa menghubungi lewat LinkedIn puguhsw1244**

