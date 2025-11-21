# 📊 Forecasting & Data Viewer Apps (Streamlit)

Repositori ini berisi dua aplikasi berbasis Streamlit yang digunakan untuk melakukan analisis data harian dan forecasting. Aplikasi pertama, `ap.py`, merupakan aplikasi forecasting yang dirancang untuk memproses banyak file sekaligus, menggabungkan data berdasarkan kolom tanggal, dan memberikan fleksibilitas kepada pengguna dalam memilih kolom target yang akan dianalisis. Di dalamnya tersedia model peramalan Holt-Winters, XGBoost, dan LightGBM, serta fitur pendukung seperti perhitungan interval kepercayaan (CI 95%), deteksi anomali, Bollinger Bands, dan visualisasi agregasi harian hingga tahunan.

Aplikasi kedua, `cek.py`, digunakan untuk menampilkan data harian dalam bentuk grafik batang yang interaktif. Aplikasi ini dapat mengubah data harian menjadi data mingguan atau bulanan menggunakan fungsi agregasi seperti penjumlahan, rata-rata, nilai maksimum atau minimum. Pengguna dapat langsung melihat grafik yang interaktif dengan fitur zoom, hover, serta kemampuan menampilkan atau menyembunyikan seri tertentu melalui legend.

---

# ⚙️ Cara Install

Sebelum menjalankan aplikasi, pastikan Python sudah terpasang (disarankan versi 3.9–3.11).  
Selanjutnya, install seluruh dependency dengan menjalankan perintah berikut di terminal:

`pip install streamlit pandas numpy matplotlib statsmodels scikit-learn plotly xgboost lightgbm openpyxl`

---

# 🚀 Cara Menjalankan Aplikasi

Untuk menjalankan aplikasi forecasting, buka terminal pada folder proyek ini lalu jalankan perintah:

`streamlit run ap.py`

Sedangkan untuk menjalankan aplikasi viewer grafik batang, gunakan perintah:

`streamlit run cek.py`

Setelah perintah dijalankan, kedua aplikasi akan terbuka secara otomatis di browser melalui alamat `http://localhost:8501`. Jika tidak terbuka otomatis, alamat tersebut bisa diketik secara manual di browser.

---

# 🧠 Alur Singkat Penggunaan

Pada aplikasi `ap.py`, pengguna dapat mengunggah satu atau beberapa file CSV atau Excel yang berisi data harian. Aplikasi akan menggabungkan data berdasarkan tanggal, menampilkan ringkasan kolom, lalu pengguna memilih kolom tanggal dan kolom target numerik yang ingin dianalisis. Setelah itu, parameter forecasting dapat diatur melalui sidebar dan hasilnya akan ditampilkan dalam bentuk grafik train–test, forecast jangka panjang, CI 95%, deteksi anomali, tabel data harian, serta grafik batang agregasi.

Pada aplikasi `cek.py`, pengguna cukup mengunggah satu file CSV berisi data harian, memilih kolom tanggal dan kolom nilai, lalu memilih apakah data ingin dilihat dalam bentuk harian, mingguan, atau bulanan. Aplikasi akan menampilkan grafik batang interaktif, dengan nilai tertinggi diberi warna hijau dan nilai terendah diberi warna merah sehingga pola puncak dan lembah data mudah dilihat.

Repositori ini dapat digunakan untuk keperluan analisis data harian, baik dalam konteks akademik maupun operasional, tanpa perlu banyak mengubah kode program.
