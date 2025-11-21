import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go   # ⬅️ tambahan

st.set_page_config(
    page_title="Viewer Grafik Batang Dinamis",
    layout="wide"
)

st.title("📊 Viewer Grafik Batang Dinamis (Harian → Mingguan/Bulanan)")

st.write(
    "Upload file **CSV** harian kamu, pilih kolom tanggal & nilai, lalu lihat grafik batang "
    "yang bisa diubah agregasinya ke **harian / mingguan / bulanan**."
)

# =========================
# FUNGSI UTIL
# =========================
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df

def get_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

# =========================
# SIDEBAR: UPLOAD & PENGATURAN
# =========================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    uploaded = st.file_uploader("Upload file CSV harian", type=["csv"])

    if uploaded is None:
        st.info("Silakan upload file `.csv` terlebih dahulu.")
    else:
        st.success("File berhasil di-upload ✅")

# =========================
# LOGIKA UTAMA
# =========================
if uploaded is not None:
    # Baca CSV
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()

    # Simpan versi asli untuk ditampilkan, versi normalize untuk pemrosesan
    df = normalize_cols(df_raw)

    st.subheader("🧾 Data Asli (5 baris pertama)")
    st.dataframe(df_raw.head())

    st.markdown("**Kolom setelah dinormalisasi (dipakai untuk pemrosesan):**")
    st.write(list(df.columns))

    # Pilih kolom tanggal
    st.subheader("1️⃣ Pilih Kolom Tanggal")
    date_col = st.selectbox(
        "Pilih kolom yang berisi tanggal",
        options=df.columns,
        index=0
    )

    # Konversi ke datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if df[date_col].isna().all():
        st.error("Semua nilai di kolom tanggal gagal dikonversi ke datetime. Coba pilih kolom lain atau cek format tanggal.")
        st.stop()

    # Drop baris yang gagal parse tanggal
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

    # Pilih kolom nilai (bisa banyak)
    st.subheader("2️⃣ Pilih Kolom Nilai (Batang)")
    numeric_cols = get_numeric_cols(df)

    if not numeric_cols:
        st.error("Tidak ada kolom numerik yang ditemukan. Pastikan ada kolom angka (int/float).")
        st.stop()

    value_cols = st.multiselect(
        "Pilih satu atau lebih kolom numerik untuk ditampilkan sebagai batang",
        options=numeric_cols,
        default=[numeric_cols[0]] if numeric_cols else []
    )

    if not value_cols:
        st.warning("Pilih minimal satu kolom nilai.")
        st.stop()

    # Pilih frekuensi agregasi
    st.subheader("3️⃣ Pilih Frekuensi Agregasi")
    freq_label = st.radio(
        "Agregasi data menjadi:",
        options=["Harian (tanpa gabung)", "Mingguan", "Bulanan"],
        index=0,
        horizontal=True
    )

    freq_map = {
        "Harian (tanpa gabung)": None,
        "Mingguan": "W",
        "Bulanan": "M"
    }
    freq = freq_map[freq_label]

    # Pilih fungsi agregasi
    agg_func_label = st.selectbox(
        "Fungsi agregasi jika data digabung (mingguan/bulanan):",
        options=["sum", "mean", "max", "min"],
        index=0
    )

    # Filter tanggal
    st.subheader("4️⃣ Filter Rentang Tanggal (Opsional)")
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()

    date_range = st.date_input(
        "Pilih rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
        df_filtered = df.loc[mask].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        st.warning("Tidak ada data dalam rentang tanggal yang dipilih.")
        st.stop()

    # =========================
    # RESAMPLE / AGREGASI
    # =========================
    df_agg = df_filtered.set_index(date_col)

    if freq is not None:
        # Agregasi
        df_agg = getattr(df_agg[value_cols].resample(freq), agg_func_label)()
        df_agg = df_agg.reset_index()
    else:
        # Harian: tetap, tapi hanya ambil kolom yang dipilih
        df_agg = df_agg.reset_index()[[date_col] + value_cols]

    # Label frekuensi untuk judul
    freq_title = {
        None: "Harian",
        "W": "Mingguan",
        "M": "Bulanan"
    }[freq]

    st.subheader(f"5️⃣ Data Setelah Agregasi ({freq_title})")
    st.dataframe(df_agg.head())

    # =========================
    # GRAFIK BATANG INTERAKTIF
    # =========================
    st.subheader("6️⃣ Grafik Batang Interaktif")

    # Palet dasar kalau kolomnya lebih dari 1
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig = go.Figure()

    if len(value_cols) == 1:
        # Satu seri saja
        series = value_cols[0]
        data = df_agg[[date_col, series]].rename(columns={series: "Value"})
        max_val = data["Value"].max()
        min_val = data["Value"].min()

        colors = [
            "green" if v == max_val else
            "red" if v == min_val else
            base_colors[0]
            for v in data["Value"]
        ]

        fig.add_bar(
            x=data[date_col],
            y=data["Value"],
            marker_color=colors,
            name=series
        )
    else:
        # Beberapa seri → satu trace per seri
        for idx, series in enumerate(value_cols):
            data = df_agg[[date_col, series]].rename(columns={series: "Value"})
            max_val = data["Value"].max()
            min_val = data["Value"].min()
            base_color = base_colors[idx % len(base_colors)]

            colors = [
                "green" if v == max_val else
                "red" if v == min_val else
                base_color
                for v in data["Value"]
            ]

            fig.add_bar(
                x=data[date_col],
                y=data["Value"],
                marker_color=colors,
                name=series
            )

    fig.update_layout(
        title=f"Grafik Batang {freq_title} per Tanggal",
        xaxis_title="Tanggal",
        yaxis_title="Nilai",
        hovermode="x unified",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💡 Tips: Ubah pilihan **frekuensi**, **fungsi agregasi**, atau **kolom nilai** "
        "di atas untuk mengeksplor data kamu dengan cara yang berbeda. "
        "Batang hijau = nilai tertinggi, batang merah = nilai terendah."
    )
