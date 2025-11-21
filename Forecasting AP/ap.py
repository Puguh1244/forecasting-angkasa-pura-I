import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io, os, re
import plotly.graph_objects as go

st.set_page_config(page_title="Forecast Dinamis (Multi-file)", layout="wide")

# =========================
# Utils
# =========================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

def read_any_file(file) -> pd.DataFrame:
    ext = os.path.splitext(getattr(file, "name", "uploaded"))[1].lower()
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(file)
        return _normalize_cols(df)
    try:
        df = pd.read_csv(file, sep=None, engine="python")
        return _normalize_cols(df)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, sep=";")
        return _normalize_cols(df)

def union_columns(dfs):
    cols = set()
    for d in dfs:
        cols |= set(d.columns)
    return sorted(cols)

def guess_date_candidates(dfs):
    union_cols = union_columns(dfs)
    name_hint = re.compile(r"(date|tanggal|tgl|time|waktu|datetime|period|bulan|month|yr|year)", re.I)
    candidates = []
    for c in union_cols:
        score = 0
        if name_hint.search(c): score += 1
        sample_vals = pd.concat([d[c] for d in dfs if c in d.columns]).head(500)
        try:
            ok_ratio = pd.to_datetime(sample_vals, errors="coerce").notna().mean()
        except Exception:
            ok_ratio = 0.0
        score += ok_ratio
        candidates.append((c, score))
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return [c for c, _ in candidates] if candidates else union_cols

def combine_by_date(dfs, date_col, target_col):
    kept = []
    for df in dfs:
        if date_col not in df.columns:
            continue
        cols = [c for c in [date_col, target_col] if c in df.columns]
        if not cols:
            continue
        tmp = df[cols].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        if target_col in tmp.columns:
            tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        kept.append(tmp)
    if not kept:
        return pd.DataFrame(columns=[date_col, target_col])
    all_df = pd.concat(kept, ignore_index=True)
    all_df = all_df.dropna(subset=[date_col]).sort_values(date_col)
    all_df = all_df.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    return all_df

def create_time_features(df, date_col):
    df['month'] = df[date_col].dt.month
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    df['dayofyear'] = df[date_col].dt.dayofyear
    return df

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    eps = 1e-8
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (denom + eps))

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Pengaturan Forecast")
model_choice = st.sidebar.selectbox(
    "Pilih Model Forecast:",
    ["Holt-Winters", "XGBoost", "LightGBM"]
)
ci_agg_choice = st.sidebar.selectbox(
    "Tampilan grafik batang CI95 (agregasi):",
    ["Per Hari", "Per Minggu", "Per Bulan", "Per Kuartal", "Per Tahun"]
)

# Param per-model
if model_choice == "Holt-Winters":
    st.sidebar.subheader("📈 Parameter Holt-Winters")
    trend_type = st.sidebar.selectbox("Tipe Trend:", ["Additive", "Multiplicative"])
    season_type = st.sidebar.selectbox("Tipe Musiman:", ["Additive", "Multiplicative"])
    damped = st.sidebar.selectbox("Gunakan Damped Trend?", ["Ya", "Tidak"])
    seasonal_input = st.sidebar.text_input("Seasonal Periods (misal 12, 52, 365):", value="365")
    try:
        seasonal_periods = int(seasonal_input)
    except ValueError:
        st.sidebar.warning("Seasonal Periods tidak valid. Dipakai 365.")
        seasonal_periods = 365
else:
    st.sidebar.subheader(f"🌿 Parameter {model_choice}")
    n_estimators = st.sidebar.slider("n_estimators", 100, 1000, 500, 50)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)
    max_depth = st.sidebar.slider("max_depth", 2, 12, 6)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8)
    if model_choice == "LightGBM":
        num_leaves = st.sidebar.slider("num_leaves", 8, 128, 31)

n_years = st.sidebar.slider("Berapa tahun forecast?", 1, 10, 5)
use_bbands = st.sidebar.checkbox("Tampilkan Bollinger Bands?", value=False)

# Anomali
st.sidebar.subheader("🔎 Deteksi Anomali (MA)")
enable_anomaly = st.sidebar.checkbox("Aktifkan deteksi anomali", value=True)
baseline_ma = st.sidebar.selectbox("Baseline MA", ["MA 7", "MA 30", "MA 90"], index=1)
anomaly_method = st.sidebar.selectbox(
    "Metode anomali",
    ["Z-Score Residual", "IQR Residual", "Top-N Residual"]
)
if anomaly_method == "Z-Score Residual":
    z_thresh = st.sidebar.slider("Batas |z|", 1.0, 5.0, 3.0, 0.1)
elif anomaly_method == "IQR Residual":
    iqr_k = st.sidebar.slider("Kelipatan IQR", 0.5, 3.0, 1.5, 0.1)
else:
    top_n = st.sidebar.slider("Top-N anomali", 5, 300, 25, 1)

# =========================
# Main
# =========================
st.title("📊 Forecast Dinamis (Multi-file, Kolom Bebas)")
st.write(
    "Upload **satu atau banyak file** (CSV/XLS/XLSX). "
    "Setelah itu pilih **kolom tanggal** & **kolom target** dari daftar kolom yang terdeteksi."
)

uploaded_files = st.file_uploader(
    "📂 Upload data",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("⬆️ Upload file dulu. Dropdown pilihan kolom akan muncul setelah file diunggah.")
    st.stop()

# Baca semua file
try:
    raw_dfs = [read_any_file(f) for f in uploaded_files]
except Exception as e:
    st.error(f"❌ Gagal membaca file: {e}")
    st.stop()

with st.expander("ℹ️ Ringkasan Kolom per File"):
    info = []
    for f, d in zip(uploaded_files, raw_dfs):
        info.append({
            "file": getattr(f, "name", "uploaded"),
            "baris": len(d),
            "contoh_10_kolom": ", ".join(list(d.columns)[:10])
            + ("..." if len(d.columns) > 10 else "")
        })
    st.dataframe(pd.DataFrame(info))

# === Dropdown kolom (baru muncul setelah upload) ===
union_cols = union_columns(raw_dfs)
date_candidates = guess_date_candidates(raw_dfs)
date_col = st.selectbox(
    "📅 Pilih kolom tanggal",
    date_candidates if date_candidates else union_cols
)

candidate_targets = [c for c in union_cols if c != date_col]
numeric_like = []
for c in candidate_targets:
    sample = pd.concat([d[c] for d in raw_dfs if c in d.columns]).head(500)
    if pd.api.types.is_numeric_dtype(sample):
        numeric_like.append(c)
    else:
        try:
            pd.to_numeric(sample, errors="coerce")
            numeric_like.append(c)
        except Exception:
            pass

options_target = numeric_like if numeric_like else candidate_targets
if not options_target:
    st.error("Tidak ada kolom numerik untuk target. Pilih kolom tanggal lain atau cek data Anda.")
    st.stop()

prefer = [
    c for c in options_target
    if re.search(r"(pax|passenger|close|total|jumlah|penumpang)", c)
]
default_target = prefer[0] if prefer else options_target[0]
target_col = st.selectbox(
    "🎯 Pilih kolom target",
    options_target,
    index=options_target.index(default_target)
)

st.success(f"Dipakai → **Tanggal**: {date_col} | **Target**: {target_col}")

# Gabung
df = combine_by_date(raw_dfs, date_col, target_col)
if df.empty or df[date_col].isna().all():
    st.error("Data gabungan kosong / kolom tanggal tidak valid setelah konversi.")
    st.stop()

# Tambahkan nama hari di data gabungan
df[date_col] = pd.to_datetime(df[date_col])
hari_map_preview = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}
df["hari"] = df[date_col].dt.dayofweek.map(hari_map_preview)

st.subheader("🔎 Cuplikan Data Gabungan")
st.dataframe(df.head(20))

run = st.button("🚀 Jalankan Forecast")
if not run:
    st.stop()

# =========================
# Pipeline Forecast
# =========================
status = st.empty()
status.info("⏳ Memproses...")

df = df.sort_values(date_col).reset_index(drop=True)
dates = pd.to_datetime(df[date_col], errors="coerce")
y = pd.to_numeric(df[target_col], errors="coerce").values
mask_valid = (~dates.isna()) & (~np.isnan(y))
dates, y = dates[mask_valid], y[mask_valid]

if len(y) < 5:
    st.error("Data valid terlalu sedikit untuk forecast.")
    st.stop()

split_idx = int(len(y) * 0.8) if len(y) >= 10 else max(1, int(len(y) * 0.7))
train, test = y[:split_idx], y[split_idx:]
train_dates, test_dates = dates[:split_idx], dates[split_idx:]

freq_guess = pd.infer_freq(dates) or "D"
if str(freq_guess).upper().startswith("D"):
    periods_per_year = 365
elif str(freq_guess).upper().startswith("W"):
    periods_per_year = 52
elif str(freq_guess).upper().startswith("M"):
    periods_per_year = 12
else:
    periods_per_year = 365

# ====== MODELING ======
if model_choice == "Holt-Winters":
    # Holt-Winters
    trend_code = "add" if trend_type == "Additive" else "mul"
    season_code = "add" if season_type == "Additive" else "mul"
    use_damped = (damped == "Ya")

    # Multiplicative butuh semua > 0
    has_nonpos = np.any(train <= 0)
    if season_code == "mul" and has_nonpos:
        st.warning("Musiman multiplicative butuh semua nilai > 0 → dialihkan ke **Additive**.")
        season_code = "add"
    if trend_code == "mul" and has_nonpos:
        st.warning("Trend multiplicative butuh semua nilai > 0 → dialihkan ke **Additive**.")
        trend_code = "add"

    sp = int(seasonal_periods)
    use_seasonal = (sp >= 2) and (len(train) >= 2 * sp)

    if not use_seasonal:
        sp_guess = (
            7
            if str(freq_guess).upper().startswith("D")
            else (12 if str(freq_guess).upper().startswith("M") else 52)
        )
        if len(train) >= 2 * sp_guess:
            st.info(
                f"Seasonal periods {seasonal_periods} terlalu besar untuk panjang data → dipakai **{sp_guess}**."
            )
            sp = sp_guess
            use_seasonal = True
        else:
            st.info("Data train belum cukup untuk komponen **musiman** → model tanpa musiman.")

    try:
        hw = ExponentialSmoothing(
            train,
            trend=trend_code,
            seasonal=(season_code if use_seasonal else None),
            seasonal_periods=(sp if use_seasonal else None),
            damped_trend=use_damped,
            initialization_method="estimated",
        ).fit(optimized=True)
    except Exception:
        try:
            st.warning("Retry HW tanpa musiman (fallback 1).")
            hw = ExponentialSmoothing(
                train,
                trend=trend_code,
                seasonal=None,
                damped_trend=use_damped,
                initialization_method="estimated",
            ).fit(optimized=True)
        except Exception:
            st.warning("Retry SimpleExponentialSmoothing (fallback 2).")
            hw = SimpleExpSmoothing(
                train,
                initialization_method="estimated"
            ).fit(optimized=True)

    model = hw
    forecast_test = model.forecast(len(test))
    forecast_future = model.forecast(periods_per_year * n_years)

else:
    # Fitur waktu & lag untuk ML
    df_feat = pd.DataFrame({date_col: dates, target_col: y})
    df_feat = create_time_features(df_feat, date_col)
    for lag in [1, 7, 30]:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)
    df_feat = df_feat.dropna()

    X = df_feat.drop(columns=[target_col, date_col])
    y_ml = df_feat[target_col]
    split_idx_ml = int(len(y_ml) * 0.8) if len(y_ml) >= 10 else max(1, int(len(y_ml) * 0.7))
    X_train, X_test = X.iloc[:split_idx_ml], X.iloc[split_idx_ml:]
    y_train, y_test = y_ml.iloc[:split_idx_ml], y_ml.iloc[split_idx_ml:]

    if model_choice == "XGBoost":
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
        )
    else:
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves if model_choice == "LightGBM" else 31,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
        )

    model.fit(X_train, y_train)
    forecast_test = model.predict(X_test)

    # Forecast ke depan (harian)
    forecast_future = []
    last_data = df_feat.iloc[-30:].copy()
    base_last_date = dates.iloc[-1]
    for i in range(periods_per_year * n_years):
        next_date = base_last_date + pd.Timedelta(days=i + 1)
        new_row = {
            "month": next_date.month,
            "dayofweek": next_date.dayofweek,
            "quarter": (next_date.month - 1) // 3 + 1,
            "year": next_date.year,
            "dayofyear": next_date.dayofyear,
        }
        for lag in [1, 7, 30]:
            new_row[f"lag_{lag}"] = last_data[target_col].iloc[-lag]
        pred = float(model.predict(pd.DataFrame([new_row]))[0])
        forecast_future.append(pred)
        last_data.loc[len(last_data)] = {
            date_col: next_date,
            target_col: pred,
            **new_row,
        }

# ===== Evaluasi =====
forecast_test = np.array(forecast_test)[: len(test)]
mae = (
    mean_absolute_error(test[: len(forecast_test)], forecast_test)
    if len(test) and len(forecast_test)
    else np.nan
)
rmse = (
    np.sqrt(mean_squared_error(test[: len(forecast_test)], forecast_test))
    if len(test) and len(forecast_test)
    else np.nan
)
smape_val = (
    smape(test[: len(forecast_test)], forecast_test)
    if len(test) and len(forecast_test)
    else np.nan
)

c1, c2, c3 = st.columns(3)
c1.metric("📉 MAE", f"{mae:,.2f}" if pd.notna(mae) else "—")
c2.metric("📊 RMSE", f"{rmse:,.2f}" if pd.notna(rmse) else "—")
c3.metric("📈 SMAPE", f"{smape_val:.2f}%" if pd.notna(smape_val) else "—")

status.success("✅ Forecast selesai!")

# ===== Plot utama + CI95 =====
future_freq = pd.infer_freq(dates) or "D"
test_index = (
    pd.date_range(
        start=test_dates.iloc[0],
        periods=len(test),
        freq=(pd.infer_freq(test_dates) or future_freq),
    )
    if len(test) > 0
    else pd.DatetimeIndex([])
)
forecast_future_index = pd.date_range(
    start=dates.iloc[-1] + pd.tseries.frequencies.to_offset(future_freq),
    periods=periods_per_year * n_years,
    freq=future_freq,
)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train_dates, train, label="Train", linewidth=2, color="#9467bd")
if len(test) > 0:
    ax.plot(test_index, test, label="Aktual Test", linewidth=2, color="#ff7f0e")
    ax.plot(
        test_index[: len(forecast_test)],
        forecast_test,
        label="Prediksi Test",
        linewidth=2.5,
        color="#2ca02c",
    )
ax.plot(
    forecast_future_index,
    forecast_future,
    label=f"Forecast {forecast_future_index[0].year}–{forecast_future_index[-1].year}",
    linewidth=2.5,
    color="#2ca02c",
    alpha=0.8,
)

residuals = (
    test[: len(forecast_test)] - forecast_test
    if len(test) and len(forecast_test)
    else np.array([0.0])
)
ci95 = 1.96 * np.std(residuals) if len(residuals) > 1 else 0.0
upper_ci = np.array(forecast_future) + ci95
lower_ci = np.array(forecast_future) - ci95
ax.fill_between(
    forecast_future_index,
    lower_ci,
    upper_ci,
    color="#2ca02c",
    alpha=0.2,
    label="CI 95%",
)

if use_bbands:
    combined_index = pd.concat(
        [train_dates, test_dates, pd.Series(forecast_future_index)]
    )
    combined_series = pd.Series(
        np.concatenate([train, test, forecast_future]), index=combined_index
    )
    rolling_mean = combined_series.rolling(window=20).mean()
    rolling_std = combined_series.rolling(window=20).std()
    ax.plot(
        rolling_mean.index,
        rolling_mean + 2 * rolling_std,
        linestyle="--",
        alpha=0.6,
        label="Upper BB",
        color="gray",
    )
    ax.plot(
        rolling_mean.index,
        rolling_mean - 2 * rolling_std,
        linestyle="--",
        alpha=0.6,
        label="Lower BB",
        color="gray",
    )
    ax.plot(
        rolling_mean.index,
        rolling_mean,
        linestyle="-",
        linewidth=2.8,
        color="#1f77b4",
        alpha=0.95,
        label="Middle BB (MA 20)",
        zorder=5,
    )

ax.set_title(f"📈 Forecast — Target: {target_col}", fontsize=15, fontweight="bold")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Nilai")
ax.legend(loc="upper left", fontsize=10, facecolor="white")
ax.grid(alpha=0.3)
st.pyplot(fig)

# ===== MA + Anomali =====
full_index = pd.concat(
    [train_dates, test_dates, pd.Series(forecast_future_index)]
)
full_series = pd.Series(
    np.concatenate([train, test, forecast_future]), index=full_index
)
ma7 = full_series.rolling(7, min_periods=1).mean()
ma30 = full_series.rolling(30, min_periods=1).mean()
ma90 = full_series.rolling(90, min_periods=1).mean()
baseline_map = {"MA 7": ma7, "MA 30": ma30, "MA 90": ma90}
base = baseline_map.get(baseline_ma, ma30)

resid_ma = (full_series - base).dropna()
hist_series = pd.Series(
    np.concatenate([train, test]), index=pd.concat([train_dates, test_dates])
).dropna()
prof = hist_series.groupby(hist_series.index.dayofyear).median()
if 366 not in prof.index:
    prof.loc[366] = prof.loc[365] if 365 in prof.index else prof.median()
seasonal_expected = pd.Series(
    prof.reindex(full_series.index.dayofyear).values, index=full_series.index
)
resid_season = (full_series - seasonal_expected).dropna()


def anomaly_mask(residual_series, method, z_thresh=None, iqr_k=None, top_n=None):
    if len(residual_series) == 0:
        return pd.Series(False, index=residual_series.index)
    if method == "Z-Score Residual":
        stdv = residual_series.std(ddof=0)
        if stdv == 0 or np.isnan(stdv):
            return pd.Series(False, index=residual_series.index)
        z = (residual_series - residual_series.mean()) / (stdv + 1e-12)
        return z.abs() >= float(z_thresh)
    elif method == "IQR Residual":
        q1, q3 = residual_series.quantile(0.25), residual_series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            return pd.Series(False, index=residual_series.index)
        lower, upper = (
            q1 - float(iqr_k) * iqr,
            q3 + float(iqr_k) * iqr,
        )
        return (residual_series < lower) | (residual_series > upper)
    else:
        n = max(1, int(top_n))
        if n >= len(residual_series):
            return pd.Series(True, index=residual_series.index)
        top_idx = residual_series.abs().nlargest(n).index
        return residual_series.index.isin(top_idx)


if anomaly_method == "Z-Score Residual":
    mask_ma = anomaly_mask(resid_ma, "Z-Score Residual", z_thresh=z_thresh)
    mask_season = anomaly_mask(
        resid_season, "Z-Score Residual", z_thresh=float(z_thresh)
    )
elif anomaly_method == "IQR Residual":
    mask_ma = anomaly_mask(resid_ma, "IQR Residual", iqr_k=iqr_k)
    mask_season = anomaly_mask(
        resid_season, "IQR Residual", iqr_k=float(iqr_k)
    )
else:
    mask_ma = anomaly_mask(resid_ma, "Top-N Residual", top_n=top_n)
    mask_season = anomaly_mask(
        resid_season, "Top-N Residual", top_n=int(top_n)
    )

idx_common = resid_ma.index.intersection(resid_season.index)
mask_red = (mask_ma.loc[idx_common] & mask_season.loc[idx_common])
mask_purp = (
    mask_ma.loc[idx_common] & (~mask_season.loc[idx_common])
)

anoms_red = full_series.loc[idx_common[mask_red]]
anoms_purp = full_series.loc[idx_common[mask_purp]]

fig_ma, ax_ma = plt.subplots(figsize=(14, 6))
ax_ma.plot(
    full_series.index,
    full_series.values,
    color="lightgray",
    alpha=0.6,
    label="Data",
)
ax_ma.plot(ma7.index, ma7.values, label="MA 7")
ax_ma.plot(ma30.index, ma30.values, label="MA 30")
ax_ma.plot(ma90.index, ma90.values, label="MA 90")
if enable_anomaly:
    if len(anoms_purp) > 0:
        ax_ma.scatter(
            anoms_purp.index,
            anoms_purp.values,
            s=36,
            color="purple",
            label=f"Outlier musiman wajar (🟣 {len(anoms_purp)})",
            zorder=3,
        )
    if len(anoms_red) > 0:
        ax_ma.scatter(
            anoms_red.index,
            anoms_red.values,
            s=40,
            color="red",
            label=f"Anomali tak wajar (🔴 {len(anoms_red)})",
            zorder=4,
        )
ax_ma.set_title(f"Moving Average — {target_col}")
ax_ma.set_xlabel("Tanggal")
ax_ma.set_ylabel("Nilai")
ax_ma.legend(loc="upper left", fontsize=10, facecolor="white")
ax_ma.grid(alpha=0.3)
st.pyplot(fig_ma)

# ===== Tabel Data Harian + Nama Hari =====
with st.expander("📋 Data Harian Lengkap (dengan nama hari)"):
    hari_map = {
        0: "Senin",
        1: "Selasa",
        2: "Rabu",
        3: "Kamis",
        4: "Jumat",
        5: "Sabtu",
        6: "Minggu",
    }

    daily_df = pd.DataFrame({"Tanggal": full_series.index})
    daily_df["Tanggal"] = pd.to_datetime(daily_df["Tanggal"])
    daily_df["Hari"] = daily_df["Tanggal"].dt.dayofweek.map(hari_map)
    daily_df["Nilai"] = full_series.values
    daily_df["MA7"] = ma7.reindex(full_series.index).values
    daily_df["MA30"] = ma30.reindex(full_series.index).values
    daily_df["MA90"] = ma90.reindex(full_series.index).values
    daily_df["Anomali_Tak_Wajar"] = daily_df["Tanggal"].isin(
        anoms_red.index
    )
    daily_df["Outlier_Musiman_Wajar"] = daily_df["Tanggal"].isin(
        anoms_purp.index
    )

    st.dataframe(daily_df)

    csv_daily = io.StringIO()
    daily_df.to_csv(csv_daily, index=False)
    st.download_button(
        label="⬇️ Download Data Harian (CSV)",
        data=csv_daily.getvalue().encode("utf-8"),
        file_name=f"data_harian_{target_col}.csv",
        mime="text/csv",
    )

# ===== CI95 Agregasi =====
st.subheader("📊 Analisis CI 95% (Agregasi)")
st.write(f"Nilai CI95 (±): **{ci95:,.2f}**")

# buat dataframe CI95 (tanggal harian forecast)
ci_df = pd.DataFrame(
    {
        "Tanggal": forecast_future_index,
        "Forecast": forecast_future,
        "Upper_CI": upper_ci,
        "Lower_CI": lower_ci,
    }
).reset_index(drop=True)

# map hari Indonesia (bisa dipakai untuk semua level agregasi; untuk agregasi non-harian 'Hari' akan mengikuti tanggal representatif)
hari_map_name = {
    'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
    'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
}
ci_df["Tanggal"] = pd.to_datetime(ci_df["Tanggal"])
ci_df["Hari"] = ci_df["Tanggal"].dt.day_name().map(hari_map_name)

# Buat index datetime agar mudah resample
ci_df = ci_df.set_index("Tanggal")

# Pilih agregasi sesuai pilihan
if ci_agg_choice == "Per Hari":
    agg_df = ci_df.copy().reset_index()
    agg_df["period_label"] = agg_df["Tanggal"].dt.strftime("%Y-%m-%d")
    width = 1
elif ci_agg_choice == "Per Minggu":
    agg_df = ci_df.resample("W").mean().reset_index()
    agg_df["period_label"] = agg_df["Tanggal"].dt.to_period("W").astype(str)
    # assign Hari dari tanggal representatif (start/end of week) — tetap gunakan hari dari tanggal index
    agg_df["Hari"] = agg_df["Tanggal"].dt.day_name().map(hari_map_name)
    width = 4
elif ci_agg_choice == "Per Bulan":
    agg_df = ci_df.resample("M").mean().reset_index()
    agg_df["period_label"] = agg_df["Tanggal"].dt.to_period("M").astype(str)
    agg_df["Hari"] = agg_df["Tanggal"].dt.day_name().map(hari_map_name)
    width = 20
elif ci_agg_choice == "Per Kuartal":
    agg_df = ci_df.resample("Q").mean().reset_index()
    agg_df["period_label"] = agg_df["Tanggal"].dt.to_period("Q").astype(str)
    agg_df["Hari"] = agg_df["Tanggal"].dt.day_name().map(hari_map_name)
    width = 30
else:  # Per Tahun
    agg_df = ci_df.resample("A").mean().reset_index()
    agg_df["period_label"] = agg_df["Tanggal"].dt.to_period("A").astype(str)
    agg_df["Hari"] = agg_df["Tanggal"].dt.day_name().map(hari_map_name)
    width = 90

# Jika semua period_label identik dengan tanggal (format YYYY-MM-DD), hapus kolom period_label supaya tidak redundant
try:
    if "period_label" in agg_df.columns:
        cmp_series = agg_df["Tanggal"].dt.strftime("%Y-%m-%d")
        if (agg_df["period_label"].astype(str) == cmp_series.astype(str)).all():
            agg_df = agg_df.drop(columns=["period_label"])


except Exception:
    pass

# Pastikan kolom Hari ada (jika belum karena resample mean menghapus string -> kita rekonstruksi dari index)
if "Hari" not in agg_df.columns:
    if "Tanggal" in agg_df.columns:
        agg_df["Hari"] = pd.to_datetime(agg_df["Tanggal"]).dt.day_name().map(hari_map_name)
    else:
        agg_df["Hari"] = ""

# Tampilkan grafik CI95 (bar + errorbar) dan tambahkan penanda nama hari di atas tiap batang
fig_ci, ax_ci = plt.subplots(figsize=(14, 6))
x = agg_df["Tanggal"]
heights = agg_df["Forecast"]
# Untuk kasus hasil resample, kolom Lower_CI/Upper_CI masih numeric (mean). Hitung error untuk errorbar:
if "Lower_CI" in agg_df.columns and "Upper_CI" in agg_df.columns:
    lower_err = agg_df["Forecast"] - agg_df["Lower_CI"]
    upper_err = agg_df["Upper_CI"] - agg_df["Forecast"]
else:
    lower_err = np.zeros_like(heights)
    upper_err = np.zeros_like(heights)

ax_ci.bar(
    x,
    heights,
    width=width,
    align="center",
    color="#777777",
    alpha=0.7,
    label=f"Forecast ({ci_agg_choice})",
)
ax_ci.errorbar(
    x,
    heights,
    yerr=[lower_err, upper_err],
    fmt="none",
    ecolor="black",
    elinewidth=1.5,
    capsize=4,
    label="CI 95%",
)

# Label angka di atas batang (jika sedikit titik, tampilkan; jika banyak, tetap dicoba tetapi bisa overlap)
for xi, yi, hari in zip(x, heights, agg_df["Hari"]):
    if pd.notna(yi):
        # angka di atas batang
        ax_ci.text(
            xi,
            yi,
            f"{yi:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        # penanda hari (singkatan) sedikit di atas angka untuk terlihat
        hari_singkat = "" if pd.isna(hari) else (hari[:3])  # misal 'Sen' untuk 'Senin'
        ax_ci.text(
            xi,
            yi + (0.02 * (np.nanmax(heights) - np.nanmin(heights)) if len(heights)>0 else 0),
            hari_singkat,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )

ax_ci.set_title(f"📆 CI 95% ({ci_agg_choice}) — {target_col}")
ax_ci.set_xlabel("Periode")
ax_ci.set_ylabel("Nilai")
ax_ci.legend(loc="upper left", fontsize=10, facecolor="white")
ax_ci.grid(alpha=0.3)
plt.xticks(rotation=30)
st.pyplot(fig_ci)

# Tampilkan tabel agregasi CI95 dan sediakan tombol download CSV (kolom Hari sudah termasuk)
with st.expander("📋 Lihat Data CI 95% (Agregasi)"):
    display_df = agg_df.copy()
    # Pastikan tanggal tampil rapi (jika datetime -> date)
    if pd.api.types.is_datetime64_any_dtype(display_df["Tanggal"]):
        display_df["Tanggal"] = display_df["Tanggal"].dt.date

    # Urutkan kolom agar lebih rapi di tabel
    cols_order = ["Tanggal", "Hari"] + [c for c in ["period_label", "Forecast", "Lower_CI", "Upper_CI"] if c in display_df.columns]
    cols_order = [c for c in cols_order if c in display_df.columns]  # filter yang ada
    st.dataframe(display_df[cols_order].reset_index(drop=True), use_container_width=True)

    csv_buffer = io.StringIO()
    # simpan CSV dengan kolom Hari terikut
    display_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download CI95 Agregasi (CSV)",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name=f"ci95_{target_col}_{ci_agg_choice.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )
