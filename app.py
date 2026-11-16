"""Dashboard segmentasi pelanggan mall dengan tema elegan bernuansa ungu muda."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Segmentasi Pelanggan Mall X",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded",
)

INCOME = "Annual_Income($)"
SPEND = "Spending Score(1-100)"
REQUIRED = [
    "CustomerID",
    "Gender",
    "Age",
    INCOME,
    SPEND,
    "Profession",
    "Work_Experience",
    "Family_Size",
]

CLUSTER_COLORS = [
    "#8B5CF6",  # violet
    "#C084FC",  # light purple
    "#EC4899",  # pink
    "#6366F1",  # indigo
    "#A855F7",
    "#D946EF",
    "#7C3AED",
    "#F472B6",
    "#9333EA",
    "#818CF8",
]


# =========================
# CSS tema elegan
# =========================
st.markdown(
    """
    <style>
        :root {
            --bg: #f7f3ff;
            --panel: #ffffff;
            --panel-soft: #f2ebff;
            --panel-deep: #ece3ff;
            --line: #d9c9ff;
            --line-strong: #b99cff;
            --text: #2e2155;
            --muted: #6e6291;
            --primary: #7c3aed;
            --primary-2: #9f67ff;
            --accent: #c084fc;
            --shadow: 0 10px 28px rgba(124, 58, 237, 0.10);
            --shadow-soft: 0 6px 18px rgba(124, 58, 237, 0.08);
            --radius: 18px;
        }

        .stApp {
            background: linear-gradient(180deg, #faf7ff 0%, #f5efff 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.15rem;
            max-width: 1380px;
        }

        h1, h2, h3 {
            color: var(--text);
            letter-spacing: -0.02em;
        }

        p, li, label, .stMarkdown, .stCaption {
            color: var(--text);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f0e8ff 0%, #ebe2ff 100%);
            border-right: 1px solid var(--line);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.1rem;
        }

        .brand-box {
            background: linear-gradient(135deg, #ffffff 0%, #f4eeff 100%);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: var(--shadow-soft);
            margin-bottom: 0.8rem;
        }

        .brand-kicker {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--primary);
            margin-bottom: 0.3rem;
        }

        .brand-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--text);
            margin: 0;
        }

        .hero-card {
            background: linear-gradient(135deg, #ffffff 0%, #f6f0ff 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.05rem 1.2rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            color: var(--primary);
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 0.35rem;
            line-height: 1.1;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            margin-bottom: 0;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #faf7ff 100%);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: var(--shadow-soft);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
        }

        .section-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: var(--shadow-soft);
            margin-bottom: 1rem;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin-bottom: 0.6rem;
        }

        button[data-baseweb="tab"] {
            background: #f3ebff;
            border: 1px solid var(--line);
            border-radius: 14px;
            color: var(--text);
            font-weight: 700;
            padding: 0.45rem 0.9rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary), var(--primary-2));
            color: white;
            border-color: transparent;
            box-shadow: var(--shadow-soft);
        }

        /* Input dan select yang lebih jelas */
        .stNumberInput > div > div,
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div,
        .stTextArea textarea {
            background: #ffffff !important;
            border: 1.5px solid var(--line-strong) !important;
            border-radius: 14px !important;
            color: var(--text) !important;
            box-shadow: none !important;
        }

        .stNumberInput input,
        .stSelectbox input,
        .stTextInput input,
        .stTextArea textarea {
            color: var(--text) !important;
            font-weight: 600 !important;
        }

        .stNumberInput label,
        .stSelectbox label,
        .stSlider label,
        .stFileUploader label,
        .stRadio label,
        .stMarkdown label {
            color: var(--text) !important;
            font-weight: 700 !important;
        }

        div[data-baseweb="select"] > div {
            min-height: 48px;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.78);
            border: 2px dashed var(--line-strong);
            border-radius: 18px;
            padding: 10px;
        }

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] div {
            color: var(--text) !important;
        }

        /* Slider */
        div[data-baseweb="slider"] > div > div > div {
            background: var(--primary) !important;
        }

        div[data-baseweb="slider"] [role="slider"] {
            background: #ffffff !important;
            border: 3px solid var(--primary) !important;
            box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.12);
        }

        /* Tombol */
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, var(--primary), var(--primary-2));
            color: #ffffff;
            border: none;
            border-radius: 14px;
            font-weight: 700;
            box-shadow: var(--shadow-soft);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            filter: brightness(1.03);
        }

        .stAlert {
            border-radius: 16px;
            border: 1px solid var(--line);
        }

        .small-note {
            color: var(--muted);
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Utilitas data
# =========================
def find_default_data() -> Path | None:
    base = Path(__file__).resolve().parent
    candidates = [
        base / "data" / "Customer_mall.csv",
        base / "data" / "Customer_mall(1).csv",
        base / "Customer_mall.csv",
        base / "Customer_mall(1).csv",
        Path("/mnt/data/Customer_mall.csv"),
        Path("/mnt/data/Customer_mall(1).csv"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_default() -> pd.DataFrame:
    data_path = find_default_data()
    if data_path is None:
        st.warning(
            "Dataset bawaan tidak ditemukan. Silakan unggah file CSV melalui sidebar."
        )
        st.stop()
    return pd.read_csv(data_path)


@st.cache_data(show_spinner=False)
def load_uploaded(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(raw))


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing))
        st.stop()
    df = df.copy()
    for col in ["Age", INCOME, SPEND, "Work_Experience", "Family_Size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare(raw: pd.DataFrame, advanced: bool):
    df = raw.copy()
    stats = {
        "Baris awal": len(df),
        "Profesi kosong (awal)": int(df["Profession"].isna().sum()),
    }

    df["Profession"] = df["Profession"].fillna("Tidak diketahui")

    if advanced:
        age_median = df.loc[df["Age"].gt(0)].groupby("Profession")["Age"].median()
        global_age = df.loc[df["Age"].gt(0), "Age"].median()
        mask_age_zero = df["Age"].eq(0)
        df.loc[mask_age_zero, "Age"] = (
            df.loc[mask_age_zero, "Profession"].map(age_median).fillna(global_age)
        )

        invalid_age = (
            ((df["Age"] < 17) | (df["Age"] > 80))
            & ~df["Profession"].isin(["Artist", "Entertainment"])
        )
        df = df.loc[~invalid_age].copy()

        mismatch = df["Work_Experience"] > df["Age"]
        df = df.loc[~mismatch].copy()

        we_median = (
            df.loc[df["Work_Experience"].gt(0)]
            .groupby("Profession")["Work_Experience"]
            .median()
        )
        global_we = df.loc[df["Work_Experience"].gt(0), "Work_Experience"].median()
        mask_we_zero = df["Work_Experience"].eq(0)
        df.loc[mask_we_zero, "Work_Experience"] = (
            df.loc[mask_we_zero, "Profession"].map(we_median).fillna(global_we)
        )

        df = df.loc[df[INCOME].gt(0)].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[INCOME, SPEND]).copy()

    stats["Baris akhir"] = len(df)
    stats["Baris dikeluarkan"] = stats["Baris awal"] - stats["Baris akhir"]
    return df, stats


@st.cache_data(show_spinner=False)
def run_model(df: pd.DataFrame, n_clusters: int, random_state: int):
    X = df[[INCOME, SPEND]].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    result = df.copy()
    result["Cluster"] = labels
    result["Cluster Label"] = result["Cluster"].map(lambda x: f"Klaster {x}")

    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=[INCOME, SPEND],
    )
    centers["Cluster"] = np.arange(n_clusters)

    sil = float(silhouette_score(X_scaled, labels)) if 1 < len(np.unique(labels)) < len(X_scaled) else np.nan
    dbi = float(davies_bouldin_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else np.nan
    return result, centers, model, scaler, sil, dbi


@st.cache_data(show_spinner=False)
def evaluate(df: pd.DataFrame, max_k: int, random_state: int) -> pd.DataFrame:
    X = StandardScaler().fit_transform(df[[INCOME, SPEND]])
    metrics = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        metrics.append(
            {
                "K": k,
                "Inertia": float(km.inertia_),
                "Silhouette": float(silhouette_score(X, labels)),
                "Davies-Bouldin": float(davies_bouldin_score(X, labels)),
            }
        )
    return pd.DataFrame(metrics)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def business_label(row: pd.Series, income_cut: float, spend_cut: float) -> str:
    high_i = row[INCOME] >= income_cut
    high_s = row[SPEND] >= spend_cut
    if high_i and high_s:
        return "VIP (pendapatan & belanja tinggi)"
    if high_i and not high_s:
        return "Potensial (pendapatan tinggi, belanja rendah)"
    if not high_i and high_s:
        return "Aktif (pendapatan rendah, belanja tinggi)"
    return "Hemat (pendapatan & belanja rendah)"


# =========================
# Header utama
# =========================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Mall X / Analytics</div>
        <div class="hero-title">Dashboard Segmentasi Pelanggan</div>
        <div class="hero-subtitle">
            Analisis data pelanggan menggunakan K-Means, lengkap dengan Inertia,
            Silhouette Score, Davies-Bouldin Index, profil klaster, dan prediksi pelanggan baru.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-kicker">Mall X / Analytics</div>
            <p class="brand-title">Control Center</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    file = st.file_uploader(
        "Ganti dataset (opsional)",
        type=["csv"],
        help="Jika tidak diunggah, aplikasi akan memakai dataset bawaan Customer_mall.csv.",
    )

    raw = validate(load_uploaded(file.getvalue()) if file else load_default())
    st.caption(f"Sumber data: {'CSV yang diunggah' if file else 'Dataset bawaan'}")

    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="Nilai random state memengaruhi inisialisasi pusat klaster agar hasil dapat direproduksi.",
    )

    mode = st.radio(
        "Mode preprocessing",
        ["Dasar (2 fitur)", "Aturan preprocessing materi PDF"],
        help="Mode PDF menerapkan aturan usia, pengalaman kerja, dan profesi seperti materi."
    )
    advanced = mode.startswith("Aturan")

    cleaned, prep_stats = prepare(raw, advanced)

    if len(cleaned) < 3 or cleaned[[INCOME, SPEND]].drop_duplicates().shape[0] < 2:
        st.error("Data valid atau variasi fitur tidak cukup untuk K-Means.")
        st.stop()

    kmax = min(10, len(cleaned) - 1, cleaned[[INCOME, SPEND]].drop_duplicates().shape[0])
    if kmax < 2:
        st.error("Minimal dua titik unik dibutuhkan.")
        st.stop()

    eval_scores = evaluate(cleaned, kmax, random_state)
    recommended_k = int(eval_scores.loc[eval_scores["Silhouette"].idxmax(), "K"])

    k_limit = st.slider(
        "Uji klaster hingga K",
        min_value=2,
        max_value=max(2, kmax),
        value=max(2, kmax),
        help="Menentukan batas maksimum jumlah klaster yang dievaluasi pada grafik metrik.",
    )
    eval_scores = evaluate(cleaned, k_limit, random_state)
    recommended_k = int(eval_scores.loc[eval_scores["Silhouette"].idxmax(), "K"])

    options_k = list(range(2, k_limit + 1))
    default_index = options_k.index(recommended_k) if recommended_k in options_k else 0
    k = st.selectbox(
        "Jumlah klaster model akhir",
        options=options_k,
        index=default_index,
        help="Pilih jumlah klaster yang akan digunakan pada model utama.",
    )

    st.success(f"K rekomendasi Silhouette = {recommended_k}")
    st.markdown(
        "<div class='small-note'>Metode: StandardScaler + KMeans (n_init=10).</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div class='small-note'><b>Sumber pemodelan</b><br>Annual Income ($) & Spending Score (1–100). Kedua fitur distandardisasi sebelum clustering.</div>",
        unsafe_allow_html=True,
    )

with st.spinner("Menghitung model segmentasi..."):
    result, centers, model, scaler, sil, dbi = run_model(cleaned, k, random_state)

income_cut = float(result[INCOME].median())
spend_cut = float(result[SPEND].median())
centers["Profil indikatif"] = centers.apply(
    lambda row: business_label(row, income_cut, spend_cut), axis=1
)
label_map = dict(zip(centers["Cluster"], centers["Profil indikatif"]))
result["Profil indikatif"] = result["Cluster"].map(label_map)

# =========================
# Tabs utama
# =========================
tabs = st.tabs(
    [
        "📊 Dashboard",
        "🔎 Eksplorasi Data",
        "🧹 Preprocessing",
        "🧠 K-Means",
        "📈 Evaluasi Model",
        "🎯 Segmentasi Pelanggan",
    ]
)

# -------------------------
# Dashboard
# -------------------------
with tabs[0]:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total pelanggan", f"{len(result):,}")
    m2.metric("Rata-rata usia", f"{result['Age'].mean():.1f} tahun")
    m3.metric("Pendapatan rata-rata", f"$ {result[INCOME].mean():,.0f}")
    m4.metric("Skor belanja rata-rata", f"{result[SPEND].mean():.1f} / 100")

    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Distribusi pelanggan per klaster")
        fig = px.scatter(
            result,
            x=SPEND,
            y=INCOME,
            color="Cluster Label",
            color_discrete_sequence=CLUSTER_COLORS,
            hover_data=["CustomerID", "Gender", "Age", "Profession", "Profil indikatif"],
            opacity=0.72,
            template="plotly_white",
        )
        fig.update_layout(
            height=480,
            margin=dict(t=20, b=10, l=10, r=10),
            legend_title_text="Klaster",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Komposisi klaster")
        counts = (
            result["Cluster Label"]
            .value_counts()
            .sort_index()
            .rename_axis("Klaster")
            .reset_index(name="Jumlah")
        )
        pie = px.pie(
            counts,
            names="Klaster",
            values="Jumlah",
            hole=0.60,
            color_discrete_sequence=CLUSTER_COLORS,
            template="plotly_white",
        )
        pie.update_layout(
            height=420,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(pie, use_container_width=True)

    st.info(
        "Semakin ke kanan berarti Spending Score lebih tinggi, sedangkan semakin ke atas berarti Annual Income lebih tinggi."
    )

# -------------------------
# Eksplorasi data
# -------------------------
with tabs[1]:
    st.subheader("Dataset pelanggan")
    st.dataframe(raw, use_container_width=True, hide_index=True, height=320)

    q1, q2, q3 = st.columns(3)
    q1.metric("Jumlah baris", len(raw))
    q2.metric("Jumlah nilai kosong", int(raw.isna().sum().sum()))
    q3.metric("Baris duplikat", int(raw.duplicated().sum()))

    with st.expander("Lihat nilai kosong per kolom"):
        st.dataframe(raw.isna().sum().rename("Jumlah kosong"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_gender = px.histogram(
            raw,
            x="Gender",
            color="Gender",
            color_discrete_sequence=["#A78BFA", "#EC4899"],
            template="plotly_white",
            title="Distribusi gender",
        )
        st.plotly_chart(fig_gender, use_container_width=True)

        fig_prof = px.histogram(
            raw,
            x="Profession",
            color_discrete_sequence=["#8B5CF6"],
            template="plotly_white",
            title="Distribusi profesi",
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    with c2:
        fig_age = px.histogram(
            raw,
            x="Age",
            nbins=22,
            color_discrete_sequence=["#C084FC"],
            template="plotly_white",
            title="Distribusi usia",
        )
        st.plotly_chart(fig_age, use_container_width=True)

        fig_age_spend = px.scatter(
            raw,
            x="Age",
            y=SPEND,
            opacity=0.55,
            color_discrete_sequence=["#7C3AED"],
            template="plotly_white",
            title="Usia vs Spending Score",
        )
        st.plotly_chart(fig_age_spend, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_we = px.histogram(
            raw,
            x="Work_Experience",
            color_discrete_sequence=["#A855F7"],
            template="plotly_white",
            title="Distribusi pengalaman kerja",
        )
        st.plotly_chart(fig_we, use_container_width=True)
    with c4:
        fig_fs = px.histogram(
            raw,
            x="Family_Size",
            color_discrete_sequence=["#D946EF"],
            template="plotly_white",
            title="Distribusi ukuran keluarga",
        )
        st.plotly_chart(fig_fs, use_container_width=True)

# -------------------------
# Preprocessing
# -------------------------
with tabs[2]:
    st.subheader("Tahapan preprocessing")
    st.write(f"**Mode aktif:** {mode}")
    st.write(
        "**Fitur utama model:** Annual_Income($) dan Spending Score(1-100)."
    )

    if advanced:
        st.markdown(
            """
            - Profesi kosong diisi menjadi **Tidak diketahui**.
            - Usia 0 diimputasi dengan median usia berdasarkan profesi.
            - Usia di luar 17–80 dieliminasi, kecuali profesi **Artist** atau **Entertainment**.
            - Data dengan **Work_Experience > Age** dibuang.
            - Work_Experience = 0 diimputasi median profesi.
            - Annual Income ≤ 0 dibuang.
            - Setelah bersih, model hanya memakai **Annual Income** dan **Spending Score**.
            """
        )
    else:
        st.markdown(
            "Mode dasar hanya membersihkan nilai kosong/tidak valid pada fitur utama serta mengganti profesi kosong menjadi **Tidak diketahui**."
        )

    p1, p2, p3 = st.columns(3)
    p1.metric("Baris awal", prep_stats["Baris awal"])
    p2.metric("Baris dikeluarkan", prep_stats["Baris dikeluarkan"])
    p3.metric("Baris siap model", prep_stats["Baris akhir"])

    st.dataframe(cleaned.head(30), use_container_width=True, hide_index=True)
    st.download_button(
        "Unduh data bersih (CSV)",
        csv_bytes(cleaned),
        "data_pelanggan_bersih.csv",
        "text/csv",
    )

# -------------------------
# K-Means
# -------------------------
with tabs[3]:
    st.subheader("Hasil pemodelan K-Means")
    a, b, c = st.columns(3)
    a.metric("Jumlah klaster", k)
    b.metric("Silhouette Score", f"{sil:.4f}")
    c.metric("Davies-Bouldin", f"{dbi:.4f}")

    cluster_fig = px.scatter(
        result,
        x=SPEND,
        y=INCOME,
        color="Cluster Label",
        color_discrete_sequence=CLUSTER_COLORS,
        hover_data=["CustomerID", "Age", "Profession", "Profil indikatif"],
        title="Visualisasi hasil clustering",
        template="plotly_white",
        opacity=0.75,
    )
    cluster_fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(cluster_fig, use_container_width=True)

    st.markdown("**Pusat klaster dalam satuan asli**")
    st.dataframe(centers, use_container_width=True, hide_index=True)

    st.download_button(
        "Unduh hasil segmentasi (CSV)",
        csv_bytes(result.drop(columns=["Cluster Label"])),
        "hasil_segmentasi_pelanggan.csv",
        "text/csv",
    )

# -------------------------
# Evaluasi model
# -------------------------
with tabs[4]:
    st.subheader("Evaluasi model")
    st.caption(
        "Inertia dan Davies-Bouldin yang lebih rendah lebih baik, sedangkan Silhouette yang lebih tinggi lebih baik."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        elbow = px.line(
            eval_scores,
            x="K",
            y="Inertia",
            markers=True,
            title="Inertia / Elbow Method",
            template="plotly_white",
        )
        elbow.update_traces(line_color="#7C3AED")
        st.plotly_chart(elbow, use_container_width=True)

    with c2:
        silfig = px.line(
            eval_scores,
            x="K",
            y="Silhouette",
            markers=True,
            title="Silhouette Score",
            template="plotly_white",
        )
        silfig.update_traces(line_color="#C084FC")
        st.plotly_chart(silfig, use_container_width=True)

    with c3:
        dbfig = px.line(
            eval_scores,
            x="K",
            y="Davies-Bouldin",
            markers=True,
            title="Davies-Bouldin Index",
            template="plotly_white",
        )
        dbfig.update_traces(line_color="#EC4899")
        st.plotly_chart(dbfig, use_container_width=True)

    st.dataframe(
        eval_scores.style.format(
            {
                "Inertia": "{:.2f}",
                "Silhouette": "{:.4f}",
                "Davies-Bouldin": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    best_s = int(eval_scores.loc[eval_scores["Silhouette"].idxmax(), "K"])
    best_db = int(eval_scores.loc[eval_scores["Davies-Bouldin"].idxmin(), "K"])

    st.success(
        f"K terbaik berdasarkan Silhouette Score adalah **{best_s}**, sedangkan berdasarkan Davies-Bouldin Index adalah **{best_db}**."
    )

# -------------------------
# Segmentasi pelanggan
# -------------------------
with tabs[5]:
    st.subheader("Karakteristik dan kesimpulan klaster")
    summary = (
        result.groupby("Cluster")
        .agg(
            Jumlah=("CustomerID", "size"),
            Pendapatan_Rata_Rata=(INCOME, "mean"),
            Skor_Belanja_Rata_Rata=(SPEND, "mean"),
            Usia_Rata_Rata=("Age", "mean"),
        )
        .reset_index()
    )
    summary["Persentase (%)"] = (100 * summary["Jumlah"] / len(result)).round(2)
    summary["Profil indikatif"] = summary["Cluster"].map(label_map)

    st.dataframe(
        summary.style.format(
            {
                "Pendapatan_Rata_Rata": "${:,.0f}",
                "Skor_Belanja_Rata_Rata": "{:.1f}",
                "Usia_Rata_Rata": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    top_cluster = summary.sort_values("Skor_Belanja_Rata_Rata", ascending=False).iloc[0]
    low_cluster = summary.sort_values("Skor_Belanja_Rata_Rata", ascending=True).iloc[0]

    st.markdown(
        f"""
        **Kesimpulan otomatis:**
        - Klaster dengan rata-rata belanja tertinggi adalah **Klaster {int(top_cluster['Cluster'])}**.
        - Klaster dengan rata-rata belanja terendah adalah **Klaster {int(low_cluster['Cluster'])}**.
        - Model saat ini menggunakan **K = {k}**, dengan **Silhouette Score = {sil:.4f}** dan **Davies-Bouldin Index = {dbi:.4f}**.
        - Profil klaster dapat dipakai sebagai dasar strategi promosi, loyalitas, dan penawaran yang lebih terarah.
        """
    )

    st.warning(
        "Profil bisnis bersifat interpretatif untuk membantu pembacaan hasil clustering, bukan label resmi dari dataset."
    )

    st.download_button(
        "Unduh ringkasan klaster",
        csv_bytes(summary),
        "ringkasan_klaster.csv",
        "text/csv",
    )

    st.subheader("Prediksi pelanggan baru")
    with st.form("form_prediksi"):
        f1, f2 = st.columns(2)
        with f1:
            income = st.number_input(
                "Pendapatan tahunan ($)", min_value=0.0, value=80000.0, step=1000.0
            )
        with f2:
            score = st.number_input(
                "Spending Score (0–100)",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
            )
        submitted = st.form_submit_button("Prediksi klaster")

    if submitted:
        x = pd.DataFrame([{INCOME: income, SPEND: score}])
        pred = int(model.predict(scaler.transform(x))[0])
        st.success(f"Pelanggan diprediksi masuk Klaster {pred} — {label_map[pred]}")
        st.caption(
            "Prediksi mengikuti model aktif dan bergantung pada data, preprocessing, serta jumlah klaster yang dipilih."
        )

st.divider()
st.caption(
    "Sumber: Customer_mall.csv dan materi 'Pemodelan Data Pelanggan Mall Dengan Metode Kmeans' (Adi Surya Basri)."
)
