"""Dashboard segmentasi pelanggan mal — K-Means (materi Mall X)."""
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentasi Pelanggan Mall X", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")
INCOME = "Annual_Income($)"
SPEND = "Spending Score(1-100)"
REQUIRED = ["CustomerID", "Gender", "Age", INCOME, SPEND, "Profession", "Work_Experience", "Family_Size"]
PALETTE = ["#0f766e", "#f59e0b", "#6366f1", "#e11d48", "#06b6d4", "#8b5cf6", "#64748b", "#84cc16", "#f97316", "#ec4899"]

st.markdown("""<style>
.block-container{padding-top:1.35rem;max-width:1350px}
[data-testid="stMetric"]{border:1px solid #d9e2e7;border-radius:12px;padding:12px 16px}
h1,h2,h3{letter-spacing:-0.025em}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_default():
    return pd.read_csv(
    Path(_file_).parent / "Customer_mall.csv"
)

@st.cache_data(show_spinner=False)
def load_uploaded(raw):
    return pd.read_csv(BytesIO(raw))


def validate(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing))
        st.stop()
    df = df.copy()
    for col in ["Age", INCOME, SPEND, "Work_Experience", "Family_Size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare(raw, advanced):
    """Mode 2 fitur: hanya data valid untuk income dan score.
       Mode PDF: aturan pembersihan terdokumentasi, tetapi clustering tetap 2 fitur."""
    df = raw.copy()
    stats = {"Baris awal": len(df), "Profesi kosong (awal)": int(df["Profession"].isna().sum())}
    df["Profession"] = df["Profession"].fillna("Tidak diketahui")
    if advanced:
        # Sesuai materi: usia 0 diimputasi median profesi; ketentuan artist/entertainment untuk usia ekstrem.
        age_median = df.loc[df["Age"].gt(0)].groupby("Profession")["Age"].median()
        global_age = df.loc[df["Age"].gt(0), "Age"].median()
        z = df["Age"].eq(0)
        df.loc[z, "Age"] = df.loc[z, "Profession"].map(age_median).fillna(global_age)
        invalid_age = ((df["Age"] < 17) | (df["Age"] > 80)) & ~df["Profession"].isin(["Artist", "Entertainment"])
        df = df.loc[~invalid_age].copy()
        we_median = df.loc[df["Work_Experience"].gt(0)].groupby("Profession")["Work_Experience"].median()
        global_we = df.loc[df["Work_Experience"].gt(0), "Work_Experience"].median()
        mismatch = df["Work_Experience"] > df["Age"]
        df = df.loc[~mismatch].copy()
        z = df["Work_Experience"].eq(0)
        df.loc[z, "Work_Experience"] = df.loc[z, "Profession"].map(we_median).fillna(global_we)
        df = df.loc[df[INCOME].gt(0)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[INCOME, SPEND]).copy()
    stats["Baris akhir"] = len(df)
    stats["Baris dikeluarkan"] = stats["Baris awal"] - stats["Baris akhir"]
    return df, stats


@st.cache_data(show_spinner=False)
def run_model(df, n_clusters):
    X = df[[INCOME, SPEND]].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=11, n_init=10)
    labels = model.fit_predict(X_scaled)
    result = df.copy()
    result["Cluster"] = labels
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=[INCOME, SPEND])
    centers["Cluster"] = np.arange(n_clusters)
    sil = float(silhouette_score(X_scaled, labels)) if 1 < len(np.unique(labels)) < len(X_scaled) else np.nan
    dbi = float(davies_bouldin_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else np.nan
    return result, centers, model, scaler, sil, dbi


@st.cache_data(show_spinner=False)
def evaluate(df, max_k):
    X = StandardScaler().fit_transform(df[[INCOME, SPEND]])
    metrics = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=11, n_init=10)
        labels = km.fit_predict(X)
        metrics.append({"K": k, "Inertia": float(km.inertia_), "Silhouette": float(silhouette_score(X, labels)), "Davies-Bouldin": float(davies_bouldin_score(X, labels))})
    return pd.DataFrame(metrics)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


st.title("🛍️ Segmentasi Pelanggan Mall X")
st.caption("Analisis Data • Preprocessing • K-Means • Evaluasi Model • Segmentasi Pelanggan")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    file = st.file_uploader("Unggah dataset CSV (opsional)", type=["csv"], help="Tanpa unggahan, aplikasi memakai dataset Mall X yang sudah disertakan.")
    raw = validate(load_uploaded(file.getvalue()) if file else load_default())
    st.caption(f"Sumber: {'CSV yang diunggah' if file else 'Customer_mall.csv (bawaan)'}")
    mode = st.radio("Mode pembersihan data", ["Dasar (2 fitur)", "Aturan preprocessing materi PDF"], help="Mode PDF menerapkan aturan usia/pengalaman/profesi sebelum clustering. Kedua mode menggunakan pendapatan dan skor belanja sebagai fitur model.")
    advanced = mode.startswith("Aturan")
    cleaned, prep_stats = prepare(raw, advanced)
    if len(cleaned) < 3 or cleaned[[INCOME, SPEND]].drop_duplicates().shape[0] < 2:
        st.error("Data valid atau variasi fitur tidak cukup untuk K-Means.")
        st.stop()
    kmax = min(10, len(cleaned)-1, cleaned[[INCOME,SPEND]].drop_duplicates().shape[0])
    if kmax < 2:
        st.error("Minimal dua titik unik dibutuhkan.")
        st.stop()
    k = st.slider("Jumlah klaster (K)", 2, kmax, min(4, kmax))
    st.markdown("---")
    st.caption("Catatan: label klaster (0, 1, dst.) adalah identitas algoritma dan dapat berganti ketika K atau data berubah.")

with st.spinner("Menghitung model segmentasi..."):
    result, centers, model, scaler, sil, dbi = run_model(cleaned, k)

# Label bisnis berdasarkan posisi relatif terhadap median seluruh sampel, tidak mengasumsikan label 0..3 tetap.
income_cut = float(result[INCOME].median())
spend_cut = float(result[SPEND].median())
def business_label(row):
    high_i = row[INCOME] >= income_cut
    high_s = row[SPEND] >= spend_cut
    if high_i and high_s: return "VIP (pendapatan & belanja tinggi)"
    if high_i and not high_s: return "Potensial (pendapatan tinggi, belanja rendah)"
    if not high_i and high_s: return "Aktif (pendapatan rendah, belanja tinggi)"
    return "Hemat (pendapatan & belanja rendah)"

centers["Profil indikatif"] = centers.apply(business_label, axis=1)
label_map = dict(zip(centers["Cluster"], centers["Profil indikatif"]))
result["Profil indikatif"] = result["Cluster"].map(label_map)
result["Cluster Label"] = result["Cluster"].map(lambda x: f"Klaster {x}")

pages = st.tabs(["📊 Dashboard", "🔎 Eksplorasi Data", "🧹 Preprocessing", "🧠 Model K-Means", "📈 Evaluasi & Profil", "🎯 Prediksi Pelanggan"])

with pages[0]:
    a,b,c,d = st.columns(4)
    a.metric("Total Pelanggan", f"{len(result):,}")
    b.metric("Rata-rata Usia", f"{result['Age'].mean():.1f} tahun")
    c.metric("Pendapatan Rata-rata", f"$ {result[INCOME].mean():,.0f}")
    d.metric("Skor Belanja Rata-rata", f"{result[SPEND].mean():.1f} / 100")
    left,right = st.columns([1.6,1])
    with left:
        st.subheader("Peta Segmentasi Pelanggan")
        fig=px.scatter(result, x=SPEND,y=INCOME,color="Cluster Label",color_discrete_sequence=PALETTE,hover_data=["CustomerID", "Gender", "Age", "Profession"],opacity=.68)
        fig.update_layout(height=450,legend_title_text="Klaster",margin=dict(t=15,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Komposisi Klaster")
        counts=result["Cluster Label"].value_counts().sort_index().rename_axis("Klaster").reset_index(name="Jumlah")
        st.plotly_chart(px.pie(counts,names="Klaster",values="Jumlah",hole=.58,color_discrete_sequence=PALETTE),use_container_width=True)
    st.info("Interpretasi: posisi kanan menunjukkan skor belanja lebih tinggi, posisi atas menunjukkan pendapatan tahunan lebih tinggi. Warna menunjukkan hasil K-Means.")

with pages[1]:
    st.subheader("Data Pelanggan")
    st.dataframe(raw, use_container_width=True,hide_index=True,height=320)
    st.subheader("Kualitas Data")
    n1,n2,n3=st.columns(3)
    n1.metric("Jumlah baris",len(raw))
    n2.metric("Nilai kosong",int(raw.isna().sum().sum()))
    n3.metric("Baris duplikat penuh",int(raw.duplicated().sum()))
    with st.expander("Nilai kosong per kolom"):
        st.dataframe(raw.isna().sum().rename("Jumlah kosong"),use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(raw,x="Gender",title="Pelanggan menurut Gender",color="Gender"),use_container_width=True)
        st.plotly_chart(px.histogram(raw,x="Profession",title="Profesi Pelanggan"),use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(raw,x="Age",nbins=22,title="Distribusi Usia"),use_container_width=True)
        st.plotly_chart(px.scatter(raw,x="Age",y=SPEND,title="Usia vs Skor Belanja",opacity=.5),use_container_width=True)
    c3,c4=st.columns(2)
    with c3:st.plotly_chart(px.histogram(raw,x="Work_Experience",title="Pengalaman Kerja"),use_container_width=True)
    with c4:st.plotly_chart(px.histogram(raw,x="Family_Size",title="Ukuran Keluarga"),use_container_width=True)
    with st.expander("Statistik deskriptif lengkap"):
        st.dataframe(raw.describe(include="all").T,use_container_width=True)

with pages[2]:
    st.subheader("Tahapan Preprocessing")
    st.write("**Mode aktif:**",mode)
    st.write("**Fitur K-Means:** Annual_Income($) dan Spending Score(1-100). CustomerID digunakan hanya sebagai identitas.")
    if advanced:
        st.markdown("""Mengikuti panduan slide 15: isi profesi yang kosong sebagai 'Tidak diketahui'; usia nol diisi median sesuai profesi; buang usia di luar 17–80 kecuali Artist atau Entertainment; buang pengalaman kerja melebihi usia; pengalaman kerja nol diisi median profesi; buang pendapatan nol. Kode menjalankan aturan tersebut dengan penyederhanaan yang dinyatakan ini, lalu melakukan standardisasi untuk dua fitur model.""")
    else:
        st.markdown("Mode dasar hanya mengeluarkan baris dengan pendapatan atau skor belanja tidak valid dan mengganti profesi kosong dengan 'Tidak diketahui'. Usia dan pengalaman kerja tidak digunakan oleh model K-Means.")
    a,b,c=st.columns(3)
    a.metric("Baris awal",prep_stats["Baris awal"])
    b.metric("Baris dikeluarkan",prep_stats["Baris dikeluarkan"])
    c.metric("Baris siap dianalisis",prep_stats["Baris akhir"])
    st.dataframe(cleaned.head(30),use_container_width=True,hide_index=True)
    st.download_button("Unduh data bersih (CSV)",csv_bytes(cleaned),"data_pelanggan_bersih.csv","text/csv")

with pages[3]:
    st.subheader("Pemodelan K-Means")
    st.markdown(f"Jumlah klaster ditetapkan **K = {k}**. Data dua fitur distandardisasi memakai StandardScaler sebelum dilatih dengan KMeans (`random_state=11`, `n_init=10`).")
    a,b,c=st.columns(3)
    a.metric("K",k)
    b.metric("Silhouette",f"{sil:.4f}")
    c.metric("Davies–Bouldin",f"{dbi:.4f}")
    st.plotly_chart(px.scatter(result,x=SPEND,y=INCOME,color="Cluster Label",color_discrete_sequence=PALETTE,hover_data=["CustomerID","Age","Profession"],title="Hasil Pengelompokan Pelanggan"),use_container_width=True)
    st.markdown("**Pusat klaster dalam satuan asli**")
    st.dataframe(centers,use_container_width=True,hide_index=True)
    st.download_button("Unduh hasil klaster (CSV)",csv_bytes(result.drop(columns=["Cluster Label"])),"hasil_segmentasi_pelanggan.csv","text/csv")

with pages[4]:
    st.subheader("Evaluasi Pilihan Jumlah Klaster")
    st.caption("Inertia dan Davies–Bouldin lebih rendah lebih baik; Silhouette lebih tinggi lebih baik. Elbow dinilai dari perubahan kemiringan kurva, bukan otomatis nilai paling kecil.")
    with st.spinner("Menghitung evaluasi K = 2 sampai 10..."):
        scores=evaluate(cleaned,kmax)
    s1,s2,s3=st.columns(3)
    with s1:st.plotly_chart(px.line(scores,x="K",y="Inertia",markers=True,title="Elbow / Inertia"),use_container_width=True)
    with s2:st.plotly_chart(px.line(scores,x="K",y="Silhouette",markers=True,title="Silhouette Score"),use_container_width=True)
    with s3:st.plotly_chart(px.line(scores,x="K",y="Davies-Bouldin",markers=True,title="Davies–Bouldin Index"),use_container_width=True)
    st.dataframe(scores.style.format({"Inertia":"{:.2f}","Silhouette":"{:.4f}","Davies-Bouldin":"{:.4f}"}),use_container_width=True,hide_index=True)
    best_s=int(scores.loc[scores["Silhouette"].idxmax(),"K"])
    best_db=int(scores.loc[scores["Davies-Bouldin"].idxmin(),"K"])
    st.info(f"Pada data dan mode ini, K terbaik berdasarkan Silhouette = {best_s}, sedangkan berdasarkan Davies–Bouldin = {best_db}. K = {k} adalah pilihan yang sedang ditampilkan. Tidak ada jaminan kedua metrik selalu sepakat.")
    st.subheader("Ringkasan Karakteristik Tiap Klaster")
    summary=result.groupby("Cluster").agg(Jumlah=("CustomerID","size"),Pendapatan_Rata_Rata=(INCOME,"mean"),Skor_Belanja_Rata_Rata=(SPEND,"mean"),Usia_Rata_Rata=("Age","mean")).reset_index()
    summary["Persentase (%)"]=(100*summary["Jumlah"]/len(result)).round(2)
    summary["Profil indikatif"] = summary["Cluster"].map(label_map)
    st.dataframe(summary.style.format({"Pendapatan_Rata_Rata":"${:,.0f}","Skor_Belanja_Rata_Rata":"{:.1f}","Usia_Rata_Rata":"{:.1f}"}),use_container_width=True,hide_index=True)
    st.warning("Profil bisnis merupakan interpretasi indikatif atas pusat klaster terhadap median data, bukan label baku atau kategori asli dari dataset. K-Means tidak mengukur profitabilitas atau kepuasan pelanggan secara langsung.")
    st.download_button("Unduh ringkasan klaster",csv_bytes(summary),"ringkasan_klaster.csv","text/csv")

with pages[5]:
    st.subheader("Prediksi Segmen Pelanggan Baru")
    st.write("Masukkan pendapatan tahunan (USD) dan skor belanja. Prediksi mengikuti model K-Means yang aktif, tanpa pelatihan ulang.")
    with st.form("form_prediksi"):
        a,b=st.columns(2)
        with a:income=st.number_input("Pendapatan tahunan ($)",min_value=0.0,value=80000.0,step=1000.0)
        with b:score=st.number_input("Spending Score (0–100)",min_value=0.0,max_value=100.0,value=65.0,step=1.0)
        submitted=st.form_submit_button("Prediksi Klaster",type="primary")
    if submitted:
        x=pd.DataFrame([{INCOME:income,SPEND:score}])
        pred=int(model.predict(scaler.transform(x))[0])
        st.success(f"Pelanggan diprediksi masuk Klaster {pred} — {label_map[pred]}")
        st.caption("Hasil bergantung pada dataset, preprocessing, dan jumlah K yang sedang dipilih; ini bukan ukuran nilai ekonomi pasti pelanggan.")
    st.markdown("**Contoh penggunaan:** Tim pemasaran dapat membandingkan pola skor belanja dan pendapatan sebagai masukan untuk merancang promosi yang lebih relevan.")

st.divider()
st.caption("Sumber: Customer_mall.csv dan materi 'Pemodelan Data Pelanggan Mall Dengan Metode Kmeans' (Adi Surya Basri). Aplikasi untuk pembelajaran dan analisis eksploratif.")
