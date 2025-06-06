# Langkah 4: Buat file app.py
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Temperature Clustering Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# Fungsi untuk membersihkan outlier
def clear_outlier(df, kolom_numerik, metode='iqr', threshold=1.5):
    Q1 = df[kolom_numerik].quantile(0.25)
    Q3 = df[kolom_numerik].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - threshold * IQR
    batas_atas = Q3 + threshold * IQR
    return df[(df[kolom_numerik] >= batas_bawah) & (df[kolom_numerik] <= batas_atas)]

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    df = pd.read_csv('IOT-temp.csv')  # Dataset diakses secara lokal
    # Membersihkan outlier
    df = clear_outlier(df, 'temp')
    df.dropna(inplace=True)
    return df

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    return joblib.load('dbscan_model.pkl')

# Header
st.title("Temperature Clustering Dashboard (DBSCAN)")

# Overview Dashboard
st.markdown("""
## Selamat Datang di Dashboard Clustering Suhu

Dashboard ini menggunakan model DBSCAN untuk mengelompokkan data suhu berdasarkan kadar suhu, lokasi (In/Out), dan waktu.

### Dataset:
Dataset berisi informasi tentang suhu (`temp`), lokasi pengukuran (`out/in`), dan waktu (`noted_date`) dari sensor IoT pada periode Juli hingga Desember 2018.
""")

# Memuat data
try:
    df = load_data()
except FileNotFoundError:
    st.error("File IOT-temp.csv tidak ditemukan. Pastikan file ada di direktori yang sama.")
    st.stop()

# Statistik dasar
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Data", df.shape[0])
with col2:
    st.metric("Pengukuran Dalam Ruangan", df[df['out/in'] == 'In'].shape[0])
with col3:
    st.metric("Pengukuran Luar Ruangan", df[df['out/in'] == 'Out'].shape[0])

# Preprocessing
df = df.drop_duplicates()
df['noted_date'] = pd.to_datetime(df['noted_date'], format='%d-%m-%Y %H:%M', errors='coerce')
df['hour'] = df['noted_date'].dt.hour
df = df.drop(columns=['id', 'room_id/id'])
df['temp'] = df['temp'].fillna(df['temp'].mean())
df['out/in'] = df['out/in'].map({'In': 0, 'Out': 1}).fillna(0)
df['hour'] = df['hour'].fillna(df['hour'].mode()[0])
scaler = StandardScaler()
df['temp_scaled'] = scaler.fit_transform(df[['temp']])

# Memuat model
model = load_model()

# Clustering menggunakan model yang dimuat
features = df[['temp_scaled', 'out/in', 'hour']]
df['cluster'] = model.fit_predict(features)

# Distribusi cluster
st.subheader("Distribusi Cluster")
fig_pie = px.pie(
    df,
    names='cluster',
    title='Distribusi Cluster DBSCAN',
    color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'],
    labels={'cluster': 'Cluster'}
)
st.plotly_chart(fig_pie)

# Visualisasi clustering
st.markdown("---")
st.subheader("Hasil Clustering Suhu")

fig = px.scatter(
    df,
    x='noted_date',
    y='temp',
    color='cluster',
    title='Temperature Clustering (DBSCAN)',
    labels={'cluster': 'Cluster'},
    color_continuous_scale='Viridis'
)
fig.update_xaxes(
    tickformat="%b %d, %Y",
    tickangle=45,
    dtick=14*24*60*60*1000  # Interval 14 hari
)
st.plotly_chart(fig, use_container_width=True)

# Contoh data per cluster
st.subheader("Contoh Data per Cluster")
for cluster_label in sorted(df['cluster'].unique()):
    st.write(f"**Cluster {cluster_label}:**")
    st.write(df[df['cluster'] == cluster_label][['noted_date', 'temp', 'out/in']].head(3))

# Footer
st.markdown("---")

# Langkah 5: Jalankan Streamlit dan Ngrok
import subprocess
import time
from pyngrok import ngrok

# Jalankan Streamlit
process = subprocess.Popen("streamlit run app.py --server.port 8501", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Tunggu beberapa detik agar Streamlit berjalan
time.sleep(5)

# Cek log Streamlit
stdout, stderr = process.communicate(timeout=30)
print("Streamlit Log (stdout):")
print(stdout.decode())
print("Streamlit Log (stderr):")
print(stderr.decode())

# Buat URL publik dengan Ngrok
public_url = ngrok.connect(addr='8501')
print(f"Public URL: {public_url}")