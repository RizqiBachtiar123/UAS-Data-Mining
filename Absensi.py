import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit
st.write("""# K-Means Clustering untuk Data Absensi""")

# Memuat data
def load_data():
    df = pd.read_csv('Absensi.csv')
    return df

# Pra-pemrosesan data
def preprocess_data(df):
    # Hapus baris duplikat
    df = df.drop_duplicates()

    # Hapus baris dengan nilai yang hilang
    df = df.dropna()

    # Pilih kolom-kolom yang relevan untuk clustering
    kolom_untuk_clustering = ['Check_In', 'Check_Out', 'Date*', 'Employee_ID*']

    # Ubah kolom waktu menjadi objek datetime
    for kolom in ['Check_In', 'Check_Out', 'Date*']:
        df[kolom] = pd.to_datetime(df[kolom], errors='coerce')

    # Buat DataFrame baru dengan selisih waktu
    df['Check_In_Out_Diff'] = (df['Check_Out'] - df['Check_In']).dt.total_seconds() / 3600

    return df

# K-Means clustering
def kmeans_clustering(df, optimal_k):
    # Pilih fitur untuk clustering
    X = df[['Check_In_Out_Diff']]

    # Standarisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Terapkan clustering K-Means dengan k yang dipilih
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    return df

# Visualisasi cluster
def visualize_clusters(df):
    # Visualisasikan cluster
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='Check_In_Out_Diff', y='Employee_ID*', hue='Cluster', data=df, palette='viridis', s=100)
    plt.title('K-Means Clustering Data Absensi Karyawan')
    plt.xlabel('Waktu Check-In ke Check-Out (jam)')
    plt.ylabel('ID Karyawan')
    st.pyplot(fig)

# Tampilkan tabel karyawan berdasarkan clustering
def display_clustered_table(df, optimal_k, cluster_explanations):
    st.subheader(f'Tabel Karyawan Berdasarkan Clustering (Jumlah Cluster = {optimal_k})')

    # Hapus duplikat ID Karyawan
    df_unique_employee = df.drop_duplicates(subset=['Employee_ID*'])

    # Urutkan DataFrame berdasarkan 'Employee_ID*' dan 'Cluster'
    df_unique_employee = df_unique_employee.sort_values(by=['Employee_ID*', 'Cluster'])

    # Reset indeks untuk mengurutkan nomor baris
    df_unique_employee = df_unique_employee.reset_index(drop=True)

    # Hapus kolom yang tidak perlu ditampilkan
    df_display = df_unique_employee[['Employee_ID*', 'Cluster']]
    
    # Tambahkan kolom penjelasan cluster pada tabel
    df_display['Cluster Explanation'] = df_unique_employee['Cluster'].map(cluster_explanations)

    st.table(df_display)

# Jelaskan setiap cluster
def explain_clusters(optimal_k):
    st.subheader('Penjelasan Cluster:')
    cluster_explanations = {}

    for cluster in range(optimal_k):
        st.write(f'**Cluster {cluster + 1}:**')
        explanation = st.text_area(f'Masukkan penjelasan tentang karakteristik Cluster {cluster + 1}:')
        cluster_explanations[cluster] = explanation

    return cluster_explanations

def main():
    # Memuat data
    df = load_data()

    # Pra-pemrosesan data
    df = preprocess_data(df)

    # Memungkinkan pengguna untuk memilih jumlah hari untuk clustering
    total_hari = len(df['Date*'].unique())
    hari_terpilih = st.slider('Pilih Jumlah Hari untuk Clustering:', 1, total_hari, 7)

    # Filter data untuk jumlah hari yang dipilih
    df = df.groupby('Employee_ID*').head(hari_terpilih)

    # Menentukan jumlah cluster optimal menggunakan Metode Elbow
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df[['Check_In_Out_Diff']])
        wcss.append(kmeans.inertia_)

    # Plot Metode Elbow
    st.subheader('Metode Elbow untuk Memilih Jumlah Cluster Optimal')
    st.line_chart(pd.DataFrame({'Jumlah Cluster': range(1, 11), 'WCSS': wcss}).set_index('Jumlah Cluster'))

    # Berdasarkan Metode Elbow, pilih jumlah cluster (k) yang optimal
    optimal_k = st.slider('Pilih Jumlah Cluster (k) yang Optimal:', 2, 10, 3)

    # K-Means clustering
    df = kmeans_clustering(df, optimal_k)

    # Visualisasi cluster
    visualize_clusters(df)

    # Jelaskan setiap cluster
    cluster_explanations = explain_clusters(optimal_k)

    # Tampilkan tabel karyawan berdasarkan clustering
    display_clustered_table(df, optimal_k, cluster_explanations)

if __name__ == '__main__':
    main()
