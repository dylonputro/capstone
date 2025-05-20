import pandas as pd
from sklearn.cluster import KMeans

def fix_column_name(df, column_mapping):
    """
    Rename columns in df according to column_mapping dictionary.
    """
    return df.rename(columns=column_mapping)

def clean_data(df):
    """
    Bersihkan data: convert tanggal, handle missing, sorting, dan reset index.
    """
    # Ubah kolom tanggal ke datetime
    df['Tanggal & Waktu'] = pd.to_datetime(df['Tanggal & Waktu'], errors='coerce')
    # Drop row yang tanggalnya invalid atau missing
    df = df.dropna(subset=['Tanggal & Waktu'])
    # Isi missing numeric dengan 0
    numeric_cols = ['Jumlah Produk', 'Harga Produk', 'nominal_transaksi']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Buat kolom nominal_transaksi jika belum ada
    if 'nominal_transaksi' not in df.columns and ('Jumlah Produk' in df.columns and 'Harga Produk' in df.columns):
        df['nominal_transaksi'] = df['Jumlah Produk'] * df['Harga Produk']

    # Urutkan berdasarkan tanggal & waktu
    df = df.sort_values('Tanggal & Waktu').reset_index(drop=True)
    return df

def prep_sales(df):
    """
    Mengelompokkan data per hari dengan agregasi pemasukan, produk, transaksi dll
    """
    df_daily = df.copy()
    df_daily['Tanggal & Waktu'] = pd.to_datetime(df_daily['Tanggal & Waktu']).dt.date

    agg_df = df_daily.groupby('Tanggal & Waktu').agg(
        nominal_transaksi=('nominal_transaksi', 'sum'),
        banyak_produk=('Jumlah Produk', 'sum'),
        banyak_transaksi=('ID Struk', 'nunique'),
        banyak_jenis_produk=('Nama Produk', 'nunique')
    ).reset_index()

    return agg_df

def prep_customer(df):
    """
    Grouping data customer dengan beberapa ringkasan
    """
    df_customer = df.groupby('Nama Pelanggan').agg(
        totSpen=('nominal_transaksi', 'sum'),
        totJum=('Jumlah Produk', 'sum'),
        totJenPro=('Nama Produk', 'nunique'),
        totKat=('Kategori', 'nunique'),
        banyak_transaksi=('ID Struk', 'nunique')
    ).reset_index()
    return df_customer

def prep_grouphour(df):
    """
    Membuat agregasi rata-rata produk per jam
    """
    df_hour = df.copy()
    df_hour['Jam'] = pd.to_datetime(df_hour['Tanggal & Waktu']).dt.hour
    df_grouped = df_hour.groupby('Jam').agg(Jumlah_produk=('Jumlah Produk', 'mean')).reset_index()
    return df_grouped

def prep_groupProduct(df):
    """
    Total produk per nama produk
    """
    df_group = df.groupby('Nama Produk').agg(Jumlah_produk=('Jumlah Produk', 'sum')).reset_index()
    return df_group

def prep_groupKategori(df):
    """
    Total omzet per kategori produk
    """
    df_group = df.groupby('Kategori').agg(Total_omset=('nominal_transaksi', 'sum')).reset_index()
    return df_group

def customer_segmentation(df_customer):
    """
    Segmentasi pelanggan menggunakan KMeans (3 cluster)
    """
    features = ['totSpen', 'totJum', 'totJenPro', 'totKat']
    df_customer_clean = df_customer.dropna(subset=features).copy()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_customer_clean['cluster'] = kmeans.fit_predict(df_customer_clean[features])
    return df_customer_clean
