import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM


# fix the name of data
def fix_column_name(df, names):
    df = df.rename(columns={v: k for k, v in names.items()})
    return df


def clean_data(df):
    df["Jumlah Produk"] = pd.to_numeric(df["Jumlah Produk"], errors="coerce")
    df = df[df["Jumlah Produk"] >= 0]
    df["Harga Produk"] = df["Harga Produk"].astype(str).str.replace(",", "", regex=True)
    df["Harga Produk"] = df["Harga Produk"].str.replace(r"[^0-9.]", "", regex=True)
    df["Harga Produk"] = pd.to_numeric(df["Harga Produk"], errors="coerce")
    df["Tanggal & Waktu"] = df["Tanggal & Waktu"].str.replace(
        r"(\d{2})\.(\d{2})", r"\1:\2", regex=True
    )
    df["Tanggal & Waktu"] = pd.to_datetime(
        df["Tanggal & Waktu"], errors="coerce", dayfirst=False
    )
    df["Total_harga"] = df["Harga Produk"] * df["Jumlah Produk"]
    df["Jam"] = df["Tanggal & Waktu"].dt.hour
    return df


def prep_sales(df):
    dff = df.copy()
    dff["Tanggal & Waktu"] = dff["Tanggal & Waktu"].dt.date
    df_grouped = (
        dff.groupby("Tanggal & Waktu", as_index=False)
        .agg(
            banyak_transaksi=("ID Struk", "nunique"),
            banyak_produk=("Jumlah Produk", "sum"),
            banyak_jenis_produk=("Nama Produk", "nunique"),
            nominal_transaksi=("Total_harga", "sum"),
        )
        .reset_index()
    )
    return df_grouped


def prep_customer(df):
    df_grouped = (
        df.groupby("ID Struk")  # Pastikan ini adalah customer ID, bukan ID transaksi
        .agg(
            totSpen=("Total_harga", "sum"),
            totJum=("Jumlah Produk", "sum"),
            totJenPro=("Nama Produk", "nunique"),
            totKat=("Kategori", "nunique"),
        )
        .reset_index()
    )
    return df_grouped


def prep_grouphour(df):
    df_grouped = (
        df.groupby("Jam").agg(Jumlah_produk=("Jumlah Produk", "mean")).reset_index()
    )
    return df_grouped


def prep_groupProduct(df):
    df_grouped = (
        df.groupby("Nama Produk")
        .agg(
            Jumlah_produk=("Jumlah Produk", "sum"),
            Total_omset=("Total_harga", "sum"),
            Harga_Satuan=("Harga Produk", "first"),
        )
        .reset_index()
    )
    return df_grouped


def prep_groupKategori(df):
    df_grouped = (
        df.groupby("Kategori")
        .agg(
            Jumlah_produk=("Jumlah Produk", "sum"),
            Total_omset=("Total_harga", "sum"),
            Harga_Satuan=("Harga Produk", "first"),
        )
        .reset_index()
    )
    return df_grouped


def customer_segmentation(df):
    # Pastikan kolom yang diperlukan ada
    required_columns = ["totSpen", "totJum", "totJenPro", "totKat"]
    df = df[required_columns]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Gunakan model yang sudah di-save
    model1 = joblib.load("Segmentasi_pembeli1.pkl")
    model2 = joblib.load("Segmentasi_pembeli22.pkl")

    labels = model1.fit_predict(df_scaled)
    df["cluster"] = labels

    # Handle outliers dengan DBSCAN
    mask = labels == -1
    X_outlier = df_scaled[mask]
    label = model2.predict(X_outlier)
    label += df["cluster"].max() + 1
    df.loc[mask, "cluster"] = label

    return df


def predict_revenue(df, periods=30):
    # Persiapan data untuk Prophet
    model_df = df[["Tanggal & Waktu", "nominal_transaksi"]].rename(
        columns={"Tanggal & Waktu": "ds", "nominal_transaksi": "y"}
    )

    # Inisialisasi model
    model = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    # Training model
    model.fit(model_df)

    # Membuat dataframe prediksi
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Evaluasi model (opsional)
    mae = mean_absolute_error(model_df["y"], forecast["yhat"][:-periods])
    print(f"MAE: {mae:.2f}")

    return forecast, model


def load_hf_model():
    model_name = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    hf_pipeline = HuggingFacePipeline(
        pipeline=pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            return_text=True,  # Pastikan output berupa text
        )
    )
    return hf_pipeline
