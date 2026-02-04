import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

# ====== TAMBAHAN UNTUK GOOGLE SHEETS ======
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

st.set_page_config(page_title="MIKOM Dashboard", layout="wide")
st.title("📊 Dashboard Mini Kompetisi (MIKOM)")


# ================= GOOGLE SHEETS =================
def connect_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "credentials.json", scope
    )
    client = gspread.authorize(creds)

    spreadsheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/1x7h1G3f56cQVm_UJU7Gc9s-KlenA58DPZP1-rxqd5ow/edit?usp=sharing"
    )

    sheet = spreadsheet.worksheet("PREDIKSI")  # <<< PENTING

    return sheet



def save_to_sheet(pagu, kategori, jumlah_penawar, jumlah_barang, persentase, hasil_rp):
    try:
        sheet = connect_sheet()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([
            now,
            pagu,
            kategori,
            jumlah_penawar,
            jumlah_barang,
            f"{persentase*100:.2f}%",
            int(hasil_rp)
        ])
    except Exception as e:
        st.error(f"Gagal menyimpan ke Google Sheets: {e}")

# ============ LOAD DATA ============
@st.cache_data
def load_mikom_data():
    """Load data dari Excel MIKOM"""
    file_path = "Data Mini Kompetisi FIX.xlsx"
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_excel(file_path, sheet_name="Table Mikom", engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

@st.cache_resource
def train_model(df):
    try:
        df_model = df.dropna(subset=['Penawaran Pemenang', 'Penawaran SAGA']).copy()
        
        df_model['Pagu'] = df_model['Pagu'] * 1_000_000
        df_model['Penawaran SAGA'] = df_model['Penawaran SAGA'] * 1_000_000
        
        le = LabelEncoder()
        df_model['Kategori_enc'] = le.fit_transform(df_model['Kategori'])
        
        # ===== SIMPAN NILAI DARI TRAINING =====
        mean_penawar = df_model['Jumlah Penawar'].mean()
        pagu_bins = pd.cut(df_model['Pagu'], bins=5, retbins=True)[1]
        
        # ===== FEATURE ENGINEERING TRAINING =====
        df_model['rasio_penawar'] = df_model['Jumlah Penawar'] / mean_penawar
        df_model['log_pagu'] = np.log1p(df_model['Pagu'])
        df_model['pagu_per_penawar'] = df_model['Pagu'] / (df_model['Jumlah Penawar'] + 1)
        df_model['pagu_kategori'] = pd.cut(df_model['Pagu'], bins=pagu_bins, labels=False)
        
        X = df_model[[
            'Pagu', 'Jumlah Penawar', 'Kategori_enc', 'rasio_penawar',
            'log_pagu', 'pagu_per_penawar', 'pagu_kategori', 'Jumlah Barang'
        ]]
        
        df_model['persentase_saga'] = df_model['Penawaran SAGA'] / df_model['Pagu']
        y = df_model['persentase_saga']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=1000, random_state=42)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, pred),
            'r2': r2_score(y_test, pred),
            'mape': np.mean(np.abs((y_test - pred) / y_test)) * 100
        }
        
        # ===== RETURN SEMUA YANG DIPERLUKAN SAAT PREDIKSI =====
        return model, le, X.columns, metrics, mean_penawar, pagu_bins
    
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None, None, None


# Load data utama
df = load_mikom_data()

if df is None:
    st.error("❌ File 'Data Mini Kompetisi FIX.xlsx' tidak ditemukan di folder ini!")
    st.info("Pastikan file berada di folder yang sama dengan streamlit_app.py")
else:
    df_clean = df.dropna(subset=["Kategori", "Status SAGA"]).copy()
    df_clean["Status SAGA"] = df_clean["Status SAGA"].str.upper().str.strip()
    df_clean["Status SAGA Group"] = df_clean["Status SAGA"].apply(
        lambda x: x if x in ["MENANG", "PENAWARAN TERLEWAT"] else "KALAH"
    )
    
    # Sidebar Navigation
    tab = st.sidebar.radio(
        "Pilih Tampilan",
        ["📈 Analisis Kategori", "🎯 Status SAGA", "👤 Per PIC", "🤖 Prediksi Model", "📊 Data Explorer"]
    )
    
    # ============ TAB 1: ANALISIS KATEGORI ============
    if tab == "📈 Analisis Kategori":
        st.subheader("Distribusi Data Berdasarkan Kategori")
        
        col1, col2 = st.columns(2)
        
        with col1:
            kategori_counts = df_clean["Kategori"].value_counts()
            fig_kat = px.pie(values=kategori_counts.values, names=kategori_counts.index, 
                            title="Distribusi Kategori")
            st.plotly_chart(fig_kat, use_container_width=True)
        
        with col2:
            st.dataframe(kategori_counts, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Persentase Status per Kategori")
        daftar_kategori = df_clean["Kategori"].unique()
        
        cols_display = st.columns(2)
        for idx, kategori in enumerate(daftar_kategori):
            df_kat = df_clean[df_clean["Kategori"] == kategori]
            status_counts = df_kat["Status SAGA"].value_counts(normalize=True) * 100
            
            fig_status = px.pie(values=status_counts.values, names=status_counts.index,
                               title=f"Status SAGA: {kategori}")
            cols_display[idx % 2].plotly_chart(fig_status, use_container_width=True)
    
    # ============ TAB 2: STATUS SAGA ============
    elif tab == "🎯 Status SAGA":
        st.subheader("Distribusi Status SAGA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = df_clean["Status SAGA Group"].value_counts()
            fig_status_group = px.pie(values=status_counts.values, names=status_counts.index,
                                      title="Status SAGA: MENANG / TERLEWAT / KALAH")
            st.plotly_chart(fig_status_group, use_container_width=True)
        
        with col2:
            st.dataframe(status_counts.rename("Jumlah"), use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Rincian Status KALAH")
        status_kalah_detail = df_clean[~df_clean["Status SAGA"].isin(["MENANG", "PENAWARAN TERLEWAT"])]
        kalah_counts = status_kalah_detail["Status SAGA"].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_kalah = px.pie(values=kalah_counts.values, names=kalah_counts.index,
                              title="Rincian Penyebab KALAH")
            st.plotly_chart(fig_kalah, use_container_width=True)
        
        with col2:
            st.dataframe(kalah_counts.rename("Jumlah"), use_container_width=True)
    
    # ============ TAB 3: PER PIC ============
    elif tab == "👤 Per PIC":
        st.subheader("Analisis Status per PIC (Perusahaan)")
        
        # Status SAGA per PIC (grouped)
        pivot_pic = pd.pivot_table(df_clean, index="PIC", columns="Status SAGA Group", aggfunc="size", fill_value=0)
        pivot_pic = pivot_pic.reindex(columns=["MENANG", "PENAWARAN TERLEWAT", "KALAH"], fill_value=0)
        
        fig_pic_grouped = px.bar(pivot_pic, title="Status SAGA per PIC (Grouped)")
        st.plotly_chart(fig_pic_grouped, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Top 5 Penyebab KALAH per PIC")
        df_kalah = df_clean[~df_clean["Status SAGA"].isin(["MENANG", "PENAWARAN TERLEWAT"])]
        pivot_kalah_pic = pd.pivot_table(df_kalah, index="PIC", columns="Status SAGA", aggfunc="size", fill_value=0)
        
        if not pivot_kalah_pic.empty:
            top5_penyebab = pivot_kalah_pic.sum().sort_values(ascending=False).head(5).index
            pivot_top5 = pivot_kalah_pic[top5_penyebab]
            
            fig_top5 = px.bar(pivot_top5, title="Top 5 Penyebab KALAH per PIC", barmode="stack")
            st.plotly_chart(fig_top5, use_container_width=True)
    
    # ============ TAB 4: PREDIKSI MODEL ============
    elif tab == "🤖 Prediksi Model":
        st.subheader("Model Prediksi Penawaran SAGA")

        def format_rupiah(x):
            return f"Rp {x:,.0f}".replace(",", ".")

        with st.spinner("Loading model..."):
            model, le, X_cols, metrics, mean_penawar, pagu_bins = train_model(df)

        if model is not None:
            # ===== METRICS =====
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['mae']:.4f}")
            col2.metric("R² Score", f"{metrics['r2']:.4f}")
            col3.metric("MAPE", f"{metrics['mape']:.2f}%")

            st.markdown("---")

            # ===== FEATURE IMPORTANCE =====
            importance_df = pd.DataFrame({
                'Fitur': X_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Fitur',
                orientation='h',
                title="Importance Fitur Model"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

            st.markdown("---")

            # ===== INPUT DATA BARU =====
            st.subheader("Prediksi Penawaran untuk Data Baru")

            col1, col2 = st.columns(2)
            with col1:
                pagu_rp = st.number_input("Pagu (Rp)", min_value=1)

            with col2:
                kategori = st.selectbox("Kategori", le.classes_)

            col1, col2 = st.columns(2)
            with col1:
                jumlah_penawar = st.number_input("Jumlah Penawar", min_value=1, value=9, step=1)

            with col2:
                jumlah_barang = st.number_input("Jumlah Barang", min_value=1, value=8, step=1)

            if st.button("🔮 Prediksi"):
                data_baru = pd.DataFrame({
                    'Pagu': [pagu_rp],
                    'Jumlah Penawar': [jumlah_penawar],
                    'Kategori': [kategori],
                    'Jumlah Barang': [jumlah_barang]
                })

                # ===== FEATURE ENGINEERING IDENTIK DENGAN TRAINING =====
                data_baru['Kategori_enc'] = le.transform(data_baru['Kategori'])
                data_baru['rasio_penawar'] = data_baru['Jumlah Penawar'] / mean_penawar
                data_baru['log_pagu'] = np.log1p(data_baru['Pagu'])
                data_baru['pagu_per_penawar'] = data_baru['Pagu'] / (data_baru['Jumlah Penawar'] + 1)
                data_baru['pagu_kategori'] = pd.cut(
                    data_baru['Pagu'],
                    bins=pagu_bins,
                    labels=False
                )

                data_baru_model = data_baru[X_cols]

                persentase_pred = model.predict(data_baru_model)[0]
                persentase_pred = np.clip(persentase_pred, 0, 0.99)
                penawaran_pred = persentase_pred * pagu_rp

                save_to_sheet(
                    pagu_rp,
                    kategori,
                    jumlah_penawar,
                    jumlah_barang,
                    persentase_pred,
                    penawaran_pred
                )

                st.success("✅ Prediksi berhasil dan tersimpan ke Google Sheets!")

                penawaran_juta = penawaran_pred / 1_000_000

                col1, col2 = st.columns(2)
                col1.metric("Persentase Penawaran", f"{persentase_pred*100:.2f}%")
                col2.metric(
                    "Estimasi Penawaran SAGA",
                    format_rupiah(penawaran_pred)
)

    
    # ============ TAB 5: DATA EXPLORER ============
    elif tab == "📊 Data Explorer":
        st.subheader("Data Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Baris", df.shape[0])
            st.metric("Total Kolom", df.shape[1])
        
        with col2:
            st.metric("Baris Bersih (setelah cleaning)", df_clean.shape[0])
            st.metric("Data Hilang", df.isnull().sum().sum())
        
        st.markdown("---")
        
        if st.checkbox("Tampilkan seluruh dataset"):
            st.dataframe(df_clean, use_container_width=True, height=400)
        else:
            st.dataframe(df_clean.head(50), use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Statistik Deskriptif")
        numeric_cols = df_clean.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            st.dataframe(df_clean[numeric_cols].describe(), use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Unduh Data")
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name="mikom_data_clean.csv", mime='text/csv')
