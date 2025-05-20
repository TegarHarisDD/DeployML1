import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------
# 1. Load model dan artefak yang disimpan
# ----------------------------------------
scaler           = joblib.load('models/scaler.pkl')
knn_tuned        = joblib.load('models/knn_tuned.pkl')
svm_tuned        = joblib.load('models/svm_tuned.pkl')
xgb_tuned        = joblib.load('models/xgb_tuned.pkl')
le               = joblib.load('models/label_encoder.pkl')
accuracy_tuned   = joblib.load('models/accuracy_tuned.pkl')
feature_columns  = joblib.load('models/feature_columns.pkl')

with open('dataset_info.md', 'r') as f:
    dataset_info = f.read()

# Fitur numerik untuk normalisasi
num_feats = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','BMI']

# ----------------------------------------
# 1.a Kamus Mapping Label → Kode
# ----------------------------------------
GENDER_MAP = {"Perempuan": 0, "Laki-laki": 1}
CALC_MAP   = {"Tidak Pernah": 0, "Kadang-kadang": 1, "Sering": 2}
FAVC_MAP   = {"Tidak": 0, "Ya": 1}
SCC_MAP    = {"Tidak": 0, "Ya": 1}
SMOKE_MAP  = {"Tidak": 0, "Ya": 1}
FAM_MAP    = {"Tidak": 0, "Ya": 1}
CAEC_MAP   = {"Tidak Pernah": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
MTRANS_MAP = {
    "Berjalan": 0,
    "Transportasi Umum": 1,
    "Mobil": 2,
    "Motor": 3,
    "Sepeda": 4
}

# ----------------------------------------
# 2. Fungsi Preprocessing
# ----------------------------------------
def preprocess_input(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    cat_cols = [
        'Gender','CALC','FAVC','SCC','SMOKE',
        'family_history_with_overweight','CAEC','MTRANS'
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    df[num_feats] = scaler.transform(df[num_feats])
    return df

# ----------------------------------------
# 3. Navigasi Sidebar
# ----------------------------------------
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Tentang Aplikasi", "Prediksi Tunggal", "Prediksi Massal"])

# ----------------------------------------
# 4. Halaman TENTANG
# ----------------------------------------
if page == "Tentang Aplikasi":
    st.title("Aplikasi Klasifikasi Obesitas")
    st.markdown(dataset_info)
    st.subheader("Akurasi Model Setelah Tuning")
    for model_name, acc in accuracy_tuned.items():
        st.write(f"- **{model_name}**: {acc:.4f}")
    st.markdown("""
    **Tujuan Aplikasi:**  
    Mengklasifikasikan tingkat obesitas berdasarkan data pribadi, pola makan, dan kebiasaan hidup  
    menggunakan model KNN, SVM, dan XGBoost yang sudah di-tuning.
    """)

# ----------------------------------------
# 5. Halaman PREDIKSI TUNGGAL
# ----------------------------------------
elif page == "Prediksi Tunggal":
    st.title("Prediksi Data Tunggal")

    with st.form("input_form"):
        age    = st.number_input("Umur (tahun)", 14, 80, 25)
        height = st.number_input("Tinggi Badan (meter)", 1.2, 2.2, 1.70, format="%.2f")
        weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0, format="%.1f")
        FCVC   = st.number_input("Frekuensi konsumsi sayur (0–10)", 0, 10, 3)
        NCP    = st.number_input("Jumlah makan per hari", 0, 10, 3)
        CH2O   = st.number_input("Asupan air (liter/hari)", 0.0, 10.0, 2.0)
        FAF    = st.number_input("Frekuensi aktivitas fisik", 0.0, 15.0, 1.0)
        TUE    = st.number_input("Jam penggunaan gadget", 0, 10, 2)

        gender_label  = st.selectbox("Jenis Kelamin", list(GENDER_MAP.keys()))
        calc_label    = st.selectbox("Minuman berkalori", list(CALC_MAP.keys()))
        favc_label    = st.selectbox("Konsumsi makanan berkalori tinggi", list(FAVC_MAP.keys()))
        scc_label     = st.selectbox("Pantau kalori yang dikonsumsi", list(SCC_MAP.keys()))
        smoke_label   = st.selectbox("Merokok", list(SMOKE_MAP.keys()))
        fam_label     = st.selectbox("Riwayat keluarga obesitas", list(FAM_MAP.keys()))
        caec_label    = st.selectbox("Makan camilan di antara waktu makan", list(CAEC_MAP.keys()))
        mtrans_label  = st.selectbox("Transportasi utama", list(MTRANS_MAP.keys()))

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': GENDER_MAP[gender_label],
            'Height': height,
            'Weight': weight,
            'FCVC': FCVC,
            'NCP': NCP,
            'CH2O': CH2O,
            'FAF': FAF,
            'TUE': TUE,
            'CALC': CALC_MAP[calc_label],
            'FAVC': FAVC_MAP[favc_label],
            'SCC': SCC_MAP[scc_label],
            'SMOKE': SMOKE_MAP[smoke_label],
            'family_history_with_overweight': FAM_MAP[fam_label],
            'CAEC': CAEC_MAP[caec_label],
            'MTRANS': MTRANS_MAP[mtrans_label]
        }])

        Xp = preprocess_input(input_df)

        # Prediksi dari masing-masing model
        p_knn = knn_tuned.predict(Xp)[0]
        p_svm = svm_tuned.predict(Xp)[0]
        p_xgb = xgb_tuned.predict(Xp)[0]

        # Label hasil klasifikasi
        c_knn = le.inverse_transform([p_knn])[0]
        c_svm = le.inverse_transform([p_svm])[0]
        c_xgb = le.inverse_transform([p_xgb])[0]

        st.success("Hasil Prediksi:")
        st.write(f"**KNN →** {c_knn}")
        st.write(f"**SVM →** {c_svm}")
        st.write(f"**XGBoost →** {c_xgb}")

# ----------------------------------------
# 6. Halaman PREDIKSI MASSAL
# ----------------------------------------
elif page == "Prediksi Massal":
    st.title("Prediksi Massal via File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Pratinjau data yang diunggah:")
        st.dataframe(data.head())

        proc = preprocess_input(data)
        preds = xgb_tuned.predict(proc)
        data['Hasil_Prediksi'] = le.inverse_transform(preds)

        st.subheader("Hasil Prediksi")
        st.dataframe(data[['Hasil_Prediksi']])

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh hasil sebagai CSV",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )
