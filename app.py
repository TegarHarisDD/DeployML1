import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------
# 1. Load saved artifacts
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

# Numeric features that were scaled
num_feats = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','BMI']

# ----------------------------------------
# Category → code mappings
# ----------------------------------------
gender_map = {'Female': 0, 'Male': 1}
calc_map   = {'no': 0, 'Sometimes': 1, 'Frequently': 2}
favc_map   = {'no': 0, 'yes': 1}
scc_map    = {'no': 0, 'yes': 1}
smoke_map  = {'no': 0, 'yes': 1}
fam_map    = {'no': 0, 'yes': 1}
caec_map   = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
mtrans_map = {
    'Public_Transportation': 0,
    'Walking':               1,
    'Automobile':            2,
    'Bike':                  3,
    'Motorbike':             4
}

# ----------------------------------------
# 2. Preprocessing helper
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
# 3. Sidebar navigation
# ----------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Single Prediction", "Bulk Prediction"])

# ----------------------------------------
# 4. ABOUT page
# ----------------------------------------
if page == "About":
    st.title("Obesity Classification App")
    st.markdown(dataset_info)
    st.subheader("Tuned Model Accuracies")
    for model_name, acc in accuracy_tuned.items():
        st.write(f"- **{model_name}**: {acc:.4f}")
    st.markdown("""
    **Purpose:**  
    Classify individuals into obesity categories using KNN, SVM, and XGBoost (all hyperparameter‑tuned).
    """)

# ----------------------------------------
# 5. SINGLE PREDICTION page
# ----------------------------------------
elif page == "Single Prediction":
    st.title("Single‑Row Prediction")

    with st.form("input_form"):
        age     = st.number_input("Age (years)", 14, 80, 25)
        gender  = st.selectbox("Gender", list(gender_map.keys()))
        height  = st.number_input("Height (meters)", 1.2, 2.2, 1.70, format="%.2f")
        weight  = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, format="%.1f")

        FCVC    = st.number_input("Vegetable freq (FCVC)", 0, 10, 3)
        NCP     = st.number_input("Meals per day (NCP)", 0, 10, 3)
        CH2O    = st.number_input("Water intake L (CH2O)", 0.0, 10.0, 2.0)
        FAF     = st.number_input("Physical activity freq (FAF)", 0.0, 15.0, 1.0)
        TUE     = st.number_input("Device use hrs (TUE)", 0, 10, 2)

        CALC    = st.selectbox("Caloric drinks (CALC)", list(calc_map.keys()))
        FAVC    = st.selectbox("High‑cal food (FAVC)", list(favc_map.keys()))
        SCC     = st.selectbox("Calories monitoring (SCC)", list(scc_map.keys()))
        SMOKE   = st.selectbox("Smoking habit (SMOKE)", list(smoke_map.keys()))
        fam     = st.selectbox("Family history overweight", list(fam_map.keys()))
        CAEC    = st.selectbox("Snacking between meals (CAEC)", list(caec_map.keys()))
        MTRANS  = st.selectbox("Transport mode (MTRANS)", list(mtrans_map.keys()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender_map[gender],
            'Height': height,
            'Weight': weight,
            'FCVC': FCVC,
            'NCP': NCP,
            'CH2O': CH2O,
            'FAF': FAF,
            'TUE': TUE,
            'CALC': calc_map[CALC],
            'FAVC': favc_map[FAVC],
            'SCC': scc_map[SCC],
            'SMOKE': smoke_map[SMOKE],
            'family_history_with_overweight': fam_map[fam],
            'CAEC': caec_map[CAEC],
            'MTRANS': mtrans_map[MTRANS]
        }])

        Xp = preprocess_input(input_df)

        # raw predictions
        p_knn = knn_tuned.predict(Xp)[0]
        p_svm = svm_tuned.predict(Xp)[0]
        p_xgb = xgb_tuned.predict(Xp)[0]

        # map back to labels
        c_knn = le.inverse_transform([p_knn])[0]
        c_svm = le.inverse_transform([p_svm])[0]
        c_xgb = le.inverse_transform([p_xgb])[0]

        st.write(f"**KNN →** {c_knn}")
        st.write(f"**SVM →** {c_svm}")
        st.write(f"**XGB →** {c_xgb}")

# ----------------------------------------
# 6. BULK PREDICTION page
# ----------------------------------------
elif page == "Bulk Prediction":
    st.title("Bulk CSV Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())

        proc = preprocess_input(data)
        preds = xgb_tuned.predict(proc)
        data['Predicted_NObeyesdad'] = le.inverse_transform(preds)

        st.subheader("Predictions")
        st.dataframe(data[['Predicted_NObeyesdad']])

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
