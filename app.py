import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------
# 1. Load saved artifacts
# ----------------------------------------
scaler           = joblib.load('scaler.pkl')
knn_tuned        = joblib.load('knn_tuned.pkl')
svm_tuned        = joblib.load('svm_tuned.pkl')
xgb_tuned        = joblib.load('xgb_tuned.pkl')
le               = joblib.load('label_encoder.pkl')
accuracy_tuned   = joblib.load('accuracy_tuned.pkl')
feature_columns  = joblib.load('feature_columns.pkl')

with open('dataset_info.md', 'r') as f:
    dataset_info = f.read()

# Numeric features that were scaled
num_feats = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','BMI']

# ----------------------------------------
# 1.a Mapping dicts for form labels → codes
# ----------------------------------------
GENDER_MAP = {"Female": 0, "Male": 1}
CALC_MAP   = {"No": 0, "Sometimes": 1, "Frequently": 2}
FAVC_MAP   = {"No": 0, "Yes": 1}
SCC_MAP    = {"No": 0, "Yes": 1}
SMOKE_MAP  = {"No": 0, "Yes": 1}
FAM_MAP    = {"No": 0, "Yes": 1}
CAEC_MAP   = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
MTRANS_MAP = {
    "Walking": 0,
    "Public Transportation": 1,
    "Automobile": 2,
    "Motorbike": 3,
    "Bike": 4
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
    Classify obesity levels based on personal, dietary, and lifestyle features  
    using hyperparameter‑tuned KNN, SVM & XGBoost models.
    """)

# ----------------------------------------
# 5. SINGLE PREDICTION page
# ----------------------------------------
elif page == "Single Prediction":
    st.title("Single‑Row Prediction")

    with st.form("input_form"):
        # Numeric inputs
        age    = st.number_input("Age (years)", 14, 80, 25)
        height = st.number_input("Height (meters)", 1.2, 2.2, 1.70, format="%.2f")
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, format="%.1f")
        FCVC   = st.number_input("Vegetable freq (FCVC)", 0, 10, 3)
        NCP    = st.number_input("Meals per day (NCP)", 0, 10, 3)
        CH2O   = st.number_input("Water intake L (CH2O)", 0.0, 10.0, 2.0)
        FAF    = st.number_input("Physical activity freq (FAF)", 0.0, 15.0, 1.0)
        TUE    = st.number_input("Device use hrs (TUE)", 0, 10, 2)

        # Categorical inputs as labels
        gender_label  = st.selectbox("Gender", list(GENDER_MAP.keys()))
        calc_label    = st.selectbox("Caloric drinks (CALC)", list(CALC_MAP.keys()))
        favc_label    = st.selectbox("High‑calorie food (FAVC)", list(FAVC_MAP.keys()))
        scc_label     = st.selectbox("Calorie monitoring (SCC)", list(SCC_MAP.keys()))
        smoke_label   = st.selectbox("Smoking habit (SMOKE)", list(SMOKE_MAP.keys()))
        fam_label     = st.selectbox("Family history overweight", list(FAM_MAP.keys()))
        caec_label    = st.selectbox("Snacking between meals (CAEC)", list(CAEC_MAP.keys()))
        mtrans_label  = st.selectbox("Transport mode (MTRANS)", list(MTRANS_MAP.keys()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Map labels back to numeric codes
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

        # Raw predictions
        p_knn = knn_tuned.predict(Xp)[0]
        p_svm = svm_tuned.predict(Xp)[0]
        p_xgb = xgb_tuned.predict(Xp)[0]

        # Inverse to category names
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
    if uploaded_file is not None:
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
            label="Download predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
