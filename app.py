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
# 2. Preprocessing helper
# ----------------------------------------
def preprocess_input(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # 2.1 Compute BMI
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # 2.2 One‑hot encode categorical columns exactly as in training
    cat_cols = [
        'Gender','CALC','FAVC','SCC','SMOKE',
        'family_history_with_overweight','CAEC','MTRANS'
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 2.3 Add any missing dummy columns (fill with 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # 2.4 Reorder to match training feature order
    df = df[feature_columns]

    # 2.5 Scale numeric features
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
    This application classifies individuals into obesity categories based on personal, dietary, 
    and lifestyle features, using three hyperparameter‑tuned models:  
    - K‑Nearest Neighbors (KNN)  
    - Support Vector Machine (SVM)  
    - XGBoost (XGB)  
    """)

# ----------------------------------------
# 5. SINGLE PREDICTION page
# ----------------------------------------
elif page == "Single Prediction":
    st.title("Single‑Row Prediction")

    with st.form("input_form"):
        age     = st.number_input("Age (years)",     min_value=14, max_value=80,  value=25)
        gender  = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
        height  = st.number_input("Height (meters)", min_value=1.2,  max_value=2.2,  value=1.70, format="%.2f")
        weight  = st.number_input("Weight (kg)",     min_value=30.0, max_value=200.0, value=70.0, format="%.1f")

        FCVC    = st.number_input("Vegetable freq (FCVC)", min_value=0, max_value=10, value=3)
        NCP     = st.number_input("Meals per day (NCP)",   min_value=0, max_value=10, value=3)
        CH2O    = st.number_input("Water intake L (CH2O)", min_value=0.0, max_value=10.0, value=2.0)
        FAF     = st.number_input("Physical activity freq (FAF)", min_value=0.0, max_value=15.0, value=1.0)
        TUE     = st.number_input("Device use hrs (TUE)", min_value=0, max_value=10, value=2)

        CALC    = st.selectbox("Caloric drinks (CALC)", [0, 1, 2])
        FAVC    = st.selectbox("High‑cal food (FAVC)", [0, 1])
        SCC     = st.selectbox("Calories monitoring (SCC)", [0, 1])
        SMOKE   = st.selectbox("Smoking habit (SMOKE)", [0, 1])
        fam     = st.selectbox("Family history overweight", [0, 1])
        CAEC    = st.selectbox("Snacking between meals (CAEC)", [0, 1, 2, 3])
        MTRANS  = st.selectbox("Transport mode (MTRANS)", [0, 1, 2, 3, 4])

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'Height': height, 'Weight': weight,
            'FCVC': FCVC, 'NCP': NCP, 'CH2O': CH2O, 'FAF': FAF, 'TUE': TUE,
            'CALC': CALC, 'FAVC': FAVC, 'SCC': SCC, 'SMOKE': SMOKE,
            'family_history_with_overweight': fam, 'CAEC': CAEC, 'MTRANS': MTRANS
        }])

        Xp = preprocess_input(input_df)

        # Raw numeric predictions
        p_knn = knn_tuned.predict(Xp)[0]
        p_svm = svm_tuned.predict(Xp)[0]
        p_xgb = xgb_tuned.predict(Xp)[0]

        # Inverse‑map to categorical labels
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

        # Map back to category labels
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
