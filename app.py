import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------
# 1. Load scaler, models, accuracy & cols
# ----------------------------------------
scaler           = joblib.load('scaler.pkl')
knn_tuned        = joblib.load('knn_tuned.pkl')
svm_tuned        = joblib.load('svm_tuned.pkl')
xgb_tuned        = joblib.load('xgb_tuned.pkl')
accuracy_tuned   = joblib.load('accuracy_tuned.pkl')
with open('dataset_info.md','r') as f:
    dataset_info = f.read()

# THIS is the crucial addition:
feature_columns = joblib.load('feature_columns.pkl')

# Numeric features you scaled
num_feats = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','BMI']

# ----------------------------------------
# 2. Preprocessing helper
# ----------------------------------------
def preprocess_input(df_in):
    df = df_in.copy()
    # 2.1 compute BMI
    df['BMI'] = df['Weight'] / df['Height']**2

    # 2.2 one‑hot encode exactly as in training
    #    drop_first=True was used, so we must replicate it
    cat_cols = ['Gender','CALC','FAVC','SCC','SMOKE',
                'family_history_with_overweight','CAEC','MTRANS']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 2.3 Add any missing dummy columns from training; fill with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # 2.4 Reorder to match training
    df = df[feature_columns]

    # 2.5 Scale numeric columns
    df[num_feats] = scaler.transform(df[num_feats])

    return df

# ----------------------------------------
# 3. Sidebar navigation
# ----------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About","Single Prediction","Bulk Prediction"])

# ----------------------------------------
# 4. ABOUT page
# ----------------------------------------
if page == "About":
    st.title("Obesity Classification App")
    st.markdown(dataset_info)
    st.subheader("Tuned Model Accuracies")
    for name, acc in accuracy_tuned.items():
        st.write(f"- **{name}**: {acc:.4f}")
    st.markdown("""
    **Purpose:**  
    Classify obesity levels using KNN, SVM & XGBoost (hyperparameter‑tuned).  
    """)

# ----------------------------------------
# 5. SINGLE PREDICTION page
# ----------------------------------------
elif page == "Single Prediction":
    st.title("Single‑Row Prediction")

    with st.form("input_form"):
        age     = st.number_input("Age", 14, 80, 25)
        gender  = st.selectbox("Gender (0=F,1=M)", [0,1])
        height  = st.number_input("Height (m)", 1.2, 2.2, 1.7, format="%.2f")
        weight  = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, format="%.1f")
        FCVC    = st.number_input("Veg freq (FCVC)", 0, 10, 3)
        NCP     = st.number_input("Main meals (NCP)", 0, 10, 3)
        CH2O    = st.number_input("Water liters (CH2O)", 0.0, 10.0, 2.0)
        FAF     = st.number_input("Phys act freq (FAF)", 0.0, 15.0, 1.0)
        TUE     = st.number_input("Device hours (TUE)", 0, 10, 2)
        CALC    = st.selectbox("Caloric drinks (CALC)", [0,1,2])
        FAVC    = st.selectbox("High‑cal food (FAVC)", [0,1])
        SCC     = st.selectbox("Calorie monitoring (SCC)", [0,1])
        SMOKE   = st.selectbox("Smoking (SMOKE)", [0,1])
        fam     = st.selectbox("Fam overweight", [0,1])
        CAEC    = st.selectbox("Snacking (CAEC)", [0,1,2,3])
        MTRANS  = st.selectbox("Transport (MTRANS)", [0,1,2,3,4])
        submit  = st.form_submit_button("Predict")

    if submit:
        inp = pd.DataFrame([{
            'Age':age,'Gender':gender,'Height':height,'Weight':weight,
            'FCVC':FCVC,'NCP':NCP,'CH2O':CH2O,'FAF':FAF,'TUE':TUE,
            'CALC':CALC,'FAVC':FAVC,'SCC':SCC,'SMOKE':SMOKE,
            'family_history_with_overweight':fam,'CAEC':CAEC,'MTRANS':MTRANS
        }])
        Xp = preprocess_input(inp)
        st.write("**KNN →**", knn_tuned.predict(Xp)[0])
        st.write("**SVM →**", svm_tuned.predict(Xp)[0])
        st.write("**XGB →**", xgb_tuned.predict(Xp)[0])

# ----------------------------------------
# 6. BULK PREDICTION page
# ----------------------------------------
elif page == "Bulk Prediction":
    st.title("Bulk CSV Prediction")
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        data = pd.read_csv(up)
        st.write(data.head())
        proc = preprocess_input(data)
        data['Predicted_NObeyesdad'] = xgb_tuned.predict(proc)
        st.write(data[['Predicted_NObeyesdad']])
        st.download_button("Download CSV", data.to_csv(index=False), "preds.csv")
