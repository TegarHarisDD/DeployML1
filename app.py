import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files and models
scaler = joblib.load('scaler.pkl')
knn_default = joblib.load('knn_default.pkl')
svm_default = joblib.load('svm_default.pkl')
xgb_default = joblib.load('xgb_default.pkl')

knn_tuned = joblib.load('knn_tuned.pkl')
svm_tuned = joblib.load('svm_tuned.pkl')
xgb_tuned = joblib.load('xgb_tuned.pkl')

accuracy_default = joblib.load('accuracy_default.pkl')
accuracy_tuned = joblib.load('accuracy_tuned.pkl')

with open('dataset_info.md', 'r') as f:
    dataset_info = f.read()

# Constants
num_feats = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','BMI']

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Single Prediction", "Bulk Prediction"])

def preprocess_input(data):
    """Preprocess a single sample or DataFrame"""
    data = data.copy()
    # Calculate BMI if not present
    if 'BMI' not in data.columns:
        data['BMI'] = data['Weight'] / (data['Height'] ** 2)
    # Encode categorical columns manually as done in training (Gender, CALC, etc.)
    # Assuming these columns:
    cat_cols = ['Gender','CALC','FAVC','SCC','SMOKE',
                'family_history_with_overweight','CAEC','MTRANS']
    for col in cat_cols:
        # For this example, assume binary and encoded as 0/1 or via get_dummies with drop_first=True
        if col in data.columns:
            data[col] = data[col].astype(int)
        else:
            data[col] = 0  # default if missing
    # Create dummy vars as needed (must match training)
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
    # Add any missing dummy columns from training set with 0
    trained_cols = scaler.feature_names_in_
    for c in trained_cols:
        if c not in data.columns:
            data[c] = 0
    # Reorder columns to match training set
    data = data[trained_cols]
    # Scale numeric features
    data[num_feats] = scaler.transform(data[num_feats])
    return data

if page == "About":
    st.title("Obesity Classification App")
    st.markdown(dataset_info)
    st.subheader("Model Accuracies")
    st.markdown("### Default Models")
    for model_name, acc in accuracy_default.items():
        st.write(f"- {model_name}: {acc:.4f}")
    st.markdown("### Tuned Models")
    for model_name, acc in accuracy_tuned.items():
        st.write(f"- {model_name}: {acc:.4f}")
    st.markdown("### Purpose")
    st.write("""
    This app classifies obesity levels based on personal and lifestyle data using
    three machine learning models (KNN, SVM, XGBoost) with and without hyperparameter tuning.
    """)

elif page == "Single Prediction":
    st.title("Single Prediction")
    with st.form("input_form"):
        # Input form fields (example subset, add all necessary)
        age = st.number_input("Age (years)", min_value=14, max_value=80, value=25)
        gender = st.selectbox("Gender (0=Female, 1=Male)", options=[0,1])
        height = st.number_input("Height (meters)", min_value=1.2, max_value=2.2, value=1.7, format="%.2f")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f")
        FCVC = st.number_input("Frequency of vegetables consumption (FCVC)", min_value=0, max_value=10, value=3)
        NCP = st.number_input("Number of main meals (NCP)", min_value=0, max_value=10, value=3)
        CH2O = st.number_input("Daily water consumption (CH2O) (liters)", min_value=0, max_value=10, value=2)
        FAF = st.number_input("Physical activity frequency (FAF)", min_value=0.0, max_value=15.0, value=1.0)
        TUE = st.number_input("Time using electronic devices (TUE) (hours)", min_value=0, max_value=10, value=2)
        
        # Categorical binary inputs
        CALC = st.selectbox("Calories consumption (CALC)", options=[0,1,2])
        FAVC = st.selectbox("High calorie food consumption (FAVC)", options=[0,1])
        SCC = st.selectbox("Smoke (SCC)", options=[0,1])
        SMOKE = st.selectbox("Smoking habit (SMOKE)", options=[0,1])
        family_history = st.selectbox("Family history with overweight", options=[0,1])
        CAEC = st.selectbox("Consumption of food between meals (CAEC)", options=[0,1,2,3])
        MTRANS = st.selectbox("Transportation used (MTRANS)", options=[0,1,2,3,4])

        submitted = st.form_submit_button("Predict")
    if submitted:
        input_df = pd.DataFrame({
            'Age': [age], 'Gender':[gender], 'Height':[height], 'Weight':[weight],
            'FCVC':[FCVC], 'NCP':[NCP], 'CH2O':[CH2O], 'FAF':[FAF], 'TUE':[TUE],
            'CALC':[CALC], 'FAVC':[FAVC], 'SCC':[SCC], 'SMOKE':[SMOKE],
            'family_history_with_overweight':[family_history], 'CAEC':[CAEC], 'MTRANS':[MTRANS]
        })
        input_processed = preprocess_input(input_df)
        pred_knn = knn_tuned.predict(input_processed)[0]
        pred_svm = svm_tuned.predict(input_processed)[0]
        pred_xgb = xgb_tuned.predict(input_processed)[0]

        st.write(f"KNN Prediction: {pred_knn}")
        st.write(f"SVM Prediction: {pred_svm}")
        st.write(f"XGB Prediction: {pred_xgb}")

elif page == "Bulk Prediction":
    st.title("Bulk Prediction")
    st.write("Upload CSV file with same features as input except target")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview uploaded data:")
        st.write(data.head())

        data_processed = preprocess_input(data)
        preds = xgb_tuned.predict(data_processed)
        data['Predicted_NObeyesdad'] = preds

        st.write("Predictions:")
        st.write(data[['Predicted_NObeyesdad']])
        st.download_button(
            label="Download predictions as CSV",
            data=data.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
