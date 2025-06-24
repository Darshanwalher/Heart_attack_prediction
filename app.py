import streamlit as st
import pandas as pd
import joblib

# --- THEME TOGGLE ---
theme = st.radio("Select Theme 🎨", ["Dark", "Light"], horizontal=True)

# --- CSS STYLING ---
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1588776814546-ec345fc6f1b3');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .main {
            background-color: rgba(0, 0, 0, 0.85);
            color: #ffffff;
        }
        html, body, [class*="css"] {
            color: white;
        }
        .title {
            font-size: 36px;
            color: #ff4c4c;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #dddddd;
            text-align: center;
        }
        .stButton>button {
            background-color: #ff4c4c;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1588776814546-ec345fc6f1b3');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.85);
            color: #000000;
        }
        html, body, [class*="css"] {
            color: black;
        }
        .title {
            font-size: 36px;
            color: #4B9CD3;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #333333;
            text-align: center;
        }
        .stButton>button {
            background-color: #4B9CD3;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- LOAD MODEL AND SCALER ---
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# --- TITLE ---
st.markdown('<div class="title">💓 Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Please enter the required health details below</div>', unsafe_allow_html=True)
st.write("")

# --- INPUT FORM ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 100, 40)
    sex = st.selectbox("⚧️ Sex", ["Male", "Female"])
    chest_pain = st.selectbox("💢 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("🩺 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("🥩 Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("📈 Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("❤️ Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("📉 ST Slope", ["Up", "Flat", "Down"])

# --- PREDICTION LOGIC ---
if st.button("🔍 Predict Heart Risk"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("---")
    if prediction == 1:
        st.error("🚨 **High Risk** of Heart Disease. Please consult a doctor.")
    else:
        st.success("✅ **Low Risk** of Heart Disease. Stay healthy!")
    st.markdown("---")
