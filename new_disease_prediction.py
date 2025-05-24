import streamlit as st
import pickle
import speech_recognition as sr
import os


# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Voice input function
def get_voice_input(label):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(f"Speak now for {label}...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError:
            st.error("Could not request results")
        except sr.WaitTimeoutError:
            st.error("Listening timed out")
    return ""

# Sidebar navigation
with st.sidebar:
    selected = st.selectbox('Choose Disease Prediction',
                            ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'])

# ==================== Diabetes Prediction ====================
if selected == 'Diabetes Prediction':
    st.title("🧪 Diabetes Prediction using ML")
    diabetes_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

    for feature in diabetes_features:
        key = f"diabetes_{feature}"
        if key not in st.session_state:
            st.session_state[key] = ""

    user_input = []
    for feature in diabetes_features:
        col1, col2 = st.columns([2, 1])
        with col1:
            val = st.text_input(f"{feature}", value=st.session_state[f"diabetes_{feature}"], key=f"text_diabetes_{feature}")
            st.session_state[f"diabetes_{feature}"] = val
        with col2:
            if st.button(f"🎤 Speak {feature}", key=f"speak_diabetes_{feature}"):
                voice_val = get_voice_input(feature)
                if voice_val:
                    st.session_state[f"diabetes_{feature}"] = voice_val
        user_input.append(st.session_state[f"diabetes_{feature}"])

    if st.button("🧬 Predict Diabetes"):
        try:
            values = [float(x) for x in user_input]
            prediction = diabetes_model.predict([values])
            result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
            st.success(result)
        except:
            st.error("Please enter valid numeric values.")

# ==================== Heart Disease Prediction ====================
elif selected == 'Heart Disease Prediction':
    st.title("❤️ Heart Disease Prediction using ML")
    heart_features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                      "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    for feature in heart_features:
        key = f"heart_{feature}"
        if key not in st.session_state:
            st.session_state[key] = ""

    user_input = []
    for feature in heart_features:
        col1, col2 = st.columns([2, 1])
        with col1:
            val = st.text_input(f"{feature}", value=st.session_state[f"heart_{feature}"], key=f"text_heart_{feature}")
            st.session_state[f"heart_{feature}"] = val
        with col2:
            if st.button(f"🎤 Speak {feature}", key=f"speak_heart_{feature}"):
                voice_val = get_voice_input(feature)
                if voice_val:
                    st.session_state[f"heart_{feature}"] = voice_val
        user_input.append(st.session_state[f"heart_{feature}"])

    if st.button("🫀 Predict Heart Disease"):
        try:
            values = [float(x) for x in user_input]
            prediction = heart_disease_model.predict([values])
            result = "The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease"
            st.success(result)
        except:
            st.error("Please enter valid numeric values.")

# ==================== Parkinson's Prediction ====================
elif selected == 'Parkinsons Prediction':
    st.title("🧠 Parkinson's Disease Prediction using ML")
    parkinsons_features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
                           "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
                           "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", 
                           "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", 
                           "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]

    for feature in parkinsons_features:
        key = f"parkinsons_{feature}"
        if key not in st.session_state:
            st.session_state[key] = ""

    user_input = []
    for feature in parkinsons_features:
        col1, col2 = st.columns([2, 1])
        with col1:
            val = st.text_input(f"{feature}", value=st.session_state[f"parkinsons_{feature}"], key=f"text_parkinsons_{feature}")
            st.session_state[f"parkinsons_{feature}"] = val
        with col2:
            if st.button(f"🎤 Speak {feature}", key=f"speak_parkinsons_{feature}"):
                voice_val = get_voice_input(feature)
                if voice_val:
                    st.session_state[f"parkinsons_{feature}"] = voice_val
        user_input.append(st.session_state[f"parkinsons_{feature}"])

    if st.button("🧠 Predict Parkinson's Disease"):
        try:
            values = [float(x) for x in user_input]
            prediction = parkinsons_model.predict([values])
            result = "The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease"
            st.success(result)
        except:
            st.error("Please enter valid numeric values.")