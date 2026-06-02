import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
import imblearn
import xgboost as xgb


@st.cache_resource
def load_nn():
    
    minimised_nn = tf.keras.models.load_model('models/minimised_nn.keras')
    multi_nn = tf.keras.models.load_model('models/multi_nn.keras')
    comprehensive_nn = tf.keras.models.load_model('models/comprehensive_nn.keras')
    
    
    minimised_ct = joblib.load('models/minimised_ct.joblib')
    multi_ct = joblib.load('models/multi_ct.joblib')
    comprehensive_ct = joblib.load('models/comprehensive_ct.joblib')
    
    
    return minimised_nn,multi_nn,comprehensive_nn,minimised_ct,multi_ct,comprehensive_ct 

minimised_nn,multi_nn,comprehensive_nn,minimised_ct,multi_ct,comprehensive_ct = load_nn()

st.set_page_config(
    page_title="Heart Disease Demo",
    layout="centered"
)

datasetChoice = ["Minimised", "Multi-Classification", "Comprehensive"]
modelChoice = ["Neural Network", "SVM", "Random Forest", "XGBoost"]

Preset_Data = {
    
    "Comprehensive": {
        "High-Risk": [64,1,1,170,227,0,1,142,0,0,2],
        "Low-Risk": [43,0,1,100,223,0,1.5,142,0,0,1],
        "Middle": [65,1,4,136,248,0,1.5,140,1,4,3],
    },
    "Multi-Classification": {
        "High-Risk": [60.0,1.0,4.0,130.0,206.0,0.0,2.0,132.0,1.0,2.4,2.0,2.0,7.0],
        "Low-Risk": [61.0,0.0,4.0,130.0,330.0,0.0,2.0,169.0,0.0,0.0,1.0,0.0,3.0],
        "Middle": [54.0,1.0,4.0,120.0,188.0,0.0,0.0,113.0,0.0,1.4,2.0,1.0,7.0],
        
    }, 
    "Minimised": {
        "High-Risk": [60.0,1.0,4.0,130.0,206.0,0.0,2.0,132.0,1.0,2.4,2.0,2.0,7.0],
        "Low-Risk": [61.0,0.0,4.0,130.0,330.0,0.0,2.0,169.0,0.0,0.0,1.0,0.0,3.0],
        "Middle": [54.0,1.0,4.0,120.0,188.0,0.0,0.0,113.0,0.0,1.4,2.0,1.0,7.0],
    
}
}

st.title("Heart Disease Diagnosis Demo")
st.markdown("Choose a Dataset and ML model to begin testing")

chosen_dataset = st.selectbox("Choose Dataset", datasetChoice)

chosen_model = st.selectbox("Chosen Model", modelChoice)

preset_options = list(Preset_Data.get(chosen_dataset, {}).keys()) + ["Manual input"]
preset_choice = st.selectbox("Choose a Preset Patient data", preset_options)

col1, col2 = st.columns(2)


data_value = Preset_Data[chosen_dataset][preset_choice]

    
if chosen_dataset == "Minimised":
    with col1:
        age = st.number_input("Age", value=data_value[0])
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1,0], index=0 if data_value[0]==1 else 1)
        cp = st.number_input("Chest Pain Type", value=data_value[2])
        bps = st.number_input("Resting Blood Pressure", value=data_value[3])
    with col2:
        chol = st.number_input("Cholesterol", value=data_value[4])
        fbs = st.number_input("Fasting Blood Sugar", value=data_value[5])
        exang = st.number_input("Exercise Induced Angina (0-1)", value=data_value[8])
elif chosen_dataset == "Comprehensive":
    with col1:
        age = st.number_input("Age", value=data_value[0])
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1,0], index=0 if data_value[1]==1 else 1)
        cp = st.number_input("Chest Pain Type", value=data_value[2])
        bps = st.number_input("Resting Blood Pressure", value=data_value[3])
        chol = st.number_input("Cholesterol", value=data_value[4])
    with col2:
        fbs = st.number_input("Fasting Blood Sugar", value=data_value[5])
        ecg = st.number_input("Resting ECG", value=data_value[6])
        maxHR = st.number_input("Max Heart Rate", value=data_value[7])
        exang = st.number_input("Exercise Induced Angina (1-4)", value=data_value[8])
        oldpeak = st.number_input("OldPeak", value=data_value[9])
        stslope = st.number_input("ST Slope", value=data_value[10])
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", value=data_value[0])
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1,0], index=0 if data_value[1]==1 else 1)
        cp = st.number_input("Chest Pain Type", value=data_value[2])
        bps = st.number_input("Resting Blood Pressure", value=data_value[3])
        chol = st.number_input("Cholesterol", value=data_value[4])
    with col2:
        fbs = st.number_input("Fasting Blood Sugar", value=data_value[5])
        ecg = st.number_input("Resting ECG", value=data_value[6])
        maxHR = st.number_input("Max Heart Rate", value=data_value[7])
        exang = st.number_input("Exercise Induced Angina (1-4)", value=data_value[8])
    with col3:
        oldpeak = st.number_input("OldPeak", value=data_value[9])
        stslope = st.number_input("ST Slope", value=data_value[10])
        ca = st.number_input("Major Vessels colored by Flurosopy (0-3)", value=data_value[11])
        thal = st.number_input("Thalassemia", value=data_value[12])
        
if st.button("Predict", type="primary"): 
    
    if chosen_model == "Random Forest":
        chosen_model = "rf"
    elif chosen_model == "Neural Network":
        chosen_model = "nn"
    elif chosen_model == "XGBoost":
        chosen_model = "xgb"
    else:
        chosen_model = "svm"
    
    
    
    if chosen_dataset == "Minimised":
        path = f"models/minimised_{chosen_model}.joblib"
        ct_path = "models/minimised_ct.joblib"
    elif chosen_dataset == "Comprehensive":
        path = f"models/comprehensive_{chosen_model}.joblib"
        ct_path = "models/comprehensive_ct.joblib"
    else: 
        path = f"models/multi_{chosen_model}.joblib"
        ct_path = "models/multi_ct.joblib"
        
    try:
        if chosen_model == "nn":
            if chosen_dataset == "Minimised":
                model = minimised_nn
                ct = minimised_ct
            elif chosen_dataset == "Comprehensive":
                model = comprehensive_nn
                ct = comprehensive_ct
            else: 
                model = multi_nn
                ct = multi_ct
        else:
            model = joblib.load(path)
    
        if chosen_dataset == "Minimised":
            df = pd.DataFrame([[age,sex,cp,bps,chol,fbs,exang]], columns=['age','sex','chest pain type','resting bp s','cholesterol','fasting blood sugar','exercise angina'])
            
        elif chosen_dataset == "Comprehensive":
            df = pd.DataFrame([[age,sex,cp, bps, chol, fbs, ecg, maxHR, exang, oldpeak, stslope]], columns=['age','sex','chest pain type','resting bp s','cholesterol','fasting blood sugar','resting ecg','max heart rate','exercise angina','oldpeak','ST slope'])
            
        else:
            df = pd.DataFrame([[age,sex,cp,bps,chol,fbs,ecg,maxHR,exang,oldpeak,stslope,ca,thal]], columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])
            
            
        if chosen_model == "nn":
            if chosen_dataset == "Multi-Classification":
                prediction = model.predict(ct.transform(df))[0]
                st.write(f"Prediction: {prediction[0]:.1%} likelihood of healthy, {prediction[1]:.1%} likelihood of mild heart disease,{prediction[2]:.1%} likelihood of moderate heart disease, {prediction[3]:.1%} likelihood of severe heart disease, {prediction[4]:.1%} likelihood of very severe heart disease")
            else:
                prediction = model.predict(ct.transform(df))[0]
                st.write(f"Prediction: {prediction:.1%} likelihood of heart disease")
        else:
            if chosen_dataset == "Multi-Classification":
                prediction = model.predict_proba(df)
                st.write(f"Prediction: {prediction[0][0]:.1%} likelihood of healthy, {prediction[0][1]:.1%} likelihood of mild heart disease,{prediction[0][2]:.1%} likelihood of moderate heart disease, {prediction[0][3]:.1%} likelihood of severe heart disease, {prediction[0][4]:.1%} likelihood of very severe heart disease")
            else:
                prediction = model.predict_proba(df)
                st.write(f"Prediction: {prediction[0][1]:.1%} likelihood of heart disease")
            
            
            
            
    except FileNotFoundError:
        st.warning("File not found")
    
    
    
            
        
        
        
        
        
    

    
    