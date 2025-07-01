import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def prepare_prediction_data(df):
    features_to_use = [
        'age', 'gender', 'family_history', 'benefits', 'care_options', 'leave',
        'mental_health_consequence', 'no_employees', 'tech_company'
    ]

    # Task 1: Predicting Treatment
    df_task1 = df.copy()
    X1 = df_task1[features_to_use]
    y1 = df_task1['treatment']

    X1_encoded = pd.get_dummies(X1, drop_first=True)
    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1_encoded)
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42, stratify=y1)

    # Task 2: Predicting Work Interference
    df_task2 = df.copy()
    df_task2.dropna(subset=['work_interfere'], inplace=True)
    df_task2 = df_task2[~df_task2['work_interfere'].isin(['N/A', 'Not applicable'])]

    X2 = df_task2[features_to_use]
    y2_raw = df_task2['work_interfere']

    le_interference = LabelEncoder()
    y2 = le_interference.fit_transform(y2_raw)
    target_names_task2 = le_interference.classes_

    X2_encoded = pd.get_dummies(X2, drop_first=True)
    X2_encoded = X2_encoded.reindex(columns=X1_encoded.columns, fill_value=0)

    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2_encoded)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42, stratify=y2)

    return (X1_train, X1_test, y1_train, y1_test, 
            X2_train, X2_test, y2_train, y2_test, 
            target_names_task2, X1_encoded.columns, scaler1, scaler2)

@st.cache_resource
def train_models(X1_train, y1_train, X2_train, y2_train):
    rf_treatment = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_treatment.fit(X1_train, y1_train)
    
    rf_interfere = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_interfere.fit(X2_train, y2_train)
    
    return {
        "treatment": rf_treatment,
        "interfere": rf_interfere
    }

def predict_treatment(model, input_data, feature_columns, scaler):
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability

def predict_work_interference(model, input_data, feature_columns, scaler, target_names):
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    prediction_label = target_names[prediction]
    return prediction_label, probability

def show(df):
    st.header("🔮 Predictive Insights")
    st.markdown("""
    **Predict mental health outcomes** based on employee characteristics and workplace factors.
    Use the input panels below to simulate different scenarios and see predicted outcomes.
    """)
    
    # Prepare data
    (X1_train, X1_test, y1_train, y1_test, 
     X2_train, X2_test, y2_train, y2_test, 
     target_names_task2, feature_columns, scaler1, scaler2) = prepare_prediction_data(df)
    
    # Train models
    models = train_models(X1_train, y1_train, X2_train, y2_train)
    
    tab1, tab2 = st.tabs(["Treatment Prediction", "Work Interference Prediction"])
    
    with tab1:
        st.subheader("Predict Likelihood of Seeking Treatment")
        
        with st.expander("💼 Employee Information"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 18, 75, 35)
                gender = st.selectbox("Gender", ["Male", "Female", "Trans", "Other"])
                family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
                
            with col2:
                no_employees = st.selectbox("Company Size", 
                                           ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
                tech_company = st.selectbox("Works in Tech Company", ["Yes", "No"])
        
        with st.expander("🏥 Workplace Mental Health Support"):
            col1, col2 = st.columns(2)
            with col1:
                benefits = st.selectbox("Mental Health Benefits", ["Yes", "No", "Don't know"])
                care_options = st.selectbox("Knowledge of Care Options", ["Yes", "No", "Not sure"])
                
            with col2:
                leave = st.selectbox("Ease of Taking Medical Leave", 
                                    ["Very easy", "Somewhat easy", "Don't know", 
                                     "Somewhat difficult", "Very difficult"])
                mental_health_consequence = st.selectbox("Fear of Negative Consequences", 
                                                        ["Yes", "No", "Maybe"])
        
        # Create input dictionary
        input_data = {
            'age': age,
            'gender': gender,
            'family_history': family_history,
            'benefits': benefits,
            'care_options': care_options,
            'leave': leave,
            'mental_health_consequence': mental_health_consequence,
            'no_employees': no_employees,
            'tech_company': tech_company
        }
        
        if st.button("Predict Treatment Seeking", type="primary"):
            prediction, probability = predict_treatment(
                models["treatment"], 
                input_data, 
                feature_columns, 
                scaler1
            )
            
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success("### This employee is LIKELY to seek treatment for mental health issues")
                st.metric("Probability", f"{probability[1]*100:.1f}%")
            else:
                st.warning("### This employee is UNLIKELY to seek treatment for mental health issues")
                st.metric("Probability", f"{probability[0]*100:.1f}%")
            
            # Show feature importance
            st.subheader("Key Influencing Factors")
            model = models["treatment"]
            importances = model.feature_importances_
            features = feature_columns
            feature_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
            feature_imp = feature_imp.sort_values('Importance', ascending=False).head(5)
            feature_imp['Feature'] = feature_imp['Feature'].str.replace('_', ' ').str.title()
            fig = px.bar(feature_imp, y='Feature', x='Importance', orientation='h',
                        title='<b>Top Influencing Factors</b>',
                        color_discrete_sequence=['#1a5a9e'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Model accuracy: {accuracy_score(y1_test, model.predict(X1_test))*100:.1f}%")
    
    with tab2:
        st.subheader("Predict Work Interference from Mental Health")
        st.info("Predict how much mental health issues might interfere with work performance")
        
        with st.expander("💼 Employee Information"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age ", 18, 75, 35)
                gender = st.selectbox("Gender ", ["Male", "Female", "Trans", "Other"])
                family_history = st.selectbox("Family History of Mental Illness ", ["Yes", "No"])
                
            with col2:
                no_employees = st.selectbox("Company Size ", 
                                           ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
                tech_company = st.selectbox("Works in Tech Company ", ["Yes", "No"])
        
        with st.expander("🏥 Workplace Mental Health Support"):
            col1, col2 = st.columns(2)
            with col1:
                benefits = st.selectbox("Mental Health Benefits ", ["Yes", "No", "Don't know"])
                care_options = st.selectbox("Knowledge of Care Options ", ["Yes", "No", "Not sure"])
                
            with col2:
                leave = st.selectbox("Ease of Taking Medical Leave ", 
                                    ["Very easy", "Somewhat easy", "Don't know", 
                                     "Somewhat difficult", "Very difficult"])
                mental_health_consequence = st.selectbox("Fear of Negative Consequences ", 
                                                        ["Yes", "No", "Maybe"])
        
        # Create input dictionary
        input_data = {
            'age': age,
            'gender': gender,
            'family_history': family_history,
            'benefits': benefits,
            'care_options': care_options,
            'leave': leave,
            'mental_health_consequence': mental_health_consequence,
            'no_employees': no_employees,
            'tech_company': tech_company
        }
        
        if st.button("Predict Work Interference", type="primary"):
            prediction, probability = predict_work_interference(
                models["interfere"], 
                input_data, 
                feature_columns, 
                scaler2,
                target_names_task2
            )
            
            st.subheader("Prediction Result")
            
            interference_levels = {
                "Never": ("😊", "Mental health rarely affects work performance"),
                "Rarely": ("😐", "Occasional minor impact on work"),
                "Sometimes": ("😟", "Noticeable impact on work performance"),
                "Often": ("😞", "Frequent significant impact on work")
            }
            
            emoji, description = interference_levels.get(prediction, ("❓", "Unknown impact level"))
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                <div style="font-size: 48px; margin-bottom: 10px;">{emoji}</div>
                <h3 style="color: #1f77b4;">{prediction}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                "Interference Level": target_names_task2,
                "Probability": [f"{p*100:.2f}%" for p in probability]
            })
            fig = px.bar(prob_df, x='Probability', y='Interference Level', orientation='h',
                        title='<b>Prediction Probability Distribution</b>',
                        color='Interference Level',
                        color_discrete_sequence=px.colors.sequential.Blues)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show key factors
            st.subheader("Key Influencing Factors")
            model = models["interfere"]
            importances = model.feature_importances_
            features = feature_columns
            feature_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
            feature_imp = feature_imp.sort_values('Importance', ascending=False).head(5)
            feature_imp['Feature'] = feature_imp['Feature'].str.replace('_', ' ').str.title()
            fig = px.bar(feature_imp, y='Feature', x='Importance', orientation='h',
                        title='<b>Top Influencing Factors</b>',
                        color_discrete_sequence=['#1a5a9e'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Model accuracy: {accuracy_score(y2_test, model.predict(X2_test))*100:.1f}%")
