# predictive_analytics.py (revised)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ... (keep prepare_prediction_data and train_models functions) ...

def get_user_input(features, df, key_suffix=""):
    user_input = {}
    with st.form(f"input_form_{key_suffix}"):
        col1, col2 = st.columns(2)
        with col1:
            user_input['age'] = st.slider("Age", 18, 100, 30, key=f"age_{key_suffix}")
            user_input['gender'] = st.selectbox("Gender", df['gender'].unique(), key=f"gender_{key_suffix}")
            user_input['family_history'] = st.selectbox("Family History of Mental Illness", 
                                                      df['family_history'].unique(), key=f"fam_{key_suffix}")
            user_input['benefits'] = st.selectbox("Employer Provides Mental Health Benefits", 
                                                df['benefits'].unique(), key=f"benefits_{key_suffix}")
        with col2:
            user_input['care_options'] = st.selectbox("Knowledge of Care Options", 
                                                    df['care_options'].unique(), key=f"care_{key_suffix}")
            user_input['leave'] = st.selectbox("Ease of Taking Medical Leave", 
                                             df['leave'].unique(), key=f"leave_{key_suffix}")
            user_input['mental_health_consequence'] = st.selectbox("Fear of Negative Consequences", 
                                                                 df['mental_health_consequence'].unique(), 
                                                                 key=f"conseq_{key_suffix}")
            user_input['no_employees'] = st.selectbox("Company Size", 
                                                    sorted(df['no_employees'].unique()), 
                                                    key=f"employees_{key_suffix}")
            user_input['tech_company'] = st.selectbox("Works in Tech", 
                                                    df['tech_company'].unique(), key=f"tech_{key_suffix}")
        
        submitted = st.form_submit_button("Predict")
    return user_input, submitted

def preprocess_input(user_input, encoded_columns, scaler):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    # Ensure same columns as training data
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)
    return scaler.transform(input_encoded)

def show(df):
    st.header("🔮 Mental Health Prediction Tool")
    st.caption("Enter your information to predict mental health outcomes")
    
    # Prepare data and models
    (X1_train, X1_test, y1_train, y1_test, 
     X2_train, X2_test, y2_train, y2_test, 
     le_interference, scaler1, scaler2, encoded_columns) = prepare_prediction_data(df)
    
    models = train_models(X1_train, y1_train, X2_train, y2_train)
    
    tab1, tab2 = st.tabs(["Treatment Prediction", "Work Interference Prediction"])
    
    with tab1:
        st.subheader("Will you seek mental health treatment?")
        model_choice = st.radio("Select Prediction Model", 
                               ["Logistic Regression", "Random Forest", "XGBoost"],
                               horizontal=True)
        
        user_input, submitted = get_user_input(encoded_columns, df, "treatment")
        
        if submitted:
            # Preprocess input
            input_scaled = preprocess_input(user_input, encoded_columns, scaler1)
            
            # Get model and predict
            model = models["treatment"][model_choice]
            proba = model.predict_proba(input_scaled)[0]
            prediction = model.predict(input_scaled)[0]
            
            # Display results
            st.divider()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Prediction", 
                         "Likely to seek treatment" if prediction == 1 else "Unlikely to seek treatment",
                         f"{proba[1]*100:.1f}%" if prediction == 1 else f"{proba[0]*100:.1f}%")
                
                # Visual probability gauge
                prob_value = proba[1] if prediction == 1 else proba[0]
                st.progress(int(prob_value * 100), 
                           text=f"Confidence: {prob_value*100:.1f}%")
            
            with col2:
                # Explain top influencing factors
                if model_choice == "Random Forest":
                    importances = model.feature_importances_
                    features = encoded_columns.tolist()
                    top_idx = np.argsort(importances)[-3:][::-1]
                    
                    st.write("**Key factors in this prediction:**")
                    for i, idx in enumerate(top_idx):
                        feature_name = features[idx].split("_")[-1]
                        st.markdown(f"{i+1}. {feature_name} ({(importances[idx]*100):.1f}%)")
                else:
                    st.info("Top factors available with Random Forest model")
    
    with tab2:
        st.subheader("How will mental health affect your work?")
        model_choice = st.radio("Select Prediction Model", 
                               ["Logistic Regression", "Random Forest", "XGBoost"],
                               horizontal=True, key="model_interfere")
        
        user_input, submitted = get_user_input(encoded_columns, df, "interfere")
        
        if submitted:
            # Preprocess input
            input_scaled = preprocess_input(user_input, encoded_columns, scaler2)
            
            # Get model and predict
            model = models["interfere"][model_choice]
            proba = model.predict_proba(input_scaled)[0]
            prediction = le_interference.inverse_transform(model.predict(input_scaled))[0]
            
            # Display results
            st.divider()
            st.subheader(f"Predicted Impact: **{prediction}**")
            
            # Probability distribution
            st.write("**Probability distribution:**")
            classes = le_interference.classes_
            for cls, p in zip(classes, proba):
                st.markdown(f"- {cls}: `{p*100:.1f}%`")
                st.progress(int(p * 100))
            
            # Business impact interpretation
            st.write("**What this means for your work:**")
            impact_info = {
                "Never": "Minimal productivity impact",
                "Rarely": "Occasional focus challenges",
                "Sometimes": "Noticeable productivity fluctuations",
                "Often": "Significant work performance impact"
            }
            st.info(impact_info.get(prediction, "Consult HR about workplace accommodations"))# predictive_analytics.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

    return X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test, target_names_task2

@st.cache_resource
def train_models(X1_train, y1_train, X2_train, y2_train):
    # Models for treatment prediction
    lr_treatment = LogisticRegression(max_iter=1000, random_state=42)
    rf_treatment = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_treatment = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    lr_treatment.fit(X1_train, y1_train)
    rf_treatment.fit(X1_train, y1_train)
    xgb_treatment.fit(X1_train, y1_train)
    
    # Models for work interference prediction
    lr_interfere = LogisticRegression(max_iter=1000, random_state=42)
    rf_interfere = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_interfere = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    lr_interfere.fit(X2_train, y2_train)
    rf_interfere.fit(X2_train, y2_train)
    xgb_interfere.fit(X2_train, y2_train)
    
    return {
        "treatment": {
            "Logistic Regression": lr_treatment,
            "Random Forest": rf_treatment,
            "XGBoost": xgb_treatment
        },
        "interfere": {
            "Logistic Regression": lr_interfere,
            "Random Forest": rf_interfere,
            "XGBoost": xgb_interfere
        }
    }

def show(df):
    st.header("🤖 Predictive Analytics")
    
    # Prepare data
    X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test, target_names_task2 = prepare_prediction_data(df)
    
    # Train models
    models = train_models(X1_train, y1_train, X2_train, y2_train)
    
    tab1, tab2 = st.tabs(["Treatment Prediction", "Work Interference Prediction"])
    
    with tab1:
        st.subheader("Predicting Treatment Seeking")
        model_choice = st.selectbox("Select Model", 
                                   ["Logistic Regression", "Random Forest", "XGBoost"])
        
        model = models["treatment"][model_choice]
        y_pred = model.predict(X1_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classification Report")
            report = classification_report(y1_test, y_pred, target_names=['No Treatment', 'Sought Treatment'], output_dict=True)
            st.json(report)
            
            st.subheader("Key Metrics")
            accuracy = report['accuracy']
            precision = report['1']['precision']
            recall = report['1']['recall']
            f1 = report['1']['f1-score']
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2f}")
            col2.metric("Precision", f"{precision:.2f}")
            col3.metric("Recall", f"{recall:.2f}")
            col4.metric("F1 Score", f"{f1:.2f}")
        
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y1_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Predicted No', 'Predicted Yes'],
                        yticklabels=['Actual No', 'Actual Yes'])
            ax.set_title(f'Confusion Matrix - {model_choice}')
            st.pyplot(fig)
            
            st.subheader("Feature Importance")
            if model_choice == "Random Forest":
                importances = model.feature_importances_
                features = pd.get_dummies(df[['age', 'gender', 'family_history', 'benefits', 'care_options', 
                                            'leave', 'mental_health_consequence', 'no_employees', 
                                            'tech_company']], drop_first=True).columns
                feature_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
                feature_imp = feature_imp.sort_values('Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis', ax=ax)
                ax.set_title('Top 10 Important Features')
                st.pyplot(fig)
            else:
                st.info("Feature importance visualization available for Random Forest only")
    
    with tab2:
        st.subheader("Predicting Work Interference")
        model_choice = st.selectbox("Select Model for Work Interference", 
                                   ["Logistic Regression", "Random Forest", "XGBoost"])
        
        model = models["interfere"][model_choice]
        y_pred = model.predict(X2_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classification Report")
            report = classification_report(y2_test, y_pred, target_names=target_names_task2, output_dict=True)
            st.json(report)
            
            st.subheader("Overall Metrics")
            accuracy = report['accuracy']
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2f}")
            col2.metric("Precision", f"{weighted_precision:.2f}")
            col3.metric("Recall", f"{weighted_recall:.2f}")
            col4.metric("F1 Score", f"{weighted_f1:.2f}")
        
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y2_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=target_names_task2, 
                        yticklabels=target_names_task2)
            ax.set_title(f'Confusion Matrix - {model_choice}')
            plt.xticks(rotation=45)
            st.pyplot(fig)
