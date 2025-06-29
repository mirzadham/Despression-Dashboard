# predictive_analytics.py
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
