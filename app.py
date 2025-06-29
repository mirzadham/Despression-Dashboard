import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================================
# 1. SETUP AND DATA LOADING
# ==============================================================================
st.set_page_config(layout="wide", page_title="Mental Health in Tech Dashboard", page_icon="🧠")

# Custom CSS to improve styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1.5rem 1rem;
    }
    h1, h2, h3 {
        color: #1f77b4;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.3rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df_2014 = pd.read_csv('2014.csv')
        df_2016 = pd.read_csv('2016.csv')
        return df_2014, df_2016
    except FileNotFoundError:
        st.error("ERROR: Make sure you have uploaded both '2014.csv' and '2016.csv'")
        st.stop()

@st.cache_data
def clean_and_merge_data(df_2014, df_2016):
    # Define mappings from original column names to standard names
    map_2014 = {
        'Age': 'age', 'Gender': 'gender', 'self_employed': 'self_employed',
        'family_history': 'family_history', 'treatment': 'treatment', 'work_interfere': 'work_interfere',
        'no_employees': 'no_employees', 'tech_company': 'tech_company', 'benefits': 'benefits',
        'care_options': 'care_options', 'wellness_program': 'wellness_program', 'seek_help': 'seek_help',
        'anonymity': 'anonymity', 'leave': 'leave', 'mental_health_consequence': 'mental_health_consequence',
        'coworkers': 'coworkers', 'supervisor': 'supervisor'
    }
    df_2014_clean = df_2014[list(map_2014.keys())].rename(columns=map_2014)

    map_2016 = {
        'What is your age?': 'age', 'What is your gender?': 'gender', 'Are you self-employed?': 'self_employed',
        'Do you have a family history of mental illness?': 'family_history',
        'Have you ever sought treatment for a mental health issue from a mental health professional?': 'treatment',
        'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?': 'work_interfere',
        'How many employees does your company or organization have?': 'no_employees',
        'Is your employer primarily a tech company/organization?': 'tech_company',
        'Does your employer provide mental health benefits as part of healthcare coverage?': 'benefits',
        'Do you know the options for mental health care available under your employer-provided coverage?': 'care_options',
        'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?': 'wellness_program',
        'Does your employer offer resources to learn more about mental health concerns and options for seeking help?': 'seek_help',
        'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?': 'anonymity',
        'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:': 'leave',
        'Do you think that discussing a mental health disorder with your employer would have negative consequences?': 'mental_health_consequence',
        'Would you feel comfortable discussing a mental health disorder with your coworkers?': 'coworkers',
        'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?': 'supervisor'
    }
    df_2016_clean = df_2016[list(map_2016.keys())].rename(columns=map_2016)

    df_merged = pd.concat([df_2014_clean, df_2016_clean], ignore_index=True)

    # Clean Age column
    df_merged['age'] = pd.to_numeric(df_merged['age'], errors='coerce')
    df_merged.dropna(subset=['age'], inplace=True)
    df_merged = df_merged[(df_merged['age'] >= 18) & (df_merged['age'] <= 75)].copy()
    df_merged['age'] = df_merged['age'].astype(int)

    # Clean Gender column
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "cis male"]
    female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]
    trans_str = ["trans-female", "something akin to female...", "woman-ish", "transgender", "female (trans)"]

    def clean_gender(gender_str):
        if pd.isna(gender_str):
            return gender_str
        gender_str = str(gender_str).lower()
        if gender_str in male_str:
            return 'Male'
        elif gender_str in female_str:
            return 'Female'
        elif gender_str in trans_str:
            return 'Trans'
        else:
            return 'Other'

    df_merged['gender'] = df_merged['gender'].apply(clean_gender)

    # Impute missing values
    df_merged['self_employed'] = df_merged['self_employed'].fillna('No')
    df_merged['work_interfere'] = df_merged['work_interfere'].fillna('N/A')
    
    # Impute other categorical columns with mode
    categorical_cols = [
        'benefits', 'care_options', 'wellness_program', 'seek_help', 
        'anonymity', 'leave', 'mental_health_consequence', 'coworkers', 
        'supervisor', 'no_employees', 'tech_company'
    ]
    for col in categorical_cols:
        df_merged[col] = df_merged[col].fillna(df_merged[col].mode()[0])

    # Standardize treatment column
    treatment_map = {'Yes': 1, 'No': 0, '1': 1, '0': 0, True: 1, False: 0}
    df_merged['treatment'] = df_merged['treatment'].astype(str).map(treatment_map)
    df_merged.dropna(subset=['treatment'], inplace=True)
    df_merged['treatment'] = df_merged['treatment'].astype(int)

    return df_merged

# Load and clean data
df_2014, df_2016 = load_data()
df = clean_and_merge_data(df_2014, df_2016)

# ==============================================================================
# 2. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("Mental Health Dashboard")
section = st.sidebar.radio("Navigate to:", [
    "📊 Data Overview",
    "📈 Descriptive Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 Predictive Analytics",
    "👥 Employee Profiling"
])

st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset Information**  
- **Records**: {} employees  
- **Years**: 2014 & 2016  
- **Variables**: 16  
- **Last Updated**: 2023-10-15  
""".format(len(df)))

st.sidebar.markdown("---")
st.sidebar.caption("""
Created by [Your Name]  
[GitHub Repository](https://github.com/your-repo)  
[Contact Me](mailto:you@example.com)  
""")

# ==============================================================================
# 3. MAIN CONTENT AREA
# ==============================================================================
st.title("🧠 Mental Health in Tech Workplace Analysis")

if section == "📊 Data Overview":
    # DATA OVERVIEW SECTION
    st.header("📊 Dataset Overview")
    st.write("This dashboard analyzes mental health attitudes in tech workplaces using survey data from 2014 and 2016.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Dataset Summary")
        st.metric("Total Records", len(df))
        st.metric("Variables", 16)
        st.metric("Years Covered", "2014 & 2016")
        
        st.subheader("Key Variables")
        st.markdown("""
        - `treatment`: Whether employee sought treatment
        - `work_interfere`: How mental health affects work
        - `benefits`: Mental health benefits provided
        - `care_options`: Knowledge of care options
        - `gender`, `age`: Demographic information
        """)
    
    with col2:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), height=300)
        
        st.subheader("Missing Values")
        missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
        st.dataframe(missing_df.style.background_gradient(cmap="Reds"))
        
        st.subheader("Treatment Distribution")
        treatment_counts = df['treatment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.pie(treatment_counts, labels=['No Treatment', 'Sought Treatment'], 
               autopct='%1.1f%%', colors=['#ff7f0e', '#1f77b4'], startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

elif section == "📈 Descriptive Analytics":
    # DESCRIPTIVE ANALYTICS
    st.header("📈 Descriptive Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['age'], bins=30, kde=True, color="#1f77b4")
        ax.set_title('Employee Age Distribution')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        st.subheader("Company Size Distribution")
        company_size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='no_employees', data=df, order=company_size_order, 
                     palette='viridis', ax=ax)
        ax.set_title('Company Size Distribution')
        ax.set_xlabel('Number of Employees')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Gender Distribution")
        gender_counts = df['gender'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
               colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], startangle=90)
        ax.set_title('Employee Gender Distribution')
        ax.axis('equal')
        st.pyplot(fig)
        
        st.subheader("Tech Company Distribution")
        tech_counts = df['tech_company'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(tech_counts.index, tech_counts.values, color=['#1f77b4', '#ff7f0e'])
        ax.set_title('Tech Company Employment')
        ax.set_xlabel('Works in Tech Company')
        ax.set_ylabel('Count')
        st.pyplot(fig)

elif section == "🔍 Diagnostic Analytics":
    # DIAGNOSTIC ANALYTICS
    st.header("🔍 Diagnostic Analytics")
    
    tab1, tab2 = st.tabs(["Treatment Analysis", "Work Interference"])
    
    with tab1:
        st.subheader("Treatment Seeking Behavior")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**By Gender**")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='gender', hue='treatment', data=df, palette='Set2', ax=ax)
            ax.set_title('Treatment Seeking by Gender')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Count')
            ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
            st.pyplot(fig)
            
        with col2:
            st.write("**By Company Size**")
            fig, ax = plt.subplots(figsize=(10, 6))
            company_size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
            sns.countplot(x='no_employees', hue='treatment', data=df, 
                         order=company_size_order, palette='viridis', ax=ax)
            ax.set_title('Treatment Seeking by Company Size')
            ax.set_xlabel('Number of Employees')
            ax.set_ylabel('Count')
            ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        st.subheader("Treatment by Family History")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='family_history', hue='treatment', data=df, palette='coolwarm', ax=ax)
        ax.set_title('Treatment Seeking by Family History')
        ax.set_xlabel('Family History of Mental Illness')
        ax.set_ylabel('Count')
        ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Work Interference Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Work Interference Distribution**")
            filtered_df = df[df['work_interfere'] != 'N/A']
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='work_interfere', data=filtered_df, 
                         order=['Never', 'Rarely', 'Sometimes', 'Often'], 
                         palette='magma', ax=ax)
            ax.set_title('Work Interference Levels')
            ax.set_xlabel('Interference Frequency')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
        with col2:
            st.write("**Interference by Benefits Availability**")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='benefits', hue='work_interfere', data=filtered_df, 
                         palette='viridis', ax=ax,
                         hue_order=['Never', 'Rarely', 'Sometimes', 'Often'])
            ax.set_title('Work Interference by Mental Health Benefits')
            ax.set_xlabel('Benefits Provided')
            ax.set_ylabel('Count')
            ax.legend(title='Interference Level')
            st.pyplot(fig)
        
        st.subheader("Interference by Company Size")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='no_employees', hue='work_interfere', data=filtered_df, 
                     order=company_size_order, palette='coolwarm', ax=ax,
                     hue_order=['Never', 'Rarely', 'Sometimes', 'Often'])
        ax.set_title('Work Interference by Company Size')
        ax.set_xlabel('Number of Employees')
        ax.set_ylabel('Count')
        ax.legend(title='Interference Level')
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif section == "🤖 Predictive Analytics":
    # PREDICTIVE ANALYTICS
    st.header("🤖 Predictive Analytics")
    
    # Prepare data for modeling
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
    
    # Load prepared data
    X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test, target_names_task2 = prepare_prediction_data(df)
    
    # Model training
    @st.cache_resource
    def train_models():
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
    
    models = train_models()
    
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

elif section == "👥 Employee Profiling":
    # PRESCRIPTIVE ANALYTICS
    st.header("👥 Employee Profiling")
    st.write("Clustering analysis to identify distinct employee segments based on mental health attitudes and behaviors.")
    
    # Perform clustering
    @st.cache_data
    def perform_clustering(df):
        cluster_features = [
            'benefits', 'care_options', 'wellness_program', 'seek_help',
            'anonymity', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor'
        ]
        df_cluster_data = df[cluster_features].copy()
        df_cluster_encoded = pd.get_dummies(df_cluster_data, drop_first=True)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_cluster_encoded)
        df_cluster_data['cluster'] = clusters
        
        return df_cluster_data
    
    df_cluster = perform_clustering(df)
    cluster_counts = df_cluster['cluster'].value_counts().sort_index()
    
    st.subheader("Cluster Distribution")
    col1, col2, col3 = st.columns(3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    with col1:
        st.metric("Cluster 0: Engaged & Supported", f"{cluster_counts[0]} employees", 
                 help="Employees aware of and utilizing mental health resources")
        st.markdown("""
        - High awareness of benefits
        - Comfortable discussing with managers
        - Moderate treatment seeking
        """)
        st.progress(0.8, text="Resource Utilization")
        
    with col2:
        st.metric("Cluster 1: At-Risk & Unsupported", f"{cluster_counts[1]} employees", 
                 help="Employees with limited access to mental health support")
        st.markdown("""
        - Low benefit awareness
        - Anonymity concerns
        - High work interference
        """)
        st.progress(0.35, text="Resource Utilization")
        
    with col3:
        st.metric("Cluster 2: Proactive & Vulnerable", f"{cluster_counts[2]} employees", 
                 help="Employees actively seeking help with existing challenges")
        st.markdown("""
        - Family history of mental illness
        - High treatment seeking
        - Concerned about consequences
        """)
        st.progress(0.65, text="Resource Utilization")
    
    st.subheader("Cluster Characteristics Comparison")
    
    # Compare cluster characteristics
    cluster_chars = df_cluster.groupby('cluster').agg({
        'benefits': lambda x: (x == 'Yes').mean(),
        'care_options': lambda x: (x == 'Yes').mean(),
        'wellness_program': lambda x: (x == 'Yes').mean(),
        'seek_help': lambda x: (x == 'Yes').mean(),
        'anonymity': lambda x: (x == 'Yes').mean(),
        'coworkers': lambda x: (x == 'Yes').mean(),
        'supervisor': lambda x: (x == 'Yes').mean()
    }).reset_index()
    
    melted = cluster_chars.melt(id_vars='cluster', var_name='feature', value_name='percentage')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='feature', y='percentage', hue='cluster', data=melted, palette=colors, ax=ax)
    ax.set_title('Mental Health Resource Awareness by Cluster')
    ax.set_xlabel('Resource Feature')
    ax.set_ylabel('Percentage Responded "Yes"')
    ax.legend(title='Cluster', labels=['Engaged & Supported', 'At-Risk & Unsupported', 'Proactive & Vulnerable'])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Recommendations by Cluster")
    st.markdown("""
    | Cluster | Recommended Actions |
    |---------|---------------------|
    | **Engaged & Supported** | Maintain current programs, focus on retention, peer support networks |
    | **At-Risk & Unsupported** | Increase resource awareness, improve anonymity guarantees, manager training |
    | **Proactive & Vulnerable** | Provide specialized support, reduce stigma, flexible work arrangements |
    """)

# Footer
st.markdown("---")
st.caption("© 2023 Mental Health in Tech Dashboard | Data Sources: 2014 & 2016 OSMI Mental Health in Tech Surveys")
