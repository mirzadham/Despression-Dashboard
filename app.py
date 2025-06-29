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
st.set_page_config(layout="wide", page_title="Mental Health in Tech Dashboard")

# Custom CSS to improve styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #1f77b4;
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
    # (Keep your existing cleaning code here)
    # ...
    return df_merged

# Load data
df_2014, df_2016 = load_data()
df = clean_and_merge_data(df_2014, df_2016)

# ==============================================================================
# 2. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Section", [
    "Data Overview",
    "Descriptive Analytics",
    "Diagnostic Analytics",
    "Predictive Analytics",
    "Employee Profiling"
])

st.sidebar.markdown("---")
st.sidebar.info("""
**Mental Health in Tech Dashboard**  
Analyzing survey data from 2014-2016  
*Dataset Size*: {} records  
*Last Updated*: 2023-10-15
""".format(len(df)))

# ==============================================================================
# 3. MAIN CONTENT AREA
# ==============================================================================
st.title("Mental Health in Tech Workplace Analysis")

if section == "Data Overview":
    # DATA OVERVIEW SECTION
    st.header("Data Overview")
    st.write("This dashboard presents an analysis of mental health attitudes in tech workplaces based on 2014-2016 survey data.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Dataset Summary")
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        
        st.subheader("Missing Values")
        missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
        st.dataframe(missing_df.style.background_gradient(cmap="Reds"))
    
    with col2:
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        st.subheader("Key Variables")
        st.markdown("""
        - **treatment**: Whether employee sought treatment (Target)
        - **work_interfere**: How mental health affects work
        - **benefits/care_options**: Mental health benefits
        - **gender/age**: Demographic info
        """)

elif section == "Descriptive Analytics":
    # DESCRIPTIVE ANALYTICS
    st.header("Descriptive Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['age'], bins=30, kde=True, color="#1f77b4")
        ax.set_title('Employee Age Distribution')
        ax.set_xlabel('Age')
        st.pyplot(fig)
        
        st.subheader("Treatment Rate")
        treatment_rate = df['treatment'].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        treatment_rate.plot(kind='bar', color=['#ff7f0e', '#2ca02c'], ax=ax)
        ax.set_title('Treatment Seeking Rate')
        ax.set_ylabel('Percentage')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        gender_counts = df['gender'].value_counts()
        gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], ax=ax)
        ax.set_title('Employee Gender Distribution')
        ax.set_ylabel('')
        st.pyplot(fig)
        
        st.subheader("Company Size Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        company_size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
        sns.countplot(x='no_employees', data=df, order=company_size_order, palette='viridis', ax=ax)
        ax.set_title('Company Size Distribution')
        ax.set_xlabel('Number of Employees')
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif section == "Diagnostic Analytics":
    # DIAGNOSTIC ANALYTICS
    st.header("Diagnostic Analytics")
    
    st.subheader("Treatment Seeking Behavior")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**By Gender**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='gender', hue='treatment', data=df, palette='Set2', ax=ax)
        ax.set_title('Treatment Seeking by Gender')
        ax.set_xlabel('Gender')
        ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
        st.pyplot(fig)
        
    with col2:
        st.write("**By Company Size**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='no_employees', hue='treatment', data=df, 
                     order=company_size_order, palette='viridis', ax=ax)
        ax.set_title('Treatment Seeking by Company Size')
        ax.set_xlabel('Number of Employees')
        ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.subheader("Work Interference Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='work_interfere', hue='treatment', data=df[df['work_interfere'] != 'N/A'], 
                 palette='coolwarm', ax=ax)
    ax.set_title('Work Interference vs Treatment Seeking')
    ax.set_xlabel('Work Interference Level')
    ax.legend(title='Sought Treatment', labels=['No', 'Yes'])
    st.pyplot(fig)

elif section == "Predictive Analytics":
    # PREDICTIVE ANALYTICS
    st.header("Predictive Analytics")
    
    tab1, tab2 = st.tabs(["Treatment Prediction", "Work Interference Prediction"])
    
    with tab1:
        st.subheader("Predicting Treatment Seeking")
        st.write("Machine learning models predicting whether an employee will seek treatment")
        
        model_choice = st.selectbox("Select Model", 
                                   ["Logistic Regression", "Random Forest", "XGBoost"])
        
        # Placeholder for model results - you'd add your actual model code here
        if model_choice == "Logistic Regression":
            st.info("Logistic Regression Results")
            st.metric("Accuracy", "0.78")
            st.metric("Precision", "0.81")
            st.metric("Recall", "0.72")
            
            # Confusion matrix visualization
            fig, ax = plt.subplots(figsize=(6, 4))
            cm = [[120, 30], [25, 95]]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Predicted No', 'Predicted Yes'],
                        yticklabels=['Actual No', 'Actual Yes'])
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
        # Similar blocks for other models...
    
    with tab2:
        st.subheader("Predicting Work Interference")
        st.write("Models predicting how mental health issues affect work performance")
        
        # Similar structure as tab1 but for interference prediction
        # ...

elif section == "Employee Profiling":
    # PRESCRIPTIVE ANALYTICS
    st.header("Employee Profiling")
    st.write("Clustering analysis to identify employee segments based on mental health attitudes")
    
    # Cluster visualization
    st.subheader("Cluster Distribution")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cluster 0 Size", "350 employees")
        st.write("**Characteristics:**")
        st.markdown("- High awareness of benefits")
        st.markdown("- Comfortable discussing with managers")
        st.markdown("- Moderate treatment seeking")
    
    with col2:
        st.metric("Cluster 1 Size", "420 employees")
        st.write("**Characteristics:**")
        st.markdown("- Low benefit awareness")
        st.markdown("- Anonymity concerns")
        st.markdown("- High work interference")
    
    with col3:
        st.metric("Cluster 2 Size", "280 employees")
        st.write("**Characteristics:**")
        st.markdown("- Family history of mental illness")
        st.markdown("- High treatment seeking")
        st.markdown("- Concerned about consequences")
    
    st.subheader("Cluster Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    # Example visualization - replace with your actual cluster analysis
    cluster_data = pd.DataFrame({
        'Cluster': [0, 0, 1, 1, 2, 2],
        'Variable': ['Benefits', 'Anonymity', 'Benefits', 'Anonymity', 'Benefits', 'Anonymity'],
        'Value': [0.8, 0.6, 0.3, 0.9, 0.7, 0.4]
    })
    sns.barplot(x='Variable', y='Value', hue='Cluster', data=cluster_data, palette='Set2', ax=ax)
    ax.set_title('Key Variables by Cluster')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Mental Health in Tech Dashboard | Data Sources: 2014 & 2016 OSMI Surveys")
