
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
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
st.title("Mental Health in Tech Workplace Analysis")
st.write("This dashboard presents an analysis of mental health attitudes and treatment-seeking behavior in the tech workplace, based on survey data from 2014 and 2016.")
st.write("Navigate through the sections below to explore descriptive statistics, diagnostic insights, predictive models, and employee clustering.")

# Load the datasets from the uploaded files
@st.cache_data
def load_data():
    try:
        df_2014 = pd.read_csv('2014.csv')
        df_2016 = pd.read_csv('2016.csv')
        return df_2014, df_2016
    except FileNotFoundError:
        st.error("ERROR: Make sure you have uploaded both '2014.csv' and '2016.csv'")
        st.stop()

df_2014, df_2016 = load_data()

# ==============================================================================
# 2. DATA CLEANING AND PRE-PROCESSING
# ==============================================================================
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
            return 'Other' # Handle other categories as 'Other'

    df_merged['gender'] = df_merged['gender'].apply(clean_gender)

    # Impute missing values
    df_merged['self_employed'] = df_merged['self_employed'].fillna('No')
    df_merged['work_interfere'] = df_merged['work_interfere'].fillna('N/A')
    # Impute other categorical columns with a placeholder 'Missing' or mode
    for col in ['benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor', 'no_employees', 'tech_company']:
         df_merged[col] = df_merged[col].fillna(df_merged[col].mode()[0]) # Fill with mode for simplicity

    # Standardize the 'treatment' column
    treatment_map = {'Yes': 1, 'No': 0, '1': 1, '0': 0, True: 1, False: 0}
    df_merged['treatment'] = df_merged['treatment'].astype(str).map(treatment_map)
    df_merged.dropna(subset=['treatment'], inplace=True)
    df_merged['treatment'] = df_merged['treatment'].astype(int)

    return df_merged

df = clean_and_merge_data(df_2014, df_2016)

st.subheader("Data Overview after Cleaning")
st.write("Initial rows of the cleaned dataset:")
st.dataframe(df.head())
st.write("Shape of the cleaned dataset:", df.shape)
st.write("Missing values after imputation:")
st.dataframe(df.isnull().sum())


# ==============================================================================
# 3. DESCRIPTIVE ANALYTICS
# ==============================================================================
st.header("1. Descriptive Analytics")

st.subheader("Distribution of Employee Age")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True, ax=ax)
ax.set_title('Distribution of Employee Age', fontsize=16)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
st.pyplot(fig)

st.subheader("Distribution of Employee Gender")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='gender', data=df, palette='plasma', order=df['gender'].value_counts().index, ax=ax)
ax.set_title('Distribution of Employee Gender', fontsize=16)
st.pyplot(fig)

st.write("These plots show the distribution of age and the cleaned gender categories in the dataset.")


# ==============================================================================
# 4. DIAGNOSTIC ANALYTICS
# ==============================================================================
st.header("2. Diagnostic Analytics")

st.subheader("Treatment Seeking Behavior by Gender")
fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(x='gender', hue='treatment', data=df, palette='magma', ax=ax)
ax.set_title('Treatment Seeking Behavior by Gender', fontsize=16)
ax.set_xlabel('Gender', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
legend = ax.legend(title='Sought Treatment')
legend_labels = ['No', 'Yes']
for t, l in zip(legend.texts, legend_labels): t.set_text(l)
st.pyplot(fig)


st.subheader("Treatment Seeking Behavior by Company Size")
fig, ax = plt.subplots(figsize=(14, 8))
sns.countplot(x='no_employees', hue='treatment', data=df, palette='cividis', order=['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'], ax=ax)
ax.set_title('Treatment Seeking Behavior by Company Size', fontsize=16)
ax.set_xlabel('Number of Employees', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
legend = ax.legend(title='Sought Treatment')
for t, l in zip(legend.texts, legend_labels): t.set_text(l)
st.pyplot(fig)

st.write("These plots show how treatment-seeking behavior varies across different demographics and company sizes.")


# ==============================================================================
# 5. PREDICTIVE ANALYTICS
# ==============================================================================
st.header("3. Predictive Analytics")

# --- Prepare data for Task 1 and Task 2 ---
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
    # Align columns after one-hot encoding - crucial for consistent feature sets
    X2_encoded = X2_encoded.reindex(columns = X1_encoded.columns, fill_value=0)


    scaler2 = StandardScaler() # Use a separate scaler for task 2 data if needed, or the same if features are consistent
    X2_scaled = scaler2.fit_transform(X2_encoded)

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42, stratify=y2)

    return X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test, target_names_task2, X1_encoded.columns.tolist() # Return feature names for consistent encoding

X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test, target_names_task2, task1_features_encoded = prepare_prediction_data(df)


# --- Train and evaluate models (Task 1 & Task 2) ---
@st.cache_resource # Use st.cache_resource for models
def train_models(X1_train, y1_train, X2_train, y2_train):
    models_task1 = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    models_task2 = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    for name, model in models_task1.items():
        model.fit(X1_train, y1_train)

    for name, model in models_task2.items():
        model.fit(X2_train, y2_train)

    return models_task1, models_task2

models_task1, models_task2 = train_models(X1_train, y1_train, X2_train, y2_train)


# --- Display results for Treatment Prediction (Task 1) ---
st.subheader("Predicting Treatment (Binary Classification)")
st.write("Evaluation results for models predicting whether an employee seeks treatment for a mental health issue:")

for name, model in models_task1.items():
    st.write(f"**{name} Results:**")
    try:
        y1_pred = model.predict(X1_test)
        report = classification_report(y1_test, y1_pred, target_names=['No', 'Yes'], output_dict=True)
        st.json(report)

        cm = confusion_matrix(y1_test, y1_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax)
        ax.set_title(f'Confusion Matrix for {name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not get results for {name}: {e}")


# --- Display results for Work Interference Prediction (Task 2) ---
st.subheader("Predicting Work Interference (Multi-Class Classification)")
st.write("Evaluation results for models predicting how a mental health issue interferes with work:")

for name, model in models_task2.items():
    st.write(f"**{name} Results:**")
    try:
        y2_pred = model.predict(X2_test)
        report = classification_report(y2_test, y2_pred, target_names=target_names_task2, output_dict=True)
        st.json(report)

        cm = confusion_matrix(y2_test, y2_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names_task2, yticklabels=target_names_task2, ax=ax)
        ax.set_title(f'Confusion Matrix for {name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not get results for {name}: {e}")

st.write("These sections present the performance of different models in predicting treatment-seeking behavior and work interference.")


# ==============================================================================
# 6. PRESCRIPTIVE ANALYTICS (CLUSTERING)
# ==============================================================================
st.header("4. Prescriptive Analytics (Employee Profiling)")
st.write("This section presents employee profiles based on clustering analysis of their attitudes towards mental health resources at work.")

@st.cache_data
def perform_clustering(df):
    cluster_features = [
        'benefits', 'care_options', 'wellness_program', 'seek_help',
        'anonymity', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor'
    ]
    df_cluster_data = df[cluster_features].copy()
    df_cluster_encoded = pd.get_dummies(df_cluster_data, drop_first=True)

    # Apply K-Means Clustering (assuming k=3 based on previous analysis)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_cluster_encoded)
    df_cluster_data['cluster'] = clusters

    return df_cluster_data, cluster_features, k

df_cluster_data, cluster_features, k = perform_clustering(df)

st.subheader(f"Cluster Profiles (k={k})")
st.write(f"The clustering analysis identified {k} distinct groups of employees based on their responses to questions about workplace mental health support.")

for i in range(k):
    st.write(f"### Cluster {i} Profile")
    cluster_profile = df_cluster_data[df_cluster_data['cluster'] == i]
    st.write(f"**Size of cluster:** {len(cluster_profile)}")

    st.write(f"**Characteristics of Cluster {i}:**")

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f'Characteristics of Cluster {i}', fontsize=20)
    axes = axes.flatten()

    for j, col in enumerate(cluster_features):
        sns.countplot(x=col, data=cluster_profile, ax=axes[j], palette='cubehelix', order=df[col].value_counts().index)
        axes[j].set_title(f'Distribution of "{col}"')
        axes[j].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

    st.write("*(Interpretation of this cluster's key characteristics goes here based on the plots)*")


st.write("Understanding these profiles can help tailor mental health initiatives and support programs to specific employee groups.")
