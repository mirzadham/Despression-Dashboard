# descriptive_analytics.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show(df):
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
