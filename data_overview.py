import streamlit as st
import pandas as pd
import plotly.express as px

def show(df):
    st.header("🔍 Data Overview")
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
        treatment_counts = df['treatment'].value_counts().reset_index()
        fig = px.pie(treatment_counts, values='count', names='treatment',
                    title='<b>Treatment Distribution</b>',
                    color_discrete_sequence=['#ff7f0e', '#1f77b4'],
                    labels={'treatment': 'Treatment Status', 'count': 'Count'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
