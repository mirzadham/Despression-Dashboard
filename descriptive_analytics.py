import streamlit as st
import plotly.express as px
import pandas as pd

def show(df):
    st.header("📈 Workforce Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(df, x='age', nbins=30, 
                          title='<b>Employee Age Distribution</b>',
                          color_discrete_sequence=['#1a5a9e'])
        fig.update_layout(
            hovermode='x unified', 
            template='plotly_white',
            font=dict(family="Inter"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Company Size Distribution")
        company_size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
        size_counts = df['no_employees'].value_counts().loc[company_size_order].reset_index()
        fig = px.bar(size_counts, x='index', y='no_employees',
                    title='<b>Company Size Distribution</b>',
                    labels={'index':'Number of Employees', 'no_employees':'Count'},
                    color_discrete_sequence=['#1a5a9e'])
        fig.update_layout(
            xaxis_tickangle=-45,
            template='plotly_white',
            font=dict(family="Inter")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Gender Distribution")
        gender_counts = df['gender'].value_counts().reset_index()
        fig = px.pie(gender_counts, values='count', names='gender',
                    title='<b>Employee Gender Distribution</b>',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            marker=dict(line=dict(color='#000000', width=0.5))
        fig.update_layout(
            font=dict(family="Inter")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Tech Company Distribution")
        tech_counts = df['tech_company'].value_counts().reset_index()
        fig = px.bar(tech_counts, x='tech_company', y='count',
                    title='<b>Tech Company Employment</b>',
                    labels={'tech_company':'Works in Tech Company', 'count':'Count'},
                    color='tech_company',
                    color_discrete_map={'Yes':'#1a5a9e', 'No':'#ff7f0e'})
        fig.update_layout(
            font=dict(family="Inter")
        )
        st.plotly_chart(fig, use_container_width=True)
