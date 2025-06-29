import streamlit as st
import plotly.express as px
import pandas as pd

def show(df):
    st.header("📉 Impact Analysis")
    
    tab1, tab2 = st.tabs(["Treatment Analysis", "Work Interference"])
    
    with tab1:
        st.subheader("Treatment Seeking Behavior")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**By Gender**")
            fig = px.histogram(df, x='gender', color='treatment', barmode='group',
                              title='<b>Treatment Seeking by Gender</b>',
                              color_discrete_sequence=['#ff7f0e', '#1f77b4'],
                              labels={'gender': 'Gender', 'count': 'Count'})
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("**By Company Size**")
            fig = px.histogram(df, x='no_employees', color='treatment', 
                              category_orders={'no_employees': ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']},
                              title='<b>Treatment Seeking by Company Size</b>',
                              color_discrete_sequence=['#ff7f0e', '#1f77b4'],
                              labels={'no_employees': 'Company Size', 'count': 'Count'})
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Treatment by Family History")
        fig = px.histogram(df, x='family_history', color='treatment', barmode='group',
                          title='<b>Treatment Seeking by Family History</b>',
                          color_discrete_sequence=['#ff7f0e', '#1f77b4'],
                          labels={'family_history': 'Family History', 'count': 'Count'})
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Work Interference Analysis")
        filtered_df = df[df['work_interfere'] != 'N/A']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Work Interference Distribution**")
            fig = px.histogram(filtered_df, x='work_interfere', 
                              category_orders={'work_interfere': ['Never', 'Rarely', 'Sometimes', 'Often']},
                              title='<b>Work Interference Levels</b>',
                              color_discrete_sequence=['#1a5a9e'],
                              labels={'work_interfere': 'Interference Level', 'count': 'Count'})
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("**Interference by Benefits Availability**")
            fig = px.histogram(filtered_df, x='benefits', color='work_interfere',
                              category_orders={'work_interfere': ['Never', 'Rarely', 'Sometimes', 'Often']},
                              title='<b>Work Interference by Benefits</b>',
                              color_discrete_sequence=px.colors.sequential.Magma,
                              labels={'benefits': 'Benefits Provided', 'count': 'Count'})
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Interference by Company Size")
        fig = px.histogram(filtered_df, x='no_employees', color='work_interfere',
                          category_orders={
                              'no_employees': ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'],
                              'work_interfere': ['Never', 'Rarely', 'Sometimes', 'Often']
                          },
                          title='<b>Work Interference by Company Size</b>',
                          color_discrete_sequence=px.colors.sequential.Cividis,
                          labels={'no_employees': 'Company Size', 'count': 'Count'})
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
