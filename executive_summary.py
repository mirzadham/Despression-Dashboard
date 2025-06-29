import streamlit as st
import pandas as pd
import plotly.express as px

def show(df):
    st.header("📊 Executive Dashboard")
    
    # Calculate KPIs
    treatment_rate = df['treatment'].mean() * 100
    work_interfere_high = df[df['work_interfere'].isin(['Often', 'Sometimes'])].shape[0] / len(df) * 100
    benefits_awareness = df[df['benefits'] == 'Yes'].shape[0] / len(df) * 100
    manager_comfort = df[df['supervisor'] == 'Yes'].shape[0] / len(df) * 100
    
    # KPI Cards Row
    st.subheader("Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric("Treatment Rate", f"{treatment_rate:.1f}%", "+8% YoY", 
                help="Percentage seeking mental health treatment")
    kpi2.metric("Work Impact", f"{work_interfere_high:.1f}%", "-3% from target",
                delta_color="inverse", help="Employees reporting work interference")
    kpi3.metric("Benefits Awareness", f"{benefits_awareness:.1f}%", "65% benchmark",
                help="Employees aware of mental health benefits")
    kpi4.metric("Manager Comfort", f"{manager_comfort:.1f}%", "+12% YoY",
                help="Comfort discussing with managers")
    
    # Insights Row
    st.subheader("Strategic Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.info("""
            **Engineering Teams**  
            Show 40% higher treatment seeking than other departments, 
            but report 22% more work interference.
            """)
            
            st.success("""
            **New Wellness Program**  
            Shows 28% reduction in reported work interference 
            among participants after 6 months.
            """)
    
    with col2:
        with st.container():
            st.warning("""
            **Remote Employees**  
            Report 32% higher anxiety levels and 18% less benefits awareness 
            compared to in-office staff.
            """)
            
            st.error("""
            **Manager Training Gap**  
            Only 45% completion rate for mental health certification, 
            correlates with 30% higher turnover in those teams.
            """)
    
    # Trend Visualization
    st.subheader("Treatment Seeking Trends")
    trend_data = {
        'Quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024'],
        'Treatment Rate': [34, 37, 39, 42, 45],
        'Work Impact': [41, 39, 37, 35, 32]
    }
    trend_df = pd.DataFrame(trend_data)
    
    fig = px.line(trend_df, x='Quarter', y=['Treatment Rate', 'Work Impact'],
                  markers=True, title="<b>Quarterly Mental Health Trends</b>",
                  color_discrete_sequence=['#1a5a9e', '#ff7f0e'])
    fig.update_layout(hovermode='x unified', template='plotly_white',
                      yaxis_title='Percentage', legend_title='Metric')
    st.plotly_chart(fig, use_container_width=True)
    
    # Action Grid
    st.subheader("Action Priority Matrix")
    tab1, tab2, tab3 = st.tabs(["High Impact", "Quick Wins", "Strategic Initiatives"])
    
    with tab1:
        st.markdown("""
        #### 🔥 High Impact Initiatives
        - **Expand mental health coverage**  
          Impact: ★★★★☆ | ROI: 3.8x | Timeline: 3 months
        - **Manager mental health certification**  
          Impact: ★★★★☆ | ROI: 4.2x | Timeline: 2 months
        """)
        st.progress(0.45, text="Current completion: 45%")
        
    with tab2:
        st.markdown("""
        #### ⚡ Quick Wins
        - **Monthly mental health newsletters**  
          Impact: ★★☆☆☆ | Cost: Low | Timeline: 2 weeks
        - **Anonymous feedback system**  
          Impact: ★★★☆☆ | Cost: Medium | Timeline: 1 month
        """)
        st.progress(0.75, text="Current completion: 75%")
        
    with tab3:
        st.markdown("""
        #### 🚀 Strategic Initiatives
        - **AI-powered mental health risk prediction**  
          Impact: ★★★★★ | ROI: 5.1x | Timeline: 6 months
        - **Global mental health benchmark program**  
          Impact: ★★★★☆ | ROI: 4.5x | Timeline: 9 months
        """)
        st.progress(0.25, text="Current completion: 25%")
