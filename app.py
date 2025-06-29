import streamlit as st
from utils import setup_page, load_data, clean_and_merge_data
from data_overview import show as show_data_overview
from descriptive_analytics import show as show_descriptive
from diagnostic_analytics import show as show_diagnostic
from predictive_analytics import show as show_predictive
from employee_profiling import show as show_profiling
from executive_summary import show as show_executive_summary

# Initialize page
setup_page()

# Load data
df_2014, df_2016 = load_data()
df = clean_and_merge_data(df_2014, df_2016)

# Sidebar navigation
st.sidebar.title("Mental Health Dashboard")
sections = {
    "📊 Executive Summary": show_executive_summary,
    "🔍 Data Overview": show_data_overview,
    "📈 Workforce Analytics": show_descriptive,
    "📉 Impact Analysis": show_diagnostic,
    "🔮 Predictive Insights": show_predictive,
    "👥 Employee Segments": show_profiling
}
selected = st.sidebar.radio("Navigate to:", list(sections.keys()))

# Smart filters
st.sidebar.markdown("---")
st.sidebar.subheader("Smart Filters")

with st.sidebar.expander("🔍 Demographic Filters"):
    age_range = st.slider("Age Range", 18, 75, (25, 55))
    gender_filter = st.multiselect("Gender", options=df['gender'].unique(), default=df['gender'].unique())
    company_size = st.multiselect("Company Size", 
                                 options=df['no_employees'].unique(), 
                                 default=df['no_employees'].unique())

with st.sidebar.expander("🏢 Workplace Filters"):
    tech_company = st.selectbox("Tech Company", ["All", "Yes", "No"])
    benefits = st.selectbox("Mental Health Benefits", ["All", "Yes", "No", "Don't know"])
    treatment_status = st.selectbox("Treatment Status", ["All", "Received", "Not Received"])

# Apply filters to dataframe
df_filtered = df.copy()
df_filtered = df_filtered[(df_filtered['age'] >= age_range[0]) & (df_filtered['age'] <= age_range[1])]
df_filtered = df_filtered[df_filtered['gender'].isin(gender_filter)]
df_filtered = df_filtered[df_filtered['no_employees'].isin(company_size)]

if tech_company != "All":
    df_filtered = df_filtered[df_filtered['tech_company'] == tech_company]
    
if benefits != "All":
    df_filtered = df_filtered[df_filtered['benefits'] == benefits]
    
if treatment_status == "Received":
    df_filtered = df_filtered[df_filtered['treatment'] == 1]
elif treatment_status == "Not Received":
    df_filtered = df_filtered[df_filtered['treatment'] == 0]

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Filtered Dataset**  
- **Records**: {len(df_filtered)} employees ({len(df_filtered)/len(df):.0%} of total)
- **Applied Filters**: {len(gender_filter)} gender, {len(company_size)} company sizes
""")

st.sidebar.markdown("---")
st.sidebar.caption("""
Created by [Your Name]  
[GitHub Repository](https://github.com/your-repo)  
[Contact Me](mailto:you@example.com)  
""")

# Main content area
st.title("🧠 Mental Health in Tech Workplace Analysis")
sections[selected](df_filtered)

# Footer
st.markdown("---")
st.caption("© 2024 Mental Health Intelligence Dashboard | Data Sources: 2014 & 2016 OSMI Mental Health in Tech Surveys")
