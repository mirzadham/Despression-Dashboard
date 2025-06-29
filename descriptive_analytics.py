# descriptive_analytics.py (replace matplotlib with Plotly)
import plotly.express as px

def show(df):
    st.header("📈 Workforce Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        # Age Distribution - Plotly
        st.subheader("Age Distribution")
        fig = px.histogram(df, x='age', nbins=30, 
                          title='<b>Employee Age Distribution</b>',
                          color_discrete_sequence=['#1a5a9e'])
        fig.update_layout(hovermode='x unified', 
                         template='plotly_white',
                         font=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Company Size Distribution - Plotly
        st.subheader("Company Size Distribution")
        company_size_order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
        fig = px.bar(df['no_employees'].value_counts().loc[company_size_order].reset_index(),
                    x='index', y='no_employees',
                    title='<b>Company Size Distribution</b>',
                    labels={'index':'Number of Employees', 'no_employees':'Count'},
                    color_discrete_sequence=['#1a5a9e'])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender Distribution - Plotly
        st.subheader("Gender Distribution")
        gender_counts = df['gender'].value_counts().reset_index()
        fig = px.pie(gender_counts, values='count', names='gender',
                    title='<b>Employee Gender Distribution</b>',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Tech Company Distribution - Plotly
        st.subheader("Tech Company Distribution")
        tech_counts = df['tech_company'].value_counts().reset_index()
        fig = px.bar(tech_counts, x='tech_company', y='count',
                    title='<b>Tech Company Employment</b>',
                    labels={'tech_company':'Works in Tech Company', 'count':'Count'},
                    color='tech_company',
                    color_discrete_map={'Yes':'#1a5a9e', 'No':'#ff7f0e'})
        st.plotly_chart(fig, use_container_width=True)
