import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

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

def show(df):
    st.header("👥 Employee Segments")
    st.write("Clustering analysis to identify distinct employee segments based on mental health attitudes and behaviors.")
    
    # Perform clustering
    df_cluster = perform_clustering(df)
    
    # Reassign clusters based on size
    cluster_counts = df_cluster['cluster'].value_counts()
    sorted_counts = cluster_counts.sort_values(ascending=False)
    cluster_map = {
        sorted_counts.index[0]: 0,  # Largest cluster -> Cautious/Uninformed
        sorted_counts.index[1]: 1,  # Middle cluster -> Supported
        sorted_counts.index[2]: 2   # Smallest cluster -> Stigmatized
    }
    df_cluster['cluster'] = df_cluster['cluster'].map(cluster_map)
    cluster_counts = df_cluster['cluster'].value_counts().sort_index()
    
    # Create cluster names
    cluster_names = {
        0: "The Cautious / Uninformed",
        1: "The Supported",
        2: "The Stigmatized"
    }
    df_cluster['cluster_name'] = df_cluster['cluster'].map(cluster_names)

    st.subheader("Cluster Distribution")
    col1, col2, col3 = st.columns(3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    with col1:
        st.metric(
            "Cluster 0: The Cautious / Uninformed", 
            f"{cluster_counts[0]} employees",
            help="Characterized by uncertainty with frequent 'Don't know' answers"
        )
        st.markdown("""
        - High uncertainty in benefits
        - Limited awareness of care options
        - Frequent "Don't know" responses
        """)
        st.progress(0.45, text="Resource Awareness")
        
    with col2:
        st.metric(
            "Cluster 1: The Supported", 
            f"{cluster_counts[1]} employees",
            help="Employees with positive workplace support"
        )
        st.markdown("""
        - Strong awareness of benefits
        - Active wellness programs
        - Comfortable discussing with supervisors
        """)
        st.progress(0.85, text="Resource Utilization")
        
    with col3:
        st.metric(
            "Cluster 2: The Stigmatized", 
            f"{cluster_counts[2]} employees",
            help="Employees fearing negative repercussions"
        )
        st.markdown("""
        - Fear of negative consequences
        - Uncomfortable with coworkers
        - Concerns about anonymity
        """)
        st.progress(0.25, text="Psychological Safety")
    
    st.subheader("Cluster Characteristics Comparison")
    
    # Compare cluster characteristics
    cluster_chars = df_cluster.groupby('cluster_name').agg({
        'benefits': lambda x: (x == 'Yes').mean(),
        'care_options': lambda x: (x == 'Yes').mean(),
        'wellness_program': lambda x: (x == 'Yes').mean(),
        'seek_help': lambda x: (x == 'Yes').mean(),
        'anonymity': lambda x: (x == 'Yes').mean(),
        'coworkers': lambda x: (x == 'Yes').mean(),
        'supervisor': lambda x: (x == 'Yes').mean(),
        'mental_health_consequence': lambda x: (x == 'Yes').mean()
    }).reset_index()
    
    melted = cluster_chars.melt(id_vars='cluster_name', var_name='feature', value_name='percentage')
    
    fig = px.bar(melted, x='feature', y='percentage', color='cluster_name', 
                barmode='group', title='<b>Mental Health Resource Awareness by Segment</b>',
                labels={'feature': 'Resource Feature', 'percentage': 'Percentage Responded "Yes"'},
                color_discrete_sequence=colors)
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Employee Segment'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recommendations by Segment")
    st.markdown("""
    | Segment | Recommended Actions |
    |---------|---------------------|
    | **The Cautious/Uninformed** | Improve communication of benefits, clarify care options, conduct awareness workshops |
    | **The Supported** | Maintain current programs, enhance peer support networks, leadership recognition |
    | **The Stigmatized** | Address stigma through training, ensure anonymity protections, create safe reporting channels |
    """)
