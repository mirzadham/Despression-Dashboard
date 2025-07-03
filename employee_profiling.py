import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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
    
    # Calculate characteristics for each cluster
    cluster_stats = []
    for c in range(3):
        cluster_data = df_cluster[df_cluster['cluster'] == c]
        stats = {
            'cluster': c,
            'dk_benefits': (cluster_data['benefits'] == "Don't know").mean(),
            'dk_care_options': (cluster_data['care_options'] == "Don't know").mean(),
            'yes_benefits': (cluster_data['benefits'] == 'Yes').mean(),
            'yes_wellness': (cluster_data['wellness_program'] == 'Yes').mean(),
            'yes_supervisor': (cluster_data['supervisor'] == 'Yes').mean(),
            'yes_consequence': (cluster_data['mental_health_consequence'] == 'Yes').mean(),
            'no_coworkers': (cluster_data['coworkers'] == 'No').mean(),
            'size': len(cluster_data)
        }
        cluster_stats.append(stats)
    
    # Create cluster mapping based on characteristics
    cluster_map = {}
    for stats in cluster_stats:
        if stats['dk_benefits'] > 0.4 and stats['dk_care_options'] > 0.4:
            cluster_map[stats['cluster']] = ("The Cautious / Uninformed", stats['size'], "#1f77b4")
        elif stats['yes_benefits'] > 0.7 and stats['yes_wellness'] > 0.7 and stats['yes_supervisor'] > 0.7:
            cluster_map[stats['cluster']] = ("The Supported", stats['size'], "#2ca02c")
        elif stats['yes_consequence'] > 0.5 and stats['no_coworkers'] > 0.4:
            cluster_map[stats['cluster']] = ("The Stigmatized", stats['size'], "#ff7f0e")
        else:
            # Fallback for any unclassified clusters
            cluster_map[stats['cluster']] = (f"Cluster {stats['cluster']}", stats['size'], "#9467bd")
    
    # Apply mapping
    df_cluster['cluster_name'] = df_cluster['cluster'].map(lambda x: cluster_map[x][0])
    df_cluster['color'] = df_cluster['cluster'].map(lambda x: cluster_map[x][2])
    
    # Get counts
    cluster_counts = df_cluster['cluster_name'].value_counts()
    
    st.subheader("Cluster Distribution")
    cols = st.columns(3)
    
    # Display each cluster
    for i, (name, count) in enumerate(cluster_counts.items()):
        color = [v[2] for k, v in cluster_map.items() if v[0] == name][0]
        
        with cols[i]:
            if "Cautious" in name:
                st.metric(f"Cluster: {name}", f"{count} employees", 
                         help="Employees with high uncertainty about mental health resources")
                st.markdown("""
                - Frequent "Don't know" responses
                - Low awareness of benefits
                - Limited knowledge of care options
                """)
                st.progress(0.35, text="Resource Awareness")
                
            elif "Supported" in name:
                st.metric(f"Cluster: {name}", f"{count} employees", 
                         help="Employees with positive workplace support")
                st.markdown("""
                - High benefit awareness
                - Active wellness programs
                - Comfortable with supervisors
                """)
                st.progress(0.85, text="Support Satisfaction")
                
            elif "Stigmatized" in name:
                st.metric(f"Cluster: {name}", f"{count} employees", 
                         help="Employees fearing negative consequences")
                st.markdown("""
                - Fear of negative repercussions
                - Uncomfortable with coworkers
                - Concerns about job security
                """)
                st.progress(0.25, text="Psychological Safety")
    
    st.subheader("Cluster Characteristics Comparison")
    
    # Prepare data for visualization
    plot_data = []
    for name, group in df_cluster.groupby('cluster_name'):
        for col in ['benefits', 'care_options', 'wellness_program', 
                   'supervisor', 'coworkers', 'mental_health_consequence']:
            yes_pct = (group[col] == 'Yes').mean()
            dk_pct = (group[col] == "Don't know").mean()
            
            plot_data.append({
                'Cluster': name,
                'Feature': f'{col} (Yes)',
                'Percentage': yes_pct
            })
            
            plot_data.append({
                'Cluster': name,
                'Feature': f'{col} (Don\'t know)',
                'Percentage': dk_pct
            })
            
    plot_df = pd.DataFrame(plot_data)
    
    # Create visualization
    fig = px.bar(plot_df, x='Feature', y='Percentage', color='Cluster',
                 barmode='group', title='<b>Mental Health Attitudes by Segment</b>',
                 color_discrete_map={
                     "The Cautious / Uninformed": "#1f77b4",
                     "The Supported": "#2ca02c",
                     "The Stigmatized": "#ff7f0e"
                 })
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text='Employee Segment',
        yaxis_tickformat=',.0%'
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
