"""
RFM Analysis & Customer Segmentation Module
Implements Recency, Frequency, Monetary analysis and K-Means clustering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_rfm(df_sales):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer
    
    Args:
        df_sales (pd.DataFrame): Cleaned sales data
        
    Returns:
        pd.DataFrame: RFM dataframe with customer metrics
    """
    # Reference date = max invoice date + 1 day
    reference_date = df_sales['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Compute RFM
    rfm = (
        df_sales
        .groupby('Customer ID')
        .agg({
            'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
            'Invoice': 'nunique',                                       # Frequency
            'TotalPrice': 'sum'                                         # Monetary
        })
    )
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    print(f"RFM calculated for {len(rfm)} customers")
    print(f"Average Recency: {rfm['Recency'].mean():.0f} days")
    print(f"Average Frequency: {rfm['Frequency'].mean():.1f} purchases")
    print(f"Average Monetary: £{rfm['Monetary'].mean():,.2f}")
    
    return rfm


def find_optimal_clusters(rfm, max_k=9):
    """
    Use elbow method and silhouette score to find optimal number of clusters
    
    Args:
        rfm (pd.DataFrame): RFM dataframe
        max_k (int): Maximum number of clusters to test
        
    Returns:
        tuple: (inertias, silhouette_scores, K_range, optimal_k)
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    inertias = []
    silhouette_scores_list = []
    K_range = range(2, max_k)
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(rfm_scaled)
        inertias.append(kmeans_temp.inertia_)
        silhouette_scores_list.append(silhouette_score(rfm_scaled, kmeans_temp.labels_))
    
    optimal_k = K_range[silhouette_scores_list.index(max(silhouette_scores_list))]
    
    print(f"\nOptimal number of clusters: {optimal_k}")
    print(f"Best Silhouette Score: {max(silhouette_scores_list):.3f}")
    
    return inertias, silhouette_scores_list, K_range, optimal_k


def segment_customers(rfm, n_clusters=4):
    """
    Perform K-Means clustering on RFM data
    
    Args:
        rfm (pd.DataFrame): RFM dataframe
        n_clusters (int): Number of clusters
        
    Returns:
        pd.DataFrame: RFM dataframe with cluster assignments
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    print(f"\nCustomers segmented into {n_clusters} clusters")
    
    return rfm


def assign_personas(rfm):
    """
    Assign business personas to clusters based on RFM characteristics
    
    Args:
        rfm (pd.DataFrame): RFM dataframe with clusters
        
    Returns:
        pd.DataFrame: RFM dataframe with persona labels
    """
    # Analyze clusters to assign personas
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    # Map clusters to personas based on characteristics
    persona_map = {
        3: 'Ultra VIP Loyalists',
        2: 'High-Value Loyal Customers',
        1: 'Active Regular Customers',
        0: 'At-Risk / Dormant Customers'
    }
    
    rfm['Persona'] = rfm['Cluster'].map(persona_map)
    
    # Print cluster summary
    print("\nCluster Summary:")
    for cluster_id in range(4):
        cluster_data = rfm[rfm['Cluster'] == cluster_id]
        count = len(cluster_data)
        pct = (count / len(rfm) * 100)
        avg_recency = cluster_data['Recency'].mean()
        avg_frequency = cluster_data['Frequency'].mean()
        avg_monetary = cluster_data['Monetary'].mean()
        
        print(f"\nCluster {cluster_id} - {persona_map[cluster_id]}:")
        print(f"  Customers: {count} ({pct:.1f}%)")
        print(f"  Avg Recency: {avg_recency:.0f} days")
        print(f"  Avg Frequency: {avg_frequency:.1f} purchases")
        print(f"  Avg Monetary: £{avg_monetary:,.2f}")
    
    return rfm


if __name__ == "__main__":
    from data_loader import load_retail_data
    from data_cleaning import clean_retail_data, engineer_features
    
    # Test the module
    df = load_retail_data()
    df_sales, df_returns = clean_retail_data(df)
    df_sales = engineer_features(df_sales)
    
    # RFM Analysis
    rfm = calculate_rfm(df_sales)
    inertias, scores, K_range, optimal_k = find_optimal_clusters(rfm)
    rfm = segment_customers(rfm, n_clusters=4)
    rfm = assign_personas(rfm)
    
    print("\nRFM Analysis Complete:")
    print(rfm.head())
