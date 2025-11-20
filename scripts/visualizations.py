"""
Visualization Module
Creates all charts and visualizations for the analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def plot_monthly_revenue(df_sales, save_path=None):
    """
    Create monthly revenue trend chart
    
    Args:
        df_sales (pd.DataFrame): Sales data
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    monthly_revenue = (
        df_sales
        .groupby('YearMonth')['TotalPrice']
        .sum()
        .reset_index()
        .sort_values('YearMonth')
    )
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_revenue['YearMonth'], monthly_revenue['TotalPrice'], marker='o')
    ax.set_xticklabels(monthly_revenue['YearMonth'], rotation=45)
    ax.set_title("Monthly Revenue Over Time", fontsize=14)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Revenue (£)")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Monthly revenue chart saved to: {save_path}")
    
    return fig


def plot_country_revenue_map(df_sales, save_path=None):
    """
    Create geographic revenue distribution map
    
    Args:
        df_sales (pd.DataFrame): Sales data
        save_path (str, optional): Path to save the figure
        
    Returns:
        plotly.graph_objs._figure.Figure: The created figure
    """
    country_revenue = (
        df_sales
        .groupby('Country', as_index=False)['TotalPrice']
        .sum()
    )
    
    fig = px.choropleth(
        country_revenue,
        locations="Country",
        locationmode="country names",
        color="TotalPrice",
        color_continuous_scale="Blues",
        title="Revenue by Country (World Map)",
    )
    
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        coloraxis_colorbar=dict(title="Revenue (£)")
    )
    
    if save_path:
        fig.write_image(save_path)
        print(f"Country revenue map saved to: {save_path}")
    
    return fig


def plot_top_products(df_sales, top_n=10, save_path=None):
    """
    Create bar chart of top products by revenue
    
    Args:
        df_sales (pd.DataFrame): Sales data
        top_n (int): Number of top products to show
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    product_revenue = (
        df_sales
        .groupby(['StockCode', 'Description'])['TotalPrice']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    
    top_products = product_revenue.head(top_n)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=top_products,
        x='TotalPrice',
        y='Description',
        palette='magma',
        ax=ax
    )
    ax.set_title(f"Top {top_n} Products by Revenue", fontsize=14)
    ax.set_xlabel("Revenue (£)")
    ax.set_ylabel("Product Description")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top products chart saved to: {save_path}")
    
    return fig


def plot_cluster_analysis(rfm, save_path=None):
    """
    Create RFM bubble chart visualization
    
    Args:
        rfm (pd.DataFrame): RFM dataframe with clusters
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scatter = ax.scatter(
        rfm['Frequency'],
        rfm['Monetary'],
        c=rfm['Cluster'],
        s=np.log1p(rfm['Monetary']) * 10,
        alpha=0.6,
        cmap='Spectral'
    )
    
    ax.set_title("RFM Bubble Chart (Log-Scaled Bubble Size)", fontsize=14)
    ax.set_xlabel("Frequency (Number of Purchases)")
    ax.set_ylabel("Monetary Value (£)")
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left")
    ax.add_artist(legend)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RFM cluster chart saved to: {save_path}")
    
    return fig


def plot_elbow_silhouette(inertias, silhouette_scores, K_range, save_path=None):
    """
    Create elbow method and silhouette score plots
    
    Args:
        inertias (list): List of inertia values
        silhouette_scores (list): List of silhouette scores
        K_range (range): Range of K values tested
        save_path (str, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax1.set_title('Elbow Method for Optimal k', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score by Number of Clusters', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Elbow/Silhouette chart saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    from data_loader import load_retail_data
    from data_cleaning import clean_retail_data, engineer_features
    from rfm_analysis import calculate_rfm, find_optimal_clusters, segment_customers
    
    # Test the module
    df = load_retail_data()
    df_sales, df_returns = clean_retail_data(df)
    df_sales = engineer_features(df_sales)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_monthly_revenue(df_sales, save_path='plots/monthly_revenue.png')
    plot_top_products(df_sales, save_path='plots/top_products.png')
    
    rfm = calculate_rfm(df_sales)
    inertias, scores, K_range, optimal_k = find_optimal_clusters(rfm)
    rfm = segment_customers(rfm, n_clusters=4)
    
    plot_elbow_silhouette(inertias, scores, K_range, save_path='plots/elbow_silhouette.png')
    plot_cluster_analysis(rfm, save_path='plots/rfm_clusters.png')
    
    print("\nAll visualizations generated successfully!")
