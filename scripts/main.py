"""
Main Analysis Runner
Orchestrates the entire analysis pipeline
"""

import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

from data_loader import load_retail_data
from data_cleaning import clean_retail_data, engineer_features, check_data_quality
from rfm_analysis import calculate_rfm, find_optimal_clusters, segment_customers, assign_personas
from visualizations import (
    plot_monthly_revenue, 
    plot_country_revenue_map, 
    plot_top_products,
    plot_cluster_analysis,
    plot_elbow_silhouette
)


def create_output_directories():
    """Create necessary output directories"""
    os.makedirs('plots', exist_ok=True)
    print("Output directories created/verified")


def run_full_analysis():
    """
    Execute the complete retail analysis pipeline
    """
    print("=" * 80)
    print("ONLINE RETAIL ANALYSIS - Full Pipeline Execution")
    print("=" * 80)
    
    # Create directories
    create_output_directories()
    
    # Step 1: Load Data
    print("\n[1/7] Loading data...")
    df = load_retail_data()
    
    # Step 2: Data Quality Check
    print("\n[2/7] Checking data quality...")
    quality_metrics = check_data_quality(df)
    
    # Step 3: Clean Data
    print("\n[3/7] Cleaning data...")
    df_sales, df_returns = clean_retail_data(df)
    
    # Step 4: Engineer Features
    print("\n[4/7] Engineering features...")
    df_sales = engineer_features(df_sales)
    
    # Step 5: RFM Analysis
    print("\n[5/7] Performing RFM analysis...")
    rfm = calculate_rfm(df_sales)
    
    print("\n[5a/7] Finding optimal clusters...")
    inertias, silhouette_scores, K_range, optimal_k = find_optimal_clusters(rfm)
    
    print("\n[5b/7] Segmenting customers...")
    rfm = segment_customers(rfm, n_clusters=4)
    rfm = assign_personas(rfm)
    
    # Step 6: Generate Visualizations
    print("\n[6/7] Generating visualizations...")
    plot_monthly_revenue(df_sales, save_path='plots/monthly_revenue.png')
    plot_top_products(df_sales, save_path='plots/top_products.png')
    plot_elbow_silhouette(inertias, silhouette_scores, K_range, save_path='plots/elbow_silhouette.png')
    plot_cluster_analysis(rfm, save_path='plots/rfm_clusters.png')
    
    # Generate key metrics
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    total_revenue = df_sales['TotalPrice'].sum()
    total_customers = df_sales['Customer ID'].nunique()
    avg_order_value = df_sales['TotalPrice'].mean()
    
    print(f"\n📊 Business Metrics:")
    print(f"  Total Revenue: £{total_revenue:,.2f}")
    print(f"  Total Customers: {total_customers:,}")
    print(f"  Average Order Value: £{avg_order_value:,.2f}")
    print(f"  Total Transactions: {len(df_sales):,}")
    
    print(f"\n👥 Customer Segments:")
    for cluster_id in range(4):
        cluster_data = rfm[rfm['Cluster'] == cluster_id]
        count = len(cluster_data)
        total_value = cluster_data['Monetary'].sum()
        pct = (count / len(rfm) * 100)
        revenue_pct = (total_value / rfm['Monetary'].sum() * 100)
        print(f"  Cluster {cluster_id}: {count} customers ({pct:.1f}%) | £{total_value:,.0f} ({revenue_pct:.1f}% of revenue)")
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check the 'plots' directory for results.")
    print("=" * 80)


if __name__ == "__main__":
    run_full_analysis()
