# Plots Directory

This directory contains all visualizations generated from the analysis.

## Generated Files

After running `main.py` or the Jupyter notebook, you'll find:

- `monthly_revenue.png` - Monthly revenue trend over time
- `top_products.png` - Top 10 products by revenue
- `rfm_clusters.png` - RFM customer segmentation bubble chart
- `elbow_silhouette.png` - K-means clustering validation plots

## Usage

These plots are automatically saved when running the analysis. You can also generate them individually:

```python
from visualizations import plot_monthly_revenue, plot_top_products
from data_loader import load_retail_data
from data_cleaning import clean_retail_data, engineer_features

df = load_retail_data()
df_sales, _ = clean_retail_data(df)
df_sales = engineer_features(df_sales)

plot_monthly_revenue(df_sales, save_path='plots/monthly_revenue.png')
plot_top_products(df_sales, save_path='plots/top_products.png')
```

## Note

Plots are gitignored by default. Run the analysis to generate them locally.
