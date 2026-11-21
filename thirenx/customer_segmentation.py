
# Step 1: Import Required Libraries

import os

# Prevent OpenBLAS memory allocation issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("Libraries imported successfully!")

# Step 2: Load Adidas Excel Dataset

from pathlib import Path

file_path = Path(__file__).resolve().parent / "Adidas US Sales Datasets.xlsx"

# Check whether the file exists
if not file_path.exists():
    raise FileNotFoundError("Adidas Excel dataset not found!")

# Load Excel dataset
df = pd.read_excel(
    file_path,
    sheet_name="Data Sales Adidas",
    header=4,
    usecols="B:N",
    engine="openpyxl"
)

# Display first five rows
print("\nFirst 5 Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())




# Step 3: Data Cleaning and Preprocessing

print("\n========== DATA CLEANING ==========")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

# Convert Invoice Date to datetime
df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])

# Display cleaned dataset information
print("\nCleaned Dataset Information:")
df.info()

# Display cleaned dataset shape
print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nData Cleaning Completed Successfully!")

# Step 4: Exploratory Data Analysis (EDA)

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# 1. Total Sales
print("\nTotal Sales:")
print(df["Total Sales"].sum())

# 2. Total Operating Profit
print("\nTotal Operating Profit:")
print(df["Operating Profit"].sum())

# 3. Total Units Sold
print("\nTotal Units Sold:")
print(df["Units Sold"].sum())

# 4. Sales by Retailer
retailer_sales = df.groupby("Retailer")["Total Sales"].sum()

print("\nSales by Retailer:")
print(retailer_sales)

# 5. Sales by Region
region_sales = df.groupby("Region")["Total Sales"].sum()

print("\nSales by Region:")
print(region_sales)

# 6. Sales by Product
product_sales = df.groupby("Product")["Total Sales"].sum()

print("\nSales by Product:")
print(product_sales)

# 7. Sales Method Analysis
method_sales = df.groupby("Sales Method")["Total Sales"].sum()

print("\nSales by Method:")
print(method_sales)


# 8. Retailer Sales Graph
plt.figure(figsize=(10, 5))
retailer_sales.sort_values().plot(kind="bar", color="skyblue")

plt.title("Total Sales by Retailer")
plt.xlabel("Retailer")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("retailer_sales.png")
plt.show()


# 9. Region Sales Graph
plt.figure(figsize=(10, 5))
region_sales.sort_values().plot(kind="bar", color="orange")

plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("region_sales.png")
plt.show()


# 10. Product Sales Graph
plt.figure(figsize=(12, 5))
product_sales.sort_values().plot(kind="bar", color="green")

plt.title("Total Sales by Product")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("product_sales.png")
plt.show()

print("\nEDA Completed Successfully!")


# Step 5: Feature Selection and K-Means Clustering

print("\n========== K-MEANS CLUSTERING ==========")

# Group data by retailer
retailer_data = df.groupby("Retailer").agg({
    "Total Sales": "sum",
    "Units Sold": "sum",
    "Operating Profit": "sum",
    "Operating Margin": "mean",
    "Price per Unit": "mean"
})

print("\nRetailer Segmentation Data:")
print(retailer_data)

# Feature selection
X = retailer_data[
    [
        "Total Sales",
        "Units Sold",
        "Operating Profit",
        "Operating Margin",
        "Price per Unit"
    ]
]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed!")

# Apply K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

retailer_data["Cluster"] = kmeans.fit_predict(X_scaled)

# Display cluster results
print("\nRetailer Clustering Results:")
print(retailer_data)

print("\nK-Means Clustering Completed Successfully!")

# Step 6: Cluster Visualization

print("\n========== CLUSTER VISUALIZATION ==========")

plt.figure(figsize=(10, 6))

plt.scatter(
    retailer_data["Total Sales"],
    retailer_data["Operating Profit"],
    c=retailer_data["Cluster"],
    cmap="viridis",
    s=200
)

for retailer in retailer_data.index:
    plt.annotate(
        retailer,
        (
            retailer_data.loc[retailer, "Total Sales"],
            retailer_data.loc[retailer, "Operating Profit"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("Retailer Segmentation Using K-Means")
plt.xlabel("Total Sales")
plt.ylabel("Operating Profit")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.tight_layout()

plt.savefig("retailer_clusters.png")
print("Graph saved as retailer_clusters.png")

plt.show(block=True)

print("Cluster Visualization Completed!")

# Step 7: Export Segmentation Results

print("\n========== EXPORTING RESULTS ==========")

# Convert retailer index into a normal column
results = retailer_data.reset_index()

# Save results as CSV
results.to_csv("retailer_segmentation_results.csv", index=False)

# Save results as Excel
results.to_excel("retailer_segmentation_results.xlsx", index=False)

print("CSV file saved successfully!")
print("Excel file saved successfully!")
print("\n========== PROJECT STEP 7 COMPLETED ==========")

# Step 8: Cluster Analysis

print("\n========== CLUSTER ANALYSIS ==========")

for cluster in sorted(retailer_data["Cluster"].unique()):

    print(f"\nCluster {cluster}:")

    cluster_data = retailer_data[
        retailer_data["Cluster"] == cluster
    ]

    print("Retailers:", ", ".join(cluster_data.index))

    print("Average Total Sales:",
          round(cluster_data["Total Sales"].mean(), 2))

    print("Average Operating Profit:",
          round(cluster_data["Operating Profit"].mean(), 2))

    print("Average Operating Margin:",
          round(cluster_data["Operating Margin"].mean(), 4))

print("\nCluster Analysis Completed Successfully!")