
# DATA CLEANING & REPORTING AUTOMATION

# STEP 1: IMPORT REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# STEP 2: CREATE OUTPUT FOLDER

os.makedirs("outputs", exist_ok=True)

# STEP 3: LOAD DATASET

file_path = "dataset/Adidas US Sales Datasets.xlsx"

df = pd.read_excel(file_path, header=4)

# STEP 4: DISPLAY DATASET INFORMATION

print("\n========== ORIGINAL DATASET ==========")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# STEP 5: REMOVE UNNECESSARY COLUMNS

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\n========== AFTER REMOVING UNNECESSARY COLUMNS ==========")
print("Dataset Shape:", df.shape)

# STEP 6: REMOVE DUPLICATE ROWS

duplicates = df.duplicated().sum()

print("\nNumber of Duplicate Rows:", duplicates)

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:", df.shape)


# STEP 7: CHECK MISSING VALUES

print("\n========== MISSING VALUE ANALYSIS ==========")

missing_values = df.isnull().sum()

print("\nMissing Values in Each Column:")
print(missing_values)

print("\nTotal Missing Values:", missing_values.sum())

# HANDLE MISSING VALUES

for column in df.columns:

    if df[column].isnull().sum() > 0:

        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])

        elif pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nTotal Missing Values After Cleaning:", df.isnull().sum().sum())


# STEP 8: CLEAN INCONSISTENT TEXT VALUES

print("\n========== TEXT DATA CLEANING ==========")

text_columns = df.select_dtypes(include=["object", "str"]).columns

for column in text_columns:
    df[column] = df[column].str.strip()
    df[column] = df[column].str.replace(r"\s+", " ", regex=True)
    df[column] = df[column].str.title()

print("\nText Values Standardized Successfully!")

print("\nSample Cleaned Data:")
print(df[text_columns].head())

# CHECK UNIQUE VALUES

for column in text_columns:
    print(f"\nUnique values in {column}:")
    print(df[column].unique())


# STEP 9: NUMERICAL DATA VALIDATION

print("\n========== NUMERICAL DATA VALIDATION ==========")

numeric_columns = [
    "Price per Unit",
    "Units Sold",
    "Total Sales",
    "Operating Profit",
    "Operating Margin"
]

print("\nNegative Values Before Validation:")

for column in numeric_columns:
    print(column, ":", (df[column] < 0).sum())

# CHECK ZERO VALUES

print("\nZero Values:")

for column in numeric_columns:
    print(column, ":", (df[column] == 0).sum())

# CHECK INVALID VALUES

invalid_rows = df[
    (df["Price per Unit"] < 0) |
    (df["Units Sold"] < 0) |
    (df["Total Sales"] < 0) |
    (df["Operating Profit"] < 0) |
    (df["Operating Margin"] < 0) |
    (df["Operating Margin"] > 1)
]

print("\nNumber of Invalid Rows:", len(invalid_rows))

print("\nNumerical Validation Completed!")

# STEP 10: SAVE CLEANED DATASET

print("\n========== EXPORTING CLEANED DATA ==========")

# Save cleaned dataset
df.to_excel(
    "outputs/cleaned_data.xlsx",
    index=False
)

print("Cleaned dataset saved successfully!")

# Generate data quality report
quality_report = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

# Save quality report
quality_report.to_excel(
    "outputs/data_quality_report.xlsx",
    index=False
)

print("Data quality report saved successfully!")

print("\nFinal Dataset Shape:", df.shape)
print("\nAll cleaning operations completed!")


# STEP 11: GENERATE SALES SUMMARY REPORTS

print("\n========== SALES SUMMARY REPORT ==========")

# 1. Overall Sales Summary

overall_summary = pd.DataFrame({
    "Metric": [
        "Total Revenue",
        "Total Units Sold",
        "Total Operating Profit",
        "Average Sales",
        "Total Transactions"
    ],
    "Value": [
        df["Total Sales"].sum(),
        df["Units Sold"].sum(),
        df["Operating Profit"].sum(),
        df["Total Sales"].mean(),
        len(df)
    ]
})

print("\nOverall Sales Summary:")
print(overall_summary)

# 2. Region-wise Sales

region_summary = df.groupby("Region").agg(
    Total_Sales=("Total Sales", "sum"),
    Total_Units=("Units Sold", "sum"),
    Total_Profit=("Operating Profit", "sum")
).reset_index()

print("\nRegion-wise Sales:")
print(region_summary)

# 3. Product-wise Sales

product_summary = df.groupby("Product").agg(
    Total_Sales=("Total Sales", "sum"),
    Total_Units=("Units Sold", "sum")
).reset_index()

print("\nProduct-wise Sales:")
print(product_summary)

# 4. Retailer-wise Sales

retailer_summary = df.groupby("Retailer").agg(
    Total_Sales=("Total Sales", "sum"),
    Total_Profit=("Operating Profit", "sum")
).reset_index()

print("\nRetailer-wise Sales:")
print(retailer_summary)

# 5. Save Reports

overall_summary.to_csv("outputs/overall_sales_summary.csv", index=False)

region_summary.to_csv("outputs/region_sales_summary.csv", index=False)

product_summary.to_csv("outputs/product_sales_summary.csv", index=False)

retailer_summary.to_csv("outputs/retailer_sales_summary.csv", index=False)

print("\nAll Sales Summary Reports Saved Successfully!")


# STEP 12: GENERATE VISUAL REPORTS

print("\n========== GENERATING VISUAL REPORTS ==========")

# GRAPH 1: REGION-WISE SALES

plt.figure(figsize=(10, 6))

sns.barplot(
    data=region_summary,
    x="Region",
    y="Total_Sales"
)

plt.title("Region-wise Sales Analysis")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/region_sales_chart.png")
plt.show()


# GRAPH 2: PRODUCT-WISE SALES

plt.figure(figsize=(12, 6))

sns.barplot(
    data=product_summary,
    x="Product",
    y="Total_Sales"
)

plt.title("Product-wise Sales Analysis")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/product_sales_chart.png")
plt.show()


# GRAPH 3: RETAILER-WISE SALES

plt.figure(figsize=(10, 6))

sns.barplot(
    data=retailer_summary,
    x="Retailer",
    y="Total_Sales"
)

plt.title("Retailer-wise Sales Analysis")
plt.xlabel("Retailer")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("outputs/retailer_sales_chart.png")
plt.show()


print("\nAll Visual Reports Generated Successfully!")

# STEP 13: CREATE CONSOLIDATED EXCEL REPORT

with pd.ExcelWriter("outputs/automated_sales_report.xlsx", engine="openpyxl") as writer:

    overall_summary.to_excel(writer, sheet_name="Overall Summary", index=False)

    region_summary.to_excel(writer, sheet_name="Region Summary", index=False)

    product_summary.to_excel(writer, sheet_name="Product Summary", index=False)

    retailer_summary.to_excel(writer, sheet_name="Retailer Summary", index=False)

    quality_report.to_excel(writer, sheet_name="Data Quality", index=False)

print("\nConsolidated Excel Report Created Successfully!")