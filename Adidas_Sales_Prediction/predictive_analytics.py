import pandas as pd

# Load Adidas dataset
df = pd.read_excel(
    "dataset/Adidas US Sales Datasets.xlsx",
    header=4
)

# Remove empty columns
df = df.dropna(axis=1, how="all")
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

print("Dataset Loaded Successfully!")

# Convert Invoice Date into date format
df["Invoice Date"] = pd.to_datetime(df["Invoice Date"])

# Convert Total Sales into numeric format
df["Total Sales"] = pd.to_numeric(
    df["Total Sales"], errors="coerce"
)

# Remove missing values
df = df.dropna(subset=["Invoice Date", "Total Sales"])

# Sort data by date
df = df.sort_values("Invoice Date")

print("\nData Cleaning Completed!")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

print("\nFirst 5 Cleaned Records:")
print(df.head())

print("\nTotal Records:", len(df))

# MONTHLY SALES ANALYSIS

# Extract month and year
df["Year-Month"] = df["Invoice Date"].dt.to_period("M")

# Calculate monthly sales
monthly_sales = df.groupby("Year-Month")["Total Sales"].sum()

# Display monthly sales
print("\nMONTHLY SALES ANALYSIS")
print(monthly_sales)

# Save monthly sales report
monthly_sales.to_csv("outputs/monthly_sales.csv")

print("\nMonthly sales report saved successfully!")

import matplotlib.pyplot as plt

# MONTHLY SALES VISUALIZATION

plt.figure(figsize=(12, 6))

monthly_sales.plot(
    kind="line",
    marker="o",
    color="blue"
)

plt.title("Adidas Monthly Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()

# Save graph
plt.savefig("outputs/monthly_sales_graph.png")

plt.show()

print("\nMonthly sales graph saved successfully!")

# STEP 16: PREPARE DATA FOR MACHINE LEARNING

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Convert monthly sales into a DataFrame
sales_df = monthly_sales.reset_index()

sales_df.columns = ["Month", "Total Sales"]

# Create numerical time feature
sales_df["Month_Number"] = range(len(sales_df))

# Prepare input and output
X = sales_df[["Month_Number"]]
y = sales_df["Total Sales"]

# Split training and testing data
X_train = X.iloc[:18]
X_test = X.iloc[18:]

y_train = y.iloc[:18]
y_test = y.iloc[18:]

print("\nMACHINE LEARNING DATA PREPARED!")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# STEP 18: TRAIN LINEAR REGRESSION MODEL

# Create model
model = LinearRegression()

# Train model using training data
model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED!")

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save actual and predicted values
results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

results.to_csv("outputs/model_evaluation.csv", index=False)

print("\nModel evaluation report saved successfully!")

# STEP 21: FUTURE SALES PREDICTION

# Create future month numbers
future_months = pd.DataFrame({
    "Month_Number": range(24, 30)
})

# Predict next 6 months
future_predictions = model.predict(future_months)

# Create future dates
future_dates = pd.date_range(
    start="2022-01-01",
    periods=6,
    freq="MS"
)

# Create prediction report
forecast_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted Sales": future_predictions
})

print("\nFUTURE ADIDAS SALES PREDICTION")
print(forecast_df)

# Save forecast
forecast_df.to_csv(
    "outputs/future_sales_prediction.csv",
    index=False
)

print("\nFuture sales prediction saved successfully!")

# STEP 22: FUTURE SALES FORECAST GRAPH

plt.figure(figsize=(12, 6))

plt.plot(
    forecast_df["Month"],
    forecast_df["Predicted Sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title("Adidas Future Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("outputs/future_sales_forecast.png")

plt.show()

print("\nFuture forecast graph saved successfully!")