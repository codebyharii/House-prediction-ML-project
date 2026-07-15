# House Price Prediction - A simple Machine Learning example
# Data source: housing.csv (California Housing dataset, downloaded from Kaggle)
# Goal: predict median_house_value using the other columns in the CSV.

# Install these first (run in terminal) if you don't have them:
# pip install pandas numpy scikit-learn matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the data
# pandas reads the CSV into a table called a DataFrame
data = pd.read_csv("housing.csv")

# Peek at the data so we know what we're working with
print(data.head())
print(data.info())

# Step 2: Handle missing values
# total_bedrooms has some empty cells (NaN). We fill them with the median
# value of that column so the model doesn't crash on missing numbers.
data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

# Step 3: Handle the text column
# ocean_proximity is a category (e.g. "NEAR BAY", "INLAND"), not a number.
# ML models need numbers, so we convert it into separate 0/1 columns
# (this is called "one-hot encoding").
data = pd.get_dummies(data, columns=["ocean_proximity"])

# Step 4: Split into features (X) and target (y)
# X = everything the model uses to predict
# y = the thing we want to predict (the house price)
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Step 5: Split into training and testing sets
# We train on 80% of the data and test on the remaining 20% to see
# how well the model performs on data it hasn't seen before.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
predictions = model.predict(X_test)

# Step 8: Evaluate how good the model is
# MAE = average dollar amount the prediction was off by (lower is better)
# R2 = how much of the price variation the model explains (closer to 1 is better)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R2 Score: {r2:.3f}")

# Step 9: Visualize actual vs predicted prices
plt.scatter(y_test, predictions, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")  # perfect-prediction line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
