import pandas as pd
import numpy as np

df = pd.read_csv("House Price Prediction Dataset.csv")
print("\n========== HOUSE PRICE PREDICTION ==========\n")

print("Dataset loaded successfully!")
print("Number of records:", len(df))

print("\nFirst 5 records:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

features = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
target = "Price"

df = df[features + [target]].dropna()

X = df[features].values
y = df[target].values

np.random.seed(42)

indices = np.random.permutation(len(X))
split = int(len(X) * 0.8)

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

y_pred = X_test @ coefficients

mse = np.mean((y_test - y_pred) ** 2)

r2 = 1 - (
    np.sum((y_test - y_pred) ** 2) /
    np.sum((y_test - np.mean(y_test)) ** 2)
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\n========== MODEL RESULTS ==========")
print("R2 Score:", r2)
print("Mean Squared Error:", mse)

sample = np.array([[1, 1500, 3, 2, 2, 2015]])

prediction = sample @ coefficients

print("\n========== SAMPLE PREDICTION ==========")
print("Area:", 1500)
print("Bedrooms:", 3)
print("Bathrooms:", 2)
print("Floors:", 2)
print("Year Built:", 2015)
print("Predicted House Price:", round(prediction[0], 2))