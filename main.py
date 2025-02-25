import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define dataset folder path
dataset_folder = "datasets"

# Load main logic circuit dataset
df_main = pd.read_csv(os.path.join(dataset_folder, "logic_circuit_dataset.csv"))

# Load all RTL datasets (1 to 7) and combine them
rtl_datasets = []
for i in range(1, 8):
    file_path = os.path.join(dataset_folder, f"rtl_dataset{i}.csv")
    if os.path.exists(file_path):
        rtl_datasets.append(pd.read_csv(file_path))

# Combine all datasets
df_combined = pd.concat([df_main] + rtl_datasets, ignore_index=True)

# Display column names to verify
print(" Dataset Columns:", df_combined.columns)

# Feature selection (excluding 'Component' if present)
X = df_combined[['Fan_In', 'Fan_Out', 'Num_Gates', 'Path_Length']]
y = df_combined['Logic_Depth']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print(f" Mean Absolute Error (MAE): {mae:.2f}")
print(f" Mean Squared Error (MSE): {mse:.2f}")
print(f" R-squared (R²): {r2:.2f}")

# Save the trained model
joblib.dump(model, "optimized_timing_model.pkl")
print(" Model saved as 'optimized_timing_model.pkl'")

# Function to predict Logic Depth for a new circuit
def predict_logic_depth(fan_in, fan_out, num_gates, path_length):
    input_data = pd.DataFrame([[fan_in, fan_out, num_gates, path_length]], 
                              columns=['Fan_In', 'Fan_Out', 'Num_Gates', 'Path_Length'])
    predicted_depth = model.predict(input_data)[0]
    return round(predicted_depth, 2)

# ✅ User Input for Prediction (Method 2)
print("\n Enter values to predict Logic Depth:")
fan_in = int(input(" Enter Fan-In: "))
fan_out = int(input(" Enter Fan-Out: "))
num_gates = int(input(" Enter Number of Gates: "))
path_length = int(input(" Enter Path Length: "))

# Get prediction
predicted_depth = predict_logic_depth(fan_in, fan_out, num_gates, path_length)
print(f"\n🔹 Predicted Logic Depth: {predicted_depth}")

# Plot Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed', label="Perfect Prediction")
plt.xlabel("Actual Logic Depth")
plt.ylabel("Predicted Logic Depth")
plt.title("Actual vs Predicted Logic Depth")
plt.legend()
plt.grid()
plt.show()
