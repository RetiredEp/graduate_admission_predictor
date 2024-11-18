# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib



# Load the dataset
df = pd.read_csv(r"D:\Desktop\CAPSTONE\Graduate-Admission-Prediction-master\Graduate-Admission-Prediction-master\Admission_Predict_Ver1.1.csv")

# Display dataset information
print(df.info())
print(df.head())

# Drop irrelevant column
df = df.drop(columns=["Serial No."])  # 'Serial No.' is not useful for predictions

# Define features (X) and target (y)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column (Chance of Admit)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the TheilSenRegressor model
model = TheilSenRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Make sample predictions
sample_data = X_test[:10]  # Select 10 samples
predictions = model.predict(sample_data)
print(f"Predictions: {predictions}")

# Save the model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved as model.pkl.")
print("Scaler saved as scaler.pkl.")

