import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Create a sample house-price dataset
np.random.seed(42)

rows = 1000

data = pd.DataFrame({
    "latitude": np.random.uniform(16.3, 17.6, rows),
    "longitude": np.random.uniform(78.2, 80.0, rows),
    "size": np.random.randint(500, 5000, rows),
    "bedrooms": np.random.randint(1, 6, rows),
    "bathrooms": np.random.randint(1, 5, rows),
    "age": np.random.randint(0, 40, rows),
    "amenity_proximity": np.random.uniform(0.2, 10, rows)
})

# Generate a realistic-looking synthetic price
data["price"] = (
    data["size"] * 5500
    + data["bedrooms"] * 300000
    + data["bathrooms"] * 200000
    - data["age"] * 50000
    - data["amenity_proximity"] * 25000
    + np.random.normal(0, 300000, rows)
)

# Make sure prices don't become negative
data["price"] = data["price"].clip(lower=500000)

# Feature engineering
data["size_bedrooms"] = data["size"] * data["bedrooms"]
data["size_bathrooms"] = data["size"] * data["bathrooms"]
data["inverse_age"] = 1 / (data["age"] + 1)
data["inverse_proximity"] = 1 / (data["amenity_proximity"] + 0.1)

features = [
    "latitude",
    "longitude",
    "size",
    "bedrooms",
    "bathrooms",
    "age",
    "amenity_proximity",
    "size_bedrooms",
    "size_bathrooms",
    "inverse_age",
    "inverse_proximity"
]

X = data[features]
y = data["price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Test the model
predictions = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model training completed!")
print(f"Mean Absolute Error: ₹{mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# Save the model and scaler
joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save dataset
data.to_csv("house_data.csv", index=False)

print("\nFiles created successfully:")
print("1. best_model.pkl")
print("2. scaler.pkl")
print("3. house_data.csv")