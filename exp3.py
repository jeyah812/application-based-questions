# ==========================================
# Experiment 4: Linear Regression
# Car Resale Price Prediction
# ==========================================

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("data/cars.csv")

# Display dataset
print("\n========== CAR DATASET ==========")
print(df)

# Features (Independent Variables)
X = df[["Age", "Kms_Driven", "Engine_CC"]]

# Target (Dependent Variable)
y = df["Resale_Price"]

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

print("\n========== MODEL STATUS ==========")
print("Linear Regression model trained successfully!")

# Accept user input
print("\nEnter details of the car:")

age = int(input("Age of Car (Years): "))
kms = int(input("Kilometers Driven: "))
engine = int(input("Engine Capacity (CC): "))

# Create input DataFrame
new_car = pd.DataFrame(
    [[age, kms, engine]],
    columns=["Age", "Kms_Driven", "Engine_CC"]
)

# Predict resale price
predicted_price = model.predict(new_car)

# Display result
print("\n========== PREDICTION ==========")
print(f"Estimated Resale Price: ₹{predicted_price[0]:,.2f}")