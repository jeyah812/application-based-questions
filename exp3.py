# ==========================================
# Experiment 3: Linear Regression
# Car Resale Price Prediction
# ==========================================

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("data/cars.csv")

# Display dataset
print("\n========== CAR DATASET ==========")
print(df)

# Convert Brand column into numerical columns
df = pd.get_dummies(df, columns=["Brand"])

# Features (Independent Variables)
X = df.drop(columns=["Car_ID", "Resale_Price"])

# Target (Dependent Variable)
y = df["Resale_Price"]

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

print("\n========== MODEL STATUS ==========")
print("Linear Regression model trained successfully!")

# Store feature names
feature_columns = X.columns

# -------------------- USER INPUT --------------------

print("\nEnter details of the car:")

brand = input("Brand (Maruti/Hyundai/Honda/Tata/Toyota/Kia/Mahindra/BMW/Mercedes-Benz/Audi): ")

age = int(input("Age of Car (Years): "))
kms = int(input("Kilometers Driven: "))
engine = int(input("Engine Capacity (CC): "))

# Create a dictionary with all feature columns initialized to 0
new_car = {col: 0 for col in feature_columns}

# Assign numerical values
new_car["Age"] = age
new_car["Kms_Driven"] = kms
new_car["Engine_CC"] = engine

# Assign selected brand
brand_column = "Brand_" + brand

if brand_column in new_car:
    new_car[brand_column] = 1
else:
    print("Invalid Brand!")
    exit()

# Convert to DataFrame
new_car = pd.DataFrame([new_car])

# Predict resale price
predicted_price = model.predict(new_car)

# Display result
print("\n========== PREDICTION ==========")
print(f"Estimated Resale Price: ₹{predicted_price[0]:,.2f}")