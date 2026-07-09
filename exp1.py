import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("data/houses.csv")

# Display the dataset
print("House Dataset:")
print(df)

# Convert required columns to a NumPy array
# Columns: Bedrooms, Sqft, Sale_Price
house_data = df[["Bedrooms", "Sqft", "Sale_Price"]].to_numpy()

# Filter houses with more than 4 bedrooms
filtered_houses = house_data[house_data[:, 0] > 4]

# Calculate average sale price
average_price = np.mean(filtered_houses[:, 2])

# Display the filtered houses
print("\nHouses with more than 4 bedrooms:")
print(filtered_houses)

# Display the average sale price
print(f"\nAverage Sale Price: ₹{average_price:,.2f}")