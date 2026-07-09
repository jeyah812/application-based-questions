# ==========================================
# Scenario 2: Logistic Regression
# Spam Email Classification
# ==========================================

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("data/emails.csv")

# Display dataset
print("\n========== EMAIL DATASET ==========")
print(df)

# Features (Independent Variables)
X = df[["Num_Links", "Num_Caps_Words", "Email_Length"]]

# Target (Dependent Variable)
y = df["Spam"]

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

print("\nModel trained successfully!")

# Take user input
print("\nEnter details of the new email:")

links = int(input("Number of Links: "))
caps = int(input("Number of Capitalized Words: "))
length = int(input("Email Length: "))

new_email = pd.DataFrame(
    [[links, caps, length]],
    columns=["Num_Links", "Num_Caps_Words", "Email_Length"]
)

prediction = model.predict(new_email)

# Display result
print("\n========== PREDICTION ==========")

if prediction[0] == 1:
    print("The email is predicted as: SPAM")
else:
    print("The email is predicted as: NOT SPAM")