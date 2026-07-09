import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Load the dataset
df = pd.read_csv("data/loan_data.csv")

# Display dataset
print("\n========== LOAN DATASET ==========")
print(df)

# Features (Independent Variables)
X = df[["Income", "Credit_Score", "Existing_Loans"]]

# Target (Dependent Variable)
y = df["Approved"]

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("\n========== MODEL STATUS ==========")
print("Decision Tree (CART) model trained successfully!")

# Accept user input
print("\nEnter applicant details:")

income = int(input("Monthly Income (₹): "))
credit_score = int(input("Credit Score: "))
existing_loans = int(input("Existing Loans: "))

# Create input DataFrame
new_applicant = pd.DataFrame(
    [[income, credit_score, existing_loans]],
    columns=["Income", "Credit_Score", "Existing_Loans"]
)

# Predict
prediction = model.predict(new_applicant)

# Display result
print("\n========== PREDICTION ==========")

if prediction[0] == 1:
    print("Loan Status: APPROVED")
else:
    print("Loan Status: REJECTED")

# Display decision rules
print("\n========== DECISION RULES ==========")
rules = export_text(
    model,
    feature_names=["Income", "Credit_Score", "Existing_Loans"]
)
print(rules)