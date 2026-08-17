import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
# STEP 1: LOAD DATASET

data = pd.read_csv("customer_churn.csv")


# STEP 2: UNDERSTAND DATASET

# First 5 rows
print("\nFirst 5 rows:")
print(data.head())


# Number of rows and columns
print("\nDataset Shape:")
print(data.shape)


# Column names
print("\nColumn Names:")
print(data.columns)


# Information about dataset
print("\nDataset Information:")
print(data.info())


# Missing values
print("\nMissing Values:")
print(data.isnull().sum())


# Duplicate rows
print("\nDuplicate Rows:")
print(data.duplicated().sum())

# Churn distribution
print("\nChurn Distribution:")
print(data["Churn"].value_counts())

# step 3: DATA CLEANING

# Check TotalCharges
print("\nTotalCharges data type:")
print(data["TotalCharges"].dtype)

# Convert TotalCharges to numbers
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Handle the missing NaN
data["TotalCharges"] = data["TotalCharges"].fillna(0)

# Drop TotalCharges
data.dropna(subset=["TotalCharges"], inplace=True)

# Check duplicate customers
print("\nDuplicate rows:")
print(data.duplicated().sum())

data = data.drop_duplicates()

# Remove customerID
data = data.drop("customerID", axis=1)

# Make a separate copy for EDA
eda_data = data.copy()

# convert churn in eda_data
eda_data["Churn"] = eda_data["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Check the cleaned dataset
print("\nCleaned dataset:")
print(data.head())

print("\nCleaned shape:")
print(data.shape)

print(data.columns.tolist())

# step 4: Encoding

# which columns contain text
print("\nCategorical columns:")
print(data.select_dtypes(include="str").columns)


# Converting the Target Churn
data["Churn"] = data["Churn"].map({
    "Yes": 1,
    "No": 0
})

# print(data["Churn"].value_counts())

# One-Hot Encoding
data = pd.get_dummies(data, drop_first=True)
data = data.astype(int)

# Check encoded dataset
print("\nDataset after encoding:")
print(data.head())

# Check new shape
print("\nNew shape:")
print(data.shape)

# Step 5: - Exploratory Data Analysis (EDA)

# Churn Distribution How many customers stayed vs left?

print("\nChurn Distribution:")
print(data["Churn"].value_counts())

# Plot churn distribution
sns.countplot(x="Churn", data = eda_data)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.show()

# 6. Internet Service vs Churn
sns.countplot(
    x="InternetService",
    hue="Churn",
    data = eda_data
)

plt.title("Internet Service vs Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.show()

# data = pd.get_dummies(data, drop_first=True)

# Step 7 — Separating X and y

# X = Features
X = data.drop("Churn", axis=1)

# y = Target
y = data["Churn"]

# Save the feature names used by the model
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\nFeature names saved successfully!")

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nFirst 5 rows of X:")
print(X.head())

print("\nFirst 5 values of y:")
print(y.head())

X = data.drop("Churn", axis=1)
y = data["Churn"]


# STEP 8 — Train/Test Split + Scaling

# Import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Check the split
print("\nTraining X shape:")
print(X_train.shape)

print("\nTesting X shape:")
print(X_test.shape)

print("\nTraining y shape:")
print(y_train.shape)

print("\nTesting y shape:")
print(y_test.shape)

# Scale the training data
X_train = scaler.fit_transform(X_train)

# Scale the testing data
X_test = scaler.transform(X_test)
print("\nScaling completed successfully!")


# STEP 9 — Logistic Regression 
# Import Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

   
# STEP 10 — Make Predictions
y_pred = model.predict(X_test)

print("\nFirst 20 Predictions:")
print(y_pred[:20])



# STEP 11 — Check How Good the Predictions Are


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("Accuracy Percentage:")
print(accuracy * 100, "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# visualizing the confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()



# STEP 12 — Precision, Recall and F1 Score

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# STEP 13 — Churn Probability
probabilities = model.predict_proba(X_test)

print("\nFirst 10 probabilities:")
print(probabilities[:10])

# STEP 14 — ROC-AUC
# Another important evaluation metric is ROC-AUC.

y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

print("\nROC-AUC Score:")
print(roc_auc)

y_pred = model.predict(X_test)

print("\nFirst 20 Predictions:")
print(y_pred[:20])

# STEP 16 — Try Random Forest

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train Random Forest
rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Make Random Forest predictions
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest First 20 Predictions:")
print(rf_pred[:20])

# Random Forest Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print("Random Forest Accuracy Percentage:")
print(rf_accuracy * 100, "%")

# Random Forest Classification Report
print("\nRandom Forest Classification Report:")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# Random Forest ROC-AUC
rf_probability = rf_model.predict_proba(X_test)[:, 1]

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probability
)

print("\nRandom Forest ROC-AUC:")
print(rf_roc_auc)

# STEP 17 — MODEL COMPARISON

print("\n========== MODEL COMPARISON ==========")

print("\nLogistic Regression:")
print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)

print("\nRandom Forest:")
print("Accuracy:", rf_accuracy)
print("ROC-AUC:", rf_roc_auc)


print("\n========== MODEL COMPARISON ==========")

print("\nLogistic Regression:")
print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)

print("\nRandom Forest:")
print("Accuracy:", rf_accuracy)
print("ROC-AUC:", rf_roc_auc)

joblib.dump(model, "churn_model.pkl")

joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")
print("Scaler saved successfully!")

# STEP 18 — Confusion Matrix for Random Forest

rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# STEP 19 — Feature Importance

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# STEP 20 — Plot Feature Importance

plt.figure(figsize=(10, 6))

feature_importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Features Affecting Customer Churn")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.show()

# STEP 21 — Find High-Risk Customers

churn_probability = model.predict_proba(X_test)[:, 1]

high_risk = churn_probability >= 0.70

print("\nNumber of High-Risk Customers:")
print(high_risk.sum())

# STEP 22 — Create a Risk Category

def risk_category(probability):

    if probability < 0.40:
        return "Low Risk"

    elif probability < 0.70:
        return "Medium Risk"

    else:
        return "High Risk"


risk_categories = [
    risk_category(prob)
    for prob in churn_probability
]

print("\nFirst 20 Risk Categories:")
print(risk_categories[:20])

# STEP 23 — Create Prediction DataFrame

results = pd.DataFrame({

    "Actual_Churn": y_test.values,

    "Predicted_Churn": y_pred,

    "Churn_Probability": churn_probability,

    "Risk_Category": risk_categories
})

print("\nPrediction Results:")
print(results.head(10))

# STEP 24 — Save Prediction Results
results.to_csv(
    "churn_predictions.csv",
    index=False
)

print("\nPrediction results saved successfully!")

# STEP 25 — Save the ML Model

import joblib

joblib.dump(model, "churn_model.pkl")

joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")

# STEP 26 — Load the Saved Model
# model = LogisticRegression()
# model.fit(X_train, y_train)

model = joblib.load("churn_model.pkl")

scaler = joblib.load("scaler.pkl")
print("\nSaved model and scaler loaded successfully!")

# STEP 27 — Build a Customer Prediction Function
def predict_churn(customer_data):

    scaled_data = scaler.transform(customer_data)

    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)[:, 1]

    return prediction[0], probability[0]
