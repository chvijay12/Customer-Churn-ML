from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")



# ==============================
# DASHBOARD
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# CUSTOMER
# ==============================

@app.route("/customer")
def customer():
    return render_template("index.html")


# ==============================
# ANALYTICS
# ==============================

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


# ==============================
# HISTORY
# ==============================

@app.route("/history")
def history():
    return render_template("history.html")


# ==============================
# SETTINGS
# ==============================

@app.route("/settings")
def settings():
    return render_template("settings.html")


# ==============================
# ABOUT
# ==============================

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from website
    tenure = float(request.form["tenure"])
    monthly_charges = float(request.form["monthly_charges"])
    total_charges = float(request.form["total_charges"])

    contract = request.form["Contract"]
    internet_service = request.form["InternetService"]


    # Create an empty row with all 30 features
    customer = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )


    # Add numerical values
    customer["tenure"] = tenure
    customer["MonthlyCharges"] = monthly_charges
    customer["TotalCharges"] = total_charges


    # Contract
    if contract == "One year":
        customer["Contract_One year"] = 1

    elif contract == "Two year":
        customer["Contract_Two year"] = 1


    # Internet Service
    if internet_service == "Fiber optic":
        customer["InternetService_Fiber optic"] = 1

    elif internet_service == "No":
        customer["InternetService_No"] = 1


    # Scale the customer data
    customer_scaled = scaler.transform(customer)


    # Make prediction
    prediction = model.predict(customer_scaled)[0]

    probability = model.predict_proba(customer_scaled)[0][1]


    # Convert prediction to text
    if prediction == 1:
        result = "Customer is likely to CHURN 🔴"
    else:
        result = "Customer is likely to STAY 🟢"


    return f"""
    <h1>Customer Churn Prediction</h1>

    <h2>{result}</h2>

    <h3>Churn Probability: {probability * 100:.2f}%</h3>

    <br>

    <a href="/">← Back to Home</a>
    """

if __name__ == "__main__":
    app.run(debug=True)