# Customer Churn Prediction using Machine Learning

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company based on customer demographic, service, and billing information.

The project includes data preprocessing, exploratory data analysis, model training, evaluation, and a Flask web application for making customer churn predictions.

## 🎯 Objectives

* Predict whether a customer is likely to churn.
* Analyze customer information and identify factors related to churn.
* Build a Machine Learning classification model.
* Provide a web interface for real-time churn prediction.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Flask
* Joblib
* HTML
* CSS

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Flask Web Application
   ↓
Customer Churn Prediction
```

## 📊 Dataset

The project uses customer information such as:

* Customer demographics
* Tenure
* Contract type
* Internet service
* Phone service
* Payment method
* Monthly charges
* Total charges
* Churn status

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Handling missing values
* Data type conversion
* Encoding categorical variables
* Feature scaling
* Splitting data into training and testing sets

## 🤖 Machine Learning Model

The project uses a classification-based Machine Learning approach to predict customer churn.

The model takes customer information as input and predicts whether the customer is likely to:

* Stay
* Churn

Model performance is evaluated using classification metrics.

## 🌐 Flask Web Application

A Flask web application was developed to provide an interactive interface for the trained Machine Learning model.

Users can enter customer details through the web interface, and the application processes the input and displays the predicted churn result.

## 📁 Project Structure

```text
Customer-Churn-ML/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── README.md
├── requirements.txt
├── app.py
├── churn_model.pkl
├── churn_prediction.py
├── churn_predictions.csv
├── customer_churn.csv
├── feature_names.pkl
└── scaler.pkl
```

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/chvijay12/Customer-Churn-ML.git
```

### 2. Open the project folder

```bash
cd Customer-Churn-ML
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```
### 5. Open the application

After starting the Flask application, open the local address shown in the terminal.


## ⭐ Key Features

* Customer churn prediction
* Data preprocessing
* Exploratory data analysis
* Feature encoding
* Feature scaling
* Machine Learning classification
* Flask web interface
* Real-time prediction
* Saved trained model

## 🚀 Future Improvements

* Deploy the application online.
* Improve model performance through hyperparameter tuning.
* Add additional Machine Learning algorithms.
* Add interactive data visualizations.
* Add customer churn probability prediction.

## 👨‍💻 Author

**Vijaya Bhaskar Chikatla**

B.Tech – Computer Science and Engineering (AI/ML)

## 📜 License

This project is created for educational and portfolio purposes.
