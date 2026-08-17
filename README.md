Customer Churn Prediction using Machine Learning
📌 Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave a company based on their demographic, service, and billing information.

The project includes data preprocessing, exploratory data analysis, machine learning model training, evaluation, and a Flask web application for making customer churn predictions.

🎯 Objectives
Predict whether a customer is likely to churn.
Analyze customer information and identify factors related to churn.
Build a machine learning classification model.
Provide an easy-to-use web interface for real-time predictions.
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Flask
Joblib
HTML & CSS
🔄 Project Workflow
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
Churn Prediction
📊 Dataset

The project uses customer data containing information such as:

Customer demographics
Contract details
Internet and phone services
Payment information
Monthly charges
Total charges
Customer tenure
Churn status
🧹 Data Preprocessing

The dataset was prepared before training the machine learning model. The preprocessing steps include:

Handling missing values
Converting data types
Encoding categorical variables
Feature scaling
Splitting the dataset into training and testing data
🤖 Machine Learning

The project uses classification techniques to predict customer churn.

The trained model takes customer information as input and predicts whether the customer is likely to:

Stay
Churn

Model performance is evaluated using appropriate classification metrics.

🌐 Flask Web Application

A Flask-based web application was developed to provide an interactive interface for the prediction model.

Users can enter customer details through the web interface, and the application processes the information and displays the predicted churn result.

📁 Project Structure
Customer-Churn-ML/
│
├── app.py
├── churn_prediction.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── model/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
└── dataset/
    └── Telco-Customer-Churn.csv

Update the file names above if your actual project folder uses different names.

▶️ How to Run the Project
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_LINK
2. Open the project folder
cd Customer-Churn-ML
3. Install the required libraries
pip install -r requirements.txt
4. Run the Flask application
python app.py
5. Open the application

Open the localhost address shown in your terminal, for example:

http://127.0.0.1:5000/
📈 Key Features
Customer churn prediction
Data preprocessing and analysis
Machine learning classification
Feature scaling and encoding
Flask-based web interface
Real-time prediction
Saved machine learning model
🚀 Future Improvements
Deploy the application online.
Improve model performance through hyperparameter tuning.
Add additional machine learning algorithms.
Add interactive data visualizations.
Add customer churn probability prediction.
👨‍💻 Author

Vijaya Bhaskar Chikatla

B.Tech – Computer Science and Engineering (AI/ML)

📜 License

This project is created for educational and portfolio purposes.