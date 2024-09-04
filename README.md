# Bank Loan Approval Prediction

This project is a web-based application for predicting bank loan approval using machine learning models. The app is built using Streamlit and allows users to input various features such as applicant income, loan amount, credit history, and more to predict whether a loan will be approved or not. The underlying model is trained on a dataset of past loan applications.
## Features

- ```User-Friendly Interface:``` Easy-to-use form for inputting applicant details.
- ```Real-Time Prediction:``` Instant loan approval prediction based on user inputs.
- ```Data Visualization:``` Displays the dataset used for training the model.
- ```Model Explanation:``` Provides an understanding of the features and transformations used in the prediction model.


## Dataset

The dataset used for this project is the [Loan Prediction Dataset](https://www.kaggle.com/datasets/premptk/loan-approval-prediction-dataset). It includes details like gender, marital status, income, loan amount, credit history, and loan status. The dataset was preprocessed and cleaned to handle missing values and categorical variables were encoded for model training.

#### Features:
- ```Gender:``` Male, Female
- ```Married:``` Yes, No
- ```Dependents:``` Number of dependents (0, 1, 2, 3+)
- ```Education:``` Graduate, Not Graduate
- ```Self_Employed:``` Yes, No
- ```ApplicantIncome:``` Income of the applicant
- ```CoapplicantIncome:``` Income of the co-applicant
- ```LoanAmount:``` Loan amount requested
- ```Loan_Amount_Term:``` Term of the loan in months
- ```Credit_History:``` Credit history (1 = Good, 0 = Bad)
- ```Property_Area:``` Urban, Rural, Semiurban
- ```Loan_Status:``` (Target Variable) Y = Approved, N = Not Approved
#### 🔗 Source
[Loan Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/premptk/loan-approval-prediction-dataset)

## Usage

After running the Streamlit app, you will see a sidebar menu with two options: "Form" and "Dataset".

- ```Form:``` Fill out the form with the applicant's details to get a prediction on whether the loan will be approved or not.
- ```Dataset:``` View the dataset used for training the model.

## Model Information

The model used in this project is a Random Forest Classifier. The model was chosen for its accuracy and robustness in handling various feature types. Here are the steps involved in building the model:

1. **Data Preprocessing:**

- Handling missing values.
- Log transformation of income and loan amount features.
- Encoding categorical variables.

2. **Model Training:**

- The dataset was split into training and testing sets.
- Several models were tested, including Logistic Regression, Decision Tree, and K-Nearest Neighbors.
- The Random Forest model was selected based on its performance (accuracy: 78.57%).

3. **Model Evaluation:**

- The model was evaluated using accuracy, precision, recall, and F1-score.
- Cross-validation was performed to ensure model stability.

4. **Model Deployment:**

- The trained model was saved as a .pkl file and loaded into the Streamlit app for real-time predictions.
## Future Improvements

- **Model Optimization:** Fine-tuning the model parameters to improve accuracy.
- **Interactive Visualizations:** Adding more data visualizations to help users understand the model's decisions.
- **Multi-Model Support:** Allowing users to select between different models for predictions.
