# Black Friday Sales Prediction

## Project Overview
This project involves analyzing the Black Friday sales dataset to understand customer purchase behavior and build a predictive model for purchase amounts. The goal is to perform Exploratory Data Analysis (EDA), feature engineering, and then train a machine learning model to predict individual purchase amounts.

## Dataset
The dataset used in this project contains transactional purchase data from Black Friday sales. It includes various features about customers and products, such as:
- User_ID
- Product_ID
- Gender
- Age
- Occupation
- City_Category
- Stay_In_Current_City_Years
- Marital_Status
- Product_Category_1, 2, 3
- Purchase (Target Variable)

## Exploratory Data Analysis (EDA)
- Initial data inspection (`.info()`, `.head()`, `.shape`).
- Checking for missing values.
- Analyzing distributions of categorical and numerical features.
- Visualizations to understand relationships between features and the target variable (`Purchase`).

## Feature Engineering
- **Handling Categorical Features:**
  - `Gender`: Mapped 'F' to 0 and 'M' to 1.
  - `Age`: Mapped age ranges to numerical values (e.g., '0-17' to 1, '55+' to 7) using `LabelEncoder`.
  - `City_Category`: One-hot encoded using `pd.get_dummies`.
- **Handling Missing Values:**
  - `Product_Category_2` and `Product_Category_3`: Filled missing values with the mode of their respective columns.
- **Feature Transformation:**
  - `Stay_In_Current_City_Years`: Removed '+' sign and converted to integer type.
- **Dropping Unnecessary Features:**
  - `User_ID` and `Product_ID` were dropped after initial use as they are not directly used in modeling.

## Model Training
- **Data Splitting:** The preprocessed data was split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using `train_test_split` from `sklearn.model_selection`.
- **Feature Scaling:** `StandardScaler` was applied to `X_train` and `X_test` for feature scaling.

## Next Steps (Further Model Training)
After feature scaling, the next steps involve:
1.  **Model Selection:** Choose appropriate regression models (e.g., Linear Regression, Decision Tree Regressor, Random Forest Regressor, XGBoost).
2.  **Model Training:** Train selected models on the `X_train` and `y_train` data.
3.  **Model Evaluation:** Evaluate model performance using metrics like R-squared, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) on the `X_test` and `y_test` data.
4.  **Hyperparameter Tuning:** Optimize model hyperparameters for better performance.
5.  **Prediction:** Use the best-performing model to make predictions on the actual test dataset (the part of `df` where `Purchase` was initially `NaN`).

## How to Run the Project
1.  Clone this repository:
    ```bash
    git clone <repository_url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd Black-Friday-Sales-Prediction
    ```
3.  Ensure you have the necessary datasets (`train.csv`, `test.csv`) in the root directory.
4.  Install the required Python packages:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
5.  Open and run the Jupyter Notebook (or Python script) provided in the repository to execute the EDA, feature engineering, and model training steps.
