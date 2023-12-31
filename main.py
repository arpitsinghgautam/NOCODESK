import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from category_encoders import TargetEncoder, BinaryEncoder, CountEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split,GridSearchCV,cross_val_score
from sklearn.model_selection import cross_val_score
import streamlit as st

def data_preprocessing(data, scaling_method=None, encoding_method=None, scaling_columns=None, encoding_columns=None):
    # Data Cleaning (you can add more data cleaning steps as needed)
    data.dropna(inplace=True)  # Example: Remove rows with missing values

    # Feature Scaling
    if scaling_columns:
        for column in scaling_columns:
            try:
                scaler = None
                if column in data.columns:
                    if scaling_method == "Standardization":
                        scaler = StandardScaler()
                    elif scaling_method == "Min-Max Scaling":
                        scaler = MinMaxScaler()
                    elif scaling_method == "Robust Scaling":
                        scaler = RobustScaler()
                    elif scaling_method == "Log Transformation":
                        data[column] = np.log1p(data[column])
                    elif scaling_method == "Box-Cox Transformation":
                        pt = PowerTransformer(method='box-cox')
                        data[column] = pt.fit_transform(data[column])

                    if scaler:
                        data[column] = scaler.fit_transform(data[[column]])
            except ValueError:
                st.warning(f"Skipping scaling for column '{column}' (not suitable for scaling).")

    # Encoding Categorical Variables
    if encoding_columns:
        encoded_data = data.copy()
        for column in encoding_columns:
            try:
                if column in data.columns:
                    if encoding_method == "One-Hot Encoding":
                        encoded_data = pd.get_dummies(encoded_data, columns=[column])
                    elif encoding_method == "Label Encoding":
                        le = LabelEncoder()
                        encoded_data[column] = le.fit_transform(data[column])
            except ValueError:
                st.warning(f"Skipping encoding for column '{column}' (not suitable for encoding).")

        data = encoded_data
    return data



def exploratory_data_analysis(data, methods_to_apply=None):

    if methods_to_apply is None:
        methods_to_apply = ["Summary Statistics", "Data Visualization", "Correlation Analysis", "Distribution Plots",
                            "Pair Plots", "Box Plots",
                            "Multivariate Analysis"]

    if not isinstance(data, pd.DataFrame):
        st.error("Input data is not a DataFrame.")
        return

    if "Summary Statistics" in methods_to_apply:
        st.subheader("Summary Statistics:")
        st.write(data.describe())
        st.write("\n")

    if "Data Visualization" in methods_to_apply:
        st.subheader("Data Visualization:")
        st.set_option('deprecation.showPyplotGlobalUse', False)  # To avoid deprecation warning
        plt.figure(figsize=(12, 6))
        sns.countplot(data=data)
        st.pyplot(plt)
        st.write("\n")

    if "Correlation Analysis" in methods_to_apply:
        encoded_data = data.copy()
        label_encoder = LabelEncoder()
        for column in data.columns:
            if data[column].dtype == 'object':
                encoded_data[column] = label_encoder.fit_transform(data[column])
        st.subheader("Correlation Analysis:")
        corr_matrix = encoded_data.corr()
        st.write(corr_matrix)
        st.write("\n")

    if "Distribution Plots" in methods_to_apply:
        st.subheader("Distribution Plots:")
        for column in data.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(data[column], kde=True)
            plt.title(f"Distribution of {column}")
            st.pyplot(plt)
        st.write("\n")

    if "Pair Plots" in methods_to_apply:
        st.subheader("Pair Plots:")
        scatter_matrix(data, alpha=0.2, figsize=(12, 8), diagonal="kde")
        st.pyplot(plt)
        st.write("\n")

    # if "Categorical Data Visualization" in methods_to_apply:
    #     st.subheader("Categorical Data Visualization:")
    #     df_cat = data.select_dtypes(exclude=['int64', 'float64'])
    #     for column in df_cat.columns:
    #         plt.figure(figsize=(6, 4))
    #         sns.countplot(data=df_cat[column])
    #         sns.histplot(df_cat[column], kde=True)
    #         plt.title(f"Distribution of {column}")
    #         st.pyplot(plt)
    #     st.write("\n")
    
    if "Box Plots" in methods_to_apply:
        st.subheader("Box Plots:")
        for column in data.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=data, y=column)
            plt.title(f"Box Plot of {column}")
            st.pyplot(plt)
        st.write("\n")

    if "Time Series Analysis" in methods_to_apply:
        st.subheader("Time Series Analysis:")
        plot_acf(data, lags=30)
        st.write("\n")

    if "Multivariate Analysis" in methods_to_apply:
        st.subheader("Multivariate Analysis:")
        df_num = data.select_dtypes(include=['int64', 'float64'])
        plt.figure(figsize=(15,6))
        correlation = df_num.corr( method='pearson' )
        sns.heatmap( correlation, annot=True );
        st.pyplot(plt)
        st.write("\n")


def model_selection(data, problem_type, model_choice):
    if problem_type == 'classification':
        if model_choice == 'Logistic Regression':
            model = LogisticRegression()
        elif model_choice == 'Decision Trees':
            model = DecisionTreeClassifier()
        elif model_choice == 'Random Forest':
            model = RandomForestClassifier()
        elif model_choice == 'Support Vector Machine (SVM)':
            model = SVC()
        elif model_choice == 'K-Nearest Neighbors (K-NN)':
            model = KNeighborsClassifier()
        else:
            return "Invalid model choice for classification."

    elif problem_type == 'regression':
        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Decision Trees':
            model = DecisionTreeRegressor()
        elif model_choice == 'Random Forest Regression':
            model = RandomForestRegressor()
        elif model_choice == 'Support Vector Machine (SVM) Regression':
            model = SVR()
        elif model_choice == 'Gradient Boosting Regression (XGBoost)':
            model = XGBRegressor()
        elif model_choice == 'Gradient Boosting Regression (LightGBM)':
            model = LGBMRegressor()
        else:
            return "Invalid model choice for regression."
    else:
        return "Invalid problem type."

    return model



def model_training(model, X_train, y_train):
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        return str(e)


def model_evaluation(model, X_test, y_test, problem_type, cv=None):
    try:
        if problem_type == 'classification':
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted') 
            recall = recall_score(y_test, y_pred, average='weighted')  
            f1 = f1_score(y_test, y_pred, average='weighted')
            evaluation_metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
        elif problem_type == 'regression':
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            evaluation_metrics = {
                'Root Mean Squared Error (RMSE)': rmse
            }
        else:
            return "Invalid problem type."

        # Perform cross-validation if cv is specified
        if cv:
            cross_val_scores = cross_val_score(model, X_test, y_test, cv=cv)
            evaluation_metrics['Cross-Validation Mean Score'] = cross_val_scores.mean()

        return evaluation_metrics
    except Exception as e:
        return str(e)




def plot_compare(original_data, preprocessed_data, model, X_test, y_test, class_labels):
    # Assuming class_labels are provided for creating a confusion matrix
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)

    # Decision Boundary and Separation
    plt.figure(figsize=(12, 6))
    
    # Plot original data (if applicable)
    if original_data is not None:
        plt.subplot(131)
        # Plot the original data points with their labels and decision boundary
        # You'll need to customize this part depending on the nature of your data and model
        # Example: plt.scatter(original_data['feature1'], original_data['feature2'], c=original_data['target'])
        plt.title("Original Data")

    # Plot preprocessed data
    plt.subplot(132)
    # Plot the preprocessed data points with their labels and decision boundary (if available)
    # Example: plt.scatter(preprocessed_data['feature1'], preprocessed_data['feature2'], c=preprocessed_data['target'])
    plt.title("Preprocessed Data")

    # Plot model predictions
    plt.subplot(133)
    # Plot the model's predictions with the actual labels and decision boundary (if available)
    # Example: plt.scatter(X_test['feature1'], X_test['feature2'], c=model.predict(X_test))
    plt.title("Model Predictions")

    plt.tight_layout()

    # Confusion Matrix
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    cm = confusion_matrix(y_test_encoded, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix")

    # Pie Chart
    plt.subplot(122)
    class_counts = [len(y_test[y_test == label]) for label in class_labels]
    plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=140)
    plt.title("Class Distribution")

    plt.tight_layout()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_encoded, model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test_encoded, model.predict_proba(X_test)[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.show()

def generate_regression_dummy_data():
    # Generate and return regression dummy data
    return pd.DataFrame(data={'X': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]})

def generate_classification_dummy_data():
    # Generate and return classification dummy data
    return pd.DataFrame(data={'Feature1': [1, 2, 3, 4, 5], 'Feature2': [4, 3, 5, 2, 1], 'Target': [1, 0, 1, 0, 1]})

def algorithm_description(algorithm):
    # Display algorithm description
    if algorithm == "Linear Regression":
        st.write("Linear regression is a simple linear modeling technique used to model the relationship between a dependent variable and one or more independent variables. It fits a linear equation to the data, making it useful for predicting a continuous target variable.")
        st.write("Mathematically, linear regression can be represented as:\n\ny = mx + b")
        st.write("Where 'y' is the dependent variable, 'x' is the independent variable, 'm' is the slope, and 'b' is the intercept.")
    
  
def algorithm_equations(algorithm):
    # Display math equations
    if algorithm == "Linear Regression":
        st.write("Math Equation:")
        st.latex(r'y = mx + b')
        # You can also show a diagram if available

    # Add equations for other algorithms as needed

def apply_algorithm(algorithm, dummy_data):
    # Apply the selected algorithm to the dummy data and return results
    if algorithm == "Linear Regression":
        # Apply linear regression to the data and return predictions
        model = LinearRegression()
        X = dummy_data[['X']]
        y = dummy_data['y']
        model.fit(X, y)
        predictions = model.predict(X)
        return predictions

    # Add algorithm-specific logic for other algorithms

def algorithm_visual_explanation(algorithm):
    # Visual explanation for algorithm
    if algorithm == "Linear Regression":
        st.write("Visual Explanation:")
        # Show a diagram or plot to explain the algorithm



def download_code(selected_functions, filename):
    # Define the code for the selected functions
    code = ""

    if "data_preprocessing" in selected_functions:
        code += """
# Data Preprocessing
def data_preprocessing(data):
    # Add your data preprocessing code here
    pass
"""

    if "exploratory_data_analysis" in selected_functions:
        code += """
# Exploratory Data Analysis
def exploratory_data_analysis(data):
    # Add your exploratory data analysis code here
    pass
"""

    if "model_selection" in selected_functions:
        code += """
# Model Selection
from sklearn.linear_model import LogisticRegression

def model_selection(data, problem_type, model_choice):
    if problem_type == 'classification':
        if model_choice == 'Logistic Regression':
            model = LogisticRegression()
        else:
            return "Invalid model choice for classification."
    else:
        return "Invalid problem type."
    return model
"""

    if "model_training" in selected_functions:
        code += """
# Model Training
def model_training(model, X_train, y_train):
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        return str(e)
"""

    if "model_evaluation" in selected_functions:
        code += """
# Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def model_evaluation(model, X_test, y_test, problem_type, cv=None):
    try:
        if problem_type == 'classification':
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            evaluation_metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
        else:
            return "Invalid problem type."

        if cv:
            cross_val_scores = cross_val_score(model, X_test, y_test, cv=cv)
            evaluation_metrics['Cross-Validation Mean Score'] = cross_val_scores.mean()

        return evaluation_metrics
    except Exception as e:
        return str(e)
"""

    # Save the code to a .py file
    with open(filename, "w") as file:
        file.write(code)

# Example usage to download the code for selected functions
selected_functions = ["data_preprocessing", "exploratory_data_analysis", "model_selection", "model_training", "model_evaluation"]
download_code(selected_functions, "Nocodesk_Coding.py")
