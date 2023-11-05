import streamlit as st
import pandas as pd
from main import *
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

def nocodesk_Main():
    # Define the pages
    pages = {
        "About NOCODESK": 1,
        "Learn ML Algos": 2,
        "Data Playground": 3,
        "Neural Networks": 4,
        "About Hackathons and Future Goals": 5,
    }
    # Define the pre-loaded datasets
    car_price_data = pd.read_csv("CarPrice.csv")  # You should replace this with the actual path to your CSV file
    iris_data = pd.read_csv("IRIS.csv")
    problemtype = None
    flag = ""
    # Page layout
    st.set_page_config(layout="wide")

    # Sidebar
    st.sidebar.title("NOCODESK")
    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    # Main content area
    if pages[selected_page] == 1:
        # Page 1: About the Product
        st.title("NOCODESK")
    
        st.write("Welcome to NOCODESK, your one-stop solution for simplified machine learning pipelines.")
    
        st.subheader("Our Mission")
        st.write("Our mission is to bridge the gap between theory and practical implementation of Artificial Intelligence (AI) and Machine Learning (ML). NOCODESK is designed to empower users with limited coding experience to perform data preprocessing, model selection, training, and evaluation with ease. We aim to make AI and ML accessible to everyone, regardless of their technical background.")
    
        st.subheader("Features")
        st.write("NOCODESK offers the following features:")
        st.write("- Easy-to-use interface for data preprocessing")
        st.write("- Model selection for both classification and regression problems")
        st.write("- Model training and evaluation with various metrics")
        st.write("- Visualizations for data exploration and model comparison")
    
        st.subheader("Get Started")
        st.write("To get started with NOCODESK, navigate to the 'Data Playground' and follow the step-by-step instructions to preprocess your data, select a model, train it, and evaluate its performance.")

    elif pages[selected_page] == 2:
        #Learning Tab
        st.title("Learn ML Algorithms")
        
        st.sidebar.header("Choose a Model Type")
        model_type = st.sidebar.selectbox("Select a model type", ["Regression", "Classification"])

        if model_type == "Regression":
            st.subheader("Linear Regression")
            st.write("Linear regression is a simple and widely used algorithm for regression tasks.")
            st.write("It models the relationship between a dependent variable 'y' and one or more independent variables 'x' as a linear equation.")
            st.write("The linear regression equation is:")
            st.latex(r'y = mx + b')
            st.write("Where 'm' is the slope and 'b' is the intercept.")
            st.write("Regression - Linear Regression Example")
            st.write("In this example, we have a simple dataset with two features 'X' and 'Y'.")
            st.write("The goal is to predict 'Y' based on 'X' using Linear Regression.")
            regression_data = pd.DataFrame({
                'X': [24, 50, 15, 38, 87, 36],
                'Y': [21.54945196, 47.46446305, 17.21865634, 36.58639803, 87.28898389, 32.46387493]
            })
            

            # Linear Regression Model
            X_reg = regression_data[['X']]
            y_reg = regression_data['Y']
            model_reg = LinearRegression()
            model_reg.fit(X_reg, y_reg)
            y_pred_reg = model_reg.predict(X_reg)

            

            # Plot original and predicted data
            plt.figure(figsize=(8, 6))
            plt.scatter(X_reg, y_reg, color='blue', label='Original Data')
            plt.plot(X_reg, y_pred_reg, color='red', label='Predicted Data')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Linear Regression Example')
            plt.legend()
            st.pyplot(plt)

            st.write("Original Data:")
            st.write(regression_data)

            st.write("Predicted Data:")
            predicted_data_reg = pd.DataFrame({'X': X_reg['X'], 'Predicted_Y': y_pred_reg})
            st.write(predicted_data_reg)


        elif model_type == "Classification":
        
            st.subheader("Logistic Regression")
            st.write("Logistic regression is a widely used algorithm for binary classification tasks.")
            st.write("It models the probability of a binary outcome as a logistic function of the input features.")
            st.write("The logistic regression equation is:")
            st.latex(r'P(y=1) = \frac{1}{{1 + e^{-(mx + b)}}')
            st.write("Where 'm' is the slope and 'b' is the intercept.")
            st.write("Classification - Logistic Regression Example")
            st.write("In this example, we have a binary classification dataset with two features 'Feature1' and 'Feature2'.")
            st.write("The goal is to predict the 'Target' based on 'Feature1' and 'Feature2 using Logistic Regression.")
            classification_data = pd.DataFrame({
                'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'Feature2': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                'Target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            })

            # Logistic Regression Model
            X_cls = classification_data[['Feature1', 'Feature2']]
            y_cls = classification_data['Target']
            model_cls = LogisticRegression()
            model_cls.fit(X_cls, y_cls)
            predictions_cls = model_cls.predict(X_cls)



            # Create a mesh grid for visualization
            h = .02
            x_min, x_max = X_cls['Feature1'].min() - 1, X_cls['Feature1'].max() + 1
            y_min, y_max = X_cls['Feature2'].min() - 1, X_cls['Feature2'].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = model_cls.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot decision boundary
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
            plt.scatter(X_cls['Feature1'], X_cls['Feature2'], c=y_cls, cmap=plt.cm.coolwarm, s=20)
            plt.xlabel('Feature1')
            plt.ylabel('Feature2')
            plt.title('Logistic Regression - Decision Boundary')
            plt.legend()
            st.pyplot(plt)

            st.write("Original Data:")
            st.write(classification_data)
            st.write("Predicted Classes:")
            st.write(predictions_cls)

    elif pages[selected_page] == 3:
        # Page 3: Pipeline Tab
        st.title("Data Playground")
        
        # Subpage selection
        subpages = {
            "Select": "Select",
            "Data Preprocessing": "data",
            "Exploratory Data Analysis": "eda",
            "Model Training & Evaluation": "model",
        }

        selected_subpage = st.sidebar.selectbox("Select a subpage", list(subpages.keys()))
        
        if subpages[selected_subpage] == "Select":
            st.write("This is the playgorund")
            st.warning("Write about the playground feature here:")

        # Dataset selection
        dataset_option = st.radio("Select a dataset option", ["Car Price Dataset (Regression)", "Iris Flower Dataset (Classification)", "Upload Your Own File"])

        if dataset_option == "Car Price Dataset (Regression)":
            # Use the pre-loaded Car Price dataset
            data = car_price_data
            st.subheader("Car Price Dataset ")
            st.write(car_price_data)

        elif dataset_option == "Iris Flower Dataset (Classification)":
            # Use the pre-loaded Iris Flower dataset
            data = iris_data
            st.subheader("Iris Flower Dataset")
            st.write(iris_data)

        else:
            # Allow the user to upload their own file
            st.subheader("Upload Your Own File")
            data = file_upload()
            try:
                st.write(data)
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
        

        try:
            st.subheader("Select Target Column")
            target_column = st.selectbox("Select the target column", data.columns)

            # Split data into numerical and categorical columns
            df_num = data.select_dtypes(include=['int64', 'float64'])
            df_cat = data.select_dtypes(exclude=['int64', 'float64'])
        
            if data[target_column].dtype in ['int64', 'float64']:
                st.success("This is a Regression based Dataset ")
                df_num = data.select_dtypes(include=['int64', 'float64'])
                problemtype = "Regression"
            else:
                st.success("This is a Classification based Dataset") 
                problemtype = "Classification"

            if problemtype == "Regression":    
                df_num = df_num.drop(target_column, axis=1)
        except Exception as e:
            st.error(f"Please Input your data first")

        

        if subpages[selected_subpage] == "data":
            
            # Define empty DataFrames for scaled and encoded data
            scaled = pd.DataFrame()
            encoded = pd.DataFrame()

            # Data Upload & Preprocessing
            st.header("Data Preprocessing")
            st.write("The Numerical Data is:")
            st.write(df_num.head())
            st.write("The Categorical Data is:")
            st.write(df_cat.head())
            print(df_num.columns)
            # Data Preprocessing Options - Step 1: Scaling
            st.subheader("Step 1: Scaling Numerical Data")
            scaling_columns = st.multiselect("Select columns for scaling", df_num.columns)

            scaling_method = st.selectbox("Select Scaling Method", ["None", "Standardization", "Min-Max Scaling", "Robust Scaling", "Log Transformation", "Box-Cox Transformation"])
            if scaling_method == "None":
                scaling_method = None

            if st.button("Scale Data"):
                scaled_data = data_preprocessing(df_num,scaling_method=scaling_method, scaling_columns=scaling_columns)
                pd.concat([scaled, scaled_data], axis= 1)
                with st.spinner('Wait for it...'):
                        time.sleep(2)
                        st.success('Done!')
                flag = flag + "a"
            if "a" in flag:
                # Show preprocessed data
                st.subheader("Scaled Data")
                st.write(scaled_data.head())

            # Data Preprocessing Options - Step 2: Encoding
            st.subheader("Step 2: Encoding for Categorical Data")
            encoding_columns = st.multiselect("Select columns for encoding", df_cat.columns)

            encoding_method = st.selectbox("Select Encoding Method", ["None", "One-Hot Encoding", "Label Encoding"])
            if encoding_method == "None":
                encoding_method = None

            if st.button("Encode Data"):
                encoded_data = data_preprocessing(df_cat, encoding_method=encoding_method, encoding_columns=encoding_columns)
                with st.spinner('Wait for it...'):
                    time.sleep(2)
                    st.success('Done!')
                flag = flag + "b"
            if "b" in flag:
                # Show Encoding data
                st.header("Encoded Data:")
                st.write(encoded_data.head())

            if st.sidebar.button("Finalize the Preprocessing"):
                df_all = pd.concat([scaled, encoded], axis= 1)
                # st.write("Final Data after Preprocessing Done")
                # st.write(df_all.head())
                st.success("Preprocessing Completed")


        elif subpages[selected_subpage] == "eda":
            st.header("Exploratory Data Analysis")

            # Define EDA options
            eda_options = st.multiselect("Select EDA methods to apply", ["Summary Statistics", "Data Visualization", "Correlation Analysis", "Distribution Plots", "Pair Plots", "Box Plots", "Time Series Analysis", "Multivariate Analysis"])

            exploratory_data_analysis(data, eda_options)

        elif subpages[selected_subpage] == "model":
            # Model Selection
            st.header("Model Training & Evaluation")
            if problemtype == "Classification":
                model_choice = st.selectbox("Select Classification Model", ["Logistic Regression", "Decision Trees", "Random Forest", "Support Vector Machine (SVM)", "K-Nearest Neighbors (K-NN)"])
            else:
                model_choice = st.selectbox("Select Regression Model", ["Linear Regression", "Decision Trees", "Random Forest Regression", "Support Vector Machine (SVM) Regression", "Gradient Boosting Regression (XGBoost)", "Gradient Boosting Regression (LightGBM)"])

            if st.button("Perform Model Training and Evaluation"):
                # Split data into features (X) and target (y)
                X = data.drop(target_column, axis=1)
                y = data[target_column]
                if y.dtype == 'object':
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)
                    # Split the data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
                else:
                    # Split the data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                
                # Perform model selection
                selected_model = model_selection(data, problemtype.lower(), model_choice)

                if not isinstance(selected_model, str):
                    # Perform model training
                    trained_model = model_training(selected_model, X_train, y_train)
                    # Perform model evaluation
                    evaluation_results = model_evaluation(trained_model, X_test, y_test, problemtype.lower(), cv=5)
                    with st.spinner('Wait for it...'):
                        time.sleep(2)
                        st.success('Done!')
                    # Display evaluation results
                    st.subheader("Model Evaluation Results")
                    st.write(evaluation_results)
                else:
                    st.warning(selected_model)


    elif pages[selected_page] == 4:
        # Page 4: Neural Networks Tab
        st.title("Neural Networks")
        st.header("Introduction to Neural Networks")
        st.write("Neural Networks are a class of machine learning models inspired by the human brain.")
        st.write("Nodes or artificial neurons in a neural network are arranged in layers. There are three types of layers:")
        st.write("1. Input Layer: Takes input features.")
        st.write("2. Hidden Layers: Process data between input and output layers.")
        st.write("3. Output Layer: Provides the final output of the network.")
    
        # Insert an image to show the structure of a neural network (e.g., from a URL)
        network_diagram_url = "https://www.researchgate.net/profile/Pavitra-Kumar-5/publication/341716390/figure/fig1/AS:896335746179074@1590714503580/Basic-structure-of-neural-network.png"
        st.image(network_diagram_url, caption="Neural Network Structure", use_column_width=True)
    
        st.header("Interactive Example: Convolutional Neural Network (CNN)")
        st.write("In this example, you can interactively change the number of hidden layers and observe the impact on the network's performance.")
    
        # Slider for the number of hidden layers
        num_hidden_layers = st.slider("Number of Hidden Layers", 1, 10, 3)
    

        # Page 4: Neural Networks Tab
        st.title("Neural Networks")
        st.write("Neural Networks (NN) are a type of machine learning model inspired by the human brain. They consist of nodes organized into input, output, and hidden layers. Each node performs a weighted sum of its inputs and passes the result through an activation function.")

        st.header("Neural Network Structure")
        st.write("A neural network typically consists of the following layers:")
        st.subheader("Input Layer")
        st.write("The input layer receives the data features. Each node in the input layer corresponds to a feature in the dataset.")
    
        num_hidden_layers = st.slider("Number of Hidden Layers", 0, 5, 1)
        st.subheader("Hidden Layers")
        st.write("Hidden layers perform intermediate processing and feature extraction. The number of hidden layers and nodes in each layer can be adjusted to improve model performance.")
    
        model = create_cnn(num_hidden_layers)

        st.subheader("Output Layer")
        st.write("The output layer produces the final predictions or classifications.")
        st.subheader("How Neural Networks Work")
        st.write("Neural networks apply mathematical transformations to input data through layers of interconnected nodes. The core equation for each node is the weighted sum of inputs passed through an activation function.")
    
        # Provide a diagram here if you have one
    
        st.header("Demo: CNN with Dummy Data")
        st.write("Let's see how a Convolutional Neural Network (CNN) works with some dummy data.")

        # Create random dummy data (images)
        num_images = 5
        image_size = 28
        dummy_data = np.random.rand(num_images, image_size, image_size, 3)  # RGB images
    
        st.write("Generated Dummy Data:")
        st.image(dummy_data, width=200)

        # Initialize a basic CNN model
        st.subheader("Convolutional Neural Network (CNN)")
        st.write("We'll use a simple CNN model to process the dummy data.")

        if st.button("Run CNN"):
            st.spinner("Running CNN...")

            # Dummy CNN model
            if num_hidden_layers > 0:
                st.write("Creating a CNN with", num_hidden_layers, "hidden layers.")
                st.success("CNN created!")

            # Provide the model architecture summary
            # st.subheader("CNN Model Architecture")
            # model.summary()

            # Make predictions with the dummy data (no real labels)
            # dummy_predictions = model.predict(dummy_data)
            # st.subheader("CNN Output")
            # st.write("The CNN has processed the dummy data. These are the model's predictions (no real labels provided):")
            # st.write(dummy_predictions)

            st.success("CNN has finished processing.")

    elif pages[selected_page] == 5:
        # Page 5: About Hackathons and Future Goals
        st.title("About Hackathons and Future Goals")
        # Add content about hackathons and future goals here


def file_upload():
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="file_upload")

    if uploaded_file is not None:
        # Check file size
        max_file_size = 50 * 1024 * 1024  # 50 MB
        if uploaded_file.size > max_file_size:
            st.error("File size exceeds the maximum allowed size (50 MB). Please upload a smaller file.")
            return None

        # Read the file into a DataFrame
        try:
            if uploaded_file.type == 'application/vnd.ms-excel' or uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                # Excel file (XLS or XLSX)
                df = pd.read_excel(uploaded_file)
            else:
                # CSV file
                df = pd.read_csv(uploaded_file)

            return df
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            return None

    return None


# Create a CNN model
def create_cnn(num_hidden_layers):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for _ in range(num_hidden_layers):
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
# Run the app
if __name__ == '__main__':
    nocodesk_Main()
