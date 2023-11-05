import streamlit as st
import pandas as pd
from main import *
import time
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
    type = None
    flag = ""
    # Page layout
    st.set_page_config(layout="wide")

    # Sidebar
    st.sidebar.title("NOCODESK")
    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))

    # Main content area
    if pages[selected_page] == 1:
        # Page 1: About the Product
        st.title("About NOCODESK")
        # Add content about the product here

    elif pages[selected_page] == 2:
        # Page 2: Learning Tab
        st.title("Learn ML Algos")
        # Add content for the learning tab here

    elif pages[selected_page] == 3:
        # Page 3: Pipeline Tab
        st.title("Data Playground")
        
        # Subpage selection
        subpages = {
            "Select": "Select",
            "Data Preprocessing": "data",
            "Exploratory Data Analysis": "eda",
            "Model Selection": "model_selection",
            "Model Training": "model_training",
            "Model Evaluation": "model_evaluation",
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
                type = "Regression"
            else:
                st.success("This is a Classification based Dataset") 
                type = "Classification"

            if type == "Regression":    
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
                st.write("Final Data after Preprocessing Done")
                st.write(df_all.head())
                st.success("Preprocessing Completed")


        elif subpages[selected_subpage] == "eda":
            st.header("Exploratory Data Analysis")

            # Define EDA options
            eda_options = st.multiselect("Select EDA methods to apply", ["Summary Statistics", "Data Visualization", "Correlation Analysis", "Distribution Plots", "Pair Plots", "Categorical Data Visualization", "Box Plots", "Time Series Analysis", "Multivariate Analysis"])

            exploratory_data_analysis(data, eda_options)

        elif subpages[selected_subpage] == "model_selection":
            # Model Selection
            st.header("Model Selection")
            # Add content for model selection here

        elif subpages[selected_subpage] == "model_training":
            # Model Training
            st.header("Model Training")
            # Add content for model training here

        elif subpages[selected_subpage] == "model_evaluation":
            # Model Evaluation
            st.header("Model Evaluation")
            # Add content for model evaluation here

    elif pages[selected_page] == 4:
        # Page 4: Neural Networks Tab
        st.title("Neural Networks")
        # Add content for the neural networks tab here

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



# Run the app
if __name__ == '__main__':
    nocodesk_Main()
