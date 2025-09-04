import streamlit as st
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, silhouette_score, mean_squared_error
)
from sklearn.cluster import KMeans
from io import BytesIO

# App Title
st.title("Comprehensive Data Processing App")
st.header("Step 1: Upload a CSV File")

# Initialize session state for dataset and functions
if "data" not in st.session_state:
    st.session_state.data = None

# File Upload Section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    delim = st.selectbox("Select a delimiter:", options=["", ";", ","])
    if delim in [";", ","]:
        st.session_state.data = pd.read_csv(uploaded_file, delimiter=delim)
    else:
        st.session_state.data = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")

    if st.checkbox("Show/Hide DataFrame"):
        st.dataframe(st.session_state.data)

    # Optimize for large datasets
    if len(st.session_state.data) > 10000:
        st.warning("Dataset has more than 10,000 rows. Subsampling to 10,000 rows for analysis.")
        st.session_state.data = st.session_state.data.sample(10000, random_state=42)

# Sidebar Functionality
if st.session_state.data is not None:
    st.sidebar.header("Options")

    # ================== Exploration Section ==================
    st.sidebar.subheader("Exploration")
    if st.sidebar.checkbox("Statistical Description"):
        st.header("Statistical Description")
        st.write(st.session_state.data.describe())

    if st.sidebar.checkbox("Check Null Values"):
        st.header("Null Values")
        st.write("Missing values per column:")
        st.write(st.session_state.data.isnull().sum())

    if st.sidebar.checkbox("Detect Outliers"):
        st.header("Outliers")
        numerical_data = st.session_state.data.select_dtypes(include=["number"])
        if not numerical_data.empty:
            Q1 = numerical_data.quantile(0.25)
            Q3 = numerical_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).sum()
            st.write("Number of potential outliers in each numerical column:")
            st.write(outliers)
        else:
            st.write("No numerical columns available to detect outliers.")

    if st.sidebar.checkbox("Check Duplicates"):
        st.header("Duplicate Rows")
        duplicates = st.session_state.data.duplicated().sum()
        st.write(f"Number of duplicated rows: {duplicates}")

    # ================== Preprocessing Section ==================
    st.sidebar.subheader("Preprocessing")

    # Fill Missing Values
    if st.sidebar.checkbox("Fill Missing Values with Mode"):
        st.header("Fill Missing Values")
        for col in st.session_state.data.columns:
            if st.session_state.data[col].isnull().sum() > 0:
                mode_value = st.session_state.data[col].mode()[0]
                st.session_state.data[col].fillna(mode_value, inplace=True)
        st.success("Missing values have been filled with mode.")

    # Drop Missing Values
    if st.sidebar.checkbox("Drop Rows with Missing Values"):
        st.header("Drop Rows with Missing Values")
        st.session_state.data.dropna(inplace=True)
        st.success("Rows with missing values have been dropped.")

    # Drop Outliers
    if st.sidebar.checkbox("Drop Outliers"):
        st.header("Drop Outliers")
        numerical_data = st.session_state.data.select_dtypes(include=["number"])
        if not numerical_data.empty:
            Q1 = numerical_data.quantile(0.25)
            Q3 = numerical_data.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).any(axis=1)
            st.session_state.data = st.session_state.data[mask]
            st.success("Outliers have been removed.")
        else:
            st.write("No numerical columns available to drop outliers.")

    # Transform Categorical to Numerical
    if st.sidebar.checkbox("Transform Categorical to Numerical"):
        st.header("Transform Categorical Columns to Numerical")
        categorical_cols = st.session_state.data.select_dtypes(include=["object", "category"]).columns
        if not categorical_cols.empty:
            select_all = st.checkbox("Select All Categorical Columns")
            if select_all:
                selected_columns = list(categorical_cols)
            else:
                selected_columns = st.multiselect("Select columns to transform:", categorical_cols)

            transformation_type = st.radio("Select transformation type:", ["Label Encoding", "One-Hot Encoding"])

            if selected_columns:
                if transformation_type == "Label Encoding":
                    for col in selected_columns:
                        st.session_state.data[col] = st.session_state.data[col].astype("category").cat.codes
                    st.success(f"Selected columns transformed using {transformation_type}.")
                elif transformation_type == "One-Hot Encoding":
                    st.session_state.data = pd.get_dummies(st.session_state.data, columns=selected_columns)
                    st.success(f"Selected columns transformed using {transformation_type}.")
            else:
                st.warning("Please select at least one column to transform.")
        else:
            st.write("No categorical columns found.")

    # Delete Duplicates
    if st.sidebar.checkbox("Delete Duplicate Rows"):
        st.header("Delete Duplicates")
        st.session_state.data.drop_duplicates(inplace=True)
        st.success("Duplicate rows have been removed.")

    # ================== Correlation Matrix Section ==================
    if st.sidebar.button("Show Correlation Matrix"):
        st.header("Correlation Matrix")
        correlation_matrix = st.session_state.data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ================== Data Visualization Section ==================
    if st.sidebar.checkbox("Visualize Data"):
        st.header("Data Visualization")
        plot_type = st.selectbox("Select Plot Type:", ["", "Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot"])

        if plot_type:
            columns = st.session_state.data.columns
            x_axis = st.selectbox("Select X-Axis:", columns)
            y_axis = st.selectbox("Select Y-Axis:", columns)

            if x_axis and y_axis:
                fig, ax = plt.subplots()

                if plot_type == "Scatter Plot":
                    ax.scatter(st.session_state.data[x_axis], st.session_state.data[y_axis])
                    ax.set_title("Scatter Plot")
                elif plot_type == "Line Plot":
                    ax.plot(st.session_state.data[x_axis], st.session_state.data[y_axis])
                    ax.set_title("Line Plot")
                elif plot_type == "Bar Plot":
                    ax.bar(st.session_state.data[x_axis], st.session_state.data[y_axis])
                    ax.set_title("Bar Plot")
                elif plot_type == "Histogram":
                    ax.hist(st.session_state.data[x_axis], bins=30)
                    ax.set_title("Histogram")
                elif plot_type == "Box Plot":
                    sns.boxplot(x=st.session_state.data[x_axis], y=st.session_state.data[y_axis], ax=ax)
                    ax.set_title("Box Plot")

                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                st.pyplot(fig)
            else:
                st.warning("Please select both X and Y axes.")

    # ================== Machine Learning Section ==================
    st.sidebar.subheader("Machine Learning Options")
    ml_option = st.sidebar.radio("Select ML Task", ["", "Classification", "Regression", "Clustering"])

    if ml_option in ["Classification", "Regression"]:
        target_column = st.sidebar.selectbox("Select Target Column:", st.session_state.data.columns)
        feature_columns = st.sidebar.multiselect("Select Feature Columns:", 
                                                  st.session_state.data.columns.drop(target_column))
        if len(feature_columns) == 0:
            st.warning("Please select at least one feature column.")
        else:
            X = st.session_state.data[feature_columns]
            y = st.session_state.data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            start_time = time.time()

            if ml_option == "Classification":
                model_type = st.radio("Select Classification Model:", 
                                      ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest"])

                if model_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=200)
                elif model_type == "K-Nearest Neighbors":
                    n_neighbors = st.slider("Select number of neighbors (k):", 1, 10, 3)
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                elif model_type == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                elif model_type == "Random Forest":
                    n_estimators = st.slider("Select number of trees:", 10, 100, 50)
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                execution_time = time.time() - start_time

                # Evaluation Metrics
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy * 100:.2f}%")
                st.write("**Accuracy**: The percentage of correct predictions out of total predictions.")

                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                st.write(f"Precision: {precision * 100:.2f}%")
                st.write("**Precision**: The proportion of true positive predictions out of all positive predictions.")

                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                st.write(f"Recall: {recall * 100:.2f}%")
                st.write("**Recall**: The proportion of true positive predictions out of all actual positives.")

                f1 = f1_score(y_test, y_pred, average='weighted')
                st.write(f"F1-Score: {f1 * 100:.2f}%")
                st.write("**F1-Score**: The harmonic mean of precision and recall, balancing the two.")

                mcc = matthews_corrcoef(y_test, y_pred)
                st.write(f"Matthews Correlation Coefficient: {mcc:.2f}")
                st.write("**Matthews Correlation Coefficient (MCC)**: A measure of the quality of binary classifications, considering all confusion matrix values.")

                st.write(f"Execution Time: {execution_time:.2f} seconds")

            elif ml_option == "Regression":
                model_type = st.radio("Select Regression Model:", ["Linear Regression", "Random Forest"])

                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Random Forest":
                    n_estimators = st.slider("Select number of trees:", 10, 100, 50)
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                execution_time = time.time() - start_time

                mse = mean_squared_error(y_test, y_pred)

                st.write(f"{model_type} Mean Squared Error:")
                st.write(f"{mse:.2f}")
                st.write(f"Execution Time: {execution_time:.2f} seconds")

    elif ml_option == "Clustering":
        numerical_columns = st.sidebar.multiselect("Select Columns for Clustering:", 
                                                    st.session_state.data.select_dtypes(include=["number"]).columns)
        if len(numerical_columns) == 0:
            st.warning("Please select at least one numerical column.")
        else:
            n_clusters = st.slider("Select number of clusters:", 2, 10, 3)
            model = KMeans(n_clusters=n_clusters, random_state=42)

            clustering_data = st.session_state.data[numerical_columns]
            cluster_labels = model.fit_predict(clustering_data)
            st.session_state.data["Cluster"] = cluster_labels

            silhouette = silhouette_score(clustering_data, cluster_labels)
            st.write("Silhouette Score:")
            st.write(f"{silhouette:.2f}")

# ================== Export Processed Data ==================
st.sidebar.subheader("Download Preprocessed Data")
if st.sidebar.button("Download Processed Data"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        st.session_state.data.to_excel(writer, index=False, sheet_name="Processed_Data")
    st.download_button(
        label="Download Data as Excel",
        data=output.getvalue(),
        file_name="processed_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Reset Operations
if st.sidebar.button("Reset Operations"):
    st.session_state.data = pd.read_csv(uploaded_file)
    st.success("Dataset has been reset.")
