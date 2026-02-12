# Import all necessary libraries
from scipy import stats
from sklearn import preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, accuracy_score, 
                           confusion_matrix, classification_report, 
                           mean_squared_error, silhouette_score, 
                           calinski_harabasz_score)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.manifold import TSNE
import umap.umap_ as umap
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import chardet
from io import StringIO, BytesIO

# Page config
st.set_page_config(page_title="Data Modeling App", layout="wide")
st.title("📊 OmniAnalyzer: From Raw Data to AI Models in One Click")

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload your data: ", type=["csv", "xlsx", "xls"])

def create_gauge(value, title, color):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 100], 'color': "blue"}]}))
    st.plotly_chart(fig)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""

# Helper functions
@st.cache_data
def load_data(file, file_type):
    if file_type == 'csv':
        # Read a sample to detect encoding
        sample = file.read(10240)  # Read first 10 KB
        file.seek(0)
        result = chardet.detect(sample)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        try:
            df = pd.read_csv(file, encoding=encoding)
            if df.empty:
                st.error("The uploaded CSV file is empty.")
                return None
            return df
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
            return None
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return None
    elif file_type in ['xlsx', 'xls']:
        try:
            df = pd.read_excel(file)
            if df.empty:
                st.error("The uploaded Excel file is empty.")
                return None
            return df
        except pd.errors.EmptyDataError:
            st.error("The uploaded Excel file is empty.")
            return None
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None
    
def drop_columns(df, columns):
    try:
        df = df.drop(columns=columns)
        return df
    except Exception as e:
        print(f"Error :{e}")
        return df

def detect_task_type(df, target_column=None):
    """
    Enhanced task type detection with more sophisticated rules
    Returns: "classification", "regression", or "clustering"
    """
    if target_column is None:
        return "clustering"
    
    target_series = df[target_column]
    
    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(target_series):
        nunique = target_series.nunique()
        unique_ratio = nunique / len(target_series)
        
        # Additional heuristics for regression vs classification
        if unique_ratio > 0.1:  # More continuous values likely regression
            return "regression"
        elif nunique <= 20:  # Few unique values likely classification
            return "classification"
        else:
            # Check if values appear categorical despite being numeric
            value_counts = target_series.value_counts(normalize=True)
            if value_counts.max() > 0.5:  # One dominant value
                return "classification"
            return "regression"
    else:
        # Non-numeric target is always classification
        return "classification"

def get_models(task_type):
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier()
        }
    elif task_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }
    elif task_type == "clustering":
        return {
            "KMeans": KMeans(),
            "DBSCAN": DBSCAN(),
            "Agglomerative": AgglomerativeClustering(),
            "Gaussian Mixture": GaussianMixture()
        }

def get_metrics(task_type):
    if task_type == "classification":
        return {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": lambda y_true, y_pred: classification_report(y_true, y_pred, output_dict=True)['weighted avg']['f1-score']
        }
    elif task_type == "regression":
        return {
            "mse": mean_squared_error,
            "r2": lambda y_true, y_pred: r2_score(y_true, y_pred)
        }
    elif task_type == "clustering":
        return {}  # Metrics handled separately in clustering section

def prepare_data(df, target_column):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Convert categorical columns to numeric
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test), X.columns.tolist()

def train_single_model(model, X_train, y_train):
    return model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test, metric_func):
    y_pred = model.predict(X_test)
    return metric_func(y_test, y_pred)

def save_model(model, feature_columns):
    model_data = {
        'model': model,
        'feature_columns': feature_columns
    }
    with open('saved_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

def load_model():
    with open('saved_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_columns']

def predict_new_data(model, feature_columns, new_df):
    # Preprocess new data to match training format
    new_data = new_df[feature_columns].copy()
    categorical_cols = new_data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        new_data[col] = le.fit_transform(new_data[col].astype(str))
    return model.predict(new_data)

# Handle uploaded file
if uploaded_file:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            st.session_state.data = df_raw.copy()
            st.session_state.df = df_raw.copy()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success("✅ New data loaded successfully!")

# Use session_state data
if not st.session_state.df.empty:
    df = st.session_state.df.copy()

    # Raw Data Preview
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head())

    # Tabs
    selected_tab = st.sidebar.selectbox(
        "📌 Select Section",
        (
            "🧹 Data Preprocessing",
            "📊 Data Visualization",
            "🤖 Model Setup & Train",
            "📁 Report"
        )
    )

    if selected_tab == "🧹 Data Preprocessing":
        st.header("Data Preprocessing")
        tab_clean, tab_impute, tab_scale, tab_adv = st.tabs(["🧼 Cleaning", "🧩 Imputation", "📐 Scaling", "📊 Advanced"])
        
        with tab_clean:
            st.subheader("Data Cleaning")

            st.markdown("### 🗑️ Drop Columns")
            columns_to_drop=st.multiselect("Choose columns to delete",df.columns)
            if st.button("Delete selected columns"):
                df = drop_columns(df,columns_to_drop)
                st.session_state['df']=df
                st.success("Columns deleted successfully")

            st.subheader("Nulls & Duplicates")
            column_options = ["All Columns"] + df.columns.tolist()
            selected_columns = st.multiselect("Select columns to check/remove nulls or duplicates:", column_options)
            
            if st.button("Check Nulls"):
                if not selected_columns:
                    st.warning("Please select at least one option.")
                else:
                    if "All Columns" in selected_columns:
                        null_counts = df.isnull().sum()
                    else:
                        null_counts = df[selected_columns].isnull().sum()
                    null_counts = null_counts[null_counts > 0]
                    if null_counts.empty:
                        st.success("No missing values found in selected columns.")
                    else:
                        null_df = null_counts.reset_index()
                        null_df.columns = ['Column Name', 'Missing Values']
                        st.dataframe(null_df)
            
            if st.button("Check Duplicates"):
                if not selected_columns:
                    st.warning("Please select at least one option.")
                else:
                    if "All Columns" in selected_columns:
                        duplicate_rows = df[df.duplicated()]
                    else:
                        duplicate_rows = df[df.duplicated(subset=selected_columns)]
                    if duplicate_rows.empty:
                        st.success("No duplicate rows found.")
                    else:
                        st.error(f"Found {duplicate_rows.shape[0]} duplicate rows:")
                        st.dataframe(duplicate_rows)
            
            if st.button("Remove Duplicates"):
                if not selected_columns:
                    st.warning("Please select at least one option.")
                else:
                    initial_count = df.shape[0]
                    if "All Columns" in selected_columns:
                        df = df.drop_duplicates().copy()
                    else:
                        df = df.drop_duplicates(subset=selected_columns).copy()
                    final_count = df.shape[0]
                    removed = initial_count - final_count
                    st.session_state.df = df.copy()
                    if removed == 0:
                        st.info("No duplicates were removed.")
                    else:
                        st.success(f"Removed {removed} duplicate rows.")

        with tab_impute:
            st.subheader("Imputation")
            method = st.selectbox("Select Imputation Method", ["Simple", "KNN"])
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if method == "Simple":
                num_selection = st.radio("Numerical columns to impute:", ["Select Specific", "All Numerical"])
                selected_num_cols = st.multiselect("Select Numerical Columns for Imputation", num_cols) if num_selection == "Select Specific" else num_cols
                cat_selection = st.radio("Categorical columns to impute:", ["Select Specific", "All Categorical"])
                selected_cat_cols = st.multiselect("Select Categorical Columns for Imputation", cat_cols) if cat_selection == "Select Specific" else cat_cols
            else:
                num_selection = st.radio("Numerical columns to impute:", ["Select Specific", "All Numerical"])
                selected_num_cols = st.multiselect("Select Numerical Columns for Imputation", num_cols) if num_selection == "Select Specific" else num_cols
            
            if method == "Simple":
                strategy = st.selectbox("Select Strategy for Imputation", ["mean", "median", "most_frequent"])
            if method == "KNN" and selected_num_cols:
                n_neighbors = st.slider("Number of Neighbors (K)", min_value=1, max_value=10, value=5)
            
            if st.button("Apply Imputation"):
                try:
                    df_copy = df.copy()
                    if method == "Simple":
                        if selected_num_cols:
                            num_imputer = SimpleImputer(strategy=strategy)
                            df_copy[selected_num_cols] = num_imputer.fit_transform(df_copy[selected_num_cols])
                        if selected_cat_cols:
                            cat_imputer = SimpleImputer(strategy="most_frequent")
                            df_copy[selected_cat_cols] = cat_imputer.fit_transform(df_copy[selected_cat_cols])
                        st.session_state.df = df_copy.copy()
                        st.success("Imputation applied successfully.")
                    elif method == "KNN":
                        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
                        df_copy[selected_num_cols] = knn_imputer.fit_transform(df_copy[selected_num_cols])
                        st.session_state.df = df_copy.copy()
                        st.success("KNN Imputation applied.")
                except Exception as e:
                    st.error(f"Error during imputation: {e}")

        with tab_scale:
            st.subheader("Scaling")
            col_selection = st.radio("Select columns to transform:", ["All Numeric", "Specific Columns"])
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            cols_to_transform = st.multiselect("Select columns:", options=numeric_cols, default=numeric_cols[:1]) if col_selection == "Specific Columns" else numeric_cols
            
            scale_method = st.selectbox("Select Scaling Method", ["standard", "minmax", "robust", "log"])
            if scale_method == "standard":
                with_mean = st.checkbox("Center data", True)
                with_std = st.checkbox("Scale to unit variance", True)
            elif scale_method == "minmax":
                min_val = st.number_input("Min value", -10.0, 10.0, 0.0, 0.1)
                max_val = st.number_input("Max value", -10.0, 10.0, 1.0, 0.1)
            elif scale_method == "robust":
                q_low = st.slider("Lower quantile", 0, 49, 25)
                q_high = st.slider("Upper quantile", 51, 100, 75)
            elif scale_method == "log":
                log_base = st.selectbox("Log base", ["natural", "10", "2"])
            
            if st.button("Apply Scaling"):
                try:
                    if not cols_to_transform:
                        raise ValueError("No columns selected for scaling.")
                    df_copy = df.copy()
                    df_copy[cols_to_transform] = df_copy[cols_to_transform].fillna(df_copy[cols_to_transform].mean())
                    
                    if scale_method == "log":
                        for col in cols_to_transform:
                            if (df_copy[col] <= 0).any():
                                df_copy[col] += abs(df_copy[col].min()) + 1
                            if log_base == "natural":
                                df_copy[col] = np.log(df_copy[col])
                            elif log_base == "10":
                                df_copy[col] = np.log10(df_copy[col])
                            elif log_base == "2":
                                df_copy[col] = np.log2(df_copy[col])
                    else:
                        if scale_method == "standard":
                            scaler = preprocessing.StandardScaler(with_mean=with_mean, with_std=with_std)
                        elif scale_method == "minmax":
                            scaler = preprocessing.MinMaxScaler(feature_range=(min_val, max_val))
                        elif scale_method == "robust":
                            scaler = preprocessing.RobustScaler(quantile_range=(q_low, q_high))
                        df_copy[cols_to_transform] = scaler.fit_transform(df_copy[cols_to_transform])
                    
                    st.session_state.df = df_copy.copy()
                    st.success(f"{scale_method.capitalize()} scaling applied successfully!")
                    st.dataframe(df_copy[cols_to_transform].head())
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with tab_adv:
            st.subheader("📊 Advanced Options")

            # Feature Selection for Clustering
            task_type = st.session_state.get("task_type", None)
            if task_type == "clustering":
                st.markdown("### 🔍 Feature Selection for Clustering")
                
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols:
                    selected_features = st.multiselect(
                        "Select features for clustering analysis",
                        numeric_cols,
                        default=numeric_cols
                    )
                    
                    # Variance threshold filter
                    var_threshold = st.slider(
                        "Minimum feature variance threshold",
                        0.0, 1.0, 0.01, 0.01,
                        help="Remove low-variance features"
                    )
                    
                    if st.button("Apply Feature Selection"):
                        selector = VarianceThreshold(threshold=var_threshold)
                        try:
                            selected_data = selector.fit_transform(df[selected_features])
                            kept_features = selector.get_support(indices=True)
                            st.success(f"Selected {len(kept_features)} features with variance > {var_threshold}")
                            st.session_state.df = df.iloc[:, kept_features].copy()
                        except Exception as e:
                            st.error(f"Feature selection failed: {str(e)}")

            # Outlier Removal
            st.markdown("### 🪓 Remove Outliers")
            outlier_method = st.selectbox("Select Outlier Detection Method", ["zscore", "isolationforest", "dbscan"])
            
            if outlier_method == "zscore":
                z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, step=0.1)
            elif outlier_method == "isolationforest":
                iso_contamination = st.slider("Contamination (Isolation Forest)", 0.01, 0.2, 0.05, step=0.01)
            elif outlier_method == "dbscan":
                dbscan_eps = st.slider("Epsilon (DBSCAN)", 1.0, 10.0, 3.0, step=0.1)
                dbscan_min_samples = st.slider("Min Samples (DBSCAN)", 2, 20, 5)
            
            if st.button("Remove Outliers"):
                df_copy = df.copy()
                numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns found for outlier removal.")
                else:
                    try:
                        if outlier_method == "zscore":
                            z_scores = np.abs(stats.zscore(df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())))
                            df_cleaned = df_copy[(z_scores < z_threshold).all(axis=1)]
                        elif outlier_method == "dbscan":
                            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                            df_numeric_filled = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
                            clusterer.fit(df_numeric_filled)
                            df_cleaned = df_copy[clusterer.labels_ != -1]
                        
                        removed_rows = len(df_copy) - len(df_cleaned)
                        st.success(f"✅ Removed {removed_rows} outliers.")
                        st.dataframe(df_cleaned.head())
                        st.session_state.df = df_cleaned.copy()
                        st.info("🔄 Data updated in session state.")
                    except Exception as e:
                        st.error(f"❌ Error during outlier removal: {e}")

            st.markdown("---")
            # Encoding Options
            st.markdown("### 🏷️ Encoding Options")
            obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not obj_cols:
                st.info("ℹ️ No categorical columns found for encoding.")
            else:
                encoding_type = st.selectbox("Select Encoding Type", ["Label Encoding", "One-Hot Encoding"])
                
                if encoding_type == "Label Encoding":
                    selected_cols = st.multiselect("Select Categorical Columns to Encode", obj_cols)
                    if st.button("Apply Label Encoding"):
                        if not selected_cols:
                            st.warning("⚠️ Please select at least one column.")
                        else:
                            df_encoded = df.copy()
                            le = LabelEncoder()
                            for col in selected_cols:
                                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                            st.session_state.df = df_encoded.copy()
                            st.success(f"✅ Encoded {len(selected_cols)} columns.")
                
                elif encoding_type == "One-Hot Encoding":
                    selected_cols = st.multiselect("Select Categorical Columns for One-Hot Encoding", obj_cols)
                    drop_first = st.checkbox("Drop First Column (Avoid Multicollinearity)", value=True)
                    if st.button("Apply One-Hot Encoding"):
                        if not selected_cols:
                            st.warning("⚠️ Please select at least one column.")
                        else:
                            df_encoded = pd.get_dummies(df, columns=selected_cols, drop_first=drop_first)
                            st.session_state.df = df_encoded.copy()
                            st.success(f"✅ Applied One-Hot Encoding on {len(selected_cols)} columns.")

            st.markdown("---")
            # Dimensionality Reduction
            st.markdown("### 📉 Apply PCA (Principal Component Analysis)")
            numeric_cols = df.select_dtypes(include=np.number).columns
            n_components = st.slider("Number of Principal Components", 1, min(len(numeric_cols), 10), 2)
            
            if st.button("Apply PCA"):
                if len(numeric_cols) < 2:
                    st.warning("⚠️ Need at least 2 numeric columns for PCA.")
                else:
                    try:
                        df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
                        pca = PCA(n_components=n_components)
                        principal_components = pca.fit_transform(df_numeric)
                        pc_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
                        
                        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                        if non_numeric_cols:
                            pc_df = pd.concat([pc_df, df[non_numeric_cols].reset_index(drop=True)], axis=1)
                        
                        st.success("✅ PCA Applied Successfully!")
                        st.dataframe(pc_df.head())
                        st.session_state.df = pc_df.copy()
                    except Exception as e:
                        st.error(f"❌ Error applying PCA: {str(e)}")

    elif selected_tab == "📊 Data Visualization":
        st.header("📊 Data Visualization")
        if "data" not in st.session_state or st.session_state.data.empty:
            st.warning("⚠️ Please upload data first to visualize.")
            st.stop()
        
        df = st.session_state.data.copy()
        viz_type = st.selectbox("Select Visualization Type", [
            "Histogram", "Box Plot", "Density Plot", "Bar Chart", "Pie Chart", "Violin Plot",
            "Scatter Plot", "Line Plot", "Heatmap", "Pair Plot", "3D Scatter"
        ])

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col_requirements = {
            "Histogram": 1,
            "Box Plot": 2,
            "Density Plot": 1,
            "Bar Chart": 1,
            "Pie Chart": 1,
            "Violin Plot": 2,
            "Scatter Plot": 2,
            "Line Plot": 2,
            "Heatmap": 0,
            "Pair Plot": 0,
            "3D Scatter": 3,
        }

        single_col_viz = ["Histogram", "Density Plot", "Bar Chart", "Pie Chart"]
        use_multi = True
        if viz_type not in single_col_viz:
            default_required = col_requirements.get(viz_type, 1)
            use_multi = st.checkbox(f"Use multi-column view ({default_required} columns)", value=True)
        else:
            use_multi = False

        selected_cols = []

        try:
            if viz_type == "Histogram":
                col1 = st.selectbox("Select column for histogram", numeric_cols)
                selected_cols = [col1]

            elif viz_type == "Box Plot":
                if use_multi:
                    cat_col = st.selectbox("Select Categorical Column", obj_cols + ["None"])
                    num_col = st.selectbox("Select Numerical Column", numeric_cols)
                    selected_cols = [cat_col, num_col] if cat_col != "None" else [num_col]
                else:
                    num_col = st.selectbox("Select Numerical Column", numeric_cols)
                    selected_cols = [num_col]

            elif viz_type == "Density Plot":
                col1 = st.selectbox("Select column for density plot", numeric_cols)
                selected_cols = [col1]

            elif viz_type == "Bar Chart":
                col1 = st.selectbox("Select categorical or discrete column", obj_cols + numeric_cols)
                selected_cols = [col1]

            elif viz_type == "Pie Chart":
                col1 = st.selectbox("Select categorical column", obj_cols)
                selected_cols = [col1]

            elif viz_type == "Violin Plot":
                if use_multi:
                    x_col = st.selectbox("Select Categorical Column", obj_cols)
                    y_col = st.selectbox("Select Numerical Column", numeric_cols)
                    selected_cols = [x_col, y_col]
                else:
                    y_col = st.selectbox("Select Numerical Column", numeric_cols)
                    selected_cols = [y_col]

            elif viz_type == "Scatter Plot":
                if use_multi:
                    x_col = st.selectbox("Select X Column", numeric_cols)
                    y_col = st.selectbox("Select Y Column", numeric_cols)
                    selected_cols = [x_col, y_col]
                else:
                    col = st.selectbox("Select single numeric column", numeric_cols)
                    selected_cols = [col]

            elif viz_type == "Line Plot":
                if use_multi:
                    x_col = st.selectbox("Select X Column", numeric_cols)
                    y_col = st.selectbox("Select Y Column", numeric_cols)
                    selected_cols = [x_col, y_col]
                else:
                    col = st.selectbox("Select single numeric column", numeric_cols)
                    selected_cols = [col]

            elif viz_type == "Heatmap":
                cols = st.multiselect("Select Numeric Columns for Heatmap", numeric_cols, default=numeric_cols[:5])
                selected_cols = cols

            elif viz_type == "Pair Plot":
                cols = st.multiselect("Select Numeric Columns for Pair Plot", numeric_cols, default=numeric_cols[:3])
                selected_cols = cols

            elif viz_type == "3D Scatter":
                if use_multi:
                    x_col = st.selectbox("Select X Column", numeric_cols)
                    y_col = st.selectbox("Select Y Column", numeric_cols)
                    z_col = st.selectbox("Select Z Column", numeric_cols)
                    selected_cols = [x_col, y_col, z_col]
                else:
                    col = st.selectbox("Select single numeric column", numeric_cols)
                    selected_cols = [col]

            # Visualization Rendering
            if len(selected_cols) == 0:
                st.warning("⚠️ No columns selected. Please choose at least one.")
            else:
                with st.spinner("Generating visualization..."):
                    if viz_type == "Histogram":
                        fig, ax = plt.subplots()
                        sns.histplot(df[selected_cols[0]], kde=True, ax=ax)
                        st.pyplot(fig)

                    elif viz_type == "Box Plot":
                        if len(selected_cols) == 2:
                            fig = px.box(df, x=selected_cols[0], y=selected_cols[1])
                        else:
                            fig = px.box(df, y=selected_cols[0])
                        st.plotly_chart(fig)

                    elif viz_type == "Density Plot":
                        fig, ax = plt.subplots()
                        sns.kdeplot(df[selected_cols[0]], fill=True, ax=ax)
                        ax.set_title(f"Density Plot of {selected_cols[0]}")
                        st.pyplot(fig)

                    elif viz_type == "Bar Chart":
                        series = df[selected_cols[0]].value_counts().reset_index()
                        series.columns = ['Category', 'Count']
                        fig = px.bar(series, x='Category', y='Count', title=f"Bar Chart of {selected_cols[0]}")
                        st.plotly_chart(fig)

                    elif viz_type == "Pie Chart":
                        counts = df[selected_cols[0]].value_counts()
                        fig = px.pie(values=counts.values, names=counts.index)
                        st.plotly_chart(fig)

                    elif viz_type == "Violin Plot":
                        if len(selected_cols) == 2:
                            fig = px.violin(df, x=selected_cols[0], y=selected_cols[1])
                        else:
                            fig = px.violin(df, y=selected_cols[0])
                        st.plotly_chart(fig)

                    elif viz_type == "Scatter Plot":
                        if len(selected_cols) == 2:
                            fig = px.scatter(df, x=selected_cols[0], y=selected_cols[1])
                        else:
                            fig = px.scatter(y=df[selected_cols[0]])
                        st.plotly_chart(fig)

                    elif viz_type == "Line Plot":
                        if len(selected_cols) == 2:
                            fig = px.line(df, x=selected_cols[0], y=selected_cols[1])
                        else:
                            fig = px.line(y=df[selected_cols[0]])
                        st.plotly_chart(fig)

                    elif viz_type == "Heatmap":
                        if len(selected_cols) >= 2:
                            corr = df[selected_cols].corr()
                            fig, ax = plt.subplots()
                            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ Select at least 2 numeric columns for heatmap.")

                    elif viz_type == "Pair Plot":
                        if len(selected_cols) >= 2:
                            fig = sns.pairplot(data=df[selected_cols])
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ Select at least 2 numeric columns for pair plot.")

                    elif viz_type == "3D Scatter":
                        if len(selected_cols) == 3:
                            fig = px.scatter_3d(df, x=selected_cols[0], y=selected_cols[1], z=selected_cols[2])
                            st.plotly_chart(fig)
                        else:
                            st.warning("⚠️ For 3D Scatter, please select exactly 3 numeric columns.")

        except Exception as e:
            st.error(f"⚠️ An error occurred while generating the plot: {str(e)}")

    elif selected_tab == "🤖 Model Setup & Train":
        st.header("🧠 Model Training")

        if "df" not in st.session_state or st.session_state.df.empty:
            st.warning("⚠️ Please upload and preprocess data first.")
            st.stop()

        df = st.session_state.df.copy()

        # Step 1: Select target column
        target_column = st.selectbox("🎯 Select the target column", df.columns)

        # Step 2: Detect task type with manual override
        task_type = detect_task_type(df, target_column)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📌 Detected task type: **{task_type.upper()}**")
        with col2:
            override_task = st.selectbox(
                "Override task type?",
                ["Auto-detected", "Force Classification", "Force Regression", "Force Clustering"],
                help="Use when automatic detection fails"
            )
            
            if override_task == "Force Classification":
                task_type = "classification"
            elif override_task == "Force Regression":
                task_type = "regression"
            elif override_task == "Force Clustering":
                task_type = "clustering"

        # Step 3: Select model with enhanced options
        models = get_models(task_type)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            model_name = st.selectbox(
                "🤖 Select Model",
                options=list(models.keys()),
                help="Choose the algorithm for your machine learning task"
            )
        
        with col2:
            if model_name in ["Logistic Regression", "Linear Regression"]:
                st.caption("🏆 Best for: Linear relationships")
            elif model_name in ["Random Forest", "Decision Tree"]:
                st.caption("🏆 Best for: Non-linear relationships, feature importance")
            elif model_name == "SVM" or model_name == "SVR":
                st.caption("🏆 Best for: High-dimensional data, clear margins")
            elif model_name == "KNN":
                st.caption("🏆 Best for: Small datasets, local patterns")
            elif model_name in ["KMeans", "DBSCAN", "Agglomerative", "Gaussian Mixture"]:
                st.caption("🏆 Best for: Unsupervised clustering, grouping similar data points")
        
        # Model-specific hyperparameter tuning
        st.markdown("### ⚙️ Model Configuration")
        
        # Initialize default model
        model = models[model_name]
        
        # Dynamic hyperparameter controls
        if model_name in ["Random Forest", "Random Forest Regressor"]:
            n_estimators = st.slider(
                "Number of trees", 
                10, 500, 100,
                help="More trees generally mean better performance but longer training time"
            )
            max_depth = st.slider(
                "Max tree depth", 
                1, 30, 5,
                help="Controls how deep each tree can grow. Deeper trees may overfit"
            )
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            ) if task_type == "classification" else RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        
        elif model_name in ["SVM", "SVR"]:
            kernel = st.selectbox(
                "Kernel type",
                ["linear"],
                help="Different kernels can capture different types of relationships"
            )
            C = st.slider(
                "Regularization (C)", 
                0.01, 10.0, 1.0, 0.01,
                help="Smaller values prevent overfitting but may underfit"
            )
            model = SVC(
                kernel=kernel,
                C=C,
                random_state=42
            ) if task_type == "classification" else SVR(
                kernel=kernel,
                C=C
            )
        
        elif model_name == "KNN":
            n_neighbors = st.slider(
                "Number of neighbors",
                1, 50, 5,
                help="Smaller values make the model more sensitive to noise"
            )
            weights = st.selectbox(
                "Weighting",
                ["uniform", "distance"],
                help="Weight points by distance or treat equally"
            )
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights
            )
        
        elif model_name == "Decision Tree" or model_name == "Decision Tree Regressor":
            max_depth = st.slider(
                "Max depth",
                1, 20, 5,
                help="Controls how deep the tree can grow"
            )
            min_samples_split = st.slider(
                "Minimum samples to split",
                2, 20, 2,
                help="Minimum number of samples required to split a node"
            )
            if task_type == "classification":
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
        
        elif model_name in ["KMeans", "DBSCAN", "Agglomerative", "Gaussian Mixture"]:
            if model_name == "KMeans":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif model_name == "DBSCAN":
                eps = st.slider("Epsilon (neighborhood radius)", 0.1, 5.0, 0.5, 0.1)
                min_samples = st.slider("Minimum samples in neighborhood", 1, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif model_name == "Agglomerative":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                linkage = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            elif model_name == "Gaussian Mixture":
                n_components = st.slider("Number of components", 1, 10, 3)
                model = GaussianMixture(n_components=n_components, random_state=42)

        # Show model summary before training
        st.markdown("### 🧠 Model Summary")
        st.json({
            "Model Type": model_name,
            "Task Type": task_type,
            "Parameters": model.get_params()
        })

        # Step 4: For supervised learning, select evaluation metric
        if task_type in ["classification", "regression"]:
            metrics = get_metrics(task_type)
            metric_name = st.selectbox("📏 Select a metric for evaluation", list(metrics.keys()))
            metric_func = metrics[metric_name]

            # Step 5: Train/test split ratio
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100

        # Step 6: Train the model or perform clustering
        if st.button("🚀 Train Selected Model" if task_type in ["classification", "regression"] else "🔍 Perform Clustering"):
            with st.spinner("Processing..."):
                try:
                    if task_type in ["classification", "regression"]:
                        # Prepare data
                        (X_train, X_test, y_train, y_test), feature_columns = prepare_data(df, target_column)
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test)
                        score = metric_func(y_test, y_pred)
                        
                        # Display results
                        if 0 <= score <= 1:
                            percent_score = f"{score * 100:.2f}%"
                        else:
                            percent_score = f"{score:.4f}"
                        
                        st.success(f"✅ {model_name} achieved **{metric_name}**: {percent_score}")
                        
                        # Additional metrics and visualizations
                        if task_type == "classification":
                            st.subheader("📊 Classification Report")
                            st.text(classification_report(y_test, y_pred))
                            
                            # Confusion matrix
                            st.subheader("🧮 Confusion Matrix")
                            fig, ax = plt.subplots()
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            
                            # Feature importance for tree-based models
                            if hasattr(model, 'feature_importances_'):
                                st.subheader("🔍 Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': feature_columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(importance_df.head(10), x='Importance', y='Feature', orientation='h')
                                st.plotly_chart(fig)
                        
                        elif task_type == "regression":
                            # Residual plot
                            st.subheader("📉 Residual Plot")
                            residuals = y_test - y_pred
                            fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'})
                            fig.add_hline(y=0, line_dash="dash")
                            st.plotly_chart(fig)
                            
                            # Actual vs Predicted
                            st.subheader("🔮 Actual vs Predicted")
                            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
                            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                            st.plotly_chart(fig)
                        
                        # Save model for later use
                        save_model(model, feature_columns)
                        st.info("💾 Model saved successfully for future predictions.")
                    
                    elif task_type == "clustering":
                        try:
                            numeric_df = df.select_dtypes(include=np.number).dropna()
                            cluster_cols = st.multiselect(
                                "Select features for clustering", 
                                numeric_df.columns.tolist(), 
                                default=numeric_df.columns.tolist()
                            )
                            
                            if len(cluster_cols) < 2:
                                st.warning("⚠️ Please select at least 2 numeric columns for clustering.")
                            else:
                                # Fit the model
                                if model_name == "Gaussian Mixture":
                                    model.fit(numeric_df[cluster_cols])
                                    labels = model.predict(numeric_df[cluster_cols])
                                else:
                                    model.fit(numeric_df[cluster_cols])
                                    labels = model.labels_
                                
                                df_with_labels = numeric_df.copy()
                                df_with_labels["Cluster"] = labels
                                st.success("✅ Clustering completed!")
                                
                                # Cluster metrics
                                try:
                                    silhouette = silhouette_score(numeric_df[cluster_cols], labels)
                                    calinski = calinski_harabasz_score(numeric_df[cluster_cols], labels)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        create_gauge(silhouette*100, "Silhouette Score", "blue")
                                    with col2:
                                        create_gauge(min(calinski/100, 100), "Calinski-Harabasz", "green")
                                        
                                    st.info(f"""
                                    - **Silhouette Score**: {silhouette:.2f} (Higher is better, range [-1, 1])
                                    - **Calinski-Harabasz**: {calinski:.2f} (Higher is better)
                                    """)
                                except Exception as e:
                                    st.warning(f"Could not compute cluster metrics: {str(e)}")
                                
                                # Cluster visualization
                                st.subheader("📊 Clustered Data Preview")
                                st.dataframe(df_with_labels.head())
                                
                                st.subheader("🗺️ Cluster Visualization")
                                viz_option = st.selectbox(
                                    "Visualization Method",
                                    ["PCA (2D)", "t-SNE (2D)", "UMAP (2D)", "3D PCA"]
                                )
                                
                                if viz_option == "PCA (2D)":
                                    pca = PCA(n_components=2)
                                    reduced = pca.fit_transform(df_with_labels[cluster_cols])
                                elif viz_option == "t-SNE (2D)":
                                    tsne = TSNE(n_components=2, perplexity=30)
                                    reduced = tsne.fit_transform(df_with_labels[cluster_cols])
                                elif viz_option == "UMAP (2D)":
                                    reducer = umap.UMAP()
                                    reduced = reducer.fit_transform(df_with_labels[cluster_cols])
                                elif viz_option == "3D PCA":  # 3D PCA
                                    pca = PCA(n_components=3)
                                    reduced = pca.fit_transform(df_with_labels[cluster_cols])
                                
                                # Plotting
                                if viz_option == "3D PCA":
                                    fig = px.scatter_3d(
                                        x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                                        color=labels, title="3D Cluster Visualization"
                                    )
                                else:
                                    fig = px.scatter(
                                        x=reduced[:, 0], y=reduced[:, 1],
                                        color=labels, title=f"{viz_option} Cluster Visualization"
                                    )
                                st.plotly_chart(fig)
                                plt.clf()
                                
                                # Save clustering results
                                st.session_state.df = df_with_labels.copy()
                                st.info("💾 Clustering results saved in session state.")
                        
                        except Exception as e:
                            st.error(f"❌ Error during clustering: {e}")
                
                except Exception as e:
                    st.error(f"❌ Error during model training/clustering: {str(e)}")

        # Prediction on new data (only for supervised learning)
        if task_type in ["classification", "regression"]:
            st.markdown("---")
            st.subheader("🔮 Make Predictions")
            new_data_file = st.file_uploader("Upload new data for prediction (without target column)", type=["csv", "xlsx"])
            
            if new_data_file:
                try:
                    new_df = load_data(new_data_file)
                    if new_df is not None:
                        loaded_model, feature_columns = load_model()
                        
                        # Check if all required features are present
                        missing_features = set(feature_columns) - set(new_df.columns)
                        if missing_features:
                            st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                        else:
                            predictions = predict_new_data(loaded_model, feature_columns, new_df)
                            
                            # Display predictions
                            st.subheader("📊 Predictions")
                            pred_df = pd.DataFrame(predictions, columns=['Prediction'])
                            
                            if task_type == "classification":
                                if hasattr(loaded_model, 'predict_proba'):
                                    proba = loaded_model.predict_proba(new_df[feature_columns])
                                    proba_df = pd.DataFrame(proba, columns=[f"Prob_{c}" for c in loaded_model.classes_])
                                    pred_df = pd.concat([pred_df, proba_df], axis=1)
                            
                            st.dataframe(pred_df)
                            
                            # Download predictions
                            csv = pred_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download Predictions",
                                data=csv,
                                file_name='predictions.csv',
                                mime='text/csv'
                            )
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")

    elif selected_tab == "📁 Report":
        st.header("📑 Automated Data Report")
        
        if "df" not in st.session_state or st.session_state.df.empty:
            st.warning("⚠️ Please upload and preprocess data first.")
            st.stop()
        
        df = st.session_state.df.copy()
        st.success(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        
        # Basic Statistics
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe())
        
        # Missing Values Summary
        st.subheader("❓ Missing Values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("No missing values found!")
        else:
            missing = missing[missing > 0]
            st.warning(f"Found {missing.sum()} missing values across {len(missing)} columns.")
            fig = px.bar(missing, title="Missing Values by Column")
            st.plotly_chart(fig)
        
        # Correlation Analysis
        st.subheader("🔄 Correlation Analysis")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto")
            st.plotly_chart(fig)
        else:
            st.info("Not enough numeric columns for correlation analysis.")