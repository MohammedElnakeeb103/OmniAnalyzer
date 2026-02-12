# from ydata_profiling import ProfileReport # type: ignore
# from streamlit_pandas_profiling import st_profile_report # type: ignore
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
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.utils import resample
from sklearn.metrics import classification_report
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from nora import detect_task_type, evaluate_model, get_metrics, get_models, load_data, load_model, predict_new_data, prepare_data, save_model, train_single_model


# Page config
st.set_page_config(page_title="Data Modeling App", layout="wide")
st.title("📊 Data Explorer, Preprocessing & ML Trainer")

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

# Handle uploaded file
if uploaded_file:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        df_raw = load_data(uploaded_file)
        st.session_state.data = df_raw.copy()
        st.session_state.df = df_raw.copy()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success("✅ New data loaded successfully!")

# Use session_state data
if not st.session_state.df.empty:
    df = st.session_state.df.copy()

    # Raw Data Preview
    st.subheader("📋 Raw Data Preview")
    num_rows = st.slider("Choose the number of rows to display:", min_value=1, max_value=len(df), value=5)
    st.dataframe(df.head(num_rows))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 Data Preprocessing",
        "📊 Data Visualization",
        "🤖 Model Setup & Train",
        "📑 Report"
    ])

    with tab1:
        st.header("Data Preprocessing")
        tab_clean, tab_impute, tab_scale, tab_adv = st.tabs(["🧼 Cleaning", "🧩 Imputation", "📐 Scaling", "📊 Advanced"])
        with tab_clean:
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
                    st.session_state.df = df.copy()  # Update the session state
                    if removed == 0:
                        st.info("No duplicates were removed.")
                    else:
                        st.success(f"Removed {removed} duplicate rows.")

        with tab_impute:
            st.subheader("Imputation")
            method = st.selectbox("Select Imputation Method", ["Simple", "KNN", "Iterative"])
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
                    df_copy = df.copy()  # Work with a copy
                    if method == "Simple":
                        if selected_num_cols:
                            num_imputer = SimpleImputer(strategy=strategy)
                            df_copy[selected_num_cols] = num_imputer.fit_transform(df_copy[selected_num_cols])
                        if selected_cat_cols:
                            cat_imputer = SimpleImputer(strategy="most_frequent")
                            df_copy[selected_cat_cols] = cat_imputer.fit_transform(df_copy[selected_cat_cols])
                        st.session_state.df = df_copy.copy()  # Update session state
                        st.success("Imputation applied successfully.")
                    elif method == "KNN":
                        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
                        df_copy[selected_num_cols] = knn_imputer.fit_transform(df_copy[selected_num_cols])
                        st.session_state.df = df_copy.copy()  # Update session state
                        st.success("KNN Imputation applied.")
                    elif method == "Iterative":
                        iterative_imputer = IterativeImputer()
                        df_copy[selected_num_cols] = iterative_imputer.fit_transform(df_copy[selected_num_cols])
                        st.session_state.df = df_copy.copy()  # Update session state
                        st.success("Iterative Imputation applied.")
                except Exception as e:
                    st.error(f"Error during imputation: {e}")

        with tab_scale:
            st.subheader("Scaling & Winsorizing")
            col_selection = st.radio("Select columns to transform:", ["All Numeric", "Specific Columns"])
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            cols_to_transform = st.multiselect("Select columns:", options=numeric_cols, default=numeric_cols[:1]) if col_selection == "Specific Columns" else numeric_cols
            scale_method = st.selectbox("Select Scaling Method", ["standard", "minmax", "maxabs", "power", "robust", "log"])
            if scale_method == "standard":
                with_mean = st.checkbox("Center data", True)
                with_std = st.checkbox("Scale to unit variance", True)
            elif scale_method == "minmax":
                min_val = st.number_input("Min value", -10.0, 10.0, 0.0, 0.1)
                max_val = st.number_input("Max value", -10.0, 10.0, 1.0, 0.1)
            elif scale_method == "power":
                power_method = st.selectbox("Power method", ["yeo-johnson", "box-cox"])
            elif scale_method == "robust":
                q_low = st.slider("Lower quantile", 0, 49, 25)
                q_high = st.slider("Upper quantile", 51, 100, 75)
            elif scale_method == "log":
                log_base = st.selectbox("Log base", ["natural", "10", "2"])
            if st.button("Apply Scaling"):
                try:
                    if not cols_to_transform:
                        raise ValueError("No columns selected for scaling.")
                    df_copy = df.copy()  # Work with a copy
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
                        elif scale_method == "maxabs":
                            scaler = preprocessing.MaxAbsScaler()
                        elif scale_method == "power":
                            scaler = preprocessing.PowerTransformer(method=power_method)
                        elif scale_method == "robust":
                            scaler = preprocessing.RobustScaler(quantile_range=(q_low, q_high))
                        df_copy[cols_to_transform] = scaler.fit_transform(df_copy[cols_to_transform])
                    st.session_state.df = df_copy.copy()  # Update session state
                    st.success(f"{scale_method.capitalize()} scaling applied successfully!")
                    st.dataframe(df_copy[cols_to_transform].head())
                except Exception as e:
                    st.error(f"Error: {str(e)}")

            st.subheader("Winsorizing")
            low_limit = st.slider("Lower limit (%)", 0.0, 20.0, 5.0, 0.5) / 100
            high_limit = st.slider("Upper limit (%)", 0.0, 20.0, 5.0, 0.5) / 100
            if st.button("Apply Winsorization"):
                try:
                    df_copy = df.copy()  # Work with a copy
                    for col in cols_to_transform:
                        na_mask = df_copy[col].isna()
                        non_na_values = df_copy.loc[~na_mask, col].values
                        if len(non_na_values) > 0:
                            winsorized_values = stats.mstats.winsorize(non_na_values, limits=(low_limit, high_limit))
                            df_copy.loc[~na_mask, col] = winsorized_values
                    st.session_state.df = df_copy.copy()  # Update session state
                    st.success("Winsorization applied successfully!")
                    st.dataframe(df_copy[cols_to_transform].head())
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with tab_adv:
            st.subheader("📊 Advanced Options")

            # --- 1. Outlier Removal ---
            st.markdown("### 🪓 Remove Outliers")
            outlier_method = st.selectbox(
                "Select Outlier Detection Method",
                ["zscore", "isolationforest", "dbscan"],
                key="adv_outlier_method"
            )
            if outlier_method == "zscore":
                z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, step=0.1)
            elif outlier_method == "isolationforest":
                iso_contamination = st.slider("Contamination (Isolation Forest)", 0.01, 0.2, 0.05, step=0.01)
            elif outlier_method == "dbscan":
                dbscan_eps = st.slider("Epsilon (DBSCAN)", 1.0, 10.0, 3.0, step=0.1)
                dbscan_min_samples = st.slider("Min Samples (DBSCAN)", 2, 20, 5)
            if st.button("Remove Outliers", key="adv_remove_outliers"):
                df_copy = df.copy()  # Work with a copy
                numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns found for outlier removal.")
                else:
                    try:
                        if outlier_method == "zscore":
                            z_scores = np.abs(stats.zscore(df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())))
                            df_cleaned = df_copy[(z_scores < z_threshold).all(axis=1)]
                        elif outlier_method == "isolationforest":
                            iso_forest = IsolationForest(contamination=iso_contamination, random_state=42)
                            df_numeric_filled = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
                            preds = iso_forest.fit_predict(df_numeric_filled)
                            df_cleaned = df_copy[preds != -1]
                        elif outlier_method == "dbscan":
                            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                            df_numeric_filled = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
                            clusterer.fit(df_numeric_filled)
                            df_cleaned = df_copy[clusterer.labels_ != -1]
                        removed_rows = len(df_copy) - len(df_cleaned)
                        st.success(f"✅ Removed {removed_rows} outliers.")
                        st.dataframe(df_cleaned.head())
                        # Update session state
                        st.session_state.df = df_cleaned.copy()
                        st.info("🔄 Data updated in session state.")
                    except Exception as e:
                        st.error(f"❌ Error during outlier removal: {e}")

            st.markdown("---")
            # --- 2. Encode Labels (Label Encoding) ---
            st.markdown("### 🏷️ Label Encoding")
            obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not obj_cols:
                st.info("ℹ️ No categorical columns found for encoding.")
            else:
                selected_obj_cols = st.multiselect(
                    "Select Categorical Columns to Encode",
                    obj_cols,
                    default=obj_cols[:1] if obj_cols else [],
                    key="adv_encode_cols"
                )
                if st.button("Encode Selected Columns", key="adv_encode_button"):
                    if not selected_obj_cols:
                        st.warning("⚠️ Please select at least one column.")
                    else:
                        df_encoded = df.copy()  # Work with a copy
                        le = LabelEncoder()
                        for col in selected_obj_cols:
                            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                        st.success(f"✅ Encoded {len(selected_obj_cols)} categorical column(s).")
                        st.dataframe(df_encoded[selected_obj_cols].head())
                        # Update session state
                        st.session_state.df = df_encoded.copy()
                        st.info("🔄 Data updated in session state.")

            st.markdown("---")
            # --- 3. One-Hot Encoding ---
            st.markdown("### 🔢 One-Hot Encoding")
            selected_onehot_cols = st.multiselect(
                "Select Categorical Columns for One-Hot Encoding",
                obj_cols,
                key="onehot_encode_cols"
            )
            drop_first = st.checkbox("Drop First Column (Avoid Multicollinearity)", value=True)
            if st.button("Apply One-Hot Encoding", key="adv_onehot_button"):
                if not selected_onehot_cols:
                    st.warning("⚠️ Please select at least one column.")
                else:
                    df_encoded = pd.get_dummies(df, columns=selected_onehot_cols, drop_first=drop_first)
                    st.success(f"✅ Applied One-Hot Encoding on {len(selected_onehot_cols)} columns.")
                    st.dataframe(df_encoded.head())
                    # Update session state
                    st.session_state.df = df_encoded.copy()
                    st.info("🔄 Data updated in session state.")

            st.markdown("---")
            # --- 4. Ordinal Encoding ---
            st.markdown("### 📊 Ordinal Encoding")
            selected_ordinal_cols = st.multiselect(
                "Select Ordinal Columns",
                obj_cols,
                key="ordinal_encode_cols"
            )
            ordinal_mappings = {}
            if selected_ordinal_cols:
                for col in selected_ordinal_cols:
                    unique_vals = sorted(df[col].dropna().unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    ordinal_mappings[col] = mapping
                    st.write(f"🔢 Mapping for `{col}`:", mapping)
            if st.button("Apply Ordinal Encoding", key="adv_ordinal_button"):
                if not selected_ordinal_cols:
                    st.warning("⚠️ Please select at least one column.")
                else:
                    df_encoded = df.copy()
                    for col in selected_ordinal_cols:
                        df_encoded[col] = df_encoded[col].map(ordinal_mappings[col])
                    st.success(f"✅ Applied Ordinal Encoding on {len(selected_ordinal_cols)} columns.")
                    st.dataframe(df_encoded[selected_ordinal_cols].head())
                    # Update session state
                    st.session_state.df = df_encoded.copy()
                    st.info("🔄 Data updated in session state.")

            st.markdown("---")
            # --- 5. Frequency Encoding ---
            st.markdown("### 📈 Frequency Encoding")
            selected_freq_cols = st.multiselect(
                "Select Columns for Frequency Encoding",
                obj_cols,
                key="freq_encode_cols"
            )
            if st.button("Apply Frequency Encoding", key="adv_freq_button"):
                if not selected_freq_cols:
                    st.warning("⚠️ Please select at least one column.")
                else:
                    df_encoded = df.copy()  # Work with a copy
                    for col in selected_freq_cols:
                        freq_map = df_encoded[col].value_counts(normalize=True)
                        df_encoded[f"{col}_freq"] = df_encoded[col].map(freq_map)
                    st.success(f"✅ Applied Frequency Encoding on {len(selected_freq_cols)} columns.")
                    st.dataframe(df_encoded[[f"{col}_freq" for col in selected_freq_cols]].head())
                    # Update session state
                    st.session_state.df = df_encoded.copy()
                    st.info("🔄 Data updated in session state.")

            st.markdown("---")
            # --- 6. Target Encoding ---
            st.markdown("### 🎯 Target Encoding")
            target_col = st.selectbox("Select Target Column", df.columns, key="target_encode_target")
            selected_target_cols = st.multiselect(
                "Select Categorical Columns for Target Encoding",
                obj_cols,
                key="target_encode_cols"
            )
            if st.button("Apply Target Encoding", key="adv_target_button"):
                if not selected_target_cols or not target_col:
                    st.warning("⚠️ Please select a target and at least one column.")
                else:
                    df_encoded = df.copy()  # Work with a copy
                    for col in selected_target_cols:
                        mapping = df_encoded.groupby(col)[target_col].mean()
                        df_encoded[f"{col}_target"] = df_encoded[col].map(mapping)
                    st.success(f"✅ Applied Target Encoding on {len(selected_target_cols)} columns.")
                    st.dataframe(df_encoded[[f"{col}_target" for col in selected_target_cols]].head())
                    # Update session state
                    st.session_state.df = df_encoded.copy()
                    st.info("🔄 Data updated in session state.")

            st.markdown("---")
            # --- 7. Apply SMOTE for Class Balancing ---
            st.markdown("### 🔁 Apply SMOTE (Class Balancing)")
            target_col = st.selectbox("Select Target Column for SMOTE", df.columns, key="adv_smote_target")
            if st.button("Apply SMOTE", key="adv_smote_button"):
                try:
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    # Only apply SMOTE on classification tasks
                    if y.dtype == 'object' or y.nunique() <= 10:
                        # Convert categorical columns to numeric
                        X_encoded = X.copy()
                        cat_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
                        for cat_col in cat_cols:
                            le = LabelEncoder()
                            X_encoded[cat_col] = le.fit_transform(X_encoded[cat_col].astype(str))
                        # Apply SMOTE
                        smote = SMOTE(random_state=42)
                        X_res, y_res = smote.fit_resample(X_encoded, y)
                        # Reconstruct the dataframe
                        df_balanced = pd.DataFrame(X_res, columns=X_encoded.columns)
                        df_balanced[target_col] = y_res
                        st.success("✅ SMOTE applied successfully!")
                        st.write("🔢 Class Distribution Before:", df[target_col].value_counts())
                        st.write("🔢 Class Distribution After:", y_res.value_counts())
                        st.dataframe(df_balanced[[target_col]].value_counts().to_frame())
                        # Update session state
                        st.session_state.df = df_balanced.copy()
                        st.info("🔄 Data updated in session state.")
                    else:
                        st.warning("⚠️ SMOTE is typically used for classification tasks with categorical targets.")
                except Exception as e:
                    st.error(f"❌ Error applying SMOTE: {str(e)}")

            st.markdown("---")
            # --- 8. Apply PCA for Dimensionality Reduction ---
            st.markdown("### 📉 Apply PCA (Principal Component Analysis)")
            numeric_cols = df.select_dtypes(include=np.number).columns
            n_components = st.slider("Number of Principal Components", 1, min(len(numeric_cols), 10), 2)
            if st.button("Apply PCA", key="adv_pca_button"):
                if len(numeric_cols) < 2:
                    st.warning("⚠️ Need at least 2 numeric columns for PCA.")
                else:
                    try:
                        # Fill missing values before PCA
                        df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
                        pca = PCA(n_components=n_components)
                        principal_components = pca.fit_transform(df_numeric)
                        pc_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
                        # Add any non-numeric columns if needed
                        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                        if non_numeric_cols:
                            pc_df = pd.concat([pc_df, df[non_numeric_cols].reset_index(drop=True)], axis=1)
                        st.success("✅ PCA Applied Successfully!")
                        st.dataframe(pc_df.head())
                        # Update session state
                        st.session_state.df = pc_df.copy()
                        st.info("🔄 Data updated in session state.")
                    except Exception as e:
                        st.error(f"❌ Error applying PCA: {str(e)}")

    with tab2:
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

            # --- Visualization Rendering ---
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
                        series.columns = ['Category', 'Count']  # Avoid name clash
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

    with tab3:
        st.header("🧠 Model Training")

        # Step 1: التأكد من وجود بيانات
        if "df" not in st.session_state or st.session_state.df.empty:
            st.warning("⚠️ Please upload and preprocess data first.")
            st.stop()

        df = st.session_state.df.copy()  # 👈 هذه هي البيانات بعد التعديل!

        # Step 2: اختيار العمود المستهدف
        target_column = st.selectbox("🎯 Select the target column", df.columns)

        # Step 3: اكتشاف نوع المهمة (تصنيف أو انحدار)
        task_type = detect_task_type(df, target_column)
        st.info(f"📌 Detected task type: **{task_type.upper()}**")

        # Step 4: اختيار النموذج
        models = get_models(task_type)
        model_name = st.selectbox("🤖 Select a model", list(models.keys()))
        model = models[model_name]

        # Step 5: اختيار المعيار للتقييم
        metrics = get_metrics(task_type)
        metric_name = st.selectbox("📏 Select a metric for evaluation", list(metrics.keys()))
        metric_func = metrics[metric_name]

        # Step 6: تدريب النموذج باستخدام البيانات المعدلة
        if st.button("🚀 Train Selected Model"):
            try:
                # تحضير البيانات باستخدام النسخة المعدلة من df
                (X_train, X_test, y_train, y_test), feature_columns = prepare_data(df, target_column)

                # تدريب النموذج
                trained_model = train_single_model(model, X_train, y_train)

                # التقييم
                score = evaluate_model(trained_model, X_test, y_test, metric_func)

                # عرض النتيجة كنسبة مئوية إن أمكن
                if 0 <= score <= 1:
                    percent_score = f"{score * 100:.2f}%"
                else:
                    percent_score = f"{score:.4f}"

                # رسالة التقييم + تفسير
                if task_type == "classification":
                    st.success(f"✅ {model_name} achieved a **{metric_name}**: `{percent_score}`")
                    if metric_name == "accuracy":
                        st.info(f"📘 Accuracy means the model correctly predicted {percent_score} of the test data.")
                    elif metric_name == "precision":
                        st.info(f"📘 Precision means when the model predicts positive, it is correct {percent_score}.")
                    elif metric_name == "recall":
                        st.info(f"📘 Recall means the model detected {percent_score} of all actual positives.")
                    elif metric_name == "f1_score":
                        st.info(f"📘 F1 Score is the harmonic mean of precision and recall: `{percent_score}`.")
                elif task_type == "regression":
                    st.success(f"✅ {model_name} achieved a **{metric_name}**: `{percent_score}`")
                    st.info(f"📘 This score indicates how well the model predicts numerical values. Lower is usually better for regression metrics like MSE or MAE.")

                # حفظ النموذج للتنبؤ لاحقًا
                save_model(trained_model, feature_columns)
                st.info("💾 Model saved successfully and ready for prediction.")

            except Exception as e:
                st.error(f"❌ An error occurred during training: {e}")

        # Step 7: التنبؤ بدون رفع ملف جديد
        st.markdown("---")
        st.subheader("📡 Predict using saved model")
        new_data_file = st.file_uploader("Upload new data for prediction (without target column)", type=["csv"], key="predict")

        if new_data_file:
            new_df = pd.read_csv(new_data_file)
            loaded_model, feature_columns = load_model()
            predictions = predict_new_data(loaded_model, feature_columns, new_df)
            st.write("### 🔮 Predictions", pd.DataFrame(predictions, columns=["Prediction"]))
# --- Download Processed Data Button ---
if not st.session_state.df.empty:
    csv = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="⬇️ Download Processed Data",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv"
    )
    with tab4:
        st.header("📑 Automated Data Report")

        if "df" not in st.session_state or st.session_state.df.empty:
            st.warning("⚠️ Please upload and preprocess data first.")
            st.stop()

        df = st.session_state.df.copy()
        st.success(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

        # --- Generate Profile Report ---
        st.markdown("### 🧾 Full Data Profiling Report")
        st.info("This includes missing values, distributions, correlations, and more.", icon="📊")

        try:
            # Generate the report using ydata-profiling
            profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
            
            # Display the report in Streamlit
            st_profile_report(profile)

            # Optional: Allow user to download the report as HTML
            report_html = profile.to_html()
            st.download_button(
                label="📄 Download Full Report (HTML)",
                data=report_html,
                file_name="data_profiling_report.html",
                mime="text/html"
            )

        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")