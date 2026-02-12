import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    silhouette_score, calinski_harabasz_score,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import joblib
import tempfile
import os

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = ""
if 'task_type' not in st.session_state:
    st.session_state.task_type = ""
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'labels' not in st.session_state:
    st.session_state.labels = None

# Helper functions
def detect_task_type(df, target_column):
    if target_column == "None" or target_column not in df.columns:
        return "clustering"
    
    unique_values = df[target_column].nunique()
    if unique_values <= 10 or df[target_column].dtype == 'object':
        return "classification"
    else:
        return "regression"

def get_models(task_type):
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
    elif task_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }
    else:  # clustering
        return {
            "KMeans": KMeans(),
            "DBSCAN": DBSCAN(),
            "Agglomerative": AgglomerativeClustering(),
            "Gaussian Mixture": GaussianMixture()
        }

def get_metrics(task_type):
    if task_type == "classification":
        return {
            "Accuracy": accuracy_score,
            "Precision": precision_score,
            "Recall": recall_score,
            "F1 Score": f1_score
        }
    elif task_type == "regression":
        return {
            "MSE": mean_squared_error,
            "MAE": mean_absolute_error,
            "R2 Score": r2_score
        }
    else:
        return {}

def prepare_data(df, target_column):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical target if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        st.session_state.label_encoder = le
    
    # Get feature columns before preprocessing
    feature_columns = X.columns.tolist()
    
    # Scale numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    st.session_state.scaler = scaler
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return (X_train, X_test, y_train, y_test), feature_columns

def save_model(model, feature_columns):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        joblib.dump(model, tmp_file.name)
        st.session_state.model_path = tmp_file.name
    st.session_state.model = model
    st.session_state.feature_columns = feature_columns

def load_model():
    if 'model_path' in st.session_state and os.path.exists(st.session_state.model_path):
        model = joblib.load(st.session_state.model_path)
        return model, st.session_state.feature_columns
    return None, []

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file)
    return None

def predict_new_data(model, feature_columns, new_df):
    # Preprocess new data same as training data
    if st.session_state.scaler:
        new_df = pd.DataFrame(
            st.session_state.scaler.transform(new_df[feature_columns]),
            columns=feature_columns
        )
    return model.predict(new_df)

def create_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': '#ff4444'},
                {'range': [40, 70], 'color': '#ffaa00'},
                {'range': [70, 100], 'color': '#44ff44'}
            ],
        }
    ))
    fig.update_layout(height=200)
    return fig

# Caching the clustering function
@st.cache
def perform_clustering(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(scaled_data)
    return kmeans.labels_

# Main app function
def model_training_tab():
    st.header("🧠 Model Training")

    if "df" not in st.session_state or st.session_state.df.empty:
        st.warning("⚠️ Please upload and preprocess data first.")
        return

    df = st.session_state.df.copy()

    # Step 1: Select target column
    target_column = st.selectbox("🎯 Select the target column", ["None"] + df.columns.tolist())
    st.session_state.target_column = target_column

    # Step 2: Detect task type with manual override
    task_type = detect_task_type(df, target_column)
    st.session_state.task_type = task_type
    
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
        st.session_state.task_type = task_type

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
        elif model_name in ["Random Forest", "Random Forest Regressor", "Decision Tree", "Decision Tree Regressor"]:
            st.caption("🏆 Best for: Non-linear relationships, feature importance")
        elif model_name in ["SVM", "SVR"]:
            st.caption("🏆 Best for: High-dimensional data, clear margins")
        elif model_name in ["KNN", "KNN Regressor"]:
            st.caption("🏆 Best for: Small datasets, local patterns")
        elif model_name in ["KMeans", "DBSCAN", "Agglomerative", "Gaussian Mixture"]:
            st.caption("🏆 Best for: Unsupervised clustering, grouping similar data points")
    
    # Model-specific hyperparameter tuning
    st.markdown("### ⚙️ Model Configuration")
    
    # Initialize default model
    model = models[model_name]
    
    # Dynamic hyperparameter controls
    if model_name in ["Random Forest", "Random Forest Regressor"]:
        n_estimators = st.slider("Number of trees", 10, 500, 100)
        max_depth = st.slider("Max tree depth", 1, 30, 5)
        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
    
    elif model_name in ["SVM", "SVR"]:
        kernel = st.selectbox("Kernel type", ["linear", "rbf", "poly"], index=0)
        C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
        if task_type == "classification":
            model = SVC(
                kernel=kernel,
                C=C,
                probability=True,
                random_state=42
            )
        else:
            model = SVR(
                kernel=kernel,
                C=C
            )
    
    elif model_name in ["KNN", "KNN Regressor"]:
        n_neighbors = st.slider("Number of neighbors", 1, 50, 5)
        weights = st.selectbox("Weighting", ["uniform", "distance"])
        if task_type == "classification":
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights
            )
        else:
            model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights
            )
    
    elif model_name in ["Decision Tree", "Decision Tree Regressor"]:
        max_depth = st.slider("Max depth", 1, 20, 5)
        min_samples_split = st.slider("Minimum samples to split", 2, 20, 2)
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
                            # Perform clustering and cache results
                            n_clusters = st.slider("Number of clusters", 2, 10, 3)
                            st.session_state.labels = perform_clustering(numeric_df[cluster_cols], n_clusters)
                            
                            df_with_labels = numeric_df.copy()
                            df_with_labels["Cluster"] = st.session_state.labels
                            st.success("✅ Clustering completed!")
                            
                            # Cluster metrics
                            try:
                                silhouette = silhouette_score(numeric_df[cluster_cols], st.session_state.labels)
                                calinski = calinski_harabasz_score(numeric_df[cluster_cols], st.session_state.labels)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.plotly_chart(create_gauge(silhouette*100, "Silhouette Score", "blue"))
                                with col2:
                                    st.plotly_chart(create_gauge(min(calinski/100, 100), "Calinski-Harabasz", "green"))
                                    
                                st.info(f"""
                                - **Silhouette Score**: {silhouette:.2f} (Higher is better, range [-1, 1])
                                - **Calinski-Harabasz**: {calinski:.2f} (Higher is better)
                                """)
                            except Exception as e:
                                st.warning(f"Could not compute cluster metrics: {str(e)}")
                            
                            # Cluster visualization
                            st.subheader("📊 Clustered Data Preview")
                            st.dataframe(df_with_labels.head())
                            
                            # Visualization method selection
                            viz_option = st.selectbox(
                                "Visualization Method",
                                ["PCA (2D)", "t-SNE (2D)", "UMAP (2D)", "3D PCA"]
                            )
                            
                            # Perform visualization based on selected method
                            if viz_option == "PCA (2D)":
                                pca = PCA(n_components=2)
                                reduced = pca.fit_transform(numeric_df[cluster_cols])
                                fig = px.scatter(
                                    x=reduced[:, 0], y=reduced[:, 1],
                                    color=st.session_state.labels, title="PCA Cluster Visualization"
                                )
                                st.plotly_chart(fig)
                            
                            elif viz_option == "t-SNE (2D)":
                                tsne = TSNE(n_components=2, random_state=42)
                                reduced = tsne.fit_transform(numeric_df[cluster_cols])
                                fig = px.scatter(
                                    x=reduced[:, 0], y=reduced[:, 1],
                                    color=st.session_state.labels, title="t-SNE Cluster Visualization"
                                )
                                st.plotly_chart(fig)

                            elif viz_option == "UMAP (2D)":
                                reducer = umap.UMAP(n_components=2, random_state=42)
                                reduced = reducer.fit_transform(numeric_df[cluster_cols])
                                fig = px.scatter(
                                    x=reduced[:, 0], y=reduced[:, 1],
                                    color=st.session_state.labels, title="UMAP Cluster Visualization"
                                )
                                st.plotly_chart(fig)

                            elif viz_option == "3D PCA":
                                pca = PCA(n_components=3)
                                reduced = pca.fit_transform(numeric_df[cluster_cols])
                                fig = px.scatter_3d(
                                    x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
                                    color=st.session_state.labels, title="3D PCA Cluster Visualization"
                                )
                                st.plotly_chart(fig)

                            st.success("✅ Clustering and visualization completed!")
                    except Exception as e:
                        st.error(f"❌ Error during clustering: {str(e)}")

            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")