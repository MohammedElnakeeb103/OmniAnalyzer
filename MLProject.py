from scipy import stats
from sklearn import preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
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
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import umap.umap_ as umap
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import time
import psutil
import os

st.set_page_config(page_title="OmniAnalyzer", layout="wide", page_icon="📊")
st.title("📊 OmniAnalyzer: From Raw Data to AI Models in One Click")

st.markdown("""
<style>
    /* Main colors */
    :root {
        --primary: #008080;
        --secondary: #2E5A88;
        --accent: #FFD700;
        --light-bg: #F5F5F5;
        --dark-text: #2E3440;
    }
    
    /* Main container */
    .stApp {
        background-color: #080808;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #080808 !important;
        border-right: 1px solid #000000;
    }
    
    /* Tabs */
    .stTabs [aria-selected="true"] {
        color: #f8f8ff !important;
        font-weight: 600;
        border-bottom: 2px solid #f8f8ff;
    }
    .stTabs [aria-selected="false"] {
        color: #2E5A88;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #008080;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #006666;
        color: white;
    }
    .stDownloadButton>button {
        background-color: var(--secondary);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricLabel"] {
        color: var(--secondary);
    }
    [data-testid="stMetricValue"] {
        color: var(--primary);
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Input widgets */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stMultiselect>div>div>div,
    .stSlider>div>div>div>div {
        border-radius: 6px !important;
        border: 1px solid #E0E0E0 !important;
    }
    
    /* Success messages */
    .stAlert [data-testid="stMarkdownContainer"] {
        color: #06d639;
    }
    
    /* Warning messages */
    .stAlert [data-testid="stMarkdownContainer"] {
        color: #ffd166;
    }
    
    /* Error messages */
    .stAlert [data-testid="stMarkdownContainer"] {
        color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your data: ", type=["csv", "xlsx", "xls"])

# Add data info in sidebar
# if 'df' in st.session_state and not st.session_state.df.empty:
#     st.sidebar.markdown("---")
#     st.sidebar.subheader("📋 Data Info")
#     st.sidebar.write(f"**Rows:** {len(st.session_state.df):,}")
#     st.sidebar.write(f"**Columns:** {len(st.session_state.df.columns):,}")
#     st.sidebar.write(f"**Memory:** {st.session_state.df.memory_usage().sum() / 1024**2:.2f} MB")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Large File Options")
chunk_processing = st.sidebar.checkbox("Enable chunk processing", value=True, 
                                      help="Process large files in chunks to reduce memory usage")
chunk_size = st.sidebar.selectbox("Chunk size (rows)", 
                                 [10000, 50000, 100000, 200000], 
                                 index=1,
                                 help="Number of rows to process at a time")
sampling_rate = st.sidebar.slider("Sampling rate (%)", 1, 100, 100,
                                help="Percentage of data to use for analysis")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Return in MB

# Function to process large CSV files in chunks
def process_large_csv(file, chunk_size, sampling_rate):
    chunks = []
    total_rows = 0
    processed_rows = 0
    
    # First pass to count rows (for progress bar)
    with st.spinner("Counting rows in file..."):
        for chunk in pd.read_csv(file, chunksize=10000):
            total_rows += len(chunk)
        file.seek(0)  # Reset file pointer
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for chunk in pd.read_csv(file, chunksize=chunk_size):
        # Apply sampling
        if sampling_rate < 100:
            chunk = chunk.sample(frac=sampling_rate/100)
        
        chunks.append(chunk)
        processed_rows += len(chunk)
        
        # Update progress
        progress = min(processed_rows / total_rows, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processed {processed_rows:,} of ~{total_rows:,} rows")
        
        # Update memory usage
        st.session_state.memory_usage = get_memory_usage()
        time.sleep(0.01)  # Small delay to allow UI updates
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.concat(chunks, ignore_index=True)

# Function to process Excel files with sampling
def process_excel(file, sampling_rate):
    xls = pd.ExcelFile(file)
    sheets = xls.sheet_names
    
    if len(sheets) > 1:
        sheet = st.selectbox("Select sheet to analyze", sheets)
    else:
        sheet = sheets[0]
    
    df = pd.read_excel(file, sheet_name=sheet)
    
    if sampling_rate < 100:
        df = df.sample(frac=sampling_rate/100)
    
    return df

def create_gauge(value, title, color):
    """Create a gauge chart for data quality metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': 80, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ff4444'},
                {'range': [40, 70], 'color': '#ffaa00'},
                {'range': [70, 100], 'color': '#44ff44'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""
if 'data_quality_score' not in st.session_state:
    st.session_state.data_quality_score = 0

def load_data(file_path):
    try:
        if file_path.name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def calculate_data_quality_score(df):
    """Calculate overall data quality score"""
    if df.empty:
        return 0
    
    # Missing values penalty
    missing_penalty = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    
    # Duplicate penalty
    duplicate_penalty = (df.duplicated().sum() / len(df)) * 100
    
    # Column variety bonus (having both numerical and categorical)
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    variety_bonus = 10 if len(numerical_cols) > 0 and len(categorical_cols) > 0 else 0
    
    # Size bonus (reasonable dataset size)
    size_bonus = min(10, len(df) / 1000)  # Max 10 points for datasets with 1000+ rows
    
    # Calculate final score
    base_score = 100
    final_score = max(0, base_score - missing_penalty - duplicate_penalty + variety_bonus + size_bonus)
    
    return min(100, final_score)  # Cap at 100

def missing_values_summary(df):
    """Enhanced missing values analysis"""
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Values': missing.values,
        'Missing (%)': missing_percent.values,
        'Data Type': df.dtypes[missing.index].values
    })
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(
        by='Missing (%)', ascending=False
    ).reset_index(drop=True)
    
    return missing_df

def duplicates_summary(df):
    """Enhanced duplicate analysis"""
    try:
        duplicates_mask = df.duplicated(keep=False)
        duplicate_rows = df[duplicates_mask]
        
        if not duplicate_rows.empty:
            # Group identical rows and count duplicates
            dup_counts = (duplicate_rows.groupby(list(df.columns))
                          .size()
                          .reset_index(name='Duplicate Count')
                          .sort_values('Duplicate Count', ascending=False))
            
            # Calculate summary metrics
            total_duplicates = len(duplicate_rows)
            unique_groups = len(dup_counts)
            duplicate_pct = (total_duplicates / len(df)) * 100
            
            return {
                'all_duplicates': duplicate_rows,
                'grouped_duplicates': dup_counts,
                'metrics': {
                    'total_duplicates': total_duplicates,
                    'unique_groups': unique_groups,
                    'duplicate_pct': duplicate_pct
                }
            }
        return None
    except Exception as e:
        st.error(f"Error analyzing duplicates: {str(e)}")
        return None

def feature_type_overview(df):
    """Enhanced feature classification"""
    numerical = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    
    # Try to identify potential datetime columns in object columns
    potential_datetime = []
    for col in categorical[:]:  # Use slice to avoid modifying during iteration
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(50)
            if len(sample) > 0:
                # Check column name for datetime keywords
                name_check = any(keyword in col.lower() for keyword in 
                               ['date', 'time', 'created', 'updated', 'timestamp', 'dt', 'day', 'month', 'year'])
                
                # Try to parse a few values to see if they're dates
                parse_check = False
                try:
                    # Try parsing first few non-null values
                    test_values = sample.head(3).tolist()
                    parsed_count = 0
                    for val in test_values:
                        try:
                            pd.to_datetime(val, errors='raise')
                            parsed_count += 1
                        except:
                            continue
                    # If at least 2 out of 3 values can be parsed as datetime
                    parse_check = parsed_count >= 2
                except:
                    parse_check = False
                
                if name_check or parse_check:
                    potential_datetime.append(col)
    
    return {
        'numerical': numerical,
        'categorical': [col for col in categorical if col not in potential_datetime],
        'datetime': datetime_cols,
        'potential_datetime': potential_datetime,
        'unhandled': [col for col in df.columns 
                      if col not in numerical + categorical + datetime_cols + potential_datetime]
    }

def numerical_summary(df, numerical_cols):
    """Enhanced numerical summary with additional statistics"""
    if not numerical_cols:
        return None
    
    summary = df[numerical_cols].describe().T
    summary['missing'] = df[numerical_cols].isnull().sum()
    summary['missing_pct'] = (summary['missing'] / len(df)) * 100
    summary['skewness'] = df[numerical_cols].skew()
    summary['kurtosis'] = df[numerical_cols].kurt()
    summary['outliers'] = df[numerical_cols].apply(
        lambda x: len([i for i in x if abs(i - x.mean()) > 2 * x.std()]) if x.std() > 0 else 0
    )
    
    return summary

def categorical_summary(df, categorical_cols):
    """Enhanced categorical summary"""
    if not categorical_cols:
        return None
    
    summary_data = []
    for col in categorical_cols:
        col_data = df[col]
        summary_data.append({
            'Column': col,
            'Unique Values': col_data.nunique(),
            'Most Frequent': col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A',
            'Frequency': col_data.value_counts().iloc[0] if not col_data.empty else 0,
            'Missing Values': col_data.isnull().sum(),
            'Missing (%)': (col_data.isnull().sum() / len(df)) * 100,
            'Cardinality': 'High' if col_data.nunique() > len(df) * 0.5 else 'Low'
        })
    
    return pd.DataFrame(summary_data)

def create_correlation_heatmap(df, numerical_cols):
    """Create an enhanced correlation heatmap"""
    if len(numerical_cols) < 2:
        return None
    
    corr = df[numerical_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        width=800,
        height=600
    )
    
    return fig

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
    if target_column == "None":
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
            "KNN Classifier": KNeighborsClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "SVM Classifier": SVC(probability=True)
        }
    elif task_type == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "KNN Regressor": KNeighborsRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "SVM Regressor": SVR()
        }
    else:
        return {
            "KMeans Clustering": KMeans(),
            "DBSCAN Clustering": DBSCAN(),
            "Agglomerative Clustering": AgglomerativeClustering(),
            "Gaussian Mixture": GaussianMixture()
        }

def get_metrics(task_type):
    if task_type == "classification":
        return {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1_score": lambda y_true, y_pred: 2 * (precision_score(y_true, y_pred, average='weighted') * recall_score(y_true, y_pred, average='weighted')) / (precision_score(y_true, y_pred, average='weighted') + recall_score(y_true, y_pred, average='weighted') + 1e-8)
        }
    else:
        return {
            "MSE": mean_squared_error,
            "R2": r2_score
        }

def prepare_data(df, target_column):
    df = df.dropna(subset=[target_column])
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical to numeric
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns.tolist()

def train_single_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, metric_func):
    y_pred = model.predict(X_test)
    return metric_func(y_test, y_pred)

def save_model(model, feature_columns):
    with open("saved_model.pkl", "wb") as f:
        pickle.dump((model, feature_columns), f)

def load_model():
    with open("saved_model.pkl", "rb") as f:
        return pickle.load(f)

def predict_new_data(model, feature_columns, new_df):
    for col in feature_columns:
        if col not in new_df.columns:
            new_df[col] = 0
    X_new = new_df[feature_columns]
    return model.predict(X_new)

def display_memory_usage():
    mem = get_memory_usage()
    st.sidebar.markdown("---")
    st.sidebar.metric("Memory Usage", f"{mem:.2f} MB")


def perform_clustering(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(scaled_data)
    return kmeans.labels_

# Main application logic
if uploaded_file and uploaded_file.name != st.session_state.uploaded_file_name:
    try:
        with st.spinner("Loading data..."):
            st.session_state.memory_usage = get_memory_usage()
            
            if chunk_processing and uploaded_file.size > 1024 * 1024:  # If file > 1MB and chunk processing enabled
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = process_large_csv(uploaded_file, chunk_size, sampling_rate/100)
                else:
                    st.warning("Chunk processing only available for CSV files. Loading full file.")
                    if uploaded_file.name.endswith(('.xlsx', '.xls')):
                        st.session_state.df = process_excel(uploaded_file, sampling_rate/100)
            else:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                    if sampling_rate < 100:
                        st.session_state.df = st.session_state.df.sample(frac=sampling_rate/100)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    st.session_state.df = process_excel(uploaded_file, sampling_rate/100)
            
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.data_quality_score = calculate_data_quality_score(st.session_state.df)
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            
            # Display memory usage after loading
            st.session_state.memory_usage = get_memory_usage()
            display_memory_usage()
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.session_state.df = pd.DataFrame()

if not st.session_state.df.empty:
    df = st.session_state.df.copy()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📊 Analysis", "🔧 Data Quality", "📈 Visualizations"])
    
    with tab1:
        st.subheader("📋 Raw Data Preview")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            num_rows = st.slider(
                "Number of rows to display:",
                min_value=1,
                max_value=len(df),
                value=10
            )
        with col2:
            show_info = st.checkbox("Show column info", value=True)
        
        st.dataframe(df.head(num_rows), use_container_width=True)
        
        if show_info:
            st.subheader("📊 Dataset Information")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{len(df):,}")
            col2.metric("Total Columns", f"{len(df.columns):,}")
            col3.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.2f} MB")
            col4.metric("File Size", f"{uploaded_file.size / 1024**2:.2f} MB" if uploaded_file else "N/A")
            
            # Quick feature type overview
            features_preview = feature_type_overview(df)
            st.markdown("**Feature Types Overview:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Numerical", len(features_preview['numerical']))
            col2.metric("Categorical", len(features_preview['categorical']))
            col3.metric("DateTime", len(features_preview['datetime']))
            col4.metric("Potential DateTime", len(features_preview['potential_datetime']))
    
    with tab2:
        st.subheader("🔍 Statistical Analysis")
        
        # Feature type classification
        features = feature_type_overview(df)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Numerical Features", len(features['numerical']))
        col2.metric("Categorical Features", len(features['categorical']))
        col3.metric("DateTime Features", len(features['datetime']))
        col4.metric("Potential DateTime", len(features['potential_datetime']))
        
        # Show potential datetime columns if found
        if features['potential_datetime']:
            st.info(f"🕒 Potential datetime columns detected: {', '.join(features['potential_datetime'])}")
            
            # Option to convert potential datetime columns
            if st.button("Convert potential datetime columns"):
                converted_cols = []
                for col in features['potential_datetime']:
                    try:
                        st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce')
                        converted_cols.append(col)
                    except Exception as e:
                        st.warning(f"Could not convert {col}: {str(e)}")
                
                if converted_cols:
                    st.success(f"✅ Successfully converted: {', '.join(converted_cols)}")
                    st.info("Please refresh the page or navigate to another tab and back to see the updated feature types.")
                    # Update session state to track conversion
                    if 'converted_datetime_cols' not in st.session_state:
                        st.session_state.converted_datetime_cols = []
                    st.session_state.converted_datetime_cols.extend(converted_cols)
                    st.rerun()
        
        # Detailed analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(["📈 Numerical", "📝 Categorical", "📅 DateTime", "🔗 Correlations"])
        
        with analysis_tab1:
            if features['numerical']:
                num_summary = numerical_summary(df, features['numerical'])
                st.dataframe(
                    num_summary.style
                    .background_gradient(cmap='Blues', subset=['mean', '50%'])
                    .background_gradient(cmap='Reds', subset=['missing_pct'])
                    .format({
                        'mean': '{:.2f}', 'std': '{:.2f}', '50%': '{:.2f}',
                        'missing_pct': '{:.1f}%', 'skewness': '{:.2f}', 'kurtosis': '{:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Feature distribution
                if len(features['numerical']) > 0:
                    selected_num = st.selectbox("Select feature for distribution analysis:", features['numerical'])
                    if selected_num:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig1 = px.histogram(df, x=selected_num, marginal="box", 
                                              title=f"Distribution of {selected_num}")
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2:
                            fig2 = px.box(df, y=selected_num, title=f"Box Plot of {selected_num}")
                            st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No numerical features found in the dataset")
        
        with analysis_tab2:
            if features['categorical']:
                cat_summary = categorical_summary(df, features['categorical'])
                st.dataframe(
                    cat_summary.style
                    .background_gradient(cmap='Greens', subset=['Unique Values'])
                    .background_gradient(cmap='Reds', subset=['Missing (%)'])
                    .format({
                        'Frequency': '{:,.0f}', 
                        'Missing Values': '{:,.0f}',
                        'Missing (%)': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Feature distribution
                if len(features['categorical']) > 0:
                    selected_cat = st.selectbox("Select feature for distribution analysis:", features['categorical'])
                    if selected_cat:
                        value_counts = df[selected_cat].value_counts().head(20)  # Limit to top 20
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   title=f"Distribution of {selected_cat} (Top 20)")
                        fig.update_xaxes(title=selected_cat)
                        fig.update_yaxes(title="Count")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical features found in the dataset")
        
        with analysis_tab3:
            datetime_features = features['datetime'] + features['potential_datetime']
            if datetime_features:
                st.markdown(f"**Found {len(datetime_features)} datetime-related features**")
                
                # Show datetime column info
                datetime_info = []
                for col in datetime_features:
                    col_data = df[col]
                    if col in features['datetime']:
                        status = "✅ Parsed"
                        date_range = f"{col_data.min()} to {col_data.max()}" if col_data.notna().any() else "No valid dates"
                    else:
                        status = "⚠️ Potential"
                        # Try to show sample values
                        sample_vals = col_data.dropna().head(3).tolist()
                        date_range = f"Sample: {', '.join(map(str, sample_vals))}"
                    
                    datetime_info.append({
                        'Column': col,
                        'Status': status,
                        'Date Range/Sample': date_range,
                        'Missing Values': col_data.isnull().sum(),
                        'Missing (%)': (col_data.isnull().sum() / len(df)) * 100
                    })
                
                st.dataframe(pd.DataFrame(datetime_info), use_container_width=True)
                
                # DateTime visualization for properly parsed columns
                parsed_datetime_cols = [col for col in datetime_features if col in features['datetime']]
                if parsed_datetime_cols:
                    selected_dt = st.selectbox("Select datetime column for analysis:", parsed_datetime_cols)
                    if selected_dt:
                        dt_data = df[selected_dt].dropna()
                        if len(dt_data) > 0:
                            # Time series plot
                            fig = px.histogram(dt_data, x=selected_dt, 
                                             title=f"Distribution of {selected_dt}")
                            fig.update_xaxes(title="Date")
                            fig.update_yaxes(title="Frequency")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show date range info
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Earliest Date", dt_data.min().strftime('%Y-%m-%d'))
                            col2.metric("Latest Date", dt_data.max().strftime('%Y-%m-%d'))
                            col3.metric("Date Span", f"{(dt_data.max() - dt_data.min()).days} days")
            else:
                st.info("No datetime features found in the dataset")
        
        with analysis_tab4:
            if len(features['numerical']) >= 2:
                corr_fig = create_correlation_heatmap(df, features['numerical'])
                if corr_fig:
                    st.plotly_chart(corr_fig, use_container_width=True)
                    
                    # Highlight strong correlations
                    corr = df[features['numerical']].corr()
                    strong_corr = []
                    for i in range(len(corr.columns)):
                        for j in range(i+1, len(corr.columns)):
                            if abs(corr.iloc[i, j]) > 0.7:
                                strong_corr.append({
                                    'Feature 1': corr.columns[i],
                                    'Feature 2': corr.columns[j],
                                    'Correlation': corr.iloc[i, j]
                                })
                    
                    if strong_corr:
                        st.subheader("🔥 Strong Correlations (|r| > 0.7)")
                        st.dataframe(pd.DataFrame(strong_corr))
            else:
                st.info("Need at least 2 numerical features for correlation analysis")
    
    with tab3:
        st.subheader("🔧 Data Quality Assessment")
        
        # Quality score
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            quality_fig = create_gauge(st.session_state.data_quality_score, "Overall Data Quality Score", "blue")
            st.plotly_chart(quality_fig, use_container_width=False)
        
        # Missing values analysis
        st.subheader("❗ Missing Values Analysis")
        missing_df = missing_values_summary(df)
        if not missing_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    missing_df.style
                    .background_gradient(cmap='Reds', subset=['Missing (%)'])
                    .format({'Missing (%)': '{:.2f}%'}),
                    use_container_width=True
                )
            with col2:
                fig = px.bar(missing_df, x='Column', y='Missing (%)',
                           title='Missing Values by Column',
                           color='Missing (%)', color_continuous_scale='reds')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
        
        # Duplicates analysis
        st.subheader("🔍 Duplicates Analysis")
        duplicates = duplicates_summary(df)
        if duplicates:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Duplicate Rows", duplicates['metrics']['total_duplicates'])
            col2.metric("Unique Groups", duplicates['metrics']['unique_groups'])
            col3.metric("Dataset Impact", f"{duplicates['metrics']['duplicate_pct']:.2f}%")
            
            if st.checkbox("Show duplicate details"):
                st.dataframe(duplicates['grouped_duplicates'].head(10))

        else:
            st.success("✅ No duplicate rows found!")
    
    with tab4:
        st.subheader("📈 Advanced Visualizations")
        
        if features['numerical']:
            viz_type = st.selectbox("Select visualization type:", 
                                  ["Pairplot", "Distribution Comparison", "Outlier Analysis"])
            
            if viz_type == "Pairplot" and len(features['numerical']) >= 2:
                selected_features = st.multiselect("Select features for pairplot:", 
                                                 features['numerical'], 
                                                 default=features['numerical'][:4])
                if len(selected_features) >= 2:
                    # Create scatter matrix
                    fig = px.scatter_matrix(df[selected_features])
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Distribution Comparison":
                if len(features['numerical']) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        feature1 = st.selectbox("Feature 1:", features['numerical'])
                    with col2:
                        feature2 = st.selectbox("Feature 2:", features['numerical'])
                    
                    if feature1 != feature2:
                        fig = px.scatter(df, x=feature1, y=feature2, 
                                       title=f"{feature1} vs {feature2}")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Outlier Analysis":
                selected_feature = st.selectbox("Select feature for outlier analysis:", features['numerical'])
                if selected_feature:
                    # Calculate outliers using IQR method
                    Q1 = df[selected_feature].quantile(0.25)
                    Q3 = df[selected_feature].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[selected_feature] < Q1 - 1.5 * IQR) | 
                                (df[selected_feature] > Q3 + 1.5 * IQR)]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Outliers Found", len(outliers))
                        st.metric("Outlier Percentage", f"{(len(outliers)/len(df))*100:.2f}%")
                    
                    with col2:
                        fig = px.box(df, y=selected_feature, title=f"Outliers in {selected_feature}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if len(outliers) > 0 and st.checkbox("Show outlier details"):
                        st.dataframe(outliers)

    
if not st.session_state.df.empty:
    df = st.session_state.df.copy()
    selected_tab = st.sidebar.selectbox(
        "📌 Select Section",
        (
            "🧹 Data Preprocessing",
            "📊 Data Visualization",
            "🤖 Model Setup & Train",
            "📁 Final Report"
        )
    )
    if selected_tab == "🧹 Data Preprocessing":
        st.subheader("🧹 Data Preprocessing")
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
        
        # Check if data exists
        if st.session_state.df.empty:
            st.warning("⚠️ Please upload data first to visualize.")
            st.stop()
        
        df = st.session_state.df.copy()
        
        # Get column types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Visualization categories
        viz_categories = {
            "📊 Single Variable": ["Histogram", "Density Plot", "Box Plot", "Violin Plot", "Bar Chart", "Pie Chart"],
            "📈 Two Variables": ["Scatter Plot", "Line Plot", "Correlation Plot"],
            "🔥 Multi Variable": ["Heatmap", "3D Scatter"],
            "📅 Time Series": ["Time Series Plot", "Seasonal Decomposition"]
        }
        
        # Select visualization category
        selected_category = st.selectbox("Select Visualization Category", list(viz_categories.keys()))
        viz_type = st.selectbox("Select Visualization Type", viz_categories[selected_category])
        
        # Create visualization based on selection
        try:
            # Single Variable Visualizations
            if selected_category == "📊 Single Variable":
                if viz_type == "Histogram":
                    if not numeric_cols:
                        st.error("No numeric columns available for histogram")
                    else:
                        col_name = st.selectbox("Select Column", numeric_cols)
                        bins = st.slider("Number of Bins", 10, 100, 30)
                        
                        if st.button("Generate Histogram"):
                            fig = px.histogram(df, x=col_name, nbins=bins, 
                                            title=f"Histogram of {col_name}")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Density Plot":
                    if not numeric_cols:
                        st.error("No numeric columns available for density plot")
                    else:
                        col_name = st.selectbox("Select Column", numeric_cols)
                        
                        if st.button("Generate Density Plot"):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.kdeplot(data=df, x=col_name, fill=True, ax=ax)
                            ax.set_title(f"Density Plot of {col_name}")
                            st.pyplot(fig)
                
                elif viz_type == "Box Plot":
                    if not numeric_cols:
                        st.error("No numeric columns available for box plot")
                    else:
                        col_name = st.selectbox("Select Column", numeric_cols)
                        group_by = st.selectbox("Group by (Optional)", ["None"] + categorical_cols)
                        
                        if st.button("Generate Box Plot"):
                            if group_by == "None":
                                fig = px.box(df, y=col_name, title=f"Box Plot of {col_name}")
                            else:
                                fig = px.box(df, x=group_by, y=col_name, 
                                        title=f"Box Plot of {col_name} by {group_by}")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Violin Plot":
                    if not numeric_cols:
                        st.error("No numeric columns available for violin plot")
                    else:
                        col_name = st.selectbox("Select Column", numeric_cols)
                        group_by = st.selectbox("Group by (Optional)", ["None"] + categorical_cols)
                        
                        if st.button("Generate Violin Plot"):
                            if group_by == "None":
                                fig = px.violin(df, y=col_name, title=f"Violin Plot of {col_name}")
                            else:
                                fig = px.violin(df, x=group_by, y=col_name,
                                            title=f"Violin Plot of {col_name} by {group_by}")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Bar Chart":
                    if not categorical_cols:
                        st.error("No categorical columns available for bar chart")
                    else:
                        col_name = st.selectbox("Select Column", categorical_cols)
                        top_n = st.slider("Show Top N Categories", 5, 20, 10)
                        
                        if st.button("Generate Bar Chart"):
                            value_counts = df[col_name].value_counts().head(top_n)
                            fig = px.bar(x=value_counts.index, y=value_counts.values,
                                    title=f"Top {top_n} Categories in {col_name}",
                                    labels={'x': col_name, 'y': 'Count'})
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Pie Chart":
                    if not categorical_cols:
                        st.error("No categorical columns available for pie chart")
                    else:
                        col_name = st.selectbox("Select Column", categorical_cols)
                        top_n = st.slider("Show Top N Categories", 5, 15, 10)
                        
                        if st.button("Generate Pie Chart"):
                            value_counts = df[col_name].value_counts().head(top_n)
                            fig = px.pie(values=value_counts.values, names=value_counts.index,
                                    title=f"Distribution of {col_name}")
                            st.plotly_chart(fig, use_container_width=True)
            
            # Two Variables Visualizations
            elif selected_category == "📈 Two Variables":
                if viz_type == "Scatter Plot":
                    if len(numeric_cols) < 2:
                        st.error("Need at least 2 numeric columns for scatter plot")
                    else:
                        x_col = st.selectbox("Select X Column", numeric_cols)
                        y_col = st.selectbox("Select Y Column", numeric_cols)
                        color_by = st.selectbox("Color by (Optional)", ["None"] + categorical_cols)
                        size_by = st.selectbox("Size by (Optional)", ["None"] + numeric_cols)
                        
                        if st.button("Generate Scatter Plot"):
                            kwargs = {'x': x_col, 'y': y_col, 'title': f"{y_col} vs {x_col}"}
                            if color_by != "None":
                                kwargs['color'] = color_by
                            if size_by != "None":
                                kwargs['size'] = size_by
                            
                            fig = px.scatter(df, **kwargs)
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Line Plot":
                    if len(numeric_cols) < 2:
                        st.error("Need at least 2 numeric columns for line plot")
                    else:
                        x_col = st.selectbox("Select X Column", numeric_cols + datetime_cols)
                        y_col = st.selectbox("Select Y Column", numeric_cols)
                        color_by = st.selectbox("Color by (Optional)", ["None"] + categorical_cols)
                        
                        if st.button("Generate Line Plot"):
                            kwargs = {'x': x_col, 'y': y_col, 'title': f"{y_col} over {x_col}"}
                            if color_by != "None":
                                kwargs['color'] = color_by
                            
                            fig = px.line(df, **kwargs)
                            st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Correlation Plot":
                    if len(numeric_cols) < 2:
                        st.error("Need at least 2 numeric columns for correlation plot")
                    else:
                        selected_cols = st.multiselect("Select Columns for Correlation", 
                                                    numeric_cols, default=numeric_cols[:5])
                        
                        if st.button("Generate Correlation Plot"):
                            if len(selected_cols) >= 2:
                                corr_matrix = df[selected_cols].corr()
                                fig = px.imshow(corr_matrix, 
                                            title="Correlation Matrix",
                                            color_continuous_scale='RdBu',
                                            aspect="auto")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Please select at least 2 columns")
            
            # Multi Variable Visualizations
            elif selected_category == "🔥 Multi Variable":
                if viz_type == "Heatmap":
                    if len(numeric_cols) < 2:
                        st.error("Need at least 2 numeric columns for heatmap")
                    else:
                        selected_cols = st.multiselect("Select Columns", numeric_cols, 
                                                    default=numeric_cols[:min(10, len(numeric_cols))])
                        
                        if st.button("Generate Heatmap"):
                            if len(selected_cols) >= 2:
                                fig, ax = plt.subplots(figsize=(12, 8))
                                corr = df[selected_cols].corr()
                                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                                ax.set_title("Correlation Heatmap")
                                st.pyplot(fig)
                            else:
                                st.error("Please select at least 2 columns")
                
                elif viz_type == "3D Scatter":
                    if len(numeric_cols) < 3:
                        st.error("Need at least 3 numeric columns for 3D scatter plot")
                    else:
                        x_col = st.selectbox("Select X Column", numeric_cols)
                        y_col = st.selectbox("Select Y Column", numeric_cols)
                        z_col = st.selectbox("Select Z Column", numeric_cols)
                        color_by = st.selectbox("Color by (Optional)", ["None"] + categorical_cols)
                        
                        if st.button("Generate 3D Scatter"):
                            kwargs = {'x': x_col, 'y': y_col, 'z': z_col, 
                                    'title': f"3D Scatter: {x_col}, {y_col}, {z_col}"}
                            if color_by != "None":
                                kwargs['color'] = color_by
                            
                            fig = px.scatter_3d(df, **kwargs)
                            st.plotly_chart(fig, use_container_width=True)
            
            # Time Series Visualizations
            elif selected_category == "📅 Time Series":
                if not datetime_cols:
                    st.error("No datetime columns available for time series analysis")
                else:
                    if viz_type == "Time Series Plot":
                        date_col = st.selectbox("Select Date Column", datetime_cols)
                        value_col = st.selectbox("Select Value Column", numeric_cols)
                        
                        if st.button("Generate Time Series Plot"):
                            fig = px.line(df, x=date_col, y=value_col,
                                        title=f"Time Series: {value_col} over {date_col}")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Seasonal Decomposition":
                        st.info("Seasonal decomposition requires regular time intervals and sufficient data points")
                        date_col = st.selectbox("Select Date Column", datetime_cols)
                        value_col = st.selectbox("Select Value Column", numeric_cols)
                        period = st.number_input("Period (e.g., 12 for monthly, 365 for daily)", 
                                            min_value=2, max_value=365, value=12)
                        
                        if st.button("Generate Seasonal Decomposition"):
                            try:
                                # Prepare data
                                ts_data = df[[date_col, value_col]].copy()
                                ts_data = ts_data.dropna().sort_values(date_col)
                                ts_data.set_index(date_col, inplace=True)
                                
                                # Perform decomposition
                                decomposition = seasonal_decompose(ts_data[value_col], 
                                                                period=period, model='additive')
                                
                                # Plot results
                                fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                                decomposition.observed.plot(ax=axes[0], title='Original')
                                decomposition.trend.plot(ax=axes[1], title='Trend')
                                decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                                decomposition.resid.plot(ax=axes[3], title='Residual')
                                plt.tight_layout()
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error in seasonal decomposition: {str(e)}")
        
        except Exception as e:
            st.error(f"⚠️ An error occurred while generating the visualization: {str(e)}")
            st.info("Please check your data and column selections.")
        
        # Add download option for plots
        st.markdown("---")
        st.info("💡 Tip: Right-click on any plot to save or download it!")

    elif selected_tab == "🤖 Model Setup & Train":
        st.header("🧠 Model Training")

        if "df" not in st.session_state or st.session_state.df.empty:
            st.warning("⚠️ Please upload and preprocess data first.")
            st.stop()

        df = st.session_state.df.copy()

        # Step 1: Select target column
        target_column = st.selectbox("🎯 Select the target column", ["None"] + df.columns.tolist())

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
                ["linear", "poly", "rbf", "sigmoid"],
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
        # if st.button("🚀 Train Selected Model" if task_type in ["classification", "regression"] ):
        if task_type in ["classification", "regression"] :
            if st.button("🚀 Train Selected Model"):
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

                    except Exception as e:
                        st.error(f"❌ Error during model training/clustering: {str(e)}")
                        
        if task_type == "clustering":
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
                    # Get clustering parameters
                    n_clusters = st.slider("Number of clusters", 2, 10, 3)
                    
                    # Add a button to trigger clustering
                    if st.button("🚀 Perform Clustering"):
                        with st.spinner("Clustering in progress..."):
                            # Perform clustering
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(numeric_df[cluster_cols])
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            st.session_state.labels = kmeans.fit_predict(scaled_data)
                            
                            # Store clustering parameters to detect changes
                            st.session_state.cluster_params = {
                                'n_clusters': n_clusters,
                                'features': cluster_cols
                            }
                            
                            st.success("✅ Clustering completed!")
                    
                    # Only show results if clustering has been performed
                    if 'labels' in st.session_state:
                        # Check if parameters have changed
                        current_params = {
                            'n_clusters': n_clusters,
                            'features': cluster_cols
                        }
                        
                        # If parameters changed, prompt user to re-run clustering
                        if ('cluster_params' in st.session_state and 
                            current_params != st.session_state.cluster_params):
                            st.warning("⚠️ Clustering parameters have changed. Please click 'Perform Clustering' to update.")
                        
                        else:
                            # Show clustering results
                            df_with_labels = numeric_df.copy()
                            df_with_labels["Cluster"] = st.session_state.labels
                            
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

            except ValueError as ve:
                st.error(f"❌ Value Error: {str(ve)}")
    

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

    elif selected_tab == "📁 Final Report":
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

if not st.session_state.df.empty:
    csv = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="⬇️ Download Processed Data",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv"
    )
    
else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to OmniAnalyzer!
    
    This powerful tool helps you analyze your data comprehensively. Here's what you can do:
    
    ### 🚀 Features:
    - **📊 Data Overview**: Quick preview and basic statistics
    - **🔍 Deep Analysis**: Statistical summaries and feature analysis  
    - **🔧 Quality Assessment**: Missing values, duplicates, and quality scoring
    - **📈 Visualizations**: Interactive charts and correlation analysis
    
    ### 📁 Supported Formats:
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    
    ### 🎯 Get Started:
    1. Upload your data file using the sidebar
    2. Explore the different tabs for comprehensive analysis
    3. Use the interactive features to dive deeper into your data
    
    **Ready to begin?** Upload a file to start your data analysis journey!
    """)
    
    # Add some example use cases
    with st.expander("💡 Use Cases & Examples"):
        st.markdown("""
        **Perfect for:**
        - Data scientists exploring new datasets
        - Business analysts preparing reports
        - Students learning data analysis
        - Anyone needing quick data insights
        
        **Example datasets to try:**
        - Sales data with customer information
        - Survey responses with multiple choice questions
        - Time series data with dates and measurements
        - Any structured data in CSV or Excel format
        """)