"""
Streamlit Dashboard for Housing Price Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import glob
import os
import sys
from datetime import datetime
import joblib
from PIL import Image

sys.path.insert(0, os.path.abspath('..'))

# Page configuration
st.set_page_config(
    page_title="Housing Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    .stAlert {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_metrics():
    """Load all metrics files"""
    # Try multiple locations - prioritize src/ directory
    metrics_files = []
    search_paths = [
        '../src/metrics/*_metrics.json',
        'src/metrics/*_metrics.json',
        '../metrics/*_metrics.json',
        'metrics/*_metrics.json',
        '*_metrics.json'
    ]
    
    for pattern in search_paths:
        metrics_files = glob.glob(pattern)
        if metrics_files:
            break
    
    metrics_data = []
    
    for file in metrics_files:
        timestamp = os.path.basename(file).split('_')[0]
        with open(file, 'r') as f:
            metrics = json.load(f)
            metrics['timestamp'] = timestamp
            metrics['datetime'] = datetime.strptime(timestamp, '%Y%m%d%H%M%S')
            metrics_data.append(metrics)
    
    return pd.DataFrame(metrics_data).sort_values('datetime', ascending=False)


@st.cache_resource
def load_model(model_path, preprocessor_path):
    """Load model and preprocessor"""
    model = joblib.load(model_path)
    
    sys.path.insert(0, os.path.abspath('../src'))
    from preprocessing import HousingPreprocessor
    preprocessor = HousingPreprocessor.load_preprocessor(preprocessor_path)
    
    return model, preprocessor


def get_latest_model():
    """Get the latest trained model"""
    # Try multiple locations - prioritize src/ directory
    model_files = []
    search_paths = [
        '../src/models/model_*_gradient_boosting.joblib',
        'src/models/model_*_gradient_boosting.joblib',
        '../models/model_*_gradient_boosting.joblib',
        'models/model_*_gradient_boosting.joblib',
        'model_*_gradient_boosting.joblib',
        '../model_*_gradient_boosting.joblib'
    ]
    
    for pattern in search_paths:
        found = glob.glob(pattern)
        if found:
            model_files.extend(found)
            break
    
    if not model_files:
        return None, None, None
    
    model_files.sort(reverse=True)
    latest_model = model_files[0]
    
    timestamp = os.path.basename(latest_model).split('_')[1]
    
    # Try multiple locations for preprocessor
    preprocessor_path = None
    preprocessor_patterns = [
        f'../src/models/preprocessor_{timestamp}.pkl',
        f'src/models/preprocessor_{timestamp}.pkl',
        f'../models/preprocessor_{timestamp}.pkl',
        f'models/preprocessor_{timestamp}.pkl',
        f'preprocessor_{timestamp}.pkl'
    ]
    
    for pattern in preprocessor_patterns:
        if os.path.exists(pattern):
            preprocessor_path = pattern
            break
    
    return latest_model, preprocessor_path, timestamp


def create_metrics_chart(df):
    """Create interactive metrics comparison chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RMSE Over Time', 'MAE Over Time', 
                       'R² Score Over Time', 'MAPE Over Time'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # RMSE
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['RMSE'], 
                  mode='lines+markers', name='RMSE',
                  line=dict(color='#FF6B6B', width=3)),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['MAE'], 
                  mode='lines+markers', name='MAE',
                  line=dict(color='#4ECDC4', width=3)),
        row=1, col=2
    )
    
    # R²
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['R2_Score'], 
                  mode='lines+markers', name='R²',
                  line=dict(color='#95E1D3', width=3)),
        row=2, col=1
    )
    
    # MAPE
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['MAPE'], 
                  mode='lines+markers', name='MAPE',
                  line=dict(color='#F38181', width=3)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Model Performance Metrics Over Time",
        title_font_size=20
    )
    
    return fig


def create_comparison_table(df):
    """Create styled comparison table"""
    display_df = df[['timestamp', 'RMSE', 'MAE', 'R2_Score', 'MAPE']].copy()
    display_df.columns = ['Timestamp', 'RMSE', 'MAE', 'R² Score', 'MAPE (%)']
    
    # Style the dataframe
    styled_df = display_df.style.background_gradient(
        subset=['RMSE', 'MAE'], 
        cmap='RdYlGn_r'
    ).background_gradient(
        subset=['R² Score'], 
        cmap='RdYlGn'
    ).format({
        'RMSE': '{:.4f}',
        'MAE': '{:.4f}',
        'R² Score': '{:.4f}',
        'MAPE (%)': '{:.2f}%'
    })
    
    return styled_df


# Main App
def main():
    # Header
    st.title("🏠 California Housing Price Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/home.png", width=100)
        st.title("Navigation")
        page = st.radio(
            "Choose a page:",
            ["📊 Overview", "🎯 Live Prediction", "📈 Model Comparison", "📋 About"]
        )
        
        st.markdown("---")
        st.markdown("### Model Info")
        
        latest_model, preprocessor_path, timestamp = get_latest_model()
        if latest_model:
            st.success(f"✅ Model Loaded")
            st.info(f"**Version:** {timestamp}")
        else:
            st.error("❌ No model found")
    
    # Page routing
    if page == "📊 Overview":
        show_overview()
    elif page == "🎯 Live Prediction":
        show_prediction()
    elif page == "📈 Model Comparison":
        show_comparison()
    else:
        show_about()


def show_overview():
    """Overview page with key metrics and visualizations"""
    st.header("📊 Model Performance Overview")
    
    # Load metrics
    try:
        df = load_metrics()
        
        if df.empty:
            st.warning("No metrics found. Please train a model first.")
            return
        
        latest = df.iloc[0]
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="RMSE",
                value=f"{latest['RMSE']:.4f}",
                delta=f"{(latest['RMSE'] - df.iloc[1]['RMSE']):.4f}" if len(df) > 1 else None,
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="MAE",
                value=f"{latest['MAE']:.4f}",
                delta=f"{(latest['MAE'] - df.iloc[1]['MAE']):.4f}" if len(df) > 1 else None,
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="R² Score",
                value=f"{latest['R2_Score']:.4f}",
                delta=f"{(latest['R2_Score'] - df.iloc[1]['R2_Score']):.4f}" if len(df) > 1 else None
            )
        
        with col4:
            st.metric(
                label="MAPE",
                value=f"{latest['MAPE']:.2f}%",
                delta=f"{(latest['MAPE'] - df.iloc[1]['MAPE']):.2f}%" if len(df) > 1 else None,
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Charts
        if len(df) > 1:
            st.plotly_chart(create_metrics_chart(df), use_container_width=True)
        
        # Display evaluation plot
        st.subheader("📈 Model Evaluation Plots")
        
        # Try multiple locations for plots
        plot_files = []
        plot_patterns = [
            f'../src/plots/evaluation_{latest["timestamp"]}_*.png',
            f'src/plots/evaluation_{latest["timestamp"]}_*.png',
            f'../plots/evaluation_{latest["timestamp"]}_*.png',
            f'plots/evaluation_{latest["timestamp"]}_*.png',
            f'evaluation_{latest["timestamp"]}_*.png'
        ]
        
        for pattern in plot_patterns:
            found = glob.glob(pattern)
            if found:
                plot_files.extend(found)
                break
        
        if plot_files:
            col1, col2 = st.columns(2)
            with col1:
                image = Image.open(plot_files[0])
                st.image(image, caption="Model Evaluation", use_column_width=True)
            
            # Feature importance
            feat_files = []
            feat_patterns = [
                f'../src/plots/feature_importance_{latest["timestamp"]}_*.png',
                f'src/plots/feature_importance_{latest["timestamp"]}_*.png',
                f'../plots/feature_importance_{latest["timestamp"]}_*.png',
                f'plots/feature_importance_{latest["timestamp"]}_*.png',
                f'feature_importance_{latest["timestamp"]}_*.png'
            ]
            
            for pattern in feat_patterns:
                found = glob.glob(pattern)
                if found:
                    feat_files.extend(found)
                    break
            
            if feat_files:
                with col2:
                    image = Image.open(feat_files[0])
                    st.image(image, caption="Feature Importance", use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading metrics: {e}")


def show_prediction():
    """Live prediction interface"""
    st.header("🎯 Make a Prediction")
    
    latest_model_path, preprocessor_path, timestamp = get_latest_model()
    
    if not latest_model_path:
        st.error("No trained model found. Please train a model first.")
        return
    
    try:
        model, preprocessor = load_model(latest_model_path, preprocessor_path)
        
        st.success(f"✅ Model loaded (Version: {timestamp})")
        
        # Input form
        st.subheader("Enter Housing Features:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            med_inc = st.number_input(
                "Median Income ($10k)",
                min_value=0.0, max_value=15.0, value=3.5, step=0.1,
                help="Median income in block group"
            )
            
            house_age = st.slider(
                "House Age (years)",
                min_value=1, max_value=52, value=25,
                help="Median house age in block group"
            )
            
            ave_rooms = st.number_input(
                "Average Rooms",
                min_value=1.0, max_value=20.0, value=5.5, step=0.1,
                help="Average number of rooms per household"
            )
        
        with col2:
            ave_bedrms = st.number_input(
                "Average Bedrooms",
                min_value=0.5, max_value=10.0, value=1.2, step=0.1,
                help="Average number of bedrooms per household"
            )
            
            population = st.number_input(
                "Population",
                min_value=1, max_value=10000, value=1200, step=10,
                help="Block group population"
            )
            
            ave_occup = st.number_input(
                "Average Occupancy",
                min_value=1.0, max_value=10.0, value=3.0, step=0.1,
                help="Average household size"
            )
        
        with col3:
            latitude = st.number_input(
                "Latitude",
                min_value=32.0, max_value=42.0, value=34.05, step=0.01,
                help="Block group latitude"
            )
            
            longitude = st.number_input(
                "Longitude",
                min_value=-125.0, max_value=-114.0, value=-118.25, step=0.01,
                help="Block group longitude"
            )
        
        # Predict button
        if st.button("🔮 Predict Price", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'MedInc': [med_inc],
                'HouseAge': [house_age],
                'AveRooms': [ave_rooms],
                'AveBedrms': [ave_bedrms],
                'Population': [population],
                'AveOccup': [ave_occup],
                'Latitude': [latitude],
                'Longitude': [longitude]
            })
            
            # Preprocess
            input_features = preprocessor.create_features(input_data)
            input_processed = preprocessor.transform(input_features)
            
            # Predict
            prediction = model.predict(input_processed)[0]
            prediction_usd = prediction * 100000
            
            # Display result
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Price", f"${prediction:.2f} × 100k")
            
            with col2:
                st.metric("Predicted Price (USD)", f"${prediction_usd:,.0f}")
            
            with col3:
                # Confidence based on R² score
                metrics = load_metrics()
                confidence = metrics.iloc[0]['R2_Score'] * 100
                st.metric("Model Confidence", f"{confidence:.1f}%")
            
            st.balloons()
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")


def show_comparison():
    """Model comparison page"""
    st.header("📈 Model Version Comparison")
    
    try:
        df = load_metrics()
        
        if df.empty:
            st.warning("No metrics found.")
            return
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Best Performing Model")
            best_model = df.loc[df['R2_Score'].idxmax()]
            st.info(f"""
            **Timestamp:** {best_model['timestamp']}  
            **R² Score:** {best_model['R2_Score']:.4f}  
            **RMSE:** {best_model['RMSE']:.4f}  
            **MAE:** {best_model['MAE']:.4f}
            """)
        
        with col2:
            st.markdown("### Latest Model")
            latest = df.iloc[0]
            st.success(f"""
            **Timestamp:** {latest['timestamp']}  
            **R² Score:** {latest['R2_Score']:.4f}  
            **RMSE:** {latest['RMSE']:.4f}  
            **MAE:** {latest['MAE']:.4f}
            """)
        
        st.markdown("---")
        
        # Detailed comparison table
        st.subheader("📋 Detailed Comparison")
        st.dataframe(create_comparison_table(df), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading comparison: {e}")


def show_about():
    """About page"""
    st.header("📋 About This Dashboard")
    
    st.markdown("""
    ## 🏠 California Housing Price Prediction
    
    This dashboard provides a comprehensive interface for monitoring and interacting 
    with machine learning models trained on the California Housing dataset.
    
    ### 🎯 Features
    
    - **Real-time Predictions**: Make predictions using the latest trained model
    - **Performance Monitoring**: Track model metrics over time
    - **Model Comparison**: Compare different model versions
    - **Interactive Visualizations**: Explore model performance visually
    
    ### 📊 Dataset
    
    The California Housing dataset contains information from the 1990 Census:
    - **20,640 observations**
    - **8 features**: Median income, house age, rooms, bedrooms, population, occupancy, location
    - **Target**: Median house value
    
    ### 🤖 Model Details
    
    - **Algorithm**: Gradient Boosting Regressor
    - **Preprocessing**: Standard scaling + feature engineering
    - **Evaluation**: RMSE, MAE, R², MAPE
    - **Tracking**: MLflow for experiment management
    - **Deployment**: FastAPI for serving predictions
    
    ### 🛠️ Technology Stack
    
    - **ML**: scikit-learn, XGBoost
    - **Tracking**: MLflow
    - **API**: FastAPI
    - **Dashboard**: Streamlit
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **CI/CD**: GitHub Actions
    
 

    """)


if __name__ == "__main__":
    main()




