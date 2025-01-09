# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost
import plotly.graph_objects as go

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
def preprocess_dataset(numeric_features, categorical_features, target, 
                      scale=True, handle_missing=True, encode_categorical=True):
    """
    Preprocess the dataset based on selected options
    
    Args:
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        target (str): Target variable name
        scale (bool): Whether to scale numeric features
        handle_missing (bool): Whether to handle missing values
        encode_categorical (bool): Whether to encode categorical variables
        
    Returns:
        tuple: Processed features (X) and target (y)
    """
    try:
        data = st.session_state.data.copy()
        features = numeric_features + categorical_features
        X = data[features]
        y = data[target]
        
        if handle_missing:
            X = handle_missing_values(X, numeric_features, categorical_features)
        
        if scale:
            X = scale_numeric_features(X, numeric_features)
        
        if encode_categorical:
            X = encode_categorical_features(X, categorical_features)
        
        return X, y
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None

def handle_missing_values(X, numeric_features, categorical_features):
    """Handle missing values in the dataset"""
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())
    X[categorical_features] = X[categorical_features].fillna('Unknown')
    return X

def scale_numeric_features(X, numeric_features):
    """Scale numeric features using StandardScaler"""
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    st.session_state.scaler = scaler
    return X

def encode_categorical_features(X, categorical_features):
    """Encode categorical features using one-hot encoding"""
    return pd.get_dummies(X, columns=categorical_features)

# =============================================================================
# MODEL TRAINING
# =============================================================================
def train_selected_model(model_type, params, X_train, X_val, y_train, y_val):
    """
    Train the selected model type with given parameters
    
    Args:
        model_type (str): Type of model to train
        params (dict): Model parameters
        X_train, X_val (pd.DataFrame): Training and validation features
        y_train, y_val (pd.Series): Training and validation targets
        
    Returns:
        object: Trained model
    """
    try:
        model = create_model(model_type, params)
        if model is None:
            return None
            
        eval_set = [(X_val, y_val)]
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            verbose=False
        )
        
        return model
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None

def create_model(model_type, params):
    """Create model instance based on type"""
    model_classes = {
        'LightGBM': lgb.LGBMRegressor,
        'XGBoost': xgb.XGBRegressor,
        'CatBoost': CatBoost
    }
    
    if model_type not in model_classes:
        st.error(f"Unsupported model type: {model_type}")
        return None
        
    return model_classes[model_type](**params)

# =============================================================================
# MODEL EVALUATION
# =============================================================================
def calculate_metrics(model, X_val, y_val):
    """
    Calculate model performance metrics
    
    Args:
        model: Trained model
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation targets
        
    Returns:
        dict: Dictionary of performance metrics
    """
    try:
        y_pred = model.predict(X_val)
        
        metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        
    Returns:
        go.Figure: Plotly figure object
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            return None
            
        importance_df = create_importance_dataframe(model, feature_names)
        fig = create_importance_plot(importance_df)
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")
        return None

def create_importance_dataframe(model, feature_names):
    """Create dataframe of feature importance scores"""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    return importance_df.sort_values('Importance', ascending=True)

def create_importance_plot(importance_df):
    """Create feature importance plot"""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color='rgb(58, 149, 136)'
        )
    )
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(400, len(importance_df) * 20),
        showlegend=False
    )
    
    return fig

# =============================================================================
# MODEL VALIDATION
# =============================================================================
def validate_model_performance(model, X_val, y_val):
    """
    Additional validation checks for model performance
    
    Args:
        model: Trained model
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation targets
        
    Returns:
        go.Figure: Plotly figure object with residual plot
    """
    try:
        y_pred = model.predict(X_val)
        residuals = y_val - y_pred
        
        fig = create_residual_plot(y_pred, residuals)
        return fig
        
    except Exception as e:
        st.error(f"Error in model validation: {str(e)}")
        return None

def create_residual_plot(y_pred, residuals):
    """Create residual plot"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='rgb(58, 149, 136)'),
            name='Residuals'
        )
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        showlegend=True
    )
    
    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Model Training Helper", page_icon="🤖", layout="wide")
    st.title("Model Training and Evaluation Helpers")