import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def calculate_confidence_interval(prediction, probabilities):
    """Calculate 95% confidence interval for prediction"""
    std_dev = np.std(probabilities) * 100
    return (
        max(0, prediction - 1.96 * std_dev),
        min(100, prediction + 1.96 * std_dev)
    )

def show_similar_medicines(features):
    """Find and display similar medicines from the dataset"""
    if st.session_state.data is None:
        return pd.DataFrame()
    
    # Prepare feature matrix
    feature_cols = ['Price', 'Composition_Length', 'Side_Effects_Count']
    X = st.session_state.data[feature_cols].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale input features
    features_scaled = scaler.transform(features[feature_cols].values.reshape(1, -1))
    
    # Calculate similarities
    similarities = cosine_similarity(features_scaled, X_scaled)
    
    # Get top 5 similar medicines
    similar_indices = similarities[0].argsort()[-5:][::-1]
    similar_medicines = st.session_state.data.iloc[similar_indices].copy()
    
    # Add similarity score
    similar_medicines['Similarity'] = similarities[0][similar_indices]
    similar_medicines['Similarity'] = similar_medicines['Similarity'].round(3)
    
    # Display results
    st.dataframe(
        similar_medicines[[
            'Medicine_Name', 'Manufacturer', 'Price', 
            'Overall_Score', 'Similarity'
        ]].style.background_gradient(subset=['Similarity'])
    )
    
    return similar_medicines

def calculate_feature_importance(model, features):
    """Calculate feature importance for the current prediction"""
    try:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': importance
        })
        return feature_importance.sort_values('Importance', ascending=False)
    except:
        return None

def generate_analysis_report(prediction, features, similar_medicines):
    """Generate a detailed analysis report"""
    report = f"""
    # Medicine Effectiveness Analysis Report
    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Prediction Results
    - Predicted Effectiveness: {prediction:.1f}%
    - Confidence Level: {max(features['probabilities']):.1f}%
    
    ## Input Features
    {features.to_markdown()}
    
    ## Similar Medicines
    {similar_medicines.to_markdown()}
    """
    
    return report

def export_results(prediction, features, similar_medicines):
    """Export analysis results to various formats"""
    results = {
        'prediction': prediction,
        'features': features.to_dict(),
        'similar_medicines': similar_medicines.to_dict()
    }
    
    return pd.DataFrame([results])