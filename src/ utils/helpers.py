# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# PREDICTION ANALYSIS
# =============================================================================
def analyze_prediction(prediction, features, model):
    """Analyze prediction with confidence intervals and similar medicines"""
    results = {
        'prediction': prediction,
        'confidence_interval': calculate_confidence_interval(prediction, features['probabilities']),
        'similar_medicines': find_similar_medicines(features),
        'feature_importance': calculate_feature_importance(model, features)
    }
    
    display_analysis_results(results)
    return results

def calculate_confidence_interval(prediction, probabilities):
    """Calculate 95% confidence interval for prediction"""
    try:
        std_dev = np.std(probabilities) * 100
        return (
            max(0, prediction - 1.96 * std_dev),
            min(100, prediction + 1.96 * std_dev)
        )
    except Exception as e:
        st.error(f"Error calculating confidence interval: {str(e)}")
        return (prediction, prediction)

# =============================================================================
# SIMILARITY ANALYSIS
# =============================================================================
def find_similar_medicines(features):
    """Find similar medicines based on input features"""
    if not validate_data():
        return pd.DataFrame()
    
    try:
        feature_matrix = prepare_feature_matrix()
        similarities = calculate_similarities(features, feature_matrix)
        return get_top_similar_medicines(similarities)
    except Exception as e:
        st.error(f"Error finding similar medicines: {str(e)}")
        return pd.DataFrame()

def validate_data():
    """Validate data availability"""
    if st.session_state.data is None:
        st.warning("No reference data available for similarity analysis")
        return False
    return True

def prepare_feature_matrix():
    """Prepare and scale feature matrix"""
    feature_cols = ['Price', 'Composition_Length', 'Side_Effects_Count']
    X = st.session_state.data[feature_cols].values
    
    scaler = MinMaxScaler()
    return {
        'matrix': scaler.fit_transform(X),
        'scaler': scaler,
        'columns': feature_cols
    }

def calculate_similarities(features, feature_matrix):
    """Calculate cosine similarities"""
    features_scaled = feature_matrix['scaler'].transform(
        features[feature_matrix['columns']].values.reshape(1, -1)
    )
    return cosine_similarity(features_scaled, feature_matrix['matrix'])

def get_top_similar_medicines(similarities, top_n=5):
    """Get top similar medicines"""
    similar_indices = similarities[0].argsort()[-top_n:][::-1]
    similar_medicines = st.session_state.data.iloc[similar_indices].copy()
    
    similar_medicines['Similarity'] = similarities[0][similar_indices].round(3)
    return similar_medicines

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
def calculate_feature_importance(model, features):
    """Calculate and format feature importance"""
    try:
        importance = model.feature_importances_
        return pd.DataFrame({
            'Feature': features.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
    except Exception as e:
        st.warning(f"Could not calculate feature importance: {str(e)}")
        return None

# =============================================================================
# VISUALIZATION
# =============================================================================
def display_analysis_results(results):
    """Display comprehensive analysis results"""
    display_prediction_summary(results)
    display_similar_medicines(results['similar_medicines'])
    display_feature_importance(results['feature_importance'])

def display_prediction_summary(results):
    """Display prediction summary with confidence interval"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Predicted Effectiveness",
            f"{results['prediction']:.1f}%",
            help="Predicted effectiveness score"
        )
    
    with col2:
        ci_low, ci_high = results['confidence_interval']
        st.metric(
            "Confidence Interval",
            f"{ci_low:.1f}% - {ci_high:.1f}%",
            help="95% confidence interval"
        )

def display_similar_medicines(similar_medicines):
    """Display similar medicines with interactive table"""
    if not similar_medicines.empty:
        st.subheader("Similar Medicines")
        st.dataframe(
            similar_medicines[[
                'Medicine_Name', 'Manufacturer', 'Price', 
                'Overall_Score', 'Similarity'
            ]].style.background_gradient(subset=['Similarity']),
            use_container_width=True
        )

def display_feature_importance(feature_importance):
    """Display feature importance visualization"""
    if feature_importance is not None:
        st.subheader("Feature Importance")
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# REPORTING
# =============================================================================
def generate_analysis_report(results):
    """Generate detailed analysis report"""
    return f"""
    # Medicine Effectiveness Analysis Report
    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Prediction Results
    - Predicted Effectiveness: {results['prediction']:.1f}%
    - Confidence Interval: {results['confidence_interval'][0]:.1f}% - {results['confidence_interval'][1]:.1f}%
    
    ## Similar Medicines
    {results['similar_medicines'].to_markdown()}
    
    ## Feature Importance
    {results['feature_importance'].to_markdown() if results['feature_importance'] is not None else 'Not available'}
    """

def export_results(results, format='csv'):
    """Export analysis results to specified format"""
    try:
        export_data = {
            'prediction': results['prediction'],
            'confidence_interval': results['confidence_interval'],
            'similar_medicines': results['similar_medicines'].to_dict(),
            'feature_importance': results['feature_importance'].to_dict() if results['feature_importance'] is not None else None
        }
        
        return pd.DataFrame([export_data])
    except Exception as e:
        st.error(f"Error exporting results: {str(e)}")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Medicine Analysis", page_icon="🔍", layout="wide")
    st.title("Medicine Analysis Tools")