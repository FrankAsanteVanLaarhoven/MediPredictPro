# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# =============================================================================
# MAIN PREDICTION INTERFACE
# =============================================================================
def enhance_prediction_interface():
    """Enhanced prediction interface with additional features"""
    st.title("MediPredictPro - Medicine Effectiveness Prediction")
    
    with st.form("prediction_form"):
        render_input_columns()
        advanced_options = render_advanced_options()
        
        if st.form_submit_button("Predict Effectiveness", type="primary"):
            process_prediction_request(advanced_options)

def render_input_columns():
    """Render input form columns"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.medicine_details = render_medicine_details()
    with col2:
        st.session_state.additional_info = render_additional_info()

# =============================================================================
# INPUT SECTIONS
# =============================================================================
def render_medicine_details():
    """Render medicine details section"""
    st.subheader("Medicine Details")
    
    return {
        'name': render_name_input(),
        'composition': render_composition_input(),
        'manufacturer': render_manufacturer_input(),
        'price': render_price_input(),
        'dosage_form': render_dosage_form_input()
    }

def render_additional_info():
    """Render additional information section"""
    st.subheader("Additional Information")
    
    return {
        'side_effects': render_side_effects_input(),
        'usage_metrics': render_usage_metrics(),
        'storage_requirements': render_storage_requirements()
    }

# =============================================================================
# INPUT COMPONENTS
# =============================================================================
def render_name_input():
    """Render medicine name input"""
    return st.text_input(
        "Medicine Name",
        help="Enter the complete medicine name"
    )

def render_composition_input():
    """Render composition input"""
    return st.text_area(
        "Composition",
        help="Enter the full medicine composition"
    )

def render_manufacturer_input():
    """Render manufacturer selection"""
    manufacturers = get_sorted_manufacturers()
    return st.selectbox(
        "Manufacturer",
        options=manufacturers,
        help="Select the medicine manufacturer"
    )

def render_price_input():
    """Render price input"""
    return st.number_input(
        "Price",
        min_value=0.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
        help="Enter medicine price"
    )

# =============================================================================
# ADVANCED OPTIONS
# =============================================================================
def render_advanced_options():
    """Render advanced prediction options"""
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_settings = render_confidence_settings()
        with col2:
            analysis_settings = render_analysis_settings()
        
        return {**confidence_settings, **analysis_settings}

def render_confidence_settings():
    """Render confidence-related settings"""
    return {
        'confidence_threshold': st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            help="Minimum confidence level for predictions"
        ),
        'include_uncertainty': st.checkbox(
            "Include Uncertainty Estimates",
            value=True,
            help="Show confidence intervals for predictions"
        )
    }

def render_analysis_settings():
    """Render analysis-related settings"""
    return {
        'analysis_mode': st.radio(
            "Analysis Mode",
            ["Standard", "Detailed", "Expert"],
            help="Select level of analysis detail"
        ),
        'compare_similar': st.checkbox(
            "Compare with Similar Medicines",
            value=True,
            help="Show comparison with similar medicines"
        )
    }

# =============================================================================
# PREDICTION PROCESSING
# =============================================================================
def process_prediction_request(advanced_options):
    """Process the prediction request"""
    with st.spinner("Analyzing medicine effectiveness..."):
        try:
            features = preprocess_inputs()
            prediction_results = get_prediction(features)
            
            if prediction_results:
                display_results(prediction_results, features, advanced_options)
                save_to_history(prediction_results)
                
        except Exception as e:
            handle_prediction_error(e)

def preprocess_inputs():
    """Preprocess input data for prediction"""
    medicine_details = st.session_state.medicine_details
    additional_info = st.session_state.additional_info
    
    # Add preprocessing logic here
    return {**medicine_details, **additional_info}

def get_prediction(features):
    """Get prediction from model"""
    try:
        validate_model()
        
        prediction = st.session_state.model.predict(features)[0]
        probabilities = st.session_state.model.predict_proba(features)[0]
        
        return {
            'prediction': prediction,
            'probabilities': probabilities
        }
    except Exception as e:
        handle_prediction_error(e)
        return None

# =============================================================================
# RESULTS DISPLAY
# =============================================================================
def display_results(results, features, options):
    """Display prediction results"""
    st.success("Prediction Complete!")
    
    display_prediction_metrics(results)
    
    if options['include_uncertainty']:
        display_uncertainty_analysis(results)
    
    if options['compare_similar']:
        display_similar_medicines(features, results)
    
    if options['analysis_mode'] == "Expert":
        display_expert_analysis(features, results)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def validate_model():
    """Validate model availability"""
    if 'model' not in st.session_state:
        raise ValueError("Model not loaded")

def handle_prediction_error(error):
    """Handle prediction errors"""
    st.error(f"Error during prediction: {str(error)}")
    st.exception(error)

def save_to_history(results):
    """Save prediction to history"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'prediction': results['prediction'],
        'confidence': max(results['probabilities'])
    })

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="Medicine Effectiveness Prediction",
        page_icon="💊",
        layout="wide"
    )
    enhance_prediction_interface()