import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def enhance_prediction_interface():
    """Enhanced prediction interface with additional features"""
    st.title("MediPredictPro - Medicine Effectiveness Prediction")
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # Left column - Basic Details
        with col1:
            medicine_details = get_medicine_details()
        
        # Right column - Additional Info
        with col2:
            additional_info = get_additional_info()
        
        # Advanced Options
        advanced_options = get_advanced_options()
        
        # Submit button
        submitted = st.form_submit_button("Predict Effectiveness", type="primary")
        
        if submitted:
            process_prediction(medicine_details, additional_info, advanced_options)

def get_medicine_details():
    """Get basic medicine details"""
    st.subheader("Medicine Details")
    
    return {
        'name': st.text_input(
            "Medicine Name",
            help="Enter the name of the medicine"
        ),
        'composition': st.text_area(
            "Composition",
            help="Enter the medicine composition"
        ),
        'manufacturer': st.selectbox(
            "Manufacturer",
            options=sorted(st.session_state.data['Manufacturer'].unique()),
            help="Select the manufacturer"
        ),
        'price': st.number_input(
            "Price",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Enter medicine price"
        ),
        'dosage_form': st.selectbox(
            "Dosage Form",
            options=['Tablet', 'Capsule', 'Injection', 'Syrup', 'Other'],
            help="Select the dosage form"
        )
    }

def get_additional_info():
    """Get additional medicine information"""
    st.subheader("Additional Information")
    
    return {
        'side_effects': st.text_area(
            "Side Effects",
            help="Enter known side effects"
        ),
        'usage_complexity': st.slider(
            "Usage Complexity",
            min_value=1,
            max_value=10,
            value=5,
            help="Rate the complexity of usage from 1 to 10"
        ),
        'treatment_duration': st.number_input(
            "Treatment Duration (days)",
            min_value=1,
            max_value=365,
            value=30,
            help="Expected duration of treatment"
        ),
        'storage_temp': st.slider(
            "Storage Temperature (°C)",
            min_value=2,
            max_value=30,
            value=25,
            help="Recommended storage temperature"
        )
    }

def get_advanced_options():
    """Get advanced prediction options"""
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                help="Minimum confidence level for predictions"
            )
            
            include_uncertainty = st.checkbox(
                "Include Uncertainty Estimates",
                value=True,
                help="Show confidence intervals for predictions"
            )
        
        with col2:
            analysis_mode = st.radio(
                "Analysis Mode",
                ["Standard", "Detailed", "Expert"],
                help="Select level of analysis detail"
            )
            
            compare_similar = st.checkbox(
                "Compare with Similar Medicines",
                value=True,
                help="Show comparison with similar medicines"
            )
        
        return {
            'confidence_threshold': confidence_threshold,
            'include_uncertainty': include_uncertainty,
            'analysis_mode': analysis_mode,
            'compare_similar': compare_similar
        }

def process_prediction(medicine_details, additional_info, advanced_options):
    """Process the prediction request"""
    with st.spinner("Analyzing medicine effectiveness..."):
        try:
            # Preprocess inputs
            features = preprocess_input(medicine_details, additional_info)
            
            # Get prediction
            prediction, probabilities = predict_effectiveness(features)
            
            if prediction is not None:
                display_prediction_results(
                    prediction,
                    probabilities,
                    features,
                    advanced_options
                )
                
                # Save prediction to history
                save_prediction_history(
                    medicine_details['name'],
                    prediction,
                    probabilities
                )
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

def predict_effectiveness(features):
    """Make effectiveness prediction"""
    try:
        if 'model' not in st.session_state:
            raise ValueError("Model not loaded")
            
        prediction = st.session_state.model.predict(features)[0]
        probabilities = st.session_state.model.predict_proba(features)[0]
        
        return prediction, probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def display_prediction_results(prediction, probabilities, features, options):
    """Display prediction results with visualizations"""
    st.success("Prediction Complete!")
    
    # Display metrics
    display_metrics(prediction, probabilities)
    
    # Show uncertainty analysis if requested
    if options['include_uncertainty']:
        display_uncertainty_analysis(prediction, probabilities)
    
    # Show similar medicines if requested
    if options['compare_similar']:
        display_similar_medicines(features, prediction, options['analysis_mode'])
    
    # Show feature importance for expert mode
    if options['analysis_mode'] == "Expert":
        display_feature_importance(features)

def save_prediction_history(medicine_name, prediction, probabilities):
    """Save prediction to history"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'medicine_name': medicine_name,
        'prediction': prediction,
        'confidence': max(probabilities)
    })

# Add other helper functions (display_metrics, display_uncertainty_analysis, etc.)