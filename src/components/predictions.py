import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

def load_model(model_name='medicprepro_gb_model.joblib'):
    """Load the specified model"""
    try:
        model_path = Path('models') / model_name
        if model_path.exists():
            return joblib.load(model_path)
        else:
            st.warning(f"Model file not found: {model_name}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_input(composition, side_effects, manufacturer, price):
    """Preprocess input data for prediction"""
    # Create feature vector
    features = {
        'Composition_Length': len(composition.split()),
        'Side_Effects_Count': len(side_effects.split()),
        'Price': float(price),
        'Manufacturer': manufacturer
    }
    return pd.DataFrame([features])

def predict_effectiveness(model, features):
    """Make prediction using the model"""
    try:
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def render_predictions():
    """Render predictions interface"""
    st.title("Medicine Effectiveness Predictions")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Could not load prediction model")
        return
    
    # Create prediction form
    with st.form("prediction_form"):
        st.subheader("Enter Medicine Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            composition = st.text_area(
                "Composition",
                help="Enter the medicine composition"
            )
            
            manufacturer = st.selectbox(
                "Manufacturer",
                options=sorted(st.session_state.data['Manufacturer'].unique()),
                help="Select the manufacturer"
            )
        
        with col2:
            side_effects = st.text_area(
                "Side Effects",
                help="Enter known side effects"
            )
            
            price = st.number_input(
                "Price",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                step=10.0,
                help="Enter medicine price"
            )
        
        submitted = st.form_submit_button("Predict Effectiveness")
        
        if submitted:
            if not composition or not side_effects:
                st.warning("Please fill in all fields")
                return
                
            # Preprocess input
            features = preprocess_input(composition, side_effects, manufacturer, price)
            
            # Make prediction
            prediction, probabilities = predict_effectiveness(model, features)
            
            if prediction is not None:
                # Display results
                st.subheader("Prediction Results")
                
                # Create columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Rating",
                        f"{prediction:.1f}%",
                        help="Predicted excellent review percentage"
                    )
                
                with col2:
                    confidence = max(probabilities) * 100
                    st.metric(
                        "Confidence",
                        f"{confidence:.1f}%",
                        help="Model confidence in prediction"
                    )
                
                with col3:
                    effectiveness = "High" if prediction > 75 else "Medium" if prediction > 50 else "Low"
                    st.metric(
                        "Effectiveness Level",
                        effectiveness,
                        help="Categorized effectiveness level"
                    )
                
                # Show detailed probabilities
                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame({
                    'Rating Range': ['0-25%', '25-50%', '50-75%', '75-100%'],
                    'Probability': probabilities
                })
                
                st.bar_chart(prob_df.set_index('Rating Range'))
                
                # Additional insights
                st.subheader("Analysis Insights")
                insights = []
                
                if features['Price'].iloc[0] > st.session_state.data['Price'].mean():
                    insights.append("• Price is above average for similar medicines")
                
                if features['Composition_Length'].iloc[0] > 5:
                    insights.append("• Complex composition may affect effectiveness")
                
                if features['Side_Effects_Count'].iloc[0] > 3:
                    insights.append("• Multiple side effects noted")
                
                for insight in insights:
                    st.write(insight)