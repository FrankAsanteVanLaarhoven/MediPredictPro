# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import plotly.express as px
from typing import Tuple, Optional, Dict, List

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================
def load_model(model_name: str = 'medicprepro_gb_model.joblib') -> Optional[object]:
    """Load the prediction model"""
    try:
        model_path = validate_model_path(model_name)
        return joblib.load(model_path)
    except Exception as e:
        handle_model_error(e)
        return None

def validate_model_path(model_name: str) -> Path:
    """Validate model file path"""
    model_path = Path('models') / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_name}")
    return model_path

def handle_model_error(error: Exception) -> None:
    """Handle model loading errors"""
    if isinstance(error, FileNotFoundError):
        st.warning(str(error))
    else:
        st.error(f"Error loading model: {str(error)}")

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
def preprocess_input(medicine_details: Dict) -> pd.DataFrame:
    """Preprocess input data for prediction"""
    return pd.DataFrame([{
        'Composition_Length': len(medicine_details['composition'].split()),
        'Side_Effects_Count': len(medicine_details['side_effects'].split()),
        'Price': float(medicine_details['price']),
        'Manufacturer': medicine_details['manufacturer']
    }])

def predict_effectiveness(model: object, 
                        features: pd.DataFrame) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Make prediction using the model"""
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        return prediction, probabilities
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# =============================================================================
# PREDICTION INTERFACE
# =============================================================================
def render_predictions() -> None:
    """Render main prediction interface"""
    st.title("Medicine Effectiveness Predictions")
    
    model = load_model()
    if model is None:
        return
    
    render_prediction_form(model)

def render_prediction_form(model: object) -> None:
    """Render prediction input form"""
    with st.form("prediction_form"):
        st.subheader("Enter Medicine Details")
        
        medicine_details = get_medicine_details()
        submitted = st.form_submit_button("Predict Effectiveness")
        
        if submitted:
            process_prediction(model, medicine_details)

def get_medicine_details() -> Dict:
    """Get medicine details from user input"""
    col1, col2 = st.columns(2)
    
    with col1:
        composition = render_composition_input()
        manufacturer = render_manufacturer_input()
    
    with col2:
        side_effects = render_side_effects_input()
        price = render_price_input()
    
    return {
        'composition': composition,
        'side_effects': side_effects,
        'manufacturer': manufacturer,
        'price': price
    }

# =============================================================================
# INPUT COMPONENTS
# =============================================================================
def render_composition_input() -> str:
    """Render composition input field"""
    return st.text_area(
        "Composition",
        help="Enter the medicine composition"
    )

def render_manufacturer_input() -> str:
    """Render manufacturer selection"""
    return st.selectbox(
        "Manufacturer",
        options=get_manufacturer_options(),
        help="Select the manufacturer"
    )

def render_side_effects_input() -> str:
    """Render side effects input field"""
    return st.text_area(
        "Side Effects",
        help="Enter known side effects"
    )

def render_price_input() -> float:
    """Render price input field"""
    return st.number_input(
        "Price",
        min_value=0.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
        help="Enter medicine price"
    )

# =============================================================================
# PREDICTION PROCESSING
# =============================================================================
def process_prediction(model: object, medicine_details: Dict) -> None:
    """Process prediction request"""
    if not validate_inputs(medicine_details):
        return
    
    features = preprocess_input(medicine_details)
    prediction, probabilities = predict_effectiveness(model, features)
    
    if prediction is not None:
        display_prediction_results(prediction, probabilities, features)

def validate_inputs(medicine_details: Dict) -> bool:
    """Validate user inputs"""
    if not medicine_details['composition'] or not medicine_details['side_effects']:
        st.warning("Please fill in all required fields")
        return False
    return True

# =============================================================================
# RESULTS DISPLAY
# =============================================================================
def display_prediction_results(prediction: float, 
                             probabilities: np.ndarray, 
                             features: pd.DataFrame) -> None:
    """Display prediction results"""
    st.subheader("Prediction Results")
    
    display_metrics(prediction, probabilities)
    display_probability_distribution(probabilities)
    display_insights(features)

def display_metrics(prediction: float, probabilities: np.ndarray) -> None:
    """Display prediction metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_rating_metric(prediction)
    with col2:
        display_confidence_metric(probabilities)
    with col3:
        display_effectiveness_metric(prediction)

def display_probability_distribution(probabilities: np.ndarray) -> None:
    """Display probability distribution chart"""
    st.subheader("Probability Distribution")
    
    prob_df = create_probability_dataframe(probabilities)
    fig = create_probability_chart(prob_df)
    st.plotly_chart(fig, use_container_width=True)

def display_insights(features: pd.DataFrame) -> None:
    """Display analysis insights"""
    st.subheader("Analysis Insights")
    insights = generate_insights(features)
    
    for insight in insights:
        st.write(insight)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_manufacturer_options() -> List[str]:
    """Get sorted list of manufacturers"""
    return sorted(st.session_state.data['Manufacturer'].unique())

def create_probability_dataframe(probabilities: np.ndarray) -> pd.DataFrame:
    """Create probability distribution dataframe"""
    return pd.DataFrame({
        'Rating Range': ['0-25%', '25-50%', '50-75%', '75-100%'],
        'Probability': probabilities
    })

def create_probability_chart(prob_df: pd.DataFrame) -> px.Figure:
    """Create probability distribution chart"""
    return px.bar(
        prob_df,
        x='Rating Range',
        y='Probability',
        title="Probability Distribution"
    )

def generate_insights(features: pd.DataFrame) -> List[str]:
    """Generate analysis insights"""
    insights = []
    
    if features['Price'].iloc[0] > st.session_state.data['Price'].mean():
        insights.append("• Price is above average for similar medicines")
    
    if features['Composition_Length'].iloc[0] > 5:
        insights.append("• Complex composition may affect effectiveness")
    
    if features['Side_Effects_Count'].iloc[0] > 3:
        insights.append("• Multiple side effects noted")
    
    return insights

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="Medicine Effectiveness Prediction",
        page_icon="💊",
        layout="wide"
    )
    render_predictions()