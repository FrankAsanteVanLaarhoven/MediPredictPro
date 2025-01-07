import streamlit as st
import json
from datetime import datetime

def health_check():
    """API endpoint for health checking"""
    return json.dumps({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })