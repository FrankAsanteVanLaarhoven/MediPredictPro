# setup.py

from setuptools import setup, find_packages

setup(
    name="MediPredictPro",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "streamlit",
        "nltk",
        "textblob",
        "networkx",
        "scipy",
        "fpdf",
        'kaggle',
        'watchdog',
        'python-dotenv'
    ],
    entry_points={
        "console_scripts": [
            "medipredictpro=src.main:main"
        ]
    }
)
