# Create verify_setup.py
import importlib
import sys

def verify_imports():
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sklearn',
        'xgboost',
        'lightgbm',
        'catboost',
        'optuna',
        'scipy',
        'statsmodels',
        'seaborn',
        'matplotlib',
        'networkx',
        'altair',
        'requests',
        'joblib',
        'mlflow',
        'pytest',
        'dotenv'
    ]

    print("Verifying package installations...")
    all_success = True

    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} successfully imported")
        except ImportError as e:
            print(f"❌ Error importing {package}: {str(e)}")
            all_success = False

    return all_success

if __name__ == "__main__":
    success = verify_imports()
    if success:
        print("\nAll packages successfully verified! ✨")
        print("You can now run: streamlit run src/app.py")
    else:
        print("\n⚠️ Some packages failed to import. Please check the errors above.")
        sys.exit(1)