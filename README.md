# MediPredictPro 🏥

Developed by: Frank Asante Van Laarhoven

A comprehensive medical data analysis and prediction platform built with Streamlit.

## Features

- Interactive dashboard with real-time data visualization
- Advanced analytics with pandas profiling
- Price prediction using machine learning
- Data import from multiple sources (CSV, Kaggle)
- 3D visualization capabilities
- Automated file monitoring

## Setup

1. Clone the repository:
```bash
git clone https://github.com/FAVL/MediPredictPro.git
cd MediPredictPro
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run src/app.py
```

## Project Structure

```
MediPredictPro/
├── data/                  # Data files
├── src/                   # Source code
│   ├── components/        # UI components
│   ├── data/             # Data handling
│   └── app.py            # Main application
├── tests/                # Test files
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Contact

- Developer: Frank Asante Van Laarhoven
- GitHub: [@FAVL](https://github.com/FAVL)

## License

MIT License
