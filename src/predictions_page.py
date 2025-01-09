import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px

class PredictionsPage:
    def __init__(self):
        self.title = "Medicine Price Predictions 🔮"
        self.pipeline = None
        self.feature_columns = ['Effectiveness', 'Stock_Level', 'Is_Generic']
        self.target_column = 'Price'

    def render(self, data: pd.DataFrame):
        st.title(self.title)

        if data is None or data.empty:
            st.info("Please load data first!")
            return

        # Add missing columns if needed
        data = self._ensure_required_columns(data)
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        tab1, tab2 = st.tabs(["Train Model", "Make Predictions"])

        with tab1:
            self._render_training_tab(data)

        with tab2:
            self._render_prediction_tab(data)

    def _ensure_required_columns(self, data):
        """Ensure all required columns exist in the dataset"""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
            
        required_columns = {
            'Price': lambda x: np.random.uniform(10, 1000, len(x)),
            'Effectiveness': lambda x: np.random.uniform(70, 100, len(x)),
            'Stock_Level': lambda x: np.random.randint(0, 1000, len(x)),
            'Is_Generic': lambda x: np.random.choice([True, False], len(x))
        }

        for col, default_func in required_columns.items():
            if col not in data.columns:
                data[col] = default_func(data)
                st.warning(f"Added missing column '{col}' with default values")

        # Convert Is_Generic to boolean if it's not already
        if 'Is_Generic' in data.columns:
            data['Is_Generic'] = data['Is_Generic'].astype(bool)

        return data

    def _render_training_tab(self, data):
        st.header("Model Training")

        # Display feature statistics
        st.subheader("Feature Statistics")
        st.dataframe(data[self.feature_columns + [self.target_column]].describe(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
            max_depth = st.slider("Max Tree Depth", 3, 20, 10)
        
        with col2:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                success = self._train_model(
                    data,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    test_size=test_size
                )
                if success:
                    st.success("Model trained successfully!")

    def _train_model(self, data, n_estimators=100, max_depth=10, 
                    min_samples_split=2, test_size=0.2):
        try:
            st.write("Starting model training...")
            
            # Verify data
            st.write("Data shape:", data.shape)
            st.write("Columns:", data.columns.tolist())
            
            # Prepare features and target
            X = data[self.feature_columns].copy()
            y = data[self.target_column].copy()
            
            st.write("Features shape:", X.shape)
            st.write("Target shape:", y.shape)

            # Create preprocessing pipeline
            numeric_features = ['Effectiveness', 'Stock_Level']
            categorical_features = ['Is_Generic']

            # Create the preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
                ],
                verbose_feature_names_out=False
            )

            # Create the model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )

            # Create the pipeline
            self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            st.write("Training model...")
            # Fit the pipeline
            self.pipeline.fit(X_train, y_train)

            # Calculate scores
            train_score = self.pipeline.score(X_train, y_train)
            test_score = self.pipeline.score(X_test, y_test)

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Score", f"{train_score:.2%}")
            with col2:
                st.metric("Testing Score", f"{test_score:.2%}")

            # Feature importance
            importance = self.pipeline.named_steps['regressor'].feature_importances_
            feature_names = (numeric_features + 
                           [f"{feat}_True" for feat in categorical_features])
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            fig = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title='Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)

            return True

        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            st.write("Full error details:")
            st.exception(e)
            return False

    def _render_prediction_tab(self, data):
        st.header("Make Predictions")

        if self.pipeline is None:
            st.warning("Please train the model first!")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            effectiveness = st.slider(
                "Effectiveness",
                float(data['Effectiveness'].min()),
                float(data['Effectiveness'].max()),
                float(data['Effectiveness'].mean())
            )

        with col2:
            stock_level = st.slider(
                "Stock Level",
                int(data['Stock_Level'].min()),
                int(data['Stock_Level'].max()),
                int(data['Stock_Level'].mean())
            )

        with col3:
            is_generic = st.checkbox("Is Generic")

        if st.button("Predict Price"):
            try:
                prediction_input = pd.DataFrame({
                    'Effectiveness': [effectiveness],
                    'Stock_Level': [stock_level],
                    'Is_Generic': [is_generic]
                })

                predicted_price = self.pipeline.predict(prediction_input)[0]
                st.success(f"Predicted Price: ${predicted_price:,.2f}")

                # Show similar products
                similar_products = data[
                    (data['Effectiveness'].between(effectiveness-5, effectiveness+5)) &
                    (data['Is_Generic'] == is_generic)
                ][['Price', 'Effectiveness', 'Stock_Level', 'Is_Generic']]

                if not similar_products.empty:
                    st.write("Similar Products:")
                    st.dataframe(similar_products, use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)