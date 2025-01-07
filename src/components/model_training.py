# Add these helper functions to the existing file

def preprocess_dataset(numeric_features, categorical_features, target, 
                      scale=True, handle_missing=True, encode_categorical=True):
    """Preprocess the dataset based on selected options"""
    try:
        data = st.session_state.data.copy()
        
        # Select features
        features = numeric_features + categorical_features
        X = data[features]
        y = data[target]
        
        # Handle missing values
        if handle_missing:
            X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())
            X[categorical_features] = X[categorical_features].fillna('Unknown')
        
        # Scale numeric features
        if scale:
            scaler = StandardScaler()
            X[numeric_features] = scaler.fit_transform(X[numeric_features])
            st.session_state.scaler = scaler
        
        # Encode categorical variables
        if encode_categorical:
            X = pd.get_dummies(X, columns=categorical_features)
        
        return X, y
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None

def train_selected_model(model_type, params, X_train, X_val, y_train, y_val):
    """Train the selected model type with given parameters"""
    try:
        # Create model based on type
        if model_type == 'LightGBM':
            model = lgb.LGBMRegressor(**params)
        elif model_type == 'XGBoost':
            model = xgb.XGBRegressor(**params)
        else:  # CatBoost
            model = CatBoost(params)
        
        # Create validation dataset
        eval_set = [(X_val, y_val)]
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            verbose=False
        )
        
        return model
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None

def calculate_metrics(model, X_val, y_val):
    """Calculate model performance metrics"""
    try:
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    try:
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return None
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color='rgb(58, 149, 136)'
            )
        )
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(feature_names) * 20),
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")
        return None

def validate_model_performance(model, X_val, y_val):
    """Additional validation checks for model performance"""
    try:
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Create residual plot
        residuals = y_val - y_pred
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(color='rgb(58, 149, 136)'),
                name='Residuals'
            )
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error in model validation: {str(e)}")
        return None