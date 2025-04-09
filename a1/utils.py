# Core data science libraries
import pandas as pd
import numpy as np
import streamlit as st
import datetime

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    classification_report, mean_absolute_error, mean_absolute_percentage_error,
    explained_variance_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.decomposition import PCA

# Set Matplotlib and Seaborn styles 
def set_plotting_style():
    """Set consistent plotting styles for the application"""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6)
    })
    # Custom color palette
    sns.set_palette("viridis")

# Data loading functions with caching
@st.cache_data
def load_dataset(dataset_name):
    """Load a dataset by name or from uploaded file"""
    try:
        if dataset_name != "Upload Custom CSV":
            data = sns.load_dataset(dataset_name)
            return data, None
        else:
            return None, "Please upload a CSV file"
    except Exception as e:
        return None, f"Failed to load {dataset_name} dataset: {str(e)}"

# Data preprocessing functions
def preprocess_data(X, y, preprocessing_config=None):
    """Preprocess features and target for modeling with customizable options
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataframe
    y : pandas.Series
        Target variable
    preprocessing_config : dict, optional
        Configuration for preprocessing steps
        
    Returns:
    --------
    X_processed : pandas.DataFrame
        Processed features
    y : pandas.Series
        Processed target variable
    """
    X_processed = X.copy()
    y_processed = y.copy()
    
    # If no config provided, use default preprocessing
    if preprocessing_config is None:
        # Handle categorical features with label encoding
        categorical_features = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_features:
            le = LabelEncoder()
            X_processed.loc[:, col] = le.fit_transform(X_processed[col].astype(str))
        
        # Simple imputation for missing values
        if X_processed.isnull().values.any():
            imputer = SimpleImputer(strategy='mean')
            X_processed = pd.DataFrame(imputer.fit_transform(X_processed), columns=X_processed.columns)
        
        return X_processed, y_processed
    
    # Get numerical and categorical columns
    numerical_features = X_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Handle Missing Values
    if preprocessing_config.get("missing_handling") == "Simple Imputation (mean/mode)":
        # Impute numerical features
        if numerical_features:
            num_imputer = SimpleImputer(strategy='mean')
            X_processed[numerical_features] = num_imputer.fit_transform(X_processed[numerical_features])
        
        # Impute categorical features
        if categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_processed[categorical_features] = cat_imputer.fit_transform(X_processed[categorical_features])
            
    elif preprocessing_config.get("missing_handling") == "Drop Rows with Missing Values":
        X_processed = X_processed.dropna()
        # Get indices of remaining rows to filter target accordingly
        remaining_indices = X_processed.index
        y_processed = y_processed.loc[remaining_indices]
    
    # 2. Process Categorical Features
    if categorical_features:
        if preprocessing_config.get("cat_preprocessing") == "One-Hot Encoding":
            # Create OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            # Fit and transform
            encoded_features = encoder.fit_transform(X_processed[categorical_features])
            # Create dataframe with encoded feature names
            feature_names = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X_processed.index)
            
            # Drop original categorical columns and add encoded ones
            X_processed = X_processed.drop(columns=categorical_features)
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
            
        elif preprocessing_config.get("cat_preprocessing") == "Label Encoding":
            for col in categorical_features:
                le = LabelEncoder()
                X_processed.loc[:, col] = le.fit_transform(X_processed[col].astype(str))
                
        elif preprocessing_config.get("cat_preprocessing") == "Target Encoding":
            # Simple target encoding (mean target value per category)
            for col in categorical_features:
                # Calculate target mean for each category
                target_means = y_processed.groupby(X_processed[col]).mean()
                # Replace categories with their target means
                X_processed.loc[:, col] = X_processed[col].map(target_means)
                # Handle unseen categories
                X_processed.loc[:, col] = X_processed[col].fillna(y_processed.mean())
    
    # Update numerical features after categorical processing
    numerical_features = X_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
    # 3. Process Numerical Features
    if numerical_features:
        # Handle outliers separately if specified
        if preprocessing_config.get("remove_outliers", False):
            # Use IQR method to identify outliers
            for col in numerical_features:
                Q1 = X_processed[col].quantile(0.25)
                Q3 = X_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Filter out rows with outliers
                outlier_mask = (X_processed[col] >= lower_bound) & (X_processed[col] <= upper_bound)
                X_processed = X_processed[outlier_mask]
                # Update target as well
                y_processed = y_processed[outlier_mask]
        
        # Apply log transformation if specified
        if preprocessing_config.get("log_transform", False):
            for col in numerical_features:
                # Only apply to positive columns
                if (X_processed[col] > 0).all():
                    X_processed[col] = np.log1p(X_processed[col])
        
        # Apply scaling if specified (only one scaling method at a time)
        scaling_methods = {
            "Standardization (z-score)": StandardScaler(),
            "Min-Max Scaling": MinMaxScaler(feature_range=(0, 1)),
            "Robust Scaling": RobustScaler()
        }
        
        # Get the single scaling option from num_preprocessing
        num_preprocessings = preprocessing_config.get("num_preprocessing", [])
        if num_preprocessings:
            scaling_type = num_preprocessings[0]  # Get the first (and only) scaling option
            if scaling_type in scaling_methods:
                scaler = scaling_methods[scaling_type]
                X_processed[numerical_features] = scaler.fit_transform(X_processed[numerical_features])
    
    return X_processed, y_processed

# Model training and evaluation functions
def train_model(X, y, model_name, model_params, test_size, random_state, use_grid_search=False, grid_search_params=None, preprocessing_config=None):
    """Train a model and return results"""
    # Preprocess data with custom config if provided
    X_processed, y_processed = preprocess_data(X, y, preprocessing_config)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=test_size, random_state=random_state
    )
    
    # Scale features for models that benefit from scaling
    # Skip if scaling was already applied in preprocessing
    skip_scaling = preprocessing_config and any(
        scaling in preprocessing_config.get("num_preprocessing", []) 
        for scaling in ["Standardization (z-score)", "Min-Max Scaling", "Robust Scaling"]
    )
    
    scale_models = ["Support Vector Machine", "KNN", "Ridge", "Lasso", "SVC", "SVR"]
    needs_scaling = any(model in model_name for model in scale_models) and not skip_scaling
    
    if needs_scaling:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Initialize the model based on selection
    if model_name == "Linear Regression":
        model = LinearRegression()
        target_type = "regression"
    elif model_name == "Ridge Regression":
        model = Ridge(
            alpha=model_params.get('alpha', 1.0),
            random_state=random_state
        )
        target_type = "regression"
    elif model_name == "Lasso Regression":
        model = Lasso(
            alpha=model_params.get('alpha', 1.0),
            random_state=random_state
        )
        target_type = "regression"
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            random_state=random_state
        )
        target_type = "regression"
    elif model_name == "Extra Trees Regressor":
        model = ExtraTreesRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            random_state=random_state
        )
        target_type = "regression"
    elif model_name == "KNN Regressor":
        model = KNeighborsRegressor(
            n_neighbors=model_params.get('n_neighbors', 5),
            weights=model_params.get('weights', 'uniform')
        )
        target_type = "regression"
    elif model_name == "SVR":
        model = SVR(
            C=model_params.get('C', 1.0),
            kernel=model_params.get('kernel', 'rbf'),
            gamma=model_params.get('gamma', 'scale')
        )
        target_type = "regression"
    elif model_name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 3),
            learning_rate=model_params.get('learning_rate', 0.1),
            random_state=random_state
        )
        target_type = "regression"
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            random_state=random_state
        )
        target_type = "classification"
    elif model_name == "Extra Trees Classifier":
        model = ExtraTreesClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 10),
            random_state=random_state
        )
        target_type = "classification"
    elif model_name == "KNN Classifier":
        model = KNeighborsClassifier(
            n_neighbors=model_params.get('n_neighbors', 5),
            weights=model_params.get('weights', 'uniform')
        )
        target_type = "classification"
    elif model_name == "SVC":
        model = SVC(
            C=model_params.get('C', 1.0),
            kernel=model_params.get('kernel', 'rbf'),
            gamma=model_params.get('gamma', 'scale'),
            probability=True,
            random_state=random_state
        )
        target_type = "classification"
    elif model_name == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 3),
            learning_rate=model_params.get('learning_rate', 0.1),
            random_state=random_state
        )
        target_type = "classification"
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            C=model_params.get('C', 1.0),
            max_iter=model_params.get('max_iter', 1000),
            random_state=random_state
        )
        target_type = "classification"
    
    # Perform grid search if requested
    best_params = None
    if use_grid_search and grid_search_params:
        grid_search = GridSearchCV(
            model,
            grid_search_params,
            cv=5,
            scoring='accuracy' if target_type == "classification" else 'r2',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        # Train the model
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics based on model type
    results = {}
    
    if target_type == "regression":
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            "mse": mse,
            "r2": r2
        }
    else:
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        
        # Calculate additional classification metrics
        # For multiclass, we use weighted average
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Get class names for classification
        if hasattr(y, 'cat'):
            class_names = y.cat.categories.tolist()
        else:
            class_names = sorted(y.unique().tolist())
        
        # Store prediction probabilities for ROC curve if applicable
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_mat,
            "class_names": class_names,
            "y_test": y_test,
            "y_pred_proba": y_pred_proba
        }
    
    # Extract feature importance if available
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = {
            "features": X.columns.tolist(),
            "importance": model.feature_importances_
        }
    elif hasattr(model, "coef_"):
        if len(model.coef_.shape) == 1:
            feature_importance = {
                "features": X.columns.tolist(),
                "importance": np.abs(model.coef_)
            }
        else:
            feature_importance = {
                "features": X.columns.tolist(),
                "importance": np.mean(np.abs(model.coef_), axis=0)
            }
    
    return {
        "model": model,
        "model_type": model_name,
        "target_type": target_type,
        "results": results,
        "feature_importance": feature_importance,
        "test_data": (X_test, y_test),
        "predictions": y_pred,
        "best_params": best_params
    }

# Visualization functions
def plot_regression_results(y_test, y_pred):
    """Create a scatter plot of predicted vs actual values for regression"""
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    
    # Add a perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual Values")
    return fig

def plot_confusion_matrix(conf_matrix, class_names):
    """Plot a confusion matrix for classification results"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    return fig

def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve for binary classification"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    return fig

def plot_feature_importance(features, importance):
    """Plot feature importance"""
    # Create a DataFrame for the feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, len(features) * 0.5))
    bar_plot = sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    # Add value labels to bars
    for i, v in enumerate(importance_df['Importance']):
        bar_plot.text(v + 0.01, i, f"{v:.3f}", va='center')
    ax.set_title('Feature Importance')
    return fig

# Display functions that use native Streamlit components instead of custom CSS
def display_model_config(model_type, target, features, test_size):
    """Display model configuration using Streamlit components"""
    with st.container():
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Model:** {model_type}")
            st.markdown(f"**Target:** {target}")
        
        with col2:
            st.markdown(f"**Test Size:** {test_size * 100:.0f}% of data")
        
        st.markdown("**Features:**")
        # Display features as pills/tags using Streamlit's beta_columns
        feature_cols = st.columns(4)
        for i, feature in enumerate(features):
            with feature_cols[i % 4]:
                st.markdown(f"<div style='background-color: {st.get_option('theme.secondaryBackgroundColor')}; "
                            f"padding: 0.3rem 0.6rem; border-radius: 15px; margin-bottom: 0.5rem; "
                            f"display: inline-block; font-size: 0.9rem;'>{feature}</div>", 
                            unsafe_allow_html=True)

def display_training_phase(phase, description=None):
    """Display the current training phase using Streamlit components"""
    phases = {
        "data_prep": {"icon": "📊", "title": "Data Preparation", "color": "blue"},
        "preprocessing": {"icon": "🧹", "title": "Data Preprocessing", "color": "blue"},
        "splitting": {"icon": "✂️", "title": "Train-Test Splitting", "color": "violet"},
        "training": {"icon": "⚙️", "title": "Model Training", "color": "red"},
        "evaluation": {"icon": "📈", "title": "Model Evaluation", "color": "green"},
        "complete": {"icon": "✅", "title": "Training Complete", "color": "green"}
    }
    
    if phase not in phases:
        return
    
    info = phases[phase]
    
    # Create a container with the phase information
    with st.container():
        cols = st.columns([1, 10])
        with cols[0]:
            st.markdown(f"<h1 style='font-size: 1.8rem; margin: 0; text-align: center;'>{info['icon']}</h1>", 
                       unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"**{info['title']}**")
            if description:
                st.markdown(f"<span style='color: gray; font-size: 0.9rem;'>{description}</span>", 
                           unsafe_allow_html=True)

# Experiment Management Functions
def save_experiment(experiment_data):
    """Save an experiment to session state."""
    st.session_state.experiment_history = st.session_state.get("experiment_history", [])
    st.session_state.experiment_history.append(experiment_data)

def create_experiment_data(name, notes, model_type, target_type, model_info=True, metrics=True, 
                         feature_info=True, predictions=True, dataset_info=True, **kwargs):
    """Create a standardized experiment data dictionary."""
    # Base experiment data
    experiment_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "name": name,
        "notes": notes,
        "model_type": model_type,
        "target_type": target_type
    }
    
    # Optional sections based on flags
    if model_info:
        experiment_data.update({
            "grid_search": kwargs.get("use_grid_search", False),
            "best_params": kwargs.get("best_params") if kwargs.get("use_grid_search") else kwargs.get("model_params", {})
        })
    
    if metrics and "model_results" in kwargs:
        experiment_data["model_results"] = kwargs["model_results"]
    
    if feature_info:
        experiment_data.update({
            "selected_features": kwargs.get("selected_features", []),
            "feature_importance": kwargs.get("feature_importance")
        })
    
    if predictions and "model_predictions" in kwargs:
        predictions_data = kwargs["model_predictions"]
        experiment_data["predictions"] = (
            predictions_data.tolist() 
            if isinstance(predictions_data, np.ndarray) 
            else predictions_data
        )
        experiment_data["test_data"] = kwargs.get("test_data")
    
    if dataset_info:
        experiment_data.update({
            "dataset_name": kwargs.get("dataset_name"),
            "target_variable": kwargs.get("target_variable")
        })
    
    return experiment_data

def get_sorted_experiments():
    """Get experiments sorted by timestamp (newest first)."""
    return sorted(
        st.session_state.get("experiment_history", []),
        key=lambda x: x["timestamp"],
        reverse=True
    )

def create_experiment_comparison_data(experiments, selected_names):
    """Create comparison data for selected experiments."""
    # Define metric formatters for each model type
    metric_formatters = {
        "regression": {
            "R²": lambda x: f"{x['r2']:.4f}",
            "MSE": lambda x: f"{x['mse']:.4f}",
            "RMSE": lambda x: f"{np.sqrt(float(x['mse'])):.4f}"
        },
        "classification": {
            "Accuracy": lambda x: f"{x['accuracy']:.4f}",
            "Precision": lambda x: f"{x['precision']:.4f}",
            "Recall": lambda x: f"{x['recall']:.4f}",
            "F1": lambda x: f"{x['f1']:.4f}"
        }
    }
    
    comparison_data = []
    for exp_name in selected_names:
        exp = next(exp for exp in experiments if exp["name"] == exp_name)
        
        # Basic experiment info
        exp_data = {
            "Experiment": exp_name,
            "Model": exp.get("model_type", "N/A"),
            "Dataset": exp.get("dataset_name", "N/A"),
            "Target": exp.get("target_variable", "N/A"),
            "Features": len(exp.get("selected_features", [])),
            "Grid Search": "Yes" if exp.get("grid_search", False) else "No"
        }
        
        # Add metrics based on model type
        if "model_results" in exp:
            metrics = metric_formatters.get(exp.get("target_type", ""))
            if metrics:
                exp_data.update({
                    metric: formatter(exp["model_results"])
                    for metric, formatter in metrics.items()
                })
        
        comparison_data.append(exp_data)
    
    return comparison_data

def plot_metrics_comparison(comparison_data, plot_type="bar"):
    """Create comparison plots for experiment metrics."""
    if not comparison_data:
        return None
        
    # Get metrics columns (excluding non-metric columns)
    non_metrics = {"Experiment", "Model", "Dataset", "Target", "Features", "Grid Search"}
    metrics = [col for col in comparison_data[0].keys() if col not in non_metrics]
    
    if not metrics:
        return None
    
    # Create a figure for each metric
    figures = []
    experiments = [d["Experiment"] for d in comparison_data]
    
    for metric in metrics:
        try:
            # Safely convert values to float, skipping any that can't be converted
            values = []
            for d in comparison_data:
                try:
                    value = float(d.get(metric, 0))
                    values.append(value)
                except (ValueError, TypeError):
                    continue
            
            if values:  # Only create plot if we have valid values
                if plot_type == "bar":
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=experiments,
                        y=values,
                        text=[f"{v:.4f}" for v in values],
                        textposition="auto"
                    ))
                    fig.update_layout(
                        title=f"{metric} Comparison",
                        xaxis_title="Experiments",
                        yaxis_title=metric,
                        showlegend=False
                    )
                else:  # radar plot
                    fig = go.Figure()
                    values.append(values[0])  # Close the polygon
                    metrics_plot = [metric] * len(values)
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics_plot,
                        fill="toself"
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(values) * 1.1]
                            )),
                        title=f"{metric} Comparison",
                        showlegend=False
                    )
                
                figures.append(fig)
        except Exception as e:
            print(f"Error processing metric {metric}: {str(e)}")
            continue
    
    return figures

def get_experiment_config_display(experiment):
    """Format configuration data for display."""
    config = {
        "Basic Info": {
            "Model Type": experiment.get("model_type", "N/A"),
            "Target Type": experiment.get("target_type", "N/A"),
            "Dataset": experiment.get("dataset_name", "N/A"),
            "Target Variable": experiment.get("target_variable", "N/A"),
            "Created": datetime.datetime.fromisoformat(experiment["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    if experiment.get("grid_search"):
        config["Grid Search"] = {
            "Best Parameters": experiment.get("best_params", {})
        }
    else:
        config["Model Parameters"] = experiment.get("best_params", {})
    
    if "selected_features" in experiment:
        config["Features"] = {
            "Count": len(experiment["selected_features"]),
            "Selected": experiment["selected_features"]
        }
    
    if "feature_importance" in experiment:
        importance = experiment["feature_importance"]
        if isinstance(importance, dict):
            # Convert values to float and handle any non-numeric values
            formatted_importance = {}
            for k, v in importance.items():
                try:
                    # Try to convert to float and format
                    formatted_importance[k] = f"{float(v):.4f}"
                except (ValueError, TypeError):
                    # If conversion fails, keep original value
                    formatted_importance[k] = str(v)
            
            # Sort by numeric value if possible, otherwise by key
            try:
                sorted_items = sorted(
                    formatted_importance.items(),
                    key=lambda x: float(x[1]) if x[1].replace('.', '').isdigit() else float('-inf'),
                    reverse=True
                )
            except (ValueError, TypeError):
                # If sorting fails, sort by key
                sorted_items = sorted(formatted_importance.items())
            
            config["Feature Importance"] = dict(sorted_items)
    
    return config

# Model Factory Function
def create_model(model_name, model_params, random_state):
    """Factory function to create model instances with consistent parameters."""
    model_map = {
        "Linear Regression": (LinearRegression, {}),
        "Ridge Regression": (Ridge, {"alpha": model_params.get('alpha', 1.0)}),
        "Lasso Regression": (Lasso, {"alpha": model_params.get('alpha', 1.0)}),
        "Random Forest Regressor": (RandomForestRegressor, {
            "n_estimators": model_params.get('n_estimators', 100),
            "max_depth": model_params.get('max_depth', 10)
        }),
        "Random Forest Classifier": (RandomForestClassifier, {
            "n_estimators": model_params.get('n_estimators', 100),
            "max_depth": model_params.get('max_depth', 10)
        }),
        # Add more model mappings here
    }
    
    model_class, default_params = model_map.get(model_name, (None, {}))
    if model_class:
        params = {**default_params, "random_state": random_state}
        return model_class(**params)
    return None

# Enhanced Experiment Analysis
def analyze_experiment_trends(experiments):
    """Analyze trends across multiple experiments."""
    trends = {
        "model_performance": {},
        "feature_importance": {},
        "parameter_impact": {}
    }
    
    # Analyze performance trends
    for exp in experiments:
        if "model_results" in exp:
            model_type = exp.get("model_type", "Unknown")
            if model_type not in trends["model_performance"]:
                trends["model_performance"][model_type] = []
            trends["model_performance"][model_type].append(exp["model_results"])
    
    # Analyze feature importance trends
    for exp in experiments:
        if "feature_importance" in exp:
            for feature, importance in exp["feature_importance"].items():
                if feature not in trends["feature_importance"]:
                    trends["feature_importance"][feature] = []
                trends["feature_importance"][feature].append(importance)
    
    return trends

# Enhanced Visualization Functions
def plot_experiment_trends(trends):
    """Create interactive plots for experiment trends."""
    fig = go.Figure()
    
    # Plot performance trends
    for model_type, performances in trends["model_performance"].items():
        if "accuracy" in performances[0]:  # Classification
            values = [p["accuracy"] for p in performances]
            fig.add_trace(go.Scatter(
                y=values,
                name=f"{model_type} Accuracy",
                mode="lines+markers"
            ))
        elif "r2" in performances[0]:  # Regression
            values = [p["r2"] for p in performances]
            fig.add_trace(go.Scatter(
                y=values,
                name=f"{model_type} R²",
                mode="lines+markers"
            ))
    
    fig.update_layout(
        title="Model Performance Trends",
        xaxis_title="Experiment Number",
        yaxis_title="Performance Metric",
        showlegend=True
    )
    return fig

def plot_feature_importance_trends(trends):
    """Create interactive plot for feature importance trends."""
    fig = go.Figure()
    
    for feature, importances in trends["feature_importance"].items():
        fig.add_trace(go.Scatter(
            y=importances,
            name=feature,
            mode="lines+markers"
        ))
    
    fig.update_layout(
        title="Feature Importance Trends",
        xaxis_title="Experiment Number",
        yaxis_title="Importance Score",
        showlegend=True
    )
    return fig

# Enhanced Experiment Comparison
def compare_experiment_parameters(experiments):
    """Compare model parameters across experiments."""
    param_comparison = {}
    
    for exp in experiments:
        model_type = exp.get("model_type", "Unknown")
        if model_type not in param_comparison:
            param_comparison[model_type] = []
        
        params = exp.get("best_params", {})
        param_comparison[model_type].append({
            "experiment": exp.get("name", "Unknown"),
            "parameters": params
        })
    
    return param_comparison

# Enhanced Data Analysis
def analyze_dataset_characteristics(data):
    """Analyze and summarize dataset characteristics."""
    analysis = {
        "basic_stats": {
            "rows": len(data),
            "columns": len(data.columns),
            "missing_values": data.isnull().sum().sum(),
            "duplicates": data.duplicated().sum()
        },
        "column_types": {
            "numeric": data.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical": data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime": data.select_dtypes(include=['datetime64']).columns.tolist()
        },
        "correlation_analysis": data.corr() if len(data.select_dtypes(include=['int64', 'float64']).columns) > 1 else None
    }
    return analysis

# Enhanced Model Evaluation
def evaluate_model_performance(y_true, y_pred, model_type):
    """Comprehensive model evaluation with additional metrics."""
    metrics = {}
    
    if model_type == "classification":
        metrics.update({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1": f1_score(y_true, y_pred, average='weighted'),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        })
    else:  # regression
        metrics.update({
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred)
        })
    
    return metrics 