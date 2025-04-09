"""
Machine Learning Model Trainer Application

This Streamlit application allows users to train and evaluate machine learning models
on different datasets. It supports both classification and regression tasks, with 
a variety of preprocessing options and model configurations.

Features:
- Dataset selection and custom CSV upload
- Automatic detection of classification/regression tasks
- Exploratory data analysis
- Feature selection and preprocessing
- Multiple model types with configurable parameters
- Grid search for hyperparameter tuning
- Performance metrics and visualizations
- Experiment history and comparison

"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from utils import (
    set_plotting_style, 
    load_dataset, 
    preprocess_data, 
    train_model, 
    plot_regression_results, 
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_feature_importance,
    display_model_config,
    display_training_phase,
    save_experiment,
    create_experiment_data,
    get_sorted_experiments,
    create_experiment_comparison_data,
    plot_metrics_comparison,
    get_experiment_config_display
)

# Only import these when needed for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json

# Set plot styles
set_plotting_style()

# Define all utility functions first

def load_dataset_with_error_handling(dataset_option):
    """
    Loads a dataset based on user selection with proper error handling
    
    Parameters:
    -----------
    dataset_option : str
        The name of the dataset to load or 'Upload Custom CSV'
        
    Returns:
    --------
    pandas.DataFrame or None
        The loaded dataset or None if loading failed
    """
    try:
        if dataset_option == "Upload Custom CSV":
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    return data
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    return None
            else:
                return None
        else:
            # Load selected dataset
            data, error_message = load_dataset(dataset_option)
            if error_message:
                st.error(error_message)
                return None
            return data
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the dataset: {str(e)}")
        return None

def validate_grid_search_params(grid_search_params):
    """
    Validates that grid search parameters have at least one value selected for each parameter
    
    Parameters:
    -----------
    grid_search_params : dict
        Dictionary of grid search parameters
        
    Returns:
    --------
    bool
        True if all parameters have at least one value, False otherwise
    """
    if not grid_search_params:
        return False
        
    for param, values in grid_search_params.items():
        if not values:  # Empty list
            return False
    
    return True

def display_regression_metrics(model_results):
    """
    Display regression metrics in a formatted way
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing regression metrics
    """
    st.subheader("Regression Metrics")
    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Mean Squared Error", f"{model_results['mse']:.4f}")
    metrics_cols[1].metric("R² Score", f"{model_results['r2']:.4f}")
    # Fix RMSE calculation
    mse_value = float(model_results['mse'])  # Ensure MSE is a float
    metrics_cols[2].metric("RMSE", f"{np.sqrt(mse_value):.4f}")

def display_classification_metrics(model_results):
    """
    Display classification metrics in a formatted way
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing classification metrics
    """
    st.subheader("Classification Metrics")
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Accuracy", f"{model_results['accuracy']:.4f}")
    metrics_cols[1].metric("Precision", f"{model_results['precision']:.4f}")
    metrics_cols[2].metric("Recall", f"{model_results['recall']:.4f}")
    metrics_cols[3].metric("F1 Score", f"{model_results['f1']:.4f}")

def configure_tree_ensemble_parameters(use_grid_search, model_params, grid_search_params):
    """
    Configure parameters for Random Forest and Extra Trees models.
    
    Parameters:
    -----------
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    model_params : dict
        Dictionary to store model parameters if not using grid search
    grid_search_params : dict
        Dictionary to store grid search parameters if using grid search
    """
    if not use_grid_search:
        model_params['n_estimators'] = st.number_input(
            "Number of Trees", 
            min_value=10, 
            max_value=500, 
            value=100, 
            step=10,
            help="Number of decision trees in the forest. More trees generally improve performance but increase training time and memory usage."
        )
        model_params['max_depth'] = st.number_input(
            "Maximum Depth", 
            min_value=1, 
            max_value=50, 
            value=10,
            help="Maximum depth of each tree. Deeper trees can model more complex patterns but may overfit. Use None for unlimited depth."
        )
    else:
        st.markdown("**Grid Search Parameters**")
        st.markdown("**Number of Trees:**")
        grid_search_params['n_estimators'] = st.multiselect(
            "Select values",
            [50, 100, 200, 300, 500],
            default=[50, 100, 200],
            help="Different numbers of trees to test. The grid search will find the optimal value."
        )
        st.markdown("**Maximum Depth:**")
        grid_search_params['max_depth'] = st.multiselect(
            "Select values",
            [None, 5, 10, 20, 30],
            default=[5, 10, 20]
        )
        st.markdown("**Minimum Samples Split:**")
        grid_search_params['min_samples_split'] = st.multiselect(
            "Select values",
            [2, 5, 10],
            default=[2, 5]
        )

def configure_gradient_boosting_parameters(use_grid_search, model_params, grid_search_params):
    """
    Configure parameters for Gradient Boosting models.
    
    Parameters:
    -----------
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    model_params : dict
        Dictionary to store model parameters if not using grid search
    grid_search_params : dict
        Dictionary to store grid search parameters if using grid search
    """
    if not use_grid_search:
        model_params['n_estimators'] = st.number_input(
            "Number of Boosting Stages", 
            min_value=10, 
            max_value=500, 
            value=100, 
            step=10,
            help="Number of boosting iterations. More stages generally improve performance but may lead to overfitting."
        )
        model_params['max_depth'] = st.number_input(
            "Maximum Depth", 
            min_value=1, 
            max_value=20, 
            value=3,
            help="Maximum depth of each tree. For boosting, shallow trees (3-5) often work best."
        )
        model_params['learning_rate'] = st.slider(
            "Learning Rate", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.1, 
            step=0.01,
            help="Controls how much each tree contributes to the final prediction. Lower values need more trees but often generalize better."
        )
    else:
        st.markdown("**Grid Search Parameters**")
        st.markdown("**Number of Boosting Stages:**")
        grid_search_params['n_estimators'] = st.multiselect(
            "Select values",
            [50, 100, 200, 300],
            default=[50, 100, 200]
        )
        st.markdown("**Maximum Depth:**")
        grid_search_params['max_depth'] = st.multiselect(
            "Select values",
            [3, 5, 7, 9],
            default=[3, 5, 7]
        )
        st.markdown("**Learning Rate:**")
        grid_search_params['learning_rate'] = st.multiselect(
            "Select values",
            [0.01, 0.05, 0.1, 0.2],
            default=[0.01, 0.1, 0.2]
        )

def configure_logistic_regression_parameters(use_grid_search, model_params, grid_search_params):
    """
    Configure parameters for Logistic Regression models.
    
    Parameters:
    -----------
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    model_params : dict
        Dictionary to store model parameters if not using grid search
    grid_search_params : dict
        Dictionary to store grid search parameters if using grid search
    """
    if not use_grid_search:
        model_params['C'] = st.number_input(
            "Regularization Strength (C)", 
            min_value=0.01, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Inverse of regularization strength. Smaller values specify stronger regularization, reducing overfitting."
        )
        model_params['max_iter'] = st.number_input(
            "Maximum Iterations", 
            min_value=100, 
            max_value=2000, 
            value=1000, 
            step=100,
            help="Maximum number of iterations for the solver to converge. Increase if the model doesn't converge."
        )
    else:
        st.markdown("**Grid Search Parameters**")
        st.markdown("**Regularization Strength (C):**")
        grid_search_params['C'] = st.multiselect(
            "Select values",
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            default=[0.1, 1.0, 10.0]
        )

def configure_regularized_regression_parameters(use_grid_search, model_params, grid_search_params):
    """
    Configure parameters for Ridge and Lasso Regression models.
    
    Parameters:
    -----------
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    model_params : dict
        Dictionary to store model parameters if not using grid search
    grid_search_params : dict
        Dictionary to store grid search parameters if using grid search
    """
    if not use_grid_search:
        model_params['alpha'] = st.slider(
            "Regularization Strength (alpha)", 
            min_value=0.001, 
            max_value=10.0, 
            value=1.0, 
            step=0.01, 
            format="%.3f",
            help="Regularization strength parameter. Higher values increase regularization, which may prevent overfitting but can underfit if too high."
        )
    else:
        st.markdown("**Grid Search Parameters**")
        st.markdown("**Regularization Strength (alpha):**")
        grid_search_params['alpha'] = st.multiselect(
            "Select values",
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            default=[0.001, 0.01, 0.1, 1.0, 10.0]
        )

def configure_knn_parameters(use_grid_search, model_params, grid_search_params):
    """
    Configure parameters for K-Nearest Neighbors models.
    
    Parameters:
    -----------
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    model_params : dict
        Dictionary to store model parameters if not using grid search
    grid_search_params : dict
        Dictionary to store grid search parameters if using grid search
    """
    if not use_grid_search:
        model_params['n_neighbors'] = st.slider(
            "Number of Neighbors (k)", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="Number of neighbors to use for prediction. Larger values reduce the impact of noise but make decision boundaries less distinct."
        )
        model_params['weights'] = st.selectbox(
            "Weight Function", 
            ["uniform", "distance"],
            help="'uniform': all neighbors weighted equally. 'distance': neighbors weighted by inverse of distance (closer neighbors have more influence)."
        )
    else:
        st.markdown("**Grid Search Parameters**")
        st.markdown("**Number of Neighbors:**")
        grid_search_params['n_neighbors'] = st.multiselect(
            "Select values",
            [3, 5, 7, 9, 11, 13, 15],
            default=[3, 5, 7, 9]
        )
        st.markdown("**Weight Function:**")
        grid_search_params['weights'] = st.multiselect(
            "Select values",
            ["uniform", "distance"],
            default=["uniform", "distance"]
        )

def configure_svm_parameters(use_grid_search, model_params, grid_search_params):
    """
    Configure parameters for SVM models (SVC and SVR).
    
    Parameters:
    -----------
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    model_params : dict
        Dictionary to store model parameters if not using grid search
    grid_search_params : dict
        Dictionary to store grid search parameters if using grid search
    """
    if not use_grid_search:
        model_params['C'] = st.slider(
            "Regularization Parameter (C)", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Regularization parameter. Trades off correct classification of training examples against maximizing the decision boundary margin."
        )
        model_params['kernel'] = st.selectbox(
            "Kernel", 
            ["rbf", "linear", "poly", "sigmoid"],
            help="Kernel type: 'rbf' (Gaussian) works well for most data, 'linear' is faster for linear problems, 'poly' for polynomial relationships."
        )
        model_params['gamma'] = st.selectbox(
            "Kernel Coefficient (gamma)", 
            ["scale", "auto"],
            help="Kernel coefficient. 'scale' uses 1/(n_features*X.var()) which is the default. 'auto' uses 1/n_features."
        )
    else:
        st.markdown("**Grid Search Parameters**")
        st.markdown("**Regularization Parameter (C):**")
        grid_search_params['C'] = st.multiselect(
            "Select values",
            [0.1, 1.0, 10.0, 100.0],
            default=[0.1, 1.0, 10.0]
        )
        st.markdown("**Kernel Type:**")
        grid_search_params['kernel'] = st.multiselect(
            "Select values",
            ["rbf", "linear", "poly", "sigmoid"],
            default=["rbf", "linear"]
        )
        st.markdown("**Kernel Coefficient:**")
        grid_search_params['gamma'] = st.multiselect(
            "Select values",
            ["scale", "auto"],
            default=["scale", "auto"]
        )

def generate_model_report():
    """
    Generate a JSON report of the current model configuration, parameters, and results
    
    Returns:
    --------
    str
        JSON string containing the model report
    """
    if st.session_state.model_results is None:
        return json.dumps({"error": "No model results available"})
    
    # Create a report dictionary
    report = {
        "model_type": st.session_state.model_type,
        "target_type": st.session_state.target_type,
        "target_variable": st.session_state.model_config["target"],
        "features": st.session_state.model_config["features"],
        "test_size": st.session_state.model_config["test_size"],
        "preprocessing": st.session_state.model_config["preprocessing"],
        "metrics": {}
    }
    
    # Add model parameters
    if "best_params" in st.session_state and st.session_state.best_params:
        report["parameters"] = st.session_state.best_params
    
    # Add appropriate metrics based on model type
    if st.session_state.target_type == "regression":
        report["metrics"] = {
            "mse": float(st.session_state.model_results["mse"]),
            "r2": float(st.session_state.model_results["r2"]),
            "rmse": float(np.sqrt(float(st.session_state.model_results["mse"])))
        }
    else:
        report["metrics"] = {
            "accuracy": float(st.session_state.model_results["accuracy"]),
            "precision": float(st.session_state.model_results["precision"]),
            "recall": float(st.session_state.model_results["recall"]),
            "f1": float(st.session_state.model_results["f1"])
        }
    
    # Add feature importance if available
    if st.session_state.feature_importance is not None:
        # Convert to dict for JSON serialization
        importance_dict = {}
        for i, feature in enumerate(st.session_state.feature_importance["features"]):
            importance_dict[feature] = float(st.session_state.feature_importance["importance"][i])
        
        report["feature_importance"] = importance_dict
    
    # Return as JSON string
    return json.dumps(report, indent=2)

# Initialize session state for the application
def initialize_session_state():
    """
    Initialize all session state variables in one place for better organization.
    
    This function ensures all required session state variables exist before they're accessed,
    preventing KeyError exceptions and reducing code duplication.
    """
    if "model_results" not in st.session_state:
        st.session_state.model_results = None
    if "feature_importance" not in st.session_state:
        st.session_state.feature_importance = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = None
    if "target_type" not in st.session_state:
        st.session_state.target_type = None
    if "test_data" not in st.session_state:
        st.session_state.test_data = None
    if "model_config" not in st.session_state:
        st.session_state.model_config = None
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = {"numerical": [], "categorical": []}
    if "previous_target" not in st.session_state:
        st.session_state.previous_target = None
    if "experiment_history" not in st.session_state:
        st.session_state.experiment_history = []
    if "show_save_form" not in st.session_state:
        st.session_state.show_save_form = False
    if "selected_num_features" not in st.session_state:
        st.session_state.selected_num_features = []
    if "best_params" not in st.session_state:
        st.session_state.best_params = None

# Call initialization function
initialize_session_state()

# App title and description
st.title("🤖 Machine Learning Model Trainer")
st.markdown("Train ML models on different datasets, configure parameters, and visualize results")

# Initialize train_button to False by default
train_button = False

# Add a function to reset model results when dataset or target changes
def reset_model_results_if_needed(current_dataset_name, current_target):
    """
    Reset model results if the dataset or target variable has changed.
    This prevents showing results that don't correspond to the current setup.
    
    Parameters:
    -----------
    current_dataset_name : str
        The name of the currently selected dataset
    current_target : str
        The name of the currently selected target variable
    """
    # Create a key to identify the current dataset and target combination
    current_key = f"{current_dataset_name}_{current_target}"
    
    # Check if we have a stored key
    if "dataset_target_key" not in st.session_state:
        st.session_state.dataset_target_key = current_key
    
    # If the dataset or target has changed, reset model results
    if st.session_state.dataset_target_key != current_key:
        st.session_state.model_results = None
        st.session_state.feature_importance = None
        st.session_state.predictions = None
        st.session_state.test_data = None
        st.session_state.model_config = None
        # Update the key
        st.session_state.dataset_target_key = current_key

# Sidebar for dataset selection and configuration
with st.sidebar:
    st.header("Model Configuration")
    
    # Dataset selection in an expander
    with st.expander("Dataset Selection", expanded=True):
        dataset_option = st.selectbox(
            "Select a dataset",
            ["iris", "tips", "diamonds", "titanic", "mpg", "penguins", "Upload Custom CSV"]
        )
        
        # Load dataset using the function
        data = load_dataset_with_error_handling(dataset_option)
    
    # Only show these options if data is loaded
    if data is not None:
        # Target and features selection in an expander
        with st.expander("Target & Features", expanded=True):
            # Target variable selection
            all_cols = data.columns.tolist()
            target_variable = st.selectbox(
                "Target Variable", 
                all_cols,
                help="The feature you want to predict. This will be treated as the dependent variable in your model."
            )
            
            # Reset model results if dataset or target changes
            reset_model_results_if_needed(dataset_option, target_variable)
            
            # Immediately filter out target from potential features
            numerical_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns.tolist() if col != target_variable]
            categorical_cols = [col for col in data.select_dtypes(include=['object', 'category']).columns.tolist() if col != target_variable]
            
            # Clear any existing selections that might include the target
            if "selected_features" in st.session_state:
                st.session_state.selected_features["numerical"] = [
                    col for col in st.session_state.selected_features["numerical"] 
                    if col != target_variable
                ]
                st.session_state.selected_features["categorical"] = [
                    col for col in st.session_state.selected_features["categorical"] 
                    if col != target_variable
                ]
            
            # Save current target for next comparison
            st.session_state.previous_target = target_variable
            
            # Determine if classification or regression
            if target_variable in data.select_dtypes(include=['object', 'category']).columns or data[target_variable].nunique() < 10:
                model_task = "classification"
                st.info(f"📊 Classification task detected")
                models = [
                    "Random Forest Classifier", 
                    "Extra Trees Classifier",
                    "Logistic Regression", 
                    "KNN Classifier",
                    "SVC",
                    "Gradient Boosting Classifier"
                ]
            else:
                model_task = "regression"
                st.info(f"📈 Regression task detected")
                models = [
                    "Linear Regression", 
                    "Ridge Regression",
                    "Lasso Regression",
                    "Random Forest Regressor",
                    "Extra Trees Regressor",
                    "KNN Regressor",
                    "SVR",
                    "Gradient Boosting Regressor"
                ]
            
            # Feature Selection - simplified UI
            st.subheader("Feature Selection")
            
            # Filter stored features to only include valid columns that exist in the current dataset
            valid_numerical = [col for col in st.session_state.selected_features["numerical"] 
                              if col in numerical_cols]
            valid_categorical = [col for col in st.session_state.selected_features["categorical"] 
                                if col in categorical_cols]
            
            # Update session state with filtered lists
            st.session_state.selected_features["numerical"] = valid_numerical
            st.session_state.selected_features["categorical"] = valid_categorical
            
            # Display multiselects with the persisted selections
            selected_num_features = st.multiselect(
                "Numerical features",
                numerical_cols,
                default=st.session_state.selected_features["numerical"],
                help="Select the numerical columns to use as input features for your model. These should have predictive power for your target."
            )
            
            selected_cat_features = st.multiselect(
                "Categorical features",
                categorical_cols,
                default=st.session_state.selected_features["categorical"],
                help="Select the categorical columns to use as input features for your model. These will be encoded according to your preprocessing settings."
            )
            
            # Update session state with new selections
            st.session_state.selected_features["numerical"] = selected_num_features
            st.session_state.selected_features["categorical"] = selected_cat_features
            
            # Combine selected features
            selected_features = selected_num_features + selected_cat_features
            
            if selected_features:
                st.success(f"✅ {len(selected_features)} features selected")
        
        # Preprocessing options
        with st.expander("Data Preprocessing", expanded=False):
            # Missing values handling
            missing_handling = st.selectbox(
                "Missing Values Strategy",
                ["Simple Imputation (mean/mode)", "Drop Rows with Missing Values", "Keep Missing Values"],
                help="Simple Imputation: Replaces missing values with mean (for numerical) or mode (for categorical). Drop Rows: Removes rows containing any missing values. Keep Missing: Leaves missing values as-is."
            )
            
            # Numerical features preprocessing - simplified to one scaling option at a time
            scaling_option = st.selectbox(
                "Numerical Features Scaling",
                ["None", "Standardization (z-score)", "Min-Max Scaling", "Robust Scaling"],
                help="Standardization: Transforms features to mean=0, std=1. Min-Max: Scales features to range [0,1]. Robust: Uses median and IQR, less affected by outliers. Scaling helps models that use distance measures."
            )
            
            # Add log transformation option separately
            log_transform = st.checkbox(
                "Apply Log Transformation", 
                value=False,
                help="Applies log(1+x) to numerical features. Useful for right-skewed data or features with exponential growth patterns. Helps normalize the distribution and reduce the impact of extreme values."
            )
            
            # Outlier removal option separately
            remove_outliers = st.checkbox(
                "Remove Outliers", 
                value=False,
                help="Removes data points that fall outside 1.5 × IQR (Interquartile Range). This typically removes about 0.7% of normally distributed data, but more if data is skewed or has actual outliers."
            )
            
            # Categorical features preprocessing
            cat_preprocessing = st.selectbox(
                "Categorical Features Encoding",
                ["One-Hot Encoding", "Label Encoding", "Target Encoding"],
                help="One-Hot: Creates binary columns for each category. Label: Converts categories to integer values. Target: Replaces categories with the mean target value for that category."
            )

        # Create a simplified preprocessing config dict
        preprocessing_config = {
            "missing_handling": missing_handling,
            "num_preprocessing": [],  # Initialize empty and add selected option below
            "cat_preprocessing": cat_preprocessing,
            "log_transform": log_transform,
            "remove_outliers": remove_outliers
        }

        # Add the selected scaling option if not "None"
        if scaling_option != "None":
            preprocessing_config["num_preprocessing"].append(scaling_option)
        
        # Model selection and configuration
        with st.expander("Model Configuration", expanded=True):
            selected_model = st.selectbox(
                "Select Model", 
                models,
                help="Choose the algorithm for training. Different models have different strengths and weaknesses depending on your data and task."
            )
            
            # Add grid search option
            use_grid_search = st.checkbox(
                "Use Grid Search", 
                value=False, 
                help="Automatically tests multiple hyperparameter combinations to find the best settings. Takes longer to train but may improve model performance."
            )
            
            # Common parameters
            test_size = st.slider(
                "Test Set Size", 
                0.1, 0.5, 0.2, 0.05,
                help="Percentage of data used for testing. Smaller values give more training data but less reliable performance estimates."
            )
            random_state = st.number_input(
                "Random State", 
                value=42,
                help="Seed for random operations. Using the same value ensures reproducible results across runs."
            )
        
        # Model parameters dict for easier passing to functions
        model_params = {}
        grid_search_params = {}
        
        # Model-specific parameters in a collapsible section
        with st.expander("Model Parameters", expanded=False):
            # Model-specific parameters
            if selected_model in ["Random Forest Classifier", "Random Forest Regressor", "Extra Trees Classifier", "Extra Trees Regressor"]:
                configure_tree_ensemble_parameters(use_grid_search, model_params, grid_search_params)
            elif selected_model in ["Gradient Boosting Classifier", "Gradient Boosting Regressor"]:
                configure_gradient_boosting_parameters(use_grid_search, model_params, grid_search_params)
            elif selected_model in ["Logistic Regression"]:
                configure_logistic_regression_parameters(use_grid_search, model_params, grid_search_params)
            elif selected_model in ["Ridge Regression", "Lasso Regression"]:
                configure_regularized_regression_parameters(use_grid_search, model_params, grid_search_params)
            elif selected_model in ["KNN Classifier", "KNN Regressor"]:
                configure_knn_parameters(use_grid_search, model_params, grid_search_params)
            elif selected_model in ["SVC", "SVR"]:
                configure_svm_parameters(use_grid_search, model_params, grid_search_params)
        
        # Train model button - keep outside expandable sections for visibility
        train_button = st.button("Train Model", type="primary", use_container_width=True)
        
        # Help section
        with st.expander("Help & Tips", expanded=False):
            st.markdown("""
            ## Quick Tips
            
            1. **Selecting Features**: Choose features that might have predictive power for your target
            2. **Data Preprocessing**: Handle missing values and scale numerical features 
            3. **Model Selection**: Choose based on your task (classification/regression)
            4. **Grid Search**: Use for finding optimal hyperparameters
            
            The app automatically detects if you need a classification or regression model based on your target variable.
            """)

# Main content
if data is not None:
    # Add tabs for main content, EDA, and experiment history
    eda_tab, main_tab, history_tab = st.tabs(["Exploratory Data Analysis", "Model Training", "Experiment History"])
    
    with eda_tab:
        st.header("Exploratory Data Analysis")
        
        # Data Summary Section
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Rows", data.shape[0])
        with col2:
            st.metric("Number of Columns", data.shape[1])
        with col3:
            missing_pct = (data.isna().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            st.metric("Missing Values (%)", f"{missing_pct:.2f}%")
        
        # Data Types Overview
        st.subheader("Data Types Overview")
        dtype_counts = data.dtypes.value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        dtype_counts.plot(kind='bar', ax=ax)
        plt.title('Distribution of Data Types')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Missing Values Analysis
        st.subheader("Missing Values Analysis")
        missing_data = data.isna().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data.plot(kind='bar', ax=ax)
            plt.title('Missing Values by Column')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.success("No missing values found in the dataset!")
        
        # Data visualization section
        st.subheader("Data Visualization")
        if len(data.select_dtypes(include=['int64', 'float64']).columns) >= 2:
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            x_axis = st.selectbox("X-axis", numeric_cols)
            y_axis = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            # Add color option for scatter plots
            color_option = st.selectbox(
                "Color points by (optional)", 
                ["None"] + categorical_cols,
                index=0
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if color_option != "None":
                scatter = sns.scatterplot(data=data, x=x_axis, y=y_axis, hue=color_option, alpha=0.7, ax=ax)
                # Move legend to the side
                scatter.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            else:
                sns.scatterplot(data=data, x=x_axis, y=y_axis, alpha=0.7, ax=ax)
            
            ax.set_title(f"{y_axis} vs {x_axis}")
            ax.grid(True, linestyle='--', alpha=0.7)
            fig.tight_layout()
            st.pyplot(fig)
        
        # Add correlation matrix
        st.subheader("Correlation Analysis")
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            
            # Create a better mask for the upper triangle
            mask = np.triu(np.ones_like(corr_matrix), k=1)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            heatmap = sns.heatmap(
                corr_matrix, 
                mask=mask,
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                center=0,
                square=True, 
                linewidths=.5,
                cbar_kws={"shrink": .8}
            )
            plt.title('Correlation Matrix', fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Top correlations (excluding self-correlations)
            st.subheader("Top Correlations")
            # Create a flattened view without the diagonal
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
            
            # Sort by absolute correlation value (descending)
            corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
            
            # Create a DataFrame for display
            if corr_pairs:
                top_correlations = pd.DataFrame(
                    corr_pairs[:10], 
                    columns=['Feature 1', 'Feature 2', 'Correlation']
                )
                st.dataframe(top_correlations)
            else:
                st.info("No correlation pairs found.")
        
        # Feature Distributions
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select feature to view distribution", numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, x=selected_feature, kde=True, ax=ax)
        plt.title(f'Distribution of {selected_feature}')
        st.pyplot(fig)
        
        # Box Plots
        st.subheader("Box Plots")
        selected_box_feature = st.selectbox("Select feature for box plot", numeric_cols)
        if len(categorical_cols) > 0:
            hue_col = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            if hue_col != "None":
                sns.boxplot(data=data, x=hue_col, y=selected_box_feature, ax=ax)
            else:
                sns.boxplot(data=data, y=selected_box_feature, ax=ax)
            plt.title(f'Box Plot of {selected_box_feature}')
            st.pyplot(fig)
        
        # Categorical Data Analysis
        if len(categorical_cols) > 0:
            st.subheader("Categorical Data Analysis")
            selected_cat = st.selectbox("Select categorical feature", categorical_cols)
            fig, ax = plt.subplots(figsize=(10, 6))
            data[selected_cat].value_counts().plot(kind='bar', ax=ax)
            plt.title(f'Distribution of {selected_cat}')
            plt.xticks(rotation=45)
            st.pyplot(fig)

    with main_tab:
        # Dataset Overview Section (in an expander)
        with st.expander("Dataset Overview", expanded=True):
            st.dataframe(data.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Shape:", data.shape)
            with col2:
                missing_values = data.isna().sum().sum()
                st.write("Missing Values:", missing_values)
                if missing_values > 0:
                    st.info(f"This dataset contains {missing_values} missing values that will be automatically handled during model training.")
            
            # Show basic statistics
            st.subheader("Statistics")
            st.dataframe(data.describe(), use_container_width=True)
        
        # Model Training (in an expander if not yet trained)
        if st.session_state.model_results is None:
            with st.expander("Train Model", expanded=True):
                st.subheader("Ready to Train")
                
                if selected_features:
                    with st.container():
                        st.info(f"**Selected Features:** {len(selected_features)}")
                        # Display features list
                        feature_text = ", ".join(selected_features)
                        st.text(feature_text)
                else:
                    st.warning("Please select at least one feature to train the model.")
        
        # Model Results Section (if available, in an expander)
        if st.session_state.model_results is not None:
            with st.expander("Model Performance", expanded=True):
                # Display model type and configuration
                display_model_config(
                    st.session_state.model_type,
                    st.session_state.model_config["target"],
                    st.session_state.model_config["features"],
                    st.session_state.model_config["test_size"]
                )
                
                # Display preprocessing information
                if "preprocessing" in st.session_state.model_config and st.session_state.model_config["preprocessing"]:
                    st.markdown("---")
                    st.subheader("Preprocessing Applied")
                    
                    preproc = st.session_state.model_config["preprocessing"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Missing Values:**")
                        st.write(preproc.get("missing_handling", "Default"))
                        
                        st.markdown("**Numerical Features:**")
                        num_preproc = preproc.get("num_preprocessing", [])
                        if num_preproc:
                            for np in num_preproc:
                                st.write(f"- {np}")
                        else:
                            st.write("- None")
                    
                    with col2:
                        st.markdown("**Categorical Features:**")
                        st.write(preproc.get("cat_preprocessing", "Default"))
                
                # Display grid search results if used
                if st.session_state.model_config["use_grid_search"] and st.session_state.best_params:
                    st.markdown("---")
                    st.subheader("Grid Search Results")
                    st.markdown("**Best Parameters Found:**")
                    for param, value in st.session_state.best_params.items():
                        st.markdown(f"- **{param}**: {value}")
                
                # Display results based on model type
                if st.session_state.target_type == "regression":
                    # Metrics in cards
                    display_regression_metrics(st.session_state.model_results)
                    
                    # Add a prediction vs actual plot
                    st.subheader("Predictions vs Actual Values")
                    y_test = st.session_state.test_data[1]
                    y_pred = st.session_state.predictions
                    
                    fig = plot_regression_results(y_test, y_pred)
                    st.pyplot(fig)
                else:
                    # Classification metrics
                    display_classification_metrics(st.session_state.model_results)
                    
                    # More elegant metrics explanation using Streamlit's native components
                    st.markdown("---")
                    with st.container():
                        st.markdown("""
                        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
                            <h4 style='color: #1E88E5;'>📊 Understanding the Metrics</h4>
                            <ul style='margin-left: 20px;'>
                                <li><b>Accuracy</b>: The proportion of correct predictions among all predictions</li>
                                <li><b>Precision</b>: How many selected items are relevant? (True positives / (True positives + False positives))</li>
                                <li><b>Recall</b>: How many relevant items are selected? (True positives / (True positives + False negatives))</li>
                                <li><b>F1 Score</b>: The harmonic mean of precision and recall, providing a balance between the two</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    conf_matrix = st.session_state.model_results['confusion_matrix']
                    class_names = st.session_state.model_results['class_names']
                    
                    fig = plot_confusion_matrix(conf_matrix, class_names)
                    st.pyplot(fig)
                    
                    # ROC Curve (for binary classification)
                    if len(class_names) == 2 and st.session_state.model_results['y_pred_proba'] is not None:
                        st.subheader("ROC Curve")
                        y_test = st.session_state.model_results['y_test']
                        y_pred_proba = st.session_state.model_results['y_pred_proba'][:, 1]
                        
                        fig = plot_roc_curve(y_test, y_pred_proba)
                        st.pyplot(fig)
                
                # Feature Importance
                if st.session_state.feature_importance is not None:
                    st.subheader("Feature Importance")
                    features = st.session_state.feature_importance["features"]
                    importance = st.session_state.feature_importance["importance"]
                    
                    fig = plot_feature_importance(features, importance)
                    st.pyplot(fig)
                
                # Add save experiment button with better styling
                st.markdown("---")
                
                # Create columns for form and download button
                save_col, download_col = st.columns([3, 1])
                
                # Use a single form for saving experiments
                with save_col.form(key="save_experiment_form", clear_on_submit=False):
                    st.subheader("Save Experiment")
                    
                    # Experiment naming
                    default_name = f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    experiment_name = st.text_input("Experiment Name", value=default_name)
                    
                    # What to save
                    st.markdown("**Select what to save:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        model_info = st.checkbox("Model Information", value=True)
                        metrics = st.checkbox("Performance Metrics", value=True)
                        feature_info = st.checkbox("Feature Information", value=True)
                    with col2:
                        predictions = st.checkbox("Predictions", value=True)
                        dataset_info = st.checkbox("Dataset Information", value=True)
                    
                    notes = st.text_area("Notes (optional)", height=100)
                    
                    # Submit button
                    submitted = st.form_submit_button("Save Experiment", type="primary", use_container_width=True)
                    
                    if submitted:
                        # Create and save experiment data using utility function
                        experiment_data = create_experiment_data(
                            name=experiment_name,
                            notes=notes,
                            model_type=st.session_state.model_type,
                            target_type=st.session_state.target_type,
                            model_info=model_info,
                            metrics=metrics,
                            feature_info=feature_info,
                            predictions=predictions,
                            dataset_info=dataset_info,
                            model_params=model_params,
                            use_grid_search=use_grid_search,
                            best_params=st.session_state.best_params,
                            model_results=st.session_state.model_results,
                            selected_features=selected_features,
                            feature_importance=st.session_state.feature_importance,
                            test_data=st.session_state.test_data,
                            model_predictions=st.session_state.predictions,
                            dataset_name=dataset_option,
                            target_variable=target_variable
                        )
                        
                        save_experiment(experiment_data)
                        st.success(f"✅ Experiment '{experiment_name}' saved successfully!")
                        st.rerun()
        
                # Add download button for model report
                with download_col:
                    st.markdown("<br><br>", unsafe_allow_html=True)  # Add some space to align with form
                    if st.button("📥 Download Report", type="primary", use_container_width=True):
                        report_data = generate_model_report()
                        st.download_button(
                            label="Download Report (JSON)",
                            data=report_data,
                            file_name=f"model_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )

    with history_tab:
        st.header("Experiment History")
        
        if len(st.session_state.experiment_history) == 0:
            st.info("No experiments saved yet. Train a model and save it to see it here!")
        else:
            # Add experiment type filter - REMOVE ALL OPTION
            st.subheader("Filter Experiments")
            experiment_type = st.radio(
                "Select Experiment Type",
                ["Classification", "Regression"],
                horizontal=True
            )
            
            # Filter experiments based on type - ALWAYS FILTER NOW
            filtered_experiments = get_sorted_experiments()
            filtered_experiments = [
                exp for exp in filtered_experiments 
                if exp.get("target_type", "").lower() == experiment_type.lower()
            ]
            
            if not filtered_experiments:
                st.warning(f"No {experiment_type.lower()} experiments found.")
            else:
                # Add experiment comparison options
                st.subheader("Compare Experiments")
                
                # Display total number of filtered experiments
                st.write(f"Total {experiment_type.lower()} experiments: {len(filtered_experiments)}")
                
                # Select experiments to compare
                experiment_names = [exp["name"] for exp in filtered_experiments]
                selected_experiments = st.multiselect(
                    "Select experiments to compare",
                    options=experiment_names,
                    default=[experiment_names[0]] if experiment_names else None
                )
                
                if selected_experiments:
                    # Create comparison data using utility function
                    comparison_data = create_experiment_comparison_data(filtered_experiments, selected_experiments)
                    
                    # Display comparison table
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                    
                    # Plot metrics comparison
                    st.subheader("Metrics Visualization")

                    # Check if we have mixed experiment types
                    experiment_types = set(exp.get("target_type", "").lower() for exp in filtered_experiments)
                    if len(experiment_types) > 1:
                        st.warning("⚠️ Cannot compare metrics between regression and classification experiments. Please filter by experiment type first.")
                    else:
                        plot_type = st.selectbox("Plot type", ["bar", "radar"])
                        
                        # Create and display plots using utility function
                        figures = plot_metrics_comparison(comparison_data, plot_type)
                        if figures:
                            for fig in figures:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No valid metrics to display for the selected experiments.")
                    
                    # Individual experiment details
                    st.subheader("Experiment Details")
                    selected_exp = st.selectbox("Select experiment to view details", selected_experiments)
                    if selected_exp:
                        exp = next(exp for exp in filtered_experiments if exp["name"] == selected_exp)
                        
                        with st.expander("Configuration", expanded=True):
                            st.json(get_experiment_config_display(exp))
                        
                        # Add option to delete experiment
                        if st.button(f"Delete '{selected_exp}'", type="secondary"):
                            st.session_state.experiment_history = [
                                e for e in st.session_state.experiment_history 
                                if e["name"] != selected_exp
                            ]
                            st.success(f"Experiment '{selected_exp}' deleted!")
                            st.rerun()

# Training logic
if train_button and selected_features:
    # Validate grid search parameters if grid search is enabled
    if use_grid_search and not validate_grid_search_params(grid_search_params):
        st.error("Please select at least one value for each grid search parameter before training.")
    else:
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            # Prepare data
            with status_container:
                display_training_phase("data_prep", "Extracting features and target variable")
            progress_bar.progress(10)
            
            # Get features and target
            X = data[selected_features]
            y = data[target_variable]
            
            # Preprocess data
            with status_container:
                display_training_phase("preprocessing", "Handling missing values and encoding categorical features")
            progress_bar.progress(30)
            
            try:
                X, y = preprocess_data(X, y)
            except Exception as e:
                st.error(f"Error during data preprocessing: {str(e)}")
                st.stop()
            
            # Train model
            with status_container:
                display_training_phase("splitting", f"Splitting data into {(1-test_size)*100:.0f}% train and {test_size*100:.0f}% test sets")
            progress_bar.progress(40)
            
            with status_container:
                if use_grid_search:
                    display_training_phase("training", f"Performing grid search for {selected_model} with {len(selected_features)} features")
                else:
                    display_training_phase("training", f"Training {selected_model} with {len(selected_features)} features")
            progress_bar.progress(60)
            
            # Get model results
            try:
                model_output = train_model(
                    X, 
                    y, 
                    selected_model, 
                    model_params, 
                    test_size, 
                    random_state, 
                    use_grid_search, 
                    grid_search_params,
                    preprocessing_config
                )
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.stop()
            
            # Evaluate model
            with status_container:
                display_training_phase("evaluation", "Calculating performance metrics and generating visualizations")
            progress_bar.progress(90)
            
            # Update session state
            st.session_state.model_type = model_output["model_type"]
            st.session_state.target_type = model_output["target_type"]
            st.session_state.model_results = model_output["results"]
            st.session_state.feature_importance = model_output["feature_importance"]
            st.session_state.test_data = model_output["test_data"]
            st.session_state.predictions = model_output["predictions"]
            st.session_state.best_params = model_output.get("best_params", None)
            
            # Store model configuration
            st.session_state.model_config = {
                "target": target_variable,
                "features": selected_features,
                "test_size": test_size,
                "use_grid_search": use_grid_search,
                "preprocessing": preprocessing_config
            }
            
            progress_bar.progress(100)
            
            with status_container:
                display_training_phase("complete", "Model training and evaluation completed successfully")
            
            # Success message
            st.success("Model training completed! Results displayed below.")
            
            # Force a rerun to update the UI with the model results
            st.rerun()
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.warning("Please check your data and model configuration and try again.")
elif train_button and not selected_features:
    st.warning("Please select at least one feature before training the model.")