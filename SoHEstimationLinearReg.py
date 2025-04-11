import numpy as np
import pandas as pd


def estimate_battery_soh(train_df: pd.DataFrame, 
                         test_df: pd.DataFrame, 
                         features: list, 
                         target: str = 'ibmu_algopksohtrue'):
    """
    Train a simple model to estimate battery State of Health (SoH) from historical data and evaluate it.
    
    Parameters:
        train_df (pd.DataFrame): Training data containing features and target.
        test_df (pd.DataFrame): Testing data for validation.
        features (list of str): List of feature column names to use for prediction.
        target (str): Name of the target column (SoH true value). Defaults to 'ibmu_algopksohtrue'.
    
    Returns:
        dict: A dictionary with the trained model parameters and performance metrics (MAE, R2).
              For linear model, returns weights and intercept as model parameters.
    """
    # 1. Ensure only available features are used
    features_to_use = [f for f in features if f in train_df.columns and f in test_df.columns]
    dropped_features = [f for f in features if f not in features_to_use]
    if dropped_features:
        print(f"Note: Dropping unavailable features: {dropped_features}")
    if not features_to_use:
        raise ValueError("No valid features available for training.")
    
    # 2. Handle missing data in the features: drop rows with NaN in any feature or target
    train_clean = train_df.dropna(subset=features_to_use + [target]).copy()
    test_clean  = test_df.dropna(subset=features_to_use + [target]).copy()
    
    # Separate input features (X) and target (y)
    X_train = train_clean[features_to_use].values
    y_train = train_clean[target].values
    X_test  = test_clean[features_to_use].values
    y_test  = test_clean[target].values
    
    # 3. Create a linear regression model   
    # Add a column of ones to X for the intercept term (bias)
    ones_train = np.ones((X_train.shape[0], 1))
    X_train_aug = np.hstack([X_train, ones_train])
    # Solve for weights using least squares: w = (X^T X)^-1 X^T y
    # (np.linalg.lstsq finds the least-squares solution)
    w_aug, *_ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
    # Split weights and intercept
    weights = w_aug[:-1]   # coefficients for each feature
    intercept = w_aug[-1]  # bias term
    
    
    # 5. Output the model and metrics
    model_params = {
        'weights': weights,
        'intercept': intercept
    }
    
    return {'model': model_params, 'metrics': metrics}
