import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap


def train_ml(X_train, y_train, X_test, random_state=42):
    """
    Trains a classical Random Forest regression model on molecular features
    and returns the fitted model and predictions on the test set.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)
    X_test : array-like of shape (n_samples_test, n_features)
    random_state : int, default=42

    Returns
    -------
    model : RandomForestRegressor
    y_pred : np.ndarray
    """
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, np.asarray(y_pred, dtype=float)


def compute_shap_explanations(model, X_sample):
    """
    Computes SHAP (SHapley Additive exPlanations) values for the Random Forest model
    to interpret feature contributions to predicted bioactivity (pIC50).

    Parameters
    ----------
    model : RandomForestRegressor
        Fitted random forest regressor.
    X_sample : pd.DataFrame or np.ndarray
        Sample of molecular features to explain.

    Returns
    -------
    dict
        Contains:
        - 'explainer': shap.TreeExplainer
        - 'shap_values': np.ndarray of shape (n_samples, n_features)
        - 'expected_value': float (base value)
        - 'mean_abs_shap': dict of {feature_name: importance}
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(X_sample, pd.DataFrame):
        feature_names = list(X_sample.columns)
    else:
        feature_names = [f"Feature_{i}" for i in range(X_sample.shape[1])]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance_dict = {f: float(v) for f, v in zip(feature_names, mean_abs)}
    # Sort descending
    importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    expected_val = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'expected_value': expected_val,
        'mean_abs_shap': importance_dict,
        'feature_names': feature_names
    }
