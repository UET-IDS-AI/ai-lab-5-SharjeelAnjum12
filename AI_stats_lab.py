"""
AIstats_lab.py

Student starter file for the Regularization & Overfitting lab.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# =========================
# Helper Functions
# =========================

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# =========================
# Q1 Lasso Regression
# =========================

def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Implement Lasso regression using gradient descent.
    """

    # TODO: Load diabetes dataset
    # TODO: Train/test split
    # TODO: Standardize features
    # TODO: Add bias column
    # TODO: Initialize theta
    # TODO: Implement gradient descent with L1 regularization
    # TODO: Compute predictions
    # TODO: Compute metrics

    data = load_diabetes()
    X = data.data
    y = data.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Adding bias column
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    # Initialize theta
    n_samples, n_features = X_train.shape
    theta = np.zeros(n_features)

    # Gradient Descent
    for _ in range(epochs):

        # Predictions
        y_pred = X_train @ theta

        # Error
        error = y_pred - y_train

        # Gradient of MSE
        grad = (1 / n_samples) * (X_train.T @ error)

        # L1 subgradient 
        l1_grad = lambda_reg * np.sign(theta)
        l1_grad[0] = 0

        # Update rule
        theta = theta - lr * (grad + l1_grad)

    # Final predictions
    train_pred = X_train @ theta
    test_pred = X_test @ theta

    # Metrics
    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q2 Polynomial Overfitting
# =========================

def polynomial_overfitting_experiment(max_degree=10):
    """
    Study overfitting using polynomial regression.
    """

    # TODO: Load dataset
    # TODO: Select BMI feature only
    # TODO: Train/test split
    # TODO: Loop through polynomial degrees
    # TODO: Create polynomial features
    # TODO: Fit regression using normal equation
    # TODO: Compute train/test errors

    data = load_diabetes()
    X = data.data[:, 2].reshape(-1, 1)   # BMI feature only
    y = data.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    degrees = []
    train_errors = []
    test_errors = []

    # Loop through polynomial degrees
    for d in range(1, max_degree + 1):
        degrees.append(d)

        # Polynomial features
        poly = PolynomialFeatures(degree=d, include_bias=True)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Normal Equation
        theta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train

        # Predictions
        train_pred = X_train_poly @ theta
        test_pred = X_test_poly @ theta

        # Compute MSE
        train_errors.append(mse(y_train, train_pred))
        test_errors.append(mse(y_test, test_pred))

    return {
        "degrees": degrees,
        "train_mse": train_errors,
        "test_mse": test_errors
    }
