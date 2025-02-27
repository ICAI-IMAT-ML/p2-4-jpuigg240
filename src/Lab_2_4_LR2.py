import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model
        Xt = np.transpose(X)

        beta = np.linalg.inv(Xt @ X) @ Xt @ y
        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        losses = []         # Histórico del error cuadrático medio
        coefficients = []   # Histórico de w
        intercepts = []     # Histórico de b


        for epoch in range(iterations):

            predictions = np.hstack([self.intercept, self.coefficients]) @ X.T
            error = predictions - y

            # Con el valor del gradiente actualizamos el valor de nuestros regresores
            gradient = (2 / m) * X.T @ error
            self.intercept -= learning_rate * gradient[0]
            self.coefficients -= learning_rate * gradient[1:]
            
            # Almacenamos el intercepto y coeficiente de esta iteración
            coefficients.append(self.coefficients.copy())
            intercepts.append(self.intercept)

            # Calculamos y almacenamos el valor del error cuadrático medio
            mse = np.mean(error ** 2)
            losses.append(mse)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: MSE = {mse}")

        coefficients = np.array(coefficients).T

        return losses, intercepts, coefficients

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")
        if np.ndim(X) == 1:
            predictions = self.intercept + self.coefficients * X
        else:
            predictions = self.intercept + X @ self.coefficients
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    n = len(y_true)    
    
    # R^2 Score
    rss = sum([(y_true[i] - y_pred[i])**2 for i in range(n)])
    tss = sum([(y_true[i] - np.mean(y_true))**2 for i in range(n)])

    r_squared = 1 - (rss / tss)

    # Root Mean Squared Error
    rmse = np.sqrt(sum([(y_true[i] - y_pred[i])**2 for i in range(n)]) / n)

    # Mean Absolute Error
    mae = sum([abs(y_true[i] - y_pred[i]) for i in range(n)]) / n

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}



def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        col = X_transformed[:, index]
        

        # TODO: Find the unique categories (works with strings)
        unique_vals = np.unique(col)

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = []
        for v in unique_vals:
            one_hot.append([1 if elem == v else 0 for elem in col])
        one_hot = np.array(one_hot).T

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        # 1) Borramos la columna categórica original
        X_transformed = np.delete(X_transformed, index, axis=1)
        
        # 2) Concatenamos las dummies a la izquierda
        X_transformed = np.hstack((one_hot, X_transformed))

    return X_transformed
