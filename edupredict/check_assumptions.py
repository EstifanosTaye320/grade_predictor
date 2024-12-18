import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_preprocessing import load_and_preprocess_data, labels
from linear_regression import LinearRegression
import tensorflow as tf

def check_assumptions():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    X_all = np.concatenate((X_train, X_val, X_test))
    y_all = np.concatenate((y_train, y_val, y_test))

    # Check for linearity with scatter plots of each feature vs. target
    for i in range(X_all.shape[1]):
        plt.figure(figsize=(6, 4))
        plt.scatter(X_all[:, i], y_all)
        plt.title(f"{labels[i]} vs. Target (Linearity Check)")
        plt.xlabel(f"Feature {i+1}: {labels[i]}")
        plt.ylabel("Final Grade (G3)")
        plt.show()

    # Check for multicollinearity (correlation between features)
    correlation_matrix = np.corrcoef(X_train.T)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
    plt.title("Feature Correlation Matrix (Collinearity Check)")
    plt.show()

    # Train a temporary linear regression model and check residuals
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    

    input_x = tf.linspace()

    # Residual plot for homoscedasticity
    residuals = y_train - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Plot (Homoscedasticity Check)")
    plt.show()

if __name__ == "__main__":
    check_assumptions()
