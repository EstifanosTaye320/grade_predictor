from data_preprocessing import load_and_preprocess_data
from linear_regression import LinearRegression
import numpy as np

def train_and_tune_model():
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    best_alpha = None
    best_mse = float('inf')
    best_model = None

    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        model = LinearRegression(alpha=alpha, epochs=1000)
        model.fit(X_train, y_train)

        # Validate the model
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        print(f"Alpha: {alpha}, Validation MSE: {mse}")

        # Choose the best alpha based on validation MSE
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_model = model

    print(f"Best alpha: {best_alpha}")
    return best_model

# Manual MSE function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    best_model = train_and_tune_model()
