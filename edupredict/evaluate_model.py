from train_model import train_and_tune_model
from data_preprocessing import load_and_preprocess_data
import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def evaluate_model():
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    print(len(X_train))
    print(len(X_test))
    print(len(X_val))   

    # Train the final model with the best hyperparameter (alpha)
    best_model = train_and_tune_model()

    # Make predictions on the test data
    y_test_pred = best_model.predict(X_test)

    # Calculate performance metrics manually
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    r2 = r_squared(y_test, y_test_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

if __name__ == "__main__":
    evaluate_model()
