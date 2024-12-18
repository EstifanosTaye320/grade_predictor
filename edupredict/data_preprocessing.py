# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

labels = ['studytime', 'failures', 'absences', 'G1', 'G2']
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv("student-mat.csv", sep=";")

    # Selecting relevant features and the target variable
    features = data[['studytime', 'failures', 'absences', 'G1', 'G2']]
    target = data['G3']  # Final grade

    # Splitting the data into training (60%), validation (20%), and test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert to numpy arrays for manual calculations
    X_train, X_val, X_test = np.array(X_train), np.array(X_val), np.array(X_test)
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
