# -*- coding: utf-8 -*-
"""Cleaned Apparent Temp Prediction RNN"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('/content/weatherHistory.csv')

# Correlation matrix (removed plot)

# Handle missing values
df.dropna(inplace=True)

# Remove duplicates
df = df.drop_duplicates()

# Outlier treatment
df_original = df.copy()
columns_with_outliers = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']

Q1 = df[columns_with_outliers].quantile(0.25)
Q3 = df[columns_with_outliers].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.0 * IQR
upper_bound = Q3 + 1.0 * IQR
outlier_mask = False
for col in columns_with_outliers:
    col_mask = (df[col] < lower_bound[col]) | (df[col] > upper_bound[col])
    outlier_mask = outlier_mask | col_mask

for col in columns_with_outliers:
    df[col] = df[col].clip(lower=lower_bound[col], upper=upper_bound[col])

# Train-test split and scaling
df = df.select_dtypes(include='number')

X = df.drop(['Apparent Temperature (C)'], axis='columns')
y = df[['Apparent Temperature (C)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

y_train_reshaped = y_train_scaled.reshape((y_train_scaled.shape[0], 1))
y_test_reshaped = y_test_scaled.reshape((y_test_scaled.shape[0], 1))

# RNN parameters
input_size = X_train_reshaped.shape[2]
hidden_size = 32
output_size = 1
lr = 0.001
epochs = 20

Wxh = np.random.randn(input_size, hidden_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(hidden_size, output_size) * 0.01
bh = np.zeros((1, hidden_size))
by = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def print_model_summary():
    print("Model Architecture:")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Output size: {output_size}")
    print("\nParameter Shapes:")
    print(f"Wxh: {Wxh.shape}")
    print(f"Whh: {Whh.shape}")
    print(f"Why: {Why.shape}")
    print(f"bh: {bh.shape}")
    print(f"by: {by.shape}")

print_model_summary()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for i in range(X_train_reshaped.shape[0]):
        x = X_train_reshaped[i]
        y_true = y_train_scaled[i]
        h_prev = np.zeros((1, hidden_size))

        h = np.tanh(np.dot(x, Wxh) + np.dot(h_prev, Whh) + bh)
        y_pred = np.dot(h, Why) + by

        loss = mse_loss(y_true, y_pred)
        total_loss += loss

        dy = 2 * (y_pred - y_true)
        dWhy = np.dot(h.T, dy)
        dby = dy

        dh = np.dot(dy, Why.T) * (1 - h ** 2)
        dWxh = np.dot(x.T, dh)
        dWhh = np.dot(h_prev.T, dh)
        dbh = dh

        Wxh -= lr * dWxh
        Whh -= lr * dWhh
        Why -= lr * dWhy
        bh -= lr * dbh
        by -= lr * dby

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/X_train_reshaped.shape[0]:.4f}")

# Inference
y_preds = []
for i in range(X_test_reshaped.shape[0]):
    x = X_test_reshaped[i]
    h = np.tanh(np.dot(x, Wxh) + np.dot(np.zeros((1, hidden_size)), Whh) + bh)
    y_pred = np.dot(h, Why) + by
    y_preds.append(y_pred)

y_preds = np.array(y_preds).reshape(-1, 1)
y_preds_orig = scaler_y.inverse_transform(y_preds)
y_test_orig = scaler_y.inverse_transform(y_test_scaled)

# Evaluation
print("MSE:", mean_squared_error(y_test_orig, y_preds_orig))
print("R2 Score:", r2_score(y_test_orig, y_preds_orig))
