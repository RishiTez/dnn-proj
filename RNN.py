import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
import gc

df = pd.read_csv('weather.csv')
df

df.describe()

df.isnull().sum()

df[df['Precip Type'].isnull()].head()

df.dropna(inplace=True)

df.duplicated().sum()

df = df.drop_duplicates()

"""Removing outliers for better model training"""

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


outliers = df[outlier_mask]
print(f"Number of rows with outliers: {len(outliers)}")


for col in columns_with_outliers:
    df[col] = df[col].clip(lower=lower_bound[col], upper=upper_bound[col])

"""Train-Test Split"""

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

print(f'Train Shape: {X_train_reshaped.shape}, Test Shape: {X_test_reshaped.shape}')
print(f'Train Shape: {y_train_reshaped.shape}, Test Shape: {y_test_reshaped.shape}')

input_size = X_train_reshaped.shape[2]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def train_rnn(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, lr, epochs=10):
    Wxh = np.random.randn(input_size, hidden_size) * 0.01
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01
    Why = np.random.randn(hidden_size, output_size) * 0.01
    bh = np.zeros((1, hidden_size))
    by = np.zeros((1, output_size))

    for epoch in range(epochs):
        total_loss = 0
        for i in range(X_train.shape[0]):
            x = X_train[i]
            y_true = y_train[i]

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

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / X_train.shape[0]:.4f}")

    return Wxh, Whh, Why, bh, by

def predict_rnn(X, Wxh, Whh, Why, bh, by, hidden_size, scaler_y, y_true_scaled):
    y_preds = []
    for i in range(X.shape[0]):
        x = X[i]
        h = np.tanh(np.dot(x, Wxh) + np.dot(np.zeros((1, hidden_size)), Whh) + bh)
        y_pred = np.dot(h, Why) + by
        y_preds.append(y_pred)

    y_preds = np.array(y_preds).reshape(-1, 1)

    y_preds_orig = scaler_y.inverse_transform(y_preds)
    y_true_orig = scaler_y.inverse_transform(y_true_scaled)

    mse = mean_squared_error(y_true_orig, y_preds_orig)
    r2 = r2_score(y_true_orig, y_preds_orig)

    print("MSE:", mse)
    print("R2 Score:", r2)

    return y_preds_orig, y_true_orig

hidden_sizes = [16, 32, 64]
learning_rates = [0.001, 0.005]
best_loss = float('inf')
best_params = None

for hs, lr in product(hidden_sizes, learning_rates):
    print(f"\nTrying hidden_size={hs}, learning_rate={lr}")

    Wxh, Whh, Why, bh, by = train_rnn(
        X_train_reshaped, y_train_reshaped,
        X_test_reshaped, y_test_reshaped,
        input_size=input_size,
        hidden_size=hs,
        output_size=1,
        lr=lr,
        epochs=10
    )

    y_pred, y_true = predict_rnn(
        X_test_reshaped, Wxh, Whh, Why, bh, by,
        hidden_size=hs,
        scaler_y=scaler_y,
        y_true_scaled=y_test_reshaped
    )

    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE: {mse:.4f}")

    if mse < best_loss:
        best_loss = mse
        best_params = (hs, lr)
        print("New best model found!")

print("\nBest Parameters:")
print(f"Hidden Size: {best_params[0]}, Learning Rate: {best_params[1]}")
