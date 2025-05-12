import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools

np.random.seed(42)

# Data preprocessing
def preprocess_data(df):
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df['hour'] = df['Formatted Date'].dt.hour
    df['day'] = df['Formatted Date'].dt.day
    df['month'] = df['Formatted Date'].dt.month
    df['dayofweek'] = df['Formatted Date'].dt.dayofweek
    categorical_cols = ['Summary', 'Precip Type', 'Daily Summary']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna('None'))
    df = df.drop(['Formatted Date'], axis=1)
    return df

def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps - 1])
    return np.array(X_seq), np.array(y_seq)

# CNN Layers
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class Conv1D:
    def __init__(self, num_filters, kernel_size, input_channels):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        limit = np.sqrt(6 / (input_channels * kernel_size + num_filters))
        self.filters = np.random.uniform(-limit, limit, (num_filters, kernel_size, input_channels))
        self.biases = np.zeros(num_filters)
        # Adam state
        self.m_f = np.zeros_like(self.filters)
        self.v_f = np.zeros_like(self.filters)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0

    def forward(self, X):
        batch_size, seq_len, in_channels = X.shape
        pad = self.kernel_size // 2
        X_padded = np.pad(X, ((0,0), (pad,pad), (0,0)), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(X_padded, (self.kernel_size, in_channels), axis=(1,2))
        windows = windows[:, :, 0, :, :]  # shape: (batch, seq_len, kernel_size, in_channels)
        out = np.tensordot(windows, self.filters, axes=([2,3],[1,2])) + self.biases
        self.input = X
        self.windows = windows
        self.output = out
        return out

    def backward(self, d_out, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        batch_size, seq_len, _ = self.input.shape
        pad = self.kernel_size // 2
        X_padded = np.pad(self.input, ((0,0),(pad,pad),(0,0)), mode='constant')
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input_padded = np.zeros_like(X_padded)

        for f in range(self.num_filters):
            d_filters[f] = np.sum(self.windows * d_out[:, :, f][..., None, None], axis=(0,1))
            d_biases[f] = np.sum(d_out[:, :, f])

        for i in range(seq_len):
            for k in range(self.kernel_size):
                d_input_padded[:, i+k, :] += np.tensordot(d_out[:, i, :], self.filters[:, k, :], axes=(1,0))

        d_input = d_input_padded[:, pad:-pad, :]

        # Adam state
        self.t += 1
        self.m_f = beta1 * self.m_f + (1 - beta1) * d_filters
        self.v_f = beta2 * self.v_f + (1 - beta2) * (d_filters ** 2)
        m_f_hat = self.m_f / (1 - beta1 ** self.t)
        v_f_hat = self.v_f / (1 - beta2 ** self.t)
        self.filters -= learning_rate * m_f_hat / (np.sqrt(v_f_hat) + eps)

        self.m_b = beta1 * self.m_b + (1 - beta1) * d_biases
        self.v_b = beta2 * self.v_b + (1 - beta2) * (d_biases ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

        return d_input

class MaxPooling1D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, X):
        batch_size, seq_len, channels = X.shape
        out_len = seq_len // self.pool_size
        X_reshaped = X[:, :out_len * self.pool_size, :].reshape(batch_size, out_len, self.pool_size, channels)
        out = X_reshaped.max(axis=2)
        self.input = X
        self.X_reshaped = X_reshaped
        self.max_indices = X_reshaped.argmax(axis=2)
        return out

    def backward(self, d_out):
        batch_size, out_len, channels = d_out.shape
        d_input = np.zeros_like(self.input)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_len):
                    idx = self.max_indices[b, i, c]
                    d_input[b, i*self.pool_size + idx, c] = d_out[b, i, c]
        return d_input

class Flatten:
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    def backward(self, d_out):
        return d_out.reshape(self.input_shape)

class Dense:
    def __init__(self, input_dim, output_dim):
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.biases = np.zeros(output_dim)
        # Adam state
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0
    def forward(self, X):
        self.input = X
        return X @ self.weights + self.biases
    def backward(self, d_out, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        d_weights = self.input.T @ d_out
        d_biases = np.sum(d_out, axis=0)
        d_input = d_out @ self.weights.T

        self.t += 1
        # Adam update for weights
        self.m_w = beta1 * self.m_w + (1 - beta1) * d_weights
        self.v_w = beta2 * self.v_w + (1 - beta2) * (d_weights ** 2)
        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
        # Adam update for biases
        self.m_b = beta1 * self.m_b + (1 - beta1) * d_biases
        self.v_b = beta2 * self.v_b + (1 - beta2) * (d_biases ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

        return d_input

class Dropout:
    def __init__(self, rate=0.2):
        self.rate = rate
        self.training = True
    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.rate) / (1.0 - self.rate)
            return X * self.mask
        else:
            return X
    def backward(self, d_out):
        if self.training:
            return d_out * self.mask
        else:
            return d_out

# CNN Model
class CNNModel:
    def __init__(
        self, input_shape, learning_rate=0.001, 
        conv1_filters=64, conv2_filters=128, 
        kernel_size=3, dropout_rate=0.2
    ):
        self.learning_rate = learning_rate
        batch_size, seq_len, channels = input_shape

        self.conv1 = Conv1D(num_filters=conv1_filters, kernel_size=kernel_size, input_channels=channels)
        self.pool1 = MaxPooling1D(pool_size=2)
        self.dropout1 = Dropout(rate=dropout_rate)

        self.conv2 = Conv1D(num_filters=conv2_filters, kernel_size=kernel_size, input_channels=conv1_filters)
        self.pool2 = MaxPooling1D(pool_size=2)
        self.dropout2 = Dropout(rate=dropout_rate)

        flattened_size = (seq_len // 4) * conv2_filters
        self.flatten = Flatten()
        self.dense1 = Dense(flattened_size, 32)
        self.dropout3 = Dropout(rate=dropout_rate)
        self.dense2 = Dense(32, 1)

    def forward(self, X):
        out = self.conv1.forward(X)
        out = relu(out)
        out = self.pool1.forward(out)
        out = self.dropout1.forward(out)

        out = self.conv2.forward(out)
        out = relu(out)
        out = self.pool2.forward(out)
        out = self.dropout2.forward(out)

        out = self.flatten.forward(out)
        out = self.dense1.forward(out)
        self.relu1_out = relu(out)
        out = self.relu1_out
        out = self.dropout3.forward(out)

        out = self.dense2.forward(out)
        return out.squeeze()

    def backward(self, d_loss):
        d_out = d_loss.reshape(-1, 1)
        d_out = self.dense2.backward(d_out, self.learning_rate)
        d_out = self.dropout3.backward(d_out)
        d_out = d_out * relu_derivative(self.relu1_out)
        d_out = self.dense1.backward(d_out, self.learning_rate)
        d_out = self.flatten.backward(d_out)

        d_out = self.dropout2.backward(d_out)
        d_out = self.pool2.backward(d_out)
        d_out = d_out * relu_derivative(self.conv2.output)
        d_out = self.conv2.backward(d_out, self.learning_rate)

        d_out = self.dropout1.backward(d_out)
        d_out = self.pool1.backward(d_out)
        d_out = d_out * relu_derivative(self.conv1.output)
        d_out = self.conv1.backward(d_out, self.learning_rate)

    def train_on_batch(self, X_batch, y_batch):
        preds = self.forward(X_batch)
        loss = np.mean((preds - y_batch) ** 2)
        d_loss = 2 * (preds - y_batch) / y_batch.size
        self.backward(d_loss)
        return loss

    def predict(self, X):
        self.dropout1.training = False
        self.dropout2.training = False
        self.dropout3.training = False
        preds = self.forward(X)
        self.dropout1.training = True
        self.dropout2.training = True
        self.dropout3.training = True
        return preds

# Training Loop
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128, patience=3):
    n_train = X_train.shape[0]
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        indices = np.arange(n_train)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        losses = []
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            loss = model.train_on_batch(X_batch, y_batch)
            losses.append(loss)
        train_loss = np.mean(losses)
        y_val_pred = model.predict(X_val)
        val_loss = np.mean((y_val_pred - y_val) ** 2)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    return train_losses, val_losses

# Evaluation Metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("\nModel Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    return mae, mse, rmse, r2

# Hyperparameter Search
def hyperparameter_search(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape):
    # Define search space (expand as needed)
    param_grid = {
        'learning_rate': [0.001],
        'batch_size': [64, 128],
        'conv1_filters': [32, 64],
        'conv2_filters': [64, 128],
        'kernel_size': [3],
        'dropout_rate': [0.2, 0.3],
        'epochs': [10],
        'patience': [3]
    }

    keys, values = zip(*param_grid.items())
    best_val_loss = float('inf')
    best_params = None
    best_model = None

    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        print(f"\nTesting params: {params}")
        model = CNNModel(
            input_shape=input_shape,
            learning_rate=params['learning_rate'],
            conv1_filters=params['conv1_filters'],
            conv2_filters=params['conv2_filters'],
            kernel_size=params['kernel_size'],
            dropout_rate=params['dropout_rate']
        )
        train_losses, val_losses = train_model(
            model, X_train_seq, y_train_seq, X_val_seq, y_val_seq,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            patience=params['patience']
        )
        val_loss = val_losses[-1]
        print(f"Validation loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_model = model
            best_train_losses = train_losses
            best_val_losses = val_losses

    print(f"\nBest params: {best_params}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return best_model, best_params, best_train_losses, best_val_losses

# Main
def main():
    df = pd.read_csv('weatherHistory.csv')  # Update path if needed
    print("Preprocessing data...")
    df = preprocess_data(df)
    target_col = 'Apparent Temperature (C)'
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    time_steps = 24
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

    print(f"Train shape: {X_train_seq.shape}, {y_train_seq.shape}")
    print(f"Val shape: {X_val_seq.shape}, {y_val_seq.shape}")
    print(f"Test shape: {X_test_seq.shape}, {y_test_seq.shape}")

    input_shape = X_train_seq.shape  # (samples, time_steps, features)
    print("Starting hyperparameter search...")
    best_model, best_params, train_losses, val_losses = hyperparameter_search(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape
    )

    # Plot training and validation loss for the best model
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss (Best Model)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    evaluate_model(best_model, X_test_seq, y_test_seq)

if __name__ == "__main__":
    main()
