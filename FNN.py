import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product
from tqdm import tqdm
import time

# Activation functions and derivatives
def relu(Z): return np.maximum(0, Z)
def relu_derivative(Z): return Z > 0

def sigmoid(Z): return 1 / (1 + np.exp(-Z))
def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def tanh(Z): return np.tanh(Z)
def tanh_derivative(Z): return 1 - np.tanh(Z)**2

def linear(Z): return Z
def linear_derivative(Z): return np.ones_like(Z)

activation_functions = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative)
}

# Loss functions
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return (y_pred - y_true)

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\nRegression Metrics:")
    print(f"  MAE   : {mae:.4f}")
    print(f"  MSE   : {mse:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  R^2   : {r2:.4f}")

class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01, task='regression'):
        print("Initializing Neural Network...")
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.task = task
        self.weights = []
        self.biases = []
        self.activations = activations

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
        print("Initialization complete.")

    def forward(self, X):
        A = X
        cache = {'A0': X}
        for idx, (W, b, activation_name) in enumerate(zip(self.weights, self.biases, self.activations)):
            Z = np.dot(A, W) + b
            activation_func, _ = activation_functions[activation_name]
            A = activation_func(Z)
            cache[f'Z{idx+1}'] = Z
            cache[f'A{idx+1}'] = A
        return A, cache

    def backward(self, X, y, cache):
        grads = {}
        m = X.shape[0]
        L = len(self.weights)

        assert f'A{L}' in cache, f"A{L} not found in cache. Available keys: {list(cache.keys())}"

        if self.task == 'regression':
            dA = mean_squared_error_derivative(y, cache[f'A{L}'])
        else:
            raise ValueError("Only regression supported currently.")

        for l in reversed(range(L)):
            activation_func, activation_derivative = activation_functions[self.activations[l]]
            dZ = dA * activation_derivative(cache[f'Z{l+1}'])
            dW = np.dot(cache[f'A{l}'].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            if l > 0:
                dA = np.dot(dZ, self.weights[l].T)

            grads[f'dW{l+1}'] = dW
            grads[f'db{l+1}'] = db

        return grads

    def update_parameters(self, grads):
        for l in range(len(self.weights)):
            self.weights[l] -= self.learning_rate * grads[f'dW{l+1}']
            self.biases[l] -= self.learning_rate * grads[f'db{l+1}']

    def fit(self, X, y, epochs=200, batch_size=32, verbose=True):
        print("Starting training...")
        loss_history = []
        m = X.shape[0]

        best_loss = float('inf')
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            bar = tqdm(total=m, desc=f"Epoch {epoch+1} Progress", unit="sample")

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred, cache = self.forward(X_batch)
                grads = self.backward(X_batch, y_batch, cache)
                self.update_parameters(grads)

                bar.update(len(X_batch))

            bar.close()

            y_pred_full, _ = self.forward(X)
            loss = mean_squared_error(y, y_pred_full)
            loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]

            if verbose:
                mae = mean_absolute_error(y, y_pred_full)
                rmse = np.sqrt(loss)
                r2 = r2_score(y, y_pred_full)
                print(f"Epoch {epoch+1} Loss: {loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R^2: {r2:.4f}")

        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        print("Training complete. Best weights restored.")
        return loss_history

    def predict(self, X):
        print("Making predictions...")
        y_pred, _ = self.forward(X)
        return y_pred

# Grid search for hyperparameter tuning
def hyperparameter_search(X_train, y_train, X_val, y_val):
    hidden_layer_options = [[8, 16], [16, 8], [32, 16], [16, 32]]
    activation_options = [['relu', 'relu', 'linear'], ['relu', 'tanh', 'linear'], ['sigmoid', 'sigmoid', 'linear']]
    learning_rates = [0.001, 0.005, 0.01]

    best_loss = float('inf')
    best_model = None
    best_params = {}

    for hidden_layers, activations, lr in product(hidden_layer_options, activation_options, learning_rates):
        print(f"\n--- Trying Configuration: Layers={hidden_layers}, Activations={activations}, LR={lr} ---")
        layer_sizes = [X_train.shape[1]] + hidden_layers + [1]

        nn = NeuralNetwork(layer_sizes=layer_sizes, activations=activations, learning_rate=lr)
        nn.fit(X_train, y_train, epochs=50, batch_size=32, verbose=True)

        y_pred = nn.predict(X_val)
        loss = mean_squared_error(y_val, y_pred)
        print(f"Validation Loss: {loss:.4f}")

        if loss < best_loss:
            print("New best model found!")
            best_loss = loss
            best_model = nn
            best_params = {'layers': hidden_layers, 'activations': activations, 'lr': lr}

        del nn
        gc.collect()

    print("\nBest Hyperparameters:", best_params)
    return best_model

# Main execution
def main():
    print("Loading data...")
    for _ in tqdm(range(100), desc="Reading and preprocessing data", unit="%"):
        time.sleep(0.005)
    data = pd.read_csv('weather.csv')
    data = data.dropna().copy()
    data = data.sample(frac=0.2, random_state=42).copy()
    data = pd.get_dummies(data)

    if 'Apparent Temperature (C)' not in data.columns:
        raise ValueError("Expected 'Apparent Temperature (C)' column not found in dataset")

    X = data.drop('Apparent Temperature (C)', axis=1).values
    y = data['Apparent Temperature (C)'].values.reshape(-1, 1)

    print("Splitting data...")
    for _ in tqdm(range(100), desc="Splitting and scaling data", unit="%"):
        time.sleep(0.005)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # print("Starting hyperparameter search...")
    # best_model = hyperparameter_search(X_train, y_train, X_val, y_val)

    '''
        Uncomment the following lines to perform hyperparameter search

        The hyperparameter search resulted in the following parameters for the best model:
            hidden_layers = [16, 8]
            activations = ['sigmoid', 'sigmoid', 'linear']
            lr = 0.005
    '''
    hidden_layers = [16, 8]
    activations = ['sigmoid', 'sigmoid', 'linear']
    learning_rate = 0.005
    layer_sizes = [X_train.shape[1]] + hidden_layers + [1]

    print("Evaluating on test set...")
    best_model = NeuralNetwork(layer_sizes=layer_sizes, activations=activations, learning_rate=learning_rate)
    best_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=True)


    y_pred_test = best_model.predict(X_test)
    evaluate_regression(y_test, y_pred_test)
 
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
