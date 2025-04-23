import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions and their derivatives
def relu(Z): return np.maximum(0, Z)
def relu_derivative(Z): return Z > 0

def sigmoid(Z): return 1 / (1 + np.exp(-Z))
def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def tanh(Z): return np.tanh(Z)
def tanh_derivative(Z): return 1 - np.tanh(Z)**2

activation_functions = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative)
}

# Loss functions
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return (y_pred - y_true)

class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01, task='regression'):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.task = task
        self.weights = []
        self.biases = []
        self.activations = activations
        
        # Initialize weights and biases
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
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

    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True):
        loss_history = []
        m = X.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred, cache = self.forward(X_batch)
                grads = self.backward(X_batch, y_batch, cache)
                self.update_parameters(grads)

            # Calculate loss after epoch
            y_pred_full, _ = self.forward(X)
            loss = mean_squared_error(y, y_pred_full)
            loss_history.append(loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} / {epochs}, Loss: {loss:.4f}")

        return loss_history

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

# Example usage
if __name__ == "__main__":
    data = pd.read_csv('weather.csv')
    data = data.dropna()
    data = pd.get_dummies(data)

    X = data.drop('Temperature', axis=1).values
    y = data['Temperature'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameters
    layer_sizes = [X_train.shape[1], 128, 64, 1]
    activations = ['relu', 'tanh', 'relu']
    learning_rate = 0.01

    nn = NeuralNetwork(layer_sizes=layer_sizes, activations=activations, learning_rate=learning_rate)
    loss_history = nn.fit(X_train, y_train, epochs=1000, batch_size=32)

    # Plot training loss
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    # Evaluate
    y_pred = nn.predict(X_test)
    test_loss = mean_squared_error(y_test, y_pred)
    print(f"\nFinal Test Loss (MSE): {test_loss:.4f}")
