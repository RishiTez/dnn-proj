import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score
from tqdm import tqdm
import time

# Build adjacency matrix using cosine similarity and k-nearest neighbors
def build_adjacency_matrix(features, k=5):
    similarity = cosine_similarity(features)
    adj = np.zeros_like(similarity)
    for i in range(similarity.shape[0]):
        neighbors = np.argsort(similarity[i])[-(k+1):]
        adj[i, neighbors] = 1
    np.fill_diagonal(adj, 1)
    return adj

# Normalize adjacency matrix: D^(-1/2) @ A @ D^(-1/2)
def normalize_adj(adj):
    D = np.diag(1.0 / np.sqrt(np.sum(adj, axis=1)))
    return D @ adj @ D

# Single GCN layer
class GCNLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01

    def forward(self, X, A_hat):
        self.X = X
        self.A_hat = A_hat
        return A_hat @ X @ self.W

    def backward(self, dZ, lr):
        dW = self.X.T @ self.A_hat.T @ dZ
        self.W -= lr * dW
        return dZ @ self.W.T

# Two-layer GCN model
class GCN:
    def __init__(self, in_features, hidden_features, out_features):
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_features)

    def forward(self, X, A_hat):
        self.Z1 = np.tanh(self.gcn1.forward(X, A_hat))
        self.Z2 = self.gcn2.forward(self.Z1, A_hat)
        return self.Z2

    def backward(self, loss_grad, lr):
        dZ2 = loss_grad
        dZ1 = self.gcn2.backward(dZ2, lr)
        _ = self.gcn1.backward(dZ1 * (1 - self.Z1**2), lr)

# Mean Squared Error loss and its gradient
def mse_loss(pred, true):
    return np.mean((pred - true) ** 2), 2 * (pred - true) / len(true)

# Training function
def train(X, y, A, hidden_dim, lr, epochs, verbose=False):
    A_hat = normalize_adj(A)
    model = GCN(X.shape[1], hidden_dim, 1)
    losses = []

    if verbose:
        print(f"\nTraining model (Hidden={hidden_dim}, LR={lr}, Epochs={epochs})...")

    pbar = tqdm(range(epochs), desc="Training", leave=False, ncols=100, disable=not verbose)

    for epoch in pbar:
        pred = model.forward(X, A_hat).flatten()
        loss, grad = mse_loss(pred, y)
        model.backward(grad.reshape(-1, 1), lr)
        losses.append(loss)

        # Show status every 50 epochs
        if verbose and (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

    return model, losses[-1]

# Main function
def main():
    # Toggle this to enable/disable hyperparameter tuning
    # perform_tuning = True
    perform_tuning = False

    print("Loading dataset...")
    df = pd.read_csv('weather.csv').head(6000)  # Limit for memory
    df = df.drop(columns=["Formatted Date", "Daily Summary"])

    print("Preprocessing...")
    df['Summary'] = LabelEncoder().fit_transform(df['Summary'])
    df['Precip Type'] = LabelEncoder().fit_transform(df['Precip Type'])

    X = df.drop(columns=["Apparent Temperature (C)"])
    y = df["Apparent Temperature (C)"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = y.to_numpy()

    print("Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Building graph (adjacency matrix)...")
    A_train = build_adjacency_matrix(X_train, k=3)

    if perform_tuning:
        print("Starting hyperparameter tuning...")
        best_loss = float('inf')
        best_params = None
        best_model = None
        param_grid = [(h, lr, ep)
                      for h in [4, 8, 16, 32]
                      for lr in [0.001, 0.01]
                      for ep in [100, 300]]

        for hidden, lr, epoch in tqdm(param_grid, desc="Tuning Models", ncols=100):
            model, loss = train(X_train, y_train, A_train, hidden, lr, epoch, verbose=False)
            if loss < best_loss:
                best_loss = loss
                best_params = (hidden, lr, epoch)
                best_model = model

        print("\nBest hyperparameters:")
        print(f"   Hidden: {best_params[0]}, LR: {best_params[1]}, Epochs: {best_params[2]}")
        print(f"   Training Loss (MSE): {best_loss:.4f}")
    else:
        print("Skipping hyperparameter tuning. Using default settings.")
        hidden, lr, epoch = 32, 0.01, 500
        best_model, best_loss = train(X_train, y_train, A_train, hidden, lr, epoch, verbose=True)
        best_params = (hidden, lr, epoch)

    print("\nEvaluating on test set...")
    A_test = build_adjacency_matrix(X_test, k=3)
    A_hat_test = normalize_adj(A_test)
    y_pred = best_model.forward(X_test, A_hat_test).flatten()

    test_loss = np.mean((y_pred - y_test) ** 2)
    r2 = r2_score(y_test, y_pred)

    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
