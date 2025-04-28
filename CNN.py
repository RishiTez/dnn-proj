import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product
from tqdm import tqdm
import time
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


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
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

class CNNRegressor:
    def __init__(self, input_shape, filters=None, kernel_sizes=None, dense_layers=None, 
                 learning_rate=0.001, activation='relu', dropout_rate=0.2):
        print("Initializing CNN Regressor...")
        
        # Default architecture parameters if not provided
        if filters is None:
            filters = [8, 16]
        if kernel_sizes is None:
            kernel_sizes = [3, 3]
        if dense_layers is None:
            dense_layers = [8, 16]
            
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dense_layers = dense_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Build the model
        self.model = self._build_model()
        print("Initialization complete.")
        
    def _build_model(self):
        model = Sequential()
        
        # Add Conv1D layers
        model.add(Conv1D(filters=self.filters[0], 
                         kernel_size=self.kernel_sizes[0],
                         activation=self.activation,
                         input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        
        # Add additional Conv1D layers if needed
        for f, k in zip(self.filters[1:], self.kernel_sizes[1:]):
            model.add(Conv1D(filters=f, kernel_size=k, activation=self.activation))
            model.add(MaxPooling1D(pool_size=2))
        
        # Flatten and add dense layers
        model.add(Flatten())
        
        for units in self.dense_layers:
            model.add(Dense(units, activation=self.activation))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer for regression
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, verbose=1):
        print("Starting training...")
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
        ]
        
        # If validation data is provided
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Training complete.")
        return history
    
    def predict(self, X):
        print("Making predictions...")
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return evaluate_regression(y_true, y_pred)
    
# Function to reshape data for CNN
def reshape_for_cnn(X, time_steps=10):
    """
    Reshape tabular data to be suitable for CNN (samples, time_steps, features)
    
    Parameters:
    - X: Original data (2D array)
    - time_steps: Number of time steps for CNNs
    
    Returns:
    - Reshaped data suitable for CNN input
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # If samples are not sufficient for reshaping, pad with zeros
    if n_samples % time_steps != 0:
        padding_size = time_steps - (n_samples % time_steps)
        padding = np.zeros((padding_size, n_features))
        X = np.vstack([X, padding])
        n_samples = X.shape[0]
    
    # Create subsequences: reshape to (n_subsequences, time_steps, n_features)
    n_subsequences = n_samples // time_steps
    X_reshaped = X.reshape(n_subsequences, time_steps, n_features)
    
    return X_reshaped

# Grid search for hyperparameter tuning
def hyperparameter_search(X_train, y_train, X_val, y_val, time_steps=10):
    filter_options = [[8, 16], [16, 8], [8, 16, 32]]
    kernel_options = [[3, 3], [5, 3], [3, 3, 3]]
    dense_options = [[8, 16], [16, 8], [8, 16, 32]]
    learning_rates = [0.001, 0.005, 0.01]
    
    # Reshape data for CNN
    X_train_cnn = reshape_for_cnn(X_train, time_steps)
    X_val_cnn = reshape_for_cnn(X_val, time_steps)
    
    # Adjust target data to match CNN input shape (if necessary)
    y_train_cnn = y_train[:X_train_cnn.shape[0] * time_steps].reshape(X_train_cnn.shape[0], time_steps, 1)
    y_train_cnn = y_train_cnn.mean(axis=1)  # Take average temperature for each sequence
    
    y_val_cnn = y_val[:X_val_cnn.shape[0] * time_steps].reshape(X_val_cnn.shape[0], time_steps, 1)
    y_val_cnn = y_val_cnn.mean(axis=1)  # Take average temperature for each sequence
    
    best_loss = float('inf')
    best_model = None
    best_params = {}
    
    input_shape = (time_steps, X_train.shape[1])
    
    for filters, kernels, dense, lr in product(filter_options, kernel_options, dense_options, learning_rates):
        # Make sure filter and kernel lists have the same length
        if len(filters) != len(kernels):
            continue
            
        print(f"\n--- Trying Configuration: Filters={filters}, Kernels={kernels}, Dense={dense}, LR={lr} ---")
        
        model = CNNRegressor(
            input_shape=input_shape,
            filters=filters,
            kernel_sizes=kernels,
            dense_layers=dense,
            learning_rate=lr
        )
        
        model.fit(X_train_cnn, y_train_cnn, X_val_cnn, y_val_cnn, epochs=20, batch_size=32, verbose=1)
        
        # Evaluate model on validation set
        metrics = model.evaluate(X_val_cnn, y_val_cnn)
        loss = metrics['mse']
        
        print(f"Validation Loss: {loss:.4f}")
        
        if loss < best_loss:
            print("New best model found!")
            best_loss = loss
            best_model = model
            best_params = {'filters': filters, 'kernels': kernels, 'dense': dense, 'lr': lr}
        
        # Clean up memory
        tf.keras.backend.clear_session()
        del model
        gc.collect()
    
    print("\nBest Hyperparameters:", best_params)
    return best_model, best_params


# Main execution
def main():
    print("Loading data...")
    data = pd.read_csv('./weather.csv')
    data = data.dropna().copy()
    data = data.sample(frac=0.2, random_state=42).copy()
    data = pd.get_dummies(data)

    if 'Apparent Temperature (C)' not in data.columns:
        raise ValueError("Expected 'Apparent Temperature (C)' column not found in dataset")

    X = data.drop('Apparent Temperature (C)', axis=1).values
    y = data['Apparent Temperature (C)'].values.reshape(-1, 1)

    print("Splitting data...")

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    print("Scaling data")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Define time steps for CNN reshaping
    time_steps = 10
    
    # Reshape data for CNN
    print("Reshaping data for CNN...")
    X_train_cnn = reshape_for_cnn(X_train, time_steps)
    X_val_cnn = reshape_for_cnn(X_val, time_steps)
    X_test_cnn = reshape_for_cnn(X_test, time_steps)
    
    # Adjust target data to match CNN input shape
    print("Adjusting target data for CNN...")
    y_train_cnn = y_train[:X_train_cnn.shape[0] * time_steps].reshape(X_train_cnn.shape[0], time_steps, 1)
    y_train_cnn = y_train_cnn.mean(axis=1)  # Take average temperature for each sequence
    
    y_val_cnn = y_val[:X_val_cnn.shape[0] * time_steps].reshape(X_val_cnn.shape[0], time_steps, 1)
    y_val_cnn = y_val_cnn.mean(axis=1)
    
    y_test_cnn = y_test[:X_test_cnn.shape[0] * time_steps].reshape(X_test_cnn.shape[0], time_steps, 1)
    y_test_cnn = y_test_cnn.mean(axis=1)
    
    input_shape = (time_steps, X_train.shape[1])
    
    # Uncomment to run hyperparameter search
    print("Starting hyperparameter search...")
    best_model, best_params = hyperparameter_search(X_train, y_train, X_val, y_val, time_steps)
    
    # Based on best hyperparameters (example values, you can replace with results from your search)
    best_filters = best_params['filters']
    best_kernels = best_params['kernels']
    best_dense = best_params['dense']
    best_learning_rate = best_params['lr']
    
    print(f"Training final model with best hyperparameters...")
    final_model = CNNRegressor(
        input_shape=input_shape,
        filters=best_filters,
        kernel_sizes=best_kernels,
        dense_layers=best_dense,
        learning_rate=best_learning_rate
    )
    
    # Train the model
    history = final_model.fit(
        X_train_cnn, y_train_cnn, 
        X_val_cnn, y_val_cnn,
        epochs=50, 
        batch_size=32,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    final_metrics = final_model.evaluate(X_test_cnn, y_test_cnn)
    
    # Get predictions for plotting if needed
    y_pred_test = final_model.predict(X_test_cnn)
    
    print("CNN model training and evaluation complete.")
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
