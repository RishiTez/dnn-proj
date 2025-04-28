import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to preprocess the data
def preprocess_data(df):
    # Convert date to datetime and extract features
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df['hour'] = df['Formatted Date'].dt.hour
    df['day'] = df['Formatted Date'].dt.day
    df['month'] = df['Formatted Date'].dt.month
    df['dayofweek'] = df['Formatted Date'].dt.dayofweek
    
    # Handle categorical variables
    categorical_cols = ['Summary', 'Precip Type', 'Daily Summary']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna('None'))
    
    # Drop the original date column and any other non-useful columns
    df = df.drop(['Formatted Date'], axis=1)
    
    return df

# Function to create sequences for CNN
def create_sequences(X, y, time_steps=24):
    """Create sequences for time series prediction."""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps - 1])
    
    return np.array(X_seq), np.array(y_seq)

# Function to evaluate model performance
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

# Function to build the CNN model
def build_cnn_model(input_shape, filters=64, kernel_size=3, dense_units=32, 
                    dropout_rate=0.2, learning_rate=0.001):
    """Build a CNN model with the specified parameters."""
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
                     input_shape=input_shape, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    # Second convolutional layer
    model.add(Conv1D(filters=filters*2, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer for regression
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Function to perform hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape):
    """Perform a simplified grid search for hyperparameters."""
    # Define hyperparameter grid
    filters_options = [32, 64]
    kernel_sizes = [3, 5]
    dense_units_options = [32, 64]
    dropout_rates = [0.2, 0.3]
    learning_rates = [0.001, 0.0005]
    
    best_val_loss = float('inf')
    best_params = {}
    
    total_combinations = (len(filters_options) * len(kernel_sizes) * 
                         len(dense_units_options) * len(dropout_rates) * 
                         len(learning_rates))
    
    print(f"Starting hyperparameter tuning with {total_combinations} combinations")
    
    # Track all results
    results = []
    
    # Simple grid search
    for filters in filters_options:
        for kernel_size in kernel_sizes:
            for dense_units in dense_units_options:
                for dropout_rate in dropout_rates:
                    for lr in learning_rates:
                        print(f"\nTesting: filters={filters}, kernel_size={kernel_size}, "
                              f"dense_units={dense_units}, dropout={dropout_rate}, lr={lr}")
                        
                        # Build and train model
                        model = build_cnn_model(
                            input_shape=input_shape,
                            filters=filters,
                            kernel_size=kernel_size,
                            dense_units=dense_units,
                            dropout_rate=dropout_rate,
                            learning_rate=lr
                        )
                        
                        # Use early stopping for faster training
                        callbacks = [
                            EarlyStopping(patience=5, restore_best_weights=True)
                        ]
                        
                        history = model.fit(
                            X_train, y_train,
                            epochs=15,  # Reduced epochs for faster tuning
                            batch_size=64,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=0
                        )
                        
                        # Evaluate the model
                        val_loss = min(history.history['val_loss'])
                        
                        # Record results
                        results.append({
                            'filters': filters,
                            'kernel_size': kernel_size,
                            'dense_units': dense_units,
                            'dropout_rate': dropout_rate,
                            'learning_rate': lr,
                            'val_loss': val_loss
                        })
                        
                        print(f"Validation loss: {val_loss:.4f}")
                        
                        # Check if this is the best model so far
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = {
                                'filters': filters,
                                'kernel_size': kernel_size,
                                'dense_units': dense_units,
                                'dropout_rate': dropout_rate,
                                'learning_rate': lr
                            }
                            
                        # Clean up to free memory
                        tf.keras.backend.clear_session()
    
    # Sort and display results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_loss')
    print("\nTop 5 hyperparameter combinations:")
    print(results_df.head())
    
    print("\nBest hyperparameters:")
    print(best_params)
    
    return best_params

# Main function to run the full pipeline
def main():
    # Load the dataset
    print("Loading and preprocessing data...")
    df = pd.read_csv('weather.csv')
    
    # Check for missing values
    print(f"Missing values before preprocessing:\n{df.isnull().sum()}")
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Define target and features
    target_col = 'Apparent Temperature (C)'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create sequences for CNN
    time_steps = 24  # Use 24 hours as a sequence
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
    
    print(f"Training data shape: {X_train_seq.shape}")
    print(f"Validation data shape: {X_val_seq.shape}")
    print(f"Test data shape: {X_test_seq.shape}")
    
    # Get input shape for the model
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    
    # Hyperparameter tuning - comment out if not needed
    best_params = hyperparameter_tuning(X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_shape)
    
    # Build the final model with the best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    final_model = build_cnn_model(
        input_shape=input_shape,
        filters=best_params['filters'],
        kernel_size=best_params['kernel_size'],
        dense_units=best_params['dense_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )
    
    # Define callbacks for final training
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
    ]
    
    # Train the final model
    history = final_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=64,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating final model on test data...")
    mae, mse, rmse, r2 = evaluate_model(final_model, X_test_seq, y_test_seq)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Make predictions on test data
    y_pred = final_model.predict(X_test_seq)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_seq[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title('Actual vs Predicted Apparent Temperature (First 100 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Apparent Temperature (C)')
    plt.legend()
    plt.savefig('predictions.png')
    plt.close()
    
    # Save the final model
    final_model.save('weather_cnn_model.h5')
    print("Model saved as 'weather_cnn_model.h5'")
    
    # Print summary of the model
    print("\nModel Summary:")
    final_model.summary()
    
    return final_model, history, scaler

if __name__ == "__main__":
    main()
