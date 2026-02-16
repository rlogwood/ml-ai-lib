"""
Model Training Module for Deep Learning Models

This module provides functions for building, training, and configuring
neural network models for loan default prediction.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

try:
    from . import text_util as tu
except ImportError:
    import text_util as tu


def calculate_class_weights(y_train):
    """
    Calculate balanced class weights for imbalanced datasets.

    Parameters:
    -----------
    y_train : array-like
        Training labels

    Returns:
    --------
    dict
        Dictionary mapping class labels to weights
    numpy.ndarray
        Array of class weights

    Examples:
    ---------
    >>> class_weight_dict, class_weights = calculate_class_weights(y_train)
    >>> print(f"Class 0 weight: {class_weight_dict[0]:.4f}")
    """
    tu.print_heading("CLASS WEIGHT CALCULATION")

    # Calculate class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )

    # Create dictionary
    class_weight_dict = dict(zip(classes, class_weights))

    print("\nCalculated Class Weights:")
    print(f"  Class 0 (Paid):    {class_weights[0]:.4f}")
    print(f"  Class 1 (Default): {class_weights[1]:.4f}")
    print(f"\n  Weight Ratio (1/0): {class_weights[1]/class_weights[0]:.2f}x")

    print("\n  Interpretation: The model will penalize misclassifying")
    print(f"  a default {class_weights[1]/class_weights[0]:.2f}x more than misclassifying a paid loan.")

    return class_weight_dict, class_weights


def build_neural_network(input_dim, layers=[32, 16], dropout_rate=0.3, learning_rate=0.001):
    """
    Build a sequential neural network for binary classification.

    Parameters:
    -----------
    input_dim : int
        Number of input features
    layers : list
        List of integers specifying units in each hidden layer
    dropout_rate : float
        Dropout rate for regularization (0.0 to 1.0)
    learning_rate : float
        Learning rate for optimizer

    Returns:
    --------
    keras.Model
        Compiled Keras model

    Examples:
    ---------
    >>> model = build_neural_network(input_dim=17, layers=[64, 32], dropout_rate=0.3)
    >>> model.summary()
    """
    from tensorflow import keras
    from tensorflow.keras import layers as keras_layers

    tu.print_heading("BUILDING NEURAL NETWORK MODEL")

    # Clear any previous models
    keras.backend.clear_session()

    # Create sequential model
    model = keras.Sequential(name='sequential')

    # Add input layer and first hidden layer
    model.add(keras_layers.Dense(
        layers[0],
        activation='relu',
        input_shape=(input_dim,),
        name='hidden_layer_1'
    ))
    model.add(keras_layers.Dropout(dropout_rate, name='dropout_1'))

    # Add additional hidden layers
    for i, units in enumerate(layers[1:], start=2):
        model.add(keras_layers.Dense(
            units,
            activation='relu',
            name=f'hidden_layer_{i}'
        ))
        model.add(keras_layers.Dropout(dropout_rate, name=f'dropout_{i}'))

    # Output layer
    model.add(keras_layers.Dense(1, activation='sigmoid', name='output'))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    print("\nModel Architecture:\n")
    model.summary()

    print(tu.bold_and_colored_text("\nMODEL READY FOR TRAINING", tu.Color.GREEN))

    return model


def create_early_stopping(patience=5, monitor='val_auc', mode='max', verbose=1):
    """
    Create early stopping callback for training.

    Parameters:
    -----------
    patience : int
        Number of epochs with no improvement after which training will be stopped
    monitor : str
        Metric to monitor
    mode : str
        'min', 'max', or 'auto'
    verbose : int
        Verbosity mode

    Returns:
    --------
    keras.callbacks.EarlyStopping
        Early stopping callback

    Examples:
    ---------
    >>> early_stop = create_early_stopping(patience=5, monitor='val_auc')
    >>> history = model.fit(X, y, callbacks=[early_stop])
    """
    from tensorflow.keras.callbacks import EarlyStopping

    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        mode=mode,
        verbose=verbose
    )


def train_model_with_class_weights(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
    epochs=50,
    batch_size=256,
    callbacks=None,
    verbose=1
):
    """
    Train a neural network model with class weights.

    Parameters:
    -----------
    model : keras.Model
        Compiled Keras model
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    class_weights : dict
        Dictionary mapping class labels to weights
    epochs : int
        Maximum number of epochs
    batch_size : int
        Batch size for training
    callbacks : list
        List of Keras callbacks
    verbose : int
        Verbosity mode

    Returns:
    --------
    keras.callbacks.History
        Training history object

    Examples:
    ---------
    >>> history = train_model_with_class_weights(
    ...     model, X_train, y_train, X_val, y_val,
    ...     class_weights={0: 0.59, 1: 3.12},
    ...     epochs=50
    ... )
    """
    tu.print_heading("TRAINING MODEL")

    print(f"\nTraining with class weights: {class_weights}")
    print("This will force the model to learn from minority class...\n")

    if callbacks is None:
        callbacks = []

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose
    )

    return history


def get_model_config_name(layers, dropout, lr):
    """
    Generate a descriptive name for a model configuration.

    Parameters:
    -----------
    layers : list
        List of layer sizes
    dropout : float
        Dropout rate
    lr : float
        Learning rate

    Returns:
    --------
    str
        Configuration name
    """
    layer_str = f"{layers[0]}-{layers[1]}"
    return f"NN_{layer_str}_d{dropout}_lr{lr}"


if __name__ == "__main__":
    # Example usage
    print("Model Training Module")
    print("=" * 50)
    print("\nThis module provides neural network training functions.")
    print("\nAvailable functions:")
    print("  - calculate_class_weights(y_train)")
    print("  - build_neural_network(input_dim, layers, dropout_rate, learning_rate)")
    print("  - create_early_stopping(patience, monitor, mode)")
    print("  - train_model_with_class_weights(...)")
    print("  - get_model_config_name(layers, dropout, lr)")
