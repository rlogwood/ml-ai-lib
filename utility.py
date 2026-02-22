import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    # When imported as part of a package
    from . import text_util as tu
except ImportError:
    # When run as a standalone script
    import text_util as tu


def inspect_variable(var, var_name="variable"):
    print(f"=== {var_name} ===")
    print(f"Class: {var.__class__}")
    print(f"Class Name: {var.__class__.__name__}")
    print(f"Type: {type(var)}")
    print(f"Type name: {type(var).__name__}")
    print(f"String representation: {str(var)[:100]}...")
    if hasattr(var, 'shape'):
        print(f"Shape: {var.shape}")
    print()


def show_breakdown(df, col1, col2, from_val: float = None, to_val: float = None):
    def print_summary(title):
        tu.print_sub_heading(title)
        # summary = df.groupby(col1)[col2].value_counts().unstack(fill_value=0)

        if from_val is not None:
            if to_val is not None:
                # print(summary.loc[from_val:to_val,:])
                print(tu.bold_text(f'Breakdown between {from_val} and {to_val}'))
                print(summary.loc[(summary.index > from_val) & (summary.index <= to_val)])
                print(tu.italic_text(tu.bold_text('Total')))
                print(summary.loc[(summary.index > from_val) & (summary.index <= to_val)].sum())
            else:
                print(tu.bold_text(f'Breakdown from {from_val}'))
                print(summary.loc[(summary.index > from_val)])
                print(tu.italic_text(tu.bold_text('Total')))
                print(summary.loc[(summary.index > from_val)].sum())
        else:
            print(summary)

    tu.print_heading(f'summary of {col1} vs {col2}')
    summary = pd.crosstab(df[col1], df[col2])
    print_summary('crosstab')

    # summary = df.groupby(col1)[col2].value_counts().unstack(fill_value=0)
    # print_summary('group_by')
    #
    # summary = df.pivot_table(index=col1, columns=col2, aggfunc='size', fill_value=0)
    # print_summary('pivot_table')


def show_env():
    import os
    from textwrap import wrap

    env_vars = dict(os.environ)

    print("\n" + "=" * 100)
    print(f"{'ENVIRONMENT VARIABLES':^100}")
    print("=" * 100)
    print(f"Total: {len(env_vars)} variables\n")

    for key in sorted(env_vars.keys()):
        value = env_vars[key]

        # Wrap long values across multiple lines with indentation
        if len(value) > 80:
            wrapped = wrap(value, width=80)
            print(f"\033[1m{key}\033[0m:")
            for line in wrapped:
                print(f"  {line}")
            print()
        else:
            print(f"\033[1m{key:<35}\033[0m = {value}")

    print("=" * 100 + "\n")


def get_predictions(model, X, threshold=0.5, verbose=0):
    """
    Get class predictions and probabilities from any model type.

    This function works with both sklearn-style classifiers (with predict_proba)
    and Keras/TensorFlow models (where predict returns probabilities).

    Parameters:
    -----------
    model : object
        Trained classifier model (sklearn or Keras)
    X : array-like
        Input features for prediction
    threshold : float, default=0.5
        Probability threshold for binary classification
    verbose : int, default=0
        Verbosity level for Keras models (0=silent, 1=progress bar)

    Returns:
    --------
    y_pred : ndarray
        Binary class predictions (0 or 1)
    y_pred_proba : ndarray
        Probability of positive class (1D array)

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier().fit(X_train, y_train)
    >>> y_pred, y_proba = get_predictions(model, X_test)

    >>> from tensorflow import keras
    >>> model = keras.Sequential([...])
    >>> model.fit(X_train, y_train)
    >>> y_pred, y_proba = get_predictions(model, X_test, verbose=0)
    """
    if hasattr(model, 'predict_proba'):
        # sklearn-style model with probability support
        try:
            y_pred_proba = model.predict_proba(X)[:, 1]
        except (TypeError, IndexError):
            # Mock object or non-standard predict_proba - fall back to predict
            try:
                y_pred_proba = model.predict(X, verbose=verbose).flatten()
            except TypeError:
                y_pred_proba = model.predict(X).flatten()
    else:
        # Keras/TensorFlow model - predict returns probabilities
        try:
            # Try with verbose parameter (Keras models)
            y_pred_proba = model.predict(X, verbose=verbose).flatten()
        except TypeError:
            # Fall back without verbose (sklearn or other models)
            y_pred_proba = model.predict(X).flatten()

    # Apply threshold to get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    return y_pred, y_pred_proba


@dataclass
class ModelArchitectureInfo:
    """
    Generic model architecture information.

    This dataclass provides a structured way to describe any model type
    (Keras, sklearn, XGBoost, etc.) with consistent attributes.

    Attributes:
    -----------
    model_type : str
        Class name of the model (e.g., "Sequential", "RandomForestClassifier")
    model_family : str
        Broad category of model: "neural_network", "tree_ensemble",
        "linear", "gradient_boosting", or "unknown"
    n_parameters : int, optional
        Total number of trainable parameters (neural networks)
    n_layers : int, optional
        Number of layers (neural networks)
    input_shape : tuple, optional
        Input shape for neural networks
    config : dict, optional
        Key hyperparameters and configuration settings
    layers : list, optional
        List of layer descriptions for neural networks

    Examples:
    ---------
    >>> info = ModelArchitectureInfo(
    ...     model_type="Sequential",
    ...     model_family="neural_network",
    ...     n_parameters=1234,
    ...     n_layers=5
    ... )
    >>> print(info)
    Sequential (neural_network), 5 layers, 1,234 parameters
    """
    model_type: str
    model_family: str
    n_parameters: Optional[int] = None
    n_layers: Optional[int] = None
    input_shape: Optional[tuple] = None
    config: Optional[Dict[str, Any]] = None
    layers: Optional[list] = None

    def layer_summary(self):
        summary = ""
        for layer in self.layers:
            if hasattr(layer, 'units'):
                units = f":{layer_info['units']}"
            else:
                units = ""

            if hasattr(layer, 'output_shape') and layer['output_shape'] is not None:
                output_shape = f"({layer['output_shape']})"
            else:
                output_shape = ""
            summary += f"  - {layer['name']} ({layer['type']}{units}) {output_shape}\n"
        return summary

    def __str__(self) -> str:
        """
        Human-readable description of the model architecture.

        Returns:
        --------
        str
            Formatted string describing the model
        """
        parts = [f"{self.model_type} ({self.model_family})"]
        if self.n_parameters:
            parts.append(f"{self.n_parameters:,} parameters")

        if self.n_layers:
            parts.append(f"{self.n_layers} layers")
            parts.append(f"\n{self.layer_summary()}")

        if self.config:
            # Show up to 3 key config items
            config_items = [(k, v) for k, v in self.config.items() if v is not None][:3]
            if config_items:
                config_str = ', '.join(f"{k}={v}" for k, v in config_items)
                parts.append(config_str)

        return ', '.join(parts)


def get_model_architecture_info(model) -> ModelArchitectureInfo:
    """
    Extract architecture information from any model type.

    This function works with Keras/TensorFlow models, sklearn models,
    XGBoost models, and provides generic fallback for other types.

    Parameters:
    -----------
    model : object
        Trained model (Keras, sklearn, XGBoost, etc.)

    Returns:
    --------
    ModelArchitectureInfo
        Structured model information with type, family, and configuration

    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(n_estimators=100, max_depth=10)
    >>> info = get_model_architecture_info(model)
    >>> print(info)
    RandomForestClassifier (tree_ensemble), n_estimators=100, max_depth=10

    >>> from tensorflow import keras
    >>> model = keras.Sequential([...])
    >>> info = get_model_architecture_info(model)
    >>> print(info)
    Sequential (neural_network), 5 layers, 1,234 parameters
    """
    model_type = type(model).__name__

    # Keras/TensorFlow models
    if hasattr(model, 'layers'):
        config = {}
        if hasattr(model, 'optimizer'):
            config['optimizer'] = model.optimizer.__class__.__name__
        if hasattr(model, 'loss'):
            loss_name = model.loss if isinstance(model.loss, str) else model.loss.__class__.__name__
            config['loss'] = loss_name

        # Extract layer information
        layer_descriptions = []
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None,
            }

            # Add units for Dense layers
            if hasattr(layer, 'units'):
                layer_info['units'] = layer.units

            # Add activation for layers that have it
            if hasattr(layer, 'activation') and layer.activation is not None:
                activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
                layer_info['activation'] = activation_name

            # Add dropout rate
            if hasattr(layer, 'rate'):
                layer_info['rate'] = layer.rate

            layer_descriptions.append(layer_info)

        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='neural_network',
            n_parameters=model.count_params() if hasattr(model, 'count_params') else None,
            n_layers=len(model.layers),
            input_shape=model.input_shape if hasattr(model, 'input_shape') else None,
            config=config if config else None,
            layers=layer_descriptions
        )

    # sklearn tree-based models
    elif 'Forest' in model_type or 'Tree' in model_type:
        params = model.get_params() if hasattr(model, 'get_params') else {}
        config = {
            k: v for k, v in {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'min_samples_split': params.get('min_samples_split')
            }.items() if v is not None
        }
        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='tree_ensemble',
            config=config if config else None
        )

    # sklearn linear models
    elif 'Logistic' in model_type or 'Linear' in model_type or 'Ridge' in model_type:
        params = model.get_params() if hasattr(model, 'get_params') else {}
        config = {
            k: v for k, v in {
                'C': params.get('C'),
                'penalty': params.get('penalty'),
                'solver': params.get('solver')
            }.items() if v is not None
        }
        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='linear',
            config=config if config else None
        )

    # XGBoost models
    elif 'XGB' in model_type:
        params = model.get_params() if hasattr(model, 'get_params') else {}
        config = {
            k: v for k, v in {
                'n_estimators': params.get('n_estimators'),
                'max_depth': params.get('max_depth'),
                'learning_rate': params.get('learning_rate')
            }.items() if v is not None
        }
        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='gradient_boosting',
            config=config if config else None
        )

    # Generic fallback for unknown model types
    else:
        config = None
        if hasattr(model, 'get_params'):
            all_params = model.get_params()
            # Try to extract a few interesting parameters
            interesting_keys = ['n_estimators', 'max_depth', 'C', 'alpha', 'learning_rate']
            config = {k: v for k, v in all_params.items() if k in interesting_keys and v is not None}

        return ModelArchitectureInfo(
            model_type=model_type,
            model_family='unknown',
            config=config if config else None
        )
