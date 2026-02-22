import pandas as pd

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
