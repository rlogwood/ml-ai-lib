"""
Notebook Setup Module

This module provides standardized setup for Jupyter notebooks including:
- All standard library imports (numpy, pandas, etc.)
- Deep learning imports (TensorFlow, Keras, etc.)
- Custom lib module imports
- Random seed initialization for reproducibility
- Environment information display

IMPORTANT: This module is only available AFTER the bootstrap process completes.

Usage in notebooks:
    # Step 1: Bootstrap (downloads lib if in Colab)
    BOOTSTRAP_URL = 'https://raw.githubusercontent.com/rlogwood/fs-ml-lib/main/colab_bootstrap.py'
    import urllib.request
    exec(urllib.request.urlopen(BOOTSTRAP_URL).read().decode())
    upload_lib(force_refresh=False)

    # Step 2: Setup imports (now lib is available)
    from lib.notebook_setup import setup_notebook
    setup_notebook()
"""

import os
import sys


def setup_notebook(show_versions=True):
    """
    Complete notebook setup including all imports and configuration.

    This function handles:
    - All standard and deep learning imports
    - Custom lib module imports and reloading
    - Random seed initialization for reproducibility
    - Optional version information display

    Parameters:
    -----------
    show_versions : bool, optional (default=True)
        If True, prints TensorFlow, Keras, and GPU availability information

    Example:
    --------
    >>> setup_notebook()
    >>> # Now use imports directly
    >>> np.random.rand(5)
    >>> tu.print_heading("My Heading")
    >>> plt.plot([1, 2, 3])
    """
    # Standard imports
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd
    import zipfile

    # Deep Learning imports
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                         GlobalAveragePooling2D, Flatten, Input)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50

    # Sklearn imports
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image

    from IPython.display import display, HTML

    # Public git repo fs-ml-lib imports
    # For local development clone the repo from:
    # https://github.com/rlogwood/fs-ml-lib.git
    import lib.text_util as tu
    import lib.wrangler as wr
    import lib.data_cleaner as dc
    import lib.analyzer as da
    import lib.corr_analysis as ca
    import lib.utility as utl
    import lib.data_downloader as ddl

    # Reload modules after code changes
    from lib.colab_bootstrap import reload_lib_modules
    reload_lib_modules()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Display version information if requested
    if show_versions:
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {keras.__version__}")
        print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

    # Make imports available in the calling scope
    # This allows direct usage like: np.array(), tu.print_heading(), etc.
    import inspect
    frame = inspect.currentframe().f_back
    frame.f_globals.update({
        'np': np,
        'pd': pd,
        'tf': tf,
        'keras': keras,
        'tu': tu,
        'wr': wr,
        'dc': dc,
        'da': da,
        'ca': ca,
        'utl': utl,
        'ddl': ddl,
        'plt': plt,
        'sns': sns,
        'os': os,
        'zipfile': zipfile,
        'warnings': warnings,
        'Sequential': Sequential,
        'Model': Model,
        'Dense': Dense,
        'Dropout': Dropout,
        'BatchNormalization': BatchNormalization,
        'GlobalAveragePooling2D': GlobalAveragePooling2D,
        'Flatten': Flatten,
        'Input': Input,
        'Adam': Adam,
        'EarlyStopping': EarlyStopping,
        'ReduceLROnPlateau': ReduceLROnPlateau,
        'ModelCheckpoint': ModelCheckpoint,
        'ImageDataGenerator': ImageDataGenerator,
        'VGG16': VGG16,
        'MobileNetV2': MobileNetV2,
        'ResNet50': ResNet50,
        'confusion_matrix': confusion_matrix,
        'classification_report': classification_report,
        'accuracy_score': accuracy_score,
        'Image': Image,
        'display': display,
        'HTML': HTML
    })


# Alias for backwards compatibility and clarity
setup_notebook_imports = setup_notebook
