"""
Feature Engineering Module for Loan Data Analysis

This module provides functions for creating derived features from loan data
to improve model performance.
"""

import pandas as pd
import numpy as np

try:
    from . import text_util as tu
except ImportError:
    import text_util as tu


def create_loan_features(df):
    """
    Create engineered features for loan default prediction.

    Creates the following features:
    - credit_util_ratio: Normalized revolving credit utilization
    - annual_inquiry_rate: Projected annual credit inquiries
    - debt_burden: Monthly payment relative to annual income
    - credit_history_years: Credit line age in years
    - high_debt: Binary flag for high-risk debt indicators
    - risk_score: Composite risk indicator

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing loan data with required columns:
        - revol.util
        - inq.last.6mths
        - installment
        - log.annual.inc
        - days.with.cr.line
        - dti

    Returns:
    --------
    pd.DataFrame
        DataFrame with 6 new engineered features added

    Examples:
    ---------
    >>> df_enhanced = create_loan_features(df)
    >>> print(df_enhanced.columns)
    """
    tu.print_heading("FEATURE ENGINEERING")

    df_fe = df.copy()

    print("\nCreating new features...")

    # 1. Credit Utilization Ratio (normalized)
    df_fe['credit_util_ratio'] = df_fe['revol.util'] / 100
    print("  + credit_util_ratio: Normalized revolving utilization")

    # 2. Annual Inquiry Rate (project 6 months to annual)
    df_fe['annual_inquiry_rate'] = df_fe['inq.last.6mths'] * 2
    print("  + annual_inquiry_rate: Projected annual inquiries")

    # 3. Debt Burden (monthly payment / annual income)
    df_fe['debt_burden'] = df_fe['installment'] / np.exp(df_fe['log.annual.inc'])
    print("  + debt_burden: Monthly payment / annual income")

    # 4. Credit History in Years
    df_fe['credit_history_years'] = df_fe['days.with.cr.line'] / 365.25
    print("  + credit_history_years: Credit line age in years")

    # 5. High Debt Indicator (binary flag)
    df_fe['high_debt'] = ((df_fe['dti'] > 20) |
                          (df_fe['revol.util'] > 80)).astype(int)
    print("  + high_debt: Binary flag for high debt indicators")

    # 6. Risk Score (composite indicator)
    df_fe['risk_score'] = (
        df_fe['credit_util_ratio'] * 0.3 +
        (df_fe['inq.last.6mths'] / 10) * 0.2 +
        (df_fe['dti'] / 30) * 0.3 +
        (1 - np.clip(df_fe['credit_history_years'] / 20, 0, 1)) * 0.2
    )
    print("  + risk_score: Composite risk indicator")

    original_count = len(df.columns)
    new_count = len(df_fe.columns)
    added_count = new_count - original_count

    print(f"\nOriginal features: {original_count}")
    print(f"Total features after engineering: {new_count}")
    print(f"New features added: {added_count}")

    return df_fe


def get_engineered_feature_names():
    """
    Returns list of feature names created by create_loan_features.

    Returns:
    --------
    list
        List of engineered feature names
    """
    return [
        'credit_util_ratio',
        'annual_inquiry_rate',
        'debt_burden',
        'credit_history_years',
        'high_debt',
        'risk_score'
    ]


def validate_features_for_engineering(df):
    """
    Validate that DataFrame has required columns for feature engineering.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate

    Returns:
    --------
    tuple
        (bool, list) - (is_valid, missing_columns)

    Examples:
    ---------
    >>> valid, missing = validate_features_for_engineering(df)
    >>> if not valid:
    ...     print(f"Missing columns: {missing}")
    """
    required_cols = [
        'revol.util',
        'inq.last.6mths',
        'installment',
        'log.annual.inc',
        'days.with.cr.line',
        'dti'
    ]

    missing = [col for col in required_cols if col not in df.columns]

    return len(missing) == 0, missing


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nThis module provides feature engineering functions for loan data.")
    print("\nAvailable functions:")
    print("  - create_loan_features(df)")
    print("  - get_engineered_feature_names()")
    print("  - validate_features_for_engineering(df)")
