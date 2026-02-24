from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


# handle lib imports
# try:
#     # When imported as part of a package
#     from . import text_util as tu, utility as utl, imputer as im
# except ImportError:
#     # When run as a standalone script
#     import text_util as tu
#     import utility as utl
#     import imputer as im

#import text_util as tu
#import utility as utl
#import imputer as im

# class_imbalance.py
try:
    import lib.text_util as tu
except ImportError:
    import text_util as tu


@dataclass
class ClassAnalysisItem:
    class_label: Any
    class_label_display: str
    count: int
    percentage: float
    ratio_to_majority: float
    ratio_to_minority: float


@dataclass
class ImbalanceAnalysisResult:
    target_column: str
    n_classes: int
    total_samples: int
    minority_index: int
    majority_index: int
    majority_class: Any
    majority_count: int
    minority_class: Any
    minority_count: int
    minority_percentage: str
    imbalance_ratio: str
    severity: str
    recommended_action: str
    class_analysis: list[ClassAnalysisItem]
    class_names: list[Any]

    def display(self):
        tu.print_heading("CLASS IMBALANCE ANALYSIS")
        tu.print_sub_heading(f"Target column: {self.target_column}")
        tu.print_sub_heading(f"Number of classes: {self.n_classes}")
        tu.print_sub_heading(f"Total samples: {self.total_samples:,}")
        print(f"\nClass Distribution:")
        print("-" * 70)

        sorted_analysis = sorted(self.class_analysis, key=lambda x: x.count, reverse=True)
        for item in sorted_analysis:
            bar_length = int(item.percentage / 2)
            bar = "█" * bar_length
            print(f"  {item.class_label_display:>15}: {item.count:>8,} ({item.percentage:>6.2f}%) {bar}")

        print("\n" + "-" * 70)
        print(f"Majority class: {self.majority_class} ({self.majority_count:,} samples)")
        print(f"Minority class: {self.minority_class} ({self.minority_count:,} samples)")
        print(f"Imbalance ratio: {self.imbalance_ratio}")
        print(f"\nSeverity: {self.severity}")
        print(f"Recommended action: {self.recommended_action}")

    def display_markdown(self) -> str:
        """Generate a markdown formatted summary of the imbalance analysis."""
        md = f"Our dataset exhibits **{self.severity}** for **{self.target_column}**:\n\n"

        # Class distribution
        sorted_analysis = sorted(self.class_analysis, key=lambda x: x.count, reverse=True)
        for item in sorted_analysis:
            md += f"- **{item.class_label_display}**: {item.count:,} samples ({item.percentage:.2f}%)\n"

        md += f"- **Imbalance Ratio**: {self.imbalance_ratio}\n"

        return md


def check_imbalance(target_series: pd.Series, class_labels: dict = None, verbose: bool = True) -> ImbalanceAnalysisResult:
    """
    Check if dataset is imbalanced for binary or multi-class problems

    Parameters:
    -----------
    target_series : pd.Series
        The target/label column from your dataframe (e.g., df['loan_status'] or y_train)
        The Series.name will be used as the target column name in the report
    class_labels : dict, optional
        Mapping of class values to human-readable labels {class_value: "Display Name"}
        Example: {0: "Paid", 1: "Default"}
    verbose : bool, default=True
        Whether to print detailed information

    Returns:
    --------
    ImbalanceAnalysisResult with comprehensive imbalance analysis
    """
    # Extract column name and calculate value counts
    target_column = target_series.name if target_series.name else "target"
    class_counts = target_series.value_counts()
    class_dict = class_counts.to_dict()
    counts = class_counts.values

    # Basic statistics
    total = counts.sum()
    n_classes = len(counts)
    majority_count = counts.max()
    minority_count = counts.min()
    ratio = majority_count / minority_count

    # Calculate imbalance ratio for each class vs majority
    majority_idx = counts.argmax()
    minority_idx = counts.argmin()

    # Determine severity based on worst-case ratio
    if ratio < 1.5:
        severity = "Balanced ✅"
        action = "No special handling needed"
    elif ratio < 3:
        severity = "Slight Imbalance ⚠️"
        action = "Monitor metrics closely"
    elif ratio < 9:
        severity = "Moderate Imbalance 🟡"
        action = "Use class weights or resampling"
    elif ratio < 99:
        severity = "Severe Imbalance 🟠"
        action = "Special techniques required (SMOTE, heavy class weights)"
    else:
        severity = "Extreme Imbalance 🔴"
        action = "Consider anomaly detection approaches"

    # Per-class analysis
    class_analysis = []
    class_values = list(class_dict.keys())

    for i, (label, count) in enumerate(class_dict.items()):
        percentage = (count / total) * 100
        ratio_to_majority = majority_count / count if count > 0 else np.inf
        ratio_to_minority = count / minority_count if minority_count > 0 else np.inf

        # Determine display label
        if class_labels and label in class_labels:
            display_label = f"Class {label} ({class_labels[label]})"
        else:
            display_label = f"Class {label}"

        class_analysis.append(ClassAnalysisItem(
            class_label=label,
            class_label_display=display_label,
            count=count,
            percentage=percentage,
            ratio_to_majority=ratio_to_majority,
            ratio_to_minority=ratio_to_minority
        ))

    result = ImbalanceAnalysisResult(
        target_column=target_column,
        n_classes=n_classes,
        total_samples=int(total),
        minority_index=minority_idx,
        majority_index=majority_idx,
        majority_class=class_values[majority_idx],
        majority_count=int(majority_count),
        minority_class=class_values[minority_idx],
        minority_count=int(minority_count),
        minority_percentage=f"{(minority_count / total) * 100:.2f}%",
        imbalance_ratio=f"{ratio:.2f}:1",
        severity=severity,
        recommended_action=action,
        class_analysis=class_analysis,
        class_names=class_values
    )

    if verbose:
        result.display()

    return result
