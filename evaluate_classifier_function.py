import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

def evaluate_classifier(model, X_val, y_val, labels=None, threshold=0.5, title_suffix='', save_path=None, show_plot=True):
    """
    Evaluates a binary classification model by displaying or saving:
      - Confusion matrix
      - Classification report
      - ROC curve with AUC

    Parameters:
        model: Trained classifier with predict_proba method.
        X_val: Validation features.
        y_val: True validation labels.
        labels: Optional list of class labels. If None, derived from y_val.
        threshold: Decision threshold for positive class.
        title_suffix: Custom suffix for plot titles.
        save_path: If provided (e.g., 'results.png'), saves the plot instead of displaying it.

    Returns:
        cm_df: Confusion matrix as pandas DataFrame.
        cr_df: Classification report as pandas DataFrame.
        accuracy: Validation accuracy as float.
        auc_score: AUC score as float.
    """
    # Predict probabilities and apply threshold
    probs = model.predict_proba(X_val)
    y_proba = probs[:, 1]
    y_pred = (y_proba > threshold).astype("int32")

    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    if labels is None:
        labels = sorted(np.unique(y_val))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Classification report
    cr = classification_report(y_val, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    for col in ['precision', 'recall', 'f1-score']:
        cr_df[col] = cr_df[col].round(3)
    cr_df['support'] = cr_df['support'].astype(int)
    cr_df.rename(index={'0': 'DOWN', '1': 'UP'}, inplace=True)

    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc_score = roc_auc_score(y_val, y_proba)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    # Confusion Matrix
    axes[0].axis('off')
    table1 = axes[0].table(cellText=cm_df.values,
                           rowLabels=cm_df.index,
                           colLabels=cm_df.columns,
                           cellLoc='center',
                           loc='center',
                           rowLoc='center',
                           colLoc='center',
                           colWidths=[0.2]*len(cm_df.columns))
    table1.auto_set_font_size(False)
    table1.set_fontsize(18)
    table1.scale(1.2, 2.0)
    for key, cell in table1.get_celld().items():
        cell.set_linewidth(1.2)
        cell.set_edgecolor('gray')
    axes[0].set_title(f'Confusion Matrix {title_suffix}', fontsize=24, pad=20)

    # Classification Report
    axes[1].axis('off')
    table2 = axes[1].table(cellText=cr_df.values,
                           rowLabels=cr_df.index,
                           colLabels=cr_df.columns,
                           cellLoc='center',
                           loc='center',
                           rowLoc='center',
                           colLoc='center',
                           colWidths=[0.2]*len(cr_df.columns))
    table2.auto_set_font_size(False)
    table2.set_fontsize(16)
    table2.scale(1.2, 2.0)
    for key, cell in table2.get_celld().items():
        cell.set_linewidth(1.2)
        cell.set_edgecolor('gray')
    axes[1].set_title(f'Classification Report {title_suffix}', fontsize=24, pad=20)

    # ROC Curve
    axes[2].plot(fpr, tpr, color='blue', linewidth=2, label=f'AUC = {auc_score:.4f}')
    axes[2].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('False Positive Rate', fontsize=14)
    axes[2].set_ylabel('True Positive Rate', fontsize=14)
    axes[2].set_title(f'ROC Curve {title_suffix}', fontsize=24, pad=20)
    axes[2].legend(loc='lower right', fontsize=14)
    axes[2].grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return cm_df, cr_df, accuracy, auc_score
