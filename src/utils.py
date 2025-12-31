"""Utility functions for visualization and performance evaluation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

logger = logging.getLogger(__name__)


def calculate_metrics(algorithm_name, y_true, y_pred, verbose=True):
    """Calculate classification metrics."""
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    fscore = f1_score(y_true, y_pred, average='macro') * 100
    
    if verbose:
        logger.info(f"{algorithm_name} - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, "
                   f"Recall: {recall:.2f}%, F-Score: {fscore:.2f}%")
    
    return {
        'algorithm': algorithm_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    }


def plot_confusion_matrix_and_roc(algorithm_name, y_true, y_pred, save_path=None):
    """Plot confusion matrix and ROC curve."""
    labels = ['Normal', 'Attack']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels,
                     annot=True, cmap="viridis", fmt="g", ax=axs[0])
    ax.set_ylim([0, len(labels)])
    axs[0].set_title(f"{algorithm_name}\nConfusion Matrix")
    axs[0].set_ylabel('True Label')
    axs[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    random_probs = [0 for _ in range(len(y_true))]
    p_fpr, p_tpr, _ = roc_curve(y_true, random_probs, pos_label=1)
    axs[1].plot(p_fpr, p_tpr, linestyle='--', color='orange', label='Random')
    
    ns_fpr, ns_tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='-', color='blue', label='Model')
    
    axs[1].set_title(f"{algorithm_name}\nROC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()



def plot_class_distribution(dataset, title="Class Distribution", save_path=None):
    """Plot class distribution pie chart."""
    labels, counts = np.unique(dataset['attack'].ravel(), return_counts=True)
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=['Normal', 'Attack'], autopct='%1.1f%%',
            colors=sns.color_palette("Set2"), startangle=90)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_protocol_distribution(dataset, save_path=None):
    """Plot protocol distribution pie chart."""
    plt.figure(figsize=(8, 8))
    dataset.groupby("proto").size().plot.pie(
        autopct='%.1f%%', startangle=90, colors=sns.color_palette("Set3")
    )
    plt.title("Protocol Distribution")
    plt.ylabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_protocol_attack_relationship(dataset, save_path=None):
    """Plot protocol vs attack relationship."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=dataset[['proto', 'attack']], x='proto', y='attack', hue='proto', legend=False)
    plt.title("Protocol vs Attack Pattern")
    plt.xlabel("Protocol")
    plt.ylabel("Attack")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison(results_list, save_path=None):
    """Plot algorithm performance comparison."""
    data = []
    for result in results_list:
        data.extend([
            [result['algorithm'], 'Accuracy', result['accuracy']],
            [result['algorithm'], 'Precision', result['precision']],
            [result['algorithm'], 'Recall', result['recall']],
            [result['algorithm'], 'F-Score', result['fscore']]
        ])
    
    df = pd.DataFrame(data, columns=['Algorithm', 'Metric', 'Value'])
    df_pivot = df.pivot(index='Metric', columns='Algorithm', values='Value')
    
    ax = df_pivot.plot(kind='bar', figsize=(10, 6), rot=0, width=0.8)
    plt.title("Algorithm Performance Comparison")
    plt.xlabel("Metrics")
    plt.ylabel("Score (%)")
    plt.legend(title="Algorithms", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_table(results_list):
    """Create formatted results table."""
    df = pd.DataFrame(results_list)
    df = df[['algorithm', 'accuracy', 'precision', 'recall', 'fscore']]
    df.columns = ['Algorithm', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F-Score (%)']
    return df.round(2)


def save_results_to_file(results_df, filepath):
    """Save results to CSV."""
    results_df.to_csv(filepath, index=False)
    logger.info(f"Results saved to {filepath}")
