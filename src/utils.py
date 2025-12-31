"""
Utility functions for visualization and performance evaluation.
Handles metrics calculation, confusion matrices, and ROC curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)


def calculate_metrics(algorithm_name, y_true, y_pred, verbose=True):
    """
    Calculate classification metrics: accuracy, precision, recall, F-score.
    
    Args:
        algorithm_name (str): Name of the algorithm
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        verbose (bool): Whether to print metrics (default True)
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Calculate metrics (macro average for multi-class compatibility)
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    fscore = f1_score(y_true, y_pred, average='macro') * 100
    
    metrics = {
        'algorithm': algorithm_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{algorithm_name} Performance Metrics")
        print(f"{'='*60}")
        print(f"Accuracy  : {accuracy:.2f}%")
        print(f"Precision : {precision:.2f}%")
        print(f"Recall    : {recall:.2f}%")
        print(f"F-Score   : {fscore:.2f}%")
        print(f"{'='*60}")
    
    return metrics


def plot_confusion_matrix_and_roc(algorithm_name, y_true, y_pred, save_path=None):
    """
    Plot confusion matrix and ROC curve side by side.
    
    Args:
        algorithm_name (str): Name of the algorithm
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        save_path (str): Path to save the plot (optional)
    """
    labels = ['Normal', 'Attack']
    
    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(
        conf_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        cmap="viridis",
        fmt="g",
        ax=axs[0]
    )
    ax.set_ylim([0, len(labels)])
    axs[0].set_title(f"{algorithm_name}\nConfusion Matrix")
    axs[0].set_ylabel('True Label')
    axs[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    # Plot random baseline (50% diagonal)
    random_probs = [0 for _ in range(len(y_true))]
    p_fpr, p_tpr, _ = roc_curve(y_true, random_probs, pos_label=1)
    axs[1].plot(p_fpr, p_tpr, linestyle='--', color='orange', label='Random Classifier')
    
    # Plot model's ROC curve
    ns_fpr, ns_tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    axs[1].plot(ns_fpr, ns_tpr, linestyle='-', color='blue', label='Model')
    
    axs[1].set_title(f"{algorithm_name}\nROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend()
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_class_distribution(dataset, title="Class Distribution", save_path=None):
    """
    Plot bar chart of class distribution.
    
    Args:
        dataset (pd.DataFrame): Dataset with 'attack' column
        title (str): Plot title
        save_path (str): Path to save the plot (optional)
    """
    labels, count = np.unique(dataset['attack'].ravel(), return_counts=True)
    
    plt.figure(figsize=(6, 4))
    bars = ['Normal', 'Attack']
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, count, color=['green', 'red'], alpha=0.7)
    plt.xticks(y_pos, bars)
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title(title)
    
    # Add count labels on bars
    for i, v in enumerate(count):
        plt.text(i, v + max(count)*0.01, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_protocol_distribution(dataset, save_path=None):
    """
    Plot pie chart of protocol distribution.
    
    Args:
        dataset (pd.DataFrame): Dataset with 'proto' column
        save_path (str): Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 8))
    dataset.groupby("proto").size().plot.pie(
        autopct='%.1f%%',
        startangle=90,
        colors=sns.color_palette("Set3")
    )
    plt.title("Network Protocol Distribution")
    plt.ylabel("")  # Remove default ylabel
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_protocol_attack_relationship(dataset, save_path=None):
    """
    Plot boxplot showing relationship between protocols and attacks.
    
    Args:
        dataset (pd.DataFrame): Dataset with 'proto' and 'attack' columns
        save_path (str): Path to save the plot (optional)
    """
    data = dataset[['proto', 'attack']]
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x='proto', y='attack', palette='rainbow')
    plt.title("Protocol Usage in Normal vs Attack Traffic")
    plt.xlabel("Protocol")
    plt.ylabel("Attack (0=Normal, 1=Attack)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pca_imbalance(X, Y, title="Dataset Imbalance Visualization (PCA)", save_path=None):
    """
    Use PCA to visualize dataset imbalance in 2D.
    
    Args:
        X (np.ndarray): Feature matrix
        Y (np.ndarray): Labels
        title (str): Plot title
        save_path (str): Path to save the plot (optional)
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='coolwarm', alpha=0.6, s=10)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title(title)
    plt.colorbar(scatter, label='Class (0=Normal, 1=Attack)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_algorithm_comparison(results_list, save_path=None):
    """
    Plot comparison bar chart of all algorithms.
    
    Args:
        results_list (list): List of metric dictionaries from calculate_metrics()
        save_path (str): Path to save the plot (optional)
    """
    # Create DataFrame for plotting
    data = []
    for result in results_list:
        data.extend([
            [result['algorithm'], 'Accuracy', result['accuracy']],
            [result['algorithm'], 'Precision', result['precision']],
            [result['algorithm'], 'Recall', result['recall']],
            [result['algorithm'], 'F-Score', result['fscore']]
        ])
    
    df = pd.DataFrame(data, columns=['Algorithm', 'Metric', 'Value'])
    
    # Pivot for plotting
    df_pivot = df.pivot(index='Metric', columns='Algorithm', values='Value')
    
    # Plot
    ax = df_pivot.plot(kind='bar', figsize=(10, 6), rot=0, width=0.8)
    plt.title("Algorithm Performance Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Score (%)", fontsize=12)
    plt.legend(title="Algorithms", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_results_table(results_list):
    """
    Create a formatted table of results.
    
    Args:
        results_list (list): List of metric dictionaries
        
    Returns:
        pd.DataFrame: Results table
    """
    df = pd.DataFrame(results_list)
    df = df[['algorithm', 'accuracy', 'precision', 'recall', 'fscore']]
    df.columns = ['Algorithm', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F-Score (%)']
    
    # Round to 2 decimal places
    for col in df.columns[1:]:
        df[col] = df[col].round(2)
    
    return df


def save_results_to_file(results_df, filepath):
    """
    Save results table to CSV file.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        filepath (str): Output file path
    """
    results_df.to_csv(filepath, index=False)
    print(f"\nResults saved to {filepath}")
