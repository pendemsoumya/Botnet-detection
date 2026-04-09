"""Training pipeline for botnet detection models."""
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_dataset_info, load_dataset
from models import BOGPDecisionTree, CNNClassifier, DefaultDecisionTree, SVMClassifier
from preprocessing import DataPreprocessor
from utils import (
    calculate_metrics,
    create_results_table,
    plot_algorithm_comparison,
    plot_class_distribution,
    plot_confusion_matrix_and_roc,
    plot_protocol_attack_relationship,
    plot_protocol_distribution,
    save_results_to_file,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = "data/UNSW_2018_IoT_Botnet_Full5pc_4.csv"
PIPELINE_STEPS = [
    ("dataset_selected", "Dataset selected"),
    ("dataset_loaded", "Dataset loaded"),
    ("sampling", "Sampling applied"),
    ("eda", "EDA visualizations"),
    ("preprocessing", "Data preprocessing"),
    ("decision_tree_default", "Default Decision Tree"),
    ("decision_tree_bogp", "BOGP Decision Tree"),
    ("svm", "SVM classifier"),
    ("cnn", "CNN classifier"),
    ("results_generated", "Results generated"),
    ("best_model_selected", "Best model selected"),
]


def ensure_output_directories():
    """Create directories required by the training pipeline."""
    for directory in ["data", "models", "results", "notebooks"]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def _emit_progress(progress_callback, step_key, status, message, extra=None):
    """Emit a structured progress update when a callback is provided."""
    if not progress_callback:
        return

    labels = dict(PIPELINE_STEPS)
    payload = {
        "step_key": step_key,
        "step_label": labels[step_key],
        "status": status,
        "message": message,
    }
    if extra:
        payload.update(extra)

    progress_callback(payload)


def _emit_log(log_callback, message):
    """Emit a log message to UI and logger."""
    logger.info(message)
    if log_callback:
        log_callback(message)


def _sample_dataset(dataset, sample_size, log_callback=None):
    """Sample the dataset while preserving all normal rows."""
    if not sample_size or sample_size >= len(dataset):
        message = f"No sampling applied. Using all {len(dataset)} rows."
        _emit_log(log_callback, message)
        return dataset, message

    normal = dataset[dataset["attack"] == 0]
    requested_attack_rows = max(sample_size - len(normal), 0)
    attack = dataset[dataset["attack"] == 1]

    if requested_attack_rows >= len(attack):
        sampled_dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        message = (
            "Sample size exceeds available attack rows after keeping all normal rows. "
            f"Using all {len(sampled_dataset)} rows."
        )
        _emit_log(log_callback, message)
        return sampled_dataset, message

    sampled_attack = attack.sample(n=requested_attack_rows, random_state=42)
    sampled_dataset = (
        pd.concat([normal, sampled_attack])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    message = f"Sampled {len(sampled_dataset)} rows from {len(dataset)} total rows."
    _emit_log(log_callback, message)
    return sampled_dataset, message


def _validate_target_classes(dataset):
    """Ensure the dataset contains at least two target classes."""
    class_counts = dataset["attack"].value_counts(dropna=False).to_dict()
    unique_classes = dataset["attack"].nunique(dropna=False)

    if unique_classes < 2:
        raise ValueError(
            "The selected dataset contains only one target class "
            f"({class_counts}). This project needs both normal (0) and attack (1) rows."
        )


def _save_evaluation_plot(name, y_test, predictions, save_results):
    filename_map = {
        "Default Decision Tree": "results/dt_default_evaluation.png",
        "BOGP Optimized Decision Tree": "results/dt_bogp_evaluation.png",
        "SVM Algorithm": "results/svm_evaluation.png",
        "Extension CNN": "results/cnn_evaluation.png",
    }
    save_path = filename_map[name] if save_results else None
    plot_confusion_matrix_and_roc(name, y_test, predictions, save_path)
    return save_path


def run_training_pipeline(
    dataset_path=DEFAULT_DATASET_PATH,
    show_visualizations=True,
    save_results=True,
    sample_size=None,
    progress_callback=None,
    log_callback=None,
):
    """
    Execute the training and evaluation pipeline.

    Returns:
        dict: Dataset info, results table, best model, and artifact paths.
    """
    ensure_output_directories()
    current_step = "dataset_selected"

    _emit_progress(
        progress_callback,
        "dataset_selected",
        "running",
        f"Preparing dataset: {dataset_path}",
        {"dataset_path": dataset_path},
    )
    _emit_log(log_callback, f"Dataset selected: {dataset_path}")
    _emit_progress(
        progress_callback,
        "dataset_selected",
        "done",
        f"Dataset selected: {dataset_path}",
        {"dataset_path": dataset_path},
    )

    try:
        current_step = "dataset_loaded"
        _emit_progress(progress_callback, "dataset_loaded", "running", "Loading dataset")
        dataset = load_dataset(dataset_path)
        dataset_info = get_dataset_info(dataset)
        _emit_log(
            log_callback,
            f"Dataset loaded with {dataset_info['rows']} rows and {dataset_info['columns']} columns.",
        )
        _emit_progress(
            progress_callback,
            "dataset_loaded",
            "done",
            f"Loaded {dataset_info['rows']} rows and {dataset_info['columns']} columns.",
            {"dataset_info": dataset_info},
        )
        _validate_target_classes(dataset)

        current_step = "sampling"
        _emit_progress(progress_callback, "sampling", "running", "Checking whether sampling is needed")
        dataset, sampling_message = _sample_dataset(dataset, sample_size, log_callback=log_callback)
        _validate_target_classes(dataset)
        _emit_progress(
            progress_callback,
            "sampling",
            "done",
            sampling_message,
            {"sampled_rows": int(len(dataset))},
        )

        artifact_paths = []
        if show_visualizations:
            current_step = "eda"
            _emit_progress(progress_callback, "eda", "running", "Generating exploratory plots")
            class_dist_path = "results/class_distribution.png" if save_results else None
            protocol_dist_path = "results/protocol_distribution.png" if save_results else None
            protocol_attack_path = "results/protocol_attack_relationship.png" if save_results else None

            plot_class_distribution(dataset, "Class Distribution (Before SMOTE)", class_dist_path)
            plot_protocol_distribution(dataset, protocol_dist_path)
            plot_protocol_attack_relationship(dataset, protocol_attack_path)

            artifact_paths.extend(
                [
                    path
                    for path in [class_dist_path, protocol_dist_path, protocol_attack_path]
                    if path
                ]
            )
            _emit_log(log_callback, "EDA visualizations generated.")
            _emit_progress(progress_callback, "eda", "done", "EDA visualizations completed")
        else:
            _emit_progress(progress_callback, "eda", "done", "EDA visualizations skipped")

        current_step = "preprocessing"
        _emit_progress(progress_callback, "preprocessing", "running", "Encoding, scaling, and balancing data")
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset, test_size=0.2)
        _emit_log(
            log_callback,
            f"Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}.",
        )
        _emit_progress(
            progress_callback,
            "preprocessing",
            "done",
            f"Prepared train/test split with {len(X_train)} training rows and {len(X_test)} test rows.",
        )

        results = []
        labels_to_step = {
            "Default Decision Tree": "decision_tree_default",
            "BOGP Optimized Decision Tree": "decision_tree_bogp",
            "SVM Algorithm": "svm",
            "Extension CNN": "cnn",
        }

        current_step = "decision_tree_default"
        _emit_progress(progress_callback, "decision_tree_default", "running", "Training Default Decision Tree")
        default_tree = DefaultDecisionTree()
        default_tree.train(X_train, y_train)
        default_predictions = default_tree.predict(X_test)
        results.append(calculate_metrics("Default Decision Tree", y_test, default_predictions))
        if show_visualizations:
            evaluation_path = _save_evaluation_plot(
                "Default Decision Tree", y_test, default_predictions, save_results
            )
            if evaluation_path:
                artifact_paths.append(evaluation_path)
        _emit_log(log_callback, "Default Decision Tree completed.")
        _emit_progress(progress_callback, "decision_tree_default", "done", "Default Decision Tree completed")

        current_step = "decision_tree_bogp"
        _emit_progress(progress_callback, "decision_tree_bogp", "running", "Training BOGP Decision Tree")
        bogp = BOGPDecisionTree()
        bogp.optimize_hyperparameters(X_train, y_train, init_points=5, n_iter=2)
        bogp.train(X_train, y_train)
        bogp_predictions = bogp.predict(X_test)
        results.append(calculate_metrics("BOGP Optimized Decision Tree", y_test, bogp_predictions))
        if show_visualizations:
            evaluation_path = _save_evaluation_plot(
                "BOGP Optimized Decision Tree", y_test, bogp_predictions, save_results
            )
            if evaluation_path:
                artifact_paths.append(evaluation_path)
        _emit_log(log_callback, "BOGP Decision Tree completed.")
        _emit_progress(progress_callback, "decision_tree_bogp", "done", "BOGP Decision Tree completed")

        current_step = "svm"
        _emit_progress(progress_callback, "svm", "running", "Training SVM classifier")
        svm_model = SVMClassifier(training_limit=50)
        svm_model.train(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)
        results.append(calculate_metrics("SVM Algorithm", y_test, svm_predictions))
        if show_visualizations:
            evaluation_path = _save_evaluation_plot(
                "SVM Algorithm", y_test, svm_predictions, save_results
            )
            if evaluation_path:
                artifact_paths.append(evaluation_path)
        _emit_log(log_callback, "SVM classifier completed.")
        _emit_progress(progress_callback, "svm", "done", "SVM classifier completed")

        current_step = "cnn"
        _emit_progress(progress_callback, "cnn", "running", "Training CNN classifier")
        cnn = CNNClassifier(
            model_path="models/cnn_weights.weights.h5",
            history_path="models/cnn_history.pckl",
        )
        cnn.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
        cnn_predictions = cnn.predict(X_test)
        results.append(calculate_metrics("Extension CNN", y_test, cnn_predictions))
        if show_visualizations:
            evaluation_path = _save_evaluation_plot(
                "Extension CNN", y_test, cnn_predictions, save_results
            )
            if evaluation_path:
                artifact_paths.append(evaluation_path)
        _emit_log(log_callback, "CNN classifier completed.")
        _emit_progress(progress_callback, "cnn", "done", "CNN classifier completed")

        current_step = "results_generated"
        _emit_progress(progress_callback, "results_generated", "running", "Building comparison results")
        results_df = create_results_table(results)
        comparison_path = "results/algorithm_comparison.png" if save_results else None
        if show_visualizations:
            plot_algorithm_comparison(results, comparison_path)
        if comparison_path:
            artifact_paths.append(comparison_path)
        if save_results:
            save_results_to_file(results_df, "results/performance_metrics.csv")
        _emit_log(log_callback, "Results table and comparison output generated.")
        _emit_progress(progress_callback, "results_generated", "done", "Results generated")

        current_step = "best_model_selected"
        _emit_progress(progress_callback, "best_model_selected", "running", "Selecting best model")
        best_row = results_df.loc[results_df["Accuracy (%)"].idxmax()]
        best_model = best_row.to_dict()
        _emit_log(
            log_callback,
            f"Best model: {best_model['Algorithm']} with {best_model['Accuracy (%)']}% accuracy.",
        )
        _emit_progress(
            progress_callback,
            "best_model_selected",
            "done",
            f"Best model: {best_model['Algorithm']}",
            {"best_model": best_model},
        )

        return {
            "dataset_path": dataset_path,
            "dataset_info": dataset_info,
            "results_df": results_df,
            "best_model": best_model,
            "artifact_paths": artifact_paths,
            "step_keys": [step_key for step_key, _ in PIPELINE_STEPS],
            "step_labels": dict(PIPELINE_STEPS),
            "model_step_map": labels_to_step,
        }
    except Exception as exc:
        message = str(exc)
        _emit_log(log_callback, f"Pipeline failed: {message}")
        _emit_progress(progress_callback, current_step, "failed", message)
        raise


if __name__ == "__main__":
    run_training_pipeline(
        dataset_path=DEFAULT_DATASET_PATH,
        show_visualizations=True,
        save_results=True,
    )
