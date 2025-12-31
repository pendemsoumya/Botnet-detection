"""Training pipeline for botnet detection models."""
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from data_loader import load_dataset, get_dataset_info
from preprocessing import DataPreprocessor
from models import DefaultDecisionTree, BOGPDecisionTree, SVMClassifier, CNNClassifier
from utils import (
    calculate_metrics, plot_confusion_matrix_and_roc,
    plot_class_distribution, plot_protocol_distribution,
    plot_protocol_attack_relationship,
    plot_algorithm_comparison, create_results_table, save_results_to_file
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_training_pipeline(dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv",
                          show_visualizations=True,
                          save_results=True,
                          sample_size=None):
    """
    Execute training and evaluation pipeline.
    
    Args:
        dataset_path: Path to dataset CSV
        show_visualizations: Generate plots
        save_results: Save results to disk
        sample_size: Limit dataset size for memory efficiency
    """
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_path)
        
        if sample_size and sample_size < len(dataset):
            logger.info(f"Sampling {sample_size} records from {len(dataset)}")
            normal = dataset[dataset['attack'] == 0]
            attack = dataset[dataset['attack'] == 1].sample(
                n=sample_size - len(normal), random_state=42
            )
            dataset = pd.concat([normal, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        get_dataset_info(dataset)
    except FileNotFoundError as e:
        logger.error(f"Dataset error: {e}")
        return
    
    # EDA
    if show_visualizations:
        logger.info("Generating exploratory visualizations")
        plot_class_distribution(dataset, "Class Distribution (Before SMOTE)")
        plot_protocol_distribution(dataset)
        plot_protocol_attack_relationship(dataset)
    
    # Preprocessing
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset, test_size=0.2)
    
    # Train models
    logger.info("Training models")
    results = []
    
    # Decision Tree
    logger.info("Training Decision Tree (default)")
    dt = DefaultDecisionTree()
    dt.train(X_train, y_train)
    results.append(calculate_metrics("Default Decision Tree", y_test, dt.predict(X_test)))
    if show_visualizations:
        plot_confusion_matrix_and_roc("Default Decision Tree", y_test, dt.predict(X_test),
                                     "results/dt_default_evaluation.png" if save_results else None)
    
    # BOGP Decision Tree
    logger.info("Training Decision Tree (BOGP-optimized)")
    bogp = BOGPDecisionTree()
    bogp.optimize_hyperparameters(X_train, y_train, init_points=5, n_iter=2)
    bogp.train(X_train, y_train)
    results.append(calculate_metrics("BOGP Optimized Decision Tree", y_test, bogp.predict(X_test)))
    if show_visualizations:
        plot_confusion_matrix_and_roc("BOGP Optimized Decision Tree", y_test, bogp.predict(X_test),
                                     "results/dt_bogp_evaluation.png" if save_results else None)
    
    # SVM
    logger.info("Training SVM")
    svm_model = SVMClassifier(training_limit=50)
    svm_model.train(X_train, y_train)
    results.append(calculate_metrics("SVM Algorithm", y_test, svm_model.predict(X_test)))
    if show_visualizations:
        plot_confusion_matrix_and_roc("SVM Algorithm", y_test, svm_model.predict(X_test),
                                     "results/svm_evaluation.png" if save_results else None)
    
    # CNN
    logger.info("Training CNN")
    cnn = CNNClassifier(model_path="models/cnn_weights.hdf5", history_path="models/cnn_history.pckl")
    cnn.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
    results.append(calculate_metrics("Extension CNN", y_test, cnn.predict(X_test)))
    if show_visualizations:
        plot_confusion_matrix_and_roc("Extension CNN", y_test, cnn.predict(X_test),
                                     "results/cnn_evaluation.png" if save_results else None)
    
    # Results
    results_df = create_results_table(results)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    if show_visualizations:
        plot_algorithm_comparison(results, "results/algorithm_comparison.png" if save_results else None)
    
    if save_results:
        save_results_to_file(results_df, "results/performance_metrics.csv")
    
    best = results_df.loc[results_df['Accuracy (%)'].idxmax()]
    logger.info(f"\nBest model: {best['Algorithm']} (Accuracy: {best['Accuracy (%)']}%)")


if __name__ == "__main__":
    run_training_pipeline(
        dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv",
        show_visualizations=True,
        save_results=True
    )
    dt_model = DefaultDecisionTree()
    dt_model.train(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_metrics = calculate_metrics("Default Decision Tree", y_test, dt_predictions)
    results.append(dt_metrics)
    
    if show_visualizations:
        plot_confusion_matrix_and_roc("Default Decision Tree", y_test, dt_predictions,
                                     save_path="results/dt_default_evaluation.png" if save_results else None)
    
    # 4.2: BOGP-Optimized Decision Tree
    print("\n[4.2] BOGP-Optimized Decision Tree")
    print("-" * 40)
    bogp_model = BOGPDecisionTree()
    bogp_model.optimize_hyperparameters(X_train, y_train, init_points=5, n_iter=2)
    bogp_model.train(X_train, y_train)
    bogp_predictions = bogp_model.predict(X_test)
    bogp_metrics = calculate_metrics("BOGP Optimized Decision Tree", y_test, bogp_predictions)
    results.append(bogp_metrics)
    
    if show_visualizations:
        plot_confusion_matrix_and_roc("BOGP Optimized Decision Tree", y_test, bogp_predictions,
                                     save_path="results/dt_bogp_evaluation.png" if save_results else None)
    
    # 4.3: Support Vector Machine
    print("\n[4.3] Support Vector Machine")
    print("-" * 40)
    svm_model = SVMClassifier(training_limit=50)
    svm_model.train(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_metrics = calculate_metrics("SVM Algorithm", y_test, svm_predictions)
    results.append(svm_metrics)
    
    if show_visualizations:
        plot_confusion_matrix_and_roc("SVM Algorithm", y_test, svm_predictions,
                                     save_path="results/svm_evaluation.png" if save_results else None)
    
    # 4.4: Convolutional Neural Network
    print("\n[4.4] Convolutional Neural Network (CNN)")
    print("-" * 40)
    cnn_model = CNNClassifier(model_path="models/cnn_weights.hdf5",
                             history_path="models/cnn_history.pckl")
    cnn_model.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
    cnn_predictions = cnn_model.predict(X_test)
    cnn_metrics = calculate_metrics("Extension CNN", y_test, cnn_predictions)
    results.append(cnn_metrics)
    
    if show_visualizations:
        plot_confusion_matrix_and_roc("Extension CNN", y_test, cnn_predictions,
                                     save_path="results/cnn_evaluation.png" if save_results else None)
    
    # Step 5: Results Comparison
    print("\n[STEP 5] Results Comparison")
    print("-" * 80)
    
    # Create results table
    results_df = create_results_table(results)
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Plot comparison
    if show_visualizations:
        print("\nGenerating comparison plot...")
        plot_algorithm_comparison(results, 
                                save_path="results/algorithm_comparison.png" if save_results else None)
    
    # Save results
    if save_results:
        save_results_to_file(results_df, "results/performance_metrics.csv")
    
    # Find best model
    best_model = results_df.loc[results_df['Accuracy (%)'].idxmax()]
    print(f"\n{'='*80}")
    print(f"BEST PERFORMING MODEL: {best_model['Algorithm']}")
    print(f"Accuracy: {best_model['Accuracy (%)']}%")
    print(f"{'='*80}")
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"Results saved to results/ directory" if save_results else "")


if __name__ == "__main__":
    # Run the complete pipeline
    run_training_pipeline(
        dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv",
        show_visualizations=True,
        save_results=True
    )
