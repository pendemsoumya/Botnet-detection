"""
Main training script for botnet detection models.
Orchestrates the complete pipeline: data loading → preprocessing → training → evaluation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, get_dataset_info
from preprocessing import DataPreprocessor
from models import DefaultDecisionTree, BOGPDecisionTree, SVMClassifier, CNNClassifier
from utils import (
    calculate_metrics, plot_confusion_matrix_and_roc,
    plot_class_distribution, plot_protocol_distribution,
    plot_protocol_attack_relationship, plot_pca_imbalance,
    plot_algorithm_comparison, create_results_table, save_results_to_file
)


def run_training_pipeline(dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv",
                          show_visualizations=True,
                          save_results=True):
    """
    Execute complete training and evaluation pipeline.
    
    Args:
        dataset_path (str): Path to the dataset CSV file
        show_visualizations (bool): Whether to display plots
        save_results (bool): Whether to save results to files
    """
    print("="*80)
    print(" IoT BOTNET DETECTION - TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Load dataset
    print("\n[STEP 1] Loading Dataset")
    print("-" * 80)
    try:
        dataset = load_dataset(dataset_path)
        get_dataset_info(dataset)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return
    
    # Step 2: Exploratory Data Analysis
    if show_visualizations:
        print("\n[STEP 2] Exploratory Data Analysis")
        print("-" * 80)
        
        print("Visualizing class distribution...")
        plot_class_distribution(dataset, "Original Class Distribution (Before SMOTE)")
        
        print("Checking for missing values...")
        null_count = dataset.isnull().sum().sum()
        print(f"Total missing values: {null_count}")
        
        print("Visualizing protocol distribution...")
        plot_protocol_distribution(dataset)
        
        print("Analyzing protocol-attack relationship...")
        plot_protocol_attack_relationship(dataset)
    
    # Step 3: Data Preprocessing
    print("\n[STEP 3] Data Preprocessing")
    print("-" * 80)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset, test_size=0.2)
    
    # Visualize dataset imbalance using PCA
    if show_visualizations:
        print("\nVisualizing class imbalance with PCA...")
        # Use a sample for visualization (full dataset is too large)
        sample_size = min(10000, len(dataset))
        dataset_sample = dataset.sample(n=sample_size, random_state=42)
        Y_sample = dataset_sample['attack'].values
        X_sample = dataset_sample.drop(['attack'], axis=1).values
        plot_pca_imbalance(X_sample, Y_sample)
    
    # Step 4: Model Training and Evaluation
    print("\n[STEP 4] Model Training and Evaluation")
    print("-" * 80)
    
    results = []
    
    # 4.1: Default Decision Tree
    print("\n[4.1] Default Decision Tree")
    print("-" * 40)
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
