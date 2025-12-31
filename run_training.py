#!/usr/bin/env python3
"""
Entry point script for IoT Botnet Detection project.
Handles dataset download and runs the complete training pipeline.
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


def download_dataset():
    """
    Download UNSW 2018 IoT Botnet Dataset.
    
    The dataset is available from the UNSW Canberra Cyber research repository.
    Note: This function provides instructions for manual download as the dataset
    requires agreeing to terms of use.
    """
    dataset_path = "data/UNSW_2018_IoT_Botnet_Full5pc_4.csv"
    
    if os.path.exists(dataset_path):
        print(f"✓ Dataset found at {dataset_path}")
        return True
    
    print("\n" + "="*80)
    print(" DATASET DOWNLOAD REQUIRED")
    print("="*80)
    print("\nThe UNSW 2018 IoT Botnet Dataset is required for training.")
    print("\nDataset Information:")
    print("  Name: UNSW_2018_IoT_Botnet_Full5pc_4.csv")
    print("  Size: ~668,522 records")
    print("  Features: 46 columns (network traffic features)")
    print("  Classes: Normal (0) and Attack (1)")
    
    print("\n" + "-"*80)
    print("DOWNLOAD OPTIONS:")
    print("-"*80)
    
    print("\n1. Official UNSW Source (Recommended):")
    print("   https://research.unsw.edu.au/projects/bot-iot-dataset")
    print("   https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE")
    
    print("\n2. Alternative Sources:")
    print("   - Kaggle: https://www.kaggle.com/datasets/piyushagni5/unsw-nb15-and-bot-iot-datasets")
    print("   - UCI Machine Learning Repository (if available)")
    
    print("\n" + "-"*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("-"*80)
    print("1. Visit one of the URLs above")
    print("2. Download the dataset file (may require registration)")
    print("3. Extract if compressed")
    print("4. Place 'UNSW_2018_IoT_Botnet_Full5pc_4.csv' in the 'data/' directory")
    print("5. Run this script again")
    
    print("\n" + "="*80)
    
    # Check if user wants to provide custom path
    response = input("\nDo you have the dataset file already? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        custom_path = input("Enter the full path to the CSV file: ").strip()
        if os.path.exists(custom_path):
            import shutil
            shutil.copy(custom_path, dataset_path)
            print(f"✓ Dataset copied to {dataset_path}")
            return True
        else:
            print(f"✗ File not found: {custom_path}")
            return False
    
    return False


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n" + "="*80)
    print(" CHECKING DEPENDENCIES")
    print("="*80)
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'sklearn', 'imblearn', 'keras', 'tensorflow',
        'bayes_opt'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package if package != 'sklearn' else 'sklearn')
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n" + "-"*80)
        print("MISSING PACKAGES DETECTED")
        print("-"*80)
        print("Please install missing packages using:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'results', 'notebooks']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories checked/created")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print(" IoT BOTNET DETECTION - TRAINING SYSTEM")
    print(" College Project - Machine Learning Based Network Security")
    print("="*80)
    
    # Step 1: Create directories
    print("\n[1] Setting up directories...")
    create_directories()
    
    # Step 2: Check dependencies
    print("\n[2] Checking dependencies...")
    if not check_dependencies():
        print("\n✗ Please install missing dependencies and run again")
        sys.exit(1)
    
    # Step 3: Check/download dataset
    print("\n[3] Checking dataset...")
    if not download_dataset():
        print("\n✗ Dataset not available. Please download and try again")
        sys.exit(1)
    
    # Step 4: Run training pipeline
    print("\n[4] Starting training pipeline...")
    print("="*80)
    
    # Import and run training
    sys.path.append('src')
    from train import run_training_pipeline
    
    try:
        run_training_pipeline(
            dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv",
            show_visualizations=True,
            save_results=True
        )
        
        print("\n" + "="*80)
        print(" TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nResults saved in:")
        print("  - results/performance_metrics.csv")
        print("  - results/*.png (evaluation plots)")
        print("  - models/cnn_weights.hdf5 (trained CNN model)")
        print("\nYou can now review the results and present to your manager!")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
