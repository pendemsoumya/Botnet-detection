#!/usr/bin/env python3
"""Training pipeline entry point for IoT Botnet Detection."""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_dataset():
    """Verify dataset file exists."""
    dataset_path = "data/UNSW_2018_IoT_Botnet_Full5pc_4.csv"
    
    if os.path.exists(dataset_path):
        return True
    
    logger.error(f"Dataset not found: {dataset_path}")
    logger.info("Download from: https://research.unsw.edu.au/projects/bot-iot-dataset")
    return False


def check_dependencies():
    """Verify required packages are installed."""
    required = ['pandas', 'numpy', 'sklearn', 'imblearn', 'keras', 'tensorflow', 'bayes_opt']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg if pkg != 'sklearn' else 'sklearn')
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    return True


def setup_directories():
    """Create required directories."""
    for directory in ['data', 'models', 'results', 'notebooks']:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    """Execute training pipeline."""
    logger.info("Starting botnet detection training")
    
    setup_directories()
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_dataset():
        sys.exit(1)
    
    sys.path.append('src')
    from train import run_training_pipeline
    
    try:
        run_training_pipeline(
            dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv",
            show_visualizations=True,
            save_results=True,
            sample_size=50000
        )
        logger.info("Training completed successfully")
        logger.info("Results saved to: results/")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
