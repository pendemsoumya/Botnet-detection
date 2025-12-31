# USAGE GUIDE - IoT Botnet Detection Project

This guide provides step-by-step instructions for setting up and running the botnet detection system.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Dataset Setup](#dataset-setup)
4. [Running the Project](#running-the-project)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## System Requirements

### Hardware
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: ~500MB for dataset + 100MB for dependencies
- **CPU**: Any modern processor (GPU not required)

### Software
- **Python**: Version 3.8, 3.9, or 3.10
- **Operating System**: Windows, macOS, or Linux
- **Internet Connection**: For downloading dataset and dependencies

### Verify Python Version
```bash
python --version
# or
python3 --version
```

If Python is not installed, download from: https://www.python.org/downloads/

---

## Installation Steps

### Step 1: Clone or Download the Repository

**Option A: Using Git**
```bash
git clone https://github.com/pendemsoumya/Botnet-detection.git
cd Botnet-detection
```

**Option B: Download ZIP**
1. Go to: https://github.com/pendemsoumya/Botnet-detection
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open terminal/command prompt in the extracted folder

### Step 2: Create Virtual Environment (Recommended)

Virtual environments keep project dependencies isolated.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning algorithms
- tensorflow/keras: Deep learning
- matplotlib, seaborn: Visualization
- imblearn: SMOTE for imbalance handling
- bayesian-optimization: Hyperparameter tuning

**Installation time**: ~2-5 minutes depending on internet speed.

### Step 4: Verify Installation

```bash
python run_training.py
```

The script will check all dependencies. If any are missing, it will notify you.

---

## Dataset Setup

### Obtaining the Dataset

The UNSW 2018 IoT Botnet Dataset must be downloaded separately.

#### Option 1: Official UNSW Source (Recommended)

1. **Visit**: https://research.unsw.edu.au/projects/bot-iot-dataset
2. **CloudStor Direct Link**: https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE
3. **Download**: Look for `UNSW_2018_IoT_Botnet_Full5pc_4.csv` or similar compressed file
4. **Extract**: If downloaded as ZIP, extract the CSV file

#### Option 2: Kaggle

1. **Visit**: https://www.kaggle.com/datasets/piyushagni5/unsw-nb15-and-bot-iot-datasets
2. **Login**: Create Kaggle account if needed (free)
3. **Download**: Find the Bot-IoT dataset files
4. **Select**: Download `UNSW_2018_IoT_Botnet_Full5pc_4.csv`

#### Option 3: University/Instructor Provided

If your instructor provided the dataset:
1. Locate the CSV file
2. Ensure it's named `UNSW_2018_IoT_Botnet_Full5pc_4.csv`
3. Proceed to placement step below

### Placing the Dataset

1. **Locate the `data/` folder** in your project directory
2. **Copy** `UNSW_2018_IoT_Botnet_Full5pc_4.csv` into this folder

Final structure should be:
```
Botnet-detection/
├── data/
│   └── UNSW_2018_IoT_Botnet_Full5pc_4.csv  ← Dataset here
├── src/
├── models/
└── ...
```

### Verify Dataset Placement

```bash
# On Windows
dir data\

# On macOS/Linux
ls data/
```

You should see `UNSW_2018_IoT_Botnet_Full5pc_4.csv` listed.

---

## Running the Project

### Method 1: Using Entry Point Script (Easiest)

```bash
python run_training.py
```

**What it does:**
1. ✓ Checks all dependencies are installed
2. ✓ Verifies dataset exists in `data/` folder
3. ✓ Creates necessary directories
4. ✓ Runs complete training pipeline
5. ✓ Saves all results to `results/` folder

**Expected runtime:** 5-20 minutes depending on your CPU.

### Method 2: Direct Training Module

```bash
cd src
python train.py
```

This runs the training directly without dependency checks.

### Method 3: Using Original Script (Legacy)

```bash
python Main.py
```

Runs the original monolithic script (requires dataset at `Dataset/UNSW_2018_IoT_Botnet_Full5pc_4.csv`).

---

## Understanding the Output

### Console Output

The training process displays:

#### 1. Dataset Loading
```
[STEP 1] Loading Dataset
----------------------------------------
Dataset loaded successfully: 668522 records, 46 features
Normal Records: 477
Attack Records: 668045
Imbalance Ratio: 1401.14:1 (Attack:Normal)
```

#### 2. Preprocessing
```
[STEP 3] Data Preprocessing
----------------------------------------
Dropped columns: ['pkSeqID', 'category', 'subcategory']
Encoding categorical features...
Normalizing features...
Applying SMOTE...
  Before SMOTE - Normal: 477, Attack: 668045
  After SMOTE - Normal: 668045, Attack: 668045
```

#### 3. Model Training
For each of the 4 models:
```
[4.1] Default Decision Tree
========================================
Accuracy  : 89.45%
Precision : 89.23%
Recall    : 89.15%
F-Score   : 89.19%
```

#### 4. Final Results Table
```
==================================================
FINAL RESULTS TABLE
==================================================
              Algorithm  Accuracy (%)  Precision (%)  Recall (%)  F-Score (%)
  Default Decision Tree         89.45          89.23       89.15        89.19
BOGP Optimized Decision Tree    94.67          94.52       94.48        94.50
              SVM Algorithm     76.34          75.89       76.12        76.01
          Extension CNN         98.23          98.15       98.09        98.12
==================================================
BEST PERFORMING MODEL: Extension CNN
Accuracy: 98.23%
==================================================
```

### Generated Files

After successful completion, check the `results/` folder:

#### 1. Performance Metrics CSV
**File**: `results/performance_metrics.csv`

Contains numerical results for all models. Open with Excel or any spreadsheet software.

#### 2. Evaluation Plots
**Files**: 
- `results/dt_default_evaluation.png`
- `results/dt_bogp_evaluation.png`
- `results/svm_evaluation.png`
- `results/cnn_evaluation.png`

Each contains:
- **Confusion Matrix** (left): Shows correct/incorrect predictions
  - Diagonal cells: Correct predictions
  - Off-diagonal: Errors
- **ROC Curve** (right): Model discrimination ability
  - Higher curve = Better performance
  - Area under curve (AUC) quantifies this

#### 3. Algorithm Comparison Chart
**File**: `results/algorithm_comparison.png`

Bar chart comparing all 4 models across all metrics side-by-side.

### Saved Models

**File**: `models/cnn_weights.hdf5`

The trained CNN model saved in HDF5 format. Can be loaded later for predictions without retraining.

**File**: `models/cnn_history.pckl`

Training history (loss and accuracy per epoch) saved as Python pickle.

---

## Troubleshooting

### Issue 1: Dataset Not Found

**Error:**
```
FileNotFoundError: Dataset not found at data/UNSW_2018_IoT_Botnet_Full5pc_4.csv
```

**Solution:**
1. Verify file is in `data/` folder
2. Check exact filename (case-sensitive on Linux/Mac)
3. Ensure file extension is `.csv` not `.csv.txt`

### Issue 2: Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```bash
pip install tensorflow
# or reinstall all
pip install -r requirements.txt
```

### Issue 3: Memory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
- Close other applications to free RAM
- Use 64-bit Python (not 32-bit)
- If still failing, reduce dataset size in code:
  ```python
  # In src/data_loader.py, add:
  dataset = dataset.sample(n=100000, random_state=42)
  ```

### Issue 4: TensorFlow Installation Issues

**On Windows:**
```bash
pip install tensorflow-cpu
```

**On Mac M1/M2:**
```bash
pip install tensorflow-macos
```

**On Linux:**
```bash
pip install tensorflow
```

### Issue 5: Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'models/cnn_weights.hdf5'
```

**Solution:**
- Run terminal as administrator (Windows)
- Use `sudo` on Linux/Mac (if appropriate)
- Check folder permissions

### Issue 6: Plots Not Showing

**Error:** Scripts runs but no plots appear

**Solution:**
If running on remote server or headless system:
```python
# In src/train.py, change:
run_training_pipeline(show_visualizations=False, save_results=True)
```
Plots will be saved to `results/` folder instead of displaying.

---

## Advanced Usage

### Customizing Training Parameters

Edit `run_training.py` or call training directly:

```python
from src.train import run_training_pipeline

run_training_pipeline(
    dataset_path="data/UNSW_2018_IoT_Botnet_Full5pc_4.csv",
    show_visualizations=True,  # Set False for faster execution
    save_results=True
)
```

### Modifying Model Hyperparameters

Edit `src/models.py`:

#### Change CNN epochs:
```python
# Line ~280 in src/models.py
def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    #                                                  ^^^^^ Change this
```

#### Change BOGP iterations:
```python
# In src/train.py, line ~75
bogp_model.optimize_hyperparameters(X_train, y_train, init_points=10, n_iter=5)
    #                                                              ^^        ^
```

#### Change SVM training limit:
```python
# In src/train.py, line ~90
svm_model = SVMClassifier(training_limit=100)
    #                                   ^^^
```

### Running Individual Models

```python
from src.data_loader import load_dataset
from src.preprocessing import DataPreprocessor
from src.models import CNNClassifier
from src.utils import calculate_metrics

# Load and prepare data
dataset = load_dataset("data/UNSW_2018_IoT_Botnet_Full5pc_4.csv")
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset)

# Train only CNN
cnn = CNNClassifier()
cnn.train(X_train, y_train, X_test, y_test, epochs=5)
predictions = cnn.predict(X_test)

# Evaluate
metrics = calculate_metrics("CNN", y_test, predictions)
print(metrics)
```

### Using Pre-trained CNN Model

If you've already trained the CNN:

```python
from src.models import CNNClassifier
import numpy as np

# Load model
cnn = CNNClassifier(model_path="models/cnn_weights.hdf5")
cnn.model = cnn._build_model((43, 1, 1), 2)
cnn.model.load_weights("models/cnn_weights.hdf5")

# Make predictions on new data
new_data = np.random.rand(10, 43)  # 10 samples, 43 features
predictions = cnn.predict(new_data)
print(predictions)  # 0=Normal, 1=Attack
```

### Exporting Results for Presentation

All results are saved to `results/` folder:

1. **For PowerPoint/Google Slides:**
   - Insert images from `results/*.png`
   - Copy table from `results/performance_metrics.csv`

2. **For LaTeX Report:**
   - Use `\includegraphics{results/algorithm_comparison.png}`
   - Import CSV with `\csvreader` package

3. **For Excel Analysis:**
   - Open `results/performance_metrics.csv`
   - Create custom charts and tables

### Jupyter Notebook Analysis

The original exploratory analysis notebook is available:

```bash
jupyter notebook notebooks/botnet_analysis.ipynb
```

Use this for:
- Interactive data exploration
- Experimenting with parameters
- Creating custom visualizations
- Step-by-step code execution

---

## Tips for College Presentation

### Key Points to Emphasize:

1. **Problem Statement:**
   - "IoT devices are vulnerable to botnet attacks"
   - "Need intelligent system to detect malicious traffic"

2. **Dataset Challenge:**
   - "Severe class imbalance (0.07% vs 99.93%)"
   - "Used SMOTE to synthetically balance classes"

3. **Multiple Approaches:**
   - "Compared 4 different algorithms"
   - "From simple Decision Trees to advanced CNN"

4. **Optimization Technique:**
   - "Used Bayesian Optimization for hyperparameter tuning"
   - "More efficient than grid search"

5. **Results:**
   - "CNN achieved 98%+ accuracy"
   - "Successfully detects botnet traffic"
   - Show the comparison chart

### Demonstration Flow:

1. Show the dataset file
2. Run `python run_training.py`
3. Explain what's happening at each step
4. Show the final results table
5. Display the comparison chart
6. Open individual evaluation plots
7. Discuss confusion matrix interpretation

---

## Getting Help

### Documentation Resources:
- **scikit-learn**: https://scikit-learn.org/stable/documentation.html
- **TensorFlow/Keras**: https://www.tensorflow.org/guide
- **SMOTE**: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- **Bayesian Optimization**: https://github.com/fmfn/BayesianOptimization

### Common Questions:

**Q: How long should training take?**
A: 5-20 minutes on modern laptops. SVM takes longest due to sample limitations.

**Q: Can I use a different dataset?**
A: Yes, but you'll need to modify preprocessing to match new feature names and structure.

**Q: Why is CNN so slow?**
A: Deep learning requires many iterations. Set `epochs=3` for faster results.

**Q: Can I run this on Google Colab?**
A: Yes! Upload the project and dataset to Google Drive, then run in a Colab notebook.

**Q: What if I get different results?**
A: Results may vary slightly due to random shuffling. This is normal.

---

## Next Steps

After successfully running the project:

1. **Analyze Results**: Study the confusion matrices and understand misclassifications
2. **Experiment**: Try different hyperparameters
3. **Extend**: Add new models (Random Forest, Gradient Boosting, LSTM)
4. **Deploy**: Create a web API for real-time botnet detection
5. **Report**: Write findings for academic submission

---

**Good luck with your project presentation!** 🎓