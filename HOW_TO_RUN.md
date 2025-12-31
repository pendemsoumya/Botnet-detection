# How to Run: IoT Botnet Detection Project

A complete, step-by-step guide to running the IoT Botnet Detection project from scratch. This guide assumes no prior setup and will walk you through every single step.

---

## 📋 Table of Contents

1. [Prerequisites Check](#1-prerequisites-check)
2. [Environment Setup](#2-environment-setup)
3. [Download the Dataset](#3-download-the-dataset)
4. [Install Dependencies](#4-install-dependencies)
5. [Run the Project](#5-run-the-project)
6. [Understanding the Output](#6-understanding-the-output)
7. [Troubleshooting](#7-troubleshooting)
8. [Advanced Options](#8-advanced-options)

---

## 1. Prerequisites Check

### What You Need

Before starting, ensure you have:

- **Python 3.8 or higher** installed
- **pip** (Python package manager)
- **4GB+ RAM** (recommended)
- **~500MB free disk space** (for dataset)
- **Internet connection** (for downloading dataset and dependencies)

### Check Your Python Version

Open a terminal/command prompt and run:

\`\`\`bash
python --version
\`\`\`

or

\`\`\`bash
python3 --version
\`\`\`

**Expected Output**: Something like \`Python 3.8.10\` or \`Python 3.10.5\`

**If you see an error or version < 3.8**:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Mac**: \`brew install python3\`
- **Linux (Ubuntu/Debian)**: \`sudo apt update && sudo apt install python3 python3-pip\`

### Check pip

\`\`\`bash
pip --version
\`\`\`

or

\`\`\`bash
pip3 --version
\`\`\`

**Expected Output**: \`pip 21.x.x from ...\`

---

## 2. Environment Setup

### Step 2.1: Clone or Download the Repository

**Option A: Using Git** (Recommended)

\`\`\`bash
git clone https://github.com/pendemsoumya/Botnet-detection.git
cd Botnet-detection
\`\`\`

**Option B: Download ZIP**

1. Go to [https://github.com/pendemsoumya/Botnet-detection](https://github.com/pendemsoumya/Botnet-detection)
2. Click green "Code" button → "Download ZIP"
3. Extract the ZIP file
4. Open terminal in the extracted folder

### Step 2.2: Verify Project Structure

Check that you're in the right directory:

\`\`\`bash
ls
\`\`\`

**You should see**:
\`\`\`
README.md
HOW_TO_RUN.md
requirements.txt
run_training.py
data/
models/
results/
src/
\`\`\`

### Step 2.3: Create Virtual Environment (Recommended)

A virtual environment keeps project dependencies isolated from your system Python.

**Create the virtual environment**:

\`\`\`bash
python3 -m venv venv
\`\`\`

**Activate the virtual environment**:

**On Linux/Mac**:
\`\`\`bash
source venv/bin/activate
\`\`\`

**On Windows (Command Prompt)**:
\`\`\`bash
venv\\Scripts\\activate
\`\`\`

**How to know it worked?**  
Your terminal prompt should now show \`(venv)\` at the beginning.

---

## 3. Download the Dataset

### Step 3.1: Understand What You Need

The project requires:
- **File**: \`UNSW_2018_IoT_Botnet_Full5pc_4.csv\`
- **Size**: ~150-200 MB
- **Location**: Must be placed in the \`data/\` folder

### Step 3.2: Download from Official Sources

**Option 1: UNSW CloudStor (Recommended)**

1. Visit: [https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE](https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE)
2. Look for: \`UNSW_2018_IoT_Botnet_Full5pc_4.csv\`
3. Click to download

**Option 2: Kaggle**

1. Visit: [https://www.kaggle.com/datasets/piyushagni5/unsw-nb15-and-bot-iot-datasets](https://www.kaggle.com/datasets/piyushagni5/unsw-nb15-and-bot-iot-datasets)
2. Click "Download" (may require Kaggle account)

### Step 3.3: Place Dataset in Correct Location

Move the downloaded file to the \`data/\` folder:

**On Linux/Mac**:
\`\`\`bash
mv ~/Downloads/UNSW_2018_IoT_Botnet_Full5pc_4.csv ./data/
\`\`\`

**On Windows**:
\`\`\`bash
move %USERPROFILE%\\Downloads\\UNSW_2018_IoT_Botnet_Full5pc_4.csv data\\
\`\`\`

### Step 3.4: Verify Dataset

\`\`\`bash
ls data/
\`\`\`

**Expected Output**: \`UNSW_2018_IoT_Botnet_Full5pc_4.csv\`

---

## 4. Install Dependencies

### Step 4.1: Install All Dependencies

Make sure your virtual environment is activated (you should see \`(venv)\` in your prompt).

Run:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

**This will take 2-5 minutes**.

### Step 4.2: Verify Installation

\`\`\`bash
python -c "import numpy, pandas, sklearn, tensorflow; print('✓ All packages installed')"
\`\`\`

**Expected Output**: \`✓ All packages installed\`

---

## 5. Run the Project

### Step 5.1: Run the Main Script

\`\`\`bash
python run_training.py
\`\`\`

or

\`\`\`bash
python3 run_training.py
\`\`\`

### Step 5.2: What Will Happen

The script will execute in this order:

1. **Dependency Check** (5 seconds)
2. **Dataset Verification** (2 seconds)
3. **Data Loading** (10-30 seconds)
4. **Data Preprocessing** (2-5 minutes) - Encoding, normalization, SMOTE
5. **Model Training** (5-15 minutes):
   - Default Decision Tree (~45 seconds)
   - BOGP-Optimized Decision Tree (~3 minutes)
   - SVM (~12 seconds)
   - CNN (~8 minutes or instant if pre-trained weights exist)
6. **Evaluation & Visualization** (1-2 minutes)
7. **Results Saved** ✓

**Total Time**: 10-20 minutes (first run), 5-10 minutes (subsequent runs with pre-trained CNN)

---

## 6. Understanding the Output

### Output Files Location

All results are saved in the \`results/\` directory:

\`\`\`bash
ls results/
\`\`\`

**You will see**:
- \`performance_metrics.csv\` - Numerical results table
- \`algorithm_comparison.png\` - Bar chart comparing all models
- \`dt_default_evaluation.png\` - Default DT confusion matrix + ROC
- \`dt_bogp_evaluation.png\` - Optimized DT confusion matrix + ROC
- \`svm_evaluation.png\` - SVM confusion matrix + ROC
- \`cnn_evaluation.png\` - CNN confusion matrix + ROC

### Performance Metrics

Open \`results/performance_metrics.csv\`:

| Algorithm | Accuracy | Precision | Recall | F-Score |
|-----------|----------|-----------|--------|---------|
| Default DT | ~87% | ~87% | ~88% | ~87% |
| BOGP DT | ~95% | ~94% | ~95% | ~95% |
| SVM | ~77% | ~75% | ~79% | ~77% |
| CNN | **~98%** | **~98%** | **~98%** | **~98%** |

**Key Insight**: CNN achieves the best performance across all metrics!

### Visualizations

**Confusion Matrix** shows:
- **True Positives (TP)**: Attacks correctly detected
- **True Negatives (TN)**: Normal traffic correctly identified
- **False Positives (FP)**: Normal traffic incorrectly flagged as attack
- **False Negatives (FN)**: Attacks that were missed (most critical!)

**ROC Curve** shows model quality:
- **AUC = 1.0**: Perfect model
- **AUC = 0.9-1.0**: Excellent
- **AUC = 0.8-0.9**: Good

---

## 7. Troubleshooting

### Problem 1: "Dataset not found"

**Solution**: Download the dataset and place it in the \`data/\` folder (see Step 3)

\`\`\`bash
ls data/  # Should show UNSW_2018_IoT_Botnet_Full5pc_4.csv
\`\`\`

### Problem 2: "Module not found"

**Solution**: Activate virtual environment and reinstall dependencies

\`\`\`bash
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
\`\`\`

### Problem 3: TensorFlow Installation Issues

**For Python 3.11+**: TensorFlow may not be compatible. Use Python 3.8-3.10 instead.

\`\`\`bash
python --version  # Check your version
\`\`\`

### Problem 4: Memory Error During SMOTE

**Solution**: Your system needs more RAM. Try reducing the dataset size by editing \`src/preprocessing.py\`:

In the \`balance_classes()\` method, add before SMOTE:
\`\`\`python
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data
\`\`\`

### Problem 5: CNN Training Too Slow

**Solution 1**: Reduce epochs in \`src/train.py\`:
\`\`\`python
cnn_model.train(X_train, y_train, epochs=2)  # Instead of 5
\`\`\`

**Solution 2**: The project saves CNN weights. On the second run, it will load pre-trained weights (much faster!).

---

## 8. Advanced Options

### Run Only Specific Models

Edit \`src/train.py\` and comment out models you don't want:

\`\`\`python
# results['SVM'] = train_svm(...)  # Skip SVM
\`\`\`

### Modify Hyperparameters

Edit \`src/train.py\`:

\`\`\`python
# Change test size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # 30% instead of 20%
)

# Change CNN epochs
cnn_model.train(X_train, y_train, epochs=10)  # 10 instead of 5
\`\`\`

### Use GPU for CNN Training

If you have an NVIDIA GPU with CUDA:

\`\`\`bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
\`\`\`

If GPU is detected, TensorFlow will automatically use it (5-10x faster!).

---

## Quick Reference

### Complete Setup (One Command)

**Linux/Mac**:
\`\`\`bash
git clone https://github.com/pendemsoumya/Botnet-detection.git && cd Botnet-detection && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
\`\`\`

**Windows**:
\`\`\`bash
git clone https://github.com/pendemsoumya/Botnet-detection.git && cd Botnet-detection && python -m venv venv && venv\\Scripts\\activate && pip install -r requirements.txt
\`\`\`

### Run Project

\`\`\`bash
python run_training.py
\`\`\`

### View Results

\`\`\`bash
ls results/
cat results/performance_metrics.csv
\`\`\`

### Clean Generated Files

\`\`\`bash
rm -rf results/*.png results/*.csv
\`\`\`

---

## Summary Checklist

- [ ] Python 3.8-3.10 installed
- [ ] Project downloaded
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Dataset downloaded and placed in \`data/\` folder
- [ ] \`python run_training.py\` executed
- [ ] Results generated in \`results/\` folder

**Congratulations!** For more technical details, see [README.md](README.md).
