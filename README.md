# IoT Botnet Detection Using Machine Learning

A comprehensive machine learning project that detects botnet attacks in IoT network traffic by comparing traditional ML algorithms (Decision Trees, SVM) with deep learning (CNN) on the UNSW 2018 IoT Botnet Dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Methodology & Algorithms](#methodology--algorithms)
  - [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
  - [Machine Learning Models](#machine-learning-models)
  - [Evaluation Metrics](#evaluation-metrics)
- [Expected Results](#expected-results)
- [Technical Details](#technical-details)
- [Educational Value](#educational-value)
- [Contributing](#contributing)

---

## 🎯 Overview

Botnets pose a critical security threat to IoT devices, enabling DDoS attacks, data theft, and network compromise. This project implements an intelligent detection system that analyzes network traffic patterns to identify malicious botnet activity using multiple machine learning approaches.

### Project Objectives

- **Compare 4 classification approaches**: Decision Trees (default and optimized), SVM, and CNN
- **Handle severe class imbalance**: 0.07% normal vs 99.93% attack traffic using SMOTE
- **Achieve high detection accuracy**: Distinguish normal traffic from botnet attacks with >95% accuracy
- **Provide comprehensive analysis**: Confusion matrices, ROC curves, and performance comparisons
- **Demonstrate ML pipeline**: From data loading to model evaluation and visualization

---

## 📊 Dataset Information

### UNSW 2018 IoT Botnet Dataset

**Source**: University of New South Wales (UNSW) Canberra Cyber Research Centre

**Dataset Characteristics**:
- **Total Records**: 668,522 network traffic instances
- **Features**: 46 columns (43 after preprocessing)
- **Classification**: Binary (Normal vs Attack)
- **Class Distribution**:
  - Normal traffic: 477 instances (0.07%)
  - Attack traffic: 668,045 instances (99.93%)
- **Attack Types**: DDoS (UDP), Data Theft (Keylogging), OS/Service Scans

### Feature Categories

The dataset includes various network flow features:

**Temporal Features**:
- `stime`, `ltime`: Start and last timestamp
- `dur`: Connection duration

**Protocol Information**:
- `proto`: Protocol type (TCP, UDP, ICMP, etc.)
- `flgs`: TCP flags (SYN, ACK, FIN, RST, PSH, URG)
- `state`: Connection state (ESTABLISHED, FIN_WAIT, etc.)

**Network Endpoints**:
- `saddr`, `daddr`: Source and destination IP addresses
- `sport`, `dport`: Source and destination ports

**Traffic Statistics**:
- `pkts`, `bytes`: Total packets and bytes
- `spkts`, `dpkts`: Source and destination packets
- `sbytes`, `dbytes`: Source and destination bytes
- `rate`: Transfer rate
- `mean`, `stddev`: Statistical metrics

**Behavioral Features**:
- Aggregated connection statistics
- Flow-level metadata

---

## 📁 Project Structure

```
Botnet-detection/
│
├── README.md                       # Project overview and documentation (this file)
├── HOW_TO_RUN.md                  # Step-by-step running instructions
├── requirements.txt                # Python dependencies
├── run_training.py                 # Main entry point script
│
├── data/                           # Dataset directory
│   └── UNSW_2018_IoT_Botnet_Full5pc_4.csv  (download required)
│
├── models/                         # Saved trained models
│   ├── cnn_weights.hdf5           # Pre-trained CNN weights
│   └── cnn_history.pckl           # Training history (generated)
│
├── results/                        # Output directory
│   ├── performance_metrics.csv    # Model comparison table
│   ├── algorithm_comparison.png   # Bar chart comparison
│   ├── dt_default_evaluation.png  # Decision Tree plots
│   ├── dt_bogp_evaluation.png     # Optimized DT plots
│   ├── svm_evaluation.png         # SVM plots
│   └── cnn_evaluation.png         # CNN plots
│
└── src/                            # Source code modules
    ├── data_loader.py             # Dataset loading and validation
    ├── preprocessing.py           # Data preprocessing pipeline
    ├── models.py                  # ML/DL model implementations
    ├── utils.py                   # Metrics, visualization, reporting
    └── train.py                   # Training orchestration pipeline
```

### Source Code Modules

#### `src/data_loader.py` (66 lines)
- **Purpose**: Load and validate dataset
- **Key Functions**:
  - `load_dataset()`: Load CSV with error handling
  - `display_statistics()`: Show dataset info

#### `src/preprocessing.py` (163 lines)
- **Purpose**: Complete preprocessing pipeline
- **Key Class**: `DataPreprocessor`
- **Methods**:
  - Feature removal (pkSeqID, category, subcategory)
  - Label encoding (7 categorical features)
  - MinMax normalization
  - SMOTE class balancing
  - Train-test split

#### `src/models.py` (380 lines)
- **Purpose**: ML/DL model implementations
- **Key Classes**:
  - `DecisionTreeDefault`: Baseline DT (max_depth=1)
  - `DecisionTreeBOGP`: Bayesian-optimized DT (~100 lines)
  - `SVMClassifier`: Support Vector Machine (~40 lines)
  - `CNNModel`: Deep learning CNN (~200 lines)

#### `src/utils.py` (250 lines)
- **Purpose**: Evaluation and visualization
- **Key Functions**:
  - `calculate_metrics()`: Accuracy, precision, recall, F-score
  - `plot_evaluation()`: Confusion matrix + ROC curve
  - `plot_comparison()`: Algorithm comparison charts
  - `save_results()`: Export metrics to CSV

#### `src/train.py` (172 lines)
- **Purpose**: Main training pipeline
- **Key Function**: `run_training_pipeline()`
- **Pipeline Steps**:
  1. Load dataset
  2. Generate EDA visualizations
  3. Preprocess data (encode, normalize, balance)
  4. Train all 4 models sequentially
  5. Evaluate and compare performance
  6. Save results and visualizations

---

## 🧠 Methodology & Algorithms

### Data Preprocessing Pipeline

#### 1. Data Cleaning
- Remove irrelevant features: `pkSeqID`, `category`, `subcategory`
- Reduces dimensionality from 46 to 43 features

#### 2. Feature Encoding
Uses **Label Encoding** for categorical features:
- `flgs`: TCP flags → numeric codes
- `proto`: Protocol types → numeric codes
- `saddr`, `daddr`: IP addresses → numeric codes
- `sport`, `dport`: Port numbers → numeric codes
- `state`: Connection states → numeric codes

**Why Label Encoding?**
- Computationally efficient for tree-based models
- Suitable for categorical features without ordinal relationships
- Preserves feature space dimensionality

#### 3. Feature Normalization
Applies **MinMaxScaler** to normalize features to [0, 1]:

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Benefits**:
- Equal feature scale for distance-based algorithms
- Improved gradient descent convergence
- Required for SVM and Neural Networks

#### 4. Class Imbalance Handling - SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** addresses severe imbalance:

**Algorithm**:
1. For each minority class sample:
   - Find k=5 nearest neighbors
   - Create synthetic samples along line segments:
   
   $$X_{synthetic} = X_i + \lambda \times (X_{neighbor} - X_i)$$
   
   where $\lambda \in [0, 1]$ is random

**Why SMOTE?**
- Avoids overfitting from simple duplication
- Preserves minority class distribution
- Creates smoother decision boundaries
- Better than undersampling (preserves attack patterns)

**Impact**:
- Before: 477 normal : 668,045 attack (1:1401 ratio)
- After: ~668,045 normal : 668,045 attack (1:1 ratio)

#### 5. Train-Test Split
- 80% training data (model learning)
- 20% test data (unbiased evaluation)

---

### Machine Learning Models

#### Model 1: Default Decision Tree

**Configuration**: 
- Algorithm: CART (Classification and Regression Trees)
- `max_depth=1`: Single decision stump

**How it works**:
- Selects one feature providing maximum information gain
- Creates single split to classify data

$$IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$$

where entropy: $H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$

**Characteristics**:
- ✅ Fast training and prediction
- ✅ Interpretable (single rule)
- ❌ Limited expressiveness
- **Purpose**: Baseline for comparison

---

#### Model 2: BOGP-Optimized Decision Tree

**Optimization Method**: Bayesian Optimization with Gaussian Process

**Hyperparameters Optimized**:
- `max_depth`: Tree depth (range: 5-10)
- `min_samples_split`: Split threshold (range: 0.1-0.9)
- `max_features`: Features per split (range: 0.1-0.9)

**BOGP Algorithm**:

1. **Surrogate Model**: Gaussian Process models objective function
   
   $$f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))$$

2. **Acquisition Function**: Expected Improvement guides search
   
   $$EI(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]$$

3. **Iterative Process**:
   - Evaluate at selected points
   - Update GP posterior
   - Select next point via acquisition
   - Repeat until convergence

**Why BOGP over Grid Search?**
- **Efficient**: Fewer evaluations (7 vs hundreds)
- **Intelligent**: Uses past results to guide search
- **Global**: Balances exploration vs exploitation
- **Effective**: Ideal for expensive cross-validation

**Characteristics**:
- ✅ Systematically finds optimal hyperparameters
- ✅ Significant improvement over default
- ✅ Better generalization

---

#### Model 3: Support Vector Machine (SVM)

**Configuration**:
- Kernel: RBF (Radial Basis Function)
- **Training Samples**: 50 (due to computational constraints)

**Objective**: Find optimal hyperplane maximizing margin

$$\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$$

subject to: $y_i(w \cdot x_i + b) \geq 1 - \xi_i$

**RBF Kernel**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$

**Computational Limitation**:
- SVM training: $O(n^2)$ to $O(n^3)$ complexity
- Full dataset (1.3M after SMOTE) would require hours/days
- Limited to 50 samples for comparison purposes

**Characteristics**:
- ✅ Effective in high-dimensional spaces
- ✅ Robust to outliers
- ❌ Computationally expensive for large datasets
- ❌ Limited training data affects performance

---

#### Model 4: Convolutional Neural Network (CNN)

**Architecture**:

```
Input: (43, 1, 1)           # 43 features as 1D "image"
    ↓
Conv2D: 32 filters, (1×1)   # 32 feature combinations
    ↓  ReLU activation
MaxPooling2D: (1×1)         # Dimensionality control
    ↓
Conv2D: 16 filters, (1×1)   # 16 higher-level abstractions
    ↓  ReLU activation
MaxPooling2D: (1×1)         # Dimensionality control
    ↓
Flatten                      # Convert to 1D
    ↓
Dense: 256 units            # Complex pattern learning
    ↓  ReLU activation
Output: 2 units             # Binary classification
    Softmax activation
```

**Training Configuration**:
- **Optimizer**: Adam (adaptive learning rate)
- **Loss**: Categorical Cross-Entropy
  
  $$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

- **Epochs**: 5
- **Batch Size**: 32
- **Model Checkpoint**: Saves best weights

**Why (1×1) Convolutions?**
- Network features lack spatial relationships (unlike images)
- (1×1) convolution = feature transformation/combination
- Learns cross-feature interactions
- Acts as learnable feature engineering

**Architecture Rationale**:
1. **First Conv Layer (32 filters)**: Learns 32 feature combinations
2. **Second Conv Layer (16 filters)**: Learns higher-level abstractions
3. **Dense Layer (256 units)**: Combines features for complex decisions
4. **Softmax Output**: Converts to class probabilities

**Characteristics**:
- ✅ Automatic feature learning
- ✅ Captures complex non-linear patterns
- ✅ Hierarchical representation
- ✅ Best overall performance
- 💾 Saves/loads pre-trained weights

---

### Evaluation Metrics

All models evaluated using:

#### Accuracy
Proportion of correct predictions:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

#### Precision
Correct positive predictions:

$$Precision = \frac{TP}{TP + FP}$$

*Importance*: Low false positives (avoid flagging normal traffic)

#### Recall
Detected actual positives:

$$Recall = \frac{TP}{TP + FN}$$

*Importance*: High detection rate (catch real attacks)

#### F-Score
Harmonic mean balancing precision and recall:

$$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

*Importance*: Balanced metric for imbalanced data

**Note**: Uses **macro averaging** to give equal weight to both classes.

### Visualizations

**Confusion Matrix**:
- True Positives (TP): Correctly identified attacks
- True Negatives (TN): Correctly identified normal traffic
- False Positives (FP): Normal flagged as attack (Type I error)
- False Negatives (FN): Missed attacks (Type II error)

**ROC Curve**:
- Plots TPR vs FPR at various thresholds
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing

---

## 📊 Expected Results

Typical performance on UNSW 2018 IoT Botnet Dataset:

| Algorithm | Accuracy | Precision | Recall | F-Score |
|-----------|----------|-----------|--------|---------|
| **Default Decision Tree** | ~85-90% | ~85-90% | ~85-90% | ~85-90% |
| **BOGP Decision Tree** | ~92-96% | ~92-96% | ~92-96% | ~92-96% |
| **SVM** (limited) | ~70-80% | ~70-80% | ~70-80% | ~70-80% |
| **CNN** | **~96-99%** | **~96-99%** | **~96-99%** | **~96-99%** |

### Key Findings

1. **CNN achieves highest performance** across all metrics
2. **BOGP optimization** significantly improves Decision Tree (6-8% gain)
3. **SVM limited by training size** - full training would improve results
4. **SMOTE successfully balances** dataset for all models
5. **All models exceed 70% accuracy** demonstrating feasibility of ML-based botnet detection

---

## 🔧 Technical Details

### Dependencies

**Core Libraries**:
- Python 3.8-3.10
- NumPy 1.21.0-1.24.0
- Pandas 1.3.0-2.0.0
- Scikit-learn 1.0.0-1.3.0
- TensorFlow 2.10.0-2.13.0

**Additional**:
- Matplotlib, Seaborn (visualization)
- Imbalanced-learn (SMOTE)
- Bayesian-optimization (BOGP)

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB+ recommended
- **Storage**: ~500MB for dataset
- **CPU**: Multi-core recommended for CNN training
- **GPU**: Optional (speeds up CNN training)

### Customization Options

**Modify hyperparameters in `src/train.py`**:
- Test split ratio (default: 0.2)
- CNN epochs (default: 5)
- CNN batch size (default: 32)
- BOGP iterations (default: 2)
- SVM training samples (default: 50)

**Add new models in `src/models.py`**:
1. Create model class
2. Implement `train()` and `predict()` methods
3. Add evaluation in `src/train.py`

**Custom visualizations in `src/utils.py`**:
- Modify plot styles and colors
- Add additional metrics
- Create custom comparison charts

---

## 🎓 Educational Value

### Learning Outcomes

This project demonstrates:

1. **Real-World Data Handling**: Imbalanced datasets in cybersecurity
2. **Feature Engineering**: Encoding and normalization techniques
3. **Class Imbalance Solutions**: SMOTE mathematical foundation
4. **Hyperparameter Optimization**: Bayesian methods
5. **Deep Learning**: CNN architecture design
6. **Model Comparison**: Systematic evaluation methodology
7. **Software Engineering**: Modular code organization
8. **Scientific Reporting**: Visualization and metrics

### Key Concepts Explained

- **Why multiple models?** Comprehensive comparison validates results
- **Why SMOTE?** Demonstrates handling imbalanced security data
- **Why BOGP?** Shows modern optimization techniques
- **Why CNN with (1×1) kernels?** Adapts image processing to tabular data
- **Why 5 epochs?** Balances training time with convergence

---

## 🤝 Contributing

This is an educational project. Suggestions welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

- **GitHub**: [@pendemsoumya](https://github.com/pendemsoumya)
- **Repository**: [Botnet-detection](https://github.com/pendemsoumya/Botnet-detection)

---

## 📄 License

Educational purposes. Check UNSW dataset license for research usage terms.

---

## 🙏 Acknowledgments

- **UNSW Canberra Cyber** for Bot-IoT dataset
- **scikit-learn, TensorFlow/Keras** communities
- **Bayesian Optimization** library contributors

---

## 📚 Citations

**Project Citation**:
```bibtex
@misc{botnet_detection_2024,
  title={IoT Botnet Detection Using Machine Learning and Deep Learning},
  author={Pendem Soumya},
  year={2024},
  howpublished={\url{https://github.com/pendemsoumya/Botnet-detection}}
}
```

**Dataset Citation**:
```bibtex
@article{koroniotis2019towards,
  title={Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset},
  author={Koroniotis, Nickolaos and Moustafa, Nour and Sitnikova, Elena and Turnbull, Benjamin},
  journal={Future Generation Computer Systems},
  volume={100},
  pages={779--796},
  year={2019},
  publisher={Elsevier}
}
```

---

**For detailed step-by-step instructions on how to run this project, see [HOW_TO_RUN.md](HOW_TO_RUN.md)**



**Information Gain Formula:**

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

where $H(S)$ is entropy: $H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$

**Advantages:**
- Fast training and prediction
- Interpretable (single rule)
- Baseline for comparison

**Limitations:**
- Limited expressiveness (max_depth=1)
- May underfit complex patterns

#### Model 2: BOGP-Optimized Decision Tree

**Bayesian Optimization with Gaussian Process (BOGP)** for hyperparameter tuning.

**Hyperparameters Optimized:**
- `max_depth`: Tree depth (range: 5-10)
- `min_samples_split`: Minimum samples to split node (range: 0.1-0.9)
- `max_features`: Features to consider per split (range: 0.1-0.9)

**How BOGP Works:**

1. **Surrogate Model**: Gaussian Process models the objective function $f(x)$ (cross-validation score)
   
   $$f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))$$
   
   where $\mu(x)$ is mean function and $k(x, x')$ is covariance kernel

2. **Acquisition Function**: Expected Improvement (EI) selects next point to evaluate:
   
   $$EI(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]$$
   
   where $f(x^+)$ is current best value

3. **Iterative Process**:
   - Evaluate objective at selected point
   - Update Gaussian Process posterior
   - Select next point using acquisition function
   - Repeat until convergence

**Why BOGP over Grid Search?**
- **Efficiency**: Requires fewer evaluations (5 init + 2 iterations vs hundreds in grid search)
- **Intelligence**: Uses past results to guide search
- **Global Optimization**: Balances exploration and exploitation
- **Handles Expensive Functions**: Ideal for cross-validation (5 folds per evaluation)

**Advantages:**
- Systematically finds optimal hyperparameters
- Significantly better than default configuration
- Generalizes well to unseen data

#### Model 3: Support Vector Machine (SVM)

**Algorithm**: SVM with RBF kernel (default)

**Objective**: Find optimal hyperplane that maximizes margin between classes:

$$\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$$

subject to: $y_i(w \cdot x_i + b) \geq 1 - \xi_i$

where:
- $w$: weight vector (hyperplane normal)
- $b$: bias term
- $C$: regularization parameter
- $\xi_i$: slack variables (allow misclassification)

**RBF Kernel**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$

**Training Limitation:**
- Only uses **50 samples** due to computational complexity
- SVM training has $O(n^2)$ to $O(n^3)$ time complexity
- Full dataset (1.3M samples after SMOTE) would require hours/days
- This is acknowledged as a limitation for comparison purposes

**Advantages:**
- Effective in high-dimensional spaces (43 features)
- Robust to outliers (margin maximization)
- Theoretically sound (maximizes generalization)

**Limitations:**
- Computationally expensive for large datasets
- Limited training data affects performance in this project

#### Model 4: Convolutional Neural Network (CNN)

**Deep Learning Architecture:**

```
Input Layer: (43, 1, 1) - 43 features reshaped as "image"
    ↓
Conv2D: 32 filters, (1×1) kernel, ReLU activation
    ↓
MaxPooling2D: (1×1) pool size
    ↓
Conv2D: 16 filters, (1×1) kernel, ReLU activation
    ↓
MaxPooling2D: (1×1) pool size
    ↓
Flatten Layer
    ↓
Dense: 256 units, ReLU activation
    ↓
Output: 2 units (Normal/Attack), Softmax activation
```

**Training Configuration:**
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: Categorical Cross-Entropy
  
  $$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
  
- **Epochs**: 5
- **Batch Size**: 32
- **Callbacks**: ModelCheckpoint (saves best model)

**Architecture Explanation:**

1. **Why (1×1) Convolution?**
   - Network traffic features don't have spatial relationships like images
   - (1×1) convolution acts as feature transformation/combination
   - Learns cross-feature interactions without assuming locality
   - Effectively a learnable feature engineering layer

2. **Two Convolutional Layers:**
   - **First layer (32 filters)**: Learns 32 different feature combinations
   - **Second layer (16 filters)**: Learns higher-level abstractions
   - Creates hierarchical feature representation

3. **MaxPooling:**
   - Provides translation invariance
   - Reduces parameters (regularization effect)
   - With (1×1) size, primarily serves as pass-through for architecture consistency

4. **Dense Layer (256 units):**
   - Combines convolutional features
   - Learns complex non-linear decision boundaries
   - ReLU prevents vanishing gradients

5. **Output Layer:**
   - Softmax converts logits to probabilities: $P(y=c|x) = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}$
   - Two neurons for binary classification

**Advantages:**
- Automatic feature learning (no manual feature engineering)
- Captures complex non-linear patterns
- Hierarchical representation learning
- Can improve with more training data

**Model Persistence:**
- Saves best weights to `models/cnn_weights.hdf5`
- Loads pre-trained model if exists (avoids retraining)
- Training history saved for analysis

### 3. Evaluation Metrics

For each model, we calculate:

#### Accuracy
Proportion of correct predictions:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

#### Precision
Proportion of positive predictions that are correct:

$$Precision = \frac{TP}{TP + FP}$$

*Why it matters:* Low false positives (don't flag normal traffic as attacks)

#### Recall (Sensitivity)
Proportion of actual positives correctly identified:

$$Recall = \frac{TP}{TP + FN}$$

*Why it matters:* High detection rate (catch real attacks)

#### F-Score (F1-Score)
Harmonic mean of precision and recall:

$$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

*Why it matters:* Balances precision and recall (important for imbalanced data)

**Note**: We use **macro averaging** across classes to give equal weight to both normal and attack classes, preventing bias toward the majority class.

### 4. Visualization and Analysis

#### Confusion Matrix
2×2 matrix showing:
- True Positives (TP): Correctly identified attacks
- True Negatives (TN): Correctly identified normal traffic
- False Positives (FP): Normal traffic flagged as attack (Type I error)
- False Negatives (FN): Missed attacks (Type II error)

#### ROC Curve (Receiver Operating Characteristic)
- Plots True Positive Rate vs False Positive Rate
- Area Under Curve (AUC) measures overall performance
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing

## 📁 Project Structure

```
Botnet-detection/
│
├── data/                           # Dataset directory
│   └── UNSW_2018_IoT_Botnet_Full5pc_4.csv
│
├── models/                         # Saved model weights
│   ├── cnn_weights.hdf5           # Trained CNN model
│   └── cnn_history.pckl           # Training history
│
├── results/                        # Output results
│   ├── performance_metrics.csv    # Performance table
│   ├── algorithm_comparison.png   # Comparison chart
│   └── *_evaluation.png           # Individual model plots
│
├── src/                            # Source code modules
│   ├── data_loader.py             # Dataset loading utilities
│   ├── preprocessing.py           # Data preprocessing pipeline
│   ├── models.py                  # ML/DL model implementations
│   ├── utils.py                   # Visualization and metrics
│   └── train.py                   # Main training orchestrator
│
├── notebooks/                      # Jupyter notebooks
│   └── botnet_analysis.ipynb      # Exploratory data analysis
│
├── run_training.py                 # Main entry point script
├── Main.py                         # Original monolithic script (legacy)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── USAGE.md                        # Detailed usage guide
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- ~500MB disk space for dataset

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/pendemsoumya/Botnet-detection.git
cd Botnet-detection
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download dataset:**
Place `UNSW_2018_IoT_Botnet_Full5pc_4.csv` in the `data/` directory.

**Dataset Sources:**
- Official: https://research.unsw.edu.au/projects/bot-iot-dataset
- CloudStor: https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE
- Kaggle: https://www.kaggle.com/datasets/piyushagni5/unsw-nb15-and-bot-iot-datasets

### Running the Project

**Option 1: Using the entry point script (recommended)**
```bash
python run_training.py
```
This script will:
- Check dependencies
- Verify dataset availability
- Run complete training pipeline
- Save results to `results/` directory

**Option 2: Using the training module directly**
```bash
cd src
python train.py
```

**Option 3: Using the original script (legacy)**
```bash
python Main.py
```

### Expected Output

The training process will:
1. Load and analyze the dataset
2. Display exploratory visualizations
3. Preprocess data (encode, normalize, balance)
4. Train all 4 models sequentially
5. Evaluate and compare performance
6. Generate result files

**Results saved:**
- `results/performance_metrics.csv`: Performance table
- `results/algorithm_comparison.png`: Bar chart comparison
- `results/*_evaluation.png`: Individual confusion matrices and ROC curves
- `models/cnn_weights.hdf5`: Trained CNN model

## 📊 Expected Results

Based on the UNSW 2018 IoT Botnet Dataset, typical performance metrics:

| Algorithm | Accuracy | Precision | Recall | F-Score |
|-----------|----------|-----------|--------|---------|
| Default Decision Tree | ~85-90% | ~85-90% | ~85-90% | ~85-90% |
| BOGP Decision Tree | ~92-96% | ~92-96% | ~92-96% | ~92-96% |
| SVM (limited) | ~70-80% | ~70-80% | ~70-80% | ~70-80% |
| Extension CNN | **~96-99%** | **~96-99%** | **~96-99%** | **~96-99%** |

**Key Findings:**
- CNN achieves highest overall performance
- BOGP optimization significantly improves Decision Tree
- SVM performance limited by training sample size
- SMOTE successfully balances dataset for all models

## 🎓 Educational Value

### Learning Outcomes

This project demonstrates:

1. **Data Preprocessing**: Handling real-world imbalanced datasets
2. **Feature Engineering**: Label encoding and normalization techniques
3. **Class Imbalance**: SMOTE and its mathematical foundation
4. **Optimization**: Bayesian optimization for hyperparameter tuning
5. **Deep Learning**: CNN architecture design and training
6. **Model Comparison**: Systematic evaluation methodology
7. **Software Engineering**: Modular code organization
8. **Visualization**: Performance analysis and reporting

### Key Concepts Explained

- **Why multiple models?** Provides comprehensive comparison and validates results
- **Why SMOTE?** Demonstrates handling of imbalanced data (common in security)
- **Why BOGP?** Shows modern hyperparameter optimization techniques
- **Why CNN with (1×1) kernels?** Adapts image processing techniques to tabular data
- **Why 5 epochs?** Balances training time with convergence for college project scope

## 🔧 Customization

### Modifying Hyperparameters

Edit `src/train.py` to adjust:
- Test split ratio (default: 0.2)
- CNN epochs (default: 5)
- CNN batch size (default: 32)
- BOGP iterations (default: 2)
- SVM training limit (default: 50)

### Adding New Models

1. Create model class in `src/models.py`
2. Implement `train()` and `predict()` methods
3. Add evaluation in `src/train.py`

### Custom Visualizations

Modify functions in `src/utils.py` to:
- Change plot styles and colors
- Add additional metrics
- Create custom comparison charts

## 📝 Citation

If you use this project in your research or coursework, please cite:

```
@misc{botnet_detection_2024,
  title={IoT Botnet Detection Using Machine Learning and Deep Learning},
  author={[Your Name]},
  year={2024},
  howpublished={\\url{https://github.com/pendemsoumya/Botnet-detection}}
}
```

**Dataset Citation:**
```
@inproceedings{koroniotis2019towards,
  title={Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset},
  author={Koroniotis, Nickolaos and Moustafa, Nour and Sitnikova, Elena and Turnbull, Benjamin},
  booktitle={Future Generation Computer Systems},
  volume={100},
  pages={779--796},
  year={2019},
  publisher={Elsevier}
}
```

## 🤝 Contributing

This is a college project, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or collaboration:
- GitHub: [pendemsoumya](https://github.com/pendemsoumya)
- Project Link: https://github.com/pendemsoumya/Botnet-detection

## 📄 License

This project is for educational purposes. Please check the UNSW dataset license for research usage terms.

## 🙏 Acknowledgments

- UNSW Canberra Cyber for providing the Bot-IoT dataset
- scikit-learn, TensorFlow/Keras communities for excellent ML libraries
- Bayesian Optimization library contributors
- College faculty and mentors for guidance

---

**Note**: This project was developed as a college assignment to demonstrate practical application of machine learning in cybersecurity. The code preserves the original implementation logic to match the accompanying academic report.