# IoT Botnet Detection Using Machine Learning and Deep Learning

A comprehensive college project implementing multiple machine learning approaches for detecting botnet attacks in IoT network traffic. This project compares the performance of traditional ML algorithms (Decision Trees, SVM) with deep learning (CNN) on the UNSW 2018 IoT Botnet Dataset.

## 📋 Project Overview

Botnets pose a significant security threat to Internet of Things (IoT) devices, enabling attackers to launch DDoS attacks, steal data, and compromise network integrity. This project develops and evaluates an intelligent detection system that analyzes network traffic patterns to identify malicious botnet activity.

### Key Objectives
- Implement and compare 4 different classification approaches
- Handle severely imbalanced dataset (0.07% normal vs 99.93% attack traffic)
- Achieve high accuracy in distinguishing normal traffic from botnet attacks
- Provide comprehensive performance analysis with confusion matrices and ROC curves

## 🎯 Dataset

### UNSW 2018 IoT Botnet Dataset
- **Source**: University of New South Wales (UNSW) Canberra Cyber
- **Records**: 668,522 network traffic instances
- **Features**: 43 network flow features (after preprocessing)
- **Classes**: Binary classification
  - Normal traffic (477 instances, 0.07%)
  - Attack traffic (668,045 instances, 99.93%)
- **Attack Types**: DDoS (UDP), Theft (Keylogging), and others

### Network Traffic Features
The dataset includes temporal, protocol, and statistical features:
- **Temporal**: `stime`, `ltime`, `dur` (timing information)
- **Protocol**: `proto`, `flgs`, `state` (connection metadata)
- **Network**: `saddr`, `sport`, `daddr`, `dport` (endpoints)
- **Traffic Statistics**: `pkts`, `bytes`, `spkts`, `dpkts`, `rate`, `mean`, `stddev`
- **Custom Metrics**: Aggregated connection statistics

## 🧠 Methodology

### 1. Data Preprocessing Pipeline

#### Step 1: Data Cleaning
- Remove irrelevant features: `pkSeqID` (sequence ID), `category`, `subcategory`
- Reduces dimensionality from 46 to 43 features

#### Step 2: Feature Encoding
Uses **Label Encoding** to convert categorical string features to numeric:
- `flgs`: TCP flags (SYN, ACK, FIN, etc.)
- `proto`: Protocol types (TCP, UDP, ICMP, etc.)
- `saddr`, `daddr`: Source and destination IP addresses
- `sport`, `dport`: Source and destination port numbers
- `state`: Connection state (ESTABLISHED, FIN_WAIT, etc.)

*Why Label Encoding?* For categorical features without ordinal relationship, Label Encoding is computationally efficient and suitable for tree-based models.

#### Step 3: Feature Normalization
Applies **MinMaxScaler** to normalize all features to [0, 1] range:

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

*Why Normalize?* 
- Ensures features are on the same scale
- Critical for distance-based algorithms (SVM, Neural Networks)
- Improves gradient descent convergence in deep learning

#### Step 4: Class Imbalance Handling with SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** addresses the severe class imbalance:

**How SMOTE Works:**
1. For each minority class sample (normal traffic):
   - Find k-nearest neighbors (default k=5)
   - Randomly select one neighbor
   - Create synthetic sample along the line segment:
   
   $$X_{synthetic} = X_i + \lambda \times (X_{neighbor} - X_i)$$
   
   where $\lambda \in [0, 1]$ is a random number

2. Repeat until classes are balanced

**Why SMOTE?**
- Simple oversampling (duplication) leads to overfitting
- Random undersampling loses valuable attack pattern information
- SMOTE generates diverse synthetic samples while preserving minority class distribution
- Creates smoother decision boundaries

**Impact on Dataset:**
- Before SMOTE: 477 normal vs 668,045 attack (1:1401 ratio)
- After SMOTE: ~668,045 normal vs 668,045 attack (1:1 ratio)

#### Step 5: Train-Test Split
- 80% training data (for model learning)
- 20% test data (for unbiased evaluation)

### 2. Machine Learning Models

#### Model 1: Default Decision Tree
**Algorithm**: CART (Classification and Regression Trees)

**Configuration:**
- `max_depth=1`: Creates decision stump (single split)
- Splits dataset based on one feature that provides maximum information gain

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