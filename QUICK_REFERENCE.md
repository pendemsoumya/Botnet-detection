# Quick Reference - IoT Botnet Detection

## 🚀 Get Started in 3 Steps

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Add Dataset
Place `UNSW_2018_IoT_Botnet_Full5pc_4.csv` in the `data/` folder

**Download from:**
- https://research.unsw.edu.au/projects/bot-iot-dataset
- https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE
- https://www.kaggle.com/datasets/piyushagni5/unsw-nb15-and-bot-iot-datasets

### 3️⃣ Run Training
```bash
python run_training.py
```

---

## 📁 Project Structure

```
Botnet-detection/
├── data/                    # Place dataset CSV here
├── models/                  # Trained models saved here
├── results/                 # Output plots and metrics
├── src/                     # Source code modules
├── notebooks/               # Jupyter notebooks
├── run_training.py         # Main entry point
├── Main.py                 # Original script (legacy)
├── README.md               # Full documentation
└── USAGE.md                # Step-by-step guide
```

---

## 🎯 What This Project Does

**Detects botnet attacks in IoT network traffic using Machine Learning**

**Dataset**: 668,522 network flows with 43 features
**Challenge**: Severe imbalance (0.07% normal vs 99.93% attack)
**Solution**: SMOTE balancing + 4 ML/DL models

**Models Compared:**
1. Default Decision Tree (baseline)
2. BOGP-Optimized Decision Tree (Bayesian optimization)
3. Support Vector Machine (SVM)
4. Convolutional Neural Network (CNN) ← **Best performer**

---

## 📊 Expected Results

| Model | Accuracy | Key Feature |
|-------|----------|-------------|
| Default DT | ~85-90% | Fast baseline |
| BOGP DT | ~92-96% | Optimized hyperparameters |
| SVM | ~70-80% | Limited training (50 samples) |
| CNN | **~96-99%** | Deep learning, best overall |

---

## 🔧 Common Commands

### Check Dependencies
```bash
python -c "import tensorflow, sklearn, pandas; print('All good!')"
```

### Run Individual Module
```bash
cd src
python train.py
```

### Open Jupyter Notebook
```bash
jupyter notebook notebooks/botnet_analysis.ipynb
```

### View Results
```bash
ls results/              # List output files
open results/*.png       # View plots (Mac)
explorer results\*.png   # View plots (Windows)
```

---

## 📝 Key Technical Concepts

### SMOTE (Balancing)
Creates synthetic minority class samples to balance dataset
- Before: 477 normal vs 668,045 attack
- After: 668,045 normal vs 668,045 attack

### BOGP (Optimization)
Bayesian Optimization with Gaussian Process
- Efficiently finds best hyperparameters
- Uses 5 init points + 2 iterations
- Optimizes: max_depth, min_samples_split, max_features

### CNN Architecture
```
Input (43 features)
  ↓
Conv2D: 32 filters (1×1)
  ↓
MaxPool (1×1)
  ↓
Conv2D: 16 filters (1×1)
  ↓
MaxPool (1×1)
  ↓
Dense: 256 units
  ↓
Output: 2 classes (Normal/Attack)
```

**Why (1×1) kernels?** Network features don't have spatial relationships like images. (1×1) convolution learns feature combinations without assuming locality.

---

## 🐛 Troubleshooting

### Dataset Not Found
```
ERROR: Dataset not found at data/UNSW_2018_IoT_Botnet_Full5pc_4.csv
```
**Fix**: Download dataset and place in `data/` folder

### TensorFlow Issues
```bash
# Windows
pip install tensorflow-cpu

# Mac M1/M2
pip install tensorflow-macos

# Linux
pip install tensorflow
```

### Memory Error
- Close other applications
- Use 64-bit Python
- Reduce dataset size in code if needed

### Import Errors
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📈 Output Files

### Metrics CSV
`results/performance_metrics.csv` - Performance table

### Comparison Chart
`results/algorithm_comparison.png` - Bar chart of all models

### Individual Evaluations
- `results/dt_default_evaluation.png`
- `results/dt_bogp_evaluation.png`
- `results/svm_evaluation.png`
- `results/cnn_evaluation.png`

Each contains: Confusion Matrix + ROC Curve

### Saved Models
`models/cnn_weights.hdf5` - Trained CNN model

---

## 🎓 For Presentation

### Talk About:
1. **Problem**: IoT botnet attacks (cybersecurity threat)
2. **Challenge**: Severe class imbalance (1401:1 ratio)
3. **Solution**: SMOTE + 4 different ML approaches
4. **Innovation**: Bayesian Optimization for tuning
5. **Results**: CNN achieves 96-99% accuracy

### Show:
1. Project structure (organized code)
2. Run the training live
3. Final results table
4. Comparison bar chart
5. CNN confusion matrix
6. Explain real-world impact

---

## 🔗 Documentation Links

- **Full Documentation**: [README.md](README.md)
- **Usage Guide**: [USAGE.md](USAGE.md)
- **Restructuring Notes**: [RESTRUCTURING_SUMMARY.md](RESTRUCTURING_SUMMARY.md)

---

## ⏱️ Time Estimates

- **Setup**: 10-15 minutes
- **Training**: 10-20 minutes
- **Total**: ~30 minutes to complete results

---

## ✅ Pre-Presentation Checklist

- [ ] Python 3.8-3.10 installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset in `data/` folder
- [ ] Training completed successfully
- [ ] Results in `results/` folder
- [ ] Reviewed README technical explanations
- [ ] Prepared to explain SMOTE and BOGP
- [ ] Can interpret confusion matrix
- [ ] Know CNN architecture details

---

**Need Help?** Check [USAGE.md](USAGE.md) Troubleshooting section

**Ready to Present?** All results in `results/` folder

**Questions?** Technical details in [README.md](README.md)

---

*Good luck with your presentation! 🎯*
