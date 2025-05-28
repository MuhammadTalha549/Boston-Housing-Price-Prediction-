# Boston-Housing-Price-Prediction-
# Boston Housing Price Prediction - From Scratch Implementation

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-From%20Scratch-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📋 Project Overview

This project implements three regression models **completely from scratch** to predict Boston housing prices without using high-level ML libraries like sklearn. All algorithms are built using only NumPy, Pandas, and Matplotlib for a deeper understanding of machine learning fundamentals.

## 🎯 Objectives

- Build Linear Regression, Random Forest, and XGBoost models from mathematical foundations
- Compare model performance using RMSE and R² metrics
- Analyze feature importance for tree-based models
- Create comprehensive visualizations for model evaluation

## 🚀 Models Implemented

### 1. **Linear Regression**
- Gradient descent optimization
- Cost function tracking
- Custom parameter initialization

### 2. **Random Forest**
- Custom Decision Tree implementation
- Bootstrap sampling for ensemble diversity
- Feature subsampling for variance reduction
- MSE-based splitting criterion

### 3. **XGBoost (Simplified Gradient Boosting)**
- Sequential tree building on residuals
- Learning rate control for regularization
- Gradient-based optimization

## 📊 Dataset

**Boston Housing Dataset** from Kaggle
- **Samples**: 506 houses
- **Features**: 13 numerical features
- **Target**: Median home value (medv) in $1000s

### Key Features:
- `lstat`: % lower status of population
- `rm`: Average number of rooms per dwelling
- `dis`: Weighted distances to employment centers
- `ptratio`: Pupil-teacher ratio by town
- `nox`: Nitric oxides concentration

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
```

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/boston-housing-price-prediction.git
cd boston-housing-price-prediction
```

## 📁 Project Structure

```
boston-housing-price-prediction/
├── README.md
├── data/
│   └── BostonHousing.csv          # Dataset file
├── notebooks/
│   └── boston_housing_analysis.ipynb  # Main analysis notebook
├── results/
│   ├── model_comparison.png        # Performance comparison
│   ├── feature_importance.png      # Feature rankings
│   └── actual_vs_predicted.png     # Prediction accuracy plots
└── requirements.txt
```

## 🏃‍♂️ How to Run

### Option 1: Jupyter Notebook (Recommended)
1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the analysis notebook**:
   ```
   notebooks/boston_housing_analysis.ipynb
   ```

3. **Run all cells** or execute step-by-step

### Option 2: Python Script
```bash
python notebooks/boston_housing_analysis.py
```

### Important: Update Dataset Path
Make sure your CSV file path is correct in the notebook:
```python
df = pd.read_csv('data/BostonHousing.csv')  # Update if needed
```

## 📈 Results

### Model Performance Comparison

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | 4.72 | 0.67 |
| Random Forest | 3.29 | 0.82 |
| **XGBoost** | **3.15** | **0.84** |

### Key Findings

1. **Best Performer**: XGBoost achieved the lowest RMSE (3.15) and highest R² (0.84)
2. **Feature Importance**: `lstat` and `rm` are consistently the most predictive features
3. **Model Complexity**: Tree-based models significantly outperform linear regression
4. **Prediction Quality**: All models show strong predictive capability (R² > 0.65)

### Top 5 Most Important Features
1. **lstat** (28-31%): Socioeconomic status indicator
2. **rm** (21-24%): Housing size (number of rooms)
3. **dis** (16-18%): Location accessibility
4. **ptratio** (12-13%): School quality indicator
5. **nox** (8-9%): Environmental quality

## 📊 Visualizations

The project generates comprehensive visualizations:

- **Performance Metrics**: Bar charts comparing RMSE and R² across models
- **Actual vs Predicted**: Scatter plots showing prediction accuracy
- **Feature Importance**: Rankings for Random Forest and XGBoost
- **Training Progress**: Cost function convergence for Linear Regression

## 🔍 Technical Implementation Details

### Data Preprocessing
- Custom normalization using training statistics only
- Manual train/test split implementation
- Feature scaling for improved convergence

### Model Architecture
- **No sklearn dependencies** for core algorithms
- All mathematical operations implemented from scratch
- Proper cross-validation and evaluation metrics

### Performance Optimization
- Efficient NumPy operations
- Memory-conscious implementations
- Reproducible results with fixed random seeds

## 🚧 Limitations & Future Work

### Current Limitations
- Simplified XGBoost (basic gradient boosting)
- Limited hyperparameter tuning
- No cross-validation implementation

### Potential Improvements
- Add k-fold cross-validation
- Implement hyperparameter optimization
- Add regularization techniques
- Include more advanced ensemble methods

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for:
- Bug fixes
- Performance improvements
- Additional model implementations
- Enhanced visualizations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/MuhammadTalha549)
- Email: talhamuhammad549@gmail.com

## 🙏 Acknowledgments

- Boston Housing Dataset from UCI ML Repository
- Kaggle for dataset hosting
- NumPy and Pandas communities for excellent documentation

---

⭐ **If you found this project helpful, please give it a star!** ⭐
