# Notebook 4: Data Preprocessing, Pipelines & Feature Scaling
### Part 4/30 – ML Mastery Series for Python Experts

This notebook serves as a critical bridge between Exploratory Data Analysis (EDA) and machine learning modeling. It focuses on building robust, production-ready preprocessing pipelines that prevent data leakage and ensure consistency across training and deployment.

## 🎯 Learning Objectives

By completing this notebook, you will master:
- **Leakage Detection**: Identifying and preventing data contamination in preprocessing.
- **Pipeline Construction**: Using `Pipeline` and `make_pipeline` to automate workflows.
- **ColumnTransformer**: Handling mixed data types (numerical and categorical) simultaneously.
- **Feature Scaling**: Understanding and applying various scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`).
- **Custom Transformers**: Creating domain-specific transformers using `BaseEstimator` and `TransformerMixin`.
- **Cross-Validation Hygiene**: Ensuring proper refitting of preprocessing steps within CV folds.

## 🛠️ Key Components

### 1. The Pipeline Architecture
The notebook emphasizes the transition from manual preprocessing to Scikit-learn Pipelines. This approach is non-negotiable for real-world ML to avoid code duplication and deployment mismatches.

### 2. Feature Scaling Comparison
A deep dive into different scaling strategies:
- **StandardScaler**: Centering and scaling to unit variance (best for Gaussian-like distributions).
- **MinMaxScaler**: Scaling to a fixed range (0 to 1), sensitive to outliers.
- **RobustScaler**: Using quartiles to handle data with significant outliers.
- **MaxAbsScaler**: Scaling by the maximum absolute value, ideal for sparse data.

### 3. Avoiding Data Leakage
Interactive demonstrations show how fitting a scaler on the entire dataset *before* splitting leads to optimistic bias. The notebook teaches the "Right Way" using pipelines to fit only on training data.

### 4. Advanced Preprocessing
- **Handling Heterogeneous Data**: Mastering `ColumnTransformer` for multi-layered feature engineering.
- **Grid Search within Pipelines**: Finding optimal preprocessing parameters alongside model hyperparameters.

## 🚀 Getting Started

Ensure you have the following libraries installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Open `data_feature_scaling.ipynb` in your Jupyter environment to begin the masterclass on data preprocessing.
