# Notebook 6: Logistic Regression & Multiclass Classification

**Part 6/30** – ML Mastery Series for Python Experts

**Author:** Tassawar Abbas  
**Email:** abbas829@gmail.com

---

## 📋 Overview

This notebook takes you from the mathematical fundamentals of logistic regression through practical implementation, multiclass classification strategies, probability calibration, and proper evaluation metrics. You'll learn not just how to build classifiers, but how to interpret them, calibrate their predictions, and evaluate them rigorously.

### Key Topics Covered

- Mathematical foundations: why linear regression fails for classification, and how the sigmoid function fixes it
- Binary and multiclass classification using scikit-learn pipelines
- Coefficient interpretation and odds ratios
- One-vs-Rest (OvR) vs Multinomial (Softmax) strategies
- Probability calibration using Platt scaling and isotonic regression
- Advanced evaluation metrics: log-loss, Brier score, ROC-AUC
- Regularization tuning and feature scaling best practices

---

## 🎯 Learning Objectives

By the end of this notebook, you will:

- 🎯 Understand why linear regression fails for classification and how the sigmoid function fixes it
- 🎯 Interpret logistic regression coefficients as log-odds and convert them to odds ratios
- 🎯 Understand cross-entropy loss (log loss) and why it's the natural choice for probabilistic classification
- 🎯 Implement binary logistic regression using scikit-learn pipelines with proper scaling
- 🎯 Distinguish between One-vs-Rest (OvR) and Multinomial (Softmax) strategies for multiclass problems
- 🎯 Calibrate predicted probabilities using Platt scaling or isotonic regression
- 🎯 Evaluate classifiers using log-loss, Brier score, and ROC-AUC beyond simple accuracy
- 🎯 Apply regularization (L1/L2) and tune the regularization strength $C$ via cross-validation
- 🎯 Recognize when probability calibration matters for downstream decision-making

---

## 📊 Notebook Sections

### 1. Binary Classification Baseline – Iris (setosa vs rest)

**What you'll learn:**
- Set up a binary classification problem using the classic Iris dataset
- Create a stratified train/test split to maintain class balance
- Build a logistic regression pipeline with proper feature scaling
- Understand the relationship between decision functions and predicted probabilities

**Key Concepts:**
- Sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Decision boundary at $p = 0.5$ (where $z = 0$)
- The importance of feature scaling for gradient-based optimization

---

### 2. Visualizing the Sigmoid & Decision Boundary

**What you'll learn:**
- Restrict to 2 features to visualize the decision boundary in 2D space
- Create contour plots showing predicted probabilities
- Understand the linear decision boundary in feature space

**Visualization:** Probability contours with the decision boundary at $p = 0.5$ overlay on scatter plot of training and test data

---

### 3. Interpreting Coefficients & Odds Ratios

**What you'll learn:**
- Extract and interpret logistic regression coefficients
- Convert coefficients to odds ratios using exponentiation
- Understand how each feature affects the log-odds and odds of the positive class

**Key Insight:** 
- A coefficient $\beta$ means that for each one-unit increase in that feature, the log-odds increase by $\beta$
- The odds ratio $e^\beta$ is the multiplicative effect on the odds
- Odds ratios > 1 increase probability; < 1 decrease it

**Visualization:** Bar charts comparing raw coefficients vs odds ratios

---

### 4. Multiclass Strategies – One-vs-Rest (OvR) vs Multinomial (Softmax)

**What you'll learn:**
- Understand the two main approaches for extending binary logistic regression to multiclass:
  - **One-vs-Rest (OvR):** Fit one binary classifier per class vs. all others
  - **Multinomial (Softmax):** Directly model the probability distribution using softmax function
  
- Compare performance using accuracy and log-loss
- Understand when probabilities sum to 1 and why it matters

**Key Difference:**
- OvR: Multiple independent binary classifiers
- Multinomial: Single joint optimization, ensures proper probability distribution

**Loss Functions:**
- Cross-entropy (log loss): $L = -\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(p_{ik})$

**Visualization:** Confusion matrices comparing OvR and Multinomial performance

---

### 5. Probability Calibration – When & Why It Matters

**What you'll learn:**
- Recognize miscalibration: predicted probability ≠ actual frequency of occurrence
- Apply two calibration methods:
  - **Sigmoid (Platt scaling):** Fits a logistic regression on classifier scores (assumes sigmoid distortion)
  - **Isotonic regression:** Non-parametric approach, more flexible but needs more data

- Evaluate calibration using Brier score
- Interpret reliability diagrams (calibration curves)

**When It Matters:** Probability calibration is critical when:
- Probabilities feed into downstream decisions (medical diagnosis, risk scoring)
- Class imbalance is present
- Model decisions have real consequences

**Visualization:** Reliability diagrams showing uncalibrated vs calibrated models

---

### 6. Evaluation Beyond Accuracy – Log Loss & ROC-AUC

**What you'll learn:**
- Move beyond accuracy to probability-based metrics
- Understand three key metrics:

| Metric | Formula | When to Use |
|--------|---------|------------|
| Log Loss | $-\frac{1}{N}\sum_{i=1}^N \sum_k y_{ik} \log(p_{ik})$ | Primary metric for probability evaluation |
| Brier Score | $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$ | Measures calibration + refinement |
| ROC-AUC | Area under ROC curve | Threshold-independent performance |

- Compute ROC curves for multiclass problems using One-vs-Rest approach
- Understand macro vs micro averaging for multiclass metrics

**Key Insight:** Accuracy can be misleading on imbalanced data. Always use metrics that evaluate probability quality.

**Visualization:** ROC curves for each class with AUC scores

---

### 7. Regularization in Logistic Regression

**What you'll learn:**
- Understand the $C$ parameter: **inverse** of regularization strength
  - Small $C$ = strong regularization = simpler model
  - Large $C$ = weak regularization = more complex model
  
- Use GridSearchCV to find optimal $C$ via cross-validation
- Observe how regularization shrinks coefficients toward zero

**Regularization Methods:**
- L2 (default): $\text{Loss} + \frac{1}{2C} \sum \beta_j^2$
- L1 (sparsity): $\text{Loss} + \frac{1}{C} \sum |\beta_j|$ (feature selection)

**Visualization:** Certificate curves showing CV performance and coefficient magnitudes across different $C$ values

---

## 🚫 Common Pitfalls & ✅ Pro Tips

### Don'ts

🚫 **Don't use accuracy on imbalanced data.** A classifier predicting the majority class can have 99% accuracy but be useless. Use F1, ROC-AUC, or log-loss instead.

🚫 **Don't interpret raw coefficients without exponentiating.** Coefficients are in log-odds space; odds ratios ($e^\beta$) are interpretable.

🚫 **Don't forget to scale features.** Logistic regression assumes features are on similar scales. Always use `StandardScaler` in pipelines.

🚫 **Don't default to OvR for multiclass.** Multinomial often performs better and gives true probability distributions.

🚫 **Don't ignore probability calibration.** If using probabilities for thresholding or downstream decisions, check calibration with reliability diagrams.

🚫 **Don't use `solver='liblinear'` for large datasets.** It's limited to OvR. Use `'lbfgs'`, `'sag'`, or `'saga'` for multiclass and large data.

🚫 **Don't ignore convergence warnings.** Increase `max_iter` if you see warnings, or check if features need scaling.

### Do's

✅ **Do check for multicollinearity.** Highly correlated features make coefficients unstable. Use VIF or PCA.

✅ **Do use stratified sampling.** Always stratify train/test splits to maintain class distributions.

✅ **Do use pipelines.** Prevents data leakage and ensures proper scaling on test data.

✅ **Do cross-validate hyperparameters.** Use GridSearchCV or RandomizedSearchCV for proper tuning.

✅ **Do check the confusion matrix.** Understand which classes are confused with each other.

---

## 🔧 Hands-On Exercises

### 🟢 Easy: Binary Classification on Breast Cancer Dataset

Load `sklearn.datasets.load_breast_cancer()`, select two features (e.g., mean radius and mean texture), and fit a binary logistic regression. Plot the decision boundary and report the odds ratios. Which feature has a stronger effect on malignancy prediction?

### 🟡 Medium: Compare OvR vs Multinomial on Digits Dataset

Load `sklearn.datasets.load_digits()` (10 classes). Compare OvR and Multinomial strategies using both accuracy and log-loss. Which performs better? Visualize the confusion matrix for the better-performing model.

### 🟡 Medium: L1 Regularization and Sparsity

Using the breast cancer dataset, fit logistic regression with `penalty='l1'` and `solver='liblinear'` (or `'saga'`). Try different $C$ values and plot how many coefficients become exactly zero (sparsity) as you decrease $C$. L1 regularization performs feature selection!

### 🔴 Hard: Calibration Function

Write a function `fit_calibrated_logreg(X_train, y_train, X_cal, y_cal, X_test, method='sigmoid')` that:
1. Fits a logistic regression on training data
2. Calibrates it on calibration data using the specified method
3. Returns calibrated probabilities and Brier score on test data

Test it on a synthetic dataset from `make_classification` with class imbalance.

### 🔵 Bonus: Wine Dataset Evaluation

Load `sklearn.datasets.load_wine()` (3 classes). Compare OvR vs Multinomial using:
- Accuracy
- Log-loss
- Macro-averaged ROC-AUC
- Calibration curves for each class

Which strategy is more appropriate for this dataset and why?

---

## 🧠 Key Formulas

### Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Log-Odds (Logit)
$$\text{logit}(p) = \log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$$

### Cross-Entropy Loss (Log Loss)
$$L = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(p_{ik})$$

### Odds Ratio
$$\text{OR} = e^\beta$$

### Softmax Function (Multinomial)
$$P(y=k|X) = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

### Brier Score
$$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$$

---

## 📚 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## 🚀 Quick Start

1. **Run cells sequentially** from top to bottom to follow the flow
2. **Modify datasets** in each section to apply concepts to your own data
3. **Adjust hyperparameters** (C, max_iter, solver) to see their effects
4. **Complete exercises** to reinforce your understanding

---

## 📖 References

- Scikit-learn Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Probability Calibration: https://scikit-learn.org/stable/modules/calibration.html
- ROC-AUC Documentation: https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics

---

## 📝 Summary

You've learned that logistic regression is much more than a simple classification algorithm:

✅ **Mathematical Foundations:** You understand the sigmoid function and log-odds
✅ **Interpretability:** You can explain the real-world impact of coefficients
✅ **Multiclass Strategies:** You know OvR vs Multinomial and their trade-offs
✅ **Probability Calibration:** You recognize and fix miscalibrated probabilities
✅ **Proper Evaluation:** You use probability-based metrics beyond accuracy
✅ **Regularization:** You tune model complexity via cross-validation

**Next Steps:** Apply these concepts to real-world classification problems, experiment with feature engineering, and compare logistic regression with more complex classifiers like random forests and gradient boosting.

---

**Happy Learning! 🎓**
