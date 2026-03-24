cell 22
# Partial Dependence Plots with sklearn
# PDP shows average prediction as function of feature value
top_features = ['MedInc', 'AveOccup', 'HouseAge']
feature_indices = [feature_names.index(f) for f in top_features]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (feature, ax) in enumerate(zip(top_features, axes)):
    PartialDependenceDisplay.from_estimator(
        rf_model, X_train, features=[feature],
        kind='average',              # 'average' = PDP, 'individual' = ICE, 'both' = both
        ax=ax,
        line_kw={'linewidth': 2, 'color': 'blue'}
    )
    ax.set_title(f'PDP: {feature}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Partial Dependence Plots: Marginal Feature Effects', y=1.02)
plt.tight_layout()
plt.show()

print("PDP Interpretation:")
print("- Shows how prediction changes as feature changes (averaged over all samples)")
print("- MedInc: Higher income → higher price (monotonic, as expected)")
print("- AveOccup: More occupants → slightly lower price (crowding effect)")
print("- HouseAge: Non-monotonic relationship (very old houses may be worth less)")
-----
cell 23
# ICE (Individual Conditional Expectation) curves
# Shows PDP for individual instances, revealing heterogeneity
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ICE for MedInc
PartialDependenceDisplay.from_estimator(
    rf_model, X_train, features=['MedInc'],
    kind='both',                     # Show both PDP (thick line) and ICE (thin lines)
    ax=axes[0],
    line_kw={'linewidth': 0.5, 'alpha': 0.5},  # ICE lines
    pd_line_kw={'linewidth': 3, 'color': 'red'}  # PDP line
)
axes[0].set_title('ICE + PDP: Median Income')
axes[0].grid(True, alpha=0.3)

# ICE for AveOccup
PartialDependenceDisplay.from_estimator(
    rf_model, X_train, features=['AveOccup'],
    kind='both',
    ax=axes[1],
    line_kw={'linewidth': 0.5, 'alpha': 0.5},
    pd_line_kw={'linewidth': 3, 'color': 'red'}
)
axes[1].set_title('ICE + PDP: Average Occupancy')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nICE Interpretation:")
print("- Each thin line = one instance's prediction curve")
- Thick red line = average (PDP)")
- Diverging lines indicate feature interactions")
- If all lines parallel → no interactions; crossing lines → strong interactions")
-----
cell 33
<details>
<summary><b>Exercise Solutions (Click to Expand)</b></summary>

### Easy Solution Outline (SHAP Summary)
```python
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
# Top 3: MedInc, AveOccup, HouseAge
```

### Medium Solution Outline (LIME on Misclassification)
```python
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train, feature_names=feature_names_cancer, mode='classification')
exp = explainer.explain_instance(X_test_c[misclassified[0]], rf_cancer.predict_proba, num_features=5)
exp.show_in_notebook()
```

### Medium Solution Outline (PDP + ICE)
```python
from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(rf_model, X_train, features=['MedInc'], 
                                        kind='both', ax=ax)
# Look for diverging lines = interactions
```

### Hard Solution Outline (Model Comparison)
```python
# SHAP for both models
shap_rf = shap.TreeExplainer(rf).shap_values(X_test)
shap_nn = shap.DeepExplainer(nn, background).shap_values(X_test_scaled)
# Compare rankings
from scipy.stats import spearmanr
corr, _ = spearmanr(np.abs(shap_rf).mean(0), np.abs(shap_nn).mean(0))
```

</details>
-----
