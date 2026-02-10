# Kagglethon 1st Runner-up Solution: Outage Risk Prediction

## Competition Overview
**Task:** Binary classification to predict power outage risk  
**Metric:** ROC-AUC  
**Result:** **2nd Place**

---

## Solution Summary

Our winning approach combines **feature selection**, **monotonic constraints**, and **seed averaging** to achieve robust predictions with minimal overfitting.

## Technical Approach

### 1. Feature Selection

Cross-validation experiments showed that:
- Most predictive signal was concentrated in **Feature_3**
- **Feature_2** provided a small but consistent complementary gain
- Additional features consistently **reduced CV ROC-AUC**, indicating noise

**Final feature set:**
```python
["Feature_2", "Feature_3"]
```
**Impact:** Using only two features outperformed models with 10+ features by ~3–5% CV AUC.

**Missing Value Handling:**
```python
imputer = SimpleImputer(strategy="median")
```
- Median imputation chosen over mean for robustness to outliers
- Applied consistently to train and test sets

---

### 2. Monotonic Constraints

Domain analysis indicated a clear relationship: **Higher values of Feature_3 should not reduce the probability of a positive outcome.**

We enforced this directly using monotonic constraints:

```python
monotone_constraints = [0, 1]  # Feature_2: None, Feature_3: Increasing
```

**Why this mattered:**
- Prevented spurious ranking inversions
- Reduced effective model complexity
- Improved ranking stability and generalization

---

### 3. Model Choice: CatBoost

CatBoost was chosen due to:
- Native monotonic constraint support
- Strong regularization and stability
- Superior ranking behavior on small–medium tabular datasets

It generalized better than alternative gradient boosting models in this setting.

---

### 4. Validation & Variance Reduction

**Strategy:**
- Stratified 5-fold cross-validation to preserve class balance
- Seed averaging (5 seeds) to reduce stochastic variance
- **Total models trained:** 25 (5 folds × 5 seeds)

**Result:** This produced smoother predictions and more stable rankings on unseen data.

---

### 5. Overfitting Prevention

**Multiple safeguards:**
1. **Early stopping:** `early_stopping_rounds=100`
2. **L2 regularization:** `l2_leaf_reg=3`
3. **Minimum leaf size:** `min_data_in_leaf=20`
4. **Bagging:** `bagging_temperature=0.5`
5. **Conservative learning rate:** `0.035`
6. **Monotonic constraints:** Prevents overfitting to noise

## Results

```
FINAL CV AUC (Seed-Averaged): 0.704625
```

## Reproducibility

### Running the Code

**Step 1: Install dependencies**
```bash
!pip install -r requirements.txt
```

**Step 2: Mount Google Drive (Colab)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 3: Update file paths**
```python
train = pd.read_csv("/content/drive/MyDrive/dataset/train.csv")
test = pd.read_csv("/content/drive/MyDrive/dataset/test.csv")
```

**Step 4: Run all code blocks sequentially**

## File Structure

```
kagglethon-solution/
│
├── README.md                              # This file
├── kagglethon_solution_code.ipynb         # Full implementation
├── submission_optimized_monotonic.csv     # Final predictions
│
└── dataset/
    ├── train.csv                          # Training data
    └── test.csv                           # Test data
```

---
