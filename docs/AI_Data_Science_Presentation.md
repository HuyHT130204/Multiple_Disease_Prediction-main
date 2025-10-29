## Multiple Disease Prediction – AI Data Science Presentation

### 0. Overview
- **Goal**: Build predictive models for three health datasets (Diabetes, Heart, Kidney), from EDA to baseline and ensemble models.
- **Notebooks**: `notebooks/Advance Project Diabetes Prediction Using ML.ipynb`, `notebooks/Advance Project Heart Disease Prediction Using ML.ipynb`, `notebooks/Advance Project Kidney Disease Prediction Using ML.ipynb`.
- **Environment**: `%pip install pandas numpy scikit-learn matplotlib seaborn plotly imbalanced-learn xgboost missingno statsmodels`.

### 1. Datasets
- **Diabetes** (`dataset/diabetes.csv`)
  - **Size**: 768 rows × 9 columns
  - **Target**: `Outcome` (0/1)
  - **Features**: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`
  - **Notes**: 0-values in clinical fields treated as missing (imputed)

- **Heart** (`dataset/heart.csv`)
  - **Size**: 303 rows × 14 columns
  - **Target**: `target` (0/1)
  - **Features**: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
  - **Notes**: Clean, compact dataset; some log-transforms used to reduce variance

- **Kidney** (`dataset/kidney_disease.csv`)
  - **Size**: ~400 rows × 25 columns
  - **Target**: `class` (mapped: ckd→0, not ckd→1)
  - **Features**: numeric + categorical (e.g., `red_blood_cells, pus_cell, bacteria`)
  - **Notes**: Data cleaning (fix stray text, type casting), missing handling (random sampling/mode), label encoding

### 2. EDA Highlights
- **Common checks**: `df.info()`, `df.describe()`, `df.shape`, missingness profiles
- **Target distribution**: Pie/Countplot to inspect class balance
- **Distributions**:
  - Hist/Distplot for numeric features; Boxplots to spot outliers (e.g., Insulin, BMI)
  - KDE/Violin split by class (Kidney) to compare distributions
- **Correlation (numeric only)**:
  - Use `df.corr(numeric_only=True)` and heatmaps
  - Diabetes: `Glucose` correlates with `Outcome`; `BMI`, `Age` moderate
  - Heart: `oldpeak`, `cp`, `thalach` relate to `target`
  - Kidney: Bio-chemistry indicators (e.g., urea, creatinine) relate to `class`

### 3. Data Preparation
- **Missing values**: 0→NaN (Diabetes), median-by-class imputation; random sampling + mode (Kidney)
- **Encoding**: Label Encoding for categorical (Kidney); One-hot where appropriate
- **Scaling**: `StandardScaler`/`RobustScaler` for numeric features (esp. Diabetes/Heart)
- **Outliers**: IQR and Local Outlier Factor (LOF) exploration (Diabetes)

### 4. Models Trained
- **Baselines**: Logistic Regression (LR), K-Nearest Neighbors (KNN), Support Vector Machine (SVM)
- **Trees/Ensembles**: Decision Tree (DT), Random Forest (RF), Gradient Boosting (GBDT), XGBoost (XGB)
- **Tuning**: GridSearch on key hyperparameters (e.g., SVM C/gamma; DT depth/splits; GBDT learning_rate/n_estimators)

### 5. Evaluation Protocol
- **Split**: Train/Test (test size ≈ 20–30%, `random_state=0`)
- **Metrics**: Accuracy, Confusion Matrix, Precision/Recall/F1, ROC-AUC
- **Visualization**: ROC curves; bar charts comparing Accuracy and AUC across models

### 6. Results Snapshot
- These ranges reflect observed outcomes from the notebooks (exact values depend on split/tuning):

- **Diabetes**
  - LR/KNN: ~0.86–0.89 Accuracy
  - SVM/GBDT/XGB: up to ~0.90–0.91 Accuracy; better ROC-AUC

- **Heart**
  - LR ~0.79; KNN ~0.76; SVM variable (~0.52–0.80 depending on params)
  - DT tuned ~0.78; RF ~0.82; XGB ~0.80; GBDT ~0.79

- **Kidney**
  - LR ~0.94; DT tuned ~0.96; GBDT ~0.97; XGB ~0.96; RF up to ~0.99
  - Note: Small dataset + strong preprocessing → verify generalization (ROC-AUC, CV)

### 7. Key Insights
- **Data quality drives performance**: Cleaning and imputation are crucial (especially Kidney, Diabetes)
- **Feature importance**:
  - Diabetes: `Glucose`, `BMI`, `Age` highly informative
  - Heart: `oldpeak`, `cp`, `thalach`; engineered transforms help
  - Kidney: Biochemical features dominate; categorical normalization is essential
- **Modeling**:
  - Ensembles (RF/GBDT/XGB) generally outperform baselines, but monitor overfitting
  - SVM needs careful tuning and scaling

### 8. Recommendations
- **Validation**: Use stratified K-Fold and report mean±std Accuracy and ROC-AUC
- **Interpretability**: Apply SHAP/feature importance to explain predictions
- **Calibration**: Calibrated probabilities (Platt/Isotonic) for thresholding
- **Deployment**: Persist best model(s) via pickle; add input validation and drift monitoring

### 9. Demo Script (Speaker Notes)
- Open each notebook; run install cell; confirm dataset load paths via `Path.cwd().parent / "dataset" / ...`
- Show target distribution and a few key hist/boxplots
- Display correlation heatmap and explain a couple of strong relations
- Walk through preprocessing steps (missing → impute; encode; scale)
- Train LR as baseline → show metrics; then RF/GBDT/XGB → compare
- Show ROC curves; conclude with best trade-off model per dataset

### 10. Appendix
- **Repro commands** (inside notebook cells):
  - Install: `%pip install --quiet pandas numpy scikit-learn matplotlib seaborn plotly imbalanced-learn xgboost missingno statsmodels`
  - Read data from notebooks folder:
    - `from pathlib import Path; root = Path.cwd().parent`
    - `pd.read_csv(root / "dataset" / "diabetes.csv")`
    - `pd.read_csv(root / "dataset" / "heart.csv")`
    - `pd.read_csv(root / "dataset" / "kidney_disease.csv")`
- **Seaborn compatibility**: Use `sns.countplot(x='col', data=df)`
- **NumPy 2.0**: Use `np.nan` (not `np.NaN`)
- **Correlation**: `df.corr(numeric_only=True)`


