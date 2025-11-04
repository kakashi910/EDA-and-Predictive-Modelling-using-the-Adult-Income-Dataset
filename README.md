# Adult Income Prediction: EDA and Predictive Modeling

## Overview

This repository contains a Jupyter Notebook implementing **Exploratory Data Analysis (EDA)** and **Predictive Modeling** on the **Adult Income Dataset** from the UCI Machine Learning Repository. The primary goal is to predict whether an individual's annual income exceeds $50K based on demographic and employment census data.

The dataset is a classic binary classification problem, often used to explore socio-economic factors influencing income levels. The notebook covers data loading, preprocessing (handling missing values, encoding categoricals), EDA (visualizations, correlations), model training (Logistic Regression, Random Forest, XGBoost), evaluation (accuracy, precision, recall, F1-score), and model comparison.

Key highlights:
- **Dataset Size**: ~32,561 training instances (from `adult.data`), ~16,281 test instances (from `adult.test`).
- **Target Variable**: Binary (`<=50K` or `>50K`), imbalanced (~76% `<=50K`).
- **Best Performing Model**: Typically XGBoost or Random Forest (F1-score ~0.70-0.80 after tuning).
- **Insights**: Education level, age, capital gains, and occupation are strong predictors of high income.

A compressed version of the notebook (`EDA and predictive Modelling_compressed.ipynb`) is included, reducing file size by ~99% by stripping outputs for easier sharing.

## Dataset Description

The dataset is derived from the 1994 US Census Bureau database and includes 14 features plus the target. Missing values are marked as `?` in categorical columns.

### Features
| Feature          | Type       | Description |
|------------------|------------|-------------|
| `age`            | Continuous | Age in years (16+) |
| `workclass`      | Categorical | Employment class (e.g., Private, Self-emp-not-inc) |
| `fnlwgt`         | Continuous | Final weight (sampling adjustment) |
| `education`      | Categorical | Highest education level (e.g., Bachelors, HS-grad) |
| `education-num`  | Continuous | Numeric education level |
| `marital-status` | Categorical | Marital status (e.g., Married-civ-spouse) |
| `occupation`     | Categorical | Job type (e.g., Exec-managerial, Craft-repair) |
| `relationship`   | Categorical | Family relationship (e.g., Husband, Not-in-family) |
| `race`           | Categorical | Race (e.g., White, Black) |
| `sex`            | Categorical | Gender (Male, Female) |
| `capital-gain`   | Continuous | Capital gains |
| `capital-loss`   | Continuous | Capital losses |
| `hours-per-week` | Continuous | Weekly work hours |
| `native-country` | Categorical | Country of origin (e.g., United-States, Mexico) |
| `income`         | Binary     | Target: `<=50K` or `>50K` |

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult)
- **Files**:
  - `adult.data`: Raw training data (comma-separated, no headers).
  - `adult.test`: Raw test data (comma-separated, starts with junk line, labels end with period).
  - `adult.names`: Metadata with attribute descriptions and class distribution.
  - `old.adult.names`: Legacy metadata file.
  - `Index`: Directory listing of original files.

### Class Distribution
- `<=50K`: 76.07% (~75.22% without unknowns)
- `>50K`: 23.93% (~24.78% without unknowns)

~7% of instances have missing values (handled by dropping or imputing in the notebook).

## Notebook Structure

1. **Introduction**: Dataset overview and goals.
2. **Import Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost.
3. **Data Loading & Preprocessing**:
   - Load `adult.data` into Pandas DataFrame.
   - Handle `?` as NaN, drop rows with missing values.
   - Encode categoricals (OneHotEncoder), scale numerics (StandardScaler).
   - 80/20 train-test split on training data (as per notebook instructions).
4. **EDA**:
   - Summary statistics, distributions (histograms, boxplots).
   - Correlations, cross-tabulations (e.g., income vs. education).
   - Visualizations: Count plots for categoricals, scatter plots for continuous.
5. **Modeling**:
   - Models: Logistic Regression, Random Forest, XGBoost.
   - Evaluation: Cross-validation, metrics table.
6. **Model Selection**: Best model based on F1-score (due to imbalance).
7. **Compression Script**: Strips notebook outputs for size reduction.

Example output from model comparison (results may vary slightly):
| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Logistic Regression | 0.85    | 0.72     | 0.65  | 0.68    |
| Random Forest  | 0.87    | 0.78     | 0.70  | 0.74    |
| XGBoost        | 0.88    | 0.80     | 0.72  | 0.76    |

## Setup & Running the Notebook

### Prerequisites
- Python 3.8+ (tested with 3.13.5 in conda base env).
- Install dependencies:
  ```
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter nbformat
  ```

### Usage
1. Clone or download the repository.
2. Place the dataset files (`adult.data`, `adult.test`, `adult.names`) in the root directory.
3. Launch Jupyter:
   ```
   jupyter notebook
   ```
4. Open `EDA and predictive Modelling.ipynb` and run all cells.
   - Note: The notebook uses an 80/20 split on `adult.data`; `adult.test` is available for external validation but not used here.
5. For the compressed version: Run the final cell to generate `EDA and predictive Modelling_compressed.ipynb`.

## Results & Insights
- **Key Predictors**: Higher education (`education-num > 13`), age (35-50), capital gains (>0), and managerial occupations strongly correlate with `>50K`.
- **Challenges**: Imbalanced classes favor precision/recall over accuracy; missing values in `workclass`/`occupation` (~5-7%).
- **Performance**: Models achieve ~85-88% accuracy, outperforming baselines like Naive Bayes (~83%). XGBoost excels due to handling interactions and non-linearity.
- **Limitations**: Dataset is from 1994 (outdated demographics); potential biases in race/sex features.
- **Future Work**: Hyperparameter tuning (GridSearchCV), ensemble methods, or fairness analysis (e.g., disparate impact).
