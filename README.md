# 🚚 Amazon Delivery Time Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-%23F7931E?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/Boosting-XGBoost-76B900)
![LightGBM](https://img.shields.io/badge/Boosting-LightGBM-3EA868)
![CatBoost](https://img.shields.io/badge/Boosting-CatBoost-FF9900)
![Optuna](https://img.shields.io/badge/Optimization-Optuna-6E49CB)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)
![Platform](https://img.shields.io/badge/Platform-Colab%20%7C%20Local-lightgrey)

> End‑to‑end regression pipeline to predict e‑commerce delivery time with robust feature engineering, PCA, multiple tree-based models, Optuna tuning, and MLflow tracking.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Notebook API (Functions)](#notebook-api-functions)
- [Prerequisites](#prerequisites)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About the Project
This repository builds a supervised learning pipeline to **predict delivery time (minutes)** for e‑commerce orders. It includes:
- Exploratory analysis and rich visualizations.
- Distance computation via **haversine**.
- Categorical encoding and time-based feature extraction.
- **Standardization + PCA (retain ≈95% variance)**.
- Baseline & advanced regressors (**RandomForest, HistGradientBoosting, LightGBM, XGBoost, CatBoost**).
- **Hyperparameter optimization with Optuna** and experiment tracking via **MLflow**.

## Key Features
- 🧭 **Feature engineering**: distance (haversine), weather×traffic interactions, time parts, and efficiency features.
- 🧼 **Data cleaning**: KNN imputation for ratings; strict dtype conversions for dates & times.
- 🧰 **Model zoo**: RandomForest, Histogram GB, LightGBM, XGBoost, CatBoost.
- 🧰 **Dimensionality reduction**: PCA to 95% variance (17 components from 21 numeric features).
- 📈 **Experiment tracking**: MLflow runs, residual plots, PCA scree plots.
- 🎯 **Hyperparameter tuning**: Optuna with TPE sampler; results exported as CSV and visualized.
- 💾 **Model artifacts**: serialized `.pkl` models (Git LFS recommended for >100 MB).

## Tech Stack
**Core:** Python, Jupyter Notebook  
**Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `mlflow`, `matplotlib`, `seaborn`, `plotly`, `statsmodels`, `haversine`, `matplotlib-venn`, `joblib`, `IPython.display`.

## Installation
> Works on Windows/macOS/Linux. For heavy training, Colab is supported.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna mlflow matplotlib seaborn plotly statsmodels haversine matplotlib-venn joblib ipython

# (Optional) For large model files in GitHub
git lfs install
git lfs track "*.pkl"
```

## Usage
### Run the Notebook
1. Clone the repository and open `Amazon_Delivery_Time (13).ipynb`.
2. Ensure `amazon_delivery.csv` is present in the repo root.
3. In **Colab**, mount Drive and update paths if needed. In **local Jupyter**, replace any `/content/drive/...` paths with local paths.

### Minimal Example (in Python)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load
df = pd.read_csv("amazon_delivery.csv")

# (Example) Build numeric matrix; drop target and IDs
X = df.drop(columns=["Delivery_Time", "Order_ID"], errors="ignore").select_dtypes(include="number")
y = df["Delivery_Time"]

# Scale + PCA (≈95% variance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = StandardScaler().fit_transform(X_train)
X_test  = StandardScaler().fit(X_train).transform(X_test)  # for brevity here; in notebook scaler is reused

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

# Train a quick XGBoost model
model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_train_pca, y_train)
pred = model.predict(X_test_pca)

rmse = (np.mean((y_test - pred)**2))**0.5
mae  = mean_absolute_error(y_test, pred)
r2   = r2_score(y_test, pred)
print(rmse, mae, r2)
```

## Project Structure
```
.
├─ Amazon_Delivery_Time (13).ipynb        # Main end‑to‑end notebook
├─ amazon_delivery.csv                    # Dataset (tabular)
├─ Models/
│  └─ RandomForest_PCA.pkl                # Example saved model (use Git LFS)
├─ optuna_results-20251004T162645Z-1-001.zip  # Optuna exports (trials CSV, plots)
├─ .gitignore
└─ README.md
```

## Dataset
Target: `Delivery_Time` (minutes)

Notable fields used/engineered in the notebook:
- **Dates/Times**: `Order_Date`, `Order_Time`, `Pickup_Time`, derived `Order_Year/Month/Day/Hour/Minute`, `Pickup_Hour/Minute`, `Is_Weekend`, `Pickup_Delay`.
- **Geo**: `Store_Latitude`, `Store_Longitude`, `Drop_Latitude`, `Drop_Longitude`, computed **`Distance`** (km) via haversine; optional `Area_Based_Distance`.
- **Categoricals**: `Weather`, `Traffic`, `Area`, `Category`, `Vehicle`, interaction **`Traffic_Weather`**.
- **Quality/Agent**: `Agent_Rating`, `Agent_Age`.
- **Derived**: frequency/target encodings for `Category`, rolling stats (e.g., `Category_RollingMean`), and **`Efficiency_TimePerKm`**.

> The notebook handles missing values (e.g., KNN imputation for `Agent_Rating`) and applies one‑hot encoding for categorical features (with `drop='first'` to avoid multicollinearity).

## Methodology
1. **Load & Inspect**: read CSV, check dtypes, missingness visualization.
2. **Cleaning & Imputation**: KNN imputation for `Agent_Rating`; coercive parsing for times (`errors='coerce'`); safe date parsing with format strings.
3. **Feature Engineering**:
   - Geographic **Distance** using `haversine((lat1, lon1), (lat2, lon2))` (km).
   - Time features: order/pickup hour, minute, day‑of‑week, weekend flag; pickup delays.
   - Categorical encodings: one‑hot for `Weather`, `Traffic_Weather`; frequency/target encodings for `Category`.
4. **Scaling & PCA**: standardize numeric features, then PCA with `n_components=0.95` → **17 components** retaining **≈95.04%** variance on **21** numeric features.
5. **Modeling**:
   - Baselines & tree models: **RandomForest**, **HistGradientBoosting**, **LightGBM**, **XGBoost**, **CatBoost**.
   - Evaluation metrics: **RMSE**, **MAE**, **R²** on test set; residual analysis.
6. **Optimization**:
   - **Optuna** TPE sampler explores XGBoost hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`).
   - Studies run for 30–100 trials; best params reused to retrain final model.
7. **Experiment Tracking**:
   - **MLflow** logs metrics, parameters, residual plots, and PCA scree plots.
8. **Artifacts**: models serialized with `joblib.dump` to `Models/` (recommend Git LFS).

## Results
### Model Comparison (test set)
| Model                     | Dataset | RMSE  | MAE   | R²    |
|--------------------------|---------|------:|------:|:-----:|
| RandomForest             | raw     | 0.010 | 0.000 | 1.000 |
| RandomForest             | scaled  | 0.009 | 0.000 | 1.000 |
| RandomForest             | PCA     | 7.121 | 5.238 | 0.981 |
| HistGradientBoosting     | PCA     | 5.206 | 3.919 | 0.990 |
| LightGBM                 | PCA     | 5.405 | 3.981 | 0.989 |
| **XGBoost**              | PCA     | **5.033** | **3.800** | **0.991** |
| CatBoost                 | PCA     | 5.646 | 4.141 | 0.988 |

**Optuna (validation)**  
- Best validation RMSE ranged from **≈0.0766** to **≈0.0616** (on the validation split during tuning).  
- The best trial’s parameters were plugged back into the final XGBoost.

**Final Tuned XGBoost (test set)**  
- **RMSE:** `4.8012`  
- **MAE:** `3.5394`  
- **R²:** `0.9915`

> PCA diagnostics: Original shape `(43648, 21)` → PCA `(43648, 17)`, total variance retained **0.9504**.

## Notebook API (Functions)
- `calculate_distance(row) -> float`  
  Computes great‑circle distance (km) between store and drop coordinates using **haversine**.

- `objective(trial) -> float`  
  Optuna objective for **XGBRegressor**. Suggests `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`. Trains on train/val split and returns validation **RMSE**.

- `setup_mlflow_drive(experiment_name, base_dir, nested_dir) -> str`  
  Configures **MLflow** to log runs under a given folder (e.g., Google Drive), sets tracking URI, and creates/returns the path.

## Prerequisites
- Python **3.9+**
- Jupyter Notebook or Google Colab
- (Optional) **Git LFS** for large `.pkl` artifacts

## Contributing
Contributions are welcome!  
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/awesome`)
3. Commit changes (`git commit -m "feat: add X"`)
4. Push to branch (`git push origin feature/awesome`)
5. Open a Pull Request

## License
No license file is present yet. If you intend the work to be open source, consider adding an **MIT License**.

## Contact
Author: **thedynasty**  
GitHub: [@thedynasty23](https://github.com/thedynasty23)

## Acknowledgments
- Libraries: scikit‑learn, XGBoost, LightGBM, CatBoost, Optuna, MLflow, pandas, numpy, plotly, seaborn, statsmodels, haversine.
- Inspiration: common delivery‑time prediction use‑cases in logistics.
