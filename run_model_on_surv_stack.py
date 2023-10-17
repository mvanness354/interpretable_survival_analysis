import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, brier_score, integrated_brier_score
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
import pickle

import sys
from surv_stack import SurvivalStacker

# Replace this with data (removed as data is private)
data = pd.read_csv("/path/to/data.csv")

censor_years = 15

data["survival_time"] = data["survival_time"] / 365

valid_data = data[data["survival_time"] > 0].copy(deep=True)

valid_data["gender"] = valid_data["gender"].astype(str)
valid_data["ethnicity"] = valid_data["ethnicity"].astype(str)

X = valid_data[[c for c in valid_data.columns if c not in [
    "survival_time", "censor", "patient", "survival_date", "index_date", "death"
]]]

y = np.array(
    list(zip(~valid_data["censor"], valid_data["survival_time"])),
    dtype=[("Status", "?"), ("Survival_Time", "f8")]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)


# Impute missing values with mean imputation
preprocessor = make_column_transformer(
    (
        make_pipeline(
            SimpleImputer(strategy="mean"),
        ),
        ["age"] + list([c for c in X.columns if X[c].isna().mean() > 0])
    ), 
    remainder="passthrough",
).set_output(transform="pandas")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Run survival stacking
stacker = SurvivalStacker(discrete_time=False, sampling_ratio=0.01)
stacker.fit(y_train["Survival_Time"], ~y_train["Status"])
X_train_stacked, y_train_stacked = stacker.transform(X_train_processed_sub)

# Fit EBM model on survival stacked data
# Replace with other binary classification models as needed
model = ExplainableBoostingClassifier()
model.fit(X_train_stacked, y_train_stacked)


# Get cumulative hazards
train_times = y_train["Survival_Time"]
pred_times = np.unique(train_times[
    np.logical_and(
        train_times >= y_test["Survival_Time"].min(),
        train_times < y_test["Survival_Time"].max()
    )
])

pred_times_sub = np.random.choice(pred_times, 500, replace=False)

cum_hazards_monte_carlo = stacker.predict_all_cum_hazard(
    model, X_test_processed, times=pred_times_sub, monte_carlo=True
)

# Get AUC
aucs, mean_auc = cumulative_dynamic_auc(
    y_train, y_test, cum_hazards_monte_carlo, pred_times_sub
)
print("EBM AUC: ", mean_auc)

# Get Brier Scores
brier = integrated_brier_score(
    y_train, y_test, np.exp(-1 * cum_hazards_monte_carlo), pred_times_sub
)
print("EBM BRIER: ", brier)