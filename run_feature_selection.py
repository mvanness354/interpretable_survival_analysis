import numpy as np
import pandas as pd
from ControlBurn.ControlBurnModel import ControlBurnClassifier
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from feature_engine.selection import DropDuplicateFeatures
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pickle
from time import time

# Replace this with data (removed as data is private)
data = pd.read_csv("/path/to/data.csv")

predict_years = 5

data["survival_time"] = data["survival_time"] / 365

valid_data = data[data["survival_time"] > 0].copy(deep=True)

valid_data["gender"] = valid_data["gender"].astype(str)
valid_data["ethnicity"] = valid_data["ethnicity"].astype(str)

X = valid_data[[c for c in valid_data.columns if c not in ["survival_time", "censor", "patient"]]]

X = pd.get_dummies(X)

y = np.array(
    list(zip(~valid_data["censor"], valid_data["survival_time"])),
    dtype=[("Status", "?"), ("Survival_Time", "f8")]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

preprocessor = make_column_transformer(
    (
        make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0, add_indicator=False),
        ),
        ["age"] + list([c for c in X.columns if X[c].isna().mean() > 0])
    ), 
    remainder="passthrough",
).set_output(transform="pandas")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Remove training samples that are censored before censored years mark
binary_train_samples = ~np.logical_and(~y_train["Status"], y_train["Survival_Time"] <= predict_years)
X_train_binary = X_train_processed[binary_train_samples]

y_train_binary = y_train[binary_train_samples]
y_train_binary = np.logical_and(y_train_binary["Status"], y_train_binary["Survival_Time"] <= predict_years).astype(int)

binary_test_samples = ~np.logical_and(~y_test["Status"], y_test["Survival_Time"] <= predict_years)
X_test_binary = X_test_processed[binary_test_samples]

y_test_binary = y_test[binary_test_samples]
y_test_binary = np.logical_and(y_test_binary["Status"], y_test_binary["Survival_Time"] <= predict_years).astype(int)

# Use bisection to find given number of features
lower = 50
upper = 100
alpha = 0.001
n_feats = -1

while n_feats < lower or n_feats > upper:
    print(f"Running alpha={alpha}")
    tic = time()
    cb = ControlBurnClassifier(
        alpha = alpha, 
    )
    cb.fit(X_train_binary, y_train_binary)
    print(f"Fit Time: {np.round((time() - tic) / 60, 4)} mins")
    
    n_feats = len(cb.features_selected_)
    print(f"Number of features: {n_feats}")
    if n_feats > 0:
    
        score = roc_auc_score(y_test_binary, cb.predict_proba(X_test_binary)[:, 1])
        print(f"Score: {score}")

        with open(f"cb_bisect_alpha{alpha}.pkl", 'wb') as f:
            pickle.dump(cb, f)
    
    if n_feats < lower:
        alpha = alpha / 2
    elif n_feats > upper:
        alpha = alpha + alpha / 2
    


print(f"Final Model: alpha={alpha}, n_feats={n_feats}")

    


