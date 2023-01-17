import numpy as np
from skcoreset import build_and_sample_lg
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")
# Load data
X, y = fetch_covtype(return_X_y=True, shuffle=False)
dataset_name = "covtype"


# Coreset parameters
coreset_perc = 0.01


# Experiment parameters
seed = 42


# Preprocess data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

coreset_size = int(X_train.shape[0] * coreset_perc)

# build and sample on the training data
idxs, weights = build_and_sample_lg(X_train, y_train, coreset_size=coreset_size)
# Train model
model = LogisticRegression(random_state=42)
X_, y_ = X_train[idxs], y_train[idxs]
model.fit(X_, y_, sample_weight=weights)

# Evaluate
c = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovo")
print("Single train auc: ", c)


# Grid search
# Grid search parameters
grid_params = {"C": np.linspace(0.01, 1, 5)}

# Create model and grid search
model = LogisticRegression(random_state=seed)
gs = GridSearchCV(model, grid_params, scoring="roc_auc_ovo")

# Fit grid search
X_, y_ = X_train[idxs], y_train[idxs]
gs.fit(X_, y_, sample_weight=weights)

c = roc_auc_score(y_test, gs.predict_proba(X_test), multi_class="ovo")

print("auc grid search: ", c)
