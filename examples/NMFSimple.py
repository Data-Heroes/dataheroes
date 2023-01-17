import numpy as np
from skcoreset import CoresetNMF
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_kddcup99
from sklearn.decomposition import NMF


# Utils
def cost_nmf(X, model):
    t = model.transform(X)
    t = model.inverse_transform(t)
    return np.sqrt(np.sum((X - t) ** 2))


# Data
X, y = fetch_kddcup99(return_X_y=True)
X = X[:, 4:].astype(np.float64)
X = StandardScaler().fit_transform(X)

k = 1
coreset_factor = 0.005
coreset_size = int(coreset_factor * X.shape[0])

print(f"Data shape: {X.shape}")
print(f"Coreset coreset_size: {coreset_size}")

# Algorithms
nmf = NMF(n_components=k)
nmf.fit(X)
cost_op = cost_nmf(X, nmf)
print(f"cost optimal: {cost_op}")

cor = CoresetNMF(k=k, coreset_size=coreset_size)
cor.fit(X)
print(f"Cost coreset / optimal: {cor.cost(X)/cost_op}")

nmf = NMF(n_components=k)
idxs = np.random.choice(len(X), coreset_size)
nmf.fit(X[idxs])
cost = cost_nmf(X, nmf)
print(f"Cost random / optimal: {cost/cost_op}")
