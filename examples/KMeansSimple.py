from random import sample
import numpy as np
from skcoreset import CoresetKMeans
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from time import time
from tqdm import tqdm

# Get data
X, y = fetch_kddcup99(return_X_y=True)
X = X[:, 4:].astype(np.float64)
X = StandardScaler().fit_transform(X)
k = 10
runs = 10

coreset_factor = 0.0005
coreset_size = int(coreset_factor * X.shape[0])
print(f"{X.shape = }")
print(f"{coreset_size = }")


t1 = time()
km = KMeans(n_clusters=k)
km.fit(X)
cost_op = -km.score(X)
seconds = time() - t1
print(f"{cost_op = :.4f} | {seconds = :.4f}")


costs = []
times = []
for _ in tqdm(range(runs)):
    t1 = time()
    cor = CoresetKMeans(
        algorithm="practical",
        k=k,
    )
    idxs, weights = cor.build(X, coreset_size=coreset_size)
    km = KMeans(n_clusters=k)
    km.fit(X[idxs], sample_weight=weights)

    times.append(time() - t1)
    costs.append(-km.score(X) / cost_op)

print(f"coreset cost / cost_op: {np.mean(costs) :.4f} | Time: {np.mean(times) :.4f}")


costs = []
times = []
for _ in tqdm(range(runs)):
    t1 = time()
    idxs = np.random.choice(len(X), coreset_size)
    km = KMeans(n_clusters=k)
    km.fit(X[idxs])
    times.append(time() - t1)
    costs.append(-km.score(X) / cost_op)
print(f"random cost / cost_op: {np.mean(costs) :.4f}. Time: {np.mean(times) :.4f} ")
print()
