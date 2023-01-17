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

print(f"{X.shape = }")
print(f"Number of clusters: {k}")

t1 = time()
km = KMeans(n_clusters=k)
km.fit(X)
cost_op = -km.score(X)
print(f"{cost_op = :.4f} | Time = {time()-t1 = :.4f}")

ccosts = []
ctimes = []
rcosts = []
rtimes = []
coreset_sizes = [37, 50, 250, 500, 5000, 50000]
runs = 5
for coreset_size in tqdm(coreset_sizes):
    print(f"{coreset_size = }")
    costs = []
    times = []
    for _ in range(runs):
        t1 = time()
        # cor = CoresetKMeans(algorithm="practical", k=k)
        cor = CoresetKMeans(algorithm="lightweight")
        idxs, weights = cor.build(
            X,
            coreset_size=coreset_size,
        )
        km = KMeans(n_clusters=k)
        km.fit(X[idxs], sample_weight=weights)
        times.append(time() - t1)
        costs.append(-km.score(X) / cost_op)

    ccosts.append(np.mean(costs))
    ctimes.append(np.mean(times))
    print(f"coreset cost / cost_op: {np.mean(costs) :.4f} | Time: {np.mean(times) :.4f}")

    costs = []
    times = []
    for _ in range(runs):
        t1 = time()
        idxs = np.random.choice(len(X), coreset_size)
        km = KMeans(n_clusters=k)
        km.fit(X[idxs])
        times.append(time() - t1)
        costs.append(-km.score(X) / cost_op)

    rcosts.append(np.mean(costs))
    rtimes.append(np.mean(times))
    print(f"random cost / cost_op: {np.mean(costs) :.4f} | Time: {np.mean(times) :.4f} ")
    print()


ccosts = np.round(ccosts, 4)
rcosts = np.round(rcosts, 4)
ctimes = np.round(ctimes, 4)
rtimes = np.round(rtimes, 4)
print("Coreset")
print(f"{ccosts = }\n{ctimes = }")

print("Random")
print(f"{rcosts = }\n{rtimes = }")

print()
