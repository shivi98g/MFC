from collections import defaultdict
import numpy as np
from kfed import algo_1
from k1_equals_k import algo_2
from send_rand_k1 import algo_3
from sklearn.metrics import pairwise_distances as sparse_cdist
from sklearn.metrics import silhouette_score
import scipy
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import time
import random

class COST:
    def distance_to_set(self, A, S, sparse=False):
        '''
        S is a list of points. Distance to set is the minimum distance of $x$ to
        points in $S$. In this case, this is computed for each row of $A$.  Note
        that this works with sparse matrices (sparse=True)
        Returns a single array of length len(A) containing corresponding distances.
        '''
        n, d = A.shape
        assert S.ndim == 2
        assert S.shape[1] == d, S.shape[1]
        assert A.shape[1] == d
        assert A.ndim == 2
        # Pair wise distances
        if sparse is False:
            pd = scipy.spatial.distance.cdist(A, S, metric='euclidean')
        else:
            pd = sparse_cdist(A, S)
        assert np.allclose(pd.shape, [A.shape[0], len(S)])
        dx = np.min(pd, axis=1)
        assert len(dx) == A.shape[0]
        assert dx.ndim == 1
        return dx


    def get_clustering(self, A, centers, sparse=False):
        '''
        Returns a list of integers of length len(A). Each integer is an index which
        tells us the cluster A[i] belongs to. A[i] is assigned to the closest
        center.
        '''
        # Pair wise distances
        if sparse is False:
            pd = scipy.spatial.distance.cdist(A, centers, metric='euclidean')
        else:
            pd = sparse_cdist(A, centers)
        assert np.allclose(pd.shape, [A.shape[0], len(centers)])
        indices = np.argmin(pd, axis=1)
        assert len(indices) == A.shape[0]
        return np.array(indices)


    def kmeans_cost(self, A, centers, sparse=False, remean=False):
        '''
        Computes the k means cost of rows of $A$ when assigned to the nearest
        centers in `centers`.
        remean: If remean is set to True, then the kmeans cost is computed with
        respect to the actual means of the clusters and not necessarily the centers
        provided in centers argument (which might not be actual mean of the
        clustering assignment).
        '''
        clustering = self.get_clustering(A, centers, sparse=sparse)
        cost = 0

        centers2 = []
        for clusterid in np.unique(clustering):
            points = A[clustering == clusterid]
            centers2.append(np.mean(points, axis=0))
        centers2 = np.array(centers2)
        dic = defaultdict(lambda: 0)
        arr = []
        # print(list(np.unique(clustering)))

        max_cost = 0
        for clusterid in np.unique(clustering):
            points = A[clustering == clusterid]
            dist = self.distance_to_set(points, centers, sparse=sparse)
            
            dist2 = self.distance_to_set(points, centers2, sparse=sparse)

            ''' CHANGED '''
            cost += np.mean(dist ** 2)
            # cost += sum(dist)
            
            dic[tuple(centers[clusterid])] += np.mean(dist**2)
            arr.append([np.mean(dist**2) - np.mean(dist2**2), points])
        arr.sort(reverse=1)
        return arr

def cleaup_max(local_estimators, weights, k, dev_k, useSKLearn=True, sparse=False):
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(local_estimators, sample_weight = weights)
    ret = (kmeans.cluster_centers_, kmeans)
    return ret

def get_optimal_k(x):
    # create an empty list to store the Within-Cluster-Sum-of-Squares (WCSS) values
    sil = []
    kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(x)
        labels = kmeans.labels_
        sil.append(silhouette_score(x, labels, metric = 'euclidean'))
    return sil.index(max(sil)) + 2



def gauss_2d(mu, sigma):
    x = random.gauss(mu, sigma)
    y = random.gauss(mu, sigma)
    return (x, y)

start = time.time()

data = []
ax = []
ay = []
cnt = 0
total = 5000
num_devices = 100
num_gaussians = 10
homogeneity = 3
K = 10
K1 = 3

for _ in range (num_devices):
    cur = []
    curx = []
    cury = []
    all_gauss = [i for i in range (num_gaussians)]
    random.shuffle(all_gauss)
    gauss = all_gauss[:homogeneity]
    for __ in range (total//num_devices):
        r = random.choice(gauss)
        x,y = gauss_2d(10*r,2)
        cur.append([x,y])
        ax.append(x)
        ay.append(y)
    data.append(cur)

data = np.array(data)


final = []
times = 1
algos = [algo_1, algo_2, algo_3]
for algo_class in algos:
    
    algo = algo_class()
    avg_cost = [0]*11
        
    centers = algo.kfed(data, K1, K)[1]

    calc = COST()

    centers_cost = defaultdict(lambda: [])

    shots = 10
    combined_data = []
    for i in data:
        combined_data.extend(i)
    combined_data = np.array(combined_data)
    cost = algo.kmeans_cost(combined_data, centers)
    if algo_class == algo_1:
        kfed_cost = cost
    avg_cost[0] += cost

    for shot in range (shots):
        new_pts = []
        weights = []
        for device_id in range (len(data)):
            device = data[device_id]
            arr = calc.kmeans_cost(device, centers)
            for cost, points in arr[:1]:
                # if len(points)<len(data[device_id])//(2*K):
                #     continue
                dim = len(points[0])
                filtered_pts = [[] for i in range (dim)]
                for point in points:
                    for axis in range (dim):
                        filtered_pts[axis].append(point[axis])
                new_pt = []
                for axis in range (dim):
                    new_pt.append(np.mean(filtered_pts[axis]))
                new_pts.append(new_pt)
                weights.append(len(points))
                break
        
        max_weight = max(weights)
        new_pts.extend(centers)
        weights.extend([max_weight]*len(centers))
        # weights = [1]*len(weights)
        new_pts = np.array(new_pts)
        weights = np.array(weights)
        centers, _ = cleaup_max(new_pts, weights, K, 1)
        cost = algo.kmeans_cost(combined_data, centers)
        avg_cost[shot+1]+=cost
    for i in range (len(avg_cost)):
        avg_cost[i]/=times
    # print(avg_cost)

    # print("Federated Cost:", algo.kmeans_cost(combined_data, centers))
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(combined_data)
    centers, _ = (kmeans.cluster_centers_, kmeans)
    kmeans_cost = algo.kmeans_cost(combined_data, centers)
    # print("Centralized Cost:", algo.kmeans_cost(combined_data, centers))


    ff = []
    for device in data:
        ff.append(algo.kmeans_cost(device, centers))
    
    mean = sum(ff)/len(ff)
    variance = 0
    for xx in ff:
        variance += (xx-mean)**2
    variance/=len(ff)
    std_dev = variance**0.5

    ff.sort()
    q1 = int((25/100)*len(ff))
    q3 = int((75/100)*len(ff))
    iqr = ff[q3] - ff[q1]
    # print(ff)
    tou = 0

    # print(mean, std_dev, iqr)

    # plt.scatter([i for i in range (len(ff))], ff)
    # plt.show()

    final.append(avg_cost)

# for center in centers:
#     for device in data:


params = {'legend.fontsize': '18',
        
        #   'figure.figsize': (15, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'16',
          'font.weight': '700',
          'figure.titlesize':'xx-large',
         'ytick.labelsize':'16'}

plt.rcParams.update(params)
print("Centralized cost:", kmeans_cost)
fig, ax = plt.subplots(figsize=(8,6))


ax.plot(final[0], label = "$k$-FED$_M$", linewidth = 3)
ax.plot(final[1], label = "$MFC$", linewidth = 3)
ax.plot(final[2], label = "$MFC_H$", linewidth = 3)
ax.plot([kmeans_cost]*len(final[0]), label = "C$k$means", linewidth = 3, linestyle='dashed')
ax.plot([kfed_cost]*len(final[0]), label = "$k$-FED", linewidth = 3, linestyle='dashed')
ax.legend()
ax.set_ylabel("Heterogeneous\n\nClustering Cost", fontweight='bold')
ax.set_xlabel("Number of iterations", fontweight='bold')
plt.grid()

plt.savefig('img.jpg', dpi=300,bbox_inches="tight")
plt.show()




# plt.scatter(ax,ay)
# plt.scatter([i[0] for i in centers], [i[1] for i in centers], color = 'red')
# plt.title("Z = %s, Num Gauss = %s, K = %s, K' = %s, Time = %s"%(num_devices, num_gaussians, K, K1, str(round(end - start,2) ) + 'sec'))
# plt.show()