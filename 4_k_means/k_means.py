################### UNSUPERVISED LEARNING ###################
'''
K-Means : "k" corresponds to the number of CENTROIDS

=> REPEAT UNTIL WE GET NO CHANGES BETWEEN OUR DATA POINTS
1. Place "k" centroids in random positions
2. If k=2, draw a line between the 2 centroids
3. Try do divide all of the points to be equally divided between the centroids
4. Find the points closest to either of the centroids
5. Place the centroids in the middle of each group of points (P1*x1+P2*x2+...+Pn*xn)/n

ALGORITHM ANALYSIS
O(p*c*i*f) time
Where 'p' is the number of points (the most influential), 'c' is the amount of centroids, 
'i' is the number of iterations (some hundreds), and 'f' is the amount of features
'''
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)  # Scale all of our features down (values between 0 and 1) to save time in computations
y = digits.target

# k = len(np.unique(y))  # amount of centroids
k = 10
samples, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean'),
            )
    )

classifier = KMeans(n_clusters=k, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300,
                    tol=0.0001)
bench_k_means(estimator=classifier, name="1", data=data)

