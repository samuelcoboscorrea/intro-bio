import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Data generation
np.random.seed(42)

N1 = 500
mu1 = [1, -1]
Sigma1 = [[0.9, 0.4], [0.4, 0.3]]
c1 = np.random.multivariate_normal(mu1, Sigma1, N1)

N2 = 500
mu2 = [-4, 3]
Sigma2 = [[1, 0], [0, 2]]
c2 = np.random.multivariate_normal(mu2, Sigma2, N2)

plt.scatter(c1[:, 0], c1[:, 1], marker='o', color='r')
plt.scatter(c2[:, 0], c2[:, 1], marker='o', color='b')

# K-NN classification
X = np.concatenate((c1, c2), axis=0)
y = np.concatenate((np.ones(N1), np.zeros(N2))).astype(int)  # Convert labels to integers

neigh = NearestNeighbors(n_neighbors=940)
neigh.fit(X)

x2_vals = np.arange(-3, 7.2, 0.2)
x1_vals = np.arange(-7, 4.2, 0.2)

for x1 in x1_vals:
    for x2 in x2_vals:
        x_test = np.array([[x1, x2]])
        distances, indices = neigh.kneighbors(x_test)
        label_counts = np.bincount(y[indices.flatten()])
        decision = np.argmax(label_counts)

        if decision == 1:
            plt.plot(x_test[0, 0], x_test[0, 1], 'r.', markersize=5)
        elif decision == 0:
            plt.plot(x_test[0, 0], x_test[0, 1], 'b.', markersize=5)
        else:
            plt.plot(x_test[0, 0], x_test[0, 1], 'g.', markersize=5)

plt.xlim([-7, 7])
plt.ylim([-3, 8])
plt.show()
