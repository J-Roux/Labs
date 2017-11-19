
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from numpy import genfromtxt

def create_data(mean, cov, N):
    return np.random.multivariate_normal(mean, cov, N)


def sum_distance(center, cls):
    return sum([(center[0] - i[0]) ** 2 + (center[1] - i[1]) ** 2 for i in cls
                ])


def clusterize(X, cluster_num=2):

    for i, color in zip(X, ['>m', '^b', 'og'] ):
        plt.plot(i[:, 0], i[:, 1], color)
    plt.show()

    X =  np.concatenate( X)
    kmeans = KMeans( n_clusters=cluster_num, random_state=0).fit(X)
    data = np.append(X, [[i] for i in kmeans.labels_], 1)

    Y = [ data[data[:, 2] == k, :2] for k in np.unique(data[:, 2])]
    for i, color in zip(Y, ['>m', '^b', 'og'] ):
        plt.plot(i[:, 0], i[:, 1], color)
    for i, color in zip(kmeans.cluster_centers_, ['xr', 'xr', 'xr']):
        plt.plot(i[0], i[ 1], color)
    plt.show()

    print [ sum_distance(center, i) for i, center in zip(Y, kmeans.cluster_centers_)]




if __name__ == '__main__':
    first_class = create_data([1, -1],[[1, -0.1], [-0.1, 1]], 1000)
    second_class = create_data([4, 2], [[2, -0.1],[-0.1, 2]], 100)


    X = np.array([first_class, second_class])

    clusterize(X)

    wine = genfromtxt('wine.txt', delimiter=',')
    x = wine[:, 2]
    y = wine[:, 11]
    labels = wine[:,0]
    data = np.array(zip(x, y, labels))

    Y = [data[data[:, 2] == k, :2] for k in np.unique(data[:, 2])]
    clusterize(Y, 3)