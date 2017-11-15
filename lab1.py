
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from numpy import genfromtxt


N = 100
mean = [0, -1]
cov = [[1, -0.5], [-0.5, 2]]


def perform(x, y):
    R, P = scipy.stats.pearsonr(x, y)
    print 'Pearson Correlation %f' % R
    print 'p-value %e' % P
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    x, y = np.random.multivariate_normal(mean, cov, N).T
    print 'Real correlation %f' % (-0.5 / 2.0 ** 0.5)
    perform(x, y)
    wine = genfromtxt('wine.txt', delimiter=',')
    x = wine[:,2]
    y = wine[:,11]
    perform(x, y)