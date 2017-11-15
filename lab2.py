import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from numpy import genfromtxt


N = 100
a = -0.5
b = 2
sigma = 0.01


def regression(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    print 'Coefficient of determination %f' % r_value ** 2
    print 'a %f' % slope
    print 'b %f' % intercept
    plt.plot(x, y, 'o', label='original data')
    plt.plot(x, intercept + slope * x, 'r', label='fitted line')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x = np.random.uniform(0, 1, N)
    y = a * x + b + np.random.normal(0, sigma**0.5, N)
    regression(x, y)
    wine = genfromtxt('wine.txt', delimiter=',')
    x = wine[:, 2]
    y = wine[:, 11]
    regression(x, y)