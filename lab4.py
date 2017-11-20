from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def create_data(mean, cov, N):
    return np.random.multivariate_normal(mean, cov, N)

colors = ['r', 'b', 'g']
markers = ['^', 'o', '>']

def show_data(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, label in zip(X, y):
        c = colors[int(label)]
        m = markers[int(label)]
        ax.scatter(i[0], i[1], i[2], c=c, marker=m)
    plt.show()



def classify(X):
    X_train, X_test, y_train, y_test = train_test_split(
        X[:, :3], X[:, -1], test_size=0.33, random_state=42)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_predict = neigh.predict(X_test)

    show_data(X_test, y_test)
    show_data(X_test, y_predict)
    print neigh.score(X_test, y_test)




first_class = create_data([2, -2, 0], [[2, -1, 0.5], [-1, 4, -1] , [0.1, -1, 2]], 1000)
second_class = create_data([4, 2, -4], [[2, -0.1, -1], [-0.1, 2, -1], [1, -1, 4]], 100)


first_class = np.hstack((first_class, np.zeros((first_class.shape[0], 1))))
second_class = np.hstack((second_class, np.ones((second_class.shape[0], 1))))

X = np.concatenate((first_class, second_class))

classify(X)

wine = genfromtxt('wine.txt', delimiter=',')
x = wine[:, 2]
y = wine[:, 11]
z = wine[:, 13]
labels = wine[:, 0] - 1
data = np.array(zip(x, y, z, labels))

classify(data)