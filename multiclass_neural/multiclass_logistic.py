import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from scipy import optimize

data = scipy.io.loadmat('ex3data1.mat')


# plt.imshow(data['X'][381].reshape(-1, 20).T)

class MultiClass:
    def __init__(self, data):
        self.data = data
        r, c = self.data['X'].shape
        self.X_ = self.data['X']
        # adding an intercept term in the data
        self.X = np.column_stack((np.ones((r, 1)), self.X_))
        self.y = data['y']
        self.theta = np.zeros((1, self.X.shape[1]))
        self.lamda = 0.01
        self.m = r
        self.cost_history = []

    def display_data(self):
        fig, axs = plt.subplots(10, 10)
        for i in range(10):
            for j in range(10):
                axs[i, j].imshow(self.X_[np.random.randint(0, 4999)].reshape(20, 20).T)
                axs[i, j].axis('off')

    def costfunction(self, *args):
        if len(args):
            theta = args[0]
            lamda = args[1]
            X = args[2]
            y = args[3]
        else:
            theta = self.theta
            X = self.X
            y = self.y
            lamda = self.lamda
        z = np.dot(X, theta.T)
        h_theta = sigmoid_function(z)
        h_theta = h_theta.T
        J_theta = ((np.dot(np.log(h_theta), y)) + (np.dot(np.log(1 - h_theta), (1 - y)))) / - self.m
        regularized_J = J_theta + (lamda / 2 * self.m) * np.dot(theta, theta.T)
        if np.isnan(J_theta):
            self.cost_history.append(np.inf)
            return np.inf

        self.cost_history.append(J_theta)
        return regularized_J

    def gradient(self, *args):
        if len(args):
            theta = args[0]
            lamda = args[1]
            X = args[2]
            y = args[3]
        else:
            theta = self.theta
            X = self.X
            y = self.y
            lamda = self.lamda
        z = np.dot(X, theta.T)
        h_theta = sigmoid_function(z)
        error = h_theta - y
        gradient = (1 / self.m) * np.dot(error.T, X)
        regularized_G = gradient + (lamda / 2 * self.m) * np.dot(theta, theta.T)
        return regularized_G.flatten()

    def minimize(self, lamda):
        theta_multi = np.zeros((10, self.X.shape[1]))
        for i in range(10):
            y_label = i if i else 10
            res = optimize.minimize(self.costfunction, self.theta, args=(lamda, self.X, (y_label == self.y).flatten()),
                                    method='BFGS', jac=self.gradient, options={'maxiter':50})
            theta_multi[i-1] = res.x
            return res


def sigmoid_function(z):
    return 1.0 / (1.0 + np.exp(-z))


multi = MultiClass(data)
multi.display_data()
h = multi.minimize(0.1)
