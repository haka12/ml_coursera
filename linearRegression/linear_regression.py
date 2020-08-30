import numpy as np
import matplotlib.pyplot as plt
from linearRegression import load_data


class Linear:
    def __init__(self, data, alpha, itr):
        self.data = data
        r, c = self.data.shape
        "adding a column of ones for x0-->>theta0*x0=theta0"
        self.X = np.row_stack((np.ones((1, r)), data['X'].values))
        self.y = data['y'].values
        self.theta = np.zeros([1, c])
        self.alpha = alpha
        self.m = len(self.y)
        self.itr = itr
        self.cost_history = []

    def plot_data(self):
        plt.scatter(y=self.data["y"], x=self.data['X'], label="Training Data")
        plt.ylabel('Profit in $10,000s')
        plt.xlabel('Population of City in 10,000')
        return plt

    def cost_function(self):
        h_theta = np.dot(self.theta, self.X)
        error = (h_theta - self.y) ** 2
        j_theta = np.sum(error) / (2 * self.m)
        self.cost_history.append(j_theta)
        return j_theta

    def gradient_descent(self):
        for i in range(self.itr):
            self.cost_function()
            error = (np.dot(self.theta, self.X)) - self.y
            self.theta -= self.alpha / self.m * (np.dot(error, self.X.T))

        return self.theta, self.cost_history

    def plot_fitted(self):
        plt = self.plot_data()
        theta, cost = self.gradient_descent()
        plt.plot(self.X, theta[0, 1] * self.X + theta[0, 0], color='red')
        plt.legend()
        plt.show()

    def predict(self, x):
        theta, cost = self.gradient_descent()
        return theta[0, 1] * x + theta[0, 0]


lin = Linear(load_data.load(), 0.01, 1500)
lin.plot_fitted()
print(lin.predict(10))
