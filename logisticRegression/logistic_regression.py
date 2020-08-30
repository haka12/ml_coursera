import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from logisticRegression import load_data


class Logistic:
    def __init__(self, data, alpha, itr):
        self.data = data
        self.itr = itr
        self.alpha = alpha
        r, c = data.shape
        self.m = r
        self.positive = data.loc[data['y'] == 1]
        self.negative = data.loc[data['y'] == 0]
        # intercept added to the original value of the X
        self.X = np.column_stack((np.ones((r, 1)), data[['x1', 'x2']].values))
        self.X = np.transpose(self.X)
        self.y = data['y'].values
        self.theta = np.zeros((1, c))
        self.cost_history = []

    def plot_data(self):
        plt.scatter(x=self.positive['x1'], y=self.positive['x2'], marker='+', label='Positive')
        plt.scatter(x=self.negative['x1'], y=self.negative['x2'], marker='_', label='Negative')
        plt.legend()
        return plt

    # *args is used to accept parameters from the minimize function called later in the code
    def cost_function(self, *args):
        if len(args):
            theta = args[0]
            X = args[1]
            y = args[2]
        else:
            theta = self.theta
            X = self.X
            y = self.y
        z = np.dot(theta, X)
        h_theta = sigmoid_function(z)
        J_theta = ((np.dot(np.log(h_theta), y)) + (np.dot(np.log(1 - h_theta), (1 - y)))) / - self.m
        if np.isnan(J_theta):
            self.cost_history.append(np.inf)
            return np.inf

        self.cost_history.append(J_theta)
        return J_theta

    def gradient(self, *args):
        if len(args):
            theta = args[0]
            X = args[1]
            y = args[2]
        else:
            theta = self.theta
            X = self.X
            y = self.y
        z = np.dot(theta, X)
        h_theta = sigmoid_function(z)
        error = h_theta - y
        gradient = (1 / self.m) * np.dot(error, X.T)
        return gradient.flatten()

    def minimize(self):
        result = optimize.minimize(self.cost_function, self.theta, args=(self.X, self.y),
                                   method=None, jac=self.gradient)
        self.theta = result.x
        return result

    def predict(self, x):
        z = np.dot(self.theta, x.T)
        return 1 if 0.5 <= sigmoid_function(z) else 0

    def plot_fitted(self):
        plt = self.plot_data()
        x = np.linspace(20, 80, 20)
        y = -(self.theta[0] + x * self.theta[1]) / self.theta[2]
        plt.plot(x, y, 'b')


def sigmoid_function(z):
    return 1.0 / (1.0 + np.exp(-z))


logi = Logistic(load_data.load(), 0.01, 1500)
logi.minimize()
ans = logi.predict(np.array([1, 20, 96]))
logi.plot_fitted()
