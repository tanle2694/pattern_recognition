import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(1234)


def generate_random_data(x_min, x_max, y_function, size, use_x_linspace=True, noise_scale=0.25):
    x = np.random.uniform(x_min, x_max, size)
    if use_x_linspace:
        x = np.linspace(x_min, x_max, size)
    noise = np.random.normal(scale=noise_scale, size=size)
    y = y_function(x) + noise
    return x, y


class PolynomialRegression():

    def __init__(self, x, y, order):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        self.poly = PolynomialFeatures(degree=order)
        X = self.poly.fit_transform(x)
        self.regressor = LinearRegression()
        self.regressor.fit(X, y)

    def predict(self, x_test):
        x_test = x_test.reshape(-1, 1)
        X_test = self.poly.fit_transform(x_test)
        y_predict = self.regressor.predict(X_test)
        y_predict = y_predict.flatten()
        return y_predict

    def compute_loss(self, x_test, y_test):
        y_predict = self.predict(x_test)
        loss = np.sum((y_predict - y_test)**2) * 0.5

        return loss

    def rmse(self, x_test, y_test):
        y_predict = self.predict(x_test)
        loss = np.sqrt(np.mean(np.square(y_predict - y_test)))
        return loss


def plot_curve(ax, function_plot, x_min=0, x_max=1, size=10000, color='r', label=''):
    x = np.linspace(x_min, x_max, size)
    y = function_plot(x)
    ax.plot(x, y, c=color, label=label)



def plot_2dcurve_with_gradient():
    function = lambda x: 2 * x ** 2
    fig, ax = plt.subplots()
    plot_curve(ax, function, x_min=-10, x_max=10, label='$y=2 x^2$')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_2dcurve_with_gradient()