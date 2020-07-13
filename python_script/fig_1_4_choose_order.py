import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
np.random.seed(1234)


def plot_model_with_order(x, y, order, i):
    plt.subplot(2, 2, i)
    poly = PolynomialFeatures(degree=order)
    X = poly.fit_transform(x)
    regressor = LinearRegression()
    regressor.fit(X, y)
    x_test = np.arange(0, 1, 0.00001).reshape(-1, 1)
    X_test = poly.fit_transform(x_test)
    y_predict = regressor.predict(X_test)
    plt.plot(x_test, y_predict, c='r', label='fitting')
    y_target = np.sin(2 * np.pi * x_test)
    plt.plot(x_test, y_target, 'g', label='$sin(2 \pi x)$')
    plt.scatter(x, y, facecolor='none', edgecolor='b', s=50, label='$training$')
    plt.annotate("M={}".format(order), xy=(0.8, 1))


def main():
    x = np.linspace(0, 1, 10)
    y = np.sin(2 * np.pi * x)
    noise = np.random.normal(scale=0.25, size=10)
    y  = y + noise
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    plot_model_with_order(x, y, 0, 1)
    plot_model_with_order(x, y, 1, 2)
    plot_model_with_order(x, y, 3, 3)
    plot_model_with_order(x, y, 9, 4)

    plt.legend(bbox_to_anchor=(1.05, 0.38), loc=2, borderaxespad=0)
    plt.show()
if __name__ == "__main__":
    main()