import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

def draw_random_function(x, order):
    x_orders = np.zeros((x.shape[0], order))
    w = np.random.uniform(-30, 30, order).T
    for i in range(order):
        x_orders[:, i] = x.T**i
    print(x_orders.shape, w.shape)
    output = np.dot(x_orders, w)
    plt.plot(x, output.T)


def main():
    x = np.linspace(0, 1, 10)
    y = np.sin(2 * np.pi * x)
    noise = np.random.normal(scale=0.25, size=10)
    plt.scatter(x, y + noise, facecolor='none', edgecolors='b', s=50, label='training data')
    x_curve = np.arange(0, 1, 0.000001)
    y_curve = np.sin(2 * np.pi * x_curve)
    plt.plot(x_curve, y_curve, 'g', label='$sin(2 \pi x)$')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()
