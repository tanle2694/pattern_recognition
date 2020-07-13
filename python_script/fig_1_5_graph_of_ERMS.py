import numpy as np
from utils import generate_random_data, PolynomialRegression
import matplotlib.pyplot as plt
np.random.seed(1234)


def compute_loss(x_train, y_train, x_test, y_test, order):
    polynomial_resolve = PolynomialRegression(x_train, y_train, order=order)
    loss_train = polynomial_resolve.rmse(x_train, y_train)
    loss_test = polynomial_resolve.rmse(x_test, y_test)
    return loss_train, loss_test


def main():
    target_function = lambda x: np.sin(2 * np.pi * x)
    train_size = 10
    test_size = 100
    x_train, y_train = generate_random_data(x_min=0, x_max=1, y_function=target_function, size=train_size)
    x_test, y_test = generate_random_data(x_min=0, x_max=1, y_function=target_function, size=test_size)

    orders = list(range(10))
    losses_train = []
    losses_test = []
    for order in orders:
        loss_train, loss_test = compute_loss(x_train, y_train, x_test, y_test, order=order)
        losses_train.append(loss_train)
        losses_test.append(loss_test)
    plt.plot(orders, losses_train, 'o-', mfc='none', mec='b', ms=10, c='b', label='Training')
    plt.plot(orders, losses_test, 'o-', mfc='none', mec='r', ms=10, c='r', label='Test')
    plt.xlabel('Degree')
    plt.ylabel('$E_{RMS}$')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()