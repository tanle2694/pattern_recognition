import numpy as np
from utils import generate_random_data, PolynomialRegression
import matplotlib.pyplot as plt
np.random.seed(1234)


def compute_loss(x_train, y_train, x_test, y_test, order, lamda_value):
    polynomial_resolve = PolynomialRegression(x_train, y_train, order=order, use_l2=True, lamda_value=lamda_value)
    loss_train = polynomial_resolve.rmse(x_train, y_train)
    loss_test = polynomial_resolve.rmse(x_test, y_test)
    return loss_train, loss_test


def main():
    target_function = lambda x: np.sin(2 * np.pi * x)
    train_size = 10
    test_size = 100
    x_train, y_train = generate_random_data(x_min=0, x_max=1, y_function=target_function, size=train_size)
    x_test, y_test = generate_random_data(x_min=0, x_max=1, y_function=target_function, size=test_size)

    lamda_values = np.linspace(1e-5, 5e-3, 1000)
    losses_train = []
    losses_test = []
    for lamda_value in lamda_values:
        loss_train, loss_test = compute_loss(x_train, y_train, x_test, y_test, order=9, lamda_value=lamda_value)
        losses_train.append(loss_train)
        losses_test.append(loss_test)
    plt.plot(list(lamda_values), losses_train, '-', mfc='none', mec='b', ms=10, c='b', label='Training')
    plt.plot(list(lamda_values), losses_test, '-', mfc='none', mec='r', ms=10, c='r', label='Test')
    plt.xlabel('Degree')
    plt.ylabel('$E_{RMS}$')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()