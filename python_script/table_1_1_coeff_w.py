import numpy as np
from utils import generate_random_data, PolynomialRegression
import pandas as pd


# np.random.seed(1234)

def compute_coefficient(x_train, y_train, order):
    polynomial_resolve = PolynomialRegression(x_train, y_train, order=order).get_coefficients()[0]
    coefs = {}
    for i in range(order + 1):
        coefs['w_{}'.format(i)] = polynomial_resolve[i]
    return coefs



def main():
    target_function = lambda x: np.sin(2 * np.pi * x)
    train_size = 10
    x_train, y_train = generate_random_data(x_min=0, x_max=1, y_function=target_function, size=train_size)
    orders = [0, 1, 6, 9]
    coefs_with_order = {}
    for order in orders:
        coefs = compute_coefficient(x_train, y_train, order=order)
        coefs_with_order['M={}'.format(order)] = coefs
    coefs_with_order_frame = pd.DataFrame(coefs_with_order)
    coefs_with_order_frame = coefs_with_order_frame.fillna(' ')
    print(coefs_with_order_frame)

if __name__ == "__main__":
    main()