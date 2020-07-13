import numpy as np
from utils import generate_random_data, PolynomialRegression, plot_curve
import matplotlib.pyplot as plt



def plot_with_number_data(ax, number_data, order):
    np.random.seed(1234)
    target_function = lambda x: np.sin(2 * np.pi * x)
    x, y = generate_random_data(x_min=0, x_max=1, y_function=target_function, size=number_data)
    polynomial_resolve = PolynomialRegression(x, y, order=order)
    ax.scatter(x, y, facecolor='none', edgecolors='b', s=50)
    plot_curve(ax, polynomial_resolve.predict, color='r')
    plot_curve(ax, target_function, color='g')
    ax.text(0.8, 0.8, 'N = {}'.format(number_data))



def main():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    plot_with_number_data(axs[0], 15, order=9)
    plot_with_number_data(axs[1], 100, order=9)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()