import numpy as np
from utils import generate_random_data, PolynomialRegression, plot_curve
import matplotlib.pyplot as plt


def main():
    target_function = lambda x: np.sin(2 * np.pi * x)

    x_random, y_random = generate_random_data(x_min=0, x_max=1, y_function=target_function, size=10)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))

    orders = [1, 3, 9]
    titles = ["Underfitting", "Good fit", "Overfitting"]

    for ax, order, title in zip(axs, orders, titles):
        polynomial_resolve = PolynomialRegression(x_random, y_random, order=order)
        y_predict = polynomial_resolve.predict(x_random)
        y_target = target_function(x_random)
        ax.scatter(x_random, y_random, facecolor='none', edgecolors='b', s=50, label='training')
        plot_curve(ax=ax, function_plot=target_function, color='g', label='$sin(2 \pi x)$')
        plot_curve(ax=ax, function_plot=lambda x: polynomial_resolve.predict(x), color='r', label='fitting')
        ax.set_title(title)

    plt.legend(bbox_to_anchor=(1.05, 0.29), loc=2, borderaxespad=0)
    plt.show()


if __name__ == "__main__":
    main()