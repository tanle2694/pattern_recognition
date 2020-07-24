import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from utils import arrowed_spines
np.random.seed(12)

def main():
    x = np.linspace(-1, 1, 7) + np.random.uniform(-0.15, 0.15, 7)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.45)
    y_zeros = np.zeros((7, ))
    y = stats.norm(0, 1).pdf(x)
    plt.scatter(x, y, color='b', s=50)
    plt.scatter(x, y_zeros, color='black', s=50)
    for i in range(x.shape[0]):
        plt.axvline(x[i], 0, y[i]/ 0.45, color='g', linewidth=3, alpha=0.7)

    x = np.linspace(-4, 4, 1000)
    y = stats.norm(0, 1).pdf(x)

    plt.plot(x, y, color='r')
    plt.annotate("$\mathcal{N}(x_n|\mu, \sigma^2 )$", xy=(0.5, 0.4))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$x_n$")
    plt.ylabel("$p(x)$", rotation=True, labelpad=15)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    arrowed_spines(plt.gca())
    plt.show()

if __name__ == "__main__":
    main()


