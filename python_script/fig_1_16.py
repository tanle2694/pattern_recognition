
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from utils import arrowed_spines

def main():
    x = np.linspace(-2, 3, 1000)
    y = x**3
    plt.xlim(-2, 3)
    plt.ylim(-5, 20)
    plt.axvline(1.5, 0, 1, color='black')
    plt.axhline(y= 1.5**3, xmax=3.5/5, linestyle='--', color='g')

    y_gaussian = np.linspace(-5, 20, 1000)
    x_gaussian = stats.norm(1.5**3, 1).pdf(y_gaussian) + 1.5
    plt.plot(x_gaussian, y_gaussian, color='b')
    plt.annotate('$y(x_0, w)$', xy=(-2.5, 1.5**3), annotation_clip=False)
    plt.annotate("y(x, w)", xy=(2, 18))
    plt.annotate("$p(t|x_0,w,\\beta)$", xy=(1.8, 2))

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    arrowed_spines(plt.gca())
    plt.plot(x, y, color='r')
    plt.show()

if __name__ == "__main__":
    main()