import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(12)
x = np.linspace(-1, 1, 7) + np.random.uniform(-0.15, 0.15, 7)
y_zeros = np.zeros((7, ))
y = stats.norm(0, 1).pdf(x)
plt.scatter(x, y, color='b', s=50)
plt.scatter(x, y_zeros, color='black', s=50)
x = np.linspace(-4, 4, 1000)
y = stats.norm(0, 1).pdf(x)

plt.plot(x, y, color='r')
plt.annotate("$\mathcal{N}(x_n|\mu, \sigma^2 )$", xy=(0.5, 0.4))
plt.xticks([])
plt.yticks([])
# plt.gca().spines['bottom'].set_position(('data', 0))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.show()



