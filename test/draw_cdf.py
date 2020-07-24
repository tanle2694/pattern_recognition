import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Create some test data
X = np.arange(-2, 3.5, 0.01)
Y_pdf = (stats.norm(0, 0.4).pdf(X) + stats.norm(1, 0.25).pdf(X)) / 2
Y_cdf = np.cumsum(Y_pdf * 0.01)


plt.plot(X, Y_pdf, 'r')
plt.plot(X, Y_cdf, 'b')
print(X.shape)
index_start = 170
sigma = 15
plt.fill_between(X[index_start: index_start + sigma], Y_pdf[index_start: index_start + sigma], -0.05, alpha=0.5, color='g')
plt.xticks([])
plt.yticks([])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_position(('data', -2))
plt.gca().spines['bottom'].set_position(('data', -0.05))
# plt.spines['right'].set_visible(False)
plt.annotate("$p(x)$", xy=(0.6, 0.8))
plt.annotate("$P(x)$", xy=(1.1, 1))
plt.annotate("$\delta_x$", xy=(X[index_start], -0.09))
plt.show()

