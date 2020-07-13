from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.random.uniform(0, 1, 20)
y = np.sin(2 * np.pi * x)
noise = np.random.normal(0, 0.1, 20)
y = y + noise
poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(x.reshape(-1, 1))


regressor = LinearRegression()
regress = regressor.fit(X, y.reshape(-1, 1))
# model = poly.fit(x.reshape(-1, 1), y.reshape(-1, 1))

print(regress.coef_)