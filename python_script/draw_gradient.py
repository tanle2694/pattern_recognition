import numpy as np
import matplotlib.pyplot as plt

function_original = lambda x: 3 * x**2
function_gradient = lambda x: 6 * x

x = np.arange(-10, 10, 0.1)
y = function_original(x)
plt.plot(x, y)


x_gradient = 5.0
y_gradient = function_gradient(x_gradient)
y_origin = function_original(x_gradient)
x_vector_draw = x_gradient
plt.show()