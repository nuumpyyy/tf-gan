# Implement true dataset

import numpy as np
#import matplotlib.pyplot as plt

# Simple quadratic function used to simplify demo
def f(x):
    return x*x + 10

def sample_data(n, scale):
    data = []

    rng = np.random.default_rng()
    x = rng.uniform(-scale/2, scale/2, n)

    for i in range(n):
        y = f(x[i])
        data.append([x[i], y])

    return np.array(data)

# Plot generated data

#data = sample_data(10000, 100)

#x = data[:, 0]
#y = data[:, 1]

#plt.scatter(x, y, s=5)
#plt.xlabel("x")
#plt.ylabel("y")
#plt.title("Sample quadratic data")
#plt.show()