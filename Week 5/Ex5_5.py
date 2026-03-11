import numpy as np
import matplotlib.pyplot as plt

def c(t):

    x = 2*t + 1
    y = -t**2

    return x, y

def compute_length_euclidean(curve, points):
    x, y = curve
    length = 0
    for i in range(points-1):
        length += np.sqrt((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)
    return length

def speed_c(t):
    return 2 * np.sqrt(1 + t**2)

def compute_length_integral(t, step_size):
    return np.sum(speed_c(t[:-1])) * step_size

points = 1000
start_t = 0
end_t = 1

t = np.linspace(start_t, end_t, points)
step_size = t[1] - t[0]

print(compute_length_euclidean(c(t), points))
print(compute_length_integral(t, step_size))
print(np.sqrt(2) + np.log(1 + np.sqrt(2)))