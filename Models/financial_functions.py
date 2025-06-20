import math
import random
import numpy as np


# Define a box of size 10 using a radius of 5

radius = 5

box_size = 2 * radius

points_to_generate = 10000

# Generate random points that are from a uniform distribution inside the box
points_generated = (np.random.sample((points_to_generate, 2)) - 0.5) * 10 # pointsDim & (x, y) that are generate from [0, 1) => [-5, 5)

count_inside_circle = 0

for point in points_generated:
    if(np.sqrt(point[0] ** 2 + point[1] ** 2)) < radius:
        count_inside_circle += 1


print((count_inside_circle / points_to_generate) * 4)