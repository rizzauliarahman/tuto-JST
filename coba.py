import numpy as np
import random


a = [[3, 4, 5], [6, 2, 1], [6, 4, 0], [8, 1, 9]]
a = np.array(a)

l = [i for i in range(len(a))]

print(a)

random.shuffle(l)

a = a[l]
print(a)
