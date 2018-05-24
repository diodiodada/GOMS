import numpy as np
import pickle
import random

# a = np.arange(0, 1, 0.1)
# print(a)
#
# a = np.linspace(0, 1, 11)
# print(a)
# for i in range(100):
#     print(random.randint(0, 3))


arr = np.arange(9).reshape((3, 3))
print(arr)
print()
np.random.shuffle(arr)
print(arr)
