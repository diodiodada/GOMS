import numpy as np
from sklearn.preprocessing import normalize

a = [[1, 2, 3],
     [4, 5, 6],
     [1, 2, 3],
     [4, 5, 6]
     ]

a = np.array(a)

print(a)
print()

print(a.mean())
print(a.std())
print(a.var())
print(a.std() * a.std())
print()

a = (a - a.mean()) / a.std()
print(a)
