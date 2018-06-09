import numpy as np
from sklearn.preprocessing import normalize

a = [[1, 2, 3],
     [4, 5, 6],
     [1, 2, 3],
     [4, 5, 6]
     ]

a = np.array(a)

print(a)
print(a.max())
print(a.min())

a = (a - a.min()) / (a.max() - a.min())
print(a)
