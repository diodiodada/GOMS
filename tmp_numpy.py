import numpy as np
import pickle

n = 0
data = pickle.load(open('FetchPickAndPlace-v0.p', 'rb'))
print(data.shape)


# for i in range(data.shape[0]):
#     if data[i][-1] == 1.0:
#         print(i+1)
#         n = n + 1
# print(n)

data = data.reshape((5000, 50, 58))

print(data.shape)

for i in range(5000):
    if data[i, 49, 57] != 1.0:
        print("wrong !!")
    else:
        print("yes !!")

