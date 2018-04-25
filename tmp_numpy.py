import numpy as np
import pickle
# a = input("输入")
# a = a.split(" ")
# a = np.array(a)
# print(a)

# a = np.ndarray((4,))
# print(a)

# A = np.array([[0, 1, 2], [0, 2, 0]])
# print(type(A))

# a = []
# for i in range(3):
#     a.append([1,2,3])
#
# a = np.array(a)
# print(type(a))
# print(a)


n = 0
data = pickle.load(open('FetchPickAndPlace-v0.p', 'rb'))
print(data.shape)

for i in range(data.shape[0]):
    if data[i][-1] == 1.0:
        print(i+1)
        n = n + 1
print(n)
