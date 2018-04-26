import numpy as np
import pickle

s_1 = [10/6]*6
s_2 = [10/4]*4
s_3 = [10/5]*5
s_4 = [10/4]*4
s_5 = [10/31]*31

s = s_1 + s_2 + s_3 + s_4 + s_5

sample_weight = [s] * 59
sample_weight = np.array(sample_weight)
print(sample_weight.shape)
