import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

#L1 normalization if outliers doesnt matter, otherwise use L2
#L1 works by making sure that the sum of absolute values of each row is 1
data_normalization_L1= preprocessing.normalize(input_data, norm='l1')
#L2 works by making sure that the sum of squares is 1
data_normalization_L2= preprocessing.normalize(input_data, norm='l2')
print("\nL1 normallized data: \n", data_normalization_L1)
print("\nL2 normallized data: \n", data_normalization_L2)

