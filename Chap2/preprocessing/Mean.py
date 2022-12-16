import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])
            
#Print mean and standard deviation
print("Before:")
print("Mean= ", input_data.mean(axis=0))
print("std deviation =", input_data.std(axis=0))

#remove mean
#scale coge cada dato, le resta la media y lo divide
#entre la desviación típica
data_scaled= preprocessing.scale(input_data)
print(data_scaled)
print("After:")
print("Mean= ", data_scaled.mean(axis=0))
print("std deviation =", data_scaled.std(axis=0))