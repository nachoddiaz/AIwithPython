import numpy as np
from sklearn import preprocessing

#Define some input labels
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

#Creating the labelobjetc and train it
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

#print the mapping
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

#encode some randomly ordered labels and see how it performs
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

#decode some numbers
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))

