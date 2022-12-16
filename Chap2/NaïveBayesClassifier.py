import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'