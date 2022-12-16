import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
    #Define max and min values for X and y that will be used in the mesh grid
    min_x, max_x = X[:,0].min() -1.0, X[:,0].max() +1.0
    min_y, max_y = X[:,0].min() -1.0, X[:,1].max() +1.0

    #Define the step size to use in plotting the mesh grid
    mesh_step_size = 0.01
    #define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),np.arange(min_y, max_y, mesh_step_size))

    #run the classifier on the mesh grid
    output=classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    #reshape the output array
    output= output.reshape(x_vals.shape)

    #Create the plot
    plt.figure()
    #chose the color scheme for the plot
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    #Draw the training point on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black',linewidth=1, cmap=plt.cm.Paired)

    #Specify the boundaries of the plot
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    #Especificamos la escala de los ejes
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1),1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1),1.0)))

    plt.show()