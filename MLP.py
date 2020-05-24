from keras.utils import np_utils
from keras.datasets import mnist
import seaborn as sns 
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import numpy as np 
import time 
import pickle


# Global parameters
num_classes=10


# the data, shuffled and split between a train and test sets 
with open('mnist_data.pkl', 'rb') as f:
    data = pickle.load(f)

data_keys=data.keys()
print("Data keys:::::",data.keys())

# Train data
X_train=np.asarray(data['trainImages'])
y_train=np.asarray(data['trainLabels'])

# Test Data
X_test=np.asarray(data['testImages'])
y_test=np.asarray(data['testLabels'])

print("Number of train images::::",len(y_train))
print("Number of test images::::",len(y_test))
print("size of each image::::", X_train.shape[1])



# Normalisation of images 
X_train=X_train/255
X_test=X_test/255

# Convert labels into one hot vector 
print("class label of first image:::: ",y_train[0] )

Y_train=np_utils.to_categorical(y_train,num_classes)
Y_test=np_utils.to_categorical(y_test,num_classes)

print("After converting one hot vector:::: ", Y_train[0])

def plot_functions(x, vy, ty, ax ,color=['b']):
    ax.plot(x,vy,'b',label='Validation loss')
    ax.plot(x,ty,'r',label='Train loss')
    plt.legend()
    plt.grid()
    fig.canvas.draw()
