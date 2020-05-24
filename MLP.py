from keras.utils import np_utils
from keras.datasets import mnist
import seaborn as sns 
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import numpy as np 
import time 
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense


# Global parameters
num_classes=10
input_layer=28*28
batch=50
epoch=100


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

# The Sequential model is a linear stack of layers. we can create a Sequential model by passing a list of layer instances to the constructor
# Reference : https://keras.io/guides/sequential_model/
# start building model 

model = keras.Sequential()
model.add(layers.Dense(num_classes,input_dim=input_layer,activation='softmax'))

# define An optimizer , A loss function, A metrics
# sgd=Stochastic gradient descent
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])


# Train 
history = model.fit(X_train,Y_train,batch_size=batch,epochs=epoch,verbose=1,validation_data=(X_test,Y_test))


print("Model summary::: ",model.summary())

# Evaluation
score=model.evaluate(X_test,Y_test,verbose=0)
print('Test score: ',score[0])
print('Test accuracy ',score[1])

fig,ax =plt.subplots(1,1)
ax.set_xlabel('epoch')
ax.set_ylabel('categorical crossentropy loss')

x =  list(range(1,epoch+1))

vy=history.history['val_loss']
ty=history.history['loss']
ax.plot(x,vy,'b',label='Validation loss')
ax.plot(x,ty,'r',label='Train loss')
plt.legend()
plt.grid()
plt.show()



fig,ax =plt.subplots(1,1)
ax.set_xlabel('epoch')
ax.set_ylabel('Accuracy')

x =  list(range(1,epoch+1))

vy=history.history['val_acc']
ty=history.history['acc']
ax.plot(x,vy,'b',label='Validation accuracy')
ax.plot(x,ty,'r',label='Train accuracy')
plt.legend()
plt.grid()
plt.show()



