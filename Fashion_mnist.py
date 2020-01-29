import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(type(train_images))
print(train_images[0]) # how the real data looks in the dataset fashion_mnist
""" the data is probably a color coded pixels"""


plt.imshow(train_images[0])#plotting it in graph we can see it represents some clothing

print(train_images[0].shape)#each data is a matrix of 28x28

print(train_images.shape)#there are exactly 60000 rows in our dataset

#gives you a better idea how the data is present in your dataset
l1 = np.array([
    
               [[1,2,3],[3,4,5],[3,4,5],[3,4,5]],
               [[1,2,3],[3,4,5],[3,4,5],[3,4,5]]
    
              ])
print(l1.shape)

l1 = l1/5.0
print(l1)

#we need to scale the data to a lower value to be used efficiently
train_images = train_images/255.0
test_images = test_images/255.0

#to confirm the data is the same as it was we will look at the image again
plt.imshow(train_images[0])

#but now the data is efficient to use take a look
print(train_images[0])

#creating the model
model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"), 
    keras.layers.Dense(10,activation="softmax") #softmax: gives you the probabilities it thinks the data belongs to each class
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


model.fit(train_images,train_labels,epochs=5)#you can tweak the number of epochs to get better results

test_loss,test_acc = model.evaluate(test_images,test_labels)
print("Test Acc:",test_acc)

