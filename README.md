# Convolutional Deep Neural Network for Digit Classification

## AIM
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Problem Statement:
The objective at hand is to create a Convolutional Neural Network (CNN) that is capable of correctly classifying handwritten numbers between 0 and 9. Scannable photos of handwritten numbers, including those not in the standard dataset, ought to be processed by this CNN.

Dataset: 
In the fields of computer vision and machine learning, the MNIST dataset is widely acknowledged as a fundamental resource. It is made up of 28x28 pixel grayscale images that each show a handwritten number from 0 to 9. Ten thousand well labeled test images and sixty thousand training images make up the dataset, which is used to assess the model. These photos are represented in grayscale as numbers between 0 and 255, where 0 denotes black and 255, white. When evaluating different machine learning models, MNIST is a useful standard, especially when it comes to tasks involving digit recognition. We want to create and assess a specific CNN for digit classification using MNIST, and we also want to see if it can generalize to handwritten images from the real world that aren't in the dataset.

## Neural Network Model
![](nn1.png)

## DESIGN STEPS
### Step 1 
involves preprocessing the MNIST dataset by transforming the labels to a one-hot encoded format and scaling the pixel values to the range [0, 1].
### Step 2: 
Use TensorFlow Keras to create a convolutional neural network (CNN) model with the desired architecture.
### Step 3: 
Put the model together using the Adam optimizer and the categorical cross-entropy loss function.
### Step 4: 
Use the preprocessed training data to train the built model for five epochs with a batch size of 64.
### Step 5:
Plotting training and validation metrics, creating a confusion matrix, and producing a classification report are the means by which the trained model's performance on the test set is assessed in Step 5. Additionally, to illustrate model inference, make predictions on sample photos.

## PROGRAM
### Name: Sudharshna Lakshmi S
### Register Number: 212221230110
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[2]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[2]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[400]
plt.imshow(single_image,cmap='gray')
y_train_onehot[200]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(keras.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters =32 , kernel_size =(3,3),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))
model.summary()


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=10,batch_size=40, validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()
print("Sudharshna Lakshmi S 212221230110")


print("Sudharshna Lakshmi S 212221230110")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print("Sudharshna Lakshmi S 212221230110")
print(confusion_matrix(y_test,x_test_predictions))

print("Sudharshna Lakshmi S 212221230110")
print(classification_report(y_test,x_test_predictions))

**Prediction for a single input**
img = image.load_img("/content/3.png")
type(img)

print("Sudharshna Lakshmi S 212221230110")
img = image.load_img('/content/3.png')
plt.imshow(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print("Sudharshna Lakshmi S 212221230110")
print(x_single_prediction)

print("Sudharshna Lakshmi S 212221230110")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![](1.png)

![](2.png)

![](3.png)

### Classification Report
![](5.png)

### Confusion Matrix
![](4.png)

### New Sample Data Prediction
![](6.png)

![](7.png)

![](8.png)

![](9.png)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
