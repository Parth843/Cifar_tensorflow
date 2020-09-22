from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

# Load the data and scale it.
cifar = tf.keras.datasets.cifar10
(xtrain,ytrain), (xtest,ytest) = cifar.load_data()
xtrain, xtest = xtrain/255.0, xtest/255.0
ytrain, ytest = ytrain.flatten(), ytest.flatten()

# Print the dimensions of the data.
print('xtrain and test shapes:', xtrain.shape, ' ', xtest.shape)
print('ytrain and test shape:', ytrain.shape,' ',ytest.shape)

# Number of output classes
K = len(set(ytrain))

# Build the model using the functional API
i = Input(shape=xtrain[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i,x)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

r = model.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=50)

# Fit with data augmentation
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(xtrain, ytrain, batch_size)
steps_per_epoch = xtrain.shape[0] // batch_size
r = model.fit(train_generator, validation_data=(xtest, ytest), steps_per_epoch=steps_per_epoch, epochs=50)

#Plot the accuraccy and error of the model
plt.plot(r.history['loss'],label='error')
plt.plot(r.history['val_loss'],label='val_error')
plt.legend()
plt.plot(r.history['accuracy'],label='acc')
plt.plot(r.history['val_accuracy'],label='val_acc')
plt.legend()

ptest = model.predict(xtest).argmax(axis=1)

errorid = np.where(ptest != ytest)[0]

i = np.random.choice(errorid)

labels = '''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck'''.split()

plt.imshow(xtest[i],cmap='gray')
plt.title('Actual: {} ; predicted: {}'.format(labels[ytest[i]],labels[ptest[i]]))