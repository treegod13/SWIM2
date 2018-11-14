from __future__ import absolute_import

import keras
from keras.layers import Activation, Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import optimizers
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.use('TkAgg')

#np.random.seed(1337)

# MNIST dataset

# this is the size of our encoded representaitons
input_dim = 784
encoding_dim = 32

# this is our input placeholder. Input returns a tensor
input_img = Input(shape=(input_dim,))

# encoded representation of input. Dense layer
encoded = Dense(encoding_dim, activation='relu')(input_img)

# decoded representation of input. Also a dense layer
decoded = Dense(input_dim, activation='relu')(encoded)

# model maps the input and its reconstruction. Creat a model include two dense layer.
autoencoder = Model(inputs = input_img, outputs = decoded)

# model maps the input and its encoded representation
encoder = Model(inputs = input_img, outputs = encoded)

# model the decoder.
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(inputs = encoded_input, outputs = decoder_layer(encoded_input))

# Obtain data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, [x_train.shape[0], np.prod(x_train.shape[1:])])
x_test = np.reshape(x_test, [x_test.shape[0], np.prod(x_test.shape[1:])])

print(x_train.shape)
print(x_test.shape)

# Train model
adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=0.0)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.compile(optimizer=adam, loss='mean_squared_error')
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Predict
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Display
for i in range(1):
    n = np.random.choice(x_test.shape[0])
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(x_test[n].reshape([28, 28]))
    plt.gray()
    plt.subplot(2, 2, 2)
    plt.imshow(decoded_imgs[n].reshape([28, 28]))
    plt.gray()
    #plt.savefig('./pics/mean_loss.png')
    plt.show()

n = 5  # how many digits we will display
plt.figure(figsize=(10, 4))
for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.savefig('./pics/binary_adadelta.png')

# Plot the learning rate
plt.figure()
plt.plot(autoencoder.history.history['loss'], label='loss')
plt.plot(autoencoder.history.history['val_loss'], label='val_loss')
plt.legend()
#plt.savefig('./pics/ae_adadelta.png')
#plt.show()











