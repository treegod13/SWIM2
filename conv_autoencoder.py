from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv2D, Activation, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Reshape, Flatten, Conv2DTranspose
from keras import optimizers
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from data_utils import *

# Define ship sensors sequence inputs
# Sensors data: sample period = 65 ms
data, label = load_sensors_data()
data = split_sensors_data(data, window_size = 60)
data = data[:, :, [1,2,3,7]]  # remove the time dimension
sample_num, window_size, sensors_dim = data.shape
data = np.reshape(data, [sample_num, window_size, sensors_dim, 1])

# Network parameters
input_shape = (window_size, sensors_dim, 1)
batch_size = 32
kernel_size = 3
latent_dim = 10
layer_filters = [16, 32, 64]
layer_dense = [64, 32, 16]

# Build encoder model
inputs = Input(shape = input_shape, name='sensors_input')
x = inputs
x = BatchNormalization()(x)
pool_index = 0
for filters in layer_filters:
    pool_index += 1
    x = Conv2D(filters = filters,
            kernel_size = kernel_size,
            strides = 1,
            activation = 'relu',
            padding = 'same')(x)
    if pool_index % 3 == 0:
        x = MaxPooling2D(pool_size = [1, 2], padding = 'same')(x)

shape = K.int_shape(x)
print(shape)

x = Flatten()(x)
for n in layer_dense:
    x = Dense(units = n, activation = 'relu')(x)

latent = Dense(latent_dim, activation = 'sigmoid', name='latent_vector')(x)

encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = latent_inputs

for n in layer_dense[::-1]:
    x = Dense(units = n, activation = 'relu')(x)

x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1],shape[2], shape[3]))(x)

pool_index = 0
for filters in layer_filters[::-1]:
    pool_index += 1
    x = Conv2DTranspose(filters = filters,
            kernel_size = kernel_size,
            padding='same',
            activation='relu',
            strides = 1)(x)
    if pool_index % 3 == 0:
        x = UpSampling2D(size = [1, 2])(x)

x = Conv2DTranspose(filters = 1,
        kernel_size = kernel_size,
        padding = 'same')(x)
outputs = x
#outputs = Activation('sigmoid', name='decoder_output')(x)
decoder = Model(latent_inputs, outputs, name='decoder_output')
decoder.summary()

autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()


# Train autoencoder
adam = optimizers.Adam(lr=5e-5, beta_1=0.9, beta_2=0.999, decay=0.0)
#sgd = optimizers.SGD(lr=5e-4)
#autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.fit(x = data, y = data, epochs = 8000, batch_size = batch_size, shuffle=True, validation_split=0.2)

# Plot the figures
print('Data shape: ' + str(data.shape))
plot_autoencoder([autoencoder, encoder], data, 0, file_name='tmp')

# Save the model
# autoencoder.save('./models/autoencoder.h5')
# encoder.save('./models/encoder.h5')
# decoder.save('./models/decoder.h5')



