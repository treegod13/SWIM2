from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from data_utils import *

# Define ship sensors sequence inputs
# Sensors data: sample period = 65 ms
sensors_dim = 2
len_sensors = 60  # window 60 = 3 second 
sensors_inputs = Input(shape=[60, 4, 1], name='sensors_input')
#sensors_inputs = Input(shape=[64,])
# Define CSI inputs
# CSI sample period = 100 ms
len_csi = 30 # 

# Sensors data encoder part
x = BatchNormalization()(sensors_inputs)
#x = Dense(32, activation='relu')(x)
#x = Dense(32, activation='relu')(x)
#encoded = Dense(16, activation='relu')(x)
#y = Dense(16, activation='relu')(encoded)
#uy = Dense(32, activation='relu')(encoded)
#decoded = Dense(64, activation='relu')(y)

x = Conv2D(filters=32, kernel_size=[3, 3], activation='relu', padding='same')(x)

x = MaxPooling2D(pool_size=(1, 2), padding='same')(x)

x = Conv2D(filters=16, kernel_size=[3, 3], activation='relu', padding='same')(x)

#x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = Conv2D(filters=1, kernel_size=[2, 2], activation='relu', padding='same')(x)

x = MaxPooling2D(pool_size=(1, 2), padding='same')(x)

x = Dense(48, activation='relu')(x)

x = Dense(32, activation='relu')(x)

encoded = Dense(32, activation='relu')(x)

# Decoder part

y = Dense(32, activation='relu')(encoded)

y = Dense(32, activation='relu')(y)

y = Dense(60, activation='relu')(y)

y = Conv2D(filters=16, kernel_size=[2, 2], activation='relu', padding='same')(y)

y = UpSampling2D(size=(1, 2))(y)

y = Conv2D(filters=32, kernel_size=[3, 3], activation='relu', padding='same')(y)

#y = UpSampling2D(size=(2, 2))(y)

y = Conv2D(filters=64, kernel_size=[3,3], activation='relu', padding='same')(y)

y = UpSampling2D(size=(1, 2))(y)

decoded = Conv2D(filters=1, kernel_size=[3, 3], activation='sigmoid', padding='same')(y)

autoencoder = Model(inputs = sensors_inputs, outputs = decoded)

encoder = Model(inputs = sensors_inputs, outputs = encoded)

#encoded_input = Input(shape=[1, 196, 1])
#decoded_layer1 = autoencoder.layers[4](encoded_input)
#decoded_layer2 = autoencoder.layers[3](decoded_layer1)
#decoder = Model(inputs = encoded_input, outputs = decoded_layer2)

adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.0)
#sgd = optimizers.SGD(lr=5e-4)
autoencoder.compile(optimizer=adam, loss='mean_squared_error')

# Train autoencoder
data, label = load_sensors_data()
data = split_sensors_data(data, window_size = 60)
data = data[0:4, :, [1,2,3,7]]  # remove the time dimension
sample_num, window_size, sensors_dim = data.shape
data = np.reshape(data, [sample_num, window_size, sensors_dim, 1])
#data = np.reshape(data, [data.shape[0], np.prod(data.shape[1:])])
#(x_train, _), (x_test, _) = mnist.load_data()
#x_train = x_train.astype('float32')/256
#x_test = x_test.astype('float32') / 255
#x_train = np.reshape(x_train, [60000, 28, 28, 1])
#x_test = np.reshape(x_test, [10000, 28, 28, 1])

#autoencoder.fit(x = x_train[0:256, :, :, :], y = x_train[0:256], epochs=50, batch_size=64, shuffle=True, validation_split=0.5)

autoencoder.fit(x = data, y = data, epochs=5000, batch_size=4, shuffle=True, validation_split=0.1)

# Plot the learning rate
plt.figure()
plt.plot(autoencoder.history.history['loss'], label='loss')
plt.plot(autoencoder.history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('./pics/ae_adadelta.png')
plt.show()

# Plot the sensors data
t = data[1,:,:,:]
plt.figure()
plt.plot(t[:,0,0], label='x_axis Acc')
plt.plot(t[:,1,0], label='y_axis Acc')
plt.plot(t[:,2,0], label='z_axis Acc')
plt.plot(t[:,3,0], label='Compass')
plt.legend()
plt.savefig('./pics/sensor_data.png')
plt.show()

# Generate hidden states for RNN
t = data[1:2,:,:,:]
e = encoder.predict(t)
plt.plot(e[0,:,0,0], label='Hidden states')
plt.legend()
plt.savefig('./pics/basis.png')
plt.show()

# Plot decoded data
t = data[1:2,:,:,:]
d = autoencoder.predict(t)
plt.figure()
plt.plot(d[0,:,0,0], label='x_axis Acc')
plt.plot(d[0,:,1,0], label='y_axis Acc')
plt.plot(d[0,:,2,0], label='z_axis Acc')
plt.plot(d[0,:,3,0], label='Compass')
plt.legend()
plt.savefig('./pics/decoded_data.png')
plt.show()



# RNN part



