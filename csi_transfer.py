from keras.models import load_model
from keras.layers import Input, LSTM, Dense
from keras import optimizers
from keras.utils import plot_model
import numpy as np
from rnn_sequence import rnn_transfer
from data_utils import *
from csi_embedding import *
import sys, getopt

# System parameters
train_able = 0
# train_able = sys.argv[1]

# Load sensors data
file_path = '/home/chen/OneDrive/rnn/'
sensors_data, sensors_label = load_sensors_data(file_path + 'data/ship_sensors_1.csv')
sensors_data = split_sensors_data(sensors_data, window_size = 60)
sensors_data = sensors_data[:, :, [1,2,3,7]]  # remove the time dimension

# Load CSI data
csi_data = load_csi_data(file_path + 'data/csi_dynamic.csv')
csi_window_size = 360  # window_size should = T_sensors / T_csi * sensors_window_size
csi_data = split_sensors_data(csi_data, csi_window_size)

# csi_x = load_csi_data('./data/csi_sailing.csv')
# csi_window_size = 10  # window_size should = T_sensors / T_csi * sensors_window_size
# csi_x = split_sensors_data(csi_x, csi_window_size)
# csi_y = load_csi_data('./data/csi_anchored.csv')
# csi_y = split_sensors_data(csi_y, csi_window_size)

# Network parameters
sensors_samples, sensors_window_size, sensors_dim = sensors_data.shape
window_size = {}
window_size['sensors'] = sensors_window_size
window_size['csi'] = csi_window_size
embedding_dim = 90   # CSI embedding dimension = 3 * 30
hidden_dim = 10   # According to encoder
csi_dim = csi_data.shape[2]  # Here, we only use amplitude
csi_samples = csi_data.shape[0]
sample_num = min(csi_samples, sensors_samples)-1 # sample num used in training
batch_size = 32

# Load the autoencoder model
rnn_model = rnn_transfer(window_size, sensors_dim, embedding_dim,
                         hidden_dim, csi_dim, encoder_path = './models/encoder.h5')
rnn_model.summary()

# Train the model
prev_csi = csi_data[0:sample_num,:,:]
curr_csi = csi_data[1:sample_num+1,:,:]
prev_sensors = sensors_data[0:sample_num, :,:]
curr_sensors = sensors_data[1:sample_num+1, :,:]
# csi_x = np.reshape(csi_x, [sample_num, csi_window_size, csi_dim, 1])
# csi_y = np.reshape(csi_y, [sample_num, csi_window_size, csi_dim, 1])
prev_sensors = np.reshape(prev_sensors, [sample_num, sensors_window_size, sensors_dim, 1])
curr_sensors = np.reshape(curr_sensors, [sample_num, sensors_window_size, sensors_dim, 1])

adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=0.0)
rnn_model.compile(optimizer=adam, loss='mse')
# rnn_model.fit(x = [prev_csi, prev_sensors, curr_sensors], y = curr_csi, epochs = 18000, 
#               batch_size = batch_size, shuffle=True, validation_split=0.2)

# # Save the model
# rnn_model.save_weights('./models/rnn_weights.h5')
# plot_loss(rnn_model, file_name='rnn')

# Test: Load model
t_model = rnn_transfer(window_size, sensors_dim, embedding_dim, 
                              hidden_dim, csi_dim, encoder_path = './models/encoder.h5')
t_model.load_weights('./models/rnn_weights.h5')

# load_model('./models/rnn.h5',
#         custom_objects={'csi_embedding_dense':csi_embedding_dense})


# Plot the data
# plot_rnn(rnn_model, prev_csi, curr_csi, prev_sensors, curr_sensors)
# plot_model(rnn_model, to_file='./pics/rnn_model.png')

# plot_csi_data(csi_x[:,0:1,:], 0, 'Collecting_CSI')
print('Data samples:' + str(sample_num))
idx = 15
plot_sensors_data(prev_sensors[idx,:,[0, 2],:], 'Previous_Sensors')
plot_sensors_data(curr_sensors[idx,:,:,:], 'Current_Sensors')
predict_csi = t_model.predict([prev_csi[idx:idx+1,:,:], prev_sensors[idx:idx+1,:,:,:], curr_sensors[idx:idx+1,:,:,:]])
plot_csi_data(prev_csi, idx, 'Previous_CSI')
plot_csi_data(curr_csi, idx, 'Current_CSI')
plot_csi_data(predict_csi, 0, 'Predict_CSI')
# plot_sensors_data(sensors_data[3,:,:,:],'sensors')












