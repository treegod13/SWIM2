from keras.models import Model, load_model
from keras.layers import Input, GRU, Dense, TimeDistributed, LSTM
import numpy as np
from data_utils import *
from keras import backend as K
from csi_embedding import *

def rnn_transfer(window_size, sensors_dim, embedding_dim,
                hidden_dim, csi_dim = 30,
                encoder_path = './models/encoder.h5'):
    """
    Build LSTM-based seq2seq model.
    csi_time_length: a sequence length of CSI data, e.g 60 samples
    Input:
        window_size: a dictionary contains {'sensors': 60, 'csi': 10}
        sensors_dim: int number 4
        embedding_dim: three times of CSI dim 90
        hidden_dim: 10 as used in autoencoder
    Comments:
    LSTM layer Inputs:
        3D tensor (batch_size, timesteps, input_dim).
    LSTM layer Outputs:

    """
    # Parameters
    window_size_sensors = window_size['sensors']
    window_size_csi = window_size['csi']
#    sensors_dim = sensors_dim
#    embedding_size = embedding_size
#    csi_size = csi_size

    # CSI embedding
#     time_length = 10
#     csi_length = 30
#     embedding_size = 90
    csi_input = Input(shape=(window_size_csi, csi_dim), name='CSI_inputs')
    x = csi_input
    x = TimeDistributed(csi_embedding_dense(embedding_dim), name='csi_embedding')(x)
    embedded_csi = x

    # sensors embeding
#     data, label = load_sensors_data()
#     data = split_sensors_data(data, window_size = 60)
#     data = data[:, :, [1,2,3,7]]  # remove the time dimension
#     sample_num, window_size, sensors_dim = data.shape
#     data = np.reshape(data, [sample_num, window_size, sensors_dim, 1])

    # Previous sensors encoding
    prev_sensors_input = Input(shape=(window_size_sensors, sensors_dim, 1), name='Previous_inputs')
    y1 = prev_sensors_input
    encoder = load_model(encoder_path)
    for layer in encoder.layers:
        layer.trainable = False
        y1 = layer(y1)
    state_c = y1

    # Current sensors encoding
    curr_sensors_input = Input(shape=(window_size_sensors, sensors_dim, 1), name='Current_inputs')
    y2 = curr_sensors_input
#    encoder = load_model(encoder_path)
    for layer in encoder.layers:
        layer.trainable = False
        y2 = layer(y2)
    state_h = y2

    # RNN transfer model
    # state_h = Input(shape=(10,), name='hidden_state_h'
#    latent_dim = 10
    hidden_states = [state_h, state_c]
    z = LSTM(hidden_dim, return_state = False, return_sequences = True)(embedded_csi, initial_state = hidden_states)

    # Reconstruct the CSI
    z = TimeDistributed(Dense(csi_dim), name='csi_reconstruction')(z)
    csi_recons = z

    # Define model
    rnn_model = Model(inputs = [csi_input, prev_sensors_input, curr_sensors_input], outputs = csi_recons)

    return rnn_model



