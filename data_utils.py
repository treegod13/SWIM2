import numpy as np
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


file_path = '/home/chen/OneDrive/rnn/pics/'
show_flag = 1

def load_sensors_data(file_name='./data/car_sensors_1.csv'):
    """ 
    This function load the sensors data.
    """
    f = np.loadtxt(file_name, dtype = np.str, delimiter = ',')
    data = f[1:, :].astype(np.float)
    label = f[0, :]
    return data, label

def split_sensors_data(data, window_size = 60):
    """
    This function split the sensors data into window size samples.
    """
    length = data.shape[0] // window_size
    data = np.split(data[0:length * window_size, :], length, axis = 0)
    data = np.array(data)
    return data

def load_csi_data(file_name='./data/csi_sailing.csv'):
    """
    This function load the CSI data.
    """
    data = np.loadtxt(file_name, dtype=np.str, delimiter=',')
    data = data.astype(np.float)
    return data

def split_csi_data(data, window_size = 10):
    """
    window_size should = T_sensors / T_csi * sensors_window_size
    # Here, T_csi = 100ms, T_sensors = 65ms, sensors_window_size=60
    """
    data = split_sensors_data(data, window_size)
    return data

def plot_loss(model, file_name):
    plt.figure()
    plt.plot(model.history.history['loss'], label='loss')
    plt.plot(model.history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(file_path = 'loss_' + file_name)
    if show_flag == 1:
        plt.show()

def plot_sensors_data(t, file_name):
    plt.figure()
    plt.plot(t[:,0,0], label='x_axis Acc')
    plt.plot(t[:,1,0], label='y_axis Acc')
    plt.plot(t[:,2,0], label='z_axis Acc')
    plt.plot(t[:,3,0], label='Compass')
    plt.legend()
    plt.savefig(file_path + 'sensors_data_' + file_name)
    if show_flag == 1:
        plt.show()

def plot_sensors_hidden(t, file_name):
    plt.figure()
    plt.plot(t[0,:], label='Hidden states')
    plt.legend()
    plt.savefig(file_path + 'basis_' + file_name)
    if show_flag == 1:
        plt.show()

def plot_sensors_decoded(t, file_name):
    plt.figure()
    plt.plot(t[0,:,0,0], label='x_axis Acc')
    plt.plot(t[0,:,1,0], label='y_axis Acc')
    plt.plot(t[0,:,2,0], label='z_axis Acc')
    plt.plot(t[0,:,3,0], label='Compass')
    plt.legend()
    plt.savefig(file_path + 'decoded_data_' + file_name)
    if show_flag == 1:
        plt.show()

def plot_autoencoder(model, data, index, file_name):
    plot_loss(model[0], file_name)

    sensors = data[index, :, :, :]
    plot_sensors_data(sensors, file_name)

    hidden = model[1].predict(data[index:index+1, :, :, :])
    plot_sensors_hidden(hidden, file_name)

    decoded = model[0].predict(data[index:index+1, :, :, :])
    plot_sensors_decoded(decoded, file_name)

def plot_csi_data(t, index, file_name):
    t = t[index, :, :]
    num = t.shape[0]
    x_axis = np.arange(1, 31)
    plt.figure()
    for i in range(num):
        plt.plot(x_axis, t[i,:])
    plt.title(file_name)
    plt.savefig(file_path + 'csi_' + file_name)
    if show_flag == 1:
        plt.show()

def plot_rnn(model, csi_x, csi_y, sensors, index=0):
    plot_loss(model, file_name='rnn')

    sensors = sensors[index, :, :, :]
    plot_sensors_data(sensors, file_name='rnn')

    plot_csi_data(csi_x, 1, 'Sailing state')

    plot_csi_data(csi_y, 1, 'Anchored state')

#     plt.figure()
#     csi_plot = csi_x[index,:,:]
#     csi_plot.dtype = 'float32'
#     plt.plot(csi_plot)
#     plt.title('Sailing state')
#     plt.savefig('./pics/rnn_csi_sail.png')
#     plt.show()

#     plt.figure()
#     csi_plot = csi_y[index,:,:]
#     csi_plot.dtype = 'float32'
#     plt.plot(csi_plot)
#     plt.title('anchor state')
#     plt.savefig('./pics/rnn_csi_anchor.png')
#     plt.show()

