from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
from keras.layers import Input
from keras.layers import TimeDistributed

class csi_embedding_dense(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim    # output_dim must be the multiple of csi_dim
        super(csi_embedding_dense, self).__init__(**kwargs)

    def build(self, input_shape):
        # creat trainable weights
        self.scale = self.output_dim // input_shape[1]
        self.embedding_weights = self.add_weight(name='embedding_weights',
                                    shape=(input_shape[1], self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
#        self.embedding_filger = self.add_weight
#        eyes = K.eye(input_shape[1])
        super(csi_embedding_dense, self).build(input_shape) # Important to call

    def call(self, x):
#        kernel_0 = K.eye(self.weights_num)
#         weights_scale = K.tile(self.embedding_weights, [9,1])
#         kernel = K.eye(self.weights_num) * weights_scale
#         for i in range(self.weights_num):
#             kernel = K.concatenate([kernel, ])
        weights = self.embedding_weights
        scale = self.scale
#        dia = K.eyes(scale)
        kernel = weights
        return K.dot(x, kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(csi_embedding_dense, self).get_config()
        return dict(list(base_config.items())+ list(config.items()))

class csi_embedding(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim    # output_dim must be the multiple of csi_dim
        super(csi_embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # creat trainable weights
        self.scale = self.output_dim // input_shape[1]
        self.embedding_weights = self.add_weight(name='embedding_weights',
                                    shape=(1, self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
        self.ishape = input_shape
        super(csi_embedding, self).build(input_shape) # Important to call

    def call(self, x):
        ishape = self.ishape
        s = self.scale
        w = self.embedding_weights
        y = K.dot(x[:,:,0:1], w[:, 0:s])
        for i in range(ishape[1]-1):
            j = i + 1
            t = K.dot(x[:,:, j:j+1], w[:, j*s:(j+1)*s])
            y = K.concatenate([y, t])
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# a = Input(shape=(1,3,2))
# b = TimeDistributed(csi_embedding(4))(a)

