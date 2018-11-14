
from __future__ import print_function
import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg") # Use TkAgg to show figures

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url
from cs231n.classifiers.rnn import *

def rel_error(x, y):
    """Return relative error."""
    return np.max(np.abs(x - y)) / (np.maximum(1e-8, np.abs(x) + np.abs(y)))

# Load COCO data. Return a dict
data = load_coco_data(pca_features=True)

# Print out the dict
#for k, v in data.items():
#    if type(v) == np.ndarray:
#        print(k, type(v), v.shape, v.dtype)
#    else:
#        print(k, type(v), len(v))

# Look at data
#batch_size = 3
#captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
#for i, (caption, url) in enumerate(zip(captions, urls)):
#    print(caption)
#    print(url)
#    plt.imshow(image_from_url(url))
#    plt.axis('off')
#    caption_str = decode_captions(caption, data['idx_to_word'])
#    plt.title(caption_str)
#    plt.show()

# RNN
#np.random.seed(231)

small_data = load_coco_data(max_train=500)

small_rnn_model = CaptioningRNN(
            cell_type = 'rnn',
            word_to_idx = data['word_to_idx'],
            input_dim = data['train_features'].shape[1],
            hidden_dim = 512,
            wordvec_dim = 256,
            )

small_rnn_solver = CaptioningSolver(small_rnn_model, small_data,
        update_rule = 'adam',
        num_epochs = 100,
        batch_size = 25,
        optim_config = {'learning_rate': 5e-3},
        lr_decay = 0.95,
        verbose = True, print_every = 10,
        )


small_rnn_solver.train()

# Plot
#plt.plot(small_rnn_solver.loss_history)
#plt.xlabel('Iteration')
#plt.ylabel('Loss')
#plt.title('Training loss')
#plt.show()

#plt.plot(small_rnn_solver.train_acc_history)
#plt.xlabel('Iteration')
#plt.ylabel('Accuracy')
#plt.title('Training Accuracy')
#plt.show()

# Test-time sampling
for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data, split=split, batch_size=5)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = small_rnn_model.sample(features)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.show()


