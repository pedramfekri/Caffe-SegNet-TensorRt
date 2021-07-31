caffe_root = '/home/pedram/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys

sys.path.insert(0, caffe_root + 'python')
import caffe

''' it should run in jupyter lab or notebook to create dataset using bash
# run scripts from caffe root
import os
os.chdir(caffe_root)
# Download data
!data/mnist/get_mnist.sh
# Prepare data
!examples/mnist/create_mnist.sh
# back to examples
os.chdir('examples')
'''

from caffe import layers as L, params as P
import matplotlib.pyplot as plt
import numpy as np
import time


# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

def segnet():
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.ImageData(
        image_data_param=dict(source='/home/pedram/caffe/models/SegNet/SegNet-Tutorial/CamVid/train.txt' , batch_size=4),
        ntop=2)

    # layer 1
    n.conv1_1 = L.Convolution(n.data, kernel_size=3, num_output=64, weight_filler=dict(type='msra'),
                            bias_filler=dict(type='constant'),
                              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.conv1_1_bn = L.BatchNorm(n.conv1_1, use_global_stats=1, in_place=True)
    n.relu1_1 = L.ReLU(n.conv1_1, in_place=True)
    n.conv1_2 = L.Convolution(n.conv1_1, kernel_size=3, num_output=64, weight_filler=dict(type='msra'),
                               bias_filler=dict(type='constant'),
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.conv1_2_bn = L.BatchNorm(n.conv1_2, use_global_stats=1, in_place=True)
    n.relu1_2 = L.ReLU(n.conv1_2, in_place=True)
    n.pool1, n.pool1_mask = L.Pooling(n.conv1_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    #
    # # layer 2
    # n.conv2_1 = L.Convolution(n.pool1, kernel_size=3, num_output=128, weight_filler=dict(type='msra'),
    #                           bias_filler=dict(type='constant'),
    #                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # n.conv2_1_bn = L.BatchNorm(n.conv2_1, use_global_stats=1, in_place=True)
    # n.relu2_1 = L.ReLU(n.conv2_1, in_place=True)
    # n.conv2_2 = L.Convolution(n.conv2_1, kernel_size=3, num_output=128, weight_filler=dict(type='msra'),
    #                           bias_filler=dict(type='constant'),
    #                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # n.conv2_2_bn = L.BatchNorm(n.conv2_2, use_global_stats=1, in_place=True)
    # n.relu2_2 = L.ReLU(n.conv2_2, in_place=True)
    # n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    #
    # # layer 3
    # n.conv3_1 = L.Convolution(n.pool2, kernel_size=3, num_output=256, weight_filler=dict(type='msra'),
    #                           bias_filler=dict(type='constant'),
    #                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # n.conv3_1_bn = L.BatchNorm(n.conv3_1, use_global_stats=1, in_place=True)
    # n.relu3_1 = L.ReLU(n.conv3_1, in_place=True)
    # n.conv3_2 = L.Convolution(n.conv3_1, kernel_size=3, num_output=256, weight_filler=dict(type='msra'),
    #                           bias_filler=dict(type='constant'),
    #                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # n.conv3_2_bn = L.BatchNorm(n.conv3_2, use_global_stats=1, in_place=True)
    # n.relu2_2 = L.ReLU(n.conv3_2, in_place=True)
    # n.pool2 = L.Pooling(n.conv2_2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    return n.to_proto()

with open(caffe_root + 'models/'+ 'SegNet/segNet_train.prototxt', 'w') as f:
    f.write(str(segnet()))
#segnet()