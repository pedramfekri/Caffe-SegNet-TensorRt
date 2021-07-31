caffe_root = '/home/pedram/caffe-segnet-cudnn5/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
import matplotlib.pyplot as plt
import numpy as np
import time

caffe.set_mode_cpu()
# caffe.set_device(0)
# caffe.set_mode_gpu()

solver = caffe.SGDSolver(caffe_root + 'models/' + 'SegNet-Tutorial/Models/segnet_solver.prototxt')

# # each output is (batch size, feature dim, spatial dim)
print([(k, v.data.shape) for k, v in solver.net.blobs.items()])
#


# just print the weight sizes (we'll omit the biases)
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])


solver.net.forward()  # train net
a = solver.test_nets[0].forward()  # test net (there can be more than one)
print(a)