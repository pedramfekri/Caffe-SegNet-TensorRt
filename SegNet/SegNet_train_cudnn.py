caffe_root = '/home/pedram/caffe-segnet-cudnn-v/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
import matplotlib.pyplot as plt
import numpy as np
import time

# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(caffe_root + 'models/' + 'SegNet-Tutorial/Models/segnet_basic_solver.prototxt')

# # each output is (batch size, feature dim, spatial dim)
print([(k, v.data.shape) for k, v in solver.net.blobs.items()])
#


# just print the weight sizes (we'll omit the biases)
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])


#solver.net.forward()  # train net
#a = solver.test_nets[0].forward()  # test net (there can be more than one)
#print(a)

# now let's train the network
niter = 20000
test_interval = 25
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
output = np.zeros((niter, 8, 10))

tik = time.time()
# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    # train_loss[it] = solver.net.blobs['accuracy'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    # solver.test_nets[0].forward(start='conv1_1')
    # output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    # if it % test_interval == 0:
    #     print 'Iteration', it, 'testing...'
    #     correct = 0
    #     for test_it in range(100):
    #         solver.test_nets[0].forward()
    #         correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
    #                        == solver.test_nets[0].blobs['label'].data)
    #     test_acc[it // test_interval] = correct / 1e4
