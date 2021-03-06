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

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


with open(caffe_root + 'examples/'+ 'mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet(caffe_root +'examples/'+'mnist/mnist_train_lmdb', 64)))

with open(caffe_root +'examples/'+ 'mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet(caffe_root +'examples/'+'mnist/mnist_test_lmdb', 100)))


# ### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver(caffe_root +'examples/'+'mnist/lenet_auto_solver.prototxt')
#
# # each output is (batch size, feature dim, spatial dim)
print([(k, v.data.shape) for k, v in solver.net.blobs.items()])
#


# just print the weight sizes (we'll omit the biases)
print([(k, v[0].data.shape) for k, v in solver.net.params.items()])


solver.net.forward()  # train net
a = solver.test_nets[0].forward()  # test net (there can be more than one)
print(a)

# we use a little trick to tile the first eight images
plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
plt.show()
print('train labels:', solver.net.blobs['label'].data[:8])

plt.imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
plt.show()
print('test labels:', solver.test_nets[0].blobs['label'].data[:8])

# Both train and test nets seem to be loading data, and to have correct labels.
#
#     Let's take one step of (minibatch) SGD and see what happens.
solver.step(1)

plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5)
       .transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray')
plt.show()


# now let's train the network
niter = 200
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
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4


print('--%s seconds--' % (time.time()-tik))
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.show()