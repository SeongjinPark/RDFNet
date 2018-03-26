import sys
#caffe_root = './caffe-master/' # Change this to the absolute directoy to Caffe
sys.path.insert(0, caffe_root + 'python')

import caffe
import matplotlib.pyplot as plt
import numpy as np

caffe.set_mode_gpu()

scale = 1.0
model = 'NYU-101'
proto = './' + model + '/res_test'
caffemodel ='./' + model + '/Trained_model/res101101_dim1_scconv1_640480_with_1CRP_iter_240000.caffemodel'

net=caffe.Net(proto + str(scale) +'.prototxt',caffemodel,caffe.TEST)

images = np.loadtxt('./data/test.txt', dtype=str)
for k in enumerate(images):
    net.forward()
    prediction = np.argmax(net.blobs['finalscore'].data[0].transpose([1,2,0]),axis=2)
    plt.imshow(prediction)
    plt.show()
