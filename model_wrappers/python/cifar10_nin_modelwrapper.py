from __future__ import print_function
import caffe
import numpy as np
import skimage
import skimage.io as skio
import sys
import os
import rpc

from skimage.transform import resize

class Cifar10NinModelWrapper(rpc.ModelWrapperBase):

    def __init__(self, batch_size, image_size, num_classes,
                 model_path, weights_path):
        # Set GPU
        caffe.set_mode_gpu()
        caffe.set_device(0)

        self.batch_size = batch_size
        self.image_size = image_size
        self.model_path = model_path
        self.weights_path = weights_path
        self.net = caffe.Net(model_path, 1, weights=weights_path)
        # Reshape to get desired batch size
        self.net.blobs['data'].reshape(batch_size, 3, image_size, image_size)
        self.net.reshape()

    def predict_ints(self, inputs):
        inputs = map(lambda x: skimage.img_as_float(x.astype(np.ubyte)), inputs)
        return self.predict_floats(inputs)

    def predict_floats(self, inputs):
        number_of_inputs = len(inputs)
        inputs = np.array(inputs).astype(np.float32)
        inputs = inputs.reshape(
            (number_of_inputs, self.image_size, self.image_size, 3))
        if len(inputs) < self.batch_size:
            top_img = inputs[0].reshape((1,) + inputs[0].shape)
            pad_imgs = np.repeat(top_img, self.batch_size - len(inputs), axis=0)
            inputs = np.concatenate((inputs, pad_imgs), axis=0)
        inputs = np.transpose(inputs, (0, 3, 1, 2))
        self.net.blobs['data'].data[...] = inputs
        preds = map(lambda x: np.argmax(x), self.net.forward()['loss'])
        preds = np.array(preds).astype(np.float64)
        return preds[:number_of_inputs]


if __name__=='__main__':
    model_path = os.environ["CLIPPER_MODEL_PATH"]
    weights_path = os.environ["CLIPPER_WEIGHTS_PATH"]
    print(model_path, file=sys.stderr)
    print(weights_path, file=sys.stderr)
    model = Cifar10NinModelWrapper(4, 32, 10, model_path, weights_path)
    rpc.start(model, 6001)
