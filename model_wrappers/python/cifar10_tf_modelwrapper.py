from __future__ import print_function
import numpy as np
import skimage
import skimage.io as skio
import sys
import os
import rpc
import tensorflow as tf

from skimage.transform import resize
from cifar10_net import cifar10

class Cifar10TfModelWrapper(rpc.ModelWrapperBase):

    def __init__(self, batch_size, image_size, num_classes, checkpoint_path):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.reuse_scope = False
        self.sess = tf.Session()
        # Perform initialization of TF cifar10 model
        input_shape = (batch_size, image_size, image_size, 3)
        self.jpegs = tf.placeholder(tf.float32, shape=input_shape)
        # Get logits from inference
        logits = cifar10.inference(self.jpegs, batch_size)
        self.top_1_op = tf.nn.top_k(logits, 1)
        # Restore variables from training checkpoint
        saver = tf.train.Saver()
        saver.restore(self.sess, self.checkpoint_path)
        # Do a sample inference to preload everything
        # self.predict_floats(np.zeros(input_shape))

    def predict_ints(self, inputs):
        inputs = map(lambda x: skimage.img_as_float(x.astype(np.ubyte)), inputs)
        return self.predict_floats(inputs)

    def predict_floats(self, inputs):
        inputs = np.array(inputs)
        inputs = inputs.reshape(
            (len(inputs), self.image_size, self.image_size))
        # Pad the inputs ndarray with the first image if necessary
        if len(inputs) < self.batch_size:
            top_img = inputs[0].reshape((1,) + inputs[0].shape)
            pad_imgs = np.repeat(top_img, self.batch_size - len(inputs), axis=0)
            inputs = np.concatenate((inputs, pad_imgs), axis=0)
        with tf.variable_scope("", reuse=self.reuse_scope) as scope:
            top_1 = self.sess.run([self.top_1_op],
                                  feed_dict={self.jpegs: inputs,
                                             self.keep: self.dropout})
        # Mark reuse_scope as True for future reuse of inception variables
        self.reuse_scope = True
        preds = top_1[0].indices.flatten()
        preds = preds.astype(np.float64)
        return preds


if __name__=='__main__':
    model_path = os.environ["CLIPPER_MODEL_PATH"]
    print(model_path, file=sys.stderr)
    model = Cifar10TfModelWrapper(32, 32, 10, model_path)
    rpc.start(model, 6001)
