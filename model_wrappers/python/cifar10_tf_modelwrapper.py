from __future__ import print_function
import faster_rpc
import numpy as np
import skimage
import skimage.io as skio
import sys
import os
import tensorflow as tf

from skimage.transform import resize
from cifar10_net import cifar10

class Cifar10TfModelWrapper(faster_rpc.ModelWrapperBase):

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
        self.predict_floats(np.zeros(input_shape))

    def predict_ints(self, inputs):
        inputs = map(lambda x: skimage.img_as_float(x.astype(np.ubyte)), inputs)
        return self.predict_floats(inputs)

    def predict_floats(self, inputs):
        num_inputs = len(inputs)
        num_batches = num_inputs / self.batch_size
        inputs = np.array(inputs)
        padded_inputs = inputs.reshape((len(inputs), self.image_size, self.image_size, 3))
        # find how much padding we need
        if num_batches * self.batch_size < num_inputs:
            padding = self.batch_size - num_inputs % self.batch_size
            top_img = padded_inputs[0].reshape((1,) + padded_inputs[0].shape)
            pad_imgs = np.repeat(top_img, padding, axis=0)
            padded_inputs = np.concatenate((padded_inputs, pad_imgs), axis=0)
            assert len(padded_inputs) / self.batch_size == num_batches + 1

        assert len(padded_inputs) % self.batch_size == 0
        num_batches = len(padded_inputs) / self.batch_size
        preds = []
        for b in range(num_batches):
            with tf.variable_scope("", reuse=self.reuse_scope) as scope:
                top_1 = self.sess.run([self.top_1_op],
                                      feed_dict={self.jpegs: padded_inputs[b*self.batch_size: (b+1)*self.batch_size]})
                cur_preds = top_1[0].indices.flatten()
                preds.extend(cur_preds.astype(np.float64))
        # Mark reuse_scope as True for future reuse of inception variables
        self.reuse_scope = True
        preds = np.array(preds[:num_inputs])
        print("Predicted: %s" % preds)
        return preds


if __name__=='__main__':
    model_path = os.environ["CLIPPER_MODEL_PATH"]
    print(model_path, file=sys.stderr)
    model = Cifar10TfModelWrapper(32, 32, 10, model_path)
    faster_rpc.start(model, 6001)
