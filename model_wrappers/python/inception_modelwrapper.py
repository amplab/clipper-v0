from __future__ import print_function
import numpy as np
import skimage
import skimage.io as skio
import sys
import os
# import rpc
import faster_rpc
import tensorflow as tf
import pandas as pd

from inception import inception_model as inception
from skimage.transform import resize

class InceptionModelWrapper(faster_rpc.ModelWrapperBase):

    def __init__(self, batch_size, image_size, num_classes, checkpoint_path):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.reuse_scope = False
        self.sess = tf.Session()
        # Perform initialization of TF inception model
        input_shape = (batch_size, image_size, image_size, 3)
        self.jpegs = tf.placeholder(tf.float32, shape=input_shape)
        images = tf.map_fn(self.preprocess_image, self.jpegs, dtype=tf.float32)
        logits, _ = inception.inference(images, num_classes + 1)
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

        # num_inputs = len(inputs)
        # inputs = np.array(inputs)
        # inputs = inputs.reshape(
        #     (len(inputs), self.image_size, self.image_size, 3))
        # Pad the inputs ndarray with the first image if necessary
        # if len(inputs) < self.batch_size:
        #     top_img = inputs[0].reshape((1,) + inputs[0].shape)
        #     pad_imgs = np.repeat(top_img, self.batch_size - len(inputs), axis=0)
        #     inputs = np.concatenate((inputs, pad_imgs), axis=0)
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

    def preprocess_image(self, image_buffer):
        """Preprocess JPEG encoded bytes to 3D float Tensor."""
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.convert_image_dtype(image_buffer, dtype=tf.float32)
        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)
        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image,
                                         [self.image_size, self.image_size],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        # Finally, rescale to [-1,1] instead of [0, 1)
        image = tf.sub(image, 0.5)
        image = tf.mul(image, 2.0)
        return image







if __name__=='__main__':
    model_path = os.environ["CLIPPER_MODEL_PATH"]
    print(model_path, file=sys.stderr)
    model = InceptionModelWrapper(3, 299, 1000, model_path)
    faster_rpc.start(model, 6001)
