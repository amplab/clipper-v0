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
import datetime

from skimage.transform import resize

class ConvNetMnistModelWrapper(faster_rpc.ModelWrapperBase):

    def __init__(self, batch_size, image_size, num_classes, checkpoint_path):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.reuse_scope = False
        self.sess = tf.Session()
        # Perform initialization of TF inception model
        input_shape = (batch_size, image_size, image_size)
        self.jpegs = tf.placeholder(tf.float32, shape=input_shape)
        self.dropout = 0.2
        self.keep = tf.placeholder(tf.float32)
        # images = tf.map_fn(self.preprocess_image, self.jpegs, dtype=tf.float32)
        # logits = inception.inference(self.jpegs, num_classes + 1)
        logits = convnet(self.jpegs, self.dropout)
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
        t1 = datetime.datetime.now()
        num_inputs = len(inputs)
        num_batches = num_inputs / self.batch_size
        # if num_batches > 0:
        #   print(num_batches, len(inputs))
        # print(len(inputs))
        inputs = np.array(inputs)
        padded_inputs = inputs.reshape((len(inputs), self.image_size, self.image_size))
        # find how much padding we need
        if num_batches * self.batch_size < num_inputs:
            padding = self.batch_size - num_inputs % self.batch_size
            top_img = padded_inputs[0].reshape((1,) + padded_inputs[0].shape)
            pad_imgs = np.repeat(top_img, padding, axis=0)
            padded_inputs = np.concatenate((padded_inputs, pad_imgs), axis=0)
            assert len(padded_inputs) / self.batch_size == num_batches + 1

        assert len(padded_inputs) % self.batch_size == 0
        num_batches = len(padded_inputs) / self.batch_size
        t2 = datetime.datetime.now()
        preds = []
        for b in range(num_batches):
            with tf.variable_scope("", reuse=self.reuse_scope) as scope:
                top_1 = self.sess.run([self.top_1_op],
                        feed_dict={self.jpegs: padded_inputs[b*self.batch_size: (b+1)*self.batch_size],
                          self.keep: self.dropout})
                cur_preds = top_1[0].indices.flatten()
                preds.extend(cur_preds.astype(np.float64))
        t3 = datetime.datetime.now()
        # Mark reuse_scope as True for future reuse of inception variables
        self.reuse_scope = True
        preds = np.array(preds[:num_inputs])
        # print("Predicted: %s" % preds)
        t4 = datetime.datetime.now()
        # return (preds, (t4-t3).microseconds, (t3-t2).microseconds, (t2-t1).microseconds)
        return preds




    # def predict_floats(self, inputs):
    #     inputs = np.array(inputs)
    #     inputs = inputs.reshape(
    #         (len(inputs), self.image_size, self.image_size))
    #     # Pad the inputs ndarray with the first image if necessary
    #     if len(inputs) < self.batch_size:
    #         top_img = inputs[0].reshape((1,) + inputs[0].shape)
    #         pad_imgs = np.repeat(top_img, self.batch_size - len(inputs), axis=0)
    #         inputs = np.concatenate((inputs, pad_imgs), axis=0)
    #     with tf.variable_scope("", reuse=self.reuse_scope) as scope:
    #         top_1 = self.sess.run([self.top_1_op],
    #                               feed_dict={self.jpegs: inputs,
    #                                          self.keep: self.dropout})
    #     # Mark reuse_scope as True for future reuse of inception variables
    #     self.reuse_scope = True
    #     preds = top_1[0].indices.flatten()
    #     preds = preds.astype(np.float64)
    #     return preds


# Useful functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Model definition
def convnet(inputs, keep):
    # Create the model
    # x = tf.placeholder(tf.float32, [None, 784])
    x = inputs
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    # Do convolution stuff
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # keep_prob = tf.placeholder(tf.float32)
    keep_prob = keep
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # y_ = tf.placeholder(tf.float32, [None, 10])

    return y_conv


def load_digits(digits_location, digits_filename):
    digits_path = os.path.join(digits_location, digits_filename)
    print("Source file:", digits_path)
    df = pd.read_csv(digits_path, sep=",", header=None)
    data = df.values
    print("Number of image files:", len(data))
    y = data[:,0]
    X = data[:,1:]
    return (X, y)

def local_test(model, num_reqs, batch_size):
    dp = "/crankshaw-local/mnist/data"
    (x,y) = load_digits(dp, "test.data")
    total_preds = 0
    setup_times = []
    inference_times = []
    teardown_times = []
    total_call_times = []
    for i in range(0, num_reqs, batch_size):
        cur_batch_start = i % len(x)
        cur_batch_end = (i + batch_size) % len(x)
        ## rollover
        if cur_batch_start >= cur_batch_end:
            cur_batch_start = 0
            cur_batch_end = batch_size
        batch = list(x[cur_batch_start: cur_batch_end])
        t1 = datetime.datetime.now()
        (preds, st, it, tt) = model.predict_floats(batch)
        t2 = datetime.datetime.now()
        setup_times.append(st)
        inference_times.append(it)
        teardown_times.append(tt)
        total_call_times.append((t2-t1).microseconds)
        total_preds += len(preds)

    return (total_preds,
            np.array(setup_times),
            np.array(inference_times),
            np.array(teardown_times),
            total_call_times)


if __name__=='__main__':
    model_path = os.environ["CLIPPER_MODEL_PATH"]
    print(model_path, file=sys.stderr)
    batch_size = int(os.environ["TF_CONV_BATCH_SIZE"])
    model = ConvNetMnistModelWrapper(batch_size, 28, 10, model_path)
    faster_rpc.start(model, 6001)
    # start = datetime.datetime.now()
    # num_reqs = 1000000
    # (num_preds, st, it, tt, tct) = local_test(model, num_reqs, batch_size) # see penalty for going over batch size
    # end = datetime.datetime.now()
    # total_time = (end-start).total_seconds()
    # throughput = float(num_preds) / total_time
    # print("Made %d predictions in %f seconds. Throughput: %f" % (num_preds, total_time, throughput))
    # print("MEANS (microseconds): setup: %f, inference: %f, teardown: %f, call time %f" % (np.mean(st), np.mean(it), np.mean(tt), np.mean(tct)))
