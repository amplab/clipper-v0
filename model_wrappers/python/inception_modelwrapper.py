from __future__ import print_function
import numpy as np
import skimage.io as skio
import sys
import os
import rpc
import tensorflow as tf

from inception import inception_model as inception
from skimage.transform import resize

# WORKING_DIR = 'inception_nn'
# SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
# METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')
DAISY_PATH = '../../clipper_server/models/inception/raw-data/train/daisy/'
ROSES_PATH = '../../clipper_server/models/inception/raw-data/train/roses/'

def preprocess_image(image_buffer, image_size):
    """Preprocess JPEG encoded bytes to 3D float Tensor."""
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    # # image = tf.image.decode_jpeg(image_buffer, channels=3)
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
                                     [image_size, image_size],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image


class InceptionModelWrapper(rpc.ModelWrapperBase):

    def __init__(self, image_size, num_classes, checkpoint_path):
        self.image_size = image_size
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.reuse_scope = False

    def predict_ints(self, inputs):
        return np.ones(len(inputs))

    def predict_floats(self, inputs):
        inputs = inputs.reshape(
            (len(inputs), self.image_size, self.image_size, 3))
        list_of_inputs = \
            map(lambda x: preprocess_image(x, self.image_size), inputs)
        list_of_inputs = tf.pack(list_of_inputs)
        with tf.variable_scope("", reuse=self.reuse_scope) as scope:
            with tf.Session() as sess:
                logits, _ = inception.inference(list_of_inputs, self.num_classes)
                # Restore variables from training checkpoint.
                saver = tf.train.Saver()
                saver.restore(sess, self.checkpoint_path)
                top_1_op = tf.nn.top_k(logits, 1)
                top_5_op = tf.nn.top_k(logits, 5)
                top_1, top_5 = sess.run([top_1_op, top_5_op])
        # Mark reuse_scope as True for future reuse of inception variables
        self.reuse_scope = True
        return top_1.indices.flatten()



if __name__=='__main__':
    imgs = map(lambda x: skio.imread(os.path.join(DAISY_PATH, x)),
                                     os.listdir(DAISY_PATH)[-10:])
    imgs = map(lambda x: resize(x, (299, 299)).flatten(), imgs)
    imgs = np.array(imgs)
    imgs2 = map(lambda x: skio.imread(os.path.join(ROSES_PATH, x)),
                                      os.listdir(ROSES_PATH)[-10:])
    imgs2 = map(lambda x: resize(x, (299, 299)).flatten(), imgs2)
    imgs2 = np.array(imgs2)
    os.environ["CLIPPER_MODEL_PATH"] = "/Users/giuliozhou/Research/RISE/clipper/model_wrappers/python/inception_nn/inception-v3-model/model.ckpt-157585"
    model_path = os.environ["CLIPPER_MODEL_PATH"]
    print(model_path, file=sys.stderr)
    model = InceptionModelWrapper(299, 1001, model_path)
    print(model.predict_floats(imgs))
    print(model.predict_floats(imgs2))
    # rpc.start(model, 6001)
    # with tf.variable_scope("") as scope:
    #     print(model.predict_floats(imgs))
    #     scope.reuse_variables()
    #     print(model.predict_floats(imgs))
