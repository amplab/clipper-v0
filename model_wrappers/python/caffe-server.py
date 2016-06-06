#!/usr/bin/env python

from __future__ import print_function, absolute_import

import sys
import time
import os
import numpy
import datetime
import cPickle

import argparse
import socket
import random
import capnp
import numpy as np
import pandas as pd
import caffe
import datetime
import logging
import exifutil
capnp.remove_import_hook()
feature_capnp = capnp.load(os.path.abspath('../../clipper_server/schema/feature.capnp'))
# from sample_feature import TestFeature


# REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
REPO_DIRNAME = os.path.expanduser("/crankshaw-local/caffe")


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_googlenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_googlenet/bvlc_googlenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim):
        logging.info('Loading net and associated files...')
        caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify_image(self, image):
        try:
            scores = self.net.predict([image], oversample=True).flatten()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]


            # Compute expected information gain
            expected_infogain = np.dot(
                self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']

            # sort the scores
            infogain_sort = expected_infogain.argsort()[::-1]
            bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
                          for v in infogain_sort[:5]]
            print('bet result: %s' % str(bet_result))
            return bet_result

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')







class CaffeFeatureImpl(feature_capnp.Feature.Server):
    

    def __init__(self):
        self.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
        self.clf.net.forward()
        print("started caffe\n")
        image_file = "/crankshaw-local/caffe/examples/images/cat.jpg"
        self.test_image = exifutil.open_oriented_im(image_file)



    def computeFeature(self, inp, _context, **kwargs):

        start = datetime.datetime.now()
        # image = caffe.io.load_image(string_buffer)
        result = self.clf.classify_image(self.test_image)
        print(result)
        pred = 0.2
        end = datetime.datetime.now()
        print("latency: %f ms" % ((end-start).total_seconds() * 1000))
        # print("SKLEARN: model predicted: %f" % pred)
        return float(pred)


def parse_args():
    parser = argparse.ArgumentParser(usage='''Runs the server bound to the\
given address/port ADDRESS may be '*' to bind to all local addresses.\
:PORT may be omitted to choose a port automatically. ''')

    parser.add_argument("address", type=str, help="ADDRESS[:PORT]")
    # parser.add_argument("framework", type=str, help="spark|sklearn")
    # parser.add_argument("modelpath", help="full path to pickled model file")


    return parser.parse_args()


def main():
    args = parse_args()
    address = args.address
    # model_path = args.modelpath
    # print(model_path)
    server = capnp.TwoPartyServer(address, bootstrap=CaffeFeatureImpl())
    server.run_forever()

if __name__ == '__main__':
    main()


