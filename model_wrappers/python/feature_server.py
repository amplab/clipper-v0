#!/usr/bin/env python

from __future__ import print_function, absolute_import

import findspark
# findspark.init('/crankshaw-local/spark-1.6.0-bin-hadoop2.4')
findspark.init()

import pyspark
import sys
import time
import os
import numpy
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.tree import RandomForestModel

import argparse
import socket
import random
import capnp
import os
from numpy.random import exponential as expn
from sklearn.externals import joblib
# print(joblib.__version__)
import numpy as np
# from collections import OrderedDict
from sklearn import linear_model as lm
import sklearn.svm as svm
capnp.remove_import_hook()
feature_capnp = capnp.load(os.path.abspath('../../clipper_server/schema/feature.capnp'))
from sample_feature import TestFeature

EXPN_SCALE_PARAM = 5.0

# import graphlab as gl 


# def load_gl_model(local_path):
#     return gl.load_model(local_path)

# import caffe

def load_scikit_model(pickle_path):
    # pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
    feature = joblib.load(pickle_path)
    name = os.path.basename(pickle_path).strip(".pkl")
    return (name, feature)


class CaffeFeatureImpl(feature_capnp.Feature.Server):
    """
        Returns the probabilities of each label after
        doing DNN Features -> Multiclass linear model.
        For now for timing purposes, just compute the features
        then return 1.
    """
    def __init__(self, path):

        # caffe_root = '/crankshaw-local/caffe/'
        # caffe.set_mode_cpu()
        # net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt',
        #                 caffe_root + 'models/bvlc_googlenet/bvlc_reference_caffenet.caffemodel',
        #                 caffe.TEST)
        #
        # # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        # transformer.set_transpose('data', (2,0,1))
        # transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
        # transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        # transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        # net.blobs['data'].reshape(1,3,227,227)
        # net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'))
        # out = net.forward()
        #
        #
        # self.model = load_gl_model(path)
        print("started Dato DNN")


    def computeFeature(self, inp, _context, **kwargs):
        pass



class ScikitFeatureImpl(feature_capnp.Feature.Server):
    

    def __init__(self, path):
        self.name, self.model = load_scikit_model(path)
        self.path = path
        print("started sklearn\n")


    # def load_feature_functions(self):
    #     feature_objects = []
    #     feature_names = [line.strip() for line in open(self.model_path, 'r')]
    #     for lf in feature_names:
    #         pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
    #         feature = joblib.load(pickle_loc) 
    #         feature_objects.append(feature)
    #     return (feature_objects, feature_names)

    def computeFeature(self, inp, _context, **kwargs):
        # print(_context.params)
        # s = name
        start = datetime.datetime.now()
        pred = self.model.predict(np.array(inp).reshape(1, -1))[0]
        # time.sleep(expn(EXPN_SCALE_PARAM) / 1000.0)
        end = datetime.datetime.now()
        print("%s: %f ms\n" % (self.path, (end-start).total_seconds() * 1000))
        # pred = 0.2
        # print("SKLEARN: model predicted: %f" % pred)
        return float(pred)

class PySparkFeatureImpl(feature_capnp.Feature.Server):
    # TODO: find out how to stop spark context
    def __init__(self, path):

        conf = SparkConf() \
            .setAppName("crankshaw-pyspark") \
            .set("spark.executor.memory", "2g") \
            .set("spark.kryoserializer.buffer.mb", "128") \
            .set("master", "local")
        sc = SparkContext(conf=conf, batchSize=10)
        self.model = LogisticRegressionModel.load(sc, path)
        # self.model = RandomForestModel.load(sc, path)
        self.path = path
        # path = '/Users/crankshaw/model-serving/tugboat/feature_servers/python/spark_model'
        # self.name, self.model = load_pyspark_model(path)

        print("started spark")

    def computeFeature(self, inp, _context, **kwargs):
        # print(_context.params)
        # s = name
        # print(type(self.model))
        start = datetime.datetime.now()
        x = [float(v)/255.0 for v in inp]
        # pred = 1.1
        pred = self.model.predict(x)
        # time.sleep(expn(EXPN_SCALE_PARAM) / 1000.0)
        # print("PYSPARK: model predicted: %f" % pred)
        end = datetime.datetime.now()
        print("%s: %f ms\n" % (self.path, (end-start).total_seconds() * 1000))
        return float(pred)

    # def load_pyspark_model(self, sc, path):
    #     # pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
    #
    #     sameModel = LogisticRegressionModel.load(sc, path)
    #     feature = joblib.load(pickle_path)
    #     name = os.path.basename(pickle_path).strip(".pkl")
    #     return (name, feature)

def parse_args():
    parser = argparse.ArgumentParser(usage='''Runs the server bound to the\
given address/port ADDRESS may be '*' to bind to all local addresses.\
:PORT may be omitted to choose a port automatically. ''')

    parser.add_argument("address", type=str, help="ADDRESS[:PORT]")
    parser.add_argument("framework", type=str, help="spark|sklearn")
    parser.add_argument("modelpath", help="full path to pickled model file")


    return parser.parse_args()


def main():
    args = parse_args()
    address = args.address
    model_path = args.modelpath
    # print(model_path)
    if args.framework == "spark":
        server = capnp.TwoPartyServer(address, bootstrap=PySparkFeatureImpl(model_path))
    elif args.framework == "sklearn":
        server = capnp.TwoPartyServer(address, bootstrap=ScikitFeatureImpl(model_path))
    else:
        print("%s is unsupported framework" % args.system)
        return
    server.run_forever()

if __name__ == '__main__':
    main()


