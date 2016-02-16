#!/usr/bin/env python

from __future__ import print_function, absolute_import

import findspark
findspark.init()

import pyspark
import sys
import time
import os
import numpy
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.tree import RandomForestModel

import argparse
import socket
import random
import capnp
import os
from sklearn.externals import joblib
print(joblib.__version__)
import numpy as np
# from collections import OrderedDict
from sklearn import linear_model as lm
import sklearn.svm as svm
capnp.remove_import_hook()
feature_capnp = capnp.load(os.path.abspath('../../clipper_server/schema/feature.capnp'))
from sample_feature import TestFeature
# import graphlab as gl 


# def load_gl_model(local_path):
#     return gl.load_model(local_path)


def load_scikit_model(pickle_path):
    # pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
    feature = joblib.load(pickle_path)
    name = os.path.basename(pickle_path).strip(".pkl")
    return (name, feature)


class DNNFeatureImpl(feature_capnp.Feature.Server):
    """
        Returns the probabilities of each label after
        doing DNN Features -> Multiclass linear model.
        For now for timing purposes, just compute the features
        then return 1.
    """
    def __init__(self, path):
        self.model = load_gl_model(path)
        print("started Dato DNN")


    def computeFeature(self, inp, _context, **kwargs):
        pass



class ScikitFeatureImpl(feature_capnp.Feature.Server):
    

    def __init__(self, path):
        self.name, self.model = load_scikit_model(path)
        print("started sklearn")


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
        print(type(self.model))
        pred = self.model.predict(np.array(inp).reshape(1, -1))[0]
        print("SKLEARN: model predicted: %f" % pred)
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
        # self.model = LogisticRegressionModel.load(sc, path)
        self.model = RandomForestModel.load(sc, path)
        # path = '/Users/crankshaw/model-serving/tugboat/feature_servers/python/spark_model'
        # self.name, self.model = load_pyspark_model(path)

        print("started spark")

    def computeFeature(self, inp, _context, **kwargs):
        # print(_context.params)
        # s = name
        # print(type(self.model))
        x = [float(v)/255.0 for v in inp]
        pred = self.model.predict(x)
        print("PYSPARK: model predicted: %f" % pred)
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


