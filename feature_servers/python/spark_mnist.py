#!/usr/bin/env python

'''
Run using:
[SparkDir]/spark/bin/spark-submit --driver-memory 2g mnist.py
'''

from __future__ import absolute_import

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


# from cvm.svm import SVC
# from cvm.kreg import KernelLogisticRegression

# Set parameters here:
NMAX = 10000
GAMMA = 0.02
C = 1.0

def objective(x):
    # prediction objective
    if x is 1:
        return 1
    else:
        return 0
    # return x

def parseData(line, obj):
    fields = line.strip().split(',')
    return LabeledPoint(obj(int(fields[0])), [float(v)/255.0 for v in fields[1:]])



if __name__ == "__main__":
    # if (len(sys.argv) != 1):
    #     print "Usage: [SPARKDIR]/bin/spark-submit --driver-memory 2g " + \
    #         "mnist.py"
    #     sys.exit(1)

    # set up environment
    conf = SparkConf() \
        .setAppName("crankshaw-pyspark") \
        .set("spark.executor.memory", "2g") \
        .set("spark.kryoserializer.buffer.mb", "128") \
        .set("master", "local")
    sc = SparkContext(conf=conf, batchSize=10)

    print 'Parsing data'
    time_start = time.time()
    trainRDD = sc.textFile('/Users/crankshaw/model-serving/data/mnist_data/train-mnist-dense-with-labels.data').map(lambda line: parseData(line, objective)).cache()
    testRDD = sc.textFile('/Users/crankshaw/model-serving/data/mnist_data/test-mnist-dense-with-labels.data').map(lambda line: parseData(line, objective)).cache()

    print 'Fitting model'

    # lrm = LogisticRegressionWithSGD.train(trainRDD, iterations=10)
    # lrm.predict(array([0.0, 1.0]))
    # lrm.predict(array([1.0, 0.0]))
    # lrm.predict(SparseVector(2, {1: 1.0}))
    # lrm.predict(SparseVector(2, {0: 1.0}))
    # lrm.predict(testRDD)

    path = '/Users/crankshaw/model-serving/tugboat/feature_servers/python/spark_model'
    # lrm.save(sc, path)
    sameModel = LogisticRegressionModel.load(sc, path)
    sameModel.predict(testRDD)
    # # sameModel.predict(array([0.0, 1.0]))
    # # sameModel.predict(SparseVector(2, {0: 1.0}))
    #
    # # model = SVC(gamma=GAMMA, C=C, nmax=NMAX)
    # # model = KernelLogisticRegression(gamma=0.01, C=2.0, nmax=3000)
    #
    # # model.train(trainRDD)
    # print("Time: {:2.2f}".format(time.time() - time_start))
    #
    # print 'Predicting outcomes training set'
    # labelsAndPredsTrain = trainRDD.map(lambda p: (p.label, sameModel.predict(p.features)))
    # trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(trainRDD.count())
    # print("Training Error = " + str(trainErr))
    # print("Time: {:2.2f}".format(time.time() - time_start))
    #
    print 'Predicting outcomes test set'
    labelsAndPredsTest = testRDD.map(lambda p: (p.label, sameModel.predict(p.features)))
    testErr = labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(testRDD.count())
    print("Test Error = " + str(testErr))
    print("Time: {:2.2f}".format(time.time() - time_start))

    # clean up
    sc.stop()
