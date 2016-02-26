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
from pyspark.mllib.classification import SVMModel, SVMWithSGD
from pyspark.mllib.tree import RandomForest



# from cvm.svm import SVC
# from cvm.kreg import KernelLogisticRegression

# Set parameters here:
NMAX = 10000
GAMMA = 0.02
C = 1.0

def objective(x, pos_label):
    # prediction objective
    if x == pos_label:
        return 1
    else:
        return 0
    # return x

def parseData(line, obj, pos_label):
    fields = line.strip().split(',')
    return LabeledPoint(obj(int(fields[0]), pos_label), [float(v)/255.0 for v in fields[1:]])

def train_logistic_regression(pos_label):
    conf = SparkConf() \
        .setAppName("crankshaw-pyspark") \
        .set("spark.executor.memory", "2g") \
        .set("spark.kryoserializer.buffer.mb", "128") \
        .set("master", "local")
    sc = SparkContext(conf=conf, batchSize=10)
    print 'Parsing data'
    trainRDD = sc.textFile("/crankshaw-local/mnist/data/train_norm.data").map(lambda line: parseData(line, objective, pos_label)).cache()
    # testRDD = sc.textFile("/crankshaw-local/mnist/data/test.data").map(lambda line: parseData(line, objective)).cache()

    print 'Fitting model'

    lrm = LogisticRegressionWithSGD.train(trainRDD, iterations=100)

    path = 'spark_models/lg_predict_%d' % pos_label
    lrm.save(sc, path)

    # sameModel = LogisticRegressionModel.load(sc, path)
    # sameModel.predict(testRDD)
    # # # sameModel.predict(array([0.0, 1.0]))
    # # # sameModel.predict(SparseVector(2, {0: 1.0}))
    # #
    # # # model = SVC(gamma=GAMMA, C=C, nmax=NMAX)
    # # # model = KernelLogisticRegression(gamma=0.01, C=2.0, nmax=3000)
    # #
    # # # model.train(trainRDD)
    # # print("Time: {:2.2f}".format(time.time() - time_start))
    # #
    # # print 'Predicting outcomes training set'
    # # labelsAndPredsTrain = trainRDD.map(lambda p: (p.label, sameModel.predict(p.features)))
    # # trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(trainRDD.count())
    # # print("Training Error = " + str(trainErr))
    # # print("Time: {:2.2f}".format(time.time() - time_start))
    # #
    # print 'Predicting outcomes test set'
    # labelsAndPredsTest = testRDD.map(lambda p: (p.label, sameModel.predict(p.features)))
    # testErr = labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(testRDD.count())
    # print("Test Error = " + str(testErr))
    # print("Time: {:2.2f}".format(time.time() - time_start))

    # clean up
    sc.stop()

def train_svm(pos_label):
    conf = SparkConf() \
        .setAppName("crankshaw-pyspark") \
        .set("spark.executor.memory", "2g") \
        .set("spark.kryoserializer.buffer.mb", "128") \
        .set("master", "local")
    sc = SparkContext(conf=conf, batchSize=10)
    print 'Parsing data'
    trainRDD = sc.textFile("/crankshaw-local/mnist/data/train_norm.data").map(lambda line: parseData(line, objective, pos_label)).cache()
    # testRDD = sc.textFile("/crankshaw-local/mnist/data/test.data").map(lambda line: parseData(line, objective)).cache()

    print 'Fitting model'

    svm = SVMWithSGD.train(trainRDD)

    path = 'spark_models/svm_predict_%d' % pos_label
    svm.save(sc, path)
    sc.stop()

def train_random_forest(num_trees):
    conf = SparkConf() \
        .setAppName("crankshaw-pyspark") \
        .set("spark.executor.memory", "2g") \
        .set("spark.kryoserializer.buffer.mb", "128") \
        .set("master", "local")
    sc = SparkContext(conf=conf, batchSize=10)

    print 'Parsing data'
    time_start = time.time()
    data_path = os.path.expanduser("~/mnist/data/train.data")
    trainRDD = sc.textFile(data_path).map(lambda line: parseData(line, objective)).cache()
    # testRDD = sc.textFile('/Users/crankshaw/model-serving/data/mnist_data/test-mnist-dense-with-labels.data').map(lambda line: parseData(line, objective)).cache()

    print 'Fitting model'
    rf = RandomForest.trainClassifier(trainRDD, 2, {}, num_trees)
    rf.save(sc, "spark_models/%drf_pred_1" % num_trees)
    # lrm.save(sc, path)

    sc.stop()


if __name__ == "__main__":

    for i in range(10):
        print "training model to predict %d" % (i + 1)
        train_logistic_regression(i + 1)
    # train_random_forest(50)
    # train_random_forest(100)
    # train_random_forest(500)
    # if (len(sys.argv) != 1):
    #     print "Usage: [SPARKDIR]/bin/spark-submit --driver-memory 2g " + \
    #         "mnist.py"
    #     sys.exit(1)

    # set up environment
