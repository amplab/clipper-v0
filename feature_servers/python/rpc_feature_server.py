from __future__ import print_function
import array
import struct
import SocketServer
import numpy as np
import time
import datetime
import sys
import os
import findspark
# findspark.init('/crankshaw-local/spark-1.6.0-bin-hadoop2.4')
findspark.init()

import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithSGD
from pyspark.mllib.classification import SVMModel, SVMWithSGD
from pyspark.mllib.tree import RandomForestModel
from numpy.random import exponential as expn
from sklearn.externals import joblib
import argparse
from sklearn import linear_model as lm
import sklearn.svm as svm
from sample_feature import TestFeature

VECTOR_LENGTH = 784
BYTES_IN_DOUBLE = 8

class FeatureTCPHandler(SocketServer.BaseRequestHandler):

    # def __init__(self, model):
    #     self.model = model

    def handle(self):
        print("HANDLING CONNECTION")

        while True:
            # print("ITER")
            # header format: 2 byte unsigned short indicating batch size
            header_bytes = 2
            data = ""
            # self.request.settimeout(0.5)
            # self.request.setblocking(1)
            # wait for header
            while len(data) < header_bytes:
                data += self.request.recv(4096)
                # print("waiting for data")
                # time.sleep(0.5)
            header, data = (data[:header_bytes], data[header_bytes:])
            batch_size = struct.unpack("<H", header)[0]
            # print("BATCH SIZE: %d" % batch_size)
            total_bytes = batch_size * VECTOR_LENGTH * BYTES_IN_DOUBLE
            # print("total bytes expected: %d" % total_bytes)
            while len(data) < total_bytes:
                data += self.request.recv(4096)
                # print("data bytes received: %d" % len(data))

            assert len(data) == total_bytes
            # print("Full batch received")

            batch_features = array.array('d', bytes(data))
            assert len(batch_features) == batch_size * VECTOR_LENGTH
            sep_feature_vecs = []
            for i in range(batch_size):
                fv_start = i * VECTOR_LENGTH
                fv_end = (i+1) * VECTOR_LENGTH
                # print("%d: (%d, %d)" % (i,fv_start,fv_end))
                fv = batch_features[fv_start: fv_end]
                sep_feature_vecs.append(fv)
                # print(fv)

            preds = self.server.model.compute_features(sep_feature_vecs)
            assert len(preds) == batch_size
            assert preds.dtype == np.dtype('float64')
            self.request.sendall(preds.tobytes())

class SparkLRServer:
    def __init__(self, path):
        conf = SparkConf() \
            .setAppName("crankshaw-pyspark") \
            .set("spark.executor.memory", "2g") \
            .set("spark.kryoserializer.buffer.mb", "128") \
            .set("master", "local")
        sc = SparkContext(conf=conf, batchSize=10)
        self.model = LogisticRegressionModel.load(sc, path)
        self.path = path
        print("started spark")

    def compute_features(self, inputs):
        start = datetime.datetime.now()
        preds = []
        for i in inputs:
            # TODO is making an RDD faster? probably not
            preds.append(float(self.model.predict(i)))
        end = datetime.datetime.now()
        print("%s: %f ms\n" % (self.path, (end-start).total_seconds() * 1000))
        preds = np.array(preds)
        assert preds.dtype == np.dtype('float64')
        return np.array(preds)

class SparkSVMServer:
    def __init__(self, path):
        conf = SparkConf() \
            .setAppName("crankshaw-pyspark") \
            .set("spark.executor.memory", "2g") \
            .set("spark.kryoserializer.buffer.mb", "128") \
            .set("master", "local")
        sc = SparkContext(conf=conf, batchSize=10)
        self.model = SVMModel.load(sc, path)
        self.path = path
        print("started spark")

    def compute_features(self, inputs):
        start = datetime.datetime.now()
        preds = []
        for i in inputs:
            # TODO is making an RDD faster? probably not
            preds.append(float(self.model.predict(i)))
        end = datetime.datetime.now()
        print("%s: %f ms\n" % (self.path, (end-start).total_seconds() * 1000))
        preds = np.array(preds)
        # assert preds.dtype == np.dtype('float64')
        return np.array(preds)

class SparkRFServer:
    def __init__(self, path):
        conf = SparkConf() \
            .setAppName("crankshaw-pyspark") \
            .set("spark.executor.memory", "2g") \
            .set("spark.kryoserializer.buffer.mb", "128") \
            .set("master", "local")
        sc = SparkContext(conf=conf, batchSize=10)
        self.model = RandomForestModel.load(sc, path)
        self.path = path
        print("started spark")

    def compute_features(self, inputs):
        start = datetime.datetime.now()
        preds = []
        for i in inputs:
            # TODO is making an RDD faster? probably not
            preds.append(self.model.predict(i))
        end = datetime.datetime.now()
        print("%s: %f ms\n" % (os.path.basename(self.path), (end-start).total_seconds() * 1000))
        preds = np.array(preds)
        assert preds.dtype == np.dtype('float64')
        return np.array(preds)

def load_scikit_model(pickle_path):
    # pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
    feature = joblib.load(pickle_path)
    name = os.path.basename(pickle_path).strip(".pkl")
    return (name, feature)

class SklearnServer:
    def __init__(self, path):
        self.name, self.model = load_scikit_model(path)
        self.path = path
        print("started sklearn\n")

    def compute_features(self, inputs):
        start = datetime.datetime.now()
        # pred = self.model.predict(np.array(inputs).reshape(1, -1))[0]
        # if len(inputs) == 1:
        #     self.model.predict(np.array(inputs).reshape(1, -1))
        # else:
        preds = self.model.predict(np.array(inputs))
        end = datetime.datetime.now()
        print("%s: %f ms\n" % (self.path, (end-start).total_seconds() * 1000))
        assert preds.dtype == np.dtype('float64')
        return preds


def parse_args():
    parser = argparse.ArgumentParser(usage='''Runs the server''')
    parser.add_argument("ip", type=str, help="ADDRESS")
    parser.add_argument("port", type=int, help="PORT")
    parser.add_argument("framework", type=str, help="spark|sklearn")
    parser.add_argument("modelpath", help="full path to pickled model file")
    return parser.parse_args()

def start_server(model, ip, port):
    print("Starting server")
    server = SocketServer.TCPServer((ip, port), FeatureTCPHandler)
    # This works, but hard to clean up from
    # server = SocketServer.ForkingTCPServer((args.ip, args.port), FeatureTCPHandler)
    server.model = model
    server.serve_forever()

def start_svm_from_mp(mp, ip, port):
    model = SparkSVMServer(mp)
    start_server(model, ip, port)


def main():
    args = parse_args()
    model_path = args.modelpath
    # print(model_path)
    model = None
    if args.framework == "sparkrf":
        model = SparkRFServer(model_path)
    elif args.framework == "sparklr":
        model = SparkLRServer(model_path)
    elif args.framework == "sparksvm":
        model = SparkSVMServer(model_path)
    elif args.framework == "sklearn":
        model = SklearnServer(model_path)
    else:
        print("%s is unsupported framework" % args.framework)
        sys.exit(1)
    start_server(model, args.ip, args.port)

if __name__=='__main__':
    main()



