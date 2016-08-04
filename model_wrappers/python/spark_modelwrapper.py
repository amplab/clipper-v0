from __future__ import print_function
import numpy as np
import time
import datetime
import sys
import os
import rpc
import findspark
findspark.init()
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionModel, LogisticRegressionWithSGD
# from pyspark.mllib.classification import SVMModel, SVMWithSGD



class PySparkLRModelWrapper(rpc.ModelWrapperBase):


    def __init__(self, path):
        conf = SparkConf() \
            .setAppName("crankshaw-pyspark") \
            .set("spark.executor.memory", "1g") \
            .set("spark.kryoserializer.buffer.mb", "128") \
            .set("master", "local")
        sc = SparkContext(conf=conf, batchSize=10)
        self.model = LogisticRegressionModel.load(sc, path)
        self.path = path
        print("Started Spark and loaded Logistic Regression")


    def predict_ints(self, inputs):
        preds = []
        for x in inputs:
            preds.append(float(self.model.predict(x)))
        # print("predicting %d integer inputs" % len(inputs))
        return np.array(preds)

    def predict_floats(self, inputs):
        preds = []
        for x in inputs:
            preds.append(float(self.model.predict(x)))
        return np.array(preds)




if __name__=='__main__':
    model_path = os.environ["CLIPPER_MODEL_PATH"]
    print(model_path)
    model = PySparkLRModelWrapper(model_path)
    rpc.start(model, 6001)

