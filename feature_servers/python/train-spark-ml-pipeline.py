
from __future__ import absolute_import

import findspark
findspark.init()
import os
import datetime


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import DenseVector
import time
import numpy as np
import argparse
import capnp
capnp.remove_import_hook()
feature_capnp = capnp.load(os.path.abspath('../../clipper_server/schema/feature.capnp'))

# # Load and parse the data file, converting it to a DataFrame.
# data = sqlContext.read.text.load("mnist/train-mnist-dense-with-labels.data")
#
# # Index labels, adding metadata to the label column.
# # Fit on whole dataset to include all labels in index.
# labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# # Automatically identify categorical features, and index them.
# # Set maxCategories so features with > 4 distinct values are treated as continuous.
# # featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
#
# # Split the data into training and test sets (30% held out for testing)
# # (trainingData, testData) = data.randomSplit([0.7, 0.3])
#
# # Train a RandomForest model.
# rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
#
# # Chain indexers and forest in a Pipeline
# pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
#
# # Train model.  This also runs the indexers.
# model = pipeline.fit(trainingData)
#
# # Make predictions.
# predictions = model.transform(testData)
#
# # Select example rows to display.
# predictions.select("prediction", "indexedLabel", "features").show(5)
#
# # Select (prediction, true label) and compute test error
# evaluator = MulticlassClassificationEvaluator(
#             labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
# accuracy = evaluator.evaluate(predictions)
# print("Test Error = %g" % (1.0 - accuracy))
#
# rfModel = model.stages[2]
# print(rfModel)  # summary only

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



def train_random_forest(sc, sql, num_trees):
    x = DenseVector([float(v)/255.0 for v in range(1, 785)])
    FeatureVec = Row("features")
    fv = FeatureVec(x)
    rdd = sc.parallelize([fv])
    test_df = sql.createDataFrame([FeatureVec(x)])

    print('Parsing data')
    time_start = time.time()
    data_path = os.path.expanduser("~/model-serving/data/mnist_data/train-mnist-dense-with-labels.data")
    # data_path = os.path.expanduser("/crankshaw-local/mnist/data/train.data")
    trainRDD = sc.textFile(data_path).map(lambda line: parseData(line, objective)).cache()
    df = sql.createDataFrame(trainRDD)
    print(df.dtypes)

    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=num_trees)
    pipeline = Pipeline(stages=[labelIndexer, rf])

    # Train model. This also runs the indexers.
    model = pipeline.fit(df)
    print(model.transform(test_df))

    rfModel = model.stages[1]
    print(rfModel)  # summary only
    return model
    # Make predictions.
    # predictions = model.transform()






    # testRDD = sc.textFile('/Users/crankshaw/model-serving/data/mnist_data/test-mnist-dense-with-labels.data').map(lambda line: parseData(line, objective)).cache()



    # print 'Fitting model'
    # rf = RandomForest.trainClassifier(trainRDD, 2, {}, num_trees)
    # rf.save(sc, "spark_models/%drf_pred_1" % num_trees)
    # lrm.save(sc, path)

    sc.stop()



class PySparkMLFeatureImpl(feature_capnp.Feature.Server):
    # TODO: find out how to stop spark context
    def __init__(self, sc, sql, model):
        self.sc = sc
        self.sql = sql
        self.model = model

    def computeFeature(self, inp, _context, **kwargs):
        # print(_context.params)
        # s = name
        # print(type(self.model))
        start = datetime.datetime.now()
        print(inp)
        x = DenseVector([float(v)/255.0 for v in inp])
        FeatureVec = Row("features")
        # fv = FeatureVec(x)
        # rdd = sc.parallelize([fv])
        test_df = sql.createDataFrame([FeatureVec(x)])
        # df = sql.createDataFrame(sc.parallelize(Row(features = x)))
        # pred = 1.1
        pred = self.model.transform(test_df)
        
        # print(pred)
        # print("PYSPARK: model predicted: %f" % pred)
        end = datetime.datetime.now()
        print("%s: %f ms\n" % ("spark.ml", (end-start).total_seconds() * 1000))
        ppp = pred.collect()[0].prediction
        print(ppp)
        return ppp

def parse_args():
    parser = argparse.ArgumentParser(usage='''Runs the server bound to the\
given address/port ADDRESS may be '*' to bind to all local addresses.\
:PORT may be omitted to choose a port automatically. ''')

    parser.add_argument("address", type=str, help="ADDRESS[:PORT]")
    parser.add_argument("numtrees", type=int, help="number of trees")
    # parser.add_argument("framework", type=str, help="spark|sklearn")
    # parser.add_argument("modelpath", help="full path to pickled model file")


    return parser.parse_args()

if __name__=='__main__':

    args = parse_args()
    address = args.address
    numtrees = args.numtrees

    conf = SparkConf() \
            .setAppName("crankshaw-pyspark") \
            .set("spark.executor.memory", "8g") \
            .set("spark.kryoserializer.buffer.mb", "128") \
            .set("master", "local")

    sc = SparkContext(conf=conf, batchSize=1)
    sql = SQLContext(sc)
    rf = train_random_forest(sc, sql, numtrees)

    server = capnp.TwoPartyServer(address, bootstrap=PySparkMLFeatureImpl(sc, sql, rf))
    server.run_forever()
    














