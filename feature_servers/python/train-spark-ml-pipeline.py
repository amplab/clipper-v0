
from __future__ import absolute_import

import findspark
findspark.init()
import os


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.regression import LabeledPoint
import time

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



def train_random_forest(num_trees):
    conf = SparkConf() \
            .setAppName("crankshaw-pyspark") \
            .set("spark.executor.memory", "2g") \
            .set("spark.kryoserializer.buffer.mb", "128") \
            .set("master", "local")

    sc = SparkContext(conf=conf, batchSize=10)
    sql = SQLContext(sc)

    print('Parsing data')
    time_start = time.time()
    data_path = os.path.expanduser("~/mnist/train-mnist-dense-with-labels.data")
    trainRDD = sc.textFile(data_path).map(lambda line: parseData(line, objective)).cache()
    df = sql.createDataFrame(trainRDD)
    print(df.dtypes)

    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=num_trees)
    pipeline = Pipeline(stages=[labelIndexer, rf])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(df)

    rfModel = model.stages[1]
    print(rfModel)  # summary only
    # Make predictions.
    # predictions = model.transform()






    # testRDD = sc.textFile('/Users/crankshaw/model-serving/data/mnist_data/test-mnist-dense-with-labels.data').map(lambda line: parseData(line, objective)).cache()



    # print 'Fitting model'
    # rf = RandomForest.trainClassifier(trainRDD, 2, {}, num_trees)
    # rf.save(sc, "spark_models/%drf_pred_1" % num_trees)
    # lrm.save(sc, path)

    sc.stop()

if __name__=='__main__':
    train_random_forest(10)
