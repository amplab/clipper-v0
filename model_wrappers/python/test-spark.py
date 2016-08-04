import findspark
# findspark.init("/tmp/spark-1.6.2-bin-hadoop2.6")
findspark.init()

import pyspark
from pyspark import SparkConf, SparkContext
conf = SparkConf() \
    .setAppName("crankshaw-pyspark") \
    .set("spark.executor.memory", "2g") \
    .set("spark.kryoserializer.buffer.mb", "128") \
    .set("master", "local")
sc = SparkContext(conf=conf, batchSize=10)
