#!/usr/bin/env bash

# for i in 1 10 25 50 100 150 200
# do
#   echo $i
#   killall python
#   sleep 10
#   cd ../feature_servers/python
#   python rpc_feature_server.py 127.0.0.1 6001 sparklr spark_models/lg_predict_1 &> /dev/null &
#   SPARK_PID=$!
#   echo $SPARK_PID
#   cd -
#   sleep 10
#   ps aux | grep python
#
#   sleep 10
#   ./target/release/clipper featurelats $i >> experiments_RAW/feature_lats/pyspark_lr_local.txt &
#   CLIPPER_PID=$!
#   echo $CLIPPER_PID
#
#   sleep 100
#   kill -9 $CLIPPER_PID
#   # kill -9 $SPARK_PID
#
# done


# for i in 200
# do
#   echo $i
#   killall python
#   sleep 30
#   cd ../feature_servers/python
#   python rpc_feature_server.py 127.0.0.1 6001 sparkrf spark_models/10rf_pred_1 &> /dev/null &
#   SPARK_PID=$!
#   echo $SPARK_PID
#   cd -
#   sleep 10
#   ps aux | grep python
#
#   sleep 10
#   ./target/release/clipper featurelats $i >> experiments_RAW/feature_lats/pyspark_10rf_local.txt &
#   CLIPPER_PID=$!
#   echo $CLIPPER_PID
#
#   sleep 100
#   kill -9 $CLIPPER_PID
#   # kill -9 $SPARK_PID
#
# done

# for i in 200
# do
#   echo $i
#   killall python
#   sleep 5
#   killall python
#   sleep 30
#   cd ../feature_servers/python
#   python rpc_feature_server.py 127.0.0.1 6001 sparkrf spark_models/100rf_pred_1 &> /dev/null &
#   SPARK_PID=$!
#   echo $SPARK_PID
#   cd -
#   sleep 10
#
#   sleep 10
#   ./target/release/clipper featurelats $i >> experiments_RAW/feature_lats/pyspark_100rf_local.txt &
#   CLIPPER_PID=$!
#   echo $CLIPPER_PID
#
#   sleep 150
#   kill -9 $CLIPPER_PID
#   # kill -9 $SPARK_PID
#
# done

for i in 200
do
  echo $i
  killall python
  sleep 30
  cd ../feature_servers/python
  python rpc_feature_server.py 127.0.0.1 6001 sklearn sklearn_models/predict_1_svm/predict_1_svm.pkl &> /dev/null &
  SPARK_PID=$!
  echo $SPARK_PID
  cd -
  sleep 10
  ps aux | grep python

  sleep 20
  ./target/release/clipper featurelats $i >> experiments_RAW/feature_lats/sklearn_svm_local.txt &
  CLIPPER_PID=$!
  echo $CLIPPER_PID

  sleep 200
  kill -9 $CLIPPER_PID
  # kill -9 $SPARK_PID

done
