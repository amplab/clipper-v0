#!/usr/bin/env bash

for i in {1..9}
do
  echo "starting spark server $i"
  python feature_server.py 127.0.0.1 600$i spark-lr spark_models/lg_predict_$i &
  # python feature_server.py 127.0.0.1:600$i sklearn sklearn_models/predict_"$i"_svm/predict_"$i"_svm.pkl &

 done

echo "starting spark server 10"
python feature_server.py 127.0.0.1 6010 spark-lr spark_models/lg_predict_10 &
