#!/usr/bin/env bash

killall python
sleep 10
# killall python
# sleep 5

for i in {1..9}
do
  echo "starting spark server $i"
  python rpc_feature_server.py 127.0.0.1 600$i sparksvm spark_models/svm_predict_$i &
  # python rpc_feature_server.py 127.0.0.1 600$i sparklr spark_models/lg_predict_$i &>  /dev/null &

 done

echo "starting spark server 10"
python rpc_feature_server.py 127.0.0.1 6010 sparksvm spark_models/svm_predict_10 &

sleep 15
ps aux | grep python
