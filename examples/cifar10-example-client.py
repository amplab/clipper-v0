from __future__ import print_function
import sys
import json
import os
import requests
import random
from datetime import datetime
import skimage
import time
import caffe
from caffe.proto import caffe_pb2
from skimage.transform import resize

import leveldb
import numpy as np
import skimage.io as skio

CIFAR10_DATA_PATH = "data/cifar10"

def load_and_predict(leveldb_path):
    db = leveldb.LevelDB(leveldb_path)
    datum = caffe_pb2.Datum()
    for key, value in db.RangeIter():
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        data = np.transpose(data, (1, 2, 0))
        data = data.flatten().tolist()
        cifar_prediction("localhost", 4, data)

def cifar_update(uid, x, y):
    url = "http://localhost:1337/update"
    req_json = json.dumps({'uid': uid, 'input': list(x), 'label': y})
    headers = {'Content-type': 'application/json'}
    start = datetime.now()
    r = requests.post(url, headers=headers, data=req_json)
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0
    print("'%s', %f ms" % (r.text, latency))

def cifar_prediction(host, uid, x):
    url = "http://%s:1337/predict" % host
    req_json = json.dumps({'uid': uid, 'input': list(x)})
    headers = {'Content-type': 'application/json'}
    # x_str = ", ".join(["%d" % a for a in x])
    start = datetime.now()
    r = requests.post(url, headers=headers, data=req_json)
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0
    print("'%s', %f ms" % (r.text, latency))

if __name__=='__main__':
    args = sys.argv
    x = load_and_predict(CIFAR10_DATA_PATH)
