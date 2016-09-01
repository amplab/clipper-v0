from __future__ import print_function
import sys
import json
import os
import requests
import random
from datetime import datetime
import skimage
import time
from skimage.transform import resize

import numpy as np
import skimage.io as skio
import sklearn.linear_model

DAISY_PATH = '../clipper_server/models/inception/raw-data/train/daisy/'

def load_inception_imgs(img_location):
    imgs = []
    img_files = os.listdir(img_location)
    # Get all the jpg images (only the first 10 or so)
    img_files = filter(lambda x: '.jpg' in x, img_files)[:10]
    # Read in the images and reshape
    for i in range(len(img_files)):
        img = skio.imread(os.path.join(img_location, img_files[i]))
        img = resize(img, (299, 299)).flatten().tolist()
        imgs.append(img)
    return imgs

def mnist_update(uid, x, y):
    url = "http://localhost:1337/update"
    req_json = json.dumps({'uid': uid, 'input': list(x), 'label': y})
    headers = {'Content-type': 'application/json'}
    start = datetime.now()
    r = requests.post(url, headers=headers, data=req_json)
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0
    print("'%s', %f ms" % (r.text, latency))

def inception_prediction(host, uid, x):
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
    x = load_inception_imgs(DAISY_PATH)
    uid = 4
    while True:
        example_num = np.random.randint(0,len(x))
        inception_prediction("localhost", uid, x[example_num])
        time.sleep(1.5)


