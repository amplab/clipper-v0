from __future__ import print_function
import sys
import json
import os
import requests
import random
from datetime import datetime


def add_model():
    # url = "http://localhost:1337/addmodel"
    url = "http://localhost:1337/addreplica"
    new_model = {
            "name": "newmodeltest-noop",
            "version": 1,
            "addrs": ["127.0.0.1:7001",]
            }
    req_json = json.dumps(new_model)
    headers = {'Content-type': 'application/json'}
    start = datetime.now()
    r = requests.post(url, headers=headers, data=req_json)
    end = datetime.now()
    latency = (end - start).total_seconds() * 1000.0
    print("'%s', %f ms" % (r.text, latency))

if __name__=='__main__':
    add_model()
