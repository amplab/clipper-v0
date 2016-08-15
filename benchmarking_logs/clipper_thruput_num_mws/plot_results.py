from __future__ import print_function
import toml
import os
import json
import sys

runs = [f.split("_results.json")[0] for f in os.listdir(".") if "json" in f]
print(runs)
num_mws = []
thrus = []
for r in runs:
    conf = "{ts}_config.toml".format(ts=r)
    with open(conf, 'r') as cf:
        conf_dict = toml.load(cf)
        # print(conf_dict)
        num_mws.append(len(conf_dict['models']))
    results = "{ts}_results.json".format(ts=r)
    with open(results, 'r') as rf:
        results_dict = json.load(rf)
        thrus.append([e['rate'] for e in results_dict["meters"] if e['name'] == "prediction_thruput"][0])
print(num_mws)
print(thrus)
        # print(results_dict)

