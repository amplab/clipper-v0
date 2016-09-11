
import os
import sys
import json


def compare_throughput_to_predicted(res):
    measured_thrus = {}
    pred_thrus = {}

    for m in res["meters"]:
        if "model" in m["name"]:
            model_name = m["name"].split(":")[0]
            # print model_name
            measured_thrus[model_name] = m["rate"]
    # print measured_thrus
    for c in res["counters"]:
        if "max_thru_gauge" in c["name"]:
            model_name = c["name"].split(":")[0]
            pred_thrus[model_name] = c["count"] / 1000.0
            # for model_name in measured_thrus:
            #     # print model_name, c["name"]
            #     if model_name in c["name"]:
            #         pred_thrus[model_name] = c["count"] / 1000.0
    # print pred_thrus
    for m in measured_thrus:
        print m, measured_thrus[m]

    
    

            # print c["count"] / 1000.0

def compare_batch_size_to_predicted(res):
    pred_batch_sizes = {}
    for c in res["counters"]:
        if "max_batch_size_gauge" in c["name"]:
            model_name = c["name"].split(":")[0]
            pred_batch_sizes[model_name] = c["count"]

    actual_batch_sizes = {}
    for h in res["histograms"]:
        if "model_batch_size" in h["name"]:
            model_name = h["name"].split(":")[0]
            actual_batch_sizes[model_name] = h

    for model in actual_batch_sizes:
        # print ("MODEL: {model_name}, pred batch: {pred_batch}, "
        #         "mean_batch: {mean_batch}, max_batch: {max_batch}, diff: {diff}").format(
        #                 model_name = model, pred_batch = pred_batch_sizes[model],
        #                 mean_batch = actual_batch_sizes[model]["mean"],
        #                 max_batch = actual_batch_sizes[model]["max"],
        #                 diff = (pred_batch_sizes[model] - actual_batch_sizes[model]["mean"]))


        print ("MODEL: {model_name} "
                "mean_batch: {mean_batch}, max_batch: {max_batch}").format(
                        model_name = model, 
                        mean_batch = actual_batch_sizes[model]["mean"],
                        max_batch = actual_batch_sizes[model]["max"])




if __name__=="__main__":
    # results_file = "learned_batching_160819-183618_results.json"
    results_file = sys.argv[1]
    with open(results_file, "r") as f:
        res = json.load(f)
    compare_throughput_to_predicted(res)
    compare_batch_size_to_predicted(res)

