from __future__ import print_function
import sys


def gen_replicas(base_text, sub_dict, num_replicas):
    docker_str = ""
    for i in range(num_replicas):
        # sub_dict["replica_num"] = i
        docker_str += base_text.format(replica_num = i, **sub_dict)

    deps_str = ""
    models_addr_str = ""
    for i in range(num_replicas):
        deps_str += """
      - {model_name}_r{replica_num}""".format(replica_num = i, **sub_dict)
        models_addr_str += "{model_name}_r{replica_num}:6001,".format(replica_num = i, **sub_dict)

    models_toml = """
[[models]]
name = "{model_name}"
addresses = ["{models_addr_str}"]
num_outputs = 1
version = 1
"""

    return (docker_str, deps_str, models_toml.format(models_addr_str = models_addr_str, **sub_dict))



def sklearn_rf_sub_dict(depth):
    sub_dict = {
            "model_name": "rf_d%d" % depth,
            "image": "clipper/sklearn-mw",
            "model_path": "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/50rf_pred3_depth%d/" % depth,
            "container_model_path": "/model/50rf_pred3_depth%d.pkl" % depth
            }
    return sub_dict

def sklearn_svm_sub_dict():
    sub_dict = {
            "model_name": "svm",
            "image": "clipper/sklearn-mw",
            "model_path": "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/svm_pred3/",
            "container_model_path": "/model/svm_pred3.pkl"
            }
    return sub_dict

def sklearn_log_regression_sub_dict():
    sub_dict = {
            "model_name": "log_reg",
            "image": "clipper/sklearn-mw",
            "model_path": "${CLIPPER_ROOT}/model_wrappers/python/sklearn_models/log_regression_pred3/",
            "container_model_path": "/model/log_regression_pred3.pkl"
            }
    return sub_dict

def spark_svm_sub_dict(label):
    sub_dict = {
            "model_name": "svm_pred%d" % label,
            "image": "clipper/spark-mw-dev",
            "model_path": "${CLIPPER_ROOT}/model_wrappers/python/spark_models/svm_predict_%d" % label,
            "container_model_path": "/model"
            }
    return sub_dict

if __name__=='__main__':
    prefix_text = """
version: '2'

services:
  redis:
    image: redis:alpine
  clipper:
    image: cl-dev
    depends_on:
      - redis{additional_deps}

    volumes:
      - "${{MNIST_PATH}}:/mnist_data:ro"
      - "${{CLIPPER_ROOT}}/digits_bench.toml:/tmp/digits_bench.toml:ro"

{deps_definitions}
"""
    base_text = """
  {model_name}_r{replica_num}:
    image: {image}
    volumes:
      - "{model_path}:/model:ro"
    environment:
      - CLIPPER_MODEL_PATH={container_model_path}

"""
    # num_replicas = int(sys.argv[1])

    

    container_def_str = ""
    deps_str = ""
    models_toml = ""

    #########################################################
    ######################## SKLEARN ########################

    ### LOG REGRESSION
    cur_container_def, cur_deps_str, cur_model_toml = gen_replicas(base_text, sklearn_log_regression_sub_dict(), 2)
    container_def_str += cur_container_def
    deps_str += cur_deps_str
    models_toml += cur_model_toml

    ### SVM
    cur_container_def, cur_deps_str, cur_model_toml = gen_replicas(base_text, sklearn_svm_sub_dict(), 2)
    container_def_str += cur_container_def
    deps_str += cur_deps_str
    models_toml += cur_model_toml

    ### RF D4
    cur_container_def, cur_deps_str, cur_model_toml = gen_replicas(base_text, sklearn_rf_sub_dict(4), 2)
    container_def_str += cur_container_def
    deps_str += cur_deps_str
    models_toml += cur_model_toml

    ### RF D8
    cur_container_def, cur_deps_str, cur_model_toml = gen_replicas(base_text, sklearn_rf_sub_dict(8), 2)
    container_def_str += cur_container_def
    deps_str += cur_deps_str
    models_toml += cur_model_toml

    ### RF D16
    cur_container_def, cur_deps_str, cur_model_toml = gen_replicas(base_text, sklearn_rf_sub_dict(16), 2)
    container_def_str += cur_container_def
    deps_str += cur_deps_str
    models_toml += cur_model_toml

    #########################################################
    ######################### SPARK #########################

    ### SVM PRED 3
    cur_container_def, cur_deps_str, cur_model_toml = gen_replicas(base_text, spark_svm_sub_dict(3), 2)
    container_def_str += cur_container_def
    deps_str += cur_deps_str
    models_toml += cur_model_toml

    text = prefix_text.format(additional_deps = deps_str, deps_definitions=container_def_str)
    print(models_toml)
    with open("docker-compose.yml", 'w') as f:
        f.write(text)
        # for i in range(num_replicas):
        #     f.write(line % i)


