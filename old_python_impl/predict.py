from sklearn.externals import joblib
import pandas as pd
import numpy as np
from collections import OrderedDict
import os
from sklearn import linear_model as lm

# TODO LIST:
# Update a feature
# Add a model
# Init code
# Serving code

def load_digits(digits_location, digits_filename = "test-mnist-dense-with-labels.data"):
    digits_path = digits_location + "/" + digits_filename
    print "Source file:", digits_path
    df = pd.read_csv(digits_path, sep=",", header=None)
    data = df.values
    print "Number of image files:", len(data)
    y = data[:,0]
    X = data[:,1:]
    return (X, y)



# TODO: Open questions?
# What does training look like for additive merge for a classifier?
# How to encode SLAs?
# Adding a feature to an already trained merge?

# Close questions:
# What to fill in for missing feature values in anytime predictions?


# Merge is associated with a single version of features
# If even a single feature changes or is added, we relearn merge
class MultClassifierMerge:

    def __init__(self):
        self.init = False
        # indicates we only need features to learn merge, not original xs
        self.features_only = True
        # self.k = k
        # self.xs = []
        # self.ys = []

    def predict(self, features):
        return self.model.predict(features)

    def update_merge(self, fs, ys):
        m = lm.LogisticRegression()
        m.fit(fs, ys)
        self.model = m

    def rank_features(self):
        return np.argsort(np.absolute(self.model.coef_)).reverse()

class MultRegressionMerge:

    def __init__(self):
        self.init = False
        # indicates we only need features to learn merge, not original xs
        self.features_only = True
        # self.k = k
        # self.xs = []
        # self.ys = []

    def predict(self, features):
        return self.model.predict(features)

    def update_merge(self, fs, ys):
        m = lm.LinearSVC()
        m.fit(fs, ys)
        self.model = m

    def rank_features(self):
        return np.argsort(np.absolute(self.model.coef_)).reverse()

# Learns a per-task residual model and adds it to the global
# shared model
class AddRegressionMerge:

    def __init__(self):
        self.init = False
        self.features_only = False

    def predict(self, x, feature):
        assert len(features) == 1
        return self.model.predict(x) + feature

    def update_merge(self, xs, fs, ys):
        residual_ys = ys - fs
        m = lm.LinearSVC()
        m.fit(xs, residual_ys)
        self.model = m

    # only one feature
    def rank_features(self):
        # raise NotImplementedError
        return [0]

class TaskData:
    def __init__(self, tid):
        self.tid = tid
        self.xs = []
        self.ys = []


    def add_example(self, x, y):
        self.xs.append(x)
        self.ys.append(y)

    def get_data(self):
        '''
        Return all training data for this task as numpy arrays

        '''
        return (np.array(x), np.array(y))

# load a csv of format:
# tid, y, x
def load_train_data(fname):
    data = np.genfromtxt ('file.csv', delimiter=",")
    data_dict = {}
    # TODO this could be much more efficient using pandas, but pandas is a pain.
    for i in range(data):
        point = data[i]
        t = point[0]
        y = point[1]
        x = point[2:]
        if not t in data_dict:
            data_dict[t] = TaskData(t)
        data_dict[t].add_example(x, y)
    return data_dict


    

class Model:
    def __init__(self, name, namespace, cache, train_data, merge_class):
        """
        name(str): the name of the model. Used to determine the directory
                   to look for this model's features.

        namespace(str): root directory of this serving system. All models in
                        the same namespace can share features, and share a cache.
                        A namespace corresponds to a git repository.

        cache(FeatureCache): The feature cache shared among all models in this
                             namespace.

        train_data: Can be str indicating a filename or a dict with the data already loaded.
        """

        self.name = name
        self.namespace = namespace
        self.model_path = "%s/models/%s" % (namespace, name) # this is a *.model file
        self.feature_objects, self.feature_names = self.load_feature_functions()
        self.merge_operators = {}
        self.feature_cache = cache
        if type(train_data) == str:
            self.train_data = load_train_data(train_data)
        else:
            self.train_data = train_data
        self.merge_class = merge_class

        self.train_all_merges()

        # Train merge operators using provided training data
    def train_all_merges(self):
        for t in self.train_data:
            train_x, train_y = self.train_data[t].get_data()
            m = self.merge_class()
            self.merge_operators[t] = m
            # Want to train with all features, so order we compute features
            # doesn't matter. Also, hopefully most of these features
            # will be cached anyway
            ranked_fs = range(len(self.feature_objects))
            fs = np.array([self.get_features(xi, ranked_fs, -1) for xs in train_x])
            if m.features_only:
                m.update_merge(fs, train_ys)
            else:
                m.update_merge(train_x, fs, train_ys)


    def online_update(self, t, x, y):
        self.train_data[t].add_example(x, y)
        train_x, train_y = self.train_data[t].get_data()
        if len(train_y) < 2:
            print "Trying to train a model with less than 2 examples, aborting."
        else:
            m = self.merge_operators[t]
            # This is a new task, create a new merge operator
            if m is None:
                m = self.merge_class()
                self.merge_operators[t] = m
            # Want to train with all features, so order we compute features
            # doesn't matter. Also, hopefully most of these features
            # will be cached anyway
            ranked_fs = range(len(self.feature_objects))
            fs = np.array([self.get_features(xi, ranked_fs, -1) for xs in train_x])
            if m.features_only:
                m.update_merge(fs, train_ys)
            else:
                m.update_merge(train_x, fs, train_ys)

    def load_feature_functions(self):
        feature_objects = []
        feature_names = [line.strip() for line in open(self.model_path, 'r')]
        for lf in feature_names:
            pickle_loc = "%s/features/%s/%s.pkl" % (self.namespace, lf, lf)
            feature = joblib.load(pickle_loc) 
            feature_objects.append(feature)
        return (feature_objects, feature_names)

    def predict(self, x, t):
        return self._merge(x,t)

    def _merge(self, x, t):
        m = self.merge_operators[t]
        if m is not None:
            ranked_fs = m.rank_features()
            fs = self.get_features(x, ranked_fs, -1)
            if m.features_only:
                return m.predict(fs)
            else:
                return m.predict(x, fs)
        else:
            print "Task %d doesn't exist" % t

    
    def get_features(self, x, ranked_features, max_features=-1):
        """
        x: input to the feature functions. This can be an arbitrary type
           as long as all the features can handle this input type.

        ranked_features(ndarray): Array of indices listing features
                                  in descending order of importance.

        max_features(int): The maximum number of features to evaluate. This is a placeholder
                           for a more sophisticated cost-based feature eval approach,
                           but is in place to demonstrate anytime predictions. A negative
                           value means use all the features. Default: -1
        """

        # TODO what to fill in for features we can't afford to compute?
        fs = np.zeros(len(self.features))
        features_used = 0
        # if max_features is -1, use all the features
        if max_features <= 0:
            max_features = len(self.features)
        for i in ranked_features:
            fname = self.feature_names[i]
            feature_i = self.feature_cache.lookup(fname, x)
            if feature_i is None:
                feature_i = self.features[i].predict(x)
                self.feature_cache.store(fname, x, feature_i)
            fs[i] = feature_i
            features_used += 1
            if features_used >= max_features:
                break
        return fs



class FeatureCache:
    def __init__(self):
        # stores cache for each feature
        self.cache = {}
        # stores feature objects for hashing
        self.features = {}
    
    def add_feature(self, fname, feature):
        self.cache[fname] = {}
        self.features[fname] = feature

    def lookup(self, fname, x):
        x_hash = self.features[fname].hash_input(x)
        if x_hash in self.cache[fname]:
            return self.cache[fname][x_hash]
        else:
            return None
    def store(self, fname, x, y):
        x_hash = self.features[fname].hash_input(x)
        self.cache[fname][x_hash] = y



class Application:
    def __init__(self, namespace):
        pass
        

if __name__=='__main__':
    namespace_path = os.path.expanduser("~/test") # full path
    digits_loc = "/Users/crankshaw/velox-centipede/data/mnist"








