import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn import svm, ensemble
from sklearn.ensemble import GradientBoostingClassifier as GBClassifier, RandomForestClassifier as RFClassifier
import digits_utils as digits
import timeit


def model_eval_latency():
    mnist_loc = '/Users/crankshaw/model-serving/data/mnist_data'
    train_x, train_y = digits.load_digits(mnist_loc, digits_filename="train-mnist-dense-with-labels.data")
    train_z = digits.normalize_digits(train_x)
    test_x, test_y = digits.load_digits(mnist_loc, digits_filename="test-mnist-dense-with-labels.data")
    test_z = digits.normalize_digits(test_x)

    # pick binary classification task, classify sevens
    is_seven = train_y == 7
    # is_one = train_ys == 1
    binary_ys = is_seven.astype(int)
    is_seven_test = test_y == 7
    binary_ys_test = is_seven_test.astype(int)
    # my_filter = is_seven | is_one
    # new_ys = train_ys[my_filter]
    # new_zs = train_zs[my_filter]
    # print len(new_ys), len(new_zs)
    # seven_zs = train_zs[is_seven]
    # one_zs = train_zs[is_one]
    print "training trees"
    clf = GBClassifier(n_estimators=5)
    clf.fit(train_z, binary_ys)
    print clf.predict(test_z[0].reshape(1, -1))[0], binary_ys_test[0]

def gb_latency():
    num_estimators = 500
    test_example_idx = 2688
    setup = '''
import digits_utils as digits
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBClassifier, RandomForestClassifier as RFClassifier

mnist_loc = '/Users/crankshaw/model-serving/data/mnist_data'
train_x, train_y = digits.load_digits(mnist_loc, digits_filename="train-mnist-dense-with-labels.data")
train_z = digits.normalize_digits(train_x)
test_x, test_y = digits.load_digits(mnist_loc, digits_filename="test-mnist-dense-with-labels.data")
test_z = digits.normalize_digits(test_x)

# pick binary classification task, classify sevens
is_seven = train_y == 7
# is_one = train_ys == 1
binary_ys = is_seven.astype(int)
is_seven_test = test_y == 7
binary_ys_test = is_seven_test.astype(int)
# my_filter = is_seven | is_one
# new_ys = train_ys[my_filter]
# new_zs = train_zs[my_filter]
# print len(new_ys), len(new_zs)
# seven_zs = train_zs[is_seven]
# one_zs = train_zs[is_one]
print "training trees"
clf = GBClassifier(n_estimators=%d)
clf.fit(train_z, binary_ys)
test_x_1 = test_z[%d].reshape(1, -1)
    ''' % (num_estimators, test_example_idx)

    statement = '''
clf.predict(test_x_1)
                '''

    t = timeit.Timer(stmt=statement, setup=setup)
    # print t.timeit(iters)
    # print t.timeit(iters)/(2.0* float(iters)) * 1000.0
    num_preds = 1000

    res = np.array(t.repeat(repeat=2, number=num_preds))
    print np.min(res)/float(num_preds) * 1000.0, np.max(res)/float(num_preds) * 1000.0

if __name__=='__main__':
    # model_eval_latency()
    gb_latency()

