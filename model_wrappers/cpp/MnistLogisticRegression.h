#ifndef __MNIST_LOGISTIC_REGRESSION_H_INCLUDED__
#define __MNIST_LOGISTIC_REGRESSION_H_INCLUDED__
#include<string>
#include<vector>
#include "Model.h"
#include "../../../vowpal_wabbit/vowpalwabbit/vw.h"
using namespace std;

class MnistLogisticRegression : public Model {
    public:
        string path;
        vw *model;
        vector<double> predictions;
        MnistLogisticRegression(string model_path);
        vector<double>& predict_bytes(vector< vector<char> >& input);
        vector<double>& predict_floats(vector< vector<double> >& input);
        vector<double>& predict_ints(vector< vector<uint32_t> >& input);
        vector<double>& predict_strings(vector< vector<string> >& input);
};

#endif
