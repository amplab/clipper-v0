#ifndef __NOOP_MODEL_WRAPPER_H_INCLUDED__
#define __NOOP_MODEL_WRAPPER_H_INCLUDED__
#include<vector>
#include "Model.h"

class NoopModelWrapper : public Model {
    public:
        vector<double>* predict_bytes(vector< vector<char> >& input);

        vector<double>* predict_floats(vector< vector<double> >& input);

        vector<double>* predict_ints(vector< vector<uint32_t> >& input);

        vector<double>* predict_strings(vector< vector<string> >& input);
};

#endif
