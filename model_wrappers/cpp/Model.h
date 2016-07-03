#ifndef __MODEL_H__
#define __MODEL_H__
#include <string>
#include <vector>
using namespace std;

class Model
{
    public:
        Model(std::string model_path) : path(model_path) {}

        std::string path;

        virtual vector<char>* predict_bytes(vector< vector<char> >& input) = 0;

        virtual vector<double>* predict_floats(vector< vector<double> >& input) = 0;

        virtual vector<uint32_t>* predict_ints(vector< vector<uint32_t> >& input) = 0;
};

#endif
