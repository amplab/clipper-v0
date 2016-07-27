#include <vector>
#include "ClipperRPC.h"
#include "MnistLogisticRegression.h"
#include "../../../vowpal_wabbit/vowpalwabbit/parser.h"
using namespace std;

/* Some utility functions */
string join(vector<char>& input, string delimiter) {
    int i;
    stringstream ss;
    ss << "| ";
    for (i = 0; i < input.size(); i++)
        ss << i << ":" << input[i] << delimiter;
    return ss.str();
}

string join(vector<uint32_t>& input, string delimiter) {
    int i;
    stringstream ss;
    ss << "| ";
    for (i = 0; i < input.size(); i++)
        if (input[i])
            ss << i << ":" << input[i] << delimiter;
    return ss.str();
}

string join(vector<double>& input, string delimiter) {
    int i;
    stringstream ss;
    ss << "| ";
    for (i = 0; i < input.size(); i++)
        ss << i << ":" << input[i] << delimiter;
    return ss.str();
}

/* Prediction functions */
vector<double>& MnistLogisticRegression::predict_bytes(
        vector< vector<char> >& input) {
    int i;
    example *ec;
    predictions.resize(input.size());
    for (i = 0; i < input.size(); i++) {
        ec = VW::read_example(*model, join(input[i], " "));
        model->l->predict(*ec);
        if (ec->pred.multiclass == 10)
            ec->pred.multiclass = 0;
        predictions[i] = ec->pred.multiclass;
        VW::finish_example(*model, ec);
    }
    return predictions;
}

vector<double>& MnistLogisticRegression::predict_floats(
        vector< vector<double> >& input) {
    int i;
    example *ec;
    predictions.resize(input.size());
    for (i = 0; i < input.size(); i++) {
        ec = VW::read_example(*model, join(input[i], " "));
        model->l->predict(*ec);
        if (ec->pred.multiclass == 10)
            ec->pred.multiclass = 0;
        predictions[i] = ec->pred.multiclass;
        VW::finish_example(*model, ec);
    }
    return predictions;
}

vector<double>& MnistLogisticRegression::predict_ints(
        vector< vector<uint32_t> >& input) {
    int i;
    example *ec;
    predictions.resize(input.size());
    for (i = 0; i < input.size(); i++) {
        ec = VW::read_example(*model, join(input[i], " "));
        model->l->predict(*ec);
        if (ec->pred.multiclass == 10)
            ec->pred.multiclass = 0;
        predictions[i] = ec->pred.multiclass;
        VW::finish_example(*model, ec);
    }
    return predictions;
}

vector<double>& MnistLogisticRegression::predict_strings(
        vector<vector<std::string> >& input) {
    int i;
    predictions.resize(input.size());
    for (i = 0; i < input.size(); i++) {
        predictions[i] = input[i].size();
    }
    return predictions;
}

MnistLogisticRegression::MnistLogisticRegression(string model_path) :
        path(model_path) {
    model = VW::initialize("-i " + path);
}

int main() {
    std::unique_ptr<Model> model(
        new MnistLogisticRegression("mnist/mnist.model"));
    ClipperRPC *clipper_rpc_server =
        new ClipperRPC(model, (char *) "127.0.0.1", 6001);
    clipper_rpc_server->serve_forever();
}
