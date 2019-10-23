# -*- coding: utf-8 -*-
#
# Transform sklearn.ensemble.AdaBoostClassifier model into C++ function
# Creates two files:
#   .h with inline decision trees function
#   .cpp with adaboost(std::vector<T> feature_vector) function to be used in C++ code
# Can be used as a standalone script or imported into Python code
#
# Decision tree part inspired by https://github.com/papkov/DecisionTreeToCpp

import numpy as np


def normalise(proba):
    normalizer = proba.sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.0] = 1.0
    proba /= normalizer
    return proba


def transform_to_vector(node):
    node = normalise(node)[0]
    values = ''
    for value in node:
        values += (str(value) + ', ')
    vector = 'std::vector<float> {%s}' % values[:-2]
    return vector


# work recursively on branches in decision tree
def branch(tree, node, spaces):
    code = ''
    if tree.tree_.threshold[node] != -2:
        code += '%sif (feature_vector.at(%s) <= %s) {\n' % (spaces*' ',
                                                            tree.tree_.feature[node],
                                                            tree.tree_.threshold[node])
        if tree.tree_.children_left[node] != -1:
            code += branch(tree, tree.tree_.children_left[node], spaces+4)  # more ifs
        code += '%s}\n%selse {\n' % (spaces*' ', spaces*' ')
        if tree.tree_.children_right[node] != -1:
            code += branch(tree, tree.tree_.children_right[node], spaces+4)  # else info
        code += '%s}\n' % (spaces*' ')
    else:
        code += '%sreturn %s;\n' % (spaces*' ', transform_to_vector(tree.tree_.value[node]))  # probability distribution
    return code


# get one tree into C++ function format
def one_tree(tree, data_type, number=0):
    code = "std::vector<float>  %s (const std::vector<%s> & feature_vector) {\n%s}" % \
           ("tree" + str(number), data_type, branch(tree, 0, 4))
    return code


# get forest prediction part into C++ code
def prediction_function(number_of_trees, number_of_classes, data_type):
    trees = ''
    for i in range(number_of_trees):
        trees += ('    tree_functions[%s] = tree%s;\n' % (i, i))

    main = """
std::vector<float> log_proba(std::vector<float> pred) {
    for (int i = 0; i < pred.size(); i++) {
        if (pred[i] == 0) {
            pred[i] = 2.220446049250313e-16;
        }
        pred[i] = std::log(pred[i]);
    }
    return pred;
}

std::vector<float> samme_proba(std::vector<float> proba, int num_classes) {
    float sum = std::accumulate(proba.begin(), proba.end(), 0.0, std::plus<float>());
    sum /= num_classes;
    for (int i = 0; i < proba.size(); i++) {
    proba[i] -= sum;
        proba[i] *= (num_classes - 1);
    }
    return proba;
}

int adaboost(std::vector<%s> feature_vector) {
    int number = %s;
    int classes = %s;

    tree tree_functions[number];
%s

    std::vector<float> res(classes, 0.0);
    for (int i = 0; i < number; i++) {
        std::transform (res.begin(), res.end(), 
                        samme_proba(log_proba(tree_functions[i](feature_vector)), classes).begin(), 
                        res.begin(), std::plus<float>());
    }

    int max = std::numeric_limits<int>::min();
    int answer = -1;
    for (int i = 0; i < res.size(); i++) {
        res[i] /= number;
        if (res[i] > max) {
            max = res[i];
            answer = i;
        }
    }

    return answer;
}

""" % (data_type, number_of_trees, number_of_classes, trees)

    return main


# create <filename.h> with decision trees functions as inline functions
# create <filename.cpp> with AdaBoost classification function
def adaboost_to_cpp(adaboost, data_type='int', filename='adaboost'):

    with open(filename + '.h', 'w') as f:
        print('#ifndef ADABOOST_H\n#define ADABOOST_H', file=f)
        print('#include <vector>\n#include <limits.h>\n#include <cmath>\n#include <algorithm>', file=f)
        print('typedef std::vector<float> (*tree) (const std::vector<%s> & feature_vector);\n' % data_type, file=f)

        for number in range(len(adaboost)):
            print(one_tree(adaboost[number], data_type, number) + '\n', file=f)

        print('int random_forest(std::vector<%s> feature_vector);\n' % data_type, file=f)
        print('#endif', file=f)

    with open(filename + '.cpp', 'w') as f:
        print('#include "%s.h"' % filename, file=f)
        print(prediction_function(len(adaboost), len(adaboost[0].tree_.value[0][0]) + 1, data_type), file=f)

    print("Classifier saved")


# standalone execution
# transforms classifier in pickle format into C++ code
if __name__ == "__main__":
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description='Save pickled AdaBoost classifier in C++ format')
    parser.add_argument('file', type=str, help='Name of pickle file to read AdaBoost classifier from')
    parser.add_argument('-t', '--data_type', default='float', help='Features data type (default:float)')
    parser.add_argument('-f', '--new_file', default='adaboost',
                        help='Name of file to save AdaBoost classifier in C++ format to (default:adaboost)')

    args = parser.parse_args()
    with open(args.file, 'rb') as ff:
        clf = pickle.load(ff)
    adaboost_to_cpp(clf, args.data_type, args.new_file)
