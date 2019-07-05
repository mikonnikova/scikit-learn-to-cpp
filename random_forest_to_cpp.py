# -*- coding: utf-8 -*-
#
# Transform sklearn.ensemble.RandomForestClassifier model into C++ function
# Creates two files:
#   .h with inline decision trees function
#   .cpp with random_forest(std::vector<float> feature_vector) function to be used in C++ code
# Can be used as a standalone script or imported into Python code
#
# Decision tree part inspired by https://github.com/papkov/DecisionTreeToCpp


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
        code += '%sreturn %s;\n' % (spaces*' ', tree.tree_.value[node].argmax())  # class label
    return code


# get one tree into C++ function format
def one_tree(tree, data_type, number=0):
    code = "inline int %s (const std::vector<%s> & feature_vector) {\n%s}" % \
           ("tree" + str(number), data_type, branch(tree, 0, 4))
    return code


# get forest prediction part into C++ code
def prediction_function(number_of_trees, number_of_classes, data_type):
    trees = ''
    for i in range(number_of_trees):
        trees += ('*tree' + str(i) + ', ')

    main = """
int random_forest(std::vector<%s> feature_vector) {
    int number = %s;
    int classes = %s;

    tree tree_functions[number] = {%s};

    std::vector<float> pred (classes, 0);
    for (int i = 0; i < number; i++) {
        pred[tree_functions[i](feature_vector)] ++;
    }

    int answer = 0;
    float value = 0;

    for (int i = 0; i < classes; i++) {
        pred[i] /= number;
        if (pred[i] > value) {
            value = pred[i];
            answer = i;
        }
    }

    return answer;
}
""" % (data_type, number_of_trees, number_of_classes, trees[:-2])

    return main


# create <filename.h> with decision trees functions as inline functions
# create <filename.cpp> with Random Forest classification function
def random_forest_to_cpp(forest, data_type='float', filename='forest'):
    with open(filename + '.h', 'w') as f:
        print('#ifndef FOREST_H\n#define FOREST_H', file=f)
        print('#include <vector>\n', file=f)
        print('typedef int (*tree) (const std::vector<%s> & feature_vector);\n' % data_type, file=f)

        for number in range(len(forest)):
            print(one_tree(forest[number], data_type, number) + '\n', file=f)

        print('int random_forest(std::vector<%s> feature_vector);\n' % data_type, file=f)
        print('#endif', file=f)

    with open(filename + '.cpp', 'w') as f:
        print('#include "forest.h"', file=f)
        print(prediction_function(len(forest), len(forest[0].tree_.value[0][0]) + 1, data_type), file=f)

    print("Forest saved")


# standalone execution
# transforms classifier in pickle format into C++ code
if __name__ == "__main__":
    import pickle
    import sys

    if len(sys.argv) == 1:
        print('Need parameters for standalone execution\nType --help for details')
        exit(1)
    file = sys.argv[1]

    if file == '--help':
        print('Parameters are:')
        print('1) Name of pickle file to read Random Forest classifier from')
        print('2) Name of file to save Random Forest classifier in C++ format to [optional, default=forest]')
        print('3) Features data type [optional, default=float]')
        exit(0)

    if len(sys.argv) > 2:
        new_file = sys.argv[2]
    else:
        new_file = "forest"

    if len(sys.argv) > 3:
        data_type = sys.argv[3]
    else:
        data_type = 'float'

    with open(file, 'rb') as ff:
        clf = pickle.load(ff)
    random_forest_to_cpp(clf, data_type, new_file)
