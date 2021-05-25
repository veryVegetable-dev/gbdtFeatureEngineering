#include "BoostingModel.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstdlib>



std::pair<int/*class_id*/, float/*score*/> DecisionTreeModel::getClassScore(const std::vector<float> &features) {
    if (node_mp->empty()) return std::make_pair<int, float>(-1, 0);
    int feature_dim = features.size();
    TreeNode node = (*node_mp)[0];
    while (node.left_node_id >= 0 && node.right_node_id >= 0) {
        if (node.split_feature_id >= feature_dim) return std::make_pair<int, float>(-1, 0);
        if (features[node.split_feature_id] < node.threshold)
            node = (*node_mp)[node.left_node_id];
        else
            node = (*node_mp)[node.right_node_id];
    }
    return std::make_pair<int, float>(reinterpret_cast<int &&>(class_id), reinterpret_cast<float &&>(node.score));
}


void BoostingModel::predict(const std::vector<float>& features, std::vector<float> *class_scores) {
    if (_num_class < class_scores->size()) return;
    for (auto &score: *class_scores) score = 0;
    for (auto &tree : *_trees_ptr) {
        std::pair<int, float> pred = tree.getClassScore(features);
        if (pred.first < 0) continue;
        (*class_scores)[pred.first] += pred.second;
    }
}


void BoostingModel::parseLine(const char *line, std::vector<char *> &fields) {
    fields.clear();
    char *res = nullptr;
    while ((res = strsep((char **)&line, "\t")))
        fields.emplace_back(res);
}


void BoostingModel::loadModel(const std::string &model_path) {
    std::ifstream in_file(model_path);
    const int MAX_LINE_LEN = 100;
    char buff[MAX_LINE_LEN];
    std::vector<char*> fields;
    std::shared_ptr<DecisionTreeModel> dt;
    int tree_id = -1;
    while (in_file.getline(buff, MAX_LINE_LEN)) {
        parseLine(buff, fields);
        int current_tree_id = atoi(fields[0]);
        if (fields.size() > 1 && current_tree_id != tree_id) {
            if (tree_id >= 0) _trees_ptr->emplace_back(*dt);
            dt = std::shared_ptr<DecisionTreeModel>(new DecisionTreeModel());
            tree_id = current_tree_id;
            dt->setClassId(atoi(fields[1]));
        }
        std::shared_ptr<TreeNode> node_ptr;
        if (fields.size() == 8) {
            int node_id = atoi(fields[2]);
            int feature_id = atoi(fields[3]);
            float threshold = atof(fields[4]);
            int left_id = atoi(fields[5]);
            int right_id = atoi(fields[6]);
            node_ptr= std::make_shared<TreeNode>(TreeNode(node_id, feature_id, threshold, left_id, right_id));
        } else if (fields.size() == 4) {
            int node_id = atoi(fields[2]);
            float score = atof(fields[3]);
            node_ptr = std::make_shared<TreeNode>(TreeNode(node_id, score));
        }
        dt->insertNode(*node_ptr);
    }
    _trees_ptr->emplace_back(*dt);
    in_file.close();
}