#include "BoostingModel.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstdlib>



DecisionTreeModel::PredictionInfo DecisionTreeModel::predict(const std::vector<float> &features) const {
    DecisionTreeModel::PredictionInfo predictionInfo;
    if (node_mp->empty()) return predictionInfo;
    int feature_dim = features.size();
    TreeNode node = (*node_mp)[0];
    while (node.left_node_id >= 0 && node.right_node_id >= 0) {
        if (node.split_feature_id >= feature_dim) return predictionInfo;
        if (features[node.split_feature_id] < node.threshold)
            node = (*node_mp)[node.left_node_id];
        else
            node = (*node_mp)[node.right_node_id];
    }
    predictionInfo.class_id = class_id;
    predictionInfo.score = node.score;
    predictionInfo.leaf_id = (*leaf_mp_ptr)[node.node_id];
    return predictionInfo;
}


void BoostingModel::predict(const std::vector<float>& features, std::vector<float> *class_scores, std::vector<int>* transformed_features) {
    if (_num_class < class_scores->size()) return;
    for (auto &score: *class_scores) score = 0;
    size_t num_trees = _trees_ptr->size();
    for (size_t i = 0; i < num_trees; i++) {
        const DecisionTreeModel &tree = (*_trees_ptr)[i];
        std::vector<int> one_tree_feature(tree.numLeaf(), 0);
        const DecisionTreeModel::PredictionInfo &predict_info = tree.predict(features);
        if (predict_info.class_id < 0) continue;
        (*class_scores)[predict_info.class_id] += predict_info.score;
        one_tree_feature[predict_info.leaf_id] = 1;
        for (const auto &e: one_tree_feature)
            transformed_features->emplace_back(e);
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
            dt->addLeaf(node_id);
        }
        dt->insertNode(*node_ptr);
    }
    _trees_ptr->emplace_back(*dt);
    in_file.close();
}

size_t BoostingModel::getLeafNum() {
    size_t res = 0;
    for (const auto &tree : *_trees_ptr)
        res += tree.numLeaf();
    return res;
}