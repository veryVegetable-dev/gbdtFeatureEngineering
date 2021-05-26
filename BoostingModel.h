#ifndef BOOSTING_PREDICT_SERVER_BOOSTINGMODEL_H
#define BOOSTING_PREDICT_SERVER_BOOSTINGMODEL_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>


struct TreeNode {
    TreeNode(int _node_id, int _feature_id, float _threshold, int _left_node_id, int _right_node_id):
            node_id(_node_id), split_feature_id(_feature_id), threshold(_threshold),
            score(-1),
            left_node_id(_left_node_id), right_node_id(_right_node_id) {};
    TreeNode(int _node_id, float _score):
        node_id(_node_id), split_feature_id(-1), threshold(-1),
        score(_score),
        left_node_id(-1), right_node_id(-1) {};
    TreeNode(){};
    int node_id;
    int split_feature_id;
    float threshold;
    float score;
    int left_node_id;
    int right_node_id;
};


class DecisionTreeModel {

public:
    struct PredictionInfo {
        PredictionInfo(): class_id(-1), score(-1), leaf_id(-1)  {}
        int class_id;
        float score;
        size_t leaf_id;
    };

    DecisionTreeModel(): class_id(-1) {
        node_mp = std::shared_ptr<std::unordered_map<int, TreeNode> >(new std::unordered_map<int, TreeNode>());
        leaf_mp_ptr = std::shared_ptr<std::unordered_map<size_t , size_t> >(new std::unordered_map<size_t , size_t>());
    }
    void setClassId(int _class_id) {class_id = _class_id; };
    void insertNode(const TreeNode &node) { (*node_mp)[node.node_id] = node; }
    void addLeaf(int node_id) {
        if (leaf_mp_ptr->find(node_id) == leaf_mp_ptr->end()) {
            (*leaf_mp_ptr)[node_id] = leaf_mp_ptr->size();
        }
    }
    DecisionTreeModel::PredictionInfo predict(const std::vector<float> &features) const ;
    size_t numLeaf() const {return leaf_mp_ptr->size(); }
private:
    std::shared_ptr<std::unordered_map<int, TreeNode> > node_mp;
    int class_id;
    std::shared_ptr<std::unordered_map<size_t , size_t> > leaf_mp_ptr;
};


class BoostingModel {
public:
    static BoostingModel* Get() {
        static BoostingModel bm;
        return &bm;
    }
    BoostingModel(): _num_class(0), _trees_ptr(nullptr) {}
    BoostingModel(BoostingModel const&) = delete;
    void operator=(BoostingModel const&) = delete;
    void Init(const std::string& model_path, int num_class) {
        _num_class = num_class;
        _trees_ptr = std::shared_ptr<std::vector<DecisionTreeModel> >(new std::vector<DecisionTreeModel>());
        loadModel(model_path);
    }
    void predict(const std::vector<float>& features, std::vector<float>* class_scores, std::vector<int>* transformed_features);
    size_t getLeafNum();
private:
    static void parseLine(const char* line, std::vector<char*>& fields);
    void loadModel(const std::string& model_path);
    std::shared_ptr<std::vector<DecisionTreeModel> > _trees_ptr;
    int _num_class;
};


#endif //BOOSTING_PREDICT_SERVER_BOOSTINGMODEL_H
