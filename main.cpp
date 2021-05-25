#include <iostream>
#include "BoostingModel.h"
#include <vector>
#include <fstream>
#include <unordered_map>


namespace Eval {
    struct Sample {
        int label;
        std::vector<float> features;
    };

    int parseSample(const char *line, const std::unordered_map<std::string, int> &label_array, Sample *sample) {
        if (!line) return 0;
        if (line[0] == 'A') return 0;
        int i = 0;
        char *field;
        while ((field = strsep((char **) &line, "\t"))) {
            std::string field_str(field);
            if (i == 16 && label_array.find(field_str) != label_array.end())
                sample->label = label_array.find(field_str)->second;
            else if (i > 0 && i < 16)
                sample->features.emplace_back(atof(field));
            i++;}
        return 1;
    }

    int findMax(const std::vector<float> &pred) {
        int n = pred.size();
        if (n == 0) return -1;
        int i = 0;
        for (int j = 1; j < n; j++) {
            if (pred[j] > pred[i]) i = j;
        }
        return i;
    }
}


int main() {
    const int NUM_CLASS = 7;
    const std::string& model_path = "/Users/menglingwu/Documents/2021/coding/personal_projects/boosting_spark/xgboost.trees.parsed";
    const std::string& data_path = "/Users/menglingwu/Documents/2021/coding/personal_projects/dataset/DryBeanDataset/Dry_Bean_Dataset.txt";

    // 加载模型
    BoostingModel bm = BoostingModel::Get()->Init(model_path, NUM_CLASS);

    // label映射关系
    std::unordered_map<std::string, int> labels_mp;
    std::vector<std::string> labels_array = {"DERMASON", "SIRA", "SEKER", "HOROZ", "CALI", "BARBUNYA", "BOMBAY"};
    int n = labels_array.size();
    for (int i = 0; i < n; ++i) {
        labels_mp[labels_array[i]] = i;
    }

    // 加载测试数据
    std::ifstream eval_file(data_path);
    const int MAX_LEN = 1000;
    char buffer[MAX_LEN];
    int error_cnt = 0, cnt = 0;
    while ((eval_file.getline(buffer, MAX_LEN))) {
        Eval::Sample sample;
        if (!Eval::parseSample(buffer, labels_mp, &sample)) continue;
        std::vector<float> pred = std::vector<float>(NUM_CLASS);
        bm.predict(sample.features, &pred);
        if (sample.label != Eval::findMax(pred)) error_cnt++;
        cnt++;
    }
    std::cout << error_cnt << "," << cnt << "," << (float)error_cnt / (float)cnt << std::endl;
    return 0;
}