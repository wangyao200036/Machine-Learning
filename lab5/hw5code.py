import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator

def find_best_split(feature_vector, target_vector):
    # 对特征值进行排序，并同时对目标值进行相应的排序
    sorted_indices = np.argsort(feature_vector)
    sorted_feature_vector = feature_vector[sorted_indices]
    sorted_target_vector = target_vector[sorted_indices]

    # 初始化变量
    thresholds = []
    ginis = []

    # 计算总的样本数和正类样本数
    total_objects = len(feature_vector)
    total_positive = np.sum(sorted_target_vector)

    # 遍历特征值，生成候选分割点
    for i in range(1, total_objects):
        if sorted_feature_vector[i] != sorted_feature_vector[i - 1]:
            threshold = (sorted_feature_vector[i - 1] + sorted_feature_vector[i]) / 2
            left_positive = np.sum(sorted_target_vector[:i])
            right_positive = total_positive - left_positive
            left_size = i
            right_size = total_objects - i

            # 计算左子集和右子集的吉尼不纯度
            if left_size > 0:
                left_gini = 1 - (left_positive / left_size) ** 2 - ((left_size - left_positive) / left_size) ** 2
            else:
                left_gini = 0

            if right_size > 0:
                right_gini = 1 - (right_positive / right_size) ** 2 - ((right_size - right_positive) / right_size) ** 2
            else:
                right_gini = 0

            # 计算总的吉尼增益
            gini = (left_size / total_objects) * left_gini + (right_size / total_objects) * right_gini

            # 存储候选分割点和对应的吉尼不纯度
            thresholds.append(threshold)
            ginis.append(gini)

    # 将列表转换为 NumPy 数组
    thresholds = np.array(thresholds)
    ginis = np.array(ginis)

    # 找到最优的分割点
    if len(ginis) > 0:
        best_index = np.argmin(ginis)  # 选择最小的吉尼不纯度
        threshold_best = thresholds[best_index]
        gini_best = ginis[best_index]
    else:
        threshold_best = None  # 没有有效分割点时返回 None
        gini_best = np.inf  # 使用一个较大的初始值

    return thresholds, ginis, threshold_best, gini_best
    
class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]) or \
           (self._max_depth is not None and depth >= self._max_depth) or \
           (self._min_samples_split is not None and sub_X.shape[0] < self._min_samples_split) or \
           (self._min_samples_leaf is not None and sub_X.shape[0] < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, np.inf, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array([categories_map.get(x, -1) for x in sub_X[:, feature]])

            if len(feature_vector) <= 1:
                continue

            # 调用 find_best_split 函数
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
            # 确保 gini 不为 None
            if gini is not None and gini < gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [key for key, value in categories_map.items() if value < threshold]
                    if not threshold_best:
                        threshold_best = [sorted_categories[0]]  # 确保至少有一个类别

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        else:
            feature_split = node["feature_split"]
            if self._feature_types[feature_split] == "real":
                if x[feature_split] < node["threshold"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            elif self._feature_types[feature_split] == "categorical":
                if x[feature_split] in node["categories_split"]:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            else:
                raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)