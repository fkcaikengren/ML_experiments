
import numpy as np

class TNode:
    def __init__(self, feat_index=None, entropy=None, num_sample=None, label=None):
        self.feat_index = feat_index #特征索引（列序号）
        self.entropy = entropy #熵
        self.num_sample = num_sample #样本数量
        self.feat_children = {} # 基于特征的分支（孩子）
        self.label = label #分类标签

    
            
class DescisionTree:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        self.unique_labels = np.unique(labels)
        self.tree = None

    def train(self):
        # 创建树
        root_entropy = DescisionTree.compute_entropy(self.data, self.labels)
        self.tree = DescisionTree.create_tree(self.data, self.labels, root_entropy)

    
    def predict(self, data):
        return [DescisionTree.find_label(self.tree, cur_data) for cur_data in data]

    @staticmethod
    def find_label(node,data):
        if not node.feat_children:
            return node.label
        else :
            feat_value = data[node.feat_index]
            if type(feat_value) != type(""):
                feat_value = str(feat_value)
            return DescisionTree.find_label(node.feat_children[feat_value] ,data)

    @staticmethod
    def create_tree(data, labels, entropy):
        # 终止分支
        if data is None or data.size == 0 :
            return None
        if np.unique(labels).size == 1:
            return TNode(None, 0, labels.shape[0], label=labels[0])

        best_feat_index, best_gain, sub_data_list, sub_labels_list, sub_entropy_list, best_feat_values = DescisionTree.choose_best_feat(data, labels)
        node = TNode(best_feat_index, entropy, data.shape[0])

        for sub_data, sub_labels, ent, value in zip(sub_data_list, sub_labels_list, sub_entropy_list, best_feat_values):
        
            # print(f"v: {value}")
            # print(" -----")
            sub_node = DescisionTree.create_tree(sub_data, sub_labels, ent)
            if sub_node:
                node.feat_children[value] = sub_node
            else :
                node.feat_children[value] = TNode(None,
                                                  ent, 
                                                  sub_data.shape[0],
                                                  label=DescisionTree.majority_label(labels))

        return node

    @staticmethod
    def choose_best_feat(data, labels):
        # 遍历所有特征，计算其信息增益(率)
        best_feat_index = -1
        best_gain = 0
        best_sub_data_list = None
        best_sub_labels_list = None
        best_sub_entropy_list = None
        best_feat_values = None
        for col_idx,col in enumerate(data.T): #取列
            unique_values = np.unique(col)
            base_entropy = DescisionTree.compute_entropy(data, labels)
            sub_entropy_mean = 0
            
            # gain_ratio = 0
            sub_data_list = []
            sub_labels_list = []
            sub_entropy_list = []
            
            for u_value in unique_values:
                sub_data = np.array(
                    [row for row in data if u_value == row[col_idx]]
                )
                sub_labels = np.array(
                    [labels[row_idx] for row_idx,row in enumerate(data) if u_value == row[col_idx]]
                )
                sub_entropy = DescisionTree.compute_entropy(sub_data, sub_labels)
                sub_data_list.append(sub_data)
                sub_labels_list.append(sub_labels)
                sub_entropy_list.append(sub_entropy)
                sub_entropy_mean += sub_entropy*(sub_data.shape[0]/data.shape[0])
            gain = base_entropy - sub_entropy_mean
            if gain > best_gain:
                best_gain = gain
                best_feat_index = col_idx
                best_sub_data_list = sub_data_list
                best_sub_labels_list = sub_labels_list
                best_sub_entropy_list = sub_entropy_list
                best_feat_values = unique_values
            
            # print(f"{col_idx}: {best_gain}")
        # 删去列
        best_sub_data_list = [np.delete(_data, [best_feat_index], axis=1) for _data in best_sub_data_list]
    
        return best_feat_index, best_gain, best_sub_data_list, best_sub_labels_list, best_sub_entropy_list, best_feat_values

    @staticmethod
    def compute_entropy(data, labels):
        unique_labels = np.unique(labels)
        n_samples = labels.shape[0]
        sum = 0.0
        for u_label in unique_labels:
            m = data[labels == u_label].shape[0]
            p = m/n_samples
            sum -= (p*np.log2(p))
        return sum
    
    @staticmethod
    def compute_gain_ratio(data, sub_data):
        # 使用增益率+启发式优化，来分支决策
        pass

    @staticmethod
    def majority_label(labels):
        label_dict = {}
        for label in labels:
            if label in label_dict:  #有key
                label_dict[label] += 1
            else:
                label_dict[label] = 1
        max_label = None
        max_cnt = 0
        for label,cnt in label_dict.items():
            if cnt >= max_cnt:
                max_cnt = cnt
                max_label = label
        return max_label



