import networkx as nx
import torch

class Metric():
    def __init__(self,label_set,son2parent):
        
        self.label_set = label_set # 这是sort过的label_set
        
        self.son2parent = son2parent
        self.G = nx.Graph()
        for son,parent in self.son2parent.items():
            self.G.add_edge(son, parent) #　添加节点并链接节点
        
        shortest_path_gen = nx.all_pairs_shortest_path(self.G) # 最短路存在一个generator数据结构中
        self.shortest_path_dict = dict(shortest_path_gen)
        self.hop_matrix = self._get_hops_matrix() # n_cls x n_cls Matrix, whose (i,j) store hops from label i to j.
    
    def _get_hops_matrix(self):
        n_cls = len(self.label_set)
        hop_matrix = torch.zeros(n_cls,n_cls)
        for i in range(n_cls):
            for j in range(n_cls):
                source, target = self.label_set[i], self.label_set[j]
                path = self.shortest_path_dict[source][target]
                hop_matrix[i,j] = len(path) - 1
        return hop_matrix   
            
    
    def hop_acc(self,ypred,ybatch, hops):
        
        correct_count = 0
        
        for i in range (len(ypred)):
            source, target = self.label_set[ypred[i]], self.label_set[ybatch[i]]
            path = self.shortest_path_dict[source][target] # 最短路
            if len(path) - 1 <= hops: # 完全对的时候path中只有一个元素，就是自己
                correct_count = correct_count + 1
   
        return correct_count / len(ybatch)
    
    def weight_acc(self, ypred, ybatch,parent_weight):     

        correct_ness_list = []
        
        for i in range (len(ypred)):
            source, target = self.label_set[ypred[i]], self.label_set[ybatch[i]]
            count = 0
            if source != target:
                count = 1
            while self.son2parent[source] != self.son2parent[target]:
                count = count + 1
                source, target = self.son2parent[source], self.son2parent[target]
            
            correct_ness = 1 if count == 0 else sum([parent_weight**(i+1) for i in range(count)]) ## 所有父辈的权重
            correct_ness_list.append(correct_ness)
        
        return sum(correct_ness_list) / len(ybatch)
    
    def hop_mAP(self, ypred_topk, ybatch, hop = 0):
        # ypred_topk: n x k
        # ybatch: n x 1
        n,k = ypred_topk.size(0), ypred_topk.size(1)

        # correct_inx = (ypred == ybatch) # broadcast automatically to n x k
        correct_inx = torch.zeros(n,k)
        for i in range(n):
            x_pos = ybatch[i] # 1 x 1
            y_pos = ypred_topk[i] # 1 x k
            hop_dist = self.hop_matrix[x_pos, y_pos] # pred_i与ybatch_i 之间的距离
            correct_inx[i,:] = (hop_dist <= hop) # i号ground truth所能认为正确的所有类别，其列号标为1

        numerator = [correct_inx[:,:i+1].sum(dim=1) for i in range(k)]
        numerator = torch.stack(numerator).t()
        denominator = torch.arange(1,k+1).repeat(n,1)
        P = numerator.float() / denominator.float()
        AP = P.mean(dim=1,keepdim=True)

        return AP.mean()