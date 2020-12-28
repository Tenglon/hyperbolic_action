import networkx as nx
import torch

class Metric():
    def __init__(self,label_set,son2parent):
        
        self.label_set = label_set # Make sure that the label is sorted as we did in data preparation.
        
        self.son2parent = son2parent
        self.G = nx.Graph()
        for son,parent in self.son2parent.items():
            self.G.add_edge(son, parent) #　Add an node and corresponding edges on the hierarchy tree
        
        shortest_path_gen = nx.all_pairs_shortest_path(self.G) # Store the shortest path for fast retrieval
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
            path = self.shortest_path_dict[source][target] # shortest path on the hierarchy
            if len(path) - 1 <= hops: # if the predction is the same class as ybatch (In contrast to pred and y are sliblings or cousins or other)
                correct_count = correct_count + 1
   
        return correct_count / len(ybatch)
    
    def hop_mAP(self, ypred_topk, ybatch, hop = 0):
        # ypred_topk: n x k
        # ybatch: n x 1
        n,k = ypred_topk.size(0), ypred_topk.size(1)

        # correct_inx = (ypred == ybatch) # broadcast automatically to n x k
        correct_inx = torch.zeros(n,k)
        for i in range(n):
            x_pos = ybatch[i] # 1 x 1
            y_pos = ypred_topk[i] # 1 x k
            hop_dist = self.hop_matrix[x_pos, y_pos] # Get the graph distance (by hop) between pred_i and ybatch_i
            correct_inx[i,:] = (hop_dist <= hop) # For all predictions whose hop distance smaller than hop, regard it as a correct prediction. (Acc 0 hop, Sibling 2 hops, Cousin 4 hops)  

        numerator = [correct_inx[:,:i+1].sum(dim=1) for i in range(k)]
        numerator = torch.stack(numerator).t()
        denominator = torch.arange(1,k+1).repeat(n,1)
        P = numerator.float() / denominator.float()
        AP = P.mean(dim=1,keepdim=True)

        return AP.mean()
