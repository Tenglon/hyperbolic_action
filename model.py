import torch.nn as nn
import torch.nn.functional as F
import torch

def loss_fn(support, query, dist_func, c, T):
    #Here we use synthesised support.
    logits = -dist_func(support,query,c) / T
    fewshot_label = torch.arange(support.size(0)).cuda()
    loss = F.cross_entropy(logits, fewshot_label)
    
    return loss

class RegressNet(nn.Module):
    def __init__(self, T, c, dist_func, eval_dist, train_loader, val_loader, emb, metric):
        super(RegressNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, emb.size(1))
        )
        self.T = T
        self.c = c
        self.dist_func = dist_func
        self.eval_dist = eval_dist
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.emb = emb
        self.metric = metric
        
    def forward(self, x):
        x = self.layers(x)
        if self.dist_func.__name__ == 'pair_wise_hyp':
            if (x.norm(dim=1) >= 1).sum() != 0:
                x = x / (x.norm(dim=1,keepdim=True) + 1e-2)
        return x

    def _train(self, optimizer, epochs, T, eval_interval):
        model = self
        model.train()
        for epoch in range(epochs):

            for i, (xbatch, ybatch, abatch) in enumerate(self.train_loader):

                xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
                optimizer.zero_grad()
                apred = model(xbatch)

                loss = loss_fn(apred, abatch, self.dist_func, self.c, self.T)

                loss.backward()
                optimizer.step()

            if epoch % eval_interval == 0:
                self.evaluation(flag = 'valid')
                self.evaluation2()

    def eval_zsl(self, unseen_idset):
        model = self
        unseen_idset = torch.tensor(unseen_idset).cuda()
        unseen_emb = self.emb[unseen_idset,:]
        
        # 0.query_candidates & GT
        aval_pred_list, yval_list = [], []
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):
            xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
            apred = model(xbatch)
            aval_pred_list.append(apred)
            yval_list.append(ybatch)
        aval_pred = torch.cat(aval_pred_list) # search candidates
        yval_order1 = torch.cat(yval_list) # ground truth in search candidates's order.
        
        GT_list, loss_list, ypred_list, ypred_topk_list = [], [], [], []
        
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):
            xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
            apred = model(xbatch)
            # 1.loss
            loss = loss_fn(apred, abatch, self.dist_func, self.c, self.T).item()
            loss_list.append(loss)
            # 2.ypred for recognition
            dist = self.eval_dist(apred, unseen_emb, self.c)
            rank = dist.sort()[1]
            top1_rank = rank[:,0]
            ypred = unseen_idset[top1_rank]
            
            # 3.ypred_topk for retrieval
            dist_retrieval = self.eval_dist(apred, aval_pred, self.c)
            rank = dist_retrieval.sort()[1]
            topk_result_inx = rank[:,:50] # k = 50
            ypred_topk = yval_order1[topk_result_inx]
            ypred_topk_list.append(ypred_topk)
            ypred_list.append(ypred)
            GT_list.append(ybatch)
        GT = torch.cat(GT_list)
        ypred = torch.cat(ypred_list)
        ypred_topk = torch.cat(ypred_topk_list)
        
        loss = torch.tensor(loss_list).mean()
        hop0_acc = (ypred == GT).float().mean().item()
        hop1_acc = self.metric.hop_acc(ypred,GT, hops = 2)
        hop2_acc = self.metric.hop_acc(ypred,GT, hops = 4)
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)
        print('loss :%.3f acc:%.3f, 1hop_acc:%.3f, 2hop_acc:%.3f, mAP:%.3f, 1hop_mAP:%.3f, 2hop_mAP:%.3f'%(loss, hop0_acc, hop1_acc, hop2_acc, hop0_mAP, hop1_mAP, hop2_mAP))
    
    def evaluation(self, flag):
        model = self

        # 0.query_candidates & GT
        aval_pred_list, yval_list = [], []
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):
            xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
            apred = model(xbatch)
            aval_pred_list.append(apred)
            yval_list.append(ybatch)
        aval_pred = torch.cat(aval_pred_list) # search candidates
        yval_order1 = torch.cat(yval_list) # ground truth in search candidates's order.
        
        GT_list = [] # ground truth in validation order
        loss_list, ypred_list, ypred_topk_list = [],[],[]
        torch.manual_seed(42)
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):
            xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
            apred = model(xbatch)
            # 1.loss
            loss = loss_fn(apred, abatch, self.dist_func, self.c, self.T).item()
            loss_list.append(loss)
            # 2.ypred for recognition
            dist = self.eval_dist(apred, self.emb, self.c)
            rank = dist.sort()[1]
            ypred = rank[:,0]
            ypred_list.append(ypred)
            # 3.ypred_topk for retrieval
            dist_retrieval = self.eval_dist(apred, aval_pred, self.c)
            rank = dist_retrieval.sort()[1]
            topk_result_inx = rank[:,:50] # k = 50
            ypred_topk = yval_order1[topk_result_inx]
            ypred_topk_list.append(ypred_topk)
            GT_list.append(ybatch)
        GT = torch.cat(GT_list)
        
        loss = torch.tensor(loss_list).mean()
        ypred = torch.cat(ypred_list)
        ypred_topk = torch.cat(ypred_topk_list)
        
        hop0_acc = (ypred == GT).float().mean().item()
        hop1_acc = self.metric.hop_acc(ypred,GT, hops = 2)
        hop2_acc = self.metric.hop_acc(ypred,GT, hops = 4)           
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)

        print('loss:%.3f,acc:%.3f,1hop_acc:%.3f,2hop_acc:%.3f,mAP:%.3f,1hop_mAP:%.3f,2hop_mAP:%.3f'%
              (loss, hop0_acc, hop1_acc, hop2_acc, hop0_mAP, hop1_mAP, hop2_mAP))
    
    def evaluation2(self):
        model = self
        torch.manual_seed(42)

        # 0.query_candidates & GT
        avalpred_list, yval_list = [], []
        for i, (xbatch, ybatch, _) in enumerate(self.val_loader):
            xbatch, ybatch = xbatch.cuda(), ybatch.cuda()
            avalpred_batch = model(xbatch)
            avalpred_list.append(avalpred_batch)
            yval_list.append(ybatch)
        aval_pred = torch.cat(avalpred_list) # search candidates
        yval = torch.cat(yval_list) # ground truth in search candidates's order.
        
        dist_retrieval = self.eval_dist(self.emb, aval_pred, self.c) # n_cls x n_val
        rank = dist_retrieval.sort()[1] # n_cls x n_val 

        topk_result_inx = rank[:,:50] # n_cls x k (k = 50)
        ypred_topk = yval[topk_result_inx] # n_cls x k
        
        GT = torch.arange(self.emb.size(0)).cuda()
              
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)

        print('mAP:%.3f,1hop_mAP:%.3f,2hop_mAP:%.3f'%(hop0_mAP, hop1_mAP, hop2_mAP))
              

class SoftmaxNet(nn.Module):
    def __init__(self, T, c, eval_dist, train_loader, val_loader, emb, metric):
        super(SoftmaxNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, emb.size(0))
        )
        self.T = T
        self.c = c
        self.eval_dist = eval_dist
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.emb = emb
        self.metric = metric
        
    def forward(self, x):
        x = self.layers(x)
        return x

    def _train(self, optimizer, epochs, eval_interval, feat_layer):
        model = self
        model.train()   
        for epoch in range(epochs):

            for i, (xbatch, ybatch, abatch) in enumerate(self.train_loader):

                xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
                optimizer.zero_grad()
                logits = model(xbatch)
                logits = logits / self.T # Temperature trick
                loss_fn = F.cross_entropy
                loss = loss_fn(logits,ybatch)

                loss.backward()
                optimizer.step()

            if epoch % eval_interval == 0:
                self.evaluation('valid', feat_layer)
                self.evaluation2()
                
    def evaluation(self, flag, feat_layer): # 2,4,6三种
        model = self

        # 0.query_candidates & GT
        feat_list, yval_list = [], []
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):
            xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
            feat_batch = model.layers[0:feat_layer](xbatch).detach()
            feat_list.append(feat_batch)
            yval_list.append(ybatch)
        feat = torch.cat(feat_list) # search candidates
        yval_order1 = torch.cat(yval_list) # ground truth in search candidates's order.
        
        GT_list = [] # ground truth in validation order
        loss_list, ypred_list, ypred_topk_list = [],[],[]
        torch.manual_seed(42)
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):
            xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
            feat_batch = model.layers[0:feat_layer](xbatch).detach()
            
            logits = model(xbatch)

            # 1.loss
            loss_fn = F.cross_entropy
            loss = loss_fn(logits,ybatch).item()
            loss_list.append(loss)
            
            # 2.ypred for recognition
            rank = logits.sort()[1]
            ypred = rank[:,-1]
            ypred_list.append(ypred)
            
            # 3.ypred_topk for retrieval
            dist_retrieval = self.eval_dist(feat_batch, feat, self.c)
            rank = dist_retrieval.sort()[1]
            topk_result_inx = rank[:,:50] # k = 50
            ypred_topk = yval_order1[topk_result_inx] # batch_size x k
            ypred_topk_list.append(ypred_topk)
            GT_list.append(ybatch)
        GT = torch.cat(GT_list)
        
        loss = torch.tensor(loss_list).mean()
        ypred = torch.cat(ypred_list)
        ypred_topk = torch.cat(ypred_topk_list)
        
        hop0_acc = (ypred == GT).float().mean().item()
        hop1_acc = self.metric.hop_acc(ypred,GT, hops = 2)
        hop2_acc = self.metric.hop_acc(ypred,GT, hops = 4)           
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)

        print('loss:%.3f,acc:%.3f,1hop_acc:%.3f,2hop_acc:%.3f,mAP:%.3f,1hop_mAP:%.3f,2hop_mAP:%.3f'%
              (loss, hop0_acc, hop1_acc, hop2_acc, hop0_mAP, hop1_mAP, hop2_mAP)) 
        
    def evaluation2(self):
        model = self
        # 0.query_candidates & GT
        logits_val_list, yval_list = [], []
        for i, (xbatch, ybatch, _) in enumerate(self.val_loader):
            xbatch, ybatch = xbatch.cuda(), ybatch.cuda()
            logits_batch = model(xbatch)
            logits_batch = logits_batch / self.T
            logits_val_list.append(logits_batch)
            yval_list.append(ybatch)
        logits_val = torch.cat(logits_val_list) # n_cls x n_val
        probs_val = F.softmax(logits_val,dim=1) # n_cls x n_val
        rank = probs_val.sort(dim=0)[1] # nval x n_cls
        topk_result_inx = rank[-50:,:] # k x n_cls (k = 50)
        yval = torch.cat(yval_list) # ground truth in search candidates's order.
        ypred_topk = yval[topk_result_inx] # k x n_cls
        ypred_topk = ypred_topk.t() # n_cls x k
        GT = torch.arange(self.emb.size(0)).unsqueeze(1).cuda() # n_cls x 1
                      
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)

        print('mAP:%.3f,1hop_mAP:%.3f,2hop_mAP:%.3f'%(hop0_mAP, hop1_mAP, hop2_mAP))
        
class CVPR19Net2(nn.Module):
    def __init__(self, T, c, parent_set, grandpa_set, son2parent, eval_dist, train_loader, val_loader, emb, metric):
        super(CVPR19Net2, self).__init__()
        
        n_pa, n_gp = len(parent_set), len(grandpa_set)
        
        self.dense1 = nn.Linear(2048, 2048)
        self.leaky1 = nn.LeakyReLU()
        self.dense2 = nn.Linear(2048, 2048)
        self.leaky2 = nn.LeakyReLU()
        self.dense_leaf = nn.Linear(2048, 200)
        self.dense_parent = nn.Linear(2048,n_pa)
        self.dense_grandpa = nn.Linear(2048,n_gp)
        
        self.c = c
        self.T = T
        self.parent_set = parent_set
        self.grandpa_set = grandpa_set
        self.eval_dist = eval_dist
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.son2parent = son2parent
        self.emb = emb
        self.metric = metric
        
    def forward(self, x):
        hidden1 = self.dense1(x)
        action1 = self.leaky1(hidden1)
        hidden2 = self.dense1(action1)
        action2 = self.leaky1(hidden2)
        
        logits_leaf = self.dense_leaf(action1)
        logits_leaf = self.dense_leaf(action2)
        logits_pa = self.dense_parent(action2)
        logits_gp = self.dense_grandpa(action2)
        
        return logits_leaf, logits_pa, logits_gp, hidden1

    def evaluation(self):
        model = self
        
        # 0.query_candidates & GT
        cand_feat_list, cand_yval_list = [], []
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):
            xbatch, ybatch, abatch, _, _ = self.batch_generation_on_gpu(xbatch, ybatch, abatch)
            _, _, _, batch_feat = model(xbatch)

            cand_feat_list.append(batch_feat)
            cand_yval_list.append(ybatch)
        cand_feat = torch.cat(cand_feat_list) # search candidates
        cand_yval = torch.cat(cand_yval_list) # ground truth in search candidates's order.
        
        GT_list = [] # ground truth in validation order
        loss_list, ypred_list, ypred_topk_list = [],[],[]
        torch.manual_seed(42)


        loss_lists = ([],[],[])
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):

            xbatch, ybatch, abatch, ybatch_pa, ybatch_gp = self.batch_generation_on_gpu(xbatch, ybatch, abatch)
            # 1.Loss
            logits_leaf, logits_pa, logits_gp, batch_feat = model(xbatch)
            loss_fn = F.cross_entropy
            
            loss_leaf = loss_fn(logits_leaf,ybatch).item()
            loss_pa = loss_fn(logits_pa,ybatch_pa).item()
            loss_gp = loss_fn(logits_gp,ybatch_gp).item()
            loss_lists[0].append(loss_leaf)
            loss_lists[1].append(loss_pa)
            loss_lists[2].append(loss_gp)
            
            # For accs
            rank = logits_leaf.sort()[1]
            ypred = rank[:,-1]
            
            # For retrievals
            dist_retrieval = self.eval_dist(batch_feat, cand_feat, self.c)
            rank = dist_retrieval.sort()[1]
            topk_result_inx = rank[:,:50] # k = 50
            ypred_topk = cand_yval[topk_result_inx]
 
            # For ground truth
            GT_list.append(ybatch)
            ypred_list.append(ypred)
            ypred_topk_list.append(ypred_topk)
     
        loss_leaf = torch.tensor(loss_lists[0]).mean()
        loss_pa = torch.tensor(loss_lists[1]).mean()
        loss_gp = torch.tensor(loss_lists[2]).mean()
        
        GT = torch.cat(GT_list)       
        ypred = torch.cat(ypred_list)
        ypred_topk = torch.cat(ypred_topk_list)
        
        hop0_acc = (ypred == GT).float().mean().item()
        hop1_acc = self.metric.hop_acc(ypred,GT, hops = 2)
        hop2_acc = self.metric.hop_acc(ypred,GT, hops = 4)           
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)

        print('loss:%.3f,loss:%.3f,loss:%.3f,acc:%.3f,1hop_acc:%.3f,2hop_acc:%.3f,mAP:%.3f,1hop_mAP:%.3f,2hop_mAP:%.3f'%(loss_leaf,loss_pa,loss_gp, hop0_acc, hop1_acc, hop2_acc, hop0_mAP, hop1_mAP, hop2_mAP))
    
    def batch_generation_on_gpu(self, xbatch, ybatch, abatch):
        son2parent = self.son2parent
        label_set = [key for key,value in son2parent.items() if key not in son2parent.values()]
        son2grandpa = {key:son2parent[value] for key,value in son2parent.items() if value in self.parent_set}
        
        ybatch_pa = [self.parent_set.index(son2parent[label_set[item]]) for item in ybatch]
        ybatch_pa = torch.tensor(ybatch_pa)
        ybatch_gp = [self.grandpa_set.index(son2grandpa[label_set[item]]) for item in ybatch]
        ybatch_gp = torch.tensor(ybatch_gp)

        xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
        ybatch_pa, ybatch_gp = ybatch_pa.cuda(), ybatch_gp.cuda()
        
        return xbatch, ybatch, abatch, ybatch_pa, ybatch_gp
    
    def _train(self, optimizer, epochs,T, eval_interval):
        model = self
        model.train()   
        for epoch in range(epochs):

            for i, (xbatch, ybatch, abatch) in enumerate(self.train_loader):

                xbatch, ybatch, abatch, ybatch_pa, ybatch_gp = self.batch_generation_on_gpu(xbatch, ybatch, abatch)
                optimizer.zero_grad()
                logits_leaf, logits_pa, logits_gp, _ = model(xbatch)
                T = self.T
                logits_leaf, logits_pa, logits_gp = logits_leaf/T, logits_pa/T, logits_gp/T # Temperature trick
                
                loss_fn = F.cross_entropy
                loss_leaf = loss_fn(logits_leaf,ybatch)
                loss_pa = loss_fn(logits_pa,ybatch_pa)
                loss_gp = loss_fn(logits_gp,ybatch_gp)
                
                loss = loss_leaf + 1* loss_pa + 1 * loss_gp

                loss.backward()
                optimizer.step()

            if epoch % eval_interval == 0:
                self.evaluation()
                self.evaluation2()
                
    def eval_zsl(self, unseen_idset):
        model = self
        logits_val_list, yval_list = [], []
        for i, (xbatch, ybatch, _) in enumerate(self.val_loader):
            xbatch, ybatch = xbatch.cuda(), ybatch.cuda()
            logits_batch, _, _, _ = model(xbatch)
            logits_batch = logits_batch / self.T
            logits_val_list.append(logits_batch)
            yval_list.append(ybatch)
        logits_val = torch.cat(logits_val_list) # n_val x n_cls
        # 关键步骤
        logits_val = logits_val[:,unseen_idset]
        
        probs_val = F.softmax(logits_val,dim=1) # n_val x n_unseencls
        rank = probs_val.sort(dim=0)[1] # n_val x n_unseencls
        topk_result_inx = rank[-50:,:] # k x n_unseencls (top k for each class)

        yval = torch.cat(yval_list) # ground truth in search candidates's order.
        ypred_topk = yval[topk_result_inx] #  k x n_unseencls # 每个name最近的k个样本的类别
        ypred_topk = ypred_topk.t()
        
        GT = torch.tensor(unseen_idset).cuda()
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)
        print('ZSL-search by name:, mAP:%.3f, 1hop_mAP:%.3f, 2hop_mAP:%.3f'%(hop0_mAP, hop1_mAP, hop2_mAP))
        
    def evaluation2(self):
        model = self
        # 0.query_candidates & GT
        logits_val_list, yval_list = [], []
        for i, (xbatch, ybatch, _) in enumerate(self.val_loader):
            xbatch, ybatch = xbatch.cuda(), ybatch.cuda()
            logits_batch, _, _, _ = model(xbatch)
            logits_batch = logits_batch / self.T
            logits_val_list.append(logits_batch)
            yval_list.append(ybatch)
        logits_val = torch.cat(logits_val_list) # n_cls x n_val
        probs_val = F.softmax(logits_val,dim=1) # n_cls x n_val
        rank = probs_val.sort(dim=0)[1] # n_cls x nval 
        topk_result_inx = rank[-50:,:] # k x nval (k = 50)
        yval = torch.cat(yval_list) # ground truth in search candidates's order.
        ypred_topk = yval[topk_result_inx] # k x nval
        ypred_topk = ypred_topk.t() # nval x k
        GT = torch.arange(self.emb.size(0)).unsqueeze(1).cuda() # n_cls x 1
                      
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)

        print('mAP:%.3f,1hop_mAP:%.3f,2hop_mAP:%.3f'%(hop0_mAP, hop1_mAP, hop2_mAP))
        
        
class CVPR19Net(nn.Module):
    def __init__(self, T, c, parent_set, grandpa_set, son2parent, eval_dist, train_loader, val_loader, emb, metric):
        super(CVPR19Net, self).__init__()
        
        n_pa, n_gp = len(parent_set), len(grandpa_set)
        
        self.dense1 = nn.Linear(2048, 2048)
        self.leaky1 = nn.LeakyReLU()
        self.dense_pa = nn.Linear(2048, 2048)
        self.leaky_pa = nn.LeakyReLU()
        self.dense_gp = nn.Linear(2048, 2048)
        self.leaky_gp = nn.LeakyReLU()
        self.dense_leaf = nn.Linear(2048, 200)
        self.dense_parent = nn.Linear(4096, n_pa)
        self.dense_grandpa = nn.Linear(6144, n_gp)
        
        self.c = c
        self.T = T
        self.parent_set = parent_set
        self.grandpa_set = grandpa_set
        self.eval_dist = eval_dist
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.son2parent = son2parent
        self.emb = emb
        self.metric = metric
        
    def forward(self, x):
        hidden1 = self.dense1(x)
        action1 = self.leaky1(hidden1)
        hidden_pa = self.dense_pa(action1)
        action_pa = self.leaky_pa(hidden_pa)
        hidden_gp = self.dense_gp(action1)
        action_gp = self.leaky_gp(hidden_gp)
        
        logits_leaf = self.dense_leaf(action1)
#         import pdb
#         pdb.set_trace()
        logits_pa = self.dense_parent(torch.cat((action1,action_pa),dim=1))
        logits_gp = self.dense_grandpa(torch.cat((action1,action_pa,action_gp),dim=1))
#         logits_leaf = self.dense_leaf(action2)
#         logits_pa = self.dense_parent(action2)
#         logits_gp = self.dense_grandpa(action2)
        
        
        return logits_leaf, logits_pa, logits_gp, hidden1
    
    def evaluation(self):
        model = self
        
        # 0.query_candidates & GT
        cand_feat_list, cand_yval_list = [], []
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):
            xbatch, ybatch, abatch, _, _ = self.batch_generation_on_gpu(xbatch, ybatch, abatch)
            _, _, _, batch_feat = model(xbatch)

            cand_feat_list.append(batch_feat)
            cand_yval_list.append(ybatch)
        cand_feat = torch.cat(cand_feat_list) # search candidates
        cand_yval = torch.cat(cand_yval_list) # ground truth in search candidates's order.
        
        GT_list = [] # ground truth in validation order
        loss_list, ypred_list, ypred_topk_list = [],[],[]
        torch.manual_seed(42)


        loss_lists = ([],[],[])
        for i, (xbatch, ybatch, abatch) in enumerate(self.val_loader):

            xbatch, ybatch, abatch, ybatch_pa, ybatch_gp = self.batch_generation_on_gpu(xbatch, ybatch, abatch)
            # 1.Loss
            logits_leaf, logits_pa, logits_gp, batch_feat = model(xbatch)
            loss_fn = F.cross_entropy
            
            loss_leaf = loss_fn(logits_leaf,ybatch).item()
            loss_pa = loss_fn(logits_pa,ybatch_pa).item()
            loss_gp = loss_fn(logits_gp,ybatch_gp).item()
            loss_lists[0].append(loss_leaf)
            loss_lists[1].append(loss_pa)
            loss_lists[2].append(loss_gp)
            
            # For accs
            rank = logits_leaf.sort()[1]
            ypred = rank[:,-1]
            
            # For retrievals
            dist_retrieval = self.eval_dist(batch_feat, cand_feat, self.c)
            rank = dist_retrieval.sort()[1]
            topk_result_inx = rank[:,:50] # k = 50
            ypred_topk = cand_yval[topk_result_inx]
 
            # For ground truth
            GT_list.append(ybatch)
            ypred_list.append(ypred)
            ypred_topk_list.append(ypred_topk)
     
        loss_leaf = torch.tensor(loss_lists[0]).mean()
        loss_pa = torch.tensor(loss_lists[1]).mean()
        loss_gp = torch.tensor(loss_lists[2]).mean()
        
        GT = torch.cat(GT_list)       
        ypred = torch.cat(ypred_list)
        ypred_topk = torch.cat(ypred_topk_list)
        
        hop0_acc = (ypred == GT).float().mean().item()
        hop1_acc = self.metric.hop_acc(ypred,GT, hops = 2)
        hop2_acc = self.metric.hop_acc(ypred,GT, hops = 4)           
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)

        print('loss:%.3f,loss:%.3f,loss:%.3f,acc:%.3f,1hop_acc:%.3f,2hop_acc:%.3f,mAP:%.3f,1hop_mAP:%.3f,2hop_mAP:%.3f'%(loss_leaf,loss_pa,loss_gp, hop0_acc, hop1_acc, hop2_acc, hop0_mAP, hop1_mAP, hop2_mAP))
    
    def batch_generation_on_gpu(self, xbatch, ybatch, abatch):
        son2parent = self.son2parent
        label_set = [key for key,value in son2parent.items() if key not in son2parent.values()]
        son2grandpa = {key:son2parent[value] for key,value in son2parent.items() if value in self.parent_set}
        
        ybatch_pa = [self.parent_set.index(son2parent[label_set[item]]) for item in ybatch]
        ybatch_pa = torch.tensor(ybatch_pa)
        ybatch_gp = [self.grandpa_set.index(son2grandpa[label_set[item]]) for item in ybatch]
        ybatch_gp = torch.tensor(ybatch_gp)

        xbatch, ybatch, abatch = xbatch.cuda(), ybatch.cuda(), abatch.cuda()
        ybatch_pa, ybatch_gp = ybatch_pa.cuda(), ybatch_gp.cuda()
        
        return xbatch, ybatch, abatch, ybatch_pa, ybatch_gp
    
    def _train(self, optimizer, epochs,T, eval_interval):
        model = self
        model.train()   
        for epoch in range(epochs):

            for i, (xbatch, ybatch, abatch) in enumerate(self.train_loader):

                xbatch, ybatch, abatch, ybatch_pa, ybatch_gp = self.batch_generation_on_gpu(xbatch, ybatch, abatch)
                optimizer.zero_grad()
                logits_leaf, logits_pa, logits_gp, _ = model(xbatch)
                T = self.T
                logits_leaf, logits_pa, logits_gp = logits_leaf/T, logits_pa/T, logits_gp/T # Temperature trick
                
                loss_fn = F.cross_entropy
                loss_leaf = loss_fn(logits_leaf,ybatch)
                loss_pa = loss_fn(logits_pa,ybatch_pa)
                loss_gp = loss_fn(logits_gp,ybatch_gp)
                
                loss = loss_leaf + 1* loss_pa + 1 * loss_gp

                loss.backward()
                optimizer.step()

            if epoch % eval_interval == 0:
                self.evaluation()
                self.evaluation2()
                
    def eval_zsl(self, unseen_idset):
        model = self
        logits_val_list, yval_list = [], []
        for i, (xbatch, ybatch, _) in enumerate(self.val_loader):
            xbatch, ybatch = xbatch.cuda(), ybatch.cuda()
            logits_batch, _, _, _ = model(xbatch)
            logits_batch = logits_batch / self.T
            logits_val_list.append(logits_batch)
            yval_list.append(ybatch)
        logits_val = torch.cat(logits_val_list) # n_val x n_cls
        # 关键步骤
        logits_val = logits_val[:,unseen_idset]
        
        probs_val = F.softmax(logits_val,dim=1) # n_val x n_unseencls
        rank = probs_val.sort(dim=0)[1] # n_val x n_unseencls
        topk_result_inx = rank[-50:,:] # k x n_unseencls (top k for each class)

        yval = torch.cat(yval_list) # ground truth in search candidates's order.
        ypred_topk = yval[topk_result_inx] #  k x n_unseencls # 每个name最近的k个样本的类别
        ypred_topk = ypred_topk.t()
        
#         top1_result_inx = rank[-1,:] # n_unseencls x 1 (k = 1)
#         ypred = yval[top1_result_inx]
#         hop0_acc = (ypred == GT).float().mean().item()
#         hop1_acc = self.metric.hop_acc(ypred,GT, hops = 2)
#         hop2_acc = self.metric.hop_acc(ypred,GT, hops = 4)
        
        GT = torch.tensor(unseen_idset).cuda()
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)
        print('ZSL-search by name:, mAP:%.3f, 1hop_mAP:%.3f, 2hop_mAP:%.3f'%(hop0_mAP, hop1_mAP, hop2_mAP))
        
#         dist = self.eval_dist(apred, unseen_emb, self.c)
        
    def evaluation2(self):
        model = self
        # 0.query_candidates & GT
        logits_val_list, yval_list = [], []
        for i, (xbatch, ybatch, _) in enumerate(self.val_loader):
            xbatch, ybatch = xbatch.cuda(), ybatch.cuda()
            logits_batch, _, _, _ = model(xbatch)
            logits_batch = logits_batch / self.T
            logits_val_list.append(logits_batch)
            yval_list.append(ybatch)
        logits_val = torch.cat(logits_val_list) # n_cls x n_val
        probs_val = F.softmax(logits_val,dim=1) # n_cls x n_val
        rank = probs_val.sort(dim=0)[1] # n_cls x nval 
        topk_result_inx = rank[-50:,:] # k x nval (k = 50)
        yval = torch.cat(yval_list) # ground truth in search candidates's order.
        ypred_topk = yval[topk_result_inx] # k x nval
        ypred_topk = ypred_topk.t() # nval x k
        GT = torch.arange(self.emb.size(0)).unsqueeze(1).cuda() # n_cls x 1
                      
        hop0_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 0)
        hop1_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 2)
        hop2_mAP = self.metric.hop_mAP(ypred_topk, GT, hop = 4)

        print('mAP:%.3f,1hop_mAP:%.3f,2hop_mAP:%.3f'%(hop0_mAP, hop1_mAP, hop2_mAP))
        
class NIP19Proto():
    def __init__(self, w2v_emb, lr, epochs, loss_lambdas):
        self.w2v_emb = w2v_emb
        self.lr = lr
        self.epochs = epochs
        self.loss_lambdas = loss_lambdas

    def similarity(self,prototypes):
        norm = torch.norm(prototypes, dim=1)  # each row is norm 1.

        deviation = (norm.sum() - prototypes.shape[0])
        if deviation > 1e1:
            print('deviation from norm 1', deviation)

        t1 = norm.unsqueeze(1)  # n_cls x 1
        t2 = norm.unsqueeze(0)  # 1 x n_cls
        denominator = torch.matmul(t1, t2)  # n_cls x n_cls, each element is a norm product
        numerator = torch.matmul(prototypes, prototypes.t())  # each element is a in-prod
        cos_sim = numerator / denominator  # n_cls x n_cls, each element is a cos_sim
        cos_sim_off_diag = cos_sim - torch.diag(torch.diag(cos_sim))
        obj = cos_sim_off_diag.max(dim=1)[0]

        return obj.mean(), cos_sim

    def order_loss(self,prototypes, w2v, lmd=0):
        B = prototypes.t()
        _, S = self.similarity(w2v)
        S = S.float()

        # Laplacian matrix L
        S1 = S - lmd
        ones = torch.ones(S1.shape[0], 1)
        L = torch.diag(S1.matmul(ones)) - S

        # Loss = Trace(BLB')
        M = B.matmul(L)
        M = M.matmul(B.t())
        o_loss = torch.trace(M) / B.shape[1]**2

        return o_loss
    
    def train(self):

        emb = self.w2v_emb.cpu()
        emb = F.normalize(emb)
        prototypes = nn.Parameter(F.normalize(torch.randn(emb.size(0), 300), p=2, dim=1))
        optimizer = torch.optim.SGD([prototypes], lr=self.lr, momentum=0.9)
        best_loss = 1000
        
        for i in range(self.epochs):
            optimizer.zero_grad()
            sim, _ = self.similarity(prototypes)
            o_loss = self.order_loss(prototypes, emb.cpu(), lmd=0)
            loss = self.loss_lambdas[0] * sim + self.loss_lambdas[1] * o_loss # 默认是1:1

            loss.backward(retain_graph=True)
            optimizer.step()
            if i % 10 == 0:
                if i % 100 == 0:
                    print(f'Loss: {loss}, Order Loss: {o_loss} and Sim Loss: {sim}')
                prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
                optimizer = torch.optim.SGD([prototypes], lr=self.lr, momentum=0.9)

        self.prototypes = prototypes
        
        
