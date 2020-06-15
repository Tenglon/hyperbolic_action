import torch, pickle, json
from torch.utils.data import Dataset, DataLoader

def get_son2parent(csv_path):
    son2parent = dict()
    with open (csv_path,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            tmp_list = line.split(',')
            for i in range(len(tmp_list) - 2): # ignore last digit.
                key, value = tmp_list[i], tmp_list[i+1]
                son2parent[key] = value
    if '' in son2parent.keys():
        del son2parent['']
    return son2parent

class My_DS(Dataset):
    def __init__(self, X, y, emb):
        self.X = X
        self.y = y
        self.emb = emb
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        x_batch = self.X[index, :]
        y_batch = self.y[index]
        a_batch = self.emb[y_batch,:]
        
        return x_batch, y_batch, a_batch

def get_dataloader(Xtr, Xval, ytr, yval, emb, batch_size):
    
    train_dataset = My_DS(X=Xtr, y=ytr, emb=emb)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = My_DS(X=Xval, y=yval, emb=emb)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return train_loader, val_loader

def split_act_data(feat, label, fns, anno_fn):

    with open(anno_fn) as json_file:
        data = json.load(json_file)

    anno_tr = {key:value for key,value in data['database'].items() if value['annotations']}

    training_inx, validation_inx = [], []

    for feat_fn in fns:
        # 找到文件名及对应clip_id
        key = feat_fn.split('.')[0].split('_')[-1]
        key = '_'.join(feat_fn.split('.')[0].split('_')[2:])
        clip_inx = feat_fn.split('.')[0].split('_')[0]
        clip_inx = int(clip_inx)

        # 取出标注
        video_anno = anno_tr[key]
        if video_anno['subset'] == 'validation':
            training_inx.append(False), validation_inx.append(True)
        elif video_anno['subset'] == 'training':
            training_inx.append(True), validation_inx.append(False)
    
    training_inx, validation_inx = torch.tensor(training_inx), torch.tensor(validation_inx)
    
    Xtr, Xval = feat[training_inx], feat[validation_inx]
    ytr, yval = label[training_inx], label[validation_inx]
    assert Xtr.dim() == 2 and ytr.dim() == 1
    
    return Xtr, Xval, ytr, yval

def get_activitynet_dataset(feat_path = './data.pickle', anno_fn = './activity_net.v1-3.json'):
    # 获得X,y
    with open(feat_path,'rb') as f:
        data1 = pickle.load(f)
        
    label_set = list(set(data1['label']))
    label_set.sort() # 有Sort很重要

    label = [label_set.index(item) for item in data1['label']]
    label = torch.tensor(label)
    feat, fns = data1['feat'], data1['fn']
    
    Xtr, Xval, ytr, yval = split_act_data(feat, label, fns, anno_fn)
    
    return Xtr, Xval, ytr, yval, label_set


def get_emb(emb_type, emb_file, label_set):
    
    n_cls = len(label_set)
    if emb_type == 'rand':
        emb = torch.rand(n_cls,300)
    
    elif emb_type == 'wacv':
        with open(emb_file, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
            emb = content['embedding']
            emb = torch.tensor(emb).cuda()
        

    elif emb_type == 'oh':
        one_hot = torch.zeros(n_cls, n_cls).long()
        emb = one_hot.scatter_(dim=1, index=torch.unsqueeze(torch.arange(n_cls), dim=1), src=torch.ones(n_cls, n_cls).long())
        emb = emb.float()

    elif emb_type == 'glove':
        data2 = torch.load(emb_file, map_location='cpu')
        emb_names = data2['objects']
        ext_emb = data2['embeddings'] # n_cls x dim

        emb = torch.zeros(n_cls,ext_emb.shape[1])
        emb = emb.cuda()
        for i in range(n_cls):
            pos = emb_names.index(label_set[i])
            emb[i] = ext_emb[pos,:]

    elif emb_type == 'hyp':
        data2 = torch.load(emb_file, map_location='cpu')
        emb_names = data2['objects']
        ext_emb = data2['embeddings'] # 272 x dim
        emb = torch.zeros(n_cls,ext_emb.shape[1])
        for i in range(n_cls):
            pos = emb_names.index(label_set[i])
            emb[i] = ext_emb[pos,:]

    elif emb_type == 'cone':
        data2 = torch.load(emb_file, map_location='cpu')
        if type(data2) is zip:
            data2 = dict(data2)
        emb_names = list(data2.keys())
        ext_emb = list(data2.values()) # 271 x dim, root is discarded 
        ext_emb = torch.tensor(ext_emb)

        emb = torch.zeros(n_cls,ext_emb.shape[1])
        for i in range(n_cls):
            pos = emb_names.index(label_set[i])
            emb[i] = ext_emb[pos,:]
    
    return emb

def get_kineticslike_dataset(train_pth_path, valid_pth_path):
    data_train = torch.load(train_pth_path)
    data_val = torch.load(valid_pth_path)

    label_set = list(set(data_train['label']))
    label_set.sort() # 有Sort很重要
    
    Xtr, Xval = data_train['feat'], data_val['feat']
    ytr = torch.tensor([label_set.index(item )for item in data_train['label']])
    yval = torch.tensor([label_set.index(item )for item in data_val['label']])
    
    return Xtr, Xval, ytr, yval, label_set

