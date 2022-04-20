import scipy.io
import urllib.request
import dgl
import math
import numpy as np
from model import *
import torch
from data_loader import data_loader
from utils.data import load_data
from utils.pytorchtools import EarlyStopping
import argparse
import time


ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
ap.add_argument('--feats-type', type=int, default=3,
                help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                    '4 - only term features (id vec for others);' + 
                    '5 - only term features (zero vec for others).')
ap.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
ap.add_argument('--patience', type=int, default=30, help='Patience.')
ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--lr', type=float, default=5e-3)
ap.add_argument('--dropout', type=float, default=0.5)
ap.add_argument('--weight-decay', type=float, default=1e-4)
ap.add_argument('--slope', type=float, default=0.05)
ap.add_argument('--dataset', type=str, default='ACM')
ap.add_argument('--edge-feats', type=int, default=64)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--schedule_step', type=int, default=300)
ap.add_argument('--use_norm', type=bool, default=False)

args = ap.parse_args()
if args.device >= 0:
    device = torch.device("cuda:"+str(args.device))
else:
    device = torch.device('cpu')
def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

dataset = data_loader('./'+args.dataset)
edge_dict = {}

for i, meta_path in dataset.links['meta'].items():
    edge_dict[(str(meta_path[0]), str(meta_path[0]) + '_' + str(meta_path[1]), str(meta_path[1]))] = (torch.tensor(dataset.links['data'][i].tocoo().row - dataset.nodes['shift'][meta_path[0]]), torch.tensor(dataset.links['data'][i].tocoo().col - dataset.nodes['shift'][meta_path[1]]))

node_count = {}
for i, count in dataset.nodes['count'].items():
    print(i, node_count)
    node_count[str(i)] = count

G = dgl.heterograph(edge_dict, num_nodes_dict = node_count, device=device)
"""
for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = dataset.nodes['attr'][ntype]
    # print(G.nodes['attr'][ntype].shape)
"""

G.node_dict = {}
G.edge_dict = {}
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype] 


feats_type = args.feats_type
features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
features_list = [mat2tensor(features).to(device) for features in features_list]

if feats_type == 0:
    in_dims = [features.shape[1] for features in features_list]
elif feats_type == 1 or feats_type == 5:
    save = 0 if feats_type == 1 else 2
    in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
    for i in range(0, len(features_list)):
        if i == save:
            in_dims.append(features_list[i].shape[1])
        else:
            in_dims.append(10)
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
elif feats_type == 2 or feats_type == 4:
    save = feats_type - 2
    in_dims = [features.shape[0] for features in features_list]
    for i in range(0, len(features_list)):
        if i == save:
            in_dims[i] = features_list[i].shape[1]
            continue
        dim = features_list[i].shape[0]
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
elif feats_type == 3:
    in_dims = [features.shape[0] for features in features_list]
    for i in range(len(features_list)):
        dim = features_list[i].shape[0]
        indices = np.vstack((np.arange(dim), np.arange(dim)))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(np.ones(dim))
        features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)


for ntype in G.ntypes:
    G.nodes[ntype].data['inp'] = features_list[int(ntype)]#.to(device)
labels = torch.LongTensor(labels).to(device)
#add
batch_size = 10
edge_labels_train = dl.train_labels
edge_labels_test = dl.test_labels

import torch.utils.data as Data

torch_dataset = Data.TensorDataset(torch.tensor(edge_labels_train['head_node_idx']), torch.tensor(edge_labels_train['tail_node_idx']), torch.tensor(np.asarray(edge_labels_train['label'], dtype=np.int64)))
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_head_idx = torch.tensor(edge_labels_test['head_node_idx']).to(device)
test_tail_idx = torch.tensor(edge_labels_test['tail_node_idx']).to(device)
test_label = torch.tensor(np.asarray(edge_labels_test['label'], dtype=np.int64)).to(device)
#------add end-------

train_idx = train_val_test_idx['train_idx']
train_idx = np.sort(train_idx)
val_idx = train_val_test_idx['val_idx']
val_idx = np.sort(val_idx)
test_idx = train_val_test_idx['test_idx']
test_idx = np.sort(test_idx)

print(len(train_idx), len(test_idx), len(val_idx))
model = HGT(G, n_inps=in_dims, n_hid=args.hidden_dim, n_out=labels.max().item()+1, n_layers=args.num_layers, n_heads=args.num_heads, use_norm = args.use_norm).to(device)

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)#, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=int(edge_labels_train['count']/batch_size)+1, max_lr = 1e-3, pct_start=0.05)

# train_lp.py
early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
train_step = 0
for epoch in range(args.epoch):
    for step, (head_idx, tail_idx, label) in enumerate(loader):

        t_start = time.time()
        model.train()
        label = label.to(device)
        logits = model(G, head_idx.to(device), tail_idx.to(device))
        train_loss = F.cross_entropy(logits, label)

        pred = logits.argmax(1)
        train_acc = (pred == label).float().mean()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_step += 1
        scheduler.step(train_step)
        t_end = time.time()
        print('Epoch {:05d} | Step {:03d} | Train_Loss: {:.4f} | Train_acc: {:.4f} |Time: {:.4f}'.format(epoch, step,train_loss.item(), train_acc.item(), t_end-t_start))
        t_start = time.time()
    model.eval()
    with torch.no_grad():
        test_logits = model(G, test_head_idx.to(device), test_tail_idx.to(device))
        test_loss = F.cross_entropy(test_logits, test_label)
        pred = test_logits.cpu().numpy().argmax(axis=1)
    test_acc = (pred == test_label.cpu().numpy()).mean()
    t_end = time.time()
    print('Epoch {:05d} | Test_Loss {:.4f} | Test_acc {:.4f} | Time(s) {:.4f}'.format(
        epoch, test_loss.item(), test_acc.item(),  t_end - t_start))
    early_stopping(test_loss, model)
    if early_stopping.early_stop:
        print('Early stopping!')
        break

model.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))

