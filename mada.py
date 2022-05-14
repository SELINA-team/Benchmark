import torch
from torch.autograd import Function
import torch.nn as nn
import glob
import ntpath
import datatable as dt
import pandas as pd
import numpy as np
from functools import reduce
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
from torch.utils.tensorboard import SummaryWriter
from imblearn.over_sampling import SMOTE
from collections import Counter

def read_expr(path):
    expr = dt.fread(path, header=True, sep='\t', nthreads=6)
    expr = expr.to_pandas()
    expr.index = expr.loc[:, 'Gene']
    del expr['Gene']
    return expr


def label2dic(label):
    label_set = list(set(label))
    dic = {}
    for i in range(len(label_set)):
        dic[label_set[i]] = i
    return dic


def preprocessing_sample(path_in,i):
    samples = sorted(glob.glob(path_in + '/*_expr.txt'))
    metas = sorted(glob.glob(path_in + '/*_meta.txt'))
    samples.pop(i)
    metas.pop(i)
    train_sets = []
    celltypes = []
    platforms = []
    print('Loading data')
    for i in range(len(samples)):
        train_sets.append(read_expr(samples[i]))
        meta = pd.read_csv(metas[i], sep='\t', header=0)
        celltype = meta['Celltype'].to_list()
        platform = meta['Platform'].to_list()
        celltypes.append(celltype)
        platforms.append(platform)
    ct_freqs = Counter([i for item in celltypes for i in item])
    max_n = max(ct_freqs.values())
    rct_freqs = {}
    if  max_n < 500:
        sample_n = 100
    elif max_n < 1000:
        sample_n = 500
    else:
        sample_n = 1000
    for ct,freq in ct_freqs.items():
        if freq <= sample_n:
            rct_freqs[ct] = freq
                
    for i in range(len(samples)):
        sample_ct_freq = {}
        ct_freq = Counter(celltypes[i])
        if len(ct_freq)>1:
            for ct,freq in rct_freqs.items():
                if (ct in ct_freq.keys()) & (ct_freq[ct] >= 6):
                    sample_ct_freq[ct] = round(sample_n * ct_freq[ct]/freq)
            smo = SMOTE(sampling_strategy = sample_ct_freq,random_state=1)
            train_sets[i],celltypes[i] = smo.fit_resample(train_sets[i].T,celltypes[i])
            train_sets[i] = train_sets[i].T         
            platforms[i] = np.unique(platforms[i]).tolist() * train_sets[i].shape[1]
    platforms = [i for item in platforms for i in item]
    celltypes = [i for item in celltypes for i in item]
    for i in range(len(samples)):
        train_sets[i] = np.divide(train_sets[i], np.sum(train_sets[i],
                                                        axis=0)) * 10000
        train_sets[i] = np.log2(train_sets[i] + 1)
    train_data = pd.concat(train_sets, axis=1)
    return train_data, celltypes, platforms


class Datasets(Dataset):
    def __init__(self, data, celltypes, platforms, ct_dic, plat_dic):
        class_labels = [ct_dic[i] for i in celltypes]
        domain_labels = [plat_dic[i] for i in platforms]
        self.class_labels = torch.as_tensor(class_labels)
        self.domain_labels = torch.as_tensor(domain_labels)
        self.expr = data.values

    def __getitem__(self, index):
        return torch.as_tensor(
            self.expr[:, index]
        ), self.class_labels[index], self.domain_labels[index]

    def __len__(self):
        return len(self.class_labels)


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MADA(nn.Module):
    def __init__(self, nfeatures, nct, nplat):
        super(MADA, self).__init__()
        self.nct = nct
        self.feature = nn.Sequential(
            nn.Linear(in_features=nfeatures, out_features=100), nn.ReLU(),
            nn.Dropout())
        self.class_classifier = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU(),
            nn.Dropout(), nn.Linear(in_features=50, out_features=nct))
        self.domain_classifier = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features=100, out_features=25),
                          nn.ReLU(),
                          nn.Linear(in_features=25, out_features=nplat))
            for _ in range(nct)
        ])

    def forward(self, input_data, alpha, nct):
        features = self.feature(input_data)
        class_logits = self.class_classifier(features)
        class_predictions = F.softmax(class_logits, dim=1)
        reverse_features = GRL.apply(features, alpha)
        domain_logits = []
        for class_idx in range(nct):
            wrf = class_predictions[:,
                                    class_idx].unsqueeze(1) * reverse_features
            domain_logits.append(self.domain_classifier[class_idx](wrf))
        return class_logits, domain_logits


def train(train_data, params, celltypes, platforms, nfeatures, nct, nplat,
          ct_dic, plat_dic, device):
    network = MADA(nfeatures, nct, nplat).train()
    lr = params[0]
    n_epoch = params[1]
    batch_size = params[2]
    train_data = Datasets(train_data, celltypes, platforms, ct_dic, plat_dic)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()
    network = network.to(device)
    loss_class = loss_class.to(device)
    loss_domain = loss_domain.to(device)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    len_train_loader = len(train_loader)
    print('Begin training')
    for epoch in tqdm(range(n_epoch)):
        loader_iter = iter(train_loader)
        for i in range(len_train_loader):
            p = float(i +
                      epoch * len_train_loader) / n_epoch / len_train_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            expr, class_label, domain_label = loader_iter.next()
            expr = expr.to(device)
            expr = expr.float()
            class_label = class_label.to(device)
            domain_label = domain_label.to(device)
            class_output, domain_output = network(input_data=expr,
                                                  alpha=alpha,
                                                  nct=nct)
            err_class = loss_class(class_output, class_label)
            err_domain = [
                loss_domain(domain_output[class_idx], domain_label)
                for class_idx in range(nct)
            ]
            loss_total = (1 -
                          alpha) * sum(err_domain) / nct + alpha * err_class
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
    print('Finish Training')
    return network


class datasets(Dataset):
    def __init__(self, data):
        self.expr = data.values

    def __getitem__(self, index):
        return torch.as_tensor(self.expr[:, index])

    def __len__(self):
        return self.expr.shape[1]


class Autoencoder(nn.Module):
    def __init__(self, network, nfeature, nct):
        super(Autoencoder, self).__init__()
        encoder = list(network.feature.children()) + list(
            network.class_classifier.children())
        encoder_index = [0, 1, 3, 4, 6]
        self.encoder = nn.Sequential(*[encoder[i] for i in encoder_index],
                                     nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(in_features=nct, out_features=50), nn.ReLU(),
            nn.Linear(in_features=50, out_features=100), nn.ReLU(),
            nn.Linear(in_features=100, out_features=nfeature))

    def forward(self, input_data):
        output = self.decoder(self.encoder(input_data))
        return (output)


class Classifier(nn.Module):
    def __init__(self, network):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(*list(network.encoder.children())[:-1],
                                        nn.Softmax(dim=1))

    def forward(self, input_data):
        output = self.classifier(input_data)
        return (output)


def tune1(test_df, network, params):
    test_dat = datasets(test_df)
    lr = params[0]
    n_epoch = params[1]
    batch_size = params[2]
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss = nn.MSELoss()
    loss = loss.to(device)
    test_loader = DataLoader(dataset=test_dat,
                             batch_size=batch_size,
                             shuffle=True)
    for name, paras in network.encoder.named_parameters():
        paras.requires_grad = False
    for name, paras in network.decoder.named_parameters():
        paras.requires_grad = True
    network = network.to(device)
    tb = tensorboard_tune()
    tb.begin_run(test_loader)
    for epoch in tqdm(range(n_epoch)):
        tb.begin_epoch()
        for batch in test_loader:
            expr = batch
            expr = expr.float()
            expr = expr.to(device)
            output = network(expr)
            err = loss(output, expr)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            tb.track_loss(err)
        tb.end_epoch()
    tb.end_run()
    print('Finish Tuning1')
    return network


def tune2(test_df, network, params):
    test_dat = datasets(test_df)
    lr = params[0]
    n_epoch = params[1]
    batch_size = params[2]
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss = nn.MSELoss()
    loss = loss.to(device)
    test_loader = DataLoader(dataset=test_dat,
                             batch_size=batch_size,
                             shuffle=True)
    for name, paras in network.encoder.named_parameters():
        paras.requires_grad = True
    for name, paras in network.decoder.named_parameters():
        paras.requires_grad = False
    network = network.to(device)
    tb = tensorboard_tune()
    tb.begin_run(test_loader)
    for epoch in tqdm(range(n_epoch)):
        tb.begin_epoch()
        for batch in test_loader:
            expr = batch
            expr = expr.float()
            expr = expr.to(device)
            output = network(expr)
            err = loss(output, expr)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            tb.track_loss(err)
        tb.end_epoch()
    tb.end_run()
    print('Finish Tuning2')
    return network


def test(test_df, network, ct_dic):
    test_dat = datasets(test_df)
    pred_prob = []
    ct_dic_rev = {v: k for k, v in ct_dic.items()}
    test_loader = DataLoader(dataset=test_dat,
                             batch_size=test_df.shape[1],
                             shuffle=False)
    with torch.no_grad():
        pred_labels = []
        for batch in test_loader:
            expr = batch
            expr = expr.float()
            expr = expr.to(device)
            class_output = network(expr)
            class_output
            pred_labels.append(
                class_output.argmax(dim=1).cpu().numpy().tolist())
            pred_prob.append(class_output.cpu().numpy())
        pred_labels = [ct_dic_rev[i] for item in pred_labels for i in item]
        pred_labels = pd.DataFrame({
            'Cell': test_df.columns,
            'Prediction': pred_labels
        })
        pred_prob = pd.DataFrame(reduce(pd.concat, pred_prob))
        pred_prob.index = test_df.columns
        pred_prob.columns = ct_dic.keys()
    return pred_labels, pred_prob

class tensorboard_tune():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.loader = None
        self.tb = None

    def begin_run(self, loader):
        self.loader = loader
        self.tb = SummaryWriter()

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_count += 1
        self.epoch_loss = 0

    def end_epoch(self):
        loss = self.epoch_loss / len(self.loader.dataset)
        self.tb.add_scalar('Loss', loss, self.epoch_count)

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

def metric(true, pred):
    true = pd.Series(true, name='true')
    pred = pd.Series(pred, name='pred')
    confmat = pd.DataFrame(pd.crosstab(true, pred))
    f1 = []
    acc = 0
    for i in range(confmat.shape[0]):
        flag = 0
        if confmat.index[i] in confmat.columns:
            flag = 1
        if flag:
            j = np.where(confmat.columns == confmat.index[i])[0][0]
            TP = confmat.iloc[i, j]
            if TP == 0:
                f1.append(0)
            else:
                precision = TP / confmat.iloc[:, j].sum()
                recall = TP / confmat.iloc[i, :].sum()
                f1.append(2 * (precision * recall) / (precision + recall))
        else:
            TP = 0
            f1.append(0)
        acc = acc + TP
    acc = acc / confmat.sum().sum()
    metric_summary = {
        'Acc': acc,
        'F1': f1,
        'Macro_F1': np.mean(f1),
        'confmat': confmat
    }
    return metric_summary

params_train = [0.0001, 50, 128]
params_tune1 = [0.0005, 50, 128]
params_tune2 = [0.0001, 20, 128]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = '/public/home/wanglab/pengfei/performance/'
tissue = 'Lung'
path_in = path + tissue + '/data/' 
path_out = path + tissue + '/res/' 

samples = sorted(glob.glob(path_in  + '*_expr.txt'))
tuned = {}

for i in range(len(samples)):
    print(i)
    sample = ntpath.basename(samples[i]).replace('_expr.txt','')
    train_data, celltypes, platforms = preprocessing_sample(path_in,i)
    ct_dic, plat_dic = label2dic(celltypes), label2dic(platforms)
    nfeatures, nct, nplat = train_data.shape[0], len(ct_dic), len(plat_dic)
    query_expr = read_expr(samples[i])
    query_expr = np.divide(query_expr, np.sum(query_expr, axis=0)) * 10000
    query_expr = np.log2(query_expr + 1) 
    network = train(train_data, params_train, celltypes, platforms, nfeatures,
                    nct, nplat, ct_dic, plat_dic, device)   
    network = Autoencoder(network, nfeatures, nct)
    print('Fine-tuning1')
    network = tune1(query_expr, network, params_tune1)
    print('Fine-tuning2')
    network = tune2(query_expr, network, params_tune2)
    network = Classifier(network).to(device)
    pred_labels, pred_prob = test(query_expr, network, ct_dic)
    true = pd.read_csv(samples[i].replace('expr','meta'),header=0,sep='\t')['Celltype'].to_list()
    tuned_res = metric(true,pred_labels['Prediction'].to_list())
    tuned[sample] = tuned_res
    print('tuned acc: ' + str(tuned_res['Acc']))
    print('tuned f1: ' + str(tuned_res['Macro_F1']))

with open(path_out + tissue + '_mada_res.pkl', 'wb') as f:
    pickle.dump(tuned, f, pickle.HIGHEST_PROTOCOL)

summary = {}
summary['res'] = []
for i in range(len(samples)):
    sample = ntpath.basename(samples[i]).replace('_expr.txt','')
    summary['res'].append(tuned[sample]['Acc'])
    summary['res'].append(tuned[sample]['Macro_F1'])
summary['Method'] = ['MADA'] * len(tuned)*2
summary['Metric'] = ['Acc','MacroF1'] * len(tuned)
pd.DataFrame(summary).to_csv(path_out + tissue + '_mada_res.txt',sep='\t',index=False)
