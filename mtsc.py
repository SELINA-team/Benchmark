import os
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import datatable as dt
import glob
import ntpath
import pickle

path = '/public/home/wanglab/pengfei/performance/'
tissue = 'Lung'
path_in = path + tissue + '/data/'
path_out = path + tissue + '/res/'


class Setting:
    """Parameters for training"""
    def __init__(self):
        self.epoch = 300
        self.lr = 0.0005


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


def read_data(path):
    expr = dt.fread(path, header=False, sep='\t', nthreads=6, skip_to_line=2)
    labels = dt.fread(path, header=False, sep='\t', max_nrows=1)
    expr = expr.to_pandas()
    expr.columns = labels.to_pandas().iloc[0, :]
    return expr


def indexn2ct(dat):
    label = list(dat.index)
    ct = [i.split(';')[0] for i in label]
    return ct


def indexn2dataset(dat):
    label = list(dat.index)
    dataset = [i.split(';')[1] for i in label]
    dataset = [i.split('.')[0] for i in dataset]
    return dataset


def coln2ct(dat):
    label = list(dat.columns)
    ct = [i.split(';')[0] for i in label]
    return ct


def coln2dataset(dat):
    label = list(dat.columns)
    dataset = [i.split(';')[1] for i in label]
    dataset = [i.split('.')[0] for i in dataset]
    return dataset


class Net(nn.Module):
    def __init__(self, feature_num):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(feature_num, 500)
        self.layer_2 = nn.Linear(500, 20)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class CellDataset(Dataset):
    def __init__(self, dat):
        self.data_frame = dat

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        train_x = self.data_frame.iloc[idx, :]
        train_x = torch.tensor(train_x, dtype=torch.float32)
        return train_x


class NPairSampler(Sampler):
    def __init__(self, labels):
        self.labels = labels

    def generate_npairs(self, labels):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        label_set, count = np.unique(labels, return_counts=True)
        label_set = label_set[count >= 2]
        pos_pairs = np.array([
            np.random.choice(np.where(labels == x)[0], 2, replace=False)
            for x in label_set
        ])
        neg_tuples = []
        for idx in range(len(pos_pairs)):
            neg_tuples.append(
                pos_pairs[np.delete(np.arange(len(pos_pairs)), idx), 1])
        neg_tuples = np.array(neg_tuples)
        sampled_npairs = [[a, p, *list(neg)]
                          for (a, p), neg in zip(pos_pairs, neg_tuples)]
        return iter(sampled_npairs)

    def __iter__(self):
        """
        This methods finds N-Pairs in a batch given by the classes provided in labels in the
        creation fashion proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'.
        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        sampled_npairs = self.generate_npairs(self.labels)
        while True:
            try:
                yield next(sampled_npairs)
            except StopIteration:
                sampled_npairs = self.generate_npairs(self.labels)
                yield next(sampled_npairs)


class NPairLoss(torch.nn.Module):
    def __init__(self, l2=0.05):
        """
        Basic N-Pair Loss as proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'
        Args:
            l2: float, weighting parameter for weight penality due to embeddings not being normalized.
        Returns:
            Nothing!
        """
        super(NPairLoss, self).__init__()
        self.l2 = l2

    def npair_distance(self, anchor, positive, negatives):
        """
        Compute basic N-Pair loss.
        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            n-pair loss (torch.Tensor())
        """
        return torch.log(1 + torch.sum(
            torch.exp(
                anchor.reshape(1, -1).mm((negatives -
                                          positive).transpose(0, 1)))))

    def weightsum(self, anchor, positive):
        """
        Compute weight penalty.
        NOTE: Only need to penalize anchor and positive since the negatives are created based on these.
        Args:
            anchor, positive: torch.Tensor(), resp. embeddings for anchor and positive samples.
        Returns:
            torch.Tensor(), Weight penalty
        """
        return torch.sum(anchor**2 + positive**2)

    def forward(self, batch):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
        Returns:
            n-pair loss (torch.Tensor(), batch-averaged)
        """
        loss = torch.stack([
            self.npair_distance(npair[0], npair[1], npair[2:])
            for npair in batch
        ])
        loss = loss + self.l2 * torch.mean(
            torch.stack(
                [self.weightsum(npair[0], npair[1]) for npair in batch]))
        return torch.mean(loss)


def get_dataloaders_and_LabelNumList_and_FeatureNum(train_dataset_list):
    dataloaders = []
    label_num_list = []
    for train_dataset in train_dataset_list:
        cell_dataset = CellDataset(train_dataset)
        tr_y = np.array(indexn2ct(train_dataset))
        npair_sampler = NPairSampler(tr_y)
        dataloader = DataLoader(cell_dataset,
                                batch_sampler=npair_sampler,
                                num_workers=5)
        label_num = len(np.unique(tr_y))
        dataloaders.append(dataloader)
        label_num_list.append(label_num)
    feature_num = cell_dataset.data_frame.shape[1]
    return dataloaders, label_num_list, feature_num


def train(model, dataloaders, optimizer, epoch, loss_function, device,
          label_num_list):
    model.train()
    for _epoch in tqdm(range(epoch)):
        loss_list = []
        for dataloader_id in range(len(dataloaders)):
            count = 0
            batch_data = []
            for i, data in enumerate(dataloaders[dataloader_id]):
                data = data.to(device)
                output = model(data)
                batch_data.append(output)
                count = count + 1
                if count % label_num_list[dataloader_id] == 0:
                    optimizer.zero_grad()
                    loss = loss_function(batch_data).view(1, -1)
                    print(loss)
                    loss_list.append(loss)
                    break
        total_loss = torch.sum(torch.cat(loss_list, dim=1))
        total_loss.backward()
        optimizer.step()
    return model


def get_MetricsList_and_LabelsList(model, train_dataset_list):
    metrics_list = []
    labels_list = []
    for l in range(len(train_dataset_list)):
        train_data = train_dataset_list[l]
        tr_x = train_data.values
        tr_y = indexn2ct(train_data)
        labels = np.unique(tr_y)
        metrics = calculate_metrics(tr_x, tr_y, model, labels)
        metrics_list.append(metrics)
        labels_list.append(labels)
    return metrics_list, labels_list


def calculate_metrics(tr_x, tr_y, model, labels):
    tr_y = np.array(tr_y)
    labels = np.array(labels)
    metrics = []
    for i in labels:
        #classify embedding data according to classes and calculate metrics
        class_indices = np.where(tr_y == i)[0]
        class_data = torch.tensor(tr_x[class_indices, :], dtype=torch.float32)
        class_embedding = model(class_data)
        class_embedding = class_embedding.detach().numpy()
        class_metric = np.median(class_embedding, axis=0)
        metrics.append(class_metric)
    return metrics


def test(model, test_data, train_dataset_list, metrics_list, labels_list):
    test_data = torch.tensor(test_data.values, dtype=torch.float32)
    max_likelihood_lists = []
    max_likelihood_classes = []
    for l in range(len(train_dataset_list)):
        max_likelihood_list, max_likelihood_class = one_model_predict(
            test_data, model, metrics_list[l], labels_list[l])
        max_likelihood_lists.append(max_likelihood_list)
        max_likelihood_classes.append(max_likelihood_class)
        # calculate f1_score
    pred_class = []
    max_likelihood_indices = np.argmax(max_likelihood_lists, axis=0)
    for k in range(len(max_likelihood_indices)):
        max_likelihood_indice = max_likelihood_indices[k]
        pred_class.append(max_likelihood_classes[max_likelihood_indice][k])
    return pred_class


def one_model_predict(test_data, model, metrics, labels):
    test_embedding = model(test_data).detach().numpy()
    max_likelihood_class = []
    max_likelihood_list = []
    for i in test_embedding:
        predict_pearsonr = []
        for k in metrics:
            predict_pearsonr.append(pearsonr(i, k)[0])
        pred = np.argmax(predict_pearsonr)
        max_likelihood_list.append(predict_pearsonr[pred])
        max_likelihood_class.append(labels[pred])
    return max_likelihood_list, max_likelihood_class


def main(train_dataset, test_dataset):
    torch.manual_seed(2)

    datasets = np.array(coln2dataset(train_dataset))
    train_dataset_list = [i for i in range(len(np.unique(datasets)))]
    for i in range(len(np.unique(datasets))):
        train_dataset_list[i] = train_dataset.iloc[:,
                                                   np.where(
                                                       datasets == np.
                                                       unique(datasets)[i])[0]]
        train_dataset_list[i] = train_dataset_list[i].T

    args = Setting()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = args.epoch

    dataloaders, label_num_list, feature_num = get_dataloaders_and_LabelNumList_and_FeatureNum(
        train_dataset_list)

    model = Net(feature_num).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = NPairLoss()
    print('train')
    model = train(model, dataloaders, optimizer, epoch, criterion, device,
                  label_num_list)

    #test
    print('test')
    model.cpu().eval()
    metrics_list, labels_list = get_MetricsList_and_LabelsList(
        model, train_dataset_list)

    pred_class = test(model, test_dataset, train_dataset_list, metrics_list,
                      labels_list)

    test_ct = indexn2ct(test_dataset)
    metric_summary = metric(test_ct, pred_class)
    return metric_summary


metric_res = {}
samples = sorted(glob.glob(path + tissue + '/rds/' + '/*rds'))

for i in range(len(samples)):
    samples[i] = ntpath.basename(samples[i]).replace('.rds', '')

for sample in samples:
    train_dataset = read_data(path_in + sample + '_train.txt')
    train_dataset = np.log2(train_dataset + 1)
    test_dataset = read_data(path_in + sample + '_test.txt')
    test_dataset = np.log2(test_dataset + 1)
    test_dataset = test_dataset.T
    metric_res[sample] = main(train_dataset, test_dataset)

with open(path_out + tissue + '_mtSC.pkl', 'wb') as f:
    pickle.dump(metric_res, f, pickle.HIGHEST_PROTOCOL)

summary = {}
summary['res'] = []
for sample in samples:
    summary['res'].append(metric_res[sample]['Acc'])
    summary['res'].append(metric_res[sample]['Macro_F1'])
summary['Method'] = ['mtSC'] * len(metric_res) * 2
summary['Metric'] = ['Acc', 'MacroF1'] * len(metric_res)
pd.DataFrame(summary).to_csv(path_out + tissue + '_mtSC_res.txt',
                             sep='\t',
                             index=False)
