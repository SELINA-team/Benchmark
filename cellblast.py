import pandas as pd
import numpy as np
import tensorflow as tf
import Cell_BLAST as cb
import glob
import ntpath
import datatable as dt
import scipy.sparse as sp_sparse
import pickle

path = '/public/home/wanglab/pengfei/performance/'
tissue = 'Lung'
path_in = path + tissue + '/data/' 
path_out = path + tissue + '/res/' 

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

def preprocessing(path_in,i):
    samples = sorted(glob.glob(path_in + '/*_expr.txt'))
    metas = sorted(glob.glob(path_in + '/*_meta.txt'))
    samples.pop(i)
    metas.pop(i)
    train_objs = []
    studys = []
    print('Loading data')
    for i in range(len(samples)):
        train_set = read_expr(samples[i])
        meta = pd.read_csv(metas[i], sep='\t', header=0)
        study = ntpath.basename(samples[i]).replace('_expr.txt','')
        studys.append(study)
        meta['study'] = study
        # print(meta)
        train_set.index.name = None
        meta.index = train_set.T.index
        train_obj = cb.data.ExprDataSet(exprs = sp_sparse.csc_matrix(train_set.T, dtype=np.float32), obs=meta, var=pd.DataFrame({}, index=train_set.index),uns=train_set.index)
        # selected_genes, axes = train_obj.find_variable_genes()
        # train_obj.uns = {'seurat_genes':selected_genes}
        train_objs.append(train_obj)

    train_objs_dict = {}
    for i in range(len(studys)):
        train_objs_dict[studys[i]] = train_objs[i]
    combind_sets = cb.data.ExprDataSet.merge_datasets(train_objs_dict, meta_col="study")
    return combind_sets


def read_expr(path):
    expr = dt.fread(path, header=True, sep='\t', nthreads=6, fill=True)
    expr = expr.to_pandas()
    expr.index = expr.loc[:, 'Gene']
    del expr['Gene']
    return expr


samples = sorted(glob.glob(path_in  + '*_expr.txt'))
res={}
for i in range(len(samples)):
    print(i)
    sample = ntpath.basename(samples[i]).replace('_expr.txt','')
    combind_sets = preprocessing(path_in,i)
    models = []
    for j in range(1):
        models.append(cb.directi.fit_DIRECTi(
            combind_sets,
            batch_effect='study', latent_dim=10, cat_dim=20, random_seed=j
        ))
    blast = cb.blast.BLAST(models, combind_sets)
    query_expr = read_expr(samples[i])
    query_expr.index.name = None
    query_exprs=query_expr.T
    query_obs = pd.read_csv(samples[i].replace('expr.txt','meta.txt'), sep='\t')
    query_obj = cb.data.ExprDataSet(exprs = sp_sparse.csc_matrix(query_exprs, dtype=np.float32), obs=query_obs, var=pd.DataFrame({}, index=query_expr.index),uns=query_expr.index)
    # query_selected_genes, axes = query_obj.find_variable_genes()
    # query_obj.uns = {'seurat_genes':query_selected_genes}
    query_hits = blast.query(query_obj)
    result=query_hits.annotate('Celltype')
    res[sample]=metric(query_obs.Celltype, result.Celltype)

summary = {}
summary['res'] = []
for i in range(len(samples)):
    sample = ntpath.basename(samples[i]).replace('_expr.txt','')
    summary['res'].append(res[sample]['Acc'])
    summary['res'].append(res[sample]['Macro_F1'])
summary['Method'] = ['CellBlast'] * len(res)*2
summary['Metric'] = ['Acc','MacroF1'] * len(res)
pd.DataFrame(summary).to_csv(path_out + tissue + '_CellBlast_res.txt',sep='\t',index=False)

with open(path_out + tissue + '_CellBlast.pkl', 'wb') as f:
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
