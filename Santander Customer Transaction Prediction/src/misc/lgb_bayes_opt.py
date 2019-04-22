import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import warnings
from bayes_opt import BayesianOptimization
from collections import deque

windows_flag = False
SEED = 42
explore_n_iter = 10
exploit_n_iter = 10
num_folds = 5

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

gc.enable()

def scale_dfs(train_df, test_df):
    public_index = np.load("./input/public_LB.npy")
    private_index = np.load("./input/private_LB.npy")
    synthetic_index = np.load("./input/synthetic_samples_indexes.npy")
    private_df, public_df, synthetic_df = test_df.iloc[private_index], test_df.iloc[public_index], test_df.iloc[synthetic_index]
    
    df = pd.concat([train_df, private_df, public_df])
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean)/std
        test_df[col] = (test_df[col] - mean)/std
    return df[:len(train_df)], test_df

def getp_vec_sum(x,x_sort,y,std,c=0.5):
    # x is sorted
    left = x - std/c
    right = x + std/c
    p_left = np.searchsorted(x_sort,left)
    p_right = np.searchsorted(x_sort,right)
    p_right[p_right>=y.shape[0]] = y.shape[0]-1
    p_left[p_left>=y.shape[0]] = y.shape[0]-1
    return (y[p_right]-y[p_left])

def get_pdf(tr,col,x_query=None,smooth=3):
    std = tr[col].std()
    df = tr.groupby(col).agg({'target':['sum','count']})
    cols = ['sum_y','count_y']
    df.columns = cols
    df = df.reset_index()
    df = df.sort_values(col)
    y,c = cols
    
    df[y] = df[y].cumsum()
    df[c] = df[c].cumsum()
    
    if x_query is None:
        rmin,rmax,res = -5.0, 5.0, 501
        x_query = np.linspace(rmin,rmax,res)
    
    dg = pd.DataFrame()
    tm = getp_vec_sum(x_query,df[col].values,df[y].values,std,c=smooth)
    cm = getp_vec_sum(x_query,df[col].values,df[c].values,std,c=smooth)+1
    dg['res'] = tm/cm
    dg.loc[cm<500,'res'] = 0.1
    return dg['res'].values

def get_pdfs(tr):
    y = []
    for i in tr.columns:
        if i != "target":
            res = get_pdf(tr,i)
            y.append(res)
    return np.vstack(y)

def print_corr(corr_mat,col,bar=0.97):
    cols = corr_mat.loc[corr_mat[col]>bar,col].index.values
    return cols

def main():
    print('Load Train Data.')
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
    print('\nShape of Train Data: {}\t Shape of Test Data: {}'
        .format(train_df.shape, test_df.shape))
    
    train_labels = train_df['target']
    train_index = np.array(train_df.index)

    train_df.drop(['ID_code', 'target'], axis=1, inplace=True)
    test_df.drop(['ID_code'], axis=1, inplace=True)

    train_df, test_df = scale_dfs(train_df, test_df)

    reverse_list = [0,1,2,3,4,5,6,7,8,11,15,16,18,19,
                22,24,25,26,27,41,29,
                32,35,37,40,48,49,47,
                55,51,52,53,60,61,62,103,65,66,67,69,
                70,71,74,78,79,
                82,84,89,90,91,94,95,96,97,99,
                105,106,110,111,112,118,119,125,128,
                130,133,134,135,137,138,
                140,144,145,147,151,155,157,159,
                161,162,163,164,167,168,
                170,171,173,175,176,179,
                180,181,184,185,187,189,
                190,191,195,196,199]
    
    df = pd.concat([train_df, test_df])
    df.iloc[:,reverse_list] = df.iloc[:,reverse_list]*-1
    train_df, test_df = df[:len(train_df)], df[len(train_df):]

    train_abs_mean = np.mean(np.abs(train_df.values),axis=1)
    test_abs_mean = np.mean(np.abs(test_df.values),axis=1)

    train_df["mean_feats"] = np.mean(train_df.values, axis=1)
    test_df["mean_feats"] = np.mean(test_df.values, axis=1)

    train_df["abs_mean_feats"] = train_abs_mean
    test_df["abs_mean_feats"] = test_abs_mean

    tr = pd.concat([train_df, train_labels],axis=1)

    pdfs = get_pdfs(tr)
    df_pdf = pd.DataFrame(pdfs.T,columns=train_df.columns)
    corr_mat = df_pdf.corr(method='pearson')
    
    groups = []
    skip_list = set()
    for col in train_df.columns:
        if col not in skip_list:
            cols = print_corr(corr_mat,col)
            if len(cols)>1:
                groups.append(cols)
                for v in cols:
                    skip_list.add(v)
    
    for i, cols in enumerate(groups):
        train_df["mean_groupfeats_{}".format(i)] = np.mean(train_df[cols].values, axis=1)
        test_df["mean_groupfeats_{}".format(i)] = np.mean(test_df[cols].values, axis=1)

    int_params = {"num_leaves", "min_data_in_leaf","bagging_freq"}
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    best_fold_params = {}
    best_fold_scores = {}

    print('\nTraining model\n\n')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print("\n\nRunning Bayesian Optimization for Fold Number {}\n".format(counter+1))

        X_train, y_train = train_df.loc[ids[0],:], train_labels.loc[ids[0]]
        X_val, y_val = train_df.loc[ids[1], :], train_labels.loc[ids[1]]

        def LGB_bayesian(
            num_leaves,
            min_data_in_leaf,
            learning_rate,
            min_sum_hessian_in_leaf,
            feature_fraction,
            lambda_l1,
            lambda_l2,
            min_gain_to_split,
            bagging_fraction,
            bagging_freq):

            num_leaves = int(num_leaves)
            min_data_in_leaf = int(min_data_in_leaf)
            bagging_freq = int(bagging_freq)

            train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
            
            params = {
                'n_estimators':99999,
                'num_leaves': num_leaves,
                'min_data_in_leaf': min_data_in_leaf,
                'learning_rate': learning_rate,
                'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'feature_fraction': feature_fraction,
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2,
                'min_gain_to_split': min_gain_to_split,
                'max_depth': -1,
                'save_binary': True, 
                'seed': SEED,
                'feature_fraction_seed': SEED,
                'bagging_seed': SEED,
                'objective': 'binary',
                'verbose': 0,
                'metric': 'auc',
                'boost_from_average': False,
                'n_jobs': -1
            }
            
            model = lgb.train(params,train_data, valid_sets=valid_data, verbose_eval=0, early_stopping_rounds=500)
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)

            return roc_auc_score(y_val, val_preds)

        bounds_LGB = {
            'num_leaves': (2, 25), 
            'min_data_in_leaf': (5, 100),  
            'learning_rate': (0.001, 0.3),
            'min_sum_hessian_in_leaf': (0.00001, 15),    
            'feature_fraction': (0.01, 1.0),
            'lambda_l1': (0, 5.0), 
            'lambda_l2': (0, 5.0), 
            'min_gain_to_split': (0, 1.0),
            'bagging_fraction':(0.001, 1.0),
            'bagging_freq': (1,10)
        }
        LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=SEED)
        
        print(LGB_BO.space.keys)
        print('-' * 130)

        #LGB_BO.probe(params = {"min_gain_to_split":0,"lambda_l1":0, "lambda_l2": 0,"num_leaves" : 13,"learning_rate" : 0.01,"bagging_freq": 5,"bagging_fraction" : 0.4,"feature_fraction" : 0.05,"min_data_in_leaf": 80,"min_sum_hessian_in_leaf": 10},lazy=True)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            LGB_BO.maximize(init_points=explore_n_iter, n_iter=exploit_n_iter, acq='ucb', xi=0.0, alpha=1e-6)

        max_params = {
            'metric':'auc',
            'objective':'binary', 
            'n_jobs':-1,
            'n_estimators':99999,
            'max_depth':-1
        }
        for key, value in LGB_BO.max["params"].items():
            if key in int_params:
                max_params[key] = int(value)
            else:
                max_params[key] = value

        print("\nBest score in Fold {} is {}".format(counter+1, LGB_BO.max["target"]))
        best_fold_scores[counter] = LGB_BO.max["target"]
        
        print("\nBest params in Fold {} are {}".format(counter+1, max_params))
        best_fold_params[counter] = max_params

    print("\nSaving best parameters and scores accross {} folds!".format(num_folds))
    np.save(r'./output/misc/lgb_optimal_magic_parameters.npy', best_fold_params)
    np.save(r'./output/misc/lgb_optimal_magic_scores.npy', best_fold_scores)

if __name__ == "__main__":
    change_log = "Crap magic for each fold."
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
