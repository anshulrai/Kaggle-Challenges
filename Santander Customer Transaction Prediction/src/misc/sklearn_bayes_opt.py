import numpy as np
import pandas as pd
import gc
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import warnings
from bayes_opt import BayesianOptimization

windows_flag = False
SEED = 42
explore_n_iter = 5
exploit_n_iter = 5
num_folds = 5

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

gc.enable()

def shuffle_features(df, target, seed):
    _seed = seed
    _df = pd.DataFrame()
    for column in df.columns:
        _df[column] = df[column].sample(frac=1,random_state=_seed).values
        _seed += 1
    return _df

def upsample_data(df, target):
    df_1 = df[target==1]
    for i in range(8):
        df = pd.concat([df, shuffle_features(df_1, 1, SEED+10*i)], axis=0, sort=False)
        target = pd.concat([target, pd.Series(np.ones(len(df_1)))], axis=0)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    target = target.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df, target

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

    int_params = {"num_leaves", "min_data_in_leaf","bagging_freq"}
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    best_fold_params = {}
    best_fold_scores = {}

    print('\nTraining model\n\n')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print("\n\nRunning Bayesian Optimization for Fold Number {}\n".format(counter+1))

        X_train, y_train = train_df.loc[ids[0],:], train_labels.loc[ids[0]]

        print("Upsample train data in fold.")
        X_train, y_train = upsample_data(X_train, y_train)

        X_val, y_val = train_df.loc[ids[1], :], train_labels.loc[ids[1]]

        def GBC_bayesian(
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

            model.fit(X_train, y_train)
            
            #model = lgb.train(params,train_data, valid_sets=valid_data, verbose_eval=0, early_stopping_rounds=500)
            val_preds = model.predict_proba(X_val)[:,1]

            return roc_auc_score(y_val, val_preds)

        bounds_GBC = {
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
        GBC_BO = BayesianOptimization(GBC_bayesian, bounds_GBC, random_state=SEED)
        
        print(GBC_BO.space.keys)
        print('-' * 130)

        GBC_BO.probe(params = {"min_gain_to_split":0,"lambda_l1":0, "lambda_l2": 0,"num_leaves" : 13,"learning_rate" : 0.01,"bagging_freq": 5,"bagging_fraction" : 0.4,"feature_fraction" : 0.05,"min_data_in_leaf": 80,"min_sum_hessian_in_leaf": 10},lazy=True)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            GBC_BO.maximize(init_points=explore_n_iter, n_iter=exploit_n_iter, acq='ucb', xi=0.0, alpha=1e-6)

        max_params = {
            'metric':'auc',
            'objective':'binary', 
            'n_jobs':-1,
            'n_estimators':99999,
            'max_depth':-1
        }
        for key, value in GBC_BO.max["params"].items():
            if key in int_params:
                max_params[key] = int(value)
            else:
                max_params[key] = value

        print("\nBest score in Fold {} is {}".format(counter+1, GBC_BO.max["target"]))
        best_fold_scores[counter] = GBC_BO.max["target"]
        
        print("\nBest params in Fold {} are {}".format(counter+1, max_params))
        best_fold_params[counter] = max_params

    print("\nSaving best parameters and scores accross {} folds!".format(num_folds))
    np.save(r'./output/misc/lgb_optimal_parameters.npy', best_fold_params)
    np.save(r'./output/misc/lgb_optimal_scores.npy', best_fold_scores)

if __name__ == "__main__":
    change_log = "Upsampling for each fold."
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
