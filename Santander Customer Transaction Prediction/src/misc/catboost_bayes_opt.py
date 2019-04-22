import numpy as np
import pandas as pd
import gc
import time
import catboost as cb
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
explore_n_iter = 10
exploit_n_iter = 10
num_folds = 5

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

gc.enable()

def main():
    print('Load Train Data.')
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
    print('\nShape of Train Data: {}\t Shape of Test Data: {}'
        .format(train_df.shape, test_df.shape))
    
    train_labels = np.array(train_df['target'])
    train_index = np.array(train_df.index)

    train_df.drop(['ID_code', 'target'], axis=1, inplace=True)
    test_df.drop(['ID_code'], axis=1, inplace=True)

    int_params = {"max_depth", "max_bin", "reg_lambda"}
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    best_fold_params = {}
    best_fold_scores = {}

    print('\nTraining model\n\n')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print("\n\nRunning Bayesian Optimization for Fold Number {}\n".format(counter+1))

        X_train, y_train = train_df.values[ids[0]], train_labels[ids[0]]
        X_val, y_val = train_df.values[ids[1]], train_labels[ids[1]]

        def CB_bayesian(
            learning_rate,
            colsample_bylevel,
            reg_lambda,
            max_depth,
            max_bin
            ):

            max_depth = int(max_depth)
            max_bin = int(max_bin)
            reg_lambda = int(reg_lambda)

            valid_data = cb.Pool(X_val, label=y_val)
            
            params = {
                'n_estimators':99999,                
                'learning_rate': learning_rate,
                'colsample_bylevel': colsample_bylevel,
                'reg_lambda': reg_lambda,
                'max_depth': max_depth,
                'max_bin' : max_bin,
                'random_seed': SEED,
                'objective': 'Logloss',
                'boosting_type': 'Plain',
                'eval_metric': 'AUC'
            }
            
            model = cb.CatBoost(params)
            model.fit(X_train, y=y_train, eval_set=valid_data, verbose_eval=1000, early_stopping_rounds=500)
            val_preds = model.predict(valid_data, prediction_type='Probability')

            return roc_auc_score(y_val, val_preds)

        bounds_CB = { 
            'learning_rate': (0.001, 0.3),  
            'colsample_bylevel': (0.01, 1.0),
            'reg_lambda': (0, 10.0), 
            'max_depth':(2,20),
            'max_bin': (20,120),
        }
        CB_BO = BayesianOptimization(CB_bayesian, bounds_CB, random_state=SEED)
    
        print(CB_BO.space.keys)
        print('-' * 130)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            CB_BO.maximize(init_points=explore_n_iter, n_iter=exploit_n_iter, acq='ucb', xi=0.0, alpha=1e-6)

        max_params = {
            'objective': 'Logloss',
            'boosting_type': 'Plain',
            'eval_metric': 'AUC',
            'verbose': 0,
            'n_estimators':99999
        }
        for key, value in CB_BO.max["params"].items():
            if key in int_params:
                max_params[key] = int(value)
            else:
                max_params[key] = value

        print("\nBest score in Fold {} is {}".format(counter+1, CB_BO.max["target"]))
        best_fold_scores[counter] = CB_BO.max["target"]
        
        print("\nBest params in Fold {} are {}".format(counter+1, max_params))
        best_fold_params[counter] = max_params

    print("\nSaving best parameters and scores accross {} folds!".format(num_folds))
    np.save(r'./output/misc/cb_optimal_parameters.npy', best_fold_params)
    np.save(r'./output/misc/cb_optimal_scores.npy', best_fold_scores)

if __name__ == "__main__":
    change_log = "Init catboost Bayes Opt for each fold."
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
