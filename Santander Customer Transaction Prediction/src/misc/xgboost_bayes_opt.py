import numpy as np
import pandas as pd
import gc
import time
import xgboost as xgb
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

    int_params = {"max_depth"}
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    best_fold_params = {}
    best_fold_scores = {}

    print('\nTraining model\n\n')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print("\n\nRunning Bayesian Optimization for Fold Number {}\n".format(counter+1))

        X_train, y_train = train_df.values[ids[0]], train_labels[ids[0]]
        X_val, y_val = train_df.values[ids[1]], train_labels[ids[1]]

        def XGB_bayesian(learning_rate, gamma, max_depth, min_child_weight,
        max_delta_step, subsample, colsample_bytree, reg_lambda, reg_alpha):

            train_data = xgb.DMatrix(X_train, label=y_train, nthread=-1)
            valid_data = xgb.DMatrix(X_val, label=y_val, nthread=-1)
            
            max_depth = int(max_depth)
            params = {
                'learning_rate': learning_rate,
                'gamma': gamma,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'max_delta_step': max_delta_step,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_lambda': reg_lambda, 
                'reg_alpha': reg_alpha,
                'objective':'binary:logistic', 
                'n_jobs':-1,
                'eval_metric':'auc'
            }
            eval_list = [(train_data, 'train'), (valid_data, 'eval')]
            model = xgb.train(params,train_data, num_boost_round=9999, evals=eval_list, verbose_eval=100, early_stopping_rounds=100)
            val_preds = model.predict(valid_data)

            return roc_auc_score(y_val, val_preds)

        #scale_pos_weight = 0.1117163789174106
        bounds_XGB = {  
            'learning_rate': (0.001, 0.3),
            'gamma': (0,5),
            'max_depth': (2,20),
            'min_child_weight': (0,10),
            'max_delta_step': (0,10),
            'subsample': (0.01, 1),
            'colsample_bytree': (0.01,1),
            'reg_lambda': (0, 5.0), 
            'reg_alpha': (0, 5.0)
        }
        XGB_BO = BayesianOptimization(XGB_bayesian, bounds_XGB, random_state=SEED)
    
        print(XGB_BO.space.keys)
        print('-' * 130)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            XGB_BO.maximize(init_points=explore_n_iter, n_iter=exploit_n_iter, acq='ucb', xi=0.0, alpha=1e-6)

        max_params = {
            'objective':'binary:logistic', 
            'n_jobs':-1,
            'eval_metric':'auc'
        }
        for key, value in XGB_BO.max["params"].items():
            if key in int_params:
                max_params[key] = int(value)
            else:
                max_params[key] = value

        print("\nBest score in Fold {} is {}".format(counter+1, XGB_BO.max["target"]))
        best_fold_scores[counter] = XGB_BO.max["target"]
        
        print("\nBest params in Fold {} are {}".format(counter+1, max_params))
        best_fold_params[counter] = max_params

    print("\nSaving best parameters and scores accross {} folds!".format(num_folds))
    np.save(r'./output/misc/xgb_optimal_parameters.npy', best_fold_params)
    np.save(r'./output/misc/xgb_optimal_scores.npy', best_fold_scores)

if __name__ == "__main__":
    change_log = "Added Bayes Opt for each fold in xgb."
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
