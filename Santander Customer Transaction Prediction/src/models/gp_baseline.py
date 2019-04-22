import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
import os
import datetime
import gplearn
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

num_folds = 5
windows_flag = False
gp_generations = 5

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

gc.enable()

def fitness_metric(y, y_pred, w):
    diffs = np.abs(y - y_pred)  
    return 100. * np.average(diffs, weights=w)

def pred_transform(pred, beta):
    return 0.5*((2*np.abs(pred-0.5))**beta)*np.sign(pred-0.5) + 0.5

def display_importance(df, file_name, auc):
    features = df.groupby("feature").mean().sort_values(by="importance", ascending=False).index
    features_df = df.loc[df.feature.isin(features)]
    features_df = features_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(16, 40))
    sns.barplot(x="importance", y="feature", data=features_df.sort_values(by="importance", ascending=False))
    plt.title('Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(r'./output/feature importance/images/{}_{}.png'.format(file_name,auc))
    features_df.to_csv(r'./output/feature importance/files/{}_{}.csv'.format(file_name,auc), index=False)

def main():
    file_name = str(__file__).split('.')[0].split('\\')[2] if windows_flag else str(__file__).split('.')[0].split('/')[2]
    print('Load Train Data.')
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
    print('\nShape of Train Data: {}\t Shape of Test Data: {}'
        .format(train_df.shape, test_df.shape))
    
    train_labels = np.array(train_df['target'])
    train_index = np.array(train_df.index)

    train_df.drop(['ID_code', 'target'], axis=1, inplace=True)
    test_df.drop(['ID_code'], axis=1, inplace=True)

    print("\nMaking GP features.\n")

    my_fit = make_fitness(fitness_metric, greater_is_better=False)
    function_set = ['add', 'sub', 'mul', 'div', 'log', 
                'sqrt', 'log', 'abs', 'neg', 'inv', 
                'max', 'min', 
                'sin', 'cos', 'tan' ]

    gp = SymbolicRegressor(function_set=function_set, metric = my_fit,
                        verbose=0, generations=gp_generations, 
                        random_state=42, n_jobs=-1)

    gp_train = np.zeros((train_df.shape))
    gp_test = np.zeros((test_df.shape))

    train_numpy = np.array(train_df)
    test_numpy = np.array(test_df)

    for i in range(train_numpy.shape[1]) :
        print("\nGP for feature {}".format(i))
        X = np.delete(train_numpy,i,1)
        y = train_numpy[:,i]
        gp.fit(X, y) 
        gp_train[:,i] = gp.predict(X)
        X = np.delete(test_numpy, i, 1)
        gp_test[:,i] = gp.predict(X)
    
    gp_train_df = pd.DataFrame(gp_train, columns=["gp_{}".format(i) for i in range(train_numpy.shape[1])])
    gp_test_df = pd.DataFrame(gp_test, columns=["gp_{}".format(i) for i in range(train_numpy.shape[1])])

    train_df = gp_train_df
    test_df = gp_test_df

    print("\nCreating GP feature files.\n")
    train_df.to_csv(r'./output/misc/{}_train_{}.csv'.format(file_name,gp_generations), index=False)
    test_df.to_csv(r'./output/misc/{}_test_{}.csv'.format(file_name,gp_generations), index=False)

    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    feature_importance = pd.DataFrame()
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    print('\nTraining model')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print('\nFold {}'.format(counter+1))
        X_train, y_train = train_df.values[ids[0]], train_labels[ids[0]]
        X_val, y_val = train_df.values[ids[1]], train_labels[ids[1]]

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        params = {
        'n_estimators':99999,
        "objective" : "binary",
        "metric" : "auc",
        "max_depth" : -1,
        "num_leaves" : 13,
        "learning_rate" : 0.01,
        "bagging_freq": 5,
        "bagging_fraction" : 0.4,
        "feature_fraction" : 0.05,
        "min_data_in_leaf": 80,
        "min_sum_hessian_in_leaf": 10,
        "boost_from_average": "false",
        "bagging_seed" : 42,
        "verbosity" : 1,
        "seed": 42
        }
    
        model = lgb.train(params,train_data, valid_sets=valid_data, verbose_eval=5000, early_stopping_rounds=500)
                    
        fold_val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(test_df, num_iteration=model.best_iteration)/num_folds

        oof_preds[ids[1]] += fold_val_preds

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = [feature for feature in train_df.columns]
        fold_importance["importance"] = model.feature_importance()

        feature_importance = pd.concat([feature_importance, fold_importance])

        del X_train, X_val, y_train, y_val
        gc.collect()

    auc  = roc_auc_score(train_labels, oof_preds)

    print('Validation AUC: {}'.format(auc))
    
    print('Saving OOF predictions!')
    file_name = str(__file__).split('.')[0].split('\\')[2] if windows_flag else str(__file__).split('.')[0].split('/')[2]
    oof_csv = pd.DataFrame(data={'target':oof_preds},index=train_index)
    oof_csv.to_csv(r'./output/oof predictions/{}_{}.csv'.format(file_name,auc), index=False)
    
    print('Making Feature Importance File!')
    display_importance(feature_importance, file_name, auc)

    print('Making Submission File!')
    submission = pd.read_csv('./input/sample_submission.csv')
    submission['target'] = test_preds
    submission.to_csv('./submissions/{}_{}.csv'.format(file_name,auc), index=False)

if __name__ == "__main__":
    change_log = "Pure GP run"
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
