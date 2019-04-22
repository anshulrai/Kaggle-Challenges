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
from sklearn.preprocessing import StandardScaler
import datetime

num_folds = 5
windows_flag = False
SEED = 42

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

def expand_data(df, target):
    df_1 = df[target==1]
    df_0 = df[target==0]
    df = pd.concat([df_1, df_0], axis=0, sort=False)
    target = pd.concat([target[target==1], target[target==0]], axis=0)
    for i in range(4):
        df = pd.concat([df, shuffle_features(df_1, 1, SEED+10*i)], axis=0, sort=False)
        target = pd.concat([target, pd.Series(np.ones(len(df_1)))], axis=0)
    for i in range(4):
        df = pd.concat([df, shuffle_features(df_0, 0, (i+2)*SEED+10*i)], axis=0, sort=False)
        target = pd.concat([target, pd.Series(np.zeros(len(df_0)))], axis=0)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    target = target.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df, target

def pred_transform(pred, beta):
    return 0.5*((2*np.abs(pred-0.5))**beta)*np.sign(pred-0.5) + 0.5

def display_importance(df, file_name, auc):
    features = df.groupby("feature").mean().sort_values(by="importance", ascending=False).index
    features_df = df.loc[df.feature.isin(features)]
    features_df = features_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(16, 25))
    sns.barplot(x="importance", y="feature", data=features_df.sort_values(by="importance", ascending=False))
    plt.title('Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(r'./output/feature importance/images/{}_{}.png'.format(file_name,auc))
    features_df.to_csv(r'./output/feature importance/files/{}_{}.csv'.format(file_name,auc), index=False)

def main():
    print('Load Train Data.')
    train_df = pd.read_csv('./input/train.csv')
    test_df = pd.read_csv('./input/test.csv')
    print('\nShape of Train Data: {}\t Shape of Test Data: {}'
        .format(train_df.shape, test_df.shape))

    train_df.drop(['ID_code'], axis=1, inplace=True)
    test_df.drop(['ID_code'], axis=1, inplace=True)

    train_labels = train_df['target']
    train_index = np.array(train_df.index)

    train_df.drop(['target'], axis=1, inplace=True)

    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    feature_importance = pd.DataFrame()
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    print('\nTraining model')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print('\nFold {}'.format(counter+1))
        X_train, y_train = train_df.loc[ids[0],:], train_labels.loc[ids[0]]

        print("Upsample train data in fold.")
        X_train, y_train = upsample_data(X_train, y_train)

        print("Expand train data in fold.")
        X_train, y_train = expand_data(X_train, y_train)

        X_val, y_val = train_df.loc[ids[1], :], train_labels.loc[ids[1]]

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
    
        model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000, early_stopping_rounds=500)
                    
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
    change_log = "Experimenting with increasing dataset of each fold x4"
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
