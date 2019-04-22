import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
import os
from sklearn.preprocessing import StandardScaler
import datetime

num_folds = 5
_num_folds = 4
num_categorical = 500
windows_flag = False

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

gc.enable()

def create_cat_vars(train_df, test_df):
    df = pd.concat([train_df, test_df])
    cat_vars = []
    for column in train_df.columns:
        df[column+"_cat"] = pd.cut(df[column], num_categorical).cat.codes
        cat_vars.append(column+"_cat")
    return df[:len(train_df)], df[len(train_df):], cat_vars

def create_target_encoding(train_df, test_df, cat_vars):
    aggs = ["mean","skew","var"]
    test_index = test_df.index
    for column in cat_vars:
        cat_df = train_df.groupby(column)["target"].agg(aggs)
        test_df = test_df.set_index(column)
        for agg in aggs:
            test_df[column+"_target_"+agg] = cat_df[agg]
        test_df = test_df.reset_index()
        test_df.drop(columns=[column],inplace=True)
        del cat_df
        gc.collect()
    test_df.index = test_index
    return train_df, test_df

def pred_transform(pred, beta):
    return 0.5*((2*np.abs(pred-0.5))**beta)*np.sign(pred-0.5) + 0.5

def display_importance(df, file_name, auc):
    features = df.groupby("feature").mean().sort_values(by="importance", ascending=False).index
    features_df = df.loc[df.feature.isin(features)]
    features_df = features_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(16, 75))
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
    
    train_labels = train_df['target']
    train_index = np.array(train_df.index)

    train_df.drop(['ID_code', 'target'], axis=1, inplace=True)
    test_df.drop(['ID_code'], axis=1, inplace=True)

    print("\nCreating categorical variables.\n")
    train_df, test_df, cat_vars = create_cat_vars(train_df, test_df)

    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    feature_importance = pd.DataFrame()
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    print('\nTraining model')
    for counter, ids in enumerate(skf.split(train_index, train_labels.values)):
        print('\nFold {}'.format(counter+1))
        y_train, y_val = train_labels.values[ids[0]], train_labels.values[ids[1]]

        _skf = StratifiedKFold(n_splits=_num_folds, shuffle=True, random_state=42)
        _train_index = ids[0]
        _train_labels = y_train
        X_train = pd.DataFrame()

        print("\nCreating target stats for training data\n")
        for _counter, _ids in enumerate(_skf.split(_train_index, _train_labels)):
            print('\nInner Fold {}'.format(_counter+1))
            _train_df = pd.concat([train_df.loc[_train_index[_ids[0]],:], train_labels.loc[_train_index[_ids[0]]]], axis=1)
            _test_df = train_df.loc[_train_index[_ids[1]],:]
            _train_df, _test_df = create_target_encoding(_train_df, _test_df, cat_vars)
            X_train = pd.concat([X_train, _test_df])
            del _train_df, _test_df
        X_train.sort_index(inplace=True)
        print("\nFinished creating target stats for training data\n")

        print("\nCreating target stats for validation and test data\n")
        fold_train_df = pd.concat([train_df.loc[ids[0],:], train_labels.loc[ids[0]]], axis=1)
        fold_train_df, X_val = create_target_encoding(fold_train_df, train_df.loc[ids[1],:], cat_vars)
        fold_train_df, fold_test_df = create_target_encoding(fold_train_df, test_df, cat_vars)
        del fold_train_df
        print("\nFinished creating target stats for validation and test data\n")

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        params = {
            'max_depth':-1,
            'n_estimators':99999,
            'learning_rate':0.02,
            'colsample_bytree':0.3,
            'num_leaves':2,
            'metric':'auc',
            'objective':'binary', 
            'n_jobs':-1,
            'boost_from_average': False
        }
    
        model = lgb.train(params,train_data, valid_sets=valid_data, verbose_eval=500, early_stopping_rounds=500)
                    
        fold_val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(fold_test_df, num_iteration=model.best_iteration)/num_folds

        oof_preds[ids[1]] += fold_val_preds

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = [feature for feature in X_train.columns]
        fold_importance["importance"] = model.feature_importance()

        feature_importance = pd.concat([feature_importance, fold_importance])

        del X_train, X_val, y_train, y_val, fold_test_df
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
    change_log = "Making target stats for all features with 500 bins"
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
