import numpy as np
import pandas as pd
import gc
import time
import xgboost as xgb
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

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

gc.enable()

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
    
    train_labels = np.array(train_df['target'])
    train_index = np.array(train_df.index)

    train_df.drop(['ID_code', 'target'], axis=1, inplace=True)
    test_df.drop(['ID_code'], axis=1, inplace=True)

    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    feature_importance = pd.DataFrame()
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    optimal_params = np.load("./output/misc/xgb_optimal_parameters.npy").item()
    optimal_scores = np.load("./output/misc/xgb_optimal_scores.npy").item()
    test_df = xgb.DMatrix(test_df.values)

    print('\nTraining model')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print('\nFold {}'.format(counter+1))
        X_train, y_train = train_df.values[ids[0]], train_labels[ids[0]]
        X_val, y_val = train_df.values[ids[1]], train_labels[ids[1]]

        train_data = xgb.DMatrix(X_train, label=y_train, nthread=-1)
        valid_data = xgb.DMatrix(X_val, label=y_val, nthread=-1)

        params = optimal_params[counter]
        params["silent"] = 1
    
        eval_list = [(train_data, 'train'), (valid_data, 'eval')]
        model = xgb.train(params,train_data, num_boost_round=9999, evals=eval_list, verbose_eval=100, early_stopping_rounds=100)
                    
        fold_val_preds = model.predict(valid_data)
        
        test_preds += model.predict(test_df)/num_folds

        oof_preds[ids[1]] += fold_val_preds

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = [feature for feature in train_df.columns]
        fold_importance["importance"] = model.get_score()

        feature_importance = pd.concat([feature_importance, fold_importance])

        fold_auc  = roc_auc_score(y_val, fold_val_preds)
        print("Fold AUC: {}\tOptimal AUC: {}".format(fold_auc, optimal_scores[counter]))
            
        del X_train, X_val, y_train, y_val
        gc.collect()

    auc  = roc_auc_score(train_labels, oof_preds)

    print('Validation AUC: {}'.format(auc))
    
    print('Saving OOF predictions!')
    file_name = str(__file__).split('.')[0].split('\\')[2] if windows_flag else str(__file__).split('.')[0].split('/')[2]
    oof_csv = pd.DataFrame(data={'target':oof_preds},index=train_index)
    oof_csv.to_csv(r'./output/oof predictions/{}_{}.csv'.format(file_name,auc), index=False)
    
    print('Making Feature Importance File!')
    #display_importance(feature_importance, file_name, auc)

    print('Making Submission File!')
    submission = pd.read_csv('./input/sample_submission.csv')
    submission['target'] = test_preds
    submission.to_csv('./submissions/{}_{}.csv'.format(file_name,auc), index=False)

if __name__ == "__main__":
    change_log = "Newest optimal hyperparams with 20 runs."
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)