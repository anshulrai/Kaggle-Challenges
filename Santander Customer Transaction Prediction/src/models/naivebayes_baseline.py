import numpy as np
import pandas as pd
import gc
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
import os
from sklearn.preprocessing import StandardScaler
import datetime
from scipy.stats.kde import gaussian_kde

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

def scale_col(col):
    max_val = col.max()
    min_val = col.min()
    return (col-min_val)/(max_val-min_val)

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

    oof_preds = np.zeros(train_df.shape[0])
    test_preds = np.zeros(test_df.shape[0])
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    print('\nTraining model')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print('\nFold {}'.format(counter+1))
        X_train, y_train = train_df.loc[ids[0],:], train_labels[ids[0]]
        X_val, y_val = train_df.loc[ids[1], :], train_labels[ids[1]]

        X_train, y_train = upsample_data(X_train, y_train)

        pos_idx = (y_train==1)
        neg_idx = (y_train==0)
        prior_pos = pos_idx.sum()/len(y_train)
        prior_neg = 1-prior_pos

        pos_kdes,neg_kdes=[],[]

        for col in X_train.columns:
            pos_kde = gaussian_kde(X_train.loc[pos_idx,col])
            neg_kde = gaussian_kde(X_train.loc[neg_idx,col])
            pos_kdes.append(pos_kde)
            neg_kdes.append(neg_kde)

        def cal_prob_KDE_col_i(df,i,num_of_bins=100):
            bins = pd.cut(df.iloc[:,i],bins=num_of_bins)
            uniq = bins.unique()
            uniq_mid = uniq.map(lambda x:(x.left+x.right)/2)
            mapping=pd.DataFrame({
                'pos':pos_kdes[i](uniq_mid),
                'neg':neg_kdes[i](uniq_mid)
            },index=uniq)
            return bins.map(mapping['pos'])/bins.map(mapping['neg'])

        fold_val_probs = [[prior_pos/prior_neg]*len(X_val)]
        for i in range(len(X_val.columns)):
            fold_val_probs.append(cal_prob_KDE_col_i(X_val,i))
        
        val_NB_df = pd.DataFrame(np.array(fold_val_probs).T,columns=['prior']+['var_'+str(i) for i in range(len(X_val.columns))])
        fold_val_preds = val_NB_df.apply(lambda x: np.prod(x),axis=1)

        test_probs = [[prior_pos/prior_neg]*len(test_df)]
        for i in range(len(test_df.columns)):
            test_probs.append(cal_prob_KDE_col_i(test_df,i))
        
        test_NB_df = pd.DataFrame(np.array(test_probs).T,columns=['prior']+['var_'+str(i) for i in range(len(test_df.columns))])

        test_preds += test_NB_df.apply(lambda x: np.prod(x),axis=1)/num_folds

        oof_preds[ids[1]] += fold_val_preds

        auc_fold  = roc_auc_score(y_val, fold_val_preds)

        print('Fold {} Validation AUC: {}'.format(counter+1, auc_fold))
        del X_train, X_val, y_train, y_val
        gc.collect()
    
    auc_lgb  = roc_auc_score(train_labels, oof_preds)

    print('Validation AUC: {}'.format(auc_lgb))
    
    print('Saving OOF predictions!')
    file_name = str(__file__).split('.')[0].split('\\')[2] if windows_flag else str(__file__).split('.')[0].split('/')[2]
    oof_csv = pd.DataFrame(data={'target':scale_col(oof_preds)},index=train_index)
    oof_csv.to_csv(r'./output/oof predictions/{}_{}.csv'.format(file_name,auc_lgb), index=False)
    
    print('Making Submission File!')
    submission = pd.read_csv('./input/sample_submission.csv')
    submission['target'] = scale_col(test_preds)
    submission.to_csv('./submissions/{}_{}.csv'.format(file_name,auc_lgb), index=False)

if __name__ == "__main__":
    change_log = "Baseline naive bayes run."
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)