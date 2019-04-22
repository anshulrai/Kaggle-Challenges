from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np 
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import gc
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata

num_folds = 5
windows_flag = False

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

gc.enable()

def prob_preds(df, stats_df):
    neg_z = (df.values - stats_df.neg_mean.values) / stats_df.neg_sd.values
    neg_p = (1 - norm.cdf(np.abs(neg_z))) * 2
    neg_prob = neg_p.prod(axis=1)

    pos_z = (df.values - stats_df.pos_mean.values) / stats_df.pos_sd.values
    pos_p = (1 - norm.cdf(np.abs(pos_z))) * 2
    pos_prob = pos_p.prod(axis=1)
    return pos_prob/neg_prob

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
        X_train, y_train = train_df.loc[ids[0],:], train_labels.values[ids[0]]
        X_val, y_val = train_df.loc[ids[1], :], train_labels.values[ids[1]]

        s = [0]*200
        m = [0]*200
        for i in range(200):
            s[i] = np.std(X_train['var_'+str(i)])
            m[i] = np.mean(X_train['var_'+str(i)])

        rmin=-5 
        rmax=5 
        res=501
        pr = 0.1 * np.ones((200,res))
        train0 = X_train[y_train==0].copy()
        train1 = X_train[y_train==1].copy()

        def getp(i,x):
            c = 3 #smoothing factor
            a = len( train1[ (train1['var_'+str(i)]>x-s[i]/c)&(train1['var_'+str(i)]<x+s[i]/c) ] ) 
            b = len( train0[ (train0['var_'+str(i)]>x-s[i]/c)&(train0['var_'+str(i)]<x+s[i]/c) ] )
            if a+b<500: return 0.1 #smoothing factor
            # RETURN PROBABILITY
            return a / (a+b)

        print("Creating probabilities!")
        for num in range(200):
            ct = 0
            for i in np.linspace(rmin,rmax,res):
                pr[num,ct] = getp(num,m[num]+i*s[num])
                ct += 1
        print("Done creating probabilities!")
        
        def getp2(i,x):
            z = (x-m[i])/s[i]
            ss = (rmax-rmin)/(res-1)
            if res%2==0: 
                idx = min( (res+1)//2 + z//ss, res-1)
            else: 
                idx = min( (res+1)//2 + (z-ss/2)//ss, res-1)
            idx = max(idx,0)
            return pr[i,int(idx)]
        
        fold_val_preds = [0]*len(X_val)
        ct = 0
        for r in range(len(X_val)):
            p = 0.1
            for i in range(200):
                p *= 10*getp2(i,X_val.iloc[r,i])
            fold_val_preds[ct]=p
            ct += 1

        test_pred = np.array([0]*len(test_df))
        ct = 0
        for r in range(len(test_df)):
            p = 0.1
            for i in range(200):
                p *= 10*getp2(i,test_df.iloc[r,i])
            test_pred[ct]=p
            ct += 1

        test_preds += test_pred/num_folds

        print("AUC score: {}".format(roc_auc_score(y_val, fold_val_preds)))
        oof_preds[ids[1]] += fold_val_preds

        del X_train, X_val, y_train, y_val
        gc.collect()

    auc  = roc_auc_score(train_labels, oof_preds)

    print('Validation AUC: {}'.format(auc))
    
    print('Saving OOF predictions!')
    file_name = str(__file__).split('.')[0].split('\\')[2] if windows_flag else str(__file__).split('.')[0].split('/')[2]
    oof_csv = pd.DataFrame(data={'target':oof_preds},index=train_index)
    oof_csv.to_csv(r'./output/oof predictions/{}_{}.csv'.format(file_name,auc), index=False)

    print('Making Submission File!')
    submission = pd.read_csv('./input/sample_submission.csv')
    submission['target'] = test_preds
    submission.to_csv('./submissions/{}_{}.csv'.format(file_name,auc), index=False)

if __name__ == "__main__":
    change_log = "init modified naive bayes run"
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
