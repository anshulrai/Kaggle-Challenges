import numpy as np
import pandas as pd
import gc
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
import os
import datetime

num_folds = 5
windows_flag = False

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

gc.enable()

class GaussianMixtureNB(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=1, reg_covar=1e-06):
        self.n_components = n_components
        self.reg_covar = reg_covar

    def fit(self, X, y):
        self.log_prior_ = np.log(np.bincount(y) / len(y))
        # shape of self.log_pdf_
        shape = (len(self.log_prior_), X.shape[1])
        self.log_pdf_ = [[GaussianMixture(n_components=self.n_components,
                                          reg_covar=self.reg_covar)
                          .fit(X[y == i, j:j + 1])
                          .score_samples for j in range(shape[1])]
                         for i in range(shape[0])]

    def predict_proba(self, X):
        # shape of log_likelihood before summing
        shape = (len(self.log_prior_), X.shape[1], X.shape[0])
        log_likelihood = np.sum([[self.log_pdf_[i][j](X[:, j:j + 1])
                                  for j in range(shape[1])]
                                 for i in range(shape[0])], axis=1).T
        log_joint = self.log_prior_ + log_likelihood
        return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
        
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

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
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    print('\nTraining model')
    for counter, ids in enumerate(skf.split(train_index, train_labels)):
        print('\nFold {}'.format(counter+1))
        X_train, y_train = train_df.values[ids[0]], train_labels[ids[0]]
        X_val, y_val = train_df.values[ids[1]], train_labels[ids[1]]

        model = make_pipeline(StandardScaler(),
                         GaussianMixtureNB(n_components=10, reg_covar=0.03))
        
        model.fit(X_train, y_train)
                    
        fold_val_preds = model.predict_proba(X_val)[:,1]
        test_preds += model.predict_proba(test_df)[:,1]/num_folds

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
    change_log = "GMM baseline"
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)
