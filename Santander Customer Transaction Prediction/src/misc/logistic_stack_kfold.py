from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import os
import datetime
import time

num_folds = 5

def main():
    submissions = []
    for submission in os.listdir('./submissions/'):
        model_type = submission.split('_')[0]
        if model_type not in ("rank", "ridge", "logisticstack", "lgbmagic"):
            submissions.append(submission)

    print("Considering {} submission outputs.".format(len(submissions)))
        
    n_val = len(pd.read_csv("./input/train.csv", usecols=["target"]))
    val_data = np.zeros((n_val, len(submissions)))
    column_names = []
    for i, submission in enumerate(submissions):
        column_names.append(submission.split('.csv')[0])
        val_data[:,i] = pd.read_csv(r"./output/oof predictions/{}".format(submission), 
                                usecols=["target"]).values[:,0]
    oof_labels = pd.read_csv("./input/train.csv", usecols=["target"]).values[:,0]

    n_test = len(pd.read_csv("./input/test.csv", usecols=["ID_code"]))
    test_data = np.zeros((n_test, len(submissions)))
    column_names = []
    for i, submission in enumerate(submissions):
        column_names.append(submission.split('.csv')[0])
        test_data[:,i] = pd.read_csv("./submissions/{}".format(submission), 
                                usecols=["target"]).values[:,0]

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(n_val)
    test_preds = np.zeros(n_test)

    for counter, ids in enumerate(skf.split(np.arange(n_val), oof_labels)):
        print('\nFold {}'.format(counter+1))

        X_train, y_train = val_data[ids[0]], oof_labels[ids[0]]
        X_val, y_val = val_data[ids[1]], oof_labels[ids[1]]

        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        fold_val_preds = clf.predict_proba(X_val)[:,1]
        oof_preds[ids[1]] += fold_val_preds

        fold_val_auc  = roc_auc_score(y_val, fold_val_preds)
        print('Fold Validation AUC: {}'.format(fold_val_auc))

        test_preds += clf.predict_proba(test_data)[:,1]/num_folds

    val_auc  = roc_auc_score(oof_labels, oof_preds)
    print('Overall Validation AUC: {}'.format(val_auc))

    submission = pd.read_csv('./input/sample_submission.csv')
    submission['target'] = test_preds
    submission.to_csv('./submissions/logisticstack_kfold_{}.csv'.format(val_auc), index=False)

if __name__ == "__main__":
    change_log = "Kfold logistic stack init"
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)