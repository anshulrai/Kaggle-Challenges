from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import datetime
import time

windows_flag = False

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

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

    clf = LogisticRegression(random_state=42)
    clf.fit(val_data, oof_labels)
    val_preds = clf.predict_proba(val_data)[:,1]

    val_auc  = roc_auc_score(oof_labels, val_preds)
    print('Validation AUC: {}'.format(val_auc))

    n_test = len(pd.read_csv("./input/test.csv", usecols=["ID_code"]))
    test_data = np.zeros((n_test, len(submissions)))
    column_names = []
    for i, submission in enumerate(submissions):
        column_names.append(submission.split('.csv')[0])
        test_data[:,i] = pd.read_csv("./submissions/{}".format(submission), 
                                usecols=["target"]).values[:,0]
    test_preds = clf.predict_proba(test_data)[:,1]

    submission = pd.read_csv('./input/sample_submission.csv')
    submission['target'] = test_preds
    submission.to_csv('./submissions/logisticstack_{}.csv'.format(val_auc), index=False)

if __name__ == "__main__":
    change_log = "Considering only 3"
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)