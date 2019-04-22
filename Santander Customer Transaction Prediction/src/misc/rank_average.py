from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import time
import datetime
import os

weighted_flag = True
windows_flag = False

print("Running on Windows!\n") if windows_flag else print("Running on Linux!\n")

def main():
    oof_val = "weighted" if weighted_flag else "notweighted"

    submissions = []
    for submission in os.listdir('./submissions/'):
        model_type = submission.split('_')[0]
        if model_type not in ("rank", "ridge", "logisticstack"):
            submissions.append(submission)
    print("Considering {} submission outputs.".format(len(submissions)))

    n_test = len(pd.read_csv("./input/test.csv", usecols=["ID_code"]))
    test_data = np.zeros((n_test, len(submissions)))
    column_names = []

    for i, submission in enumerate(submissions):
        column_names.append(submission.split('.csv')[0])
        test_data[:,i] = pd.read_csv("./submissions/{}".format(submission), 
                                usecols=["target"]).values[:,0]

    test_df = pd.DataFrame(data=test_data, columns=column_names)
    test_ranks = test_df.rank()

    if weighted_flag:
        n_val = len(pd.read_csv("./input/train.csv", usecols=["target"]))
        val_data = np.zeros((n_val, len(submissions)))
        column_names = []

        for i, submission in enumerate(submissions):
            column_names.append(submission.split('.csv')[0])
            val_data[:,i] = pd.read_csv(r"./output/oof predictions/{}".format(submission), 
                                    usecols=["target"]).rank().values[:,0]

        oof_labels = pd.read_csv("./input/train.csv", usecols=["target"]).rank().values[:,0]
        clf = Ridge(random_state=42)
        clf.fit(val_data, oof_labels)
        test_ranks["pred_rank"] = clf.predict(test_ranks.values)
    else:
        test_ranks["pred_rank"] = test_ranks.mean(axis=1)

    rank_preds = test_ranks["pred_rank"].values
    rank_preds = rank_preds.reshape(-1, 1)

    ss = MinMaxScaler()
    submission = pd.read_csv('./input/sample_submission.csv')
    submission['target'] = ss.fit_transform(rank_preds)
    submission.to_csv('./submissions/rank_preds_{}.csv'.format(oof_val), index=False)

if __name__ == "__main__":
    change_log = "Removed hardcoded submission input"
    start_time = str(datetime.datetime.now().time()).split('.')[0]

    begin_time = time.time()
    main()
    end_time = time.time()

    with open('./logs/'+str(datetime.date.today())+'.txt', 'a') as f:
        log = "File Name: {}\nChange Log: {}\nStart Time: {}\nExecution Time(seconds): {}\n\n\n".format(__file__, change_log, start_time, end_time-begin_time)
        f.write(log)