from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import my_utils
import lightgbm as lgb
from sklearn.linear_model import  LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import scipy as sp
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

stacking_lr_train = pd.read_csv('../data/stacking_lr_train_1.csv')
stackin_lr_test = pd.read_csv('../data/stacking_lr_test_1.csv')
stacking_train = pd.read_csv('../data/stacking_train_1.csv')
stacking_test = pd.read_csv('../data/stacking_test_1.csv')
stacking_trainy = pd.read_csv('../data/stacking_trainy_1.csv')
df_test = pd.read_csv("../data/round2_ijcai_18_test_b_20180510.txt",
                      usecols=['instance_id'], sep=' ')

stacking_train = stacking_train.merge(stacking_lr_train, on=['instance_id'], how='left')
stacking_test = stacking_test.merge(stackin_lr_test, on=['instance_id'], how='left')
print('merge DONE!')
train_X, val_X, train_y, val_y = train_test_split(stacking_train, stacking_trainy, test_size=0.3, random_state=7)
clf = lgb.LGBMClassifier(learning_rate=0.01, num_leaves=10, n_estimators=3000, min_child_weight=5, nthread=6,n_jobs=32)
clf.fit(train_X, train_y,
        eval_set=(val_X, val_y),
        early_stopping_rounds=50,
        verbose=20,
)
pred = clf.predict_proba(stacking_test)[:, 1]
df_res = pd.DataFrame({"instance_id": df_test["instance_id"].values, "predicted_score": pred})
df_res.sort_values("instance_id", inplace=True)
df_test = pd.read_csv("../data/round2_ijcai_18_test_b_20180510.txt",
                      usecols=['instance_id'], sep=' ')
df_test = df_test.merge(df_res, on='instance_id', how='left')
print(df_test['predicted_score'].mean())
df_test.to_csv("../submission/submission_stacking.txt", index=False, sep=' ', line_terminator='\n')