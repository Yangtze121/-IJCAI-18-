# coding=utf-8
# @author:bryan
# blog: https://blog.csdn.net/bryan__
# github: https://github.com/YouChouNoBB/2018-tencent-ad-competition-baseline
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import my_utils
import scipy as sp
import numpy as np


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


data = pd.read_csv('../data/df_merge.csv')
df_test = pd.read_csv("../data/round2_ijcai_18_test_a_20180425.txt",
                      usecols=['instance_id'], sep=' ')
data=data.fillna('-1')
one_hot_feature=['context_page_id', 'item_brand_id', 'item_city_id', 'item_collected_level', 'item_id',
                 'item_price_level', 'item_pv_level', 'item_sales_level', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                 'user_age_level', 'user_gender_id', 'user_id', 'user_occupation_id', 'user_star_level', 'context_hour', 'context_min',
                 'context_sec', 'item_property_num', 'predict_category_property_num',
                 'pre_usr_clk_cnt', 'pre_usr_clk_same_item_id_cnt', 'his_usr_clk_cnt',
                 'his_usr_clk_same_item_id_cnt', 'pre_usr_act_cnt',
                 'pre_usr_act_same_item_id_cnt', 'his_usr_clk_same_item_id_fir_las',
                 'his_usr_clk_same_item_id_seq', 'item_id_clk_gap_bf',
                 'item_id_clk_gap_af', 'item_id_act_gap_bf', 'item_id_clk_cnt_bf',
                 'item_id_clk_cnt_af', 'item_id_clk_gap_2_fir', 'item_id_clk_cnt_bf_3h',
                 'item_id_clk_cnt_af_3h', 'clk_cnt_bf', 'clk_cnt_af', 'clk_cnt_bf_3h',
                 'clk_cnt_af_3h',
                 # 'gender_item_mean',
                 # 'gender_item_count',
                 # 'gender_category_1_mean',
                 # 'gender_category_1_count',
                 # 'age_item_mean',
                 # 'age_item_count',
                 # 'age_category_1_mean',
                 # 'age_category_1_count',
                 # 'occupation_item_count',
                 # 'occupation_item_mean',
                 # 'occupation_category_1_count',
                 # 'occupation_category_1_mean',
                 # 'occupation_hour_mean',
                 # 'occupation_hour_count',
                 ]
vector_feature=['item_category_list' , 'item_property_list', 'predict_category_property',
                'item_category_split', 'item_property_split', 'predict_category_property_split']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train = my_utils.select_range_by_day(data, 12, 14, 0, 3)
val = my_utils.select_range_by_day(data, 15, 15, 0, 0)
test = my_utils.select_range_by_day(data, 16, 16, 0, 0)
train_y = train.pop('is_trade')
val_y = val.pop('is_trade')
test_y = test.pop('is_trade')
# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
res = test[['instance_id']]
enc = OneHotEncoder()
train_x = train[['shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate']]
test_x = test[['shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate']]
val_x = val[['shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate']]

for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a = enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    val_a = enc.transform(val[feature].values.reshape(-1, 1))
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
    val_x = sparse.hstack((val_x, val_a))
print('one-hot prepared !')

cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    val_a = cv.transform(val[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
    val_x = sparse.hstack((val_x, val_a))
print('cv prepared !')


def LGB_predict(train_x, train_y, test_x, test_y, val_x, val_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=7, n_jobs=-1
    )
    clf.fit(train_x, train_y.astype(int), eval_set=[(val_x, val_y.astype(int))], eval_metric='logloss',early_stopping_rounds=50, verbose=5)
    pred = clf.predict_proba(test_x)[:, 1]
    print(logloss(test_y.astype(int), pred))
    ans = pd.DataFrame({"instance_id": test['instance_id'].values, "predicted_score": pred})
    ans.sort_values("instance_id", inplace=True)
    # ans = df_test.merge(res, on='instance_id', how='left')
    ans.to_csv('../submission/submission_new1.txt', index=False, sep=' ', line_terminator='\n')
    return clf


model = LGB_predict(train_x, train_y, test_x, test_y, val_x, val_y)