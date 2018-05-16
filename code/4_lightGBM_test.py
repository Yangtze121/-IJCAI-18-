import lightgbm as lgb
import zipfile
import sys
import time
import datetime
import os
import pandas as pd
import scipy as sp
import numpy as np
from sklearn.model_selection import train_test_split
import my_utils

df_test = pd.read_csv("../data/round2_ijcai_18_test_b_20180510.txt",
                      usecols=['instance_id'], sep=' ')
print(df_test.info())

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def logloss_baseline(y_train, y_val):
    print('\n##### baseline #####')
    mean_cvr_train = y_train.mean()
    pred_by_train = len(y_val) * [mean_cvr_train]
    print('mean_cvr_train:', mean_cvr_train)
    print('pred by mean_cvr_train:', logloss(act=y_val, pred=pred_by_train))
    mean_cvr_val = y_val.mean()
    pred_by_val = len(y_val) * [mean_cvr_val]
    print('mean_cvr_val:', mean_cvr_val)
    print('pred by mean_cvr_val:', logloss(act=y_val, pred=pred_by_val))
    print('######################\n')


def get_data_and_label(df, feature_cols, label_col):
    x = df[feature_cols].values
    y = df[label_col].values
    return x, y


lgb_clf = lgb.LGBMClassifier(
            # max_depth=param1,
            learning_rate=0.03,
            num_leaves=25,
            n_estimators=3000,
            min_child_weight=5,
            nthread=6,
            n_jobs=32,
            )

df = pd.read_csv("../data/df_merge.csv")
print(df.columns)
df.fillna(0, inplace=True)

feature_cols = my_utils.feature_cols
# feature_cols = [
#                 # 'instanceID',
#                 # 'context_id',
#                  'user_id',
#                 #  'context_timestamp',
#                   'shop_score_service',
#                  'item_id',
#                   'shop_score_delivery',
#                    'shop_score_description',
#                   'shop_review_positive_rate',
#                 'shop_id',
#                 'item_id_clk_gap_bf',
#                 'item_brand_id',
#                 'item_id_clk_gap_2_fir',
#                 'item_id_clk_gap_af',
#
#                 'category_1',
#                 'clk_cnt_af_3h',
#                 'item_price_level',
#                 'item_city_id',
#                 'item_id_clk_cnt_bf_3h',
#                 'his_usr_clk_cnt',
#                 'item_sales_level',
#                 'clk_cnt_af',
#                 'his_usr_clk_same_item_id_seq',
#                 'user_gender_id',
#
#                 'clk_cnt_bf',
#                 'clk_cnt_bf_3h',
#                 'item_id_clk_cnt_bf',
#                 'item_id_clk_cnt_af_3h',
#                 'item_id_clk_cnt_af',
#                 'item_property_num',
#                 'his_usr_clk_same_item_id_fir_las',
#                 'item_id_act_gap_bf',
#                 'predict_category_property_num', #binary_logloss:0.0807734
#                 'shop_star_level',
#                 'shop_review_num_level',
#                 #'context_hour',
#                 'user_age_level',
#                 'item_collected_level',
#                 'user_star_level',
#                 #'context_sec',
#                 'his_usr_clk_same_item_id_cnt',
#                 #'context_min',
#                 'item_pv_level',                #binary_logloss:0.080126
#                 'context_page_id',
#                 'category_2',                   #binary_logloss:0.0800587
#                 'user_occupation_id',
#                 'pre_usr_clk_cnt',
#                 'pre_usr_clk_same_item_id_cnt', #binary_logloss:0.0800121
#                 # 'context_day',
#                 'pre_usr_act_cnt',                 #binary_logloss:0.0800121
#                 'pre_usr_act_same_item_id_cnt',
#                 'gender_item_mean',
#                 'gender_item_count',
#                 'gender_category_1_mean',
#                 'gender_category_1_count',
#                 'age_item_mean',
#                 'age_item_count',
#                  'age_category_1_mean',
#                  'age_category_1_count',
#                  'occupation_item_count',
#                 'occupation_item_mean',
#                 'occupation_category_1_count',
#                 'occupation_category_1_mean',
#                 #  'occupation_hour_mean',
#                 # 'occupation_hour_count',
#
#
#                 #'user_gender_id_-1', 'user_gender_id_0', 'user_gender_id_1', 'user_gender_id_2'
#                 ]
feature_cols = [
        # 'instanceID',
    'context_id',
    'context_page_id',
    # 'context_timestamp',
    #    'instance_id',
    #'is_trade',
    'item_brand_id',
    # 'item_category_list',
    'item_city_id',
    'item_collected_level',
    'item_id',
    'item_price_level',
       # 'item_property_list',
    'item_pv_level',
    'item_sales_level',
       # 'predict_category_property',
    'shop_id',
    'shop_review_num_level',
    'shop_review_positive_rate',
    'shop_score_delivery',
    'shop_score_description', 'shop_score_service', 'shop_star_level',
    'user_age_level',
    'user_id',
    'user_occupation_id',
    'user_star_level',
    'user_gender_id',
    'context_timestamp_string',
    'category_0',
    'category_1',
    'category_2',
    'item_property_num',
    'predict_category_property_num',
    'pre_usr_clk_same_context_page_id_cnt',
    'his_usr_clk_same_context_page_id_cnt',
    'pre_usr_act_same_context_page_id_cnt',
    'his_usr_clk_same_context_page_id_fir_las',
    'his_usr_clk_same_context_page_id_seq',
    'his_usr_clk_same_context_page_id_arr',
    'context_page_id_clk_gap_bf',
    'context_page_id_clk_gap_af',
    'context_page_id_act_gap_bf',
    'context_page_id_clk_cnt_bf',
    'context_page_id_clk_cnt_af',
    'context_page_id_clk_gap_2_fir',
    'context_page_id_clk_cnt_bf_3h',
    'context_page_id_clk_cnt_af_3h',
    'pre_usr_clk_same_item_brand_id_cnt',
    'his_usr_clk_same_item_brand_id_cnt',
    'pre_usr_act_same_item_brand_id_cnt',
    'his_usr_clk_same_item_brand_id_fir_las',
    'his_usr_clk_same_item_brand_id_seq',
    'his_usr_clk_same_item_brand_id_arr',
    'item_brand_id_clk_gap_bf',
    'item_brand_id_clk_gap_af',
    'item_brand_id_act_gap_bf',
    'item_brand_id_clk_cnt_bf',
    'item_brand_id_clk_cnt_af',
    'item_brand_id_clk_gap_2_fir',
    'item_brand_id_clk_cnt_bf_3h',
    'item_brand_id_clk_cnt_af_3h',
    'pre_usr_clk_cnt',
    'pre_usr_clk_same_item_id_cnt',
    'his_usr_clk_cnt',
    'his_usr_clk_same_item_id_cnt',
    'pre_usr_act_cnt',
    'pre_usr_act_same_item_id_cnt',
    'his_usr_clk_same_item_id_fir_las',
    'his_usr_clk_same_item_id_seq',
    'his_usr_clk_same_item_id_arr',
    'item_id_clk_gap_bf',
    'item_id_clk_gap_af',
    'item_id_act_gap_bf',
    'item_id_clk_cnt_bf',
    'item_id_clk_cnt_af',
    'item_id_clk_gap_2_fir',
    'item_id_clk_cnt_bf_3h',
    'item_id_clk_cnt_af_3h',
    'clk_cnt_bf',
    'clk_cnt_af',
    'clk_cnt_bf_3h',
    'clk_cnt_af_3h',
    'pre_usr_clk_same_shop_id_cnt',
    'his_usr_clk_same_shop_id_cnt',
    'pre_usr_act_same_shop_id_cnt',
    'his_usr_clk_same_shop_id_fir_las',
    'his_usr_clk_same_shop_id_seq',
    'his_usr_clk_same_shop_id_arr',
    'shop_id_clk_gap_bf',
    'shop_id_clk_gap_af',
    'shop_id_act_gap_bf',
    'shop_id_clk_cnt_bf',
    'shop_id_clk_cnt_af',
    'shop_id_clk_gap_2_fir',
    'shop_id_clk_cnt_bf_3h',
    'shop_id_clk_cnt_af_3h',
    'pre_usr_clk_same_item_city_id_cnt',
    'his_usr_clk_same_item_city_id_cnt',
    'pre_usr_act_same_item_city_id_cnt',
    'his_usr_clk_same_item_city_id_fir_las',
    'his_usr_clk_same_item_city_id_seq',
    'his_usr_clk_same_item_city_id_arr',
    'item_city_id_clk_gap_bf',
    'item_city_id_clk_gap_af',
    'item_city_id_act_gap_bf',
    'item_city_id_clk_cnt_bf',
    'item_city_id_clk_cnt_af',
    'item_city_id_clk_gap_2_fir',
    'item_city_id_clk_cnt_bf_3h',
    'item_city_id_clk_cnt_af_3h',
    'pre_usr_clk_same_category_2_cnt',
    'his_usr_clk_same_category_2_cnt',
    'pre_usr_act_same_category_2_cnt',
    'his_usr_clk_same_category_2_fir_las',
    'his_usr_clk_same_category_2_seq',
    'his_usr_clk_same_category_2_arr',
    'category_2_clk_gap_bf',
    'category_2_clk_gap_af',
    'category_2_act_gap_bf',
    'category_2_clk_cnt_bf',
    'category_2_clk_cnt_af',
    'category_2_clk_gap_2_fir',
    'category_2_clk_cnt_bf_3h',
    'category_2_clk_cnt_af_3h',
]

label_col = 'is_trade'

df_train = my_utils.select_range_by_day(df, 12, 14, 8, 11)
df_val = my_utils.select_range_by_day(df, 15, 15, 0, 0)
df_test = my_utils.select_range_by_day(df, 16, 16, 0, 0)
print(df_test.head())
# train_x, train_y = get_data_and_label(df_train, feature_cols, label_col)
X_train, y_train = get_data_and_label(df_train, feature_cols, label_col)
X_val, y_val = get_data_and_label(df_val, feature_cols, label_col)
# X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.3, random_state=2022)
X_test, y_test = get_data_and_label(df_test, feature_cols, label_col)

logloss_baseline(y_train, y_val)

model = lgb_clf
model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          early_stopping_rounds=50,
          verbose=5,
        )

pred = model.predict_proba(X_test)[:,1]
pred_loss = logloss(y_test, pred)
print(pred_loss)

df_res = pd.DataFrame({"instance_id": df_test["instance_id"].values, "predicted_score": pred})
df_res.sort_values("instance_id", inplace=True)

df_test = pd.read_csv("../data/round2_ijcai_18_test_b_20180510.txt",
                      usecols=['instance_id'], sep=' ')

df_test = df_test.merge(df_res, on='instance_id', how='left')
#print(df_test.info())
df_test.to_csv("../submission/submission6.txt", index=False, sep=' ', line_terminator='\n')



'''
原始数据 valid_0's binary_logloss: 0.0824322
加入用户维度数据 valid_0's binary_logloss: 0.0801978
'''

# y_pred = np.round(y_pred, 12)
# y_pred = np.round(y_pred * my_utils.mean_cvr / np.mean(y_pred), 12)
# df = pd.DataFrame({"instanceID": df_instanceID["instanceID"].values, "proba": y_pred})