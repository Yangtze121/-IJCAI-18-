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


def get_data_and_label(df, feature_cols, label_col):
    x = df[feature_cols].values
    y = df[label_col].values
    return x, y


data = pd.read_csv('../data/df_merge.csv')
one_hot_feature = feature_cols = [
    'context_page_id',
    'item_brand_id',
    'item_city_id',
    'shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate',
    'item_collected_level',
    'item_id',
    'item_price_level',
    'item_pv_level',
    'item_sales_level',
    'shop_id',
    'shop_review_num_level',
    'shop_star_level',
    'user_age_level',
    #'user_id',
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
]
del_feature = ['shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate', 'context_id']
label_col = 'is_trade'
for feature in one_hot_feature:
    if not (feature in del_feature):
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
enc = OneHotEncoder()
df_train = my_utils.select_range_by_day(data, 14, 15, 0, 0)
df_test = my_utils.select_range_by_day(data, 16, 16, 0, 0)
train_X, train_y = get_data_and_label(df_train, feature_cols, label_col)
test_X, test_y = get_data_and_label(df_test, feature_cols, label_col)
kf = KFold(n_splits=5, shuffle=False)
clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, solver='liblinear', max_iter=1000, verbose=20, n_jobs=32)
X_train_2j = pd.DataFrame()
X_test_2j = pd.DataFrame()
test_2j = np.zeros(len(test_X))
col_name = '3_predicted_score'
for i, (train_index, val_index) in enumerate(kf.split(df_train)):
    print('%d fold' % (i))
    X_train = train_X[train_index]
    y_train = train_y[train_index]
    X_val = train_X[val_index]
    y_val = train_y[val_index]
    model = clf
    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_val = pd.DataFrame(X_val, columns=feature_cols)
    train_x = X_train[
        ['shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate']]
    test_x = df_test[
        ['shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate']]
    val_x = X_val[
        ['shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate']]
    enc = OneHotEncoder()
    for feature in one_hot_feature:
        if not (feature in del_feature):
            enc.fit(data[feature].values.reshape(-1, 1))
            train_a = enc.transform(X_train[feature].values.reshape(-1, 1))
            val_a = enc.transform(X_val[feature].values.reshape(-1, 1))
            test_a = enc.transform(df_test[feature].values.reshape(-1, 1))
            train_x = sparse.hstack((train_x, train_a))
            val_x = sparse.hstack((val_x, val_a))
            test_x = sparse.hstack((test_x, test_a))
    model.fit(train_x, y_train)
    pred = model.predict_proba(test_x)[:, 1] / 5
    pred_val = model.predict_proba(val_x)[:, 1]
    test_2j += pred
    df_pred_val = pd.DataFrame(pred_val, columns=[col_name])
    X_train_2j = pd.concat([X_train_2j, df_pred_val], axis=0)
    print('ont-hot DONE!')
X_train_2j = X_train_2j.reset_index()
X_train_2j = X_train_2j.drop(['index'], axis=1)
X_train_2j = pd.DataFrame({'instance_id': df_train['instance_id'].values, col_name: X_train_2j[col_name].values})
test_2j = pd.DataFrame(test_2j, columns=[col_name])
test_2j = pd.DataFrame({'instance_id': df_test['instance_id'].values, col_name: test_2j[col_name].values})
train2_dir = '../data/stacking_lr_train_1.csv'
test2_dir = '../data/stacking_lr_test_1.csv'
X_train_2j.to_csv(train2_dir, index=False)
print('%s saved!'%(train2_dir))
test_2j.to_csv(test2_dir, index=False)
print('%s saved'%(test2_dir))
