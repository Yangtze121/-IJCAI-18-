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
label_col = 'is_trade'
del_feature = ['shop_score_delivery', 'shop_score_service', 'shop_score_description', 'shop_review_positive_rate', 'context_id']
df_train = my_utils.select_range_by_day(data, 14, 15, 0, 0)
df_test = my_utils.select_range_by_day(data, 16, 16, 0, 0)
X_train, y_train = get_data_and_label(df_train, feature_cols, label_col)
X_test, y_test = get_data_and_label(df_test, feature_cols, label_col)
kf = KFold(n_splits=5, shuffle=False)
clfs = [
    lgb.LGBMClassifier(learning_rate=0.05, num_leaves=25, n_estimators=3000, min_child_weight=5, nthread=6,n_jobs=32),
    RandomForestClassifier(max_depth=32, random_state=2018, n_estimators=100, min_samples_leaf=20, min_samples_split=10, n_jobs=32),
    xgb.XGBClassifier(learning_rate=0.05, min_child_weight=5, nthread=6, max_depth=18, n_estimators=3000, n_jobs=32),
]

X_train_2 = pd.DataFrame(df_train['instance_id'])
y_train2 = pd.DataFrame(df_train['is_trade'])
test_2 = pd.DataFrame(df_test['instance_id'])
for j in range(0, 3):
    X_train_2j = pd.DataFrame()
    X_test_2j = pd.DataFrame()
    test_2j = np.zeros(len(X_test))
    for i, (train_index, val_index) in enumerate(kf.split(X_train)):
        print('%d fold' % (i))
        col_name = '%d_predicted_score' % (j)
        train_X = X_train[train_index]
        train_y = y_train[train_index]
        val_X = X_train[val_index]
        val_y = y_train[val_index]
        model = clfs[j]
        if j == 0:
            model.fit(train_X, train_y,
                      eval_set=(val_X, val_y),
                      early_stopping_rounds=50,
                      verbose=20,
                      )
        elif j == 1:
            model.fit(train_X, train_y,

                      )
        elif j == 2:
            model.fit(train_X, train_y,
                      eval_set=((train_X, train_y), (val_X, val_y)),
                      early_stopping_rounds=50,
                      verbose=20,
                      eval_metric='logloss',
                      )
        pred = model.predict_proba(X_test)[:, 1] / 5
        pred_val = model.predict_proba(val_X)[:, 1]
        test_2j += pred
        df_pred_val = pd.DataFrame(pred_val, columns=[col_name])
        X_train_2j = pd.concat([X_train_2j, df_pred_val], axis=0)
    X_train_2j = X_train_2j.reset_index()
    X_train_2j = X_train_2j.drop(['index'], axis=1)
    X_train_2j = pd.DataFrame({'instance_id': df_train['instance_id'].values, col_name: X_train_2j[col_name].values})
    test_2j = pd.DataFrame(test_2j, columns=[col_name])
    test_2j = pd.DataFrame({'instance_id': df_test['instance_id'].values, col_name: test_2j[col_name].values})
    X_train_2 = X_train_2.merge(X_train_2j, on=['instance_id'], how='left')
    test_2 = test_2.merge(test_2j, on=['instance_id'], how='left')
train2_dir = '../data/stacking_train_1.csv'
trainy2_dir = '../data/stacking_trainy_1.csv'
test2_dir = '../data/stacking_test_1.csv'
X_train_2.to_csv(train2_dir, index=False)
print('%s saved!'%(train2_dir))
y_train2.to_csv(trainy2_dir, index=False)
print('%s saved!'%(trainy2_dir))
test_2.to_csv(test2_dir, index=False)
print('%s saved'%(test2_dir))

