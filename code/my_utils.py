import numpy as np
import pandas as pd

root_path = '/Users/zhaobinw/PycharmProjects/tianchi_alimama_ad'

day_start = 20

mean_cvr = 0.02723450258


def calc_time_gap(t1, t2):
    return t1 - t2


def calc_time_gap_bak(t1, t2):
    t1_s = t1 % 100
    t2_s = t2 % 100
    t1_m = (t1 // 100) % 100
    t2_m = (t2 // 100) % 100
    t1_h = (t1 // 10000) % 100
    t2_h = (t2 // 10000) % 100
    t1_d = t1 // 1000000
    t2_d = t2 // 1000000
    gap = (t1_d - t2_d) * 24 * 3600 + (t1_h - t2_h) * 3600 + (t1_m - t2_m) * 60 + (t1_s - t2_s)
    return gap


def select_range(df, context_day, gap1, gap2):
    mask = (df['context_day'] <= (context_day - gap1)) & \
           (df['context_day'] >= (context_day - gap2))
    df_tmp = df.ix[mask, :]
    return df_tmp


def select_range_by_day(df, day_begin, day_end, day_begin1, day_end1):
    if (day_begin1 == 0) & (day_end1 == 0):
        mask = (df['context_day'] <= day_end) & (df['context_day'] >= day_begin)
        df_tmp = df.ix[mask]
    else:
        mask = (df['context_day'] <= day_end) & (df['context_day'] >= day_begin)
        df_tmp = df.ix[mask]
        mask = (df['context_day'] <= day_end1) & (df['context_day'] >= day_begin1)
        df_tmp1 = df.ix[mask]
        df_tmp = pd.concat([df_tmp, df_tmp1], axis=0)
    return df_tmp


def fix_is_trade(df, context_day):
    mask = df['context_day'] >= context_day #????
    df.ix[mask, ['is_trade']] = "0"
    return df


def col2str(df):
    for col in df.columns:
        if df[col].dtype != str:
            df.ix[:, col] = df[col].values.astype(str)
    return df

cols_basic = ['instance_id', 'is_trade', 'item_id', 'user_id', 'context_id', 'shop_id']
cols_ad_product = ['item_id', 'item_category_list', 'item_property_list', 'item_brand_id',
                       'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level',
                       'item_pv_level']
cols_user = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
cols_context = ['context_id', 'context_timestamp', 'context_page_id', 'predict_category_property']
cols_shop = ['shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
                 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
cols_extra = ['context_hour', 'context_day',
              'category_0', 'category_1', 'category_2',
              'item_property_num', 'predict_category_property_num']

all_cols = cols_basic + cols_ad_product + cols_user + cols_context + cols_shop + cols_extra
not_use_cols = ['instance_id', 'user_id', 'item_category_list', 'item_property_list',
                'context_timestamp', 'predict_category_property']

use_cols = list(set(all_cols) - set(not_use_cols))

not_feature_cols = ['is_trade', 'context_day']

feature_cols = list(set(use_cols) - set(not_feature_cols))