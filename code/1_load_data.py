import pandas as pd
import datetime
import my_utils
import numpy as np


def timestamp_2_date(x):
    return datetime.datetime.fromtimestamp(x).strftime('%Y%m%d%H%M%S')

def extract_datetime(df, col):
    # 20180918000001
    # YYYYMMDDHHMMSS
    df[col] = df[col].astype(np.int64)
    df['context_year'] = df[col].values//10000000000
    df['context_month'] = (df[col].values//100000000) % 100
    df['context_day'] = (df[col].values//1000000) % 100
    df['context_hour'] = (df[col].values//10000) % 100
    df['context_min'] = (df[col].values//100) % 100
    df['context_sec'] = df[col].values % 100
    return df


def pre_process_data(df):

    def split_str(x, s_str):
        return x.split(s_str)

    def get_arr_len(x):
        return len(x)

    def get_str_in_list(x, idx):
        if idx+1 > len(x):
            return '-1'
        return x[idx]

    # pre process category
    df['item_category_split'] = df['item_category_list'].apply(split_str, args=(';',))
    df['category_0'] = df['item_category_split'].apply(get_str_in_list, args=(0,))
    df['category_1'] = df['item_category_split'].apply(get_str_in_list, args=(1,))
    df['category_2'] = df['item_category_split'].apply(get_str_in_list, args=(2,))
    # print(df[['category_0', 'category_1', 'category_2']].head())

    # pre process property
    df['item_property_split'] = df['item_property_list'].apply(split_str, args=(';',))
    df['item_property_num'] = df['item_property_split'].apply(get_arr_len)
    # print(df[['item_property_split', 'item_property_num']].head())

    # pre process predict_category_property
    df['predict_category_property_split'] = df['predict_category_property'].apply(split_str, args=(';',))
    df['predict_category_property_num'] = df['predict_category_property_split'].apply(get_arr_len)
    # print(df[['predict_category_property_split', 'predict_category_property_num']].head())
    # 长度不定

    return df


df_train = pd.read_csv("../data/round2_train.txt", sep=' ')
df_test = pd.read_csv("../data/round2_ijcai_18_test_b_20180510.txt", sep=' ')
print(len(df_test))

df_concat = pd.concat([df_train, df_test], axis=0) #axis=0为在下方行合并

del df_train, df_test

df_concat.sort_values(by='context_timestamp', inplace=True) #直接在原DataFrame中排序
df_cols = df_concat.columns
df_concat = df_concat.reindex_axis(df_cols, axis=1) #axis=1为列合并
df_concat = df_concat.reset_index()
df_concat.rename(columns={'index': 'instanceID'}, inplace=True)

df_concat['context_timestamp_string'] = df_concat['context_timestamp'].apply(timestamp_2_date)
df_concat = extract_datetime(df_concat, 'context_timestamp_string')
df_concat = pre_process_data(df_concat)
df_concat.loc[df_concat['context_day'] == 31, 'context_day'] = 0

df_concat.loc[(df_concat['context_hour'] < 12), 'context_day'] = df_concat['context_day'] * 2
df_concat.loc[(df_concat['context_hour'] >= 12), 'context_day'] = df_concat['context_day'] * 2 + 1
df_concat.loc[(df_concat['context_day'] == 15 ), 'context_day'] = 16
df_concat.loc[((df_concat['context_day'] == 14) & (df_concat['context_hour'] >= 6)), 'context_day'] = 15


save_dir = '../data/df_concat.csv'
df_concat.to_csv(save_dir, index=False)

print('df_concat saved to %s' % save_dir)



