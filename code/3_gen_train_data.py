import pandas as pd
import numpy as np
import bayes_smoothing as bs

print('LOADING')
df = pd.read_csv("../data/df_concat.csv")
df_merge = pd.DataFrame()

for context_day in range(0, 17):
    df_ = df.ix[df['context_day'] == context_day]
    var_list = ['item_id',
                'item_brand_id',
                'shop_id',
                'context_page_id',
                'item_city_id',
                'category_2'
                ]
    for var in var_list:
        df_interest = pd.read_csv('../intermediates/usr_feature_%s_%d.csv' % (var, context_day))
        df_ = df_.merge(df_interest, on=['instanceID'], how='left')
        del df_interest
    # df_['pre_usr_cvr'] = bs.smooth(df_['pre_usr_act_cnt'].values,
    #                               df_['pre_usr_clk_cnt'].values,
    #                               0.0091,
    #                               0.3288
    #                               )
    print(context_day, 'DONE')
    if len(df_merge) == 0:
        df_merge = df_
    else:
        df_merge = pd.concat([df_merge, df_], axis=0)

# dummies_gender = pd.get_dummies(df_merge['user_gender_id'], prefix= 'user_gender_id')
# df_merge = pd.concat([df_merge,dummies_gender], axis=1)
# df_merge.drop(['user_gender_id'], axis=1, inplace=True)
print(df_merge.head())
print(df_merge.columns)
df_merge.to_csv('../data/df_merge.csv', index=False)
print(df_merge.columns)
print(df_merge.dtypes)
print('COMPLETE')