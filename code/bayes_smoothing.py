#!/usr/bin/python
# coding=utf-8

import numpy
import random
import scipy.special as special
import pandas as pd
import numpy as np

def round(x, d = 6):
    return np.round(x,d)

class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, ctr_list):

        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(ctr_list)
        # print 'mean and variance: ', mean, var
        # self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)

        self.alpha = (mean + 1e-10) * ((mean + 1e-10) * (1 + 1e-10 - mean) / (var + 1e-10) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1 + 1e-10 - mean) * ((mean + 1e-10) * (1+ 1e-10 - mean) / (var + 1e-10) - 1)

    def __compute_moment(self, ctr_list):
        '''moment estimation'''
        mean = np.sum(ctr_list)/len(ctr_list)
        if len(ctr_list) == 1:
            return mean, 0
        var = np.sum(np.power(ctr_list-mean,2))
        return mean, var/(len(ctr_list)-1)

def smooth(act, clk, alpha, beta):
    return round((act + alpha) / (clk+ alpha + beta), 6)

def hierarchy_smooth(df, key, var_list, clickDay, gap2):

    # calc root
    df_ = df[var_list + ['label']]
    df_['root'] = 0
    df_params = df_.copy()
    var_list = ['root'] + var_list
    df_params['root_clk_cnt'] = df_['label'].count()
    df_params['root_act_cnt'] = df_['label'].sum()

    df_params = df_params.drop('label', axis = 1)
    df_params.drop_duplicates(inplace=True)

    depth = len(var_list)

    for i in reversed(range(1, depth)):
        var_child = var_list[i]
        var_parent = var_list[i-1]
        df_merge = pd.DataFrame()
        path = '../train_count/hier_cnt/%d_%s_%d_%d.csv' % (clickDay, key, gap2, i)
        df_cnt = pd.read_csv(path)
        for parent in df_[var_parent].unique():
            df_tmp = df_.ix[df_[var_parent] == parent, var_list + ['label']]
            df_cnt_ = df_cnt[df_cnt[var_parent]==parent]
            df_tmp = df_tmp.merge(df_cnt_, on=var_list[:(i + 1)], how='left')
            df_tmp.fillna(0, inplace=True)
            hyper = HyperParam(1, 1)
            hyper.update_from_data_by_moment(
                df_tmp[ '%s_act_cnt' % (var_child)].values/df_tmp[ '%s_clk_cnt' % (var_child)].values)
            df_tmp['alpha_%s' % var_parent] = hyper.alpha
            df_tmp['beta_%s' % var_parent] = hyper.beta
            if df_merge.empty:
                df_merge = df_tmp
            else:
                df_merge = df_merge.append(df_tmp)
        df_merge = df_merge.drop(['label'], axis = 1)
        df_merge.drop_duplicates(inplace = True)
        df_params = df_params.merge(df_merge, on = var_list, how = 'left')

    for n in range(0, depth):
        var_target = var_list[n]
        if n == 0:
            clk = df_params['root_clk_cnt'].values
            act = df_params['root_act_cnt'].values
            cvr = act/(clk)
        for i in reversed(range(n)):
            var_parent = var_list[i]
            var_child = var_list[i + 1]
            alpha = df_params['alpha_%s' % var_parent]
            beta = df_params['beta_%s' % var_parent]
            clk = df_params['%s_clk_cnt' % (var_child)].values
            if i == (n - 1):
                act = df_params['%s_act_cnt' % (var_child)].values
            else:
                act = clk * cvr
            cvr = smooth(act, clk, alpha, beta)
        df_params['cvr_%s' % var_target] = cvr

    keep_cols = var_list + ['cvr_%s' %x for x in var_list[:]]
    df_params = df_params[keep_cols]

    df_params.drop_duplicates(inplace=True)
    return df_params

def hierarchy_smooth_ema(df, key, var_list, clickDay, gap2, gamma):

    # calc root
    df_ = df[var_list + ['label']]
    df_['root'] = 0
    df_params = df_.copy()
    var_list = ['root'] + var_list
    df_params['root_clk_cnt'] = df_['label'].count()
    df_params['root_act_cnt'] = df_['label'].sum()

    df_params = df_params.drop('label', axis = 1)
    df_params.drop_duplicates(inplace=True)

    depth = len(var_list)

    # calc alpha, beta from the bottom leaves to parents
    for i in reversed(range(1, depth)):
        var_child = var_list[i]
        var_parent = var_list[i-1]
        df_merge = pd.DataFrame()
        path = '../train_count/ema_cnt/%d_%d_%s_%d_%d_%s.csv' % (clickDay, clickDay, key, gap2, i, gamma)
        df_cnt = pd.read_csv(path)
        df_cnt = df_cnt.rename(columns={'%s_pre_clk_cnt' % (var_child): '%s_clk_cnt' % (var_child),
                                        '%s_pre_act_cnt' % (var_child): '%s_act_cnt' % (var_child)})
        for parent in df_[var_parent].unique():
            df_tmp = df_.ix[df_[var_parent] == parent, var_list + ['label']]
            df_cnt_ = df_cnt[df_cnt[var_parent]==parent]
            df_tmp = df_tmp.merge(df_cnt_, on=var_list[:(i + 1)], how='left')
            df_tmp.fillna(0, inplace=True)
            hyper = HyperParam(1, 1)
            hyper.update_from_data_by_moment(
                df_tmp['%s_act_cnt' % (var_child)].values/df_tmp['%s_clk_cnt' % (var_child)].values)
            df_tmp['alpha_%s' % var_parent] = hyper.alpha
            df_tmp['beta_%s' % var_parent] = hyper.beta
            if df_merge.empty:
                df_merge = df_tmp
            else:
                df_merge = df_merge.append(df_tmp)
        df_merge = df_merge.drop(['label'], axis = 1)
        df_merge.drop_duplicates(inplace = True)
        df_params = df_params.merge(df_merge, on = var_list, how = 'left')

    for n in range(0, depth):
        var_target = var_list[n]
        if n == 0:
            clk = df_params['root_clk_cnt'].values
            act = df_params['root_act_cnt'].values
            cvr = act/(clk)
        for i in reversed(range(n)):
            var_parent = var_list[i]
            var_child = var_list[i + 1]
            alpha = df_params['alpha_%s' % var_parent]
            beta = df_params['beta_%s' % var_parent]
            clk = df_params['%s_clk_cnt' % (var_child)].values
            if i == (n - 1):
                act = df_params['%s_act_cnt' % (var_child)].values
            else:
                act = clk * cvr
            cvr = smooth(act, clk, alpha, beta)
        df_params['cvr_%s' % var_target] = cvr

    keep_cols = var_list + ['cvr_%s' %x for x in var_list[:]]
    df_params = df_params[keep_cols]

    df_params.drop_duplicates(inplace=True)
    return df_params

def main():
    # df_main = pd.read_csv('dataSet/test.csv')
    # var_list = ['advertiserID', 'camgaignID', 'adID']
    # df_params = hierarchy_smooth(df_main, var_list)
    # df = df_main.merge(df_params, on = var_list, how = 'left')
    # print(df.iloc[:,-1].mean())
    # print(df['label'].mean())
    pass

if __name__ == '__main__':
    main()