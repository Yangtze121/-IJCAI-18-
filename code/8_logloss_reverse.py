import numpy as np
import scipy as sp
import pandas as pd



def logloss(act, pred):
    epsilon = 1e-8
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


cur_logloss = 0.65870
num = 10000000

ll1 = logloss([1], [0])
ll2 = logloss([0], [0])

tp = (cur_logloss * num - ll2 * num) / (ll1 - ll2)
mean_cvr = tp/num
print(mean_cvr)

submission1 = pd.read_csv('../submission/submission4.txt', sep=' ')
submission2 = pd.read_csv('../submission/submission5.txt', sep=' ')
submission3 = pd.read_csv('../submission/submission6.txt', sep=' ')
# submission4 = pd.read_csv('../submission/submission4.txt', sep=' ')
# submission5 = pd.read_csv('../submission/submission5.txt', sep=' ')
# print('1.mean:', submission1['predicted_score'].mean())
# print('2.mean:', submission2['predicted_score'].mean())
# print('3.mean:', submission3['predicted_score'].mean())
# print('4.mean:', submission4['predicted_score'].mean())
# print('5.mean:', submission5['predicted_score'].mean())
submission = submission1
# submission['predicted_score'] = (submission1['predicted_score'] + submission2['predicted_score'] + submission3['predicted_score'] + submission4['predicted_score'] + submission5['predicted_score']) / 5
submission['predicted_score'] = (submission1['predicted_score'] + submission2['predicted_score'] + submission3['predicted_score'] ) / 3
sub_mean_cvr = submission['predicted_score'].mean()
print(sub_mean_cvr)
submission['predicted_score'] = submission['predicted_score'] * mean_cvr / sub_mean_cvr
# print('after:', logloss(act, submission['predicted_score'].values))
print(submission['predicted_score'].mean())
df_ = submission.ix[submission['predicted_score'] > 0.2, :]
print('predicted_score > 0.2 :', len(df_))
submission.to_csv("../submission/submission_fix.txt", index=False, sep=' ', line_terminator='\n')