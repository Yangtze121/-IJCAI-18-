# IJCAI-18 阿里妈妈搜索广告转化预测复赛第94名方案
感谢wzb同学的倾情相助！<br />
-------
赛题以阿里巴巴海量真实交易数据为背景，通过构建预测模型预估用户的购买意向<br />
方案以数据清洗、特征提取、模型训练、模型融合的步骤进行，最终复赛排名94/5204<br />
1_load_data:数据读入，简单的预处理<br />
2_feature_extract:特征提取，包括一些统计特征和组合特征，将结果写入中间文件<br />
3_gen_train_data:将中间文件合并成最终的训练测试数据<br />
4_lightGBM_test:构建LGB单模型预测<br />
5_stacking_lr:比赛后期进行模型stacking时使用的lr模型，由于lr模型要对特征进行ont-hot处理，故单独列出<br />
6_stacking_modle:使用LGB,XGB和RF模型的stacking<br />
7_stacking_2nd:第二层stacking<br />
8_logloss_reverse:均值平滑<br />
bayes_smoothing:贝叶斯平滑<br />
my_utils:常用函数的封装<br />
vector_methond:对多值特征的向量化处理（线上成绩一般，最后未使用）<br />
