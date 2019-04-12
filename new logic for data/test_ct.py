# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso,ElasticNet
# from sklearn.linear_model import AdaptiveLasso

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
target_url = ("D:/Personal/Desktop/Ct_data_1546700.xlsx")
fatigue = pd.read_excel(target_url,encoding="utf-8")
r = [fatigue.min(), fatigue.max(), fatigue.mean(), fatigue.std(),fatigue.quantile()]
r = pd.DataFrame(r, index=['Min', 'Max', 'Mean', 'STD', 'Quantile']).T
np.round(r,2) #保留两位小数
print(r)
corr = np.round(fatigue.corr(method='pearson'),2) #计算相关系数矩阵，保留两位小数
corr.to_csv("corr.csv")
# print(corr)

#利用AdaptiveLasso分析变量和特征系数
model = Lasso(alpha=0.1)
fatigueVariable = ['BR_Mx_m4',	'BR_My_m4', 'BR_Mz_m4','BR_Fx_m4','BR_Fy_m4','BR_Fz_m4','BR_Mx_m9','BR_My_m9','BR_Mz_m9','BR_Fx_m9','BR_Fy_m9','BR_Fz_m9','Hubfix_Mx_m9','Hubfix_My_m9',	'Hubfix_Mz_m9','Hubfix_Fx_m9','Hubfix_Fy_m9','Hubfix_Fz_m9','Hubrot_Mx_m9','Hubrot_My_m9','Hubrot_Mz_m9',	'Hubrot_Fx_m9','Hubrot_Fy_m9','Hubrot_Fz_m9','TT_Mx','TT_My','TT_Mz','TT_Fx', 'TT_Fy', 'TT_Fz','TB_Mx','TB_My','TB_Mz','TB_Fx','TB_Fy','TB_Fz']
feature = ['A', 'K', 'annual mean wind speed', 'V50', 'turbulent intensity', 'wind shear', 'inflow angle', 'power']
for v in fatigueVariable:
    # model.fit(fatigue.iloc[:,0:8],fatigue[v])
    # model.coef_
    # print("变量\n",v)
    # print(model.coef_)
# # feature = [0,1,2,3,4,5,6,7,8]
    data_train = fatigue.loc[range(1,41)].copy()
    # print(data_train)
    data_mean = data_train.mean()
    data_std = data_train.std()
    data_train = (data_train - data_mean)/data_std
    x_train = data_train[feature].as_matrix()
    y_train = data_train[v].as_matrix()

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    import time

    start = time.clock()

    # 输入层为３个节点，隐藏层６个节点
    model = Sequential()  # 建立模型
    model.add(Dense(output_dim=6, input_dim=8))  # 添加输入层、隐藏层节点
    # model.add(Dense(input_dim=8,12))
    model.add(Activation('relu'))  # 使用relu作为激活函数，可以大幅度提高准确率
    model.add(Dense(units=1, input_dim=8))  # 添加输出层节点
    model.compile(loss='mean_squared_error', optimizer='adam')  # 编译模型
    model.fit(x_train, y_train, nb_epoch = 3000, batch_size=16) #训练模型，学习一千次
    end = time.clock()
    usetime = end-start
    print ('训练该模型耗时'+ str(usetime) +'s!')
    aa = v + '.model'
    model.save_weights(aa) # 将该模型存储

    x = ((fatigue[feature] - data_mean[feature])/data_std[feature]).as_matrix()
    pre = v + 'pre'
    fatigue[pre] = model.predict(x) * data_std[v] + data_mean[v]
    #保存的表名命名格式为“2_2_3_1k此表功能名称”，是此小节生成的第1张表格，功能为revenue：增值税预测结果
    fatigue.to_excel('2_2_3_1zengzhi.xlsx')

    import matplotlib.pyplot as plt
    plt.rc('figure',figsize=(7,7))
    p = fatigue[[v,pre]].plot(subplots = True, style=['b-o', 'r-*'])
    plt.savefig('zengzhi.jpg')
    plt.show()




