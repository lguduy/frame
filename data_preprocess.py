#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017/6/27 22:08

数据：excel表格，火焰图像对应的传感器测量的过程变量

@author: liangyu
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


project_dir = os.getcwd()
data_path = os.path.join(project_dir, 'data', 'Data2_Full.xls')
init_data = pd.read_excel(data_path, header=0, index_col=0)

# 过程变量：取前面有用的数据
use_data = init_data[0:10492]

# 过程变量：remove 0
data_0 = use_data[use_data['O2(%)'] == 0]
time_data_0 = data_0[['Hour', 'Minute', 'Second']].values
use_data_remove = use_data[use_data['O2(%)'] > 0]

# 图像：删除过程变量中 O2(%) 为0时对应的图像
m = time_data_0.shape[0]
for i in xrange(m):
    figure_name = '20090713{:0>2}{:0>2}{:0>2}.jpg'.format(time_data_0[i, 0],
                                time_data_0[i, 1],
                                time_data_0[i, 2])
    figure_dir_path = './Data/preprocess_figure/'
    figure_path = figure_dir_path + figure_name
    if os.path.exists(figure_path):
        print "Exist"
        os.remove(figure_path)
        print "Ok"
    else:
        print "Not exist"

# 利用过程变量的时间和图片文件名做匹配
use_data_remove = use_data_remove[[u'Hour', u'Minute', u'Second', u'O2(%)']]

# process path list
n = use_data_remove.shape[0]
process_path_list = []

for i in xrange(n):
    figure_name = '20090713{:0>2}{:0>2}{:0>2}.jpg'.format(int(use_data_remove.iloc[i, 0]),
                                int(use_data_remove.iloc[i, 1]),
                                int(use_data_remove.iloc[i, 2]))
    process_path_list.append(figure_name)

# figure path list
figure_path_list = []

for figure_path in os.listdir(os.path.dirname(figure_dir_path)):
        figure_path_list.append(figure_path)

# 匹配
use_data_remove['is_exist'] = 1

count = 0
for index, process_path in enumerate(process_path_list):
    if process_path not in figure_path_list:
        use_data_remove.iloc[index, 4] = 0
        count += 1
        print index, count

if count == (len(process_path_list) - len(figure_path_list)):
    print "Okey!"

# 删除对应图片不存在的过程变量
use_data_remove_match = use_data_remove[use_data_remove['is_exist'] == 1]
use_data_remove_match.drop([4731], inplace=True)

# 将燃烧状态分类
# 画图观察分几类，并确定分类界限
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
axes.plot(use_data_remove_match.values[:, 3], 'b.', markersize=1)
# xlim
axes.set_xlim(0, 9500)
# xticks
axes.set_xticks(xrange(0, 10000, 500))
# xtickl
# ylim
axes.set_ylim(0, 8)
# save
plt.savefig('./figure_', dpi = 1000)

# 
use_data_remove_match['label'] = -1

for i in range(use_data_remove_match.shape[0]):
    O2 = use_data_remove_match.iloc[i, 3]
    if O2 >= 7.1:
        use_data_remove_match.iloc[i, 5] = 0
    elif O2 >= 6.7 and O2 <= 6.9:
        use_data_remove_match.iloc[i, 5] = 1
    elif O2 >= 6.2 and O2 <= 6.4:
        use_data_remove_match.iloc[i, 5] = 2
    elif O2 >= 5.7 and O2 <= 5.9:
        use_data_remove_match.iloc[i, 5] = 3
    elif O2 >= 5.0 and O2 <= 5.5:
        use_data_remove_match.iloc[i, 5] = 4
    elif O2 >= 4.5 and O2 <= 4.7:
        use_data_remove_match.iloc[i, 5] = 5
    elif O2 >= 3.9 and O2 <= 4.2:
        use_data_remove_match.iloc[i, 5] = 6
    elif O2 >= 3.4 and O2 <= 3.6:
        use_data_remove_match.iloc[i, 5] = 7
    elif O2 >= 2.6 and O2 <= 2.8:
        use_data_remove_match.iloc[i, 5] = 8
    elif O2 >= 1.9 and O2 <= 2.1:
        use_data_remove_match.iloc[i, 5] = 9
    elif O2 <= 1.3:
        use_data_remove_match.iloc[i, 5] = 10

use_data_remove_match_label = use_data_remove_match[use_data_remove_match['label'] >= 0]
