#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Figure preprocess

@author: liangyu
"""

import cv2
import os


def test():
    """验证方案的合理性
    1. 裁边，长宽不等，且width方向有黑边，剪裁为长宽相等的图片（为建模准备）
    2. resize. 考虑到原图片尺寸太大，网络模型参数太多，resize到 128 × 128 × 3
    """
    figure_path_1 = ''
    figure_1 = cv2.imread(figure_path_1)
    print figure_1.shape

    # cut
    figure_cut = figure_1[:, 83:575, :]

    # resize
    figure_cut_resize = cv2.resize(figure_cut, (128, 128))

    # 在Python开启窗口线程
    cv2.startWindowThread()

    # 先加载窗口，再加载图片
    cv2.namedWindow('init figure', cv2.WINDOW_NORMAL)
    cv2.imshow('init figure', figure_1)

    cv2.namedWindow('cut figure', cv2.WINDOW_NORMAL)
    cv2.imshow('cut figure', figure_cut)

    cv2.namedWindow('cut resize figure', cv2.WINDOW_NORMAL)
    cv2.imshow('cut resize figure', figure_cut_resize)

    # 等待键盘输入
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()


def preprocess_figure(init_figure_dir, init_figure_filename, new_figure_dir, figsize=(128, 128)):
    """
    1. 裁边. 长宽不等，且width方向有黑边，剪裁为长宽相等的图片
    2. resize. 考虑到原图片尺寸太大，网络模型参数太多，resize到 128 × 128 × 3
    
    Parameters:
    -----------------
    init_figure_dir : str, init figure dir
    init_figure_filename : str, init figure filename
    new_figure_dir : str, new figure dir
    figsize : resize figsize
    
    Return:
    ---------
    """
    figure = cv2.imread(init_figure_dir +init_figure_filename)
    figure_cut = figure[:, 83:575, :]                                           # cut   
    figure_cut_resize = cv2.resize(figure_cut, figsize)            # resize
    new_figure_path = new_figure_dir + init_figure_filename
    cv2.imwrite(new_figure_path, figure_cut_resize)
    
    print 'Done!' 
    

# Main
if __name__ == "__main__":
    project_dir = os.getcwd()
    init_figure_dir = os.path.join(project_dir, 'data', 'init_figure/')
    new_figure_dir = os.path.join(project_dir, 'data', 'preprocess_figure/')
        
    # init figure name list
    init_figure_name_list = []

    for init_figure_name in os.listdir(os.path.dirname(init_figure_dir)):
        init_figure_name_list.append(init_figure_name)
        
    print len(init_figure_name_list)
    
    count = 0
    for figure_name in init_figure_name_list:
           preprocess_figure(init_figure_dir, figure_name, new_figure_dir)
           count += 1
           print count
 