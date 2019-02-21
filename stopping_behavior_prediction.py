# -*- coding: utf-8 -*-
'''
@ref: Grid-based evaluation metrics for Web image search
@author: anonymous author(s)
@desc: Formulas of different evaluation metrics for behavior prediction
'''
import math
from utils import *
import scipy.stats

'''
[Input] 
N - a vector records the number of images in each row; 
(sr, sc) - row and column number of the stopping position; 
**kawargs: parameter(s);
[Output]
log-likelihood;
'''

#list-based RBP model
def RBP(N,sr,sc,**kwargs):
    ll = 0.0
    num = 0
    alpha = kwargs["alpha"]
    stop_flag = False
    for i in range(sr + 1):
        for j in range(len(N[i])):
            if i == sr and j == sc:
                ll += math.log(1-alpha)
                stop_flag = True
            else:
                ll += math.log(alpha)
            num += 1
            if stop_flag:
                break
    return ll * 1.0 / num

#RBP with slower decay assumption (RBP-SD)
def RBP_SD(N,sr,sc,**kwargs):
    ll = 0.0
    num = 0
    alpha, beta = kwargs["alpha"], kwargs["beta"]
    stop_flag = False
    cum_beta = 1.0
    for i in range(sr + 1):
        for j in range(len(N[i])):
            if i == sr and j == sc:
                ll += math.log(1-alpha)
                ll += math.log(cum_beta)
                if ll > 0:
                    ll = 0
                stop_flag = True
            else:
                ll += math.log(alpha)
            num += 1
            if stop_flag:
                break
        cum_beta *= beta
    return ll * 1.0 / num

#RBP with row skipping assumption (RBP-RS)
def RBP_RS(N,sr,sc,**kwargs):
    ll = 0.0
    num = 0
    alpha, gamma = kwargs["alpha"], kwargs["gamma"]
    num_each_row = []
    for i in range(sr):
        num_each_row.append(len(N[i]))
        num += len(N[i])
    ll = calc_binary(alpha, gamma, num_each_row)
    ll = max(ll, 0.00001)
    ll = math.log(ll)
    ll += math.log(1 - gamma)
    for i in range(sc):
        ll += math.log(alpha)
    ll += math.log(1-alpha)
    num += sc+1
    return ll * 1.0 / num

#RBP with middle bias assumption (RBP-MB)
def RBP_MB(N,sr,sc,**kwargs):
    ll = 0.0
    num = 0
    ori_alpha, sigma = kwargs["alpha"], kwargs["sigma"]
    stop_flag = False
    for i in range(sr + 1):
        for j in range(len(N[i])):
            middle_position = (len(N[i]) - 1) * 1.0 / 2
            alpha = ori_alpha
            if i == sr and j == sc:
                normal_f = math.e ** get_normal_dis(sigma, j - middle_position)
                ll += math.log(1 - alpha)
                ll += math.log(normal_f)
                if ll > 0:
                    ll = 0
                stop_flag = True
            else:
                ll += math.log(alpha)
            num += 1
            if stop_flag:
                break
    return ll * 1.0 / num

if __name__ == "__main__":
    pass