# -*- coding: utf-8 -*-
'''
@ref: Grid-based evaluation metrics for Web image search
@author: anonymous author(s)
@desc: Some basic algorithms
'''

import math
import scipy.stats

maxm = 50
alpn = [0 for i in range(maxm)]
alp0 = -1.0
def calc_binary(alpha,gamma,n):
    global maxm,alpn,alp0
    if not (alpha == alp0):
        alp0 = alpha
        alpn[0] = 1.0
        for i in range(1,maxm):
            alpn[i] = alpn[i-1] * alpha
    ret = 1.0
    for i in range(len(n)):
        assert(1 <= n[i] and n[i] < maxm)
        if i < 0:
            ret = ret * alpn[n[i]]
        else:
            ret = ret * (gamma + (1-gamma) * alpn[n[i]])
    return ret

sigma_map = {}
def get_normal_dis(sigma, val):
    if sigma not in sigma_map:
        sigma_map[sigma] = {}
    if val not in sigma_map[sigma]:
        sigma_map[sigma][val] = scipy.stats.norm(0, sigma).pdf(val)
    return sigma_map[sigma][val]

def get_max(relevance_list, flag):
    if flag == 0:
        return max(relevance_list)
    else:
        max_set = []
        for i in range(len(relevance_list)):
            max_set.append(max(relevance_list[i]))
        return max(max_set)