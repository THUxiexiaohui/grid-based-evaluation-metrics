# -*- coding: utf-8 -*-
'''
@ref: Grid-based evaluation metrics for Web image search
@author: anonymous author(s)
@desc: Formulas of different evaluation metrics
'''

import math
from utils import *
import scipy.stats

'''
[Input] 
relevance_list for list-based evaluation metrics;
relevance_matrix for grid-based evaluation metrics;
[Output]
metric-specific score;
'''

def RBP(relevance_list, **kwargs):
    alpha = kwargs["alpha"]
    metric = 0
    relevance_sum_so_far = 0
    for i in range(len(relevance_list)):
        relevance_sum_so_far += relevance_list[i]
        metric += (1 - alpha) * math.pow(alpha, i) * relevance_sum_so_far
    return metric

def RBP_SD(relevance_list, **kwargs):
    alpha, beta = kwargs["alpha"], kwargs["beta"]
    metric = 0
    cum_p = 1.0
    relevance_sum_so_far = 0
    rank = 0
    cum_beta = 1.0
    for i in range(len(relevance_list)):
        for j in range(len(relevance_list[i])):
            relevance_sum_so_far += relevance_list[i][j]
            stopping_probability = min(cum_beta * (1-alpha) * cum_p, 0.9999)
            st += str(stopping_probability) + "\t"
            metric += stopping_probability * relevance_sum_so_far
            cum_p *= alpha
            rank += 1
        cum_beta *= beta
    return metric

def RBP_RS(relevance_list, **kwargs):
    alpha, gamma = kwargs["alpha"], kwargs["gamma"]
    metric = 0
    continue_cnt = 1.0
    relevance_sum_so_far = 0
    for i in range(len(relevance_list)):
        for j in range(len(relevance_list[i])):
            if i < 1:
                relevance_sum_so_far += relevance_list[i][j]
                metric += continue_cnt * math.pow(alpha, j) * (1 - alpha) * relevance_sum_so_far
            else:
                relevance_sum_so_far += (1-gamma) * relevance_list[i][j]
                metric += continue_cnt * (1-gamma) * math.pow(alpha, j) * (1 - alpha) * relevance_sum_so_far
        if i < 1:
            continue_cnt *= math.pow(alpha, len(relevance_list[i]))
        else:
            continue_cnt *= (gamma + (1-gamma) * math.pow(alpha, len(relevance_list[i])))
    return metric

def RBP_MB(relevance_list, **kwargs):
    ori_alpha, sigma = kwargs["alpha"], kwargs["sigma"]
    metric = 0
    continue_cnt = 1.0
    relevance_sum_so_far = 0
    rank = 0
    for i in range(len(relevance_list)):
        alpha_multi = 1.0
        for j in range(len(relevance_list[i])):
            relevance_sum_so_far += relevance_list[i][j]
            middle_position = (len(relevance_list[i]) - 1) * 1.0 / 2
            normal_f = math.e ** get_normal_dis(sigma,j - middle_position)
            alpha = ori_alpha
            stopping_probability = normal_f * continue_cnt * alpha_multi * (1 - alpha)
            stopping_probability = min(0.9999, stopping_probability)
            metric += stopping_probability * relevance_sum_so_far
            rank += 1
            alpha_multi *= alpha
        continue_cnt *= alpha_multi
    return metric

def DCG(relevance_list, **kwargs):
    metric = 0
    relevance_sum_so_far = 0
    cum_c = 1.0
    for i in range(len(relevance_list)):
        rank = i
        relevance_sum_so_far += relevance_list[i]
        alpha = math.log(rank +2,2)/math.log(rank +3,2)
        metric += (1 - alpha) * cum_c * relevance_sum_so_far
        cum_c *= alpha
    return metric

def DCG_SD(relevance_list, **kwargs):
    beta = kwargs["beta"]
    metric = 0
    cum_c = 1.0
    cum_beta = 1.0
    relevance_sum_so_far = 0
    rank = 0
    for i in range(len(relevance_list)):
        for j in range(len(relevance_list[i])):
            relevance_sum_so_far += relevance_list[i][j]
            alpha = math.log(rank +2,2)/math.log(rank +3,2)
            stopping_probability = min(cum_beta * (1 - alpha) * cum_c, 0.9999)
            metric += stopping_probability * relevance_sum_so_far
            cum_c *= alpha
            rank += 1
        cum_beta *= beta
    return metric

def DCG_RS(relevance_list, **kwargs):
    gamma = kwargs["gamma"]
    metric = 0
    continue_cnt = 1.0
    relevance_sum_so_far = 0
    rank = 0
    for i in range(len(relevance_list)):
        alpha_multi = 1.0
        for j in range(len(relevance_list[i])):
            alpha = math.log(rank + 2, 2)/math.log(rank + 3, 2)
            if i < 1:
                relevance_sum_so_far += relevance_list[i][j]
                metric += continue_cnt * alpha_multi * (1 - alpha) * relevance_sum_so_far
            else:
                relevance_sum_so_far += (1-gamma) * relevance_list[i][j]
                metric += continue_cnt * (1-gamma)  * alpha_multi * (1 - alpha) * relevance_sum_so_far
            rank += 1
            alpha_multi *= alpha
        if i < 1:
            continue_cnt *= alpha_multi
        else:
            continue_cnt *= (gamma + (1-gamma) * alpha_multi)
    return metric / rank

def DCG_MB(relevance_list, **kwargs):
    sigma = kwargs["sigma"]
    metric = 0
    continue_cnt = 1.0
    relevance_sum_so_far = 0
    rank = 0
    for i in range(len(relevance_list)):
        alpha_multi = 1.0
        for j in range(len(relevance_list[i])):
            relevance_sum_so_far += relevance_list[i][j]
            middle_position = (len(relevance_list[i]) - 1) * 1.0 / 2
            normal_f = math.e ** get_normal_dis(sigma, j - middle_position)
            alpha = math.log(rank + 2, 2) / math.log(rank + 3, 2)
            stopping_probability = normal_f * continue_cnt * alpha_multi * (1 - alpha)
            stopping_probability = min(0.9999, stopping_probability)
            metric += stopping_probability * relevance_sum_so_far
            rank += 1
            alpha_multi *= alpha
        continue_cnt *= alpha_multi
    return metric

def ERR(relevance_list, **kwargs):
    max_rel = get_max(relevance_list,0)
    max_rel = 100
    metric = 0.0
    relevance_sum_so_far = 0.0
    cum_c = 1.0
    for i in range(len(relevance_list)):
        relevance_sum_so_far += relevance_list[i]
        alpha = (2 ** relevance_list[i] - 1) / (2 ** max_rel)
        metric += (1 - alpha) * cum_c * relevance_sum_so_far
        cum_c *= alpha
    return metric

def ERR_SD(relevance_list, **kwargs):
    max_rel = get_max(relevance_list,1)
    beta = kwargs["beta"]
    metric = 0
    cum_c = 1.0
    cum_beta = 1.0
    relevance_sum_so_far = 0
    rank = 0
    for i in range(len(relevance_list)):
        for j in range(len(relevance_list[i])):
            relevance_sum_so_far += relevance_list[i][j]
            alpha = (2 ** relevance_list[i][j] - 1) / (2 ** max_rel)
            stopping_probability = min(cum_beta * (1 - alpha) * cum_c, 0.9999)
            metric += stopping_probability * relevance_sum_so_far
            cum_c *= alpha
            rank += 1
        cum_beta *= beta
    return metric

def ERR_MB(relevance_list, **kwargs):
    max_rel = get_max(relevance_list, 1)
    sigma = kwargs["sigma"]
    metric = 0
    continue_cnt = 1.0
    relevance_sum_so_far = 0
    rank = 0
    for i in range(len(relevance_list)):
        alpha_multi = 1.0
        for j in range(len(relevance_list[i])):
            relevance_sum_so_far += relevance_list[i][j]
            middle_position = (len(relevance_list[i]) - 1) * 1.0 / 2
            normal_f = math.e ** get_normal_dis(sigma, j - middle_position)
            alpha = (2 ** relevance_list[i][j] - 1) / (2 ** max_rel)
            stopping_probability = normal_f * continue_cnt * alpha_multi * (1 - alpha)
            stopping_probability = min(0.9999, stopping_probability)
            metric += stopping_probability * relevance_sum_so_far
            alpha_multi *= alpha
    continue_cnt *= alpha_multi
    return metric

def ERR_RS(relevance_list, **kwargs):
    max_rel = get_max(relevance_list,1)
    gamma = kwargs["gamma"]
    metric = 0
    continue_cnt = 1.0
    relevance_sum_so_far = 0
    rank = 0
    for i in range(len(relevance_list)):
        alpha_multi = 1.0
        for j in range(len(relevance_list[i])):
            alpha = (2 ** relevance_list[i][j] - 1) / (2 ** max_rel)
            if i < 1:
                relevance_sum_so_far += relevance_list[i][j]
                metric += continue_cnt * alpha_multi * (1 - alpha) * relevance_sum_so_far
            else:
                relevance_sum_so_far += (1-gamma) * relevance_list[i][j]
                metric += continue_cnt * (1-gamma) * alpha_multi * (1 - alpha) * relevance_sum_so_far
            rank += 1
            alpha_multi *= alpha
        if i < 1:
            continue_cnt *= alpha_multi
        else:
            continue_cnt *= (gamma+(1-gamma)*alpha_multi)
    return metric / rank

if __name__ == "__main__":
    pass