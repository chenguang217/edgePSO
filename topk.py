# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:45:45 2021

@author: 王家殷
"""

import collections
import math
import random
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)
wMax = 440000000

def randomIntList(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def sort_key(old_dict, reverse=False):
    keys = sorted(old_dict.keys(), reverse=reverse)
    new_dict = collections.OrderedDict()
    for key in keys:
        new_dict[key] = old_dict[key]
    return new_dict


def fitness(baseStationSet, X, M):
    PowerSum = 0
    count = 0
    for i in range(len(X)):
        if X[i] == 1:
            count += 1
            traffic = 0
            for j in range(len(X)):
                if M[j][i] == 1:
                    traffic += baseStationSet[j].traffic
            PowerSum += traffic / wMax * 800 + 1200
    # print('result', count)
    return 1 / PowerSum


'''
    topk算法,k值取的是3，即负载最大的3个基站被选作边缘服务器部署位置
'''
def topkplacer(baseStationSet, r):
    # temp = -1
    trafficSum = 0
    varNum = len(baseStationSet)
    base_stations = sorted(baseStationSet, key=lambda x: x.traffic, reverse=True)
    for base in baseStationSet:
        trafficSum += base.traffic
    
    M = np.zeros((varNum, varNum))
    X = np.zeros(varNum)
    flag = 0
    while True:
        unAssign = []
        for base in range(len(M)):
            if 1 not in M[base]:
                unAssign.append(base)
        if len(unAssign) != 0:
            if flag < 3:
                k = base_stations[flag].tid
                flag = flag + 1
            else:
                k = random.choice(unAssign)
            M[k][k] = 1
            X[k] = 1
            queue = {}
            for i in range(varNum):
                if i != k and i in unAssign:
                    q = r / baseStationSet[k].distanceCal(baseStationSet[i]) + baseStationSet[i].traffic / trafficSum
                    queue[q] = i
            queue = sort_key(queue, True)
            traffic = baseStationSet[k].traffic
            for q, i in queue.items():
                if baseStationSet[k].distanceCal(baseStationSet[i]) < r:
                    traffic += baseStationSet[i].traffic
                    if traffic <= wMax:
                        M[i][k] = 1
        else:
            return M, X
                    
        
        
class baseStation:
    def __init__(self, x, y, traffic, tid):
        self.x = x
        self.y = y
        self.traffic = 5 ** (traffic + 6)
        self.tid = tid
    def distanceCal(self, target):
        xDiv = abs(self.x - target.x)
        yDiv = abs(self.y - target.y)
        distance = (math.sqrt(xDiv ** 2 + yDiv ** 2)) / 2
        return distance


if __name__ == '__main__':
    r = 5
    # servernum_up = 25
    # servernum_down = 15
    totalNum = 5533
    baseStationSet = []
    max_fit = 0
    count = -1
    with open('baseStations.csv', 'r') as file:
        while True:
            line = file.readline().strip()
            if len(line) == 0:
                break
            tmp = line.split(',')
            block = int(tmp[0])
            traffic = int(tmp[1])
            x = int((int(block) - 1) / 120) + 0.5
            y = ((int(block) - 1) % 120) + 0.5
            count = count + 1
            baseStationSet.append(baseStation(x, y, traffic, count))
        for i in range(10):
            M, X = topkplacer(baseStationSet, r)
            fit = fitness(baseStationSet, X, M)
            if max_fit < fit:
                M_final = M
                X_final = X
                max_fit = fit
        np.savetxt('X_topk.txt', X_final, fmt='%d')
        np.savetxt('M_topk.txt', M_final, fmt='%d')
        # print(max_fit)