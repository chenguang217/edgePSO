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


class PSO:
    def __init__(self, baseStationSet, NGEN, popSize, r):
        # 初始化
        self.NGEN = NGEN                        # 迭代的代数
        self.popSize = popSize                  # 种群大小
        self.varNum = len(baseStationSet)       # 变量个数
        self.popM = []
        self.popX = []
        self.popV = []
        self.p_best = []
        self.r = r
        self.baseStationSet = baseStationSet
        self.trafficSum = 0
        self.fits = []
        self.pfits = []
        temp = -1
        for base in baseStationSet:
            self.trafficSum += base.traffic

        for a in range(popSize):
            M = np.zeros((self.varNum, self.varNum))
            X = np.zeros(self.varNum)
            while True:
                unAssign = []
                for base in range(len(M)):
                    if 1 not in M[base]:
                        unAssign.append(base)
                if len(unAssign) != 0:
                    k = random.choice(unAssign)
                    M[k][k] = 1
                    X[k] = 1
                    queue = {}
                    for i in range(self.varNum):
                        if i != k and i in unAssign:
                            q = r / baseStationSet[k].distanceCal(baseStationSet[i]) + baseStationSet[i].traffic / self.trafficSum
                            queue[q] = i
                    queue = sort_key(queue, True)
                    traffic = baseStationSet[k].traffic
                    for q, i in queue.items():
                        if baseStationSet[k].distanceCal(baseStationSet[i]) < r:
                            traffic += baseStationSet[i].traffic
                            if traffic <= wMax:
                                M[i][k] = 1
                else:
                    self.popM.append(M)
                    self.popX.append(X)
                    self.popV.append(randomIntList(0, 1, self.varNum))
                    self.p_best.append(deepcopy(X))
                    fit = self.fitness(X, M)
                    print(fit)
                    self.fits.append(fit)
                    self.pfits.append(fit)
                    if fit > temp:
                        self.g_best = deepcopy(X)
                        self.gM = deepcopy(M)
                        self.gfit = fit
                        temp = fit
                    break

    def fitness(self, X, M):
        PowerSum = 0
        count = 0
        for i in range(len(X)):
            if X[i] == 1:
                count += 1
                traffic = 0
                for j in range(len(X)):
                    if M[j][i] == 1:
                        traffic += self.baseStationSet[j].traffic
                PowerSum += traffic / wMax * 800 + 1200
        # print('result', count)
        return 1 / PowerSum

    def update_operator(self):
        for i in range(self.popSize):
            # 计算速度
            p1 = self.fits[i] / (self.fits[i] + self.pfits[i] + self.gfit)
            p2 = self.pfits[i] / (self.fits[i] + self.pfits[i] + self.gfit)
            # p3 = self.gfit / (self.fits[i] + self.pfits[i] + self.gfit)
            # print(p1,p2,p3)
            tmp1 = [b for b in self.popV[i]]
            tmp2 = [(int(self.popX[i][j]) ^ int(self.p_best[i][j])) for j in range(self.varNum)]
            # for j in range(self.varNum):
            #     print(self.popX[i][j], self.p_best[i][j])
            #     print(int(self.popX[i][j]) ^ int(self.p_best[i][j]))
            tmp3 = [(int(self.popX[i][j]) ^ int(self.g_best[j])) for j in range(self.varNum)]
            Vnew = [0] * self.varNum
            for j in range(self.varNum):
                rand = random.random()
                if rand >= 0 and rand <= p1:
                    Vnew[j] = tmp1[j]
                elif rand > p1 and rand <= p1 + p2:
                    Vnew[j] = tmp2[j]
                elif rand > p1 + p2 and rand < 1:
                    Vnew[j] = tmp3[j]
            self.popV[i] = Vnew
            X = self.popX[i]
            M = self.popM[i]
            # print(Vnew)
            # 应用速度，并去除非法分配
            for j in range(self.varNum):
                if Vnew[j] == 1:
                    if X[j] == 1:
                        for k in range(self.varNum):
                            M[k][j] = 0
                    else:
                        for k in range(self.varNum):
                            M[j][k] = 0
                        M[j][j] = 1
                    X[j] = int(X[j]) ^ 1
            # print(X)
            # 分配未分配的基站
            for j in range(self.varNum):
                if 1 not in M[j]:
                    for k in range(self.varNum):
                        if X[k] == 1:
                            if self.baseStationSet[k].distanceCal(self.baseStationSet[j]) < self.r:
                                traffic = self.baseStationSet[j].traffic
                                for l in range(self.varNum):
                                    if M[l][k] == 1:
                                        traffic += self.baseStationSet[l].traffic
                                if traffic <= wMax:
                                    M[j][k] = 1
                                    break
                    else:
                        M[j][j] = 1
                        X[j] = 1
            for j in range(self.varNum):
                if np.count_nonzero(M[j] == 1) != 1:
                    print(j)
                    print('base error')
                if X[j] == 1 and M[j][j] != 1:
                    print(j)
                    print('edge error')
            # 更新lbest gbest
            self.popX[i] = X
            self.popM[i] = M
            fit = self.fitness(X, M)
            self.fits[i] = fit
            print(fit)
            # print(self.popX[i])
            # self.xDistance(self.g_best, X)
            if fit > self.pfits[i]:
                self.p_best[i] = deepcopy(X)
                self.pfits[i] = fit
                print('update')
            if fit > self.gfit:
                self.g_best = deepcopy(X)
                self.gM = deepcopy(M)
                self.gfit = fit

    def main(self):
        popobj = []
        self.ng_best = 0
        for gen in range(self.NGEN):
            self.update_operator()
            popobj.append(self.gfit)
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.gfit > self.ng_best:
                self.ng_best = self.gfit
                self.best = self.g_best
                self.bestM = self.gM
            # print('最好的位置：{}'.format(self.ng_best))
            print('最大的函数值：{}'.format(self.ng_best))
        np.savetxt('X.txt', self.best, fmt='%d')
        np.savetxt('M.txt', self.bestM, fmt='%d')
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()

class baseStation:
    def __init__(self, x, y, traffic):
        self.x = x
        self.y = y
        self.traffic = 5 ** (traffic + 6)
    def distanceCal(self, target):
        xDiv = abs(self.x - target.x)
        yDiv = abs(self.y - target.y)
        distance = (math.sqrt(xDiv ** 2 + yDiv ** 2)) / 2
        return distance

if __name__ == '__main__':
    r = 5                                                       # 单位为100m
    totalNum = 5533
    popSize = 2
    NGEN = 400
    baseStationSet = []
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
            baseStationSet.append(baseStation(x, y, traffic))
    pso = PSO(baseStationSet, NGEN, popSize, r)
    pso.main()
