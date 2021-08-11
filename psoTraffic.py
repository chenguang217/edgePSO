import os
import math

import collections
import math
import random
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)
wMax = [220000000, 2200000000]
userLimit = 41
largeTraffic = 0.6
largeTraffic2 = 74559107
powerNorm = 242488

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
    def __init__(self, baseStationSet, NGEN, popSize, r1, r2, alpha):
        # 初始化
        self.NGEN = NGEN                            # 迭代的代数
        self.popSize = popSize                      # 种群大小
        self.varNum = len(baseStationSet)           # 变量个数
        self.popM = []                              # 基站分配矩阵M
        self.popX = []                              # 粒子位置X
        self.popV = []                              # 粒子速度V
        self.p_best = []                            # 局部最优
        self.r = [r1, r2]                           # 覆盖范围设置
        self.baseStationSet = baseStationSet        # 基站集
        self.trafficSum = 0                         # 流量总和，q计算中需要
        self.fits = []                              # 适应度
        self.pfits = []                             # 局部最优适应度
        self.alpha = alpha                          # 平衡参数
        temp = -1
        # ---------------计算流量总和---------------
        for base in baseStationSet:
            self.trafficSum += base.traffic

        for a in range(popSize):
            M = np.zeros((self.varNum, self.varNum))
            X = np.zeros(self.varNum)
            while True:
                unAssign = []
                for base in range(len(M)):
                    if 1 not in M[base] and 2 not in M[base]:
                        unAssign.append(base)
                if len(unAssign) != 0:
                    # print(len(unAssign))
                    # ---------------随机选取位置部署边缘服务器---------------
                    k = random.choice(unAssign)
                    M[k][k] = 1
                    X[k] = 1
                    queue = {}
                    for i in range(self.varNum):
                        if i != k and i in unAssign:
                            # ---------------对高可用节点延迟惩罚---------------
                            if baseStationSet[i].users >= userLimit:
                                q = (r1 / (baseStationSet[k].distanceCal(baseStationSet[i]) + self.r[0] * 0.3)) + ((baseStationSet[i].traffic + largeTraffic * baseStationSet[i].traffic) / self.trafficSum)
                            else:
                                q = r1 / baseStationSet[k].distanceCal(baseStationSet[i]) + (baseStationSet[i].traffic / self.trafficSum)
                            queue[q] = i
                    # ---------------排序q队列并进行分配操作---------------
                    queue = sort_key(queue, True)
                    traffic = 0
                    count = 0
                    for q, i in queue.items():
                        # traffic += baseStationSet[i].traffic
                        # count += 1
                        if baseStationSet[i].users >= userLimit and (baseStationSet[k].distanceCal(baseStationSet[i]) + self.r[1] * 0.3) < self.r[1]:
                            traffic += baseStationSet[i].traffic
                            count += 1
                        elif baseStationSet[i].users < userLimit and baseStationSet[k].distanceCal(baseStationSet[i]) < self.r[1]:
                            traffic += baseStationSet[i].traffic
                            count += 1
                        # -------------前10个节点超过4G上限则部署5G-------------
                        if count == 10:
                            # print(k, traffic - trafficTmp)
                            if traffic > wMax[0]:
                                # print(k, traffic - trafficTmp)
                                allocation = 2
                                X[k] = 2
                                M[k][k] = 2
                            else:
                                allocation = 1
                                X[k] = 1
                                M[k][k] = 1
                            break
                    else:
                        # print(k, traffic - trafficTmp)
                        if traffic > wMax[0]:
                            # print(k, traffic - trafficTmp)
                            allocation = 2
                            X[k] = 2
                            M[k][k] = 2
                        else:
                            allocation = 1
                            X[k] = 1
                            M[k][k] = 1
                    traffic = 0
                    # ---------------实际分配过程---------------
                    for q, i in queue.items():
                        if baseStationSet[i].users < userLimit:
                            if baseStationSet[k].distanceCal(baseStationSet[i]) < self.r[allocation - 1] and baseStationSet[i].traffic + traffic < wMax[allocation - 1]:
                                M[i][k] = allocation
                                traffic += baseStationSet[i].traffic
                        else:
                            if baseStationSet[k].distanceCal(baseStationSet[i]) + self.r[allocation - 1] * 0.3 < self.r[allocation - 1] and baseStationSet[i].traffic + traffic + largeTraffic * baseStationSet[i].traffic < wMax[allocation - 1]:
                                M[i][k] = allocation
                                traffic += baseStationSet[i].traffic + largeTraffic * baseStationSet[i].traffic
                else:
                    self.popM.append(M)
                    self.popX.append(X)
                    self.popV.append(randomIntList(0, 2, self.varNum))
                    self.p_best.append(deepcopy(X))
                    fit = self.fitness(X, M)
                    print(np.count_nonzero(X == 1))
                    print(np.count_nonzero(X == 2))
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
        delay = 0
        for i in range(len(X)):
            # ---------------统计4G---------------
            if X[i] == 1:
                count = 0
                traffic = 0
                for j in range(len(X)):
                    # ---------------判断是否为高可用---------------
                    if self.baseStationSet[j].users < userLimit:
                        if M[j][i] == 1:
                            traffic += self.baseStationSet[j].traffic
                            delay += self.baseStationSet[j].distanceCal(self.baseStationSet[i])
                    else:
                        if M[j][i] == 1:
                            traffic += self.baseStationSet[j].traffic + largeTraffic * baseStationSet[j].traffic
                            delay += self.baseStationSet[j].distanceCal(self.baseStationSet[i]) + self.r[0] * 0.3
                PowerSum += traffic / wMax[0] * 800 + 1200
            # ---------------统计5G---------------
            elif X[i] == 2:
                traffic = 0
                for j in range(len(X)):
                    # ---------------判断是否为高可用---------------
                    if self.baseStationSet[j].users < userLimit:
                        if M[j][i] == 2:
                            traffic += self.baseStationSet[j].traffic
                            delay += self.baseStationSet[j].distanceCal(self.baseStationSet[i])
                    else:
                        if M[j][i] == 2:
                            traffic += self.baseStationSet[j].traffic + largeTraffic * baseStationSet[j].traffic
                            delay += self.baseStationSet[j].distanceCal(self.baseStationSet[i]) + self.r[0] * 0.3
                # PowerSum += traffic / wMax[1] * 800 + 1200
                PowerSum += traffic / wMax[1] * 1400 + 2100
        delay = delay / self.varNum
        totalCost = self.alpha * PowerSum / powerNorm + (1 - self.alpha) * delay / self.r[0] * 2
        # print(0.5 * PowerSum / 121244, 0.5 * delay / self.r[0] * 2)
            
        return 1 / totalCost

    def update_operator(self):
        for i in range(self.popSize):
            # ---------------计算速度---------------
            p1 = self.fits[i] / (self.fits[i] + self.pfits[i] + self.gfit)
            p2 = self.pfits[i] / (self.fits[i] + self.pfits[i] + self.gfit)
            # p3 = self.gfit / (self.fits[i] + self.pfits[i] + self.gfit)
            tmp1 = [b for b in self.popV[i]]
            tmp2 = [(int(self.popX[i][j]) ^ int(self.p_best[i][j])) for j in range(self.varNum)]
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
            # ---------------应用速度，并去除非法分配---------------
            for j in range(self.varNum):
                if Vnew[j] == 1 or Vnew[j] == 2:
                    if X[j] == 1 and Vnew[j] == 1:
                        for k in range(self.varNum):
                            M[k][j] = 0
                    elif X[j] == 0 and Vnew[j] == 1:
                        for k in range(self.varNum):
                            M[j][k] = 0
                        M[j][j] = 1
                    elif X[j] == 2 and Vnew[j] == 2:
                        for k in range(self.varNum):
                            M[k][j] = 0
                    elif X[j] == 2 and Vnew[j] == 1:
                        for k in range(self.varNum):
                            M[k][j] = 0
                        M[j][j] = 1
                    elif X[j] == 1 and Vnew[j] == 2:
                        for k in range(self.varNum):
                            M[k][j] = 0
                        M[j][j] = 2
                    elif X[j] == 2 and Vnew[j] == 1:
                        for k in range(self.varNum):
                            M[j][k] = 0
                        M[j][j] = 2
                    if X[j] == 1 and Vnew[j] == 2:
                        X[j] = 2
                    else:
                        X[j] = int(abs(X[j] - Vnew[j]))
            # ---------------计算可接受的平均消耗---------------
            noAllo = 0
            for j in range(self.varNum):
                if 1 not in M[j]:
                    noAllo += 1
            if noAllo != 0:
                tmpCost = self.fitness(X, M)
                avgCost = (self.pfits[i] - 1 / tmpCost) / noAllo
                if avgCost< 0:
                    avgCost = 2
                # ---------------分配未分配的基站---------------
                for j in range(self.varNum):
                    if 1 not in M[j] and 2 not in M[j] and self.baseStationSet[j].users < userLimit:
                        bestChoice = -1
                        bestCost = 2
                        for k in range(self.varNum):
                            if k == j:
                                continue
                            if X[k] == 0:
                                continue
                            if self.baseStationSet[k].distanceCal(self.baseStationSet[j]) < self.r[int(X[k] - 1)]:
                                traffic = self.baseStationSet[j].traffic
                                for l in range(self.varNum):
                                    if M[l][k] == X[k]:
                                        if self.baseStationSet[l].users < userLimit:
                                            traffic += self.baseStationSet[l].traffic
                                        else:
                                            traffic += self.baseStationSet[l].traffic + largeTraffic * self.baseStationSet[l].traffic
                                if traffic <= wMax[int(X[k] - 1)]:
                                    w1 = [800, 1400]
                                    appendCost = self.alpha * (self.baseStationSet[j].traffic * w1[int(X[k] - 1)] / wMax[int(X[k] - 1)]) / powerNorm + (1 - self.alpha) * (self.baseStationSet[j].distanceCal(self.baseStationSet[k]) / self.varNum * self.r[int(X[k] - 1)] / 2)
                                    if bestCost > appendCost:
                                        bestCost = appendCost
                                        bestChoice = k
                                    if appendCost < avgCost:
                                        M[j][k] = X[k]
                                        break
                        else:
                            if bestChoice != -1:
                                M[j][bestChoice] = X[bestChoice]
                            else:
                                if self.baseStationSet[j].traffic >= largeTraffic2:
                                    M[j][j] = 2
                                    X[j] = 2
                                else:
                                    M[j][j] = 1
                                    X[j] = 1
                        # else:
                        #     M[j][minChoice] = X[minChoice]
                    elif 1 not in M[j] and 2 not in M[j] and self.baseStationSet[j].users >= userLimit:
                        bestChoice = -1
                        bestCost = 2
                        for k in range(self.varNum):
                            if k == j:
                                continue
                            if X[k] == 0:
                                continue
                            if self.baseStationSet[k].distanceCal(self.baseStationSet[j]) + self.r[int(X[k] - 1)] * 0.3 < self.r[int(X[k] - 1)]:
                                traffic = self.baseStationSet[j].traffic + largeTraffic * self.baseStationSet[j].traffic
                                for l in range(self.varNum):
                                    if M[l][k] == X[k]:
                                        if self.baseStationSet[l].users < userLimit:
                                            traffic += self.baseStationSet[l].traffic
                                        else:
                                            traffic += self.baseStationSet[l].traffic + largeTraffic * self.baseStationSet[l].traffic
                                if traffic <= wMax[int(X[k] - 1)]:
                                    w1 = [800, 1400]
                                    appendCost = self.alpha * ((1 + largeTraffic) * self.baseStationSet[j].traffic * w1[int(X[k] - 1)] / wMax[int(X[k] - 1)]) / powerNorm + (1 - self.alpha) * ((self.baseStationSet[j].distanceCal(self.baseStationSet[k]) + self.r[int(X[k] - 1)] * 0.3) / self.varNum * self.r[int(X[k] - 1)] / 2)
                                    if bestCost > appendCost:
                                        bestCost = appendCost
                                        bestChoice = k
                                    if appendCost < avgCost:
                                        M[j][k] = X[k]
                                        break
                                    M[j][k] = X[k]
                                    break
                        else:
                            if bestChoice != -1:
                                M[j][bestChoice] = X[bestChoice]
                            else:
                                if self.baseStationSet[j].traffic >= largeTraffic2:
                                    M[j][j] = 2
                                    X[j] = 2
                                else:
                                    M[j][j] = 1
                                    X[j] = 1

                    
                    # if 1 not in M[j] and 2 not in M[j] and self.baseStationSet[j].users < userLimit:
                    #     maxProfit = -1
                    #     maxChoice = -1
                    #     for k in range(self.varNum):
                    #         if k == j:
                    #             continue
                    #         if X[k] == 0:
                    #             continue
                    #         if self.baseStationSet[k].distanceCal(self.baseStationSet[j]) < self.r[int(X[k] - 1)]:
                    #             traffic = self.baseStationSet[j].traffic
                    #             for l in range(self.varNum):
                    #                 if M[l][k] == 1:
                    #                     traffic += self.baseStationSet[l].traffic
                    #             if traffic <= wMax[int(X[k] - 1)]:
                    #                 old = self.fitness(X, M)
                    #                 Mtmp = deepcopy(M)
                    #                 Mtmp[j][k] = 1
                    #                 new = self.fitness(X, Mtmp)
                    #                 if (new - old) > maxProfit:
                    #                     maxProfit = new - old
                    #                     maxChoice = k
                    #     print(maxChoice)
                    #     if maxChoice != -1:
                    #         M[j][maxChoice] = X[maxChoice]
                    #     else:
                    #         print('new edge')
                    #         if self.baseStationSet[j].traffic > largeTraffic2:
                    #             M[j][j] = 2
                    #             X[j] = 2
                    #         else:
                    #             M[j][j] = 1
                    #             X[j] = 1
                    # elif 1 not in M[j] and 2 not in M[j] and self.baseStationSet[j].users >= userLimit:
                    #     maxProfit = -1
                    #     maxChoice = -1
                    #     for k in range(self.varNum):
                    #         if k == j:
                    #             continue
                    #         if X[k] == 0:
                    #             continue
                    #         if self.baseStationSet[k].distanceCal(self.baseStationSet[j]) + self.r[0] * 0.3 < self.r[int(X[k] - 1)]:
                    #             traffic = self.baseStationSet[j].traffic + largeTraffic * self.baseStationSet[j].traffic
                    #             for l in range(self.varNum):
                    #                 if M[l][k] == 1:
                    #                     traffic += self.baseStationSet[l].traffic
                    #             if traffic <= wMax[int(X[k] - 1)]:
                    #                 old = self.fitness(X, M)
                    #                 Mtmp = deepcopy(M)
                    #                 Mtmp[j][k] = 1
                    #                 new = self.fitness(X, Mtmp)
                    #                 if (new - old) > maxProfit:
                    #                     maxProfit = new - old
                    #                     maxChoice = k
                    #     if maxChoice != -1:
                    #         M[j][maxChoice] = X[maxChoice]
                    #     else:
                    #         print('new edge')
                    #         if self.baseStationSet[j].traffic > largeTraffic2:
                    #             M[j][j] = 2
                    #             X[j] = 2
                    #         else:
                    #             M[j][j] = 1
                    #             X[j] = 1
            # ---------------更新lbest gbest---------------
            self.popX[i] = X
            self.popM[i] = M
            fit = self.fitness(X, M)
            self.fits[i] = fit
            # print(np.count_nonzero(X == 1))
            # print(np.count_nonzero(X == 2))
            print(fit)
            if fit > self.pfits[i]:
                self.p_best[i] = deepcopy(X)
                self.pfits[i] = fit
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
        np.savetxt('X-1.txt', self.best, fmt='%d')
        np.savetxt('M-1.txt', self.bestM, fmt='%d')
        print(np.count_nonzero(self.best == 1))
        print(np.count_nonzero(self.best == 2))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, popobj, color='b', linewidth=2)
        plt.show()


class baseStation:
    def __init__(self, x, y, traffic, users):
        self.x = x
        self.y = y
        self.users = users
        self.traffic = 3 ** (traffic + 9.5)
        
    def distanceCal(self, target):
        xDiv = abs(self.x - target.x)
        yDiv = abs(self.y - target.y)
        distance = (math.sqrt(xDiv ** 2 + yDiv ** 2)) / 2
        return distance


if __name__ == "__main__":
    r1 = 5                                                       # 4G覆盖范围单位为100m
    r2 = 2
    popSize = 40
    NGEN = 100
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
            users = int(tmp[3])
            # if block == 973:
            #     users = 38
            # else:
            #     users = 38
            baseStationSet.append(baseStation(x, y, traffic, users))
    pso = PSO(baseStationSet, NGEN, popSize, r1, r2, 0.7)
    pso.main()