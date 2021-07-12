import collections
import math
import random

import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(threshold=np.inf)
wMax = [440000000, 2200000000]
totalNum = 5951

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
    def __init__(self, baseStationSet, NGEN, popSize, r1, r2):
        # 初始化
        self.NGEN = NGEN                        # 迭代的代数
        self.popSize = popSize                  # 种群大小
        self.varNum = len(baseStationSet)       # 变量个数
        self.popM = []
        self.popX = []
        self.popV = []
        self.p_best = []
        self.r = [r1, r2]
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
                    queue = {}
                    for i in range(self.varNum):
                        if i != k and i in unAssign:
                            q = r1 / baseStationSet[k].distanceCal(baseStationSet[i]) + baseStationSet[i].traffic / self.trafficSum
                            queue[q] = i
                    queue = sort_key(queue, True)
                    traffic = 0
                    count = 0
                    for q, i in queue.items():
                        traffic += baseStationSet[i].traffic
                        count += 1
                        if count == 10:
                            if traffic > wMax[0]:
                                allocation = 2
                                X[k] = 2
                                M[k][k] = 2
                            else:
                                allocation = 1
                                X[k] = 1
                                M[k][k] = 1
                            print(allocation)
                            traffic = 0
                            break
                    for q, i in queue.items():
                        if baseStationSet[k].distanceCal(baseStationSet[i]) < self.r[allocation - 1] and baseStationSet[i].traffic + traffic < wMax[allocation - 1]:
                            M[i][k] = allocation
                            traffic += baseStationSet[i].traffic
                else:
                    self.popM.append(M)
                    self.popX.append(X)
                    self.popV.append(randomIntList(0, 2, self.varNum))
                    self.p_best.append(X)
                    fit = self.fitness(X, M)
                    self.fits.append(fit)
                    self.pfits.append(fit)
                    print(fit)
                    if fit > temp:
                        self.g_best = X
                        self.gfit = fit
                        temp = fit
                    break

    def fitness(self, X, M):
        PowerSum = 0
        delay = 0
        for i in range(len(X)):
            if X[i] == 1:
                count = 0
                traffic = 0
                for j in range(len(X)):
                    if M[j][i] == 1:
                        traffic += self.baseStationSet[j].traffic
                        delay += self.baseStationSet[j].distanceCal(self.baseStationSet[j])
                PowerSum += traffic / wMax[0] * 520 + 780
            elif X[i] == 2:
                traffic = 0
                for j in range(len(X)):
                    if M[j][i] == 2:
                        traffic += self.baseStationSet[j].traffic
                        delay += self.baseStationSet[j].distanceCal(self.baseStationSet[j])
                PowerSum += traffic / wMax[1] * 1400 + 2100
        delay = delay / totalNum
        totalCost = 0.0005 * PowerSum + 2 * delay
            
        return 1 / totalCost

    def update_operator(self):
        for i in range(self.popSize):
            # 计算速度
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
            #更新lbest gbest
            self.popX[i] = X
            self.popM[i] = M
            fit = self.fitness(X, M)
            self.fits[i] = fit
            print(fit)
            if fit > self.pfits[i]:
                self.p_best[i] = X
                self.pfits[i] = fit
            if fit > self.gfit:
                self.g_best = X
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
            # print('最好的位置：{}'.format(self.ng_best))
            print('最大的函数值：{}'.format(self.ng_best))
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
        if traffic == 0:
            self.traffic = 5 ** 5.0175
        elif traffic == 1:
            self.traffic = 5 ** 5.767
        elif traffic == 2:
            self.traffic = 5 ** 6.5165
        elif traffic == 3:
            self.traffic = 5 ** 7.266
        elif traffic == 4:
            self.traffic = 5 ** 8.0155
        elif traffic == 5:
            self.traffic = 5 ** 8.765
        elif traffic == 6:
            self.traffic = 5 ** 9.5145
        elif traffic == 7:
            self.traffic = 5 ** 10.264
        elif traffic == 8:
            self.traffic = 5 ** 11.014
    def distanceCal(self, target):
        xDiv = abs(self.x - target.x)
        yDiv = abs(self.y - target.y)
        distance = (math.sqrt(xDiv ** 2 + yDiv ** 2)) / 2
        return distance

if __name__ == '__main__':
    # NGEN = 100
    # popsize = 100
    # low = [1, 1, 1, 1]
    # up = [30, 30, 30, 30]
    # parameters = [NGEN, popsize, low, up]
    # pso = PSO(parameters)
    # pso.main()

    r1 = 5                                                       # 单位为100m
    r2 = 2
    totalNum = 5951
    popSize = 2
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
            baseStationSet.append(baseStation(x, y, traffic))
    pso = PSO(baseStationSet, NGEN, popSize, r1, r2)
    # pso.main()
