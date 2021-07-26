import collections
import math
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(threshold=np.inf)
wMax = [440000000, 2200000000]
totalNum = 5951
userLimit = 39
largeTraffic = 9765625

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
        temp = -1
        # ---------------计算流量总和---------------
        for base in baseStationSet:
            self.trafficSum += base.traffic

        # ---------------开始生成粒子群---------------
        for a in range(popSize):
            M = np.zeros((self.varNum, self.varNum))
            X = np.zeros(self.varNum)
            oldLength = 0
            while True:
                # ---------------统计未分配基站---------------
                unAssign = []
                
                for base in range(len(M)):
                    if 1 not in M[base] and 2 not in M[base]:
                        unAssign.append(base)
                    # ---------------对高可用节点进行统计---------------
                    elif baseStationSet[base].users >= userLimit:
                        count = 0
                        for i in range(len(M[base])):
                            if M[base][i] != 0:
                                count += 1
                        if count < 2 :
                            unAssign.append(base)
                            # print(base, M[base])
                if oldLength == len(unAssign):
                    print('length not change')
                oldLength = len(unAssign)
                if 0 < len(unAssign) <= 10:
                    print(len(unAssign))
                    tmp = []
                    flag = False
                    for base in unAssign:
                        if M[base][base] != 0:
                            print(base, '已经是边节点')
                            for i in range(self.varNum):
                                if i == base:
                                    continue
                                if M[i][i] == 1:
                                    if baseStationSet[base].users >= userLimit:
                                        if baseStationSet[base].distanceCal(baseStationSet[i]) + self.r[0] * 0.3 < self.r[0]:
                                            traffic = self.baseStationSet[i].traffic
                                            for j in range(self.varNum):
                                                if M[j][i] == 1:
                                                    traffic += self.baseStationSet[j].traffic
                                            if (traffic + self.baseStationSet[base].traffic < wMax[0]):
                                                M[base][i] = 1
                                                print(i, '可以分配')
                                                break
                                    else:
                                        if baseStationSet[base].distanceCal(baseStationSet[i]) < self.r[0]:
                                            traffic = self.baseStationSet[i].traffic
                                            for j in range(self.varNum):
                                                if M[j][i] == 1:
                                                    traffic += self.baseStationSet[j].traffic
                                            if (traffic + self.baseStationSet[base].traffic < wMax[0]):
                                                M[base][i] = 1
                                                print(i, '可以分配')
                                                break
                                if M[i][i] == 2:
                                    if baseStationSet[base].users >= userLimit:
                                        if baseStationSet[base].distanceCal(baseStationSet[i]) + self.r[1] * 0.3 < self.r[1]:
                                            traffic = self.baseStationSet[i].traffic
                                            for j in range(self.varNum):
                                                if M[j][i] == 1:
                                                    traffic += self.baseStationSet[j].traffic
                                            if (traffic + self.baseStationSet[base].traffic < wMax[1]):
                                                M[base][i] = 2
                                                print(i, '可以分配')
                                                break
                                    else:
                                        if baseStationSet[base].users >= userLimit:
                                            if baseStationSet[base].distanceCal(baseStationSet[i]) < self.r[1]:
                                                traffic = self.baseStationSet[i].traffic
                                                for j in range(self.varNum):
                                                    if M[j][i] == 1:
                                                        traffic += self.baseStationSet[j].traffic
                                                if (traffic + self.baseStationSet[base].traffic < wMax[1]):
                                                    M[base][i] = 2
                                                    print(i, '可以分配')
                                                    break
                            else:
                                for i in unAssign:
                                    if i == base:
                                        continue
                                    if baseStationSet[base].distanceCal(baseStationSet[i]) + self.r[0] * 0.3 < self.r[0]:
                                        print(i, '可以分配新的')
                                        M[base][i] = 1
                                        M[i][i] = 1
                                        if np.count_nonzero(M[i] == 1) + np.count_nonzero(M[i] == 2) == 2 and baseStationSet[i].users > userLimit:
                                            try:
                                                tmp.remove(i)
                                            except:
                                                pass
                                        elif np.count_nonzero(M[i] == 1) + np.count_nonzero(M[i] == 2) == 1 and baseStationSet[i].users <= userLimit:
                                            try:
                                                tmp.remove(i)
                                            except:
                                                pass
                                        break
                                else:
                                    M[base] = np.zeros(self.varNum)
                                    X[base] = 0
                                    tmp.append(base)
                                    print('wrong')
                                    flag = True
                        elif 1 not in M[base] and 2 not in M[base]:
                            tmp.append(base)
                            print(base, '完全未分配')
                            M[base][base] = 1
                        else:
                            tmp.append(base)
                            print(base, '不是边节点')
                            M[base][base] = 1
                    if len(tmp) == 0:
                        continue
                    if len(tmp) == 1 and flag:
                        # ---------------剩一个，无法完成---------------
                        M = np.zeros((self.varNum, self.varNum))
                        X = np.zeros(self.varNum)
                        continue
                    k = random.choice(tmp)
                    queue = {}
                    for i in tmp:
                        if i != k:
                            # ---------------对高可用节点延迟惩罚---------------
                            if baseStationSet[i].users >= userLimit:
                                q = r1 / (baseStationSet[k].distanceCal(baseStationSet[i]) + self.r[0] * 0.3) + baseStationSet[i].traffic / self.trafficSum
                            else:
                                q = r1 / baseStationSet[k].distanceCal(baseStationSet[i]) + baseStationSet[i].traffic / self.trafficSum
                            queue[q] = i
                    queue = sort_key(queue, True)
                    traffic = 0
                    X[k] = 1
                    M[k][k] = 1
                    for q, i in queue.items():
                        if baseStationSet[i].users < userLimit:
                            if baseStationSet[k].distanceCal(baseStationSet[i]) < self.r[allocation - 1] and baseStationSet[i].traffic + traffic < wMax[allocation - 1]:
                                M[i][k] = 1
                                traffic += baseStationSet[i].traffic
                        else:
                            if baseStationSet[k].distanceCal(baseStationSet[i]) + self.r[allocation - 1] * 0.3 < self.r[allocation - 1] and baseStationSet[i].traffic + traffic < wMax[allocation - 1]:
                                M[i][k] = 1
                                traffic += baseStationSet[i].traffic
                    # for base in unAssign:
                    #     print(M[base])
                    # exit()
                elif len(unAssign) != 0:
                    print(len(unAssign))
                    # ---------------随机选取位置部署边缘服务器---------------
                    k = random.choice(unAssign)
                    queue = {}
                    for i in range(self.varNum):
                        if i != k and i in unAssign:
                            # ---------------对高可用节点延迟惩罚---------------
                            if baseStationSet[i].users >= userLimit:
                                q = r1 / (baseStationSet[k].distanceCal(baseStationSet[i]) + self.r[0] * 0.3) + baseStationSet[i].traffic / self.trafficSum
                            else:
                                q = r1 / baseStationSet[k].distanceCal(baseStationSet[i]) + baseStationSet[i].traffic / self.trafficSum
                            queue[q] = i
                    # ---------------排序q队列并进行分配操作---------------
                    queue = sort_key(queue, True)
                    traffic = 0
                    count = 0
                    for q, i in queue.items():
                        traffic += baseStationSet[i].traffic
                        count += 1
                        # -------------前10个节点超过4G上限则部署5G-------------
                        if count == 10:
                            if traffic > wMax[0]:
                                allocation = 2
                                X[k] = 2
                                M[k][k] = 2
                            else:
                                allocation = 1
                                X[k] = 1
                                M[k][k] = 1
                            break
                    traffic = 0
                    # ---------------实际分配过程---------------
                    for q, i in queue.items():
                        if baseStationSet[i].users < userLimit:
                            if baseStationSet[k].distanceCal(baseStationSet[i]) < self.r[allocation - 1] and baseStationSet[i].traffic + traffic < wMax[allocation - 1]:
                                M[i][k] = allocation
                                traffic += baseStationSet[i].traffic
                        else:
                            if baseStationSet[k].distanceCal(baseStationSet[i]) + self.r[allocation - 1] * 0.3 < self.r[allocation - 1] and baseStationSet[i].traffic + traffic < wMax[allocation - 1]:
                                M[i][k] = allocation
                                traffic += baseStationSet[i].traffic
                        
                else:
                    # ------------更新粒子状态并计算全局最优及局部最优------------
                    self.popM.append(M)
                    self.popX.append(X)
                    self.popV.append(randomIntList(0, 2, self.varNum))
                    self.p_best.append(deepcopy(X))
                    fit = self.fitness(X, M)
                    self.fits.append(fit)
                    self.pfits.append(fit)
                    print(fit)
                    if fit > temp:
                        self.g_best = deepcopy(X)
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
                            traffic += self.baseStationSet[j].traffic
                            delay += self.baseStationSet[j].distanceCal(self.baseStationSet[i]) + self.r[0] * 0.3
                PowerSum += traffic / wMax[0] * 520 + 780
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
                            traffic += self.baseStationSet[j].traffic
                            delay += self.baseStationSet[j].distanceCal(self.baseStationSet[i]) + self.r[0] * 0.3
                PowerSum += traffic / wMax[1] * 1400 + 2100
        delay = delay / totalNum
        totalCost = 0.0005 * PowerSum + 2 * delay
            
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
                if Vnew[j] == 1 or Vnew == 2:
                    if X[j] == 1 and Vnew == 1:
                        for k in range(self.varNum):
                            M[k][j] = 0
                    elif X[j] == 0 and Vnew == 1:
                        for k in range(self.varNum):
                            M[j][k] = 0
                        M[j][j] = 1
                    elif X[j] == 2 and Vnew == 2:
                        for k in range(self.varNum):
                            M[k][j] = 0
                    elif X[j] == 2 and Vnew == 1:
                        for k in range(self.varNum):
                            M[k][j] = 0
                        X[j][j] = 1
                    elif X[j] == 1 and Vnew == 2:
                        for k in range(self.varNum):
                            M[k][j] = 0
                        X[j][j] = 2
                    elif X[j] == 0 and Vnew == 1:
                        for k in range(self.varNum):
                            M[j][k] = 0
                        M[j][j] = 2
                    if X[j] == 1 and Vnew[j] == 2:
                        X[j] = 2
                    else:
                        X[j] = int(abs(X[j] - Vnew[j]))
            # ---------------分配未分配的基站---------------
            for j in range(self.varNum):
                count = 0
                for k in range(self.varNum):
                    if M[j][k] == 1 or M[j][k] == 2:
                        count += 1
                if count < 1 and self.baseStationSet[j].users < userLimit:
                    maxProfit = -1
                    maxChoice = -1
                    for k in range(self.varNum):
                        if self.baseStationSet[k].distanceCal(self.baseStationSet[j]) < self.r[X[k] - 1]:
                            traffic = self.baseStationSet[j].traffic
                            for l in range(self.varNum):
                                if M[l][k] == 1:
                                    traffic += self.baseStationSet[l].traffic
                            if traffic <= wMax[X[k] - 1]:
                                old = self.fitness(X, M)
                                Mtmp = deepcopy(M)
                                Mtmp[j][k] = 1
                                new = self.fitness(X, Mtmp)
                                if (new - old) > maxProfit:
                                    maxProfit = new - old
                                    maxChoice = k
                    if maxChoice != -1:
                        M[j][maxChoice] = X[maxChoice]
                    else:
                        if self.baseStationSet[j].traffic > largeTraffic:
                            M[j][j] = 2
                            X[j] = 2
                        else:
                            M[j][j] = 1
                            X[j] = 1
                elif count < 2 and self.baseStationSet[j].users >= userLimit:
                    for m in range(2):
                        maxProfit = -1
                        maxChoice = -1
                        for k in range(self.varNum):
                            if self.baseStationSet[k].distanceCal(self.baseStationSet[j]) + self.r[X[k] - 1] * 0.3 < self.r[X[k] - 1]:
                                traffic = self.baseStationSet[j].traffic
                                for l in range(self.varNum):
                                    if M[l][k] == 1:
                                        traffic += self.baseStationSet[l].traffic
                                if traffic <= wMax[X[k] - 1]:
                                    old = self.fitness(X, M)
                                    Mtmp = deepcopy(M)
                                    Mtmp[j][k] = 1
                                    new = self.fitness(X, Mtmp)
                                    if (new - old) > maxProfit:
                                        maxProfit = new - old
                                        maxChoice = k
                        if maxChoice != -1:
                            M[j][maxChoice] = X[maxChoice]
                        else:
                            if self.baseStationSet[j].traffic > largeTraffic:
                                M[j][j] = 2
                                X[j] = 2
                            else:
                                M[j][j] = 1
                                X[j] = 1
            # ---------------更新lbest gbest---------------
            self.popX[i] = X
            self.popM[i] = M
            fit = self.fitness(X, M)
            self.fits[i] = fit
            print(fit)
            if fit > self.pfits[i]:
                self.p_best[i] = deepcopy(X)
                self.pfits[i] = fit
            if fit > self.gfit:
                self.g_best = deepcopy(X)
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
    def __init__(self, x, y, traffic, users):
        self.x = x
        self.y = y
        self.users = users
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
            # users = tmp[2]
            if block == 973:
                users = 38
            else:
                users = 40
            baseStationSet.append(baseStation(x, y, traffic, users))
    pso = PSO(baseStationSet, NGEN, popSize, r1, r2)
    pso.main()
