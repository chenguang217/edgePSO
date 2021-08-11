import os
import numpy as np
import math

totalNum = 6443
userLimit = 39
largeTraffic = 74559107
r = [5, 2]
wMax = [90000000, 2200000000]

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

def fitness(X, M, baseStationSet):
    PowerSum = 0
    delay = 0
    for i in range(len(X)):
        # ---------------统计4G---------------
        if X[i] == 1:
            traffic = 0
            for j in range(len(X)):
                # ---------------判断是否为高可用---------------
                if baseStationSet[j].users < userLimit:
                    if M[j][i] == 1:
                        traffic += baseStationSet[j].traffic
                        delay += baseStationSet[j].distanceCal(baseStationSet[i])
                else:
                    if M[j][i] == 1:
                        traffic += baseStationSet[j].traffic + largeTraffic
                        delay += baseStationSet[j].distanceCal(baseStationSet[i]) + r[0] * 0.3
            PowerSum += traffic / wMax[0] * 800 + 1200
        # ---------------统计5G---------------
        elif X[i] == 2:
            traffic = 0
            for j in range(len(X)):
                # ---------------判断是否为高可用---------------
                if baseStationSet[j].users < userLimit:
                    if M[j][i] == 2:
                        traffic += baseStationSet[j].traffic
                        delay += baseStationSet[j].distanceCal(baseStationSet[i])
                else:
                    if M[j][i] == 2:
                        traffic += baseStationSet[j].traffic + largeTraffic
                        delay += baseStationSet[j].distanceCal(baseStationSet[i]) + r[1] * 0.3
            PowerSum += traffic / wMax[1] * 1400 + 2100
    delay = delay / totalNum
    totalCost = 0.0005 * PowerSum + 2 * delay
            
    return totalCost


if __name__ == "__main__":
    X = np.loadtxt('X-1.txt')
    M = np.loadtxt('M-1.txt')
    # for i in range(len(M)):
    #     print(np.count_nonzero(M[i] == 1))
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
                users = 38
            baseStationSet.append(baseStation(x, y, traffic, users))
    cost = fitness(X, M, baseStationSet)
    print(cost)