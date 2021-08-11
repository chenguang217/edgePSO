import random
import numpy as np

def randomIntList(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

if __name__ == "__main__":
    blocks = []
    count = 0
    import random
import numpy as np
import math

def randomIntList(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

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
    blocks = []
    count = 0
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
    trafficSum = 0
    for base in baseStationSet:
        trafficSum += base.traffic
    print(trafficSum / 90000000 * 2000)
    print(count)