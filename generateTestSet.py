import random
import math
import collections

def generateFromFull(blocks):
    with open('baseStations.csv', 'w') as file:
        for block in blocks:
            file.write(block + ',' + str(random.randint(0, 5)) + '\n')
            # file.write(block + ',' + str(8) + '\n')

def distanceCal(block1, block2):
    x1 = int((int(block1) - 1) / 120) + 0.5
    y1 = ((int(block1) - 1) % 120) + 0.5
    x2 = int((int(block2) - 1) / 120) + 0.5
    y2 = ((int(block2) - 1) % 120) + 0.5
    xDiv = abs(x1 - x2)
    yDiv = abs(y1 - y2)
    distance = (math.sqrt(xDiv ** 2 + yDiv ** 2)) / 2
    return distance

def sortKey(old_dict, reverse=False):
    keys = sorted(old_dict.keys(), reverse=reverse)
    new_dict = collections.OrderedDict()
    for key in keys:
        new_dict[key] = old_dict[key]
    return new_dict

if __name__ == "__main__":
    r = 5
    with open('blocks.txt', 'r') as file:
        blocks = file.read().split(',')
    with open('blocksFull.txt', 'r') as file:
        blocksFull = file.read().split(',')
    trainSet = {}
    with open('baseStations.csv', 'r') as file:
        while True:
            line = file.readline().strip()
            if len(line) == 0:
                break
            tmp = line.split(',')
            block = int(tmp[0])
            traffic = int(tmp[1])
            trainSet[block] = traffic
    result = {}
    # for block, traffic in trainSet.items():
    #     result[block] = traffic
    #     for blockTmp in blocksFull:
    #         block2 = int(blockTmp)
    #         if block == block2:
    #             continue
    #         if distanceCal(block, block2) < r:
    #             if block2 not in result.keys():
    #                 result[block2] = traffic
    #             else:
    #                 try:
    #                     result[block2][0]
    #                     result[block2].append(traffic)
    #                 except:
    #                     result[block2] = [result[block2], traffic]
    print(len(blocksFull))
    for blockTmp in blocksFull:
        nearest = 1000000
        trafficTmp = -1
        block = int(blockTmp)
        for block2, traffic in trainSet.items():
            # result[block2] = traffic
            dis = distanceCal(block, block2)
            if dis < nearest:
                nearest = dis
                trafficTmp = traffic
        result[block] = trafficTmp
    sortKey(result)
    with open('trainSetForPso.csv', 'w') as file:
        for block, traffic in result.items():
            file.write(str(block) + ',' + str(traffic) + '\n')
        

