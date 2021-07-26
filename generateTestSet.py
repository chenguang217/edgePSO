import random

if __name__ == "__main__":
    with open('blocks.txt', 'r') as file:
        blocks = file.read().split(',')
    with open('baseStations.csv', 'w') as file:
        for block in blocks:
            file.write(block + ',' + str(random.randint(0, 5)) + '\n')
            # file.write(block + ',' + str(8) + '\n')
