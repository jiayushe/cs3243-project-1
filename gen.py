import os
import sys
import random

if __name__ == "__main__":
    n = int(sys.argv[1], 10)
    max_num = n ** 2

    random_list = [i for i in range(0, max_num)]
    random.shuffle(random_list)
    random_state = [[0 for i in range(n)] for j in range(n)]

    for i in range(0, max_num):
        random_state[i//n][i%n] = random_list[i]

    with open(sys.argv[2], 'a') as f:
        for line in random_state:
            first = 1
            for word in line:
                if first == 1:
                    first = 0
                else:
                    f.write(' ')
                f.write(str(word))
            f.write('\n')
