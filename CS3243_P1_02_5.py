import os
import sys
import random
import time
import shlex
from subprocess import Popen
from threading import Timer

def echo_cyan(string):
    os.system("echo '\033[0;36m> '" + string + "'\033[0m'")

def kill(process):
    try:
        process.kill()
    except OSError: 
        pass

def init():
    if os.path.exists('./experiment'):
        os.system('rm -rf ./experiment')
    os.mkdir('./experiment')

def is_valid_action(n, state, action):
    index = state.index(0)
    zr = index / n
    zc = index % n
    if action == "LEFT":
        return zc < n-1
    elif action == "RIGHT":
        return zc >= 1
    elif action == "UP":
        return zr < n-1
    else:
        return zr >= 1

def number_to_move(move):
    if move == 1:
        return "LEFT"
    elif move == 2:
        return "RIGHT"
    elif move == 3:
        return "UP"
    else:
        return "DOWN"

def move(N, state, action):
    new_state = list(state)
    index = new_state.index(0)
    zr = index / N
    zc = index % N
    if action == 1: # LEFT
        new_state[index] = new_state[zr*N + zc + 1]
        new_state[zr*N + zc + 1] = 0
    elif action == 2: # RIGHT
        new_state[index] = new_state[zr*N + zc - 1]
        new_state[zr*N + zc - 1] = 0
    elif action == 3: # UP
        new_state[index] = new_state[(zr+1)*N + zc]
        new_state[(zr+1)*N + zc] = 0
    elif action == 4: # DOWN
        new_state[index] = new_state[(zr-1)*N + zc]
        new_state[(zr-1)*N + zc] = 0
    else:
        raise Exception("Illegal action found in move function: " + action)
    return tuple(new_state)

def generate_puzzle(n):
    init_state = None
    if n == 3:
        init_state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    elif n == 4:
        init_state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
    elif n == 5:
        init_state = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0)
    else:
        raise Exception("exception at generate_puzzle")
    cur_state = init_state
    for i in range(100):
        r = random.randrange(4)+1
        action = number_to_move(r)
        while not is_valid_action(n, cur_state, action):
            r = random.randrange(4)+1
            action = number_to_move(r)
        cur_state = move(n, cur_state, r)
    return cur_state

def generate_input(dimension, sample_size, prefix):
    max_num = dimension ** 2

    for cnt in range(0, sample_size):
        random_list = generate_puzzle(dimension)
        random_state = [[0 for i in range(dimension)] for j in range(dimension)]
        for i in range(0, max_num):
            random_state[i//dimension][i%dimension] = random_list[i]
        with open(prefix + str(cnt) + '.in', 'a') as f:
            for line in random_state:
                first = 1
                for word in line:
                    if first == 1:
                        first = 0
                    else:
                        f.write(' ')
                    f.write(str(word))
                f.write('\n')

def mean(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

def stdev(lst):
    if len(lst) == 0:
        return 0
    m = mean(lst)
    res = 0
    for i in lst:
        res += (i - m) ** 2
    return (res / len(lst)) ** 0.5

def maximum(lst):
    if len(lst) == 0:
        return 0
    return max(lst)

if __name__ == '__main__':
    init()
    random.seed(1)
    time_limit = 60
    sample_size = 10
    for dimension in {3, 4, 5}:
        echo_cyan('Dimension = ' + str(dimension))
        prefix = 'experiment/test_' + str(dimension) + '_'
        generate_input(dimension, sample_size, prefix)
        for solution_index in {1, 2, 3, 4}:
            solution_file = 'CS3243_P1_02_' + str(solution_index) + '.py'
            echo_cyan('Solution = ' + str(solution_file))
            runtime_arr = []
            for cnt in range(0, sample_size):
                input_file = prefix + str(cnt) + '.in'
                output_file = prefix + str(cnt) + '_' + str(solution_index) + '.out'
                cmd = 'python ' + solution_file + ' ' + input_file + ' ' + output_file
                echo_cyan(input_file)
                process = Popen(shlex.split(cmd))
                timer = Timer(time_limit, kill, [process])
                start_time = time.time()
                timer.start()
                return_code = process.wait()
                timer.cancel()
                runtime = time.time() - start_time
                if return_code == 0:
                    runtime_arr.append(runtime)
                    sys.stdout.write('Puzzle solved (' + str(runtime) + 's)\n')
                else:
                    sys.stdout.write('Time limit exceeded (' + str(time_limit) + 's)\n')
            echo_cyan('Pass Rate:')
            sys.stdout.write(str(len(runtime_arr) / sample_size * 100) + '%\n')
            echo_cyan('Average Runtime:')
            sys.stdout.write(str(mean(runtime_arr)) + 's\n')
            echo_cyan('Standard Deviation:')
            sys.stdout.write(str(stdev(runtime_arr)) + 's\n')
            echo_cyan('Max Runtime:')
            sys.stdout.write(str(maximum(runtime_arr)) + 's\n')
