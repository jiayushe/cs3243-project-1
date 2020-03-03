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

def is_solvable(state):
        n = len(state)
        dimension = n ** 0.5
        inversions = 0
        blank = 0
        for i in range(n):
            if state[i] == 0:
                blank = i
                continue
            for j in range(i + 1, n):
                if state[j] == 0:
                    continue
                if (state[i] > state[j]):
                    inversions += 1
        if dimension % 2 == 1:
            return inversions % 2 == 0
        else:
            row_from_bottom = dimension - blank // dimension - 1
            return row_from_bottom % 2 == inversions % 2

def generate_input(dimension, sample_size):
    if os.path.exists('./experiment'):
        os.system("rm -rf ./experiment")
    os.mkdir('./experiment')

    max_num = dimension ** 2

    for cnt in range(0, sample_size):
        random_list = [i for i in range(0, max_num)]
        random.shuffle(random_list)
        while is_solvable(random_list) == False:
            random.shuffle(random_list)
        random_state = [[0 for i in range(dimension)] for j in range(dimension)]
        for i in range(0, max_num):
            random_state[i//dimension][i%dimension] = random_list[i]
        with open('experiment/test' + str(cnt) + '.in', 'a') as f:
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
    return sum(lst) / len(lst)

def stdev(lst):
    m = mean(lst)
    res = 0
    for i in lst:
        res += (i - m) ** 2
    return (res / len(lst)) ** 0.5

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.stderr.write("Parameters: dimension, solution_index, time_limit, sample_size\n")
        sys.stderr.write("Example: python test.py 4 0 10 10\n")
        exit(0)
    
    dimension = int(sys.argv[1], 10)
    solution_file = "CS3243_P1_02_" + sys.argv[2] + ".py"
    time_limit = int(sys.argv[3], 10)
    sample_size = int(sys.argv[4], 10)

    generate_input(dimension, sample_size)

    runtime_arr = []

    for i in range(0, sample_size):
        input_file = "experiment/test" + str(i) + ".in"
        output_file = "experiment/test" + str(i) + ".out"
        cmd = "python " + solution_file + " " + input_file + " " + output_file
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
            sys.stdout.write("Puzzle solved (" + str(runtime) + "s)\n")
        else:
            sys.stdout.write("Time limit exceeded (" + str(time_limit) + "s)\n")
    
    echo_cyan("Pass Rate:")
    sys.stdout.write(str(len(runtime_arr) / sample_size * 100) + "%\n")
    echo_cyan("Average Runtime:")
    sys.stdout.write(str(mean(runtime_arr)) + "s\n")
    echo_cyan("Standard Deviation:")
    sys.stdout.write(str(stdev(runtime_arr)) + "s\n")
    echo_cyan("Max Runtime:")
    sys.stdout.write(str(max(runtime_arr)) + "s\n")
   