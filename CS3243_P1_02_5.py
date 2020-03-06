import os
import sys
import random
import time
import shlex
from subprocess import Popen, PIPE
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

def verify_solution(N, init_state, moves):
    curr_state = init_state
    goal_state_dict = {3: (1, 2, 3, 4, 5, 6, 7, 8, 0),\
                       4: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0),\
                       5: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0)}
    move_dict = {"LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4}
    for move in moves:
        curr_state = move(N, curr_state, move_dict[move])
    return curr_state == goal_state_dict[N]

def tsplit(string, delimiters):
    # Behaves str.split but supports multiple delimiters
    # from http://code.activestate.com/recipes/577616-split-strings-w-multiple-separators/v
    delimiters = tuple(delimiters)
    stack = [string,]
    
    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i+j, _substring)
    print(stack)
    return stack

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
    return float(sum(lst)) / len(lst)

def stdev(lst):
    if len(lst) == 0:
        return 0
    m = mean(lst)
    res = 0
    for i in lst:
        res += (i - m) ** 2
    return (float(res) / len(lst)) ** 0.5

def maximum(lst):
    if len(lst) == 0:
        return 0
    return max(lst)

if __name__ == '__main__':
    init()
    random.seed(1)
    time_limit = 60
    sample_size = 10
    result = []
    for dimension in [3, 4, 5]:
        echo_cyan('Dimension = ' + str(dimension))
        prefix = 'experiment/test_' + str(dimension) + '_'
        generate_input(dimension, sample_size, prefix)
        for solution_index in [1, 2, 3, 4]:
            if solution_index == 1 and dimension != 3:
                continue # skip n=4,5 for BFS
            solution_file = 'CS3243_P1_02_' + str(solution_index) + '.py'
            echo_cyan('Solution = ' + str(solution_file))
            runtime_arr = []
            solution_depth_arr = []
            explored_states_arr = []
            generated_states_arr = []
            max_heap_size_arr = []
            for cnt in range(0, sample_size):
                input_file = prefix + str(cnt) + '.in'
                output_file = prefix + str(cnt) + '_' + str(solution_index) + '.out'
                cmd = 'python ' + solution_file + ' ' + input_file + ' ' + output_file
                echo_cyan(input_file)
                process = Popen(shlex.split(cmd), stderr=PIPE, stdout=PIPE)
                timer = Timer(time_limit, kill, [process])
                start_time = time.time()
                timer.start()
                return_code = process.wait()
                timer.cancel()
                runtime = time.time() - start_time
                
                if return_code == 0:
                    runtime_arr.append(runtime)
                    stdout, stderr = process.communicate()
                    input_f = open(prefix + str(cnt) + '.in','r')
                    init_state = tuple(map(int, tsplit(input_f.read(),('\n',' '))[:-1]))
                    moves_f = open(prefix + str(cnt) + '_' + str(solution_index) + '.out','r')
                    moves_list = moves_f.read().split("\n")
                    solution_correct = verify_solution(dimension, init_state, moves_list)
                    if not solution_correct:
                        raise Exception("Solution incorrect: \n"\
                                        + init_state + "\n" +\
                                        moves_list)
                        sys.exit(0)
                    res = stderr.split('\n')
                    solution_depth_arr.append(int(res[0].split(':')[1]))
                    explored_states_arr.append(int(res[1].split(':')[1]))
                    generated_states_arr.append(int(res[2].split(':')[1]))
                    max_heap_size_arr.append(int(res[3].split(':')[1]))
                    sys.stdout.write('Puzzle solved (' + str(runtime) + 's)\n')
                else:
                    sys.stdout.write('Time limit exceeded (' + str(time_limit) + 's)\n')
            echo_cyan('Pass rate:')
            pass_rate = float(len(runtime_arr)) / sample_size * 100
            sys.stdout.write(str(pass_rate) + '%\n')
            echo_cyan('Average runtime:')
            sys.stdout.write(str(mean(runtime_arr)) + 's\n')
            echo_cyan('Runtime standard deviation:')
            sys.stdout.write(str(stdev(runtime_arr)) + 's\n')
            echo_cyan('Maximum runtime:')
            sys.stdout.write(str(maximum(runtime_arr)) + 's\n')
            echo_cyan('Average solution depth:')
            sys.stdout.write(str(mean(solution_depth_arr)) + '\n')
            echo_cyan('Average explored states:')
            sys.stdout.write(str(mean(explored_states_arr)) + '\n')
            echo_cyan('Average generated states:')
            sys.stdout.write(str(mean(generated_states_arr)) + '\n')
            echo_cyan('Average max heap size:')
            sys.stdout.write(str(mean(max_heap_size_arr)) + '\n')
            sys.stdout.write('\n')
            result.append(( "(%d, %d)" % (dimension, solution_index),
                            pass_rate, 
                            mean(runtime_arr), 
                            stdev(runtime_arr), 
                            maximum(runtime_arr), 
                            mean(solution_depth_arr), 
                            mean(explored_states_arr), 
                            mean(generated_states_arr), 
                            mean(max_heap_size_arr)))
    print("test case,pass rate,avg runtime,runtime stdev,max runtime,avg sol depth,avg explore states,avg generated states,avg max heap size")
    for res in result:
        for item in res:
            sys.stdout.write(str(item) + ",")
        sys.stdout.write("\n")
