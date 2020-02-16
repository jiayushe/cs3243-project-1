import os
import sys

import time


class Puzzle(object):
    def __init__(self, init_state, goal_state):
        # you may add more attributes if you think is useful
        self.init_state = self.list_to_tuple(init_state)
        self.goal_state = self.list_to_tuple(goal_state)
        self.N = len(init_state)
        self.actions = list()
        self.visited_state = set()

    def solve(self):
        #TODO
        # implement your search algorithm here
        
        # return ["LEFT", "RIGHT"] # sample output 
        return self.BFS()
    
    def BFS(self):
        frontier = []
        frontier_state_set = set()
        frontier.append((self.init_state, []))
        frontier_state_set.add(self.init_state)
        count = 0
        while frontier:
            current = frontier.pop()
            frontier_state_set.remove(current[0])
            self.visited_state.add(current[0])

            if current[0] == self.goal_state:
                return current[1]
            
            expanded = self.next_step(current)
            for node in expanded:
                if not ( (node[0] in self.visited_state) or (node[0] in frontier_state_set) ):
                    frontier.append(node)
                    frontier_state_set.add(node[0])
        return ["UNSOLVABLE"]

    def beautify_print(self, state):
        print("---------------")
        for i in range(self.N):
            for j in range(self.N):
                print("%d" % (state[i*self.N+j])),
            print
        print("---------------")

    def next_step(self, node):
        current_state = node[0]
        current_actions = node[1]
        expanded = []
        stop = False
        for r in range(self.N):
            for c in range(self.N):
                if current_state[r * self.N + c] == 0:
                    if c >= 1: # RIGHT
                        self.expand(current_state, current_actions, r, c, "RIGHT", expanded)
                    if c < self.N - 1: # LEFT
                        self.expand(current_state, current_actions, r, c, "LEFT", expanded)
                    if r >= 1: # DOWN
                        self.expand(current_state, current_actions, r, c, "DOWN", expanded)
                    if r < self.N - 1: # UP
                        self.expand(current_state, current_actions, r, c, "UP", expanded)
                    stop = True
                    break
            if stop:
                break
        return expanded
    
    def move(self, state, zr, zc, action):
        new_state = list(state)
        if action == "LEFT":
            # s(zr,zc) = s(zr, zc+1), s(zr, zc+1) = 0
            new_state[zr*self.N + zc] = new_state[zr*self.N + zc + 1]
            new_state[zr*self.N + zc + 1] = 0
        elif action == "RIGHT":
            # s(zr,zc) = s(zr, zc-1), s(zr, zc-1) = 0
            new_state[zr*self.N + zc] = new_state[zr*self.N + zc - 1]
            new_state[zr*self.N + zc - 1] = 0
        elif action == "UP":
            # s(zr,zc) = s(zr+1, zc), s(zr, zc+1) = 0
            new_state[zr*self.N + zc] = new_state[(zr+1)*self.N + zc]
            new_state[(zr+1)*self.N + zc] = 0
        elif action == "DOWN":
            # s(zr,zc) = s(zr-1, zc), s(zr, zc-1) = 0
            new_state[zr*self.N + zc] = new_state[(zr-1)*self.N + zc]
            new_state[(zr-1)*self.N + zc] = 0
        else:
            raise Exception("Illegal action found in move function: " + action)
        return tuple(new_state)
            
    def expand(self, current_state, current_actions, r, c, action, expanded):
        new_state = self.move(current_state, r, c, action)
        new_action = list(current_actions)
        new_action.append(action)
        expanded.append((new_state, new_action))

    # you may add more functions if you think is useful

    # flatten the nested list to one-dimensional tuple
    def list_to_tuple(self, lst):
        return tuple([elem for t in lst for elem in t])

if __name__ == "__main__":
    # do NOT modify below

    # argv[0] represents the name of the file that is being executed
    # argv[1] represents name of input file
    # argv[2] represents name of destination output file
    if len(sys.argv) != 3:
        raise ValueError("Wrong number of arguments!")

    try:
        f = open(sys.argv[1], 'r')
    except IOError:
        raise IOError("Input file not found!")

    lines = f.readlines()
    
    # n = num rows in input file
    n = len(lines)
    # max_num = n to the power of 2 - 1
    max_num = n ** 2 - 1

    # Instantiate a 2D list of size n x n
    init_state = [[0 for i in range(n)] for j in range(n)]
    goal_state = [[0 for i in range(n)] for j in range(n)]
    

    i,j = 0, 0
    for line in lines:
        for number in line.split(" "):
            if number == '':
                continue
            value = int(number , base = 10)
            if  0 <= value <= max_num:
                init_state[i][j] = value
                j += 1
                if j == n:
                    i += 1
                    j = 0

    for i in range(1, max_num + 1):
        goal_state[(i-1)//n][(i-1)%n] = i
    goal_state[n - 1][n - 1] = 0

    puzzle = Puzzle(init_state, goal_state)
    ans = puzzle.solve()

    with open(sys.argv[2], 'a') as f:
        for answer in ans:
            f.write(answer+'\n')







