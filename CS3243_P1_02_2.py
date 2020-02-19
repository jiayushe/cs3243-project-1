import os
import sys
from heapq import *

class Puzzle(object):
    def __init__(self, init_state, goal_state):
        # you may add more attributes if you think is useful
        self.init_state = init_state
        self.goal_state = goal_state
        self.n = len(goal_state)
        self.actions = list()
        self.closedSet = set([])
        self.pq = []

    def solve(self):
        zeroI, zeroJ = -1, -1

        state = self.toList(init_state)
        self.goal_state = self.toList(goal_state)
        # Find starting '0'
        for i in range(self.n):
            for j in range(self.n):
                if (state[i * self.n + j] == 0):
                    zeroI, zeroJ = i, j
            if zeroI >= 0:
                break
        state = State(state, 0, zeroI, zeroJ, self.n, [])
        initial = (state.generateHeuristicScore(), state)
        heappush(self.pq, initial)
        while len(self.pq) > 0:
            state = heappop(self.pq)
            state = state[1]
            self.closedSet.add(tuple(state.getState()))
            for i in range(4):
                currState = state.generateNewState(i)
                if currState == [] or tuple(currState.getState()) in self.closedSet:
                    continue
                if currState.getState() == self.goal_state:
                    return currState.getSteps()
                stateTuple = (currState.generateHeuristicScore(), currState)
                heappush(self.pq, stateTuple)

        return ["UNSOLVABLE"]


    def toList(self, array_2d):
        state = []
        for i in range(self.n):
            for j in range(self.n):
                state.append(array_2d[i][j])
        return state

class State(object):
    def __init__(self, state, gx, i, j, n, steps):
        self.state = state
        self.gx = gx
        self.zeroI = i
        self.zeroJ = j
        self.n = n
        self.steps = steps
    
    def generateNewState(self, action):
        if (action == 0): # Left
            if (self.zeroJ == n - 1):
                return []
            newState = list(self.state)
            newState[self.zeroI * n + self.zeroJ] = self.state[self.zeroI * n + self.zeroJ + 1]
            newState[self.zeroI * n + self.zeroJ + 1] = 0
            newJ, newI = self.zeroJ + 1, self.zeroI
            newSteps = list(self.steps)
            newSteps.append("LEFT")
        elif (action == 1): # Right
            if (self.zeroJ == 0):
                return []
            newState = list(self.state)
            newState[self.zeroI * n + self.zeroJ] = self.state[self.zeroI * n + self.zeroJ - 1]
            newState[self.zeroI * n + self.zeroJ - 1] = 0
            newJ, newI = self.zeroJ - 1, self.zeroI
            newSteps = list(self.steps)
            newSteps.append("RIGHT")
        elif (action == 2): # Up
            if (self.zeroI == n - 1):
                return []
            newState = list(self.state)
            newState[self.zeroI * n + self.zeroJ] = self.state[self.zeroI * n + self.zeroJ + n]
            newState[self.zeroI * n + self.zeroJ + n] = 0
            newI, newJ = self.zeroI + 1, self.zeroJ
            newSteps = list(self.steps)
            newSteps.append("UP")
        elif (action == 3): # Down
            if (self.zeroI == 0):
                return []
            newState = list(self.state)
            newState[self.zeroI * n + self.zeroJ] = self.state[self.zeroI * n + self.zeroJ - n]
            newState[self.zeroI * n + self.zeroJ - n] = 0
            newI, newJ = self.zeroI - 1, self.zeroJ
            newSteps = list(self.steps)
            newSteps.append("DOWN")

        result = State(newState, self.gx + 1, newI, newJ, self.n, newSteps)
        return result

    def __gt__(self, state2):
        return self.gx > state2.gx

    def generateHeuristicScore(self):
        return self.getLinearConflict() + self.gx

    def getManhattanDistance(self):
        count = 0
        for i in range(self.n):
            for j in range(self.n):
                num = self.state[i * self.n + j]
                idealI = num // self.n
                idealJ = num % self.n - 1
                count += (abs(idealI - i) + abs(idealJ - j))
        return count
    
    def getLinearConflict(self):
        conflicts = 0
        for i in range(self.n):
            for j in range(self.n):
                index = i * self.n + j
                num = self.state[index]
                if num == index + 1: # If number in correct position
                    continue
                idealI = num // self.n
                idealJ = num % self.n - 1
                if idealI == i: # If correct row
                    for fromCol in range(j + 1, self.n):
                        currNum = self.state[i * self.n + fromCol]
                        if num > currNum and currNum >= i * n:
                            conflicts += 1
                elif idealJ == j: # If correct column    
                    for fromRow in range(i + 1, self.n):
                        currNum = self.state[fromRow * self.n + j]
                        correctJ = currNum % n - 1 # Check if goal is in same column
                        if num > currNum and correctJ == j:
                            conflicts += 1                    

        return self.getManhattanDistance() + 2 * conflicts

    def getState(self):
        return self.state

    def getSteps(self):
        return self.steps

    # you may add more functions if you think is useful

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
