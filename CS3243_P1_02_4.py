import os
import sys

from collections import deque
from heapq import heappush, heappop

class Node(object):
    def __init__(self, state, parent, move, cost, key):
        self.state = state
        self.parent = parent
        self.move = move
        self.cost = cost
        self.key = key

class Puzzle(object):
    def __init__(self, init_state, goal_state):
        self.init_state = self.list_to_tuple(init_state)
        self.goal_state = self.list_to_tuple(goal_state)
        self.N = len(init_state)
        self.goal_node = None
        self.goal_position = [0] * (len(self.init_state)) # a map from number to its goal position
        for i in range(len(self.goal_state)):
            self.goal_position[self.goal_state[i]] = i
        # BEGIN profiling
        self.state_explored_count = 0
        self.state_visited_count = 0
        self.frontier_size = 0
        self.max_depth = 0
        # END profiling

    def solve(self):
        if self.is_solvable() == False:
            sys.stderr.write("Unsolvable\n")
            sys.stderr.flush()
            return ["UNSOLVABLE"]
        self.AStar()
        res = self.backtrace()
        sys.stderr.write("Solution Depth: " + str(self.goal_node.cost) + "\n")
        sys.stderr.write("Max Search Depth: " + str(self.max_depth) + "\n")
        sys.stderr.write("State Explored: " + str(self.state_explored_count) + "\n")
        sys.stderr.write("State Generated: " + str(self.state_visited_count) + "\n")
        sys.stderr.write("Frontier Size: " + str(self.frontier_size) + "\n")
        sys.stderr.flush()
        return res

    def is_solvable(self):
        state = self.init_state
        n = len(state)
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
        if self.N % 2 == 1:
            return inversions % 2 == 0
        else:
            row_from_bottom = self.N - blank // self.N - 1
            return row_from_bottom % 2 == inversions % 2

    def hash(self, state):
        res = ""
        for i in state:
            if i < 10:
                res += "0"
            res += str(i)
        return res

    # manhattan distance
    def manhattan_distance(self, state):
        count = 0;
        for i in range(len(state)):
            if state[i] == 0:
                continue
            goal_X, goal_Y = self.goal_position[state[i]] // self.N, self.goal_position[state[i]] % self.N
            X, Y = i // self.N, i % self.N
            count += abs(goal_X-X) + abs(goal_Y-Y)
        return count

    # linear conflict
    def linear_conflict(self, state):
        count = 0
        for row in range(self.N): # for each row r_i
            lc = 0
            C = [[] for i in range(self.N)]
            for k in range(self.N): # for each tile t_k in row r_i, calculate C
                if state[row*self.N + k] == 0:
                    continue
                for j in range(k+1, self.N):
                    if state[row*self.N + j] == 0:
                        continue
                    # now t_j is guaranteed to be on the same line, right of t_k
                    goal_pos_j = self.goal_position[state[row*self.N + j]]
                    goal_pos_k = self.goal_position[state[row*self.N + k]]
                    if (goal_pos_j // self.N == row) and (goal_pos_j // self.N == goal_pos_k // self.N) and (goal_pos_j % self.N < goal_pos_k % self.N):
                        C[k].append(j)
                        C[j].append(k)
            while not all(len(v)==0 for v in C):
                lens = [len(v) for v in C]
                j = lens.index(max(lens))
                C[j] = []
                for k in range(self.N):
                    if state[row*self.N + k] == 0:
                        continue
                    if j in C[k]:
                        C[k].remove(j)
                lc += 1
            count += lc

        for col in range(self.N):
            lc = 0
            C = [[] for i in range(self.N)]
            for k in range(self.N):
                if state[k*self.N + col] == 0:
                    continue
                for j in range(k+1, self.N):
                    if state[j*self.N + col] == 0:
                        continue
                    # now t_j is guaranteed to be on the same line, bottom of t_k
                    goal_pos_j = self.goal_position[state[j*self.N + col]]
                    goal_pos_k = self.goal_position[state[k*self.N + col]]
                    if (goal_pos_j % self.N == col) and (goal_pos_j % self.N == goal_pos_k % self.N) and (goal_pos_j // self.N < goal_pos_k // self.N):
                        C[k].append(j)
                        C[j].append(k)
            while not all(len(v)==0 for v in C):
                lens = [len(v) for v in C]
                j = lens.index(max(lens))
                C[j] = []
                for k in range(self.N):
                    if state[k*self.N + col] == 0:
                        continue
                    if j in C[k]:
                        C[k].remove(j)
                lc += 1
            count += lc
        return count * 2 + self.manhattan_distance(state)

    def heuristic(self, state):
        return self.linear_conflict(state)

    def AStar(self):
        explored = set()
        heap = list()
        frontier_cost = dict()

        key = self.heuristic(self.init_state)
        root = Node(self.init_state, None, None, 0, key)
        entry = (key, root)
        self.state_visited_count += 1
        heappush(heap, entry)
        frontier_cost[hash(root.state)] = root.cost

        while heap:
            self.frontier_size = max(self.frontier_size, len(heap))
            heap_node = heappop(heap)
            if hash(heap_node[1].state) in frontier_cost and frontier_cost[hash(heap_node[1].state)] < heap_node[1].cost:
                # lazy deletion: if the popped node has a worse cost than recorded, it's a replaced node, ignore it
                continue
            self.state_explored_count += 1
            explored.add(hash(heap_node[1].state)) # add ONLY explored node to the explored set

            if heap_node[1].state == self.goal_state: # run goal test ONLY on explored node
                self.goal_node = heap_node[1]
                return

            neighbors = self.expand(heap_node[1])

            for neighbor in neighbors:
                neighbor.key = neighbor.cost + self.heuristic(neighbor.state)
                entry = (neighbor.key, neighbor)
                hashed_neighbor_state = hash(neighbor.state)
                if hashed_neighbor_state not in explored and hashed_neighbor_state not in frontier_cost:
                    # if the child isn't in explored set or the frontier, add it to the frontier
                    self.state_visited_count += 1
                    if self.max_depth < neighbor.cost:
                        self.max_depth = neighbor.cost
                    heappush(heap, entry)
                    frontier_cost[hashed_neighbor_state] = neighbor.cost
                elif hashed_neighbor_state in frontier_cost and frontier_cost[hashed_neighbor_state] > neighbor.cost:
                    # if the child state is already in the frontier, replace the node in frontier if the cost is bettre
                    heappush(heap, entry)
                    frontier_cost[hashed_neighbor_state] = neighbor.cost

    def expand(self, node):
        neighbors = list()
        neighbors.append(Node(self.move(node.state, 1), node, 1, node.cost+1, 0))
        neighbors.append(Node(self.move(node.state, 2), node, 2, node.cost+1, 0))
        neighbors.append(Node(self.move(node.state, 3), node, 3, node.cost+1, 0))
        neighbors.append(Node(self.move(node.state, 4), node, 4, node.cost+1, 0))
        return [neighbor for neighbor in neighbors if neighbor.state]
    
    def backtrace(self):
        current_node = self.goal_node
        if not current_node:
            return ["UNSOLVABLE"]

        moves = []
        while current_node.state != self.init_state:
            if current_node.move == 1:
                moves.append("LEFT")
            elif current_node.move == 2:
                moves.append("RIGHT")
            elif current_node.move == 3:
                moves.append("UP")
            elif current_node.move == 4:
                moves.append("DOWN")
            else:
                raise Exception("Illegal action found in backtrace function: " + current_node.move)
            current_node = current_node.parent
        moves.reverse()
        return moves
    
    def move(self, state, action):
        new_state = list(state)
        index = new_state.index(0)
        zr = index / self.N
        zc = index % self.N
        if action == 1: # LEFT
            # s(zr,zc) = s(zr, zc+1), s(zr, zc+1) = 0
            if zc < self.N-1:
                new_state[index] = new_state[zr*self.N + zc + 1]
                new_state[zr*self.N + zc + 1] = 0
            else:
                return None
        elif action == 2: # RIGHT
            # s(zr,zc) = s(zr, zc-1), s(zr, zc-1) = 0
            if zc >= 1:
                new_state[index] = new_state[zr*self.N + zc - 1]
                new_state[zr*self.N + zc - 1] = 0
            else:
                return None
        elif action == 3: # UP
            # s(zr,zc) = s(zr+1, zc), s(zr, zc+1) = 0
            if zr < self.N-1:
                new_state[index] = new_state[(zr+1)*self.N + zc]
                new_state[(zr+1)*self.N + zc] = 0
            else:
                return None
        elif action == 4: # DOWN
            # s(zr,zc) = s(zr-1, zc), s(zr, zc-1) = 0
            if zr >= 1:
                new_state[index] = new_state[(zr-1)*self.N + zc]
                new_state[(zr-1)*self.N + zc] = 0
            else:
                return None
        else:
            raise Exception("Illegal action found in move function: " + action)
        return tuple(new_state)

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
