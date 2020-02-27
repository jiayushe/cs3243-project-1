import os
import sys

from collections import deque
from heapq import heappush, heappop, heapify

class Node(object):
    def __init__(self, state, parent, move, cost, key):
        self.state = state
        self.parent = parent
        self.move = move
        self.cost = cost
        self.key = key

class Puzzle(object):
    def __init__(self, init_state, goal_state):
        # you may add more attributes if you think is useful
        self.init_state = self.list_to_tuple(init_state)
        self.goal_state = self.list_to_tuple(goal_state)
        self.N = len(init_state)
        self.goal_node = None
        self.goal_position = [0] * (len(self.init_state)) # a map from number to its goal position
        for i in range(len(self.goal_state)):
            self.goal_position[self.goal_state[i]] = i
        # BEGIN profiling
        self.state_visited_count = 0
        self.max_depth = 0
        # END profiling

    def solve(self):
        if self.is_solvable() == False:
            sys.stderr.write("Unsolvable\n")
            sys.stderr.flush()
            return ["UNSOLVABLE"]
        self.AStar()
        res = self.backtrace()
        sys.stderr.write("State Visited: " + str(self.state_visited_count) + "\n")
        sys.stderr.write("Max Depth: " + str(self.max_depth) + "\n")
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

    # manhattan
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
        for row in range(self.N):
            for k in range(self.N):
                if state[row*self.N + k] == 0:
                    continue
                for j in range(k+1, self.N):
                    if state[row*self.N + j] == 0:
                        continue
                    # now t_j is guaranteed to be on the same line, right of t_k
                    goal_pos_j = self.goal_position[state[row*self.N + j]]
                    goal_pos_k = self.goal_position[state[row*self.N + k]]
                    if (goal_pos_j // self.N == row) and (goal_pos_j // self.N == goal_pos_k // self.N) and (goal_pos_j % self.N < goal_pos_k % self.N):
                        count += 1
        return count * 2 + self.manhattan_distance(state)

    # misplaced tile count
    def misplaced_tile(self, state):
        count = 0
        for i in range(len(self.goal_state)):
            if self.goal_state[i] != state[i]:
                count += 1
        return count

    def AStar(self):
        explored = set()
        heap = list()

        key = self.linear_conflict(self.init_state)
        root = Node(self.init_state, None, None, 0, key)
        entry = (key, root)
        heappush(heap, entry)
        frontier_set = set([hash(root.state)])

        while heap:
            heap_node = heappop(heap)
            frontier_set.remove(hash(heap_node[1].state))
            explored.add(hash(heap_node[1].state))

            if heap_node[1].state == self.goal_state:
                self.goal_node = heap_node[1]
                return heap

            neighbors = self.expand(heap_node[1])

            for neighbor in neighbors:
                neighbor.key = neighbor.cost + self.linear_conflict(neighbor.state)
                entry = (neighbor.key, neighbor)
                if hash(neighbor.state) not in explored and hash(neighbor.state) not in frontier_set:
                    self.state_visited_count += 1
                    if self.max_depth < neighbor.cost:
                        self.max_depth = neighbor.cost
                    heappush(heap, entry)
                    frontier_set.add(hash(neighbor.state))
    
    def BFS(self):
        explored = set()
        frontier = deque([Node(self.init_state, None, None, 0, 0)])

        while frontier:
            node = frontier.popleft()
            explored.add(node.state)

            if node.state == self.goal_state:
                self.goal_node = node
                return frontier

            neighbors = self.expand(node)
            for neighbor in neighbors:
                if neighbor.state not in explored:
                    frontier.append(neighbor)
                    explored.add(neighbor.state)

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
