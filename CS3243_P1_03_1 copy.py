import os
import sys

from collections import deque
from heapq import heappush, heappop, heapify
import itertools

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

    def solve(self):
        self.AStar()
        return self.backtrace()

    def h(self, state):
        return sum(abs(b % self.N - g % self.N) + abs(b // self.N - g // self.N)
               for b, g in ((state.index(i), self.goal_state.index(i)) for i in range(1, self.N)))

    def AStar(self):
        explored = set()
        heap = list()
        heap_entry = {}
        counter =  itertools.count()

        key = self.h(self.init_state)
        root = Node(self.init_state, None, None, 0, key)
        entry = (key, 0, root)
        heappush(heap, entry)
        heap_entry[root.state] = entry

        while heap:
            heap_node = heappop(heap)
            explored.add(heap_node[2].state)

            if heap_node[2].state == self.goal_state:
                self.goal_node = heap_node[2]
                return heap

            neighbors = self.expand(heap_node[2])

            for neighbor in neighbors:
                neighbor.key = neighbor.cost + self.h(neighbor.state)
                entry = (neighbor.key, neighbor.move, neighbor)
                if neighbor.state not in explored:
                    heappush(heap, entry)
                    explored.add(neighbor.state)
                    heap_entry[neighbor.state] = entry
                elif neighbor.state in heap_entry and neighbor.key < heap_entry[neighbor.state][2].key:
                    hindex = heap.index((heap_entry[neighbor.state][2].key,
                                        heap_entry[neighbor.state][2].move,
                                        heap_entry[neighbor.state][2]))
                    heap[int(hindex)] = entry
                    heap_entry[neighbor.state] = entry
                    heapify(heap)

    
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
                raise Exception("Illegal action found in backtrace function: " + action)
            current_node = current_node.parent
        moves.reverse()
        return moves
    
    def move(self, state, action):
        new_state = list(state)
        index =new_state.index(0)
        zr = index / self.N
        zc = index % self.N
        if action == 1: # LEFT
            # s(zr,zc) = s(zr, zc+1), s(zr, zc+1) = 0
            if zc < self.N-1:
                new_state[zr*self.N + zc] = new_state[zr*self.N + zc + 1]
                new_state[zr*self.N + zc + 1] = 0
            else:
                return None
        elif action == 2: # RIGHT
            # s(zr,zc) = s(zr, zc-1), s(zr, zc-1) = 0
            if zc >= 1:
                new_state[zr*self.N + zc] = new_state[zr*self.N + zc - 1]
                new_state[zr*self.N + zc - 1] = 0
            else:
                return None
        elif action == 3: # UP
            # s(zr,zc) = s(zr+1, zc), s(zr, zc+1) = 0
            if zr < self.N-1:
                new_state[zr*self.N + zc] = new_state[(zr+1)*self.N + zc]
                new_state[(zr+1)*self.N + zc] = 0
            else:
                return None
        elif action == 4: # DOWN
            # s(zr,zc) = s(zr-1, zc), s(zr, zc-1) = 0
            if zr >= 1:
                new_state[zr*self.N + zc] = new_state[(zr-1)*self.N + zc]
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







