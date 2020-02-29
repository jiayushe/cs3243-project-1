import os
import sys
import time
import random

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
    def __init__(self, init_state, goal_state, heuristic="linear_conflict"):
        # you may add more attributes if you think is useful
        self.init_state = self.list_to_tuple(init_state)
        self.goal_state = self.list_to_tuple(goal_state)
        self.N = len(init_state)
        self.goal_node = None
        self.goal_position = [0] * (len(self.init_state)) # a map from number to its goal position
        for i in range(len(self.goal_state)):
            self.goal_position[self.goal_state[i]] = i
        # BEGIN experiment
        self.state_explored_count = 0
        self.node_generated_count = 0
        self.max_heap_size = 0
        self.max_depth = 0
        self.heuristic = heuristic
        # END experiment

    def solve(self):
        if self.is_solvable() == False:
            return {"result": ["UNSOLVABLE"]}
        self.AStar()
        res = self.backtrace()
        if not self.verify_solution(res):
            raise Exception("Incorrect solution with initial state " + self.init_state)
        return {"result": res, "solution_depth": self.goal_node.cost, "max_search_depth": self.max_depth, "state_explored": self.state_explored_count, "node_generated": self.node_generated_count, "max_heap_size": self.max_heap_size}

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

    def h(self, state):
        if self.heuristic == "linear_conflict":
            return self.linear_conflict(state)
        elif self.heuristic == "manhattan_distance":
            return self.manhattan_distance(state)
        elif self.heuristic == "misplaced_tile":
            return self.misplaced_tile(state)

    def AStar(self):
        explored = set()
        heap = list()
        frontier_cost = dict()

        key = self.h(self.init_state)
        root = Node(self.init_state, None, None, 0, key)
        entry = (key, root)
        heappush(heap, entry)
        frontier_cost[hash(root.state)] = root.cost

        while heap:
            self.max_heap_size = max(self.max_heap_size, len(heap))
            heap_node = heappop(heap)
            if hash(heap_node[1].state) in frontier_cost and frontier_cost[hash(heap_node[1].state)] < heap_node[1].cost:
                # lazy deletion: if the popped node has a worse cost than recorded, it's a replaced node, ignore it
                continue
            explored.add(hash(heap_node[1].state)) # add ONLY explored node to the explored set

            if heap_node[1].state == self.goal_state: # run goal test ONLY on explored node
                self.goal_node = heap_node[1]
                self.state_explored_count = len(explored)
                return heap

            neighbors = self.expand(heap_node[1])
            self.node_generated_count += len(neighbors)

            for neighbor in neighbors:
                neighbor.key = neighbor.cost + self.h(neighbor.state)
                entry = (neighbor.key, neighbor)
                hashed_neighbor_state = hash(neighbor.state)
                if hashed_neighbor_state not in explored and hashed_neighbor_state not in frontier_cost:
                    # if the child isn't in explored set or the frontier, add it to the frontier
                    if self.max_depth < neighbor.cost:
                        self.max_depth = neighbor.cost
                    heappush(heap, entry)
                    frontier_cost[hashed_neighbor_state] = neighbor.cost
                elif hashed_neighbor_state in frontier_cost and frontier_cost[hashed_neighbor_state] > neighbor.cost:
                    # if the child state is already in the frontier, replace the node in frontier if the cost is bettre
                    heappush(heap, entry)
                    frontier_cost[hashed_neighbor_state] = neighbor.cost
    
    def BFS(self):
        explored = set()
        frontier = deque([Node(self.init_state, None, None)])

        while frontier:
            node = frontier.popleft()
            explored.add(node.state)

            neighbors = self.expand(node)
            for neighbor in neighbors:
                if neighbor.state not in explored:
                    if neighbor.state == self.goal_state:
                        self.goal_node = node
                        return frontier
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

    # BEGIN experiment
    def move_to_number(self, move):
        if move == "LEFT":
            return 1
        elif move == "RIGHT":
            return 2
        elif move == "UP":
            return 3
        else:
            return 4

    def verify_solution(self, moves):
        if self.goal_node != None:
            curr_state = self.init_state
            for move in moves:
                curr_state = self.move(curr_state, self.move_to_number(move))
            return tuple(curr_state) == self.goal_state
        return True # return true if it was unsolvable
    # END experiment


public_test_3 = [[[1, 2, 3], [4, 5, 6], [8, 7, 0]],
                [[1, 8, 3], [5, 2, 4], [0, 7, 6]],
                [[8, 6, 7], [2, 5, 4], [3, 0, 1]]]
public_test_4 = [[[1, 2, 3, 4], [5, 6, 7, 8], [10, 11, 0, 12], [9, 13, 15, 14]],
                [[12, 15, 6, 10], [4, 9, 5, 8], [14, 13, 0, 2], [1, 7, 11, 3]],
                [[13, 5, 3, 4], [2, 1, 8, 0], [9, 15, 10, 11], [14, 12, 6, 7]],
                [[9, 5, 12, 4], [0, 1, 3, 10], [14, 13, 11, 2], [15, 7, 6, 8]]]
public_test_5 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 0, 14, 15], [16, 17, 13, 20, 19], [21, 22, 23, 18, 24]],
                [[1, 3, 4, 10, 5], [7, 2, 8, 0, 14], [6, 11, 12, 9, 15], [16, 17, 13, 18, 19], [21, 22, 23, 24, 20]],
                [[1, 3, 4, 0, 10], [7, 2, 12, 8, 5], [6, 11, 13, 15, 14], [17, 23, 18, 9, 19], [16, 21, 22, 24, 20]],
                [[1, 3, 4, 10, 5], [7, 2, 12, 8, 14], [6, 11, 13, 15, 0], [17, 23, 18, 9, 19], [16, 21, 22, 24, 20]],
                [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 14, 0, 15], [16, 17, 13, 18, 19], [21, 22, 23, 24, 20]],
                [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 0, 14, 15], [16, 17, 13, 18, 19], [21, 22, 23, 20, 24]]]


def print_result(d):
    print("Steps: %d; Solution depth: %d; Max search depth: %d; State explored: %d; Node generated: %d; Max heap size: %d" % (len(d["result"]), d["solution_depth"], d["max_search_depth"], d["state_explored"], d["node_generated"], d["max_heap_size"]))

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

def generate_puzzle(n, step):
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
    for i in range(step):
        r = random.randrange(4)+1
        action = number_to_move(r)
        while not is_valid_action(n, cur_state, action):
            r = random.randrange(4)+1
            action = number_to_move(r)
        cur_state = move(n, cur_state, r)
    return cur_state

if __name__ == "__main__":

    f = open("./experiment_result.txt", "w")

    # Run public tests
    public_test_3_result = []
    public_test_4_result = []
    public_test_5_result = []

    print("Running public tests on n=3")
    print("----------------------------")
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    count = 0
    for init_state in public_test_3:
        count = count + 1
        puzzle = Puzzle(init_state, goal_state)
        start_time = time.time()
        res = puzzle.solve()
        if res["result"] != ["UNSOLVABLE"]:
            print_result(res)
        else:
            print("UNSOLVABLE")
        time_taken = time.time() - start_time
        print("--- %s seconds ---" % (time_taken))
        res["time"] = time_taken
        public_test_3_result.append(res)
    print("")

    print("Running public tests on n=4")
    print("----------------------------")
    goal_state = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
    count = 0
    for init_state in public_test_4:
        count = count + 1
        puzzle = Puzzle(init_state, goal_state)
        start_time = time.time()
        res = puzzle.solve()
        if res["result"] != ["UNSOLVABLE"]:
            print_result(res)
        else:
            print("UNSOLVABLE")
        time_taken = time.time() - start_time
        print("--- %s seconds ---" % (time_taken))
        res["time"] = time_taken
        public_test_4_result.append(res)
    print("")

    print("Running public tests on n=5")
    print("----------------------------")
    goal_state = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 0]]
    count = 0
    for init_state in public_test_5:
        count = count + 1
        puzzle = Puzzle(init_state, goal_state)
        start_time = time.time()
        res = puzzle.solve()
        if res["result"] != ["UNSOLVABLE"]:
            print_result(res)
        else:
            print("UNSOLVABLE")
        time_taken = time.time() - start_time
        print("--- %s seconds ---" % (time_taken))
        res["time"] = time_taken
        public_test_5_result.append(res)
    print("")
    f.write("public test cases\n")
    f.write("n=3\n")
    f.write(str(public_test_3_result)+"\n")
    f.write("n=4\n")
    f.write(str(public_test_4_result)+"\n")
    f.write("n=5\n")
    f.write(str(public_test_5_result)+"\n")

    print(generate_puzzle(5, 100))