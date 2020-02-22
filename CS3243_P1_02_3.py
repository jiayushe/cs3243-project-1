import os
import sys
from heapq import heappush, heappop, heapify
import time

## GLOBAL CONSTANTS ##

left = "LEFT"
right = "RIGHT"
down = "DOWN"
up = "UP"

######################

## helper class ##

# simple min heap
class PriorityQueue:

    def __init__(self):
        self.queue = []

    def add(self, item):
        heappush(self.queue, item)

    def poll(self):
        return heappop(self.queue)

    def peek(self):
        return self.queue[0]

    def remove(self, item):
        value = self.queue.remove(item)
        heapify(self.queue)
        return value != None

    def is_empty(self):
        return self.queue == []

    def __len__(self):
        return len(self.queue)

'''
Node:

Help store info about state

compared by f value.

'''
class Node:
    
    def __init__(self, fval, state, parent, move, depth, blank_pos):
        self.fval = fval
        self.parent = parent
        self.move = move
        self.state = state
        self.tree_depth = depth
        self.blank = blank_pos

class Puzzle(object):
    def __init__(self, init_state, goal_state):
        self.grid_size = len(init_state)
        self.init_state = self.list_to_tuple(init_state)
        self.goal_state = self.list_to_tuple(goal_state)
        self.goal = None
        self.pq = PriorityQueue()
        self.seen_states = set()
        self.goal_position = {} # a map from number to its goal position
        for i in range(len(self.goal_state)):
            self.goal_position[self.goal_state[i]] = i

    ## Helper Functions ##

    def list_to_tuple(self, lst):
        return tuple([elem for t in lst for elem in t])
        
    def pretty_print(self, state): 
        formatted_output = ""
        k = self.grid_size
        for i in range(k):
            formatted_output += "\n"
            for j in range(k):
                formatted_output += str(state[i*k + j])
                formatted_output += " "
        print(formatted_output)

    def find_blank(self): # 1D array
        for i in range(self.grid_size**2):
            if self.init_state[i] == 0:
                return i
    
    
    def hash(self, state):
        return "".join(map(str,state))

    #########################
    
    '''
        Test solvability of puzzle:
        Idea adapted from https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html
    
    '''
    def is_solvable(self, state, blank):
        inversions = self.count_inversions(state)
        if self.grid_size % 2 == 1:  # grid size odd
            if inversions % 2 == 1:
                return False
        else:
            blank_pos = blank
            blank_row = blank_pos // self.grid_size
            row_from_bottom = self.grid_size - blank_row
            if row_from_bottom % 2 == 0:
                if inversions % 2 == 0:
                    return False
            else:     
                if inversions % 2 == 1:
                    return False
        return True

    def count_inversions(self, state): 
        n = len(state)
        inversions = 0
        for i in range(n):
            if state[i] == 0:
                continue
            for j in range(i + 1, n):
                if state[j] == 0:
                    continue
                if (state[i] > state[j]): 
                    inversions += 1
        return inversions
        
    def solve(self):
        k = self.grid_size
        upper_bound = k**4
        blank_position = self.find_blank()
        if not self.is_solvable(self.init_state, blank_position):
            return ["UNSOLVABLE"]
        heap = []
        moves = [left,right,down,up]
        curr_f_val = self.heuristic(self.init_state)
        
        root = Node(curr_f_val, self.init_state, None, None, 0, blank_position)
        heappush(heap,(curr_f_val,0,root))
        while True:
            curr_node = heappop(heap)[2]
            while curr_node.tree_depth > upper_bound:
                curr_node = heappop(heap)[2]
            self.seen_states.add(hash(curr_node.state))
            if curr_node.state == self.goal_state:
                self.goal = curr_node
                break
            for move in moves:
                new_node = self.make_move(move,curr_node)
                if new_node is not None:
                    if hash(new_node.state) not in self.seen_states:
                        heappush(heap,(new_node.fval,new_node.tree_depth, new_node))
                        self.seen_states.add(hash(new_node.state))
                        
        solution = []
        
        # travel from goal node to initial, recording moves.
        while curr_node.parent != None:
            solution.insert(0,curr_node.move)
            curr_node = curr_node.parent  
        return solution

    def make_move(self, move, curr_node):
        blank_position = curr_node.blank
        k = self.grid_size
        new_state = list(curr_node.state)  
        blank_column = blank_position % k 
        blank_row = blank_position // k
        g_n = curr_node.tree_depth + 1
        
        if move == left:
            # blank switches place with number to it's right, can't be last col
            if blank_column < k-1:
                new_state[blank_position + 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position + 1]
                func_value = self.heuristic(new_state) + g_n
                new_blank_position = blank_position+1
            else:
                return None

        elif move == right:
            # blank switches place with number to it's left, can't be first col
            if blank_column > 0:
                new_state[blank_position - 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position - 1]
                func_value = self.heuristic(new_state) + g_n
                new_blank_position = blank_position-1
            else:
                return None

        elif move == down:
            # blank switches place with number above it, can't be first row
            if blank_row > 0:
                new_state[(blank_row-1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row-1) * k + blank_column]
                func_value = self.heuristic(new_state) + g_n
                new_blank_position = (blank_row-1) * k + blank_column
            else:
                return None

        else:
            # blank switches place with number below it, can't be last row
            if blank_row < k-1:
                new_state[(blank_row+1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row+1) * k + blank_column]
                func_value = self.heuristic(new_state) + g_n
                new_blank_position = (blank_row+1) * k + blank_column
            else:
                return None

        new_node = Node(func_value, tuple(new_state), curr_node, move, g_n, new_blank_position)
        return new_node

    ## Heuristic ##

    '''
        Made as a seperate fn so many heuristics can be tried

        Manhattan distance measures the number of moves each tile is away from
        its correct place on the board
    '''

    def heuristic(self, state):
    
        return self.manhattan_distance(state)

    def manhattan_distance(self, state): 
        k = self.grid_size
        total_tile_distance = 0
        for row in range(k): # row 
            for col in range(k):  # column
                curr_tile = state[row*k+col]
                if curr_tile == 0:
                    continue
                if curr_tile%k == 0:
                    # account for list indexing, usually should be one row above
                    correct_row = curr_tile//k - 1
                    # since k-multiple always on last col
                    correct_column = k-1 
                else:
                    correct_row = curr_tile//k
                    # account for mod k vs list indexing 
                    correct_column = curr_tile%k  - 1 
                
                num_tiles_away = abs(correct_row - row) + abs(correct_column - col)
                total_tile_distance += num_tiles_away
                
        return total_tile_distance

    def linear_conflict(self, state):
        count = 0
        for row in range(self.grid_size):
            for k in range(self.grid_size):
                if state[row*self.grid_size + k] == 0:
                    continue
                for j in range(k+1, self.grid_size):
                    if state[row*self.grid_size + j] == 0:
                        continue
                    # now t_j is guaranteed to be on the same line, right of t_k
                    goal_pos_j = self.goal_position[state[row*self.grid_size + j]]
                    goal_pos_k = self.goal_position[state[row*self.grid_size + k]]
                    if (goal_pos_j // self.grid_size == goal_pos_k // self.grid_size) and (goal_pos_j % self.grid_size < goal_pos_k % self.grid_size):
                        count += 1
        return count * 2 + self.manhattan_distance(state)

    ## Verifying correctness of solution ##
    
    def make_move_simple(self, move, state, blank_position):
        k = self.grid_size
        new_state = list(state)  
        blank_column = blank_position % k 
        blank_row = blank_position // k
        
        if move == left:  
            # blank switches place with number to it's right
            new_state[blank_position + 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position + 1]
            
            new_blank_position = blank_position+1

        elif move == right: 
            # blank switches place with number to it's left 
            new_state[blank_position - 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position - 1]
           
            new_blank_position = blank_position-1

        elif move == down: 
            # blank switches place with number above it
            new_state[(blank_row-1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row-1) * k + blank_column]
           
            new_blank_position = (blank_row-1) * k + blank_column

        else:   
            # blank switches place with number below it
            new_state[(blank_row+1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row+1) * k + blank_column]
            
            new_blank_position = (blank_row+1) * k + blank_column

        return (new_blank_position,new_state)

    def verify_solution(self, moves):
        if self.goal != None:
            
            curr_state = self.init_state
            blank_pos = self.find_blank()
            for move in moves:
                self.pretty_print(curr_state)
                blank_pos, curr_state = self.make_move_simple(move, curr_state,blank_pos)

            self.pretty_print(curr_state)

#n = 3 cases
#init_state = [[2,3,6],[1,5,8],[4,7,0]]
#init_state = [[1,2,3],[4,5,6],[8,7,0]] # unsolvable 
#init_state = [[1,8,3],[5,2,4],[0,7,6]]
#init_state = [[1,2,3],[4,5,6],[7,0,8]]
#init_state = [[8,6,7],[2,5,4],[3,0,1]]
#goal_state = [[1,2,3],[4,5,6],[7,8,0]]

# n = 4 cases

#init_state = [[1,2,3,4],[5,6,7,8],[10,11,0,12],[9,13,15,14]]  # unsolvable
#init_state = [[12,15,6,10],[4,9,5,8],[14,13,0,2],[1,7,11,3]]

#init_state = [[13,5,3,4],[2,1,8,0],[9,15,10,11],[14,12,6,7]]
#init_state = [[9,5,12,4],[0,1,3,10],[14,13,11,2],[15,7,6,8]]
#goal_state = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]


# n = 5 cases

#init_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,0,14,15],[16,17,13,20,19],[21,22,23,18,24]]
#init_state = [[1,3,4,10,5],[7,2,8,0,14],[6,11,12,9,15],[16,17,13,18,19],[21,22,23,24,20]]

#init_state = [[1,3,4,0,10],[7,2,12,8,5],[6,11,13,15,14],[17,23,18,9,19],[16,21,22,24,20]]
#init_state = [[1,3,4,10,5],[7,2,12,8,14],[6,11,13,15,0],[17,23,18,9,19],[16,21,22,24,20]]
#init_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,14,0,15],[16,17,13,18,19],[21,22,23,24,20]]
#init_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,0,14,15],[16,17,13,18,19],[21,22,23,20,24]]   # unsolvable 

#goal_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,0]]


t0 = time.time()

puzzle = Puzzle(init_state, goal_state)
ans = puzzle.solve()
t1 = time.time()
puzzle.verify_solution(ans)
print ans
print len(ans)
print t1-t0

'''
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

'''
