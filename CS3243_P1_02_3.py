import os
import sys
from heapq import heappush, heappop, heapify
import copy
import math

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

    def __cmp__(self, obj):
        if not isinstance(obj, Node):
            raise Exception("Node compared to a non-node object!")
        if self.fval > obj.fval:
            return 1
        elif self.fval < obj.fval:
            return -1
        else:
            if self.tree_depth < obj.tree_depth:
                return -1
            elif self.tree_depth > obj.tree_depth:
                return 1
            else:
                return 0
    def __ne__(self, obj):
        if obj == None:
            return True
        if not isinstance(obj, Node):
            raise Exception("Node compared to a non-node object!")
        return self.state != obj.state

class Puzzle(object):
    def __init__(self, init_state, goal_state):
        self.grid_size = len(init_state)
        self.init_state = self.list_to_tuple(init_state)
        self.goal_state = self.list_to_tuple(goal_state)
        self.goal = None

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

    #########################
    
    '''
        Test solvability of puzzle:
        Idea adapted from https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html
    
    '''
    def solvable(self, state, blank):
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
        
    # todo optimise to pass n=4 case 2 and 3 in < 5s
    def solve(self):
        
        blank_position = self.find_blank()
        if not self.solvable(self.init_state, blank_position):
            return ["UNSOLVABLE"]
        
        moves = [left,right,down,up]
        pq = PriorityQueue()
        seen_states = set()
        seen_states.add(self.init_state)
        curr_f_val = self.heuristic(self.init_state)
        curr_node = Node(curr_f_val, self.init_state, None, None, 0, blank_position)
        
        while True:        
            for move in moves:
                if self.is_valid_move(move, curr_node.blank):
                    new_node = self.make_move(move, curr_node)
                
                    if new_node.state not in seen_states:
                        pq.add(new_node)
                        seen_states.add(new_node.state)
                        
            # choose state with lowest f value
            curr_node = pq.poll()  
            if curr_node.fval - curr_node.tree_depth == 0:
                # goal reached, since heuristic is 0
                self.goal = curr_node
                break
            
        solution = []
        # travel from goal node to initial, recording moves.
        while curr_node.parent != None:
            solution.insert(0,curr_node.move)
            curr_node = curr_node.parent  
        return solution

    #todo merge validity with making move so no check needed 
    def is_valid_move(self, move, blank_position): # ensure move generates new state
        
        k = self.grid_size
        blank_column = blank_position % k 
        blank_row = blank_position // k
        
        if move == left:
            if blank_column == k-1:
                # blank on rightmost column
                return False
        elif move == right:
            if blank_column == 0:
                # blank on leftmost column
                return False 
        elif move == up:
            if blank_row == k-1:
                # blank on top row
                return False  
        else:
            if blank_row == 0:
                # blank on bottom row
                return False  
        return True

    def make_move(self, move, curr_node):
        blank_position = curr_node.blank
        k = self.grid_size
        new_state = list(curr_node.state)  
        blank_column = blank_position % k 
        blank_row = blank_position // k
        g_n = curr_node.tree_depth + 1
        
        if move == left:
            # blank switches place with number to it's right 
            
            new_state[blank_position + 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position + 1]
            func_value = self.heuristic(new_state) + g_n
            new_blank_position = blank_position+1

        elif move == right:
            # blank switches place with number to it's left 
            
            new_state[blank_position - 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position - 1]
            func_value = self.heuristic(new_state) + g_n
            new_blank_position = blank_position-1

        elif move == down:
            # blank switches place with number above it
            
            new_state[(blank_row-1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row-1) * k + blank_column]
            func_value = self.heuristic(new_state) + g_n
            new_blank_position = (blank_row-1) * k + blank_column

        else:
            # blank switches place with number below it
            
            new_state[(blank_row+1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row+1) * k + blank_column]
            func_value = self.heuristic(new_state) + g_n
            new_blank_position = (blank_row+1) * k + blank_column

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

'''
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
#init_state = [[14,10,5,13],[11,8,1,3],[2,9,12,6],[15,4,0,7]]
#goal_state = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]


# n = 5 cases

#init_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,0,14,15],[16,17,13,20,19],[21,22,23,18,24]]
#init_state = [[1,3,4,10,5],[7,2,8,0,14],[6,11,12,9,15],[16,17,13,18,19],[21,22,23,24,20]]

#init_state = [[1,3,4,0,10],[7,2,12,8,5],[6,11,13,15,14],[17,23,18,9,19],[16,21,22,24,20]]
#init_state = [[1,3,4,10,5],[7,2,12,8,14],[6,11,13,15,0],[17,23,18,9,19],[16,21,22,24,20]]
init_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,14,0,15],[16,17,13,18,19],[21,22,23,24,20]]
#init_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,0,14,15],[16,17,13,18,19],[21,22,23,20,24]]   # unsolvable 

goal_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,0]]


puzzle = Puzzle(init_state, goal_state)
ans = puzzle.solve()
puzzle.verify_solution(ans)
print ans

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
