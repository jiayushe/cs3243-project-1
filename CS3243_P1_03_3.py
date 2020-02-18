import os
import sys
from heapq import heappush, heappop
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

class Puzzle(object):
    def __init__(self, init_state, goal_state):
        self.grid_size = len(init_state)
        self.init_state = self.list_to_tuple(init_state)
        self.goal_state = self.list_to_tuple(goal_state)

    def list_to_tuple(self, lst):
        return tuple([elem for t in lst for elem in t])
        
    ## debugging helper function O(K**2) ##
    ## works ## 
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

    def solve(self):
        
        moves = [left,right,down,up]
        k = self.grid_size
        blank_position = self.find_blank()
        solution = []
        pq = PriorityQueue()
        seen_states = set()
        curr_state = self.init_state
        curr_f_val = heuristic(curr_state)
        
        while True:
            
            if curr_state == self.goal_state:  # home free!
                break
            
            seen_states.add(curr_state) # detect loop
            
            for move in moves:
                
                if self.is_valid_move(move, blank_position):
                    
                    #state_copy = copy.deepcopy(curr_state)
    
                    func_value,new_state,new_moves,new_blank_position = self.make_move(curr_f_val,move, curr_state, blank_position,solution)
                    
                    if new_state not in seen_states:
                        pq.add((func_value,new_state,new_moves,new_blank_position))

            if pq.is_empty():
                return ["UNSOLVABLE"] 
            winning_move = pq.poll()
            curr_state = winning_move[1]  # make best move according to heuristic
            solution = winning_move[2]  # add move to solution set
            blank_position = winning_move[3]
            print(curr_f_val)
            curr_f_val = winning_move[0]
            self.pretty_print(curr_state)
                           
        return solution

    def is_valid_move(self, move, blank_position): # ensure move generates new state
        
        k = self.grid_size
        blank_column = blank_position % k 
        blank_row = blank_position // k
        
        if move == left:
            if blank_column == k-1:  # blank on rightmost column
                return False
        elif move == right:
            if blank_column == 0:
                return False # blank on leftmost column
        elif move == up:
            if blank_row == k-1:
                return False  # blank on top row
        else:
            if blank_row == 0:
                return False  # blank on bottom row
        return True

    def make_move(self, prev_f_val, move, state, blank_position,past_moves):
        k = self.grid_size
        new_state = list(state)  
        blank_column = blank_position % k 
        blank_row = blank_position // k
        
        if move == left: # blank switches place with number to it's right 
            
            new_state[blank_position + 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position + 1]
            func_value = heuristic(new_state) + prev_f_val + 1
            new_blank_position = blank_position+1

        elif move == right: # blank switches place with number to it's left 
            
            new_state[blank_position - 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position - 1]
            func_value = heuristic(new_state) + prev_f_val + 1
            new_blank_position = blank_position-1

        elif move == down: # blank switches place with number above it
            
            new_state[(blank_row-1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row-1) * k + blank_column]
            func_value = heuristic(new_state) + prev_f_val + 1
            new_blank_position = (blank_row-1) * k + blank_column

        else:   # blank switches place with number below it
            
            new_state[(blank_row+1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row+1) * k + blank_column]
            func_value = heuristic(new_state) + prev_f_val + 1
            new_blank_position = (blank_row+1) * k + blank_column

        new_moves = past_moves + [move,]
        return (func_value,tuple(new_state),new_moves,new_blank_position)

    def verify_move(self, move, state, blank_position):
        k = self.grid_size
        new_state = list(state)  
        blank_column = blank_position % k 
        blank_row = blank_position // k
        
        if move == left: # blank switches place with number to it's right 
            
            new_state[blank_position + 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position + 1]
            
            new_blank_position = blank_position+1

        elif move == right: # blank switches place with number to it's left 
            
            new_state[blank_position - 1], new_state[blank_position] = new_state[blank_position], new_state[blank_position - 1]
           
            new_blank_position = blank_position-1

        elif move == down: # blank switches place with number above it
            
            new_state[(blank_row-1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row-1) * k + blank_column]
           
            new_blank_position = (blank_row-1) * k + blank_column

        else:   # blank switches place with number below it
            
            new_state[(blank_row+1) * k + blank_column], new_state[blank_position] = new_state[blank_position], new_state[(blank_row+1) * k + blank_column]
            
            new_blank_position = (blank_row+1) * k + blank_column

        return (new_blank_position,new_state)

    def verify_solution(self, moves):
        curr_state = self.init_state
        blank_pos = self.find_blank()
        for move in moves:
            self.pretty_print(curr_state)
            blank_pos, curr_state = self.verify_move(move, curr_state,blank_pos)

        self.pretty_print(curr_state)

def heuristic(state):
    
    return manhattan_distance_1d(state)

def manhattan_distance_1d(state): # works
    k = int(math.sqrt(len(state)))
    num_tiles = len(state)- 1
    total_tile_distance = 0

    for i in range(k): # row 
        for j in range(k):  # column
            curr_tile = state[i*k+j]
            if curr_tile == 0:
                continue
            if curr_tile%k == 0:  # multiples of k treated diff
                correct_row = curr_tile//k - 1 
                correct_column = k-1 # always on last col
                num_tiles_away = abs(correct_row - i) + abs(correct_column - j)
                total_tile_distance += num_tiles_away
            else:
                correct_row = curr_tile//k  # quotient of curr_tile/k
                correct_column = curr_tile%k  - 1 # remainder of curr_tile/k
                num_tiles_away = abs(correct_row - i) + abs(correct_column - j)
                total_tile_distance += num_tiles_away
    return total_tile_distance




#n = 3 cases


#init_state = [[2,3,6],[1,5,8],[4,7,0]]
#init_state = [[1,2,3],[4,8,5],[7,6,0]]
#init_state = [[1,8,3],[5,2,4],[0,7,6]]
#init_state = [[1,2,3],[4,5,6],[7,0,8]]
#init_state = [[8,6,7],[2,5,4],[3,0,1]]

#goal_state = [[1,2,3],[4,5,6],[7,8,0]]

# n = 4 cases

init_state = [[1,2,3,4],[5,6,7,8],[10,11,0,12],[9,13,15,14]]
#init_state = [[12,15,6,10],[4,9,5,8],[14,13,0,2],[1,7,11,3]]
#init_state = [[14,10,5,13],[11,8,1,3],[2,9,12,6],[15,4,0,7]]

goal_state = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]


# n = 5 cases

#init_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,0,14,15],[16,17,13,18,19],[21,22,23,20,24]]
#init_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,0,22,23,24]]
#goal_state = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,0]]


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



'''


