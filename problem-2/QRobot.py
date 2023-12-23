import random
import numpy as np
from config import *

# these trajectories were given in the assignment pdf respectively
GIVEN_TRAJECTORIES_1 = [0, 0, 0, 0, 3, 3, 3, 3, 1] # U-U-U-U-R-R-R-R-D
GIVEN_TRAJECTORIES_2 = [3, 3, 0, 0, 0, 0, 3, 3, 1] # R-R-U-U-U-U-R-R-D

class QRobot: # a not so naive bot that learns with q learning
    def __init__(self, rows = 5, columns = 5, x = 0 , y = 4, q_table_mode="zeros",given_trajectories = None, robot_speed=10):
        self.q_table_mode = q_table_mode
        if self.q_table_mode == "zeros":
            self.q_table =  np.zeros((rows, columns, 4)) # 4 -> up down left right 
        elif self.q_table_mode == "random":
            self.q_table = 2 * np.random.rand(rows, columns, 4) - 1 # random.rand -> [0,1) | 2 * [0,1) = [0,2) | [0,2) - 1 = [-1, 1) 
        self.x = x
        self.y = y
        self.robot_speed = robot_speed
        self.epsilon = EPS_MAX # randomization factor
        self.jetpack_taken = False
        self.given_trajectories = given_trajectories
        self.trajectory_iterator = 0

    def choose_action(self, state):
        # if trajectory is given -> move accordingly
        if self.given_trajectories == 1:
            if(self.trajectory_iterator < len(GIVEN_TRAJECTORIES_1)):
                action =  GIVEN_TRAJECTORIES_1[self.trajectory_iterator]
                print(action)
                self.trajectory_iterator += 1
                self.move_robot(action)
                return action
            else:
                action = -1 # halts
                self.move_robot(action)
                return action
        elif self.given_trajectories == 2:
            if(self.trajectory_iterator < len(GIVEN_TRAJECTORIES_2)):
                action =  GIVEN_TRAJECTORIES_2[self.trajectory_iterator]
                self.trajectory_iterator += 1
                self.move_robot(action)
                return action
            else:
                action = -1 # halts
                self.move_robot(action)
                return action
        # if no given trajectory -> explore or exploit based on epsilon
        # extra conditions for if jetpack is taken -> move 1 or 2 at a time
        if not self.jetpack_taken:
            possible_actions = []

            if self.y > 0:  # do not move up if y == 0
                possible_actions.append(0)
            if self.y < 4:  # do not move down if y == 4
                possible_actions.append(1)
            if self.x > 0:  # do not move left if x == 0
                possible_actions.append(2)
            if self.x < 4:  # do not move right if x == 4
                possible_actions.append(3)

            if random.uniform(0, 1) < self.epsilon: # explore -> act random 
                action = random.choice(possible_actions)     
                self.move_robot(action)
                return action
            else:
                # exploit -> choose action with highest q value
                action = np.argmax(self.q_table[state])
                self.move_robot(action)
                return action
        else: 
            move = random.randint(1,2)
            possible_actions = []

            if self.y > 0 + (move - 1):  # do not move up if y == 0
                possible_actions.append(0)
            if self.y < 4 - (move - 1):  # do not move down if y == 4
                possible_actions.append(1)
            if self.x > 0 + (move - 1):  # do not move left if x == 0
                possible_actions.append(2)
            if self.x < 4 - (move - 1):  # do not move right if x == 4
                possible_actions.append(3)

            if random.uniform(0, 1) < self.epsilon: # explore -> act random 
                action = random.choice(possible_actions)     
                self.move_robot(action)
                if(move == 2):
                    self.move_robot(action)
                return action
            else:
                # exploit -> choose action with highest q value
                action = np.argmax(self.q_table[state])
                self.move_robot(action)
                if(move == 2): # if move is 2 repeat 
                    self.move_robot(action)
                return action
        
    def move_right(self): # moves robots position to right by one grid box
        if self.x < 4:
            self.x += 1

    def move_left(self): # moves robots position to left by one grid box
        if self.x > 0:
            self.x -= 1

    def move_up(self): # moves robots position to up by one grid box
        if self.y > 0:
            self.y -= 1

    def move_down(self): # moves robots position to down by one grid box
        if self.y < 4:
            self.y += 1

    def mov_NOT(self): # halts (for given trajectory endings)
        pass

    def move_robot(self, action): # operates moves with given action
        if action == 0:
            self.move_up()
        elif action == 1:
            self.move_down()
        elif action == 2:
            self.move_left()
        elif action == 3:
            self.move_right()
        elif action == -1:
            self.mov_NOT()

    def update_q_table(self, state, action, reward, next_state): # calculates new q values and updates table
        current_q_value = self.q_table[state[0]][state[1]][action]
        max_next_q_value = np.max(self.q_table[next_state[0]][next_state[1]])
        new_q_value = current_q_value + ALPHA * (reward + GAMMA * max_next_q_value - current_q_value)
        self.q_table[state[0]][state[1]][action] = new_q_value        

    def decay_epsilon(self): # epsilon calculation 
        self.epsilon = max(EPS_MIN, self.epsilon * DECAY_RATE )

    def print_q_table(self): # prints q table into console (was useful before i made visualazation)
        for i in range(5):
            for j in range(5):
                print(f"State: ({i}, {j})")
                for action in range(4):
                    print(f"  Action {action}: Q-value = {self.q_table[i, j, action]}")
        