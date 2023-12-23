import random
import numpy as np


class NaiveRobot: # a naive bot that act all randomly
    def __init__(self, rows = 5, columns = 5, x = 0 , y = 4):
        self.q_table = np.zeros((rows, columns, 4))  
        self.x = x
        self.y = y

    def choose_action(self, state):
        return random.randint(0, 3)  # acts random

    def update_q_table(self, state, action, reward, next_state):
        pass

    def decay_epsilon(self):
        pass