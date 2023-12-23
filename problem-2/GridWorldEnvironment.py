import pygame
import sys
import numpy as np
# color values in RGB format
BLACK = (0, 0, 0)
GRAY = (166, 166, 166)
BLUE = (91, 155, 213)
WHITE = (255, 255, 255)
YELLOW = (255, 192, 0)
GREEN = (112, 173, 71)
ORANGE = (197, 90, 17)
# positions
STARTING_POSITION = (0, 4)
WATER_POSITIONS = [(0, 2), (1, 2), (3, 1), (3, 2), (4, 2)]
JETPACK_POSITION = (4, 4)
GOAL_POSITION = (4, 1)
# positive and negative feedbacks
REWARD = 1
PUNISHMENT = -1
# a beautiful font
lexend = "assets\LexendExa-Regular.ttf"

class GridWorldEnv:
    def __init__(self, rows=5, columns=5, cell_size=100, robot= None,mode="vanilla"):        
        self.rows = rows
        self.columns = columns
        self.cell_size = cell_size
        self.robot = robot
        self.mode = mode
        # screen setup with a little bit padding
        self.screen_width = (columns + 2) * cell_size
        self.screen_height = (rows + 2) * cell_size
        self.screen = pygame.display.set_mode((self.screen_width , self.screen_height ))
        pygame.init() # pygame initialization
        pygame.display.set_caption("GridWorld Q-Learning Environment")  
        self.clock = pygame.time.Clock()

        # setup rewards and punishments
        self.grid = np.zeros((columns, rows))
        for position in WATER_POSITIONS:
            self.grid[position[0]][position[1]] = PUNISHMENT      
        self.grid[GOAL_POSITION[0]][GOAL_POSITION[1]] = REWARD

    def draw_grid(self): # sets up grid environment with boxes and lines
        # white background
        self.screen.fill(WHITE) 
        # color boxes
        for i in range(self.columns):
            for j in range(self.rows):
                color = self.get_grid_color(i, j) 
                pygame.draw.rect(self.screen, color, (i * self.cell_size + self.cell_size, j * self.cell_size + self.cell_size, self.cell_size, self.cell_size))

        # vertical lines
        for x in range(self.columns + 1):
            pygame.draw.line(self.screen, BLACK, ((x + 1) * self.cell_size, self.cell_size), ((x + 1) * self.cell_size, self.screen_height - self.cell_size))

        # horizontal lines
        for y in range(self.rows + 1):
            pygame.draw.line(self.screen, BLACK, (self.cell_size, (y + 1) * self.cell_size), (self.screen_width- self.cell_size, (y + 1) * self.cell_size))

        self.draw_q_table_diagonals() # adds diagonal lines for q table visualization

    def draw_q_table_diagonals(self): # draws diagonal lines for q table
        for x in range(6):
            pygame.draw.line(self.screen, BLACK, (self.cell_size, (x + 1) * self.cell_size), ((x + 1) * self.cell_size, self.cell_size))
            pygame.draw.line(self.screen, BLACK, (self.screen_width - self.cell_size, (x + 1) * self.cell_size), ( (x + 1) * self.cell_size, self.screen_height - self.cell_size))
            pygame.draw.line(self.screen, BLACK, (self.cell_size, (x + 1) * self.cell_size), (self.screen_width - (x + 1) * self.cell_size, self.screen_height -  self.cell_size))
            pygame.draw.line(self.screen, BLACK, (self.cell_size * (x + 1), self.cell_size), (self.screen_width -  self.cell_size, self.screen_height -  (x + 1) * self.cell_size))
                
    def draw_robot(self): # draws robot at its positon
        robot_x = (self.robot.x + 1) * self.cell_size + self.cell_size // 2
        robot_y = (self.robot.y + 1) * self.cell_size + self.cell_size // 2
        if self.robot.jetpack_taken: # symbolizes that jetpack is taken
            pygame.draw.circle(self.screen, ORANGE, (robot_x, robot_y), self.cell_size // 6)
        else: 
            pygame.draw.circle(self.screen, BLACK, (robot_x, robot_y), self.cell_size // 6)

    def get_grid_color(self, i, j): # returns grid box color based on given position (i, j)
        position = (i, j)
        if( self.mode == "jetpack" and position == JETPACK_POSITION and not self.robot.jetpack_taken):
            return ORANGE
        elif position in WATER_POSITIONS:
            return BLUE
        elif position == STARTING_POSITION:
            return YELLOW
        elif position == GOAL_POSITION:
            return GREEN
        else:
            return GRAY
        
    def draw_q_values(self): # adds info message, epsilon value and q values to the screen
        font = pygame.font.Font(lexend, 12) 
        for i in range(self.columns):
            for j in range(self.rows):
                state = (i, j)
                q_values = self.robot.q_table[state[0], state[1], :]
                for k in range(len(q_values)):
                    x_offset = 0
                    y_offset = 0 
                    if(k == 0):
                        y_offset = - self.cell_size / 4
                    if(k == 1):
                        y_offset = + self.cell_size / 4
                    if(k == 2):
                        x_offset = - self.cell_size / 4
                    if(k == 3):
                        x_offset = + self.cell_size / 4
                    x = (i + 1) * self.cell_size + self.cell_size // 2 + x_offset
                    y = (j + 1) * self.cell_size + self.cell_size // 2 + y_offset

                    text = font.render(f"{q_values[k]:.2f}", True, BLACK)
                    text_rect = text.get_rect(center=(x, y))

                    self.screen.blit(text, text_rect)

        eps_font = pygame.font.Font(lexend, 18) 
        message = eps_font.render(f"QRobot {self.mode.capitalize()} Environment", True, BLACK)
        message_rect = message.get_rect(center = (self.screen_width / 2, self.cell_size / 2))
        self.screen.blit(message, message_rect)
        epsilon = eps_font.render(f"epsilon: {self.robot.epsilon:.2f}", True, BLACK)
        epsilon_rect = epsilon.get_rect(center=(self.screen_width / 2, 13 * self.cell_size / 2)) 
        self.screen.blit(epsilon, epsilon_rect)

    def update_q_learning(self): # feeds robot with state and reward to update its q value table
        state = (self.robot.x, self.robot.y)
        if state == JETPACK_POSITION and self.mode == "jetpack":
            self.robot.jetpack_taken = True
            for position in WATER_POSITIONS: # remove negative feedback from the water positions 
                self.grid[position[0]][position[1]] = 0

        action = self.robot.choose_action(state)
        next_state = (self.robot.x, self.robot.y)

        if(state == next_state):
            reward = PUNISHMENT # negative feedback if not moved (stuck on edges)
        else:
            reward = self.grid[self.robot.x, self.robot.y]
        
        if(action != -1): # action retusn -1 only if trajectories given to the robot
            self.robot.update_q_table(state, action, reward, next_state)
            self.robot.decay_epsilon()

    def run(self):
        running = True
        while running:
            self.draw_grid()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update_q_learning() # robot makes a move then updates its q table based on that
            self.draw_robot() # draws robot in its new position
            self.draw_q_values() # updates q values on the screen

            pygame.display.flip()
            self.clock.tick(self.robot.robot_speed) # game speed set here

        pygame.quit()
        sys.exit()


    
