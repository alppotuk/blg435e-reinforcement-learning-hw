import pygame
import sys

lexend = "assets\LexendExa-Regular.ttf"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (112, 173, 71) # starting city
RED = (198, 42, 17) # ending city

width, height = 800, 600

class SolutionVisualizer:
    def __init__(self, width= 800, height= 600,cities= None, cost= 0 ):
        self.width = 800
        self.height = 600
        self.cities = cities
        self.cost= cost
        pygame.init() # initialize pygame
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Solution Visualizer')

        for city in self.cities:
            city.x = (city.coordinate[0] +100)  * 4
            city.y = (city.coordinate[1] + 130) * 2
            city.coordinate = (city.x, city.y)

    def draw_cities(self): # draws cities and puts texts to the screen
        font = pygame.font.Font(lexend, 12) 

        for i in range(len(self.cities) - 1): # draw lines
             pygame.draw.line(self.screen, BLACK, self.cities[i].coordinate, self.cities[i + 1].coordinate, 2) 

        for i, city in enumerate(self.cities): # draw circles
            color = BLACK 
            if i == 0:
                color = GREEN
            elif i == len(self.cities) - 1:
                color = RED
            pygame.draw.circle(self.screen, color, city.coordinate, 5)
            text = font.render(f"{city.name}", True, color)
            text_rect = text.get_rect(center = (city.x, city.y + - 20))
            self.screen.blit(text, text_rect)

            font = pygame.font.Font(lexend, 15) 
            cost = font.render(f"Cost: {self.cost:.2f}", True, BLACK)
            cost_rect = cost.get_rect(center = (600, 100))
            self.screen.blit(cost, cost_rect)
            
    def visualize(self): # visualizer function
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill(WHITE) 
            self.draw_cities()
           
            pygame.display.flip()
        

