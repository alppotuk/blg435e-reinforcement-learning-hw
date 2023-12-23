# METE ALP POTUK
# 150190013
# AI HW2 PROBLEM-1
from City import City
from TSPGA import TSPGA
from SolutionVisualizer import SolutionVisualizer

cities = []
with open('cities12.txt','r') as file:
    for line in file:
        values = line.split()
        if len(values) == 2:
            x = values[0]
            y = values[1]
            cities.append(City(x, y))

MyGAHandler = TSPGA(cities, population_size=100)
solution = MyGAHandler.run_algorithm('advanced', generations=1000)
# running modes => 'basic', 'elitism', 'advanced'  
solution_visualizer = SolutionVisualizer(cities= solution[0], cost=solution[1])
solution_visualizer.visualize()

