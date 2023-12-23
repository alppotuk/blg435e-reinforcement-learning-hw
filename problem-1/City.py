import random
city_names = ["Boston", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Erzincan", "Toronto", "Rome", "Oslo", "Bangkok",  "Vienna"]

class City: # structure that holds city information
    used_names = set()
    index_counter = 0

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.coordinate = (self.x, self.y)
        self.index = City.index_counter 
        self.name = city_names[City.index_counter]
        City.index_counter += 1  
        

    
