import math
class DistanceDict: # structure that holds distances between cities
    distance_dict = {}
    def __init__(self, cities):
        for city in cities:
            self.distance_dict[city] = {}
            for other_city in cities:
                if city != other_city:
                    self.distance_dict[city][other_city] = self.calculate_distance(city, other_city)

    def calculate_distance(self, city1, city2):
        return math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)