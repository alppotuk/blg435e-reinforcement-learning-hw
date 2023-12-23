import sys
sys.path.append(".")

from ple.games.waterworld import WaterWorld
from ple import PLE
import numpy as np

class NaiveAgent():
    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs):
        a = self.actions[np.random.randint(0, len(self.actions))]
        return a

game = WaterWorld(width=800,height=600)
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

myAgent = NaiveAgent(p.getActionSet())

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
    if p.game_over(): #check if the game is over
        p.reset_game()
    low_level_obs = p.game.getGameState()
    high_level_obs = p.getScreenRGB()
    obs = high_level_obs
    action = myAgent.pickAction(reward, obs)
    reward = p.act(action)
    print(reward)