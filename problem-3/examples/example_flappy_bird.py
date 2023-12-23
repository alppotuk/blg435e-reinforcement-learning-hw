from itertools import count
import math
import random
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model.DQN import DQN
import torch.optim as optim
from model.ReplayMemory import ReplayMemory, Transition
from model.config import *

sys.path.append(".")

from ple.games.flappybird import FlappyBird
from ple import PLE


class NaiveAgent():
    def __init__(self, actions):
        self.actions = actions
        self.c = 0
        self.r = 0

    def act_random(self):  
        return p.getActionSet()[0] if random.random() > 0.5 else 0; 
        

    def pickAction(self, reward, obs):
        # a = self.actions[np.random.randint(0, len(self.actions))]
        # return a
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.c = self.c + 1
                return policy_net(obs).max(1).indices.view(1, 1)
        else:
            self.r = self.r + 1
            return torch.tensor([[self.act_random()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_tensors = [torch.tensor(r) for r in batch.reward]
    reward_tensors =  [tensor.view(1) if tensor.dim() == 0 else tensor for tensor in reward_tensors]
    reward_tuple = tuple(reward_tensors)
    reward_batch = torch.cat(reward_tuple)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    action_batch = torch.clamp(action_batch, 0, 1)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()




game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
episode_durations = []
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

myAgent = NaiveAgent(p.getActionSet())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get number of actions from gym action space
n_actions = len(p.getActionSet())
print("p.actionSet: ",p.getActionSet())
# Get the number of state observations
# p.game.getGameState -> low level 
# p.getScreenRgb -> high level
# observations = p.game.getGameState()
observations = p.game.getGameState()
n_observations = len(observations)

print("n_actions:", n_actions, "n_observations:",n_observations);

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

nb_episodes = 800
reward = 0.0

for episode in range(nb_episodes):
    print(episode, "c: ", myAgent.c, "r: ",myAgent.r)
    myAgent.c = 0
    myAgent.r = 0
    low_level_obs = p.game.getGameState()
    high_level_obs = p.getScreenRGB()

    obs = low_level_obs
    obs_values = np.array(list(obs.values()), dtype=np.float32)
    state = torch.tensor(obs_values, dtype=torch.float32,
                        device=device).unsqueeze(0)
    for frame in count():        
        action = myAgent.pickAction(reward, state)
        reward = p.act(action) 

        if p.game_over():
            next_state = None
        else:
            next_obs = p.game.getGameState()
            next_obs_values = np.array(list(next_obs.values()), dtype=np.float32)
            next_state = torch.tensor(next_obs_values, dtype=torch.float32,
                            device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, torch.tensor(reward).unsqueeze(0))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if p.game_over():
            episode_durations.append(frame + 1)
            plot_durations()
            break

    p.reset_game()
    